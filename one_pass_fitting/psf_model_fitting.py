import numpy as np

from scipy.optimize import curve_fit

class FlattenedModel:
    """This is a wrapper class to return the evaluated psf as a 1d array"""
    # Could probably make this a subclass of FittableImageModel instead.
    def __init__(self, psf_fittable_model):
        self.mod = psf_fittable_model

    def evaluate(self, x_y, flux, x_0, y_0):
        """Evaluate the model, and flatten the output"""
        x, y = x_y
        return np.ravel(self.mod.evaluate(x, y, flux=flux,
                                          x_0=x_0, y_0=y_0))

def fit_star(xi, yi, bg_est, model, im_data, fit_shape=(5,5)):
    """
    Fit object at some (x,y) in data array with PSF model

    This is the function that fits each object with the PSF.  It cuts out
    a 5x5 pixel box around the peak of the object, and fits the PSF to that
    cutout.  If it fails, all return values are nans.  Uses curve_fit in
    scipy.optimize for fitting.  The input pixels are given sigmas of
    sqrt(abs(im_data)) for weighting the fit.

    Parameters
    ----------
    xi : int or float
        x position of objects peak, in pixel coordinates.  Can just be the
        integer pixel position.
    yi : int or float
        y position of objects peak, in pixel coordinates.  Can just be the
        integer pixel position.
    bg_est : float
        Sky background level around the object to be subtracted before fitting
    model : `FlattenedModel`
        The GriddedPSFModel, with an evaluate method that returns a flattened
        version of the output (instead of shape being (n,m), get (n*m,))
    im_data : `numpy.ndarray`
        The full image (single chip) data array from which the object should
        be cut out.
    fit_shape : length-2 array_like
        Shape of the cutout of `im_data` to be fit, in pixels. Default (5,5)


    Returns
    -------
    f : float
        The fitted flux of the model
    x : float
        The fitted x position of the model
    y : float
        The fitted y position of the model
    q : float
        The calculated fit quality
    cx : float
        The scaled residual of the central pixel
    pix_flux : float
        Sum of the pixel values of the fitted model (excludes nan pixels)
    npix_fit : float
        Number of pixels actually fit by the model (excludes nan pixels)

    """
    # Define a 5x5 pixel box around the peak of the object to fit the PSF
    _validate_fit_shape(fit_shape)
    midx = np.median(np.arange(fit_shape[1])).astype(int)
    midy = np.median(np.arange(fit_shape[0])).astype(int)
    yg, xg = np.mgrid[-midy:midy+1,-midx:midx+1]

    yf, xf = yg+int(yi+.5), xg+int(xi+.5) # Add 0.5 to deal with coordinates -> indices offset
    # Use this to handle out of bounds slices, preserves shape of cutout
    cutout = slice_array_with_nan(im_data, xf, yf)

    # Estimate initial flux guess for the model, subtracting the sky
    f_guess = np.nansum(cutout - bg_est)

    # Set initial guess for the fit parameters and bounds for curve_fit
    p0 = [f_guess, xi+.5, yi+.5]

    # Turned this off to improve fitting accuracy
    # bounds = ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])

    # Have to use the Flattened model in the curve_fit call
    fmodel = FlattenedModel(model)
    
    # temporarily mask the nans, set their weights and values to 0
    # weights are like 1/sigma
    # TODO: Remove this hack when JWST upgrades scipy to 1.11
    nanmask = np.isnan(cutout)
    sigmas = np.sqrt(np.abs(cutout))
    sigmas[nanmask] = np.inf
    cutout[nanmask] = 0.

    # TODO: Add the npix and total fit flux computations
    try:
        # Fit the model using curve_fit from scipy.optimize, using flattened
        # PSF model from FlattenedModel and pixel weights
        # Weight is just sqrt of the cutout, roughly the SNR of each pixel.
        popt, pcov = curve_fit(fmodel.evaluate, (xf,yf),
                               np.ravel(cutout)-bg_est, p0=p0,
                               sigma=np.ravel(sigmas))

        # Set the flagged values back to nan
        cutout[nanmask] = np.nan

        # Calculate quality of fit and scaled residual of the central pixel
        resid = cutout - bg_est - model.evaluate(xf, yf, *popt)
        q = np.sum(np.abs(resid)[~nanmask])/np.sum(model.evaluate(xf, yf, *popt)[~nanmask])
        cx = resid[midy,midx]/popt[0]
        pix_flux = np.sum(model.evaluate(xf, yf, *popt)[~nanmask])
        npix_fit = np.sum(~nanmask)

    except (RuntimeError, ValueError):
        # If fitting fails, set all return values to nan
        popt = [np.nan, np.nan, np.nan]
        q = np.nan
        cx = np.nan
        pix_flux = np.nan
        npix_fit = np.nan

    # Return the fitted flux, x and y positions, fit quality and scaled
    # residual of the central pixel
    f, x, y = popt
    return f, x, y, q, cx, pix_flux, npix_fit

def _validate_fit_shape(fit_shape: tuple):
    if (not fit_shape[0]%2) or (not fit_shape[1]%2):
        raise ValueError(f'fit_shape must be a 2-tuple of ODD numbers, got:{fit_shape}')
    
def slice_array_with_nan(array, xf, yf):
    """
    Slice an array using mesh grid indices with out-of-bounds filled with NaN.

    Parameters
    ----------
    array : numpy.ndarray
        Input array to be sliced.
    xf : array
        2D array of x values to slice out
    yf : slice
        2D array of y values to slice out

    Returns
    -------
    sliced_array : numpy.ndarray
        Sliced array using mesh grid indices with out-of-bounds filled with NaN.
    """

    # Generate mesh grids using np.mgrid with the provided slices

    # Mask out-of-bounds indices using the array shape
    mask = (xf >= array.shape[1]) | (yf >= array.shape[0]) | (xf <= 0) | (yf <= 0)

    # Create a new array with NaN at out-of-bounds indices
    sliced_array = np.empty_like(xf, dtype=float)
    sliced_array[~mask] = array[yf[~mask], xf[~mask]]
    sliced_array[mask] = np.nan

    return sliced_array