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


def fit_star(xi, yi, bg_est, model, im_data):
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

    """
    # Define a 5x5 pixel box around the peak of the object to fit the PSF
    yg, xg = np.mgrid[-2:3,-2:3]
    yf, xf = yg+int(yi+.5), xg+int(xi+.5) # Add 0.5 to deal with coordinates -> indices offset
    cutout = im_data[yf, xf]

    # Estimate initial flux guess for the model, subtracting the sky
    f_guess = np.sum(cutout - bg_est)

    # Set initial guess for the fit parameters and bounds for curve_fit
    p0 = [f_guess, xi+.5, yi+.5]

    # Turned this off to improve fitting accuracy
    # bounds = ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])

    # Have to use the Flattened model in the curve_fit call
    fmodel = FlattenedModel(model)

    try:
        # Fit the model using curve_fit from scipy.optimize, using flattened
        # PSF model from FlattenedModel and pixel weights
        # Weight is just sqrt of the cutout, roughly the SNR of each pixel.
        popt, pcov = curve_fit(fmodel.evaluate, (xf,yf),
                               np.ravel(cutout)-bg_est, p0=p0,
                               sigma=np.ravel(np.sqrt(np.abs(cutout))))

        # Calculate quality of fit and scaled residual of the central pixel
        resid = cutout - bg_est - model.evaluate(xf, yf, *popt)
        q = np.sum(np.abs(resid))/popt[0]
        cx = resid[2,2]/popt[0]

    except (RuntimeError, ValueError):
        # If fitting fails, set all return values to nan
        popt = [np.nan, np.nan, np.nan]
        q = np.nan
        cx = np.nan

    # Return the fitted flux, x and y positions, fit quality and scaled
    # residual of the central pixel
    f, x, y = popt
    return f, x, y, q, cx
