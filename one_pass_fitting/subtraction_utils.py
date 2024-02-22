import numpy as np
try:
    from collections import Iterable
except ImportError:
    from collections.abc import Iterable

__all__ = ['make_model_image', 'subtract_psfs', 'get_subtrahend']

def compute_cutout(x, y, flux, mod, shape, size=None):
    """
    Computes a cutout of the evaluated PSF model at the specified position.

    Parameters
    ----------
    x : float
        X-coordinate of the center of the cutout (zero-indexed pixel position).
    y : float
        Y-coordinate of the center of the cutout (zero-indexed pixel position).
    flux : float
        Flux of the PSF model.
    mod : `~photutils.psf.GriddedPSFModel`
        The PSF model to evaluate.
    shape : tuple
        Shape of the image data array.
    size : int or iterable of length 2 or ``None``, optional
        Size of the cutout. If None, the size is determined based on the PSF model shape.

    Returns
    -------
    cutout : `~numpy.ndarray`
        The evaluated PSF model cutout.
    x_grid, y_grid : `~numpy.ndarray`
        Arrays of pixel coordinates for the cutout.
    """

    if isinstance(size, int):
        sy = size
        sx = size
    elif isinstance(size, Iterable):
        if len(size) != 2:
            raise ValueError('If size is iterable, it must contain only 2 elements (y_size, x_size)')
        else:
            sy = size[0]
            sx = size[1]
    elif size is None:
        # Determine size based on the PSF model shape
        modsizey = mod.data.shape[-2]
        modsizex = mod.data.shape[-1]
        oversampling = mod.oversampling
        sy = int(modsizey/oversampling)
        sx = int(modsizex/oversampling)
    else:
        raise ValueError('Cannot understand size. size must be either an int, 2-tuple or None.')

    # Get the half sizes for left/right bound distances
    hx = sx//2
    hy = sy//2

    # Get integer pixel positions of centers, with appropriate
    # edge convention, i.e. edges of pixels are integers, 0 indexed
    x_cen = int(x+.5)
    y_cen = int(y+.5)

    # Handle cases where model spills over edge of data array
    x1 = max(0, x_cen - hx)
    y1 = max(0, y_cen - hy)

    # upper bound is exclusive, so add 1 more
    x2 = min(x_cen + hx+1, shape[1])
    y2 = min(y_cen + hy+1, shape[0])

    # Create grids of pixel coordinates
    y_grid, x_grid = np.mgrid[y1:y2, x1:x2]

    # Evaluate the PSF model at the cutout position
    cutout = mod.evaluate(x_grid, y_grid, flux, x, y)
    return cutout, x_grid, y_grid

def get_subtrahend(xs, ys, fluxes, mod, shape, size=None):
    """
    Constructs the image to be subtracted based on PSF models at given positions.

    Parameters
    ----------
    xs : array_like
        X-coordinates of PSF positions.
    ys : array_like
        Y-coordinates of PSF positions.
    fluxes : array_like
        Fluxes of the PSF models.
    mod : `~photutils.psf.GriddedPSFModel`
        The PSF model to evaluate.
    shape : tuple
        Shape of the data array.
    size : int or iterable of length 2 or ``None``, optional
        Size of the cutout. If None, the size is determined based on the PSF model shape.

    Returns
    -------
    subtrahend : `~numpy.ndarray`
        The constructed image to be subtracted.
    """

    # Initialize the array to be subtracted
    subtrahend = np.zeros(shape, dtype=float)

    for x, y, flux in zip(xs, ys, fluxes):
        if flux == np.nan: # Skip ones without good fluxes
            continue
        # Compute the cutout for each PSF position
        cutout, x_grid, y_grid = compute_cutout(x, y, flux, mod, shape, size=size)

        # Important: use += to account for overlapping cutouts
        subtrahend[y_grid, x_grid] += cutout

    return subtrahend

def make_model_image(data, cat, mod, size=None):
    """
    Reconstructs image from psf models at given positions/fluxes

    Parameters
    ----------
    data : `~numpy.ndarray`
        The image from which to get the desired shape of the model image
    cat : `~astropy.table.Table`
        Astropy table containing flux and positions.  The flux column must be labeled 'f' and the positions 
        must be 'x' and 'y'.  The fluxes must be in the same units as the `data` and the positions must be in 
        zero-indexed pixels
    mod : `~photutils.psf.GriddedPSFModel`
        The PSF model to generate the star images from the fit values
    size : int or iterable of length 2 or ``None``, optional
        Size of the cutout. If None, the size is determined based on the PSF model shape.

    Returns
    -------
    model_image : `~numpy.ndarray`
        The data array with PSF models subtracted.
    """

    shape = data.shape

    fluxes = cat['f']
    xs = cat['x'].data
    ys = cat['y'].data

    # Evaluate the PSF at each x, y, flux, and place it in subtrahend
    model_image = get_subtrahend(xs, ys, fluxes, mod, shape, size=size)
    return model_image

def subtract_psfs(data, cat, mod, size=None):
    """
    Subtracts PSF models from the specified positions in the data.

    Parameters
    ----------
    data : `~numpy.ndarray`
        The image from which to subtract the psfs
    cat : `~astropy.table.Table`
        Astropy table containing flux and positions.  The flux column must be labeled 'f' and the positions 
        must be 'x' and 'y'.  The fluxes must be in the same units as the `data` and the positions must be in 
        zero-indexed pixels
    mod : `~photutils.psf.GriddedPSFModel`
        The PSF model to generate the star images from the fit values
    size : int or iterable of length 2 or ``None``, optional
        Size of the cutout. If None, the size is determined based on the PSF model shape.

    Returns
    -------
    difference : `~numpy.ndarray`
        The data array with PSF models subtracted.
    """

    # Evaluate the PSF at each x, y, flux, and place it in subtrahend
    subtrahend = make_model_image(data, cat, mod, size=size)

    # Subtract the image
    difference = data - subtrahend
    return difference