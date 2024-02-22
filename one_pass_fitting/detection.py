import numpy as np
from photutils.aperture import CircularAperture
from photutils.segmentation import SourceFinder, SourceCatalog
from scipy.ndimage import maximum_filter, convolve, label
from scipy.spatial import cKDTree


def _filter_images(data, hmin):
    """
    Performs maximum filtering on image with a circular kernel.

    If the maximum filtered image is equal to the original data at a given pixel,
    then that pixel is a local maximum.

    Parameters:
    -----------
    data : 2D array
        An array of image data.
    hmin : int
        Size of the circular kernel, desired distance between local maxima

    Returns:
    --------
    filt_image : ndarray
            A 2D array containing the maximum filtered image.
    """

    # Laziest way to get a circle mask
    fp = CircularAperture((0, 0), r=hmin).to_mask().data > 0.1
    fp = fp.astype(bool)

    # Apply maximum filter, flux filter
    filt_image = maximum_filter(data, footprint=fp, mode="constant", cval=0)

    return filt_image


def _conv_origin(data, origin):
    """
    Shifts the convolution kernel around for peak finding.

    This function effectively does 2x2 binning of data, but allows you
    to keep the same pixel coordinates via convolutions rather than direct binning,
    and shifts the origin of the convolution around to allow measurement of all 4 of
    the possibilites of a 2x2 box containing a given pixel.

    Parameters:
    -----------
    data : 2D array
        An array of image data.
    origin : tuple
        A tuple of two integers that represent the x and y coordinates of
        the origin of the convolution kernel.

    Returns:
    --------
    convolved_data : 2D array
        Array of data convolved with 2x2 kernel of ones, effectively summing up 2x2 boxes.

    """
    # Create a convolution kernel of size (2, 2) filled with ones
    kernel = np.ones((2, 2))

    # Perform convolution with the kernel and the data
    convolved_data = convolve(
        data, weights=kernel, mode="constant", cval=0, origin=origin
    )

    return convolved_data


def calc_peak_fluxes(data):
    """
    Calculates the peak fluxes of an image.

    Parameters:
    -----------
    data : 2D array
        An array of image data.

    Returns:
    --------
    max_4sum : 2D array
        An array of data representing the peak fluxes of the image.

    """
    # Define the origins for the convolution kernel
    origins = [(0, 0), (0, -1), (-1, 0), (-1, -1)]

    # Perform convolution for each origin and find the maximum value along
    # the axis of the array of results
    max_4sum = np.amax([_conv_origin(data, o) for o in origins], axis=0)

    return max_4sum


def _find_sources(data, filt_image, max_4sum, fmin=1e3, pmax=7e4):
    """
    Finds the locations of the sources in the image data.

    This function applies the criteria to find potential sources in image.
    It checks where `data==filt_image` to find local maxima, if the 2x2 summed
    peaks are > `fmin`, and if the brightest pixel in the source are < `pmax`.

    Parameters:
    -----------
    data : ndarray
        A 2D array of image data.
    filt_image : ndarray
        A 2D array containing the maximum filtered image.
    max_4sum : ndarray
        A 2D array containing the maximum of the sums of 2x2 neighborhoods around each pixel.
    fmin : float, optional
        The minimum flux required for a source to be considered valid.
    pmax : float, optional
        The maximum pixel value allowed for a source to be considered valid.

    Returns:
    --------
    xi : ndarray
        An array containing the integer x-coordinates of the sources.
    yi : ndarray
        An array containing the integer y-coordinates of the sources.
    """
    # find indices where the maximum filtered image is equal to data
    # why this is >= rather than == I don't remember :(
    yi, xi = np.where(data >= filt_image)

    # Mask out sources with 2x2 peaks that are too faint or single peak pixels that are too bright
    mask1 = (max_4sum[yi, xi] > fmin) & (data[yi, xi] < pmax)

    # Mask out anything within 3 pixels of the edge, as the fitting uses a 5x5 region
    mask2 = (xi > 2) & (yi > 2) & (xi < data.shape[1] - 3) & (yi < data.shape[0] - 3)

    # Combine the masks
    mask = mask1 & mask2

    yi = yi[mask]
    xi = xi[mask]
    return (xi, yi)


def detect_peaks(data, hmin, fmin, pmax):
    """
    Detect sources in an astronomical image.

    This function performs source detection on a 2D image by finding local
    maxima in an image that within a range of brightness.  Specifically, it
    first finds local maxima in `data` that are at least `hmin` pixels separated
    from any higher pixel value via using a maximum filter on `data`, with a
    circular kernel of radius `hmin`.  For each of the detected maxima, it then
    calculates the sum of the  brightest 2x2 pixel box  containing the given
    maximum (for any pixel, there are 4 2x2 blocks that contain it).  If the sum
    is less than `fmin` the source is rejected.  It then throws out any source
    with a brightest pixel value greater than `pmax`.  After all these steps,
    the x and y pixel indices of the remaining sources are returned in two
    arrays.

    Parameters:
    -----------
    data : ndarray
        A 2D array of image data to perform source detection on.
    hmin : int
        Size of the circular kernel used for maximum filtering, which controls the
        desired distance between local maxima.
    fmin : float
        The minimum flux required for a source to be considered a valid detection.
    pmax : float
        The maximum pixel value allowed for a source to be considered valid.

    Returns:
    --------
    xi : ndarray
        An array containing the integer x-coordinates of the detected sources.
    yi : ndarray
        An array containing the integer y-coordinates of the detected sources.

    Example
    -------
    >>> data = 2D array containing image data
    >>> hmin = 5  # Circular kernel size for maximum filtering
    >>> fmin = 1000.0  # Minimum flux for a valid source
    >>> pmax = 50000.0  # Maximum pixel value for a valid source
    >>> xi, yi = detect_sources(data, hmin, fmin, pmax)
    >>>  # `xi` and `yi` contain the coordinates of the detected sources.

    Notes
    -----
    - This function combines various techniques, including maximum filtering, convolution,
      and peak finding, to locate sources in the input image.
    - Detected sources must meet the specified criteria of minimum flux (`fmin`) and
      maximum pixel value (`pmax`) to be considered valid detections.
    """
    filt_image = _filter_images(data, hmin)
    max_4sum = calc_peak_fluxes(data)
    xi, yi = _find_sources(data, filt_image, max_4sum, fmin, pmax)
    return xi, yi


def remove_nearby(seg_tbl, distance_factor=2.0):
    tree = cKDTree(np.array([seg_tbl["xcentroid"], seg_tbl["ycentroid"]]).T)
    approx_rad = seg_tbl["area"] ** 0.5

    ball_inds = tree.query_ball_point(tree.data, approx_rad * distance_factor)
    n_close = [len(bi) - 1 for bi in ball_inds]

    dmask = np.ones(len(seg_tbl), dtype=bool)
    inds_to_check = np.where(np.array(n_close) > 0)[0]
    to_remove = set()
    for ind in inds_to_check:
        if ind in to_remove:
            continue
        for close_ind in ball_inds[ind][1:]:
            to_remove.add(close_ind)

    if len(to_remove) > 0:
        dmask[np.array(list(to_remove))] = False

    return seg_tbl[dmask]


def detect_sat_jwst(dq, distance_factor=2.5):
    """
    Detect saturated pixels of saturated stars in the JWST data.

    Parameters
    ----------
    dq : numpy.ndarray
        Data quality array indicating pixel flags.
    distance_factor : float, optional
        Scaling factor for masking nearby saturated sources. Default is 2.5.

    Returns
    -------
    seg_tbl : astropy.table.Table
        Table containing information about detected saturated sources, sorted by area.

    Notes
    -----
    This function identifies saturated pixels in JWST data using a data quality array (dq).
    It combines multiple flag values to isolate saturated pixels, considering pipeline-related issues.
    The `distance_factor` scales the masking of nearby saturated sources during detection.
    The function employs SourceFinder and SourceCatalog from photutils.segmentation
    to identify and create a table (`seg_tbl`) of saturated sources sorted by area.



    """
    sat = (np.bitwise_and(dq, 2) / 2).astype(bool)
    # This is a kludge to get rid of high flag value pixels, and to deal with pipeline issues
    sat1 = np.logical_and(np.bitwise_and(dq, 1).astype(bool), (dq < 8192))

    sat_combined = np.logical_or(sat, sat1).astype(float)

    finder = SourceFinder(npixels=2, connectivity=4, deblend=False)
    segmap_combined = finder(sat_combined, threshold=0.5)

    seg_tbl = SourceCatalog(sat_combined, segmap_combined).to_table()
    seg_tbl.sort("area", reverse=True)

    seg_tbl = seg_tbl[~np.isnan(seg_tbl["xcentroid"])]
    # Check the if not empty
    if len(seg_tbl) > 0:
        seg_tbl = remove_nearby(seg_tbl, distance_factor)
    else:
        print("WARNING: no saturated sources detected")

    return seg_tbl


def compute_nan_areas(xs, ys, data):
    """Computes the size (number of pixels) of contiguous blocks of nans"""
    ix = np.array(xs).astype(int)
    iy = np.array(ys).astype(int)
    binary_image = np.isnan(data)

    labeled, _ = label(binary_image)
    areas = []
    for x, y in zip(ix, iy):
        val = labeled[y, x]
        if val == 0:
            areas.append(0)
        else:
            count = np.sum(labeled == val)
            areas.append(count)

    return np.array(areas)
