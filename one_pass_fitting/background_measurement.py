import numpy as np


from astropy.table import Table
from scipy.stats import sigmaclip
from photutils.aperture import CircularAnnulus


def aperture_stats_tbl(data, apertures, method="exact", sigma_clip=True):
    """Computes mean/median/mode/std in Photutils apertures.

    Compute statistics for custom local background methods.
    This is primarily intended for estimating backgrounds
    via annulus apertures.  The intent is that this falls easily
    into other code to provide background measurements.

    Parameters
    ----------
    data : array
        The data for the image to be measured.
    apertures : photutils PixelAperture object (or subclass)
        The phoutils aperture object to measure the stats in.
        i.e. the object returned via CirularAperture,
        CircularAnnulus, or RectangularAperture etc.
    method: str
        The method by which to handle the pixel overlap.
        Defaults to computing the exact area.
        NOTE: Currently, this will actually fully include a
        pixel where the aperture has ANY overlap, as a median
        is also being performed.  If the method is set to 'center'
        the pixels will only be included if the pixel's center
        falls within the aperture.
    sigma_clip: bool
        Flag to activate sigma clipping of background pixels

    Returns
    -------
    stats_tbl : astropy.table.Table
        An astropy Table with the colums X, Y, aperture_mean,
        aperture_median, aperture_mode, aperture_std, aperture_area
        and a row for each of the positions of the apertures.

    """

    # Get the masks that will be used to identify our desired pixels.
    masks = apertures.to_mask(method=method)

    # Compute the stats of pixels within the masks
    try:
        from tqdm import tqdm

        imp_tqdm = True
    except ImportError:
        imp_tqdm = False
    if imp_tqdm:
        aperture_stats = [
            calc_aperture_mmm(data, mask, sigma_clip) for mask in tqdm(masks)
        ]
    else:
        aperture_stats = [calc_aperture_mmm(data, mask, sigma_clip) for mask in masks]

    aperture_stats = np.array(aperture_stats)

    # Place the array of the x y positions alongside the stats
    stacked = np.hstack([apertures.positions, aperture_stats])
    # Name the columns
    names = [
        "X",
        "Y",
        "aperture_mean",
        "aperture_median",
        "aperture_mode",
        "aperture_std",
        "aperture_area",
    ]
    # Make the table
    stats_tbl = Table(data=stacked, names=names)

    return stats_tbl


def calc_aperture_mmm(data, mask, sigma_clip):
    """Helper function to actually calculate the stats for pixels
    falling within some Photutils aperture mask on some array
    of data.
    """
    cutout = mask.cutout(data, fill_value=np.nan)
    if cutout is None:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    else:
        values = cutout * mask.data / mask.data
        values = values[~np.isnan(values)]
        if sigma_clip:
            values, clow, chigh = sigmaclip(values, low=3, high=3)

        mean = np.mean(values)
        median = np.median(values)
        std = np.std(values)

        mode = 3 * median - 2 * mean
        actual_area = (~np.isnan(values)).sum()
        return (mean, median, mode, std, actual_area)


def estimate_all_backgrounds(xs, ys, r_in, r_out, data, stat="aperture_mode"):
    """
    Compute sky values around (xs, ys) in data with various parameters

    See photometry_tools.aperture_stats_tbl for more details.
    """
    ans = CircularAnnulus(positions=zip(xs, ys), r_in=r_in, r_out=r_out)
    bg_ests = aperture_stats_tbl(apertures=ans, data=data, sigma_clip=True)
    return np.array(bg_ests[stat])
