import numpy as np
from astropy.table import Table
from photutils import aperture_photometry

from .background_measurement import aperture_stats_tbl


def iraf_style_photometry(
        phot_apertures,
        bg_apertures,
        data,
        error_array=None,
        bg_method='mode',
        epadu=1.0):
    """Computes photometry with PhotUtils apertures, with IRAF formulae

    Parameters
    ----------
    phot_apertures : photutils PixelAperture object (or subclass)
        The PhotUtils apertures object to compute the photometry.
        i.e. the object returned via CirularAperture.
    bg_apertures : photutils PixelAperture object (or subclass)
        The phoutils aperture object to measure the background in.
        i.e. the object returned via CircularAnnulus.
    data : array
        The data for the image to be measured.
    error_array: array, optional
        The array of pixelwise error of the data.  If none, the
        Poisson noise term in the error computation will just be the
        square root of the flux/epadu. If not none, the
        aperture_sum_err column output by aperture_photometry
        (divided by epadu) will be used as the Poisson noise term.
    bg_method: {'mean', 'median', 'mode'}, optional
        The statistic used to calculate the background.
        All measurements are sigma clipped.
        NOTE: From DAOPHOT, mode = 3 * median - 2 * mean.
    epadu: float, optional
        Gain in electrons per adu (only use if image units aren't e-).

    Returns
    -------
    final_tbl : astropy.table.Table
        An astropy Table with the colums X, Y, flux, flux_error, mag,
        and mag_err measurements for each of the sources.

    """

    if bg_method not in ['mean', 'median', 'mode']:
        raise ValueError('Invalid background method, choose either \
                          mean, median, or mode')

    phot = aperture_photometry(data, phot_apertures, error=error_array)
    bg_phot = aperture_stats_tbl(data, bg_apertures, sigma_clip=True)

    if callable(phot_apertures.area):        # Handle photutils change
        ap_area = phot_apertures.area()
    else:
        ap_area = phot_apertures.area
    bg_method_name = 'aperture_{}'.format(bg_method)

    flux = phot['aperture_sum'] - bg_phot[bg_method_name] * ap_area

    # Need to use variance of the sources
    # for Poisson noise term in error computation.
    #
    # This means error needs to be squared.
    # If no error_array error = flux ** .5
    if error_array is not None:
        flux_error = compute_phot_error(phot['aperture_sum_err']**2.0,
                                        bg_phot, bg_method, ap_area,
                                        epadu)
    else:
        flux_error = compute_phot_error(flux, bg_phot,
                                        ap_area, epadu)

    mag = -2.5 * np.log10(flux)
    mag_err = 1.0857 * flux_error / flux

    # Make the final table
    X, Y = phot_apertures.positions.T
    stacked = np.stack([X, Y, flux, flux_error, mag, mag_err], axis=1)
    names = ['X', 'Y', 'flux', 'flux_error', 'mag', 'mag_error']

    final_tbl = Table(data=stacked, names=names)
    return final_tbl

def compute_phot_error(
        flux_variance,
        bg_phot,
        ap_area,
        epadu=1.0):
    """Computes the flux errors using the DAOPHOT style computation"""
    bg_variance_terms = (ap_area * bg_phot['aperture_std'] ** 2. ) \
                        * (1. + ap_area/bg_phot['aperture_area'])
    variance = flux_variance / epadu + bg_variance_terms
    flux_error = variance ** .5
    return flux_error
