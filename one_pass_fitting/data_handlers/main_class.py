import numpy as np

from astropy.table import Table
from photutils.aperture import CircularAperture, CircularAnnulus
from scipy.stats import sigmaclip

from ..aperture_photometry_utils import iraf_style_photometry


class ImageHandler:
    """Base class containing methods for corrections relevant for all detectors"""

    def __init__(self, image: str, catalog: Table):
        self.name = image
        self.catalog = catalog
        if "f" not in catalog.colnames:
            catalog["f"] = 10.0 ** (catalog["m"] / -2.5)
        self.update_sky_coords()

    def correct_catalog(
        self, aperture=True, pixel_area=True, encircled=False, **kwargs
    ):
        self._orig_catalog = self.catalog.copy()
        if aperture:
            self.calculate_apcorr(**kwargs)
            self.catalog["f"] *= self.ap_corr
        if pixel_area:
            self.area_corr()
            self.catalog["f"] *= self.area_corr
        if encircled:
            self.encircled_corr()
            self.catalog["f"] *= self.ee_corr

        if not self.exptime_corrected:
            self.catalog["f"] /= self.exptime
        self.catalog["m"] = -2.5 * np.log10(self.catalog["f"])

    def calculate_apcorr(self, radius: float = 5.0, sky_in=None, sky_out=None, sky_stat="mode"):
        """
        Calculate the aperture correction for photometric measurements.

        This method calculates the scalar offset between the PSF fit fluxes and aperture photometry 
        measurements at the provided aperture/annulus radii.  This is necessary to calibrate the 
        PSF fluxes.  It is recommended that `radius` be equal to an aperture where the PSF and 
        thus encircled energy are known to be stable, e.g. 0.4" for WFC3

        Parameters:
        -----------
        radius : float, optional
            The radius of the circular aperture used for photometry.

        sky_in : float or None, optional
            The inner radius of the annulus for sky background estimation. If None, it is set to 2 * `radius`.

        sky_out : float or None, optional
            The outer radius of the annulus for sky background estimation. If None, it is set to 3 * `radius`.

        sky_stat : str, optional
            The statistic to use for sky background estimation, one of "mean", "median" or "mode".

        Returns:
        --------
        None

        Notes:
        -----
        - This method calculates the aperture correction by comparing the instrumental magnitudes to
        calibrated magnitudes of stars in the catalog.
        - The computed aperture correction is stored in the object's `ap_corr` attribute.

        Example:
        --------
        # Create an ImagePhotometry object and calculate the aperture correction
        img_phot = ImagePhotometry(image_data, catalog)
        img_phot.calculate_apcorr(radius=5.0, sky_stat="mode")
        # The aperture correction is now available as `img_phot.ap_corr`.
        """
        # Set default values for sky annulus if not provided
        if sky_in is None:
            sky_in = 2 * radius
        if sky_out is None:
            sky_out = 3 * radius

        # Extract object coordinates
        x = self.catalog["x"]
        y = self.catalog["y"]

        # Create circular aperture and annulus
        ap = CircularAperture(zip(x, y), radius)
        anns = CircularAnnulus(zip(x, y), r_in=sky_in, r_out=sky_out)

        # Perform aperture photometry with specified sky statistic
        phot_tbl = iraf_style_photometry(ap, anns, self.data, bg_method=sky_stat)

        # Calculate the magnitude difference between catalog and photometry
        delta = self.catalog["m"] - phot_tbl["mag"]

        # Define quality metrics for filtering
        nonzero_q = self.catalog["q"] > 0
        q_perc = np.nanpercentile(self.catalog["q"][nonzero_q], 20)
        qmask = nonzero_q & (self.catalog["q"] < q_perc)
        ap_merr_perc = np.nanpercentile(phot_tbl["mag_error"][nonzero_q], 20)
        ap_mask = phot_tbl["mag_error"] < ap_merr_perc

        # Apply quality masks and perform sigma clipping
        mask = qmask & ap_mask
        clip = sigmaclip(delta[mask])[0]

        # Compute the aperture correction
        ap_corr = 10.0 ** (np.nanmedian(clip) / 2.5)
        print(
            f"Computed aperture correction of {ap_corr} using {len(clip)} stars for {self.name}"
        )

        # Store the computed aperture correction in the object
        self.ap_corr = ap_corr


    def area_corr(self):
        """For each position in catalog, return the correction factor for pixel area"""
        intx = (self.catalog["x"] + 0.5).astype(int).data
        inty = (self.catalog["y"] + 0.5).astype(int).data

        self.area_corr = self.area[inty, intx]

    def encircled_corr(self):
        # TODO: Implement this properly
        self.ee_corr = 1.0

    def update_sky_coords(self):
        """Updates the RA and Dec columns in self.catalog to be consistent with self.wcs"""
        x = self.catalog["x"]
        y = self.catalog["y"]
        r, d = self.wcs.pixel_to_world_values(x, y)
        self.catalog["RA"] = r
        self.catalog["Dec"] = d
