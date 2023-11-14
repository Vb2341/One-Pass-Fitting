import glob
import numpy as np

from astropy.table import Table
from photutils.aperture import CircularAperture, CircularAnnulus
from scipy.stats import sigmaclip

from ..aperture_photometry_utils import iraf_style_photometry


class ImageHandler:
    """
    Base class containing methods for corrections relevant for multiple detectors
    
    This is base class for working with images and their PSF fit catalogs.  It 
    serves as the parent class for subclasses specific to different instruments or 
    detectors.  It contains some base utilities that are to be used by many of the 
    subclasses.  It also does some basic cleaning of the catalogs, by removing rows 
    with invalid fluxes, or positions that aren't actually on the detector.  It also
    "updates" the calculated sky coordinates to make sure the RA/Dec coordinates are 
    consistent with the WCS of the data.
    """

    def __init__(self, catalog: Table):
        self.catalog = catalog
        if "f" not in self.catalog.colnames:
            self.catalog["f"] = 10.0 ** (self.catalog["m"] / -2.5)
        sy, sx = self.data.shape
        mask = (
            (self.catalog["x"] < 0)
            | (self.catalog["y"] < 0)
            | (self.catalog["x"] > sx)
            | (self.catalog["y"] > sy)
        )
        mask = mask | (np.isnan(self.catalog["m"])) | (self.catalog["q"] < 0.0)
        if np.sum(mask) > 0:
            print(
                f"removing {np.sum(mask)} sources for being off of frame or being bad fits"
            )
        self.catalog = self.catalog[~mask]
        self.update_sky_coords()

    def correct_catalog(
        self, aperture=False, pixel_area=False, encircled=False, **kwargs
    ):
        """
        Applies various corrections to catalogs, and calculates physical mags.

        Depending on the PSF model used when generating the catalogs, certain 
        corrections need to be applied in order to 1) remove systematic offsets 
        between fits from different images, and 2) photometrically calibrate them.
        For instance the if the PSF model is normalized at a radius of 5.5 pixels,
        then the fit fluxes do not account for the light outside of 5.5 pixels. This 
        then requires an aperture correction to calibrate out.  Furthermore, if combining 
        catalogs from different regions of a detector, then the flux may be off by a factor 
        of the pixel area.  Lastly, for HST and JWST, the units of the image (and thus the 
        measured flux from the PSF fit) are not physical flux units, and must be scaled 
        by the appropriate factors.  This function does these corrections.

        The first (optional) correction is the aperture correction.  This measures the 
        aperture photometric flux for good sources in the image at a larger radius, 
        and computes the correction factor between it and the PSF fit flux.  This is needed
        when using Jay Anderson's STDPSF models, as the tapering of these models to be 0 at 
        some radius causes some flux to be left out.  IF USING WEBBPSF MODELS THIS IS NOT NEEDED,
        SO ``aperture`` SHOULD REMAIN ``False``.  See `calculate_apcorr` for more info.

        The next (optional) correction is the pixel area correction.  This is necessary as 
        the flatfielding step of the HST pipeline corrects images to make surface brightness 
        flat across the image, which make point source fluxes show a spatial systematic.  
        Puzzlingly, this is not needed when using WebbPSF models either?  So keep `pixel_area` False for 
        JWST catalogs if using Webb PSF Models.  See `calculate_area_corr` for more info.

        The third (optional) correction is the encircled energy correction.  This is typically needed
        if doing the aperture correction above, as above a certain radius, it is difficult to measure the 
        aperture correction (depending on crowding).  This means the flux from the aperture correction radius 
        out to infinity is left out from the flux. This correction relies upon pre-tabulated encircled 
        energy tables, usually provided by the instrument team, and accounts for all the remaining flux 
        outside of the aperture correction radius.  Since these models are often unreliable at small radii,
        we must measure the aperture correction in the image out to some radius, and then apply this correction 
        to account for all the light. `encircled` SHOULD ALSO BE `False` IF USING WEBBPSF MODELS.  See 
        `calc_encircled_corr` for details.

        The last correction is the correction to ST and AB mag systems.  This is pretty much 
        just applying zeropoints and unit conversion.  Vegamag will come soon. See `convert_mags` for details.

        Parameters:
        -----------
        aperture : bool, optional
            Perform the intermediate aperture correction?  Default ``False``

        pixel_area : bool, optional
            Perform the pixel area correction?  Default ``False``

        encircled : bool, optional
            Perform the intermediate aperture to infinite aperture correction?  Default ``False``

        **kwargs : optional
            Keyword arguments to be passed to aperture correction.  See `calculate_apcorr` for details

        Returns:
        --------
        None
        """
        self._orig_catalog = self.catalog.copy()
        if aperture:
            self.calculate_apcorr(**kwargs)
            self.catalog["f"] *= self.ap_corr
        if pixel_area:
            self.calculate_area_corr()
            self.catalog["f"] *= self.area_corr
        if encircled:
            self.calc_encircled_corr()
            self.catalog["f"] *= self.ee_corr

        self.convert_mags()
        # In the case where "f" is in electrons or electrons/sec, this is an instrumental mag
        self.catalog["m"] = -2.5 * np.log10(self.catalog["f"])

    def calculate_apcorr(
        self, radius: float = None, sky_in=None, sky_out=None, sky_stat="mode"
    ):
        """
        Calculate the aperture correction for photometric measurements.

        This method calculates the scalar offset between the PSF fit fluxes and aperture photometry
        measurements at the provided aperture/annulus radii.  This is necessary to calibrate the
        PSF fluxes.  It is recommended that ``radius`` be equal to an aperture where the PSF and
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
        ------
        - This method calculates the aperture correction by comparing the instrumental magnitudes to
        calibrated magnitudes of stars in the catalog.
        - The computed aperture correction is stored in the object's `ap_corr` attribute.


        Example:
        --------
        # Create an ``ImageHandler`` object and calculate the aperture correction
        img_hand = ImageHandler(image, catalog)
        img_hand.calculate_apcorr(radius=1.0, sky_stat="mode")
        # The aperture correction is now available as `img_phot.ap_corr`.
        """
        # Set default values for sky annulus if not provided
        if radius is None:
            radius = self.standard_aperture
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
        delta = self.catalog["f"] / phot_tbl["flux"]

        # Define quality metrics for filtering
        nonzero_q = self.catalog["q"] > 0
        q_perc = np.nanpercentile(self.catalog["q"][nonzero_q], 20)
        qmask = nonzero_q & (self.catalog["q"] < q_perc)
        ap_merr_perc = np.nanpercentile(phot_tbl["mag_error"][nonzero_q], 20)
        ap_mask = phot_tbl["mag_error"] < ap_merr_perc

        # Apply quality masks and perform sigma clipping
        mask = qmask & ap_mask
        clip = sigmaclip(delta[mask])[0]
        print(np.nanmedian(clip))

        # Compute the aperture correction
        ap_corr = clip
        print(
            f"Computed aperture correction of {ap_corr} using {len(clip)} stars for {self.name}"
        )

        # Store the computed aperture correction in the object
        self.ap_corr = ap_corr

    def calculate_area_corr(self):
        """For each position in catalog, return the correction factor for pixel area"""
        intx = (self.catalog["x"] + 0.5).astype(int).data
        inty = (self.catalog["y"] + 0.5).astype(int).data

        self.area_corr = self.area[inty, intx]

    def calc_encircled_corr(self):
        # TODO: Implement this properly
        self.ee_corr = 1.0

    def update_sky_coords(self):
        """Updates the RA and Dec columns in ``self.catalog`` to be consistent with ``self.wcs``"""
        x = self.catalog["x"]
        y = self.catalog["y"]
        r, d = self.wcs.pixel_to_world_values(x, y)
        self.catalog["RA"] = r
        self.catalog["Dec"] = d

    def convert_mags(self):
        """
        Converts flux values in the catalog to ABMag and STMag.

        This method applies scaling factors and zeropoints to convert the fluxes measured in the image
        to ABMag and STMag units. It handles various starting units for HST and JWST data and performs
        necessary calculations to obtain the final magnitudes.

        Notes
        -----
        - For HST data with units in ELECTRONS, the method corrects for exposure time and applies inverse sensitivity factors to convert flux to STMag, then converts to ABMag
        - For HST data with units in ELECTRONS/s, the method applies inverse sensitivity factors to convert flux to STMag, then converts to ABMag
        - For JWST data with units in MJy/sr, the method calculates first ABMag using pixel area (in steradians) to convert to Jy, then to ABMag, and then to STMag.
        - For unit conversion definitions see `here https://www.stsci.edu/hst/instrumentation/acs/data-analysis/zeropoints#section-c5a9e052-dcf3-4b57-b783-5c66578b479c`

        Warning
        -------
        This method modifies the 'ST' and 'AB' columns in the catalog in-place.
        """
        # Do all of this with Quantities and units and your life will probably
        # be so much easier
        if "ELECTRONS" in self.bunit.upper():
            # HST Cases
            if not self.exptime_corrected:
                self.catalog["f"] /= self.exptime
                self.exptime_corrected = True
            tmp_flam = self.catalog["f"] * self.sensitivity
            st = -2.5 * np.log10(tmp_flam) - 21.1
            ab = st + 21.1 - 5.0 * np.log10(self.pivot) - 2.408

        elif self.bunit == "MJy/sr":
            # JWST Cases
            tmp_fnu = self.catalog["f"] * self.pixelarea_steradians * 1.0e6
            ab = -2.5 * np.log10(tmp_fnu / 3631.0)
            st = ab + 5 * np.log10(self.pivot) + 2.408 - 21.1

        self.catalog["ST"] = st
        self.catalog["AB"] = ab


def find_catalogs(image: str) -> list[Table]:
    """
    For a given FITS image, return a list of Astropy Tables representing catalogs for each
    science extension.

    Parameters
    ----------
    image : str
        The path to the FITS image file.

    Returns
    -------
    list of astropy.table.Table
        A list containing Astropy Tables, each representing a catalog for a science extension
        in the input image.  Each of the returned tables also contains the "sci_ext" key in the 
        table's metadata dictionary, which is an integer corresponding to the science extension of 
        the catalog.

    Notes
    -----
    The function looks for catalog files associated with the input image by searching for files
    with names following the pattern ``{image_root}_sci?_xyrd.cat`` or
    ``{image_root}_sci?_xyrd*.ecsv``, where ``?`` represents a single character wildcard and {image_root} is 
    the name of the image with the ``.fits`` extension removed from the name.
    `DO NOT` have both a ``.cat`` and ``.ecsv`` file for the same catalog in the directory as the image 
    as this could potentially give quite confusing results.

    Examples
    --------
    >>> image_path = 'path/to/image1.fits'
    >>> catalogs = find_catalogs(image_path)
    >>> for catalog in catalogs:
    ...     print(catalog)
    >>> # prints "path/to/image1_sci1_xyrd.cat" (or .ecsv for JWST)
    >>> # "path/to/image1_sci2_xyrd.cat" (If image is an HST file with multiple SCI exts)
    """
    cat_root = image.replace(".fits", "")
    cat_str = f"{cat_root}_sci?_xyrd.cat"
    cats = sorted(glob.glob(cat_str))
    if len(cats) == 0:
        cats = sorted(glob.glob(f"{cat_root}_sci?_xyrd*.ecsv"))

    tbls = []
    for tfile in cats:
        tbls.append(_read_tbl(tfile))
    return tbls


def _read_tbl(tbl_path):
    """Reads in Table from a file"""
    if tbl_path.endswith(".ecsv"):
        t = Table.read(tbl_path)
    elif tbl_path.endswith(".cat"):
        t = Table.read(tbl_path, format="ascii.commented_header")
    sci_strs = [sub for sub in tbl_path.split("_") if sub.startswith("sci")]
    if len(sci_strs)==0:
        print(f'WARNING: Cannot determine sci extension for {tbl_path}, assuming it is for "SCI,1"')
        sci_ext = 1
    else:
        sci_ext = int(sci_strs[-1].replace("sci", ""))

    t.meta["sci_ext"] = sci_ext
    return t


def _read_and_check_tbls(tbls):
    """Reads in the tables and makes sure the ``sci_ext`` metadata value is populated"""
    output_tbls = []
    if isinstance(tbls, list):
        for tbl in tbls:
            if isinstance(tbl, str):
                output_tbls.append(_read_tbl(tbl))
            elif isinstance(tbl, Table):
                output_tbls.append(tbl)
    elif isinstance(tbls, str):
        output_tbls.append(_read_tbl(tbls))
    elif isinstance(tbls, Table):
        output_tbls.append(tbls)

    for ot in output_tbls:
        if "sci_ext" not in ot.meta:
            if len(output_tbls) == 1:
                print('"sci_ext" not detected in table metadata, assuming its 1')
                ot.meta["sci_ext"] = 1
            else:
                raise KeyError(
                    '"sci_ext" not found in table metadata, and multiple tables were found for a single image.  Please set the tbl.meta["sci_ext"] equal to the index of the science extension for each catalog'
                )
        else:
            continue
    return output_tbls