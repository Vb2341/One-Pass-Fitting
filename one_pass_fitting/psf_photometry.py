"""
This is the main class for running one pass photometry.
"""
from dataclasses import dataclass
from functools import partial

import numpy as np
import tqdm
from astropy.table import Table, vstack
from photutils.psf import GriddedPSFModel
from scipy.spatial import cKDTree

from .background_measurement import estimate_all_backgrounds
from .detection import detect_peaks, detect_sat_jwst, compute_nan_areas
from .psf_model_fitting import fit_star


@dataclass
class OnePassPhot:
    """
    Perform one-pass photometry on an image using a PSF model.

    This class is designed for performing photometry on an astronomical image using a provided PSF model.
    It includes methods for detecting stars, fitting the PSF model to the stars, and calculating photometric
    parameters. It can operate in parallel for improved performance.

    Parameters:
    -----------
    psf_model : GriddedPSFModel
        The PSF model to use for photometry.

    hmin : int, optional
        Minimum separation between stars for detection (default is 5 pixels).

    fmin : float, optional
        Minimum flux value for a star to be considered (default is 1000.0).

    pmax : float, optional
        Maximum pixel value for a star to be considered (default is 70000.0).

    bkg_stat : str, optional
        Background estimation method (default is "mode"). Supported values: "mean," "median," or "mode."

    sky_in : float, optional
        Inner radius for sky background estimation (default is 8.5 pixels).

    sky_out : float, optional
        Outer radius for sky background estimation (default is 13.5 pixels).

    Methods:
    --------
    __call__(data, data_wcs=None, output_name=None, do_sat=False, dq=None):
        Perform one-pass photometry on the input image.

    fit_stars(data, xs=None, ys=None, mod=None):
        Fit the PSF model to detected stars.

    sat_phot(data, dq, mod=None, data_wcs=None, output_name=None):
        Detect and fit the saturated sources

    Attributes:
    -----------
    xdets : array-like
        X positions of detected stars.

    ydets : array-like
        Y positions of detected stars.
    """

    psf_model: GriddedPSFModel
    hmin: int = 5
    fmin: float = 1000.0
    pmax: float = 70000.0
    bkg_stat: str = "mode"
    sky_in: float = 8.5
    sky_out: float = 13.5

    def __post_init__(self):
        """Extra checks to make sure the values are valid"""
        if self.bkg_stat not in ["mean", "median", "mode"]:
            raise ValueError("bkg_stat must be either mean, median or mode")
        if self.sky_out <= self.sky_in:
            raise ValueError("sky_out must be greater than sky_in")

    def __call__(self, data, data_wcs=None, output_name=None, do_sat=False, dq=None):
        """
        Perform one-pass photometry on the input image.

        Parameters:
        -----------
        data : numpy.ndarray
            The image data on which to perform photometry.

        data_wcs : WCS, optional
            World Coordinate System information for the input data (default is ``None``).

        output_name : str, optional
            Output name for writing catalog to file.  For JWST data, ``.ecsv`` is recommended,
            for HST, ``.cat`` is recommended (saved as ascii.commented_header).  If ``None`` no
            file is written out.

        do_sat : bool, optional
            Find and measure the saturated stars as well (default is ``False``). Currently only well
            implemented for JWST

        dq : numpy.ndarray
            The data quality array corresponding to ``data``.  Must be passed in if `do_sat` is ``True``
        Returns:
        --------
        astropy.table.Table
            A table containing the photometric measurements of the detected stars.  See ``fit_stars`` for details.

        Notes:
        ------
        This method performs one-pass photometry on the input image using the provided PSF model.
        It detects stars in the image, fits the PSF model to the stars, and returns a table with
        photometric measurements.  If `do_sat` is ``True`` then it also finds and detects the saturated sources (currently
        a beta version that works for just JWST). The data quality array must be passed in the `dq` argument if
        `do_sat` is ``True``.

        If `data_wcs` is provided, the method will also calculate the celestial coordinates (RA and Dec)
        of the detected stars and include them in the output table.

        If output_name is not ``None``, it is strongly recommeneded to name the catalog to have the same name
        as the image filename, with the ``.fits`` file extension replaced with ``_sci<X>_xyrd.cat`` for HST
        or ``_sci<X>_xyrd.ecsv`` for JWST, where <X> is the EXTVER of the relevant SCI extension, e.g.
        the ``output_name`` for extension ``SCI, 2`` of ``iaab01hxq_flc.fits`` (an HST image) should be
        ``iaab01hxq_flc_sci2_xyrd.cat``.

        """

        self.xdets, self.ydets = detect_peaks(data, self.hmin, self.fmin, self.pmax)
        if not do_sat:
            output_tbl = self.fit_stars(
                data, self.xdets, self.ydets, data_wcs=data_wcs, output_name=output_name
            )
        else:
            if dq is None:
                raise ValueError(
                    "dq array must be provided for saturated star photometry"
                )
            unsat_tbl = self.fit_stars(
                data, self.xdets, self.ydets, data_wcs=data_wcs, output_name=None
            )
            sat_tbl = self.sat_phot(data, dq, data_wcs=data_wcs, output_name=None)
            if (len(unsat_tbl) == 0) or (len(sat_tbl) == 0):
                output_tbl = vstack([unsat_tbl, sat_tbl])
            else:
                output_tbl = self._merge_unsat_sat(unsat_tbl, sat_tbl)

            self.write_tbl(output_tbl, output_name=output_name)

        return output_tbl

    def fit_stars(
        self, data, xs=None, ys=None, mod=None, data_wcs=None, output_name=None
    ):
        """
        Fits PSF model to objects in ``data`` located at (``xs``,``ys``).

        Parameters:
        -----------
        data : numpy.ndarray
            Data array from which to measure stars

        xs : array-like, optional
            The x positions of the stars.  Only needs to fall within central (brightest) pixel of star.  If ``None``, sources are detected first.

        ys : array-like, optional
            The y positions of the stars.  Only needs to fall within central (brightest) pixel of star

        mod : GriddedPSFModel, EPSFModel, FittableImageModel, optional
            The PSF model to fit to the stars.  Should usually be GriddedPSFModel.  If ``None`` (default), then sets value to ``self.psf_model``.

        data_wcs : WCS, optional
            World Coordinate System information for the input data (default is ``None``).

        output_name : str, optional
            Output name for writing catalog to file.  For JWST data, ``.ecsv`` is recommended,
            for HST, ``.cat``is recommended (saved as ascii.commented_header).  If ``None`` no
            file is written out.

        Returns:
        --------
        astropy.table.Table
            A table of the fitted parameters for each star, include columns:
                - ``x`` : x positions of fitted stars (0 indexed)
                - ``y`` : y positions of fitted stars (0 indexed)
                - ``m`` : instrumental magnitude of fitted stars
                - ``q`` : quality of fit value (qfit)
                - ``s`` : sky values measured around the stars
                - ``cx`` : central excess values
                - ``f`` : flux of fitted stars
                - ``pix_flux`` : Sum of the model on the pixels on which it was fit
                - ``npix_fit`` : Number of pixels fit by the model
                - ``sat`` : Whether the core of the star was saturated (nan) or not
                - ``r`` : RA of fit position (only if data_wcs is not None)
                - ``d`` : Dec of fit position (only if data_wcs is not None)

        """
        if mod is None:
            mod = self.psf_model
        if xs is None:
            if not hasattr(self, "xdets"):
                self.xdets, self.ydets = detect_peaks(
                    data, self.hmin, self.fmin, self.pmax
                )
            xs = self.xdets
            ys = self.ydets

        if len(xs) == 0:
            print("WARNING: No stars detected to measure!")
            return Table()

        skies = estimate_all_backgrounds(
            xs,
            ys,
            self.sky_in,
            self.sky_out,
            data,
            stat=f"aperture_{self.bkg_stat}",
            progress_bar=False,
        )

        # Flatten the model to make it easier to pass to the fitting function
        # Create a partial function that includes the flattened model and the image data
        fit_func = partial(fit_star, model=mod, im_data=data)
        # Combine the x, y, and sky values into a single list of tuples
        xy_sky = list(zip(xs, ys, skies))

        # The partial is useful when using multiprocessing to parallelize this loop.
        # Add multiprocessing later
        print("Fitting stars")
        result = []
        for xys in tqdm.tqdm(xy_sky):
            result.append(fit_func(*xys))

        tbl = self.compile_results(
            result, skies, data_wcs=data_wcs, output_name=output_name
        )
        return tbl

    def sat_phot(self, data, dq, xs=None, ys=None, mod=None, data_wcs=None, output_name=None):
        """
        Perform photometry on saturated stars detected in the data.

        Parameters
        ----------
        data : numpy.ndarray
            Data array from which to measure stars
        dq : numpy.ndarray
            Data quality array indicating pixel flags, needed for finding saturated stars.
        xs : array-like, optional
            The x positions of the stars.  Only needs to fall within central block of nans of star.  If ``None``, sources are detected first.
        ys : array-like, optional
            The y positions of the stars.  Only needs to fall within central block of nans of star.  If ``None``, sources are detected first.
        areas: array-like, optional
            How many nans in block.  Used to compute fitting cutout size.  Only used if `xs` and `ys` are also passed in.  If ``None`` will attempt to compute size.
        mod : GriddedPSFModel, EPSFModel, FittableImageModel, optional
            The PSF model to fit to the stars.  Should usually be GriddedPSFModel.  If ``None`` (default), then sets value to ``self.psf_model``.
        data_wcs : WCS, optional
            World Coordinate System information for the input data (default is ``None``).
        output_name : str, optional
            Output name for writing catalog to file.  For JWST data, ``.ecsv`` is recommended,
            for HST, ``.cat``is recommended (saved as ascii.commented_header).  If ``None`` no
            file is written out.

        Notes:
        ------
        This function uses the data quality array to detect blocks of pixels that have flags
        indicating there is a saturated object present.  It then determines an approximate center
        of the object, calculates an appropriate size of cutout to fit, calculates backgrounds and then
        fits the unsaturated wings of the PSF.  For more information, see ``fit_stars()``.

        WARNING: This is a very early version of this function, and has only been tested on a couple of cases.
        In addition, it is only meant to work for JWST imaging data at the moment
        """
        if mod is None:
            mod = self.psf_model

        if xs is None:
            seg_tbl = detect_sat_jwst(dq, distance_factor=2.0)
            self.sat_xdets = seg_tbl["xcentroid"]
            self.sat_ydets = seg_tbl["ycentroid"]
            approx_sat_rad = np.array(seg_tbl["area"].value ** 0.5)

            xs = self.sat_xdets
            ys = self.sat_ydets
        
        else:
            if (ys is None) or len(ys) != len(xs):
                raise ValueError('xs and ys must be the same size.')
            areas = compute_nan_areas(xs, ys, data)
            approx_sat_rad = np.sqrt(areas)
            
        approx_sat_rad[approx_sat_rad<2.5] = 3

        if len(xs) < 1:
            print("WARNING: No saturated stars measured!")
            return Table()
        
        # max_rad = np.nanmax(approx_sat_rad)
        skies = estimate_all_backgrounds(
            xs,
            ys,
            approx_sat_rad * 2.5,
            approx_sat_rad * 3.5,
            data,
            stat=f"aperture_{self.bkg_stat}",
            progress_bar=False,
        )

        # The partial is useful when using multiprocessing to parallelize this loop.
        # Add multiprocessing later
        print("Fitting saturated stars")
        result = []
        xyskyrad = list(zip(xs, ys, skies, approx_sat_rad))
        for x, y, sky, approx_rad in tqdm.tqdm(xyskyrad):
            size = max(int(approx_rad) * 2 + 1, 5)
            result.append(
                fit_star(x, y, sky, model=mod, im_data=data, fit_shape=(size, size))
            )

        tbl = self.compile_results(
            result, skies, data_wcs=data_wcs, output_name=output_name
        )
        return tbl

    def compile_results(self, result, skies, data_wcs=None, output_name=None):
        """
        Compile photometry results into a table and potentially save it to a file.

        Parameters
        ----------
        result : list or numpy.ndarray
            List or array containing photometry results.
        skies : list or numpy.ndarray
            List or array containing sky values.
        data_wcs : astropy.wcs.WCS, optional
            WCS information for the data.
        output_name : str, optional
            Name of the output file to save the results.

        Returns
        -------
        tbl : astropy.table.Table
            Compiled table containing selected columns of photometry results.

        Notes
        -----
        This method compiles the input 'result' into an astropy table, adding sky values ('skies'),
        converting flux ('f') to magnitude ('m').  The order for the values for each sublist in `result` must be:
        flux, x positon, y postion, q fit, cx, pix flux, and finally npix_fit (see ``fit_stars`` for more info)
        Additionally, it creates a column 'sat' to mark stars with NaNs at the peak but not as a failed fit.
        If 'data_wcs' is provided, RA and Dec columns are added to the table based on pixel values.
        If 'output_name' is provided, the resulting table can be saved to a file using the 'write_tbl' method.

        """
        # Convert the list of results to a numpy array and create an astropy table with column names
        result = np.array(result)
        tbl = Table(result, names=["f", "x", "y", "q", "cx", "pix_flux", "npix_fit"])
        # Add the sky values and convert the flux to magnitude
        tbl["s"] = skies
        tbl["m"] = -2.5 * np.log10(tbl["f"])
        # Select only the columns we care about and return the table
        tbl = tbl["x", "y", "m", "q", "s", "cx", "f", "pix_flux", "npix_fit"]
        # This is a quick and dirty way to determine if the star had nans at the peak,
        # but wasn't just a failed fit
        tbl["sat"] = (np.isnan(tbl["cx"]) & ~np.isnan(tbl["m"])).astype(float)

        if data_wcs:
            r, d = data_wcs.pixel_to_world_values(tbl["x"], tbl["y"])
            tbl["RA"] = r
            tbl["Dec"] = d

        if output_name:
            self.write_tbl(tbl, output_name=output_name)

        return tbl

    def write_tbl(self, tbl, output_name):
        """Writes table to file, handles the different formats"""
        if output_name.endswith(".txt") or output_name.endswith(".cat"):
            fmt = "ascii.commented_header"
        elif output_name.endswith(".ecsv"):
            fmt = None

        tbl.write(output_name, overwrite=True, format=fmt)

    def _merge_unsat_sat(self, unsat_tbl, sat_tbl):
        """
        Eliminate duplicates that get fit in the saturated table, merge saturated and unsaturated tables
        """
        tree = cKDTree(np.array([unsat_tbl["x"], unsat_tbl["y"]]).T)
        dist, idx = tree.query(np.array([sat_tbl["x"], sat_tbl["y"]]).T)
        distmask = dist < 1.0

        return vstack([unsat_tbl, sat_tbl[~distmask]])
