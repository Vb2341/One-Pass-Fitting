"""
This is the main class for running one pass photometry.
"""
from dataclasses import dataclass
from functools import partial

import numpy as np
import tqdm
from astropy.table import Table
from photutils.psf import GriddedPSFModel

from .background_measurement import estimate_all_backgrounds
from .detection import detect_peaks
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
    __call__(data, data_wcs=None, output_name=None):
        Perform one-pass photometry on the input image.

    fit_stars(data, xs=None, ys=None, mod=None):
        Fit the PSF model to detected stars.

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

    def __call__(self, data, data_wcs=None, output_name=None):
        """
        Perform one-pass photometry on the input image.

        Parameters:
        -----------
        data : array-like
            The image data on which to perform photometry.

        data_wcs : WCS, optional
            World Coordinate System information for the input data (default is ``None``).

        output_name : str, optional
            Output name for writing catalog to file.  For JWST data, ``.ecsv`` is recommended,
            for HST, ``.cat``is recommended (saved as ascii.commented_header).  If ``None`` no
            file is written out.

        Returns:
        --------
        astropy.table.Table
            A table containing the photometric measurements of the detected stars.  See ``fit_stars`` for details.


        Notes:
        ------
        This method performs one-pass photometry on the input image using the provided PSF model.
        It detects stars in the image, fits the PSF model to the stars, and returns a table with
        photometric measurements.

        If `data_wcs` is provided, the method will also calculate the celestial coordinates (RA and Dec)
        of the detected stars and include them in the output table.

        If output_name is not ``None``, it is strongly recommeneded to name the catalog to have the same name
        as the image filename, with the ``.fits`` file extension replaced with ``_sci<X>_xyrd.cat`` for HST
        or ``_sci<X>_xyrd.ecsv`` for JWST, where <X> is the EXTVER of the relevant SCI extension, e.g.
        the ``output_name`` for extension ``SCI, 2`` of ``iaab01hxq_flc.fits`` (an HST image) should be
        ``iaab01hxq_flc_sci2_xyrd.cat``.
        """

        self.xdets, self.ydets = detect_peaks(data, self.hmin, self.fmin, self.pmax)
        output_tbl = self.fit_stars(
            data, self.xdets, self.ydets, data_wcs=data_wcs, output_name=output_name
        )

        return output_tbl

    def fit_stars(
        self, data, xs=None, ys=None, mod=None, data_wcs=None, output_name=None
    ):
        """
        Fits PSF model to objects in ``data`` located at (``xs``,``ys``).

        Parameters:
        -----------
        data : array-like
            The image data from which to extract the cutouts of the star

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

        skies = estimate_all_backgrounds(
            xs, ys, self.sky_in, self.sky_out, data, stat=f"aperture_{self.bkg_stat}"
        )

        # Flatten the model to make it easier to pass to the fitting function
        # Create a partial function that includes the flattened model and the image data
        fit_func = partial(fit_star, model=mod, im_data=data)
        # Combine the x, y, and sky values into a single list of tuples
        xy_sky = list(zip(xs, ys, skies))

        # The partial is useful when using multiprocessing to parallelize this loop.
        # Add multiprocessing later
        result = []
        for xys in tqdm.tqdm(xy_sky):
            result.append(fit_func(*xys))

        # Convert the list of results to a numpy array and create an astropy table with column names
        result = np.array(result)
        tbl = Table(result, names=["f", "x", "y", "q", "cx"])
        # Add the sky values and convert the flux to magnitude
        tbl["s"] = skies
        tbl["m"] = -2.5 * np.log10(tbl["f"])
        # Select only the columns we care about and return the table
        tbl = tbl["x", "y", "m", "q", "s", "cx", "f"]

        if output_name:
            if output_name.endswith(".txt") or output_name.endswith(".cat"):
                fmt = "ascii.commented_header"
            elif output_name.endswith(".ecsv"):
                fmt = None

        if data_wcs:
            r, d = data_wcs.pixel_to_world_values(tbl["x"], tbl["y"])
            tbl["RA"] = r
            tbl["Dec"] = d
        if output_name:
            tbl.write(output_name, overwrite=True, format=fmt)

        return tbl
