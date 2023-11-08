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
    __call__(data, data_wcs=None):
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

    def __post_init__(self, bkg_stat):
        """Extra checks to make sure the values are valid"""
        if bkg_stat not in ["mean", "median", "mode"]:
            raise ValueError("bkg_stat must be either mean, median or mode")
        if self.sky_out <= self.sky_in:
            raise ValueError("sky_out must be greater than sky_in")

    def __call__(self, data, data_wcs=None):
        """
        Perform one-pass photometry on the input image.

        Parameters:
        -----------
        data : array-like
            The image data on which to perform photometry.

        data_wcs : WCS, optional
            World Coordinate System information for the input data (default is None).

        Returns:
        --------
        astropy.table.Table
            A table containing the photometric measurements of the detected stars.


        Notes:
        ------
        This method performs one-pass photometry on the input image using the provided PSF model.
        It detects stars in the image, fits the PSF model to the stars, and returns a table with
        photometric measurements.

        If `data_wcs` is provided, the method will also calculate the celestial coordinates (RA and Dec)
        of the detected stars and include them in the output table.
        """
        self.xdets, self.ydets = detect_peaks(data, self.hmin, self.fmin, self.pmax)
        output_tbl = self.fit_stars(data, self.xdets, self.ydets)
        if data_wcs:
            r, d = data_wcs.to_pixel(output_tbl["x"], output_tbl["y"])
            output_tbl["RA"] = r
            output_tbl["Dec"] = d
        return output_tbl

    def fit_stars(self, data, xs=None, ys=None, mod=None):
        """
        Fits a model to a set of stars in parallel using multiple processes.

        Parameters:
        -----------
        data : array-like
            The image data from which to extract the cutouts of the star
        xs : array-like
            The x positions of the stars.  Only needs to fall within central (brightest) pixel of star
        ys : array-like
            The y positions of the stars.  Only needs to fall within central (brightest) pixel of star
        mod : GriddedPSFModel, EPSFModel, FittableImageModel
            The PSF model to fit to the stars.  Should be GriddedPSFModel,

        Returns:
        --------
        astropy.table.Table
            A table of the fitted parameters for each star, include columns:
                - x : x positions of fitted stars
                - y : y positions of fitted stars
                - m : instrumental magnitude of fitted stars
                - q : quality of fit value (qfit)
                - s : sky values measured around the stars
                - cx : central excess values
                - f : flux of fitted stars
        """
        if mod is None:
            mod = self.mod
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
        return tbl
