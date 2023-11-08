import numpy as np
from astropy.io import fits
from astropy.modeling import models
from astropy.wcs import WCS

from .main_class import ImageHandler


class WFC3UVISHandler(ImageHandler):
    def __init__(self, image, catalog):
        self.sci_ext = catalog.meta['sci_ext']
        hdu = fits.open(image)
        self.data = hdu["SCI", self.sci_ext].data
        self.pri_header = hdu[0].header
        self._sci_header = hdu["SCI", self.sci_ext].header

        self.filter = self.pri_header["FILTER"]
        self.exptime = self.pri_header["exptime"]
        self.sensitivity = self.pri_header["photflam"]

        self.bunit = self._sci_header["bunit"]
        self._ccdchip = self._sci_header["ccdchip"]
        self.wcs = WCS(self._sci_header, hdu)
        self.make_area_array()
        hdu.close()

        self.exptime_corrected = False

        super().__init__(image, catalog)

    def make_area_array(self):
        _pamfunc = get_pam_func(f"uvis{self._ccdchip}")
        area = np.zeros(self.data.shape)
        pixy, pixx = np.mgrid[: area.shape[0], : area.shape[1]]
        pixx -= int(self._sci_header["LTV1"])
        pixy -= int(self._sci_header["LTV2"])
        self.area = _pamfunc(pixx, pixy)

    def ee_func(self):
        pass


class WFC3IRHandler(ImageHandler):
    def __init__(self, image, catalog):
        hdu = fits.open(image)
        self.data = hdu["SCI"].data
        self.pri_header = hdu[0].header
        self._sci_header = hdu["SCI"].header

        self.filter = self.pri_header["FILTER"]
        self.exptime = self.pri_header["exptime"]
        self.sensitivity = self.pri_header["photflam"]

        self.bunit = self._sci_header["bunit"]
        self.wcs = WCS(self._sci_header, hdu)
        self.make_area_array()
        hdu.close()

        self.exptime_corrected = True

        super().__init__(image, catalog)

    def make_area_array(self):
        _pamfunc = get_pam_func("ir")
        area = np.zeros(self.data.shape)
        pixy, pixx = np.mgrid[: area.shape[0], : area.shape[1]]
        pixx -= int(self._sci_header["LTV1"])
        pixy -= int(self._sci_header["LTV2"])
        self.area = _pamfunc(pixx, pixy)

    def ee_func(self):
        pass


def get_pam_func(detchip):
    degrees = {"ir": 2, "uvis1": 3, "uvis2": 3}

    # Store polynomial coefficient values for each chip
    coeff_dict = {}
    coeff_dict["ir"] = [
        9.53791038e-01,
        -3.68634734e-07,
        -3.14690506e-10,
        8.27064384e-05,
        1.48205135e-09,
        2.12429722e-10,
    ]
    coeff_dict["uvis1"] = [
        9.83045965e-01,
        8.41184852e-06,
        1.59378242e-11,
        -2.02027686e-20,
        -8.69187898e-06,
        1.65696133e-11,
        1.31974097e-17,
        -3.86520105e-11,
        -5.92106744e-17,
        -9.87825173e-17,
    ]
    coeff_dict["uvis2"] = [
        1.00082580e00,
        8.66150267e-06,
        1.61942281e-11,
        -1.01349112e-20,
        -9.07898503e-06,
        1.70183371e-11,
        4.10618989e-18,
        -8.02421371e-11,
        3.54901127e-16,
        -1.01400681e-16,
    ]

    coeffs = coeff_dict[detchip.lower()]
    pam_func = models.Polynomial2D(degree=degrees[detchip.lower()])

    pam_func.parameters = coeffs
    return pam_func
