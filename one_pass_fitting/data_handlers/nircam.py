from astropy.io import fits
from jwst.datamodels import ImageModel

from .main_class import ImageHandler


class NIRCamHandler(ImageHandler):
    """ImageHandler subclass for JWST NIRCam"""
    def __init__(self, image, catalog):
        if isinstance(image, str):
            self.name = image
            with ImageModel(image) as mod:
                self._read(mod)
        elif isinstance(image, ImageModel):
            self._read(image)
            self.name = image.meta.filename

        # self.pri_header = fits.getheader(image)
        # self._sci_header = fits.getheader(image, 'SCI')

        self.exptime_corrected = True

        super().__init__(catalog)

    def _read(self, mod):
        self.data = mod.data
        self.area = mod.area
        self.exptime = mod.meta.exposure.duration
        self._filter = mod.meta.instrument.filter
        self._pupil = mod.meta.instrument.pupil

        if 'CLEAR' not in self._pupil:
            self.filter = self._pupil
        else:
            self.filter = self._filter

        self.bunit = mod.meta.bunit_data
        self.pixelarea_steradians = mod.meta.photometry.pixelarea_steradians
        self.pivot = self.get_pivot()
        self.standard_aperture = 10.0 # THIS IS JUST A PLACEHOLDER, SHOULD NOT BE USED

        self.wcs = mod.meta.wcs

    def ee_func(self):
        pass
    
    def get_pivot(self):
        """Return pivot wavelength of self.filter in Angstroms"""
        # Pivot wavelength dictionary, in microns
        pivot_dict = {
            'F070W': 0.704,
            'F090W': 0.901,
            'F115W': 1.154,
            'F140M': 1.404,
            'F150W': 1.501,
            'F162M': 1.626,
            'F164N': 1.644,
            'F150W2': 1.671,
            'F182M': 1.845,
            'F187N': 1.874,
            'F200W': 1.99,
            'F210M': 2.093,
            'F212N': 2.12,
            'F250M': 2.503,
            'F277W': 2.786,
            'F300M': 2.996,
            'F322W2': 3.247,
            'F323N': 3.237,
            'F335M': 3.365,
            'F356W': 3.563,
            'F360M': 3.621,
            'F405N': 4.051,
            'F410M': 4.105,
            'F430M': 4.3,
            'F444W': 4.441,
            'F460M': 4.602,
            'F466N': 4.653,
            'F470N': 4.71
        }

        return pivot_dict[self.filter.upper()] * 1.0E4


