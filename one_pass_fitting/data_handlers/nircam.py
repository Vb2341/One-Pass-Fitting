from astropy.io import fits
from jwst.datamodels import ImageModel

from .main_class import ImageHandler


class NIRCamHandler(ImageHandler):
    def __init__(self, image, catalog):
        with ImageModel(image) as mod:
            self.data = mod.data
            self.area = mod.area
            self.exptime = mod.meta.exposure.duration
            self._filter = mod.meta.instrument.filter
            self._pupil = mod.meta.instrument.pupil

            if 'CLEAR' not in self._pupil:
                self.filter = self._pupil
            else:
                self.filter = self._filter

            self.wcs = mod.meta.wcs
        self.pri_header = fits.getheader(image)
        self._sci_header = fits.getheader(image, 'SCI')

        self.exptime_corrected = True

        super().__init__(image, catalog)

    def ee_func(self):
        pass
