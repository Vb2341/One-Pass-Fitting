from astropy.io import fits
from jwst.datamodels import ImageModel

from .main_class import ImageHandler


class MIRIHandler(ImageHandler):
    def __init__(self, image, catalog):
        with ImageModel(image) as mod:
            self.data = mod.data
            self.area = mod.area
            self.exptime = mod.meta.exposure.duration
            self.filter = mod.meta.instrument.filter

            self.wcs = mod.meta.wcs

        self.pri_header = fits.getheader(image)
        self._sci_header = fits.getheader(image, 'SCI')

        self.exptime_corrected = True

        super().__init__(image, catalog)

    def ee_func(self):
        pass
