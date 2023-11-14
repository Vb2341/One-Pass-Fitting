from astropy.io import fits
from jwst.datamodels import ImageModel

from .main_class import ImageHandler


class MIRIHandler(ImageHandler):
    """ImageHandler subclass for JWST MIRI"""
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
        self.filter = mod.meta.instrument.filter

        self.bunit = mod.meta.bunit_data
        self.pixelarea_steradians = mod.meta.photometry.pixelarea_steradians
        self.pivot = self.get_pivot()
        self.standard_aperture = 10. # THIS IS JUST A PLACEHOLDER
        self.wcs = mod.meta.wcs

    def ee_func(self):
        pass
    
    def get_pivot(self):
        """Return pivot wavelength of self.filter in Angstroms"""
        # Contains pivot wavelength in microns
        pivot_dict = {
            'F560W': 5.635,
            'F770W': 7.639,
            'F1000W': 9.953,
            'F1130W': 11.309,
            'F1280W': 12.810,
            'F1500W': 15.064,
            'F1800W': 17.984,
            'F2100W': 20.795,
            'F2550W': 25.365,
            'F2550WR': 25.365,
            'FND': 12.900,
            'Opaque': 'N/A'
        }
        return pivot_dict[self.filter.upper()]*1.0E4