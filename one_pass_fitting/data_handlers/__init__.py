"""
This subpackage contains the classes that are used to set up the images/catalogs for 
merging (matching/averaging).

To extend this suite of tools for other instruments, subclass the `ImageHandler` class.
More documentation regarding this subclassing coming soon.
"""
from .wfc3 import WFC3UVISHandler, WFC3IRHandler
from .miri import MIRIHandler
from .nircam import NIRCamHandler
from .main_class import ImageHandler, find_catalogs, _read_and_check_tbls

__all__ = [
    "ImageHandler",
    "WFC3UVISHandler",
    "WFC3IRHandler",
    "NIRCamHandler",
    "MIRIHandler",
    "find_catalogs"
]
