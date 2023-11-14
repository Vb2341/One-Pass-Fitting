# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *  # noqa
# ----------------------------------------------------------------------------

__all__ = []

from .psf_photometry import OnePassPhot
from .catalog_merging import *
# from .data_handlers import (
#     WFC3UVISHandler,
#     WFC3IRHandler,
#     NIRCamHandler,
#     MIRIHandler,
#     ImageHandler,
# )

from . import catalog_merging

# Then you can be explicit to control what ends up in the namespace,
__all__ += ["OnePassPhot"]  # noqa
__all__ += catalog_merging.__all__
# __all__ += [
#     "WFC3UVISHandler",
#     "WFC3IRHandler",
#     "NIRCamHandler",
#     "MIRIHandler",
#     "ImageHandler",
# ]
# or you can keep everything from the subpackage with the following instead
# __all__ += example_mod.__all__
