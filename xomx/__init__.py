from . import plotting as pl
from . import tools as tl
from . import data_importation as di
from . import classifiers as cl
from . import embeddings as em
from . import feature_selection as fs

__version__ = "0.1.16"  # -version-

import sys

sys.modules.update(
    {
        f"{__name__}.{m}": globals()[m]
        for m in [
            "pl",
            "tl",
            "di",
            "cl",
            "em",
            "fs",
        ]
    }
)
