# segab_yolo 🚀 AGPL-3.0 License - https://segab_yolo.com/license

__version__ = "8.4.52"

import sys
import importlib
import os
from typing import TYPE_CHECKING

# Shim: allow old checkpoints referencing 'ultralytics.*' to resolve via 'segab_yolo.*'
sys.modules.setdefault("ultralytics", sys.modules[__name__])

# Set ENV variables (place before imports)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training

from segab_yolo.utils import ASSETS, SETTINGS
from segab_yolo.utils.checks import check_yolo as checks
from segab_yolo.utils.downloads import download

settings = SETTINGS

MODELS = ("YOLO", "YOLOWorld", "YOLOE", "NAS", "SAM", "FastSAM", "RTDETR")

__all__ = (
    "__version__",
    "ASSETS",
    *MODELS,
    "checks",
    "download",
    "settings",
)

if TYPE_CHECKING:
    # Enable hints for type checkers
    from segab_yolo.models import YOLO, YOLOWorld, YOLOE, NAS, SAM, FastSAM, RTDETR  # noqa


def __getattr__(name: str):
    """Lazy-import model classes on first access."""
    if name in MODELS:
        return getattr(importlib.import_module("segab_yolo.models"), name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    """Extend dir() to include lazily available model names for IDE autocompletion."""
    return sorted(set(globals()) | set(MODELS))


if __name__ == "__main__":
    print(__version__)
