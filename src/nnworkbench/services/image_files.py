"""Recognize the image file formats nnU-Net can read, and strip their endings."""

from __future__ import annotations

import os
from pathlib import Path

# Formats supported by nnU-Net's readers. Compound and longer endings come first,
# so ".nii.gz" wins over ".nii" and ".tiff" over ".tif".
IMAGE_SUFFIXES = (".nii.gz", ".nii", ".nrrd", ".mha", ".gipl", ".tiff", ".tif", ".png", ".bmp")


def image_suffix(name: str | Path) -> str | None:
    """Return the file's image ending as written in the name, or ``None`` if it is no image."""
    name = os.path.basename(os.fspath(name))  # far cheaper than Path(name).name, called once per file
    folded = name.casefold()
    for suffix in IMAGE_SUFFIXES:
        if folded.endswith(suffix):
            return name[-len(suffix):]
    return None


def has_image_suffix(name: str | Path) -> bool:
    return image_suffix(name) is not None


def strip_image_suffix(name: str | Path) -> str:
    """Remove an image ending (keeping compound endings intact), or the last suffix otherwise."""
    name = os.path.basename(os.fspath(name))
    suffix = image_suffix(name)
    return name[:-len(suffix)] if suffix else Path(name).stem


# nnU-Net reads these 2D formats with NaturalImage2DIO: a colour file (RGB/RGBA) gives 3 or 4 channels
NATURAL_IMAGE_SUFFIXES = (".png", ".bmp", ".tif")


def color_components(path: str | Path) -> int | None:
    """The channels nnU-Net's 2D reader gets from one image file (1 grey, 3 RGB, 4 RGBA; 2 for grey + alpha,
    which nnU-Net refuses), from the header only. None when the file is no 2D natural image or cannot be read."""
    suffix = image_suffix(path)
    if suffix is None or suffix.lower() not in NATURAL_IMAGE_SUFFIXES:
        return None
    try:
        from PIL import Image

        with Image.open(path) as image:
            if image.mode == "P":  # scikit-image (nnU-Net's reader) turns a colour palette into RGB(A)
                palette = image.getpalette() or []
                grey = all(palette[index] == palette[index + 1] == palette[index + 2]
                           for index in range(0, len(palette) - 2, 3))
                return 1 if grey else 4 if "transparency" in image.info else 3
            return len(image.getbands())
    except Exception:  # noqa: BLE001  (any unreadable or unknown file: no information)
        return None
