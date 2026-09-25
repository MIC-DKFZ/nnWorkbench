"""Recognize the image file formats nnU-Net can read, and strip their endings."""

from __future__ import annotations

from pathlib import Path

# Formats supported by nnU-Net's readers. Compound and longer endings come first,
# so ".nii.gz" wins over ".nii" and ".tiff" over ".tif".
IMAGE_SUFFIXES = (".nii.gz", ".nii", ".nrrd", ".mha", ".gipl", ".tiff", ".tif", ".png", ".bmp")


def image_suffix(name: str | Path) -> str | None:
    """Return the file's image ending as written in the name, or ``None`` if it is no image."""
    name = Path(name).name
    folded = name.casefold()
    for suffix in IMAGE_SUFFIXES:
        if folded.endswith(suffix):
            return name[-len(suffix):]
    return None


def has_image_suffix(name: str | Path) -> bool:
    return image_suffix(name) is not None


def strip_image_suffix(name: str | Path) -> str:
    """Remove an image ending (keeping compound endings intact), or the last suffix otherwise."""
    name = Path(name).name
    suffix = image_suffix(name)
    return name[:-len(suffix)] if suffix else Path(name).stem
