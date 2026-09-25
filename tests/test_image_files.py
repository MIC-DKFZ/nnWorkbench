from pathlib import Path

from nnworkbench.services.image_files import has_image_suffix, image_suffix, strip_image_suffix


def test_image_suffix_recognizes_nnunet_formats() -> None:
    assert image_suffix("Case001_0000.nii.gz") == ".nii.gz"
    assert image_suffix(Path("labels/Case001.TIFF")) == ".TIFF"
    assert image_suffix("Case001_0000.png") == ".png"
    assert image_suffix("class_distribution.json") is None
    assert has_image_suffix("Case001.nrrd")
    assert not has_image_suffix("Case001.jpg")


def test_strip_image_suffix_keeps_compound_endings_intact() -> None:
    assert strip_image_suffix("Case001_0000.nii.gz") == "Case001_0000"
    assert strip_image_suffix("Case.001_0000.tif") == "Case.001_0000"
    assert strip_image_suffix("points.json") == "points"
