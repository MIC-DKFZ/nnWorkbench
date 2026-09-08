from pathlib import Path

import pytest

from nnworkbench.services import image_stack


def test_image_files_are_filtered_and_naturally_ordered(tmp_path: Path) -> None:
    for name in ("slice_10.png", "slice_2.png", "slice_1.png", "notes.txt"):
        (tmp_path / name).touch()

    assert [path.name for path in image_stack.image_files(tmp_path, "png")] == [
        "slice_1.png", "slice_2.png", "slice_10.png"
    ]
    assert image_stack.default_output_path(tmp_path) == tmp_path.parent / f"{tmp_path.name}.nii.gz"
    assert image_stack.default_output_path(tmp_path, 2) == tmp_path.parent / f"{tmp_path.name}_scale2.nii.gz"


def test_stack_image_folder_downsamples_and_updates_spacing(tmp_path: Path, monkeypatch) -> None:
    np = pytest.importorskip("numpy")
    slices = tmp_path / "slices"; slices.mkdir()
    paths = [slices / f"slice_{index}.tif" for index in range(3)]
    for path in paths:
        path.touch()
    arrays = {path: np.full((4, 4), index, dtype=np.uint16) for index, path in enumerate(paths)}
    written: dict[str, object] = {}
    monkeypatch.setattr(image_stack, "_read_image_slice", lambda path: arrays[path])
    monkeypatch.setattr(image_stack, "_write_volume", lambda volume, output, spacing: written.update(
        volume=volume.copy(), output=output, spacing=spacing))

    output = image_stack.stack_image_folder(
        slices, file_type=".tif", workers=1, scale_factor=2, spacing=(0.5, 1.0, 2.0), quiet=False,
    )

    assert output == slices.parent / "slices_scale2.nii.gz"
    assert written["output"] == output
    assert written["spacing"] == (1.0, 2.0, 4.0)
    assert written["volume"].shape == (2, 2, 2)
    assert np.array_equal(written["volume"][:, 0, 0], [0, 2])


def test_stack_image_folder_rejects_inconsistent_slices(tmp_path: Path, monkeypatch) -> None:
    np = pytest.importorskip("numpy")
    slices = tmp_path / "slices"; slices.mkdir()
    first, second = slices / "slice_1.tif", slices / "slice_2.tif"
    first.touch(); second.touch()
    arrays = {first: np.zeros((2, 2), dtype=np.uint8), second: np.zeros((3, 2), dtype=np.uint8)}
    monkeypatch.setattr(image_stack, "_read_image_slice", lambda path: arrays[path])

    with pytest.raises(ValueError, match="shape"):
        image_stack.stack_image_folder(slices, file_type=".tif", workers=1, quiet=False)


def test_stack_image_folder_keeps_default_spacing_when_downsampled(tmp_path: Path, monkeypatch) -> None:
    np = pytest.importorskip("numpy")
    slices = tmp_path / "slices"; slices.mkdir()
    paths = [slices / f"slice_{index}.png" for index in range(2)]
    for path in paths:
        path.touch()
    monkeypatch.setattr(image_stack, "_read_image_slice",
                        lambda path: np.zeros((4, 4), dtype=np.uint8))
    written: dict[str, object] = {}
    monkeypatch.setattr(image_stack, "_write_volume",
                        lambda volume, output, spacing: written.update(spacing=spacing))

    image_stack.stack_image_folder(slices, file_type=".png", workers=1, scale_factor=2, quiet=False)

    assert written["spacing"] == (1.0, 1.0, 1.0)
