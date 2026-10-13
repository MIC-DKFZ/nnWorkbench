from pathlib import Path

import pytest

from nnworkbench.services.annotation_conversion import convert_annotations


def _manifest(path: Path, labels: str = '{"background": 0, "tumour": 1, "organ": 2, "ignore": 3}') -> None:
    path.write_text(f'{{"labels": {labels}, "file_ending": ".nii.gz"}}', encoding="utf-8")


def test_convert_annotations_shifts_classes_and_moves_ignore_to_last_id(tmp_path: Path) -> None:
    mitk = pytest.importorskip("mitk")
    np = pytest.importorskip("numpy")
    annotations = tmp_path / "annotationsTr"; annotations.mkdir()
    _manifest(tmp_path / "dataset.json")
    mitk.Image(np.array([[[0, 1], [2, 3]]], dtype=np.uint8), spacing=(1.1, 2.2, 3.3)).save(
        annotations / "case_001.nii.gz"
    )

    result = convert_annotations(tmp_path)

    converted = tmp_path / "labelsTr" / "case_001.nii.gz"
    assert result.converted == (converted,)
    assert not result.warnings
    assert np.array_equal(np.asarray(mitk.Image(converted)), np.array([[[3, 0], [1, 2]]], dtype=np.uint16))


def test_convert_annotations_skips_empty_and_out_of_range_cases(tmp_path: Path) -> None:
    mitk = pytest.importorskip("mitk")
    np = pytest.importorskip("numpy")
    annotations = tmp_path / "annotationsTr"; annotations.mkdir()
    _manifest(tmp_path / "dataset.json")
    mitk.Image(np.zeros((1, 2, 2), dtype=np.uint8)).save(annotations / "empty.nii.gz")
    mitk.Image(np.array([[[0, 9], [1, 0]]], dtype=np.uint8)).save(annotations / "invalid.nii.gz")
    mitk.Image(np.array([[[0, 1], [2, 3]]], dtype=np.uint8)).save(annotations / "valid.nii.gz")

    result = convert_annotations(tmp_path)

    assert result.converted == (tmp_path / "labelsTr" / "valid.nii.gz",)
    assert len(result.warnings) == 2
    assert not (tmp_path / "labelsTr" / "empty.nii.gz").exists()
    assert not (tmp_path / "labelsTr" / "invalid.nii.gz").exists()


def test_convert_annotations_requires_last_ignore_label_in_raw_manifest(tmp_path: Path) -> None:
    annotations = tmp_path / "annotationsTr"; annotations.mkdir()
    _manifest(tmp_path / "dataset.json", '{"background": 0, "ignore": 1, "tumour": 2}')

    with pytest.raises(ValueError, match="last"):
        convert_annotations(tmp_path)
