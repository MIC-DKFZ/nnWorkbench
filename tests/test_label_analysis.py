import json
from pathlib import Path

import pytest

from nnworkbench.services.label_analysis import (
    aggregate_class_counts,
    analyse_label_distribution,
    analysis_path,
    labeled_data_percentage,
    label_validity,
    load_label_analysis,
    undefined_label_ids,
)


def test_aggregate_class_counts_includes_declared_and_encountered_classes() -> None:
    classes = aggregate_class_counts([{0: 15, 1: 4}, {0: 7, 2: 3}, {1: 0}], {0: "background", 1: "organ"})

    assert classes == [
        {"id": 0, "name": "background", "images": 2, "voxels": 22},
        {"id": 1, "name": "organ", "images": 1, "voxels": 4},
        {"id": 2, "name": "Class 2", "images": 1, "voxels": 3},
    ]


def test_aggregate_class_counts_excludes_the_configured_ignore_label() -> None:
    classes = aggregate_class_counts([{0: 2, 1: 4, 2: 8}],
                                     {0: "background", 1: "tumour", 2: "ignore"}, {2})

    assert classes == [
        {"id": 0, "name": "background", "images": 1, "voxels": 2},
        {"id": 1, "name": "tumour", "images": 1, "voxels": 4},
    ]


def test_undefined_label_ids_reports_each_affected_file() -> None:
    result = undefined_label_ids(
        [Path("case_001.nii.gz"), Path("case_002.nii.gz")],
        [{0: 10, 1: 2, 7: 1}, {0: 5, 9: 3}],
        {0, 1, 2},
    )

    assert result == {"case_001.nii.gz": [7], "case_002.nii.gz": [9]}


def test_label_validity_accumulates_empty_missing_image_and_undefined_id_findings() -> None:
    assert label_validity({0: 10, 2: 4}, Path("case_001.nii.gz"), set(), {0}, {2}, set()) == [
        "Empty (background/ignore only)",
        "No matching imagesTr file",
    ]
    assert label_validity({0: 10, 7: 1}, Path("case_001.nii.gz"), set(), {0}, {2}, {7}) == [
        "No matching imagesTr file",
        "Undefined label IDs: 7",
    ]


def test_labeled_data_percentage_excludes_ignore_voxels() -> None:
    assert labeled_data_percentage({0: 5, 1: 3, 2: 2}, {2}) == 80.0
    assert labeled_data_percentage({0: 5, 1: 3}, set()) == 100.0


def test_load_label_analysis_reads_cached_raw_file(tmp_path: Path) -> None:
    analysis_path(tmp_path).write_text(json.dumps({"classes": []}), encoding="utf-8")

    assert load_label_analysis(tmp_path) == {"classes": []}


def test_label_analysis_counts_images_voxels_and_writes_cache(tmp_path: Path) -> None:
    mitk = pytest.importorskip("mitk")
    np = pytest.importorskip("numpy")
    labels = tmp_path / "labelsTr"; labels.mkdir()
    (tmp_path / "dataset.json").write_text(
        '{"labels": {"background": 0, "tumour": 1, "ignore": 2}}', encoding="utf-8"
    )
    mitk.Image(np.array([[[0, 1], [1, 2]]], dtype=np.uint8)).save(labels / "case_001.nii.gz")
    mitk.Image(np.array([[[0, 1], [2, 0]]], dtype=np.uint8)).save(labels / "case_002.nii.gz")

    progress: list[str] = []
    analysis = analyse_label_distribution(
        tmp_path, workers=1, progress=progress.append, include_labeled_data=True,
    )

    assert analysis["files_analyzed"] == 2
    assert analysis["classes"] == [
        {"id": 0, "name": "background", "images": 2, "voxels": 3},
        {"id": 1, "name": "tumour", "images": 2, "voxels": 3},
    ]
    assert analysis["per_image"] == [
        {
            "image": "case_001.nii.gz",
            "voxels": {"0": 1, "1": 2},
            "shape": [2, 2, 1],
            "spacing": [1.0, 1.0, 1.0],
            "labeled_data_percentage": 75.0,
            "validity": ["No matching imagesTr file"],
        },
        {
            "image": "case_002.nii.gz",
            "voxels": {"0": 2, "1": 1},
            "shape": [2, 2, 1],
            "spacing": [1.0, 1.0, 1.0],
            "labeled_data_percentage": 75.0,
            "validity": ["No matching imagesTr file"],
        },
    ]
    assert analysis["undefined_label_ids"] == {}
    assert load_label_analysis(tmp_path) == analysis
    assert progress == [
        "Analyzed label 1/2: case_001.nii.gz",
        "Analyzed label 2/2: case_002.nii.gz",
    ]
