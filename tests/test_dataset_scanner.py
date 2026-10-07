from pathlib import Path

import pytest

from nnworkbench.services.dataset_creator import (
    create_dataset,
    create_next_iteration,
    next_free_dataset_id,
    next_iteration_name,
)
from nnworkbench.services.dataset_scanner import scan_datasets


def test_scanner_merges_optional_roots(tmp_path: Path):
    raw, preprocessed, results = (tmp_path / name for name in ("raw", "preprocessed", "results"))
    for root in (raw, preprocessed, results): root.mkdir()
    (raw / "Dataset001_Liver").mkdir()
    (preprocessed / "Dataset001_Liver").mkdir()
    (results / "Dataset002_Kidney").mkdir()
    records = scan_datasets(raw, preprocessed, results)
    assert [(record.dataset_id, bool(record.raw_path), bool(record.preprocessed_path), bool(record.results_path))
            for record in records] == [(1, True, True, False), (2, False, False, True)]


def test_creator_rejects_id_present_in_another_root(tmp_path: Path):
    raw, results = tmp_path / "raw", tmp_path / "results"
    raw.mkdir(); results.mkdir(); (results / "Dataset007_Existing").mkdir()
    with pytest.raises(ValueError, match="already exists"):
        create_dataset(raw, [raw, None, results], 7, "New")


def test_creator_creates_raw_structure(tmp_path: Path):
    raw = tmp_path / "raw"; raw.mkdir()
    created = create_dataset(raw, [raw, None, None], 3, "My task")
    assert created.name == "Dataset003_My_task"
    assert all((created / directory).is_dir() for directory in ("imagesTr", "labelsTr"))
    assert not (created / "imagesTs").exists()
    assert (created / "dataset.json").is_file()


def test_next_iteration_copies_images_labels_and_manifest(tmp_path: Path):
    raw = tmp_path / "raw"; raw.mkdir()
    source = raw / "Dataset203_Project_iter1"
    for folder, filename in (("imagesTr", "case_001_0000.nii.gz"), ("labelsTr", "case_001.nii.gz"),
                             ("predTr", "case_001.nii.gz")):
        directory = source / folder; directory.mkdir(parents=True)
        (directory / filename).touch()
    (source / "dataset.json").write_text('{"labels": {"background": 0}}', encoding="utf-8")

    assert next_iteration_name("Project") == "Project_iter1"
    assert next_iteration_name("Project_iter1") == "Project_iter2"
    assert next_free_dataset_id(204, [raw, None, None]) == 204
    created = create_next_iteration(raw, [raw, None, None], source, 204, "Project_iter2")

    assert created.name == "Dataset204_Project_iter2"
    assert (created / "imagesTr" / "case_001_0000.nii.gz").is_file()
    assert (created / "labelsTr" / "case_001.nii.gz").is_file()
    assert (created / "dataset.json").read_text(encoding="utf-8") == '{"labels": {"background": 0}}'
    assert not (created / "predTr").exists()
