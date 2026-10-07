"""Tests for the read-only dataset status used by the stage pages."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from nnworkbench.services.commands import prediction_command, preprocessing_command, training_command
from nnworkbench.services.label_mapping import annotation_to_label, label_to_annotation
from nnworkbench.services.workflow_status import (
    fold_info,
    ignore_label_id,
    label_mode,
    models_overview,
    plans_overview,
    prediction_status,
    raw_overview,
    validation_cases,
)


def test_label_mode_follows_ignore_label() -> None:
    assert label_mode({"labels": {"background": 0, "cell": 1, "ignore": 2}}) == "sparse"
    assert ignore_label_id({"labels": {"background": 0, "cell": 1, "ignore": 2}}) == 2
    assert label_mode({"labels": {"background": 0, "cell": 1}}) == "dense"
    assert label_mode(None) == "dense"


def test_raw_overview_matches_cases_channels_and_labels(tmp_path: Path) -> None:
    images = tmp_path / "imagesTr"; labels = tmp_path / "labelsTr"
    images.mkdir(); labels.mkdir()
    for name in ("a_0000.nii.gz", "a_0001.nii.gz", "b_0000.nii.gz"):
        (images / name).touch()
    (labels / "a.nii.gz").touch()

    overview = raw_overview(tmp_path, {"channel_names": {"0": "T2", "1": "ADC"}})

    assert [case.case_id for case in overview.cases] == ["a", "b"]
    assert overview.cases[0].channels_present and overview.cases[0].has_label
    assert not overview.cases[1].channels_present and not overview.cases[1].has_label
    assert overview.missing_labels == 1 and overview.names_valid


def test_label_mapping_round_trip_sparse_and_dense() -> None:
    np = pytest.importorskip("numpy")
    label = np.array([0, 1, 2, 3])  # 3 = ignore
    annotation = label_to_annotation(label, 3)
    assert annotation.tolist() == [1, 2, 3, 0]
    assert annotation_to_label(annotation, 3).tolist() == [0, 1, 2, 3]
    assert annotation_to_label(np.zeros(3), 3).tolist() == [3, 3, 3]  # new sparse label: all ignore
    assert label_to_annotation(np.array([0, 2]), None).tolist() == [0, 2]
    assert annotation_to_label(np.zeros(3), None).tolist() == [0, 0, 0]  # new dense label: background


def test_plans_overview_reads_configurations_and_outdated(tmp_path: Path) -> None:
    raw = tmp_path / "raw"; (raw / "labelsTr").mkdir(parents=True)
    pre = tmp_path / "pre"; pre.mkdir()
    (pre / "nnUNetPlans.json").write_text(json.dumps({
        "plans_name": "nnUNetPlans", "experiment_planner_used": "ExperimentPlanner",
        "configurations": {"2d": {"data_identifier": "nnUNetPlans_2d", "patch_size": [64, 64], "spacing": [1, 1],
                                  "batch_size": 12},
                           "3d_fullres": {"data_identifier": "nnUNetPlans_3d_fullres"}}}), encoding="utf-8")
    (pre / "nnUNetPlans_2d").mkdir()
    for name in ("a.b2nd", "a_seg.b2nd", "b.b2nd"):
        (pre / "nnUNetPlans_2d" / name).touch()
    old = time.time() - 100
    os.utime(pre / "nnUNetPlans_2d", (old, old))
    (raw / "labelsTr" / "a.nii.gz").touch()

    plans, = plans_overview(pre, raw)

    assert plans.name == "nnUNetPlans" and plans.planner == "ExperimentPlanner"
    assert [(config.name, config.preprocessed_cases) for config in plans.configurations] == [("2d", 2), ("3d_fullres", 0)]
    assert plans.outdated


def test_fold_info_status_dice_and_models(tmp_path: Path) -> None:
    model = tmp_path / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    done = model / "fold_0"; running = model / "fold_1"; stopped = model / "fold_2"
    for fold in (done, running, stopped):
        fold.mkdir(parents=True)
    (done / "checkpoint_final.pth").touch(); (done / "checkpoint_best.pth").touch()
    (done / "validation").mkdir()
    (done / "validation" / "summary.json").write_text(json.dumps({"foreground_mean": {"Dice": 0.8}}), encoding="utf-8")
    log = "2026-01-01 10:00:00: Epoch 41 \n2026-01-01 10:00:01: Pseudo dice [np.float32(0.5), np.float32(0.7)] \n"
    (running / "training_log_1.txt").write_text(log, encoding="utf-8")
    (stopped / "training_log_1.txt").write_text(log, encoding="utf-8")
    old = time.time() - 3600
    os.utime(stopped / "training_log_1.txt", (old, old))
    (running / "debug.json").write_text(json.dumps({"num_epochs": "1000"}), encoding="utf-8")

    assert fold_info(done).status == "done" and fold_info(done).dice == pytest.approx(0.8)
    info = fold_info(running)
    assert info.status == "training" and info.epoch == 41 and info.total_epochs == 1000
    assert info.pseudo_dice == pytest.approx(0.6)
    assert fold_info(stopped).status == "stopped"
    found, = models_overview(tmp_path)
    assert (found.trainer, found.plans, found.configuration) == ("nnUNetTrainer", "nnUNetPlans", "3d_fullres")
    assert [fold.name for fold in found.finished_folds] == ["0"] and found.mean_dice == pytest.approx(0.8)


def test_validation_cases_mean_and_weakest_class() -> None:
    summary = {"metric_per_case": [{"reference_file": "/gt/case_1.nii.gz", "prediction_file": "/p/case_1.nii.gz",
                                     "metrics": {"1": {"Dice": 0.9}, "2": {"Dice": 0.5}}}]}
    case, = validation_cases(summary, {1: "cell", 2: "nucleus"})
    assert case.case_id == "case_1" and case.mean_dice == pytest.approx(0.7)
    assert case.weakest == ("nucleus", 0.5)


def test_prediction_status(tmp_path: Path) -> None:
    source = tmp_path / "in"; target = tmp_path / "out"; source.mkdir(); target.mkdir()
    (source / "a_0000.nii.gz").touch(); (source / "b_0000.nii.gz").touch(); (target / "a.nii.gz").touch()
    assert prediction_status(source, target) == [("a", True, True), ("b", True, False)]


def test_extended_command_options() -> None:
    assert preprocessing_command(10, ["3d_fullres", "2d"], "nnUNetPlannerResEncL", processes=8) == (
        "nnUNetv2_plan_and_preprocess -d 10 -c 3d_fullres 2d -pl nnUNetPlannerResEncL -np 8 --verify_dataset_integrity")
    assert "--verify_dataset_integrity" not in preprocessing_command(10, "2d", "P", verify=False)
    assert training_command(10, "nnUNetPlans", "2d", "T", "3", continue_training=True).endswith("-p nnUNetPlans --c")
    assert prediction_command(10, "/i", "/o", "P", "2d", folds="0 1", disable_tta=True).endswith("-f 0 1 --disable_tta")
