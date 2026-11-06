"""Runs can be configured and their command copied when the model or the preprocessed data is not on this
computer (e.g. on a cluster): Predict with a model or folds that are not here, Train with plans that are not here."""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest
from test_stress_stages import _Env, _model, _preprocessed, _raw


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    environment = _Env(tmp_path, monkeypatch)
    yield environment
    environment.close()


def _arguments(predict) -> list[str]:
    return shlex.split(predict._command.command())


def test_without_a_local_model_the_fields_hold_the_defaults_and_only_the_command_is_available(env, tmp_path: Path) -> None:
    _raw(env.root)
    source = env.root / "input"
    source.mkdir()
    (source / "x_0000.nii.gz").touch()
    env.open()
    predict = env.page.pages["predict"]
    assert not predict._models_empty.isHidden() and predict._models_table.isHidden()
    assert predict._fields() == ("nnUNetTrainer", "nnUNetResEncUNetLPlans", "3d_fullres")  # as Train would preselect
    assert predict._folds.selected() == ["0", "1", "2", "3", "4"]  # the cross-validation folds, "all" offered too
    predict._input.setText(str(source))
    predict._output.setText(str(env.root / "out"))
    predict._folders_changed()
    env.wait()
    arguments = _arguments(predict)
    assert arguments[0] == "nnUNetv2_predict"
    assert arguments[arguments.index("-tr") + 1] == "nnUNetTrainer"
    assert arguments[arguments.index("-p") + 1] == "nnUNetResEncUNetLPlans"
    assert arguments[arguments.index("-f") + 1:arguments.index("-f") + 6] == ["0", "1", "2", "3", "4"]
    assert not predict._predict.isEnabled() and "not in nnUNet_results on this computer" in predict._busy_hint.text()
    # a typed model is used as typed (e.g. a custom trainer on the cluster)
    predict._trainer.setCurrentText("nnUNetTrainer_250epochs")
    predict._configuration.setCurrentText("2d")
    arguments = _arguments(predict)
    assert arguments[arguments.index("-tr") + 1] == "nnUNetTrainer_250epochs"
    assert arguments[arguments.index("-c") + 1] == "2d"
    predict._trainer.setCurrentText("")
    assert predict._command.command() == "" and "Enter the trainer" in predict._busy_hint.text()


def test_a_local_model_fills_the_fields_and_typing_another_one_keeps_the_command(env, tmp_path: Path) -> None:
    _raw(env.root)
    _preprocessed(env.root)
    _model(env.root, folds=("0", "1"))
    source = env.root / "input"
    source.mkdir()
    (source / "x_0000.nii.gz").touch()
    env.open()
    predict = env.page.pages["predict"]
    predict._input.setText(str(source))
    predict._output.setText(str(env.root / "out"))
    predict._folders_changed()
    env.wait()
    assert predict._fields() == ("nnUNetTrainer", "nnUNetPlans", "3d_fullres")  # the selected local model
    assert predict._folds.selected() == ["0", "1"] and predict._predict.isEnabled()
    # folds that are not finished here can be chosen for a run elsewhere (e.g. a test run of fold 0 on a laptop,
    # the other folds on the cluster): the command has them, running here is refused
    assert predict._folds.options() == ["0", "1", "2", "3", "4", "all"]
    predict._folds.set_selected(["0", "1", "2"])
    assert "-f 0 1 2" in predict._command.command()
    assert not predict._predict.isEnabled() and "Fold 2 is not finished" in predict._busy_hint.text()
    predict._folds.set_selected(["0", "1"])
    assert predict._predict.isEnabled()
    predict._plans.setCurrentText("nnUNetResEncUNetLPlans")  # not trained here: the table selection goes
    assert predict._current_model() is None and not predict._models_table.selectionModel().selectedRows()
    assert not predict._predict.isEnabled() and "not in nnUNet_results" in predict._busy_hint.text()
    assert "-p nnUNetResEncUNetLPlans" in predict._command.command()
    assert predict._folds.selected() == ["0", "1", "2", "3", "4"]
    predict._plans.setCurrentText("nnUNetPlans")  # back to the local model: selected again, its folds
    assert predict._current_model() is not None and predict._models_table.selectionModel().selectedRows()
    assert predict._folds.selected() == ["0", "1"] and predict._predict.isEnabled()
    # the model chosen on Train is followed even when it is not trained here
    predict.follow_model("nnUNetTrainerBetterIgnoreSampling__nnUNetResEncUNetMPlans__3d_lowres")
    assert predict._fields() == ("nnUNetTrainerBetterIgnoreSampling", "nnUNetResEncUNetMPlans", "3d_lowres")
    assert predict._current_model() is None


def test_train_with_plans_that_are_not_preprocessed_here_only_copies(env, tmp_path: Path) -> None:
    _raw(env.root)
    env.open()
    train = env.page.pages["train"]
    train._open_defaults()  # nothing preprocessed here: the default plans and configuration, run disabled
    assert train._plans.currentText() == "nnUNetResEncUNetLPlans"
    assert train._configuration.currentText() == "3d_fullres"
    assert "-p nnUNetResEncUNetLPlans" in train._command.command() and " 3d_fullres 0 " in train._command.command()
    assert not train._start.isEnabled() and "not preprocessed on this computer" in train._busy_hint.text()
    assert train._busy_hint.property("tone") == "orange"
    assert train._plans.findText("nnUNetPlans") >= 0  # the plans of every standard planner can be chosen
    train._configuration.setCurrentText("2d")
    assert " 2d 0 " in train._command.command() and not train._start.isEnabled()
    train._run_dialog.close()


def test_train_with_local_plans_offers_the_other_plans_for_copying(env, tmp_path: Path) -> None:
    _raw(env.root, cases=("a", "b", "c", "d", "e"))
    _preprocessed(env.root, cases=5)
    env.open()
    train = env.page.pages["train"]
    train._open_defaults()
    assert train._plans.currentText() == "nnUNetPlans" and train._start.isEnabled()  # preprocessed here
    assert train._plans_hint.text() == ""
    train._plans.setCurrentText("nnUNetResEncUNetLPlans")  # preprocessed on the cluster only
    assert train._configuration.currentText() == "3d_fullres"
    assert not train._start.isEnabled() and "nnUNetResEncUNetLPlans are not preprocessed" in train._busy_hint.text()
    assert "preprocessed here: nnUNetPlans" in train._plans_hint.text()
    train._plans.setCurrentText("nnUNetPlans")
    assert train._start.isEnabled() and train._busy_hint.property("tone") != "orange"
    train._run_dialog.close()


def test_a_trainer_or_planner_that_is_not_installed_only_copies(env, tmp_path: Path) -> None:
    from nnworkbench.services.nnunet_discovery import NnunetClasses

    _raw(env.root, cases=("a", "b", "c", "d", "e"))
    _preprocessed(env.root, cases=5)
    _model(env.root, folds=("0",))
    source = env.root / "input"
    source.mkdir()
    (source / "x_0000.nii.gz").touch()
    env.open()
    # while the installed classes are unknown (or could not be listed), a typed name is not checked
    assert env.context.nnunet_classes is not None and not env.context.nnunet_classes.complete
    train = env.page.pages["train"]
    train._open_defaults()
    train._trainer.setCurrentText("nnUNetTrainer_250epochs")
    assert train._start.isEnabled()
    env.context._classes_discovered(NnunetClasses(("ExperimentPlanner", "nnUNetPlannerResEncL"),
                                                  ("nnUNetTrainer", "nnUNetTrainerBetterIgnoreSampling"), complete=True))
    train.update_actions()
    assert not train._start.isEnabled() and "nnUNetTrainer_250epochs is not installed" in train._busy_hint.text()
    assert "-tr nnUNetTrainer_250epochs" in train._command.command()  # the command can still be copied
    train._trainer.setCurrentText("nnUNetTrainer")
    assert train._start.isEnabled()
    train._run_dialog.close()
    preprocess = env.page.pages["preprocess"]
    preprocess._run_dialog.open()
    preprocess._planner.setCurrentText("MyPlanner")
    assert not preprocess._run.isEnabled() and "Planner MyPlanner is not installed" in preprocess._run_hint.text()
    assert "-pl MyPlanner" in preprocess._command.command()
    preprocess._planner.setCurrentText("nnUNetPlannerResEncL")
    assert preprocess._run.isEnabled()
    preprocess._run_dialog.close()
    predict = env.page.pages["predict"]
    predict._input.setText(str(source))
    predict._output.setText(str(env.root / "out"))
    predict._folders_changed()
    env.wait()
    assert predict._predict.isEnabled()
    predict._trainer.setCurrentText("nnUNetTrainer_250epochs")
    assert not predict._predict.isEnabled() and "is not installed" in predict._busy_hint.text()
    assert "-tr nnUNetTrainer_250epochs" in predict._command.command()
