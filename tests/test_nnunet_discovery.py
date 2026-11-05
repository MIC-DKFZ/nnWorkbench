"""The planner and trainer lists: read from the installed nnU-Net in a separate process."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nnworkbench.services import nnunet_discovery
from nnworkbench.services.nnunet_discovery import (
    NnunetClasses,
    available_trainers,
    discover_nnunet_classes,
    parse_discovery,
)

_NNUNET_INSTALLED = importlib.util.find_spec("nnunetv2") is not None


def test_the_result_line_is_found_among_other_output() -> None:
    output = ("Some library prints on import\n"
              'NNWORKBENCH_CLASSES {"planners": ["ExperimentPlanner", "MyPlanner"], '
              '"trainers": ["nnUNetTrainer", "MyTrainer", "", 3]}\n')
    assert parse_discovery(output) == NnunetClasses(("ExperimentPlanner", "MyPlanner"), ("nnUNetTrainer", "MyTrainer"))
    # the default comes first, the rest in alphabetical order, and it is always listed
    assert parse_discovery('NNWORKBENCH_CLASSES {"planners": ["b", "A"], "trainers": []}') == NnunetClasses(
        ("ExperimentPlanner", "A", "b"), ("nnUNetTrainer",))
    for broken in ("", "nothing here", "NNWORKBENCH_CLASSES not json", "NNWORKBENCH_CLASSES [1, 2]"):
        assert parse_discovery(broken) == NnunetClasses()


def test_a_failed_discovery_process_gives_the_defaults(monkeypatch) -> None:
    def failing(*_args, **_kwargs):
        raise OSError("no python")

    monkeypatch.setattr(subprocess, "run", failing)
    assert discover_nnunet_classes() == NnunetClasses()
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, "", "boom"))
    assert discover_nnunet_classes() == NnunetClasses()


def test_the_discovery_process_gets_the_trainer_folders_and_no_console_window(monkeypatch) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, 'NNWORKBENCH_CLASSES {"planners": [], "trainers": ["X"]}', "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("nnUNet_extTrainer", "/my/trainers")
    assert discover_nnunet_classes() == NnunetClasses(("ExperimentPlanner",), ("nnUNetTrainer", "X"))
    command, kwargs = calls[0]
    assert command[:3] == [sys.executable, "-m", "nnworkbench.services.nnunet_discovery"]
    folders = kwargs["env"]["nnUNet_extTrainer"]
    assert "/my/trainers" in folders and "nnunet_trainers" in folders  # the user's own and the bundled folder
    assert kwargs["env"]["PYTHONIOENCODING"] == "utf-8" and kwargs["timeout"] > 0


@pytest.mark.skipif(not _NNUNET_INSTALLED, reason="nnU-Net is not installed")
def test_the_separate_process_lists_nnunet_and_bundled_trainers() -> None:
    torch_was_loaded = "torch" in sys.modules
    classes = discover_nnunet_classes()
    assert "ExperimentPlanner" in classes.planners and "nnUNetPlannerResEncL" in classes.planners
    assert classes.trainers[0] == "nnUNetTrainer"
    for name in ("nnUNetTrainerBetterIgnoreSampling", "nnUNetTrainerBetterIgnoreSamplingOnlyMirror01",
                 "nnUNetTrainerBetterIgnoreSamplingNoMirroring"):
        assert name in classes.trainers  # the bundled trainers (nnworkbench.nnunet_trainers)
    assert ("torch" in sys.modules) == torch_was_loaded  # the whole point: this process stays without torch


@pytest.mark.skipif(not _NNUNET_INSTALLED, reason="nnU-Net is not installed")
def test_custom_trainers_in_the_users_own_folder_are_listed(tmp_path: Path, monkeypatch) -> None:
    # a folder in nnUNet_extTrainer, as nnU-Net searches it: plain Python files with trainer subclasses
    folder = tmp_path / "my trainers"
    folder.mkdir()
    (folder / "my_trainer.py").write_text(
        "from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer\n\n\n"
        "class nnUNetTrainerCustomFromFolder(nnUNetTrainer):\n    pass\n", encoding="utf-8")
    (folder / "broken.py").write_text("import module_that_does_not_exist\n", encoding="utf-8")
    # a second folder whose file has the same name, and a file named like a module that is already imported:
    # each file is loaded on its own, not answered from sys.modules by its file name
    second = tmp_path / "more trainers"
    second.mkdir()
    (second / "my_trainer.py").write_text(
        "from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer\n\n\n"
        "class nnUNetTrainerCustomFromSecondFolder(nnUNetTrainer):\n    pass\n", encoding="utf-8")
    (second / "json.py").write_text(
        "from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer\n\n\n"
        "class nnUNetTrainerCustomShadowingJson(nnUNetTrainer):\n    pass\n", encoding="utf-8")
    monkeypatch.setenv("nnUNet_extTrainer", os.pathsep.join([str(folder), str(second)]))
    code = ("import json, sys\nfrom nnworkbench.services.nnunet_discovery import available_trainers\n"
            "print(json.dumps(available_trainers()))")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                            encoding="utf-8")
    trainers = json.loads(result.stdout.strip().splitlines()[-1])
    assert "nnUNetTrainerCustomFromFolder" in trainers and "nnUNetTrainerBetterIgnoreSampling" in trainers
    assert "nnUNetTrainerCustomFromSecondFolder" in trainers and "nnUNetTrainerCustomShadowingJson" in trainers
    assert trainers[0] == "nnUNetTrainer"
    # the bundled folder is not an "own" folder, a missing one is skipped
    monkeypatch.setenv("nnUNet_extTrainer", os.pathsep.join([str(folder), str(tmp_path / "gone"), str(second)]))
    assert nnunet_discovery._extension_folders() == [str(folder), str(second)]


@pytest.mark.skipif(not _NNUNET_INSTALLED, reason="nnU-Net is not installed")
def test_classes_added_to_a_custom_nnunet_installation_are_listed(tmp_path: Path, monkeypatch) -> None:
    """A modified nnU-Net (a copy of the installed package with an extra trainer and planner inside it) is what
    the GUI must list: its classes come from the installation the discovery process imports."""
    import shutil

    import nnunetv2

    site = tmp_path / "custom site"
    shutil.copytree(Path(nnunetv2.__file__).parent, site / "nnunetv2",
                    ignore=shutil.ignore_patterns("__pycache__", "tests"))
    (site / "nnunetv2" / "training" / "nnUNetTrainer" / "variants" / "my_lab_trainer.py").write_text(
        "from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer\n\n\n"
        "class nnUNetTrainerMyLabCustom(nnUNetTrainer):\n    pass\n\n\n"
        "class nnUNetTrainerMyLabCustomDeep(nnUNetTrainerMyLabCustom):\n    pass\n", encoding="utf-8")
    (site / "nnunetv2" / "experiment_planning" / "experiment_planners" / "my_lab_planner.py").write_text(
        "from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import "
        "ExperimentPlanner\n\n\nclass MyLabPlanner(ExperimentPlanner):\n    pass\n", encoding="utf-8")
    # the discovery process imports this nnU-Net instead (and still finds nnworkbench through an existing path)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(site), os.environ.get("PYTHONPATH", "")])))
    classes = discover_nnunet_classes()
    assert "nnUNetTrainerMyLabCustom" in classes.trainers and "nnUNetTrainerMyLabCustomDeep" in classes.trainers
    assert "MyLabPlanner" in classes.planners
    assert "nnUNetTrainerBetterIgnoreSampling" in classes.trainers  # the bundled ones are still there


def test_in_process_discovery_without_nnunet_gives_the_default(monkeypatch) -> None:
    real_import = importlib.import_module

    def no_nnunet(name, *args):
        if name.startswith("nnunetv2"):
            raise ImportError(name)
        return real_import(name, *args)

    monkeypatch.setattr(nnunet_discovery.importlib, "import_module", no_nnunet)
    assert available_trainers() == ("nnUNetTrainer",)
    assert nnunet_discovery.available_planners() == ("ExperimentPlanner",)
