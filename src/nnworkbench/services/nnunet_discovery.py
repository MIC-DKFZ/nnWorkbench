"""Discover the planner and trainer classes of the installed nnU-Net, including custom ones.

Importing nnU-Net's packages is the only way to know which classes it can use (a custom installation, extra
modules, the folders in ``nnUNet_extTrainer``), but that import loads torch: about 4 s of CPU time and 400 MB that
the GUI would keep until it quits. ``discover_nnunet_classes`` therefore runs the import in a separate, short-lived
Python process (as the device check does) and reads the names from its output.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import os
import pkgutil
import subprocess
import sys
from dataclasses import dataclass

from nnworkbench.path_setup import own_trainer_folders, trainer_folders

DEFAULT_PLANNER = "ExperimentPlanner"
DEFAULT_TRAINER = "nnUNetTrainer"
_MARKER = "NNWORKBENCH_CLASSES "  # nnU-Net modules may print on import: the result is the marked line
_WINDOWS = os.name == "nt"


@dataclass(frozen=True)
class NnunetClasses:
    """The class names the GUI offers; the defaults alone when nnU-Net could not be inspected (``complete`` is
    False then: the GUI cannot tell whether a typed name exists and does not block a run on it)."""

    planners: tuple[str, ...] = (DEFAULT_PLANNER,)
    trainers: tuple[str, ...] = (DEFAULT_TRAINER,)
    complete: bool = False


def _subclass_names(package, base_class, prefix: str) -> set[str]:
    """The names of ``base_class`` subclasses in every module of *package* (recursively); modules that fail to
    import are skipped (a broken installed extension must not break the GUI)."""
    names: set[str] = set()
    for module_info in pkgutil.walk_packages(package.__path__, prefix, onerror=lambda _name: None):
        try:
            module = importlib.import_module(module_info.name)
        except Exception:  # noqa: BLE001, S112
            continue
        for _name, candidate in inspect.getmembers(module, inspect.isclass):
            try:
                if candidate is not base_class and issubclass(candidate, base_class):
                    names.add(candidate.__name__)
            except TypeError:
                continue
    return names


def _extension_folders() -> list[str]:
    return [folder for folder in own_trainer_folders() if os.path.isdir(folder)]


def _folder_subclass_names(folder: str, base_class, number: int) -> set[str]:
    """Subclasses in the Python files (and packages) of the user's folder number *number*. Its plain files are
    loaded under a name of their own (``my_trainer.py`` in two folders, or a file called like a module that is
    already imported, e.g. ``utils.py``, would otherwise be answered from ``sys.modules`` and never looked at);
    the folder is on ``sys.path`` meanwhile, so a file can import its neighbours as nnU-Net runs it."""
    names: set[str] = set()
    sys.path.insert(0, folder)
    try:
        for module_info in pkgutil.walk_packages([folder], onerror=lambda _name: None):
            try:
                if module_info.ispkg or "." in module_info.name:
                    module = importlib.import_module(module_info.name)
                else:
                    module = _load_file(os.path.join(folder, f"{module_info.name}.py"),
                                        f"_nnworkbench_trainers_{number}_{module_info.name}")
            except Exception:  # noqa: BLE001, S112
                continue
            for _name, candidate in inspect.getmembers(module, inspect.isclass):
                try:
                    if candidate is not base_class and issubclass(candidate, base_class):
                        names.add(candidate.__name__)
                except TypeError:
                    continue
    finally:
        if folder in sys.path:
            sys.path.remove(folder)
    return names


def _load_file(path: str, name: str):
    """Import one Python file under *name* (not through ``sys.modules`` lookup by its file name)."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses and pickling in the module expect to find it there
    spec.loader.exec_module(module)
    return module


def _ordered(names: set[str], default: str) -> tuple[str, ...]:
    return tuple(sorted(names | {default}, key=lambda name: (name != default, name.casefold())))


def available_planners() -> tuple[str, ...]:
    """Every installed nnU-Net experiment-planner class usable by the CLI. Imports nnU-Net (and torch): call it in
    a separate process (``discover_nnunet_classes``) from the GUI."""
    try:
        package = importlib.import_module("nnunetv2.experiment_planning.experiment_planners")
        base_module = importlib.import_module(
            "nnunetv2.experiment_planning.experiment_planners.default_experiment_planner")
        base_class = base_module.ExperimentPlanner
    except Exception:  # noqa: BLE001  (a partially installed or broken nnU-Net must not break the GUI)
        return (DEFAULT_PLANNER,)
    return _ordered(_subclass_names(package, base_class, f"{package.__name__}."), DEFAULT_PLANNER)


def available_trainers() -> tuple[str, ...]:
    """Every installed nnU-Net trainer class usable by the training CLI: nnU-Net's own (also those of a custom
    installation), the bundled ones and those in the user's ``nnUNet_extTrainer`` folders. Imports nnU-Net (and
    torch): call it in a separate process (``discover_nnunet_classes``) from the GUI."""
    try:
        package = importlib.import_module("nnunetv2.training.nnUNetTrainer")
        base_module = importlib.import_module("nnunetv2.training.nnUNetTrainer.nnUNetTrainer")
        base_class = base_module.nnUNetTrainer
    except Exception:  # noqa: BLE001  (a partially installed or broken nnU-Net must not break the GUI)
        return (DEFAULT_TRAINER,)
    names = _subclass_names(package, base_class, f"{package.__name__}.")
    try:
        bundled = importlib.import_module("nnworkbench.nnunet_trainers")
    except Exception:  # noqa: BLE001  (a broken installed extension must not break the GUI)
        bundled = None
    if bundled is not None:
        names |= _subclass_names(bundled, base_class, f"{bundled.__name__}.")
    for number, folder in enumerate(_extension_folders()):
        names |= _folder_subclass_names(folder, base_class, number)
    return _ordered(names, DEFAULT_TRAINER)


def parse_discovery(output: str) -> NnunetClasses:
    """The classes from the discovery process's output; the defaults when it printed no usable result."""
    for line in reversed(output.splitlines()):
        if not line.startswith(_MARKER):
            continue
        try:
            data = json.loads(line[len(_MARKER):])
        except ValueError:
            break
        if not isinstance(data, dict):
            break
        return NnunetClasses(_names(data.get("planners"), DEFAULT_PLANNER),
                             _names(data.get("trainers"), DEFAULT_TRAINER), complete=True)
    return NnunetClasses()


def _names(value: object, default: str) -> tuple[str, ...]:
    names = {name for name in value if isinstance(name, str) and name} if isinstance(value, list) else set()
    return _ordered(names, default)


def discover_nnunet_classes(timeout: float = 300) -> NnunetClasses:
    """List the installed planners and trainers in a separate Python process (it imports nnU-Net and torch, which
    takes a few seconds). The defaults when the process fails, e.g. without nnU-Net."""
    environment = {**os.environ, "nnUNet_extTrainer": trainer_folders(), "PYTHONIOENCODING": "utf-8"}
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "nnworkbench.services.nnunet_discovery"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout, check=False, env=environment,
            # no console window flashing up when the GUI was started without one (Windows)
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if _WINDOWS else 0)
    except (OSError, subprocess.SubprocessError):
        return NnunetClasses()
    return parse_discovery(completed.stdout)


def main() -> None:
    """Print the installed classes for ``discover_nnunet_classes``
    (``python -m nnworkbench.services.nnunet_discovery``)."""
    print(_MARKER + json.dumps({"planners": available_planners(), "trainers": available_trainers()}), flush=True)


if __name__ == "__main__":
    main()
