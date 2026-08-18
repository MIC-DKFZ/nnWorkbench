"""Discover optional nnU-Net extension classes without making them a GUI dependency."""

from __future__ import annotations

import importlib
import inspect
import pkgutil


def available_planners() -> tuple[str, ...]:
    """Return every installed nnU-Net experiment-planner class usable by the CLI."""
    default = "ExperimentPlanner"
    try:
        package = importlib.import_module("nnunetv2.experiment_planning.experiment_planners")
        base_module = importlib.import_module(
            "nnunetv2.experiment_planning.experiment_planners.default_experiment_planner")
        base_class = base_module.ExperimentPlanner
    # A partially installed or broken custom extension must not prevent the
    # default workflow from opening.
    except Exception:
        return (default,)

    names = {default}
    for module_info in pkgutil.walk_packages(package.__path__, f"{package.__name__}.", onerror=lambda _name: None):
        try:
            module = importlib.import_module(module_info.name)
        except Exception:
            continue
        for _name, candidate in inspect.getmembers(module, inspect.isclass):
            try:
                if candidate is not base_class and issubclass(candidate, base_class):
                    names.add(candidate.__name__)
            except TypeError:
                continue
    return tuple(sorted(names, key=lambda name: (name != default, name.casefold())))


def available_trainers() -> tuple[str, ...]:
    """Return every installed nnU-Net trainer class usable by the training CLI."""
    default = "nnUNetTrainer"
    try:
        package = importlib.import_module("nnunetv2.training.nnUNetTrainer")
        base_module = importlib.import_module("nnunetv2.training.nnUNetTrainer.nnUNetTrainer")
        base_class = base_module.nnUNetTrainer
    # A partially installed or broken custom extension must not prevent the
    # default workflow from opening.
    except Exception:
        return (default,)

    names = {default}
    for module_info in pkgutil.walk_packages(package.__path__, f"{package.__name__}.", onerror=lambda _name: None):
        try:
            module = importlib.import_module(module_info.name)
        except Exception:
            continue
        for _name, candidate in inspect.getmembers(module, inspect.isclass):
            try:
                if candidate is not base_class and issubclass(candidate, base_class):
                    names.add(candidate.__name__)
            except TypeError:
                continue
    return tuple(sorted(names, key=lambda name: (name != default, name.casefold())))
