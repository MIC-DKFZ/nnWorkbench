"""`nnworkbench-run status -d 42`: what the GUI's stage tabs show, for terminals and clusters (no Qt needed)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .services.dataset_scanner import scan_datasets
from .ui.dataset_state import DatasetState, load_state

# ASCII markers, so the report prints in every console (including the Windows command prompt)
_MARKERS = {"ok": "[ok]", "todo": "[ready]", "warn": "[outdated]", "error": "[fix]", "": "[ - ]"}
_STAGES = (("Data", "data_state"), ("Preprocess", "preprocess_state"), ("Train", "train_state"),
           ("Validate", "validate_state"), ("Predict", "predict_state"))


def find_state(dataset_id: int, roots: dict[str, Path | None]) -> DatasetState:
    records = scan_datasets(roots.get("raw"), roots.get("preprocessed"), roots.get("results"))
    record = next((item for item in records if item.dataset_id == dataset_id), None)
    if record is None:
        raise ValueError(f"No dataset with ID {dataset_id} in the saved nnU-Net folders.")
    return load_state(record)


def build_status(state: DatasetState) -> dict:
    """The status as plain data (also the --json output)."""
    raw = state.raw
    stages = {}
    for title, method in _STAGES:
        stage_state, hint = getattr(state, method)()
        detail = hint.split(" · ", 1)[-1]
        if title == "Data" and stage_state == "error" and state.data_problems():
            detail = "must be fixed before preprocessing (see below)"  # the problems are listed once, below
        stages[title.lower()] = {"state": stage_state or "none", "detail": detail}
    return {
        "dataset": state.record.directory_name,
        "folders": {"raw": _text(state.record.raw_path), "preprocessed": _text(state.record.preprocessed_path),
                    "results": _text(state.record.results_path)},
        "cases": len(raw.cases),
        "channels": raw.channels,
        "labels": "sparse" if state.sparse else "dense",
        "ignore_label": state.ignore_id,
        "problems": state.data_problems(),
        "stages": stages,
        "plans": [{"name": plans.name, "planner": plans.planner,
                   "configurations": [config.name for config in plans.configurations if config.preprocessed_cases],
                   "outdated": plans.outdated} for plans in state.plans],
        "models": [{"trainer": model.trainer, "plans": model.plans, "configuration": model.configuration,
                    "mean_dice": model.mean_dice,
                    "folds": [{"fold": fold.name, "status": fold.status, "epoch": fold.epoch,
                               "total_epochs": fold.total_epochs, "dice": fold.dice}
                              for fold in model.folds]}
                   for model in state.models],
    }


def format_status(status: dict) -> str:
    """A short readable report: stage states first, then plans and models."""
    lines = [status["dataset"],
             f"  {status['cases']} cases · {status['channels']} channel(s) · {status['labels']} labels"
             + (f" (ignore = {status['ignore_label']})" if status["ignore_label"] is not None else ""), ""]
    for title, _method in _STAGES:
        stage = status["stages"][title.lower()]
        lines.append(f"  {title:<11} {_MARKERS.get(stage['state'], '[ - ]'):<10} {stage['detail']}")
    for problem in status["problems"]:
        lines.append(f"    - {problem}")
    if status["plans"]:
        lines += ["", "  Plans"]
        for plans in status["plans"]:
            configurations = ", ".join(plans["configurations"]) or "not preprocessed"
            lines.append(f"    {plans['name']} ({plans['planner']}): {configurations}"
                         + ("  [outdated]" if plans["outdated"] else ""))
    if status["models"]:
        lines += ["", "  Models"]
        for model in status["models"]:
            dice = f"{model['mean_dice']:.3f}" if model["mean_dice"] is not None else "-"
            lines.append(f"    {model['trainer']} · {model['plans']} · {model['configuration']}   mean Dice {dice}")
            folds = []
            for fold in model["folds"]:
                detail = f"Dice {fold['dice']:.3f}" if fold["dice"] is not None else (
                    f"epoch {fold['epoch']}" if fold["epoch"] is not None else "")
                folds.append(f"{fold['fold']}: {fold['status']}" + (f" ({detail})" if detail else ""))
            lines.append("      folds  " + " | ".join(folds))
    return "\n".join(lines)


def run_status(dataset_id: int, roots: dict[str, Path | None], as_json: bool = False) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # never fail on a console without UTF-8
    status = build_status(find_state(dataset_id, roots))
    print(json.dumps(status, indent=2) if as_json else format_status(status))
    return 0


def _text(path: Path | None) -> str | None:
    return str(path) if path is not None else None
