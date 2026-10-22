"""State shared by all pages: paths, settings, the job runner and Napari launching."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from qtpy.QtCore import QObject, QProcess, QSettings, QUrl, Signal
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import QMessageBox

from nnworkbench.services.commands import shell_join
from nnworkbench.services.device_probe import DeviceInfo, probe_device

from .jobs import JobRunner, echo, start_process_group, stop_process_group
from .tasks import TaskRunner

PREFETCH_RADIUS_RANGE = (1, 20)  # the limits of the settings dialog, also applied to saved values


def worker_range() -> tuple[int, int]:
    return 1, os.cpu_count() or 1


@dataclass
class Settings:
    annotation_workflow: bool = False
    workers: int = 8
    prefetch_previous: bool = True
    prefetch_next: bool = True
    prefetch_radius: int = 1


class AppContext(QObject):
    """One instance per window; pages read paths and settings from here."""

    settings_changed = Signal()
    annotation_changed = Signal(bool)
    howto_changed = Signal(bool)
    device_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.store = QSettings("nnWorkbench", "nnWorkbench")
        self.settings = Settings(
            annotation_workflow=self.store.value("annotation_workflow/enabled", False, type=bool),
            workers=self._stored_number("analysis/workers", min(8, os.cpu_count() or 1), worker_range()),
            prefetch_previous=self.store.value("napari/prefetch_previous", True, type=bool),
            prefetch_next=self.store.value("napari/prefetch_next", True, type=bool),
            prefetch_radius=self._stored_number("napari/prefetch_radius", 1, PREFETCH_RADIUS_RANGE),
        )
        self.roots: dict[str, Path | None] = {"raw": None, "preprocessed": None, "results": None}
        # The model last chosen on Train, Validate or Predict (trainer__plans__configuration); the other stages
        # select it too when they are opened. Cleared when another dataset opens.
        self.model_label: str | None = None
        self.jobs = JobRunner(self)
        self.tasks = TaskRunner(self)
        self._napari: list[QProcess] = []
        self.device: DeviceInfo | None = None  # None while the check runs (see detect_device)

    def _stored_number(self, key: str, default: int, limits: tuple[int, int]) -> int:
        """A saved whole number within *limits* (the settings file may have been edited, or the CPU changed)."""
        try:
            value = int(self.store.value(key, default))
        except (TypeError, ValueError):
            value = default
        return max(limits[0], min(limits[1], value))

    # ------------------------------------------------------------ device

    def detect_device(self) -> None:
        """Check in the background which device PyTorch can use (GPU, Apple GPU or CPU)."""
        self.device = None
        self.device_changed.emit()
        self.tasks.run("device", probe_device, self._device_detected,
                       lambda error: self._device_detected(DeviceInfo("unknown", error=error)))

    def _device_detected(self, info: object) -> None:
        self.device = info if isinstance(info, DeviceInfo) else DeviceInfo("unknown")
        self.device_changed.emit()

    # ------------------------------------------------------------ settings

    def save_settings(self, settings: Settings) -> None:
        annotation_changed = settings.annotation_workflow != self.settings.annotation_workflow
        self.settings = settings
        self.store.setValue("annotation_workflow/enabled", settings.annotation_workflow)
        self.store.setValue("analysis/workers", settings.workers)
        self.store.setValue("napari/prefetch_previous", settings.prefetch_previous)
        self.store.setValue("napari/prefetch_next", settings.prefetch_next)
        self.store.setValue("napari/prefetch_radius", settings.prefetch_radius)
        self.settings_changed.emit()
        if annotation_changed:
            self.annotation_changed.emit(settings.annotation_workflow)

    @property
    def howto_expanded(self) -> bool:
        """Whether the how-to column is shown (expanded by default, remembered across sessions)."""
        return self.store.value("ui/howto_expanded", True, type=bool)

    def set_howto_expanded(self, expanded: bool) -> None:
        self.store.setValue("ui/howto_expanded", expanded)
        self.howto_changed.emit(expanded)

    def set_annotation_workflow(self, enabled: bool) -> None:
        if enabled != self.settings.annotation_workflow:
            self.save_settings(Settings(enabled, self.settings.workers, self.settings.prefetch_previous,
                                        self.settings.prefetch_next, self.settings.prefetch_radius))

    def local_roots_available(self, *names: str) -> list[str]:
        """Return the nnU-Net folders that are missing locally (empty when all exist)."""
        return [f"nnUNet_{name}" for name in names if self.roots.get(name) is None or not self.roots[name].is_dir()]

    # ------------------------------------------------------------ Napari

    def _prefetch_arguments(self) -> list[str]:
        arguments = ["--prefetch-radius", str(self.settings.prefetch_radius)]
        if not self.settings.prefetch_previous:
            arguments.append("--no-prefetch-previous")
        if not self.settings.prefetch_next:
            arguments.append("--no-prefetch-next")
        return arguments

    def inspect_case(self, path: Path, raw_dataset: Path | None = None) -> None:
        """``raw_dataset``: its dataset.json may list the case files elsewhere (the "dataset" key)."""
        self._napari_module([str(path), *self._prefetch_arguments(), *_raw_dataset_arguments(raw_dataset)])

    def annotate_case(self, path: Path, raw_dataset: Path | None = None) -> None:
        self._napari_module([str(path), "--annotate", *_raw_dataset_arguments(raw_dataset)])

    def inspect_preprocessed(self, path: Path) -> None:
        self._napari_module([str(path), "--preprocessed-case", *self._prefetch_arguments()])

    def inspect_validation(self, path: Path, raw_dataset: Path | None) -> None:
        arguments = [str(path), "--validation-prediction", *self._prefetch_arguments()]
        if raw_dataset is not None:
            arguments += ["--raw-dataset", str(raw_dataset)]
        self._napari_module(arguments)

    def inspect_predictions(self, predictions: Path, images: Path | None) -> None:
        arguments = ["--prediction-folder", str(predictions), *self._prefetch_arguments()]
        if images is not None:
            arguments += ["--image-folder", str(images)]
        self._napari_module(arguments)

    def _napari_module(self, arguments: list[str]) -> None:
        command = ["-m", "nnworkbench.services.napari_inspector", *arguments]
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.ForwardedChannels)
        self._napari.append(process)
        process.finished.connect(lambda *_: self._forget(process))
        process.errorOccurred.connect(lambda error: self._napari_failed(process, error))
        start_process_group(process, sys.executable, command)
        echo(f"Napari: {shell_join([sys.executable, *command])}")

    @property
    def napari_windows(self) -> int:
        """Napari windows opened from nnWorkbench that are still open (they close with nnWorkbench)."""
        return len(self._napari)

    def _forget(self, process: QProcess) -> None:
        if process in self._napari:
            self._napari.remove(process)
        process.deleteLater()

    def _napari_failed(self, process: QProcess, error) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self._forget(process)
            QMessageBox.warning(None, "Could not start Napari", process.errorString())

    @staticmethod
    def open_folder(path: Path | None) -> None:
        if path is not None and path.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    def shutdown(self) -> None:
        self.tasks.shutdown()
        self.jobs.shutdown()
        for process in tuple(self._napari):
            stop_process_group(process, wait_ms=2_000)
        self._napari.clear()


def _raw_dataset_arguments(raw_dataset: Path | None) -> list[str]:
    return ["--raw-dataset", str(raw_dataset)] if raw_dataset is not None else []
