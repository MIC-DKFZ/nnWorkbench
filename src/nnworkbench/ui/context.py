"""State shared by all pages: paths, settings, the job runner and Napari launching."""

from __future__ import annotations

import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

from qtpy.QtCore import QObject, QProcess, QSettings, QUrl, Signal
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import QMessageBox

from .jobs import JobRunner, start_process_group, stop_process_group
from .tasks import TaskRunner


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

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.store = QSettings("nnWorkbench", "nnWorkbench")
        self.settings = Settings(
            annotation_workflow=self.store.value("annotation_workflow/enabled", False, type=bool),
            workers=self.store.value("analysis/workers", min(8, os.cpu_count() or 1), type=int),
            prefetch_previous=self.store.value("napari/prefetch_previous", True, type=bool),
            prefetch_next=self.store.value("napari/prefetch_next", True, type=bool),
            prefetch_radius=self.store.value("napari/prefetch_radius", 1, type=int),
        )
        self.roots: dict[str, Path | None] = {"raw": None, "preprocessed": None, "results": None}
        self.jobs = JobRunner(self)
        self.tasks = TaskRunner(self)
        self._napari: list[QProcess] = []

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

    def inspect_case(self, path: Path) -> None:
        self._napari_module([str(path), *self._prefetch_arguments()])

    def annotate_case(self, path: Path) -> None:
        self._napari_module([str(path), "--annotate"])

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
        print(f"Napari: {shlex.join([sys.executable, *command])}")
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.ForwardedChannels)
        self._napari.append(process)
        process.finished.connect(lambda *_: self._forget(process))
        process.errorOccurred.connect(lambda error: self._napari_failed(process, error))
        start_process_group(process, sys.executable, command)

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
