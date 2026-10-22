"""Main window: app bar, overview or dataset page, and the console."""

from __future__ import annotations

import json

from qtpy.QtGui import QGuiApplication, QKeySequence, QShortcut
from qtpy.QtWidgets import QDialog, QMainWindow, QMessageBox, QStackedWidget, QVBoxLayout, QWidget

from nnworkbench.models.dataset import DatasetRecord
from nnworkbench.path_setup import configuration_path, configured_paths, save_paths
from nnworkbench.services.dataset_creator import (
    create_dataset,
    create_next_iteration,
    next_free_dataset_id,
    next_iteration_name,
)
from nnworkbench.services.dataset_description import create_description
from nnworkbench.services.dataset_scanner import scan_datasets

from .app_bar import AppBar
from .console import ConsolePanel
from .context import AppContext
from .dataset_page import DatasetPage
from .new_dataset_dialog import NewDatasetDialog
from .overview_page import OverviewPage
from .settings_dialog import SettingsDialog
from .switcher import DatasetSwitcher


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("nnWorkbench — nnU-Net workflows and nnInteractive-powered annotation")
        # fits a 1366x768 laptop (about 728 px usable); smaller screens (1920x1080 at 150 % with a task bar: 688 px)
        # get their available height, so the console bar is not under the task bar
        screen = QGuiApplication.primaryScreen()
        available = screen.availableGeometry() if screen else None
        self.setMinimumSize(min(1_200, available.width()) if available else 1_200,
                            min(700, available.height() - 40) if available else 700)
        self.resize(1600, 1000)
        self.context = AppContext(self)
        self.records: list[DatasetRecord] = []
        root = QWidget()
        root.setObjectName("appRoot")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.app_bar = AppBar()
        layout.addWidget(self.app_bar)
        self.pages = QStackedWidget()
        self.pages.setObjectName("pages")
        self.overview = OverviewPage(self.context)
        self.dataset_page = DatasetPage(self.context)
        self.pages.addWidget(self.overview)
        self.pages.addWidget(self.dataset_page)
        layout.addWidget(self.pages, 1)
        self.console = ConsolePanel(self.context.jobs)
        layout.addWidget(self.console)
        self.switcher = DatasetSwitcher(self)
        self._guide: QDialog | None = None
        self._save_warned = False  # a failed save of paths.json is reported once per session
        # Wiring
        self.app_bar.back_requested.connect(self.show_overview)
        self.app_bar.switcher_requested.connect(self._open_switcher)
        self.app_bar.settings_requested.connect(self._open_settings)
        self.app_bar.guide_requested.connect(lambda: self._open_guide("README.md"))
        self.app_bar.about_requested.connect(self._open_about)
        self.app_bar.annotation_toggled.connect(self.context.set_annotation_workflow)
        self.app_bar.set_annotation(self.context.settings.annotation_workflow)
        self.context.annotation_changed.connect(self.app_bar.set_annotation)
        self.switcher.dataset_chosen.connect(self.open_dataset)
        self.switcher.overview_requested.connect(self.show_overview)
        self.switcher.new_dataset_requested.connect(self._create_dataset)
        self.switcher.hide()
        self.overview.open_requested.connect(self.open_dataset)
        self.overview.new_dataset_requested.connect(self._create_dataset)
        self.overview.refresh_requested.connect(self.refresh)
        self.overview.paths.set_paths(*self._saved_paths())
        if not configuration_path().exists() and any(self._saved_paths()):  # from nnU-Net's variables: keep them
            self._paths_changed()
        self.overview.paths.set_profiles(self._remote_profiles())
        self.overview.paths.paths_changed.connect(self._paths_changed)
        self.overview.paths.profiles_changed.connect(self._save_profiles)
        self.dataset_page.guide_requested.connect(self._open_guide)
        self.dataset_page.rescan_requested.connect(self.refresh)
        QShortcut(QKeySequence.StandardKey.Find, self, activated=self._find)
        QShortcut(QKeySequence.StandardKey.Refresh, self, activated=self._refresh_shown)
        self.overview.new_iteration_requested.connect(self._create_next_iteration)
        self.context.jobs.job_finished.connect(lambda _job: self._job_finished())
        self.context.jobs.job_started.connect(lambda _job: self.dataset_page.reload())  # e.g. the fold appears
        self.refresh()
        self.show_overview()

    # ------------------------------------------------------------ paths and datasets

    @staticmethod
    def _saved_paths() -> tuple[str, str, str]:
        try:
            paths = configured_paths()
        except (RuntimeError, TypeError, OSError, ValueError):
            return "", "", ""
        return tuple(paths.get(f"nnUNet_{key}", "") for key in ("raw", "preprocessed", "results"))  # type: ignore

    def _remote_profiles(self) -> object:
        try:
            return json.loads(self.context.store.value("remote_profiles", "{}", type=str))
        except json.JSONDecodeError:
            return {}

    def _save_profiles(self, profiles: object) -> None:
        self.context.store.setValue("remote_profiles", json.dumps(profiles, sort_keys=True))

    def _paths_changed(self) -> None:
        try:
            save_paths(*(str(path) if path else "" for path in self.overview.paths.local_paths()))
        except OSError as error:  # read-only configuration folder: the paths still apply until nnWorkbench closes
            if not self._save_warned:
                self._save_warned = True
                QMessageBox.warning(self, "Paths not saved",
                                    f"The nnU-Net folders could not be saved to {configuration_path()}:\n"
                                    f"{error.strerror or error}\n\nThey are used until nnWorkbench is closed.")
        self.refresh()

    def refresh(self) -> None:
        raw, preprocessed, results = self.overview.paths.local_paths()
        self.context.roots = {"raw": raw, "preprocessed": preprocessed, "results": results}
        current = self.dataset_page.record
        self.records = scan_datasets(raw, preprocessed, results)
        self.overview.set_records(self.records, keep=current.dataset_id if current else None)
        if current is not None:
            updated = next((record for record in self.records if record.dataset_id == current.dataset_id), None)
            if updated is not None:
                self.dataset_page.record = updated
                self.app_bar.show_dataset(updated) if self.pages.currentWidget() is self.dataset_page else None

    def show_overview(self) -> None:
        self.pages.setCurrentWidget(self.overview)
        self.app_bar.show_overview()

    def open_dataset(self, record: DatasetRecord) -> None:
        current = self.dataset_page.record
        if current is not None and current.dataset_id != record.dataset_id and not self._confirm_leave():
            return
        self.overview.select_dataset(record.dataset_id)
        self.dataset_page.open(record)
        self.pages.setCurrentWidget(self.dataset_page)
        self.app_bar.show_dataset(record)

    def _find(self) -> None:
        """Ctrl+F: the dataset search on the overview, the dataset switcher on a dataset."""
        if self.pages.currentWidget() is self.overview:
            self.overview.focus_search()
        else:
            self._open_switcher()

    def _refresh_shown(self) -> None:
        """F5 (Ctrl+R on macOS): rescan the folders, and reload the open dataset."""
        self.refresh()
        if self.pages.currentWidget() is self.dataset_page:
            self.dataset_page.reload()

    def _open_switcher(self) -> None:
        self.app_bar.switcher.setChecked(False)
        self.switcher.open(self.app_bar.switcher_anchor(), self.records, self.dataset_page.record)

    def _job_finished(self) -> None:
        self.refresh()
        self.dataset_page.reload()

    # ------------------------------------------------------------ dialogs

    def _open_about(self) -> None:
        from .about_dialog import AboutDialog

        dialog = AboutDialog(self, device=self.context.device)
        dialog.exec()
        dialog.deleteLater()  # a new one is created next time

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self.context.settings, self)
        if dialog.exec():
            self.context.save_settings(dialog.settings())

    def _open_guide(self, file_name: str, anchor: str = "") -> None:
        from .workflow_guide import AnnotationWorkflowGuide

        if self._guide is None:
            self._guide = QDialog(self)
            self._guide.setWindowTitle("nnWorkbench guide")
            # never taller or wider than the screen (minus task bar and title bar), e.g. on a 1366x768 laptop
            screen = (self.screen() if hasattr(self, "screen") else None) or QGuiApplication.primaryScreen()
            width, height = 980, 820
            if screen is not None:
                room = screen.availableGeometry()
                width, height = max(480, min(width, room.width() - 40)), max(400, min(height, room.height() - 60))
            self._guide.resize(width, height)
            layout = QVBoxLayout(self._guide)
            self._guide_view = AnnotationWorkflowGuide(parent=self._guide)
            layout.addWidget(self._guide_view)
        self._guide.show()  # first, so the page is laid out before scrolling to the anchor
        self._guide_view.show_file(file_name, anchor)
        self._guide.raise_()

    def _create_dataset(self) -> None:
        raw, preprocessed, results = self.overview.paths.local_paths()
        if not _is_dir(raw):
            QMessageBox.warning(self, "nnUNet_raw required", "Choose an existing nnUNet_raw folder on the overview first.")
            return
        dialog = NewDatasetDialog(self, dataset_id=next_free_dataset_id(1, [raw, preprocessed, results]))
        if not dialog.exec():
            return
        try:
            created = create_dataset(raw, [raw, preprocessed, results], dialog.dataset_id.value(), dialog.name.text())
        except (ValueError, OSError) as error:  # OSError: read-only nnUNet_raw, unreadable roots, ...
            QMessageBox.warning(self, "Dataset not created", str(error))
            return
        if dialog.description is not None and dialog.description.isChecked():
            try:
                create_description(created)
            except (ValueError, OSError) as error:
                QMessageBox.warning(self, "dataset-description.md not created", str(error))
        self.refresh()
        record = next((item for item in self.records if item.raw_path == created), None)
        if record is not None:
            self.open_dataset(record)

    def _create_next_iteration(self, selected: DatasetRecord | None) -> None:
        raw, preprocessed, results = self.overview.paths.local_paths()
        sources = [record for record in self.records if record.raw_path is not None]
        if not sources or not _is_dir(raw):
            QMessageBox.warning(self, "nnUNet_raw required", "The source dataset must be available in nnUNet_raw.")
            return
        roots = [raw, preprocessed, results]
        dialog = NewDatasetDialog(
            self, title="Create next iteration", sources=sources,
            source=next((record for record in sources if selected and record.dataset_id == selected.dataset_id),
                        sources[0]),
            suggest=lambda record: (next_free_dataset_id(record.dataset_id + 1, roots), next_iteration_name(record.name)))
        if not dialog.exec():
            return
        source = dialog.source()
        try:
            created = create_next_iteration(raw, roots, source.raw_path, dialog.dataset_id.value(), dialog.name.text(),
                                            link_images=bool(dialog.link_images and dialog.link_images.isChecked()))
        except (ValueError, OSError) as error:
            QMessageBox.warning(self, "Iteration not created", str(error))
            return
        self.refresh()
        record = next((item for item in self.records if item.raw_path == created), None)
        if record is not None:
            self.open_dataset(record)

    def closeEvent(self, event) -> None:
        if not self._confirm_quit():
            event.ignore()
            return
        self.context.shutdown()
        super().closeEvent(event)

    def _confirm_leave(self) -> bool:
        """Unsaved edits of the open dataset: Save / Discard / Cancel. On Cancel its page is shown again."""
        if self.dataset_page.confirm_leave():
            return True
        if self.dataset_page.record is not None:
            self.pages.setCurrentWidget(self.dataset_page)
            self.app_bar.show_dataset(self.dataset_page.record)
        return False

    def _confirm_quit(self) -> bool:
        """Ask before quitting with unsaved edits, while a job runs (keep it running or stop it) or Napari windows
        are open."""
        if not self._confirm_leave():
            return False
        jobs, napari = self.context.jobs, self.context.napari_windows
        windows = (f"\n\nThe {napari} Napari window{'s' if napari != 1 else ''} opened from nnWorkbench close as "
                   "well; save your annotations first." if napari else "")
        job = jobs.current
        if job is not None:
            answer = self._ask_about_job(job, windows)
            if answer == "keep":
                jobs.detach()
            return answer in {"keep", "stop"}
        if napari:
            return QMessageBox.question(
                self, "Quit nnWorkbench?", windows.strip(),
                QMessageBox.StandardButton.Close | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel) == QMessageBox.StandardButton.Close
        return True

    def _ask_about_job(self, job, windows: str) -> str:
        """"keep", "stop" or "cancel"."""
        progress = f" ({job.progress})" if job.progress else ""
        later = ""
        if job.steps:
            names = ", ".join(step[0] or "the next step" for step in job.steps)
            later = f" The steps after it ({names}) do not start; start them again later."
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("A job is running")
        box.setText(f"{job.label}{progress} is still running.")
        box.setInformativeText(
            "Keep it running: it goes on without nnWorkbench and writes its output to the log file\n"
            f"{job.log}\nA training shows up on the Train page again when you reopen nnWorkbench." + later + windows)
        keep = box.addButton("Keep running and quit", QMessageBox.ButtonRole.AcceptRole)
        stop = box.addButton("Stop job and quit", QMessageBox.ButtonRole.DestructiveRole)
        cancel = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(cancel)
        box.exec()
        clicked = box.clickedButton()
        box.deleteLater()
        return "keep" if clicked is keep else "stop" if clicked is stop else "cancel"


def _is_dir(path) -> bool:
    try:
        return path is not None and path.is_dir()
    except OSError:
        return False
