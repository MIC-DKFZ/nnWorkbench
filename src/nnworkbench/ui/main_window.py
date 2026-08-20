from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, QSignalBlocker, QSize, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton,
                               QSplitter, QVBoxLayout, QWidget)

from nnworkbench.models.dataset import DatasetRecord
from nnworkbench.services.dataset_creator import (create_dataset, create_next_iteration, next_free_dataset_id,
                                                  next_iteration_name)
from nnworkbench.services.dataset_scanner import scan_datasets
from .metadata_editor import MetadataEditor
from .dataset_workspace import DatasetWorkspace
from .new_dataset_dialog import NewDatasetDialog
from .paths_header import PathsHeader


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("nnWorkbench — nnU-Net workflows and nnInteractive-powered annotation"); self.resize(1200, 760)
        self.settings = QSettings("nnWorkbench", "nnWorkbench"); self.records: list[DatasetRecord] = []
        root = QWidget(); root.setObjectName("appRoot"); self.setCentralWidget(root); layout = QVBoxLayout(root)
        layout.setContentsMargins(20, 20, 20, 20); layout.setSpacing(14)
        self.header = PathsHeader(); layout.addWidget(self.header)
        self.header.annotation_enabled_changed.connect(
            lambda enabled: self.settings.setValue("annotation_workflow/enabled", enabled))
        self.header.set_annotation_enabled(self.settings.value("annotation_workflow/enabled", False, type=bool))
        self.header.set_paths(*(self.settings.value(f"paths/{key}", "", type=str) for key in ("raw", "preprocessed", "results")))
        self.header.paths_changed.connect(self.refresh)
        self.splitter = QSplitter(); layout.addWidget(self.splitter, 1)
        left = QWidget(); left.setObjectName("datasetsPane"); left_layout = QVBoxLayout(left); left_layout.setContentsMargins(16, 16, 16, 16); left_layout.setSpacing(10)
        datasets_title = QLabel("Datasets"); datasets_title.setObjectName("paneTitle"); left_layout.addWidget(datasets_title)
        self.new_dataset = QPushButton("New dataset"); self.new_dataset.setObjectName("primaryButton"); self.new_dataset.clicked.connect(self._create_dataset); left_layout.addWidget(self.new_dataset)
        self.list = QListWidget(); self.list.currentRowChanged.connect(self._select); left_layout.addWidget(self.list, 1); self.splitter.addWidget(left)
        acknowledgements = QWidget(); acknowledgements.setObjectName("acknowledgementLogos")
        acknowledgements_layout = QVBoxLayout(acknowledgements)
        acknowledgements_layout.setContentsMargins(0, 8, 0, 0); acknowledgements_layout.setSpacing(4)
        asset_directory = Path(__file__).parent.parent / "assets"
        for filename, size, tooltip in (("HI_Logo.svg", QSize(145, 30), "Helmholtz Imaging"),
                                        ("DKFZ_Logo.svg", QSize(195, 32), "German Cancer Research Center (DKFZ)")):
            logo = QLabel(); logo.setPixmap(QIcon(str(asset_directory / filename)).pixmap(size)); logo.setToolTip(tooltip)
            acknowledgements_layout.addWidget(logo)
        left_layout.addWidget(acknowledgements)
        centre = DatasetWorkspace(); centre.setObjectName("workspacePane"); self.workspace = centre; self.splitter.addWidget(centre)
        self.header.annotation_documentation_requested.connect(self.workspace.show_annotation_documentation)
        self.header.annotation_documentation_closed.connect(self.workspace.hide_annotation_documentation)
        self.workspace.annotation_documentation_visibility_changed.connect(
            self.header.set_annotation_documentation_visible)
        self.header.annotation_enabled_changed.connect(self.workspace.set_annotation_workflow_enabled)
        self.workspace.set_annotation_workflow_enabled(self.header.annotation_enabled())
        self.workspace.next_iteration_requested.connect(self._create_next_iteration)
        self.editor = MetadataEditor(); self.splitter.addWidget(self.editor)
        self.editor.definition_errors_changed.connect(self.workspace.set_metadata_definition_errors)
        self.editor.metadata_saved.connect(self.workspace.refresh_consistency_check)
        self.header.annotation_enabled_changed.connect(self.editor.set_annotation_workflow_enabled)
        self.editor.set_annotation_workflow_enabled(self.header.annotation_enabled())
        # Allocate Datasets, the main workspace, and dataset.json at a 1:3:1 ratio.
        self.splitter.setStretchFactor(0, 1); self.splitter.setStretchFactor(1, 3); self.splitter.setStretchFactor(2, 1)
        self.splitter.setSizes((240, 720, 240))
        self._splitter_sizes_initialized = False
        self.refresh()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._splitter_sizes_initialized:
            self._splitter_sizes_initialized = True
            QTimer.singleShot(0, self._apply_initial_splitter_sizes)

    def closeEvent(self, event) -> None:
        self.workspace.shutdown()
        super().closeEvent(event)

    def _apply_initial_splitter_sizes(self) -> None:
        fifth = self.splitter.width() // 5
        self.splitter.setSizes((fifth, self.splitter.width() - 2 * fifth, fifth))

    def _paths(self) -> tuple[Path | None, Path | None, Path | None]: return self.header.paths()

    def refresh(self) -> None:
        raw, preprocessed, results = self._paths()
        self.workspace.set_environment_roots(raw, preprocessed, results)
        for key, path in zip(("raw", "preprocessed", "results"), (raw, preprocessed, results)):
            self.settings.setValue(f"paths/{key}", str(path) if path else "")
        previous_id = self.records[self.list.currentRow()].dataset_id if 0 <= self.list.currentRow() < len(self.records) else None
        self.records = scan_datasets(raw, preprocessed, results)
        index = next((i for i, record in enumerate(self.records) if record.dataset_id == previous_id), -1)
        selected_row = index if index >= 0 else 0 if self.records else -1
        with QSignalBlocker(self.list):
            self.list.clear()
            for record in self.records:
                self.list.addItem(QListWidgetItem(record.directory_name))
            self.list.setCurrentRow(selected_row)
        self._select(selected_row)

    def _select(self, row: int) -> None:
        self.workspace.hide_annotation_documentation()
        if not 0 <= row < len(self.records):
            self.workspace.set_dataset(None); self.editor.set_dataset(None); return
        record = self.records[row]
        self.workspace.set_dataset(record)
        self.editor.set_dataset(record.raw_path, record.preprocessed_path, record.results_path)

    def _create_dataset(self) -> None:
        raw, preprocessed, results = self._paths()
        if raw is None or not raw.is_dir():
            QMessageBox.warning(self, "Raw path required", "Choose an existing nnUNet_raw directory first."); return
        dialog = NewDatasetDialog(self)
        if not dialog.exec(): return
        try:
            created = create_dataset(raw, [raw, preprocessed, results], dialog.dataset_id.value(), dialog.name.text(),
                                     include_annotations=self.header.annotation_enabled())
        except ValueError as error:
            QMessageBox.warning(self, "Dataset not created", str(error)); return
        self.refresh()
        row = next((i for i, record in enumerate(self.records) if record.raw_path == created), -1); self.list.setCurrentRow(row)
        QMessageBox.information(self, "Dataset created", f"Created {created.name} in the raw directory.")

    def _create_next_iteration(self) -> None:
        row = self.list.currentRow()
        if not 0 <= row < len(self.records):
            return
        source = self.records[row]
        raw, preprocessed, results = self._paths()
        if raw is None or not raw.is_dir() or source.raw_path is None:
            QMessageBox.warning(self, "Raw path required", "Choose a dataset available in nnUNet_raw first.")
            return
        dataset_id = next_free_dataset_id(source.dataset_id + 1, [raw, preprocessed, results])
        dialog = NewDatasetDialog(self, dataset_id=dataset_id, name=next_iteration_name(source.name),
                                  title="Create next iteration")
        if not dialog.exec():
            return
        try:
            created = create_next_iteration(raw, [raw, preprocessed, results], source.raw_path,
                                            dialog.dataset_id.value(), dialog.name.text())
        except ValueError as error:
            QMessageBox.warning(self, "Iteration not created", str(error))
            return
        except OSError as error:
            QMessageBox.critical(self, "Iteration not created", f"Could not copy dataset files:\n{error}")
            return
        self.refresh()
        new_row = next((index for index, record in enumerate(self.records) if record.raw_path == created), -1)
        self.list.setCurrentRow(new_row)
        QMessageBox.information(self, "Iteration created", f"Created {created.name} with copied images and annotations.")
