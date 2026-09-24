from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import (
    QObject,
    QPoint,
    QProcess,
    QProcessEnvironment,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QPainter, QPixmap, QPolygon, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QSizePolicy,
    QStyle,
    QStyleOptionSpinBox,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nnworkbench.models.dataset import DatasetRecord
from nnworkbench.services.annotation_conversion import (
    AnnotationConversionResult,
    annotation_files,
    convert_annotations,
    has_annotation_files,
)
from nnworkbench.services.dataset_contents import (
    fold_training_status,
    json_value,
    model_folds,
    model_trainer_plan_configuration,
    preprocessed_files,
    raw_files,
    trained_models,
    validation_files,
)
from nnworkbench.services.dataset_json import ImageCountCheck, image_count_check
from nnworkbench.services.label_analysis import (
    analysis_path,
    label_files,
    load_label_analysis,
)
from nnworkbench.services.nnunet_discovery import available_planners, available_trainers

from .label_statistics_dialog import LabelStatisticsView
from .prediction_dialog import PredictionDialog
from .prediction_inspection_dialog import PredictionInspectionDialog
from .preprocessing_dialog import PreprocessingDialog
from .training_dialog import TrainingDialog


class _WorkerSignals(QObject):
    completed = Signal(object)
    failed = Signal(str)


class _LoadWorker(QRunnable):
    """Run filesystem work outside Qt's UI thread."""

    def __init__(self, task: Callable[[], object], cancelled: Callable[[], bool]) -> None:
        super().__init__()
        self.task, self.cancelled = task, cancelled
        self.signals = _WorkerSignals()

    def run(self) -> None:
        if self.cancelled():
            return
        try:
            result = self.task()
            if not self.cancelled():
                self.signals.completed.emit(result)
        except Exception as error:
            if not self.cancelled():
                self.signals.failed.emit(str(error))


class _SettingsSpinBox(QSpinBox):
    """A styled spin box that keeps visible arrows after QSS customisation."""

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        option = QStyleOptionSpinBox(); self.initStyleOption(option)
        style = self.style()
        up = style.subControlRect(QStyle.ComplexControl.CC_SpinBox, option, QStyle.SubControl.SC_SpinBoxUp, self)
        down = style.subControlRect(QStyle.ComplexControl.CC_SpinBox, option, QStyle.SubControl.SC_SpinBoxDown, self)
        painter = QPainter(self); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(QColor("#475569"))
        for rect, points in ((up, ((0, -3), (-4, 2), (4, 2))), (down, ((0, 3), (-4, -2), (4, -2)))):
            center = rect.center()
            painter.drawPolygon(QPolygon([QPoint(center.x() + x, center.y() + y) for x, y in points]))
        painter.end()


class DatasetWorkspace(QWidget):
    """Lazy, read-only browser for the files and outputs of a selected dataset."""

    _RAW_FOLDER_ROLE = Qt.ItemDataRole.UserRole + 1
    _RAW_STATISTICS_ROLE = Qt.ItemDataRole.UserRole + 2
    _FOLD_ROLE = Qt.ItemDataRole.UserRole + 3
    _VALIDATION_FILE_ROLE = Qt.ItemDataRole.UserRole + 4
    annotation_documentation_visibility_changed = Signal(bool)
    next_iteration_requested = Signal()
    nnunet_command_finished = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._record: DatasetRecord | None = None
        self._dataset_version = 0
        self._loaded_tabs: set[str] = set()
        self._loading_tabs: set[str] = set()
        self._plan_request = 0
        self._summary_request = 0
        self._training_request = 0
        self._planner_request = 0
        self._consistency_request = 0
        self._analysis_running = False
        self._conversion_request = 0
        self._conversion_running = False
        self._session_conversion_warnings: dict[Path, tuple[str, ...]] = {}
        self._annotation_workflow_enabled = False
        self._metadata_errors: tuple[str, ...] = ()
        self._analysis_workers = 8
        self._napari_prefetch_previous = True
        self._napari_prefetch_next = True
        self._napari_prefetch_radius = 1
        self._preprocessed_paths: list[Path] = []
        self._trained_model_paths: list[Path] = []
        self._raw_folder_files: dict[str, list[Path]] = {}
        self._loaded_raw_folders: set[str] = set()
        self._raw_render_version = 0
        self._workers: list[_LoadWorker] = []
        self._extension_workers: list[_LoadWorker] = []
        self._processes: list[QProcess] = []
        self._napari_processes: list[QProcess] = []
        self._active_process: QProcess | None = None
        self._analysis_process: QProcess | None = None
        self._closing = False
        # Filesystem discovery is I/O bound. A small dedicated pool keeps navigation
        # responsive without competing with nnU-Net or analysis worker processes.
        self._thread_pool = QThreadPool(self)
        self._thread_pool.setMaxThreadCount(min(4, max(2, (os.cpu_count() or 2) // 2)))
        self._thread_pool.setExpiryTimeout(10_000)
        self._extension_pool = QThreadPool(self)
        self._extension_pool.setMaxThreadCount(2)
        self._available_planners: tuple[str, ...] = ("ExperimentPlanner",)
        self._available_trainers: tuple[str, ...] = ("nnUNetTrainer",)
        self._planners_loaded = self._planners_loading = False
        self._trainers_loaded = self._trainers_loading = False
        self._pending_preprocessing: tuple[int, int] | None = None
        self._pending_training: tuple[int, int, object] | None = None
        self._environment_roots: dict[str, Path | None] = {"nnUNet_raw": None, "nnUNet_preprocessed": None,
                                                            "nnUNet_results": None}

        root_layout = QVBoxLayout(self); root_layout.setContentsMargins(20, 16, 20, 16); root_layout.setSpacing(10)
        self.content_splitter = QSplitter(Qt.Orientation.Vertical); root_layout.addWidget(self.content_splitter, 1)
        content = QWidget(); layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(10)
        self.content_splitter.addWidget(content)
        title_row = QHBoxLayout()
        self.title = QLabel("Dataset"); self.title.setObjectName("paneTitle"); title_row.addWidget(self.title)
        title_row.addStretch()
        self.open_dataset_folder = QPushButton("Open folder")
        self.open_dataset_folder.setToolTip("Open the selected dataset folder in the system file manager")
        self.open_dataset_folder.clicked.connect(self._open_dataset_folder)
        self.open_dataset_folder.setEnabled(False)
        title_row.addWidget(self.open_dataset_folder)
        layout.addLayout(title_row)
        self.summary = QLabel("Select a dataset."); self.summary.setObjectName("workspaceSummary")
        self.summary.setWordWrap(True); layout.addWidget(self.summary)
        self.metadata_warning = QLabel(); self.metadata_warning.setObjectName("consistencyWarning")
        self.metadata_warning.setWordWrap(True); self.metadata_warning.hide(); layout.addWidget(self.metadata_warning)
        self.consistency_warning = QLabel(); self.consistency_warning.setObjectName("consistencyWarning")
        self.consistency_warning.setWordWrap(True); self.consistency_warning.hide(); layout.addWidget(self.consistency_warning)
        self.metadata_error = QLabel(); self.metadata_error.setObjectName("metadataError")
        self.metadata_error.setWordWrap(True); self.metadata_error.hide(); layout.addWidget(self.metadata_error)
        self.tabs = QTabWidget(); self.tabs.currentChanged.connect(self._load_active_tab); layout.addWidget(self.tabs, 1)
        self.documentation_panel = QWidget(); documentation_layout = QVBoxLayout(self.documentation_panel)
        documentation_layout.setContentsMargins(0, 0, 0, 0); documentation_layout.setSpacing(8)
        documentation_header = QHBoxLayout()
        documentation_title = QLabel("Annotation workflow"); documentation_title.setObjectName("paneTitle")
        documentation_header.addWidget(documentation_title); documentation_header.addStretch()
        back_to_workspace = QPushButton("Back to workspace")
        back_to_workspace.clicked.connect(self.hide_annotation_documentation)
        documentation_header.addWidget(back_to_workspace); documentation_layout.addLayout(documentation_header)
        self.documentation_content = QWidget(); self.documentation_content.setObjectName("documentationContent")
        self.documentation_content_layout = QVBoxLayout(self.documentation_content)
        self.documentation_content_layout.setContentsMargins(0, 0, 0, 0)
        documentation_layout.addWidget(self.documentation_content, 1)
        self.documentation_panel.hide(); layout.addWidget(self.documentation_panel, 1)
        self.console_panel = QWidget(); console_layout = QVBoxLayout(self.console_panel)
        console_layout.setContentsMargins(0, 0, 0, 0); console_layout.setSpacing(4)
        console_header = QHBoxLayout()
        console_title = QLabel("Console"); console_title.setObjectName("consoleTitle"); console_header.addWidget(console_title)
        console_header.addStretch()
        self.terminate_command = QPushButton("Terminate command"); self.terminate_command.setEnabled(False)
        self.terminate_command.clicked.connect(self._terminate_command); console_header.addWidget(self.terminate_command)
        self.hide_console = QPushButton("Hide console")
        self.hide_console.clicked.connect(self.console_panel.hide); console_header.addWidget(self.hide_console)
        console_layout.addLayout(console_header)
        self.console = QPlainTextEdit(); self.console.setObjectName("commandConsole"); self.console.setReadOnly(True)
        self.console.setMinimumHeight(100); console_layout.addWidget(self.console)
        self.console_panel.hide(); self.content_splitter.addWidget(self.console_panel)
        self.content_splitter.setStretchFactor(0, 3); self.content_splitter.setStretchFactor(1, 1)

        raw = QWidget(); raw_layout = QVBoxLayout(raw); raw_layout.setContentsMargins(0, 0, 0, 0); raw_layout.setSpacing(8)
        basic_actions = QHBoxLayout()
        self.convert_annotations = QPushButton("Convert Annotations")
        self.convert_annotations.clicked.connect(self._convert_annotations)
        self.next_iteration = QPushButton("Create next iteration")
        self.next_iteration.clicked.connect(self.next_iteration_requested.emit)
        self.analyze_labels = QPushButton("Analyze Labels")
        self.analyze_labels.clicked.connect(self._analyze_labels)
        self.analysis_settings = QPushButton("⚙")
        self.analysis_settings.setObjectName("settingsButton")
        self.analysis_settings.setToolTip("Analysis and Napari settings")
        self.analysis_settings.clicked.connect(self._configure_workspace_settings)
        self.run_preprocessing = QPushButton("Run Preprocessing")
        self.run_preprocessing.clicked.connect(self._configure_preprocessing)
        basic_actions.addWidget(self.analyze_labels); basic_actions.addWidget(self.run_preprocessing)
        basic_actions.addStretch(); basic_actions.addWidget(self.analysis_settings); raw_layout.addLayout(basic_actions)
        self.annotation_actions = QWidget()
        annotation_actions = QHBoxLayout(self.annotation_actions)
        annotation_actions.setContentsMargins(0, 0, 0, 0)
        annotation_actions.addWidget(self.convert_annotations); annotation_actions.addWidget(self.next_iteration)
        annotation_actions.addStretch()
        self.annotation_actions.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.annotation_actions.hide(); raw_layout.addWidget(self.annotation_actions)
        self.raw_tree = QTreeWidget(); self.raw_tree.setColumnCount(1); self.raw_tree.setHeaderHidden(True)
        self.raw_tree.itemExpanded.connect(self._expand_raw_folder)
        raw_header = self.raw_tree.header()
        raw_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.raw_tree.currentItemChanged.connect(self._show_raw_statistics)
        self.raw_tree.itemClicked.connect(self._toggle_raw_statistics)
        self._ignore_next_statistics_click: QTreeWidgetItem | None = None
        self.raw_splitter = QSplitter(Qt.Orientation.Vertical); self.raw_splitter.addWidget(self.raw_tree)
        self.raw_statistics_container = QWidget(); self.raw_statistics_layout = QVBoxLayout(self.raw_statistics_container)
        self.raw_statistics_layout.setContentsMargins(0, 0, 0, 0)
        self.raw_statistics_scroll = QScrollArea(); self.raw_statistics_scroll.setObjectName("rawStatisticsScroll")
        self.raw_statistics_scroll.setWidgetResizable(True)
        self.raw_statistics_scroll.setWidget(self.raw_statistics_container)
        self.raw_splitter.addWidget(self.raw_statistics_scroll); self.raw_statistics_scroll.hide()
        raw_layout.addWidget(self.raw_splitter); self.tabs.addTab(raw, "Raw")

        preprocessed = QWidget(); preprocessed_layout = QVBoxLayout(preprocessed)
        preprocessed_layout.setContentsMargins(0, 0, 0, 0)
        preprocessed_actions = QHBoxLayout()
        self.run_training = QPushButton("Run Training"); self.run_training.setEnabled(False)
        self.run_training.clicked.connect(self._configure_training)
        self.preprocessed_settings = QPushButton("⚙")
        self.preprocessed_settings.setObjectName("settingsButton")
        self.preprocessed_settings.setToolTip("Analysis and Napari settings")
        self.preprocessed_settings.clicked.connect(self._configure_workspace_settings)
        preprocessed_actions.addWidget(self.run_training); preprocessed_actions.addStretch()
        preprocessed_actions.addWidget(self.preprocessed_settings)
        preprocessed_layout.addLayout(preprocessed_actions)
        self.preprocessed_splitter = QSplitter(); preprocessed_layout.addWidget(self.preprocessed_splitter)
        self.preprocessed_tree = QTreeWidget(); self.preprocessed_tree.setColumnCount(1); self.preprocessed_tree.setHeaderHidden(True)
        self.preprocessed_tree.currentItemChanged.connect(self._show_json_content)
        preprocessed_header = self.preprocessed_tree.header()
        preprocessed_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.preprocessed_splitter.addWidget(self.preprocessed_tree)
        self.json_content = QPlainTextEdit(); self.json_content.setReadOnly(True)
        self.preprocessed_splitter.addWidget(self.json_content)
        self.preprocessed_splitter.setStretchFactor(0, 1); self.preprocessed_splitter.setStretchFactor(1, 1)
        self.preprocessed_splitter.setSizes((500, 500)); self.json_content.hide()
        self.tabs.addTab(preprocessed, "Preprocessed")

        results = QWidget(); results_layout = QVBoxLayout(results); results_layout.setContentsMargins(0, 0, 0, 0)
        results_actions = QHBoxLayout()
        self.run_prediction = QPushButton("Run Predict"); self.run_prediction.setEnabled(False)
        self.run_prediction.clicked.connect(self._configure_prediction)
        self.inspect_predictions = QPushButton("Inspect predictions")
        self.inspect_predictions.clicked.connect(self._configure_prediction_inspection)
        self.results_settings = QPushButton("⚙")
        self.results_settings.setObjectName("settingsButton")
        self.results_settings.setToolTip("Analysis and Napari settings")
        self.results_settings.clicked.connect(self._configure_workspace_settings)
        results_actions.addWidget(self.run_prediction); results_actions.addWidget(self.inspect_predictions)
        results_actions.addStretch(); results_actions.addWidget(self.results_settings); results_layout.addLayout(results_actions)
        self.results_splitter = QSplitter(); results_layout.addWidget(self.results_splitter)
        self.models = QTreeWidget(); self.models.setColumnCount(1); self.models.setHeaderHidden(True)
        self.models.currentItemChanged.connect(self._show_fold_summary)
        results_header = self.models.header()
        results_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.results_splitter.addWidget(self.models)
        self.result_summary = QPlainTextEdit(); self.result_summary.setReadOnly(True)
        self.results_splitter.addWidget(self.result_summary)
        self.results_splitter.setStretchFactor(0, 1); self.results_splitter.setStretchFactor(1, 1)
        self.results_splitter.setSizes((500, 500)); self.result_summary.hide()
        self.tabs.addTab(results, "Results")
        self._preload_nnunet_extensions()

    def show_annotation_documentation(self) -> None:
        """Show the bundled annotation-workflow PDF in the main workspace."""
        if self.documentation_content_layout.count() == 0:
            pdf_path = Path(__file__).parent.parent / "assets" / "Annotation_Workflow.pdf"
            try:
                from PySide6.QtPdf import QPdfDocument
                from PySide6.QtPdfWidgets import QPdfView

                document = QPdfDocument(self.documentation_content)
                document.load(str(pdf_path))
                view = QPdfView(self.documentation_content)
                view.setObjectName("annotationDocumentationView")
                view.setStyleSheet("background: #ffffff; border: 0;")
                view.setDocument(document)
                multi_page = getattr(QPdfView.PageMode, "MultiPage", None)
                if multi_page is not None:
                    view.setPageMode(multi_page)
                fit_to_width = getattr(QPdfView.ZoomMode, "FitToWidth", None)
                if fit_to_width is not None:
                    view.setZoomMode(fit_to_width)
                self.documentation_content_layout.addWidget(view)
            except ImportError:
                message = QLabel("PDF preview is unavailable in this PySide6 installation.")
                message.setWordWrap(True); self.documentation_content_layout.addWidget(message)

        self.summary.hide(); self.metadata_warning.hide(); self.consistency_warning.hide(); self.tabs.hide(); self.console_panel.hide()
        self.documentation_panel.show()
        self.annotation_documentation_visibility_changed.emit(True)

    def hide_annotation_documentation(self) -> None:
        self.documentation_panel.hide(); self.summary.show(); self.tabs.show()
        self.annotation_documentation_visibility_changed.emit(False)
        if self._active_process is not None or self._analysis_process is not None:
            self.console_panel.show()

    def set_annotation_workflow_enabled(self, enabled: bool) -> None:
        """Expose annotation-only actions only while that workflow is active."""
        self._annotation_workflow_enabled = enabled
        self.annotation_actions.setVisible(enabled)
        self._update_convert_annotations_enabled()
        self._update_next_iteration_enabled()
        self._refresh_raw_file_actions()
        current_item = self.raw_tree.currentItem()
        if current_item and current_item.data(0, self._RAW_STATISTICS_ROLE):
            self._show_raw_statistics(current_item, None)

    def _update_next_iteration_enabled(self) -> None:
        self.next_iteration.setEnabled(bool(self._annotation_workflow_enabled and self._record
                                             and self._record.raw_path and self._record.raw_path.is_dir()))

    def _refresh_raw_file_actions(self) -> None:
        """Update already-expanded raw folders when the workflow switch changes."""
        for index in range(self.raw_tree.topLevelItemCount()):
            folder = self.raw_tree.topLevelItem(index)
            folder_name = folder.data(0, self._RAW_FOLDER_ROLE)
            if not folder_name:
                continue
            for child_index in range(folder.childCount()):
                item = folder.child(child_index)
                path_text = item.toolTip(0)
                if path_text:
                    self.raw_tree.setItemWidget(item, 0, self._raw_file_row(Path(path_text), folder_name))

    def set_dataset(self, record: DatasetRecord | None) -> None:
        # Drop unstarted discovery work from the previous selection. Running tasks
        # are version-guarded and therefore cannot update the newly selected dataset.
        self._thread_pool.clear()
        self._dataset_version += 1
        self._plan_request += 1; self._summary_request += 1; self._training_request += 1; self._consistency_request += 1
        self._conversion_running = False
        self._record = record; self._loaded_tabs.clear(); self._loading_tabs.clear()
        self._metadata_errors = (); self.metadata_warning.hide(); self.metadata_error.hide()
        self._preprocessed_paths = []; self.run_training.setText("Run Training")
        self._trained_model_paths = []
        self._raw_folder_files = {}; self._loaded_raw_folders.clear(); self._raw_render_version += 1
        if self._active_process is None and self._analysis_process is None:
            self.console_panel.hide()
        self.json_content.hide(); self.result_summary.hide()
        raw_actions_enabled = record is not None and record.raw_path is not None
        self._update_convert_annotations_enabled()
        self._update_next_iteration_enabled()
        self.analyze_labels.setEnabled(raw_actions_enabled and not self._analysis_running)
        self.analyze_labels.setText("Analyzing labels…" if self._analysis_running else "Analyze Labels")
        self.analysis_settings.setEnabled(raw_actions_enabled)
        self._clear_raw_statistics()
        self._refresh_run_actions()
        if record is None:
            self.title.setText("Dataset")
            self._update_open_dataset_button()
            self.summary.setText("Select a dataset to browse its available data and outputs.")
            self.consistency_warning.hide()
            self._set_message(self.raw_tree, "No raw dataset selected.")
            self._set_message(self.preprocessed_tree, "No preprocessed dataset selected.")
            self._set_message(self.models, "No results dataset selected.")
            return
        availability = ", ".join(name for name, path in (("Raw", record.raw_path),
                                                            ("Preprocessed", record.preprocessed_path),
                                                            ("Results", record.results_path)) if path)
        self.title.setText(record.directory_name)
        self._update_open_dataset_button()
        self.summary.setText(f"Available: {availability or 'none'}")
        self.consistency_warning.hide()
        self._set_message(self.raw_tree, "Open the Raw tab to load files.")
        self._set_message(self.preprocessed_tree, "Open the Preprocessed tab to load files.")
        self._set_message(self.models, "Open the Results tab to load trained models.")
        self._load_active_tab()
        self._start_consistency_check(record)

    def _selected_dataset_folder(self) -> Path | None:
        if self._record is None:
            return None
        kind = ("raw", "preprocessed", "results")[self.tabs.currentIndex()]
        return getattr(self._record, f"{kind}_path")

    def _update_open_dataset_button(self) -> None:
        directory = self._selected_dataset_folder()
        self.open_dataset_folder.setEnabled(directory is not None and directory.is_dir())

    def _open_dataset_folder(self) -> None:
        directory = self._selected_dataset_folder()
        if directory is not None and directory.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory.resolve())))

    def _start_consistency_check(self, record: DatasetRecord) -> None:
        self._consistency_request += 1; request, version = self._consistency_request, self._dataset_version
        self._start_task(lambda: image_count_check(record.raw_path, record.preprocessed_path, record.results_path),
                         lambda check: self._show_consistency_check(request, version, check), lambda error: None)

    def _show_consistency_check(self, request: int, version: int, check: object) -> None:
        if request != self._consistency_request or version != self._dataset_version or not isinstance(check, ImageCountCheck):
            return
        if self._metadata_errors:
            self.consistency_warning.hide()
            return
        stored_warnings = tuple(filter(None, (self._conversion_warning(), self._label_analysis_warning())))
        if stored_warnings:
            self.consistency_warning.setText("\n".join(stored_warnings))
            self.consistency_warning.show()
            return
        if not check.labels_directory_exists:
            self.consistency_warning.setText("Warning: labelsTr folder is missing.")
            self.consistency_warning.show()
            return
        if check.labels == 0:
            self.consistency_warning.setText("Warning: labelsTr folder is empty.")
            self.consistency_warning.show()
            return
        if check.matches:
            self.consistency_warning.hide()
            return
        self.consistency_warning.setText(
            f"Warning: imagesTr contains {check.images} image file(s) across {check.channels} channel(s), "
            f"which does not match {check.labels} file(s) in labelsTr.")
        self.consistency_warning.show()

    def _label_analysis_warning(self) -> str | None:
        """Return a persistent warning stored by label analysis, if present."""
        if self._record is None:
            return None
        analysis = load_label_analysis(self._record.raw_path)
        unexpected = analysis.get("undefined_label_ids") if isinstance(analysis, dict) else None
        if not isinstance(unexpected, dict) or not unexpected:
            return None
        details = "; ".join(
            f"{filename}: {', '.join(str(class_id) for class_id in class_ids)}"
            for filename, class_ids in unexpected.items()
            if isinstance(filename, str) and isinstance(class_ids, list)
        )
        if not details:
            return None
        return (
            "Warning: labelsTr contains class IDs not defined in dataset.json. "
            f"Affected files (file: undefined IDs): {details}"
        )

    def _conversion_warning(self) -> str | None:
        """Return a session-only warning for annotations skipped during conversion."""
        if self._record is None or self._record.raw_path is None:
            return None
        details = self._session_conversion_warnings.get(self._record.raw_path, ())
        if not details:
            return None
        return "Warning: Some annotations were not converted. " + " ".join(details)

    def set_metadata_definition_errors(self, errors: tuple[str, ...]) -> None:
        """Show manifest-definition problems in the main pane, not only as field tooltips."""
        had_errors = bool(self._metadata_errors)
        self._metadata_errors = errors
        if errors:
            self.metadata_error.setText("Error: " + " ".join(errors))
            self.metadata_error.show(); self.consistency_warning.hide()
        else:
            self.metadata_error.hide()
            if had_errors and self._record is not None:
                self._start_consistency_check(self._record)

    def set_training_count_warning(self, message: str) -> None:
        """Show a live numTraining mismatch below the selected dataset name."""
        if message:
            self.metadata_warning.setText(f"Warning: {message}")
            self.metadata_warning.show()
        else:
            self.metadata_warning.hide()

    def refresh_consistency_check(self) -> None:
        if self._record is not None:
            self._start_consistency_check(self._record)

    def set_environment_roots(self, raw: Path | None, preprocessed: Path | None, results: Path | None) -> None:
        self._environment_roots = {"nnUNet_raw": raw, "nnUNet_preprocessed": preprocessed,
                                   "nnUNet_results": results}

    def _load_active_tab(self, _index: int | None = None) -> None:
        self._update_open_dataset_button()
        if self._record is None:
            return
        tab = ("raw", "preprocessed", "results")[self.tabs.currentIndex()]
        if tab in self._loaded_tabs or tab in self._loading_tabs:
            return
        record, version = self._record, self._dataset_version
        self._loading_tabs.add(tab)
        if tab == "raw":
            self._set_message(self.raw_tree, "Loading raw files…")
            task = lambda: raw_files(record.raw_path)
        elif tab == "preprocessed":
            self._set_message(self.preprocessed_tree, "Loading preprocessed files…")
            task = lambda: preprocessed_files(record.preprocessed_path)
        else:
            self._set_message(self.models, "Loading trained models…")
            task = lambda: [
                (model, [(fold, validation_files(fold), fold_training_status(fold))
                         for fold in model_folds(record.results_path / model)])
                for model in trained_models(record.results_path)
            ] if record.results_path else []
        self._start_task(task, lambda data: self._apply_tab_data(tab, version, record, data),
                         lambda error: self._apply_tab_error(tab, version, error))

    def _apply_tab_data(self, tab: str, version: int, record: DatasetRecord, data: object) -> None:
        if version != self._dataset_version or record != self._record:
            return
        self._loading_tabs.discard(tab); self._loaded_tabs.add(tab)
        if tab == "raw":
            self._render_raw(record.raw_path, data)
        elif tab == "preprocessed":
            self._render_preprocessed(record.preprocessed_path, data)
        else:
            self._render_results(data)

    def _apply_tab_error(self, tab: str, version: int, error: str) -> None:
        if version != self._dataset_version:
            return
        self._loading_tabs.discard(tab)
        tree = {"raw": self.raw_tree, "preprocessed": self.preprocessed_tree, "results": self.models}[tab]
        self._set_message(tree, f"Could not load content: {error}")

    def _render_raw(self, dataset_path: Path | None, data: object) -> None:
        self.raw_tree.clear()
        contents = data if isinstance(data, dict) else {}
        if not contents:
            message = "No raw dataset directory available." if dataset_path is None else "No raw folders found."
            self.raw_tree.addTopLevelItem(QTreeWidgetItem([message]))
            return
        self._raw_render_version += 1
        self._raw_folder_files = contents
        self._loaded_raw_folders.clear()
        for folder_name, files in contents.items():
            folder = QTreeWidgetItem([f"{folder_name} ({len(files)})"]); self.raw_tree.addTopLevelItem(folder)
            folder.setData(0, self._RAW_FOLDER_ROLE, folder_name)
            folder.setChildIndicatorPolicy(QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator)
            folder.setExpanded(False)
        if dataset_path and analysis_path(dataset_path).is_file():
            self._add_statistics_item(dataset_path)

    def _add_statistics_item(self, dataset_path: Path) -> None:
        target = str(analysis_path(dataset_path))
        for index in range(self.raw_tree.topLevelItemCount()):
            existing = self.raw_tree.topLevelItem(index)
            if existing.data(0, self._RAW_STATISTICS_ROLE) == target:
                return
        item = QTreeWidgetItem(["Label analysis"])
        item.setIcon(0, self._statistics_icon())
        item.setToolTip(0, target)
        item.setData(0, self._RAW_STATISTICS_ROLE, target)
        self.raw_tree.addTopLevelItem(item)

    @staticmethod
    def _statistics_icon() -> QIcon:
        """Return a compact bar-chart icon for cached statistics files."""
        pixmap = QPixmap(16, 16); pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(QColor("#2f6fca"))
        for x, height in ((2, 5), (6, 10), (10, 7)):
            painter.drawRoundedRect(x, 13 - height, 3, height, 1, 1)
        painter.end()
        return QIcon(pixmap)

    def _expand_raw_folder(self, folder: QTreeWidgetItem) -> None:
        folder_name = folder.data(0, self._RAW_FOLDER_ROLE)
        if not folder_name or folder_name in self._loaded_raw_folders:
            return
        self._loaded_raw_folders.add(folder_name)
        folder.addChild(QTreeWidgetItem(["Loading files…"]))
        version = self._raw_render_version
        QTimer.singleShot(0, lambda: self._append_raw_file_batch(folder, folder_name, 0, version))

    def _append_raw_file_batch(self, folder: QTreeWidgetItem, folder_name: str, start: int, version: int) -> None:
        if version != self._raw_render_version:
            return
        if start == 0:
            folder.takeChildren()
        files = self._raw_folder_files.get(folder_name, [])
        for path in files[start:start + 100]:
            item = QTreeWidgetItem([""]); item.setToolTip(0, str(path)); item.setSizeHint(0, QSize(0, 34))
            folder.addChild(item)
            self.raw_tree.setItemWidget(item, 0, self._raw_file_row(path, folder_name))
        next_start = start + 100
        if next_start < len(files):
            QTimer.singleShot(0, lambda: self._append_raw_file_batch(folder, folder_name, next_start, version))

    def _raw_file_row(self, path: Path, folder_name: str) -> QWidget:
        actions = {"imagesTr": ("Inspect", "Annotate"), "annotationsTr": ("Inspect", "Edit"),
                   "labelsTr": ("Inspect",)}.get(folder_name, ())
        if not self._annotation_workflow_enabled and folder_name in {"imagesTr", "annotationsTr"}:
            actions = ("Inspect",)
        widget, layout = self._file_row(path)
        for action in actions:
            if action == "Inspect":
                callback = lambda _=False, selected=path: self._inspect_file(selected)
            elif action in {"Annotate", "Edit"}:
                callback = lambda _=False, selected=path: self._annotate_file(selected)
            layout.addWidget(self._file_action_button(action, callback))
        return widget

    @staticmethod
    def _file_row(path: Path) -> tuple[QWidget, QHBoxLayout]:
        """Create one tree-column row with a filename and optional right-aligned actions."""
        widget = QWidget(); layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 4, 0); layout.setSpacing(4)
        filename = QLabel(path.name); filename.setToolTip(str(path))
        filename.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout.addWidget(filename, 1)
        return widget, layout

    @staticmethod
    def _file_action_button(label: str, callback: Callable[..., None]) -> QPushButton:
        """Create uniformly sized inline actions for all file trees."""
        button = QPushButton(label); button.setObjectName("rawActionButton")
        button.setFixedWidth(QPushButton("Annotate").sizeHint().width())
        button.clicked.connect(callback)
        return button

    def _inspect_file(self, path: Path) -> None:
        command = ["-m", "nnworkbench.services.napari_inspector", str(path),
                   "--prefetch-radius", str(self._napari_prefetch_radius)]
        if not self._napari_prefetch_previous:
            command.append("--no-prefetch-previous")
        if not self._napari_prefetch_next:
            command.append("--no-prefetch-next")
        self._start_napari(command, path)

    def _annotate_file(self, path: Path) -> None:
        self._start_napari(["-m", "nnworkbench.services.napari_inspector", str(path), "--annotate"], path)

    def _inspect_validation_file(self, path: Path) -> None:
        command = ["-m", "nnworkbench.services.napari_inspector", str(path), "--validation-prediction"]
        if self._record is not None and self._record.raw_path is not None:
            command.extend(("--raw-dataset", str(self._record.raw_path)))
        if not self._napari_prefetch_previous:
            command.append("--no-prefetch-previous")
        if not self._napari_prefetch_next:
            command.append("--no-prefetch-next")
        command.extend(("--prefetch-radius", str(self._napari_prefetch_radius)))
        self._start_napari(command, path)

    def _inspect_preprocessed_file(self, path: Path) -> None:
        command = ["-m", "nnworkbench.services.napari_inspector", str(path), "--preprocessed-case",
                   "--prefetch-radius", str(self._napari_prefetch_radius)]
        if not self._napari_prefetch_previous:
            command.append("--no-prefetch-previous")
        if not self._napari_prefetch_next:
            command.append("--no-prefetch-next")
        self._start_napari(command, path)

    def _configure_prediction_inspection(self) -> None:
        raw_dataset = self._record.raw_path if self._record else None
        dialog = PredictionInspectionDialog(raw_dataset, self)
        if not dialog.exec():
            return
        images, predictions, reference = dialog.directories()
        command = ["-m", "nnworkbench.services.napari_inspector", "--prediction-folder", str(predictions),
                   "--prefetch-radius", str(self._napari_prefetch_radius)]
        if images is not None:
            command.extend(("--image-folder", str(images)))
        if reference is not None:
            command.extend(("--reference-folder", str(reference)))
        if not self._napari_prefetch_previous:
            command.append("--no-prefetch-previous")
        if not self._napari_prefetch_next:
            command.append("--no-prefetch-next")
        self._start_napari(command, predictions)

    def _start_napari(self, command: list[str], path: Path) -> None:
        print(f"Napari: {shlex.join([sys.executable, *command])}")
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.ForwardedChannels)
        self._napari_processes.append(process)
        process.finished.connect(lambda _code, _status, active=process: self._forget_napari_process(active))
        process.errorOccurred.connect(lambda error, active=process: self._napari_failed(active, error))
        self._start_process_group(process, sys.executable, command)

    def _forget_napari_process(self, process: QProcess) -> None:
        if process in self._napari_processes:
            self._napari_processes.remove(process)
        process.deleteLater()

    def _napari_failed(self, process: QProcess, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self._forget_napari_process(process)
            QMessageBox.warning(self, "Could not start Napari", process.errorString())

    def shutdown(self) -> None:
        """Cancel queued loaders and stop all child processes before application exit."""
        if self._closing:
            return
        self._closing = True
        self._dataset_version += 1
        self._thread_pool.clear()  # Queued lazy loads are no longer useful at shutdown.
        self._extension_pool.clear()
        for process in tuple(self._processes) + tuple(self._napari_processes):
            self._stop_process_group(process, wait_ms=2_000)
        self._processes.clear(); self._napari_processes.clear()
        self._active_process = None; self._analysis_process = None; self._analysis_running = False

    @staticmethod
    def _start_process_group(process: QProcess, program: str, arguments: list[str]) -> None:
        """Launch a child in a separate group so its worker children can be stopped too."""
        process.setProperty("isolated_process_group", False)
        if os.name == "posix":
            process.setProperty("isolated_process_group", True)
            process.start(sys.executable, ["-m", "nnworkbench.services.process_runner", program, *arguments])
        else:
            process.start(program, arguments)

    @staticmethod
    def _stop_process_group(process: QProcess, wait_ms: int = 0) -> None:
        if process.state() == QProcess.ProcessState.NotRunning:
            return
        process_id = process.processId()
        if (os.name == "posix" and bool(process.property("isolated_process_group"))
                and process_id > 0):
            try:
                os.killpg(process_id, signal.SIGTERM)
            except ProcessLookupError:
                return
            except PermissionError:
                process.terminate()
        elif os.name == "nt" and process_id > 0:
            # QProcess only owns the direct child on Windows. taskkill's tree mode
            # also closes worker processes created by a command launched by the GUI.
            subprocess.run(["taskkill", "/PID", str(process_id), "/T"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        else:
            process.terminate()
        if wait_ms and not process.waitForFinished(wait_ms):
            if (os.name == "posix" and bool(process.property("isolated_process_group"))
                    and process_id > 0):
                try:
                    os.killpg(process_id, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    process.kill()
            elif os.name == "nt" and process_id > 0:
                subprocess.run(["taskkill", "/PID", str(process_id), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            else:
                process.kill()
            process.waitForFinished(1_000)

    def _print_raw_action(self, action: str) -> None:
        if self._record and self._record.raw_path:
            print(f"{action}: {self._record.raw_path}")

    def _analyze_labels(self) -> None:
        if self._record is None or self._record.raw_path is None or self._analysis_running:
            return
        files = label_files(self._record.raw_path)
        if not files:
            QMessageBox.information(self, "No labels found", "No NIfTI files were found in labelsTr.")
            return
        raw_path, workers = self._record.raw_path, self._analysis_workers
        self._analysis_running = True; self.analyze_labels.setEnabled(False); self.analyze_labels.setText("Analyzing labels…")
        self._log(f"Run: label analysis for {raw_path} ({len(files)} file(s), up to {workers} worker(s))")
        process = QProcess(self)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(environment)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._processes.append(process); self._analysis_process = process; self._show_console()
        process.readyReadStandardOutput.connect(lambda active=process: self._read_process_output(active))
        process.finished.connect(
            lambda code, status, active=process, directory=raw_path:
            self._analysis_process_finished(active, directory, code, status))
        process.errorOccurred.connect(lambda error, active=process: self._analysis_process_failed(active, error))
        self._start_process_group(
            process, sys.executable,
            ["-m", "nnworkbench.services.label_analysis", str(raw_path), "--workers", str(workers)],
        )

    def _convert_annotations(self) -> None:
        if self._record is None or self._record.raw_path is None or self._conversion_running:
            return
        raw_path, workers = self._record.raw_path, self._analysis_workers
        files = annotation_files(raw_path)
        if not files:
            QMessageBox.information(self, "No annotations found", "No NIfTI files were found in annotationsTr.")
            return
        self._conversion_request += 1; request, version = self._conversion_request, self._dataset_version
        self._conversion_running = True
        self.convert_annotations.setEnabled(False); self.convert_annotations.setText("Converting annotations…")
        availability = ", ".join(name for name, path in (("Raw", self._record.raw_path),
                                                            ("Preprocessed", self._record.preprocessed_path),
                                                            ("Results", self._record.results_path)) if path)
        self.summary.setText(
            f"Available: {availability}\nConverting {len(files)} annotation files in the background with up to {workers} workers. "
            "This can take a while for large datasets."
        )
        self._start_task(lambda: convert_annotations(raw_path, workers),
                         lambda result: self._conversion_finished(request, version, result),
                         lambda error: self._conversion_failed(request, version, error))

    def _update_convert_annotations_enabled(self) -> None:
        raw_path = self._record.raw_path if self._record else None
        self.convert_annotations.setEnabled(
            self._annotation_workflow_enabled and raw_path is not None and not self._conversion_running
            and has_annotation_files(raw_path)
        )

    def _conversion_finished(self, request: int, version: int, result: object) -> None:
        if request != self._conversion_request or version != self._dataset_version:
            return
        self._conversion_running = False
        self.convert_annotations.setText("Convert Annotations"); self._update_convert_annotations_enabled()
        self._restore_availability_summary()
        if not isinstance(result, AnnotationConversionResult):
            return
        for warning in result.warnings:
            self._log(f"Annotation conversion warning: {warning}")
        if self._record is not None and self._record.raw_path is not None:
            self._session_conversion_warnings[self._record.raw_path] = result.warnings
        self._loaded_tabs.discard("raw")
        if self.tabs.currentIndex() == 0:
            self._load_active_tab()
        self._start_consistency_check(self._record)
        message = f"Converted {len(result.converted)} annotation file(s) into labelsTr using {result.workers} worker(s)."
        if result.warnings:
            message += "\n\nSkipped files:\n" + "\n".join(result.warnings)
            QMessageBox.warning(self, "Some annotations were not converted", message)
        else:
            QMessageBox.information(self, "Annotation conversion completed", message)

    def _conversion_failed(self, request: int, version: int, error: str) -> None:
        if request != self._conversion_request or version != self._dataset_version:
            return
        self._conversion_running = False
        self.convert_annotations.setText("Convert Annotations"); self._update_convert_annotations_enabled()
        self._restore_availability_summary()
        QMessageBox.warning(self, "Could not convert annotations", error)

    def _configure_workspace_settings(self) -> None:
        dialog = QDialog(self); dialog.setWindowTitle("Raw workspace settings")
        layout = QVBoxLayout(dialog)
        analysis_group = QGroupBox("Label analysis and annotation conversion"); analysis_form = QFormLayout(analysis_group)
        workers = _SettingsSpinBox(); workers.setObjectName("settingsSpinBox")
        workers.setRange(1, os.cpu_count() or 1); workers.setValue(self._analysis_workers)
        workers.setToolTip("Used by both Analyze Labels and Convert Annotations.")
        analysis_form.addRow("Worker processes (both actions):", workers); layout.addWidget(analysis_group)
        napari_group = QGroupBox("Napari visualization"); napari_form = QFormLayout(napari_group)
        previous = QCheckBox("Prefetch previous cases"); previous.setChecked(self._napari_prefetch_previous)
        following = QCheckBox("Prefetch next cases"); following.setChecked(self._napari_prefetch_next)
        radius = _SettingsSpinBox(); radius.setObjectName("settingsSpinBox")
        radius.setRange(1, 20); radius.setValue(self._napari_prefetch_radius)
        napari_form.addRow(previous); napari_form.addRow(following); napari_form.addRow("Prefetch radius:", radius)
        layout.addWidget(napari_group)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Save)
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject); layout.addWidget(buttons)
        if not dialog.exec():
            return
        self._analysis_workers = workers.value()
        self._napari_prefetch_previous = previous.isChecked()
        self._napari_prefetch_next = following.isChecked()
        self._napari_prefetch_radius = radius.value()

    def _analysis_process_finished(self, process: QProcess, raw_path: Path, exit_code: int,
                                   status: QProcess.ExitStatus) -> None:
        self._read_process_output(process)
        if process in self._processes:
            self._processes.remove(process)
        if process is not self._analysis_process:
            return
        self._analysis_process = None; self._analysis_running = False
        self._update_console_controls()
        outcome = "completed" if status == QProcess.ExitStatus.NormalExit and exit_code == 0 else "failed"
        self._log(f"Label analysis {outcome} (exit code {exit_code}).")
        self.analyze_labels.setText("Analyze Labels")
        self.analyze_labels.setEnabled(self._record is not None and self._record.raw_path is not None)
        if self._record is None or self._record.raw_path != raw_path:
            return
        self._restore_availability_summary()
        if exit_code == 0:
            self._start_consistency_check(self._record)
            if "raw" in self._loaded_tabs:
                self._add_statistics_item(raw_path)
        else:
            QMessageBox.warning(self, "Label analysis failed", "See the console for details.")

    def _analysis_process_failed(self, process: QProcess, error: QProcess.ProcessError) -> None:
        if error != QProcess.ProcessError.FailedToStart:
            return
        if process in self._processes:
            self._processes.remove(process)
        if process is not self._analysis_process:
            return
        self._analysis_process = None; self._analysis_running = False
        self._update_console_controls()
        self._log(f"Could not start label analysis: {process.errorString()}")
        self.analyze_labels.setText("Analyze Labels")
        self.analyze_labels.setEnabled(self._record is not None and self._record.raw_path is not None)

    def _show_raw_statistics(self, item: QTreeWidgetItem | None, previous: QTreeWidgetItem | None) -> None:
        path = item.data(0, self._RAW_STATISTICS_ROLE) if item else None
        if not path:
            self._clear_raw_statistics()
            return
        analysis = load_label_analysis(Path(path).parent)
        if analysis is None:
            self._clear_raw_statistics()
            return
        self._clear_raw_statistics()
        statistics_view = LabelStatisticsView(
            analysis, annotation_workflow_enabled=self._annotation_workflow_enabled,
            parent=self.raw_statistics_container,
        )
        self.raw_statistics_layout.addWidget(statistics_view)
        self.raw_statistics_scroll.show()
        self._ignore_next_statistics_click = item
        QTimer.singleShot(0, self._split_raw_statistics)

    def _toggle_raw_statistics(self, item: QTreeWidgetItem, _column: int) -> None:
        """Hide the statistics panel when its already-selected tree entry is clicked again."""
        if not item.data(0, self._RAW_STATISTICS_ROLE):
            return
        if self._ignore_next_statistics_click is item:
            self._ignore_next_statistics_click = None
            return
        self._clear_raw_statistics()
        self.raw_tree.setCurrentItem(None)

    def _clear_raw_statistics(self) -> None:
        while self.raw_statistics_layout.count():
            child = self.raw_statistics_layout.takeAt(0).widget()
            if child is not None:
                child.deleteLater()
        self.raw_statistics_scroll.hide()

    def _split_raw_statistics(self) -> None:
        tree_height = self.raw_splitter.height() * 3 // 10
        self.raw_splitter.setSizes((tree_height, max(1, self.raw_splitter.height() - tree_height)))

    def _restore_availability_summary(self) -> None:
        if self._record is None:
            return
        availability = ", ".join(name for name, path in (("Raw", self._record.raw_path),
                                                             ("Preprocessed", self._record.preprocessed_path),
                                                             ("Results", self._record.results_path)) if path)
        self.summary.setText(f"Available: {availability or 'none'}")

    def _preload_nnunet_extensions(self) -> None:
        """Discover optional planner/trainer classes without delaying the GUI."""
        self._load_planners(); self._load_trainers()

    def _load_planners(self) -> None:
        if self._planners_loaded or self._planners_loading:
            return
        self._planners_loading = True
        self._start_extension_task(available_planners, self._planners_ready,
                                   lambda _error: self._planners_ready(("ExperimentPlanner",)))

    def _load_trainers(self) -> None:
        if self._trainers_loaded or self._trainers_loading:
            return
        self._trainers_loading = True
        self._start_extension_task(available_trainers, self._trainers_ready,
                                   lambda _error: self._trainers_ready(("nnUNetTrainer",)))

    def _planners_ready(self, planners: object) -> None:
        self._planners_loading = False; self._planners_loaded = True
        choices = tuple(item for item in planners if isinstance(item, str)) if isinstance(planners, tuple) else ()
        self._available_planners = choices or ("ExperimentPlanner",)
        if self._pending_preprocessing is not None:
            request, version = self._pending_preprocessing; self._pending_preprocessing = None
            self._show_preprocessing_dialog(request, version, self._available_planners)

    def _trainers_ready(self, trainers: object) -> None:
        self._trainers_loading = False; self._trainers_loaded = True
        choices = tuple(item for item in trainers if isinstance(item, str)) if isinstance(trainers, tuple) else ()
        self._available_trainers = choices or ("nnUNetTrainer",)
        if self._pending_training is not None:
            request, version, plan_data = self._pending_training; self._pending_training = None
            self._show_training_dialog(request, version, plan_data)

    def _configure_preprocessing(self) -> None:
        if self._record is None:
            return
        self._planner_request += 1; request, version = self._planner_request, self._dataset_version
        if self._planners_loaded:
            self._show_preprocessing_dialog(request, version, self._available_planners)
            return
        self.run_preprocessing.setEnabled(False); self.run_preprocessing.setText("Loading planners…")
        self._pending_preprocessing = request, version
        self._load_planners()

    def _show_preprocessing_dialog(self, request: int, version: int, planners: object) -> None:
        if request != self._planner_request or version != self._dataset_version or self._record is None:
            return
        self.run_preprocessing.setText("Run Preprocessing"); self._refresh_run_actions()
        choices = tuple(item for item in planners if isinstance(item, str)) if isinstance(planners, tuple) else ()
        dialog = PreprocessingDialog(
            self._record.dataset_id,
            choices or ("ExperimentPlanner",),
            local_run_available=self._local_preprocessing_available(),
            parent=self,
        )
        if dialog.exec():
            self._run_nnunet_command(dialog.command(), ("nnUNet_raw", "nnUNet_preprocessed"))

    def _configure_training(self) -> None:
        if self._record is None:
            return
        plan_paths = [path for path in self._preprocessed_paths
                      if path.suffix.casefold() == ".json" and "plan" in path.name.casefold()]
        self._training_request += 1; request, version = self._training_request, self._dataset_version
        if not plan_paths or self._record.preprocessed_path is None:
            self._show_training_dialog(request, version, [])
            return
        root = self._record.preprocessed_path
        self.run_training.setEnabled(False); self.run_training.setText("Loading plans…")
        self._start_task(lambda: [(path, json_value(root / path)) for path in plan_paths],
                         lambda data: self._show_training_dialog(request, version, data),
                         lambda error: self._show_training_error(request, version, error))

    def _show_training_dialog(self, request: int, version: int, data: object) -> None:
        if request != self._training_request or version != self._dataset_version:
            return
        if not self._trainers_loaded:
            self.run_training.setText("Loading trainers…")
            self._pending_training = request, version, data
            self._load_trainers()
            return
        self.run_training.setText("Run Training"); self._refresh_run_actions()
        plan_data = data if isinstance(data, list) else ()
        plans: list[tuple[str, tuple[str, ...]]] = []
        for path, content in plan_data:
            configurations = content.get("configurations", {}) if isinstance(content, dict) else {}
            names = tuple(configurations) if isinstance(configurations, dict) else ()
            if names:
                plans.append((str(path), names))
        has_local_plans = bool(plans)
        if not plans:
            plans = [("nnUNetPlans.json", ("2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"))]
        default_trainer = ("nnUNetTrainerBetterIgnoreSampling" if self._annotation_workflow_enabled
                           else "nnUNetTrainer")
        dialog = TrainingDialog(self._record.dataset_id, plans, self._available_trainers,
                                default_trainer=default_trainer,
                                annotation_workflow_enabled=self._annotation_workflow_enabled,
                                local_run_available=has_local_plans and self._local_run_available(
                                    ("nnUNet_preprocessed", "nnUNet_results")), parent=self)
        if dialog.exec() and self._record:
            self._run_nnunet_command(dialog.command(), ("nnUNet_preprocessed", "nnUNet_results"))

    def _show_training_error(self, request: int, version: int, error: str) -> None:
        if request == self._training_request and version == self._dataset_version:
            self.run_training.setText("Run Training"); self._refresh_run_actions()
            print(f"Could not load training plans: {error}")

    def _configure_prediction(self) -> None:
        options = [option for path in self._trained_model_paths
                   if (option := model_trainer_plan_configuration(path))]
        has_local_models = bool(options)
        if not options:
            options = [("nnUNetTrainer", "nnUNetPlans", configuration)
                       for configuration in ("2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres")]
        dialog = PredictionDialog(
            self._record.dataset_id,
            options,
            local_run_available=has_local_models and self._local_run_available(("nnUNet_results",)),
            parent=self,
        )
        if dialog.exec() and self._record:
            self._run_nnunet_command(dialog.command(), ("nnUNet_results",))

    def _run_nnunet_command(self, command: str, required_roots: tuple[str, ...]) -> None:
        if self._active_process is not None:
            QMessageBox.warning(self, "nnU-Net command running",
                                "Wait for the active nnU-Net command to finish before starting another one.")
            return
        invalid = [name for name in required_roots
                   if self._environment_roots[name] is None or not self._environment_roots[name].is_dir()]
        if invalid:
            QMessageBox.warning(self, "nnU-Net paths required",
                                f"Configure existing directories before running nnU-Net:\n{', '.join(invalid)}")
            return
        from nnworkbench.path_setup import save_paths
        save_paths(*(str(self._environment_roots[name]) if self._environment_roots[name] else ""
                     for name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        self._log("export PYTHONUNBUFFERED=1")
        from nnworkbench.services.commands import wrapped_command
        command = wrapped_command(command)
        self._log(f"Run: {command}")
        process = QProcess(self); process.setProcessEnvironment(environment)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._processes.append(process); self._active_process = process; self._show_console(); self._refresh_run_actions()
        process.readyReadStandardOutput.connect(lambda active=process: self._read_process_output(active))
        process.finished.connect(lambda code, status, active=process: self._command_finished(active, code, status))
        process.errorOccurred.connect(lambda error, active=process: self._command_failed(active, error))
        arguments = shlex.split(command)
        self._start_process_group(process, arguments[0], arguments[1:])

    def _local_run_available(self, required_roots: tuple[str, ...]) -> bool:
        return all(self._environment_roots[name] is not None and self._environment_roots[name].is_dir()
                   for name in required_roots)

    def _local_preprocessing_available(self) -> bool:
        raw_path = self._record.raw_path if self._record else None
        images = raw_path / "imagesTr" if raw_path is not None else None
        return (self._local_run_available(("nnUNet_raw", "nnUNet_preprocessed"))
                and images is not None and images.is_dir() and any(images.iterdir()))

    def _command_finished(self, process: QProcess, exit_code: int, status: QProcess.ExitStatus) -> None:
        self._read_process_output(process)
        if process in self._processes:
            self._processes.remove(process)
        if process is self._active_process:
            self._active_process = None; self._update_console_controls(); self._refresh_run_actions()
        outcome = "completed" if status == QProcess.ExitStatus.NormalExit and exit_code == 0 else "failed"
        self._log(f"nnU-Net command {outcome} (exit code {exit_code}).")
        self.nnunet_command_finished.emit()

    def _command_failed(self, process: QProcess, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            if process in self._processes:
                self._processes.remove(process)
            if process is self._active_process:
                self._active_process = None; self._update_console_controls(); self._refresh_run_actions()
            self._log(f"Could not start nnU-Net command: {process.errorString()}")

    def _terminate_command(self) -> None:
        process = self._active_process or self._analysis_process
        if process is None:
            return
        description = "nnU-Net command" if process is self._active_process else "label analysis"
        self._log(f"Termination requested for active {description}.")
        self.terminate_command.setEnabled(False)
        self._stop_process_group(process)
        # Some frameworks take a moment to drain their workers after SIGTERM.
        QTimer.singleShot(5_000, lambda active=process: self._stop_process_group(active, wait_ms=1))

    def _read_process_output(self, process: QProcess) -> None:
        output = bytes(process.readAllStandardOutput()).decode(errors="replace")
        if output:
            print(output, end="")
            cursor = self.console.textCursor(); cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertText(output); self.console.setTextCursor(cursor)
            self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

    def _log(self, message: str) -> None:
        print(message)
        self.console.appendPlainText(message)

    def _show_console(self) -> None:
        self.console_panel.show()
        console_height = max(160, self.content_splitter.height() // 3)
        self.content_splitter.setSizes((max(1, self.content_splitter.height() - console_height), console_height))
        self._update_console_controls()

    def _update_console_controls(self) -> None:
        running = self._active_process is not None or self._analysis_process is not None
        self.terminate_command.setEnabled(running)
        self.hide_console.setEnabled(not running)

    def _refresh_run_actions(self) -> None:
        running = self._active_process is not None
        has_dataset = self._record is not None
        self.run_preprocessing.setEnabled(has_dataset and not running)
        self.run_training.setEnabled(has_dataset and not running)
        self.run_prediction.setEnabled(has_dataset and not running)
        # Inspection accepts any user-selected prediction folder, including outputs
        # produced on another machine or outside the configured Results directory.
        self.inspect_predictions.setEnabled(True)

    def _render_preprocessed(self, dataset_path: Path | None, data: object) -> None:
        self.preprocessed_tree.clear(); self.json_content.hide()
        paths = data if isinstance(data, list) else []
        self._preprocessed_paths = paths
        self._refresh_run_actions()
        if not paths:
            message = "No preprocessed dataset directory available." if dataset_path is None else "No preprocessed files found."
            self.preprocessed_tree.addTopLevelItem(QTreeWidgetItem([message]))
            return
        assert dataset_path is not None
        file_counts: dict[tuple[str, ...], int] = {}
        for relative_path in paths:
            for depth in range(1, len(relative_path.parts)):
                folder_parts = relative_path.parts[:depth]
                file_counts[folder_parts] = file_counts.get(folder_parts, 0) + 1
        folders: dict[tuple[str, ...], QTreeWidgetItem] = {}
        for relative_path in paths:
            parent = self.preprocessed_tree.invisibleRootItem(); parts = relative_path.parts
            for depth, folder_name in enumerate(parts[:-1], start=1):
                folder_parts = parts[:depth]; folder = folders.get(folder_parts)
                if folder is None:
                    folder = QTreeWidgetItem([f"{folder_name} ({file_counts[folder_parts]})"])
                    parent.addChild(folder); folder.setExpanded(False); folders[folder_parts] = folder
                parent = folder
            file_item = QTreeWidgetItem([relative_path.name]); parent.addChild(file_item)
            target = dataset_path / relative_path
            file_item.setData(0, Qt.ItemDataRole.UserRole, str(target))
            if target.name.endswith(".b2nd") and not target.name.endswith("_seg.b2nd"):
                file_item.setText(0, "")
                file_item.setSizeHint(0, QSize(0, 34))
                self.preprocessed_tree.setItemWidget(file_item, 0, self._preprocessed_file_row(target))

    def _preprocessed_file_row(self, path: Path) -> QWidget:
        """Offer inspection for a B2ND data file; the matching segmentation loads automatically."""
        widget, layout = self._file_row(path)
        layout.addWidget(self._file_action_button(
            "Inspect", lambda _=False, selected=path: self._inspect_preprocessed_file(selected)))
        return widget

    def _render_results(self, data: object) -> None:
        self.models.clear(); self.result_summary.hide()
        models = data if isinstance(data, list) else []
        self._trained_model_paths = [path for path, _folds in models]
        self._refresh_run_actions()
        if not models:
            self.models.addTopLevelItem(QTreeWidgetItem(["No trained models found."]))
            return
        assert self._record is not None and self._record.results_path is not None
        for relative_model_path, folds in models:
            model_item = QTreeWidgetItem([str(relative_model_path)]); self.models.addTopLevelItem(model_item)
            for fold_path, files, status in folds:
                suffix = "(complete)" if status.complete else (
                    f"(epoch {status.epoch})" if status.epoch is not None else ""
                )
                fold_item = QTreeWidgetItem([" ".join(part for part in (fold_path.name, suffix) if part)])
                model_item.addChild(fold_item)
                fold_item.setData(0, Qt.ItemDataRole.UserRole, str(fold_path))
                fold_item.setData(0, self._FOLD_ROLE, True)
                validation = fold_path / "validation"
                if validation.is_dir():
                    validation_item = QTreeWidgetItem([f"validation ({len(files)})"])
                    validation_item.setToolTip(0, str(validation)); validation_item.setExpanded(False)
                    fold_item.addChild(validation_item)
                    self._add_validation_files(validation_item, validation, files)
            model_item.setExpanded(False)

    def _add_validation_files(self, parent: QTreeWidgetItem, validation: Path, files: list[Path]) -> None:
        """Render validation files, keeping nested directories available but collapsed."""
        folders: dict[tuple[str, ...], QTreeWidgetItem] = {}
        for relative_path in files:
            container = parent
            for depth, part in enumerate(relative_path.parts[:-1], start=1):
                parts = relative_path.parts[:depth]
                folder = folders.get(parts)
                if folder is None:
                    folder = QTreeWidgetItem([part]); folder.setExpanded(False)
                    container.addChild(folder); folders[parts] = folder
                container = folder
            target = validation / relative_path
            item = QTreeWidgetItem([relative_path.name]); item.setToolTip(0, str(target))
            item.setSizeHint(0, QSize(0, 34))
            item.setData(0, Qt.ItemDataRole.UserRole, str(target))
            item.setData(0, self._VALIDATION_FILE_ROLE, True)
            container.addChild(item)
            if target.name.endswith((".nii", ".nii.gz")):
                item.setText(0, "")
                self.models.setItemWidget(item, 0, self._validation_file_row(target))

    def _validation_file_row(self, path: Path) -> QWidget:
        """Match labelsTr's Inspect-button layout in the Results tree."""
        widget, layout = self._file_row(path)
        layout.addWidget(self._file_action_button(
            "Inspect", lambda _=False, selected=path: self._inspect_validation_file(selected)))
        return widget

    def _show_json_content(self, item: QTreeWidgetItem | None, previous: QTreeWidgetItem | None) -> None:
        file_path = self._item_path(item)
        if file_path is None or file_path.suffix.casefold() != ".json":
            self.json_content.hide()
            return
        self._plan_request += 1; request, version = self._plan_request, self._dataset_version
        self.json_content.show(); self.json_content.setPlainText("Loading JSON…")
        QTimer.singleShot(0, lambda: self._prioritize_content(self.preprocessed_splitter))
        self._start_task(lambda: json_value(file_path), lambda data: self._apply_json_content(request, version, file_path, data),
                         lambda error: self._apply_json_error(request, version, file_path, error))

    def _apply_json_content(self, request: int, version: int, file_path: Path, data: object) -> None:
        if request != self._plan_request or version != self._dataset_version:
            return
        self.json_content.setPlainText(json.dumps(data, indent=2, ensure_ascii=False) if data is not None
                                        else f"{file_path.name} is missing or is not valid JSON.")

    def _apply_json_error(self, request: int, version: int, file_path: Path, error: str) -> None:
        if request == self._plan_request and version == self._dataset_version:
            self.json_content.setPlainText(f"Could not load {file_path.name}: {error}")

    def _show_fold_summary(self, item: QTreeWidgetItem | None, previous: QTreeWidgetItem | None) -> None:
        if item is None or not item.data(0, self._FOLD_ROLE):
            self.result_summary.hide()
            return
        fold_path = self._item_path(item)
        if fold_path is None:
            self.result_summary.hide()
            return
        self._summary_request += 1; request, version = self._summary_request, self._dataset_version
        self.result_summary.show(); self.result_summary.setPlainText("Loading validation summary…")
        QTimer.singleShot(0, lambda: self._prioritize_content(self.results_splitter))

        def task() -> tuple[str, object | None]:
            validation = fold_path / "validation"
            if not validation.is_dir():
                return "missing_validation", None
            return "summary", json_value(validation / "summary.json")

        self._start_task(task, lambda data: self._apply_summary(request, version, fold_path, data),
                         lambda error: self._apply_summary_error(request, version, fold_path, error))

    def _apply_summary(self, request: int, version: int, fold_path: Path, data: object) -> None:
        if request != self._summary_request or version != self._dataset_version:
            return
        status, summary = data
        if status == "missing_validation":
            self.result_summary.setPlainText(f"No validation folder is available for {fold_path.name}.")
        elif summary is None:
            self.result_summary.setPlainText(
                f"A validation folder exists for {fold_path.name}, but summary.json is missing or is not valid JSON.")
        else:
            self.result_summary.setPlainText(json.dumps(summary, indent=2, ensure_ascii=False))

    def _apply_summary_error(self, request: int, version: int, fold_path: Path, error: str) -> None:
        if request == self._summary_request and version == self._dataset_version:
            self.result_summary.setPlainText(f"Could not load {fold_path.name}: {error}")

    def _start_task(self, task: Callable[[], object], complete: Callable[[object], None],
                    failed: Callable[[str], None]) -> None:
        if self._closing:
            return
        worker = _LoadWorker(task, lambda: self._closing); self._workers.append(worker)

        def finish(result: object) -> None:
            if worker in self._workers:
                self._workers.remove(worker)
            complete(result)

        def fail(error: str) -> None:
            if worker in self._workers:
                self._workers.remove(worker)
            failed(error)

        worker.signals.completed.connect(finish); worker.signals.failed.connect(fail)
        self._thread_pool.start(worker)

    def _start_extension_task(self, task: Callable[[], object], complete: Callable[[object], None],
                              failed: Callable[[str], None]) -> None:
        """Run startup extension discovery independently from cancellable file loading."""
        if self._closing:
            return
        worker = _LoadWorker(task, lambda: self._closing); self._extension_workers.append(worker)

        def finish(result: object) -> None:
            if worker in self._extension_workers:
                self._extension_workers.remove(worker)
            complete(result)

        def fail(error: str) -> None:
            if worker in self._extension_workers:
                self._extension_workers.remove(worker)
            failed(error)

        worker.signals.completed.connect(finish); worker.signals.failed.connect(fail)
        self._extension_pool.start(worker)

    @staticmethod
    def _item_path(item: QTreeWidgetItem | None) -> Path | None:
        value = item.data(0, Qt.ItemDataRole.UserRole) if item else None
        return Path(value) if value else None

    @staticmethod
    def _set_message(tree: QTreeWidget, message: str) -> None:
        tree.clear(); tree.addTopLevelItem(QTreeWidgetItem([message]))

    @staticmethod
    def _prioritize_content(splitter: QSplitter) -> None:
        """Keep navigation compact and reserve most space for the selected content."""
        navigation_width = max(280, splitter.width() // 3)
        splitter.setSizes((navigation_width, max(1, splitter.width() - navigation_width)))
