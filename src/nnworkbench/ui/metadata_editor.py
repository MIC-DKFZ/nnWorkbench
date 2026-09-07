from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, QTimer, Signal
from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nnworkbench.services.dataset_json import (
    labels_tr_count,
    manifest_source,
    read_manifest,
    write_manifest,
)


class _ChannelIdDelegate(QStyledItemDelegate):
    """Restrict channel ID editing to non-negative integers."""

    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        editor.setValidator(QIntValidator(0, 2_147_483_647, editor))
        return editor


class MetadataEditor(QWidget):
    definition_errors_changed = Signal(tuple)
    training_count_warning_changed = Signal(str)
    metadata_saved = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("metadataEditor")
        self._directory: Path | None = None
        self._source_directory: Path | None = None
        self._source_kind: str | None = None
        self._manifest: dict = {}
        self._annotation_workflow_enabled = False
        self._loading = False
        self._saving = False
        self._pending_saves: set[str] = set()
        layout = QVBoxLayout(self); layout.setContentsMargins(16, 16, 16, 16); layout.setSpacing(10)
        self.notice = QLabel("Select a dataset with a raw dataset.json to edit metadata.")
        self.notice.setObjectName("metadataNotice"); self.notice.setWordWrap(True)
        self.name = QLineEdit(); self.name.setReadOnly(True)
        self.training = QLineEdit(); self.training.setObjectName("trainingCount")
        self.training.setValidator(QIntValidator(0, 2_147_483_647, self.training))
        self.training.textChanged.connect(self._update_training_validation)
        self.training.editingFinished.connect(lambda: self._auto_save("training"))
        self.channels = self._table(); self.channels.setObjectName("channelTable")
        self.labels = self._table(); self.labels.setObjectName("labelTable")
        self.channels.itemChanged.connect(lambda _item: self._table_changed("channels"))
        self.labels.itemChanged.connect(lambda _item: self._table_changed("labels"))
        self.file_ending = QLineEdit()
        self.file_ending.editingFinished.connect(lambda: self._auto_save("file_ending"))
        self.description = QPlainTextEdit(); self.description.installEventFilter(self)
        self.license = QLineEdit()
        self.license.editingFinished.connect(lambda: self._auto_save("license"))
        form = QFormLayout()
        form.addRow("Name", self.name); form.addRow("numTraining", self.training)
        self.channels.setItemDelegateForColumn(0, _ChannelIdDelegate(self.channels))
        self.labels.setItemDelegateForColumn(0, _ChannelIdDelegate(self.labels))
        self.channel_controls = self._table_controls(self.channels, channel_ids=True)
        form.addRow("Channels", self.channel_controls)
        form.addRow("File ending", self.file_ending)
        self.label_controls = self._table_controls(self.labels, channel_ids=False)
        form.addRow("Labels", self.label_controls)
        form.addRow("Description", self.description); form.addRow("License", self.license)
        layout.addWidget(self.notice); layout.addLayout(form)
        self.set_dataset(None)

    @staticmethod
    def _table() -> QTableWidget:
        table = QTableWidget(0, 2)
        table.setMaximumHeight(180)
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setVisible(False)
        table.setColumnWidth(0, 56)
        table.setAlternatingRowColors(True)
        vertical_header = table.verticalHeader()
        vertical_header.setVisible(False)
        vertical_header.setDefaultSectionSize(32)
        return table

    def _table_controls(self, table: QTableWidget, *, channel_ids: bool) -> QWidget:
        widget = QWidget(); layout = QVBoxLayout(widget); layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(table)
        controls = QHBoxLayout(); add = QPushButton("Add"); remove = QPushButton("Remove selected")
        field = "channels" if channel_ids else "labels"
        add.clicked.connect(lambda: self._add_row(table, channel_ids=channel_ids))
        remove.clicked.connect(lambda: self._remove_selected_row(table, field))
        controls.addWidget(add); controls.addWidget(remove); controls.addStretch(); layout.addLayout(controls)
        return widget

    def set_dataset(self, raw_directory: Path | None, preprocessed_directory: Path | None = None,
                    results_directory: Path | None = None) -> None:
        self._loading = True
        source = manifest_source(raw_directory, preprocessed_directory, results_directory)
        create_raw_manifest = bool(raw_directory and raw_directory.is_dir()
                                   and not (raw_directory / "dataset.json").is_file())
        self._source_kind, self._source_directory = (
            ("raw", raw_directory) if create_raw_manifest else source if source else (None, None)
        )
        self._directory = self._source_directory if self._source_kind == "raw" else None
        self._set_editing_enabled(False)
        if self._source_directory is None:
            self.notice.setText("No readable dataset.json is available in raw, preprocessed, or results.")
            self.name.clear(); self.training.clear(); self._set_table(self.channels, {}); self._set_table(self.labels, {})
            self.file_ending.clear(); self.description.clear(); self.license.clear()
            self.definition_errors_changed.emit(())
            self.training_count_warning_changed.emit("")
            self._loading = False
            return
        try:
            self._manifest = (read_manifest(source[1]) if create_raw_manifest and source else
                              self._starter_manifest(self._source_directory) if create_raw_manifest else
                              read_manifest(self._source_directory))
        except (OSError, ValueError) as error:
            self._directory = None; self._source_directory = None
            self.notice.setText(f"Could not read {self._source_kind} dataset.json: {error}")
            self.name.clear(); self.training.clear(); self._set_table(self.channels, {}); self._set_table(self.labels, {})
            self.file_ending.clear(); self.description.clear(); self.license.clear()
            self.definition_errors_changed.emit(())
            self.training_count_warning_changed.emit("")
            self._loading = False
            return
        self._set_editing_enabled(self._directory is not None)
        self.notice.setText(
            "A raw dataset.json will be created when you finish a valid edit." if create_raw_manifest else
            "" if self._directory else f"Viewing {self._source_kind} dataset.json (read-only)."
        )
        self.name.setText(self._source_directory.name)
        self.training.setText(str(self._manifest.get("numTraining", 0)))
        self._set_table(self.channels, self._manifest.get("channel_names", {}), id_first=True)
        self._set_table(self.labels, self._manifest.get("labels", {}), id_first=False)
        self.file_ending.setText(str(self._manifest.get("file_ending", "")))
        self.description.setPlainText(str(self._manifest.get("description", "")))
        self.license.setText(str(self._manifest.get("license", "")))
        self._update_training_validation(); self._update_table_validation()
        self._loading = False

    @staticmethod
    def _starter_manifest(directory: Path) -> dict:
        """Provide valid initial metadata until the first raw manifest is written."""
        return {
            "channel_names": {"0": "image"},
            "labels": {"background": 0},
            "numTraining": labels_tr_count(directory),
            "file_ending": "",
            "name": directory.name,
            "description": "",
            "license": "",
        }

    def _set_editing_enabled(self, enabled: bool) -> None:
        for control in (self.training, self.channel_controls, self.label_controls, self.file_ending, self.description,
                        self.license):
            control.setEnabled(enabled)

    def eventFilter(self, watched, event) -> bool:
        if watched is self.description and event.type() == QEvent.Type.FocusOut:
            self._auto_save("description")
        return super().eventFilter(watched, event)

    def _update_training_validation(self) -> None:
        expected = labels_tr_count(self._directory) if self._directory else None
        specified = int(self.training.text()) if self.training.text().isdigit() else None
        valid = expected is None or specified == expected
        self.training.setProperty("invalid", not valid)
        self.training.setToolTip("" if valid else f"numTraining must match labelsTr ({expected} file(s) found).")
        self.training.style().unpolish(self.training); self.training.style().polish(self.training)
        mismatch = expected is not None and specified is not None and specified != expected
        message = (f"numTraining is {specified}, but labelsTr contains {expected} file(s). "
                   "Update numTraining or the files in labelsTr before preprocessing.") if mismatch else ""
        self.training_count_warning_changed.emit(message if self._source_directory is not None else "")

    def _update_table_validation(self, *_ignored) -> None:
        channel_error = self._set_table_validation(self.channels, "Channel")
        label_error = self._set_table_validation(self.labels, "Label", require_ignore=self._annotation_workflow_enabled)
        errors = tuple(error for error in (channel_error, label_error) if error)
        self.definition_errors_changed.emit(errors if self._source_directory is not None else ())

    def _table_changed(self, field: str) -> None:
        self._update_table_validation()
        self._auto_save(field)

    @staticmethod
    def _set_table_validation(table: QTableWidget, singular: str, *, require_ignore: bool = False) -> str:
        identifiers: list[str] = []
        for row in range(table.rowCount()):
            identifier = table.item(row, 0).text().strip() if table.item(row, 0) else ""
            name = table.item(row, 1).text().strip() if table.item(row, 1) else ""
            if not identifier or not name:
                message = f"Every {singular.lower()} needs both an ID and a name."
                break
            identifiers.append(identifier)
        else:
            if not identifiers:
                message = f"Add at least one {singular.lower()}."
            elif not all(identifier.isdigit() for identifier in identifiers):
                message = f"{singular} IDs must be non-negative integers."
            elif sorted(int(identifier) for identifier in identifiers) != list(range(len(identifiers))):
                message = f"{singular} IDs must be continuous, starting at 0."
            elif require_ignore:
                ignore_id = next((int(table.item(row, 0).text()) for row in range(table.rowCount())
                                  if table.item(row, 1) and table.item(row, 1).text().strip() == "ignore"), None)
                message = "Add an ignore label for the annotation workflow." if ignore_id is None else (
                    "The ignore label must use the last (highest) ID." if ignore_id != len(identifiers) - 1 else "")
            else:
                message = ""
        table.setProperty("invalid", bool(message)); table.setToolTip(""); table.viewport().setToolTip("")
        table.style().unpolish(table); table.style().polish(table)
        return message

    def set_annotation_workflow_enabled(self, enabled: bool) -> None:
        self._annotation_workflow_enabled = enabled
        self._update_table_validation()

    @staticmethod
    def _set_table(table: QTableWidget, values: dict, *, id_first: bool = True) -> None:
        table.setRowCount(0)
        entries = [(key, value) if id_first else (value, key) for key, value in values.items()]
        for identifier, name in sorted(entries, key=lambda entry: MetadataEditor._id_sort_key(str(entry[0]))):
            row = table.rowCount(); table.insertRow(row)
            table.setItem(row, 0, QTableWidgetItem(str(identifier)))
            table.setItem(row, 1, QTableWidgetItem(str(name)))

    @staticmethod
    def _id_sort_key(identifier: str) -> tuple[int, int | str]:
        return (0, int(identifier)) if identifier.isdigit() else (1, identifier.casefold())

    def _add_row(self, table: QTableWidget, *, channel_ids: bool) -> None:
        identifiers = [table.item(row, 0).text().strip() for row in range(table.rowCount())
                       if table.item(row, 0)]
        has_non_numeric_id = any(not identifier.isdigit() for identifier in identifiers)
        identifier = "" if not channel_ids and has_non_numeric_id else str(
            max((int(value) for value in identifiers if value.isdigit()), default=-1) + 1)
        row = table.rowCount(); table.insertRow(row)
        table.setItem(row, 0, QTableWidgetItem(identifier)); table.setItem(row, 1, QTableWidgetItem(""))
        self._sort_table(table)

    def _remove_selected_row(self, table: QTableWidget, field: str) -> None:
        if table.currentRow() < 0:
            return
        table.removeRow(table.currentRow())
        self._update_table_validation()
        self._auto_save(field)

    @classmethod
    def _sort_table(cls, table: QTableWidget) -> None:
        entries = [(table.item(row, 0).text() if table.item(row, 0) else "",
                    table.item(row, 1).text() if table.item(row, 1) else "")
                   for row in range(table.rowCount())]
        table.setRowCount(0)
        for identifier, name in sorted(entries, key=lambda entry: cls._id_sort_key(entry[0])):
            row = table.rowCount(); table.insertRow(row)
            table.setItem(row, 0, QTableWidgetItem(identifier)); table.setItem(row, 1, QTableWidgetItem(name))

    @classmethod
    def _table_data(cls, table: QTableWidget, *, id_is_value: bool = False,
                    require_numeric_ids: bool = False, singular: str = "Entry") -> dict:
        cls._sort_table(table)
        if not table.rowCount():
            raise ValueError(f"Add at least one {singular.lower()}")
        values: dict = {}
        identifiers: list[int] = []
        for row in range(table.rowCount()):
            first, second = table.item(row, 0), table.item(row, 1)
            if not first or not second or not first.text().strip() or not second.text().strip():
                raise ValueError(f"Every {singular.lower()} needs both an ID and a name")
            identifier, name = first.text().strip(), second.text().strip()
            if require_numeric_ids and not identifier.isdigit():
                raise ValueError(f"{singular} IDs must be non-negative integers")
            if id_is_value and not identifier.isdigit():
                raise ValueError(f"{singular} IDs must be non-negative integers")
            value = int(identifier) if id_is_value and identifier.lstrip("-").isdigit() else identifier
            values[name if id_is_value else identifier] = value if id_is_value else name
            identifiers.append(int(identifier))
        if sorted(identifiers) != list(range(len(identifiers))):
            raise ValueError(f"{singular} IDs must be continuous, starting at 0")
        return values

    def _auto_save(self, field: str) -> None:
        if self._loading or self._saving or self._directory is None:
            return
        was_empty = not self._pending_saves
        self._pending_saves.add(field)
        if was_empty:
            QTimer.singleShot(0, self._flush_auto_saves)

    def _flush_auto_saves(self) -> None:
        """Save after Qt has finished committing the active table-cell editor."""
        fields, self._pending_saves = tuple(self._pending_saves), set()
        if self._loading or self._saving or self._directory is None:
            return
        for field in fields:
            self._save(field)

    def _save(self, field: str) -> bool:
        """Persist one valid metadata field without blocking on unrelated invalid fields."""
        if self._directory is None or self._saving:
            return False
        self._saving = True
        data = dict(self._manifest)
        try:
            if field == "channels":
                data["channel_names"] = self._table_data(
                    self.channels, require_numeric_ids=True, singular="Channel")
            elif field == "labels":
                data["labels"] = self._table_data(self.labels, id_is_value=True, singular="Label")
            elif field == "training":
                training = self.training.text().strip()
                if not training.isdigit():
                    raise ValueError("numTraining must be a non-negative integer")
                data["numTraining"] = int(training)
            elif field == "file_ending":
                data["file_ending"] = self.file_ending.text().strip()
            elif field == "description":
                data["description"] = self.description.toPlainText()
            elif field == "license":
                data["license"] = self.license.text().strip()
            else:
                raise ValueError(f"Unknown metadata field: {field}")
        except ValueError:
            self._saving = False
            return False
        try:
            write_manifest(self._directory, data)
        except OSError as error:
            QMessageBox.critical(self, "Could not save metadata", str(error))
            self._saving = False
            return False
        self._manifest = data
        self._saving = False
        self._update_training_validation()
        self.metadata_saved.emit()
        return True
