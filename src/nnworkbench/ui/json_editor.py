"""dataset.json on the right of the Data page: a read-only view and an in-place editor.

The editor makes the JSON itself editable: the top-level keys are fixed, only their values can be
changed. Labels and channels can be renamed, renumbered, added and removed.
"""

from __future__ import annotations

import json
from pathlib import Path

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from nnworkbench.services.dataset_json import (
    definition_errors,
    file_ending_error,
    num_training_error,
    write_manifest,
)

from .widgets import badge, button, label, scroll, set_badge

_STRUCTURED = ("channel_names", "labels")
# Keys nnWorkbench always shows; their names are fixed. Every other key is a custom entry.
_STANDARD = ("channel_names", "labels", "numTraining", "file_ending", "name", "description", "licence")


class _PairEditor(QWidget):
    """Editable ``key: value`` rows of a nested object with ✕ to remove and ＋ to add."""

    def __init__(self, values: dict, *, value_is_id: bool, add_text: str, parent=None) -> None:
        super().__init__(parent)
        self._value_is_id = value_is_id
        self._rows: list[tuple[QWidget, QLineEdit, QLineEdit]] = []
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(16, 0, 0, 0)
        self._layout.setSpacing(4)
        for key, value in values.items():
            self._add_row(str(key), str(value))
        add = QPushButton(add_text)
        add.setObjectName("link")
        add.clicked.connect(lambda: self._add_row("", ""))
        self._layout.addWidget(add, 0, Qt.AlignmentFlag.AlignLeft)

    def _add_row(self, key: str, value: str) -> None:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        key_field, value_field = QLineEdit(key), QLineEdit(value)
        for field in (key_field, value_field):
            field.setProperty("mono", True)
        if self._value_is_id:  # labels: "name": id
            value_field.setFixedWidth(56)
            value_field.setAlignment(Qt.AlignmentFlag.AlignRight)
            layout.addWidget(key_field, 1)
            layout.addWidget(label(":", muted=True))
            layout.addWidget(value_field)
        else:  # channels: "id": name
            key_field.setFixedWidth(56)
            layout.addWidget(key_field)
            layout.addWidget(label(":", muted=True))
            layout.addWidget(value_field, 1)
        remove = QPushButton("✕")
        remove.setObjectName("link")
        remove.setStyleSheet("color: #94a3b8;")
        remove.setToolTip("Remove")
        layout.addWidget(remove)
        entry = (row, key_field, value_field)
        remove.clicked.connect(lambda: self._remove(entry))
        self._rows.append(entry)
        self._layout.insertWidget(len(self._rows) - 1, row)

    def _remove(self, entry) -> None:
        if entry in self._rows:
            self._rows.remove(entry)
            entry[0].deleteLater()

    def value(self) -> dict:
        result: dict = {}
        for _row, key_field, value_field in self._rows:
            key, value = key_field.text().strip(), value_field.text().strip()
            if not key and not value:
                continue
            if self._value_is_id:
                try:
                    result[key] = int(value)
                except ValueError:
                    result[key] = value
            else:
                result[key] = value
        return result


def _custom_value(text: str, was_number: bool = False) -> object:
    """Objects, lists, true/false/null (and values that were numbers) are read as JSON; everything else stays
    text, so e.g. a version "1.0" is not turned into a number."""
    if text.startswith(("{", "[")) or text in ("true", "false", "null") or was_number:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return text


class DatasetJsonPanel(QFrame):
    """Shows dataset.json; Edit switches to the in-place editor."""

    saved = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self._directory: Path | None = None
        self._manifest: dict | None = None
        self._editable = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)
        head = QHBoxLayout()
        head.setSpacing(8)
        head.addWidget(label("dataset.json", name="h3"))
        self._badge = badge("✓ valid", "green")
        head.addWidget(self._badge)
        head.addStretch()
        self._edit = button("Edit", small=True, icon="edit", clicked=self._start_edit)
        self._cancel = button("Cancel", small=True, clicked=self._cancel_edit)
        self._save = button("Save", small=True, primary=True, clicked=self._save_edit)
        for widget in (self._edit, self._cancel, self._save):
            head.addWidget(widget)
        layout.addLayout(head)
        self._stack = QStackedWidget()
        self._view = QPlainTextEdit()
        self._view.setObjectName("codeView")
        self._view.setReadOnly(True)
        self._stack.addWidget(self._view)
        self._editor_holder = QWidget()
        self._stack.addWidget(self._editor_holder)
        layout.addWidget(self._stack, 1)
        self._note = label("", muted=True, wrap=True, size=12, rich=True)
        layout.addWidget(self._note)
        self._editors: dict[str, object] = {}
        self._cases: int | None = None
        self._set_mode(False)

    def set_manifest(self, directory: Path | None, manifest: dict | None, editable: bool, note: str = "",
                     cases: int | None = None) -> None:
        """``cases`` is the number of training cases in imagesTr; numTraining is checked against it."""
        self._directory, self._manifest, self._editable, self._cases = directory, manifest, editable, cases
        self._view.setPlainText(json.dumps(manifest, indent=2, ensure_ascii=False) if manifest is not None
                                else "No dataset.json found.")
        errors = ((*definition_errors(manifest), *filter(None, [file_ending_error(manifest)]))
                  if isinstance(manifest, dict) else ("missing",))
        count_error = num_training_error(manifest, cases) if isinstance(manifest, dict) and cases else None
        if manifest is None:
            set_badge(self._badge, "missing or unreadable", "red")
        elif errors:
            set_badge(self._badge, "! invalid", "red", " ".join(errors))
        elif count_error:
            set_badge(self._badge, "! check numTraining", "red", count_error)
        else:
            set_badge(self._badge, "✓ valid", "green")
        self._default_note = note
        self._note.setText(note)
        self._set_mode(False)

    def _set_mode(self, editing: bool) -> None:
        self._stack.setCurrentIndex(1 if editing else 0)
        self._edit.setVisible(not editing and self._editable)
        self._cancel.setVisible(editing)
        self._save.setVisible(editing)
        self.setProperty("active", editing)
        self.setStyleSheet("QFrame#card { border: 1px solid #1f5fbf; }" if editing else "")

    def _start_edit(self) -> None:
        manifest = dict(self._manifest or {})
        for key in _STANDARD:
            manifest.setdefault(key, {} if key in _STRUCTURED else "")
        holder = QWidget()
        grid = QGridLayout(holder)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        self._editors = {}
        row = 0
        self._custom: list[tuple[QWidget, QLineEdit, QLineEdit]] = []
        for key, value in manifest.items():
            if key not in _STANDARD:
                continue
            key_label = label(f'"{key}":', muted=True, mono=True)
            if key in _STRUCTURED and isinstance(value, dict):
                grid.addWidget(key_label, row, 0, 1, 2)
                row += 1
                editor = _PairEditor(value, value_is_id=key == "labels",
                                     add_text="＋ add label" if key == "labels" else "＋ add channel")
                grid.addWidget(editor, row, 0, 1, 2)
            else:
                field = QLineEdit(json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value)
                field.setProperty("mono", True)
                field.setProperty("json", not isinstance(value, str))
                grid.addWidget(key_label, row, 0)
                if key == "numTraining" and self._cases and value != self._cases:
                    cell = QWidget()
                    cell_row = QHBoxLayout(cell)
                    cell_row.setContentsMargins(0, 0, 0, 0)
                    cell_row.setSpacing(6)
                    cell_row.addWidget(field, 1)
                    fix = QPushButton(f"Set to {self._cases}")
                    fix.setObjectName("link")
                    fix.setToolTip("numTraining must be the number of training cases in imagesTr.")
                    fix.clicked.connect(lambda _=False, f=field, c=self._cases: f.setText(str(c)))
                    cell_row.addWidget(fix)
                    grid.addWidget(cell, row, 1)
                else:
                    grid.addWidget(field, row, 1)
                editor = field
            self._editors[key] = editor
            row += 1
        grid.addWidget(label("Custom entries", name="sectionLabel"), row, 0, 1, 2)
        row += 1
        self._custom_layout = QVBoxLayout()
        self._custom_layout.setSpacing(4)
        grid.addLayout(self._custom_layout, row, 0, 1, 2)
        row += 1
        for key, value in manifest.items():
            if key not in _STANDARD:
                self._add_custom(key, value)
        add = QPushButton("＋ add entry")
        add.setObjectName("link")
        add.setToolTip("Add a custom key, e.g. reference or release. Values are read as JSON when possible, "
                       "otherwise as text.")
        add.clicked.connect(lambda: self._add_custom("", ""))
        grid.addWidget(add, row, 0, 1, 2, Qt.AlignmentFlag.AlignLeft)
        row += 1
        grid.setRowStretch(row, 1)
        grid.setColumnStretch(1, 1)
        old = self._stack.widget(1)
        self._stack.removeWidget(old)
        old.deleteLater()
        area = scroll(holder)
        self._stack.insertWidget(1, area)
        count = f" ({self._cases})" if self._cases else ""
        self._note.setText("The standard keys are fixed; only their values can be changed. Custom entries can be "
                           "added and removed. Labels and channels can be renamed, "
                           "renumbered, added and removed. Labels must stay consecutive, and <span style='font-family:"
                           f"monospace'>ignore</span> must be the highest ID. numTraining must match the training cases in imagesTr{count}.")
        self._set_mode(True)

    def _add_custom(self, key: str, value: object) -> None:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        key_field = QLineEdit(key)
        key_field.setPlaceholderText("key")
        value_field = QLineEdit(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
        value_field.setProperty("number", isinstance(value, (int, float)) and not isinstance(value, bool))
        value_field.setPlaceholderText("value")
        for field in (key_field, value_field):
            field.setProperty("mono", True)
        key_field.setFixedWidth(130)
        layout.addWidget(key_field)
        layout.addWidget(label(":", muted=True))
        layout.addWidget(value_field, 1)
        remove = QPushButton("✕")
        remove.setObjectName("link")
        remove.setStyleSheet("color: #94a3b8;")
        remove.setToolTip("Remove this entry")
        entry = (row, key_field, value_field)
        remove.clicked.connect(lambda: self._remove_custom(entry))
        layout.addWidget(remove)
        self._custom.append(entry)
        self._custom_layout.addWidget(row)

    def _remove_custom(self, entry) -> None:
        if entry in self._custom:
            self._custom.remove(entry)
            entry[0].deleteLater()

    def _collect(self) -> dict:
        result = {}
        for key, editor in self._editors.items():
            if isinstance(editor, _PairEditor):
                result[key] = editor.value()
            elif editor.property("json"):
                text = editor.text().strip()
                try:
                    result[key] = json.loads(text) if text else ""
                except json.JSONDecodeError:
                    result[key] = text
            else:
                result[key] = editor.text()
        for _row, key_field, value_field in self._custom:
            key, text = key_field.text().strip(), value_field.text().strip()
            if key:
                result[key] = _custom_value(text, was_number=bool(value_field.property("number")))
        return result

    def _save_edit(self) -> None:
        if self._directory is None:
            return
        manifest = self._collect()
        problems = list(definition_errors(manifest))
        custom_keys = [key_field.text().strip() for _row, key_field, _value in self._custom if key_field.text().strip()]
        duplicates = sorted({key for key in custom_keys if custom_keys.count(key) > 1 or key in _STANDARD})
        if duplicates:
            problems.append(f"Custom keys must be unique and differ from the standard keys: {', '.join(duplicates)}.")
        labels = manifest.get("labels", {})
        ignore = next((value for name, value in labels.items() if name.strip().casefold() == "ignore"), None)
        numeric = [value for value in labels.values() if isinstance(value, int)]
        if ignore is not None and numeric and ignore != max(numeric):
            problems.append("ignore must be the highest label ID.")
        if problems:
            self._note.setText("<span style='color:#c2413b'>Not saved: " + " ".join(problems) + "</span>")
            return
        try:
            write_manifest(self._directory, manifest)
        except (OSError, ValueError) as error:
            self._note.setText(f"<span style='color:#c2413b'>Could not save: {error}</span>")
            return
        self._manifest = manifest
        self.set_manifest(self._directory, manifest, self._editable, self._default_note, self._cases)
        self.saved.emit()

    def _cancel_edit(self) -> None:
        self._note.setText(self._default_note)
        self._set_mode(False)
