"""Start screen: nnU-Net folders with path profiles, all datasets and the selected dataset's details."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path

from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtGui import QFontMetrics, QGuiApplication
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from nnworkbench.models.dataset import DatasetRecord
from nnworkbench.path_setup import environment_conflicts, set_paths_command, typed_path
from nnworkbench.services.dataset_contents import json_object, model_folds, trained_models
from nnworkbench.services.dataset_json import manifest_source
from nnworkbench.services.dataset_scanner import id_badge

from .context import AppContext
from .widgets import (
    AvailabilityDelegate,
    Card,
    ElidedLabel,
    FlowLayout,
    availability_item,
    availability_width,
    badge,
    breakable,
    button,
    cell_widget,
    clear_layout,
    dataset_id_label,
    elide_to,
    empty_state,
    format_time,
    label,
    make_table,
    scroll,
    section_label,
    separator,
    set_badge,
    set_elided_text,
    svg_icon,
    table_item,
)

_KEYS = ("raw", "preprocessed", "results")
_DESCRIPTION_LIMIT = 1000  # characters of the dataset.json description shown in the details
_CHIP_LIMIT = 100  # label chips shown in the details, then "+N more"
_CHIP_WIDTH = 400  # at most (the details panel is 470 wide): longer label names end with "…"
_PROFILES_HEIGHT = 84  # path profiles shown before the list scrolls (two), so the card stays low


def _expanded(text: str) -> Path:
    """The folder typed or pasted (quotes, file: URL, ~, $VAR), absolute: nnU-Net runs from another folder."""
    path = Path(typed_path(text))
    try:
        return path.absolute()
    except OSError:  # the working folder was deleted
        return path


def _folder_state(text: str) -> str:
    """"found", "missing" or "unreadable" (exists, but its content cannot be listed)."""
    path = _expanded(text)
    try:
        if not path.is_dir():
            return "missing"
    except (OSError, ValueError):  # name too long, null byte, ...
        return "missing"
    try:
        with os.scandir(path):
            return "found"
    except OSError:
        return "unreadable"


def _field_text(text: str) -> str:
    """Surrounding spaces are typos, unless the folder name really has them (e.g. "raw ")."""
    stripped = text.strip()
    if stripped != text and stripped:
        try:
            if _expanded(text).exists():
                return text
        except (OSError, ValueError):
            pass
    return stripped


_FOLDER_NAMES = ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")


def _layout_problem(texts: list[str]) -> tuple[str, str, str] | None:
    """Badge text, colour and note when the three found folders cannot work: the same folder or one inside another
    (preprocessing then writes into the raw dataset and fails), or a parent folder of nnUNet_raw & co. chosen."""
    try:
        paths = [_expanded(text).resolve() for text in texts]
    except (OSError, RuntimeError, ValueError):
        return None
    for index, first in enumerate(paths):
        for second in paths[index + 1:]:
            if first == second or first in second.parents or second in first.parents:
                return ("folders overlap", "red", ("nnUNet_raw, nnUNet_preprocessed and nnUNet_results must be "
                                                   "three different folders, none inside another: choose them "
                                                   "again with Browse…"))
    for path, expected in zip(paths, _FOLDER_NAMES):
        try:
            inside = path.name != expected and (path / expected).is_dir()
        except OSError:
            inside = False
        if inside:
            return "check folders", "orange", f"{path} contains a folder {expected}: did you mean {path / expected}?"
    return None


def _short(value: object, limit: int) -> str:
    """``str(value)`` cut to ``limit`` characters, also for deeply nested values."""
    try:
        text = str(value)
    except (RecursionError, ValueError):
        return "…"
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _wrap_anywhere(text: str) -> str:
    """``text`` that a word-wrapping label may break after "_" and inside long words (folder names)."""
    words = (word if len(word) <= 30 else "\u200b".join(word[index:index + 20] for index in range(0, len(word), 20))
             for word in text.split(" "))
    return breakable(" ".join(words))


class _ValueLabel(QLabel):
    """A value of the details grid and a muted note ("· 8 channels"): the value ends with "…" when it is longer
    than the panel is wide (the note stays visible), the full value is the tooltip."""

    def __init__(self, value: str, note: str) -> None:
        super().__init__()
        self._value, self._note = value, f"· {note}" if note else ""
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(40)
        self._show(10_000)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._show(self.contentsRect().width())

    def _show(self, width: int) -> None:
        metrics = self.fontMetrics()
        room = width - (metrics.horizontalAdvance(" " + self._note) if self._note else 0) - 2
        shown = metrics.elidedText(self._value, Qt.TextElideMode.ElideRight, max(room, 20))
        self.setToolTip(self._value if shown != self._value else "")
        self.setText(f"{html.escape(shown)} <span style='color:#64748b'>{html.escape(self._note)}</span>")


def _plain(widget: QLabel) -> QLabel:
    """dataset.json values and folder names are shown as written, never as HTML."""
    widget.setTextFormat(Qt.TextFormat.PlainText)
    return widget


class PathsCard(Card):
    """The three nnU-Net folders plus path profiles (Local is used by the GUI; others are exported)."""

    paths_changed = Signal()
    profiles_changed = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._fields: dict[str, QLineEdit] = {}
        self._marks: dict[str, QLabel] = {}
        self._browse: list[QPushButton] = []
        self._local: tuple[str, ...] = ("", "", "")
        self._profiles: dict[str, dict[str, str]] = {}
        self._active = "Local"
        outer = QHBoxLayout()
        outer.setContentsMargins(20, 16, 20, 16)
        outer.setSpacing(24)
        self.body.addLayout(outer)
        left = QVBoxLayout()
        left.setSpacing(8)
        title = QHBoxLayout()
        title.setSpacing(10)
        title.addWidget(label("nnU-Net folders", name="h3"))
        self._status = badge("", "green")
        title.addWidget(self._status)
        self._variables = badge("differs from nnU-Net variables", "orange")  # hover: which path is used
        self._variables.hide()
        title.addWidget(self._variables)
        self._note = label("shared with the command line (nnworkbench-set-paths)", muted=True, size=12)
        title.addWidget(self._note)
        title.addStretch()
        left.addLayout(title)
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        for row, key in enumerate(_KEYS):
            name = f"nnUNet_{key}"
            grid.addWidget(label(name, muted=True), row, 0)
            field = QLineEdit()
            field.setProperty("mono", True)
            field.setPlaceholderText(f"not set: choose the {name} folder with Browse…")
            field.editingFinished.connect(self._edited)
            grid.addWidget(field, row, 1)
            browse = button("Browse…", small=True, clicked=lambda item=key: self._choose(item))
            grid.addWidget(browse, row, 2)
            mark = QLabel()
            mark.setFixedWidth(24)
            mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(mark, row, 3)
            self._fields[key], self._marks[key] = field, mark
            self._browse.append(browse)
        grid.setColumnStretch(1, 1)
        left.addLayout(grid)
        left.addStretch()  # a higher profile list: the folders stay at the top
        outer.addLayout(left, 1)
        right = QVBoxLayout()
        right.setSpacing(8)
        right.addWidget(section_label("Path profile"))
        profiles = QWidget()
        self._profile_list = QVBoxLayout(profiles)
        self._profile_list.setContentsMargins(0, 0, 0, 0)
        self._profile_list.setSpacing(8)
        self._profile_scroll = _FitScrollArea(profiles, _PROFILES_HEIGHT)  # many profiles: scroll, card stays low
        right.addWidget(self._profile_scroll)
        self._add = button("＋ Add remote profile", clicked=self._add_profile)
        self._add.setStyleSheet("text-align: left; color: #64748b; font-weight: 500;")
        right.addWidget(self._add)
        right.addStretch()
        holder = QWidget()
        holder.setFixedWidth(290)
        holder.setLayout(right)
        outer.addWidget(holder)

    # ------------------------------------------------------------ state

    def set_paths(self, raw: str, preprocessed: str, results: str) -> None:
        self._local = (raw, preprocessed, results)
        if self._active == "Local":
            self._load_fields()

    def set_profiles(self, profiles: object) -> None:
        self._profiles = {
            name: {key: str(values.get(key, "")) for key in _KEYS}
            for name, values in (profiles.items() if isinstance(profiles, dict) else [])
            if isinstance(name, str) and isinstance(values, dict)
        }
        self._render_profiles()

    def local_paths(self) -> tuple[Path | None, Path | None, Path | None]:
        return tuple(_expanded(value) if value.strip() else None for value in self._local)  # type: ignore

    def _texts(self) -> tuple[str, ...]:
        return tuple(_field_text(self._fields[key].text()) for key in _KEYS)

    def _load_fields(self) -> None:
        values = self._local if self._active == "Local" else tuple(self._profiles[self._active][key] for key in _KEYS)
        for key, value in zip(_KEYS, values):
            self._fields[key].setText(value)
            self._fields[key].setCursorPosition(0)  # a long path shows its start, not its end
        for browse in self._browse:
            browse.setEnabled(self._active == "Local")
        self._update_marks()
        self._render_profiles()

    def _update_marks(self) -> None:
        """✓ found, orange ! not set yet (normal on the first start), red ✗ set but not found or not readable."""
        missing = unset = unreadable = 0
        for key in _KEYS:
            text = _field_text(self._fields[key].text())
            mark = self._marks[key]
            if self._active != "Local":
                mark.setText("")
                mark.setToolTip("")
                continue
            if not text:
                state, tooltip = "unset", "Not set yet: choose the folder with Browse… (an empty folder is fine)"
            else:
                state = _folder_state(text)
                tooltip = {"found": "Folder found", "missing": f"Folder not found: {text}",
                           "unreadable": f"Folder not readable (no permission): {text}"}[state]
            mark.setText({"found": "✓", "unset": "!", "missing": "✗", "unreadable": "✗"}[state])
            mark.setObjectName({"found": "statusYes", "unset": "statusUnset"}.get(state, "statusNo"))
            mark.setToolTip(tooltip)
            mark.style().unpolish(mark)
            mark.style().polish(mark)
            missing += state == "missing"
            unset += state == "unset"
            unreadable += state == "unreadable"
        if self._active != "Local":
            set_badge(self._status, f"profile: {self._active}", "purple")
            self._note.setText("paths for another machine · used only for Export command")
        elif missing or unreadable:
            parts = [f"{missing} not found" if missing else "", f"{unreadable} not readable" if unreadable else "",
                     f"{unset} not set" if unset else ""]
            set_badge(self._status, " · ".join(part for part in parts if part), "red")
            self._note.setText("Check the path or choose another folder with Browse…" if missing else
                               "Check the folder's permissions or choose another folder with Browse…")
        elif unset:
            set_badge(self._status, f"{unset} not set", "orange")
            self._note.setText("Please set the paths: choose each folder with Browse… (empty folders are fine)")
        elif problem := _layout_problem([_field_text(self._fields[key].text()) for key in _KEYS]):
            set_badge(self._status, *problem[:2])
            self._note.setText(problem[2])
        else:
            set_badge(self._status, "✓ all found", "green")
            self._note.setText("shared with the command line (nnworkbench-set-paths)")
        self._update_variables()

    def _update_variables(self) -> None:
        """Orange badge when nnU-Net's folder variables (nnUNet_raw, …) point elsewhere; nnWorkbench uses its own."""
        conflicts = environment_conflicts({f"nnUNet_{key}": str(path) for key, path
                                           in zip(_KEYS, self.local_paths()) if path}) if self._active == "Local" else []
        self._variables.setVisible(bool(conflicts))
        if conflicts and self._status.text() == "✓ all found":  # room for the badge; its hover text says the same
            self._note.setText("")
        self._variables.setToolTip("\n\n".join(
            f"{name}\n  used (nnWorkbench): {used or 'not set'}\n  variable (not used): {value}"
            for name, value, used in conflicts) + "\n\nnnWorkbench uses its own folders, also for the nnU-Net "
            "commands it runs.")

    def _edited(self) -> None:
        texts = self._texts()
        if self._active == "Local":
            if texts != self._local:
                self._local = texts
                self._update_marks()
                self.paths_changed.emit()
        else:
            self._profiles[self._active] = dict(zip(_KEYS, texts))
            self.profiles_changed.emit(self._profiles)

    def _choose(self, key: str) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Choose nnU-Net directory", self._fields[key].text())
        if selected:
            self._fields[key].setText(selected)
            self._edited()

    # ------------------------------------------------------------ profiles

    def _render_profiles(self) -> None:
        while self._profile_list.count():
            item = self._profile_list.takeAt(0)
            if item.widget():
                item.widget().hide()
                item.widget().deleteLater()
        self._profile_list.addWidget(self._profile_row("Local", "used by the GUI"))
        for name in sorted(self._profiles, key=str.casefold):
            self._profile_list.addWidget(self._profile_row(name, ""))
        self._profile_scroll.updateGeometry()

    def _profile_row(self, name: str, note: str) -> QWidget:
        active = name == self._active
        row = QFrame()
        row.setObjectName("card")
        row.setProperty("active", active)
        if active:
            row.setStyleSheet("QFrame#card { background: #e8f0fc; border-color: #1f5fbf; }")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(10, 4, 6, 4)
        layout.setSpacing(6)
        title = QPushButton()
        title.setObjectName("link")
        # left-aligned and shortened with "…" at the end, so a long name keeps its start (full name on hover)
        title.setStyleSheet("text-align: left; " + ("color: #1f5fbf;" if active else "color: #1e293b; font-weight: 500;"))
        set_elided_text(title, ("● " if active else "") + ("☁ " if name != "Local" else "") + name)
        title.clicked.connect(lambda _=False, selected=name: self._activate(selected))
        layout.addWidget(title, 1)
        if note:
            layout.addWidget(label(note, size=11, bold=True))
        if name != "Local":
            export = QPushButton("Export")
            export.setObjectName("link")
            export.setToolTip("Copy an nnworkbench-set-paths command for this profile")
            export.clicked.connect(lambda _=False, selected=name: self._export(selected))
            layout.addWidget(export)
            delete = QPushButton("✕")
            delete.setObjectName("link")
            delete.setStyleSheet("color: #94a3b8;")
            delete.setToolTip("Delete this profile")
            delete.clicked.connect(lambda _=False, selected=name: self._delete(selected))
            layout.addWidget(delete)
        return row

    def _activate(self, name: str) -> None:
        if name != self._active:
            self._active = name
            self._load_fields()

    def _add_profile(self) -> None:
        name, accepted = QInputDialog.getText(self, "Add remote profile", "Profile name (e.g. Cluster)")
        name = name.strip()
        if not accepted or not name:
            return
        if name == "Local" or name in self._profiles:
            QMessageBox.warning(self, "Profile exists", f'A profile named "{name}" already exists.')
            return
        self._profiles[name] = {key: "" for key in _KEYS}
        self._active = name
        self._load_fields()
        self.profiles_changed.emit(self._profiles)

    def _delete(self, name: str) -> None:
        answer = QMessageBox.question(self, "Delete profile", f'Delete the profile "{name}"?')
        if answer == QMessageBox.StandardButton.Yes:
            self._profiles.pop(name, None)
            self._active = "Local"
            self._load_fields()
            self.profiles_changed.emit(self._profiles)

    def _export(self, name: str) -> None:
        values = self._profiles.get(name, {})
        command = set_paths_command(*(values.get(key, "") for key in _KEYS))
        QGuiApplication.clipboard().setText(command)
        QMessageBox.information(self, "Command copied",
                                f"Run this in the remote shell to use the {name} paths there:\n\n{command}")


class _FitScrollArea(QScrollArea):
    """A vertical scroll area as high as its content, but at most ``maximum`` pixels (then it scrolls)."""

    def __init__(self, widget: QWidget, maximum: int, parent=None) -> None:
        super().__init__(parent)
        self._maximum = maximum
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.setWidget(widget)
        widget.setAutoFillBackground(False)
        self.viewport().setAutoFillBackground(False)

    def sizeHint(self) -> QSize:
        return QSize(super().sizeHint().width(), min(self.widget().sizeHint().height(), self._maximum))

    def minimumSizeHint(self) -> QSize:
        return QSize(super().minimumSizeHint().width(), min(self.widget().sizeHint().height(), self._maximum))


def _record_modified(record: DatasetRecord) -> float | None:
    times = []
    for path in (record.raw_path, record.preprocessed_path, record.results_path):
        try:
            if path is not None:
                times.append(path.stat().st_mtime)
        except OSError:
            pass
    return max(times) if times else None


def dataset_details(record: DatasetRecord) -> dict:
    """Read what the details panel shows: dataset.json plus a few folder listings."""
    source = manifest_source(record.raw_path, record.preprocessed_path, record.results_path)
    manifest = json_object(source[1] / "dataset.json") if source else None
    raw_folders = [name for name in ("imagesTr", "labelsTr", "imagesTs")
                   if record.raw_path is not None and (record.raw_path / name).is_dir()]
    plans = []
    if record.preprocessed_path is not None and record.preprocessed_path.is_dir():
        plans = sorted(path.stem for path in record.preprocessed_path.glob("*.json")
                       if "plans" in path.name.casefold())
    models = trained_models(record.results_path)
    folds = sum(len(model_folds(record.results_path / model)) for model in models) if record.results_path else 0
    return {"manifest": manifest, "source": source[0] if source else None, "raw_folders": raw_folders,
            "plans": plans, "models": len(models), "folds": folds}


class DetailsPanel(Card):
    open_requested = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(470)
        self._record: DatasetRecord | None = None
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)
        self.body.addLayout(layout)
        head = QHBoxLayout()
        head.setSpacing(12)
        self._id = dataset_id_label(0)
        self._id.setFixedHeight(32)
        head.addWidget(self._id)
        names = QVBoxLayout()
        names.setSpacing(0)
        self._name = _plain(label("", name="h2"))  # long names end with "…" (set_elided_text)
        self._folder = _plain(label("", muted=True, size=12))
        names.addWidget(self._name)
        names.addWidget(self._folder)
        head.addLayout(names, 1)
        self._open = button("Open dataset →", primary=True, clicked=self._open_clicked)
        head.addWidget(self._open)
        layout.addLayout(head)
        self._stack = QStackedWidget()
        layout.addWidget(self._stack, 1)
        info = QWidget()
        self._info = QVBoxLayout(info)
        self._info.setContentsMargins(0, 0, 4, 0)  # room for the scroll bar
        self._info.setSpacing(10)
        # scrolls when the window is low or the dataset has much to show, instead of squashing the rows
        self._scroll = scroll(info)
        self._scroll.viewport().setAutoFillBackground(False)
        info.setAutoFillBackground(False)
        self._stack.addWidget(self._scroll)
        self._json = QPlainTextEdit()
        self._json.setObjectName("codeView")
        self._json.setReadOnly(True)
        self._stack.addWidget(self._json)
        foot = QHBoxLayout()
        foot.addWidget(label("ⓘ Loaded from dataset.json on selection", muted=True, size=12))
        foot.addStretch()
        self._toggle_json = button("{ } Show JSON", ghost=True, small=True, clicked=self._toggle)
        foot.addWidget(self._toggle_json)
        layout.addLayout(foot)
        self.set_record(None, None)

    def set_record(self, record: DatasetRecord | None, details: dict | None) -> None:
        self._record = record
        while self._info.count():
            item = self._info.takeAt(0)
            if item.widget():
                item.widget().hide()
                item.widget().deleteLater()
            elif item.layout():
                clear_layout(item.layout())
        self._open.setEnabled(record is not None)
        self._scroll.verticalScrollBar().setValue(0)
        if record is None:
            self._id.setText("—")
            set_elided_text(self._name, "Select a dataset")
            set_elided_text(self._folder, "")
            self._json.setPlainText("")
            return
        self._id.setText(f"{record.dataset_id:03d}")
        set_elided_text(self._name, record.name)
        set_elided_text(self._folder, record.directory_name, Qt.TextElideMode.ElideMiddle)
        if (id_warning := id_badge(record)) is not None:
            self._info.addWidget(badge(*id_warning))
            self._info.addWidget(_plain(label(_wrap_anywhere(id_warning[2]), muted=True, wrap=True)))
        if details is None:
            self._info.addWidget(label("Loading dataset.json…", muted=True))
            return
        if "error" in details:  # unreadable folder or dataset.json
            self._info.addWidget(_plain(label(_wrap_anywhere(f"Could not read the dataset: {details['error']}"),
                                              muted=True, wrap=True)))
            self._json.setPlainText(str(details["error"]))
            return
        manifest = details.get("manifest")
        try:
            text = json.dumps(manifest, indent=2, ensure_ascii=False) if manifest else "No dataset.json found."
        except (RecursionError, ValueError, TypeError) as error:  # nested too deeply
            text = f"dataset.json cannot be shown: {type(error).__name__}"
        self._json.setPlainText(text)
        if not isinstance(manifest, dict):
            self._info.addWidget(label("No dataset.json found in nnUNet_raw, nnUNet_preprocessed or nnUNet_results.",
                                       muted=True, wrap=True))
            return
        description = manifest.get("description")
        if isinstance(description, str) and description.strip():
            self._info.addWidget(_plain(label(_wrap_anywhere(_short(description.strip(), _DESCRIPTION_LIMIT)),
                                              muted=True, wrap=True)))
        channels = manifest.get("channel_names") if isinstance(manifest.get("channel_names"), dict) else {}
        labels = manifest.get("labels") if isinstance(manifest.get("labels"), dict) else {}
        classes = [name for name in labels if name != "ignore"]
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        rows = (("Modality", _short(" · ".join(_short(value, 40) for value in list(channels.values())[:20]), 400) or "—",
                 f"{len(channels)} channel{'s' if len(channels) != 1 else ''}"),
                ("Cases", _short(manifest.get("numTraining", "—"), 40), "numTraining"),
                ("Labels", f"{len(classes)} classes", "+ ignore" if len(classes) != len(labels) else "dense"),
                ("File ending", _short(manifest.get("file_ending", "—"), 40), ""))
        for row, (key, value, note) in enumerate(rows):
            grid.addWidget(label(key, muted=True), row, 0, Qt.AlignmentFlag.AlignTop)
            grid.addWidget(_ValueLabel(value, note), row, 1)
        grid.setColumnStretch(1, 1)
        self._info.addLayout(grid)
        chips_holder = QWidget()
        chips = FlowLayout(chips_holder)
        for name, value in list(labels.items())[:_CHIP_LIMIT]:  # one widget per chip: thousands would freeze
            chip = _plain(QLabel())
            chip.setObjectName("chip")
            if name == "ignore":
                chip.setProperty("ignore", True)
            elide_to(chip, f"{_short(name, 60)} · {_short(value, 40)}", _CHIP_WIDTH)
            chips.addWidget(chip)
        if len(labels) > _CHIP_LIMIT:
            more = QLabel(f"+{len(labels) - _CHIP_LIMIT} more")
            more.setObjectName("chip")
            more.setToolTip("All labels: { } Show JSON")
            chips.addWidget(more)
        self._info.addWidget(chips_holder)
        self._info.addWidget(separator())
        folders = QGridLayout()
        folders.setHorizontalSpacing(12)
        folders.setVerticalSpacing(6)
        mode = "sparse labels" if any(name == "ignore" for name in labels) else "dense labels"
        entries = (("RAW", [", ".join(details["raw_folders"]) + f" · {mode}" if details["raw_folders"] else "—"]),
                   ("PRE", details["plans"] or ["—"]),  # one line per plans file, long names end with "…"
                   ("RES", [f"{details['models']} model{'s' if details['models'] != 1 else ''} · "
                            f"{details['folds']} fold{'s' if details['folds'] != 1 else ''}" if details["models"] else "—"]))
        for row, (key, values) in enumerate(entries):
            folders.addWidget(label(key, muted=True), row, 0, Qt.AlignmentFlag.AlignTop)
            lines = QVBoxLayout()
            lines.setSpacing(2)
            for value in values:
                line = _plain(label())
                set_elided_text(line, value)
                lines.addWidget(line)
            folders.addLayout(lines, row, 1)
        folders.setColumnStretch(1, 1)
        self._info.addLayout(folders)
        self._info.addStretch()

    def _toggle(self) -> None:
        showing_json = self._stack.currentIndex() == 1
        self._stack.setCurrentIndex(0 if showing_json else 1)
        self._toggle_json.setText("{ } Show JSON" if showing_json else "✕ Hide JSON")

    def _open_clicked(self) -> None:
        if self._record is not None:
            self.open_requested.emit(self._record)


class OverviewPage(QWidget):
    open_requested = Signal(object)
    new_dataset_requested = Signal()
    new_iteration_requested = Signal(object)
    refresh_requested = Signal()

    def __init__(self, context: AppContext, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("page")
        self._context = context
        self._records: list[DatasetRecord] = []
        self._modified: dict[int, float | None] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 20)
        layout.setSpacing(18)
        head = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(4)
        titles.addWidget(label("Datasets", name="h1"))
        titles.addWidget(label("All datasets found in your nnU-Net folders. Select one to see its details, "
                               "open it to work on it.", muted=True))
        head.addLayout(titles, 1)
        self._iteration = button("＋ New iteration", purple=True, clicked=self._new_iteration)
        head.addWidget(self._iteration, 0, Qt.AlignmentFlag.AlignBottom)
        head.addWidget(button("＋ New dataset", primary=True, clicked=self.new_dataset_requested.emit),
                       0, Qt.AlignmentFlag.AlignBottom)
        context.annotation_changed.connect(lambda _enabled: self._update_iteration())
        self._selected_record: DatasetRecord | None = None
        self._records: list[DatasetRecord] = []
        self._update_iteration()  # hidden from the start while the Annotation Workflow is off
        layout.addLayout(head)
        self.paths = PathsCard()
        layout.addWidget(self.paths)
        split = QHBoxLayout()
        split.setSpacing(18)
        left = QVBoxLayout()
        left.setSpacing(12)
        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search datasets by name or ID…")
        self._search.addAction(svg_icon("search", "#94a3b8"), QLineEdit.ActionPosition.LeadingPosition)
        self._search.setMinimumWidth(180)
        self._search.setMaximumWidth(360)
        self._search.textChanged.connect(self._filter)  # hides rows: fast also with 1000 datasets
        toolbar.addWidget(self._search)
        self._count = QLabel()
        self._count.setStyleSheet("background: #1e293b; color: white; border-radius: 12px; padding: 4px 12px;"
                                  "font-weight: 600; font-size: 12px;")
        toolbar.addWidget(self._count)
        toolbar.addStretch()
        toolbar.addWidget(button("⟳ Refresh", ghost=True, small=True, tooltip="Reload datasets from the nnU-Net folders",
                                 clicked=self.refresh_requested.emit))
        self._sort = QComboBox()
        self._sort.addItems(["Sort: ID", "Sort: name", "Sort: modified"])
        self._sort.currentIndexChanged.connect(self._render)
        toolbar.addWidget(self._sort)
        left.addLayout(toolbar)
        table_card = Card()
        self._table = make_table(["ID", "Dataset", "Available", "Modified", ""], stretch=1, row_height=40)
        self._table.setItemDelegateForColumn(2, AvailabilityDelegate(self._table))  # painted: no widget per row
        header = self._table.horizontalHeader()
        for column in (0, 2, 3, 4):  # widths set in _fit_columns: measuring 1000 rows on every change is slow
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
        self._table.itemSelectionChanged.connect(self._selection_changed)
        self._table.cellDoubleClicked.connect(lambda row, _column: self._open_row(row))
        self._table_stack = QStackedWidget()
        self._table_stack.addWidget(self._table)
        self._empty = empty_state("")
        self._table_stack.addWidget(self._empty)
        table_card.body.addWidget(self._table_stack)
        left.addWidget(table_card, 1)
        split.addLayout(left, 1)
        self.details = DetailsPanel()
        self.details.open_requested.connect(self.open_requested.emit)
        split.addWidget(self.details)
        layout.addLayout(split, 1)

    def set_records(self, records: list[DatasetRecord], keep: int | None = None) -> None:
        self._records = records
        self._modified = {record.dataset_id: _record_modified(record) for record in records}
        self._render(keep=keep)
        self._update_iteration()

    def _ordered(self) -> list[DatasetRecord]:
        records = list(self._records)
        order = self._sort.currentIndex()
        if order == 1:
            records.sort(key=lambda record: record.name.casefold())
        elif order == 2:
            records.sort(key=lambda record: -(self._modified.get(record.dataset_id) or 0))
        return records

    def _render(self, *_args, keep: int | None = None) -> None:
        """Rebuild the rows (new records or another order); the search only hides rows (``_filter``)."""
        selected = keep if keep is not None else self._selected_id()
        records = self._ordered()
        self._count.setText(f"All {len(self._records)}")
        self._table.blockSignals(True)
        self._table.setUpdatesEnabled(False)
        self._table.setRowCount(0)  # also removes the cell widgets
        self._table.setRowCount(len(records))
        for row, record in enumerate(records):
            self._table.setItem(row, 0, table_item(f"{record.dataset_id:03d}", mono=True, data=record))
            # with an ID problem the name is drawn by the cell widget (name and red badge) instead of the item
            id_warning = id_badge(record)
            self._table.setItem(row, 1, table_item("" if id_warning else record.name, bold=True))
            if id_warning:
                name = ElidedLabel(record.name)  # shrinks with "…", so the badge stays visible inside the column
                name.setTextFormat(Qt.TextFormat.PlainText)
                font = name.font()
                font.setBold(True)
                name.setFont(font)
                self._table.setCellWidget(row, 1, cell_widget(name, badge(*id_warning)))
            self._table.setItem(row, 2, availability_item(
                record.raw_path is not None, record.preprocessed_path is not None, record.results_path is not None))
            self._table.setItem(row, 3, table_item(format_time(self._modified.get(record.dataset_id)), muted=True))
            arrow = table_item("›")
            arrow.setForeground(Qt.GlobalColor.blue)
            self._table.setItem(row, 4, arrow)
        self._table.blockSignals(False)
        self._fit_columns(records)
        self._table.setUpdatesEnabled(True)
        self._filter(selected=selected)

    def _fit_columns(self, records: list[DatasetRecord]) -> None:
        header = self._table.horizontalHeader()
        titles = header.fontMetrics()
        mono = QFontMetrics(self._table.item(0, 0).font()) if records else self._table.fontMetrics()
        texts = {format_time(self._modified.get(record.dataset_id)) for record in records}
        contents = {
            0: max((mono.horizontalAdvance(f"{record.dataset_id:03d}") for record in records), default=0),
            2: availability_width(),
            3: max((self._table.fontMetrics().horizontalAdvance(text) for text in texts), default=0),
            4: self._table.fontMetrics().horizontalAdvance("›"),
        }
        for column, width in contents.items():
            title = self._table.horizontalHeaderItem(column).text()
            header.resizeSection(column, max(width + 28, titles.horizontalAdvance(title) + 32))

    def select_dataset(self, dataset_id: int) -> None:
        """Select the dataset that was opened, so the overview (and ＋ New iteration) starts from it."""
        self._filter(selected=dataset_id)

    def focus_search(self) -> None:
        self._search.setFocus()
        self._search.selectAll()

    def _filter(self, *_args, selected: int | None = None) -> None:
        """Hide the rows that do not match the search; keep the selection when its row is still shown."""
        needle = self._search.text().strip().casefold()
        if selected is None:
            selected = self._selected_id()
        select_row = current_row = -1
        for row in range(self._table.rowCount()):
            record: DatasetRecord = self._table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            hidden = not record.matches(needle)
            if self._table.isRowHidden(row) != hidden:
                self._table.setRowHidden(row, hidden)
            if not hidden:
                if select_row < 0:
                    select_row = row
                if record.dataset_id == selected:
                    current_row = row
        self._show_empty(select_row >= 0)
        if current_row >= 0:
            select_row = current_row
        if select_row >= 0:
            self._table.selectRow(select_row)
        else:
            self._table.clearSelection()
            self.details.set_record(None, None)

    def _show_empty(self, visible: bool) -> None:
        """Instead of an empty table: what to do when there is no dataset (yet), or none matches the search."""
        if visible:
            self._table_stack.setCurrentWidget(self._table)
            return
        message = self._empty.findChild(QLabel)
        if self._records:
            message.setText("<b style='color:#1e293b; font-size:15px'>No dataset matches the search</b>")
        else:
            message.setText("<b style='color:#1e293b; font-size:15px'>No datasets found</b><br><br>Set your "
                            "nnU-Net folders above and/or create a new dataset with <b>＋ New dataset</b>.")
        self._table_stack.setCurrentWidget(self._empty)

    def _update_iteration(self) -> None:
        """Only with the Annotation Workflow; the source dataset is chosen in the dialog (default: the selected one)."""
        self._iteration.setVisible(self._context.settings.annotation_workflow)
        available = any(record.raw_path is not None for record in getattr(self, "_records", []))
        self._iteration.setEnabled(available)
        self._iteration.setToolTip(
            "Annotation Workflow: copies the images, labels and dataset.json of a dataset (the selected one by "
            "default) into a new dataset for the next iteration" if available
            else "No dataset in nnUNet_raw yet, so no iteration can be created.")

    def _new_iteration(self) -> None:
        self.new_iteration_requested.emit(self._selected_record)  # None: the dialog picks the first dataset

    def _selected_id(self) -> int | None:
        rows = self._table.selectionModel().selectedRows() if self._table.selectionModel() else []
        if not rows:
            return None
        record = self._table.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)
        return record.dataset_id if record else None

    def _selection_changed(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        record: DatasetRecord = self._table.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)
        self._selected_record = record
        self._update_iteration()
        self.details.set_record(record, None)
        self._context.tasks.run("overview-details", lambda: dataset_details(record),
                                lambda details: self.details.set_record(record, details),
                                lambda error: self.details.set_record(record, {"error": error}))

    def _open_row(self, row: int) -> None:
        item = self._table.item(row, 0)
        if item is not None:
            self.open_requested.emit(item.data(Qt.ItemDataRole.UserRole))
