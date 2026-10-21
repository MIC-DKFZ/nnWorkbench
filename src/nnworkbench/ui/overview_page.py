"""Start screen: nnU-Net folders with path profiles, all datasets and the selected dataset's details."""

from __future__ import annotations

import json
from pathlib import Path

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QGuiApplication
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from nnworkbench.models.dataset import DatasetRecord
from nnworkbench.path_setup import set_paths_command
from nnworkbench.services.dataset_contents import json_object, model_folds, trained_models
from nnworkbench.services.dataset_json import manifest_source

from .context import AppContext
from .widgets import (
    Card,
    FlowLayout,
    availability,
    badge,
    button,
    cell_widget,
    dataset_id_label,
    empty_state,
    fit_columns,
    format_time,
    label,
    make_table,
    section_label,
    separator,
    set_badge,
    svg_icon,
    table_item,
)

_KEYS = ("raw", "preprocessed", "results")


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
        outer.addLayout(left, 1)
        right = QVBoxLayout()
        right.setSpacing(8)
        right.addWidget(section_label("Path profile"))
        self._profile_list = QVBoxLayout()
        self._profile_list.setSpacing(8)
        right.addLayout(self._profile_list)
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
        return tuple(Path(value).expanduser() if value.strip() else None for value in self._local)  # type: ignore

    def _texts(self) -> tuple[str, ...]:
        return tuple(self._fields[key].text().strip() for key in _KEYS)

    def _load_fields(self) -> None:
        values = self._local if self._active == "Local" else tuple(self._profiles[self._active][key] for key in _KEYS)
        for key, value in zip(_KEYS, values):
            self._fields[key].setText(value)
        for browse in self._browse:
            browse.setEnabled(self._active == "Local")
        self._update_marks()
        self._render_profiles()

    def _update_marks(self) -> None:
        """✓ found, orange ! not set yet (normal on the first start), red ✗ set but not found."""
        missing = unset = 0
        for key in _KEYS:
            text = self._fields[key].text().strip()
            mark = self._marks[key]
            if self._active != "Local":
                mark.setText("")
                mark.setToolTip("")
                continue
            if not text:
                state, tooltip = "unset", "Not set yet: choose the folder with Browse… (an empty folder is fine)"
            elif Path(text).expanduser().is_dir():
                state, tooltip = "found", "Folder found"
            else:
                state, tooltip = "missing", f"Folder not found: {text}"
            mark.setText({"found": "✓", "unset": "!", "missing": "✗"}[state])
            mark.setObjectName({"found": "statusYes", "unset": "statusUnset", "missing": "statusNo"}[state])
            mark.setToolTip(tooltip)
            mark.style().unpolish(mark)
            mark.style().polish(mark)
            missing += state == "missing"
            unset += state == "unset"
        if self._active != "Local":
            set_badge(self._status, f"profile: {self._active}", "purple")
            self._note.setText("paths for another machine · used only for Export command")
        elif missing:
            set_badge(self._status, f"{missing} not found" + (f" · {unset} not set" if unset else ""), "red")
            self._note.setText("Check the path or choose another folder with Browse…")
        elif unset:
            set_badge(self._status, f"{unset} not set", "orange")
            self._note.setText("Please set the paths: choose each folder with Browse… (empty folders are fine)")
        else:
            set_badge(self._status, "✓ all found", "green")
            self._note.setText("shared with the command line (nnworkbench-set-paths)")

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
        add = button("＋ Add remote profile", clicked=self._add_profile)
        add.setStyleSheet("text-align: left; color: #64748b; font-weight: 500;")
        self._profile_list.addWidget(add)

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
        title = QPushButton(("● " if active else "") + ("☁ " if name != "Local" else "") + name)
        title.setObjectName("link")
        title.setStyleSheet("color: #1f5fbf;" if active else "color: #1e293b; font-weight: 500;")
        title.clicked.connect(lambda _=False, selected=name: self._activate(selected))
        layout.addWidget(title)
        layout.addStretch()
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
        self._name = label("", name="h2")
        self._folder = label("", muted=True, size=12)
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
        self._info.setContentsMargins(0, 0, 0, 0)
        self._info.setSpacing(10)
        self._stack.addWidget(info)
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
        self._layout = layout
        self._bottom = QWidget()  # absorbs the free space below the footnote while the details are shown
        layout.addWidget(self._bottom, 1)
        self._fit_stack()
        self.set_record(None, None)

    def _fit_stack(self) -> None:
        showing_json = self._stack.currentIndex() == 1
        self._layout.setStretchFactor(self._stack, 1 if showing_json else 0)
        self._bottom.setVisible(not showing_json)
        policy = QSizePolicy.Policy.Expanding if showing_json else QSizePolicy.Policy.Maximum
        self._stack.setSizePolicy(QSizePolicy.Policy.Preferred, policy)

    def set_record(self, record: DatasetRecord | None, details: dict | None) -> None:
        self._record = record
        while self._info.count():
            item = self._info.takeAt(0)
            if item.widget():
                item.widget().hide()
                item.widget().deleteLater()
            elif item.layout():
                while item.layout().count():
                    child = item.layout().takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
        self._open.setEnabled(record is not None)
        if record is None:
            self._id.setText("—")
            self._name.setText("Select a dataset")
            self._folder.setText("")
            self._json.setPlainText("")
            return
        self._id.setText(f"{record.dataset_id:03d}")
        self._name.setText(record.name)
        self._folder.setText(record.directory_name)
        if details is None:
            self._info.addWidget(label("Loading dataset.json…", muted=True))
            return
        manifest = details.get("manifest")
        self._json.setPlainText(json.dumps(manifest, indent=2, ensure_ascii=False) if manifest else
                                "No dataset.json found.")
        if not isinstance(manifest, dict):
            self._info.addWidget(label("No dataset.json found in nnUNet_raw, nnUNet_preprocessed or nnUNet_results.",
                                       muted=True, wrap=True))
            return
        description = manifest.get("description")
        if isinstance(description, str) and description.strip():
            self._info.addWidget(label(description, muted=True, wrap=True))
        channels = manifest.get("channel_names") if isinstance(manifest.get("channel_names"), dict) else {}
        labels = manifest.get("labels") if isinstance(manifest.get("labels"), dict) else {}
        classes = [name for name in labels if str(name).casefold() != "ignore"]
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        rows = (("Modality", " · ".join(str(value) for value in channels.values()) or "—",
                 f"{len(channels)} channel{'s' if len(channels) != 1 else ''}"),
                ("Cases", str(manifest.get("numTraining", "—")), "numTraining"),
                ("Labels", f"{len(classes)} classes", "+ ignore" if len(classes) != len(labels) else "dense"),
                ("File ending", str(manifest.get("file_ending", "—")), ""))
        for row, (key, value, note) in enumerate(rows):
            grid.addWidget(label(key, muted=True), row, 0)
            text = label(f"{value} <span style='color:#64748b'>{'· ' + note if note else ''}</span>", rich=True)
            grid.addWidget(text, row, 1)
        grid.setColumnStretch(1, 1)
        self._info.addLayout(grid)
        chips_holder = QWidget()
        chips = FlowLayout(chips_holder)
        for name, value in labels.items():
            chip = QLabel(f"{name} · {value}")
            chip.setObjectName("chip")
            if str(name).casefold() == "ignore":
                chip.setProperty("ignore", True)
            chips.addWidget(chip)
        self._info.addWidget(chips_holder)
        self._info.addWidget(separator())
        folders = QGridLayout()
        folders.setHorizontalSpacing(12)
        folders.setVerticalSpacing(6)
        mode = "sparse labels" if any(str(name).casefold() == "ignore" for name in labels) else "dense labels"
        entries = (("RAW", ", ".join(details["raw_folders"]) + f" · {mode}" if details["raw_folders"] else "—"),
                   ("PRE", ", ".join(details["plans"]) or "—"),
                   ("RES", f"{details['models']} model{'s' if details['models'] != 1 else ''} · "
                           f"{details['folds']} fold{'s' if details['folds'] != 1 else ''}" if details["models"] else "—"))
        for row, (key, value) in enumerate(entries):
            folders.addWidget(label(key, muted=True), row, 0)
            folders.addWidget(label(value, wrap=True), row, 1)
        folders.setColumnStretch(1, 1)
        self._info.addLayout(folders)

    def _toggle(self) -> None:
        showing_json = self._stack.currentIndex() == 1
        self._stack.setCurrentIndex(0 if showing_json else 1)
        self._toggle_json.setText("{ } Show JSON" if showing_json else "✕ Hide JSON")
        self._fit_stack()

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
        self._search.setFixedWidth(360)
        self._search.textChanged.connect(self._render)
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

    def _visible(self) -> list[DatasetRecord]:
        needle = self._search.text().strip().casefold()
        records = [record for record in self._records
                   if not needle or needle in record.name.casefold() or needle in f"{record.dataset_id:03d}"]
        order = self._sort.currentIndex()
        if order == 1:
            records.sort(key=lambda record: record.name.casefold())
        elif order == 2:
            records.sort(key=lambda record: -(self._modified.get(record.dataset_id) or 0))
        return records

    def _render(self, *_args, keep: int | None = None) -> None:
        selected = keep if keep is not None else self._selected_id()
        visible = self._visible()
        self._count.setText(f"All {len(self._records)}")
        self._table.blockSignals(True)
        self._table.setRowCount(len(visible))
        select_row = 0 if visible else -1
        for row, record in enumerate(visible):
            self._table.setItem(row, 0, table_item(f"{record.dataset_id:03d}", mono=True, data=record))
            self._table.setItem(row, 1, table_item(record.name, bold=True))
            self._table.setCellWidget(row, 2, cell_widget(availability(
                record.raw_path is not None, record.preprocessed_path is not None, record.results_path is not None)))
            self._table.setItem(row, 3, table_item(format_time(self._modified.get(record.dataset_id)), muted=True))
            arrow = table_item("›")
            arrow.setForeground(Qt.GlobalColor.blue)
            self._table.setItem(row, 4, arrow)
            if record.dataset_id == selected:
                select_row = row
        self._table.blockSignals(False)
        fit_columns(self._table)
        self._show_empty(visible)
        if select_row >= 0:
            self._table.selectRow(select_row)
        else:
            self.details.set_record(None, None)

    def _show_empty(self, visible: list[DatasetRecord]) -> None:
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
            "default) into a new dataset for the next annotation round" if available
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
                                lambda details: self.details.set_record(record, details))

    def _open_row(self, row: int) -> None:
        item = self._table.item(row, 0)
        if item is not None:
            self.open_requested.emit(item.data(Qt.ItemDataRole.UserRole))
