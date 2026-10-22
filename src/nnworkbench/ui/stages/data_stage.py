"""Data stage: the raw dataset, its cases, statistics and dataset.json."""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from nnworkbench.services.dataset_scanner import id_badge
from nnworkbench.services.workflow_status import CaseRow, geometries_match, read_geometry

from ..context import AppContext
from ..dataset_state import DatasetState
from ..description_panel import DescriptionPanel
from ..json_editor import DatasetJsonPanel
from ..label_statistics_dialog import LabelStatisticsView
from ..widgets import (
    Bar,
    Card,
    ElidedLabel,
    HowToCard,
    IconButtonsDelegate,
    ToggleGroup,
    badge,
    breakable,
    button,
    clear_layout,
    empty_state,
    format_size,
    format_time,
    icon_html,
    icon_label,
    label,
    make_table,
    mark_item,
    polish,
    scroll,
    section_label,
    separator,
    table_item,
    tip,
)
from .common import StagePage, label_mode_badge

MAX_CASE_CLASSES = 30  # rows in the case panel's class list


def _howto() -> HowToCard:
    return HowToCard("The raw dataset", [
        "Images go to <b>imagesTr</b>, one file per channel: <span style='font-family:monospace'>cells_001_0000.nii.gz</span>.",
        "Labels go to <b>labelsTr</b> with the same case name, without the channel suffix.",
        ("<b>dataset.json</b> describes channels, labels and the file ending. An optional <b>ignore</b> label marks "
        "voxels that are not annotated; it must have the highest label ID."),
        (f"{icon_html('search')} inspects a case in Napari; {icon_html('edit')} annotates it (saved to <b>labelsTr</b>; "
         "shown when <b>Annotation Workflow</b> is on in the top bar)."),
    ], [], "Setup · Create a dataset")


def _elided(text: str, *, mono: bool = False, bold: bool = False, size: int | None = None,
            mode=Qt.TextElideMode.ElideRight) -> ElidedLabel:
    """A one-line label that ends in "…" when it does not fit (full text on hover)."""
    widget = ElidedLabel(text, mode=mode)
    if mono:
        widget.setProperty("mono", True)
    if bold or size:
        font = widget.font()
        if size:
            font.setPixelSize(size)
        font.setBold(bold)
        widget.setFont(font)
    return widget


class CaseStatisticsPanel(QFrame):
    """Right-hand panel with files, header geometry and classes of one case."""

    def __init__(self, stage: DataStage) -> None:
        super().__init__()
        self.setObjectName("card")
        self.setStyleSheet("QFrame#card { border: 1px solid #1f5fbf; }")
        self._stage = stage
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)
        head = QHBoxLayout()
        head.addWidget(icon_label("chart", 16))
        names = QVBoxLayout()
        names.setSpacing(0)
        self._case = _elided("", mono=True, bold=True)
        self._position = label("", muted=True, size=11)
        names.addWidget(self._case)
        names.addWidget(self._position)
        head.addLayout(names, 1)
        head.addWidget(button("‹", ghost=True, small=True, clicked=lambda: stage.step_case(-1)))
        head.addWidget(button("›", ghost=True, small=True, clicked=lambda: stage.step_case(1)))
        head.addWidget(button("✕", ghost=True, small=True, clicked=stage.close_case))
        layout.addLayout(head)
        # files, geometry and classes scroll (many channels or classes); the head and the actions stay in view
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 4, 0)
        self._content = QVBoxLayout()
        self._content.setSpacing(8)
        body_layout.addLayout(self._content)
        body_layout.addStretch()
        layout.addWidget(scroll(body), 1)
        actions = QHBoxLayout()
        self._inspect = button("Open in Napari", small=True, icon="search",
                               clicked=lambda: self._row is not None and stage.inspect(self._row))
        self._annotate = button("Annotate", small=True, icon="edit",
                                clicked=lambda: self._row is not None and stage.annotate(self._row))
        actions.addWidget(self._inspect)
        actions.addWidget(self._annotate)
        actions.addStretch()
        layout.addLayout(actions)
        self._row: CaseRow | None = None

    def show_case(self, row: CaseRow, index: int, total: int, state: DatasetState) -> None:
        self._row = row
        self._case.setText(row.case_id)
        self._position.setText(f"Case {index + 1} of {total}")
        clear_layout(self._content)
        self._content.addWidget(separator())
        self._content.addWidget(section_label("Files"))
        for path in (*row.image_files, *([row.label_file] if row.label_file else [])):
            line = QHBoxLayout()
            line.addWidget(_elided(f"{path.parent.name}/{path.name}", mono=True, size=12,
                                   mode=Qt.TextElideMode.ElideMiddle), 1)
            try:
                stat = path.stat()
                line.addWidget(label(f"{format_size(stat.st_size)} · {format_time(stat.st_mtime)}", muted=True, size=12))
            except OSError:
                pass
            self._content.addLayout(line)
        if row.label_file is None:
            self._content.addWidget(label("No label yet. Annotate creates it (all ignore for sparse labels, "
                                          "all background for dense labels).", muted=True, wrap=True, size=12))
        self._content.addWidget(separator())
        self._content.addWidget(label("<b style='color:#64748b'>GEOMETRY</b> <span style='color:#64748b'>"
                                      "· from file headers</span>", rich=True, size=11))
        self._geometry = QVBoxLayout()
        self._geometry.addWidget(label("Reading headers…", muted=True, size=12))
        self._content.addLayout(self._geometry)
        self._content.addWidget(separator())
        self._add_classes(row, state)
        self._annotate.setVisible(self._stage.context.settings.annotation_workflow and not state.regions)
        image = row.image_files[0] if row.image_files else None
        label_file = row.label_file
        self._stage.context.tasks.run(
            "case-geometry",
            lambda: (read_geometry(image) if image else None, read_geometry(label_file) if label_file else None),
            lambda result: self._show_geometry(row, result))

    def _show_geometry(self, row: CaseRow, result: object) -> None:
        if row is not self._row:
            return
        clear_layout(self._geometry)
        image, label_geometry = result
        if image is None:
            self._geometry.addWidget(label("The header could not be read.", muted=True, size=12))
            return
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(3)
        entries = (("Shape", " × ".join(str(value) for value in image.shape)),
                   ("Spacing", " × ".join(f"{value:g}" for value in image.spacing) + " mm"),
                   ("Origin", "[" + ", ".join(f"{value + 0.0:g}" for value in image.origin) + "]"))
        for index, (key, value) in enumerate(entries):
            grid.addWidget(label(key, muted=True, size=12), index, 0)
            grid.addWidget(label(value, mono=True, selectable=True), index, 1)
        direction = label("[" + ", ".join(f"{value + 0.0:g}" for value in image.direction) + "]", mono=True, wrap=True, selectable=True)
        grid.addWidget(label("Direction", muted=True, size=12), len(entries), 0, Qt.AlignmentFlag.AlignTop)
        grid.addWidget(direction, len(entries), 1)
        grid.setColumnStretch(1, 1)
        self._geometry.addLayout(grid)
        match = geometries_match(image, label_geometry)
        if match is True:
            self._geometry.addWidget(badge("✓ image and label match", "green"), 0, Qt.AlignmentFlag.AlignLeft)
        elif match is False:
            self._geometry.addWidget(badge("! image and label geometry differ", "red"), 0, Qt.AlignmentFlag.AlignLeft)

    def _add_classes(self, row: CaseRow, state: DatasetState) -> None:
        analysis = state.analysis or {}
        entry = None
        if row.label_file is not None:
            entry = next((item for item in analysis.get("per_image", []) if item.get("image") == row.label_file.name),
                         None)
        head = QHBoxLayout()
        head.addWidget(section_label("Classes"))
        head.addStretch()
        self._content.addLayout(head)
        if entry is None:
            message = ("No label yet." if row.label_file is None else
                       "Not in the statistics yet. Run Compute statistics to see the classes of this case.")
            self._content.addWidget(label(message, muted=True, wrap=True, size=12))
            return
        names = state.names
        voxels = {int(key): int(value) for key, value in entry.get("voxels", {}).items()}
        classes = [class_id for class_id in names if names[class_id] != "ignore"]
        # IDs that dataset.json does not define are counted apart and flagged
        undefined = {key: value for key, value in voxels.items() if value > 0 and key not in names}
        present = {key: value for key, value in voxels.items() if value > 0 and key in names}
        head.addWidget(label(f"{len(present)} of {len(classes)}", muted=True, size=11))
        if undefined:
            listed = ", ".join(f"{key} ({value:,} voxel{'s' if value != 1 else ''})"
                               for key, value in sorted(undefined.items()))
            self._content.addWidget(tip(f"! Undefined label IDs: {listed}. dataset.json does not define them; "
                                        "nnU-Net fails on them.", "orange"))
        top = max(present.values(), default=1)
        # one row per class; a label with hundreds of values (e.g. an image saved as label) shows the largest only
        shown = sorted(sorted(present, key=lambda class_id: -present[class_id])[:MAX_CASE_CLASSES])
        for class_id in shown:
            line = QHBoxLayout()
            name = _elided(names.get(class_id, str(class_id)), size=12)
            name.setToolTip(f"{class_id}: {names.get(class_id, str(class_id))} · {present[class_id]:,} voxels")
            name.setFixedWidth(128)
            line.addWidget(name)
            line.addWidget(Bar(present[class_id] / top), 1)
            value = present[class_id]
            line.addWidget(label(f"{value / 1e6:.2f} M" if value >= 1e5 else f"{value:,}", muted=True, size=12))
            self._content.addLayout(line)
        if len(present) > len(shown):
            self._content.addWidget(label(f"and {len(present) - len(shown):,} more with fewer voxels (all in "
                                          "Statistics, Per image)", muted=True, wrap=True, size=11))
        missing = [names[class_id] for class_id in classes if class_id not in present]
        if missing:
            more = f" and {len(missing) - MAX_CASE_CLASSES:,} more" if len(missing) > MAX_CASE_CLASSES else ""
            # long names are shortened and may break after "_", so the list wraps inside the panel
            shown_names = [name if len(name) <= 24 else breakable(name if len(name) <= 40 else name[:39] + "…")
                           for name in missing[:MAX_CASE_CLASSES]]
            text = label("Missing: " + ", ".join(shown_names) + more, muted=True, wrap=True, size=11)
            text.setToolTip("Missing: " + ", ".join(missing[:MAX_CASE_CLASSES]) + more)
            self._content.addWidget(text)


class DataStage(StagePage):
    key, title, guide = "data", "Data", ("setup.md", "create-a-dataset")

    def __init__(self, context: AppContext, parent=None) -> None:
        super().__init__(context, _howto(), parent)
        # both label modes stay visible; the one of this dataset is highlighted in set_state
        self._sparse_tip = self.howto.add_tip(
            f"{icon_html('edit')} <b>Sparse labels</b> (with ignore label): an empty mask is initialised with <i>ignore</i>; "
            "annotate only some regions, the rest is ignored in training.", "gray")
        self._dense_tip = self.howto.add_tip(
            f"{icon_html('edit')} <b>Dense labels</b> (no ignore label): an empty mask is initialised with background; annotate "
            "every object.", "gray")
        self._rows: list[CaseRow] = []
        self._case_index = -1
        self._analyze = button("Compute statistics", primary=True, clicked=self._analyze_dataset)
        self.header_actions.addWidget(self._analyze)
        # Files / Statistics card
        files = Card()
        head = files.add_header()
        self._path = _elided("", mono=True, mode=Qt.TextElideMode.ElideMiddle)
        head.addWidget(self._path, 1)
        self._toggle = ToggleGroup(["Files", "Statistics", "Description"], icons=["file", "chart", "edit"])
        self._toggle.set_tooltip(1, "Statistics of the whole dataset (run Compute statistics first); "
                                    "the chart button in a row shows one case")
        self._toggle.set_tooltip(2, "dataset-description.md: describe the images and define the classes "
                                    "(guide: 1. Describe and 2. Define), with example images")
        self._toggle.changed.connect(self._toggle_view)
        head.addWidget(self._toggle)
        self._views = QStackedWidget()
        files_view = QWidget()
        files_layout = QVBoxLayout(files_view)
        files_layout.setContentsMargins(0, 0, 0, 0)
        files_layout.setSpacing(0)
        band = QFrame()
        band.setObjectName("cardBand")
        band_row = QHBoxLayout(band)
        band_row.setContentsMargins(16, 8, 16, 8)
        self._counts = label("", muted=True, mono=True, rich=True)
        band_row.addWidget(self._counts)
        files_layout.addWidget(band)
        self._table = make_table(["Case", "Images", "Labels", ""], stretch=0)  # last edit: hover over Labels
        self._table.cellClicked.connect(lambda row, _column: self._select_row(row))
        self._table_stack = QStackedWidget()
        self._table_stack.addWidget(self._table)
        self._table_stack.addWidget(empty_state("Loading files…"))
        files_layout.addWidget(self._table_stack, 1)
        self._fill_version = 0
        self._table.setItemDelegateForColumn(3, IconButtonsDelegate(self._table, self._row_buttons, self._row_clicked))
        # Fixed widths: resizing to contents would re-measure every row and freeze large datasets.
        header = self._table.horizontalHeader()
        for column, width in ((1, 80), (2, 80)):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
            self._table.setColumnWidth(column, width)
        self._counts_text = ""
        self._views.addWidget(files_view)
        self._stats_holder = QWidget()
        self._stats_layout = QVBoxLayout(self._stats_holder)
        self._stats_layout.setContentsMargins(0, 0, 0, 0)
        self._views.addWidget(self._stats_holder)
        self._stats_state: DatasetState | None = None  # built when the Statistics view is shown (large datasets)
        self._views.currentChanged.connect(lambda index: index == 1 and self._build_statistics())
        self._description = DescriptionPanel()
        self._views.addWidget(self._description)
        self._description.dataset_json_changed.connect(self.reload_requested.emit)  # dataset.json panel and checks
        files.body.addWidget(self._views, 1)
        self.center.addWidget(files, 1)
        # Right: dataset.json or case statistics
        self._json = DatasetJsonPanel()
        self._json.saved.connect(self.reload_requested.emit)
        self.right.addWidget(self._json)
        self._case_panel = CaseStatisticsPanel(self)
        self.right.addWidget(self._case_panel)

    # ------------------------------------------------------------ state

    def confirm_leave(self) -> bool:
        """Before another dataset opens or nnWorkbench quits: Save / Discard / Cancel for each editor with unsaved
        changes. False: stay (Cancel, or saving failed)."""
        for name, editor in (("the description", self._description), ("dataset.json", self._json)):
            if not editor.has_unsaved_changes():
                continue
            answer = QMessageBox.question(
                self, "Unsaved changes", f"Save the changes to {name} of {self.state.record.directory_name}?"
                if self.state else f"Save the changes to {name}?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Save)
            if answer == QMessageBox.StandardButton.Save:
                if not editor.save():
                    return False
            elif answer == QMessageBox.StandardButton.Discard:
                editor.discard()
            else:
                return False
        return True

    def set_state(self, state: DatasetState) -> None:
        super().set_state(state)
        record = state.record
        annotation = self.context.settings.annotation_workflow
        for widget, active in ((self._sparse_tip, state.sparse), (self._dense_tip, not state.sparse)):
            widget.setVisible(annotation)
            widget.setProperty("tone", "purple" if active else "gray")
            polish(widget)
        raw = state.raw
        self._rows = list(raw.cases)
        self._path.setText(f"nnUNet_raw/{record.directory_name}" if record.raw_path else "no raw data in nnUNet_raw")
        self._counts_text = (f"imagesTr <b style='color:#1e293b'>{raw.image_files}</b>  ·  labelsTr "
                             f"<b style='color:#1e293b'>{raw.label_files}</b>  ·  dataset.json")
        self._description.set_dataset(record.raw_path)
        self._fill_table(annotation)
        self._fill_statistics(state)
        note = ""
        if state.manifest_kind not in (None, "raw"):
            note = f"Read from nnUNet_{state.manifest_kind}; it can only be edited in nnUNet_raw."
        self._json.set_manifest(record.raw_path, state.manifest, editable=record.raw_path is not None, note=note,
                                cases=len(raw.cases) if state.manifest_kind == "raw" and raw.cases else None)
        if self.right.currentWidget() is self._case_panel and 0 <= self._case_index < len(self._rows):
            self._case_panel.show_case(self._rows[self._case_index], self._case_index, len(self._rows), state)
        else:
            self.close_case()
        self.update_actions()

    BATCH = 1000  # rows per event-loop turn, so very large datasets keep the GUI responsive

    def show_loading(self) -> None:
        """Shown while the dataset's files are collected in the background."""
        self._fill_version += 1
        self._rows = []
        self._table.setRowCount(0)
        self._table_stack.setCurrentIndex(1)
        self._counts.setText("Loading files…")

    def _row_buttons(self, index: int) -> list:
        buttons = [("search", "Open in Napari · image and label", False)]
        if self.context.settings.annotation_workflow and not (self.state and self.state.regions):
            buttons.append(("edit", "Annotate in Napari · saves to labelsTr", False))
        open_case = self._case_index if self.right.currentWidget() is self._case_panel else -1
        buttons.append(("chart", "Case statistics · classes, geometry, files", index == open_case))
        return buttons

    def _row_clicked(self, index: int, button_number: int) -> None:
        if not 0 <= index < len(self._rows):
            return
        names = [icon for icon, _tip, _checked in self._row_buttons(index)]
        action = names[button_number]
        if action == "search":
            self.inspect(self._rows[index])
        elif action == "edit":
            self.annotate(self._rows[index])
        else:
            self.show_case(index)

    def _fill_table(self, annotation: bool) -> None:
        self._fill_version += 1
        self._table.setRowCount(0)
        self._table.setRowCount(len(self._rows))
        self._table_stack.setCurrentIndex(0)
        width = 16 + (3 if annotation else 2) * 34
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(3, width)
        self._fill_batch(0, self._fill_version)

    def _fill_batch(self, start: int, version: int) -> None:
        if version != self._fill_version:
            return  # a newer dataset or refresh replaced this table
        stop = min(start + self.BATCH, len(self._rows))
        self._table.setUpdatesEnabled(False)
        for index in range(start, stop):
            row = self._rows[index]
            self._table.setItem(index, 0, table_item(row.case_id, mono=True))
            self._table.setItem(index, 1, mark_item(row.channels_present))
            labels = mark_item(row.has_label)
            labels.setToolTip(f"Last edit: {format_time(row.last_edit)}" if row.has_label else "No label yet")
            self._table.setItem(index, 2, labels)
        self._table.setUpdatesEnabled(True)
        if stop < len(self._rows):
            self._counts.setText(f"{self._counts_text}  ·  loading {stop} / {len(self._rows)}…")
            QTimer.singleShot(0, lambda: self._fill_batch(stop, version))
        else:
            self._counts.setText(self._counts_text)

    def _update_stats_buttons(self) -> None:
        self._table.viewport().update()

    def _fill_statistics(self, state: DatasetState) -> None:
        clear_layout(self._stats_layout)
        self._stats_state = state
        if self._views.currentWidget() is self._stats_holder:
            self._build_statistics()

    def _build_statistics(self) -> None:
        state, self._stats_state = self._stats_state, None
        if state is None:
            return  # built already
        if state.analysis is None:
            band = QFrame()
            band.setObjectName("cardBand")
            band_row = QHBoxLayout(band)
            band_row.setContentsMargins(16, 8, 16, 8)
            band_row.addWidget(label("Statistics · not computed yet", muted=True, size=12))
            self._stats_layout.addWidget(band)
            self._stats_layout.addWidget(empty_state(
                "<b style='color:#1e293b; font-size:15px'>No statistics yet</b><br><br>The statistics (class "
                "distribution, voxels per class, image shape and spacing) are computed from the label files once and "
                "then reused. They have not been computed yet: run <b>Compute statistics</b> (above) first. It reads every "
                "label file and can take a few minutes for large datasets.", "chart"), 1)
            return
        view = LabelStatisticsView(state.analysis, annotation_workflow_enabled=state.sparse)
        if state.analysis_outdated:
            view.header_row.insertWidget(1, badge(f"{state.analysis_outdated} label"
                                                  f"{'s' if state.analysis_outdated != 1 else ''} changed since", "orange",
                                                  "Run Update statistics to include the changed labels"))
        self._stats_layout.addWidget(view)

    def header_badges(self, state: DatasetState) -> list:
        """Only problems are listed (in red); without problems the dataset is ready. Label mode comes last."""
        raw = state.raw
        items = []
        if raw.missing_labels:
            items.append(badge(f"labels incomplete · {raw.missing_labels} image{'s' if raw.missing_labels != 1 else ''} "
                               "without label", "red"))
        if (id_warning := id_badge(state.record)) is not None:
            items.append(badge(*id_warning))
        if not raw.names_valid:
            items.append(badge("names invalid · images need a _0000 channel suffix", "red"))
        if not raw.channels_complete:
            items.append(badge("channels missing", "red"))
        count = state.manifest.get("numTraining") if isinstance(state.manifest, dict) else None
        if any(error.startswith("numTraining") for error in state.manifest_errors):
            items.append(badge(f"numTraining {count} ≠ {len(raw.cases)} cases", "red",
                               next(error for error in state.manifest_errors if error.startswith("numTraining"))))
        other = [error for error in state.manifest_errors if not error.startswith("numTraining")]
        if other:
            items.append(badge("dataset.json invalid", "red", " ".join(other)))
        if not raw.cases and state.raw_path is not None:
            items.append(badge("no images in imagesTr yet", "red", "Add images (and labels) to imagesTr and labelsTr."))
        if not items and (problems := state.data_problems()):  # e.g. file endings, colour channels, file list
            items.append(badge(f"must be fixed · {len(problems)} problem{'s' if len(problems) != 1 else ''}", "red",
                               " · ".join(problems)))
        if not items and raw.cases:
            items.append(badge("✓ ready for preprocessing", "green"))
        if state.inherits_from:
            items.append(badge(f"from {state.inherits_from}" + (" · images linked" if raw.file_list else ""), "gray",
                               f"A later round of {state.inherits_from} (inherits_from in dataset.json)"
                               + (". Its images are listed in dataset.json (\"dataset\"), not copied: nnU-Net "
                                  "reads only the listed files." if raw.file_list else ".")))
        if state.analysis and state.analysis_outdated:  # not a problem for preprocessing, only for the statistics
            changed = state.analysis_outdated
            items.append(badge(f"statistics outdated · {changed} label{'s' if changed != 1 else ''} changed", "orange",
                               "Labels changed after the last Compute statistics; click Update statistics."))
        return [*items, label_mode_badge(state)]

    def update_actions(self) -> None:
        if self.state is None:
            return
        reason = self.busy_or_missing("raw")
        has_labels = self.state.raw.label_files > 0
        self._analyze.setEnabled(not reason and has_labels)
        self._analyze.setText("Update statistics" if self.state.analysis is not None else "Compute statistics")
        tooltip = "Computes the dataset statistics: class distribution, voxels per class, image shape and spacing."
        if reason:
            tooltip += f" Disabled: {reason}"
        elif not has_labels:
            tooltip += " Disabled: labelsTr has no label files yet."
        self._analyze.setToolTip(tooltip)

    # ------------------------------------------------------------ actions

    def _toggle_view(self, index: int) -> None:
        self._views.setCurrentIndex(index)

    def _analyze_dataset(self) -> None:
        if self.state is None or self.state.raw_path is None:
            return
        self.context.jobs.start_analysis(self.state.record.name, self.state.raw_path, self.context.settings.workers)

    def inspect(self, row: CaseRow) -> None:
        target = row.image_files[0] if row.image_files else row.label_file
        if target is not None:
            self.context.inspect_case(target, self.state.raw_path if self.state else None)

    def annotate(self, row: CaseRow) -> None:
        image = _first_channel(row.image_files)
        if image is not None:
            self.context.annotate_case(image, self.state.raw_path if self.state else None)

    def _select_row(self, row: int) -> None:
        if self.right.currentWidget() is self._case_panel:
            self.show_case(row)

    def show_case(self, index: int) -> None:
        if self.state is None or not 0 <= index < len(self._rows):
            return
        if index == self._case_index and self.right.currentWidget() is self._case_panel:
            self.close_case()
            return
        self._case_index = index
        self.right.setCurrentWidget(self._case_panel)
        self._case_panel.show_case(self._rows[index], index, len(self._rows), self.state)
        self._table.selectRow(index)
        self._update_stats_buttons()

    def step_case(self, offset: int) -> None:
        if self._rows:
            index = (self._case_index + offset) % len(self._rows)
            self._case_index = -1
            self.show_case(index)

    def close_case(self) -> None:
        self._case_index = -1
        self.right.setCurrentWidget(self._json)
        self._update_stats_buttons()


def _first_channel(paths: tuple[Path, ...]) -> Path | None:
    return next((path for path in paths if "_0000." in path.name), paths[0] if paths else None)
