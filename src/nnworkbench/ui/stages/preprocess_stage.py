"""Preprocess stage: plans files, their configurations, the preprocessed folder and the run panel."""

from __future__ import annotations

import json
import os
from pathlib import Path

from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QColor, QFont, QFontMetrics
from qtpy.QtWidgets import (
    QAbstractScrollArea,
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QPlainTextEdit,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nnworkbench.services.commands import preprocessing_command
from nnworkbench.services.dataset_contents import json_value
from nnworkbench.services.defaults import (
    CONFIGURATIONS,
    DEFAULT_CONFIGURATION,
    DEFAULT_PLANNER,
    PLANNERS,
    preprocessing_configuration,
    preprocessing_processes,
)
from nnworkbench.services.workflow_status import PlansInfo

from .. import style
from ..context import AppContext
from ..dataset_state import DatasetState
from ..editable_dropdown import EditableDropdownComboBox
from ..widgets import (
    Card,
    ChoiceGroup,
    ElidedLabel,
    HowToCard,
    NumberField,
    badge,
    button,
    clear_layout,
    empty_state,
    form_row,
    format_time,
    icon_button,
    icon_html,
    icon_label,
    label,
    make_table,
    scroll,
    section_label,
    svg_icon,
    table_item,
    tip,
    with_stepper,
)
from .common import (
    CommandBox,
    RunDialog,
    StagePage,
    arguments_error,
    arguments_field,
    detail_row,
    header_note,
    wrappable,
)

_FILE_ROLE = Qt.ItemDataRole.UserRole + 1  # file rows of an expanded folder: the file to open in Napari


def _numbers(values: tuple, number_format: str = "") -> str:
    """Patch size or spacing from a plans file as "64 × 64 × 64"; entries that are not numbers are shown as they are."""
    return " × ".join(format(value, number_format) if isinstance(value, (int, float)) and not isinstance(value, bool)
                      else str(value) for value in values) or "—"


def _howto() -> HowToCard:
    return HowToCard("What preprocessing does", [
        "Analyses your images (<b>fingerprint</b>): sizes, spacing, intensities.",
        "Plans the network, patch size and preprocessing for your GPU (<b>plans</b>).",
        ("Resamples and normalises all cases and stores them in a <b>plans folder</b> (e.g. "
        "<span style='font-family:monospace'>nnUNetResEncUNetLPlans&#8203;_3d_fullres</span>). Training reads its data from "
        "this folder."),
    ], [("blue", (f"{icon_html('bulb')} Select a suitable planner (GPU memory, training time per fold on an A100):<br>"
                 "<b>ExperimentPlanner</b> <span style='white-space:nowrap'>(~8 GB, ~9 h)</span><br>"
                 "<b>nnUNetPlannerResEncM</b> <span style='white-space:nowrap'>(9–11 GB, ~12 h)</span><br>"
                 "<b>nnUNetPlannerResEncL</b> <span style='white-space:nowrap'>(24 GB, ~35 h)</span> "
                 "<span style='background:#1f5fbf; color:white; font-weight:600'>&nbsp;default&nbsp;</span><br>"
                 "<b>nnUNetPlannerResEncXL</b> <span style='white-space:nowrap'>(40 GB, ~66 h)</span><br>"
                 "or type a custom planner name."))], "Train · Preprocess")


class PreprocessStage(StagePage):
    key, title, guide = "preprocess", "Preprocess", ("4-train.md", "4-preprocess")
    folder = "preprocessed"

    def __init__(self, context: AppContext, parent=None) -> None:
        super().__init__(context, _howto(), parent)
        self._selected_plans: tuple[str, str] | None = None  # (plans, configuration)
        self._rows: list = []
        self._planners_loaded = False  # the planner list holds the installed planners (context.nnunet_classes)
        context.classes_changed.connect(self._set_planners)
        # Plans
        plans_card = Card()
        head = plans_card.add_header("Plans")
        self._plans_path = header_note(mono=True)
        head.addWidget(self._plans_path, 1)
        self._plans_count = label("", muted=True, size=12)
        head.addWidget(self._plans_count)
        # One row per preprocessed configuration; planned-only configurations are not listed.
        self._plans_table = make_table(["Plans", "Planner", "Configuration", "Preprocessed"], stretch=None)
        self._plans_table.horizontalHeaderItem(0).setToolTip("Plans = what preprocessing writes (e.g. nnUNetResEncUNetLPlans). The planner is what you run to create them (e.g. nnUNetPlannerResEncL).")
        self._plans_table.horizontalHeaderItem(1).setToolTip("Planner = what you run to create plans (e.g. nnUNetPlannerResEncL writes nnUNetResEncUNetLPlans).")
        self._plans_table.itemSelectionChanged.connect(self._plans_selected)
        # all entries up to five, shrinking to one (then it scrolls) in a low window
        self._plans_table.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        plans_card.body.addWidget(self._plans_table)
        self._plans_empty = empty_state("This dataset has not been preprocessed yet. Click ▶ Preprocess… to plan and preprocess it.")
        plans_card.body.addWidget(self._plans_empty)
        self.center.addWidget(plans_card)
        # Folder
        folder_card = Card()
        folder_head = folder_card.add_header()
        self._folder_path = ElidedLabel()
        self._folder_path.setProperty("mono", True)
        self._folder_path.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        folder_head.addWidget(self._folder_path, 1)
        folder_head.addWidget(label("listed from the folder", muted=True, size=12))
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setIndentation(18)
        # Files of an expanded folder are plain items (name, and a magnifier in the second column for cases):
        # one widget per row made a folder with thousands of cases take minutes to expand.
        self._tree.setColumnCount(2)
        self._tree.header().setStretchLastSection(False)
        self._tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self._tree.setColumnWidth(1, 40)
        self._tree.itemExpanded.connect(self._expand_folder)
        self._tree.itemClicked.connect(lambda item, column: column == 1 and self._file_activated(item))
        self._tree.itemActivated.connect(lambda item, _column: self._file_activated(item))
        self._file_font = QFont(style.MONO)
        self._file_font.setPixelSize(12)
        # Rows are widgets with their own spacing; the default item padding would cut off their buttons.
        self._tree.setStyleSheet("QTreeWidget::item { padding: 0px; border-bottom: 1px solid #f0f3f7; }")
        self._case_folder = None
        folder_card.body.addWidget(self._tree, 1)
        self.center.addWidget(folder_card, 1)
        # Right: run panel or JSON viewer
        # Right: details of the selected configuration (or the raw JSON of a file). New runs start from the
        # "▶ Preprocess…" button in the header, which opens the run form in a dialog.
        self.right.addWidget(scroll(self._build_details()))
        self._run_dialog = RunDialog(self, "Preprocess", self._build_run_panel(), on_open=self.update_actions)
        self.header_actions.addWidget(button("▶ Preprocess…", primary=True, clicked=self._run_dialog.open,
                                             tooltip="Plan and preprocess this dataset (choose planner and "
                                                     "configurations, see the command)"))

    def _build_details(self) -> QWidget:
        panel = Card()
        layout = QVBoxLayout()
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)
        panel.body.addLayout(layout)
        self._details_title = label("", name="h3", wrap=True)
        layout.addWidget(self._details_title)
        self._details_body = QVBoxLayout()
        self._details_body.setSpacing(8)
        layout.addLayout(self._details_body)
        # Like the details on Validate: the raw plans file below the summary.
        self._plans_json_label = section_label("Details")
        self._plans_json_label.setWordWrap(True)  # long plans file names
        self._plans_json_label.setMinimumWidth(1)
        layout.addWidget(self._plans_json_label)
        self._plans_json = QPlainTextEdit()
        self._plans_json.setObjectName("codeView")
        self._plans_json.setReadOnly(True)
        self._plans_json.setMinimumHeight(160)
        layout.addWidget(self._plans_json, 1)
        layout.addStretch()  # keeps title and text at the top while the plans file is hidden
        return panel

    def _show_details(self, plans: PlansInfo | None, config) -> None:
        clear_layout(self._details_body)
        if plans is None or config is None:
            self._details_title.setText("Not preprocessed yet")
            self._details_body.addWidget(label("This dataset has not been preprocessed yet. Click ▶ Preprocess… to plan and preprocess it.",
                                               muted=True, wrap=True))
            self._plans_json_label.hide()
            self._plans_json.hide()
            return
        self._show_json(plans.file)
        self._details_title.setText(wrappable(f"{plans.name} · {config.name}"))
        total = self.state.raw.label_files or len(self.state.raw.cases) if self.state else 0
        when = format_time(plans.preprocessed_time) if plans.preprocessed_time else "—"
        status = (badge("! outdated · labelsTr changed afterwards", "orange") if plans.outdated
                  else badge("✓ up to date", "green"))
        rows = (("Planner", label(wrappable(plans.planner), mono=True, wrap=True)),
                ("Preprocessed", f"{config.preprocessed_cases} / {total or config.preprocessed_cases} cases"),
                ("Date", when),
                ("Status", status),
                ("Patch size", label(_numbers(config.patch_size), mono=True)),
                ("Spacing", label(_numbers(config.spacing, ".3g"), mono=True)),
                ("Batch size", str(config.batch_size or "—")),
                ("Folder", label(wrappable(f"{config.data_identifier}/"), mono=True, wrap=True, selectable=True)))
        for name, value in rows:
            self._details_body.addLayout(detail_row(name, value))
        actions = QHBoxLayout()
        actions.addWidget(button("↻ Run again…", small=True, clicked=lambda: self._rerun(plans, config),
                                 tooltip="Open the preprocessing form with this planner and configuration prefilled, "
                                         "e.g. after labels changed"))
        actions.addStretch()
        self._details_body.addLayout(actions)

    def _rerun(self, plans: PlansInfo, config) -> None:
        """Prefill the run form with the planner and configuration of this entry and open it."""
        if plans.planner and plans.planner != "—":
            self._planner.setCurrentText(plans.planner)
        options = list(CONFIGURATIONS) + ([config.name] if config.name not in CONFIGURATIONS else [])
        self._configurations.set_options(options, selected=[config.name])
        self._run_dialog.open()

    def _build_run_panel(self) -> QWidget:
        panel = Card()
        layout = QVBoxLayout()
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)
        panel.body.addLayout(layout)
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        self._planner = EditableDropdownComboBox()
        self._planner.setEditable(True)
        self._planner.setProperty("mono", True)
        self._planner.addItems(list(PLANNERS))
        self._planner.setCurrentText(DEFAULT_PLANNER)

        self._planner.currentTextChanged.connect(self._update_command)
        self._configurations = ChoiceGroup(exclusive=False)
        self._configurations.set_options(CONFIGURATIONS, selected=[DEFAULT_CONFIGURATION])
        self._defaults_for: tuple | None = None  # dataset ID and default configuration selected for it
        self._configurations.changed.connect(self._update_command)
        self._processes = NumberField(1, 99, 8)
        self._processes.setRange(1, max(1, os.cpu_count() or 1))
        self._processes.setValue(preprocessing_processes())
        self._processes.valueChanged.connect(self._update_command)
        self._verify = QCheckBox("Verify dataset integrity")
        self._verify.setChecked(True)
        self._verify.toggled.connect(self._update_command)
        self._arguments = arguments_field()
        self._arguments.setPlaceholderText("Optional, e.g. -npfp 8")
        self._arguments.textChanged.connect(self._update_command)
        info = {
            "Planner": "The planner chooses a suitable network size, patch size and preprocessing for your data. "
                       "Planners differ in network size, needed GPU memory and training time.",
            "Configurations": "The configuration defines the data handling and the model input: <b>3d_fullres</b> "
                              "3D at full resolution (the usual choice), <b>2d</b> single slices (faster), "
                              "<b>3d_lowres</b> downsampled 3D (only for very large images).",
        }
        for name, widget in (("Planner", self._planner), ("Configurations", self._configurations),
                             ("Processes", with_stepper(self._processes)), ("Checks", self._verify),
                             ("Additional arguments", self._arguments)):
            layout.addLayout(form_row(name, widget, info=info.get(name, "")))
        layout.addWidget(section_label("Command"))
        self._command = CommandBox()
        layout.addWidget(self._command)
        actions = QHBoxLayout()
        actions.addWidget(self._command.copy_button())
        self._run = button("▶ Preprocess", small=True, primary=True, clicked=self._run_preprocessing)
        actions.addWidget(self._run)
        actions.addStretch()
        layout.addLayout(actions)
        self._run_hint = tip("", "orange")
        layout.addWidget(self._run_hint)
        layout.addStretch()
        return panel

    # ------------------------------------------------------------ state

    def set_state(self, state: DatasetState) -> None:
        super().set_state(state)
        self.context.discover_classes()  # once per session; _set_planners fills the list when it is known
        if not self._planners_loaded and self.context.nnunet_classes is not None:
            self._set_planners()
        record = state.record
        default = preprocessing_configuration(state.manifest, (config.name for plans in state.plans
                                                                for config in plans.configurations))
        if (record.dataset_id, default) != self._defaults_for:  # another dataset or new plans: 2d for 2D images
            self._defaults_for = (record.dataset_id, default)
            self._configurations.set_options(CONFIGURATIONS, selected=[default])
        self._plans_path.setText(f"nnUNet_preprocessed/{record.directory_name}")
        self._folder_path.setText(f"nnUNet_preprocessed/{record.directory_name}")
        self._plans_count.setText(f"{len(state.plans)} plan file{'s' if len(state.plans) != 1 else ''}")
        rows = [(plans, config) for plans in state.plans for config in plans.configurations if config.preprocessed]
        has_plans = bool(rows)
        self._plans_table.setVisible(has_plans)
        self._plans_empty.setVisible(not has_plans)
        self._rows = rows
        total = state.raw.label_files or len(state.raw.cases)
        self._plans_table.blockSignals(True)
        self._plans_table.clearSpans()
        self._plans_table.setRowCount(2 * len(rows))  # each entry is followed by its details line
        selected_row = 0
        for number, (plans, config) in enumerate(rows):
            row = 2 * number
            self._plans_table.setItem(row, 0, table_item(plans.name, mono=True, data=number))
            self._plans_table.setItem(row, 1, table_item(plans.planner, mono=True, muted=True))
            self._plans_table.setItem(row, 2, table_item(config.name, mono=True, bold=True))
            done = table_item(f"✓ {config.preprocessed_cases} / {total or config.preprocessed_cases}", bold=True)
            done.setForeground(QColor("#2f8f4e"))
            self._plans_table.setItem(row, 3, done)
            size, spacing = _numbers(config.patch_size), _numbers(config.spacing, ".3g")
            details = table_item(f"      patch size {size}  ·  spacing {spacing}  ·  batch size {config.batch_size or '—'}",
                                 muted=True, data=number)
            details.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self._plans_table.setItem(row + 1, 0, details)
            self._plans_table.setSpan(row + 1, 0, 1, 4)
            self._plans_table.setRowHeight(row + 1, 26)
            if (plans.name, config.name) == self._selected_plans:
                selected_row = row
        self._plans_table.blockSignals(False)
        self._fix_plan_columns()
        header = self._plans_table.horizontalHeader().sizeHint().height() + 4
        self._plans_table.setMinimumHeight(header + sum(self._plans_table.rowHeight(index)
                                                        for index in range(min(2 * len(rows), 2))))
        self._plans_table.setMaximumHeight(header + sum(self._plans_table.rowHeight(index)
                                                        for index in range(min(2 * len(rows), 10))))
        if has_plans:
            self._plans_table.selectRow(selected_row)
            self._plans_selected()
        else:
            self._fill_tree(None)
            self._show_details(None, None)
        self._update_command()

    def _fix_plan_columns(self) -> None:
        """Plans takes the free width (long names end with "…", the full name is the hover text); Planner,
        Configuration and Preprocessed fit their text, measured with the item font (mono, bold). Only the entry
        rows count: the spanning details lines must not widen a column."""
        table = self._plans_table
        header = table.horizontalHeader()
        header.setStretchLastSection(False)
        header_font = QFont(header.font())
        header_font.setBold(True)
        for column in range(1, table.columnCount()):
            widths = [QFontMetrics(header_font).horizontalAdvance(table.horizontalHeaderItem(column).text())]
            widths += [QFontMetrics(item.font()).horizontalAdvance(item.text())
                       for row in range(0, table.rowCount(), 2) if (item := table.item(row, column)) is not None]
            width = max(widths) + 28
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
            table.setColumnWidth(column, min(width, 220) if column == 1 else width)  # a custom planner may be long
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)

    def _set_planners(self) -> None:
        """The installed planners (also custom ones) in the dropdown; the typed or chosen name stays."""
        planners = self.context.nnunet_classes.planners if self.context.nnunet_classes else ()
        self._planners_loaded = bool(planners)
        if planners:
            current = self._planner.currentText()
            self._planner.blockSignals(True)
            self._planner.clear()
            self._planner.addItems(list(planners))
            self._planner.setCurrentText(current or DEFAULT_PLANNER)
            self._planner.blockSignals(False)

    def _plans_selected(self) -> None:
        selected = self._plans_table.selectionModel().selectedRows()
        if not selected or self.state is None:
            return
        plans, config = self._rows[self._plans_table.item(selected[0].row(), 0).data(Qt.ItemDataRole.UserRole)]
        self._selected_plans = (plans.name, config.name)
        self._show_details(plans, config)
        self._fill_tree(plans)

    ROW_HEIGHT = 34

    def _fill_tree(self, plans: PlansInfo | None) -> None:
        self._tree.clear()
        state = self.state
        root = state.record.preprocessed_path if state else None
        if root is None or not root.is_dir():
            item = QTreeWidgetItem(["This dataset has not been preprocessed yet."])
            self._tree.addTopLevelItem(item)
            item.setFirstColumnSpanned(True)
            return
        notes = {"dataset.json": "copy of raw", "dataset_fingerprint.json": "intensity & spacing statistics",
                 "splits_final.json": "folds for training"}
        for path in sorted(root.glob("*.json"), key=lambda item: item.name.casefold()):
            note = notes.get(path.name, "network & patch settings" if "plans" in path.name.casefold() else "")
            self._add_row(None, "file", path.name, note, "{ }", "Show JSON", lambda selected=path: self._show_json(selected))
        self._case_folder = next((root / config.data_identifier for config in (plans.configurations if plans else ())
                                  if config.preprocessed), None)
        gt = root / "gt_segmentations"
        if gt.is_dir():
            self._add_folder(gt, "gt_segmentations/", "ground-truth segmentations")
        for config in (plans.configurations if plans else ()):
            if config.preprocessed:
                self._add_folder(root / config.data_identifier, f"{config.data_identifier}/",
                                 f"{config.preprocessed_cases} cases · .b2nd + _seg.b2nd")

    def _add_row(self, parent: QTreeWidgetItem | None, icon: str, name: str, note: str, button_icon: str = "",
                 tooltip: str = "", clicked=None) -> QTreeWidgetItem:
        item = QTreeWidgetItem([""])
        item.setSizeHint(0, QSize(0, self.ROW_HEIGHT))
        if parent is None:
            self._tree.addTopLevelItem(item)
        else:
            parent.addChild(item)
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(4, 2, 8, 2)
        layout.setSpacing(8)
        if icon:
            layout.addWidget(icon_label(icon, 14))
        # long plans folders keep start and end (e.g. ..._3d_fullres/); full name on hover, Copy in the menu
        name_label = ElidedLabel(name, mode=Qt.TextElideMode.ElideMiddle)
        name_label.setProperty("mono", True)
        name_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout.addWidget(name_label, 1)
        if note:
            layout.addWidget(label(note, muted=True, size=12))
        if button_icon:
            layout.addWidget(icon_button(button_icon, tooltip, clicked))
        else:
            spacer = QWidget()
            spacer.setFixedWidth(30)
            layout.addWidget(spacer)
        item.setFirstColumnSpanned(True)
        self._tree.setItemWidget(item, 0, row)
        return item

    def _add_folder(self, folder: Path, name: str, note: str) -> None:
        item = self._add_row(None, "folder", name, note)
        item.setData(0, Qt.ItemDataRole.UserRole, str(folder))
        item.setChildIndicatorPolicy(QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator)
        item.setExpanded(False)  # folders stay collapsed; files are listed when expanded

    def _pair_target(self, path: Path) -> Path | None:
        """The preprocessed data file to open for any file of a case (image, _seg or gt segmentation)."""
        if path.name.endswith(".b2nd"):
            return path
        if self._case_folder is None:
            return None
        from nnworkbench.services.image_files import strip_image_suffix

        candidate = self._case_folder / f"{strip_image_suffix(path)}.b2nd"
        return candidate if candidate.is_file() else None

    def _inspect_pair(self, path: Path) -> None:
        target = self._pair_target(path)
        if target is not None:
            self.context.inspect_preprocessed(target)

    def _expand_folder(self, item: QTreeWidgetItem) -> None:
        folder = item.data(0, Qt.ItemDataRole.UserRole)
        if not folder or item.childCount():
            return
        try:
            with os.scandir(folder) as entries:
                names = sorted(entry.name for entry in entries if entry.is_file())
        except OSError as error:  # removed or unreadable since the tree was filled
            item.addChild(self._note_item(f"Cannot list this folder: {error.strerror or error}"))
            return
        if not names:
            item.addChild(self._note_item("empty folder"))
            return
        item.addChildren([self._file_item(Path(folder) / name) for name in names])

    def _note_item(self, text: str) -> QTreeWidgetItem:
        child = QTreeWidgetItem([text, ""])
        child.setForeground(0, QColor(style.MUTED))
        child.setSizeHint(0, QSize(0, self.ROW_HEIGHT))
        return child

    def _file_item(self, path: Path) -> QTreeWidgetItem:
        child = QTreeWidgetItem([path.name, ""])
        child.setFont(0, self._file_font)
        child.setToolTip(0, path.name)
        child.setSizeHint(0, QSize(0, self.ROW_HEIGHT))
        if path.name.endswith(".b2nd") or path.parent.name == "gt_segmentations":
            child.setData(0, _FILE_ROLE, str(path))
            child.setIcon(1, self._search_icon())
            child.setToolTip(1, "Open in Napari · preprocessed image and segmentation")
        return child

    def _search_icon(self):
        if getattr(self, "_search", None) is None:
            self._search = svg_icon("search")
        return self._search

    def _file_activated(self, item: QTreeWidgetItem) -> None:
        path = item.data(0, _FILE_ROLE)
        if path:
            self._inspect_pair(Path(path))

    # ------------------------------------------------------------ JSON panel

    def _show_json(self, path: Path) -> None:
        """Show any JSON file of the folder in the Details box (where the plans file is shown)."""
        self._plans_json_label.setText(wrappable(f"DETAILS · {path.name}"))
        self._plans_json_label.show()
        self._plans_json.show()
        self._plans_json.setToolTip(str(path))
        self._plans_json.setPlainText("Loading…")

        def text() -> str:
            data = json_value(path)
            return (json.dumps(data, indent=2, ensure_ascii=False) if data is not None
                    else f"{path.name} is missing, not valid JSON or not UTF-8.")

        self.context.tasks.run("preprocess-plans", text, lambda shown: self._plans_json.setPlainText(str(shown)),
                               lambda error: self._plans_json.setPlainText(f"Cannot show {path.name}: {error}"))

    # ------------------------------------------------------------ run

    def _command_text(self) -> str:
        if self.state is None:
            return ""
        configurations = self._configurations.selected() or [DEFAULT_CONFIGURATION]
        return preprocessing_command(self.state.record.dataset_id, configurations,
                                     self._planner.currentText().strip() or DEFAULT_PLANNER,
                                     self._arguments.text(), processes=self._processes.value(),
                                     verify=self._verify.isChecked())

    def _update_command(self, *_args) -> None:
        error = arguments_error(self._arguments.text())  # e.g. an unclosed quote while typing
        if error:
            self._command.set_error(error)
        else:
            self._command.set_command(self._command_text())
        self.update_actions()

    def update_actions(self) -> None:
        if self.state is None or not getattr(self, "_run_dialog", None) or not self._run_dialog.visible:
            return  # nothing to show while the run dialog is closed
        reason = arguments_error(self._arguments.text()) or self.busy_or_missing("raw", "preprocessed")
        if not reason:
            reason = self.unknown_class("planner", self._planner.currentText())
        missing = self.state.raw.missing_labels
        if not reason and missing:
            reason = (f"{missing} image{'s' if missing != 1 else ''} without label · cannot preprocess: annotate "
                      f"{'them' if missing != 1 else 'it'}, or move the images you will not annotate out of imagesTr "
                      "(see Data)")
        if not reason and not self.state.raw.cases:
            reason = "no raw data · cannot preprocess"
        problems = self.state.data_problems() if not reason else []
        if problems:  # nnU-Net would fail on them later, often with an unrelated message
            reason = "fix the data first (see Data) · " + "; ".join(problem.rstrip(".") for problem in problems)
        self._run.setEnabled(not reason)
        self._run.setToolTip(reason)
        self.show_run_reason(self._run_hint, reason)

    def _run_preprocessing(self) -> None:
        if arguments_error(self._arguments.text()):
            self.update_actions()
            return
        if self.state is not None and self.context.jobs.start_nnunet("Preprocess", self.state.record.name,
                                                                      self._command_text()):
            self._run_dialog.close()

    def header_badges(self, state: DatasetState) -> list:
        """What is preprocessed, then whether preprocessing can run (it needs complete raw data)."""
        preprocessed = [plans for plans in state.plans if plans.preprocessed_time]
        if not preprocessed:
            items = [badge("no preprocessed data", "gray")]
        elif any(plans.outdated for plans in preprocessed):
            items = [badge("preprocessing outdated · labelsTr changed afterwards", "red")]
        else:
            items = [badge("✓ preprocessed", "green")]
        missing = state.raw.missing_labels
        if not state.raw.cases:
            items.append(badge("no raw data · cannot preprocess", "gray",
                               "Preprocessing needs the images and labels in nnUNet_raw."))
        elif missing:
            items.append(badge(f"{missing} image{'s' if missing != 1 else ''} without label · cannot preprocess", "red",
                               "nnU-Net needs a label for every image: annotate the images, or move the images you "
                               "will not annotate out of imagesTr (see Data)."))
        return items
