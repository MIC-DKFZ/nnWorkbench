"""Preprocess stage: plans files, their configurations, the preprocessed folder and the run panel."""

from __future__ import annotations

import json
import os
from pathlib import Path

from qtpy.QtCore import QSize, Qt, QTimer
from qtpy.QtGui import QColor, QFontMetrics
from qtpy.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nnworkbench.services.commands import preprocessing_command
from nnworkbench.services.dataset_contents import json_value
from nnworkbench.services.nnunet_discovery import available_planners
from nnworkbench.services.workflow_status import PlansInfo

from ..context import AppContext
from ..dataset_state import DatasetState
from ..editable_dropdown import EditableDropdownComboBox
from ..widgets import (
    Card,
    ChoiceGroup,
    HowToCard,
    badge,
    button,
    empty_state,
    form_row,
    icon_button,
    label,
    scroll,
    make_table,
    section_label,
    table_item,
    tip,
)
from .common import CommandBox, StagePage, arguments_field

DEFAULT_PLANNER = "nnUNetPlannerResEncL"
CONFIGURATIONS = ("3d_fullres", "2d", "3d_lowres")


def _howto() -> HowToCard:
    return HowToCard("What preprocessing does", [
        "Analyses your images (<b>fingerprint</b>): sizes, spacing, intensities.",
        "Plans the network and patch size for your GPU (<b>plans</b>).",
        "Resamples and normalises all cases for training.",
        "Needed once per dataset, and again after labels change.",
    ], [("blue", "💡 <span style='font-family:monospace'>nnUNetPlannerResEncL</span> needs a GPU with about 24 GB. "
                 "Use <span style='font-family:monospace'>nnUNetPlannerResEncM</span> for 8–12 GB. You can also type "
                 "a custom planner name.")], "Train · Preprocess")


def _chip(text: str, state: str) -> QLabel:
    chip = QLabel(text)
    chip.setObjectName("configChip")
    chip.setProperty("state", state)
    return chip


class PreprocessStage(StagePage):
    key, title, guide_stage = "preprocess", "Preprocess", 8
    folder = "preprocessed"

    def __init__(self, context: AppContext, parent=None) -> None:
        super().__init__(context, _howto(), parent)
        self._selected_plans: tuple[str, str] | None = None  # (plans, configuration)
        self._rows: list = []
        self._planners_loaded = False
        # Plans
        plans_card = Card()
        head = plans_card.add_header("Plans")
        self._plans_path = label("", muted=True, mono=True, size=12)
        head.addWidget(self._plans_path)
        head.addStretch()
        self._plans_count = label("", muted=True, size=12)
        head.addWidget(self._plans_count)
        # One row per preprocessed configuration; planned-only configurations are not listed.
        self._plans_table = make_table(["Plans", "Planner", "Configuration", "Preprocessed"], stretch=None)
        self._plans_table.itemSelectionChanged.connect(self._plans_selected)
        plans_card.body.addWidget(self._plans_table)
        self._plans_empty = empty_state("This dataset has not been preprocessed yet. Choose a planner and click "
                                        "▶ Run preprocessing.")
        plans_card.body.addWidget(self._plans_empty)
        self.center.addWidget(plans_card)
        # Folder
        folder_card = Card()
        folder_head = folder_card.add_header()
        self._folder_path = label("", mono=True)
        folder_head.addWidget(self._folder_path)
        folder_head.addStretch()
        folder_head.addWidget(label("listed from the folder", muted=True, size=12))
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setIndentation(18)
        self._tree.itemExpanded.connect(self._expand_folder)
        # Rows are widgets with their own spacing; the default item padding would cut off their buttons.
        self._tree.setStyleSheet("QTreeWidget::item { padding: 0px; border-bottom: 1px solid #f0f3f7; }")
        self._case_folder = None
        self._tree_version = 0
        folder_card.body.addWidget(self._tree, 1)
        self.center.addWidget(folder_card, 1)
        # Right: run panel or JSON viewer
        self.right.addWidget(scroll(self._build_run_panel()))
        self._json_panel = Card()
        json_layout = QVBoxLayout()
        json_layout.setContentsMargins(18, 16, 18, 16)
        json_layout.setSpacing(10)
        self._json_panel.body.addLayout(json_layout)
        json_head = QHBoxLayout()
        json_head.addWidget(label("{ }", mono=True, bold=True))
        names = QVBoxLayout()
        names.setSpacing(0)
        self._json_title = label("", mono=True, bold=True)
        names.addWidget(self._json_title)
        names.addWidget(label("raw file · read only", muted=True, size=11))
        json_head.addLayout(names, 1)
        json_head.addWidget(button("✕", ghost=True, small=True, clicked=self._close_json))
        json_layout.addLayout(json_head)
        self._json_view = QPlainTextEdit()
        self._json_view.setObjectName("codeView")
        self._json_view.setReadOnly(True)
        json_layout.addWidget(self._json_view, 1)
        self.right.addWidget(self._json_panel)

    def _build_run_panel(self) -> QWidget:
        panel = Card()
        layout = QVBoxLayout()
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)
        panel.body.addLayout(layout)
        layout.addWidget(label("Run preprocessing", name="h3"))
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        self._planner = EditableDropdownComboBox()
        self._planner.setEditable(True)
        self._planner.setProperty("mono", True)
        self._planner.addItems(["ExperimentPlanner", "nnUNetPlannerResEncM", DEFAULT_PLANNER, "nnUNetPlannerResEncXL"])
        self._planner.setCurrentText(DEFAULT_PLANNER)
        self._planner.setToolTip("Type a custom planner name, or choose an installed planner.")
        self._planner.currentTextChanged.connect(self._update_command)
        self._configurations = ChoiceGroup(exclusive=False)
        self._configurations.set_options(CONFIGURATIONS, selected=["3d_fullres"])
        self._configurations.changed.connect(self._update_command)
        self._processes = QSpinBox()
        self._processes.setRange(1, max(1, os.cpu_count() or 1))
        self._processes.setValue(min(8, os.cpu_count() or 1))
        self._processes.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self._processes.valueChanged.connect(self._update_command)
        self._verify = QCheckBox("Verify dataset integrity")
        self._verify.setChecked(True)
        self._verify.toggled.connect(self._update_command)
        self._arguments = arguments_field()
        self._arguments.setPlaceholderText("Optional, e.g. -npfp 8")
        self._arguments.textChanged.connect(self._update_command)
        for name, widget in (("Planner", self._planner), ("Configurations", self._configurations),
                             ("Processes", self._processes), ("Checks", self._verify),
                             ("Additional arguments", self._arguments)):
            layout.addLayout(form_row(name, widget))
        layout.addWidget(section_label("Command"))
        self._command = CommandBox()
        layout.addWidget(self._command)
        actions = QHBoxLayout()
        actions.addWidget(button("Copy command", small=True, clicked=lambda: self._command.copy()))
        self._run = button("▶ Run preprocessing", small=True, primary=True, clicked=self._run_preprocessing)
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
        if not self._planners_loaded:
            self._planners_loaded = True
            self.context.tasks.run("planners", available_planners, self._set_planners)
        record = state.record
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
            size = " × ".join(str(value) for value in config.patch_size) or "—"
            spacing = " × ".join(f"{value:.3g}" for value in config.spacing) or "—"
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
        header = self._plans_table.horizontalHeader().sizeHint().height()
        self._plans_table.setFixedHeight(header + 4 + sum(self._plans_table.rowHeight(index)
                                                          for index in range(min(2 * len(rows), 10))))
        if has_plans:
            self._plans_table.selectRow(selected_row)
            self._plans_selected()
        else:
            self._fill_tree(None)
        self._update_command()

    def _fix_plan_columns(self) -> None:
        """Size columns from the entry rows only; the spanning details lines must not widen the first column
        (with resize-to-contents it grew on every refresh)."""
        table = self._plans_table
        header = table.horizontalHeader()
        for column in range(table.columnCount()):
            texts = [table.horizontalHeaderItem(column).text()]
            texts += [table.item(row, column).text() for row in range(0, table.rowCount(), 2)
                      if table.item(row, column) is not None]
            width = max(QFontMetrics(table.font()).horizontalAdvance(text) for text in texts) + 40
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
            table.setColumnWidth(column, width)
        header.setSectionResizeMode(table.columnCount() - 1, QHeaderView.ResizeMode.Stretch)

    def _set_planners(self, planners: object) -> None:
        if isinstance(planners, tuple) and planners:
            current = self._planner.currentText()
            self._planner.blockSignals(True)
            self._planner.clear()
            self._planner.addItems(list(planners))
            self._planner.setCurrentText(current or DEFAULT_PLANNER)
            self._planner.blockSignals(False)

    def _current_plans(self) -> PlansInfo | None:
        if self.state is None:
            return None
        return next((plans for plans in self.state.plans if plans.name == (self._selected_plans or ("",))[0]), None)

    def _plans_selected(self) -> None:
        selected = self._plans_table.selectionModel().selectedRows()
        if not selected or self.state is None:
            return
        plans, config = self._rows[self._plans_table.item(selected[0].row(), 0).data(Qt.ItemDataRole.UserRole)]
        self._selected_plans = (plans.name, config.name)
        self._fill_tree(plans)

    ROW_HEIGHT = 34
    BATCH = 200

    def _fill_tree(self, plans: PlansInfo | None) -> None:
        self._tree.clear()
        self._tree_version = getattr(self, "_tree_version", 0) + 1
        state = self.state
        root = state.record.preprocessed_path if state else None
        if root is None or not root.is_dir():
            self._tree.addTopLevelItem(QTreeWidgetItem(["No preprocessed folder for this dataset yet."]))
            return
        notes = {"dataset.json": "copy of raw", "dataset_fingerprint.json": "intensity & spacing statistics",
                 "splits_final.json": "folds for training"}
        for path in sorted(root.glob("*.json"), key=lambda item: item.name.casefold()):
            note = notes.get(path.name, "network & patch settings" if "plans" in path.name.casefold() else "")
            self._add_row(None, "📄", path.name, note, "{ }", "Show JSON", lambda selected=path: self._show_json(selected))
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
            layout.addWidget(label(icon, size=12))
        layout.addWidget(label(name, mono=True), 1)
        if note:
            layout.addWidget(label(note, muted=True, size=12))
        if button_icon:
            layout.addWidget(icon_button(button_icon, tooltip, clicked))
        else:
            spacer = QWidget()
            spacer.setFixedWidth(30)
            layout.addWidget(spacer)
        self._tree.setItemWidget(item, 0, row)
        return item

    def _add_folder(self, folder: Path, name: str, note: str) -> None:
        item = self._add_row(None, "📁", name, note)
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
        files = sorted((path for path in Path(folder).iterdir() if path.is_file()), key=lambda path: path.name)
        self._append_files(item, files, 0, self._tree_version)

    def _append_files(self, item: QTreeWidgetItem, files: list, start: int, version: int) -> None:
        if version != self._tree_version:
            return
        for path in files[start:start + self.BATCH]:
            is_case = path.name.endswith(".b2nd") or path.parent.name == "gt_segmentations"
            self._add_row(item, "", path.name, "", "🔍" if is_case else "",
                          "Inspect in Napari · preprocessed image and segmentation",
                          lambda selected=path: self._inspect_pair(selected))
        if start + self.BATCH < len(files):
            QTimer.singleShot(0, lambda: self._append_files(item, files, start + self.BATCH, version))

    # ------------------------------------------------------------ JSON panel

    def _show_json(self, path: Path) -> None:
        self._json_title.setText(path.name)
        self._json_view.setPlainText("Loading…")
        self.right.setCurrentWidget(self._json_panel)
        self.context.tasks.run("preprocess-json", lambda: json_value(path),
                               lambda data: self._json_view.setPlainText(
                                   json.dumps(data, indent=2, ensure_ascii=False) if data is not None
                                   else f"{path.name} is missing or not valid JSON."))

    def _close_json(self) -> None:
        self.right.setCurrentIndex(0)

    # ------------------------------------------------------------ run

    def _command_text(self) -> str:
        if self.state is None:
            return ""
        configurations = self._configurations.selected() or ["3d_fullres"]
        return preprocessing_command(self.state.record.dataset_id, configurations,
                                     self._planner.currentText().strip() or DEFAULT_PLANNER,
                                     self._arguments.text(), processes=self._processes.value(),
                                     verify=self._verify.isChecked())

    def _update_command(self, *_args) -> None:
        try:
            self._command.set_command(self._command_text())
        except ValueError:  # unbalanced quotes while typing additional arguments
            pass
        self.update_actions()

    def update_actions(self) -> None:
        if self.state is None:
            return
        reason = self.busy_or_missing("raw", "preprocessed")
        missing = self.state.raw.missing_labels
        if not reason and missing:
            reason = (f"{missing} image{'s have' if missing != 1 else ' has'} no label (see Data). nnU-Net needs a label "
                      "for every image, so preprocessing is disabled until they exist.")
        if not reason and not self.state.raw.cases:
            reason = "imagesTr has no images yet."
        self._run.setEnabled(not reason)
        self._run.setToolTip(reason)
        self._run_hint.setText(("⏳ " if self.context.jobs.busy else "") + reason)
        self._run_hint.setVisible(bool(reason))

    def _run_preprocessing(self) -> None:
        if self.state is not None:
            self.context.jobs.start_nnunet("Preprocess", self.state.record.name, self._command_text())

    def header_badges(self, state: DatasetState) -> list:
        """Preprocessing status first (red when something blocks training), label mode last."""
        items = []
        if state.raw.missing_labels:
            items.append(badge(f"labels incomplete · {state.raw.missing_labels} image"
                               f"{'s' if state.raw.missing_labels != 1 else ''} without label", "red"))
        preprocessed = [plans for plans in state.plans if plans.preprocessed_time]
        if not preprocessed:
            items.append(badge("not preprocessed yet", "red"))
        elif any(plans.outdated for plans in preprocessed):
            items.append(badge("preprocessing outdated · labelsTr changed afterwards", "red"))
        elif not items:
            items.append(badge("✓ preprocessed", "green"))
        return items + super().header_badges(state)
