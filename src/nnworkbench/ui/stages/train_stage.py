"""Train stage: all trained models with their folds, the selected fold's summary, and the train panel."""

from __future__ import annotations


from qtpy.QtCore import Qt
from qtpy.QtWidgets import QCheckBox, QHeaderView, QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from nnworkbench.services.commands import training_command
from nnworkbench.services.nnunet_discovery import available_trainers
from nnworkbench.services.workflow_status import FoldInfo, ModelInfo

from ..context import AppContext
from ..dataset_state import DatasetState
from ..editable_dropdown import EditableDropdownComboBox
from ..widgets import (
    Card,
    ChoiceGroup,
    HowToCard,
    IconButtonsDelegate,
    badge,
    button,
    cell_widget,
    fit_columns,
    empty_state,
    form_hint,
    form_row,
    label,
    scroll,
    make_table,
    section_label,
    table_item,
    tip,
)
from .common import FOLD_SIZE, CommandBox, StagePage, arguments_field, mean_dice_item, models_table, size_models_table

SPARSE_TRAINER = "nnUNetTrainerBetterIgnoreSampling"
DENSE_TRAINER = "nnUNetTrainer"
_STATUS = {"done": ("✓ done", "green"), "training": ("● training", "orange"), "stopped": ("■ stopped", "orange"),
           "reduced": ("final checkpoint only", "gray")}


def _howto() -> HowToCard:
    return HowToCard("How training works", [
        "A <b>model</b> is one trainer + plans + configuration. You can train several and compare them.",
        "Each model is trained in <b>folds</b>. <span style='font-family:monospace'>splits_final.json</span> defines how "
        "many (5 by default, fewer with custom splits); <b>all</b> trains on every case.",
        "One fold can take hours to days. The GUI runs one job at a time; use Copy command to run more folds in "
        "parallel, e.g. on the cluster.",
        "<b>BetterIgnoreSampling</b> is the trainer for sparse labels (with an ignore label).",
    ], [("blue", "💡 <b>Dice</b> comes from validation/summary.json once a fold is finished; <b>pseudo Dice</b> is "
                 "read from the training log. Fold <b>all</b> is validated on its own training cases, so its Dice is "
                 "not a validation result."),
        ("blue", "A fold counts as <b>training</b> while its log was updated in the last 15 minutes; otherwise it is "
                 "shown as <b>stopped</b> and can be continued.")], "Train")


def _fold_chip(fold: FoldInfo) -> QLabel:
    chip = QLabel(fold.name)
    chip.setFixedSize(*FOLD_SIZE)
    chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
    chip.setObjectName("foldChip")
    chip.setProperty("state", fold.status)
    chip.setToolTip(_STATUS[fold.status][0])
    return chip


def _short(value: float) -> str:
    return f"{value / 1e6:.2f} M" if value >= 1e6 else f"{value / 1e3:.1f} k" if value >= 1e3 else f"{value:.0f}"


class TrainStage(StagePage):
    key, title, guide_stage = "train", "Train", 8
    folder = "results"

    def __init__(self, context: AppContext, parent=None) -> None:
        super().__init__(context, _howto(), parent)
        self._model: ModelInfo | None = None
        self._fold: FoldInfo | None = None
        self._trainers_loaded = False
        models_card = Card()
        head = models_card.add_header("Models")
        self._models_path = label("", muted=True, mono=True, size=12)
        head.addWidget(self._models_path)
        head.addStretch()
        self._models_count = label("", muted=True, size=12)
        head.addWidget(self._models_count)
        self._models = models_table("Folds")
        self._models.itemSelectionChanged.connect(self._model_selected)
        models_card.body.addWidget(self._models)
        legend = QHBoxLayout()
        legend.setContentsMargins(16, 6, 16, 8)
        legend.setSpacing(12)
        legend.addWidget(section_label("Fold colors"))
        for state, text in (("done", "trained"), ("training", "training"), ("stopped", "stopped"),
                            ("reduced", "reduced checkpoint (final checkpoint only)")):
            chip = QLabel("")
            chip.setFixedSize(16, 16)
            chip.setObjectName("foldChip")
            chip.setProperty("state", state)
            legend.addWidget(chip)
            legend.addWidget(label(text, muted=True, size=12))
        legend.addStretch()
        models_card.body.addLayout(legend)
        self._models_empty = empty_state("No trained models yet. Choose trainer, plans, configuration and fold, then "
                                         "click ▶ Start training.")
        models_card.body.addWidget(self._models_empty)
        self.center.addWidget(models_card)
        self._folds_card = Card()
        folds_head = self._folds_card.add_header("Folds")
        self._folds_model = label("", muted=True, mono=True, size=12)
        folds_head.addWidget(self._folds_model)
        folds_head.addStretch()
        self._folds = make_table(["Fold", "Status", "Epoch", "Dice", "Pseudo Dice", ""], stretch=4)
        self._folds.setItemDelegateForColumn(5, IconButtonsDelegate(
            self._folds, lambda _row: [("📁", "Open the folder of this fold", False)],
            lambda row, _number: self._model is not None and self.context.open_folder(self._model.folds[row].path)))
        self._folds.itemSelectionChanged.connect(self._fold_selected)
        self._folds_card.body.addWidget(self._folds)
        self._folds_card.body.addStretch()
        self.center.addWidget(self._folds_card, 1)
        self.right.addWidget(scroll(self._build_panel()))

    def _build_panel(self) -> QWidget:
        panel = Card()
        layout = QVBoxLayout()
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)
        panel.body.addLayout(layout)
        layout.addWidget(label("Train folds", name="h3"))
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        self._trainer = EditableDropdownComboBox()
        self._trainer.setEditable(True)
        self._trainer.setProperty("mono", True)
        self._trainer.addItems([DENSE_TRAINER, SPARSE_TRAINER])
        self._trainer.currentTextChanged.connect(self._update_command)
        self._trainer_hint = label("", muted=True, wrap=True, size=11)
        self._plans = EditableDropdownComboBox()
        self._plans.setEditable(True)
        self._plans.setProperty("mono", True)
        self._plans.currentTextChanged.connect(self._plans_changed)
        self._configuration = EditableDropdownComboBox()
        self._configuration.setEditable(True)
        self._configuration.setProperty("mono", True)
        self._configuration.currentTextChanged.connect(self._update_command)
        for combo in (self._trainer, self._plans, self._configuration):
            combo.setSizeAdjustPolicy(combo.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
        self._fold_choice = ChoiceGroup(exclusive=True)
        self._fold_choice.changed.connect(self._fold_choice_changed)
        self._fold_hint = label("", muted=True, size=11)
        self._continue = QCheckBox("Continue an existing training  --c")
        self._continue.toggled.connect(self._update_command)
        self._arguments = arguments_field()
        self._arguments.textChanged.connect(self._update_command)
        self._trainer_hint = form_hint("")
        self._fold_hint = form_hint("")
        layout.addLayout(form_row("Trainer", self._trainer))
        layout.addWidget(self._trainer_hint)
        layout.addLayout(form_row("Plans", self._plans))
        layout.addLayout(form_row("Configuration", self._configuration))
        layout.addLayout(form_row("Folds", self._fold_choice))
        layout.addWidget(self._fold_hint)
        layout.addLayout(form_row("Continue", self._continue))
        layout.addWidget(form_hint("Resumes the fold from its latest checkpoint, e.g. after a crash or a time limit on "
                                   "the cluster. Checked automatically when the selected fold exists but is not "
                                   "finished."))
        layout.addLayout(form_row("Additional arguments", self._arguments))
        layout.addWidget(section_label("Command"))
        self._command = CommandBox()
        self._command.setFixedHeight(80)
        layout.addWidget(self._command)
        actions = QHBoxLayout()
        actions.addWidget(button("Copy command", small=True, clicked=lambda: self._command.copy()))
        self._start = button("▶ Start training", small=True, primary=True, clicked=self._start_training)
        actions.addWidget(self._start)
        actions.addStretch()
        layout.addLayout(actions)
        self._busy_hint = tip("", "blue")
        layout.addWidget(self._busy_hint)
        layout.addStretch()
        layout.addWidget(label("Selecting a model or fold in the tables fills in trainer, plans, configuration and fold. "
                               "Shared models with reduced checkpoints (reduce-checkpoints) are listed, but cannot be "
                               "trained further.", muted=True, wrap=True, size=12))
        return panel

    # ------------------------------------------------------------ state

    def set_state(self, state: DatasetState) -> None:
        super().set_state(state)
        if not self._trainers_loaded:
            self._trainers_loaded = True
            self.context.tasks.run("trainers", available_trainers, self._set_trainers)
        record = state.record
        self._models_path.setText(f"nnUNet_results/{record.directory_name}")
        self._models_count.setText(f"{len(state.models)} model folder{'s' if len(state.models) != 1 else ''}")
        has_models = bool(state.models)
        self._models.setVisible(has_models)
        self._models_empty.setVisible(not has_models)
        self._folds_card.setVisible(has_models)
        previous = self._model.label if self._model else None
        self._models.blockSignals(True)
        self._models.setRowCount(len(state.models))
        select = 0
        for row, model in enumerate(state.models):
            # Same layout as the Validate models table: trainer with plans · configuration below
            self._models.setItem(row, 0, table_item("", data=row))
            name = label(f"{model.trainer}<br><span style='color:#64748b'>{model.plans} · {model.configuration}</span>",
                         mono=True, rich=True)
            self._models.setCellWidget(row, 0, cell_widget(name))
            self._models.setCellWidget(row, 2, cell_widget(*(_fold_chip(fold) for fold in model.folds)))
            self._models.setItem(row, 1, mean_dice_item(model))
            if model.label == previous:
                select = row
        self._models.blockSignals(False)
        size_models_table(self._models, state.models)
        # Panel defaults
        default_trainer = SPARSE_TRAINER if state.sparse else DENSE_TRAINER
        if self._model is None:
            self._trainer.setCurrentText(default_trainer)
        self._trainer_hint.setText(
            "default, because dataset.json has an ignore label (sparse labels)" if state.sparse else
            "default for dense labels")
        plans_names = [plans.name for plans in state.plans] or ["nnUNetResEncUNetLPlans"]
        current_plans = self._plans.currentText()
        self._plans.blockSignals(True)
        self._plans.clear()
        self._plans.addItems(plans_names)
        self._plans.setCurrentText(current_plans if current_plans in plans_names else plans_names[0])
        self._plans.blockSignals(False)
        self._plans_changed()
        folds = [str(index) for index in range(state.fold_count)] + ["all"]
        selected = self._fold_choice.selected() or [self._first_free_fold(folds)]
        self._fold_choice.set_options(folds, selected=selected)
        self._fold_hint.setText(f"One fold per command · {state.fold_count} folds in splits_final.json")
        if has_models:
            self._models.selectRow(select)
            self._model_selected()
        self._update_command()

    def _first_free_fold(self, folds: list[str]) -> str:
        if self.state is None:
            return "0"
        model = next((item for item in self.state.models if item.trainer == self._trainer.currentText()
                      and item.plans == self._plans.currentText()), None)
        existing = {fold.name for fold in model.folds} if model else set()
        return next((fold for fold in folds if fold not in existing), folds[0])

    def _set_trainers(self, trainers: object) -> None:
        if isinstance(trainers, tuple) and trainers:
            current = self._trainer.currentText()
            self._trainer.blockSignals(True)
            self._trainer.clear()
            self._trainer.addItems(list(trainers))
            self._trainer.setCurrentText(current)
            self._trainer.blockSignals(False)

    def _plans_changed(self, *_args) -> None:
        if self.state is None:
            return
        plans = next((item for item in self.state.plans if item.name == self._plans.currentText()), None)
        configurations = [config.name for config in plans.configurations] if plans else ["3d_fullres", "2d"]
        current = self._configuration.currentText()
        self._configuration.blockSignals(True)
        self._configuration.clear()
        self._configuration.addItems(configurations)
        self._configuration.setCurrentText(current if current in configurations else configurations[0])
        self._configuration.blockSignals(False)
        self._update_command()

    # ------------------------------------------------------------ tables

    def _model_selected(self) -> None:
        rows = self._models.selectionModel().selectedRows()
        if not rows or self.state is None:
            return
        model = self.state.models[self._models.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)]
        self._model = model
        self._folds_model.setText(f"{model.trainer} · {model.plans} · {model.configuration}")
        self._trainer.setCurrentText(model.trainer)
        self._plans.setCurrentText(model.plans)
        self._configuration.setCurrentText(model.configuration)
        self._folds.blockSignals(True)
        self._folds.setRowCount(len(model.folds))
        select = 0
        for row, fold in enumerate(model.folds):
            self._folds.setItem(row, 0, table_item(fold.name, bold=True, data=row))
            text, tone = _STATUS[fold.status]
            self._folds.setCellWidget(row, 1, cell_widget(badge(text, tone)))
            epoch = "—" if fold.epoch is None else (
                f"{fold.epoch + (1 if fold.finished else 0)} / {fold.total_epochs}" if fold.total_epochs
                else str(fold.epoch))
            self._folds.setItem(row, 2, table_item(epoch))
            if fold.dice is None:
                dice = table_item("—", muted=True)
            elif fold.name == "all":
                # fold "all" trains on every case, so its validation cases were also training cases
                dice = table_item(f"{fold.dice:.2f}  (training cases)", muted=True)
                dice.setToolTip("Fold all trains on every case. Its validation Dice is measured on cases it was "
                                "trained on, so it is not a validation result and is not included in the mean Dice.")
            else:
                dice = table_item(f"{fold.dice:.2f}", bold=True)
            self._folds.setItem(row, 3, dice)
            pseudo = table_item(f"{fold.pseudo_dice:.2f}" if fold.pseudo_dice is not None else "—", muted=True)
            pseudo.setToolTip("Pseudo Dice from the training log (last logged epoch)")
            self._folds.setItem(row, 4, pseudo)
            if fold.finished and fold.has_validation and (self._fold is None or self._fold.path == fold.path):
                select = row
        self._folds.blockSignals(False)
        self._folds.setFixedHeight(42 + 38 * min(len(model.folds), 6))
        fit_columns(self._folds)
        self._folds.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        self._folds.setColumnWidth(5, 56)
        if model.folds:
            self._folds.selectRow(select)
            self._fold_selected()

    def _fold_selected(self) -> None:
        rows = self._folds.selectionModel().selectedRows()
        if not rows or self._model is None:
            return
        fold = self._model.folds[self._folds.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)]
        self._fold = fold
        if fold.name in self._fold_choice.options():
            self._fold_choice.select(fold.name)

    # ------------------------------------------------------------ panel

    def _fold_choice_changed(self) -> None:
        if self.state is None:
            return
        selected = self._fold_choice.selected()
        model = next((item for item in self.state.models if item.trainer == self._trainer.currentText()
                      and item.plans == self._plans.currentText()
                      and item.configuration == self._configuration.currentText()), None)
        fold = next((item for item in (model.folds if model else ()) if selected and item.name == selected[0]), None)
        self._continue.setChecked(bool(fold and not fold.finished))
        self._update_command()

    def _command_text(self) -> str:
        if self.state is None:
            return ""
        fold = (self._fold_choice.selected() or ["0"])[0]
        return training_command(self.state.record.dataset_id, self._plans.currentText() or "nnUNetPlans",
                                self._configuration.currentText() or "3d_fullres",
                                self._trainer.currentText() or DENSE_TRAINER, fold, self._arguments.text(),
                                continue_training=self._continue.isChecked())

    def _update_command(self, *_args) -> None:
        try:
            self._command.set_command(self._command_text())
        except ValueError:
            pass
        self.update_actions()

    def update_actions(self) -> None:
        if self.state is None:
            return
        reason = self.busy_or_missing("preprocessed", "results")
        if not reason and not any(plans.preprocessed_time for plans in self.state.plans):
            reason = "Preprocess the dataset first."
        self._start.setEnabled(not reason)
        self._start.setToolTip(reason)
        self._busy_hint.setText(("⏳ " if self.context.jobs.busy else "") + reason)
        self._busy_hint.setVisible(bool(reason))

    def _start_training(self) -> None:
        if self.state is not None:
            fold = (self._fold_choice.selected() or ["0"])[0]
            self.context.jobs.start_nnunet(f"Train · fold {fold}", self.state.record.name, self._command_text())

    def header_badges(self, state: DatasetState) -> list:
        """Training status first, label mode last."""
        items = []
        if not any(plans.preprocessed_time for plans in state.plans):
            items.append(badge("not preprocessed yet", "red"))
        running = sum(1 for model in state.models for fold in model.folds if fold.status == "training")
        if running:
            items.append(badge(f"● {running} fold{'s' if running != 1 else ''} training", "orange"))
        finished = sum(len(model.finished_folds) for model in state.models)
        if finished:
            items.append(badge(f"✓ {finished} fold{'s' if finished != 1 else ''} finished", "green"))
        elif not running and not items:
            items.append(badge("no trained fold yet", "gray"))
        return items + super().header_badges(state)
