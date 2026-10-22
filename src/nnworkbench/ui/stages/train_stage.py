"""Train stage: all trained models with their folds, the selected fold's summary, and the train panel."""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractScrollArea,
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from nnworkbench.services.commands import split_arguments, training_command
from nnworkbench.services.defaults import (
    DENSE_TRAINER,
    SPARSE_TRAINER,
    default_configuration,
    default_plans,
    next_fold,
    trainer_for,
)
from nnworkbench.services.device_probe import (
    DEVICE_CHOICES,
    default_choice,
    device_argument,
    device_hint,
)
from nnworkbench.services.nnunet_discovery import available_trainers
from nnworkbench.services.workflow_status import FoldInfo, ModelInfo

from ..context import AppContext
from ..dataset_state import DatasetState
from ..editable_dropdown import EditableDropdownComboBox
from ..widgets import (
    Card,
    ChoiceGroup,
    FlowLayout,
    HowToCard,
    IconButtonsDelegate,
    ToggleGroup,
    badge,
    button,
    cell_widget,
    clear_layout,
    empty_state,
    fit_columns,
    form_hint,
    form_row,
    icon_html,
    label,
    make_table,
    scroll,
    section_label,
    separator,
    table_item,
    tip,
)
from .common import (
    DEVICE_INFO,
    FOLD_ALL_NOTE,
    FOLD_STATUS,
    CommandBox,
    RunDialog,
    StagePage,
    arguments_error,
    arguments_field,
    detail_row,
    fold_cells,
    fold_chip,
    header_note,
    mean_dice_item,
    model_name_cell,
    models_table,
    size_models_table,
    wrappable,
)


def _howto() -> HowToCard:
    return HowToCard("How training works", [
        "Pick a <b>trainer</b>, <b>plans</b> and <b>configuration</b> to define a model.",
        ("Train a <b>cross-validation</b>: each <b>fold</b> splits the cases into a training and a validation set "
        "(5 folds by default, <span style='font-family:monospace'>splits_final.json</span>). Fold <b>all</b> uses "
        "every case for training, without a dedicated validation set."),
    ], [("blue", (f"{icon_html('bulb')} <b>Dice</b>: model performance on the validation set after training.<br>"
                 "<b>Pseudo Dice</b>: quick approximation during training on random patches of the validation set.<br>"
                 "Fold <b>all</b>: its Dice is measured on training cases.")),
        ("blue", ("<b>Training status</b>: <i>training</i> while the log was updated in the last 15 minutes, "
                 "otherwise <i>stopped</i> (can be continued).")),
        ("purple", f"{icon_html('bulb')} Use <b>BetterIgnoreSampling</b> (or a variant) for sparse labels.")], "Train")


class TrainStage(StagePage):
    key, title, guide = "train", "Train", ("4-train.md", "")
    folder = "results"

    def __init__(self, context: AppContext, parent=None) -> None:
        super().__init__(context, _howto(), parent)
        self._model: ModelInfo | None = None
        self._fold: FoldInfo | None = None
        self._trainers_loaded = False
        models_card = Card()
        head = models_card.add_header("Models")
        self._models_path = header_note(mono=True)
        head.addWidget(self._models_path, 1)
        self._models_count = label("", muted=True, size=12)
        head.addWidget(self._models_count)
        self._models = models_table("Folds")
        self._models.itemSelectionChanged.connect(self._model_selected)
        models_card.body.addWidget(self._models)
        # Below the table, never on it: the table shrinks (and scrolls) in a low window. Wraps when narrow.
        models_card.body.addWidget(separator())  # a table cut off in a low window ends here, visibly
        legend_holder = QWidget()
        legend_margins = QVBoxLayout(legend_holder)
        legend_margins.setContentsMargins(16, 8, 16, 8)
        legend = FlowLayout(spacing=12)
        legend_margins.addLayout(legend)
        legend.addWidget(section_label("Training status"))
        for state, text in (("done", "done"), ("training", "training"), ("stopped", "stopped"),
                            ("reduced", "final checkpoint only")):
            chip = QLabel("")
            chip.setFixedSize(16, 16)
            chip.setObjectName("foldChip")
            chip.setProperty("state", state)
            legend.addWidget(cell_widget(chip, label(text, muted=True, size=12), spacing=6))
        models_card.body.addWidget(legend_holder)
        self._models_empty = empty_state("No trained models yet. Choose trainer, plans, configuration and fold, then "
                                         "click ▶ Train….")
        models_card.body.addWidget(self._models_empty)
        self.center.addWidget(models_card)
        self._folds_card = Card()
        folds_head = self._folds_card.add_header("Folds")
        self._folds_model = header_note(mono=True)
        folds_head.addWidget(self._folds_model, 1)
        self._folds = make_table(["Fold", "Status", "Epoch", "Dice", "Pseudo Dice", ""], stretch=3)  # Dice: fold all has a note
        # all folds up to six rows, shrinking to two (then it scrolls) in a low window
        self._folds.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        self._folds.setItemDelegateForColumn(5, IconButtonsDelegate(
            self._folds, lambda _row: [("folder", "Open folder · this fold", False)],
            lambda row, _number: self._model is not None and self.context.open_folder(self._model.folds[row].path)))
        self._folds.itemSelectionChanged.connect(self._fold_selected)
        self._folds_card.body.addWidget(self._folds)
        self._folds_card.body.addStretch()
        self.center.addWidget(self._folds_card, 1)
        # Right: details of the selected fold. New runs start from "▶ Train…" in the header (form in a dialog).
        self.right.addWidget(scroll(self._build_details()))
        self._run_dialog = RunDialog(self, "Train", self._build_panel(), on_open=self.update_actions)
        self.header_actions.addWidget(button("▶ Train…", primary=True, clicked=self._open_defaults,
                                             tooltip="Train one or more folds with the default trainer, plans and "
                                                     "configuration. To train the next fold of an existing model, "
                                                     "select it and use ▶ Next fold…"))

    def _build_details(self) -> QWidget:
        panel = Card()
        layout = QVBoxLayout()
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)
        panel.body.addLayout(layout)
        self._details_title = label("", name="h3", wrap=True)
        layout.addWidget(self._details_title)
        self._details_model = label("", muted=True, mono=True, wrap=True, size=11)
        layout.addWidget(self._details_model)
        self._details_body = QVBoxLayout()
        self._details_body.setSpacing(8)
        layout.addLayout(self._details_body)
        layout.addStretch()
        return panel

    def _show_details(self) -> None:
        clear_layout(self._details_body)
        model, fold = self._model, self._fold
        if model is None or fold is None:
            self._details_title.setText("No trained model yet")
            self._details_model.setText("")
            self._details_body.addWidget(label("Click ▶ Train… in the header to train a fold.", muted=True, wrap=True))
            return
        self._details_title.setText(f"Fold {fold.name}")
        self._details_model.setText(wrappable(f"{model.trainer}\n{model.plans} · {model.configuration}"))
        text, tone = FOLD_STATUS[fold.status]
        epoch = "—" if fold.epoch is None else (
            f"{fold.epoch + (1 if fold.finished else 0)} / {fold.total_epochs}" if fold.total_epochs else str(fold.epoch))
        if fold.dice is None:
            dice = "—"
        elif fold.name == "all":
            dice = f"{fold.dice:.3f} (trained on these cases)"
        else:
            dice = f"{fold.dice:.3f}"
        status = badge(text, tone)
        note = fold.log_note()
        status.setToolTip(note)
        rows = (("Status", status), *((("Last log", label(note, muted=True, wrap=True)),) if note else ()),
                ("Epoch", epoch), ("Dice", dice),
                ("Pseudo Dice", f"{fold.pseudo_dice:.3f}" if fold.pseudo_dice is not None else "—"),
                ("Folder", label(fold.path.name, mono=True)))
        for name, value in rows:
            self._details_body.addLayout(detail_row(name, value))
        if fold.status == "stopped":
            self._details_body.addWidget(tip("Stopped before it finished. ▶ Continue… resumes it (check first that it "
                                             "does not still run elsewhere, e.g. on a cluster).", "orange"))
        if fold.name == "all" and fold.dice is not None:
            self._details_body.addWidget(label(FOLD_ALL_NOTE, muted=True, wrap=True, size=12))
        actions = QHBoxLayout()
        if fold.status == "reduced":
            self._details_body.addWidget(label("Shared model with reduced checkpoints: it cannot be trained further.",
                                               muted=True, wrap=True, size=12))
        else:
            if not fold.finished:
                actions.addWidget(button("▶ Continue…", small=True, primary=True,
                                         clicked=lambda: self._prefill(model, fold, continue_training=True),
                                         tooltip="Resume this fold from its latest checkpoint (--c)"))
            actions.addWidget(button("↻ Run again…", small=True,
                                     clicked=lambda: self._prefill(model, fold, continue_training=False),
                                     tooltip="Train this fold again from the start with the same trainer, plans and "
                                             "configuration (the existing results of the fold are replaced)"))
            free = self._free_fold(model)
            next_fold = button("▶ Next fold…", small=True,
                               clicked=lambda: self._prefill(model, free, continue_training=False),
                               tooltip=f"Train fold {free} with the same trainer, plans and configuration" if free
                               else "All folds of this model exist already")
            next_fold.setEnabled(free is not None)
            actions.addWidget(next_fold)
        actions.addStretch()
        self._details_body.addLayout(actions)

    def _prefill(self, model: ModelInfo, fold: FoldInfo | str, *, continue_training: bool) -> None:
        """Open the training form with this model and fold (Continue…, Run again…, Next fold…)."""
        self._fill_form(model.trainer, model.plans, model.configuration,
                        fold.name if isinstance(fold, FoldInfo) else fold, continue_training)
        if self._plans.currentText():
            self._source_note.setText(f"From {model.trainer} · {model.plans} · {model.configuration}.")
        else:  # the model's plans are not preprocessed on this computer, so they are not filled in
            self._source_note.setText(f"From {model.trainer}. Its plans ({model.plans} · {model.configuration}) "
                                      "are not preprocessed on this computer.")
        self._run_dialog.open()

    def _open_defaults(self) -> None:
        """▶ Train… in the header: always the defaults, independent of the selected model."""
        if self.state is None:
            return
        trainer = trainer_for(self.state.sparse)
        plans = default_plans(plans.name for plans in self.state.plans)  # only plans in nnUNet_preprocessed
        self._plans.setCurrentText(plans)  # refreshes the configurations of these plans
        configuration = default_configuration(self._configuration.itemText(index)
                                              for index in range(self._configuration.count()))
        model = next((item for item in self.state.models if (item.trainer, item.plans, item.configuration)
                      == (trainer, plans, configuration)), None)
        self._fill_form(trainer, plans, configuration, self._free_fold(model) or "0", False)
        self._source_note.setText("Defaults for this dataset.")
        self._run_dialog.open()

    def _fill_form(self, trainer: str, plans: str, configuration: str, fold: str, continue_training: bool) -> None:
        self._trainer.setCurrentText(trainer)
        # Plans and configuration only when they are preprocessed here; otherwise the fields stay empty.
        preprocessed = self.state.plans if self.state else ()
        plans_info = next((item for item in preprocessed if item.name == plans), None)
        self._plans.setCurrentText(plans if plans_info else "")
        names = [config.name for config in plans_info.configurations] if plans_info else []
        self._configuration.setCurrentText(configuration if configuration in names else "")
        if fold in self._fold_choice.options():
            self._fold_choice.set_selected([fold])
        self._arguments.clear()
        self._continue.setChecked(continue_training)
        self._update_command()

    def _free_fold(self, model: ModelInfo | None) -> str | None:
        """First fold of splits_final.json (then all) that this model has not trained yet."""
        if self.state is None:
            return None
        return next_fold(self.state.fold_count, (fold.name for fold in model.folds) if model else ())

    def _build_panel(self) -> QWidget:
        panel = Card()
        layout = QVBoxLayout()
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)
        panel.body.addLayout(layout)
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
        # several folds train one after another; at least one stays selected
        self._fold_choice = ChoiceGroup(exclusive=False, keep_one=True)
        self._fold_choice.changed.connect(self._fold_choice_changed)
        self._fold_hint = label("", muted=True, size=11)
        self._continue = QCheckBox("Resume from the latest checkpoint  --c")
        self._continue.toggled.connect(self._update_command)
        self._device = ToggleGroup(DEVICE_CHOICES)
        self._device.changed.connect(lambda _index: self._update_command())
        self._device_hint = form_hint("")
        self._device.set_index(DEVICE_CHOICES.index(default_choice(self.context.device)))
        self.context.device_changed.connect(self._preselect_device)
        self._arguments = arguments_field()
        self._arguments.textChanged.connect(self._update_command)
        self._trainer_hint = form_hint("")
        self._fold_hint = form_hint("")
        layout.addLayout(form_row("Trainer", self._trainer, info=(
            "A trainer defines how the network is trained, e.g. number of epochs, data augmentation or patch "
            "sampling.")))
        layout.addWidget(self._trainer_hint)
        layout.addLayout(form_row("Plans", self._plans, info=(
            "The preprocessed data and network settings to train with. Only preprocessed plans are listed.")))
        layout.addLayout(form_row("Configuration", self._configuration, info=(
            "The configuration defines the data handling and the model input: <b>3d_fullres</b> 3D at full "
            "resolution (the usual choice), <b>2d</b> single slices (faster), <b>3d_lowres</b> downsampled 3D (only "
            "for very large images).")))
        layout.addLayout(form_row("Folds", self._fold_choice))
        layout.addWidget(self._fold_hint)
        layout.addLayout(form_row("Continue", self._continue))
        layout.addLayout(form_row("Device", self._device, info=DEVICE_INFO))
        layout.addWidget(self._device_hint)
        layout.addLayout(form_row("Additional arguments", self._arguments))
        layout.addWidget(section_label("Command"))
        self._command = CommandBox()
        self._command.setFixedHeight(80)
        layout.addWidget(self._command)
        actions = QHBoxLayout()
        actions.addWidget(self._command.copy_button())
        self._start = button("▶ Train", small=True, primary=True, clicked=self._start_training)
        actions.addWidget(self._start)
        actions.addStretch()
        layout.addLayout(actions)
        self._busy_hint = tip("", "blue")
        layout.addWidget(self._busy_hint)
        layout.addStretch()
        self._source_note = label("", muted=True, wrap=True, size=12)
        layout.addWidget(self._source_note)
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
        previous = self.context.model_label or (self._model.label if self._model else None)
        self._models.blockSignals(True)
        self._models.setRowCount(len(state.models))
        select = 0
        for row, model in enumerate(state.models):
            # Same layout as the Validate models table: trainer with plans · configuration below
            self._models.setItem(row, 0, table_item("", data=row))
            self._models.setCellWidget(row, 0, model_name_cell(model))
            self._models.setCellWidget(row, 2, fold_cells(fold_chip(fold) for fold in model.folds))
            self._models.setItem(row, 1, mean_dice_item(model))
            if model.label == previous:
                select = row
        self._models.blockSignals(False)
        size_models_table(self._models, state.models)
        # Panel defaults
        plans_names = [plans.name for plans in state.plans]
        current_plans = self._plans.currentText()
        self._plans.blockSignals(True)
        self._plans.clear()
        self._plans.addItems(plans_names)
        self._plans.setCurrentText(current_plans if current_plans in plans_names else
                                   (plans_names[0] if plans_names else ""))
        self._plans.blockSignals(False)
        self._plans_changed()
        folds = [str(index) for index in range(state.fold_count)] + ["all"]
        selected = [fold for fold in self._fold_choice.selected() if fold in folds] or [self._first_free_fold(folds)]
        self._fold_choice.set_options(folds, selected=selected)
        self._fold_hint.setText(f"{state.fold_count} folds in splits_final.json · several folds train one after "
                                "another and stop at the first failure")
        if has_models:
            self._loading = True  # selecting while loading does not change the shared model
            self._models.selectRow(select)
            self._model_selected()
            self._loading = False
        else:
            self._model, self._fold = None, None
            self._show_details()
        self._update_command()

    def follow_model(self, label: str) -> None:
        if self.state is None or (self._model is not None and self._model.label == label):
            return
        row = next((index for index, model in enumerate(self.state.models) if model.label == label), None)
        if row is not None:
            self._models.selectRow(row)

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
        configurations = [config.name for config in plans.configurations] if plans else []
        current = self._configuration.currentText()
        self._configuration.blockSignals(True)
        self._configuration.clear()
        self._configuration.addItems(configurations)
        self._configuration.setCurrentText(current if current in configurations else
                                           (configurations[0] if configurations else ""))
        self._configuration.blockSignals(False)
        self._update_command()

    # ------------------------------------------------------------ tables

    def _model_selected(self) -> None:
        rows = self._models.selectionModel().selectedRows()
        if not rows or self.state is None:
            return
        model = self.state.models[self._models.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)]
        self._model = model
        if not getattr(self, "_loading", False):
            self.context.model_label = model.label
        self._folds_model.setText(f"{model.trainer} · {model.plans} · {model.configuration}")
        self._folds.blockSignals(True)
        self._folds.setRowCount(len(model.folds))
        select = 0
        for row, fold in enumerate(model.folds):
            self._folds.setItem(row, 0, table_item(fold.name, bold=True, data=row))
            text, tone = FOLD_STATUS[fold.status]
            status = badge(text, tone)
            status.setToolTip(fold.log_note())  # how sure: read from the folder, the training may run elsewhere
            self._folds.setCellWidget(row, 1, cell_widget(status))
            epoch = "—" if fold.epoch is None else (
                f"{fold.epoch + (1 if fold.finished else 0)} / {fold.total_epochs}" if fold.total_epochs
                else str(fold.epoch))
            self._folds.setItem(row, 2, table_item(epoch))
            if fold.dice is None:
                dice = table_item("—", muted=True)
            elif fold.name == "all":
                # fold "all" trains on every case, so its validation cases were also training cases
                dice = table_item(f"{fold.dice:.2f}  (trained on these cases)", muted=True)
                dice.setToolTip(FOLD_ALL_NOTE)
            else:
                dice = table_item(f"{fold.dice:.2f}", bold=True)
            self._folds.setItem(row, 3, dice)
            pseudo = table_item(f"{fold.pseudo_dice:.2f}" if fold.pseudo_dice is not None else "—", muted=True)
            pseudo.setToolTip("Pseudo Dice from the training log (last logged epoch)")
            self._folds.setItem(row, 4, pseudo)
            if fold.finished and fold.has_validation and (self._fold is None or self._fold.path == fold.path):
                select = row
        self._folds.blockSignals(False)
        header = self._folds.horizontalHeader().sizeHint().height() + 4
        self._folds.setMinimumHeight(header + 38 * min(len(model.folds), 2))
        self._folds.setMaximumHeight(header + 38 * min(len(model.folds), 6))
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
        self._show_details()

    # ------------------------------------------------------------ panel

    def _fold_choice_changed(self) -> None:
        if self.state is None:
            return
        # Continue when a selected fold exists but is unfinished (nnU-Net starts new folds anyway with --c)
        self._continue.setChecked(any(not fold.finished for fold in self._existing_folds()))
        self._update_command()

    def _existing_folds(self) -> list[FoldInfo]:
        """The selected folds that the model of the form (trainer, plans, configuration) already has."""
        if self.state is None:
            return []
        key = (self._trainer.currentText(), self._plans.currentText(), self._configuration.currentText())
        model = next((item for item in self.state.models
                      if (item.trainer, item.plans, item.configuration) == key), None)
        selected = set(self._fold_choice.selected())
        return [fold for fold in (model.folds if model else ()) if fold.name in selected]

    def _command_text(self) -> list[str]:
        """One nnUNetv2_train command per selected fold (nnU-Net trains one fold per call)."""
        if self.state is None:
            return []
        # placeholders while plans or configuration are empty (no preprocessed data; the run stays disabled)
        return [training_command(self.state.record.dataset_id, self._plans.currentText() or "PLANS",
                                 self._configuration.currentText() or "CONFIGURATION",
                                 self._trainer.currentText() or DENSE_TRAINER, fold, self._arguments.text(),
                                 continue_training=self._continue.isChecked(), device=self._device_argument())
                for fold in self._fold_choice.selected()]

    def _device_argument(self) -> str | None:
        return device_argument(DEVICE_CHOICES[self._device.index()])

    def _preselect_device(self) -> None:
        """Preselect the device found on this computer (GPU, or the Apple GPU on a Mac)."""
        self._device.set_index(DEVICE_CHOICES.index(default_choice(self.context.device)))
        self._update_command()

    def _update_command(self, *_args) -> None:
        error = arguments_error(self._arguments.text())  # e.g. an unclosed quote while typing
        if error:
            self._command.set_error(error)
        else:
            self._command.set_command(self._command_text())
        self._update_trainer_hint()
        self._device_hint.setText(device_hint(DEVICE_CHOICES[self._device.index()], self.context.device))
        self.update_actions()

    def _update_trainer_hint(self) -> None:
        sparse = bool(self.state and self.state.sparse)
        default = self._trainer.currentText() == trainer_for(sparse)
        self._trainer_hint.setText(("default, because dataset.json has an ignore label (sparse labels)" if sparse
                                    else "default for dense labels") if default else "")
        self._trainer_hint.setVisible(default)

    def _too_few_cases(self) -> str:
        """nnU-Net splits the cases into 5 folds (unless splits_final.json exists): fewer cases fail in sklearn."""
        plans = next((item for item in self.state.plans if item.name == self._plans.currentText()), None)
        config = next((item for item in plans.configurations if item.name == self._configuration.currentText()),
                      None) if plans else None
        cases = config.preprocessed_cases if config else 0
        preprocessed = self.state.record.preprocessed_path
        if not cases or cases >= 5 or self._fold_choice.selected() == ["all"] or (
                preprocessed is not None and (preprocessed / "splits_final.json").is_file()):
            return ""
        return (f"Only {cases} preprocessed case{'s' if cases != 1 else ''}: the 5-fold cross-validation needs at "
                "least 5. Annotate more images (then preprocess again), or train fold all.")

    def update_actions(self) -> None:
        if self.state is None or not getattr(self, "_run_dialog", None) or not self._run_dialog.visible:
            return  # nothing to show while the run dialog is closed
        reason = arguments_error(self._arguments.text()) or self.busy_or_missing("preprocessed", "results")
        if not reason and not any(plans.preprocessed_time for plans in self.state.plans):
            reason = "no preprocessed data · cannot train"
        folds = len(self._fold_choice.selected())
        if not reason and not folds:
            reason = "Select at least one fold."
        if not reason:
            reason = self._too_few_cases()
        self._start.setText(f"▶ Train {folds} folds" if folds > 1 else "▶ Train")
        self._start.setEnabled(not reason)
        self._start.setToolTip(reason)
        self._busy_hint.setText(reason)
        self._busy_hint.setVisible(bool(reason))
        self._run_dialog.refit()

    def _start_training(self) -> None:
        if arguments_error(self._arguments.text()):
            self.update_actions()
            return
        if self.state is not None:
            folds = self._fold_choice.selected()
            warning = self._overwrite_warning()
            if warning:
                answer = QMessageBox.question(
                    self._run_dialog.dialog, "Train this fold?", warning,
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Cancel)
                if answer != QMessageBox.StandardButton.Yes:
                    return
            title = f"Train · fold {folds[0]}" if len(folds) == 1 else f"Train · folds {', '.join(folds)}"
            if self.context.jobs.start_nnunet(title, self.state.record.name, self._command_text(),
                                              step_labels=[f"fold {fold}" for fold in folds]):
                self._run_dialog.close()

    def _overwrite_warning(self) -> str:
        """Text for the confirmation when the run would start an existing fold from scratch, or a second training
        of a fold that is still training, else ""."""
        if self.state is None:
            return ""
        existing = self._existing_folds()
        running = [fold.name for fold in existing if fold.status == "training"]
        if running:  # e.g. kept running after quitting, or started in a terminal
            return (f"Fold {', '.join(running)} of this model seems to be still training (its log changed in the last "
                    "minutes). A second training of the same fold writes into the same folder. Start anyway?")
        try:
            # --c (continue) or --val (only validate) typed as additional arguments replace nothing either
            extra = set(split_arguments(self._arguments.text()))
        except ValueError:
            extra = set()
        if not existing or self._continue.isChecked() or extra & {"--c", "--val"}:
            return ""
        if len(existing) == 1:
            fold = existing[0]
            if fold.finished:
                return (f"Fold {fold.name} of this model is finished. Train it again from the start and replace its "
                        "checkpoints and validation results?")
            epoch = f" at epoch {fold.epoch}" if fold.epoch is not None else ""
            return (f"Fold {fold.name} of this model already has a training{epoch}. Start it again from the start "
                    "and replace it? To resume it instead, check Continue.")
        names = ", ".join(fold.name for fold in existing)
        return (f"Folds {names} of this model already exist. Train them again from the start and replace their "
                "results? To resume unfinished folds instead, check Continue.")

    def header_badges(self, state: DatasetState) -> list:
        """Trained folds first, then whether a new training can run (it needs preprocessed data)."""
        items = []
        running = sum(1 for model in state.models for fold in model.folds if fold.status == "training")
        if running:
            items.append(badge(f"● {running} fold{'s' if running != 1 else ''} training", "orange"))
        finished = sum(len(model.finished_folds) for model in state.models)
        models = len(state.models)
        count = f"{models} model{'s' if models != 1 else ''} · " if models else ""
        if finished:
            items.append(badge(f"✓ {count}{finished} fold{'s' if finished != 1 else ''} finished", "green"))
        elif not running:
            items.append(badge(f"{count}no trained fold yet", "gray"))
        stopped = sum(1 for model in state.models for fold in model.folds if fold.status == "stopped")
        if stopped:
            items.append(badge(f"{stopped} fold{'s' if stopped != 1 else ''} stopped · can be continued", "orange",
                               "Select the fold and click ▶ Continue… to resume it from its latest checkpoint."))
        if any(plans.preprocessed_time and plans.outdated for plans in state.plans):
            items.append(badge("preprocessing outdated · new trainings use old labels", "orange",
                               "labelsTr changed after preprocessing. Preprocess again (see Preprocess) so new "
                               "trainings use the current labels."))
        if not any(plans.preprocessed_time for plans in state.plans):
            items.append(badge("no preprocessed data · cannot train", "red" if state.raw.cases else "gray",
                               "Training needs preprocessed data (see Preprocess)."))
        return items
