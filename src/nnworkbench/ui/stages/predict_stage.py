"""Predict stage: model to use, input and output folders, prediction status per image, and the run panel."""

from __future__ import annotations

import os
from pathlib import Path

from qtpy.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from nnworkbench.path_setup import NON_ASCII_HINT, typed_path, unusable_for_nnunet
from nnworkbench.services.commands import prediction_command
from nnworkbench.services.defaults import (
    CONFIGURATIONS,
    DEFAULT_CONFIGURATION,
    DEFAULT_PLANS,
    DENSE_TRAINER,
    KNOWN_PLANS,
    SPARSE_TRAINER,
    prediction_folds,
    prediction_processes,
    trainer_for,
)
from nnworkbench.services.device_probe import (
    DEVICE_CHOICES,
    default_choice,
    device_argument,
    device_hint,
)
from nnworkbench.services.workflow_status import ModelInfo, prediction_status

from ..context import AppContext
from ..dataset_state import DatasetState
from ..editable_dropdown import EditableDropdownComboBox
from ..style import ORANGE
from ..widgets import (
    Card,
    ChoiceGroup,
    HowToCard,
    IconButtonsDelegate,
    NumberField,
    ToggleGroup,
    badge,
    button,
    empty_state,
    form_hint,
    form_row,
    icon_button,
    label,
    make_table,
    mark_item,
    scroll,
    section_label,
    table_item,
    tip,
    with_stepper,
)
from .common import (
    DEVICE_INFO,
    CommandBox,
    StagePage,
    arguments_error,
    arguments_field,
    fold_cells,
    fold_chip,
    header_note,
    mean_dice_item,
    model_name_cell,
    models_table,
    size_models_table,
)

# Narrower row titles than in the run dialogs: the panel is 340 px wide at the minimum window width, and
# "Skip already predicted cases" or the processes fields did not fit next to the default 112 px titles.
_TITLE_WIDTH = 84
_ALL_FOLDS = ["0", "1", "2", "3", "4", "all"]
_HINT_INDENT = _TITLE_WIDTH + 10


def _howto() -> HowToCard:
    return HowToCard("How to predict", [
        ("Choose a folder with new images, named like in imagesTr (<span style='font-family:monospace'>_0000</span> "
        "suffix)."),
        "Choose the model and the folds. More folds usually give better predictions, but take longer.",
        ("A model that is not in nnUNet_results on this computer (e.g. trained on a cluster): enter its trainer, "
         "plans and configuration on the right and use <b>Copy command</b> where the model is."),
        "Results are label files with the same names in the output folder.",
    ], [], "Predict")


class PredictStage(StagePage):
    key, title, guide = "predict", "Predict", ("6-predict.md", "")
    folder = "results"

    def __init__(self, context: AppContext, parent=None) -> None:
        super().__init__(context, _howto(), parent)
        self._models: list[ModelInfo] = []
        self._selected: ModelInfo | None = None
        self._input_cases: int | None = None  # cases in the input folder; None while it is listed
        self._listed_source: Path | None = None  # the input folder of the last completed listing
        self._input_error = ""  # why the input folder cannot be listed
        self._not_ready = ""  # why cases of the input folder cannot be predicted (missing channels, no suffix)
        self._folds_note = ""  # the fold hint without the warning about folds that are not here
        self._trainers_loaded = False  # the trainer list holds the installed trainers (context.nnunet_classes)
        models_card = Card()
        head = models_card.add_header("Models")
        head.addWidget(header_note("models with a finished fold · click one, or enter a model trained elsewhere on "
                                   "the right"), 1)
        self._models_table = models_table("Folds")
        self._models_table.itemSelectionChanged.connect(self._model_selected)
        models_card.body.addWidget(self._models_table)
        self._models_empty = empty_state("No model with a finished fold in nnUNet_results on this computer. Train one "
                                         "first, or enter the trainer, plans and configuration of a model on a "
                                         "remote machine on the right and copy the command.")
        models_card.body.addWidget(self._models_empty)
        self.center.addWidget(models_card)
        folders = Card()
        grid = QGridLayout()
        grid.setContentsMargins(20, 14, 20, 14)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        self._input = QLineEdit()
        self._output = QLineEdit()
        for row, (name, field) in enumerate((("Input", self._input), ("Output", self._output))):
            field.setProperty("mono", True)
            field.setPlaceholderText(f"{name} folder")
            field.editingFinished.connect(self._folders_changed)
            grid.addWidget(label(name, muted=True), row, 0)
            grid.addWidget(field, row, 1)
            browse = button("Browse…", small=True, clicked=lambda target=field: self._browse(target))
            grid.addWidget(browse, row, 2)
            grid.addWidget(icon_button("folder", f"Open folder · {name.lower()}",
                                       lambda target=field: context.open_folder(
                                           Path(target.text()) if target.text().strip() else None)), row, 3)
        grid.setColumnStretch(1, 1)
        folders.body.addLayout(grid)
        self.center.addWidget(folders)
        images = Card()
        head = images.add_header("Input cases")
        self._images_note = header_note()
        head.addWidget(self._images_note, 1)
        self._table = make_table(["Case", "Images", "Prediction", ""], stretch=0)
        self._predicted: list[bool] = []
        self._table.setItemDelegateForColumn(3, IconButtonsDelegate(
            self._table, lambda row: [("search", "Open in Napari · image and prediction", False)]
            if row < len(self._predicted) and self._predicted[row] else [],
            lambda _row, _number: self._open_predictions()))
        header = self._table.horizontalHeader()
        for column, width in ((1, 90), (2, 100), (3, 60)):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
            self._table.setColumnWidth(column, width)
        images.body.addWidget(self._table, 1)
        self._images_empty = empty_state("Choose an input folder with images.")
        images.body.addWidget(self._images_empty, 1)
        self.center.addWidget(images, 1)
        self.right.addWidget(scroll(self._build_panel()))

    def _build_panel(self) -> QWidget:
        panel = Card()
        layout = QVBoxLayout()
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)
        panel.body.addLayout(layout)
        layout.addWidget(label("Predict", name="h3"))
        # Trainer, plans and configuration: filled from the selected model, or typed for a model that is not in
        # nnUNet_results on this computer (then only Copy command works, see _current_model).
        self._trainer = EditableDropdownComboBox()
        self._trainer.addItems([DENSE_TRAINER, SPARSE_TRAINER])
        self._plans = EditableDropdownComboBox()
        self._plans.addItems(list(KNOWN_PLANS))
        self._configuration = EditableDropdownComboBox()
        self._configuration.addItems(list(CONFIGURATIONS))
        for combo in (self._trainer, self._plans, self._configuration):
            combo.setEditable(True)
            combo.setProperty("mono", True)
            combo.setSizeAdjustPolicy(combo.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
            combo.setCurrentIndex(-1)  # empty until a model is selected or the defaults are filled in
            combo.currentTextChanged.connect(self._fields_changed)
        self.context.classes_changed.connect(self._set_trainers)
        self._folds = ChoiceGroup(exclusive=False)
        self._folds.changed.connect(self._update_command)
        self._folds_hint = label("", muted=True, wrap=True, size=11)
        self._tta = ToggleGroup(["On · best", "Off · faster"])
        self._tta.changed.connect(lambda _index: self._update_command())
        self._continue = QCheckBox("Skip already predicted cases")
        self._continue.setToolTip("--continue_prediction: skips images that already have a prediction in the output folder. Without it, "
                                  "nnU-Net predicts every image again and overwrites existing predictions.")
        self._continue.setChecked(True)  # default: keep existing predictions and only predict the missing cases
        self._continue.toggled.connect(self._update_command)
        self._continue_hint = form_hint("", width=_HINT_INDENT)
        # Processes for preprocessing (-npp) and segmentation export (-nps); nnU-Net's default is 3 each.
        self._npp, self._nps = NumberField(1, 99, prediction_processes()), NumberField(1, 99, prediction_processes())
        processes = QWidget()
        processes_grid = QGridLayout(processes)  # under each other: side by side is too wide for the panel
        processes_grid.setContentsMargins(0, 0, 0, 0)
        processes_grid.setHorizontalSpacing(8)
        processes_grid.setVerticalSpacing(6)
        fields = [(self._npp, "preprocessing", "-npp: processes for preprocessing the images"),
                  (self._nps, "export", "-nps: processes for writing the segmentations")]
        for row, (field, name, purpose) in enumerate(fields):
            field.setRange(1, max(1, os.cpu_count() or 1))
            field.setValue(prediction_processes())
            field.setFixedWidth(60)
            field.setToolTip(purpose)
            field.valueChanged.connect(self._update_command)
            processes_grid.addWidget(label(name, muted=True, size=12), row, 0)
            processes_grid.addWidget(with_stepper(field), row, 1)
        processes_grid.setColumnStretch(2, 1)
        self._processes_hint = form_hint("Fewer processes use less RAM and CPU but take longer. Reduce them if you "
                                         "run out of RAM.", width=_HINT_INDENT)
        self._device = ToggleGroup(DEVICE_CHOICES)
        self._device.changed.connect(lambda _index: self._update_command())
        self._device_hint = form_hint("", width=_HINT_INDENT)
        self._device.set_index(DEVICE_CHOICES.index(default_choice(self.context.device)))
        self.context.device_changed.connect(self._preselect_device)
        self._arguments = arguments_field()
        self._arguments.setPlaceholderText("Optional, e.g. --save_probabilities")
        self._arguments.textChanged.connect(self._update_command)
        self._folds_hint = form_hint("", width=0)  # below the fold buttons
        # the fold buttons below their title: six of them next to the title are wider than a narrow panel
        model_grid = QGridLayout()  # compact: the three fields close together, no explanation below them
        model_grid.setContentsMargins(0, 0, 0, 0)
        model_grid.setHorizontalSpacing(10)
        model_grid.setVerticalSpacing(4)
        for row, (name, combo) in enumerate((("Trainer", self._trainer), ("Plans", self._plans),
                                             ("Configuration", self._configuration))):
            title = label(name, muted=True)
            title.setFixedWidth(_TITLE_WIDTH)
            model_grid.addWidget(title, row, 0)
            model_grid.addWidget(combo, row, 1)
        layout.addLayout(model_grid)
        layout.addWidget(label("Folds", muted=True))
        layout.addWidget(self._folds)
        layout.addWidget(self._folds_hint)
        layout.addLayout(form_row("Mirroring (TTA)", self._tta, width=_TITLE_WIDTH))
        layout.addLayout(form_row("Continue", self._continue, width=_TITLE_WIDTH))
        layout.addWidget(self._continue_hint)
        layout.addLayout(form_row("Processes", processes, width=_TITLE_WIDTH))
        layout.addWidget(self._processes_hint)
        layout.addLayout(form_row("Device", self._device, width=_TITLE_WIDTH, info=DEVICE_INFO))
        layout.addWidget(self._device_hint)
        layout.addLayout(form_row("Additional arguments", self._arguments, width=_TITLE_WIDTH))
        layout.addWidget(section_label("Command"))
        self._command = CommandBox()
        self._command.setFixedHeight(130)
        layout.addWidget(self._command)
        actions = QHBoxLayout()
        actions.addWidget(self._command.copy_button())
        self._predict = button("▶ Predict", small=True, primary=True, clicked=self._run_prediction)
        actions.addWidget(self._predict)
        actions.addStretch()
        layout.addLayout(actions)
        self._busy_hint = tip("", "blue")
        layout.addWidget(self._busy_hint)
        layout.addStretch()
        return panel

    # ------------------------------------------------------------ state

    def set_state(self, state: DatasetState) -> None:
        previous_record = self.state.record if self.state else None
        super().set_state(state)
        self._listed_source = None  # channels and file ending may differ now: the input is checked again
        self.context.discover_classes()  # once per session; _set_trainers fills the list when it is known
        if not self._trainers_loaded and self.context.nnunet_classes is not None:
            self._set_trainers()
        if previous_record is None:  # the last input/output folders (one pair in total, kept across sessions)
            self._input.setText(str(self.context.store.value("predict/input", "") or ""))
            self._output.setText(str(self.context.store.value("predict/output", "") or ""))
        self._models = [model for model in state.models if model.finished_folds]
        has_models = bool(self._models)
        self._models_table.setVisible(has_models)
        self._models_empty.setVisible(not has_models)
        previous = self.context.model_label or (self._selected.label if self._selected else None)
        self._models_table.blockSignals(True)
        self._models_table.setRowCount(len(self._models))
        select = 0
        for row, model in enumerate(self._models):
            self._models_table.setCellWidget(row, 0, model_name_cell(model))
            self._models_table.setItem(row, 1, mean_dice_item(model))
            self._models_table.setCellWidget(row, 2, fold_cells(fold_chip(fold) for fold in model.finished_folds))
            if model.label == previous:
                select = row
        self._models_table.blockSignals(False)
        size_models_table(self._models_table, self._models)
        self._loading = True  # selecting or filling while loading does not change the shared model
        if has_models:
            self._models_table.selectRow(select)
            self._selected = self._models[select]
            self._fill_fields(self._selected)
        elif not self._fields_filled():  # the defaults the GUI would train with, or the model chosen on Train
            parts = (previous or "").split("__")
            if len(parts) == 3 and all(parts):
                self._set_fields(*parts)
            else:
                self._set_fields(trainer_for(state.sparse), default_plans_for(state), DEFAULT_CONFIGURATION)
        self._loading = False
        self._selected = self._current_model()
        self._model_changed()
        self._folders_changed()

    def header_badges(self, state: DatasetState) -> list:
        usable = sum(1 for model in state.models if model.finished_folds)
        items = [badge(f"✓ {usable} model{'s' if usable != 1 else ''} ready for prediction", "green")
                 if usable else badge("no model with a finished fold yet", "red")]
        return items

    def _current_model(self) -> ModelInfo | None:
        """The local model (with a finished fold) that matches the fields, or None for a model that is not in
        nnUNet_results on this computer: its command can still be copied, but not run here."""
        fields = self._fields()
        return next((model for model in self._models
                     if (model.trainer, model.plans, model.configuration) == fields), None)

    def _fields(self) -> tuple[str, str, str]:
        return self._trainer.currentText(), self._plans.currentText(), self._configuration.currentText()

    def _fields_filled(self) -> bool:
        return all(self._fields())

    def _set_fields(self, trainer: str, plans: str, configuration: str) -> None:
        for combo, text in ((self._trainer, trainer), (self._plans, plans), (self._configuration, configuration)):
            combo.blockSignals(True)
            combo.setCurrentText(text)
            combo.blockSignals(False)

    def _fill_fields(self, model: ModelInfo) -> None:
        self._set_fields(model.trainer, model.plans, model.configuration)

    def _set_trainers(self) -> None:
        """The installed trainers (also custom ones) in the dropdown; the typed or chosen name stays."""
        trainers = self.context.nnunet_classes.trainers if self.context.nnunet_classes else ()
        self._trainers_loaded = bool(trainers)
        if trainers:
            current = self._trainer.currentText()
            self._trainer.blockSignals(True)
            self._trainer.clear()
            self._trainer.addItems(list(trainers))
            self._trainer.setCurrentText(current)
            self._trainer.blockSignals(False)

    def follow_model(self, label: str) -> None:
        if self._selected is not None and self._selected.label == label:
            return
        row = next((index for index, model in enumerate(self._models) if model.label == label), None)
        if row is not None:
            self._models_table.selectRow(row)
            return
        parts = label.split("__")  # a model typed on Train that is not trained here
        if len(parts) == 3 and all(parts):
            self._set_fields(*parts)
            self._fields_changed()

    def _model_selected(self) -> None:
        rows = self._models_table.selectionModel().selectedRows()
        if rows and 0 <= rows[0].row() < len(self._models):
            self._selected = self._models[rows[0].row()]
            self._fill_fields(self._selected)
            if not getattr(self, "_loading", False):
                self.context.model_label = self._selected.label
            self._model_changed()

    def _fields_changed(self, *_args) -> None:
        """Typed or chosen trainer, plans or configuration: select the matching local model, if any."""
        if getattr(self, "_loading", False) or self.state is None:
            return
        model = self._current_model()
        self._selected = model
        self._models_table.blockSignals(True)
        if model is not None:
            self._models_table.selectRow(self._models.index(model))
            self.context.model_label = model.label
        else:
            self._models_table.clearSelection()
        self._models_table.blockSignals(False)
        self._model_changed()

    def _model_changed(self, *_args) -> None:
        model = self._current_model()
        tooltips: dict[str, str] = {}
        if model is not None:
            finished = [fold.name for fold in model.finished_folds]
            # every fold can be chosen: folds that are not finished here may be finished on a remote machine
            # (then only Copy command works, see update_actions)
            options = _ALL_FOLDS + [fold.name for fold in model.folds if fold.name not in _ALL_FOLDS]
            unfinished = [fold.name for fold in model.folds if not fold.finished]
            elsewhere = [name for name in options if name not in finished and name not in unfinished]
            tooltips = {name: f"fold {name} is not finished on this computer" for name in unfinished}
            tooltips.update({name: f"fold {name} is not in nnUNet_results on this computer" for name in elsewhere})
            notes = [f"finished here: {', '.join(finished)}"]
            if unfinished:
                notes.append(f"not finished: {', '.join(unfinished)}")
            self._folds_note = " · ".join(notes) + " · other folds only with Copy command"
        elif self._fields_filled():
            finished, options = _ALL_FOLDS, _ALL_FOLDS
            self._folds_note = "the folds cannot be checked here: select the ones the model has"
        else:
            finished, options = [], []
            self._folds_note = ""
        self._folds_hint.setText(self._folds_note)
        self._folds.set_options(options, selected=prediction_folds(finished), tooltips=tooltips)  # as `suggest`
        self._update_command()

    def _folds_missing_here(self) -> list[str]:
        """Selected folds that have no finished training in nnUNet_results on this computer."""
        model = self._current_model()
        if model is None:
            return []
        finished = {fold.name for fold in model.finished_folds}
        return [name for name in self._folds.selected() if name not in finished]

    def _browse(self, field: QLineEdit) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Choose folder", field.text())
        if selected:
            field.setText(selected)
            self._folders_changed()

    def _folders_changed(self) -> None:
        self.context.store.setValue("predict/input", self._input.text().strip())
        self.context.store.setValue("predict/output", self._output.text().strip())
        source, target = _folder(self._input), _folder(self._output)
        if source != self._listed_source or self._input_error:
            # a new input folder: Predict waits for its listing. Only the output changed: the cases and their
            # problems stay, so ▶ Predict stays enabled (the listing only updates the "already predicted" marks)
            self._input_cases, self._input_error, self._not_ready = None, "", ""
        channels = self.state.raw.required_suffixes if self.state else 1  # an RGB file covers three channels
        ending = self.state.manifest.get("file_ending") if self.state and isinstance(self.state.manifest, dict) else None
        ending = ending if isinstance(ending, str) and ending.startswith(".") else None

        def listing() -> list:
            if source is not None and not source.is_dir():
                raise NotADirectoryError(f"{source} does not exist or is not a folder")
            # every channel of dataset.json needs its file (case_0000, case_0001, …), as nnU-Net expects
            return [case.for_channels(channels) for case in prediction_status(source, target, file_ending=ending)]

        self.context.tasks.run("predict-status", listing, self._fill_images, self._listing_failed)
        self._update_command()

    def _listing_failed(self, error: str) -> None:
        """The input folder is missing or cannot be read: no cases from an earlier folder stay, Predict is off."""
        self._fill_images([])
        self._listed_source = None
        self._input_error = error.split(": ", 1)[-1]  # without the exception type
        self._images_note.setText(f"Cannot list the input folder · {self._input_error}")
        self.update_actions()

    def _fill_images(self, rows: object) -> None:
        rows = rows if isinstance(rows, list) else []
        self._input_cases, self._input_error = len(rows), ""
        self._listed_source = _folder(self._input)
        self._table.setVisible(bool(rows))
        self._images_empty.setVisible(not rows)
        predicted = sum(1 for _case, _channels, done in rows if done)
        not_ready = [case for case in rows if not case[1]]
        ending = self.state.manifest.get("file_ending") if self.state and isinstance(self.state.manifest, dict) else ""
        self._not_ready = _not_ready_reason(not_ready, self.state.raw.required_suffixes if self.state else ("0000",),
                                            str(ending))
        self._images_note.setText(f"{len(rows)} cases" + (f" · {len(not_ready)} not ready" if not_ready else "")
                                  + f" · {predicted} already in the output folder" if rows else "")
        self._continue_hint.setText(
            f"{predicted} of {len(rows)} cases are already predicted. Without Continue they are predicted again and "
            "overwritten." if predicted else "No predictions in the output folder yet.")
        self._predicted = [done for _case, _channels, done in rows]
        self._table.setRowCount(0)
        self._table.setRowCount(len(rows))
        self._table.setUpdatesEnabled(False)
        for index, (case, channels, done) in enumerate(rows):
            self._table.setItem(index, 0, table_item(case, mono=True))
            images = mark_item(channels)
            if not channels:
                images.setToolTip(_case_problem(rows[index]))
            self._table.setItem(index, 1, images)
            self._table.setItem(index, 2, mark_item(done))
        self._table.setUpdatesEnabled(True)
        self.update_actions()

    def _open_predictions(self) -> None:
        source, target = _folder(self._input), _folder(self._output)
        if target is not None:
            self.context.inspect_predictions(target, source)

    def _command_text(self) -> str:
        if self.state is None or not self._fields_filled():
            return ""
        trainer, plans, configuration = self._fields()
        # placeholders instead of '' until the folders are chosen (the run stays disabled until then)
        source, target = _folder(self._input), _folder(self._output)
        return prediction_command(self.state.record.dataset_id, str(source or "INPUT_FOLDER"),
                                  str(target or "OUTPUT_FOLDER"), plans,
                                  configuration, self._arguments.text(), trainer=trainer,
                                  folds=self._folds.selected(), disable_tta=self._tta.index() == 1,
                                  continue_prediction=self._continue.isChecked(),
                                  preprocessing_processes=self._npp.value(), export_processes=self._nps.value(),
                                  device=device_argument(DEVICE_CHOICES[self._device.index()]))

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
        self._device_hint.setText(device_hint(DEVICE_CHOICES[self._device.index()], self.context.device))
        self.update_actions()

    def update_actions(self) -> None:
        if self.state is None:
            return
        reason = arguments_error(self._arguments.text()) or self.busy_or_missing("results")
        source, target = _folder(self._input), _folder(self._output)
        if not reason and not self._fields_filled():
            reason = "Enter the trainer, plans and configuration of the model."
        if not reason:
            reason = self.unknown_class("trainer", self._trainer.currentText())
        if not reason:
            reason = self.device_missing(DEVICE_CHOICES[self._device.index()])
        if not reason and self._current_model() is None:
            reason = ("This model is not in nnUNet_results on this computer. Use Copy command and run it where the "
                      "model is, or train it here first.")
        if not reason and source is None:
            reason = "Choose an input folder."
        if not reason and target is None:
            reason = "Choose an output folder."
        if not reason and not (source.is_absolute() and target.is_absolute()):  # would be in the GUI's start folder
            reason = "Use the full path of the input and output folder (Browse… gives it)."
        if not reason and self._input_error:
            reason = f"Cannot list the input folder · {self._input_error}"
        if not reason and self._input_cases is None:
            reason = "Listing the input folder…"
        if not reason and not self._input_cases:
            reason = "The input folder has no images (named like in imagesTr, e.g. case_0000.nii.gz)."
        if not reason and self._not_ready:
            reason = self._not_ready
        if not reason and _same_folder(source, target):
            reason = "Choose an output folder different from the input folder."
        if not reason and (unusable := [str(path) for path in (source, target) if unusable_for_nnunet(path)]):
            reason = f"{NON_ASCII_HINT}: {', '.join(unusable)}."
        if not reason and not self._folds.selected():
            reason = "Select at least one fold."
        missing = self._folds_missing_here()
        if not reason and missing:
            reason = (f"Fold {', '.join(missing)} {'is' if len(missing) == 1 else 'are'} not finished in "
                      "nnUNet_results on this computer. Use Copy command and run it where the folds are.")
        # the warning next to the folds too: the reason below the button may be scrolled out of view
        self._folds_hint.setText(self._folds_note + (
            f" · <span style='color:{ORANGE}'>fold {', '.join(missing)} not here: Copy command only</span>"
            if missing else ""))
        self._predict.setEnabled(not reason)
        self._predict.setToolTip(reason)
        self.show_run_reason(self._busy_hint, reason)

    def _run_prediction(self) -> None:
        self.update_actions()
        if self.state is not None and self._predict.isEnabled():
            self.context.jobs.start_nnunet("Predict", self.state.record.name, self._command_text())


def default_plans_for(state: DatasetState) -> str:
    """The plans the Train page would preselect: the default planner's, or the plans preprocessed here."""
    names = [plans.name for plans in state.plans]
    return DEFAULT_PLANS if DEFAULT_PLANS in names or not names else names[0]


def _case_problem(case) -> str:
    """Why one input case cannot be predicted (hover text of its ✗)."""
    if getattr(case, "wrong_ending", ()):
        return (f"{case.wrong_ending[0]} does not end with the dataset's file ending (dataset.json file_ending): "
                "nnU-Net skips it. Rename or convert it.")
    if getattr(case, "unsuffixed", False):
        return f"A file without channel suffix: rename it to {case[0]}_0000 (and so on for more channels)."
    if getattr(case, "missing", ()):
        return "No file for channel " + ", ".join(f"_{suffix}" for suffix in case.missing) + "."
    return "Not ready."


def _not_ready_reason(cases: list, channels: tuple[str, ...], file_ending: str = "") -> str:
    """Why Predict is disabled when input cases lack channels or the channel suffix or end differently, or ""."""
    if not cases:
        return ""
    endings = [name for case in cases for name in getattr(case, "wrong_ending", ())]
    if endings:
        return (f"Cannot predict: {len(endings)} file{'s' if len(endings) != 1 else ''} (e.g. {endings[0]}) "
                f"{'do' if len(endings) != 1 else 'does'} not end with the dataset's file ending \"{file_ending}\" "
                "(also upper/lower case); nnU-Net would skip them. Rename or convert them.")
    unsuffixed_ids = {case[0] for case in cases if getattr(case, "unsuffixed", False)}
    missing = [case for case in cases if getattr(case, "missing", ()) and case[0] not in unsuffixed_ids]
    unsuffixed = [case for case in cases if getattr(case, "unsuffixed", False)]
    expected = f"_{channels[0]}" if len(channels) == 1 else f"_{channels[0]} to _{channels[-1]}"
    parts = []
    if missing:
        parts.append(f"{len(missing)} case{'s' if len(missing) != 1 else ''} without a file for every channel of "
                     f"dataset.json (e.g. {missing[0][0]}: "
                     f"{', '.join('_' + suffix for suffix in missing[0].missing)} missing)")
    if unsuffixed:
        parts.append(f"{len(unsuffixed)} file name{'s' if len(unsuffixed) != 1 else ''} without channel suffix "
                     f"(e.g. {unsuffixed[0][0]})")
    return (f"Cannot predict: {' and '.join(parts)}. nnU-Net needs one file per channel, named like in imagesTr "
            f"(suffix {expected}). Add or rename the files, then the cases are listed as ready.")


def _folder(field: QLineEdit) -> Path | None:
    """The folder typed or pasted in a field: quotes, file: URL, ``~`` and ``$VAR`` resolved (nnU-Net gets the
    path quoted, so no shell expands it)."""
    text = typed_path(field.text()).strip()
    return Path(text) if text else None


def _same_folder(first: Path | None, second: Path | None) -> bool:
    if first is None or second is None:
        return False
    try:
        return first.resolve() == second.resolve()
    except (OSError, RuntimeError, ValueError):
        return first == second
