"""Predict stage: model to use, input and output folders, prediction status per image, and the run panel."""

from __future__ import annotations

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

from nnworkbench.services.commands import prediction_command
from nnworkbench.services.workflow_status import ModelInfo, prediction_status

from ..context import AppContext
from ..dataset_state import DatasetState
from ..widgets import (
    Card,
    ChoiceGroup,
    HowToCard,
    IconButtonsDelegate,
    ToggleGroup,
    badge,
    breakable,
    button,
    cell_widget,
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
)
from .common import (
    CommandBox,
    StagePage,
    arguments_field,
    fold_chip,
    mean_dice_item,
    model_name_cell,
    models_table,
    size_models_table,
)


def _howto() -> HowToCard:
    return HowToCard("How to predict", [
        ("Choose a folder with new images, named like in imagesTr (<span style='font-family:monospace'>_0000</span> "
        "suffix)."),
        "Choose the model and the folds. More folds usually give better predictions, but take longer.",
        "Results are label files with the same names in the output folder.",
    ], [], "Predict")


class PredictStage(StagePage):
    key, title, guide_stage = "predict", "Predict", 10
    folder = "results"

    def __init__(self, context: AppContext, parent=None) -> None:
        super().__init__(context, _howto(), parent)
        self._models: list[ModelInfo] = []
        self._selected: ModelInfo | None = None
        models_card = Card()
        head = models_card.add_header("Models")
        head.addWidget(label("models with at least one finished fold · click one to predict with it", muted=True,
                             size=12))
        head.addStretch()
        self._models_table = models_table("Folds")
        self._models_table.itemSelectionChanged.connect(self._model_selected)
        models_card.body.addWidget(self._models_table)
        self._models_empty = empty_state("No model with a finished fold yet. Train at least one fold first.")
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
        self._images_note = label("", muted=True, size=12)
        head.addWidget(self._images_note)
        head.addStretch()
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
        self._model_name = label("", muted=True, mono=True, wrap=True, size=11)
        layout.addWidget(self._model_name)
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
        self._continue_hint = form_hint("")
        self._arguments = arguments_field()
        self._arguments.setPlaceholderText("Optional, e.g. --save_probabilities")
        self._arguments.textChanged.connect(self._update_command)
        self._folds_hint = form_hint("")
        layout.addLayout(form_row("Folds", self._folds))
        layout.addWidget(self._folds_hint)
        layout.addLayout(form_row("Mirroring (TTA)", self._tta))
        layout.addLayout(form_row("Continue", self._continue))
        layout.addWidget(self._continue_hint)
        layout.addLayout(form_row("Additional arguments", self._arguments))
        layout.addWidget(section_label("Command"))
        self._command = CommandBox()
        self._command.setFixedHeight(130)
        layout.addWidget(self._command)
        actions = QHBoxLayout()
        actions.addWidget(button("Copy command", small=True, clicked=lambda: self._command.copy()))
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
            self._models_table.setCellWidget(row, 2, cell_widget(*(fold_chip(fold) for fold in model.finished_folds)))
            if model.label == previous:
                select = row
        self._models_table.blockSignals(False)
        size_models_table(self._models_table, self._models)
        if has_models:
            self._loading = True  # selecting while loading does not change the shared model
            self._models_table.selectRow(select)
            self._loading = False
        self._selected = self._models[select] if has_models else None
        self._model_changed()
        self._folders_changed()

    def header_badges(self, state: DatasetState) -> list:
        usable = sum(1 for model in state.models if model.finished_folds)
        items = [badge(f"✓ {usable} model{'s' if usable != 1 else ''} ready for prediction", "green")
                 if usable else badge("no model with a finished fold yet", "red")]
        return items

    def _current_model(self) -> ModelInfo | None:
        return self._selected

    def follow_model(self, label: str) -> None:
        if self._selected is not None and self._selected.label == label:
            return
        row = next((index for index, model in enumerate(self._models) if model.label == label), None)
        if row is not None:
            self._models_table.selectRow(row)

    def _model_selected(self) -> None:
        rows = self._models_table.selectionModel().selectedRows()
        if rows and 0 <= rows[0].row() < len(self._models):
            self._selected = self._models[rows[0].row()]
            if not getattr(self, "_loading", False):
                self.context.model_label = self._selected.label
            self._model_changed()

    def _model_changed(self, *_args) -> None:
        model = self._current_model()
        self._model_name.setText(breakable(f"{model.trainer}\n{model.plans} · {model.configuration}") if model
                                 else "No model with a finished fold yet")
        finished = [fold.name for fold in model.finished_folds] if model else []
        # Default: the cross-validation folds; fold "all" only when it is the only finished fold.
        default = [name for name in finished if name != "all"] or finished
        self._folds.set_options(finished, selected=default)
        running = [fold.name for fold in model.folds if not fold.finished] if model else []
        self._folds_hint.setText("finished folds of this model" + (
            f" · fold {', '.join(running)} {'is' if len(running) == 1 else 'are'} not finished" if running else ""))
        self._update_command()

    def _browse(self, field: QLineEdit) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Choose folder", field.text())
        if selected:
            field.setText(selected)
            self._folders_changed()

    def _folders_changed(self) -> None:
        self.context.store.setValue("predict/input", self._input.text().strip())
        self.context.store.setValue("predict/output", self._output.text().strip())
        source = Path(self._input.text()) if self._input.text().strip() else None
        target = Path(self._output.text()) if self._output.text().strip() else None
        self.context.tasks.run("predict-status", lambda: prediction_status(source, target),
                               self._fill_images)
        self._update_command()

    def _fill_images(self, rows: object) -> None:
        rows = rows if isinstance(rows, list) else []
        self._table.setVisible(bool(rows))
        self._images_empty.setVisible(not rows)
        predicted = sum(1 for _case, _channels, done in rows if done)
        self._images_note.setText(f"{len(rows)} cases · {predicted} already in the output folder" if rows else "")
        self._continue_hint.setText(
            f"{predicted} of {len(rows)} cases are already predicted. Without Continue they are predicted again and "
            "overwritten." if predicted else "No predictions in the output folder yet.")
        self._predicted = [done for _case, _channels, done in rows]
        self._table.setRowCount(0)
        self._table.setRowCount(len(rows))
        self._table.setUpdatesEnabled(False)
        for index, (case, channels, done) in enumerate(rows):
            self._table.setItem(index, 0, table_item(case, mono=True))
            self._table.setItem(index, 1, mark_item(channels))
            self._table.setItem(index, 2, mark_item(done))
        self._table.setUpdatesEnabled(True)

    def _open_predictions(self) -> None:
        source = Path(self._input.text()) if self._input.text().strip() else None
        target = Path(self._output.text()) if self._output.text().strip() else None
        if target is not None:
            self.context.inspect_predictions(target, source)

    def _command_text(self) -> str:
        model = self._current_model()
        if self.state is None or model is None:
            return ""
        # placeholders instead of '' until the folders are chosen (the run stays disabled until then)
        source = self._input.text().strip() or "INPUT_FOLDER"
        target = self._output.text().strip() or "OUTPUT_FOLDER"
        return prediction_command(self.state.record.dataset_id, source, target, model.plans,
                                  model.configuration, self._arguments.text(), trainer=model.trainer,
                                  folds=" ".join(self._folds.selected()), disable_tta=self._tta.index() == 1,
                                  continue_prediction=self._continue.isChecked())

    def _update_command(self, *_args) -> None:
        try:
            self._command.set_command(self._command_text())
        except ValueError:
            pass
        self.update_actions()

    def update_actions(self) -> None:
        if self.state is None:
            return
        reason = self.busy_or_missing("results")
        if not reason and self._current_model() is None:
            reason = "No model with a finished fold yet. Train at least one fold first."
        if not reason and not self._input.text().strip():
            reason = "Choose an input folder."
        if not reason and not self._output.text().strip():
            reason = "Choose an output folder."
        if not reason and not self._folds.selected():
            reason = "Select at least one fold."
        self._predict.setEnabled(not reason)
        self._predict.setToolTip(reason)
        self._busy_hint.setText(reason)
        self._busy_hint.setVisible(bool(reason))

    def _run_prediction(self) -> None:
        if self.state is not None:
            self.context.jobs.start_nnunet("Predict", self.state.record.name, self._command_text())
