"""Predict stage: input and output folders, prediction status per image, and the model panel."""

from __future__ import annotations

from pathlib import Path

from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QComboBox,
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
    badge,
    IconButtonsDelegate,
    mark_item,
    ToggleGroup,
    button,
    empty_state,
    form_hint,
    form_row,
    icon_button,
    label,
    make_table,
    scroll,
    section_label,
    table_item,
    tip,
)
from .common import CommandBox, StagePage, arguments_field


def _howto() -> HowToCard:
    return HowToCard("How to predict", [
        "Choose a folder with new images, named like in imagesTr (<span style='font-family:monospace'>_0000</span> "
        "suffix).",
        "Choose the model and the folds. More folds usually give better predictions, but take longer.",
        "Results are label files with the same names in the output folder.",
    ], [("orange", "Only the finished folds of a model can be used."),
        ("blue", "💡 Open a few predictions in Napari to check them before using all results.")], "Predict")


class PredictStage(StagePage):
    key, title, guide_stage = "predict", "Predict", 10
    folder = "results"

    def __init__(self, context: AppContext, parent=None) -> None:
        super().__init__(context, _howto(), parent)
        self._models: list[ModelInfo] = []
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
        grid.addWidget(icon_button("📁", "Open output folder",
                                   lambda: context.open_folder(Path(self._output.text()) if self._output.text() else None)),
                       1, 3)
        grid.setColumnStretch(1, 1)
        folders.body.addLayout(grid)
        self.center.addWidget(folders)
        images = Card()
        head = images.add_header("Input images")
        self._images_note = label("", muted=True, size=12)
        head.addWidget(self._images_note)
        head.addStretch()
        self._table = make_table(["Image", "Images", "Prediction", ""], stretch=0)
        self._predicted: list[bool] = []
        self._table.setItemDelegateForColumn(3, IconButtonsDelegate(
            self._table, lambda row: [("🔍", "Open in Napari · image and prediction", False)]
            if row < len(self._predicted) and self._predicted[row] else [],
            lambda _row, _number: self._open_predictions()))
        header = self._table.horizontalHeader()
        for column, width in ((1, 90), (2, 140), (3, 60)):
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
        layout.addWidget(label("Model", name="h3"))
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        self._model = QComboBox()
        self._model.setProperty("mono", True)
        self._model.currentIndexChanged.connect(self._model_changed)
        self._folds = ChoiceGroup(exclusive=False)
        self._folds.changed.connect(self._update_command)
        self._folds_hint = label("", muted=True, wrap=True, size=11)
        self._tta = ToggleGroup(["On · best", "Off · 8× faster"])
        self._tta.changed.connect(lambda _index: self._update_command())
        self._arguments = arguments_field()
        self._arguments.setPlaceholderText("Optional, e.g. --save_probabilities")
        self._arguments.textChanged.connect(self._update_command)
        self._folds_hint = form_hint("")
        layout.addLayout(form_row("Model", self._model))
        layout.addLayout(form_row("Folds", self._folds))
        layout.addWidget(self._folds_hint)
        layout.addLayout(form_row("Mirroring (TTA)", self._tta))
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
        if previous_record is not None and previous_record.dataset_id != state.record.dataset_id:
            self._input.clear()
            self._output.clear()
        self._models = [model for model in state.models if model.finished_folds]
        current = self._model.currentText()
        self._model.blockSignals(True)
        self._model.clear()
        for model in self._models:
            self._model.addItem(model.label)
        if current:
            self._model.setCurrentText(current)
        self._model.blockSignals(False)
        if not self._models:
            self._model.addItem("No model with a finished fold yet")
        self._model_changed()
        self._folders_changed()

    def header_badges(self, state: DatasetState) -> list:
        usable = sum(1 for model in state.models if model.finished_folds)
        items = [badge(f"✓ {usable} model{'s' if usable != 1 else ''} ready for prediction", "green")
                 if usable else badge("no model with a finished fold yet", "red")]
        return items + super().header_badges(state)

    def _current_model(self) -> ModelInfo | None:
        index = self._model.currentIndex()
        return self._models[index] if 0 <= index < len(self._models) else None

    def _model_changed(self, *_args) -> None:
        model = self._current_model()
        finished = [fold.name for fold in model.finished_folds] if model else []
        self._folds.set_options(finished, selected=finished)
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
        source = Path(self._input.text()) if self._input.text().strip() else None
        target = Path(self._output.text()) if self._output.text().strip() else None
        self.context.tasks.run("predict-status", lambda: prediction_status(source, target),
                               lambda rows: self._fill_images(rows, source, target))
        self._update_command()

    def _fill_images(self, rows: object, source: Path | None, target: Path | None) -> None:
        rows = rows if isinstance(rows, list) else []
        self._table.setVisible(bool(rows))
        self._images_empty.setVisible(not rows)
        predicted = sum(1 for _case, _channels, done in rows if done)
        self._images_note.setText(f"{len(rows)} images · {predicted} already in the output folder" if rows else "")
        self._predicted = [done for _case, _channels, done in rows]
        self._table.setRowCount(0)
        self._table.setRowCount(len(rows))
        self._table.setUpdatesEnabled(False)
        for index, (case, channels, done) in enumerate(rows):
            self._table.setItem(index, 0, table_item(case, mono=True))
            self._table.setItem(index, 1, mark_item(channels))
            status = table_item("✓ predicted" if done else "not yet", muted=not done, bold=done)
            if done:
                status.setForeground(QColor("#2f8f4e"))
            self._table.setItem(index, 2, status)
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
        return prediction_command(self.state.record.dataset_id, self._input.text(), self._output.text(), model.plans,
                                  model.configuration, self._arguments.text(), trainer=model.trainer,
                                  folds=" ".join(self._folds.selected()), disable_tta=self._tta.index() == 1)

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
        if not reason and not self._folds.selected():
            reason = "Select at least one fold."
        self._predict.setEnabled(not reason)
        self._predict.setToolTip(reason)
        self._busy_hint.setText(("⏳ " if self.context.jobs.busy else "") + reason)
        self._busy_hint.setVisible(bool(reason))

    def _run_prediction(self) -> None:
        if self.state is not None:
            self.context.jobs.start_nnunet("Predict", self.state.record.name, self._command_text())
