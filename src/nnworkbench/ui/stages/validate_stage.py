"""Validate stage: models with their mean Dice, fold results, cases and the raw summary.json."""

from __future__ import annotations

import json
from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QButtonGroup,
    QHeaderView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nnworkbench.services.dataset_contents import json_object
from nnworkbench.services.workflow_status import (
    FoldInfo,
    ModelInfo,
    ValidationCase,
    mean_class_dice,
    validation_cases,
)

from ..context import AppContext
from ..dataset_state import DatasetState
from ..widgets import (
    Bar,
    badge,
    Card,
    HowToCard,
    IconButtonsDelegate,
    button,
    cell_widget,
    clear_layout,
    empty_state,
    label,
    polish,
    make_table,
    section_label,
    table_item,
)
from .common import FOLD_BUTTON_HEIGHT, FOLD_SIZE, StagePage, mean_dice_item, models_table, size_models_table


def _howto() -> HowToCard:
    return HowToCard("How to validate", [
        "Each fold predicts the cases it did <b>not</b> train on, so the Dice shows how well the model generalises.",
        "<b>Mean Dice</b> per model is the mean over its finished folds; compare models with the same folds for a fair "
        "comparison.",
        "Click a fold to see its results and cases; open low-scoring cases with 🔍.",
    ], [("purple", "✎ Annotation Workflow: annotate the weak classes in more images and train a new round.")], "Validate")


def _fold_button(name: str, dice: str, muted_dice: bool) -> QPushButton:
    """A fold button with the fold name and, below it, its Dice (grey for fold all, not a validation result)."""
    widget = QPushButton()
    column = QVBoxLayout(widget)
    column.setContentsMargins(0, 3, 0, 3)
    column.setSpacing(0)
    labels = []
    for text, role in ((name, "name"), (dice, "mutedDice" if muted_dice else "dice")):
        part = QLabel(text)
        part.setObjectName("foldButtonText")
        part.setProperty("role", role)
        part.setAlignment(Qt.AlignmentFlag.AlignCenter)
        part.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        column.addWidget(part)
        labels.append(part)

    def restyle(checked: bool) -> None:
        for part in labels:
            part.setProperty("checked", checked)
            polish(part)

    widget.toggled.connect(restyle)
    return widget


class ValidateStage(StagePage):
    key, title, guide_stage = "validate", "Validate", 9
    folder = "results"

    def __init__(self, context: AppContext, parent=None) -> None:
        super().__init__(context, _howto(), parent)
        self._annotation_tip = self.howto._tips.itemAt(0).widget()
        self._fold: FoldInfo | None = None
        self._model: ModelInfo | None = None
        self._summary: dict | None = None
        self._cases: tuple[ValidationCase, ...] = ()
        self._case: ValidationCase | None = None
        self._fold_buttons = QButtonGroup(self)
        self._fold_buttons.setExclusive(True)
        models_card = Card()
        head = models_card.add_header("Models")
        head.addWidget(label("mean Dice over the finished folds · click a fold for its results", muted=True, size=12))
        head.addStretch()
        self._models = models_table("Finished folds")
        models_card.body.addWidget(self._models)
        self._empty = empty_state("No validation results yet. They appear here as soon as the first fold has finished "
                                  "training.")
        models_card.body.addWidget(self._empty)
        self.center.addWidget(models_card)
        self._cases_card = Card()
        cases_head = self._cases_card.add_header()
        self._cases_title = label("", name="h3")
        cases_head.addWidget(self._cases_title)
        self._cases_note = label("", muted=True, size=12)
        cases_head.addWidget(self._cases_note)
        cases_head.addStretch()
        self._cases_table = make_table(["Case", "Mean Dice", "Weakest class", ""], stretch=0)
        self._cases_table.cellClicked.connect(lambda row, column: column != 3 and self._select_case(row))
        self._cases_table.setItemDelegateForColumn(3, IconButtonsDelegate(
            self._cases_table, lambda _row: [("🔍", "Inspect in Napari · prediction and label", False)],
            lambda row, _number: self._inspect(self._cases[row])))
        header = self._cases_table.horizontalHeader()
        for column, width in ((1, 110), (2, 220), (3, 60)):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
            self._cases_table.setColumnWidth(column, width)
        self._cases_card.body.addWidget(self._cases_table, 1)
        self.center.addWidget(self._cases_card, 1)
        self.right.addWidget(self._build_panel())
        from .common import empty_panel
        self._right_empty = empty_panel("Dice per class and the raw summary.json appear here once a fold has "
                                        "validation results.")
        self.right.addWidget(self._right_empty)

    def _build_panel(self) -> QWidget:
        panel = Card()
        layout = QVBoxLayout()
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)
        panel.body.addLayout(layout)
        head = QHBoxLayout()
        head.addWidget(label("summary.json", name="h3"))
        self._panel_subject = label("", muted=True, mono=True, size=11)
        head.addWidget(self._panel_subject, 1)
        self._close_case = button("✕", ghost=True, small=True, clicked=lambda: self._select_case(-1))
        head.addWidget(self._close_case)
        layout.addLayout(head)
        mean = QHBoxLayout()
        mean.setSpacing(12)
        self._foreground = label("—", name="bigNumber")
        mean.addWidget(self._foreground)
        mean_text = QVBoxLayout()
        mean_text.setSpacing(0)
        mean_text.addWidget(section_label("Foreground mean"))
        self._foreground_note = label("Dice", muted=True, size=12)
        mean_text.addWidget(self._foreground_note)
        mean.addLayout(mean_text)
        mean.addStretch()
        layout.addLayout(mean)
        dice_head = QHBoxLayout()
        dice_head.addWidget(label("Dice per class", name="h3"))
        self._dice_note = label("", muted=True, size=11)
        dice_head.addWidget(self._dice_note)
        dice_head.addStretch()
        layout.addLayout(dice_head)
        self._dice_bars = QGridLayout()
        self._dice_bars.setHorizontalSpacing(10)
        self._dice_bars.setVerticalSpacing(3)
        layout.addLayout(self._dice_bars)
        layout.addWidget(section_label("Details"))
        self._details = QPlainTextEdit()
        self._details.setObjectName("codeView")
        self._details.setReadOnly(True)
        layout.addWidget(self._details, 1)
        return panel

    # ------------------------------------------------------------ state

    def set_state(self, state: DatasetState) -> None:
        super().set_state(state)
        annotation = self.context.settings.annotation_workflow
        self._annotation_tip.setVisible(annotation)
        models = [model for model in state.models if any(fold.has_validation for fold in model.folds)]
        self._models.setVisible(bool(models))
        self._empty.setVisible(not models)
        self._cases_card.setVisible(bool(models))
        self.right.setCurrentIndex(0 if models else 1)
        for button_widget in self._fold_buttons.buttons():
            self._fold_buttons.removeButton(button_widget)
        self._models.setRowCount(len(models))
        first_button = None
        keep_button = None
        for row, model in enumerate(models):
            name = label(f"{model.trainer}<br><span style='color:#64748b'>{model.plans} · {model.configuration}</span>",
                         mono=True, rich=True)
            self._models.setCellWidget(row, 0, cell_widget(name))
            self._models.setItem(row, 1, mean_dice_item(model))
            buttons = []
            for fold in model.folds:
                if not fold.has_validation:
                    continue
                dice = f"{fold.dice:.2f}" if fold.dice is not None else "—"
                fold_button = _fold_button(fold.name, dice, muted_dice=fold.name == "all")
                fold_button.setObjectName("foldButton")
                fold_button.setFixedSize(FOLD_SIZE[0], FOLD_BUTTON_HEIGHT)
                fold_button.setCheckable(True)
                fold_button.setCursor(Qt.CursorShape.PointingHandCursor)
                tip = f"Fold {fold.name} · Dice {fold.dice:.3f}" if fold.dice is not None else f"Fold {fold.name}"
                if fold.name == "all":
                    tip += ("\nFold all is validated on its own training cases, so this is not a validation result "
                            "and it is not included in the mean Dice.")
                fold_button.setToolTip(tip)
                fold_button.clicked.connect(lambda _=False, m=model, f=fold: self._show_fold(m, f))
                self._fold_buttons.addButton(fold_button)
                buttons.append(fold_button)
                first_button = first_button or (fold_button, model, fold)
                if self._fold is not None and fold.path == self._fold.path:
                    keep_button = (fold_button, model, fold)
            self._models.setCellWidget(row, 2, cell_widget(*buttons))
        size_models_table(self._models, models)
        chosen = keep_button or first_button
        if chosen:
            chosen[0].setChecked(True)
            self._show_fold(chosen[1], chosen[2])

    def header_badges(self, state: DatasetState) -> list:
        validated = sum(1 for model in state.models if any(fold.has_validation for fold in model.folds))
        items = [badge(f"✓ {validated} model{'s' if validated != 1 else ''} with validation results", "green")
                 if validated else badge("no validation results yet", "gray")]
        return items + super().header_badges(state)

    def _show_fold(self, model: ModelInfo, fold: FoldInfo) -> None:
        self._model, self._fold = model, fold
        self._cases_title.setText(f"Fold {fold.name} · cases")
        path = fold.path / "validation" / "summary.json"
        self.context.tasks.run("validate-summary", lambda: json_object(path), lambda data: self._apply_summary(fold, data))

    def _apply_summary(self, fold: FoldInfo, summary: object) -> None:
        if fold is not self._fold or self.state is None:
            return
        self._summary = summary if isinstance(summary, dict) else None
        names = self.state.names
        self._cases = validation_cases(self._summary, names)
        self._cases_note.setText(f"{len(self._cases)} validation cases · click a case for its metrics")
        self._cases_table.setRowCount(len(self._cases))
        for row, case in enumerate(self._cases):
            self._cases_table.setItem(row, 0, table_item(case.case_id, mono=True))
            self._cases_table.setItem(row, 1, table_item(f"{case.mean_dice:.2f}" if case.mean_dice is not None else "—",
                                                         bold=True))
            weakest = f"{case.weakest[0]} {case.weakest[1]:.2f}" if case.weakest else "—"
            self._cases_table.setItem(row, 2, table_item(weakest, muted=True))

        self._select_case(-1)

    def _select_case(self, row: int) -> None:
        self._case = self._cases[row] if 0 <= row < len(self._cases) else None
        if self._case is None:
            self._cases_table.clearSelection()
        self._close_case.setVisible(self._case is not None)
        names = self.state.names if self.state else {}
        clear_layout(self._dice_bars)
        if self._case is not None:
            self._panel_subject.setText(self._case.case_id)
            self._dice_note.setText("")
            self._foreground.setText(f"{self._case.mean_dice:.2f}" if self._case.mean_dice is not None else "—")
            self._foreground_note.setText(f"Dice · case {self._case.case_id}")
            rows = []
            for key, values in self._case.metrics.items():
                dice = values.get("Dice") if isinstance(values, dict) else None
                if isinstance(dice, (int, float)):
                    try:
                        rows.append((names.get(int(key), key), float(dice)))
                    except ValueError:
                        rows.append((key, float(dice)))
            entry = {"prediction_file": self._case.prediction_file, "reference_file": self._case.reference_file,
                     "metrics": self._case.metrics}
            self._details.setPlainText(json.dumps(entry, indent=2))
        else:
            fold = self._fold.name if self._fold else ""
            self._panel_subject.setText(f"fold {fold} · foreground_mean and mean")
            self._dice_note.setText(f"fold {fold} · mean")
            foreground = (self._summary or {}).get("foreground_mean", {}).get("Dice")
            self._foreground.setText(f"{foreground:.2f}" if isinstance(foreground, (int, float)) else "—")
            self._foreground_note.setText(f"Dice · fold {fold}")
            rows = mean_class_dice(self._summary, names)
            summary = self._summary or {}
            self._details.setPlainText(json.dumps({"foreground_mean": summary.get("foreground_mean"),
                                                   "mean": summary.get("mean")}, indent=2))
        for index, (name, dice) in enumerate(rows):
            name_label = label(str(name), size=12)
            name_label.setFixedWidth(130)
            self._dice_bars.addWidget(name_label, index, 0)
            self._dice_bars.addWidget(Bar(dice), index, 1)
            self._dice_bars.addWidget(label(f"{dice:.2f}", muted=True, size=12), index, 2)
        self._dice_bars.setColumnStretch(1, 1)

    def _inspect(self, case: ValidationCase) -> None:
        if self._fold is None:
            return
        prediction = self._fold.path / "validation" / Path(case.prediction_file).name
        self.context.inspect_validation(prediction, self.state.raw_path if self.state else None)
