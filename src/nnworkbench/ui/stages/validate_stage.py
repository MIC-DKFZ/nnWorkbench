"""Validate stage: models with their mean Dice, fold results, cases and the raw summary.json."""

from __future__ import annotations

import json
from pathlib import Path

from qtpy.QtCore import QEvent, QRect, QSize, Qt
from qtpy.QtGui import QColor, QFont, QFontMetrics, QPainter
from qtpy.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QHeaderView,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from nnworkbench.services.dataset_contents import json_object
from nnworkbench.services.workflow_status import (
    FoldInfo,
    ModelInfo,
    ValidationCase,
    class_name,
    finite_number,
    mean_class_dice,
    summary_foreground_dice,
    validation_cases,
)

from .. import style
from ..context import AppContext
from ..dataset_state import DatasetState
from ..widgets import (
    Card,
    HowToCard,
    IconButtonsDelegate,
    badge,
    button,
    empty_state,
    icon_html,
    label,
    make_table,
    scroll,
    section_label,
    table_item,
)
from .common import (
    FOLD_ALL_NOTE,
    FOLD_SIZE,
    StagePage,
    fold_cells,
    header_note,
    mean_dice_item,
    model_name_cell,
    models_table,
    size_models_table,
    wrappable,
)

WEAKEST_CLASSES = 15  # with more than SHOW_ALL_FROM classes only the weakest are listed until "Show all"
SHOW_ALL_FROM = 20


class DiceBars(QWidget):
    """Name, bar and Dice for each class (or fold), painted in one widget.

    One label, bar and label widget per class took 4 s (150 classes) to 13 s (500) per click; painting only the
    visible rows is instant. Long names end with "…"; the hover text has the full name and the Dice.
    """

    ROW_HEIGHT = 20
    NAME_WIDTH = 160  # at most; 45 % of a narrow panel

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._rows: list[tuple[str, float, str, str, bool]] = []  # name, Dice, value text, tooltip, muted
        font = QFont(self.font())
        font.setPixelSize(12)
        self.setFont(font)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(0)

    @property
    def rows(self) -> list[tuple[str, float, str, str]]:
        """The shown rows as (name, Dice, value text, tooltip)."""
        return [row[:4] for row in self._rows]

    def set_rows(self, rows) -> None:
        """``(name, Dice)`` or ``(name, Dice, value text, tooltip, muted)`` per row."""
        self._rows = []
        for row in rows:
            name, dice = str(row[0]), max(0.0, min(1.0, float(row[1])))
            text = row[2] if len(row) > 2 else f"{row[1]:.2f}"
            tooltip = row[3] if len(row) > 3 and row[3] else f"{name}\nDice {row[1]:.3f}"
            self._rows.append((name, dice, text, tooltip, bool(row[4]) if len(row) > 4 else False))
        self.setFixedHeight(len(self._rows) * self.ROW_HEIGHT)
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(200, len(self._rows) * self.ROW_HEIGHT)

    def _columns(self) -> tuple[int, int]:
        """Width of the name and the value column."""
        metrics = self.fontMetrics()
        value = max((metrics.horizontalAdvance(row[2]) for row in self._rows), default=0) + 4
        return min(self.NAME_WIDTH, max(60, self.width() * 45 // 100)), value

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        metrics = self.fontMetrics()
        name_width, value_width = self._columns()
        bar_left = name_width + 10
        bar_width = max(20, self.width() - bar_left - value_width - 10)
        first = max(0, event.rect().top() // self.ROW_HEIGHT)
        last = min(len(self._rows), event.rect().bottom() // self.ROW_HEIGHT + 1)
        text, muted = self.palette().color(self.foregroundRole()), QColor(style.MUTED)
        for index in range(first, last):
            name, dice, value, _tooltip, gray = self._rows[index]
            top = index * self.ROW_HEIGHT
            painter.setPen(muted if gray else text)
            painter.drawText(QRect(0, top, name_width, self.ROW_HEIGHT), Qt.AlignmentFlag.AlignVCenter,
                             metrics.elidedText(name, Qt.TextElideMode.ElideRight, name_width))
            painter.setPen(Qt.PenStyle.NoPen)
            bar = QRect(bar_left, top + (self.ROW_HEIGHT - 6) // 2, bar_width, 6)
            painter.setBrush(QColor("#edf1f6"))
            painter.drawRoundedRect(bar, 3, 3)
            painter.setBrush(QColor("#94a3b8" if gray else style.BLUE))
            painter.drawRoundedRect(QRect(bar.x(), bar.y(), int(bar_width * dice), 6), 3, 3)
            painter.setPen(muted)
            painter.drawText(QRect(bar_left + bar_width + 10, top, value_width, self.ROW_HEIGHT),
                             Qt.AlignmentFlag.AlignVCenter, value)
        painter.end()

    def event(self, event) -> bool:
        if event.type() == QEvent.Type.ToolTip:
            index = event.pos().y() // self.ROW_HEIGHT
            if 0 <= index < len(self._rows):
                QToolTip.showText(event.globalPos(), self._rows[index][3], self)
            else:
                QToolTip.hideText()
            return True
        return super().event(event)


def _howto() -> HowToCard:
    return HowToCard("How to validate", [
        "Each fold predicts the cases it did <b>not</b> train on, so the Dice shows how well the model generalises.",
        ("<b>Mean Dice</b> per model is the mean over its finished folds; compare models with the same folds for a fair "
        "comparison."),
        (f"Click a fold to see its results and cases. Always inspect predictions with {icon_html('search')}: metrics alone can hide "
        "errors."),
    ], [("purple", (f"{icon_html('edit')} Annotation Workflow (optional): if another iteration is needed, start it with "
                    "<b>＋ New iteration</b> on the overview, annotate more and train again. During annotation, focus "
                    "on the weak classes."))], "Validate")


class ValidateStage(StagePage):
    key, title, guide = "validate", "Validate", ("5-validate.md", "")
    folder = "results"

    def __init__(self, context: AppContext, parent=None) -> None:
        super().__init__(context, _howto(), parent)
        self._annotation_tip = self.howto._tips.itemAt(0).widget()
        self._fold: FoldInfo | None = None
        self._model: ModelInfo | None = None
        self._summary: dict | None = None
        self._cases: tuple[ValidationCase, ...] = ()
        self._case: ValidationCase | None = None
        self._case_sort: tuple[int, bool] = (0, False)  # (column, descending): case name by default
        self._fold_buttons = QButtonGroup(self)
        self._fold_buttons.setExclusive(False)  # a second click on the open fold goes back to the model
        models_card = Card()
        head = models_card.add_header("Models")
        self._models_hint = header_note("click a fold for its results, again to go back")
        head.addWidget(self._models_hint, 1)
        self._models = models_table("Folds")
        self._models.cellClicked.connect(lambda row, _column: self._choose_model(self._model_rows[row]))
        self._model_rows: list[ModelInfo] = []
        models_card.body.addWidget(self._models)
        self._empty = empty_state("No validation results yet. They appear here as soon as the first fold has finished "
                                  "training.")
        models_card.body.addWidget(self._empty)
        self.center.addWidget(models_card)
        self._cases_card = Card()
        cases_head = self._cases_card.add_header()
        self._cases_title = label("", name="h3")
        cases_head.addWidget(self._cases_title)
        self._cases_note = header_note()
        cases_head.addWidget(self._cases_note, 1)
        # The weakest class (Dice first, then the name) takes the free width; long names end with "…".
        self._cases_table = make_table(["Case", "Dice", "Weakest class", ""], stretch=2)
        self._cases_table.cellClicked.connect(lambda row, column: column != 3 and self._select_case(row))
        self._cases_table.setItemDelegateForColumn(3, IconButtonsDelegate(
            self._cases_table, lambda _row: [("search", "Open in Napari · prediction and label", False)],
            lambda row, _number: self._inspect(self._cases[row])))
        header = self._cases_table.horizontalHeader()
        for column, width in ((0, 160), (1, 70), (3, 52)):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
            self._cases_table.setColumnWidth(column, width)
        # Click "Case" or "Dice" to sort (again to reverse); the rows are re-filled, not sorted by Qt.
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(True)
        header.setSortIndicator(0, Qt.SortOrder.AscendingOrder)
        header.sectionClicked.connect(self._sort_cases)
        self._cases_table.horizontalHeaderItem(1).setToolTip("Click to sort by Dice (weakest first), again to reverse")
        self._cases_table.horizontalHeaderItem(0).setToolTip("Click to sort by case name")
        self._cases_stack = QStackedWidget()
        self._cases_stack.addWidget(self._cases_table)
        self._cases_stack.addWidget(empty_state("Select a fold above to see its validation cases."))
        self._cases_card.body.addWidget(self._cases_stack, 1)
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
        self._panel_title = label("summary.json", name="h3")  # "Model" at model level (no single file there)
        head.addWidget(self._panel_title, 0, Qt.AlignmentFlag.AlignTop)
        # Wraps instead of being cut off when the trainer or case name is long or the window is narrow.
        self._panel_subject = label("", muted=True, mono=True, size=11, wrap=True, selectable=True)
        self._panel_subject.setMinimumWidth(1)
        head.addWidget(self._panel_subject, 1)
        self._close_case = button("✕", ghost=True, small=True, clicked=lambda: self._select_case(-1))
        head.addWidget(self._close_case, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(head)
        mean = QHBoxLayout()
        mean.setSpacing(12)
        self._foreground = label("—", name="bigNumber")
        mean.addWidget(self._foreground)
        mean_text = QVBoxLayout()
        mean_text.setSpacing(0)
        self._mean_title = section_label("Dice (foreground mean)")
        mean_text.addWidget(self._mean_title)
        self._foreground_note = label("Dice", muted=True, size=12, wrap=True)
        self._foreground_note.setMinimumWidth(1)
        mean_text.addWidget(self._foreground_note)
        mean.addLayout(mean_text, 1)
        layout.addLayout(mean)
        self._fold_section = QWidget()
        fold_section = QVBoxLayout(self._fold_section)
        fold_section.setContentsMargins(0, 0, 0, 0)
        fold_section.setSpacing(4)
        fold_section.addWidget(label("Dice per fold", name="h3"))
        self._fold_bars = DiceBars()
        fold_section.addWidget(self._fold_bars)
        layout.addWidget(self._fold_section)
        dice_head = QHBoxLayout()
        dice_head.addWidget(label("Dice per class", name="h3"))
        dice_head.addStretch()
        # Many classes: the weakest first; "Show all" lists every class in label order.
        self._show_all = button("Show all", ghost=True, small=True, clicked=self._toggle_all_classes)
        dice_head.addWidget(self._show_all)
        layout.addLayout(dice_head)
        self._dice_note = label("", muted=True, size=11, wrap=True)
        self._dice_note.setMinimumWidth(1)
        layout.addWidget(self._dice_note)
        self._dice_bars = DiceBars()
        self._class_rows: list[tuple[str, float]] = []
        self._dice_base_note = ""
        self._all_classes = False
        layout.addWidget(self._dice_bars)
        # The real summary.json (fold) or its entry (case); hidden at model level, where the bars say it all.
        self._details_label = section_label("Details")
        layout.addWidget(self._details_label)
        self._details = QPlainTextEdit()
        self._details.setObjectName("codeView")
        self._details.setReadOnly(True)
        self._details.setMinimumHeight(160)
        layout.addWidget(self._details, 1)
        layout.addStretch()  # keeps the summary at the top while the details are hidden
        # In a scroll area: with many classes ("Show all") or a low window the details stay reachable.
        return scroll(panel)

    def _set_details_visible(self, visible: bool) -> None:
        self._panel_title.setText("summary.json" if visible else "Model")
        self._details_label.setVisible(visible)
        self._details.setVisible(visible)

    # ------------------------------------------------------------ state

    def set_state(self, state: DatasetState) -> None:
        super().set_state(state)
        annotation = self.context.settings.annotation_workflow
        self._annotation_tip.setVisible(annotation)
        models = [model for model in state.models if any(fold.has_validation for fold in model.folds)]
        self._models_hint.setText("click a fold for its results, again or the model to go back" if len(models) > 1
                                  else "click a fold for its results, again to go back")
        self._models.setVisible(bool(models))
        self._empty.setVisible(not models)
        self._cases_card.setVisible(bool(models))
        self.right.setCurrentIndex(0 if models else 1)
        for button_widget in self._fold_buttons.buttons():
            self._fold_buttons.removeButton(button_widget)
        self._models.setRowCount(len(models))
        self._model_rows = models
        self._button_of: dict = {}
        keep_button = None
        for row, model in enumerate(models):
            self._models.setCellWidget(row, 0, model_name_cell(model))
            self._models.setItem(row, 1, mean_dice_item(model))
            buttons = []
            for fold in model.folds:
                if not fold.has_validation:
                    continue
                fold_button = QPushButton(fold.name)  # the Dice per fold is shown in the right panel
                fold_button.setObjectName("foldButton")
                fold_button.setFixedSize(*FOLD_SIZE)
                fold_button.setCheckable(True)
                fold_button.setCursor(Qt.CursorShape.PointingHandCursor)
                tip = f"Fold {fold.name} · Dice {fold.dice:.3f}" if fold.dice is not None else f"Fold {fold.name}"
                if fold.name == "all":
                    tip += "\n" + FOLD_ALL_NOTE
                fold_button.setToolTip(tip)
                fold_button.clicked.connect(lambda _=False, m=model, f=fold: self._fold_clicked(m, f))
                self._fold_buttons.addButton(fold_button)
                self._button_of[fold.path] = fold_button
                buttons.append(fold_button)
                if self._fold is not None and fold.path == self._fold.path:
                    keep_button = (model, fold)
            self._models.setCellWidget(row, 2, fold_cells(buttons))
        size_models_table(self._models, models)
        if keep_button:
            self._show_fold(*keep_button)
        elif models:
            previous = self.context.model_label or (self._model.label if self._model else None)
            self._show_model(next((model for model in models if model.label == previous), models[0]))

    def header_badges(self, state: DatasetState) -> list:
        validated = sum(1 for model in state.models if any(fold.has_validation for fold in model.folds))
        items = [badge(f"✓ {validated} model{'s' if validated != 1 else ''} with validation results", "green")
                 if validated else badge("no validation results yet", "gray")]
        # finished cross-validation folds without validation/summary.json (e.g. stopped before the validation step)
        missing = sum(1 for model in state.models for fold in model.folds
                      if fold.finished and fold.name != "all" and not fold.has_validation)
        if missing:
            items.append(badge(f"{missing} finished fold{'s' if missing != 1 else ''} without validation results",
                               "gray", "These folds have a final checkpoint but no validation/summary.json, e.g. "
                               "because the run ended before the validation step or the folder was reduced. "
                               "nnUNetv2_train … --val runs only the validation."))
        return items

    def _check_only(self, fold: FoldInfo | None) -> None:
        for path, fold_button in self._button_of.items():
            fold_button.setChecked(fold is not None and path == fold.path)

    def _choose_model(self, model: ModelInfo) -> None:
        self.context.model_label = model.label
        self._show_model(model)

    def follow_model(self, label: str) -> None:
        if self._model is not None and self._model.label == label:
            return
        model = next((item for item in self._model_rows if item.label == label), None)
        if model is not None:
            self._show_model(model)

    def _fold_clicked(self, model: ModelInfo, fold: FoldInfo) -> None:
        self.context.model_label = model.label
        if self._fold is not None and self._fold.path == fold.path:
            self._show_model(model)  # second click on the open fold: back to the model overview
        else:
            self._show_fold(model, fold)

    def _show_model(self, model: ModelInfo) -> None:
        """Model level: mean Dice, Dice per fold and the per-class Dice averaged over the folds."""
        self._model, self._fold, self._case, self._summary, self._cases = model, None, None, None, ()
        self._check_only(None)
        row = self._model_rows.index(model) if model in self._model_rows else -1
        self._models.selectRow(row) if row >= 0 else self._models.clearSelection()
        self._cases_title.setText("Cases")
        self._cases_note.setText("")
        self._cases_stack.setCurrentIndex(1)
        self._close_case.setVisible(False)
        self._fold_section.setVisible(True)
        mean = model.mean_dice
        self._foreground.setText(f"{mean:.2f}" if mean is not None else "—")
        self._mean_title.setText("MEAN DICE")
        self._foreground_note.setText("over the folds · fold all excluded")
        self._panel_subject.setText(wrappable(f"{model.trainer} · {model.plans} · {model.configuration}"))
        folds = [fold for fold in model.folds if fold.has_validation and fold.dice is not None]
        self._fold_bars.set_rows([
            (f"fold {fold.name}" + (" · trained on these cases" if fold.name == "all" else ""), fold.dice,
             f"{fold.dice:.2f}",
             f"Fold {fold.name} · Dice {fold.dice:.3f}" + (f"\n{FOLD_ALL_NOTE}" if fold.name == "all" else ""),
             fold.name == "all") for fold in folds])
        self._set_class_rows([], "mean over the folds")
        self._set_details_visible(False)
        paths = [fold.path for fold in folds if fold.name != "all"]
        self.context.tasks.run("validate-model", lambda: [json_object(path / "validation" / "summary.json")
                                                           for path in paths],
                               lambda summaries: self._apply_model(model, summaries))

    def _apply_model(self, model: ModelInfo, summaries: object) -> None:
        if self._model is not model or self._fold is not None or self.state is None:
            return
        names = self.state.names
        per_class: dict[str, list[float]] = {}
        for summary in summaries if isinstance(summaries, list) else []:
            for name, dice in mean_class_dice(summary, names):
                per_class.setdefault(name, []).append(dice)
        rows = [(name, sum(values) / len(values)) for name, values in per_class.items()]
        self._set_class_rows(rows, "mean over the folds")

    def _set_class_rows(self, rows: list, note: str) -> None:
        """Dice per class: all classes in label order, or with many classes the weakest until "Show all"."""
        self._class_rows, self._dice_base_note = list(rows), note
        self._show_class_rows()

    def _toggle_all_classes(self) -> None:
        self._all_classes = not self._all_classes
        self._show_class_rows()

    def _show_class_rows(self) -> None:
        rows, note = self._class_rows, self._dice_base_note
        many = len(rows) > SHOW_ALL_FROM
        self._show_all.setVisible(many)
        if many and not self._all_classes:
            shown = sorted(rows, key=lambda row: row[1])[:WEAKEST_CLASSES]
            note = f"{note} · weakest {len(shown)} of {len(rows)}" if note else f"weakest {len(shown)} of {len(rows)}"
            self._show_all.setText(f"Show all {len(rows)}")
            self._show_all.setToolTip("List every class in label order")
        else:
            shown = rows
            self._show_all.setText("Weakest only")
            self._show_all.setToolTip(f"List only the {WEAKEST_CLASSES} weakest classes")
        self._dice_note.setText(note)
        self._dice_note.setVisible(bool(note))
        self._dice_bars.set_rows(shown)

    def _show_fold(self, model: ModelInfo, fold: FoldInfo) -> None:
        self._model, self._fold = model, fold
        self._set_details_visible(True)
        self._details.setPlainText("Loading…")
        self._check_only(fold)
        row = self._model_rows.index(model) if model in self._model_rows else -1
        if row >= 0:
            self._models.selectRow(row)
        self._cases_stack.setCurrentIndex(0)
        self._fold_section.setVisible(False)
        self._mean_title.setText("DICE (FOREGROUND MEAN)")
        self._cases_title.setText(f"Cases · fold {fold.name}")
        path = fold.path / "validation" / "summary.json"
        self.context.tasks.run("validate-summary", lambda: json_object(path), lambda data: self._apply_summary(fold, data))

    def _apply_summary(self, fold: FoldInfo, summary: object) -> None:
        if fold is not self._fold or self.state is None:
            return
        self._summary = summary if isinstance(summary, dict) else None
        names = self.state.names
        self._cases = self._sorted_cases(validation_cases(self._summary, names))
        self._cases_note.setText(f"{len(self._cases)} validation cases · click a case for its metrics")
        self._fill_cases()
        self._select_case(-1)

    def _sorted_cases(self, cases) -> tuple[ValidationCase, ...]:
        column, descending = self._case_sort
        if column == 1:  # Dice; cases without a Dice stay at the end
            known = sorted((case for case in cases if case.mean_dice is not None), key=lambda case: case.mean_dice,
                           reverse=descending)
            return (*known, *(case for case in cases if case.mean_dice is None))
        return tuple(sorted(cases, key=lambda case: case.case_id.casefold(), reverse=descending))

    def _sort_cases(self, column: int) -> None:
        header = self._cases_table.horizontalHeader()
        if column not in (0, 1):
            header.setSortIndicator(self._case_sort[0], Qt.SortOrder.DescendingOrder if self._case_sort[1]
                                    else Qt.SortOrder.AscendingOrder)
            return
        previous_column, descending = self._case_sort
        self._case_sort = (column, not descending if column == previous_column else False)
        header.setSortIndicator(column, Qt.SortOrder.DescendingOrder if self._case_sort[1]
                                else Qt.SortOrder.AscendingOrder)
        selected = self._case
        self._cases = self._sorted_cases(self._cases)
        self._fill_cases()
        if selected is not None and selected in self._cases:
            row = self._cases.index(selected)
            self._cases_table.selectRow(row)
            self._select_case(row)

    def _fill_cases(self) -> None:
        self._cases_table.setRowCount(len(self._cases))
        # Case column: the longest case name, measured with the item font (longer names end with "…")
        metrics = QFontMetrics(table_item("x", mono=True).font())
        longest = max((metrics.horizontalAdvance(case.case_id) for case in self._cases), default=60)
        self._cases_table.setColumnWidth(0, max(90, min(longest + 24, 240)))
        for row, case in enumerate(self._cases):
            self._cases_table.setItem(row, 0, table_item(case.case_id, mono=True))
            self._cases_table.setItem(row, 1, table_item(f"{case.mean_dice:.2f}" if case.mean_dice is not None else "—",
                                                         bold=True))
            # the Dice first: a long class name is cut at the end of the column, never the value
            weakest = f"{case.weakest[1]:.2f} · {case.weakest[0]}" if case.weakest else "—"
            self._cases_table.setItem(row, 2, table_item(weakest, muted=True))

    def _select_case(self, row: int) -> None:
        if self._fold is None:
            return
        self._case = self._cases[row] if 0 <= row < len(self._cases) else None
        if self._case is None:
            self._cases_table.clearSelection()
        self._close_case.setVisible(self._case is not None)
        names = self.state.names if self.state else {}
        if self._case is not None:
            self._panel_subject.setText(wrappable(self._case.case_id))
            note = ""  # the case is the subject above
            self._foreground.setText(f"{self._case.mean_dice:.2f}" if self._case.mean_dice is not None else "—")
            self._foreground_note.setText(f"Dice · case {self._case.case_id}")
            rows = []
            for key, values in self._case.metrics.items():
                dice = finite_number(values.get("Dice") if isinstance(values, dict) else None)
                if dice is not None:
                    rows.append((class_name(key, names), dice))
            entry = {"prediction_file": self._case.prediction_file, "reference_file": self._case.reference_file,
                     "metrics": self._case.metrics}
            self._details.setPlainText(json.dumps(entry, indent=2))
        else:
            fold = self._fold.name if self._fold else ""
            self._panel_subject.setText(f"fold {fold} · foreground_mean and mean")
            note = f"fold {fold} · mean"
            foreground = summary_foreground_dice(self._summary)  # also when foreground_mean is null or a list
            self._foreground.setText(f"{foreground:.2f}" if foreground is not None else "—")
            self._foreground_note.setText(f"Dice · fold {fold}")
            rows = mean_class_dice(self._summary, names)
            summary = self._summary or {}
            self._details.setPlainText(json.dumps({"foreground_mean": summary.get("foreground_mean"),
                                                   "mean": summary.get("mean")}, indent=2))
        self._set_class_rows(rows, note)

    def _inspect(self, case: ValidationCase) -> None:
        if self._fold is None:
            return
        prediction = self._fold.path / "validation" / Path(case.prediction_file).name
        self.context.inspect_validation(prediction, self.state.raw_path if self.state else None)
