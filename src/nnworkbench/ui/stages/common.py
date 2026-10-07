"""Shared frame of a stage page: how-to on the left, main content in the middle, a panel on the right."""

from __future__ import annotations

import shlex

from qtpy.QtCore import Signal
from qtpy.QtGui import QGuiApplication
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QPlainTextEdit,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..context import AppContext
from ..dataset_state import DatasetState
from ..widgets import Card, HowToCard, badge, button, clear_layout, label


def format_command(command: str) -> str:
    """Show one option per line (shell line continuation), so commands never break inside a word."""
    parts = shlex.split(command)
    lines: list[str] = []
    current: list[str] = []
    for part in parts:
        if part.startswith("-") and current and not _is_number(part):
            lines.append(shlex.join(current))
            current = []
        current.append(part)
    if current:
        lines.append(shlex.join(current))
    return " \\\n  ".join(lines)


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


class CommandBox(QPlainTextEdit):
    """Read-only command preview."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("commandView")
        self.setReadOnly(True)
        self.setFixedHeight(120)
        self._command = ""

    def set_command(self, command: str) -> None:
        from nnworkbench.services.commands import wrapped_command

        self._command = command
        self.setPlainText(format_command(wrapped_command(command)) if command else "")

    def command(self) -> str:
        return self._command

    def copy(self) -> None:
        from nnworkbench.services.commands import wrapped_command

        if self._command:
            QGuiApplication.clipboard().setText(wrapped_command(self._command))


def arguments_field() -> QLineEdit:
    field = QLineEdit()
    field.setProperty("mono", True)
    field.setPlaceholderText("Optional, e.g. --npz")
    return field


class StagePage(QWidget):
    """Base class: subclasses fill ``self.center`` and ``self.right`` and implement ``set_state``."""

    key = ""
    title = ""
    guide_stage = 0
    folder = "raw"  # nnU-Net folder this stage works in (raw, preprocessed, results)
    navigate = Signal(str)
    guide_requested = Signal(int)
    reload_requested = Signal()

    def __init__(self, context: AppContext, howto: HowToCard, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("page")
        self.context = context
        self.state: DatasetState | None = None
        row = QHBoxLayout(self)
        row.setContentsMargins(24, 16, 24, 16)
        row.setSpacing(18)
        self.howto = howto
        self.howto.guide_requested.connect(lambda: self.guide_requested.emit(self.guide_stage))
        row.addWidget(self.howto)
        center = QWidget()
        self.center = QVBoxLayout(center)
        self.center.setContentsMargins(0, 0, 0, 0)
        self.center.setSpacing(14)
        row.addWidget(center, 1)
        self._build_header()
        self.right = QStackedWidget()
        self.right.setFixedWidth(410)
        row.addWidget(self.right)
        context.jobs.busy_changed.connect(lambda _busy: self.update_actions())
        context.annotation_changed.connect(lambda _enabled: self.annotation_changed())

    def _build_header(self) -> None:
        """Same header on every stage: dataset name, Refresh and Open in the stage's nnU-Net folder."""
        card = Card()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 14, 20, 14)
        layout.setSpacing(10)
        card.body.addLayout(layout)
        top = QHBoxLayout()
        names = QVBoxLayout()
        names.setSpacing(2)
        self._name = label("", name="h2")
        self._sub = label("", muted=True, size=12, rich=True, wrap=True)
        names.addWidget(self._name)
        names.addWidget(self._sub)
        top.addLayout(names, 1)
        top.addWidget(button("⟳ Refresh", tooltip="Reload the dataset files from disk",
                             clicked=self.reload_requested.emit))
        self._open = button("📁 Open", tooltip=f"Open the dataset folder in nnUNet_{self.folder}", clicked=self._open_folder)
        top.addWidget(self._open)
        self.header_actions = QHBoxLayout()
        self.header_actions.setSpacing(8)
        top.addLayout(self.header_actions)
        layout.addLayout(top)
        self._badges = QHBoxLayout()
        self._badges.setSpacing(6)
        layout.addLayout(self._badges)
        self.center.addWidget(card)

    def stage_folder(self):
        if self.state is None:
            return None
        return getattr(self.state.record, f"{self.folder}_path")

    def _open_folder(self) -> None:
        self.context.open_folder(self.stage_folder())

    def header_subtitle(self, state: DatasetState) -> str:
        channels = state.raw.channels
        cases = len(state.raw.cases)
        if not cases and isinstance(state.manifest, dict):
            cases = state.manifest.get("numTraining", 0)
        classes = len([name for name in state.names.values() if name.strip().casefold() != "ignore"])
        return (f"{cases} training case{'s' if cases != 1 else ''} · {channels} channel{'s' if channels != 1 else ''} · "
                f"{classes} class{'es' if classes != 1 else ''}")

    def header_badges(self, state: DatasetState) -> list:
        """Badges below the name; the label mode is shown on every stage."""
        ignore = state.ignore_id
        if ignore is not None:
            return [badge(f"Sparse labels · ignore = {ignore}", "purple", "dataset.json has an ignore label")]
        return [badge("Dense labels · no ignore", "gray", "dataset.json has no ignore label")]

    def show_header_loading(self, name: str) -> None:
        self._name.setText(name)
        self._sub.setText("Loading files…")
        clear_layout(self._badges)

    def show_loading(self) -> None:
        """Called while the dataset is read in the background; stages may override it."""

    def set_state(self, state: DatasetState) -> None:
        self.state = state
        self._name.setText(state.record.directory_name)
        self._sub.setText(self.header_subtitle(state))
        path = self.stage_folder()
        self._open.setEnabled(path is not None and path.is_dir())
        self._open.setToolTip(f"Open the dataset folder in nnUNet_{self.folder}" if self._open.isEnabled()
                              else f"This dataset has no folder in nnUNet_{self.folder}.")
        clear_layout(self._badges)
        for item in self.header_badges(state):
            self._badges.addWidget(item)
        self._badges.addStretch()

    def update_actions(self) -> None:
        """Enable or disable run buttons (one job at a time, missing folders)."""

    def annotation_changed(self) -> None:
        if self.state is not None:
            self.set_state(self.state)

    def busy_or_missing(self, *roots: str) -> str:
        """Return why a run button is disabled, or an empty string when it can run."""
        if self.context.jobs.busy:
            return self.context.jobs.busy_reason()
        missing = self.context.local_roots_available(*roots)
        if missing:
            return f"{', '.join(missing)} is not available on this computer. Copy command still works."
        return ""


FOLD_SIZE = (48, 26)  # fold chips (Train) and fold buttons (Validate) share one width
FOLD_BUTTON_HEIGHT = 40
MODEL_ROW_HEIGHT = 56  # Validate buttons show the fold and its Dice on two lines


def models_table(folds_header: str):
    """The models table used on Train and Validate: same columns, widths and row heights."""
    from ..widgets import make_table

    table = make_table(["Model", "Mean Dice", folds_header], stretch=0, row_height=MODEL_ROW_HEIGHT)
    header = table.horizontalHeader()
    from qtpy.QtWidgets import QHeaderView

    header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
    header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
    table.setColumnWidth(1, 110)
    return table


def size_models_table(table, models) -> None:
    """Folds column fits the model with the most folds; height fits up to five rows."""
    most = max((len(model.folds) for model in models), default=1)
    table.setColumnWidth(2, max(150, 24 + most * (FOLD_SIZE[0] + 6)))
    table.setFixedHeight(table.horizontalHeader().sizeHint().height() + 4 + MODEL_ROW_HEIGHT * min(max(len(models), 1), 5))


def mean_dice_item(model):
    from ..widgets import table_item

    mean = model.mean_dice
    count = sum(1 for fold in model.folds if fold.name != "all" and fold.finished and fold.dice is not None)
    item = table_item(f"{mean:.3f}" if mean is not None else "—", bold=mean is not None, muted=mean is None)
    item.setToolTip(f"mean over {count} finished fold{'s' if count != 1 else ''} (fold all is not included)"
                    if mean is not None else "no finished fold with validation results (fold all is not included)")
    return item


def empty_panel(text: str) -> QWidget:
    holder = QWidget()
    column = QVBoxLayout(holder)
    column.addStretch()
    message = label(text, muted=True, wrap=True)
    column.addWidget(message)
    column.addStretch()
    return holder
