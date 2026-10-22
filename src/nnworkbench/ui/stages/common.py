"""Shared frame of a stage page: how-to on the left, main content in the middle, a panel on the right."""

from __future__ import annotations

import html
import re

from qtpy.QtCore import QEvent, QObject, QSize, Qt, QTimer, Signal
from qtpy.QtGui import QGuiApplication, QTextOption
from qtpy.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from nnworkbench.services.dataset_json import REGIONS_NOTE

from ..context import AppContext
from ..dataset_state import DatasetState
from ..widgets import Card, ElidedLabel, FlowLayout, HowToCard, badge, button, clear_layout, label


def format_command(command: str) -> str:
    """The wrapped command on one line, quoted for bash/zsh and PowerShell alike (no line continuation: the two
    shells have none in common). The command box wraps it on screen; a selected line is the whole command."""
    from nnworkbench.services.commands import wrapped_command

    return wrapped_command(command)


class CommandBox(QPlainTextEdit):
    """Read-only command preview."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("commandView")
        self.setReadOnly(True)
        # one command per line, wrapped on screen only (no line continuation): break anywhere in long paths
        self.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.setFixedHeight(120)
        self._command = ""
        self._commands: list[str] = []
        self._copy_buttons: list = []

    def set_command(self, command: str | list[str]) -> None:
        """One command, or several that run one after another (shown and copied one per line)."""
        from nnworkbench.services.commands import join_commands

        self._commands = [command] if isinstance(command, str) else list(command)
        self._commands = [text for text in self._commands if text]
        self._command = join_commands(self._commands)
        self.setPlainText(join_commands([format_command(text) for text in self._commands]))
        self._update_copy()

    def set_error(self, message: str) -> None:
        """No command (e.g. the additional arguments cannot be split): show why; Copy has nothing to copy."""
        self._commands, self._command = [], ""
        self.setPlainText(message)
        self._update_copy()

    def command(self) -> str:
        return self._command

    def copy_button(self):
        """The "Copy command" button; it is disabled while there is no command."""
        widget = button("Copy command", small=True, clicked=self.copy)
        self._copy_buttons.append(widget)
        self._update_copy()
        return widget

    def _update_copy(self) -> None:
        for widget in self._copy_buttons:
            widget.setEnabled(bool(self._command))

    def copy(self) -> None:
        from nnworkbench.services.commands import join_commands, wrapped_command

        if self._command:
            QGuiApplication.clipboard().setText(join_commands([wrapped_command(text) for text in self._commands]))


def arguments_error(text: str) -> str:
    """Why the additional arguments cannot be used (an unclosed quote, a trailing backslash), or ""."""
    from nnworkbench.services.commands import split_arguments

    try:
        arguments = split_arguments(text)
    except ValueError as error:
        return (f"Additional arguments cannot be read ({error}): close the quote or remove the trailing "
                "backslash.")
    if any(argument.startswith(("nnworkbench-run", "nnUNetv2_")) for argument in arguments):
        return "Additional arguments hold a whole command: enter only the extra options, e.g. --npz."
    # dashes starting an option, quotes around an argument, non-breaking spaces (a path may contain ’ or –)
    typographic = sorted({character for argument in arguments for character in
                          {argument[:1]} & set("–—‑“”„‘’‚") | {argument[-1:]} & set("“”‘’") | set(argument) & {"\u00a0"}})
    if typographic:
        shown = ", ".join("non-breaking space" if character == "\u00a0" else character for character in typographic)
        return (f"Additional arguments contain {shown} (copied from a web page or document?): type the quotes, "
                "dashes and spaces again on the keyboard.")
    return ""


def arguments_field() -> QLineEdit:
    field = QLineEdit()
    field.setProperty("mono", True)
    field.setPlaceholderText("Optional, e.g. --npz")
    return field


class StagePage(QWidget):
    """Base class: subclasses fill ``self.center`` and ``self.right`` and implement ``set_state``."""

    key = ""
    title = ""
    guide: tuple[str, str] = ("README.md", "")  # guide file and heading anchor opened by "More in the guide"
    folder = "raw"  # nnU-Net folder this stage works in (raw, preprocessed, results)
    navigate = Signal(str)
    guide_requested = Signal(str, str)
    reload_requested = Signal()

    def __init__(self, context: AppContext, howto: HowToCard, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("page")
        self.context = context
        self.state: DatasetState | None = None
        row = QHBoxLayout(self)
        row.setContentsMargins(24, 16, 24, 16)
        row.setSpacing(18)
        self._row = row
        self.howto = howto
        self.howto.guide_requested.connect(lambda: self.guide_requested.emit(*self.guide))
        from ..widgets import scroll

        self._howto_area = scroll(self.howto)
        self._howto_area.setFixedWidth(HOWTO_WIDTH[1] + 10)  # card + room for a scrollbar (console expanded)
        row.addWidget(self._howto_area)
        # Collapsed: a slim strip that brings the how-to back. The choice is shared by all stages and remembered.
        self._howto_strip = button("»", tooltip="Show the how-to", clicked=lambda: context.set_howto_expanded(True))
        self._howto_strip.setFixedWidth(34)
        self._howto_strip.setStyleSheet("padding: 0; font-size: 18px; color: #64748b;")
        self._howto_strip.setSizePolicy(self._howto_strip.sizePolicy().horizontalPolicy(),
                                        QSizePolicy.Policy.Expanding)
        row.addWidget(self._howto_strip)
        self.howto.collapse_requested.connect(lambda: context.set_howto_expanded(False))
        context.howto_changed.connect(self._show_howto)
        self._show_howto(context.howto_expanded)
        center = QWidget()
        self.center = QVBoxLayout(center)
        self.center.setContentsMargins(0, 0, 0, 0)
        self.center.setSpacing(14)
        row.addWidget(center, 1)
        self._build_header()
        self.right = QStackedWidget()
        self.right.setFixedWidth(RIGHT_WIDTH[1])
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
        # The name next to the buttons, or (narrow window, long name) the buttons in a row below it; the name is
        # shortened with "…" only when even its own row is too narrow. The summary line takes the full card width,
        # so it does not wrap word by word. Buttons keep their text (fixed horizontal size).
        self._top = QGridLayout()
        self._top.setHorizontalSpacing(10)
        self._top.setVerticalSpacing(8)
        self._name = ElidedLabel()
        self._name.setObjectName("h2")
        self._name.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._header_buttons = QWidget()
        buttons = QHBoxLayout(self._header_buttons)
        buttons.setContentsMargins(0, 0, 0, 0)
        refresh = button("⟳ Refresh", tooltip="Reload the dataset files from disk", clicked=self.reload_requested.emit)
        buttons.addWidget(refresh)
        self._open = button("Open", icon="folder", tooltip=f"Open folder · nnUNet_{self.folder}", clicked=self._open_folder)
        buttons.addWidget(self._open)
        self.header_actions = QHBoxLayout()
        self.header_actions.setSpacing(8)
        buttons.addLayout(self.header_actions)
        self._header_buttons.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self._top.addWidget(self._name, 0, 0)
        self._top.addWidget(self._header_buttons, 0, 1)
        self._top.setColumnStretch(0, 1)
        self._buttons_below = False
        layout.addLayout(self._top)
        self._header_card = card
        card.installEventFilter(self)  # re-place the buttons whenever the card changes its width
        self._sub = label("", muted=True, size=12, rich=True, wrap=True)
        self._sub.setMinimumWidth(1)
        layout.addWidget(self._sub)
        # Wraps into more lines when the badges do not fit next to each other (instead of cutting them off).
        self._badges = FlowLayout(spacing=6)
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
        classes = len([name for name in state.names.values() if name != IGNORE_KEY])
        return (f"{cases} training case{'s' if cases != 1 else ''} · {channels} channel{'s' if channels != 1 else ''} · "
                f"{classes} class{'es' if classes != 1 else ''}")

    def header_badges(self, state: DatasetState) -> list:
        """Badges below the name: what exists in this stage and whether a new run is possible."""
        return []

    def show_header_loading(self, name: str) -> None:
        """Another dataset is opened: until it is loaded nothing runs for the previous one (an open run form
        would still hold its ID)."""
        self.state = None
        run_dialog = getattr(self, "_run_dialog", None)
        if isinstance(run_dialog, RunDialog):
            run_dialog.close()
        self._name.setText(name)
        self._sub.setText("Loading files…")
        QTimer.singleShot(0, self._place_header_buttons)
        clear_layout(self._badges)

    def show_load_error(self, message: str) -> None:
        """The dataset files could not be read (or this stage failed to show them): say so in the header, with
        the way to try again, instead of "Loading files…"."""
        from ..style import RED

        self._sub.setText(f"<span style='color:{RED}'>{html.escape(message)}</span> Fix the file or folder, "
                          "then click ⟳ Refresh to try again.")

    def show_loading(self) -> None:
        """Called while the dataset is read in the background; stages may override it."""

    def confirm_leave(self) -> bool:
        """Before another dataset opens or nnWorkbench quits; stages with editors ask about unsaved changes."""
        return True

    def follow_model(self, label: str) -> None:
        """Select the model chosen on another stage, if this stage lists it (Train, Validate, Predict)."""

    def _place_header_buttons(self) -> None:
        """Buttons next to the name when both fit, otherwise in their own row below it."""
        if not self._header_card.isVisible():
            return
        room = self._header_card.width() - 40  # card margins
        below = self._name.sizeHint().width() + 10 + self._header_buttons.sizeHint().width() > room
        if below != self._buttons_below:
            self._buttons_below = below
            self._top.removeWidget(self._header_buttons)
            if below:
                self._top.addWidget(self._header_buttons, 1, 0, 1, 2, Qt.AlignmentFlag.AlignLeft)
            else:
                self._top.addWidget(self._header_buttons, 0, 1)

    def eventFilter(self, watched, event) -> bool:
        if watched is self._header_card and event.type() == QEvent.Type.Resize:
            self._place_header_buttons()
        return super().eventFilter(watched, event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Towards the minimum window width the how-to, the right panel and the margins narrow, so the tables in
        # the middle keep room for the model names and all folds.
        narrow = self.width() < 1300
        if self._row.spacing() != (12 if narrow else 18):
            self._row.setContentsMargins(12 if narrow else 24, 16, 12 if narrow else 24, 16)
            self._row.setSpacing(12 if narrow else 18)
        for widget, (low, high), extra in ((self.right, RIGHT_WIDTH, 0), (self.howto, HOWTO_WIDTH, 0),
                                           (self._howto_area, HOWTO_WIDTH, 10)):
            width = max(low, min(high, low + (self.width() - 1200) * (high - low) // 240)) + extra
            if widget.width() != width:
                widget.setFixedWidth(width)

    def minimumSizeHint(self) -> QSize:
        """The minimum at the narrow widths: a hidden stage keeps the widths of the size it was last shown at, and
        its wider minimum would hold the window wider than the minimum window width."""
        hint = super().minimumSizeHint()
        extra = self.right.minimumWidth() - RIGHT_WIDTH[0]
        if self._howto_area.isVisibleTo(self):
            extra += self._howto_area.minimumWidth() - HOWTO_WIDTH[0] - 10
        if self._row.spacing() != 12:
            extra += 2 * (24 - 12) + 2 * (18 - 12)
        return QSize(max(0, hint.width() - max(0, extra)), hint.height())

    def _show_howto(self, expanded: bool) -> None:
        self._howto_area.setVisible(expanded)
        self._howto_strip.setVisible(not expanded)

    def set_state(self, state: DatasetState) -> None:
        self.state = state
        self._name.setText(state.record.directory_name)
        self._sub.setText(self.header_subtitle(state))
        path = self.stage_folder()
        self._open.setEnabled(path is not None and path.is_dir())
        self._open.setToolTip(f"Open folder · nnUNet_{self.folder}/{state.record.directory_name}" if self._open.isEnabled()
                              else f"This dataset has no folder in nnUNet_{self.folder}.")
        clear_layout(self._badges)
        for item in self.header_badges(state):
            self._badges.addWidget(item)
        QTimer.singleShot(0, self._place_header_buttons)

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


IGNORE_KEY = "ignore"  # nnU-Net's ignore label: exactly this key in dataset.json "labels"
RIGHT_WIDTH = (340, 410)  # right panel at the minimum window width (1200 px) and from 1440 px on
HOWTO_WIDTH = (245, 280)  # how-to card, likewise
FOLD_SIZE = (38, 26)  # fold chips (Train, Predict) and fold buttons (Validate) share one width
FOLD_SPACING = 3
MODEL_ROW_HEIGHT = 48  # trainer on the first line, plans · configuration on the second
MODEL_MIN_WIDTH = 180  # narrower, model variants could not be told apart; the table scrolls sideways instead


class _ModelColumn(QObject):
    """Gives the Model column of a models table the free width, but never less than MODEL_MIN_WIDTH."""

    def __init__(self, table) -> None:
        super().__init__(table)
        self._table = table
        table.viewport().installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.Resize:
            self.fit()
        return False

    def fit(self) -> None:
        table = self._table
        others = sum(table.columnWidth(column) for column in range(1, table.columnCount()))
        width = max(MODEL_MIN_WIDTH, table.viewport().width() - others)
        if table.columnWidth(0) != width:
            table.setColumnWidth(0, width)


def models_table(folds_header: str):
    """The models table used on Train, Validate and Predict: same columns, widths and row heights."""
    from qtpy.QtWidgets import QAbstractScrollArea, QHeaderView

    from ..widgets import make_table

    table = make_table(["Model", "Mean Dice", folds_header], stretch=None, row_height=MODEL_ROW_HEIGHT)
    header = table.horizontalHeader()
    header.setStretchLastSection(False)
    for column in range(3):
        header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
    table.setColumnWidth(1, 84)
    # A narrow window: scroll sideways to the last folds instead of squeezing the model names.
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    # Height: all rows up to five (then it scrolls), and it may shrink to one row in a low window, so the
    # cards below never overlap it.
    table.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
    table._model_column = _ModelColumn(table)
    return table


def size_models_table(table, models) -> None:
    """Folds column fits the model with the most folds; height fits up to five rows."""
    most = max((len(model.folds) for model in models), default=1)
    # the cell widget is 16 px narrower than its column (item margins); it needs 12 + its chips and spacing
    table.setColumnWidth(2, max(120, 28 + most * (FOLD_SIZE[0] + FOLD_SPACING)))
    table._model_column.fit()
    header = table.horizontalHeader().sizeHint().height() + 4
    rows = min(max(len(models), 1), 5)
    bar = table.horizontalScrollBar().sizeHint().height()
    table.setMinimumHeight(header + MODEL_ROW_HEIGHT + bar)
    table.setMaximumHeight(header + MODEL_ROW_HEIGHT * rows + bar)


FOLD_STATUS = {"done": ("✓ done", "green"), "training": ("● training", "orange"), "stopped": ("■ stopped", "orange"),
           "reduced": ("final checkpoint only", "gray")}


DEVICE_INFO = ("Where nnU-Net runs: <b>GPU</b> an NVIDIA GPU (nnU-Net's default, no -device), <b>MPS</b> the "
               "Apple GPU of a Mac with Apple silicon (-device mps; slower than an NVIDIA GPU), <b>CPU</b> the "
               "processor (-device cpu; works everywhere, but training takes days per fold). The device found on "
               "this computer is preselected; without one, copy the command to a GPU machine or cluster. Details "
               "in Info.")


FOLD_ALL_NOTE = ("Fold all was trained on all cases, so its Dice is measured on training cases. It is not a "
                 "validation result and is not included in the mean Dice.")


def label_mode_badge(state: DatasetState):
    """Sparse or dense labels (or nnU-Net regions); only relevant while working on the raw data."""
    if state.regions:
        return badge("Region-based labels", "blue", REGIONS_NOTE)
    ignore = state.ignore_id
    if ignore is not None:
        return badge(f"Sparse labels · ignore = {ignore}", "purple", "dataset.json has an ignore label")
    return badge("Dense labels · no ignore", "gray", "dataset.json has no ignore label")


def wrappable(text: str) -> str:
    """Like ``breakable`` (line breaks after "_"), and also inside long runs without one (e.g. 40 x in a trainer
    name), so a wrapped title or model name in a side panel never runs past its edge."""
    from ..widgets import breakable

    return re.sub(r"[^\s\u200b]{16}(?=[^\s\u200b])", lambda match: match.group(0) + "\u200b", breakable(text))


def header_note(text: str = "", *, mono: bool = False) -> ElidedLabel:
    """A muted hint or path in a card header: shortened with "…" (full text on hover) instead of making the page
    wider than the window. Add it with stretch 1."""
    note = ElidedLabel(text)
    note.setProperty("muted", True)
    if mono:
        note.setProperty("mono", True)
    else:
        font = note.font()
        font.setPixelSize(12)
        note.setFont(font)
    note.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    return note


def fold_cells(widgets) -> QWidget:
    """The fold chips or buttons of a models table row (narrow spacing, so six folds fit a narrow window)."""
    from ..widgets import cell_widget

    widgets = list(widgets)
    holder = cell_widget(*widgets, spacing=FOLD_SPACING)
    # the chips and buttons have a fixed size; their size hint (used by cell_widget) may be larger
    holder.setMinimumWidth(12 + len(widgets) * (FOLD_SIZE[0] + FOLD_SPACING))
    return holder


def fold_chip(fold) -> QLabel:
    chip = QLabel(fold.name)
    chip.setFixedSize(*FOLD_SIZE)
    chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
    chip.setObjectName("foldChip")
    chip.setProperty("state", fold.status)
    note = fold.log_note()
    chip.setToolTip(FOLD_STATUS[fold.status][0] + (f" · {note}" if note else ""))
    return chip


def model_name_cell(model):
    """Trainer with plans · configuration below. Long names are shortened with "…" at the end (the column is
    at least MODEL_MIN_WIDTH wide, so variants such as ..._variant07_... stay apart); the full name is the hover
    text."""
    cell = QWidget()
    layout = QVBoxLayout(cell)
    layout.setContentsMargins(6, 0, 6, 0)
    layout.setSpacing(0)
    layout.addStretch()
    for text, muted in ((model.trainer, False), (f"{model.plans} · {model.configuration}", True)):
        line = ElidedLabel(text)
        line.setProperty("mono", True)
        line.setProperty("muted", muted)
        line.setMinimumWidth(1)
        line.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        line.setToolTip(f"{model.trainer}__{model.plans}__{model.configuration}")
        layout.addWidget(line)
    layout.addStretch()
    cell.setMinimumWidth(1)
    cell.setToolTip(f"{model.trainer}__{model.plans}__{model.configuration}")
    return cell


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


class RunDialog:
    """Hosts a stage's run form (planner/trainer options, command, Copy and Run) in a dialog.

    The form widget is built once by the stage and kept, so its choices survive between openings.
    """

    def __init__(self, parent: QWidget, title: str, form: QWidget, on_open=None) -> None:
        from qtpy.QtWidgets import QDialog

        self._on_open = on_open
        self.dialog = QDialog(parent)
        self.dialog.setWindowTitle(title)
        self.dialog.setMinimumWidth(480)
        layout = QVBoxLayout(self.dialog)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(form)

    def open(self) -> None:
        self.dialog.show()
        if self._on_open is not None:
            self._on_open()  # the form's hints are only computed while it is visible
        # fit the form after its hints are shown or hidden: nothing squeezed, no empty space at the bottom.
        # Again on the next event loop turn, because showing the dialog resizes it once more.
        self._fit()
        QTimer.singleShot(0, self._fit)
        self.dialog.raise_()
        self.dialog.activateWindow()

    def _fit(self) -> None:
        layout = self.dialog.layout()
        layout.activate()
        width = max(self.dialog.width(), 480)
        self.dialog.resize(width, layout.heightForWidth(width) if layout.hasHeightForWidth()
                           else self.dialog.sizeHint().height())

    def refit(self) -> None:
        """Fit again after hints changed while the dialog is open (e.g. the device was detected)."""
        if self.visible:
            QTimer.singleShot(0, self._fit)

    def close(self) -> None:
        self.dialog.accept()

    @property
    def visible(self) -> bool:
        return self.dialog.isVisible()


def detail_row(name: str, value: QWidget | str, *, mono: bool = False) -> QHBoxLayout:
    """A key/value line of a details panel."""
    row = QHBoxLayout()
    row.setSpacing(10)
    key = label(name, muted=True)
    key.setFixedWidth(110)
    row.addWidget(key)
    if isinstance(value, str):
        value = label(value, mono=mono, wrap=True, selectable=True)
    if isinstance(value, QLabel) and value.objectName() == "badge":
        row.addWidget(value, 0, Qt.AlignmentFlag.AlignLeft)  # badges keep their size, left-aligned
        row.addStretch(1)
    else:
        row.addWidget(value, 1)
    return row
