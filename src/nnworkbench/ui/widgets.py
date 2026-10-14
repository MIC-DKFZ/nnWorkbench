"""Small building blocks shared by all pages."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from qtpy.QtCore import QByteArray, QPoint, QRect, QSize, Qt, QUrl, Signal
from qtpy.QtGui import (
    QAction,
    QColor,
    QFont,
    QGuiApplication,
    QIcon,
    QIntValidator,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
)
from qtpy.QtSvg import QSvgRenderer
from qtpy.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from . import style

# ------------------------------------------------------------ icons
# Small SVG line icons (assets/icons/*.svg) instead of emoji, which look different on every system and font.
ICONS = ("search", "edit", "chart", "folder", "file", "bulb", "info")
_ICON_DIR = Path(__file__).parent.parent / "assets" / "icons"
_ICON_COLOR = "#334155"  # the stroke colour in the files; replaced for other colours
_icon_cache: dict[tuple[str, str, int], QPixmap] = {}


def icon_pixmap(name: str, size: int = 16, color: str = _ICON_COLOR) -> QPixmap:
    key = (name, color, size)
    if key not in _icon_cache:
        svg = (_ICON_DIR / f"{name}.svg").read_text(encoding="utf-8").replace(_ICON_COLOR, color)
        ratio = 2  # sharp on high-DPI screens
        pixmap = QPixmap(size * ratio, size * ratio)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        QSvgRenderer(QByteArray(svg.encode())).render(painter)
        painter.end()
        pixmap.setDevicePixelRatio(ratio)
        _icon_cache[key] = pixmap
    return _icon_cache[key]


def svg_icon(name: str, color: str = _ICON_COLOR) -> QIcon:
    return QIcon(icon_pixmap(name, 16, color))


def icon_html(name: str, size: int = 14) -> str:
    """The icon inline in rich text (how-to steps, tips)."""
    # a file:// URL: a plain Windows path ("C:/…") would be read as a URL with the scheme "c"
    url = QUrl.fromLocalFile(str(_ICON_DIR / f"{name}.svg")).toString()
    return f"<img src='{url}' width='{size}' height='{size}' style='vertical-align: middle'>"


def icon_label(name: str, size: int = 16) -> QLabel:
    widget = QLabel()
    widget.setPixmap(icon_pixmap(name, size))
    return widget


def polish(widget: QWidget) -> None:
    """Re-apply the stylesheet after a dynamic property changed."""
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def label(text: str = "", *, name: str | None = None, muted: bool = False, mono: bool = False,
          wrap: bool = False, size: int | None = None, bold: bool = False, rich: bool = False,
          selectable: bool = False) -> QLabel:
    widget = QLabel(text)
    if selectable:  # file names, paths and IDs can be selected and copied
        widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        widget.setCursor(Qt.CursorShape.IBeamCursor)
    if name:
        widget.setObjectName(name)
    if muted:
        widget.setProperty("muted", True)
    if mono:
        widget.setProperty("mono", True)
    if wrap:
        widget.setWordWrap(True)
    if rich:
        widget.setTextFormat(Qt.TextFormat.RichText)
    if size or bold:
        font = widget.font()
        if size:
            font.setPixelSize(size)
        if bold:
            font.setBold(True)
        widget.setFont(font)
    return widget


class ElidedLabel(QLabel):
    """A one-line label that shortens long text with "…" and shows the full text on hover."""

    def __init__(self, text: str = "", parent=None, mode=Qt.TextElideMode.ElideRight) -> None:
        super().__init__(parent)
        self._full = ""
        self._mode = mode
        self.setMinimumWidth(40)
        # Preferred: full width when there is room, shrinks (with "…") when there is not.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.setText(text)

    def minimumSizeHint(self) -> QSize:
        return QSize(40, super().minimumSizeHint().height())

    def setText(self, text: str) -> None:
        self._full = text
        self.setToolTip(text)
        self._elide()

    def contextMenuEvent(self, event) -> None:
        from qtpy.QtWidgets import QMenu

        menu = QMenu(self)
        menu.addAction("Copy", lambda: QGuiApplication.clipboard().setText(self._full))
        menu.exec(event.globalPos())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._elide()

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        return QSize(self.fontMetrics().horizontalAdvance(self._full) + 4, hint.height())

    def _elide(self) -> None:
        width = max(self.width(), 40)
        super().setText(self.fontMetrics().elidedText(self._full, self._mode, width))


def breakable(text: str) -> str:
    """Allow line breaks after "_" in long identifiers (trainer and plans names), so wrapped labels do not widen."""
    return text.replace("_", "_\u200b")


def section_label(text: str) -> QLabel:
    return label(text.upper(), name="sectionLabel")


def badge(text: str, tone: str = "gray", tooltip: str = "") -> QLabel:
    widget = QLabel(text)
    widget.setObjectName("badge")
    widget.setProperty("tone", tone)
    if tooltip:
        widget.setToolTip(tooltip)
    widget.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    return widget


def set_badge(widget: QLabel, text: str, tone: str, tooltip: str = "") -> None:
    widget.setText(text)
    widget.setProperty("tone", tone)
    widget.setToolTip(tooltip)
    polish(widget)


def tip(text: str, tone: str = "blue") -> QLabel:
    widget = QLabel(text)
    widget.setObjectName("tip")
    widget.setProperty("tone", tone)
    widget.setWordWrap(True)
    widget.setTextFormat(Qt.TextFormat.RichText)
    return widget


def button(text: str, *, primary: bool = False, ghost: bool = False, small: bool = False, tooltip: str = "",
           clicked: Callable[[], None] | None = None, purple: bool = False, danger: bool = False,
           icon: str = "") -> QPushButton:
    widget = QPushButton(text)
    if icon:
        widget.setIcon(svg_icon(icon, "#ffffff" if primary or purple or danger else _ICON_COLOR))
        widget.setIconSize(QSize(14, 14))
    for flag, name in ((primary, "primary"), (ghost, "ghost"), (small, "small"), (purple, "purple"),
                       (danger, "danger")):
        if flag:
            widget.setProperty(name, True)
    if tooltip:
        widget.setToolTip(tooltip)
    if clicked is not None:
        widget.clicked.connect(lambda _checked=False: clicked())
    widget.setCursor(Qt.CursorShape.PointingHandCursor)
    return widget


def icon_button(text: str, tooltip: str, clicked: Callable[[], None] | None = None) -> QPushButton:
    """A small square button; ``text`` is an icon name (see ICONS) or a short text such as "{ }"."""
    widget = QPushButton("" if text in ICONS else text)
    if text in ICONS:
        widget.setIcon(svg_icon(text))
        widget.setIconSize(QSize(15, 15))
    widget.setObjectName("iconButton")
    widget.setToolTip(tooltip)
    widget.setCursor(Qt.CursorShape.PointingHandCursor)
    if clicked is not None:
        widget.clicked.connect(lambda _checked=False: clicked())
    return widget


def separator() -> QFrame:
    line = QFrame()
    line.setObjectName("separator")
    return line


def dataset_id_label(dataset_id: int, current: bool = False) -> QLabel:
    widget = QLabel(f"{dataset_id:03d}")
    widget.setObjectName("dsid")
    widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
    widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    if current:
        widget.setProperty("current", True)
    return widget


def availability(raw: bool, preprocessed: bool, results: bool) -> QWidget:
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    for text, present in (("RAW", raw), ("PRE", preprocessed), ("RES", results)):
        item = QLabel(text)
        item.setObjectName("availOn" if present else "availOff")
        item.setAlignment(Qt.AlignmentFlag.AlignCenter)
        item.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)  # pill, not row-high
        row.addWidget(item, 0, Qt.AlignmentFlag.AlignVCenter)
    row.addStretch()
    return holder


def format_time(timestamp: float | None) -> str:
    if timestamp is None:
        return "—"
    moment = datetime.fromtimestamp(timestamp)
    today = datetime.now().date()
    if moment.date() == today:
        return f"today {moment:%H:%M}"
    if (today - moment.date()).days == 1:
        return f"yesterday {moment:%H:%M}"
    return f"{moment:%Y-%m-%d %H:%M}"


def format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


class Card(QFrame):
    """White rounded panel with an optional header row."""

    def __init__(self, parent=None, *, padding: int = 0) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(padding, padding, padding, padding)
        self.body.setSpacing(0)

    def add_header(self, title: str = "", *, subtitle: str = "", mono_subtitle: bool = False,
                   band: bool = False) -> QHBoxLayout:
        header = QFrame()
        header.setObjectName("cardBand" if band else "cardHeader")
        row = QHBoxLayout(header)
        row.setContentsMargins(16, 10, 16, 10)
        row.setSpacing(10)
        if title:
            row.addWidget(label(title, name="h3"))
        if subtitle:
            row.addWidget(label(subtitle, muted=True, mono=mono_subtitle, size=12))
        self.body.addWidget(header)
        return row


class FlowLayout(QLayout):
    """Lay out children left to right and wrap them into new lines."""

    def __init__(self, parent=None, spacing: int = 6) -> None:
        super().__init__(parent)
        self._items = []
        self._spacing = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._layout(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._layout(rect, apply=True)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        return size

    def clear(self) -> None:
        while self._items:
            item = self._items.pop()
            if item.widget():
                item.widget().deleteLater()

    def _layout(self, rect: QRect, apply: bool) -> int:
        x, y, line_height = rect.x(), rect.y(), 0
        for item in self._items:
            hint = item.sizeHint()
            if x + hint.width() > rect.right() and line_height > 0:
                x = rect.x()
                y += line_height + self._spacing
                line_height = 0
            if apply:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + self._spacing
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y()


class HowToCard(QFrame):
    """Left column: numbered steps, tips and a link to the guide chapter."""

    guide_requested = Signal()
    collapse_requested = Signal()

    def __init__(self, title: str, steps: Iterable[str], tips: Iterable[tuple[str, str]], guide: str) -> None:
        super().__init__()
        self.setObjectName("howto")
        self.setFixedWidth(280)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        head = QHBoxLayout()
        head.addWidget(section_label(title))
        head.addStretch()
        collapse = QPushButton("«")
        collapse.setObjectName("link")
        collapse.setToolTip("Hide the how-to (remembered)")
        collapse.setCursor(Qt.CursorShape.PointingHandCursor)
        collapse.clicked.connect(self.collapse_requested.emit)
        head.addWidget(collapse)
        layout.addLayout(head)
        items = "".join(f"<li style='margin-bottom:6px'>{step}</li>" for step in steps)
        text = label(f"<ol style='margin-left:-22px'>{items}</ol>", wrap=True, rich=True)
        layout.addWidget(text)
        self._tips = QVBoxLayout()
        self._tips.setSpacing(10)
        layout.addLayout(self._tips)
        for tone, message in tips:
            self.add_tip(message, tone)
        layout.addStretch()
        link = QLabel(f"<span style='color:#64748b'>More in the guide ·</span> "
                      f"<a href='guide' style='color:#1f5fbf; text-decoration:none; font-weight:600'>{guide} ›</a>")
        link.setWordWrap(True)
        link.linkActivated.connect(lambda _link: self.guide_requested.emit())
        layout.addWidget(link)

    def add_tip(self, message: str, tone: str = "blue") -> QLabel:
        widget = tip(message, tone)
        self._tips.addWidget(widget)
        return widget


class StageTab(QAbstractButton):
    """A stage tab with a round status icon and an underline when selected."""

    _ICONS: ClassVar[dict[str, str]] = {"ok": "✓", "warn": "!", "error": "!"}

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(parent)
        self.setText(text)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._state = ""
        font = self.font()
        font.setBold(True)
        self.setFont(font)

    def set_state(self, state: str) -> None:
        self._state = state
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(self.fontMetrics().horizontalAdvance(self.text()) + 62, 51)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        selected = self.isChecked()
        circle = QRect(14, 15, 20, 20)
        fill = {"ok": QColor(style.GREEN), "warn": QColor(style.ORANGE), "error": QColor(style.RED)}.get(self._state)
        if self._state == "todo":  # orange ring: ready to run (or running)
            painter.setPen(QPen(QColor(style.ORANGE), 2))
            painter.setBrush(QColor("white"))
        else:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(fill or QColor("#eef2f7"))
        painter.drawEllipse(circle)
        icon = self._ICONS.get(self._state, "")
        if icon:
            painter.setPen(QColor("white"))
            font = QFont(self.font())
            font.setPixelSize(11)
            painter.setFont(font)
            painter.drawText(circle, Qt.AlignmentFlag.AlignCenter, icon)
        painter.setFont(self.font())
        color = style.BLUE if selected else (style.TEXT if self._state else style.MUTED)
        painter.setPen(QColor(color))
        painter.drawText(QRect(42, 0, self.width() - 48, self.height()), Qt.AlignmentFlag.AlignVCenter, self.text())
        if selected:
            painter.fillRect(QRect(0, self.height() - 3, self.width(), 3), QColor(style.BLUE))
        painter.end()


class ToggleSwitch(QAbstractButton):
    """A pill switch with a text label, used for the global Annotation Workflow setting."""

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(parent)
        self.setText(text)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        font = self.font()
        font.setBold(True)
        font.setPixelSize(12)
        self.setFont(font)

    def sizeHint(self) -> QSize:
        return QSize(self.fontMetrics().horizontalAdvance(self.text()) + 66, 32)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        on = self.isChecked()
        outline = QRect(1, 1, self.width() - 2, self.height() - 2)
        painter.setPen(QPen(QColor("#c9b9ef" if on else style.LINE), 1))
        painter.setBrush(QColor(style.PURPLE_SOFT if on else "white"))
        painter.drawRoundedRect(outline, outline.height() / 2, outline.height() / 2)
        track = QRect(12, (self.height() - 17) // 2, 30, 17)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(style.PURPLE if on else "#cbd5e1"))
        painter.drawRoundedRect(track, 8.5, 8.5)
        painter.setBrush(QColor("white"))
        knob_x = track.x() + (15 if on else 2)
        painter.drawEllipse(QRect(knob_x, track.y() + 2, 13, 13))
        painter.setPen(QColor(style.PURPLE if on else style.MUTED))
        painter.drawText(QRect(52, 0, self.width() - 56, self.height()), Qt.AlignmentFlag.AlignVCenter, self.text())
        painter.end()


class ToggleGroup(QFrame):
    """Segmented control (e.g. Files | Statistics)."""

    changed = Signal(int)

    def __init__(self, options: Iterable[str], parent=None, icons: Iterable[str] = ()) -> None:
        super().__init__(parent)
        self.setObjectName("toggleGroup")
        icons = list(icons)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        for index, text in enumerate(options):
            option = QPushButton(text)
            if index < len(icons) and icons[index]:
                option.setIcon(svg_icon(icons[index]))
                option.setIconSize(QSize(14, 14))
            option.setObjectName("toggleOption")
            option.setCheckable(True)
            option.setCursor(Qt.CursorShape.PointingHandCursor)
            self._group.addButton(option, index)
            row.addWidget(option)
        self._group.button(0).setChecked(True)
        self._group.idClicked.connect(self.changed.emit)

    def set_tooltip(self, index: int, text: str) -> None:
        self._group.button(index).setToolTip(text)

    def index(self) -> int:
        return self._group.checkedId()


class ChoiceGroup(QWidget):
    """A row of checkable buttons, single or multi selection (folds, configurations)."""

    changed = Signal()

    def __init__(self, exclusive: bool = True, object_name: str = "choice", parent=None) -> None:
        super().__init__(parent)
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(6)
        self._row.addStretch()
        self._group = QButtonGroup(self)
        self._group.setExclusive(exclusive)
        self._group.buttonToggled.connect(lambda *_: self.changed.emit())
        self._buttons: dict[str, QPushButton] = {}
        self._object_name = object_name

    def set_options(self, options: Iterable[str], selected: Iterable[str] = (), disabled: Iterable[str] = (),
                    tooltips: dict[str, str] | None = None) -> None:
        selected, disabled, tooltips = set(selected), set(disabled), tooltips or {}
        self.blockSignals(True)
        for button_widget in self._buttons.values():
            self._group.removeButton(button_widget)
            self._row.removeWidget(button_widget)  # remove now; deleteLater alone can leave it visible for a while
            button_widget.hide()
            button_widget.deleteLater()
        self._buttons.clear()
        for index, option in enumerate(options):
            widget = QPushButton(option)
            widget.setObjectName(self._object_name)
            widget.setCheckable(True)
            widget.setCursor(Qt.CursorShape.PointingHandCursor)
            widget.setEnabled(option not in disabled)
            widget.setToolTip(tooltips.get(option, ""))
            self._group.addButton(widget)
            self._row.insertWidget(index, widget)
            widget.setChecked(option in selected)
            self._buttons[option] = widget
        self.blockSignals(False)
        self.changed.emit()

    def selected(self) -> list[str]:
        return [option for option, widget in self._buttons.items() if widget.isChecked()]

    def select(self, option: str) -> None:
        if option in self._buttons:
            self._buttons[option].setChecked(True)

    def set_selected(self, options: Iterable[str]) -> None:
        """Select exactly these options (for multi-selection groups)."""
        wanted = set(options)
        self.blockSignals(True)
        for option, widget in self._buttons.items():
            widget.setChecked(option in wanted)
        self.blockSignals(False)
        self.changed.emit()

    def options(self) -> list[str]:
        return list(self._buttons)


class InfoIcon(QLabel):
    """A small (i) with a short explanation: a tooltip on hover; a click opens it as a small box that stays
    open until the next click anywhere (no need to keep the mouse button pressed)."""

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(parent)
        self.setPixmap(icon_pixmap("info", 14, "#94a3b8"))
        self.setToolTip(text)
        self._text = text

    def mouseReleaseEvent(self, event) -> None:
        QToolTip.hideText()
        box = QFrame(self, Qt.WindowType.Popup)
        box.setObjectName("infoBox")
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(12, 10, 12, 10)
        text = label(self._text, wrap=True, rich=True)
        text.setFixedWidth(300)
        layout.addWidget(text)
        box.adjustSize()
        box.move(self.mapToGlobal(QPoint(0, self.height() + 4)))
        box.show()


class NumberField(QLineEdit):
    """A whole-number field. A styled line edit instead of a QSpinBox: spin boxes are drawn per platform and
    on Windows the number was not visible with the stylesheet. Type, scroll, use ↑/↓ or the steppers."""

    valueChanged = Signal(int)

    def __init__(self, minimum: int = 0, maximum: int = 99, value: int = 0, parent=None) -> None:
        super().__init__(parent)
        self._minimum, self._maximum, self._value = minimum, maximum, value
        self.setValidator(QIntValidator(minimum, maximum, self))
        self.setText(str(value))
        self.textEdited.connect(self._typed)
        self.editingFinished.connect(lambda: self.setText(str(self._value)))  # e.g. an empty field again

    def setRange(self, minimum: int, maximum: int) -> None:
        self._minimum, self._maximum = minimum, max(minimum, maximum)
        self.setValidator(QIntValidator(self._minimum, self._maximum, self))
        self.setValue(self._value)

    def value(self) -> int:
        return self._value

    def setValue(self, value: int) -> None:
        value = max(self._minimum, min(self._maximum, int(value)))
        if self.text() != str(value):
            self.setText(str(value))
        if value != self._value:
            self._value = value
            self.valueChanged.emit(value)

    def stepUp(self) -> None:
        self.setValue(self._value + 1)

    def stepDown(self) -> None:
        self.setValue(self._value - 1)

    def _typed(self, text: str) -> None:
        if text.isdigit() and self._minimum <= int(text) <= self._maximum and int(text) != self._value:
            self._value = int(text)
            self.valueChanged.emit(self._value)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            self.stepUp() if event.key() == Qt.Key.Key_Up else self.stepDown()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event) -> None:
        if self.hasFocus():  # only when focused, so scrolling a page does not change values by accident
            self.stepUp() if event.angleDelta().y() > 0 else self.stepDown()
            event.accept()
        else:
            super().wheelEvent(event)


def with_stepper(field) -> QWidget:
    """A number field with two small buttons (▴ ▾) that change it by one."""
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(2)
    row.addWidget(field)
    arrows = QVBoxLayout()
    arrows.setContentsMargins(0, 0, 0, 0)
    arrows.setSpacing(1)
    for text, step, tooltip in (("▴", field.stepUp, "Increase by one"), ("▾", field.stepDown, "Decrease by one")):
        arrow = QPushButton(text)
        arrow.setObjectName("stepButton")
        arrow.setFixedSize(18, 15)
        arrow.setToolTip(tooltip)
        arrow.setCursor(Qt.CursorShape.PointingHandCursor)
        arrow.setAutoRepeat(True)  # keep pressed to count on
        arrow.clicked.connect(lambda _checked=False, step=step: step())
        arrows.addWidget(arrow)
    row.addLayout(arrows)
    return holder


def form_row(name: str, widget: QWidget, width: int = 112, info: str = "") -> QHBoxLayout:
    """A labelled row of a settings panel; ``info`` adds a small (i) with a short explanation."""
    row = QHBoxLayout()
    row.setSpacing(10)
    title = label(name, muted=True)
    title.setWordWrap(True)
    if info:
        holder = QWidget()
        holder.setFixedWidth(width)
        holder_row = QHBoxLayout(holder)
        holder_row.setContentsMargins(0, 0, 0, 0)
        holder_row.setSpacing(5)
        holder_row.addWidget(title)
        holder_row.addWidget(InfoIcon(info))
        holder_row.addStretch()
        row.addWidget(holder)
    else:
        title.setFixedWidth(width)
        row.addWidget(title)
    row.addWidget(widget, 1)
    return row


def form_hint(text: str, width: int = 122) -> QLabel:
    hint = label(text, muted=True, wrap=True, size=11)
    hint.setContentsMargins(width, 0, 0, 0)
    return hint


def make_table(headers: Iterable[str], *, stretch: int | None = 0, row_height: int = 38) -> QTableWidget:
    headers = list(headers)
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels([header.upper() for header in headers])
    table.verticalHeader().hide()
    table.verticalHeader().setDefaultSectionSize(row_height)
    table.setShowGrid(False)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    table.setWordWrap(False)
    header = table.horizontalHeader()
    header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    header.setHighlightSections(False)
    header.setStretchLastSection(stretch is None)
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    for column in range(len(headers)):
        header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
    if stretch is not None:
        header.setSectionResizeMode(stretch, QHeaderView.ResizeMode.Stretch)
    _add_copy(table)
    return table


def _copy_cell(table: QTableWidget) -> None:
    item = table.currentItem()
    if item is None and table.selectionModel() and table.selectionModel().selectedRows():
        item = table.item(table.selectionModel().selectedRows()[0].row(), 0)
    if item is not None and item.text():
        QGuiApplication.clipboard().setText(item.text())


def _add_copy(table: QTableWidget) -> None:
    """Ctrl+C and right-click → Copy copy the text of the clicked cell (e.g. a file or case name)."""
    action = QAction("Copy", table)
    action.setShortcut(QKeySequence.StandardKey.Copy)
    action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
    action.triggered.connect(lambda: _copy_cell(table))
    table.addAction(action)
    table.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)


def table_item(text: str, *, mono: bool = False, muted: bool = False, bold: bool = False, data=None) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    if text:
        item.setToolTip(text)  # full text when the column is too narrow
    font = item.font()
    if mono:
        font = QFont(style.MONO)
        font.setPixelSize(12)
    if bold:
        font.setBold(True)
    item.setFont(font)
    if muted:
        item.setForeground(QColor(style.MUTED))
    if data is not None:
        item.setData(Qt.ItemDataRole.UserRole, data)
    return item


def cell_widget(*widgets: QWidget, align_right: bool = False, spacing: int = 4) -> QWidget:
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(6, 0, 6, 0)
    row.setSpacing(spacing)
    if align_right:
        row.addStretch()
    width = 12
    for widget in widgets:
        widget.ensurePolished()  # stylesheet padding must be known for the column width
        row.addWidget(widget)
        width += widget.sizeHint().width() + spacing
    if not align_right:
        row.addStretch()
    holder.setMinimumWidth(width)
    return holder


def fit_columns(table: QTableWidget) -> None:
    """Widen columns so that cell widgets (badges, chips, buttons) are not clipped."""
    header = table.horizontalHeader()
    for column in range(table.columnCount()):
        if header.sectionResizeMode(column) == QHeaderView.ResizeMode.Stretch:
            continue
        needed = max((table.cellWidget(row, column).minimumWidth() for row in range(table.rowCount())
                      if table.cellWidget(row, column) is not None), default=0)
        if needed > header.sectionSize(column):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
            header.resizeSection(column, needed)


def scroll(widget: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    area.setWidget(widget)
    return area


def empty_state(text: str, icon: str = "") -> QWidget:
    holder = QWidget()
    column = QVBoxLayout(holder)
    column.setContentsMargins(24, 28, 24, 28)  # near the top of its card instead of centred in a large area
    if icon:
        mark = icon_label(icon, 28) if icon in ICONS else label(icon, size=30)
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.addWidget(mark)
    message = label(text, muted=True, wrap=True, rich=True)
    message.setAlignment(Qt.AlignmentFlag.AlignCenter)
    column.addWidget(message)
    column.addStretch()
    return holder


def clear_layout(layout: QLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item.widget() is not None:
            item.widget().hide()
            item.widget().deleteLater()
        elif item.layout() is not None:
            clear_layout(item.layout())


class Bar(QWidget):
    """A thin horizontal value bar (0–1)."""

    def __init__(self, value: float, color: str = style.BLUE, parent=None) -> None:
        super().__init__(parent)
        self._value = max(0.0, min(1.0, value))
        self._color = color
        self.setFixedHeight(6)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#edf1f6"))
        painter.drawRoundedRect(self.rect(), 3, 3)
        painter.setBrush(QColor(self._color))
        painter.drawRoundedRect(QRect(0, 0, int(self.width() * self._value), self.height()), 3, 3)
        painter.end()


class IconButtonsDelegate(QStyledItemDelegate):
    """Paints small icon buttons in a table cell without creating widgets (fast for thousands of rows).

    ``buttons(row)`` returns ``[(icon, tooltip, checked), …]``; ``clicked(row, index)`` is called on a click.
    """

    SIZE, GAP = (30, 26), 4

    def __init__(self, table: QTableWidget, buttons: Callable, clicked: Callable) -> None:
        super().__init__(table)
        self._buttons, self._clicked = buttons, clicked
        table.viewport().setMouseTracking(True)
        table.viewport().installEventFilter(self)
        self._table = table

    def _rects(self, rect: QRect, count: int) -> list[QRect]:
        width, height = self.SIZE
        x = rect.right() - 8 - count * width - (count - 1) * self.GAP
        y = rect.center().y() - height // 2
        return [QRect(x + index * (width + self.GAP), y, width, height) for index in range(count)]

    def paint(self, painter: QPainter, option, index) -> None:
        buttons = self._buttons(index.row())
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for rect, (icon, _tip, checked) in zip(self._rects(option.rect, len(buttons)), buttons):
            painter.setPen(QPen(QColor(style.BLUE if checked else style.LINE), 1))
            painter.setBrush(QColor(style.BLUE if checked else "white"))
            painter.drawRoundedRect(rect.adjusted(0, 0, -1, -1), 7, 7)
            if icon in ICONS:
                pixmap = icon_pixmap(icon, 15, "#ffffff" if checked else _ICON_COLOR)
                painter.drawPixmap(rect.center().x() - 7, rect.center().y() - 7, pixmap)
            else:
                painter.setPen(QColor("white" if checked else style.TEXT))
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, icon)
        painter.restore()

    def sizeHint(self, option, index) -> QSize:
        count = len(self._buttons(index.row()))
        return QSize(16 + count * self.SIZE[0] + (count - 1) * self.GAP, self.SIZE[1] + 8)

    def _hit(self, position) -> tuple[int, int] | None:
        index = self._table.indexAt(position)
        if not index.isValid() or self._table.itemDelegateForColumn(index.column()) is not self:
            return None
        rect = self._table.visualRect(index)
        for number, button_rect in enumerate(self._rects(rect, len(self._buttons(index.row())))):
            if button_rect.contains(position):
                return index.row(), number
        return None

    def eventFilter(self, watched, event) -> bool:
        from qtpy.QtCore import QEvent

        if event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
            hit = self._hit(event.position().toPoint() if hasattr(event, "position") else event.pos())
            if hit is not None:
                self._clicked(*hit)
                return True
        elif event.type() == QEvent.Type.ToolTip:
            hit = self._hit(event.pos())
            if hit is not None:
                QToolTip.showText(event.globalPos(), self._buttons(hit[0])[hit[1]][1], self._table)
                return True
        elif event.type() == QEvent.Type.MouseMove:
            hit = self._hit(event.position().toPoint() if hasattr(event, "position") else event.pos())
            self._table.viewport().setCursor(Qt.CursorShape.PointingHandCursor if hit else Qt.CursorShape.ArrowCursor)
        return False


def mark_item(present: bool) -> QTableWidgetItem:
    """A green ✓ or red ✗ as a plain table item (no widget)."""
    item = QTableWidgetItem("✓" if present else "✗")
    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
    item.setForeground(QColor(style.GREEN if present else style.RED))
    font = item.font()
    font.setBold(True)
    item.setFont(font)
    return item
