"""Dropdown that lists all datasets, with the current one scrolled to the middle."""

from __future__ import annotations

from qtpy.QtCore import QPoint, QRect, QRectF, QSize, Qt, Signal
from qtpy.QtGui import QColor, QFont, QFontMetrics, QPainter
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from nnworkbench.models.dataset import DatasetRecord

from . import style
from .widgets import availability_width, button, label, paint_availability

_ROW_HEIGHT = 46


class _RowDelegate(QStyledItemDelegate):
    """Paints a dataset row (ID, name with "…", RAW PRE RES) instead of one widget per row: 1000 datasets
    took a second to open as widgets. The current dataset's ID is filled blue."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.current_id: int | None = None

    def paint(self, painter: QPainter, option, index) -> None:
        record: DatasetRecord = index.data(Qt.ItemDataRole.UserRole)
        background = QStyleOptionViewItem(option)
        self.initStyleOption(background, index)
        background.text = ""  # the name is drawn below, elided
        widget = option.widget
        (widget.style() if widget else QApplication.style()).drawControl(
            QStyle.ControlElement.CE_ItemViewItem, background, painter, widget)
        rect = option.rect.adjusted(16, 0, -18, 0)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        id_font = QFont(style.MONO)
        id_font.setPixelSize(11)
        id_font.setBold(True)
        painter.setFont(id_font)
        text = f"{record.dataset_id:03d}"
        metrics = QFontMetrics(id_font)
        pill = QRectF(rect.x(), rect.center().y() - (metrics.height() + 4) / 2 + 1,
                      metrics.horizontalAdvance(text) + 12, metrics.height() + 4)
        current = record.dataset_id == self.current_id
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(style.BLUE if current else "#eef2f7"))
        painter.drawRoundedRect(pill, 6, 6)
        painter.setPen(QColor("white" if current else "#475569"))
        painter.drawText(pill, Qt.AlignmentFlag.AlignCenter, text)
        pills = availability_width()
        name_font = QFont(option.font)
        name_font.setBold(True)
        painter.setFont(name_font)
        painter.setPen(QColor(style.TEXT))
        left = int(pill.right()) + 10
        name_rect = QRect(left, rect.y(), max(0, rect.right() - pills - 10 - left), rect.height())
        name = QFontMetrics(name_font).elidedText(record.name, Qt.TextElideMode.ElideRight, name_rect.width())
        painter.drawText(name_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, name)
        painter.restore()
        paint_availability(painter, QRect(rect.right() - pills, rect.y(), pills, rect.height()),
                           (record.raw_path is not None, record.preprocessed_path is not None,
                            record.results_path is not None))

    def sizeHint(self, option, index) -> QSize:
        return QSize(430, _ROW_HEIGHT)


class DatasetSwitcher(QFrame):
    dataset_chosen = Signal(object)
    overview_requested = Signal()
    new_dataset_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent, Qt.WindowType.Popup)
        self.setObjectName("popup")
        self.setFixedWidth(470)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        search_holder = QWidget()
        search_row = QVBoxLayout(search_holder)
        search_row.setContentsMargins(12, 12, 12, 12)
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search datasets by name or ID…")
        self._search.textChanged.connect(self._filter)
        self._search.returnPressed.connect(self._open_current)
        search_row.addWidget(self._search)
        layout.addWidget(search_holder)
        self._list = QListWidget()
        self._list.setFixedHeight(322)
        self._list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self._list.setUniformItemSizes(True)
        self._delegate = _RowDelegate(self._list)
        self._list.setItemDelegate(self._delegate)
        self._list.itemActivated.connect(lambda item: self._choose(item))
        self._list.itemClicked.connect(lambda item: self._choose(item))
        layout.addWidget(self._list)
        footer = QFrame()
        footer.setObjectName("cardBand")
        footer_row = QHBoxLayout(footer)
        footer_row.setContentsMargins(12, 10, 12, 10)
        footer_row.addWidget(button("▦ All datasets (overview)", small=True, clicked=self._overview))
        footer_row.addWidget(label("↑ ↓ navigate · ↵ open", muted=True, size=11))
        footer_row.addStretch()
        footer_row.addWidget(button("＋ New dataset", small=True, clicked=self._new_dataset))
        layout.addWidget(footer)
        self._records: list[DatasetRecord] = []

    def open(self, anchor: QWidget, records: list[DatasetRecord], current: DatasetRecord | None) -> None:
        self._records = records
        self._search.clear()
        self._list.clear()
        self._delegate.current_id = current.dataset_id if current is not None else None
        current_item = None
        for record in records:
            item = QListWidgetItem(record.name)  # the text: keyboard search; painted by _RowDelegate
            item.setData(Qt.ItemDataRole.UserRole, record)
            item.setToolTip(f"{record.name}\n{record.directory_name}")  # long names are shortened with "…"
            self._list.addItem(item)
            if current is not None and record.dataset_id == current.dataset_id:
                current_item = item
        # Below the button, but always inside the window: shift left if needed and shorten the list on low windows.
        window = anchor.window().geometry()  # global coordinates of the main window's content
        position = anchor.mapToGlobal(QPoint(-60, anchor.height() + 4))
        x = max(window.left() + 8, min(position.x(), window.right() - self.width() - 8))
        other = self.sizeHint().height() - self._list.height()  # search field and footer
        room = window.bottom() - position.y() - 8
        self._list.setFixedHeight(max(120, min(322, room - other)))
        self.adjustSize()
        self.move(x, position.y())
        self.show()
        self._search.setFocus()
        if current_item is not None:
            self._list.setCurrentItem(current_item)
            self._list.scrollToItem(current_item, QAbstractItemView.ScrollHint.PositionAtCenter)

    def _filter(self, text: str) -> None:
        needle = text.strip().casefold()
        first_visible = None
        for index in range(self._list.count()):
            item = self._list.item(index)
            record: DatasetRecord = item.data(Qt.ItemDataRole.UserRole)
            visible = record.matches(needle)
            item.setHidden(not visible)
            if visible and first_visible is None:
                first_visible = item
        if first_visible is not None and needle:
            self._list.setCurrentItem(first_visible)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            self._list.setFocus()
            self._list.keyPressEvent(event)
            return
        super().keyPressEvent(event)

    def _open_current(self) -> None:
        item = self._list.currentItem()
        if item is not None and not item.isHidden():  # hidden: the search matches no dataset
            self._choose(item)

    def _choose(self, item: QListWidgetItem) -> None:
        self.hide()
        self.dataset_chosen.emit(item.data(Qt.ItemDataRole.UserRole))

    def _overview(self) -> None:
        self.hide()
        self.overview_requested.emit()

    def _new_dataset(self) -> None:
        self.hide()
        self.new_dataset_requested.emit()
