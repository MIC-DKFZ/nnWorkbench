"""Dropdown that lists all datasets, with the current one scrolled to the middle."""

from __future__ import annotations

from qtpy.QtCore import QPoint, QSize, Qt, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nnworkbench.models.dataset import DatasetRecord

from .widgets import availability, button, dataset_id_label, label


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
        current_item = None
        for record in records:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, record)
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(16, 0, 18, 0)
            row_layout.setSpacing(10)
            is_current = current is not None and record.dataset_id == current.dataset_id
            row_layout.addWidget(dataset_id_label(record.dataset_id, current=is_current))
            name = label(record.name, bold=True)
            name.setTextFormat(Qt.TextFormat.PlainText)  # folder names are not HTML
            row_layout.addWidget(name, 1)
            row_layout.addWidget(availability(record.raw_path is not None, record.preprocessed_path is not None,
                                              record.results_path is not None))
            item.setSizeHint(QSize(430, 46))
            self._list.addItem(item)
            self._list.setItemWidget(item, row)
            if is_current:
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
            visible = not needle or needle in record.name.casefold() or needle in f"{record.dataset_id:03d}"
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
