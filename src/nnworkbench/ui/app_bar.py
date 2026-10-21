"""Top bar: logo, back to the overview, dataset switcher, Annotation Workflow, Guide, Settings and Info."""

from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QFrame, QHBoxLayout, QPushButton, QWidget

from nnworkbench.models.dataset import DatasetRecord

from .widgets import AppMark, ElidedLabel, ToggleSwitch, button, dataset_id_label, label


class AppBar(QFrame):
    back_requested = Signal()
    switcher_requested = Signal()
    guide_requested = Signal()
    about_requested = Signal()
    settings_requested = Signal()
    annotation_toggled = Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("appBar")
        self.setFixedHeight(60)
        row = QHBoxLayout(self)
        row.setContentsMargins(24, 0, 24, 0)
        row.setSpacing(12)
        row.addWidget(AppMark(30))
        row.addWidget(label("nnWorkbench", name="h2"))
        divider = QFrame()
        divider.setFixedSize(1, 26)
        divider.setStyleSheet("background: #e3e8ef;")
        row.addWidget(divider)
        self._title = label("Overview", bold=True)
        row.addWidget(self._title)
        self._back = QPushButton("← All datasets")
        self._back.setObjectName("backButton")
        self._back.setCursor(Qt.CursorShape.PointingHandCursor)
        self._back.clicked.connect(self.back_requested.emit)
        row.addWidget(self._back)
        self.switcher = QPushButton()
        self.switcher.setObjectName("switcherButton")
        self.switcher.setCheckable(True)
        self.switcher.setCursor(Qt.CursorShape.PointingHandCursor)
        self.switcher.clicked.connect(self.switcher_requested.emit)
        switcher_row = QHBoxLayout(self.switcher)
        switcher_row.setContentsMargins(10, 4, 12, 4)
        switcher_row.setSpacing(8)
        self._switcher_id = dataset_id_label(0)
        self._switcher_name = ElidedLabel()
        font = self._switcher_name.font()
        font.setBold(True)
        self._switcher_name.setFont(font)
        self._switcher_folder = label("", muted=True, size=12)
        switcher_row.addWidget(self._switcher_id)
        switcher_row.addWidget(self._switcher_name)  # elides long names when the button gets narrow
        switcher_row.addWidget(self._switcher_folder)
        switcher_row.addStretch()
        switcher_row.addWidget(label("▾", muted=True))
        for child in (self._switcher_id, self._switcher_name, self._switcher_folder):
            child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.switcher.setFixedHeight(40)
        row.addWidget(self.switcher)
        row.addStretch()
        self.annotation = ToggleSwitch("Annotation Workflow")
        self.annotation.setToolTip("Applies to all datasets: adds ✎ to annotate cases in Napari (saved directly to labelsTr) "
                                   "and the annotation hints")
        self.annotation.toggled.connect(self.annotation_toggled.emit)
        row.addWidget(self.annotation)
        row.addWidget(button("? Guide", ghost=True, clicked=self.guide_requested.emit))
        row.addWidget(button("⚙ Settings", ghost=True, clicked=self.settings_requested.emit))
        # Rarely needed, so last: version, credits (logos) and licence
        row.addWidget(button("Info", ghost=True, icon="info", clicked=self.about_requested.emit,
                             tooltip="About nnWorkbench: version, credits and licence"))
        self.show_overview()

    def show_overview(self) -> None:
        self._title.show()
        self._back.hide()
        self.switcher.hide()

    def show_dataset(self, record: DatasetRecord) -> None:
        self._title.hide()
        self._back.show()
        self.switcher.show()
        self._switcher_id.setText(f"{record.dataset_id:03d}")
        self._switcher_name.setText(record.name)
        self._switcher_folder.setText(f"· Dataset{record.dataset_id:03d}")

    def set_annotation(self, enabled: bool) -> None:
        self.annotation.blockSignals(True)
        self.annotation.setChecked(enabled)
        self.annotation.blockSignals(False)
        self.annotation.update()

    def switcher_anchor(self) -> QWidget:
        return self.switcher
