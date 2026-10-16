"""One dataset: stage tabs and the stage pages."""

from __future__ import annotations

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QButtonGroup, QFrame, QHBoxLayout, QStackedWidget, QVBoxLayout, QWidget

from nnworkbench.models.dataset import DatasetRecord

from .context import AppContext
from .dataset_state import DatasetState, load_state
from .stages.data_stage import DataStage
from .stages.predict_stage import PredictStage
from .stages.preprocess_stage import PreprocessStage
from .stages.train_stage import TrainStage
from .stages.validate_stage import ValidateStage
from .widgets import StageTab

STAGES = (("data", "Data"), ("preprocess", "Preprocess"), ("train", "Train"), ("validate", "Validate"),
          ("predict", "Predict"))


class DatasetPage(QWidget):
    guide_requested = Signal(str, str)

    def __init__(self, context: AppContext, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("page")
        self._context = context
        self.record: DatasetRecord | None = None
        self.state: DatasetState | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        bar = QFrame()
        bar.setObjectName("stageBar")
        bar.setFixedHeight(52)
        bar_row = QHBoxLayout(bar)
        bar_row.setContentsMargins(24, 0, 24, 0)
        bar_row.setSpacing(0)
        self._tabs: dict[str, StageTab] = {}
        group = QButtonGroup(self)
        group.setExclusive(True)
        for key, title in STAGES:
            tab = StageTab(title)
            tab.clicked.connect(lambda _=False, selected=key: self.show_stage(selected))
            group.addButton(tab)
            bar_row.addWidget(tab)
            self._tabs[key] = tab
        bar_row.addStretch()
        layout.addWidget(bar)
        self._stack = QStackedWidget()
        self._stack.setObjectName("pages")
        self.pages = {
            "data": DataStage(context),
            "preprocess": PreprocessStage(context),
            "train": TrainStage(context),
            "validate": ValidateStage(context),
            "predict": PredictStage(context),
        }
        for page in self.pages.values():
            self._stack.addWidget(page)
            page.navigate.connect(self.show_stage)
            page.guide_requested.connect(self.guide_requested.emit)
            page.reload_requested.connect(self.reload)
        layout.addWidget(self._stack, 1)
        self._current = "data"
        self.show_stage("data")

    def open(self, record: DatasetRecord) -> None:
        if self.record is None or record.dataset_id != self.record.dataset_id:
            self.state = None  # the stage stays the same when switching datasets
            self._context.model_label = None
            for page in self.pages.values():
                page.show_header_loading(record.directory_name)
                page.show_loading()
            for tab in self._tabs.values():
                tab.set_state("")
                tab.setToolTip("")
        self.record = record
        self.reload()

    def reload(self) -> None:
        if self.record is None:
            return
        record = self.record
        self._context.tasks.run("dataset-state", lambda: load_state(record), self._apply_state,
                                lambda error: print(f"Could not load {record.directory_name}: {error}"))

    def _apply_state(self, state: object) -> None:
        if not isinstance(state, DatasetState) or self.record is None or state.record != self.record:
            return
        self.state = state
        for key, method in (("data", state.data_state), ("preprocess", state.preprocess_state),
                            ("train", state.train_state), ("validate", state.validate_state),
                            ("predict", state.predict_state)):
            tab_state, hint = method()
            self._tabs[key].set_state(tab_state)
            self._tabs[key].setToolTip(hint)
        for page in self.pages.values():
            page.set_state(state)

    def show_stage(self, key: str) -> None:
        self._current = key
        self._tabs[key].setChecked(True)
        self._stack.setCurrentWidget(self.pages[key])
        if self._context.model_label:
            self.pages[key].follow_model(self._context.model_label)
