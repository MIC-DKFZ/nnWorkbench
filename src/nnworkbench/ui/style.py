from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

STYLESHEET = """
QMainWindow, QDialog {
    background: #f7f8fa;
    color: #252525;
}
QWidget {
    color: #252525;
    font-size: 13px;
}
QLabel {
    color: #464646;
}
#appRoot {
    background: #f7f8fa;
}
#pathsHeader, #datasetsPane, #workspacePane, #metadataEditor {
    background: #ffffff;
    border: 1px solid #d7d7d7;
    border-radius: 10px;
}
#pathsHeader {
    padding: 6px;
}
QWidget#acknowledgementLogos {
    background: transparent;
    border: 0;
}
QWidget#annotationWorkflow {
    background: #eeeeee;
    border: 1px solid #c7c7c7;
    border-radius: 8px;
}
QWidget#annotationWorkflow[active="true"] {
    background: #f2f8ef;
    border: 2px solid #5b9137;
}
QLabel#annotationWorkflowTitle {
    color: #666666;
    font-weight: 700;
}
QPushButton#annotationToggle, QPushButton#annotationDocumentation {
    background: #ffffff;
    border: 1px solid #c9c9c9;
    border-radius: 6px;
    min-width: 48px;
    max-width: 48px;
    min-height: 30px;
    max-height: 30px;
    padding: 0;
}
QPushButton#annotationToggle {
    color: #555555;
    font-weight: 600;
}
QPushButton#annotationToggle:hover, QPushButton#annotationDocumentation:hover {
    background: #edf3fa;
    border-color: #7f9fc9;
}
QPushButton#annotationToggle[active="true"] {
    color: #315f1f;
}
QPushButton#annotationDocumentation {
    font-size: 17px;
}
QLabel#paneTitle {
    color: #1f1f1f;
    font-size: 16px;
    font-weight: 700;
    padding-bottom: 4px;
}
QLabel#metadataNotice {
    color: #666666;
    padding-bottom: 4px;
}
QLabel#workspaceSummary {
    color: #666666;
    padding-bottom: 2px;
}
QLabel#analysisSummary {
    color: #666666;
    font-weight: 600;
}
QLabel#consistencyWarning {
    background: #fff1e7;
    border: 1px solid #f29a57;
    border-radius: 6px;
    color: #a74400;
    padding: 7px 9px;
}
QLabel#metadataError {
    background: #fff1f2;
    border: 1px solid #dc2626;
    border-radius: 6px;
    color: #b91c1c;
    padding: 7px 9px;
}
QLineEdit, QPlainTextEdit, QTableWidget, QSpinBox, QComboBox {
    background: #ffffff;
    border: 1px solid #c9c9c9;
    border-radius: 6px;
    color: #252525;
    padding: 7px 9px;
    selection-background-color: #dceaff;
    selection-color: #0a4697;
}
QLineEdit:focus, QPlainTextEdit:focus, QTableWidget:focus, QSpinBox:focus, QComboBox:focus {
    border: 2px solid #2f6fca;
}
QLineEdit:read-only {
    background: #f5f5f5;
    color: #666666;
}
QLineEdit#trainingCount[invalid="true"] {
    border: 2px solid #dc2626;
}
QTableWidget#channelTable[invalid="true"], QTableWidget#labelTable[invalid="true"] {
    border: 2px solid #dc2626;
}
QSpinBox#settingsSpinBox {
    min-width: 72px;
    padding: 4px 30px 4px 8px;
}
QSpinBox#settingsSpinBox::up-button, QSpinBox#settingsSpinBox::down-button {
    width: 24px;
    background: #f5f5f5;
    border-left: 1px solid #c9c9c9;
}
QSpinBox#settingsSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    border-top-right-radius: 5px;
    border-bottom: 1px solid #d7d7d7;
}
QSpinBox#settingsSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    border-bottom-right-radius: 5px;
}
QSpinBox#settingsSpinBox::up-button:hover, QSpinBox#settingsSpinBox::down-button:hover {
    background: #e3efff;
}
QLineEdit:disabled, QPlainTextEdit:disabled, QTableWidget:disabled, QSpinBox:disabled, QComboBox:disabled {
    background: #f5f5f5;
    color: #777777;
}
QComboBox::drop-down {
    border: 0;
    width: 28px;
}
QComboBox QAbstractItemView {
    background: #ffffff;
    border: 1px solid #c9c9c9;
    color: #252525;
    outline: 0;
    selection-background-color: #e3efff;
    selection-color: #0a4697;
}
QPushButton {
    background: #ffffff;
    border: 1px solid #c9c9c9;
    border-radius: 6px;
    color: #464646;
    font-weight: 600;
    padding: 7px 12px;
}
QPushButton:hover {
    background: #f2f5fa;
    border-color: #7f9fc9;
}
QPushButton:pressed {
    background: #dfe9f7;
}
QPushButton#primaryButton {
    background: #2f6fca;
    border-color: #2f6fca;
    color: #ffffff;
}
QPushButton#primaryButton:hover {
    background: #0a4697;
    border-color: #0a4697;
}
QPushButton#refreshDatasetsButton {
    padding: 0;
}
QPushButton#rawActionButton {
    border-radius: 4px;
    padding: 3px 8px;
}
QPushButton#settingsButton {
    color: #555555;
    font-size: 17px;
    padding: 0;
    min-width: 30px;
    max-width: 30px;
    min-height: 30px;
    max-height: 30px;
}
QPushButton:disabled {
    background: #f3f3f3;
    border-color: #dddddd;
    color: #999999;
}
QCheckBox {
    color: #464646;
    spacing: 7px;
}
QCheckBox::indicator {
    width: 15px;
    height: 15px;
    background: #ffffff;
    border: 1px solid #8b8b8b;
    border-radius: 3px;
}
QCheckBox::indicator:hover {
    border-color: #2f6fca;
    background: #edf4ff;
}
QCheckBox::indicator:checked {
    background: #2f6fca;
    border-color: #2f6fca;
}
QCheckBox::indicator:disabled {
    background: #f3f3f3;
    border-color: #c9c9c9;
}
QListWidget {
    background: #ffffff;
    border: 0;
    outline: 0;
    padding: 2px;
}
QListWidget::item {
    border-radius: 6px;
    margin: 2px 0;
    padding: 9px 10px;
}
QListWidget::item:hover {
    background: #edf4ff;
}
QListWidget::item:selected {
    background: #e3efff;
    color: #0a4697;
    font-weight: 600;
}
QTreeWidget {
    background: #ffffff;
    border: 1px solid #c9c9c9;
    border-radius: 6px;
    color: #252525;
    outline: 0;
    padding: 4px;
}
QTreeWidget::item {
    border-radius: 5px;
    padding: 5px 6px;
}
QTreeWidget::item:hover {
    background: #edf4ff;
}
QTreeWidget::item:selected {
    background: #e3efff;
    color: #0a4697;
}
QTabWidget::pane {
    border: 1px solid #d7d7d7;
    border-radius: 6px;
    top: -1px;
}
QTabBar::tab {
    background: #e5e5e5;
    border: 1px solid #d7d7d7;
    border-bottom: 0;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    color: #666666;
    font-weight: 600;
    margin-right: 4px;
    padding: 7px 14px;
}
QTabBar::tab:selected {
    background: #ffffff;
    color: #0a4697;
}
QTabBar QToolButton {
    background: #ffffff;
    border: 1px solid #c9c9c9;
    border-radius: 5px;
    color: #464646;
    margin: 2px;
    padding: 2px;
}
QTabBar QToolButton:hover {
    background: #edf4ff;
    border-color: #7f9fc9;
    color: #0a4697;
}
QTabBar QToolButton:pressed {
    background: #dceaff;
}
QTabBar QToolButton:disabled {
    background: #f3f3f3;
    border-color: #dddddd;
    color: #999999;
}
QTabBar#pathProfileTabs::tab {
    padding: 3px 10px;
    font-weight: 400;
}
QTabBar#pathProfileTabs::tab:first {
    background: #e7f0fa;
    border-color: #9dbbd9;
    color: #0a4697;
    font-weight: 600;
}
QTabBar#pathProfileTabs::tab:first:selected {
    background: #ffffff;
}
QPushButton#pathProfileAction {
    min-height: 22px;
    max-height: 22px;
    padding: 1px 8px;
}
QPushButton#pathProfileInfo, QPushButton#pathProfileDelete {
    min-width: 22px;
    max-width: 22px;
    min-height: 22px;
    max-height: 22px;
    padding: 0;
}
QPushButton#pathProfileDelete {
    border-color: #dca0a0;
}
QPushButton#pathProfileDelete:hover {
    background: #fdecec;
    border-color: #d97a7a;
}
QPushButton#pathProfileDelete:disabled {
    background: #f5f5f5;
    border-color: #c9c9c9;
}
QTableWidget {
    gridline-color: #dedede;
    padding: 0;
}
QTableWidget::viewport {
    background: #ffffff;
}
QTableWidget::item {
    background: #ffffff;
    border-bottom: 1px solid #ebebeb;
    color: #252525;
    padding: 6px;
}
QTableWidget::item:alternate {
    background: #f7f7f7;
}
QTableWidget::item:selected {
    background: #e3efff;
    color: #0a4697;
}
QTableWidget QLineEdit, QTableWidget QLineEdit:focus {
    background: #ffffff;
    border: 0;
    border-radius: 3px;
    color: #252525;
    min-height: 22px;
    padding: 0 4px;
    selection-background-color: #dceaff;
    selection-color: #0a4697;
}
QHeaderView::section {
    background: #f7f7f7;
    border: 0;
    border-bottom: 1px solid #d7d7d7;
    color: #666666;
    font-weight: 700;
    padding: 7px;
}
QTableCornerButton::section {
    background: #f7f7f7;
    border: 0;
    border-right: 1px solid #d7d7d7;
    border-bottom: 1px solid #d7d7d7;
}
QAbstractScrollArea::corner {
    background: #ffffff;
}
QScrollArea#rawStatisticsScroll {
    background: #ffffff;
    border: 0;
}
QScrollArea#rawStatisticsScroll > QWidget > QWidget {
    background: #ffffff;
}
QScrollBar:vertical {
    background: #f7f7f7;
    width: 10px;
}
QScrollBar::handle:vertical {
    background: #bdbdbd;
    border-radius: 5px;
    min-height: 24px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background: #f7f7f7;
    height: 10px;
}
QScrollBar::handle:horizontal {
    background: #bdbdbd;
    border-radius: 5px;
    min-width: 24px;
}
QScrollBar::handle:horizontal:hover, QScrollBar::handle:vertical:hover {
    background: #777777;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}
QScrollBar::add-page, QScrollBar::sub-page {
    background: transparent;
}
QSplitter::handle {
    background: #f1f1f1;
    width: 8px;
}
QSplitter::handle:hover {
    background: #dceaff;
}
"""


def apply_application_style(app: QApplication) -> None:
    """Apply the shared visual language for the desktop application."""
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(STYLESHEET)
