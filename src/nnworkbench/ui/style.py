"""Visual language of the application: colours, buttons, cards, tables and badges."""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QFont, QFontDatabase, QFontInfo, QPalette
from qtpy.QtWidgets import QApplication

BG = "#f4f6fa"
PANEL = "#ffffff"
LINE = "#e3e8ef"
MUTED = "#64748b"
TEXT = "#1e293b"
BLUE = "#1f5fbf"
BLUE_SOFT = "#e8f0fc"
GREEN = "#2f8f4e"
GREEN_SOFT = "#e6f4ea"
ORANGE = "#d9822b"
ORANGE_SOFT = "#fdf0e1"
RED = "#c2413b"
RED_SOFT = "#fbe9e8"
PURPLE = "#6b4fbb"
PURPLE_SOFT = "#efeafb"
MONO = "DejaVu Sans Mono"  # replaced at start-up by an installed fixed-width font (see apply_application_style)
CHECK_ICON = (Path(__file__).parent.parent / "assets" / "check.svg").as_posix()

TONES = {
    "green": (GREEN_SOFT, GREEN),
    "blue": (BLUE_SOFT, BLUE),
    "orange": (ORANGE_SOFT, ORANGE),
    "red": (RED_SOFT, RED),
    "gray": ("#eef1f5", MUTED),
    "purple": (PURPLE_SOFT, PURPLE),
}

_BADGES = "\n".join(
    f'QLabel#badge[tone="{tone}"] {{ background: {soft}; color: {strong}; }}'
    for tone, (soft, strong) in TONES.items()
)
_TIPS = "\n".join(
    f'QLabel#tip[tone="{tone}"] {{ background: {soft}; color: {strong}; }}'
    for tone, (soft, strong) in TONES.items()
)

def _stylesheet() -> str:
    return f"""
QMainWindow, QDialog {{ background: {BG}; }}
QWidget {{ color: {TEXT}; font-size: 13px; }}
QWidget#appRoot, QWidget#page, QStackedWidget#pages {{ background: {BG}; }}
QLabel {{ background: transparent; }}
QLabel[muted="true"] {{ color: {MUTED}; }}
QLabel[mono="true"] {{ font-family: "{MONO}"; font-size: 12px; }}
QLabel#h1 {{ font-size: 22px; font-weight: 700; }}
QLabel#h2 {{ font-size: 15px; font-weight: 700; }}
QLabel#h3 {{ font-size: 13px; font-weight: 700; }}
QLabel#sectionLabel {{ color: {MUTED}; font-size: 11px; font-weight: 700; letter-spacing: 1px; }}
QLabel#bigNumber {{ font-size: 28px; font-weight: 700; }}

QFrame#card {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 10px; }}
QFrame#card[active="true"] {{ border: 1px solid {BLUE}; }}
QFrame#cardHeader {{ background: transparent; border: 0; border-bottom: 1px solid {LINE}; }}
QFrame#cardBand {{ background: #fafbfd; border: 0; border-bottom: 1px solid {LINE}; }}
QFrame#separator {{ background: {LINE}; border: 0; max-height: 1px; min-height: 1px; }}
QFrame#appBar, QFrame#stageBar, QFrame#footer {{ background: {PANEL}; border: 0; }}
QFrame#appBar, QFrame#stageBar {{ border-bottom: 1px solid {LINE}; }}
QFrame#footer {{ border-top: 1px solid {LINE}; }}

QPushButton {{
    background: {PANEL}; border: 1px solid {LINE}; border-radius: 8px;
    padding: 7px 14px; font-weight: 600; color: {TEXT};
}}
QPushButton:hover {{ background: #f6f8fb; }}
QPushButton:disabled {{ color: #a3adbb; background: #fafbfc; }}
QPushButton[primary="true"] {{ background: {BLUE}; border-color: {BLUE}; color: white; }}
QPushButton[primary="true"]:hover {{ background: #1a54aa; }}
QPushButton[primary="true"]:disabled {{ background: #9db7e0; border-color: #9db7e0; color: #eef3fb; }}
QPushButton[ghost="true"] {{ background: transparent; border-color: transparent; color: {MUTED}; }}
QPushButton[ghost="true"]:hover {{ background: #eef1f5; color: {TEXT}; }}
QPushButton[small="true"] {{ padding: 4px 10px; font-size: 12px; }}
QPushButton[purple="true"] {{ color: {PURPLE}; border-color: #c9b9ef; }}
QPushButton[danger="true"] {{ color: {RED}; }}
QPushButton#iconButton {{ padding: 0; min-width: 30px; max-width: 30px; min-height: 26px; max-height: 26px; border-radius: 7px; }}
QPushButton#iconButton:checked {{ background: {BLUE}; border-color: {BLUE}; color: white; }}
QPushButton#toggleOption {{ border-radius: 0; border: 0; border-right: 1px solid {LINE}; padding: 5px 12px; color: {MUTED}; font-size: 12px; }}
QPushButton#toggleOption:checked {{ background: {BLUE_SOFT}; color: {BLUE}; }}
QFrame#toggleGroup {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 8px; }}
QPushButton#choice {{ padding: 4px 8px; min-width: 18px; border-radius: 6px; font-size: 12px; }}
QPushButton#choice:checked {{ background: {BLUE_SOFT}; border-color: {BLUE}; color: {BLUE}; }}
QPushButton#choice:disabled {{ border-style: dashed; color: #b6c0cd; }}
QPushButton#foldButton {{ padding: 0; border-radius: 7px; font-size: 12px; }}
QPushButton#foldButton:checked {{ background: {BLUE}; border-color: {BLUE}; color: white; }}
QPushButton#link {{ background: transparent; border: 0; color: {BLUE}; padding: 2px 4px; font-weight: 600; }}
QPushButton#backButton {{ background: transparent; border: 0; color: {MUTED}; padding: 7px 10px; }}
QPushButton#backButton:hover {{ background: #f1f4f8; color: {TEXT}; }}
QPushButton#switcherButton {{ text-align: left; padding: 6px 12px; min-width: 300px; border-radius: 9px; }}
QPushButton#switcherButton:checked {{ border-color: {BLUE}; }}

QLineEdit, QSpinBox, QComboBox, QPlainTextEdit {{
    background: {PANEL}; border: 1px solid {LINE}; border-radius: 8px; padding: 6px 9px;
    selection-background-color: {BLUE_SOFT}; selection-color: {TEXT};
}}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{ border-color: {BLUE}; }}
QLineEdit[mono="true"], QComboBox[mono="true"] {{ font-family: "{MONO}"; font-size: 12px; }}
QLineEdit:read-only {{ background: #fbfcfe; }}
QComboBox::drop-down {{ border: 0; width: 22px; }}
QComboBox QAbstractItemView {{ background: {PANEL}; border: 1px solid {LINE}; selection-background-color: {BLUE_SOFT}; }}
QPlainTextEdit#codeView, QPlainTextEdit#commandView {{
    background: #fafbfd; font-family: "{MONO}"; font-size: 12px; color: #334155;
}}
QPlainTextEdit#consoleLog {{
    background: #0f172a; color: #cbd5e1; border: 0; border-radius: 8px;
    font-family: "{MONO}"; font-size: 12px; padding: 8px 10px;
}}
QCheckBox {{ spacing: 8px; background: transparent; }}
QCheckBox::indicator {{ width: 15px; height: 15px; border: 1.5px solid #cbd5e1; border-radius: 4px; background: {PANEL}; }}
QCheckBox::indicator:checked {{ background: {BLUE}; border-color: {BLUE}; image: url("{CHECK_ICON}"); }}

QTableWidget, QTreeWidget, QListWidget {{
    background: {PANEL}; border: 0; gridline-color: transparent; outline: 0;
    selection-background-color: {BLUE_SOFT}; selection-color: {TEXT};
}}
QTableWidget::item, QTreeWidget::item {{ border-bottom: 1px solid #f0f3f7; padding: 6px 8px; }}
QTableWidget::item:selected, QTreeWidget::item:selected {{ background: {BLUE_SOFT}; color: {TEXT}; }}
QListWidget::item {{ border-bottom: 1px solid #f3f5f8; }}
QListWidget::item:selected {{ background: {BLUE_SOFT}; }}
QHeaderView {{ background: {PANEL}; border: 0; }}
QHeaderView::section {{
    background: {PANEL}; color: {MUTED}; border: 0; border-bottom: 1px solid {LINE};
    padding: 8px 10px; font-size: 11px; font-weight: 700;
}}
QScrollArea {{ background: transparent; border: 0; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 9px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #cfd7e2; border-radius: 4px; min-height: 30px; }}
QScrollBar:horizontal {{ background: transparent; height: 9px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #cfd7e2; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}

QLabel#badge {{ border-radius: 9px; padding: 2px 9px; font-size: 11px; font-weight: 600; }}
{_BADGES}
QLabel#tip {{ border-radius: 8px; padding: 9px 11px; }}
{_TIPS}
QLabel#dsid {{
    background: #eef2f7; color: #475569; border-radius: 6px; padding: 2px 6px;
    font-family: "{MONO}"; font-size: 11px; font-weight: 700;
}}
QLabel#dsid[current="true"] {{ background: {BLUE}; color: white; }}
QLabel#availOn {{ background: {BLUE_SOFT}; color: {BLUE}; border-radius: 5px; padding: 1px 6px; font-size: 10px; font-weight: 700; min-width: 24px; }}
QLabel#availOff {{ background: #eef1f5; color: #b6c0cd; border-radius: 5px; padding: 1px 6px; font-size: 10px; font-weight: 700; min-width: 24px; }}
QLabel#chip {{ border: 1px solid {LINE}; background: #fafbfd; border-radius: 10px; padding: 2px 9px; font-size: 12px; }}
QLabel#chip[ignore="true"] {{ border-style: dashed; color: {MUTED}; }}
QLabel#foldChip {{ border-radius: 7px; padding: 0; font-size: 12px; font-weight: 700; qproperty-alignment: AlignCenter; }}
QLabel#foldChip[state="done"] {{ background: #d6efdf; color: {GREEN}; }}
QLabel#foldChip[state="training"] {{ background: #fbe0c4; color: #b8661c; }}
QLabel#foldChip[state="stopped"] {{ background: #fff; border: 1px dashed {ORANGE}; color: #b8661c; }}
QLabel#foldChip[state="reduced"] {{ background: #e2e8f0; color: #475569; }}
QLabel#jobChip {{ border: 1px solid {LINE}; border-radius: 10px; padding: 2px 10px; font-size: 12px; font-weight: 600; color: {MUTED}; }}
QLabel#jobChip[active="true"] {{ border-color: {BLUE}; background: {BLUE_SOFT}; color: {BLUE}; }}
QFrame#infoBox {{ background: white; border: 1px solid {LINE}; border-radius: 8px; }}
QLabel#jobChip[failed="true"] {{ border-color: {RED}; background: {RED_SOFT}; color: {RED}; }}
QLabel[failed="true"] {{ color: {RED}; }}
QLabel#statusYes {{ color: {GREEN}; font-weight: 700; }}
QLabel#statusNo {{ color: {RED}; font-weight: 700; }}

QFrame#howto {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 10px; }}
QFrame#popup {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 12px; }}
QTextBrowser#annotationGuide {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 6px; font-size: 14px; }}
QTabBar::tab {{ background: transparent; color: {MUTED}; font-weight: 600; padding: 7px 12px; border: 0; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {BLUE}; border-bottom-color: {BLUE}; }}
QTabWidget::pane {{ border: 0; border-top: 1px solid {LINE}; }}
QMenu {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 8px; padding: 4px; }}
QMenu::item {{ background: transparent; color: {TEXT}; padding: 6px 22px 6px 12px; border-radius: 6px; }}
QMenu::item:selected {{ background: {BLUE_SOFT}; color: {BLUE}; }}
QMenu::item:disabled {{ color: #a3adbb; }}
QMenu::separator {{ height: 1px; background: {LINE}; margin: 4px 6px; }}
QToolTip {{ background: #1e293b; color: white; border: 0; padding: 5px 8px; }}
"""


STYLESHEET = _stylesheet()


def _installed(family: str) -> bool:
    try:
        families = QFontDatabase.families()  # static in Qt 6
    except TypeError:
        families = QFontDatabase().families()  # instance method in Qt 5
    return family in families


def monospace_family() -> str:
    """DejaVu Sans Mono where installed (Linux), otherwise the platform's fixed-width font (Consolas, Menlo, …)."""
    if _installed("DejaVu Sans Mono"):
        return "DejaVu Sans Mono"
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    return font.family() if QFontInfo(font).fixedPitch() else "Courier New"


def _use_light_colours(app: QApplication) -> None:
    """Light colours on every system. The stylesheet is made for them; with a system dark mode (macOS, Windows,
    some Linux desktops) Qt 6 would otherwise use a dark palette, giving dark text on dark backgrounds."""
    hints = app.styleHints()
    if hasattr(hints, "setColorScheme"):  # Qt 6.8+: also native parts such as title bars and file dialogs
        hints.setColorScheme(Qt.ColorScheme.Light)
    palette = QPalette()
    for role, colour in ((QPalette.ColorRole.Window, BG), (QPalette.ColorRole.WindowText, TEXT),
                         (QPalette.ColorRole.Base, PANEL), (QPalette.ColorRole.AlternateBase, "#f8fafc"),
                         (QPalette.ColorRole.Text, TEXT), (QPalette.ColorRole.Button, PANEL),
                         (QPalette.ColorRole.ButtonText, TEXT), (QPalette.ColorRole.BrightText, "#ffffff"),
                         (QPalette.ColorRole.Highlight, BLUE), (QPalette.ColorRole.HighlightedText, "#ffffff"),
                         (QPalette.ColorRole.ToolTipBase, TEXT), (QPalette.ColorRole.ToolTipText, "#ffffff"),
                         (QPalette.ColorRole.PlaceholderText, "#94a3b8"), (QPalette.ColorRole.Link, BLUE)):
        palette.setColor(role, QColor(colour))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor("#94a3b8"))
    app.setPalette(palette)


def apply_application_style(app: QApplication) -> None:
    """Apply the shared visual language for the desktop application."""
    global MONO
    app.setStyle("Fusion")
    _use_light_colours(app)
    # Noto Sans where installed (most Linux desktops), otherwise the platform's UI font (Segoe UI, SF Pro, …).
    font = QFont("Noto Sans", 10) if _installed("Noto Sans") else QFontDatabase.systemFont(
        QFontDatabase.SystemFont.GeneralFont)
    font.setPointSize(10)
    app.setFont(font)
    MONO = monospace_family()
    app.setStyleSheet(_stylesheet())
