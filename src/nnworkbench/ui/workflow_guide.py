"""Interactive, stage-by-stage viewer for the bundled annotation-workflow Markdown guide."""

from __future__ import annotations

import re
from pathlib import Path

from qtpy.QtCore import Qt, QTimer, QUrl
from qtpy.QtGui import (
    QColor,
    QDesktopServices,
    QFont,
    QFontDatabase,
    QFontInfo,
    QImage,
    QTextBlock,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
    QTextFormat,
    QTextTable,
)
from qtpy.QtWidgets import QTabBar, QTextBrowser, QVBoxLayout, QWidget

GUIDE_DIRECTORY = Path(__file__).parent.parent / "assets" / "annotation_workflow"
GUIDE_STAGES = (
    ("Overview", "README.md"),
    ("Setup", "setup.md"),
    ("1. Describe", "1-describe.md"),
    ("2. Define", "2-define.md"),
    ("3. Annotate", "3-annotate.md"),
    ("4. Train", "4-train.md"),
    ("5. Validate", "5-validate.md"),
    ("6. Predict", "6-predict.md"),
)
CODE_BLOCK_BACKGROUND = QColor("#f3f5f8")
TAGLINE_COLOR = QColor("#0a4697")
DOCUMENT_MARGIN = 20


def _monospace_font() -> QFont:
    """Return the system's fixed-width font, or a generic monospace font if the platform reports none."""
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    if not QFontInfo(font).fixedPitch():
        font = QFont("monospace")
        font.setStyleHint(QFont.StyleHint.TypeWriter)
    return font


def _enum_value(value) -> int:
    """Return a Qt enum's integer value; Qt 6 bindings use Python enums, Qt 5 bindings plain ints."""
    return int(getattr(value, "value", value))


def heading_anchor(text: str) -> str:
    """Return the GitHub-style anchor for a Markdown heading, so in-page links work in both places."""
    return re.sub(r"[^\w\- ]", "", text.strip().lower()).replace(" ", "-")


class GuideBrowser(QTextBrowser):
    """Markdown browser that scales images down to the available width."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._rendered_width = 0
        self._resize_timer = QTimer(self); self._resize_timer.setSingleShot(True); self._resize_timer.setInterval(150)
        self._resize_timer.timeout.connect(self._reflow_images)

    def load(self, path: Path) -> None:
        self.setSource(QUrl.fromLocalFile(str(path)), QTextDocument.ResourceType.MarkdownResource)
        self._rendered_width = self._image_width()
        self.document().setDocumentMargin(DOCUMENT_MARGIN)
        self._format_blocks()
        self._apply_code_font()
        self._collapse_table_borders()

    def loadResource(self, resource_type: int, url: QUrl):
        if _enum_value(resource_type) == _enum_value(QTextDocument.ResourceType.ImageResource):
            image = QImage(self.source().resolved(url).toLocalFile())
            if not image.isNull():
                width = self._image_width()
                if image.width() > width:
                    image = image.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)
                return image
        return super().loadResource(resource_type, url)

    def scroll_to_anchor(self, anchor: str) -> bool:
        block = self.document().begin()
        while block.isValid():
            if block.blockFormat().headingLevel() and heading_anchor(block.text()) == anchor:
                self.verticalScrollBar().setValue(int(self.document().documentLayout().blockBoundingRect(block).top()))
                return True
            block = block.next()
        return False

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._resize_timer.start()

    def _image_width(self) -> int:
        # Always reserve the vertical scroll bar so images fit whether or not it is shown.
        available = self.width() - 2 * self.frameWidth() - self.verticalScrollBar().sizeHint().width()
        return max(160, available - 2 * DOCUMENT_MARGIN - 8)

    def _reflow_images(self) -> None:
        if self.source().isEmpty() or abs(self._image_width() - self._rendered_width) < 8:
            return
        scroll_bar = self.verticalScrollBar()
        position = scroll_bar.value() / max(1, scroll_bar.maximum())
        self.load(Path(self.source().toLocalFile()))
        scroll_bar.setValue(round(position * scroll_bar.maximum()))

    def _format_blocks(self) -> None:
        """Add paragraph spacing and shade code blocks, which Qt's Markdown import leaves plain."""
        cursor = QTextCursor(self.document())
        block = self.document().begin()
        while block.isValid():
            block_format = block.blockFormat()
            if block_format.hasProperty(QTextFormat.Property.BlockCodeFence):
                block_format.setBackground(CODE_BLOCK_BACKGROUND)
                block_format.setLeftMargin(block_format.leftMargin() + 8)
                previous, following = block.previous(), block.next()
                if not previous.blockFormat().hasProperty(QTextFormat.Property.BlockCodeFence):
                    block_format.setTopMargin(6)
                if not following.blockFormat().hasProperty(QTextFormat.Property.BlockCodeFence):
                    block_format.setBottomMargin(10)
            elif block_format.headingLevel():
                block_format.setTopMargin(18 if block.previous().isValid() else 0)
                block_format.setBottomMargin(6)
            elif block.previous().blockFormat().headingLevel() and _is_tagline(block):
                block_format.setBottomMargin(16)
                self._style_tagline(block)
            elif block.textList() is not None:
                ends_list = block.next().textList() is None and not block.next().blockFormat().hasProperty(
                    QTextFormat.Property.BlockCodeFence)
                block_format.setBottomMargin(10 if ends_list else 3)
            else:
                block_format.setBottomMargin(10)
            cursor.setPosition(block.position()); cursor.setBlockFormat(block_format)
            block = block.next()
        self.document().setModified(False)

    def _apply_code_font(self) -> None:
        """Show code in a monospace font; Qt 5's Markdown import marks code but keeps the normal font."""
        # Only change the family, so code keeps the size of the surrounding text.
        code = QTextCharFormat()
        family = QFontInfo(_monospace_font()).family()  # the resolved family, e.g. "DejaVu Sans Mono"
        if hasattr(code, "setFontFamilies"):
            code.setFontFamilies([family])
        else:
            code.setFontFamily(family)
        code.setFontFixedPitch(True)
        code.setFontStyleHint(QFont.StyleHint.TypeWriter)
        block = self.document().begin()
        while block.isValid():
            if block.blockFormat().hasProperty(QTextFormat.Property.BlockCodeFence):
                cursor = QTextCursor(block)
                cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
                cursor.mergeCharFormat(code)
            else:
                fragments = block.begin()
                while not fragments.atEnd():
                    fragment = fragments.fragment()
                    if fragment.charFormat().fontFixedPitch():
                        cursor = QTextCursor(self.document())
                        cursor.setPosition(fragment.position())
                        cursor.setPosition(fragment.position() + fragment.length(), QTextCursor.MoveMode.KeepAnchor)
                        cursor.mergeCharFormat(code)
                    fragments += 1
            block = block.next()
        self.document().setModified(False)

    def _collapse_table_borders(self) -> None:
        """Draw single table borders; Qt 6 does this by default, Qt 5 draws double borders."""
        for frame in self.document().rootFrame().childFrames():
            if isinstance(frame, QTextTable):
                table_format = frame.format()
                table_format.setBorderCollapse(True)
                frame.setFormat(table_format)
        self.document().setModified(False)

    def _style_tagline(self, block: QTextBlock) -> None:
        """Highlight a bold-italic subtitle below a heading, e.g. ***Release the model and use it in the wild.***"""
        tagline = QTextCharFormat()
        tagline.setForeground(TAGLINE_COLOR)
        tagline.setFontWeight(QFont.Weight.Bold)
        tagline.setFontItalic(False)
        tagline.setProperty(QTextFormat.Property.FontPixelSize, round(QFontInfo(self.font()).pixelSize() * 1.2))
        cursor = QTextCursor(block)
        cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
        cursor.mergeCharFormat(tagline)


def _is_tagline(block: QTextBlock) -> bool:
    fragments = []
    iterator = block.begin()
    while not iterator.atEnd():
        fragments.append(iterator.fragment()); iterator += 1
    return bool(block.text().strip()) and all(
        fragment.charFormat().fontItalic() and fragment.charFormat().fontWeight() >= QFont.Weight.Bold
        for fragment in fragments)


class AnnotationWorkflowGuide(QWidget):
    """Stage selector plus Markdown view of the annotation workflow instructions."""

    def __init__(self, guide_directory: Path = GUIDE_DIRECTORY, parent=None) -> None:
        super().__init__(parent)
        self._directory = guide_directory
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(0)
        self.stages = QTabBar(); self.stages.setObjectName("guideStages")
        self.stages.setExpanding(False); self.stages.setUsesScrollButtons(True); self.stages.setDrawBase(False)
        for title, _file_name in GUIDE_STAGES:
            self.stages.addTab(title)
        layout.addWidget(self.stages)
        self.browser = GuideBrowser(); self.browser.setObjectName("annotationGuide")
        self.browser.setOpenLinks(False)
        self.browser.anchorClicked.connect(self._open_link)
        layout.addWidget(self.browser, 1)
        self.stages.currentChanged.connect(lambda index: self.show_stage(index))
        self.show_stage(0)

    def show_stage(self, index: int, anchor: str = "") -> None:
        if self.stages.currentIndex() != index:
            self.stages.blockSignals(True); self.stages.setCurrentIndex(index); self.stages.blockSignals(False)
        self.browser.load(self._directory / GUIDE_STAGES[index][1])
        if not anchor or not self.browser.scroll_to_anchor(anchor):
            self.browser.verticalScrollBar().setValue(0)

    def current_file(self) -> str:
        return GUIDE_STAGES[self.stages.currentIndex()][1]

    def _open_link(self, url: QUrl) -> None:
        if url.scheme() in {"http", "https", "mailto"}:
            QDesktopServices.openUrl(url)
            return
        if not url.path():
            self.browser.scroll_to_anchor(url.fragment())
            return
        file_name = Path(self.browser.source().resolved(url).toLocalFile()).name
        for index, (_title, stage_file) in enumerate(GUIDE_STAGES):
            if stage_file == file_name:
                self.show_stage(index, url.fragment())
                return
