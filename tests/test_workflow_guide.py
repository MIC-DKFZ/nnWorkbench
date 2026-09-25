from __future__ import annotations

import os
import re
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication

from nnworkbench.ui.workflow_guide import (
    GUIDE_DIRECTORY,
    GUIDE_STAGES,
    AnnotationWorkflowGuide,
    heading_anchor,
)

LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")


def _headings(path: Path) -> set[str]:
    text = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.DOTALL)
    return {heading_anchor(line.lstrip("#")) for line in text.splitlines() if line.startswith("#")}


def test_every_guide_file_is_listed_as_a_stage() -> None:
    assert sorted(path.name for path in GUIDE_DIRECTORY.glob("*.md")) == sorted(name for _, name in GUIDE_STAGES)


@pytest.mark.parametrize("file_name", [name for _, name in GUIDE_STAGES])
def test_guide_links_resolve(file_name: str) -> None:
    source = GUIDE_DIRECTORY / file_name
    for target in LINK.findall(source.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://")):
            continue
        path, _, anchor = target.partition("#")
        linked = (source.parent / path).resolve() if path else source
        assert linked.is_file(), f"{file_name}: broken link {target}"
        if anchor:
            assert anchor in _headings(linked), f"{file_name}: unknown anchor {target}"


def test_heading_anchor_matches_github_style() -> None:
    assert heading_anchor("3.2.5. nnInteractive") == "325-nninteractive"
    assert heading_anchor("Target Structures") == "target-structures"


def test_guide_widget_switches_stages_from_links() -> None:
    app = QApplication.instance() or QApplication([])
    guide = AnnotationWorkflowGuide()
    guide.resize(900, 700)
    guide.show()
    app.processEvents()

    assert guide.current_file() == "README.md"
    assert "Annotation Workflow" in guide.browser.toPlainText()

    guide.stages.setCurrentIndex(3)
    assert guide.current_file() == "2-define.md"
    assert "Class Definitions" in guide.browser.toPlainText()

    guide.browser.anchorClicked.emit(QUrl("3-annotate.md#scribble-annotations"))
    app.processEvents()
    assert guide.current_file() == "3-annotate.md"
    assert guide.stages.currentIndex() == 4
    assert guide.browser.verticalScrollBar().value() > 0
    guide.close()
