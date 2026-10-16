from __future__ import annotations

import os
import re
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from qtpy.QtCore import QUrl
from qtpy.QtWidgets import QApplication

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


def test_every_stage_page_links_to_an_existing_guide_section() -> None:
    from nnworkbench.ui.stages.data_stage import DataStage
    from nnworkbench.ui.stages.predict_stage import PredictStage
    from nnworkbench.ui.stages.preprocess_stage import PreprocessStage
    from nnworkbench.ui.stages.train_stage import TrainStage
    from nnworkbench.ui.stages.validate_stage import ValidateStage

    expected = {DataStage: "setup.md", PreprocessStage: "4-train.md", TrainStage: "4-train.md",
                ValidateStage: "5-validate.md", PredictStage: "6-predict.md"}
    for page, file_name in expected.items():
        guide_file, anchor = page.guide
        assert guide_file == file_name, page.__name__
        assert not anchor or anchor in _headings(GUIDE_DIRECTORY / guide_file), f"{page.__name__}: #{anchor}"


def test_show_file_opens_the_file_at_its_anchor() -> None:
    app = QApplication.instance() or QApplication([])
    guide = AnnotationWorkflowGuide()
    guide.resize(700, 500)
    guide.show()
    app.processEvents()

    guide.show_file("4-train.md", "4-preprocess")
    app.processEvents()
    assert guide.current_file() == "4-train.md"
    assert guide.browser.verticalScrollBar().value() > 0

    guide.show_file("unknown.md")
    assert guide.current_file() == "README.md"
    guide.close()
