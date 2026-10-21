import json
from pathlib import Path

import pytest

from nnworkbench.services.dataset_creator import create_next_iteration
from nnworkbench.services.dataset_description import (
    IMAGES_FOLDER,
    add_image,
    create_description,
    description_path,
    description_template,
)


def _dataset(root: Path, labels: dict) -> Path:
    folder = root / "Dataset042_Cells"
    (folder / "imagesTr").mkdir(parents=True)
    (folder / "labelsTr").mkdir()
    for case in ("cells_001", "cells_002"):
        (folder / "imagesTr" / f"{case}_0000.nii.gz").touch()
    (folder / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "fluorescence"}, "labels": labels, "file_ending": ".nii.gz"}), encoding="utf-8")
    return folder


def test_template_fills_in_facts_and_one_section_per_class(tmp_path: Path) -> None:
    sparse = description_template(_dataset(tmp_path / "a", {"background": 0, "cell": 1, "nucleus": 2, "ignore": 3}))
    assert "Images in imagesTr: 2" in sparse and "| 0 | fluorescence |" in sparse and ".nii.gz" in sparse
    assert sparse.index("| 0 | fluorescence |") < sparse.index("## 2. Describe Target Structures")
    assert "### 1. cell" in sparse and "### 2. nucleus" in sparse and "### 3. ignore" not in sparse
    assert "sparse: only parts of each image are annotated" in sparse
    dense = description_template(_dataset(tmp_path / "b", {"background": 0, "liver": 1}))
    assert "dense: every voxel gets a class" in dense and "### 0. background" in dense
    headings = [line for line in sparse.split("\n") if line.startswith("## ")]
    assert headings == ["## 1. Describe Images", "## 2. Describe Target Structures", "## 3. Define Classes",
                        "## Notes (optional)"]
    assert sparse.index("<!-- field: type -->") < sparse.index("<!-- channels -->") < sparse.index(
        "<!-- field: origin -->")
    cell = sparse[sparse.index("### 1. cell"):sparse.index("### 2. nucleus")]
    assert [line.split(":**")[0] for line in cell.split("\n") if line.startswith("**")] == [
        "**Description", "**Visual Description", "**Labeling Instructions (optional)", "**Example Images"]


def test_create_never_replaces_and_new_iterations_copy_it(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    source = _dataset(raw, {"background": 0, "cell": 1})
    path = create_description(source)
    assert path == source / "nnworkbench" / "dataset-description.md" and (path.parent / IMAGES_FOLDER).is_dir()
    path.write_text("# my notes\n", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        create_description(source)
    assert path.read_text(encoding="utf-8") == "# my notes\n"

    picture = tmp_path / "cell example.png"; picture.write_bytes(b"png")
    assert add_image(source, picture) == f"{IMAGES_FOLDER}/cell_example.png"  # no spaces in Markdown links
    other = tmp_path / "other"; other.mkdir(); (other / "cell example.png").write_bytes(b"other")
    assert add_image(source, other / "cell example.png") == f"{IMAGES_FOLDER}/cell_example_2.png"  # no overwrite

    (path.parent / "project-notes.md").write_text("decisions", encoding="utf-8")
    (path.parent / "class_distribution.json").write_text("{}", encoding="utf-8")

    # a new round gets the description, its images and the notes, not the statistics of the old labels
    copy = create_next_iteration(raw, [raw], source, 43, "Cells_iter1")
    assert description_path(copy).read_text(encoding="utf-8") == "# my notes\n"
    assert (description_path(copy).parent / IMAGES_FOLDER / "cell_example_2.png").is_file()
    assert (copy / "nnworkbench" / "project-notes.md").is_file()
    assert not (copy / "nnworkbench" / "class_distribution.json").exists()


def test_form_answers_round_trip_and_keep_text_added_by_hand(tmp_path: Path) -> None:
    from nnworkbench.services.dataset_description import Answer, read_answers, save_answers

    folder = _dataset(tmp_path, {"background": 0, "cell": 1, "ignore": 2})
    path = create_description(folder)
    assert read_answers(path.read_text(encoding="utf-8"), folder)["type"] == Answer()  # placeholders read as empty
    older = path.read_text(encoding="utf-8").replace("**Visual Description:**", "**Visual description:**")
    assert read_answers(older, folder)["class:cell:visual"] == Answer()  # also under an older label
    path.write_text(path.read_text(encoding="utf-8") + "\nA paragraph written by hand.\n", encoding="utf-8")

    answers = {"type": Answer("Confocal microscopy.\n\nTwo sessions."),
               "class:cell:examples": Answer("", (("dividing cell", "description_images/dividing.png"),))}
    save_answers(folder, answers)
    text = path.read_text(encoding="utf-8")
    assert "**Type:** Confocal microscopy.\n\nTwo sessions." in text
    assert "![dividing cell](description_images/dividing.png)" in text
    assert "A paragraph written by hand." in text  # untouched
    read = read_answers(text, folder)
    assert read["type"].text == "Confocal microscopy.\n\nTwo sessions."
    assert read["class:cell:examples"].images == (("dividing cell", "description_images/dividing.png"),)

    # the facts are refreshed on saving; the class table is the description's own (dataset.json changes show
    # as a difference until they are imported)
    (folder / "dataset.json").write_text(json.dumps({"labels": {"background": 0, "cell": 1, "nucleus": 2,
                                                                "ignore": 3}}), encoding="utf-8")
    (folder / "imagesTr" / "cells_003_0000.nii.gz").touch()
    save_answers(folder, read_answers(path.read_text(encoding="utf-8"), folder))
    text = path.read_text(encoding="utf-8")
    assert "Images in imagesTr: 3" in text and "### 2. nucleus" not in text
    from nnworkbench.services.dataset_description import import_classes

    text = import_classes(text, folder)
    assert "### 2. nucleus" in text and "| 2 | nucleus |" in text
    assert text.index("### 2. nucleus") < text.index("## Notes (optional)")
    assert "Confocal microscopy." in text and "A paragraph written by hand." in text


def test_class_table_compare_import_and_push(tmp_path: Path) -> None:
    from nnworkbench.services.dataset_description import (
        class_differences,
        class_table_problems,
        description_classes,
        import_classes,
        push_classes,
        save_answers,
    )

    folder = _dataset(tmp_path, {"background": 0, "cell": 1, "ignore": 2})
    path = create_description(folder)
    assert description_classes(path.read_text(encoding="utf-8")) == [(0, "background"), (1, "cell"), (2, "ignore")]

    edited = [(0, "background"), (1, "cell"), (2, "nucleus"), (3, "ignore")]
    save_answers(folder, {}, edited)
    text = path.read_text(encoding="utf-8")
    assert description_classes(text) == edited and "### 2. nucleus" in text
    differences = class_differences(edited, [(0, "background"), (1, "cell"), (2, "ignore")])
    assert differences == ["nucleus (2) is only in the description", "ignore: ID 3 in the description, 2 in dataset.json"]

    assert class_table_problems([(0, "background"), (2, "cell")]) == ["IDs must be continuous, starting at 0."]
    assert "ignore must have the highest ID." in class_table_problems([(0, "background"), (1, "ignore"), (2, "x")])
    with pytest.raises(ValueError, match="unique"):
        push_classes(folder, [(0, "cell"), (1, "Cell")])

    push_classes(folder, edited)
    manifest = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
    assert manifest["labels"] == {"background": 0, "cell": 1, "nucleus": 2, "ignore": 3}
    assert manifest["channel_names"] == {"0": "fluorescence"}  # everything else stays

    (folder / "dataset.json").write_text(json.dumps({**manifest, "labels": {"background": 0, "cell": 1}}),
                                         encoding="utf-8")
    text = import_classes(path.read_text(encoding="utf-8"), folder)
    assert description_classes(text) == [(0, "background"), (1, "cell")]
    assert "### 2. nucleus" not in text  # the class sections follow the table


def test_class_sections_follow_the_table_and_renames_keep_their_text(tmp_path: Path) -> None:
    from nnworkbench.services.dataset_description import (
        Answer,
        classes_losing_text,
        import_classes,
        read_answers,
        renames_by_id,
        save_answers,
    )

    folder = _dataset(tmp_path, {"background": 0, "cell": 1, "ignore": 2})
    path = create_description(folder)
    save_answers(folder, {"class:cell:visual": Answer("Bright membrane.")})
    path.write_text(path.read_text(encoding="utf-8").replace(
        "<!-- classes-end -->", "A note on cells written by hand.\n\n<!-- classes-end -->"), encoding="utf-8")

    # renamed in the form: the text moves to the new name, the hand-written note stays with the class
    save_answers(folder, {}, [(0, "background"), (1, "cells"), (2, "nucleus"), (3, "ignore")],
                 renames={"class:cells": "class:cell"})
    text = path.read_text(encoding="utf-8")
    assert "### 1. cell\n" not in text and "### 1. cells" in text and "### 2. nucleus" in text
    assert read_answers(text, folder)["class:cells:visual"].text == "Bright membrane."
    assert text.index("A note on cells written by hand.") < text.index("### 2. nucleus")
    assert text.index("### 2. nucleus") < text.index("<!-- classes-end -->") < text.index("## Notes (optional)")

    # dataset.json renamed a class under the same ID: import keeps its text; removing a described class is reported
    assert renames_by_id([(1, "cells")], [(1, "cell")]) == {"class:cell": "class:cells"}
    defined = [(0, "background"), (1, "cell"), (2, "ignore")]
    assert classes_losing_text(text, defined, renames_by_id([(1, "cells"), (2, "nucleus")], defined)) == []
    (folder / "dataset.json").write_text(json.dumps({"labels": {name: value for value, name in defined}}),
                                         encoding="utf-8")
    text = import_classes(text, folder)
    assert read_answers(text, folder)["class:cell:visual"].text == "Bright membrane." and "nucleus" not in text
    assert classes_losing_text(text, [(0, "background")]) == ["1. cell"]


def test_channel_table_compare_import_and_push(tmp_path: Path) -> None:
    from nnworkbench.services.dataset_description import (
        Answer,
        channel_differences,
        channel_table_problems,
        description_channels,
        import_channels,
        push_channels,
        read_answers,
        save_answers,
    )

    folder = _dataset(tmp_path, {"background": 0, "cell": 1})
    path = create_description(folder)
    assert description_channels(path.read_text(encoding="utf-8")) == [(0, "fluorescence")]
    edited = [(0, "membrane"), (1, "DAPI")]
    save_answers(folder, {"type": Answer("Confocal.")}, channels=edited)
    text = path.read_text(encoding="utf-8")
    assert description_channels(text) == edited and read_answers(text, folder)["type"].text == "Confocal."
    assert channel_differences(edited, [(0, "fluorescence")]) == [
        "channel 0: membrane in the description, fluorescence in dataset.json", "channel 1 (DAPI) is only in the description"]
    assert channel_table_problems([(0, "a"), (2, "b")]) == ["Channels must be numbered continuously, starting at 0."]
    assert channel_table_problems([(0, " ")]) == ["Every channel needs a name."]

    push_channels(folder, edited)
    manifest = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
    assert manifest["channel_names"] == {"0": "membrane", "1": "DAPI"} and manifest["labels"] == {"background": 0, "cell": 1}
    (folder / "dataset.json").write_text(json.dumps({**manifest, "channel_names": {"0": "CT"}}), encoding="utf-8")
    assert description_channels(import_channels(text, folder)) == [(0, "CT")]


def test_questions_of_an_earlier_template_still_read_as_unanswered(tmp_path: Path) -> None:
    from nnworkbench.services.dataset_description import FORMER_QUESTIONS, Answer, read_answers

    folder = _dataset(tmp_path, {"background": 0, "cell": 1})
    text = create_description(folder).read_text(encoding="utf-8")
    start = text.index("<!-- field: class:cell:visual -->")
    end = text.index("<!-- end -->", start)
    older = text[:start] + f"<!-- field: class:cell:visual -->\n**Visual description:** *{FORMER_QUESTIONS[0]}*\n" + text[end:]
    assert read_answers(older, folder)["class:cell:visual"] == Answer()
    # Variation and Frequency of section 2 with their earlier wording
    for key, label in (("class_variation", "Variation"), ("frequency", "Frequency")):
        start = text.index(f"<!-- field: {key} -->")
        end = text.index("<!-- end -->", start)
        text = text[:start] + f"<!-- field: {key} -->\n**{label}:** *A question of an older template?*\n" + text[end:]
    assert read_answers(text, folder)["class_variation"] == Answer() == read_answers(text, folder)["frequency"]


def test_a_question_added_to_the_template_later_is_inserted_on_saving(tmp_path: Path) -> None:
    from nnworkbench.services.dataset_description import Answer, read_answers, save_answers

    folder = _dataset(tmp_path, {"background": 0, "cell": 1})
    path = create_description(folder)
    text = path.read_text(encoding="utf-8")
    start = text.index("<!-- field: classes -->")
    path.write_text(text[:start] + text[text.index("<!-- end -->", start) + len("<!-- end -->"):].lstrip("\n"),
                    encoding="utf-8")
    save_answers(folder, {"classes": Answer("cell and nucleus; dividing cells are not a class of their own.")})
    text = path.read_text(encoding="utf-8")
    assert read_answers(text, folder)["classes"].text.startswith("cell and nucleus")
    assert text.index("<!-- field: structures -->") < text.index("<!-- field: classes -->") < text.index(
        "<!-- field: class_variation -->")


def test_why_these_structures_follows_the_structures_of_interest(tmp_path: Path) -> None:
    from nnworkbench.services.dataset_description import Answer, read_answers

    folder = _dataset(tmp_path, {"background": 0, "cell": 1})
    text = create_description(folder).read_text(encoding="utf-8")
    assert text.index("<!-- field: structures -->") < text.index("<!-- field: purpose -->") < text.index(
        "<!-- field: classes -->")
    # the combined question of the earlier template still reads as unanswered
    start = text.index("<!-- field: structures -->")
    end = text.index("<!-- end -->", start)
    older = (text[:start] + "<!-- field: structures -->\n**Structures of interest:** "
             "*What do you want to find or measure, and why?*\n" + text[end:])
    assert read_answers(older, folder)["structures"] == Answer()


def test_tables_end_with_a_blank_line_so_markdown_adds_no_empty_row(tmp_path: Path) -> None:
    from nnworkbench.services.dataset_description import channel_table_block, class_table_block

    for block in (channel_table_block([(0, "CT")]), class_table_block([(0, "background"), (1, "liver")])):
        assert block.endswith("|\n\n<!-- end -->")
