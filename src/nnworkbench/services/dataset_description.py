"""dataset-description.md: a template for describing the data and defining the classes (guide: 1. Describe, 2. Define).

Lives in the nnworkbench folder of the raw dataset; images go into description_images/ next to it and are embedded
with relative links. nnU-Net ignores the folder. The file is plain Markdown, readable anywhere; invisible markers
(``<!-- field: key -->`` … ``<!-- end -->``) let the GUI form read and update the answers without touching anything
else. Blocks marked ``<!-- auto: … -->`` are facts nnWorkbench refreshes (images, channels, labels).

Section 1 holds the description's channel table (``<!-- channels -->`` … ``<!-- end -->``), compared with
dataset.json's channel_names. Section 3 (Define Classes) follows the description's own class table (``<!-- classes -->`` … ``<!-- end -->``): one
``### ID. name`` block per class (ignore excepted), rebuilt in table order up to ``<!-- classes-end -->``.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from .dataset_contents import workbench_folder

DESCRIPTION_FILE = "dataset-description.md"
IMAGES_FOLDER = "description_images"


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    question: str
    images: bool = False


IMAGES_SECTION = "1. Describe Images"
STRUCTURES_SECTION = "2. Describe Target Structures"
CLASSES_SECTION = "3. Define Classes"
NOTES_SECTION = "Notes (optional)"
SECTIONS: tuple[tuple[str, tuple[Field, ...]], ...] = (
    (IMAGES_SECTION, (
        Field("type", "Type",
              "What kind of images are these? e.g. 3D confocal fluorescence microscopy of cultured cells."),
        Field("origin", "Number and origin",
              "How many images are there in total, from how many samples, patients or sessions?"),
        Field("variation", "Variation",
              "How much do the images differ in contrast, brightness, noise, resolution, artifacts, background and "
              "acquisition conditions?"),
        Field("representative", "Representative image", "Which image is typical, and why?", images=True),
        Field("difficult", "Most difficult image",
              "Which is the hardest image the model should still handle, and what makes it difficult?", images=True),
    )),
    (STRUCTURES_SECTION, (
        Field("structures", "Structures of interest", "Which structures do you want to find in the images?"),
        Field("purpose", "Why these structures",
              "Why are you interested in them: what do you want to measure, count or compare, and which question "
              "should that answer? This decides which classes you need and how exact their boundaries must be."),
        Field("classes", "Classes",
              "Which semantic classes do you need? Each class must be distinguishable from the others in the image; "
              "merge what cannot be told apart."),
        Field("class_variation", "Variation",
              "How much do the structures differ in size, shape, appearance, orientation, position and context?"),
        Field("frequency", "Frequency",
              "How often does each structure occur, and in how many images? Are some rare?"),
    )),
)
CLASS_FIELDS = (
    Field("description", "Description", "What is the structure?"),
    Field("visual", "Visual Description",
          "What does it look like in the images (brightness, texture, shape, size), and where is its boundary?"),
    Field("instructions", "Labeling Instructions (optional)",
          "How is it annotated (nnInteractive or brush), and how are unclear, unusual or touching objects handled?"),
    Field("examples", "Example Images", "One or two typical examples and a difficult case.", images=True),
)
# questions of earlier templates: still read as an unanswered field
FORMER_QUESTIONS = (
    ("What does it look like? What is included or excluded, where exactly is the boundary, and how are unclear or "
     "unusual cases handled?"),
    "How is it annotated: nnInteractive or brush, and what to watch out for?",
    "How much do objects of a class differ in size, shape, appearance, orientation, position and context?",
    "How often does each class occur, and in how many images? Are some classes rare?",
    "Open questions, decisions made while annotating, and what changed between annotation rounds.",
    "What do you want to find or measure, and why?",
)
# a whole answer that is one italic question is a placeholder, also of a template version not listed above
_ITALIC_QUESTION = re.compile(r"^\*[^*\n]+\?\*$")
NOTES_FIELD = Field("notes", "Notes", "Anything else worth writing down, e.g. open questions or decisions.")
_FORMER_HEADINGS = {"## Notes and Decisions": f"## {NOTES_SECTION}"}

_LABEL = re.compile(r"^\*\*[^*\n]+:\*\*")
_IMAGE_LINE = re.compile(r"^!\[(?P<caption>[^\]]*)\]\((?P<path>[^)]+)\)\s*$")
_FIELD = re.compile(r"<!-- field: (?P<key>\S+) -->\n(?P<body>.*?)<!-- end -->", re.DOTALL)
_AUTO = re.compile(r"<!-- auto: (?P<key>\S+) -->\n.*?<!-- end -->", re.DOTALL)
_CLASSES_END = "<!-- classes-end -->"
_CLASS_TABLE = re.compile(r"<!-- classes -->\n.*?<!-- end -->", re.DOTALL)
_CHANNEL_TABLE = re.compile(r"<!-- channels -->\n.*?<!-- end -->", re.DOTALL)
_CHANNEL_NOTE = "*Channels: the file ending `_0000`, `_0001`, … of each image. The table is compared with dataset.json.*"
CHANNELS_AFTER = "type"  # the channel table follows this field
_CLASS_HEADING = re.compile(r"(?m)^(?=### )")
_CLASS_KEY = re.compile(r"<!-- field: (?P<key>class:[^:\s]+):")
_TABLE_ROW = re.compile(r"^\|\s*(?P<id>\d+)\s*\|\s*(?P<name>[^|]*?)\s*\|\s*$")


@dataclass
class Answer:
    text: str = ""
    images: tuple[tuple[str, str], ...] = ()  # (caption, relative path)


def description_path(dataset_directory: Path) -> Path:
    return workbench_folder(dataset_directory) / DESCRIPTION_FILE


# ---------------------------------------------------------------- facts from the dataset


def _manifest(dataset_directory: Path) -> dict:
    try:
        data = json.loads((dataset_directory / "dataset.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def dataset_classes(dataset_directory: Path) -> list[tuple[int, str]]:
    """Classes from dataset.json in ID order, ignore included."""
    labels = _manifest(dataset_directory).get("labels")
    if not isinstance(labels, dict):
        return []
    return sorted(((value, str(name)) for name, value in labels.items()
                   if isinstance(value, int) and not isinstance(value, bool)), key=lambda item: item[0])


def dataset_channels(dataset_directory: Path) -> list[tuple[int, str]]:
    """Channels from dataset.json's channel_names in index order."""
    channels = _manifest(dataset_directory).get("channel_names")
    if not isinstance(channels, dict):
        return []
    return sorted((int(key), str(name)) for key, name in channels.items() if str(key).isascii() and str(key).isdigit())


def _is_ignore(name: str) -> bool:
    """The ignore label as nnU-Net reads it: exactly "ignore" ("Ignore" is an ordinary class there)."""
    return name.strip() == "ignore"


def class_key(name: str) -> str:
    return "class:" + (re.sub(r"[^a-z0-9]+", "_", name.strip().casefold()).strip("_") or "unnamed")


def _image_cases(dataset_directory: Path) -> int:
    folder = dataset_directory / "imagesTr"
    if not folder.is_dir():
        return 0
    return len({path.name.split("_0000")[0] for path in folder.iterdir() if "_0000" in path.name})


def _auto_blocks(dataset_directory: Path) -> dict[str, str]:
    manifest = _manifest(dataset_directory)
    sparse = any(_is_ignore(name) for _value, name in dataset_classes(dataset_directory))
    facts = "\n".join([
        "**Facts** (filled in by nnWorkbench):",
        "",
        f"- Images in imagesTr: {_image_cases(dataset_directory)}",
        f"- File ending: {manifest.get('file_ending') or '-'}",
    ])
    if sparse:
        labeling = ("**Dense or sparse:** sparse: only parts of each image are annotated; the rest is *ignore* and "
                    "not used for training (dataset.json has an ignore label).")
    else:
        labeling = ("**Dense or sparse:** dense: every voxel gets a class; what is not outlined is background "
                    "(dataset.json has no ignore label).")
    return {"facts": facts, "labeling": labeling}


# ---------------------------------------------------------------- writing


def _field_block(key: str, field: Field, answer: Answer | None = None) -> str:
    answer = answer or Answer()
    first = answer.text.strip() or f"*{field.question}*"
    lines = [f"<!-- field: {key} -->", f"**{field.label}:** {first}"]
    if answer.images:
        lines.append("")
        lines += [f"![{caption}]({path})" for caption, path in answer.images]
    lines.append("<!-- end -->")
    return "\n".join(lines)


def _auto_block(key: str, content: str) -> str:
    return f"<!-- auto: {key} -->\n{content}\n<!-- end -->"


def class_table_block(classes: list[tuple[int, str]]) -> str:
    """The editable class table (ID order); ignore gets a note that it needs no description."""
    rows = [f"| {value} | {name} |" for value, name in sorted(classes)]
    note = ("\n\n*Ignore* (highest ID) marks voxels that are not annotated; it needs no description."
            if any(_is_ignore(name) for _value, name in classes) else "")
    # a blank line before the end marker: right below a table, Markdown would render it as an empty row
    return "<!-- classes -->\n" + "\n".join(["| ID | Class |", "| --- | --- |", *rows]) + note + "\n\n<!-- end -->"


def description_classes(text: str) -> list[tuple[int, str]] | None:
    """The classes of the description's table, or None if the file has no class table."""
    return _table_rows(_CLASS_TABLE.search(text))


def channel_table_block(channels: list[tuple[int, str]]) -> str:
    """The editable channel table: the channel index is the file ending _0000, _0001, … of the images."""
    rows = [f"| {index} | {name} |" for index, name in sorted(channels)]
    return "<!-- channels -->\n" + "\n".join(["| Channel | Name |", "| --- | --- |", *rows]) + "\n\n<!-- end -->"


def _table_rows(block: re.Match | None) -> list[tuple[int, str]] | None:
    if block is None:
        return None
    rows = (_TABLE_ROW.match(line.strip()) for line in block.group(0).split("\n"))
    return [(int(row["id"]), row["name"]) for row in rows if row]


def description_channels(text: str) -> list[tuple[int, str]] | None:
    """The channels of the description's table, or None if the file has no channel table."""
    return _table_rows(_CHANNEL_TABLE.search(text))


def table_problems(rows: list[tuple[int, str]], kind: str = "class") -> list[str]:
    """Why the form cannot save a table into the description without losing rows or class texts (IDs below 0 stand
    for IDs that are not whole numbers)."""
    problems = []
    if any(value < 0 for value, _name in rows):
        problems.append(f"Every {kind} needs an ID that is a whole number: 0, 1, 2, …")
    if any("|" in name for _value, name in rows):
        problems.append(f'{kind.capitalize()} names cannot contain "|" (it separates the columns of the table).')
    if kind == "class":
        # the class texts are stored under class_key(name): names with the same key would share or lose them
        names = [name.strip() for _value, name in rows if name.strip() and not _is_ignore(name)]
        keys = [class_key(name) for name in names]
        clashes = [name for name, key in zip(names, keys, strict=True) if keys.count(key) > 1]
        if len({name.casefold() for name in clashes}) > 1:
            problems.append("These class names differ only in characters other than a-z and 0-9, so their "
                            "descriptions cannot be kept apart: " + ", ".join(clashes) + ". Rename one of them.")
    return problems


def channel_differences(described: list[tuple[int, str]], defined: list[tuple[int, str]]) -> list[str]:
    """How the description's channels differ from dataset.json, as short sentences (empty when they match)."""
    described_by_index, defined_by_index = dict(described), dict(defined)
    differences = []
    for index in sorted(set(described_by_index) | set(defined_by_index)):
        if index not in defined_by_index:
            differences.append(f"channel {index} ({described_by_index[index]}) is only in the description")
        elif index not in described_by_index:
            differences.append(f"channel {index} ({defined_by_index[index]}) is only in dataset.json")
        elif described_by_index[index].strip() != defined_by_index[index].strip():
            differences.append(f"channel {index}: {described_by_index[index] or '(no name)'} in the description, "
                               f"{defined_by_index[index] or '(no name)'} in dataset.json")
    return differences


def channel_table_problems(channels: list[tuple[int, str]]) -> list[str]:
    """Why a channel table cannot be written to dataset.json."""
    problems = []
    if not channels:
        problems.append("Add at least one channel.")
    if any(not name.strip() for _index, name in channels):
        problems.append("Every channel needs a name.")
    if sorted(index for index, _name in channels) != list(range(len(channels))):
        problems.append("Channels must be numbered continuously, starting at 0.")
    return problems


def push_channels(dataset_directory: Path, channels: list[tuple[int, str]]) -> None:
    """Write the channels into dataset.json's channel_names (all other keys stay); raises ValueError on invalid
    tables."""
    from .dataset_json import read_manifest, write_manifest

    problems = channel_table_problems(channels)
    if problems:
        raise ValueError(" ".join(problems))
    manifest = read_manifest(dataset_directory)
    manifest["channel_names"] = {str(index): name.strip() for index, name in sorted(channels)}
    write_manifest(dataset_directory, manifest)


def import_channels(text: str, dataset_directory: Path) -> str:
    """Replace the description's channel table with dataset.json's channel_names."""
    return update_description(text, dataset_directory, {}, channels=dataset_channels(dataset_directory))


def class_differences(described: list[tuple[int, str]], defined: list[tuple[int, str]]) -> list[str]:
    """How the description's classes differ from dataset.json, as short sentences (empty when they match)."""
    by_name_described = {name.strip().casefold(): (value, name) for value, name in described}
    by_name_defined = {name.strip().casefold(): (value, name) for value, name in defined}
    differences = []
    for key, (value, name) in by_name_described.items():
        if key not in by_name_defined:
            differences.append(f"{name} ({value}) is only in the description")
        elif by_name_defined[key][0] != value:
            differences.append(f"{name}: ID {value} in the description, {by_name_defined[key][0]} in dataset.json")
    differences += [f"{name} ({value}) is only in dataset.json" for key, (value, name) in by_name_defined.items()
                    if key not in by_name_described]
    return differences


def class_table_problems(classes: list[tuple[int, str]]) -> list[str]:
    """Why a class table cannot be written to dataset.json (the rules of the dataset.json editor)."""
    problems = []
    names = [name.strip() for _value, name in classes]
    if not classes:
        problems.append("Add at least one class.")
    if any(not name for name in names):
        problems.append("Every class needs a name.")
    if len({name.casefold() for name in names}) != len(names):
        problems.append("Class names must be unique.")
    if sorted(value for value, _name in classes) != list(range(len(classes))):
        problems.append("IDs must be continuous, starting at 0.")
    ignore = [value for value, name in classes if _is_ignore(name)]
    if ignore and ignore[0] != max(value for value, _name in classes):
        problems.append("ignore must have the highest ID.")
    return problems


def push_classes(dataset_directory: Path, classes: list[tuple[int, str]]) -> None:
    """Write the classes into dataset.json's labels (all other keys stay); raises ValueError on invalid tables."""
    from .dataset_json import REGIONS_NOTE, has_regions, read_manifest, write_manifest

    problems = class_table_problems(classes)
    if problems:
        raise ValueError(" ".join(problems))
    manifest = read_manifest(dataset_directory)
    if has_regions(manifest):  # the class table has single IDs only: pushing would delete the regions
        raise ValueError(f"Not written: dataset.json has region-based labels. {REGIONS_NOTE}")
    manifest["labels"] = {name.strip(): value for value, name in sorted(classes)}
    write_manifest(dataset_directory, manifest)


def renames_by_id(old: list[tuple[int, str]], new: list[tuple[int, str]]) -> dict[str, str]:
    """Classes that kept their ID but changed their name ({new key: old key}), so their texts move along."""
    old_names = {name.strip().casefold() for _value, name in old}
    new_names = {name.strip().casefold() for _value, name in new}
    old_by_id = dict(old)
    return {class_key(name): class_key(old_by_id[value]) for value, name in new
            if value in old_by_id and name.strip().casefold() not in old_names
            and old_by_id[value].strip().casefold() not in new_names}


def classes_losing_text(text: str, classes: list[tuple[int, str]],
                        renames: dict[str, str] | None = None) -> list[str]:
    """Names of described classes with written answers that a new class table would remove."""
    kept = {renames.get(class_key(name), class_key(name)) for _value, name in classes} if renames else {
        class_key(name) for _value, name in classes}
    lost = []
    for key, (heading, answers, extra) in _class_chunks(text).items():
        if key not in kept and (extra or any(answer != Answer() for answer in answers.values())):
            lost.append(heading)
    return lost


def import_classes(text: str, dataset_directory: Path) -> str:
    """Replace the description's class table with dataset.json's labels; the class sections follow (a class that
    kept its ID under a new name keeps its texts)."""
    classes = dataset_classes(dataset_directory)
    renames = renames_by_id(description_classes(text) or [], classes)
    return update_description(text, dataset_directory, {}, classes=classes, renames=renames)


def _class_block(value: int, name: str, answers: dict[str, Answer] | None = None, extra: str = "") -> str:
    answers = answers or {}
    key = class_key(name)
    parts = [f"### {value}. {name or '(class name)'}"]
    parts += [_field_block(f"{key}:{field.key}", field, answers.get(f"{key}:{field.key}")) for field in CLASS_FIELDS]
    if extra:
        parts.append(extra)  # text written by hand into the class section
    return "\n\n".join(parts)


def _class_region(text: str) -> tuple[int, int] | None:
    """Start and end of the class sections: after the class table, up to the classes-end marker."""
    table = _CLASS_TABLE.search(text)
    end = text.find(_CLASSES_END, table.end()) if table else -1
    return (table.end(), end) if table and end >= 0 else None


def _class_chunks(text: str) -> dict[str, tuple[str, dict[str, Answer], str]]:
    """The class sections by class key: heading text, answers (by field key) and text added by hand."""
    region = _class_region(text)
    if region is None:
        return {}
    chunks = {}
    for chunk in _CLASS_HEADING.split(text[region[0]:region[1]])[1:]:
        key = _CLASS_KEY.search(chunk)
        if key is None:
            continue
        heading, _newline, body = chunk.partition("\n")
        answers = {match["key"].rsplit(":", 1)[1]: _parse_body(match["body"], _field_for(match["key"], {}))
                   for match in _FIELD.finditer(body)}
        extra = _FIELD.sub("", body).strip()
        chunks[key["key"]] = (heading.removeprefix("### ").strip(), answers, extra)
    return chunks


def _rebuild_classes(text: str, classes: list[tuple[int, str]], answers: dict[str, Answer],
                     renames: dict[str, str]) -> str:
    """Rewrite the class sections in table order: answers from the form first, then the saved ones (under the old
    key for renamed classes); sections of removed classes are dropped."""
    region = _class_region(text)
    if region is None:
        return text
    chunks = _class_chunks(text)
    prefix = _CLASS_HEADING.split(text[region[0]:region[1]])[0].strip()
    blocks = []
    for value, name in sorted(classes):
        if _is_ignore(name):
            continue
        key = class_key(name)
        _heading, saved, extra = chunks.get(renames.get(key, key), ("", {}, ""))
        merged = {f"{key}:{field}": answer for field, answer in saved.items()}
        merged.update({item: answer for item, answer in answers.items() if item.startswith(key + ":")})
        blocks.append(_class_block(value, name, merged, extra))
    middle = "\n\n".join(part for part in (prefix, *blocks) if part)
    return text[:region[0]] + "\n\n" + (middle + "\n\n" if middle else "") + text[region[1]:]


def description_template(dataset_directory: Path) -> str:
    """A new description with the facts filled in and a placeholder question in every field."""
    auto = _auto_blocks(dataset_directory)
    parts = [
        f"# {dataset_directory.name}: dataset description",
        ("*Describe your images and define the classes before annotating (guide: 1. Describe and 2. Define). In "
         f"nnWorkbench: Data page, Description, Edit. Images go into the folder `{IMAGES_FOLDER}/` next to this "
         "file.*"),
    ]
    for title, fields in SECTIONS:
        parts += ["---", f"## {title}"]
        if title == IMAGES_SECTION:
            parts.append(_auto_block("facts", auto["facts"]))
        for field in fields:
            parts.append(_field_block(field.key, field))
            if field.key == CHANNELS_AFTER:
                parts += [_CHANNEL_NOTE, channel_table_block(dataset_channels(dataset_directory))]
    classes = dataset_classes(dataset_directory)
    parts += ["---", f"## {CLASSES_SECTION}",
              "*One section per class of the table. The table is compared with dataset.json, which nnU-Net uses.*",
              _auto_block("labeling", auto["labeling"]), class_table_block(classes)]
    parts += [_class_block(value, name) for value, name in classes if not _is_ignore(name)]
    parts += [_CLASSES_END, "---", f"## {NOTES_SECTION}", _field_block(NOTES_FIELD.key, NOTES_FIELD)]
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------- reading and updating (the GUI form)


def all_fields(dataset_directory: Path, classes: list[tuple[int, str]] | None = None) -> list[tuple[str, str, Field]]:
    """(section title, key, field) for every field of the file, one block per class of the description's table
    (dataset.json's labels when the description has no table)."""
    result = [(title, field.key, field) for title, fields in SECTIONS for field in fields]
    for value, name in (classes if classes is not None else dataset_classes(dataset_directory)):
        if not _is_ignore(name):
            result += [(f"{CLASSES_SECTION}: {value}. {name or '(class name)'}", f"{class_key(name)}:{field.key}",
                        field) for field in CLASS_FIELDS]
    result.append((NOTES_SECTION, NOTES_FIELD.key, NOTES_FIELD))
    return result


def _field_for(key: str, fields: dict[str, Field]) -> Field | None:
    return fields.get(key) or next((item for item in CLASS_FIELDS if key.endswith(":" + item.key)), None)


def _parse_body(body: str, field: Field | None) -> Answer:
    text_lines, images = [], []
    for line in body.strip("\n").split("\n"):
        match = _IMAGE_LINE.match(line.strip())
        if match:
            images.append((match["caption"], match["path"]))
        else:
            text_lines.append(line)
    text = _LABEL.sub("", "\n".join(text_lines).strip(), count=1).strip()  # any label, also an older wording
    if field is not None and (_ITALIC_QUESTION.match(text) or text.casefold() in {
            f"*{question}*".casefold() for question in (field.question, *FORMER_QUESTIONS)}):
        text = ""  # still the placeholder
    return Answer(text, tuple(images))


def read_answers(text: str, dataset_directory: Path) -> dict[str, Answer]:
    """The answers of every field found in the file (missing fields are empty)."""
    fields = {key: field for _title, key, field in all_fields(dataset_directory, description_classes(text))}
    return {match["key"]: _parse_body(match["body"], _field_for(match["key"], fields)) for match in _FIELD.finditer(text)}


def has_form_fields(text: str) -> bool:
    return _FIELD.search(text) is not None


def _cut(text: str, start: int, end: int) -> str:
    """Remove text[start:end] and leave one blank line where it was."""
    return text[:start].rstrip("\n") + "\n\n" + text[end:].lstrip("\n")


def _place_channel_table(text: str, channels: list[tuple[int, str]] | None) -> str:
    """Write ``channels`` into the channel table (if given) and keep the table right after the Type field."""
    existing = _CHANNEL_TABLE.search(text)
    if channels is None and existing is None:
        return text
    table = channel_table_block(channels) if channels is not None else existing.group(0)
    if existing is not None:
        text = _cut(text, existing.start(), existing.end())
        if (note := text.find(_CHANNEL_NOTE)) >= 0:
            text = _cut(text, note, note + len(_CHANNEL_NOTE))
    block = f"{_CHANNEL_NOTE}\n\n{table}"
    anchor = re.search(rf"<!-- field: {CHANNELS_AFTER} -->\n.*?<!-- end -->", text, re.DOTALL)
    if anchor is None:
        return text.rstrip("\n") + "\n\n" + block + "\n"
    return text[:anchor.end()] + "\n\n" + block + "\n\n" + text[anchor.end():].lstrip("\n")


def _add_missing_fields(text: str, answers: dict[str, Answer]) -> str:
    """Insert fields of sections 1 and 2 the file does not have yet (a question added to the template later), after
    the field before them; files without that field stay as they are."""
    for _title, fields in SECTIONS:
        for previous, field in pairwise(fields):
            if f"<!-- field: {field.key} -->" in text:
                continue
            anchor = re.search(rf"<!-- field: {previous.key} -->\n.*?<!-- end -->", text, re.DOTALL)
            if anchor is None:
                continue
            end = anchor.end()
            if (channels := _CHANNEL_TABLE.search(text, end)) is not None and previous.key == CHANNELS_AFTER:
                end = channels.end()
            block = _field_block(field.key, field, answers.get(field.key))
            text = text[:end] + "\n\n" + block + text[end:]
    return text


def update_description(text: str, dataset_directory: Path, answers: dict[str, Answer],
                       classes: list[tuple[int, str]] | None = None, renames: dict[str, str] | None = None,
                       channels: list[tuple[int, str]] | None = None) -> str:
    """Write the answers into their fields, refresh the facts, write ``channels`` and ``classes`` into their tables
    (if given) and rebuild the class sections to match the class table; keep everything else.

    ``renames`` maps the key of a renamed class to its old key, so its saved texts move along."""
    for former, heading in _FORMER_HEADINGS.items():
        text = text.replace(former + "\n", heading + "\n")
    text = _place_channel_table(text, channels)
    if classes is not None:
        table = class_table_block(classes)
        if _CLASS_TABLE.search(text):
            text = _CLASS_TABLE.sub(lambda _match: table, text, count=1)
        elif f"## {CLASSES_SECTION}" in text:
            text = text.replace(f"## {CLASSES_SECTION}", f"## {CLASSES_SECTION}\n\n{table}", 1)
    described = description_classes(text)
    if described is not None:
        text = _rebuild_classes(text, described, answers, renames or {})
    fields = {key: field for _title, key, field in all_fields(dataset_directory, described)}

    def replace_field(match: re.Match) -> str:
        field = _field_for(match["key"], fields)
        if match["key"] not in answers or field is None:
            return match.group(0)
        return _field_block(match["key"], field, answers[match["key"]])

    text = _FIELD.sub(replace_field, text)
    text = _add_missing_fields(text, answers)
    auto = _auto_blocks(dataset_directory)
    return _AUTO.sub(lambda match: _auto_block(match["key"], auto.get(match["key"], "")), text)


# ---------------------------------------------------------------- files


def create_description(dataset_directory: Path) -> Path:
    """Write the template; never replaces an existing description."""
    path = description_path(dataset_directory)
    if path.exists():
        raise ValueError(f"{path} already exists.")
    (path.parent / IMAGES_FOLDER).mkdir(parents=True, exist_ok=True)
    path.write_text(description_template(dataset_directory), encoding="utf-8")
    return path


def read_description(path: Path) -> str:
    """The description's text; a file saved in another encoding (e.g. ANSI by Windows Notepad) is read as Windows-1252
    instead of failing, and saving writes it as UTF-8. Raises OSError."""
    data = path.read_bytes()
    for encoding in ("utf-8", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass
    return data.decode("latin-1")  # never fails


def save_answers(dataset_directory: Path, answers: dict[str, Answer],
                 classes: list[tuple[int, str]] | None = None, renames: dict[str, str] | None = None,
                 channels: list[tuple[int, str]] | None = None) -> Path:
    """Update the description file with the form's answers and tables (created from the template if missing)."""
    path = description_path(dataset_directory)
    text = read_description(path) if path.is_file() else description_template(dataset_directory)
    path.parent.mkdir(exist_ok=True)
    path.write_text(update_description(text, dataset_directory, answers, classes, renames, channels),
                    encoding="utf-8")
    return path


def add_image(dataset_directory: Path, image: Path) -> str:
    """Copy an image into description_images/ next to the description (without overwriting) and return its relative
    link target. Raises OSError."""
    return add_image_next_to(description_path(dataset_directory), image)


def add_image_next_to(description: Path, image: Path) -> str:
    """Copy an image into description_images/ next to the Markdown file ``description`` (see add_image)."""
    folder = description.parent / IMAGES_FOLDER
    if not description.parent.parent.is_dir():  # creating it would leave a second dataset folder with this ID
        raise FileNotFoundError(f"{description.parent.parent} no longer exists (moved or renamed?)")
    folder.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^\w.-]+", "_", image.name)  # no spaces: they would break the Markdown link
    target = folder / name
    counter = 2
    while target.exists() and not target.samefile(image):
        target = folder / f"{Path(name).stem}_{counter}{Path(name).suffix}"
        counter += 1
    if not target.exists():
        shutil.copy2(image, target)
    return f"{IMAGES_FOLDER}/{target.name}"
