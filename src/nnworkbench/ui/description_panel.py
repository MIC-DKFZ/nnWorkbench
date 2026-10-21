"""The Description view on the Data page: dataset-description.md rendered, and a form to fill it in."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from qtpy.QtCore import Qt, QUrl, Signal
from qtpy.QtGui import QDesktopServices, QPixmap
from qtpy.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from nnworkbench.services.dataset_description import (
    CHANNELS_AFTER,
    CLASS_FIELDS,
    CLASSES_SECTION,
    DESCRIPTION_FILE,
    NOTES_FIELD,
    NOTES_SECTION,
    SECTIONS,
    Answer,
    Field,
    add_image,
    channel_differences,
    channel_table_problems,
    class_differences,
    class_key,
    class_table_problems,
    classes_losing_text,
    create_description,
    dataset_channels,
    dataset_classes,
    description_channels,
    description_classes,
    description_path,
    has_form_fields,
    import_channels,
    import_classes,
    push_channels,
    push_classes,
    read_answers,
    read_description,
    renames_by_id,
    save_answers,
    table_problems,
    update_description,
)

from .widgets import button, empty_state, label, svg_icon, tip
from .workflow_guide import GuideBrowser

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.gif *.svg *.bmp *.webp)"
THUMBNAIL_HEIGHT = 72


class _FieldEditor(QWidget):
    """One question: label, the question as hint, a text box and, for example images, thumbnails."""

    def __init__(self, field: Field, answer: Answer, dataset_directory: Path) -> None:
        super().__init__()
        self._field, self._directory = field, dataset_directory
        self._images = list(answer.images)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 6)
        layout.setSpacing(4)
        layout.addWidget(label(field.label, bold=True))
        self.text = QPlainTextEdit(answer.text)
        self.text.setPlaceholderText(field.question)
        self.text.setTabChangesFocus(True)
        self.text.setFixedHeight(76)
        layout.addWidget(self.text)
        self._image_row = QHBoxLayout()
        self._image_row.setSpacing(8)
        if field.images:
            layout.addLayout(self._image_row)
            self._render_images()

    def _render_images(self) -> None:
        while self._image_row.count():
            item = self._image_row.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        for index, (caption, path) in enumerate(self._images):
            holder = QFrame()
            holder.setObjectName("card")
            column = QVBoxLayout(holder)
            column.setContentsMargins(4, 4, 4, 4)
            column.setSpacing(2)
            picture = QLabel()
            pixmap = QPixmap(str(description_path(self._directory).parent / path))
            if not pixmap.isNull():
                picture.setPixmap(pixmap.scaledToHeight(THUMBNAIL_HEIGHT, Qt.TransformationMode.SmoothTransformation))
            else:
                picture.setText("missing")
            picture.setToolTip(path)
            column.addWidget(picture)
            remove = QPushButton("✕ remove")
            remove.setObjectName("link")
            remove.setToolTip(f"Remove {caption or path} from the description (the file stays in the folder)")
            remove.clicked.connect(lambda _checked=False, position=index: self._remove_image(position))
            column.addWidget(remove)
            self._image_row.addWidget(holder)
        self._image_row.addWidget(button("Add image…", small=True, icon="file", clicked=self._add_image,
                                         tooltip="Copy a picture or screenshot into description_images/"))
        self._image_row.addStretch()

    def _add_image(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(self, "Add image", str(Path.home()), IMAGE_FILTER)
        if not selected:
            return
        try:
            target = add_image(self._directory, Path(selected))
        except OSError as error:
            QMessageBox.warning(self, "Image not added", f"The image could not be copied: {error}")
            return
        self._images.append((Path(selected).stem.replace("_", " "), target))
        self._render_images()

    def _remove_image(self, index: int) -> None:
        del self._images[index]
        self._render_images()

    def answer(self) -> Answer:
        return Answer(self.text.toPlainText().strip(), tuple(self._images))


class _TableEditor(QWidget):
    """A table of the description (classes or channels): number and name per row, add and remove (written to
    dataset.json by Push)."""

    changed = Signal()

    def __init__(self, classes: list[tuple[int, str]], kind: str = "class") -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 6)
        layout.setSpacing(4)
        self._rows_layout = QVBoxLayout()
        self._rows_layout.setSpacing(4)
        layout.addLayout(self._rows_layout)
        self._rows: list[tuple[int, QWidget, QLineEdit, QLineEdit]] = []  # (token, row, ID field, name field)
        self._tokens = 0
        self.origins: dict[int, str] = {}  # row token -> class key the row had when loaded (texts move along)
        self.confirm_remove: Callable[[int], bool] | None = None
        buttons = QHBoxLayout()
        self._kind = kind
        actions = [(f"＋ add {kind}", lambda: self._add_row(self._next_id(), ""))]
        if kind == "class":
            actions.append(("＋ add ignore", self._add_ignore))
        for text, action in actions:
            link = QPushButton(text)
            link.setObjectName("link")
            link.clicked.connect(lambda _checked=False, run=action: run())
            buttons.addWidget(link)
        buttons.addStretch()
        layout.addLayout(buttons)
        self.set_classes(classes)

    def set_classes(self, classes: list[tuple[int, str]]) -> None:
        for _token, row, _value, _name in self._rows:
            row.deleteLater()
        self._rows, self.origins = [], {}
        for value, name in sorted(classes):
            self._add_row(value, name, origin=class_key(name))
        self.changed.emit()

    def _next_id(self) -> int:
        values = [value for value, _name in self.classes()]
        return max(values) + 1 if values else 0

    def _add_ignore(self) -> None:
        if not any(name.strip().casefold() == "ignore" for _value, name in self.classes()):
            self._add_row(self._next_id(), "ignore")

    def _add_row(self, value: int, name: str, origin: str | None = None) -> None:
        row = QWidget()
        line = QHBoxLayout(row)
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(6)
        identifier, title = QLineEdit(str(value)), QLineEdit(name)
        identifier.setFixedWidth(56)
        identifier.setAlignment(Qt.AlignmentFlag.AlignRight)
        title.setPlaceholderText("class name, e.g. nucleus" if self._kind == "class"
                                 else "channel name, e.g. fluorescence or CT")
        remove = QPushButton("✕")
        remove.setObjectName("link")
        remove.setToolTip(f"Remove this {self._kind} from the table")
        line.addWidget(identifier)
        line.addWidget(label(":", muted=True))
        line.addWidget(title, 1)
        line.addWidget(remove)
        self._tokens += 1
        entry = (self._tokens, row, identifier, title)
        if origin is not None:
            self.origins[self._tokens] = origin
        remove.clicked.connect(lambda _checked=False: self._remove(entry))
        for field in (identifier, title):
            field.textChanged.connect(lambda _text: self.changed.emit())
        self._rows.append(entry)
        self._rows_layout.addWidget(row)
        self.changed.emit()

    def _remove(self, entry) -> None:
        if self.confirm_remove is not None and not self.confirm_remove(entry[0]):
            return
        if entry in self._rows:
            self._rows.remove(entry)
            entry[1].deleteLater()
            self.changed.emit()

    def rows(self) -> list[tuple[int, int, str]]:
        """(row token, ID, name) per row that has an ID or a name; the token stays the same while a row is edited."""
        result = []
        for token, _row, identifier, title in self._rows:
            text = identifier.text().strip()
            valid = text.isascii() and text.isdigit()  # not "²", "-1" or "--1"
            if valid or title.text().strip():
                result.append((token, int(text) if valid else -1, title.text().strip()))  # -1: not a valid ID
        return result

    def classes(self) -> list[tuple[int, str]]:
        return [(value, name) for _token, value, name in self.rows()]


class _CollapsibleCard(QFrame):
    """A form section: a title that folds the section in and out (folded, a short summary shows instead)."""

    def __init__(self, title: str, hint: str = "", *, name: str = "formSection",
                 summary: Callable[[], str] | None = None) -> None:
        super().__init__()
        self.setObjectName("card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 12, 16, 12) if name == "formSection" else outer.setContentsMargins(14, 10, 14, 10)
        outer.setSpacing(6)
        head = QHBoxLayout()
        self._toggle = QPushButton(title)
        self._toggle.setObjectName(name)
        self._toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle.setToolTip("Fold in or out")
        self._toggle.clicked.connect(lambda _checked=False: self.set_collapsed(not self.collapsed))
        head.addWidget(self._toggle)
        head.addStretch()
        self._summary_label = label("", muted=True, size=12)
        head.addWidget(self._summary_label)
        outer.addLayout(head)
        self._summary = summary
        self._body = QWidget()
        self.column = QVBoxLayout(self._body)
        self.column.setContentsMargins(0, 0, 0, 0)
        self.column.setSpacing(6)
        if hint:
            self.column.addWidget(label(hint, muted=True, wrap=True, size=12))
        outer.addWidget(self._body)
        self.set_collapsed(False)

    @property
    def collapsed(self) -> bool:
        return self._body.isHidden()

    def set_title(self, title: str) -> None:
        self._toggle.setText(title)

    def set_collapsed(self, collapsed: bool) -> None:
        self._body.setVisible(not collapsed)
        self._toggle.setIcon(svg_icon("chevron-right" if collapsed else "chevron-down"))
        self._summary_label.setText(self._summary() if collapsed and self._summary is not None else "")


def _has_files(folder: Path) -> bool:
    try:
        return folder.is_dir() and any(path.is_file() for path in folder.iterdir())
    except OSError:
        return False


def _answered(editors) -> str:
    filled = sum(editor.answer() != Answer() for editor in editors)
    return f"{filled} of {len(editors)} answered"


class _ClassSection(_CollapsibleCard):
    """The form fields of one class; its title follows the class table."""

    def __init__(self, answers: dict[str, Answer], dataset_directory: Path) -> None:
        super().__init__("", name="formClass", summary=lambda: _answered(list(self.editors.values())))
        self.editors = {field.key: _FieldEditor(field, answers.get(field.key, Answer()), dataset_directory)
                        for field in CLASS_FIELDS}
        for editor in self.editors.values():
            self.column.addWidget(editor)

    def set_class(self, value: int, name: str) -> None:
        self.set_title(f"Class {value if value >= 0 else '?'}: {name or '(class name)'}")

    def answers(self) -> dict[str, Answer]:
        return {key: editor.answer() for key, editor in self.editors.items()}

    def has_text(self) -> bool:
        return any(answer != Answer() for answer in self.answers().values())


class DescriptionPanel(QWidget):
    """Empty state, rendered view or form for one dataset's description."""

    dataset_json_changed = Signal()  # Push wrote the class table into dataset.json

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._directory: Path | None = None
        self._editors: dict[str, _FieldEditor] = {}  # the fields of sections 1, 2 and the notes
        self._class_table: _TableEditor | None = None
        self._channel_table: _TableEditor | None = None
        self._cards: list[_CollapsibleCard] = []
        self._class_sections: dict[int, _ClassSection] = {}  # row token -> section
        self._class_answers: dict[str, dict[str, Answer]] = {}  # texts by class key, for rows created later
        self._class_column: QVBoxLayout | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        bar = QFrame()
        bar.setObjectName("cardBand")
        bar_row = QHBoxLayout(bar)
        bar_row.setContentsMargins(16, 8, 16, 8)
        self._title = label(DESCRIPTION_FILE, muted=True, mono=True)
        bar_row.addWidget(self._title)
        bar_row.addStretch()
        self._edit = button("Edit", small=True, icon="edit", clicked=self._start_form)
        self._text = button("Edit as text", small=True, clicked=self._edit_as_text,
                            tooltip="Edit the Markdown file directly (for experienced users)")
        self._folder = button("Open folder", small=True, icon="folder", clicked=self._open_folder)
        self._cancel = button("Cancel", small=True, clicked=self._cancel_form)
        self._save = button("Save", small=True, primary=True, clicked=self._save_form)
        for widget in (self._edit, self._text, self._folder, self._cancel, self._save):
            bar_row.addWidget(widget)
        layout.addWidget(bar)
        # shown when a table of the description (channels, classes) and dataset.json differ
        self._sync_rows: dict[str, tuple[QWidget, QLabel]] = {}
        for kind, import_action, push_action in (("channels", self._import_channels, self._push_channels),
                                                 ("classes", self._import_classes, self._push_classes)):
            row = QWidget()
            sync_row = QHBoxLayout(row)
            sync_row.setContentsMargins(16, 8, 16, 8)
            sync_row.setSpacing(8)
            text = tip("", "orange")
            sync_row.addWidget(text, 1)
            sync_row.addWidget(button("Import from dataset.json", small=True, clicked=import_action,
                                      tooltip=f"Take over the {kind} of dataset.json into the description"))
            sync_row.addWidget(button("Push to dataset.json", small=True, clicked=push_action,
                                      tooltip=f"Write the description's {kind} table into dataset.json"))
            layout.addWidget(row)
            self._sync_rows[kind] = (row, text)
        self._sync, self._sync_text = self._sync_rows["classes"]
        self._stack = QStackedWidget()
        # empty: no description yet
        empty = QWidget()
        empty_layout = QVBoxLayout(empty)
        empty_layout.addStretch()
        empty_layout.addWidget(empty_state("No description yet. It documents your images and defines the classes "
                                           "(guide: 1. Describe and 2. Define), with example images."))
        create_row = QHBoxLayout()
        create_row.addStretch()
        create_row.addWidget(button("Create description", primary=True, clicked=self._create))
        create_row.addStretch()
        empty_layout.addLayout(create_row)
        empty_layout.addStretch()
        self._stack.addWidget(empty)
        self._viewer = GuideBrowser()
        self._viewer.setOpenExternalLinks(True)
        self._stack.addWidget(self._viewer)
        self._form_scroll = QScrollArea()
        self._form_scroll.setWidgetResizable(True)
        self._form_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._stack.addWidget(self._form_scroll)
        layout.addWidget(self._stack, 1)
        self._show(None)

    # ------------------------------------------------------------ states

    def set_dataset(self, raw_path: Path | None) -> None:
        if self.editing and raw_path == self._directory:
            return  # keep the form open while the page refreshes
        self._directory = raw_path
        self._show(None)

    @property
    def editing(self) -> bool:
        return self._stack.currentWidget() is self._form_scroll

    def _path(self) -> Path | None:
        return description_path(self._directory) if self._directory is not None else None

    def _show(self, _unused) -> None:
        path = self._path()
        exists = path is not None and path.is_file()
        self._stack.setCurrentIndex(1 if exists else 0)
        if exists:
            self._viewer.load(path)
        for widget, visible in ((self._edit, exists), (self._text, exists), (self._folder, self._directory is not None),
                                (self._cancel, False), (self._save, False)):
            widget.setVisible(visible)
        self._title.setText(DESCRIPTION_FILE if exists else "")
        self._update_sync()

    # ------------------------------------------------------------ class table vs dataset.json

    def _saved_text(self) -> str | None:
        """The description file's text; None without a file or when it cannot be read."""
        path = self._path()
        try:
            return read_description(path) if path is not None and path.is_file() else None
        except OSError:
            return None

    def _read(self) -> str | None:
        """The description file's text, or None with a message when it cannot be read."""
        path = self._path()
        if path is None:
            return None
        try:
            return read_description(path)
        except OSError as error:
            QMessageBox.warning(self, "Description not read", f"{DESCRIPTION_FILE} could not be read: {error}")
            return None

    def _write(self, text: str) -> bool:
        try:
            self._path().write_text(text, encoding="utf-8")
        except OSError as error:
            QMessageBox.warning(self, "Not saved", f"{DESCRIPTION_FILE} could not be written: {error}")
            return False
        return True

    def _described_classes(self) -> list[tuple[int, str]] | None:
        if self._class_table is not None and self.editing:
            return self._class_table.classes()
        text = self._saved_text()
        return description_classes(text) if text is not None else None

    def _described_channels(self) -> list[tuple[int, str]] | None:
        if self._channel_table is not None and self.editing:
            return self._channel_table.classes()
        text = self._saved_text()
        return description_channels(text) if text is not None else None

    def _update_sync(self) -> None:
        for kind, described, compare, defined in (
                ("classes", self._described_classes(), class_differences, dataset_classes),
                ("channels", self._described_channels(), channel_differences, dataset_channels)):
            differences = (compare(described, defined(self._directory))
                           if described is not None and self._directory is not None else [])
            row, text = self._sync_rows[kind]
            row.setVisible(bool(differences))
            if differences:
                name = "class" if kind == "classes" else "channel"
                text.setText(f"<b>The {name} table differs from dataset.json:</b> " + "; ".join(differences)
                             + ". nnU-Net uses dataset.json.")

    def _import_channels(self) -> None:
        if self._directory is None:
            return
        if self.editing and self._channel_table is not None:
            self._channel_table.set_classes(dataset_channels(self._directory))
            return
        text = self._read()
        if text is not None and self._write(import_channels(text, self._directory)):
            self._show(None)

    def _push_channels(self) -> None:
        channels = self._described_channels()
        if self._directory is None or channels is None:
            return
        problems = channel_table_problems(channels)
        if problems:
            QMessageBox.warning(self, "Not written to dataset.json", " ".join(problems))
            return
        message = "Write the description's channel table into dataset.json?\n\n" + "\n".join(
            f"- {item}" for item in channel_differences(channels, dataset_channels(self._directory)))
        if _has_files(self._directory / "imagesTr"):
            message += ("\n\nThe image files in imagesTr are not changed: their endings _0000, _0001, … must match "
                        "the channels.")
        self._push(lambda: push_channels(self._directory, channels), message)

    def _push(self, write: Callable[[], None], message: str) -> None:
        if QMessageBox.question(self, "Push to dataset.json", message,
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                                QMessageBox.StandardButton.Cancel) != QMessageBox.StandardButton.Yes:
            return
        try:
            write()
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Not written to dataset.json", str(error))
            return
        if self.editing:
            if not self._save_form():
                self._update_sync()
        else:
            self._show(None)
        self.dataset_json_changed.emit()

    def _import_classes(self) -> None:
        if self._directory is None:
            return
        defined = dataset_classes(self._directory)
        if self.editing and self._class_table is not None:
            current = self._class_table.rows()
            renames = renames_by_id([(value, name) for _token, value, name in current], defined)
            texts = {class_key(name): self._class_sections[token].answers() for token, _value, name in current
                     if token in self._class_sections}
            lost = [name for token, _value, name in current if token in self._class_sections
                    and self._class_sections[token].has_text() and class_key(name) not in renames.values()
                    and class_key(name) not in {class_key(item) for _id, item in defined}]
            if lost and not self._confirm_losing(lost):
                return
            self._class_answers.update({key: texts[old] for key, old in renames.items() if old in texts})
            self._class_answers.update({key: answers for key, answers in texts.items() if key not in renames.values()})
            self._class_table.set_classes(defined)
            return
        text = self._read()
        if text is None:
            return
        lost = classes_losing_text(text, defined, renames_by_id(description_classes(text) or [], defined))
        if lost and not self._confirm_losing(lost):
            return
        if self._write(import_classes(text, self._directory)):
            self._show(None)

    def _confirm_losing(self, names: list[str]) -> bool:
        return QMessageBox.question(
            self, "Remove class descriptions?",
            "These classes are not in the new table; their descriptions are removed:\n\n"
            + "\n".join(f"- {name}" for name in names),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel) == QMessageBox.StandardButton.Yes

    def _push_classes(self) -> None:
        classes = self._described_classes()
        if self._directory is None or classes is None:
            return
        problems = class_table_problems(classes)
        if problems:
            QMessageBox.warning(self, "Not written to dataset.json", " ".join(problems))
            return
        differences = class_differences(classes, dataset_classes(self._directory))
        existing = _has_files(self._directory / "labelsTr")
        message = "Write the description's class table into dataset.json?\n\n" + "\n".join(
            f"- {item}" for item in differences)
        if existing:
            message += ("\n\nlabelsTr already contains label files. They keep their numbers: if IDs change, the "
                        "existing labels change their meaning.")
        self._push(lambda: push_classes(self._directory, classes), message)

    def _create(self) -> None:
        if self._directory is None:
            return
        try:
            create_description(self._directory)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Description not created", str(error))
            return
        self._start_form()

    # ------------------------------------------------------------ form

    def _start_form(self) -> None:
        text = self._read()
        if text is None:
            return
        if not has_form_fields(text):
            QMessageBox.information(self, "Edit as text", "This description was written without the form fields; "
                                    "it opens as text instead.")
            self._edit_as_text()
            return
        answers = read_answers(text, self._directory)
        form = QWidget()
        column = QVBoxLayout(form)
        column.setContentsMargins(18, 14, 18, 14)
        column.setSpacing(14)
        intro = QHBoxLayout()
        intro.addWidget(label("Answer in your own words; empty fields keep their question. Click a section title to "
                              "fold it in or out.", muted=True, wrap=True, size=12), 1)
        for caption, collapsed in (("Collapse all", True), ("Expand all", False)):  # not `text`: the file's text
            link = QPushButton(caption)
            link.setObjectName("link")
            link.clicked.connect(lambda _checked=False, state=collapsed: self._set_all_collapsed(state))
            intro.addWidget(link)
        column.addLayout(intro)
        self._editors, self._cards = {}, []
        for title, fields in SECTIONS:
            editors = [_FieldEditor(field, answers.get(field.key, Answer()), self._directory) for field in fields]
            card = _CollapsibleCard(title, summary=lambda items=editors: _answered(items))
            for field, editor in zip(fields, editors, strict=True):
                self._editors[field.key] = editor
                card.column.addWidget(editor)
                if field.key == CHANNELS_AFTER:
                    channels = description_channels(text)
                    card.column.addWidget(label("Channels", bold=True))
                    card.column.addWidget(label("The file ending _0000, _0001, … of each image and its name, as in "
                                                "dataset.json.", muted=True, wrap=True, size=12))
                    self._channel_table = _TableEditor(
                        channels if channels is not None else dataset_channels(self._directory), kind="channel")
                    self._channel_table.changed.connect(self._update_sync)
                    card.column.addWidget(self._channel_table)
            column.addWidget(card)
            self._cards.append(card)
        # 3. Define Classes: the table, then one section per class (follows the table while editing)
        card = _CollapsibleCard(CLASSES_SECTION, "ID and name of each class, as in dataset.json (background is 0, "
                                "ignore has the highest ID and needs no description). Below, one section per class.",
                                summary=self._classes_summary)
        card_column = card.column
        self._class_answers = {}
        for key, answer in answers.items():
            if key.startswith("class:"):
                class_name, _separator, field = key.rpartition(":")
                self._class_answers.setdefault(class_name, {})[field] = answer
        self._class_sections = {}
        classes = description_classes(text)
        self._class_table = _TableEditor(classes if classes is not None else dataset_classes(self._directory))
        self._class_table.confirm_remove = self._confirm_remove
        card_column.addWidget(self._class_table)
        self._class_column = QVBoxLayout()
        self._class_column.setSpacing(10)
        card_column.addLayout(self._class_column)
        column.addWidget(card)
        self._cards.append(card)
        self._class_table.changed.connect(self._sync_class_sections)
        self._class_table.changed.connect(self._update_sync)
        self._sync_class_sections(collapsed=True)
        notes = _FieldEditor(NOTES_FIELD, answers.get(NOTES_FIELD.key, Answer()), self._directory)
        self._editors[NOTES_FIELD.key] = notes
        card = _CollapsibleCard(NOTES_SECTION, summary=lambda: "written" if notes.answer() != Answer() else "empty")
        card.column.addWidget(notes)
        column.addWidget(card)
        self._cards.append(card)
        column.addStretch()
        self._set_all_collapsed(True)  # an overview first; a title opens its section
        self._form_scroll.setWidget(form)
        self._stack.setCurrentWidget(self._form_scroll)
        for widget, visible in ((self._edit, False), (self._text, False), (self._folder, True),
                                (self._cancel, True), (self._save, True)):
            widget.setVisible(visible)
        self._title.setText(f"{DESCRIPTION_FILE} · editing")
        self._update_sync()

    def _classes_summary(self) -> str:
        sections = [section for section in self._class_sections.values() if not section.isHidden()]
        described = sum(section.editors["description"].answer() != Answer() for section in sections)
        return f"{len(sections)} classes, {described} described"

    def _set_all_collapsed(self, collapsed: bool) -> None:
        for card in [*self._cards, *self._class_sections.values()]:
            card.set_collapsed(collapsed)

    def _sync_class_sections(self, collapsed: bool = False) -> None:
        """One section per class row (in ID order, none for ignore); renamed rows keep their texts. New sections
        start open (a class just added), or folded when the form opens."""
        if self._class_table is None or self._class_column is None:
            return
        rows = self._class_table.rows()
        tokens = {token for token, _value, _name in rows}
        for token in [token for token in self._class_sections if token not in tokens]:
            section = self._class_sections.pop(token)
            self._class_column.removeWidget(section)
            section.deleteLater()
        for token, value, name in sorted(rows, key=lambda row: row[1]):
            section = self._class_sections.get(token)
            if section is None:
                origin = self._class_table.origins.get(token)
                section = _ClassSection(self._class_answers.get(origin, {}) if origin else {}, self._directory)
                section.set_collapsed(collapsed)
                self._class_sections[token] = section
            section.set_class(value, name)
            section.setVisible(name.strip().casefold() != "ignore")
            self._class_column.removeWidget(section)
            self._class_column.addWidget(section)

    def _confirm_remove(self, token: int) -> bool:
        section = self._class_sections.get(token)
        if section is None or not section.has_text():
            return True
        return QMessageBox.question(
            self, "Remove class?", "This class has a description; it is removed with the class.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel) == QMessageBox.StandardButton.Yes

    def _form_answers(self) -> dict[str, Answer]:
        answers = {key: editor.answer() for key, editor in self._editors.items()}
        if self._class_table is not None:
            for token, _value, name in self._class_table.rows():
                if token in self._class_sections and name.strip().casefold() != "ignore":
                    key = class_key(name)
                    answers.update({f"{key}:{field}": answer
                                    for field, answer in self._class_sections[token].answers().items()})
        return answers

    def _form_renames(self) -> dict[str, str]:
        if self._class_table is None:
            return {}
        return {class_key(name): self._class_table.origins[token] for token, _value, name in self._class_table.rows()
                if token in self._class_table.origins and class_key(name) != self._class_table.origins[token]}

    def _form_text(self, current: str) -> str:
        """The file ``current`` as Save would write it."""
        classes = self._class_table.classes() if self._class_table is not None else None
        channels = self._channel_table.classes() if self._channel_table is not None else None
        return update_description(current, self._directory, self._form_answers(), classes, self._form_renames(),
                                  channels)

    def _form_problems(self) -> list[str]:
        """Why the form cannot be saved without losing table rows or class texts."""
        problems = []
        if self._class_table is not None:
            classes = self._class_table.classes()
            names = [name.strip().casefold() for _value, name in classes]
            if any(not name for name in names) or len(set(names)) != len(names):
                problems.append("Every class in the table needs a name, and the names must be unique.")
            problems += table_problems(classes, "class")
        if self._channel_table is not None:
            problems += table_problems(self._channel_table.classes(), "channel")
        return problems

    def _save_form(self) -> bool:
        if self._directory is not None:
            problems = self._form_problems()
            if problems:
                QMessageBox.warning(self, "Not saved", "\n\n".join(problems))
                return False
            try:
                save_answers(self._directory, self._form_answers(),
                             self._class_table.classes() if self._class_table is not None else None,
                             self._form_renames(),
                             self._channel_table.classes() if self._channel_table is not None else None)
            except OSError as error:
                QMessageBox.warning(self, "Not saved", f"{DESCRIPTION_FILE} could not be written: {error}")
                return False
        self._close_form()
        return True

    def _cancel_form(self) -> None:
        path = self._path()
        current = self._saved_text()
        if current is not None:
            changed = self._form_text(current) != update_description(current, self._directory, {})
        else:
            changed = path is not None and path.exists()  # unreadable: ask before discarding
        if changed and QMessageBox.question(
                self, "Discard changes?", "Discard the unsaved changes?",
                QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel) != QMessageBox.StandardButton.Discard:
            return
        self._close_form()

    def _close_form(self) -> None:
        self._class_table, self._class_column, self._class_sections = None, None, {}
        self._channel_table, self._cards = None, []
        self._show(None)

    # ------------------------------------------------------------ text and folder

    def _edit_as_text(self) -> None:
        path = self._path()
        if path is None:
            return
        from .description_dialog import DescriptionDialog

        dialog = DescriptionDialog(path, self)
        dialog._start_edit()
        dialog.exec()
        dialog.deleteLater()
        self._show(None)

    def _open_folder(self) -> None:
        """The folder with the description and its images (the dataset folder while there is none)."""
        if self._directory is not None:
            folder = description_path(self._directory).parent
            target = folder if folder.is_dir() else self._directory
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.resolve())))
