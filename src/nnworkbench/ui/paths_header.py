from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import QSignalBlocker, QSize, Signal
from qtpy.QtGui import QGuiApplication, QIcon
from qtpy.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from nnworkbench.path_setup import set_paths_command


class PathsHeader(QWidget):
    paths_changed = Signal()
    annotation_documentation_requested = Signal()
    annotation_documentation_closed = Signal()
    annotation_enabled_changed = Signal(bool)
    remote_profiles_changed = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("pathsHeader")
        self._fields: dict[str, QLineEdit] = {}
        self._browse_buttons: list[QPushButton] = []
        self._saved_path_texts: tuple[str, ...] = ()
        self._remote_profiles: dict[str, dict[str, str]] = {}
        self._local_path_texts: tuple[str, ...] = ("", "", "")
        self._active_profile = "Local"
        self._documentation_visible = False
        layout = QGridLayout(self)
        layout.setContentsMargins(12, 8, 12, 6); layout.setHorizontalSpacing(10); layout.setVerticalSpacing(5)
        for row, (key, title) in enumerate((("raw", "nnUNet_raw"), ("preprocessed", "nnUNet_preprocessed"),
                                             ("results", "nnUNet_results")), start=1):
            field = QLineEdit()
            field.setPlaceholderText(f"Optional {title} directory")
            field.editingFinished.connect(self._emit_paths_changed_if_modified)
            browse = QPushButton("Browse…")
            browse.clicked.connect(lambda _=False, item=key: self._browse(item))
            layout.addWidget(QLabel(title), row, 0); layout.addWidget(field, row, 1)
            layout.addWidget(browse, row, 2)
            self._fields[key] = field
            self._browse_buttons.append(browse)

        self.annotation_group = QWidget()
        self.annotation_group.setObjectName("annotationWorkflow")
        self.annotation_group.setFixedWidth(150)
        annotation_layout = QVBoxLayout(self.annotation_group)
        annotation_layout.setContentsMargins(10, 9, 10, 9); annotation_layout.setSpacing(7)
        annotation_heading = QHBoxLayout(); annotation_heading.setSpacing(7)
        logo = QLabel()
        logo_icon = QIcon(str(Path(__file__).parent.parent / "assets" / "AnnotationWorkflowLogo.svg"))
        logo.setPixmap(logo_icon.pixmap(QSize(48, 48)))
        annotation_title = QLabel("Annotation\nworkflow")
        annotation_title.setObjectName("annotationWorkflowTitle")
        annotation_heading.addWidget(logo); annotation_heading.addWidget(annotation_title); annotation_heading.addStretch()
        annotation_layout.addLayout(annotation_heading)
        annotation_actions = QHBoxLayout(); annotation_actions.setSpacing(7)
        self.annotation = QPushButton("Off")
        self.annotation.setObjectName("annotationToggle")
        self.annotation.setCheckable(True)
        self.annotation.setToolTip("Enable the optional interactive annotation workflow")
        self.annotation.toggled.connect(self._set_annotation_state)
        self.annotation_documentation = QPushButton("ⓘ")
        self.annotation_documentation.setObjectName("annotationDocumentation")
        self.annotation_documentation.setToolTip("Show the annotation workflow documentation")
        self.annotation_documentation.clicked.connect(self._toggle_annotation_documentation)
        annotation_actions.addWidget(self.annotation); annotation_actions.addWidget(self.annotation_documentation)
        annotation_layout.addLayout(annotation_actions)
        self._set_annotation_state(False)
        self.profile_tabs = QTabBar()
        self.profile_tabs.setObjectName("pathProfileTabs")
        self.profile_tabs.setDrawBase(False)
        self.profile_tabs.setExpanding(False)
        self.profile_tabs.setToolTip("The GUI always uses Local paths. Other profiles are for export.")
        self.profile_tabs.currentChanged.connect(self._select_profile)
        profile_actions = QHBoxLayout(); profile_actions.setContentsMargins(0, 0, 0, 0); profile_actions.setSpacing(5)
        profile_actions.addWidget(self.profile_tabs)
        profile_actions.addStretch()
        self.export_profile = QPushButton("Export")
        self.export_profile.setObjectName("pathProfileAction")
        self.export_profile.setToolTip("Copy a command that configures these paths on a remote shell")
        self.export_profile.clicked.connect(self._export_profile)
        profile_actions.addWidget(self.export_profile)
        self.delete_profile = QPushButton()
        self.delete_profile.setObjectName("pathProfileDelete")
        assets = Path(__file__).parent.parent / "assets"
        self._trash_icon = QIcon(str(assets / "trash.svg"))
        self._disabled_trash_icon = QIcon(str(assets / "trash-disabled.svg"))
        self.delete_profile.setIconSize(QSize(15, 15))
        self.delete_profile.setToolTip("Delete the selected path profile")
        self.delete_profile.clicked.connect(self._delete_profile)
        profile_actions.addWidget(self.delete_profile)
        self.profile_information = QPushButton("ⓘ")
        self.profile_information.setObjectName("pathProfileInfo")
        self.profile_information.setToolTip("Explain Local paths and remote profiles")
        self.profile_information.clicked.connect(self._show_profile_information)
        profile_actions.addWidget(self.profile_information)
        layout.addLayout(profile_actions, 0, 0, 1, 3)
        layout.addWidget(self.annotation_group, 0, 3, 4, 1)
        self._update_profile_tabs()
        self._update_profile_actions()
        self._saved_path_texts = self._path_texts()

    def _path_texts(self) -> tuple[str, ...]:
        return tuple(field.text().strip() for field in self._fields.values())

    def _emit_paths_changed_if_modified(self) -> None:
        """Refresh datasets only after the user actually changed a path."""
        path_texts = self._path_texts()
        if path_texts == self._saved_path_texts:
            return
        self._saved_path_texts = path_texts
        self._store_active_paths()
        if self._active_profile == "Local":
            self.paths_changed.emit()

    def _store_active_paths(self) -> None:
        path_texts = self._path_texts()
        if self._active_profile == "Local":
            self._local_path_texts = path_texts
        else:
            self._remote_profiles[self._active_profile] = dict(zip(("raw", "preprocessed", "results"), path_texts))
            self.remote_profiles_changed.emit(self._remote_profiles)

    def _update_profile_tabs(self) -> None:
        with QSignalBlocker(self.profile_tabs):
            while self.profile_tabs.count():
                self.profile_tabs.removeTab(0)
            self.profile_tabs.addTab("Local")
            for name in sorted(self._remote_profiles, key=str.casefold):
                self.profile_tabs.addTab(name)
            self.profile_tabs.addTab("+")
            self.profile_tabs.setTabToolTip(self.profile_tabs.count() - 1, "Add or edit a path profile")
            active_index = next(
                (index for index in range(self.profile_tabs.count())
                 if self.profile_tabs.tabText(index) == self._active_profile),
                0,
            )
            self.profile_tabs.setCurrentIndex(active_index)

    def _select_profile(self, index: int) -> None:
        if index < 0:
            return
        name = self.profile_tabs.tabText(index)
        if name == "+":
            active_index = next(
                (item for item in range(self.profile_tabs.count())
                 if self.profile_tabs.tabText(item) == self._active_profile),
                0,
            )
            with QSignalBlocker(self.profile_tabs):
                self.profile_tabs.setCurrentIndex(active_index)
            self._create_profile()
            return
        profile_name = "Local" if name == "Local (GUI)" else name
        if not name or profile_name == self._active_profile:
            return
        self._store_active_paths()
        self._active_profile = profile_name
        self._load_active_profile_paths()
        self._update_profile_actions()

    def _load_active_profile_paths(self) -> None:
        values = self._local_path_texts if self._active_profile == "Local" else tuple(
            self._remote_profiles.get(self._active_profile, {}).get(key, "")
            for key in ("raw", "preprocessed", "results")
        )
        for field, value in zip(self._fields.values(), values):
            field.setText(value)
        self._saved_path_texts = self._path_texts()

    def _update_profile_actions(self) -> None:
        can_delete = self._active_profile != "Local"
        self.delete_profile.setEnabled(can_delete)
        self.delete_profile.setIcon(self._trash_icon if can_delete else self._disabled_trash_icon)
        for browse in self._browse_buttons:
            browse.setEnabled(self._active_profile == "Local")

    def _show_profile_information(self) -> None:
        QMessageBox.information(
            self,
            "Path profiles",
            "The GUI always uses Local paths for datasets and nnU-Net workflows.\n\n"
            "Additional profiles store paths for remote environments only. Export copies an "
            "nnworkbench-set-paths command; paste and run it in the remote environment to set "
            "that profile there.",
        )

    def _profile_path_texts(self) -> tuple[str, ...]:
        if self._active_profile == "Local":
            return self._local_path_texts
        return tuple(self._remote_profiles.get(self._active_profile, {}).get(key, "")
                     for key in ("raw", "preprocessed", "results"))

    def _set_annotation_state(self, active: bool) -> None:
        """Show the future annotation-workflow switch state without changing workflow behavior."""
        self.annotation.setText("On" if active else "Off")
        self.annotation_group.setProperty("active", active)
        self.annotation_group.style().unpolish(self.annotation_group); self.annotation_group.style().polish(self.annotation_group)
        self.annotation_enabled_changed.emit(active)

    def set_annotation_enabled(self, active: bool) -> None:
        if self.annotation.isChecked() != active:
            self.annotation.setChecked(active)
        else:
            self._set_annotation_state(active)

    def annotation_enabled(self) -> bool:
        return self.annotation.isChecked()

    def set_annotation_documentation_visible(self, visible: bool) -> None:
        self._documentation_visible = visible
        self.annotation_documentation.setToolTip(
            "Close the annotation workflow documentation" if visible else "Show the annotation workflow documentation")

    def _toggle_annotation_documentation(self) -> None:
        if self._documentation_visible:
            self.annotation_documentation_closed.emit()
        else:
            self.annotation_documentation_requested.emit()

    def _browse(self, key: str) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Choose nnU-Net directory", self._fields[key].text())
        if selected:
            self._fields[key].setText(selected)
            self._emit_paths_changed_if_modified()

    def _create_profile(self) -> None:
        name, accepted = QInputDialog.getText(self, "Add path profile", "Profile name")
        name = name.strip()
        if not accepted or not name:
            return
        if name in {"Local", "+"} or name in self._remote_profiles:
            QMessageBox.warning(self, "Profile already exists", f'A profile named "{name}" already exists.')
            return
        self._store_active_paths()
        self._remote_profiles[name] = {key: "" for key in ("raw", "preprocessed", "results")}
        self._active_profile = name
        self._load_active_profile_paths()
        self._update_profile_tabs()
        self._update_profile_actions()
        self.remote_profiles_changed.emit(self._remote_profiles)

    def _delete_profile(self) -> None:
        name = self._active_profile
        if name == "Local":
            return
        answer = QMessageBox.question(
            self,
            "Delete path profile",
            f'Delete the profile "{name}"? Its stored paths will be lost.',
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        del self._remote_profiles[name]
        self._active_profile = "Local"
        self._load_active_profile_paths()
        self._update_profile_tabs()
        self._update_profile_actions()
        self.remote_profiles_changed.emit(self._remote_profiles)

    def _export_profile(self) -> None:
        self._store_active_paths()
        command = set_paths_command(*self._profile_path_texts())
        QGuiApplication.clipboard().setText(command)
        print(f"Copied: {command}")

    def set_remote_profiles(self, profiles: object) -> None:
        if not isinstance(profiles, dict):
            self._remote_profiles = {}
            self._update_profile_tabs()
            self._update_profile_actions()
            return
        self._remote_profiles = {
            name: {key: str(values.get(key, "")) for key in ("raw", "preprocessed", "results")}
            for name, values in profiles.items()
            if isinstance(name, str) and isinstance(values, dict)
        }
        self._update_profile_tabs()
        self._update_profile_actions()

    def paths(self) -> tuple[Path | None, Path | None, Path | None]:
        return tuple(Path(field.text()).expanduser() if field.text().strip() else None
                     for field in self._fields.values())  # type: ignore[return-value]

    def set_paths(self, raw: str, preprocessed: str, results: str) -> None:
        self._local_path_texts = (raw, preprocessed, results)
        if self._active_profile != "Local":
            return
        for field, value in zip(self._fields.values(), (raw, preprocessed, results)):
            field.setText(value)
        self._saved_path_texts = self._path_texts()

    def local_paths(self) -> tuple[Path | None, Path | None, Path | None]:
        return tuple(Path(value).expanduser() if value.strip() else None for value in self._local_path_texts)  # type: ignore[return-value]
