import importlib.util

import pytest

from nnworkbench import app


def test_gui_start_in_the_core_installation_says_to_install_all(monkeypatch) -> None:
    gui_modules = {"qtpy", "PySide6", "PyQt5", "napari", "mitk"}
    original = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *args: None if name in gui_modules else original(name, *args))

    assert app.missing_gui_packages() == ["qtpy", "a Qt binding", "napari", "mitk-python"]
    with pytest.raises(SystemExit) as stop:
        app.main()
    assert 'pip install "nnworkbench[all]"' in str(stop.value)


def test_one_missing_package_is_named() -> None:
    assert "napari is not installed" in app.missing_gui_message(["napari"])
