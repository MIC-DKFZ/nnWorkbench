"""Persist and retrieve nnWorkbench's nnU-Net directory configuration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from shlex import quote
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from .json_output import print_json, use_utf8_output


def configuration_path() -> Path:
    """Return the platform-appropriate user configuration file."""
    if sys.platform == "darwin":
        directory = Path.home() / "Library" / "Application Support"
    elif os.name == "nt":
        directory = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        directory = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return directory / "nnworkbench" / "paths.json"


def trainer_directory() -> Path:
    """Return the installed directory supplied to ``nnUNet_extTrainer``."""
    return Path(__file__).parent / "nnunet_trainers"


def typed_path(text: str) -> str:
    """A folder as users paste it: surrounding quotes (Windows "Copy as path"), a ``file:`` URL, non-breaking
    spaces, ``~`` and ``$VAR`` / ``%VAR%`` become the plain path. Surrounding spaces are kept (callers decide)."""
    text = text.replace("\u00a0", " ")
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in "\"'":
        text = stripped[1:-1]
    if text.strip().lower().startswith("file:"):
        url = urlparse(text.strip())
        text = url2pathname(url.path) if not url.netloc or url.netloc == "localhost" \
            else f"//{url.netloc}{unquote(url.path)}"
    text = os.path.expandvars(text)
    try:
        return os.path.expanduser(text)
    except RuntimeError:  # "~nosuchuser/...": used as typed
        return text


NON_ASCII_HINT = ("nnU-Net cannot use folders whose path contains non-ASCII characters (e.g. ä, é, ß) on Windows: "
                  "its image reader (SimpleITK) cannot open such files and preprocessing (blosc2) cannot write there. "
                  r"Move the folders to a path with only ASCII characters (e.g. C:\nnUNet\nnUNet_raw) and set them "
                  "again")


def _utf8_code_page() -> bool:
    """Whether Windows uses UTF-8 as its ANSI code page ("Beta: Use Unicode UTF-8 for worldwide language support")."""
    try:
        import ctypes

        return ctypes.windll.kernel32.GetACP() == 65001
    except (AttributeError, OSError):
        return False


def unusable_for_nnunet(path: str | os.PathLike, windows: bool | None = None) -> bool:
    """Whether nnU-Net cannot read or write files under *path*: on Windows, a path with non-ASCII characters (unless
    the system's code page is UTF-8). nnWorkbench's own tools (MITK, Napari) handle such paths; nnU-Net does not."""
    if windows is None:
        windows = os.name == "nt"
    return windows and not os.fspath(path).isascii() and not _utf8_code_page()


def own_trainer_folders() -> list[str]:
    """The user's own ``nnUNet_extTrainer`` folders (nnU-Net searches them for trainers), without the bundled one."""
    bundled = os.path.abspath(trainer_directory())
    return [folder for folder in os.environ.get("nnUNet_extTrainer", "").split(os.pathsep)
            if folder.strip() and os.path.abspath(folder) != bundled]


def trainer_folders() -> str:
    """The bundled trainers, followed by the user's own ``nnUNet_extTrainer`` folders (nnU-Net takes a list)."""
    return os.pathsep.join([str(trainer_directory()), *own_trainer_folders()])


def save_paths(raw: str, preprocessed: str, results: str) -> None:
    """Store paths in the user's nnWorkbench configuration."""
    path = configuration_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {"raw": raw, "preprocessed": preprocessed, "results": results}
    path.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")


_FOLDER_VARIABLES = ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")


def environment_paths() -> dict[str, str]:
    """nnU-Net's own folder variables, when set (experienced users have them in their shell)."""
    return {name: os.environ[name] for name in _FOLDER_VARIABLES if os.environ.get(name, "").strip()}


def environment_conflicts(paths: dict[str, str]) -> list[tuple[str, str, str]]:
    """(variable, its value, the folder nnWorkbench uses) for each nnU-Net folder variable that points elsewhere
    than ``paths`` (nnWorkbench's saved paths, which win)."""
    def same(first: str, second: str) -> bool:
        try:
            return os.path.normcase(os.path.abspath(os.path.expanduser(first.strip()))) == \
                os.path.normcase(os.path.abspath(os.path.expanduser(second.strip())))
        except ValueError:  # e.g. a null byte
            return first == second
    return [(name, value, paths.get(name, "")) for name, value in environment_paths().items()
            if not same(value, paths.get(name, ""))]


def environment_differences(paths: dict[str, str]) -> list[str]:
    """environment_conflicts as text, e.g. for the CLI note and ``get-paths --json``."""
    return [f"{name}={value} (nnWorkbench uses {used or 'nothing'})"
            for name, value, used in environment_conflicts(paths)]


def note_environment_differences(paths: dict[str, str]) -> None:
    """Say on stderr (stdout stays clean for --json) when nnU-Net's folder variables point elsewhere."""
    differences = environment_differences(paths)
    if differences:
        print("Note: nnWorkbench uses its saved folders, not these variables: " + "; ".join(differences)
              + ". Change the saved folders with nnworkbench-set-paths.", file=sys.stderr)


def configured_paths() -> dict[str, str]:
    """Load the configured paths and bundled trainer extension directory. Without a saved configuration,
    nnU-Net's folder variables are used when they are set."""
    try:
        values = json.loads(configuration_path().read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        if environment_paths():
            return {**{name: environment_paths().get(name, "") for name in _FOLDER_VARIABLES},
                    "nnUNet_extTrainer": trainer_folders()}
        raise RuntimeError("No nnWorkbench paths are configured. Run nnworkbench-set-paths first (or set "
                           "nnUNet_raw, nnUNet_preprocessed and nnUNet_results).") from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError(f"Invalid nnWorkbench path configuration: {configuration_path()}") from error
    except OSError as error:  # a folder, no permission, ...
        raise RuntimeError(f"Cannot read the nnWorkbench path configuration {configuration_path()}: "
                           f"{error.strerror or error}") from error
    if not isinstance(values, dict):
        raise TypeError(f"Invalid nnWorkbench path configuration: {configuration_path()}")

    def text(key: str) -> str:
        value = values.get(key)
        return value if isinstance(value, str) else ""

    return {
        "nnUNet_raw": text("raw"),
        "nnUNet_preprocessed": text("preprocessed"),
        "nnUNet_results": text("results"),
        "nnUNet_extTrainer": trainer_folders(),
    }


def paths_report() -> str:
    """Return the configured paths as copyable assignments: bash/zsh (``nnUNet_raw='/data/raw'``), on Windows
    PowerShell (``$env:nnUNet_raw = 'C:\\data\\raw'``)."""
    if os.name == "nt":
        return "\n".join(f"$env:{name} = '{value.replace(chr(39), chr(39) * 2)}'"
                         for name, value in configured_paths().items())
    return "\n".join(f"{name}={quote(value)}" for name, value in configured_paths().items())


def set_paths_command(raw: str, preprocessed: str, results: str) -> str:
    """Return a pasteable command (bash/zsh or PowerShell) that persists these paths on a remote system."""
    from .services.commands import shell_join

    return shell_join(["nnworkbench-set-paths", "--raw", raw, "--preprocessed", preprocessed, "--results", results])


def set_paths_main() -> None:
    """Save the supplied nnU-Net directories."""
    parser = argparse.ArgumentParser(description="Save the nnU-Net folders for nnWorkbench.")
    parser.add_argument("--raw", required=True, help="Path to nnUNet_raw")
    parser.add_argument("--preprocessed", required=True, help="Path to nnUNet_preprocessed")
    parser.add_argument("--results", required=True, help="Path to nnUNet_results")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON, e.g. for scripts")
    arguments = parser.parse_args()
    use_utf8_output()
    # Absolute and expanded: the remote shell got "~/x" quoted from Export, and nnU-Net runs from any folder.
    paths = [os.path.abspath(typed_path(value).strip()) if value.strip() else ""
             for value in (arguments.raw, arguments.preprocessed, arguments.results)]
    save_paths(*paths)
    missing = [path for path in paths if path and not os.path.isdir(path)]
    non_ascii = [path for path in paths if path and unusable_for_nnunet(path)]
    warnings = [f"These folders do not exist yet: {', '.join(missing)}. Create them (empty is fine) before creating "
                "or preprocessing a dataset." if missing else "",
                f"{NON_ASCII_HINT}: {', '.join(non_ascii)}." if non_ascii else ""]
    if arguments.json:
        print_json({"saved": str(configuration_path()), "paths": configured_paths(), "missing": missing,
                    "non_ascii": non_ascii})
    else:
        print("\n".join([f"Saved nnWorkbench paths to {configuration_path()}", *filter(None, warnings)]))


def get_paths_main() -> None:
    """Print paths saved by :func:`set_paths_main`."""
    parser = argparse.ArgumentParser(description="Show the nnU-Net folders saved for nnWorkbench.")
    parser.add_argument("--json", action="store_true", help="Print the paths as JSON, e.g. for scripts")
    arguments = parser.parse_args()
    use_utf8_output()
    try:
        if arguments.json:
            paths = configured_paths()
            print_json({"config": str(configuration_path()), "paths": paths,
                        "environment_differs": environment_differences(paths)})
        else:
            print(paths_report())
            note_environment_differences(configured_paths())
    except (RuntimeError, TypeError) as error:
        if arguments.json:
            print_json({"error": str(error)})
            raise SystemExit(1) from error
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    set_paths_main()
