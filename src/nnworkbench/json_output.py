"""JSON output for `--json`: only the result goes to stdout, everything else (progress, library logs) to stderr."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TextIO

SCHEMA_VERSION = 1  # raise when a JSON field is renamed or removed; new fields keep the version
_result_stream: TextIO | None = None  # the real stdout while it is redirected


def use_utf8_output() -> None:
    """Write stdout and stderr as UTF-8 when they are not already (Windows: a pipe or file uses the ANSI code page,
    e.g. cp1252, which cannot encode "→", "＋" or many path characters and crashes or turns "ä" into a byte that a
    UTF-8 reader such as an AI agent shows as "�"). A console window already gets UTF-8 from Python."""
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "").replace("_", "")
        if stream is None or not hasattr(stream, "reconfigure") or encoding == "utf8":
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # already read from, or closed
            pass


def print_json(value: object) -> None:
    """Write the machine-readable result (to the real stdout, also while other output is redirected)."""
    if isinstance(value, dict) and "schema_version" not in value:
        value = {"schema_version": SCHEMA_VERSION, **value}
    stream = _result_stream or sys.stdout
    if stream is sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    stream.write(json.dumps(value, indent=2) + "\n")
    stream.flush()


@contextmanager
def stdout_to_stderr() -> Iterator[None]:
    """Send everything printed to stdout to stderr, at the file-descriptor level.

    Native libraries (MITK/ITK) and worker processes write to file descriptor 1 directly; redirecting the
    descriptor keeps their logs out of the JSON. print_json still writes to the real stdout.
    """
    global _result_stream
    sys.stdout.flush()
    saved = os.dup(1)
    os.dup2(2, 1)
    _result_stream = os.fdopen(os.dup(saved), "w", encoding="utf-8", errors="replace")
    try:
        yield
    finally:
        sys.stdout.flush()
        _result_stream.flush()
        _result_stream.close()
        _result_stream = None
        os.dup2(saved, 1)
        os.close(saved)
