"""Launch one command for the GUI: in its own POSIX process group, optionally with its output in a log file.

``process_runner PROGRAM ARGS…`` replaces itself with the command (POSIX). ``process_runner --log FILE PROGRAM ARGS…``
runs the command with its output going to FILE and copies FILE to its own output while the command runs. The GUI
reads that copy; when the GUI quits and stops only this runner, the command keeps running and keeps writing to FILE.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


def run_logged(program: str, arguments: list[str], log: Path, poll_seconds: float = 0.2) -> int:
    """Run the command with its output appended to *log*, copy the new output to stdout, return its exit code."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as output:
        start = output.tell()
        # no console window on Windows (this runner has none, so Windows would open one for a console program)
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        child = subprocess.Popen([program, *arguments], stdin=subprocess.DEVNULL, stdout=output,
                                 stderr=subprocess.STDOUT, creationflags=flags)
    forwarding = True
    with log.open("rb") as source:
        source.seek(start)
        while True:
            finished = child.poll() is not None
            chunk = source.read()
            if chunk and forwarding:
                try:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.flush()
                except OSError:  # nobody reads any more (the GUI quit); the command and its log go on
                    forwarding = False
                    # also for the final flush at exit, which would fail with a broken pipe again
                    os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
            if finished:
                break
            time.sleep(poll_seconds)
    code = child.returncode
    return code if code >= 0 else 128 - code  # stopped by a signal: the shell's convention


def main() -> None:
    arguments = sys.argv[1:]
    if os.name == "posix":
        os.setsid()
    if arguments[:1] == ["--log"]:
        if len(arguments) < 3:
            raise SystemExit("usage: process_runner --log FILE PROGRAM [ARGUMENTS…]")
        raise SystemExit(run_logged(arguments[2], arguments[3:], Path(arguments[1])))
    if not arguments:
        raise SystemExit("A command is required.")
    os.execvp(arguments[0], arguments)


if __name__ == "__main__":
    main()
