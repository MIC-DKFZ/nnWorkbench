"""Launch one command for the GUI: in its own POSIX process group, optionally with its output in a log file.

``process_runner PROGRAM ARGS…`` replaces itself with the command (POSIX). ``process_runner --log FILE PROGRAM ARGS…``
runs the command with its output going to FILE and copies FILE to its own output while the command runs. The GUI
reads that copy; the log is complete even when nobody reads the copy any more.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path, PureWindowsPath


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
    return exit_status(child.returncode)


# Windows status codes of a crashed program (NTSTATUS), e.g. from a CUDA driver or a native library
WINDOWS_CRASHES = {
    0xC0000005: "access violation",
    0xC0000374: "heap corruption",
    0xC0000409: "fail-fast exception (often a crash in a native library or driver)",
    0xC000001D: "illegal instruction",
    0xC00000FD: "stack overflow",
    0xC000013A: "interrupted with Ctrl+C",
    0xC0000017: "out of memory",
}


def exit_status(code: int, windows: bool | None = None) -> int:
    """A child's exit code as this process should exit with it.

    POSIX: killed by signal N (a negative return code) → 128 + N, the shell's convention. Windows: a crashed program
    returns a 32-bit status such as 0xC0000005, which Python reports as a large positive number that SystemExit
    cannot take (it overflows and exits with -1): it is passed on as the same 32 bits (a negative int), so the caller
    sees 0xC0000005 again."""
    windows = os.name == "nt" if windows is None else windows
    if code < 0:  # POSIX only (Windows reports unsigned codes)
        return 128 - code
    if windows and code > 0x7FFFFFFF:
        return code - (1 << 32)
    return code


def crash_note(program: str, code: int, windows: bool | None = None) -> str:
    """A readable note for a Windows crash status ("" for ordinary exit codes)."""
    windows = os.name == "nt" if windows is None else windows
    if not windows or not 0xC0000000 <= code <= 0xFFFFFFFF:
        return ""
    meaning = WINDOWS_CRASHES.get(code, "the program crashed")
    return f"{PureWindowsPath(program).name} ended with Windows status 0x{code:08X} ({meaning})."


def main() -> None:
    arguments = sys.argv[1:]
    if os.name == "posix":
        os.setsid()
    else:
        # Ctrl+C in the console the GUI was started from reaches this runner too (it shares that console); the
        # command it started has its own hidden console and would keep running unseen, so the runner ignores Ctrl+C.
        # The GUI stops jobs with taskkill, which does not depend on it.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    if arguments[:1] == ["--log"]:
        if len(arguments) < 3:
            raise SystemExit("usage: process_runner --log FILE PROGRAM [ARGUMENTS…]")
        raise SystemExit(run_logged(arguments[2], arguments[3:], Path(arguments[1])))
    if not arguments:
        raise SystemExit("A command is required.")
    os.execvp(arguments[0], arguments)


if __name__ == "__main__":
    main()
