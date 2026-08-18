"""Launch one command in an isolated POSIX process group for GUI lifecycle control."""

from __future__ import annotations

import os
import sys


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("A command is required.")
    if os.name == "posix":
        os.setsid()
    program, arguments = sys.argv[1], sys.argv[2:]
    os.execvp(program, [program, *arguments])


if __name__ == "__main__":
    main()
