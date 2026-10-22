from __future__ import annotations

import os
import re
import shlex
from collections.abc import Sequence
from pathlib import Path

# Commands are kept as POSIX-quoted strings inside nnWorkbench (created with shlex.join, so Windows paths survive).
# What the user sees or copies uses one quoting that bash/zsh and PowerShell (5.1 and 7) read the same way:
# single quotes, one command per line. The Windows command prompt (cmd) is not supported.
WINDOWS = os.name == "nt"
_PLAIN = re.compile(r"[A-Za-z0-9_./=:-]+")  # needs no quotes in either shell (PowerShell: no , @ $ ( ) { } ; &)
_SINGLE_QUOTES = "'‘’‚‛"  # PowerShell also ends a single-quoted string at the typographic ones


def split_arguments(text: str) -> list[str]:
    """Split additional arguments typed by the user: '…' and "…" both quote, as in bash and PowerShell; on Windows
    backslashes are kept (paths), elsewhere they escape as in a POSIX shell. Raises ValueError (unclosed quote)."""
    if WINDOWS:
        lexer = shlex.shlex(text, posix=True)
        lexer.whitespace_split = True
        lexer.escape = ""  # C:\data stays C:\data
        lexer.commenters = ""
        return list(lexer)
    return shlex.split(text)


def shell_quote(argument: str) -> str:
    """One argument for bash/zsh and PowerShell: as it is when plain, else in single quotes. A value with a single
    quote has no form both read alike: it gets the quoting of the platform's shell."""
    if _PLAIN.fullmatch(argument):
        return argument
    if not any(quote in argument for quote in _SINGLE_QUOTES):
        return f"'{argument}'"
    if WINDOWS:  # PowerShell doubles a single quote inside single quotes
        return "'" + "".join(character * 2 if character in _SINGLE_QUOTES else character
                             for character in argument) + "'"
    return shlex.quote(argument)


def shell_join(arguments: Sequence[str]) -> str:
    """A command line for bash/zsh and PowerShell (see shell_quote)."""
    return " ".join(shell_quote(argument) for argument in arguments)


def join_commands(commands: Sequence[str]) -> str:
    """Several commands to paste at once: one per line (``&&`` does not exist in Windows PowerShell 5.1)."""
    return "\n".join(commands)


def preprocessing_command(dataset_id: int, configuration: str | Sequence[str], planner: str,
                          additional_arguments: str = "", *, processes: int | None = None,
                          verify: bool = True) -> str:
    configurations = [configuration] if isinstance(configuration, str) else list(configuration)
    arguments = ["nnUNetv2_plan_and_preprocess", "-d", str(dataset_id), "-c", *configurations, "-pl", planner]
    if processes:
        arguments += ["-np", str(processes)]
    if verify:
        arguments.append("--verify_dataset_integrity")
    return shlex.join((*arguments, *split_arguments(additional_arguments)))


def training_command(dataset_id: int, plans_file: str, configuration: str, trainer: str, fold: str,
                     additional_arguments: str = "", *, continue_training: bool = False,
                     device: str | None = None) -> str:
    arguments = ["nnUNetv2_train", str(dataset_id), configuration, fold, "-tr", trainer, "-p", Path(plans_file).stem]
    if device:  # None keeps nnU-Net's default (cuda)
        arguments += ["-device", device]
    if continue_training:
        arguments.append("--c")
    return shlex.join((*arguments, *split_arguments(additional_arguments)))


def prediction_command(dataset_id: int, input_directory: str, output_directory: str, plans: str,
                       configuration: str, additional_arguments: str = "", trainer: str = "nnUNetTrainer",
                       folds: str | Sequence[str] = "0 1 2 3 4", *, disable_tta: bool = False,
                       continue_prediction: bool = False, preprocessing_processes: int = 3,
                       export_processes: int = 3, device: str | None = None) -> str:
    arguments = ["nnUNetv2_predict", "-i", input_directory, "-o", output_directory, "-d",
                 str(dataset_id), "-c", configuration, "-p", plans, "-tr", trainer]
    # a list of fold names as they are; a string is split at spaces ("0 1 2")
    if selected_folds := folds.split() if isinstance(folds, str) else list(folds):
        arguments.extend(("-f", *selected_folds))
    # processes for preprocessing and for exporting the segmentations (nnU-Net's default: 3 each)
    arguments += ["-npp", str(preprocessing_processes), "-nps", str(export_processes)]
    if device:  # None keeps nnU-Net's default (cuda)
        arguments += ["-device", device]
    if disable_tta:
        arguments.append("--disable_tta")
    if continue_prediction:
        arguments.append("--continue_prediction")  # skip images that already have a prediction
    return shlex.join((*arguments, *split_arguments(additional_arguments)))


def wrapped_command(command: str) -> str:
    """Prefix an nnU-Net command with the saved-path nnWorkbench runner, quoted for the platform's shell."""
    return shell_join(("nnworkbench-run", *shlex.split(command)))
