from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from .workflow import accept_label, case_ids, dataset_dir, initialise, load_config, status


def parser() -> argparse.ArgumentParser:
    app = argparse.ArgumentParser(
        prog="nnworkbench", description="A GUI for nnU-Net workflows and nnInteractive-powered annotation"
    )
    app.add_argument("--config", type=Path, default=Path("config/local.yaml"))
    commands = app.add_subparsers(dest="command", required=True)
    commands.add_parser("init")
    commands.add_parser("status")
    commands.add_parser("next")
    accept = commands.add_parser("accept")
    accept.add_argument("case_id")
    accept.add_argument("mask", type=Path)
    accept.add_argument("--replace", action="store_true")
    run = commands.add_parser("run")
    run.add_argument("adapter", choices=("interactive", "predict", "plan_and_preprocess", "train"))
    run.add_argument("--image", default="")
    run.add_argument("--scribbles", default="")
    run.add_argument("--output", default="")
    return app


def main() -> None:
    args = parser().parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    base_dir = config_path.parent.parent
    if args.command == "init":
        print(f"Initialised {initialise(config, base_dir)}")
    elif args.command == "status":
        print("\n".join(f"{key}: {value}" for key, value in status(config, base_dir).items()))
    elif args.command == "next":
        root = dataset_dir(config, base_dir)
        labeled = {p.name.removesuffix(".nii.gz") for p in (root / "labelsTr").glob("*.nii.gz")}
        pending = sorted(case_ids(root / "imagesTr") - labeled)
        print(pending[0] if pending else "No pending cases.")
    elif args.command == "accept":
        print(f"Accepted: {accept_label(config, base_dir, args.case_id, args.mask, args.replace)}")
    else:
        command = config.get("adapters", {}).get(args.adapter, [])
        if not command:
            raise SystemExit(f"Adapter '{args.adapter}' is not configured in {args.config}")
        values = {"image": args.image, "scribbles": args.scribbles, "output": args.output,
                  "dataset_id": config["dataset_id"]}
        rendered = [part.format(**values) for part in command]
        print("Running:", shlex.join(rendered))
        subprocess.run(rendered, check=True)


if __name__ == "__main__":
    main()
