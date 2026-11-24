"""Command-line dispatcher for QLF stage 3.

The command implementations live in sibling modules.  They are imported lazily,
after argument parsing and environment setup, so that global help remains
available when optional runtime dependencies are not installed.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from collections.abc import Sequence

# Public command name -> module (relative to this package).
COMMANDS = {
    "features": ".feature_export",
    "area25": ".area25",
    "thresholds": ".thresholds",
    "huber-thresholds": ".huber",
    "atlas": ".atlas",
    "root-exposure": ".root_exposure",
    "baseline-masks": ".baseline_masks",
    "c04-masks": ".build_masks",
    "direct-features": ".direct_features",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qlf-stage3",
        description="Run QLF stage 3 processing commands.",
        epilog=(
            "commands: "
            + ", ".join(f"{name} -> {module}" for name, module in COMMANDS.items())
        ),
    )
    parser.add_argument(
        "--data-root",
        metavar="PATH",
        default=None,
        help="root directory for input data (exported as QLF_DATA_ROOT)",
    )
    parser.add_argument(
        "command",
        metavar="COMMAND",
        choices=COMMANDS,
        help="processing command to run",
    )
    parser.add_argument(
        "args",
        metavar="ARG",
        nargs=argparse.REMAINDER,
        help="arguments forwarded verbatim to the command",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> object:
    """Parse the command line, then dispatch to the selected module's ``main``."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.data_root is not None:
        os.environ["QLF_DATA_ROOT"] = os.path.abspath(args.data_root)

    module_name = COMMANDS[args.command]
    package = __package__ or "qlf_stage3"
    module = importlib.import_module(module_name, package=package)

    previous_argv = sys.argv
    program = previous_argv[0] if previous_argv else parser.prog
    sys.argv = [program, *args.args]
    try:
        return module.main()
    finally:
        sys.argv = previous_argv
