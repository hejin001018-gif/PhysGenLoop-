"""Single CLI entry point for the canonical Learning Repair package."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from . import action_value_cli, classification_cli


CLASSIFICATION_COMMANDS = frozenset(
    {"validate", "collect", "split", "train", "export", "predict"}
)
ACTION_VALUE_COMMANDS = frozenset(
    {
        "verify-baseline",
        "check-compatibility",
        "validate-campaign",
        "run-campaign",
        "build-targets",
        "audit-targets",
        "train-values",
        "adapt-proxy-targets",
        "closed-loop-report",
        "integration-review",
    }
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pavg-repair",
        description=(
            "Canonical Learning Repair CLI for proxy data, Actual Trials, "
            "Action-Value training, execution campaigns, evaluation, and export."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""commands:
  validate / collect / split / train / export / predict
  adapt-proxy-targets / verify-baseline / check-compatibility
  validate-campaign / run-campaign / build-targets / audit-targets
  train-values / evaluate / closed-loop-report / integration-review

`evaluate --manifest ...` evaluates a classification/proxy checkpoint.
`evaluate --targets ... --compatibility ...` evaluates Action-Value policies.
""",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not arguments:
        parser.print_help()
        return 2
    if arguments[0] in {"-h", "--help"}:
        parser.print_help()
        return 0

    command = arguments[0]
    if command in CLASSIFICATION_COMMANDS:
        return classification_cli.main(arguments)
    if command in ACTION_VALUE_COMMANDS:
        return action_value_cli.main(arguments)
    if command == "evaluate":
        has_manifest = "--manifest" in arguments
        has_targets = "--targets" in arguments
        if has_manifest == has_targets:
            parser.error(
                "evaluate requires exactly one input mode: --manifest or --targets"
            )
        delegate = classification_cli if has_manifest else action_value_cli
        return delegate.main(arguments)

    parser.error(f"unknown command: {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
