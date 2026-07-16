"""Replay cached observations and export rule-level PAVG diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from pavg_critic.benchmarking.datasets import load_manifest
from pavg_critic.benchmarking.diagnostics import write_diagnostics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose cached PAVG observations")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=("B1_RULE", "M1_GRAPH", "M2_CHECKLIST", "M3_MECHANICS"),
        default="B1_RULE",
    )
    args = parser.parse_args(argv)
    summary = write_diagnostics(
        load_manifest(args.manifest),
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        mode=args.mode,
    )
    return 2 if summary["failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
