"""Prepare immutable VideoPhy manifests for PAVG evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pavg_critic.benchmarking.datasets import (
    load_manifest,
    load_videophy_csv,
    materialize_video_csv,
    select_smoke_samples,
    split_diagnostic_samples,
    write_source_smoke_csv,
    write_manifest,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare immutable VideoPhy benchmark manifests"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("inspect")
    inspect_parser.add_argument("--csv", required=True, type=Path)
    download_parser = commands.add_parser("download")
    download_parser.add_argument("--csv", required=True, type=Path)
    download_parser.add_argument("--video-dir", required=True, type=Path)
    download_parser.add_argument("--output-csv", required=True, type=Path)
    normalize_parser = commands.add_parser("normalize")
    normalize_parser.add_argument("--csv", required=True, type=Path)
    normalize_parser.add_argument("--benchmark", required=True)
    normalize_parser.add_argument("--split", required=True)
    normalize_parser.add_argument("--output", required=True, type=Path)
    smoke_parser = commands.add_parser("smoke")
    smoke_parser.add_argument("--manifest", required=True, type=Path)
    smoke_parser.add_argument("--count", required=True, type=int)
    smoke_parser.add_argument("--seed", required=True, type=int)
    smoke_parser.add_argument("--output", required=True, type=Path)
    source_smoke_parser = commands.add_parser("source-smoke")
    source_smoke_parser.add_argument("--csv", required=True, type=Path)
    source_smoke_parser.add_argument("--count", required=True, type=int)
    source_smoke_parser.add_argument("--seed", required=True, type=int)
    source_smoke_parser.add_argument("--output-csv", required=True, type=Path)
    split_parser = commands.add_parser("split")
    split_parser.add_argument("--manifest", required=True, type=Path)
    split_parser.add_argument("--dev-count", required=True, type=int)
    split_parser.add_argument("--seed", required=True, type=int)
    split_parser.add_argument("--dev-output", required=True, type=Path)
    split_parser.add_argument("--eval-output", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.command == "inspect":
        with args.csv.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            rows = sum(1 for _ in reader)
            print(
                json.dumps(
                    {"columns": reader.fieldnames or [], "rows": rows},
                    ensure_ascii=False,
                )
            )
        return 0
    if args.command == "download":
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        failures = materialize_video_csv(
            args.csv,
            video_dir=args.video_dir,
            output_csv=args.output_csv,
        )
        failure_path = args.output_csv.with_name("download_failures.jsonl")
        failure_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in failures),
            encoding="utf-8",
        )
        return 2 if failures else 0
    if args.command == "normalize":
        write_manifest(
            load_videophy_csv(
                args.csv,
                benchmark=args.benchmark,
                split=args.split,
            ),
            args.output,
        )
        return 0
    if args.command == "source-smoke":
        write_source_smoke_csv(
            args.csv,
            args.output_csv,
            count=args.count,
            seed=args.seed,
        )
        return 0
    if args.command == "split":
        dev, evaluation = split_diagnostic_samples(
            load_manifest(args.manifest),
            dev_count=args.dev_count,
            seed=args.seed,
        )
        write_manifest(dev, args.dev_output)
        write_manifest(evaluation, args.eval_output)
        return 0
    selected = select_smoke_samples(
        load_manifest(args.manifest),
        count=args.count,
        seed=args.seed,
    )
    write_manifest(selected, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
