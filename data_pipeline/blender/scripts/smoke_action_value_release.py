"""Exercise a packaged Action-Value release on one target per repair action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    release = args.release.resolve()
    records = [
        json.loads(line)
        for line in args.targets.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    representatives = {}
    for item in records:
        representatives.setdefault(item["target_action"], item)
    results = []
    with tempfile.TemporaryDirectory(dir=release.parent) as raw:
        work = Path(raw)
        for action, item in sorted(representatives.items()):
            critic = work / f"{action}.critic.json"
            context = work / f"{action}.context.json"
            output = work / f"{action}.decision.json"
            critic.write_text(
                json.dumps(item["critic_report"], ensure_ascii=False), encoding="utf-8"
            )
            context.write_text(
                json.dumps(item["context"], ensure_ascii=False), encoding="utf-8"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(release / "inference.py"),
                    "--critic-report",
                    str(critic),
                    "--context",
                    str(context),
                    "--device",
                    args.device,
                    "--output",
                    str(output),
                ],
                cwd=release.parent,
                capture_output=True,
                text=True,
            )
            predicted = None
            if output.is_file():
                predicted = json.loads(output.read_text(encoding="utf-8"))["decision"][
                    "action"
                ]
            results.append(
                {
                    "sample_id": item["sample_id"],
                    "expected": action,
                    "predicted": predicted,
                    "returncode": result.returncode,
                    "matches": predicted == action,
                    "stderr": result.stderr[-500:],
                }
            )
    valid = (
        len(results) == 4
        and all(item["returncode"] == 0 and item["matches"] for item in results)
    )
    payload = {"valid": valid, "checked": len(results), "results": results}
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
