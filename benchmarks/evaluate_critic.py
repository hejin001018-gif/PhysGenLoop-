"""运行 PAVG Critic 的冻结 B1/M1–M5 评估并输出 JSON。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from pavg_critic.evaluation import (
    ABLATION_MODES,
    load_evaluation_samples,
    run_rule_evaluation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("evaluation/fixtures/critic_mini.json"),
    )
    parser.add_argument(
        "--mode",
        choices=["B1_RULE", "M1_GRAPH", "M2_CHECKLIST", "M3_MECHANICS"],
        default="B1_RULE",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    samples = load_evaluation_samples(args.fixture)
    records, metrics = run_rule_evaluation(samples, mode=args.mode)
    payload = {
        "mode": args.mode,
        "fixture": str(args.fixture.as_posix()),
        "metrics": asdict(metrics),
        "records": [asdict(record) for record in records],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
