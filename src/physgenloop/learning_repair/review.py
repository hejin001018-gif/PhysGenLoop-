"""Milestone 5 integration-review evidence without merging shared contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .compatibility import CompatibilityManifest
from .contracts import utc_now


def build_integration_review(
    *,
    compatibility: CompatibilityManifest,
    executor_manifest: Mapping[str, Any],
    baseline_verification: Mapping[str, Any],
    test_evidence: Mapping[str, Any],
    evaluation_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    masked = list(executor_manifest.get("masked_actions", ()))
    gates = {
        "proxy_baseline_archived": bool(baseline_verification.get("valid")),
        "critic_compatibility_frozen": bool(compatibility.compatibility_id),
        "source_revision_deployable": compatibility.deployment_ready,
        "reject_executor_available": any(
            item.get("action") == "reject"
            for item in executor_manifest.get("capabilities", ())
        ),
        "fake_backend_tests_passed": bool(test_evidence.get("passed")),
        "domain_separated_evaluation_present": bool(
            evaluation_evidence and evaluation_evidence.get("by_domain")
        ),
    }
    all_research_gates = all(
        value
        for name, value in gates.items()
        if name not in {"source_revision_deployable", "domain_separated_evaluation_present"}
    )
    return {
        "schema_version": "learning-repair-integration-review/1.0",
        "created_at": utc_now(),
        "compatibility": compatibility.to_dict(),
        "executor_capabilities": dict(executor_manifest),
        "masked_actions": masked,
        "gates": gates,
        "research_entry_ready": all_research_gates,
        "mainline_merge_ready": all(gates.values()) and not masked,
        "recommendation": "use_canonical_learning_repair_loop_runner",
        "required_team_decision": (
            "Team approval is required before any shared Controller, Protocol, or schema change."
        ),
        "test_evidence": dict(test_evidence),
        "evaluation_evidence": None if evaluation_evidence is None else dict(evaluation_evidence),
    }


def write_integration_review(
    review: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, Path]:
    json_destination = Path(json_path)
    markdown_destination = Path(markdown_path)
    if json_destination.exists() or markdown_destination.exists():
        raise FileExistsError("integration review outputs already exist")
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    markdown_destination.parent.mkdir(parents=True, exist_ok=True)
    json_destination.write_text(
        json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    gate_lines = "\n".join(
        f"- [{'x' if passed else ' '}] `{name}`"
        for name, passed in review["gates"].items()
    )
    markdown = f"""# Learning Repair Agent 主线集成评审

生成时间：{review['created_at']}

## 结论

- 研究入口就绪：`{str(review['research_entry_ready']).lower()}`
- 主线合并就绪：`{str(review['mainline_merge_ready']).lower()}`
- 当前建议：`{review['recommendation']}`
- 团队决策要求：{review['required_team_decision']}

## 门禁

{gate_lines}

## 边界

本评审只提交设计、私有契约、独立 Runner 与测试证据。未经团队批准，不修改
`LoopController`、共享 Protocol、`CriticReport` 或现有模型输入输出。
"""
    markdown_destination.write_text(markdown, encoding="utf-8")
    return json_destination, markdown_destination
