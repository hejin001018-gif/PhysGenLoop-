from pathlib import Path

from physgenloop.contracts import GeneratedCandidate
from physgenloop.learning_repair.base_contracts import RepairAction
from physgenloop.learning_repair.contracts import ExecutionRequest, LocalEditTarget, RepairDecision
from physgenloop.learning_repair.executors import PromptRepairExecutor

from generators.wanphysics.v2.executors import AuditedRejectExecutor, MaskSequenceLocalEditingExecutor


class _Generator:
    def generate(self, *, prompt, seed):
        return GeneratedCandidate("new", "/tmp/new.mp4", prompt, seed)


class _Rewriter:
    def repair(self, *, prompt, report):
        return prompt + "\nphysics correction"


class _Report:
    violations = ()


def _decision(action, target=None):
    probabilities = {item: (0.8 if item is action else 0.1) for item in RepairAction}
    return RepairDecision(
        action,
        0.8,
        "fix",
        probabilities,
        probabilities,
        local_target=target,
        source="test_policy",
    )


def _request(decision, **metadata):
    candidate = GeneratedCandidate("src", "/tmp/src.mp4", "current", 1)
    return ExecutionRequest(
        decision=decision,
        candidate=candidate,
        critic_report=_Report(),
        prompt="current",
        seed=7,
        history=(type("Eval", (), {"candidate": candidate, "report": type("R", (), {"decision": "violation", "physics_score": 0.2, "confidence": 0.8})()})(),),
        metadata=metadata,
    )


def test_original_prompt_executor_rewrites_then_generates():
    executor = PromptRepairExecutor(prompt_rewriter=_Rewriter(), generator=_Generator())
    result = executor.execute(_request(_decision(RepairAction.PROMPT_REPAIR)))
    assert result.next_prompt != "current"
    assert result.metadata["prompt_rewriter"] == "_Rewriter"


def test_prompt_executor_rejects_no_change():
    executor = PromptRepairExecutor(
        prompt_rewriter=lambda **kwargs: kwargs["prompt"],
        generator=_Generator(),
    )
    try:
        executor.execute(_request(_decision(RepairAction.PROMPT_REPAIR)))
    except ValueError as exc:
        assert str(exc) == "no_safe_prompt_change"
    else:
        raise AssertionError("no-change prompt must fail")


def test_local_executor_passes_manifest_without_physics_plan(tmp_path):
    manifest = tmp_path / "mask_manifest.json"
    manifest.write_text("{}")
    target = LocalEditTarget("src", ("ball",), 1, 2, (1, 2), str(manifest))

    class _Editor:
        def edit(self, **kwargs):
            assert "physics_plan" not in kwargs
            return GeneratedCandidate(
                "propainter-x",
                "/tmp/propainter-x.mp4",
                kwargs["candidate"].prompt,
                kwargs["seed"],
                {"propainter": {}, "output_validation": {"decode_ok": True}},
            )

    result = MaskSequenceLocalEditingExecutor(editor=_Editor()).execute(
        _request(_decision(RepairAction.LOCAL_EDITING, target))
    )
    assert result.artifacts["mask_manifest"] == str(manifest)
    assert result.metadata["editor_backend"] == "ProPainter"


def test_reject_is_terminal():
    class _Selector:
        def select(self, history):
            return history[-1]

    result = AuditedRejectExecutor(selector=_Selector()).execute(
        _request(_decision(RepairAction.REJECT))
    )
    assert result.status == "rejected" and result.terminal
