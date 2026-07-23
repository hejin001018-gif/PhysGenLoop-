from pathlib import Path

import yaml


_ROOT = Path(__file__).resolve().parents[2]


def _cfg():
    return yaml.safe_load((_ROOT / "configs" / "loop_v2.yaml").read_text(encoding="utf-8"))


def test_build_v2_runner_registers_only_authoritative_three_actions(tmp_path):
    from generators.wanphysics.v2.build_backends import build_v2_runner
    from physgenloop.learning_repair.base_contracts import RepairAction

    runner, critic, artifacts, preflight = build_v2_runner(
        cfg=_cfg(),
        run_dir=str(tmp_path),
        sample_dir=str(tmp_path / "s1"),
        sample_id="s1",
        original_prompt="a red ball falls",
    )
    assert tuple(action.value for action in RepairAction) == (
        "prompt_repair",
        "local_editing",
        "reject",
    )
    assert runner.executor_registry.actions == tuple(RepairAction)
    prompt_executor = runner.executor_registry._executors[RepairAction.PROMPT_REPAIR]
    assert prompt_executor.__class__.__name__ == "PromptRepairExecutor"
    assert prompt_executor.prompt_rewriter.__class__.__name__ == "InstructionPromptRepairer"
    assert runner.executor_registry._executors[
        RepairAction.LOCAL_EDITING
    ].editor._backend.__class__.__name__ == "StrictProPainterLocalEditor"
    assert "global_regeneration" not in runner.capability_fn()


def test_active_build_rejects_shadow(tmp_path):
    import pytest
    from generators.wanphysics.v2.build_backends import build_v2_runner

    cfg = _cfg()
    cfg["acceptance"]["mode"] = "shadow"
    with pytest.raises(ValueError, match="acceptance.mode=enforce"):
        build_v2_runner(
            cfg=cfg,
            run_dir=str(tmp_path),
            sample_dir=str(tmp_path / "s1"),
            sample_id="s1",
            original_prompt="p",
        )
