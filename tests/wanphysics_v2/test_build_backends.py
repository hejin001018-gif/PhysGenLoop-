"""V2 真实后端装配（无 GPU）：分卡 + 低分辨率 + capability mask。"""
import yaml
import pytest

_CFG = "/root/PhysGenLoop-/configs/loop_v2.yaml"


def _cfg():
    with open(_CFG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_build_v2_runner_dual_gpu_and_resolution(tmp_path):
    from generators.wanphysics.v2.build_backends import build_v2_runner
    cfg = _cfg()
    runner, critic, artifacts, preflight = build_v2_runner(
        cfg=cfg, run_dir=str(tmp_path), sample_dir=str(tmp_path / "s1"),
        sample_id="s1", allow_proxy_policy=True,
    )
    # 低分辨率落地
    assert runner.generator._height == 480
    assert runner.generator._width == 832
    # 双卡：Wan→GPU0
    assert str(runner.generator._gpu_id) == "0"
    # ProPainter 资产 readiness 与 run-time 开关分离：最终 capability = enabled && preflight。
    assert runner.capability_fn()["reject"] is True
    assert runner.capability_fn()["local_editing"] is (
        bool(cfg.get("local_editing", {}).get("enabled", False))
        and bool(preflight.capability_mask.get("local_editing", False))
    )
    from physgenloop.learning_repair.base_contracts import RepairAction
    local_exec = runner.executor_registry._executors[RepairAction.LOCAL_EDITING]
    assert local_exec.editor.__class__.__name__ == "StrictProPainterLocalEditor"
