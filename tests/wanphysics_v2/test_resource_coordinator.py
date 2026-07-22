"""GPU 分配 + vLLM owner PID 所有权（只停自己）。"""
from generators.wanphysics.v2.resource_coordinator import (
    plan_gpu_assignment, VllmOwner, write_owner, read_owner, stop_owned_vllm,
    GPU_MODE_DUAL, GPU_MODE_SINGLE,
)


def test_gpu_plan():
    assert plan_gpu_assignment(2, requested_mode=GPU_MODE_DUAL).mode == GPU_MODE_DUAL
    assert plan_gpu_assignment(1, requested_mode=GPU_MODE_DUAL).mode == GPU_MODE_SINGLE
    assert plan_gpu_assignment(None, requested_mode=GPU_MODE_DUAL).mode == GPU_MODE_SINGLE


def test_dual_assigns_distinct_gpus():
    a = plan_gpu_assignment(2, requested_mode=GPU_MODE_DUAL)
    assert a.generator_env()["CUDA_VISIBLE_DEVICES"] == "0"
    assert a.critic_env()["CUDA_VISIBLE_DEVICES"] == "1"


def test_owner_write_read(tmp_path):
    owner = VllmOwner(pid=99999, run_id="r1", port=18000)
    write_owner(tmp_path, owner)
    got = read_owner(tmp_path)
    assert got.pid == 99999 and got.run_id == "r1"


def test_stop_noop_without_owner(tmp_path):
    assert stop_owned_vllm(tmp_path)["action"] == "noop"


def test_stop_noop_dead_pid(tmp_path):
    write_owner(tmp_path, VllmOwner(pid=2, run_id="r1", port=1))  # pid 2 基本不属于本进程可控
    res = stop_owned_vllm(tmp_path)
    assert res["action"] == "noop" or "pid" in res
