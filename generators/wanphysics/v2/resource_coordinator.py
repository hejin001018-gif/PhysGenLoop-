"""GPU 资源协调与 vLLM 进程所有权（V2）。

修复 P2-2 / 方案 §19：现有 Critic 用 ``pkill -9 -f vllm`` 停服务，会误杀同机其他
vLLM 进程。V2 只管理**本 run 自己启动**的 PID：写 ``vllm.pid`` / ``vllm.owner.json``，
停止时只终止 owner manifest 内的 PID 及其进程组，禁止宽泛 pkill。

同时提供双卡亲和策略（GPU0=Wan/ProPainter，GPU1=vLLM/SAM2）与单卡交接回退的声明。

本模块的纯逻辑部分（owner manifest 读写、PID 归属判断、端口检查、GPU 解析）可在无
GPU 的 CPU 环境测试；真正启动/停止进程的部分依赖显式调用，不在导入时触发。
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RESOURCE_SCHEMA_VERSION = "v2-resource/1.0"

GPU_MODE_DUAL = "dual_gpu"
GPU_MODE_SINGLE = "single_handoff"


@dataclass(frozen=True)
class GpuAssignment:
    mode: str
    generator_gpu: int
    critic_gpu: int

    def __post_init__(self) -> None:
        if self.mode not in {GPU_MODE_DUAL, GPU_MODE_SINGLE}:
            raise ValueError(f"invalid gpu mode: {self.mode!r}")

    def generator_env(self) -> dict[str, str]:
        return {"CUDA_VISIBLE_DEVICES": str(self.generator_gpu)}

    def critic_env(self) -> dict[str, str]:
        gpu = self.generator_gpu if self.mode == GPU_MODE_SINGLE else self.critic_gpu
        return {"CUDA_VISIBLE_DEVICES": str(gpu)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RESOURCE_SCHEMA_VERSION,
            "mode": self.mode,
            "generator_gpu": self.generator_gpu,
            "critic_gpu": self.critic_gpu,
        }


def plan_gpu_assignment(gpu_count: int | None, *, requested_mode: str) -> GpuAssignment:
    """根据实际 GPU 数与请求模式决定分配；不足两卡时回退单卡交接。"""

    if gpu_count is None or gpu_count <= 1 or requested_mode == GPU_MODE_SINGLE:
        return GpuAssignment(mode=GPU_MODE_SINGLE, generator_gpu=0, critic_gpu=0)
    return GpuAssignment(mode=GPU_MODE_DUAL, generator_gpu=0, critic_gpu=1)


@dataclass
class VllmOwner:
    """本 run 拥有的 vLLM 进程记录。"""

    pid: int
    run_id: str
    port: int
    host: str = "127.0.0.1"
    started_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RESOURCE_SCHEMA_VERSION,
            "pid": self.pid,
            "run_id": self.run_id,
            "port": self.port,
            "host": self.host,
            "started_at": self.started_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "VllmOwner":
        return cls(
            pid=int(raw["pid"]),
            run_id=str(raw["run_id"]),
            port=int(raw["port"]),
            host=str(raw.get("host", "127.0.0.1")),
            started_at=raw.get("started_at"),
        )


def write_owner(run_dir: str | Path, owner: VllmOwner) -> Path:
    d = Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "vllm.pid").write_text(str(owner.pid), encoding="utf-8")
    owner_path = d / "vllm.owner.json"
    owner_path.write_text(json.dumps(owner.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return owner_path


def read_owner(run_dir: str | Path) -> VllmOwner | None:
    p = Path(run_dir) / "vllm.owner.json"
    if not p.exists():
        return None
    try:
        return VllmOwner.from_dict(json.loads(p.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        return None


def pid_alive(pid: int) -> bool:
    """进程是否存活（跨平台尽力：POSIX 用 signal 0）。"""

    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def port_in_use(host: str, port: int, timeout: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        try:
            return sock.connect_ex((host, port)) == 0
        except OSError:
            return False


def stop_owned_vllm(run_dir: str | Path, *, grace_seconds: int = 10) -> dict[str, Any]:
    """只停止 owner manifest 记录的 PID 及其进程组，绝不宽泛 pkill。

    返回操作审计字典。owner 不存在或 PID 已死时安全 no-op。
    """

    owner = read_owner(run_dir)
    if owner is None:
        return {"action": "noop", "reason": "no owner manifest"}
    if not pid_alive(owner.pid):
        return {"action": "noop", "reason": "pid not alive", "pid": owner.pid}
    result: dict[str, Any] = {"pid": owner.pid, "run_id": owner.run_id, "steps": []}
    # 优先停整个进程组（start_new_session 启动时 pgid==pid）。
    try:
        os.killpg(os.getpgid(owner.pid), signal.SIGTERM)
        result["steps"].append("SIGTERM pgid")
    except Exception as exc:  # noqa: BLE001
        result["steps"].append(f"SIGTERM failed: {exc}")
        try:
            os.kill(owner.pid, signal.SIGTERM)
            result["steps"].append("SIGTERM pid")
        except Exception as exc2:  # noqa: BLE001
            result["steps"].append(f"SIGTERM pid failed: {exc2}")
    result["action"] = "stopped"
    return result


def nvidia_smi_pid_map() -> dict[int, list[int]] | None:
    """返回 {gpu_index: [pid,...]}，用于审计进程是否落在预期卡上。

    nvidia-smi 不可用时返回 None（不报错），供 CPU 环境优雅跳过。
    """

    if not _which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except Exception:  # noqa: BLE001
        return None
    if out.returncode != 0:
        return None
    mapping: dict[int, list[int]] = {}
    # gpu_uuid 无法直接映射到 index，这里退化为 {0:[pids]} 以提供 PID 清单。
    pids: list[int] = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[1].isdigit():
            pids.append(int(parts[1]))
    if pids:
        mapping[0] = pids
    return mapping


def _which(cmd: str) -> bool:
    import shutil

    return shutil.which(cmd) is not None
