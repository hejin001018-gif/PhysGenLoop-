"""就绪门禁 preflight（V2）。

修复 P0-3 / 方案 §3、§24：在跑任何 GPU 任务前，检查资产/后端/环境是否就绪，并据此
决定 capability mask——ProPainter 缺失时必须把 local_editing 从能力集移除，绝不注册
一个运行时必然失败的 Executor。

本模块只做**只读探测**（路径存在性、端口占用、命令可用性），不下载模型、不起服务、
不写业务产物，可在 CPU 环境安全运行。GPU/显存探测在 nvidia-smi 不可用时优雅降级。
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PREFLIGHT_SCHEMA_VERSION = "v2-preflight/1.0"


# ProPainter full backend requires all three official weights.  The previous
# "any non-empty .pth" check was too permissive: it could mark local editing
# ready when only RAFT was present.  Minimum sizes are intentionally coarse
# lower bounds, not exact hashes, so mirrors/repacked files still pass while
# truncated downloads fail closed.
REQUIRED_PROPAINTER_WEIGHTS: dict[str, int] = {
    "raft-things.pth": 1_000_000,
    "recurrent_flow_completion.pth": 1_000_000,
    "ProPainter.pth": 10_000_000,
}


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass(frozen=True)
class PreflightReport:
    checks: tuple[CheckResult, ...]
    capability_mask: dict[str, bool]
    schema_version: str = PREFLIGHT_SCHEMA_VERSION

    @property
    def all_ok(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.checks if not c.ok)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "all_ok": self.all_ok,
            "missing": list(self.missing),
            "capability_mask": dict(self.capability_mask),
            "checks": [c.to_dict() for c in self.checks],
        }


def _exists(path: str | None) -> bool:
    return bool(path) and Path(path).exists()


def _command_available(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def port_in_use(host: str, port: int, timeout: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        try:
            return sock.connect_ex((host, port)) == 0
        except OSError:
            return False


def check_propainter(repo: str | None) -> CheckResult:
    """ProPainter 仓库 + 推理脚本 + 权重完整性。"""

    if not _exists(repo):
        return CheckResult("propainter_repo", False, f"missing: {repo}")
    script = Path(repo) / "inference_propainter.py"
    if not script.exists():
        return CheckResult("propainter_script", False, f"missing: {script}")
    weights = Path(repo) / "weights"
    if not weights.exists():
        return CheckResult("propainter_weights", False, f"missing weights dir: {weights}")

    # 未下完的 .part 临时文件 → 判失败。
    parts = sorted(p.name for p in weights.glob("*.part"))
    if parts:
        return CheckResult("propainter_weights", False, f"incomplete downloads: {parts}")

    missing: list[str] = []
    too_small: list[str] = []
    for name, min_bytes in REQUIRED_PROPAINTER_WEIGHTS.items():
        path = weights / name
        if not path.exists():
            missing.append(name)
            continue
        size = path.stat().st_size
        if size < int(min_bytes):
            too_small.append(f"{name}:{size}<{min_bytes}")
    if missing or too_small:
        detail = []
        if missing:
            detail.append(f"missing={missing}")
        if too_small:
            detail.append(f"too_small={too_small}")
        return CheckResult("propainter_weights", False, "; ".join(detail))

    return CheckResult(
        "propainter",
        True,
        f"{repo}; weights={sorted(REQUIRED_PROPAINTER_WEIGHTS)}",
    )


def check_ffmpeg() -> CheckResult:
    # 优先系统 ffmpeg；缺失时回退项目内 imageio_ffmpeg 自带二进制。
    if _command_available("ffmpeg"):
        return CheckResult("ffmpeg", True, "system")
    try:
        import imageio_ffmpeg  # noqa: PLC0415

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return CheckResult("ffmpeg", True, f"imageio_ffmpeg:{exe}")
    except Exception:  # noqa: BLE001
        pass
    return CheckResult("ffmpeg", False, "")


def check_cv2() -> CheckResult:
    try:
        import cv2  # noqa: F401,PLC0415

        return CheckResult("cv2", True, "")
    except Exception as exc:  # noqa: BLE001
        return CheckResult("cv2", False, f"{type(exc).__name__}: {exc}")


def check_path(name: str, path: str | None) -> CheckResult:
    return CheckResult(name, _exists(path), f"{path}")


def gpu_count() -> int | None:
    """通过 nvidia-smi 查 GPU 数；不可用返回 None（不报错）。"""

    if not _command_available("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except Exception:  # noqa: BLE001
        return None
    if out.returncode != 0:
        return None
    lines = [ln for ln in out.stdout.strip().splitlines() if ln.strip()]
    return len(lines)


def run_preflight(
    *,
    propainter_repo: str | None = None,
    wan_model: str | None = None,
    vllm_model: str | None = None,
    sam2_ckpt: str | None = None,
    env_file: str | None = None,
    vllm_host: str = "127.0.0.1",
    vllm_port: int = 18000,
    require_local_editing: bool = False,
) -> PreflightReport:
    """汇总资产/后端就绪检查，产出 capability mask。

    capability mask 里 local_editing 仅在 ProPainter+ffmpeg+cv2 全部就绪时为 True，
    否则强制 False（无论配置是否想开）。prompt_repair/global_regeneration/reject
    只要生成器可达即视为可用（生成器就绪属 W4/GPU 授权范围，这里默认 True）。
    """

    checks: list[CheckResult] = []
    pp = check_propainter(propainter_repo)
    ff = check_ffmpeg()
    cv = check_cv2()
    checks.extend([pp, ff, cv])
    if wan_model is not None:
        checks.append(check_path("wan_model", wan_model))
    if vllm_model is not None:
        checks.append(check_path("vllm_model", vllm_model))
    if sam2_ckpt is not None:
        checks.append(check_path("sam2_ckpt", sam2_ckpt))
    if env_file is not None:
        checks.append(check_path("env_file", env_file))

    n_gpu = gpu_count()
    checks.append(CheckResult("gpu", n_gpu is not None and n_gpu > 0, f"count={n_gpu}"))
    checks.append(
        CheckResult("vllm_port_free", not port_in_use(vllm_host, vllm_port), f"{vllm_host}:{vllm_port}")
    )

    local_ok = pp.ok and ff.ok and cv.ok
    capability_mask = {
        "prompt_repair": True,
        "global_regeneration": True,
        "local_editing": bool(local_ok),
        "reject": True,
    }
    # require_local_editing 但后端缺失：保留 mask=False，让上层据 missing 决定是否中止。
    if require_local_editing and not local_ok:
        checks.append(CheckResult("local_editing_required", False, "requested but backend missing"))

    return PreflightReport(checks=tuple(checks), capability_mask=capability_mask)
