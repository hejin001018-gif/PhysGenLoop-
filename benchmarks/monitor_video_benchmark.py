"""Append non-secret progress/GPU/endpoint heartbeats for benchmark runs."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


def _prediction_stats(path: Path, method: str) -> dict[str, int]:
    if not path.is_file():
        return {"prediction_count": 0, "failure_count": 0}
    keys: set[tuple[str, str]] = set()
    failures = 0
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        try:
            raw = json.loads(line)
            sample_id = str(raw["sample_id"])
            method_id = str(raw["method_id"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(
                f"invalid prediction heartbeat input line {line_number}: {path}"
            ) from exc
        if method_id != method:
            continue
        key = (sample_id, method_id)
        if key in keys:
            raise ValueError(f"duplicate heartbeat prediction key: {key}")
        keys.add(key)
        failures += raw.get("failure") is not None
    return {"prediction_count": len(keys), "failure_count": failures}


def query_gpu() -> dict[str, int] | dict[str, None]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return {
            "utilization_percent": None,
            "memory_used_mib": None,
            "memory_total_mib": None,
        }
    first = result.stdout.splitlines()[0]
    try:
        utilization, used, total = (int(value.strip()) for value in first.split(","))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid nvidia-smi monitor output: {first!r}") from exc
    return {
        "utilization_percent": utilization,
        "memory_used_mib": used,
        "memory_total_mib": total,
    }


def probe_endpoint(endpoint: str, *, timeout_sec: float = 5.0) -> bool:
    parsed = urllib.parse.urlsplit(endpoint)
    health_url = urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, "/health", "", "")
    )
    try:
        with urllib.request.urlopen(health_url, timeout=timeout_sec) as response:
            return 200 <= int(response.status) < 300
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False


def build_snapshot(
    run_specs: Mapping[str, str | Path],
    *,
    expected_per_method: int,
    previous: Mapping[str, Any] | None,
    now: float,
    stall_sec: float,
    gpu_query: Callable[[], Mapping[str, Any]],
    endpoint_probe: Callable[[], bool],
) -> dict[str, Any]:
    if expected_per_method <= 0:
        raise ValueError("expected_per_method must be positive")
    if stall_sec <= 0:
        raise ValueError("stall_sec must be positive")
    methods = {}
    previous_methods = (
        previous.get("methods", {})
        if previous is not None and isinstance(previous.get("methods", {}), Mapping)
        else {}
    )
    for method, raw_path in sorted(run_specs.items()):
        path = Path(raw_path)
        if path.is_dir():
            path = path / "predictions.jsonl"
        stats = _prediction_stats(path, method)
        previous_method = previous_methods.get(method)
        if not isinstance(previous_method, Mapping):
            previous_method = None
        if previous_method is None and previous is not None and len(run_specs) == 1:
            previous_method = previous
        previous_method_count = (
            None
            if previous_method is None
            else int(previous_method.get("prediction_count", 0))
        )
        method_progressed = (
            previous_method_count is None
            or stats["prediction_count"] > previous_method_count
        )
        method_last_progress = (
            now
            if method_progressed
            else float(previous_method.get("last_progress_epoch", now))
        )
        method_eta = None
        if (
            previous is not None
            and previous_method_count is not None
            and stats["prediction_count"] > previous_method_count
        ):
            elapsed = now - float(previous.get("timestamp_epoch", now))
            rate = (
                (stats["prediction_count"] - previous_method_count) / elapsed
                if elapsed > 0
                else 0.0
            )
            if rate > 0:
                method_eta = (
                    expected_per_method - stats["prediction_count"]
                ) / rate
        methods[method] = {
            **stats,
            "last_progress_epoch": method_last_progress,
            "stalled": stats["prediction_count"] < expected_per_method
            and now - method_last_progress >= stall_sec,
            "eta_sec": method_eta,
        }
    prediction_count = sum(item["prediction_count"] for item in methods.values())
    failure_count = sum(item["failure_count"] for item in methods.values())
    expected_count = expected_per_method * len(methods)
    previous_count = (
        None if previous is None else int(previous.get("prediction_count", 0))
    )
    progressed = previous_count is None or prediction_count > previous_count
    last_progress = (
        now
        if progressed
        else float(previous.get("last_progress_epoch", now))
    )
    eta_sec = None
    if previous is not None and prediction_count > previous_count:
        elapsed = now - float(previous.get("timestamp_epoch", now))
        rate = (prediction_count - previous_count) / elapsed if elapsed > 0 else 0.0
        if rate > 0:
            eta_sec = (expected_count - prediction_count) / rate
    return {
        "schema_version": "1.0",
        "timestamp_epoch": now,
        "timestamp_utc": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "methods": methods,
        "prediction_count": prediction_count,
        "expected_count": expected_count,
        "failure_count": failure_count,
        "last_progress_epoch": last_progress,
        "stalled": any(item["stalled"] for item in methods.values()),
        "eta_sec": eta_sec,
        "endpoint_healthy": bool(endpoint_probe()),
        "gpu": dict(gpu_query()),
        "secrets_recorded": False,
    }


def append_heartbeat(path: str | Path, snapshot: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                snapshot,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _last_heartbeat(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return None
    try:
        raw = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid final heartbeat line: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"final heartbeat must be an object: {path}")
    return raw


def _run_spec(value: str) -> tuple[str, Path]:
    method, separator, raw_path = value.partition("=")
    if not separator or not method or not raw_path:
        raise argparse.ArgumentTypeError("--run must use METHOD=PATH")
    return method, Path(raw_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor PAVG benchmark progress")
    parser.add_argument("--run", required=True, action="append", type=_run_spec)
    parser.add_argument("--expected-per-method", required=True, type=int)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--heartbeat", required=True, type=Path)
    parser.add_argument("--interval-sec", type=float, default=300.0)
    parser.add_argument("--stall-sec", type=float, default=900.0)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_specs = dict(args.run)
    if len(run_specs) != len(args.run):
        raise ValueError("--run method IDs must be unique")
    if args.interval_sec <= 0:
        raise ValueError("--interval-sec must be positive")
    previous = _last_heartbeat(args.heartbeat)
    while True:
        snapshot = build_snapshot(
            run_specs,
            expected_per_method=args.expected_per_method,
            previous=previous,
            now=time.time(),
            stall_sec=args.stall_sec,
            gpu_query=query_gpu,
            endpoint_probe=lambda: probe_endpoint(args.endpoint),
        )
        append_heartbeat(args.heartbeat, snapshot)
        if args.once:
            return 2 if snapshot["stalled"] else 0
        previous = snapshot
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    raise SystemExit(main())
