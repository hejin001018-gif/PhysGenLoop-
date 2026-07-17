"""Crash-recoverable paired prediction and diagnostics benchmark output."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .contracts import BenchmarkPrediction, BenchmarkSample


AUDITED_METHOD_IDS = frozenset(
    {
        "B1_RULE",
        "M1_GRAPH",
        "M2_CHECKLIST",
        "M3_MECHANICS",
        "M4_VLM",
        "M5_FULL",
        "M5_SHUFFLED_PROMPT_300",
        "M5_ORACLE_PLAN_300",
    }
)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _process_is_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        )
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return ctypes.get_last_error() == 5
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _lock_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="ascii").strip().removeprefix("pid="))
    except (OSError, UnicodeError, ValueError):
        return None


class AuditedBenchmarkMethod(Protocol):
    method_id: str

    def evaluate_audited(
        self,
        sample: BenchmarkSample,
    ) -> tuple[BenchmarkPrediction, dict[str, object]]: ...


class AuditedBenchmarkRunner:
    """Append paired artifacts using a recoverable write-ahead record."""

    def __init__(
        self,
        prediction_path: str | Path,
        diagnostics_path: str | Path,
        *,
        max_new_failures: int = 1,
    ) -> None:
        if (
            isinstance(max_new_failures, bool)
            or not isinstance(max_new_failures, int)
            or max_new_failures < 1
        ):
            raise ValueError("max_new_failures must be a positive integer")
        self.prediction_path = Path(prediction_path)
        self.diagnostics_path = Path(diagnostics_path)
        if self.prediction_path.resolve(strict=False) == self.diagnostics_path.resolve(
            strict=False
        ):
            raise ValueError("prediction and diagnostics paths must differ")
        self.max_new_failures = max_new_failures
        self.lock_path = self.prediction_path.with_suffix(
            self.prediction_path.suffix + ".lock"
        )
        self.pending_path = self.prediction_path.with_suffix(
            self.prediction_path.suffix + ".pending.json"
        )

    @contextmanager
    def _exclusive_lock(self):
        while True:
            try:
                descriptor = os.open(
                    self.lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
                break
            except FileExistsError as exc:
                if _process_is_alive(_lock_pid(self.lock_path)):
                    raise RuntimeError(
                        "benchmark run is already running; "
                        f"lock exists: {self.lock_path}"
                    ) from exc
                try:
                    self.lock_path.unlink()
                    _fsync_directory(self.lock_path.parent)
                except FileNotFoundError:
                    pass
        try:
            os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
            os.fsync(descriptor)
            _fsync_directory(self.lock_path.parent)
            os.close(descriptor)
            descriptor = -1
            yield
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            self.lock_path.unlink(missing_ok=True)
            _fsync_directory(self.lock_path.parent)

    @staticmethod
    def _prediction_key(raw: Mapping[str, Any]) -> tuple[str, str]:
        try:
            return str(raw["sample_id"]), str(raw["method_id"])
        except KeyError as exc:
            raise ValueError("prediction record is missing its key") from exc

    @staticmethod
    def _diagnostic_key(raw: Mapping[str, Any]) -> tuple[str, str]:
        try:
            key = raw["key"]
            if not isinstance(key, Mapping):
                raise TypeError("diagnostic key must be an object")
            return str(key["sample_id"]), str(key["method_id"])
        except (KeyError, TypeError) as exc:
            raise ValueError("diagnostic record is missing its key") from exc

    @staticmethod
    def _load_jsonl(
        path: Path,
        *,
        kind: str,
        key_reader,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        if not path.is_file():
            return {}
        result: dict[tuple[str, str], dict[str, Any]] = {}
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise TypeError(f"{kind} record must be an object")
                key = key_reader(raw)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid {kind} JSONL line {line_number}: {path}"
                ) from exc
            if key in result:
                raise ValueError(
                    f"duplicate {kind} key at JSONL line {line_number}: {key}"
                )
            result[key] = raw
        return result

    def _indexes(self):
        predictions = self._load_jsonl(
            self.prediction_path,
            kind="prediction",
            key_reader=self._prediction_key,
        )
        diagnostics = self._load_jsonl(
            self.diagnostics_path,
            kind="diagnostics",
            key_reader=self._diagnostic_key,
        )
        return predictions, diagnostics

    def _validate_pair(
        self,
        method_ids: set[str],
    ) -> set[tuple[str, str]]:
        predictions, diagnostics = self._indexes()
        diagnostic_keys = set(diagnostics)
        prediction_keys = set(predictions)
        orphan_diagnostics = diagnostic_keys - prediction_keys
        missing_diagnostics = {
            key
            for key in prediction_keys - diagnostic_keys
            if key[1] in AUDITED_METHOD_IDS
        }
        if orphan_diagnostics or missing_diagnostics:
            raise ValueError(
                "asymmetric prediction/diagnostics keys: "
                f"orphan_diagnostics={sorted(orphan_diagnostics)!r}, "
                f"missing_diagnostics={sorted(missing_diagnostics)!r}"
            )
        return prediction_keys & diagnostic_keys

    @staticmethod
    def _append_fsync(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    payload,
                    allow_nan=False,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)

    def _write_pending(
        self,
        prediction: BenchmarkPrediction,
        diagnostics: Mapping[str, Any],
    ) -> None:
        if self.pending_path.exists():
            raise RuntimeError(f"pending benchmark record already exists: {self.pending_path}")
        self.pending_path.parent.mkdir(parents=True, exist_ok=True)
        raw_temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.pending_path.parent,
                prefix=f".{self.pending_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                raw_temporary = handle.name
                json.dump(
                    {
                        "schema_version": "1.0",
                        "prediction": prediction.to_dict(),
                        "diagnostics": diagnostics,
                    },
                    handle,
                    allow_nan=False,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(raw_temporary, self.pending_path)
            raw_temporary = None
            _fsync_directory(self.pending_path.parent)
        finally:
            if raw_temporary is not None:
                Path(raw_temporary).unlink(missing_ok=True)

    def _clear_pending(self) -> None:
        self.pending_path.unlink()
        _fsync_directory(self.pending_path.parent)

    def _load_pending(self) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            raw = json.loads(self.pending_path.read_text(encoding="utf-8"))
            prediction = raw["prediction"]
            diagnostics = raw["diagnostics"]
            if not isinstance(prediction, dict) or not isinstance(diagnostics, dict):
                raise TypeError("pending records must be objects")
            if self._prediction_key(prediction) != self._diagnostic_key(diagnostics):
                raise ValueError("pending prediction and diagnostics keys differ")
            BenchmarkPrediction.from_dict(prediction)
            return prediction, diagnostics
        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(f"invalid pending benchmark record: {self.pending_path}") from exc

    def _recover_pending(self) -> None:
        if not self.pending_path.is_file():
            return
        prediction, diagnostics = self._load_pending()
        key = self._prediction_key(prediction)
        predictions, diagnostic_records = self._indexes()
        if key in predictions and predictions[key] != prediction:
            raise ValueError(f"pending prediction conflicts with completed key: {key}")
        if key in diagnostic_records and diagnostic_records[key] != diagnostics:
            raise ValueError(f"pending diagnostics conflict with completed key: {key}")
        if key not in predictions:
            self._append_fsync(self.prediction_path, prediction)
        if key not in diagnostic_records:
            self._append_fsync(self.diagnostics_path, diagnostics)
        self._clear_pending()

    def _validate_result(
        self,
        expected_key: tuple[str, str],
        prediction: BenchmarkPrediction,
        diagnostics: Mapping[str, Any],
    ) -> None:
        if (prediction.sample_id, prediction.method_id) != expected_key:
            raise ValueError(
                f"method returned mismatched prediction key: {expected_key}"
            )
        if self._diagnostic_key(diagnostics) != expected_key:
            raise ValueError(
                f"method returned mismatched diagnostics key: {expected_key}"
            )

    def run(
        self,
        samples: Sequence[BenchmarkSample],
        methods: Sequence[AuditedBenchmarkMethod],
    ) -> tuple[BenchmarkPrediction, ...]:
        self.prediction_path.parent.mkdir(parents=True, exist_ok=True)
        self.diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        method_ids = {method.method_id for method in methods}
        if len(method_ids) != len(methods):
            raise ValueError("audited methods must have unique method IDs")
        with self._exclusive_lock():
            self._recover_pending()
            completed = self._validate_pair(method_ids)
            new_records: list[BenchmarkPrediction] = []
            new_failures = 0
            for sample in samples:
                for method in methods:
                    key = (sample.sample_id, method.method_id)
                    if key in completed:
                        continue
                    prediction, diagnostics = method.evaluate_audited(sample)
                    self._validate_result(key, prediction, diagnostics)
                    self._write_pending(prediction, diagnostics)
                    self._append_fsync(self.prediction_path, prediction.to_dict())
                    self._append_fsync(self.diagnostics_path, diagnostics)
                    self._clear_pending()
                    completed.add(key)
                    new_records.append(prediction)
                    if prediction.failure is not None:
                        new_failures += 1
                        if new_failures >= self.max_new_failures:
                            raise RuntimeError("new failure budget reached")
        return tuple(new_records)
