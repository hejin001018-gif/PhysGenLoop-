"""Append-only, resumable benchmark execution."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol, Sequence

from .contracts import BenchmarkPrediction, BenchmarkSample


class BenchmarkMethod(Protocol):
    method_id: str

    def evaluate(self, sample: BenchmarkSample) -> BenchmarkPrediction: ...


class BenchmarkRunner:
    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)

    def _completed(self) -> set[tuple[str, str]]:
        if not self.output_path.exists():
            return set()
        completed: set[tuple[str, str]] = set()
        for line_number, line in enumerate(
            self.output_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            try:
                raw = json.loads(line)
                key = (str(raw["sample_id"]), str(raw["method_id"]))
                if key in completed:
                    raise ValueError(
                        f"duplicate prediction key at JSONL line {line_number}: {key}"
                    )
                completed.add(key)
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"invalid prediction JSONL line {line_number}"
                ) from exc
        return completed

    @contextmanager
    def _exclusive_lock(self):
        lock_path = self.output_path.with_suffix(self.output_path.suffix + ".lock")
        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError as exc:
            raise RuntimeError(
                f"benchmark run is already running; lock exists: {lock_path}"
            ) from exc
        try:
            os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
            os.close(descriptor)
            descriptor = -1
            yield
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            lock_path.unlink(missing_ok=True)

    def run(
        self,
        samples: Sequence[BenchmarkSample],
        methods: Sequence[BenchmarkMethod],
        *,
        max_new_failures: int | None = None,
    ) -> tuple[BenchmarkPrediction, ...]:
        if (
            max_new_failures is not None
            and (
                isinstance(max_new_failures, bool)
                or not isinstance(max_new_failures, int)
                or max_new_failures < 1
            )
        ):
            raise ValueError("max_new_failures must be a positive integer")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._exclusive_lock():
            completed = self._completed()
            new_records: list[BenchmarkPrediction] = []
            new_failures = 0
            with self.output_path.open("a", encoding="utf-8") as handle:
                for sample in samples:
                    for method in methods:
                        key = (sample.sample_id, method.method_id)
                        if key in completed:
                            continue
                        prediction = method.evaluate(sample)
                        if (prediction.sample_id, prediction.method_id) != key:
                            raise ValueError(
                                f"method returned mismatched prediction key: {key}"
                            )
                        handle.write(
                            json.dumps(prediction.to_dict(), ensure_ascii=False) + "\n"
                        )
                        handle.flush()
                        os.fsync(handle.fileno())
                        completed.add(key)
                        new_records.append(prediction)
                        if prediction.failure is not None:
                            new_failures += 1
                            if (
                                max_new_failures is not None
                                and new_failures >= max_new_failures
                            ):
                                raise RuntimeError("new failure budget reached")
        return tuple(new_records)


def load_predictions(path: str | Path) -> tuple[BenchmarkPrediction, ...]:
    source = Path(path)
    if not source.is_file():
        return ()
    result: list[BenchmarkPrediction] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        try:
            result.append(BenchmarkPrediction.from_dict(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid prediction JSONL line {line_number}") from exc
    return tuple(result)
