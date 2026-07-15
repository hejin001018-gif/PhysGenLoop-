# Critic Benchmark Stage A Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible 20-video VideoPhy-2 smoke evaluation comparing direct VLM judging, structured direct VLM judging, and PAVG B1/M3 outputs without making benchmark-performance claims.

**Architecture:** Add a focused `pavg_critic.benchmarking` package that normalizes external samples, samples identical visual evidence for matched VLM baselines, records append-only predictions, adapts PAVG reports, computes smoke metrics, and renders an auditable report. External data is downloaded through a manifest-preparation CLI; production Critic code remains unchanged except for reuse of its public protocols and report schemas.

**Tech Stack:** Python 3.12, frozen dataclasses, JSON/JSONL, CSV, OpenCV, existing `MultimodalStructuredModel` and `PhysicsCritic` APIs, pytest, stdlib statistics and file hashing.

**Approved design:** `docs/superpowers/specs/2026-07-15-critic-benchmark-evaluation-design.md`

---

## File map

- Create `src/pavg_critic/benchmarking/__init__.py`: public benchmarking exports.
- Create `src/pavg_critic/benchmarking/contracts.py`: normalized samples, predictions and run failures.
- Create `src/pavg_critic/benchmarking/datasets.py`: VideoPhy CSV normalization, immutable manifests and deterministic smoke selection.
- Create `src/pavg_critic/benchmarking/frames.py`: deterministic uniform frame sampling and JPEG data URLs.
- Create `src/pavg_critic/benchmarking/baselines.py`: D0/D1 schema-constrained VLM judges.
- Create `src/pavg_critic/benchmarking/metrics.py`: smoke-safe classification, ordinal and failure metrics.
- Create `src/pavg_critic/benchmarking/runner.py`: append-only resumable paired runner.
- Create `src/pavg_critic/benchmarking/pavg_methods.py`: B1/M3 adapters and reusable observation cache.
- Create `src/pavg_critic/benchmarking/report.py`: JSON and Markdown smoke reports.
- Create `benchmarks/prepare_videophy_manifest.py`: external CSV inspection, normalization, download and selection CLI.
- Create `benchmarks/evaluate_video_benchmark.py`: model construction and run orchestration CLI.
- Create `tests/benchmarking/`: focused unit and contract tests.
- Modify `README.md`: Stage A commands, claims boundary and artifact locations.

The first implementation does not move or rename `src/pavg_critic/evaluation.py`; its six frozen trajectory fixtures remain a separate regression suite.

### Task 1: Define canonical benchmark contracts

**Files:**
- Create: `src/pavg_critic/benchmarking/__init__.py`
- Create: `src/pavg_critic/benchmarking/contracts.py`
- Test: `tests/benchmarking/test_contracts.py`

- [ ] **Step 1: Write failing contract tests**

```python
from pathlib import Path

import pytest

from pavg_critic.benchmarking.contracts import BenchmarkPrediction, BenchmarkSample


def test_sample_requires_existing_local_video(tmp_path: Path):
    with pytest.raises(ValueError, match="video does not exist"):
        BenchmarkSample(
            sample_id="vp2-1",
            benchmark="videophy2",
            split="test",
            prompt="A ball rolls down a ramp.",
            video_path=str(tmp_path / "missing.mp4"),
            prompt_group_id="rolling_ball",
            generator="model-a",
            semantic_label="adherent",
            physics_label="physical",
        )


def test_prediction_round_trip_preserves_unknown_and_failure():
    prediction = BenchmarkPrediction(
        sample_id="vp2-1",
        method_id="D0_DIRECT_VLM",
        model_id="fake-vlm",
        semantic_score=None,
        physics_score=None,
        semantic_label="unknown",
        physics_label="unknown",
        confidence=0.0,
        coverage=0.0,
        latency_sec=1.25,
        visible_frame_count=8,
        violation_categories=(),
        evidence_frames=(),
        repair_instruction=None,
        failure={"type": "TimeoutError", "message": "timed out"},
    )
    assert BenchmarkPrediction.from_dict(prediction.to_dict()) == prediction
```

- [ ] **Step 2: Run the tests and verify the import fails**

Run: `\.venv\Scripts\python.exe -m pytest tests/benchmarking/test_contracts.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'pavg_critic.benchmarking'`.

- [ ] **Step 3: Implement frozen contracts with strict labels**

```python
# src/pavg_critic/benchmarking/contracts.py
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

SemanticLabel = Literal["adherent", "not_adherent", "unknown"]
PhysicsLabel = Literal["physical", "violation", "unknown"]


@dataclass(frozen=True)
class BenchmarkSample:
    sample_id: str
    benchmark: str
    split: str
    prompt: str
    video_path: str
    prompt_group_id: str
    generator: str
    semantic_label: SemanticLabel
    physics_label: PhysicsLabel
    semantic_score: float | None = None
    physics_score: float | None = None
    physical_rules: tuple[str, ...] = ()
    raw_labels: Mapping[str, Any] = field(default_factory=dict)
    source_url: str | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.sample_id.strip() or not self.prompt_group_id.strip():
            raise ValueError("sample_id and prompt_group_id must not be empty")
        if not Path(self.video_path).is_file():
            raise ValueError(f"video does not exist: {self.video_path}")
        if self.semantic_label not in {"adherent", "not_adherent", "unknown"}:
            raise ValueError(f"invalid semantic label: {self.semantic_label}")
        if self.physics_label not in {"physical", "violation", "unknown"}:
            raise ValueError(f"invalid physics label: {self.physics_label}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BenchmarkSample":
        data = dict(raw)
        data["physical_rules"] = tuple(data.get("physical_rules", ()))
        return cls(**data)


@dataclass(frozen=True)
class BenchmarkPrediction:
    sample_id: str
    method_id: str
    model_id: str | None
    semantic_score: float | None
    physics_score: float | None
    semantic_label: SemanticLabel
    physics_label: PhysicsLabel
    confidence: float
    coverage: float
    latency_sec: float
    visible_frame_count: int
    violation_categories: tuple[str, ...] = ()
    evidence_frames: tuple[int, ...] = ()
    repair_instruction: str | None = None
    failure: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        for value, name in ((self.confidence, "confidence"), (self.coverage, "coverage")):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.visible_frame_count < 0 or self.latency_sec < 0:
            raise ValueError("frame count and latency must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BenchmarkPrediction":
        data = dict(raw)
        data["violation_categories"] = tuple(data.get("violation_categories", ()))
        data["evidence_frames"] = tuple(data.get("evidence_frames", ()))
        return cls(**data)
```

Export both classes from `src/pavg_critic/benchmarking/__init__.py`.

- [ ] **Step 4: Run focused tests**

Run: `\.venv\Scripts\python.exe -m pytest tests/benchmarking/test_contracts.py -q`

Expected: `2 passed`.

- [ ] **Step 5: Commit the contracts**

```powershell
git add src/pavg_critic/benchmarking tests/benchmarking/test_contracts.py
git commit -m "feat: add benchmark evaluation contracts"
```

### Task 2: Normalize VideoPhy data and select a deterministic smoke subset

**Files:**
- Create: `src/pavg_critic/benchmarking/datasets.py`
- Create: `benchmarks/prepare_videophy_manifest.py`
- Test: `tests/benchmarking/test_datasets.py`

- [ ] **Step 1: Write failing tests for aliases, labels and selection**

```python
import csv
from pathlib import Path

from pavg_critic.benchmarking.datasets import load_videophy_csv, select_smoke_samples


def test_videophy_csv_aliases_and_thresholds(tmp_path: Path):
    video = tmp_path / "a.mp4"
    video.write_bytes(b"fake")
    csv_path = tmp_path / "data.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "caption", "videopath", "sa", "pc", "model", "action"])
        writer.writeheader()
        writer.writerow({
            "id": "1", "caption": "ball rolls", "videopath": str(video),
            "sa": "4", "pc": "3", "model": "gen-a", "action": "roll",
        })
    sample = load_videophy_csv(csv_path, benchmark="videophy2", split="test")[0]
    assert sample.semantic_label == "adherent"
    assert sample.physics_label == "violation"
    assert sample.prompt_group_id == "roll"


def test_smoke_selection_is_stable_and_balanced(sample_factory):
    samples = tuple(sample_factory(index=i, physical=(i % 2 == 0), generator=f"g{i % 3}") for i in range(30))
    first = select_smoke_samples(samples, count=20, seed=20260715)
    second = select_smoke_samples(samples, count=20, seed=20260715)
    assert [item.sample_id for item in first] == [item.sample_id for item in second]
    assert {item.physics_label for item in first} == {"physical", "violation"}
    assert len({item.generator for item in first}) == 3
```

Create the shared fixtures used by all later benchmark tests:

```python
# tests/benchmarking/conftest.py
import pytest

from pavg_critic.benchmarking.contracts import BenchmarkPrediction, BenchmarkSample
from pavg_critic.schemas import FrameState


@pytest.fixture
def sample_factory(tmp_path):
    video = tmp_path / "fixture.mp4"
    video.write_bytes(b"fixture")

    def make(*, index: int, physical: bool, generator: str):
        return BenchmarkSample(
            sample_id=str(index),
            benchmark="videophy2",
            split="test",
            prompt=f"prompt {index}",
            video_path=str(video),
            prompt_group_id=f"action-{index // 2}",
            generator=generator,
            semantic_label="adherent",
            physics_label="physical" if physical else "violation",
            semantic_score=5.0,
            physics_score=5.0 if physical else 2.0,
        )

    return make


@pytest.fixture
def prediction_factory():
    def make(sample_id: str, label: str, score: float | None, *, method_id: str = "D0_DIRECT_VLM"):
        return BenchmarkPrediction(
            sample_id=sample_id,
            method_id=method_id,
            model_id="fake",
            semantic_score=5.0,
            physics_score=score,
            semantic_label="adherent",
            physics_label=label,
            confidence=0.8 if label != "unknown" else 0.0,
            coverage=1.0 if label != "unknown" else 0.0,
            latency_sec=0.1,
            visible_frame_count=4,
        )

    return make


@pytest.fixture
def frame_state_factory():
    def make():
        return FrameState(
            frame=0,
            timestamp_sec=0.0,
            object="ball",
            center=(10.0, 10.0),
            bbox=(5.0, 5.0, 15.0, 15.0),
            track_id="ball-1",
        )

    return make
```

- [ ] **Step 2: Run tests and verify the loader is missing**

Run: `\.venv\Scripts\python.exe -m pytest tests/benchmarking/test_datasets.py -q`

Expected: FAIL importing `pavg_critic.benchmarking.datasets`.

- [ ] **Step 3: Implement alias resolution and normalization**

```python
# src/pavg_critic/benchmarking/datasets.py
from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .contracts import BenchmarkSample

ALIASES = {
    "sample_id": ("sample_id", "id", "video_id"),
    "prompt": ("prompt", "caption", "text"),
    "video_path": ("local_path", "video_path", "videopath", "path", "video_url", "video", "url"),
    "semantic_score": ("semantic_score", "semantic_adherence", "sa"),
    "physics_score": ("physics_score", "physical_commonsense", "pc"),
    "generator": ("generator", "model", "video_model"),
    "prompt_group_id": ("prompt_group_id", "action", "action_id", "prompt_id"),
}


def _value(row: Mapping[str, str], logical_name: str) -> str:
    for column in ALIASES[logical_name]:
        value = row.get(column)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError(f"missing {logical_name}; available columns: {sorted(row)}")


def _score(row: Mapping[str, str], logical_name: str) -> float | None:
    try:
        return float(_value(row, logical_name))
    except ValueError as exc:
        if str(exc).startswith("missing"):
            return None
        raise


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_videophy_csv(path: str | Path, *, benchmark: str, split: str) -> tuple[BenchmarkSample, ...]:
    base = Path(path).resolve().parent
    result = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row_index, row in enumerate(csv.DictReader(handle), start=1):
            semantic = _score(row, "semantic_score")
            physics = _score(row, "physics_score")
            raw_path = _value(row, "video_path")
            video_path = Path(raw_path)
            if not video_path.is_absolute():
                video_path = base / video_path
            result.append(BenchmarkSample(
                sample_id=_value(row, "sample_id"),
                benchmark=benchmark,
                split=split,
                prompt=_value(row, "prompt"),
                video_path=str(video_path.resolve()),
                prompt_group_id=_value(row, "prompt_group_id"),
                generator=_value(row, "generator"),
                semantic_label="unknown" if semantic is None else ("adherent" if semantic >= 4 else "not_adherent"),
                physics_label="unknown" if physics is None else ("physical" if physics >= 4 else "violation"),
                semantic_score=semantic,
                physics_score=physics,
                raw_labels={key: value for key, value in row.items() if key in {"sa", "pc", "semantic_adherence", "physical_commonsense"}},
                sha256=_file_sha256(video_path.resolve()),
            ))
    if not result:
        raise ValueError(f"no rows found in {path}")
    return tuple(result)


def select_smoke_samples(samples: Sequence[BenchmarkSample], *, count: int, seed: int) -> tuple[BenchmarkSample, ...]:
    if count <= 0 or count > len(samples):
        raise ValueError("count must be between 1 and the number of samples")
    buckets: dict[tuple[str, str], list[BenchmarkSample]] = {}
    for sample in samples:
        buckets.setdefault((sample.physics_label, sample.generator), []).append(sample)
    rng = random.Random(seed)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: item.sample_id)
        rng.shuffle(bucket)
    chosen: list[BenchmarkSample] = []
    keys = sorted(buckets)
    while len(chosen) < count and any(buckets.values()):
        for key in keys:
            if buckets[key] and len(chosen) < count:
                chosen.append(buckets[key].pop())
    return tuple(sorted(chosen, key=lambda item: item.sample_id))


def write_manifest(samples: Iterable[BenchmarkSample], path: str | Path) -> None:
    payload = {"schema_version": "1.0", "samples": [item.to_dict() for item in samples]}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_manifest(path: str | Path) -> tuple[BenchmarkSample, ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != "1.0" or not isinstance(raw.get("samples"), list):
        raise ValueError("benchmark manifest must use schema 1.0 with a samples array")
    samples = tuple(BenchmarkSample.from_dict(item) for item in raw["samples"])
    ids = [item.sample_id for item in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("benchmark manifest contains duplicate sample IDs")
    return samples
```

Add the download helper used before `BenchmarkSample` validation:

```python
def materialize_video_csv(
    input_csv: str | Path,
    *,
    video_dir: str | Path,
    output_csv: str | Path,
    timeout_sec: float = 60.0,
) -> tuple[dict[str, str], ...]:
    import shutil
    import urllib.parse
    import urllib.request

    destination = Path(video_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with Path(input_csv).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or ())
    if not rows:
        raise ValueError(f"no rows found in {input_csv}")
    if "local_path" not in fieldnames:
        fieldnames.append("local_path")
    failures = []
    for row_index, row in enumerate(rows, start=1):
        sample_id = _value(row, "sample_id")
        source = _value(row, "video_path")
        parsed = urllib.parse.urlparse(source)
        suffix = Path(parsed.path).suffix.lower() or ".mp4"
        target = destination / f"{sample_id}{suffix}"
        try:
            if target.is_file() and target.stat().st_size > 0:
                row["local_path"] = str(target.resolve())
                continue
            temporary = target.with_suffix(target.suffix + ".part")
            if parsed.scheme in {"http", "https"}:
                request = urllib.request.Request(source, headers={"User-Agent": "PAVG-benchmark/1.0"})
                with urllib.request.urlopen(request, timeout=timeout_sec) as response, temporary.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
            else:
                local_source = Path(source)
                if not local_source.is_absolute():
                    local_source = Path(input_csv).resolve().parent / local_source
                shutil.copyfile(local_source, temporary)
            if temporary.stat().st_size == 0:
                raise ValueError("downloaded file is empty")
            temporary.replace(target)
            row["local_path"] = str(target.resolve())
        except Exception as exc:
            failures.append({"row": row_index, "sample_id": sample_id, "source": source, "error": type(exc).__name__, "message": str(exc)[:500]})
    with Path(output_csv).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return tuple(failures)
```

The preparation CLI has four explicit subcommands:

- `inspect --csv PATH`: print headers and row count without writing data;
- `download --csv PATH --video-dir DIR --output-csv PATH`: materialize videos, write `download_failures.jsonl` beside the output CSV and exit non-zero if any row fails;
- `normalize --csv PATH --benchmark videophy2 --split test --output PATH`;
- `smoke --manifest PATH --count 20 --seed 20260715 --output PATH`.

It exits non-zero with the loader's exact missing-column diagnostic.

Implement its command dispatch as:

```python
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Prepare immutable VideoPhy benchmark manifests")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("inspect")
    inspect_parser.add_argument("--csv", required=True, type=Path)
    download_parser = commands.add_parser("download")
    download_parser.add_argument("--csv", required=True, type=Path)
    download_parser.add_argument("--video-dir", required=True, type=Path)
    download_parser.add_argument("--output-csv", required=True, type=Path)
    normalize_parser = commands.add_parser("normalize")
    normalize_parser.add_argument("--csv", required=True, type=Path)
    normalize_parser.add_argument("--benchmark", required=True)
    normalize_parser.add_argument("--split", required=True)
    normalize_parser.add_argument("--output", required=True, type=Path)
    smoke_parser = commands.add_parser("smoke")
    smoke_parser.add_argument("--manifest", required=True, type=Path)
    smoke_parser.add_argument("--count", required=True, type=int)
    smoke_parser.add_argument("--seed", required=True, type=int)
    smoke_parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "inspect":
        with args.csv.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            rows = sum(1 for _ in reader)
            print(json.dumps({"columns": reader.fieldnames or [], "rows": rows}, ensure_ascii=False))
        return 0
    if args.command == "download":
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        failures = materialize_video_csv(args.csv, video_dir=args.video_dir, output_csv=args.output_csv)
        failure_path = args.output_csv.with_name("download_failures.jsonl")
        failure_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in failures), encoding="utf-8")
        return 2 if failures else 0
    if args.command == "normalize":
        write_manifest(load_videophy_csv(args.csv, benchmark=args.benchmark, split=args.split), args.output)
        return 0
    selected = select_smoke_samples(load_manifest(args.manifest), count=args.count, seed=args.seed)
    write_manifest(selected, args.output)
    return 0
```

Add the necessary `argparse` import and standard `if __name__ == "__main__": raise SystemExit(main())` guard.

- [ ] **Step 4: Run focused tests**

Run: `\.venv\Scripts\python.exe -m pytest tests/benchmarking/test_datasets.py -q`

Expected: PASS.

- [ ] **Step 5: Commit data adapters**

```powershell
git add src/pavg_critic/benchmarking/datasets.py benchmarks/prepare_videophy_manifest.py tests/benchmarking
git commit -m "feat: normalize VideoPhy benchmark manifests"
```

### Task 3: Add deterministic uniform frame sampling

**Files:**
- Create: `src/pavg_critic/benchmarking/frames.py`
- Test: `tests/benchmarking/test_frames.py`

- [ ] **Step 1: Write failing tests for indices and decoder failures**

```python
import pytest

from pavg_critic.benchmarking.frames import uniform_indices


@pytest.mark.parametrize(("total", "count", "expected"), [
    (1, 8, (0,)),
    (5, 5, (0, 1, 2, 3, 4)),
    (10, 3, (0, 4, 9)),
])
def test_uniform_indices_cover_endpoints(total, count, expected):
    assert uniform_indices(total, count) == expected


def test_uniform_indices_reject_invalid_counts():
    with pytest.raises(ValueError, match="positive"):
        uniform_indices(10, 0)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `\.venv\Scripts\python.exe -m pytest tests/benchmarking/test_frames.py -q`

Expected: FAIL importing `frames`.

- [ ] **Step 3: Implement frame extraction and data URLs**

```python
# src/pavg_critic/benchmarking/frames.py
from __future__ import annotations

import base64
from dataclasses import dataclass


def uniform_indices(total_frames: int, count: int) -> tuple[int, ...]:
    if total_frames <= 0 or count <= 0:
        raise ValueError("total_frames and count must be positive")
    if count >= total_frames:
        return tuple(range(total_frames))
    return tuple(round(index * (total_frames - 1) / (count - 1)) for index in range(count))


@dataclass(frozen=True)
class SampledFrames:
    indices: tuple[int, ...]
    data_urls: tuple[str, ...]
    total_frames: int
    fps: float


def sample_video_frames(video_path: str, *, count: int = 16, jpeg_quality: int = 85) -> SampledFrames:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    if total <= 0:
        cap.release()
        raise ValueError(f"video reports no frames: {video_path}")
    requested = uniform_indices(total, count)
    actual: list[int] = []
    urls: list[str] = []
    for frame_index in requested:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if ok:
            actual.append(frame_index)
            urls.append("data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii"))
    cap.release()
    if not urls:
        raise ValueError(f"no requested frames could be decoded: {video_path}")
    return SampledFrames(tuple(actual), tuple(urls), total, fps)
```

- [ ] **Step 4: Add one generated-video contract test and run it**

```python
def test_sample_video_frames_decodes_requested_frames(tmp_path):
    import cv2
    import numpy as np

    path = tmp_path / "five.avi"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 64),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG writer is unavailable")
    for value in range(5):
        writer.write(np.full((64, 64, 3), value * 40, dtype=np.uint8))
    writer.release()

    result = sample_video_frames(str(path), count=3)
    assert result.indices == (0, 2, 4)
    assert len(result.data_urls) == 3
    assert all(item.startswith("data:image/jpeg;base64,") for item in result.data_urls)
    assert result.fps > 0
```

Run: `\.venv\Scripts\python.exe -m pytest tests/benchmarking/test_frames.py -q`

Expected: PASS or SKIP only when the local OpenCV build cannot create the test codec; decoding an existing fixture must still be covered.

- [ ] **Step 5: Commit frame sampling**

```powershell
git add src/pavg_critic/benchmarking/frames.py tests/benchmarking/test_frames.py
git commit -m "feat: add deterministic benchmark frame sampling"
```

### Task 4: Implement matched D0 and D1 VLM judges

**Files:**
- Create: `src/pavg_critic/benchmarking/baselines.py`
- Test: `tests/benchmarking/test_baselines.py`

- [ ] **Step 1: Write failing tests with a scripted multimodal model**

```python
from pavg_critic.benchmarking.baselines import DirectVLMJudge


class ScriptedModel:
    def __init__(self):
        self.calls = []

    def generate_json_with_images(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "semantic_score": 5,
            "physics_score": 2,
            "confidence": 0.8,
            "violation_categories": ["gravity"],
            "reason": "The ball accelerates upward without a force.",
        }


def test_direct_and_structured_judges_use_same_images(sample_factory, monkeypatch):
    model = ScriptedModel()
    frames = type("Frames", (), {"data_urls": ("data:image/jpeg;base64,AA==",) * 4, "indices": (0, 1, 2, 3)})()
    monkeypatch.setattr("pavg_critic.benchmarking.baselines.sample_video_frames", lambda *args, **kwargs: frames)
    sample = sample_factory(index=1, physical=False, generator="g")
    d0 = DirectVLMJudge(model, model_id="fake", structured=False).evaluate(sample)
    d1 = DirectVLMJudge(model, model_id="fake", structured=True).evaluate(sample)
    assert d0.physics_label == d1.physics_label == "violation"
    assert model.calls[0]["image_data_urls"] == model.calls[1]["image_data_urls"]
    assert "checklist" not in model.calls[0]["system_prompt"].lower()
    assert "checklist" in model.calls[1]["system_prompt"].lower()
```

- [ ] **Step 2: Run tests and verify the judge is missing**

Run: `\.venv\Scripts\python.exe -m pytest tests/benchmarking/test_baselines.py -q`

Expected: FAIL importing `DirectVLMJudge`.

- [ ] **Step 3: Implement the fixed schema and prompts**

```python
# src/pavg_critic/benchmarking/baselines.py
from __future__ import annotations

from time import perf_counter

from pavg_critic.interfaces import MultimodalStructuredModel

from .contracts import BenchmarkPrediction, BenchmarkSample
from .frames import sample_video_frames

JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["semantic_score", "physics_score", "confidence", "violation_categories", "reason"],
    "properties": {
        "semantic_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "physics_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "violation_categories": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
}

DIRECT_SYSTEM = """You are evaluating a generated video. Score semantic adherence and physical commonsense from 1 to 5. Use only visible evidence. If evidence is insufficient, lower confidence. Return only the requested JSON object."""

STRUCTURED_SYSTEM = """You are evaluating a generated video with a fixed physical checklist. Check object permanence, gravity and support, contact and collision order, conservation of mass/momentum when visible, material behavior, and temporal continuity. Then score semantic adherence and physical commonsense from 1 to 5. Do not assume events that are not visible. Return only the requested JSON object."""


class DirectVLMJudge:
    def __init__(self, model: MultimodalStructuredModel, *, model_id: str, structured: bool, frame_count: int = 16):
        self.model = model
        self.model_id = model_id
        self.structured = structured
        self.frame_count = frame_count
        self.method_id = "D1_STRUCTURED_VLM" if structured else "D0_DIRECT_VLM"

    def evaluate(self, sample: BenchmarkSample) -> BenchmarkPrediction:
        started = perf_counter()
        frames = sample_video_frames(sample.video_path, count=self.frame_count)
        try:
            result = self.model.generate_json_with_images(
                system_prompt=STRUCTURED_SYSTEM if self.structured else DIRECT_SYSTEM,
                user_prompt=f"Prompt: {sample.prompt}\nFrame indices: {list(frames.indices)}",
                image_data_urls=frames.data_urls,
                schema=JUDGE_SCHEMA,
            )
            semantic_score = float(result["semantic_score"])
            physics_score = float(result["physics_score"])
            return BenchmarkPrediction(
                sample_id=sample.sample_id,
                method_id=self.method_id,
                model_id=self.model_id,
                semantic_score=semantic_score,
                physics_score=physics_score,
                semantic_label="adherent" if semantic_score >= 4 else "not_adherent",
                physics_label="physical" if physics_score >= 4 else "violation",
                confidence=float(result["confidence"]),
                coverage=1.0,
                latency_sec=perf_counter() - started,
                visible_frame_count=len(frames.data_urls),
                violation_categories=tuple(str(item) for item in result["violation_categories"]),
                evidence_frames=frames.indices,
            )
        except Exception as exc:
            return BenchmarkPrediction(
                sample_id=sample.sample_id,
                method_id=self.method_id,
                model_id=self.model_id,
                semantic_score=None,
                physics_score=None,
                semantic_label="unknown",
                physics_label="unknown",
                confidence=0.0,
                coverage=0.0,
                latency_sec=perf_counter() - started,
                visible_frame_count=len(frames.data_urls),
                failure={"type": type(exc).__name__, "message": str(exc)[:500]},
            )
```

- [ ] **Step 4: Test success and timeout records**

```python
class TimeoutModel:
    def generate_json_with_images(self, **kwargs):
        raise TimeoutError("provider timeout")


def test_timeout_becomes_explicit_unknown(sample_factory, monkeypatch):
    frames = type("Frames", (), {"data_urls": ("data:image/jpeg;base64,AA==",), "indices": (0,)})()
    monkeypatch.setattr("pavg_critic.benchmarking.baselines.sample_video_frames", lambda *args, **kwargs: frames)
    sample = sample_factory(index=2, physical=False, generator="g")
    prediction = DirectVLMJudge(TimeoutModel(), model_id="timeout", structured=False).evaluate(sample)
    assert prediction.physics_label == "unknown"
    assert prediction.failure["type"] == "TimeoutError"
    assert prediction.coverage == 0.0
```

Run: `\.venv\Scripts\python.exe -m pytest tests/benchmarking/test_baselines.py -q`

Expected: PASS.

- [ ] **Step 5: Commit direct baselines**

```powershell
git add src/pavg_critic/benchmarking/baselines.py tests/benchmarking/test_baselines.py
git commit -m "feat: add matched direct VLM benchmark baselines"
```

### Task 5: Add smoke metrics with explicit unknown handling

**Files:**
- Create: `src/pavg_critic/benchmarking/metrics.py`
- Test: `tests/benchmarking/test_metrics.py`

- [ ] **Step 1: Write failing metric tests**

```python
from pavg_critic.benchmarking.metrics import compute_smoke_metrics


def test_metrics_count_unknown_as_missed_violation(sample_factory, prediction_factory):
    samples = (
        sample_factory(index=1, physical=False, generator="g"),
        sample_factory(index=2, physical=True, generator="g"),
        sample_factory(index=3, physical=False, generator="g"),
    )
    predictions = (
        prediction_factory("1", "violation", 0.2),
        prediction_factory("2", "physical", 0.9),
        prediction_factory("3", "unknown", None),
    )
    metrics = compute_smoke_metrics(samples, predictions)
    assert metrics["count"] == 3
    assert metrics["macro_f1"] < 1.0
    assert metrics["unknown_rate"] == 1 / 3
    assert metrics["failure_rate"] == 0.0
```

- [ ] **Step 2: Run tests and verify failure**

Run: `\.venv\Scripts\python.exe -m pytest tests/benchmarking/test_metrics.py -q`

Expected: FAIL importing `compute_smoke_metrics`.

- [ ] **Step 3: Implement pure-Python confusion and ordinal metrics**

Implement `compute_smoke_metrics(samples, predictions)` with exact sample-ID alignment and these outputs:

```python
{
    "count": int,
    "accuracy": float,
    "balanced_accuracy": float,
    "macro_f1": float,
    "violation_precision": float,
    "violation_recall": float,
    "unknown_rate": float,
    "failure_rate": float,
    "mean_latency_sec": float,
    "mean_visible_frames": float,
    "physics_spearman": float | None,
}
```

Use the following rules in code:

- missing or duplicated sample IDs raise `ValueError`;
- `unknown` is incorrect for accuracy and a false negative when the gold label is `violation`;
- macro-F1 averages the physical and violation class F1 values with zero-division returning `0.0`;
- Spearman uses average ranks for ties and returns `None` with fewer than two paired ordinal scores or zero rank variance;
- failures are predictions whose `failure` field is not `None`.

Implement them directly:

```python
# src/pavg_critic/benchmarking/metrics.py
from __future__ import annotations

from math import sqrt
from statistics import mean
from typing import Sequence

from .contracts import BenchmarkPrediction, BenchmarkSample


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average_rank
        cursor = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    x = _rank(left)
    y = _rank(right)
    x_mean, y_mean = mean(x), mean(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    x_scale = sqrt(sum((a - x_mean) ** 2 for a in x))
    y_scale = sqrt(sum((b - y_mean) ** 2 for b in y))
    return None if x_scale == 0 or y_scale == 0 else numerator / (x_scale * y_scale)


def compute_smoke_metrics(
    samples: Sequence[BenchmarkSample],
    predictions: Sequence[BenchmarkPrediction],
) -> dict[str, float | int | None]:
    gold = {item.sample_id: item for item in samples}
    predicted = {item.sample_id: item for item in predictions}
    if len(gold) != len(samples) or len(predicted) != len(predictions):
        raise ValueError("duplicate sample IDs are not allowed")
    if set(gold) != set(predicted):
        raise ValueError("gold and prediction sample IDs must match exactly")
    pairs = [(gold[key], predicted[key]) for key in sorted(gold)]
    classes = ("physical", "violation")
    class_f1 = []
    class_recall = []
    precision_by_class = {}
    for target in classes:
        tp = sum(g.physics_label == target and p.physics_label == target for g, p in pairs)
        fp = sum(g.physics_label != target and p.physics_label == target for g, p in pairs)
        fn = sum(g.physics_label == target and p.physics_label != target for g, p in pairs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_by_class[target] = precision
        class_recall.append(recall)
        class_f1.append(f1)
    ordinal = [(g.physics_score, p.physics_score) for g, p in pairs if g.physics_score is not None and p.physics_score is not None]
    return {
        "count": len(pairs),
        "accuracy": sum(g.physics_label == p.physics_label for g, p in pairs) / len(pairs),
        "balanced_accuracy": mean(class_recall),
        "macro_f1": mean(class_f1),
        "violation_precision": precision_by_class["violation"],
        "violation_recall": class_recall[1],
        "unknown_rate": sum(p.physics_label == "unknown" for _, p in pairs) / len(pairs),
        "failure_rate": sum(p.failure is not None for _, p in pairs) / len(pairs),
        "mean_latency_sec": mean(p.latency_sec for _, p in pairs),
        "mean_visible_frames": mean(p.visible_frame_count for _, p in pairs),
        "physics_spearman": _spearman([a for a, _ in ordinal], [b for _, b in ordinal]),
    }
```

- [ ] **Step 4: Run metric tests**

Run: `\.venv\Scripts\python.exe -m pytest tests/benchmarking/test_metrics.py -q`

Expected: PASS, including perfect, all-unknown, tied-score and duplicate-ID cases.

- [ ] **Step 5: Commit metrics**

```powershell
git add src/pavg_critic/benchmarking/metrics.py tests/benchmarking/test_metrics.py
git commit -m "feat: add benchmark smoke metrics"
```

### Task 6: Build an append-only resumable runner

**Files:**
- Create: `src/pavg_critic/benchmarking/runner.py`
- Test: `tests/benchmarking/test_runner.py`

- [ ] **Step 1: Write a failing interruption/resume test**

```python
import json

from pavg_critic.benchmarking.runner import BenchmarkRunner


class CountingMethod:
    method_id = "D0_DIRECT_VLM"

    def __init__(self, prediction_factory):
        self.calls = []
        self.prediction_factory = prediction_factory

    def evaluate(self, sample):
        self.calls.append(sample.sample_id)
        return self.prediction_factory(sample.sample_id, "physical", 4.0)


def test_runner_skips_completed_sample_method_pairs(tmp_path, sample_factory, prediction_factory):
    output = tmp_path / "predictions.jsonl"
    output.write_text(json.dumps(prediction_factory("0", "physical", 4.0).to_dict()) + "\n", encoding="utf-8")
    method = CountingMethod(prediction_factory)
    samples = (sample_factory(index=0, physical=True, generator="g"), sample_factory(index=1, physical=True, generator="g"))
    BenchmarkRunner(output).run(samples, (method,))
    assert method.calls == ["1"]
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2
```

- [ ] **Step 2: Run tests and verify failure**

Run: `\.venv\Scripts\python.exe -m pytest tests/benchmarking/test_runner.py -q`

Expected: FAIL importing `BenchmarkRunner`.

- [ ] **Step 3: Implement resume keys and durable appends**

```python
# src/pavg_critic/benchmarking/runner.py
from __future__ import annotations

import json
import os
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
        completed = set()
        for line_number, line in enumerate(self.output_path.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                raw = json.loads(line)
                completed.add((str(raw["sample_id"]), str(raw["method_id"])))
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"invalid prediction JSONL line {line_number}") from exc
        return completed

    def run(self, samples: Sequence[BenchmarkSample], methods: Sequence[BenchmarkMethod]) -> tuple[BenchmarkPrediction, ...]:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        completed = self._completed()
        new_records = []
        with self.output_path.open("a", encoding="utf-8") as handle:
            for sample in samples:
                for method in methods:
                    key = (sample.sample_id, method.method_id)
                    if key in completed:
                        continue
                    prediction = method.evaluate(sample)
                    if (prediction.sample_id, prediction.method_id) != key:
                        raise ValueError(f"method returned mismatched prediction key: {key}")
                    handle.write(json.dumps(prediction.to_dict(), ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    completed.add(key)
                    new_records.append(prediction)
        return tuple(new_records)


def load_predictions(path: str | Path) -> tuple[BenchmarkPrediction, ...]:
    source = Path(path)
    if not source.is_file():
        return ()
    result = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            result.append(BenchmarkPrediction.from_dict(json.loads(line)))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid prediction JSONL line {line_number}") from exc
    return tuple(result)
```

- [ ] **Step 4: Test malformed files and mismatched keys**

```python
def test_runner_rejects_corrupt_existing_jsonl(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text("{not-json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        BenchmarkRunner(path).run((), ())


def test_runner_rejects_method_key_mismatch(tmp_path, sample_factory, prediction_factory):
    class WrongMethod:
        method_id = "B1_RULE"
        def evaluate(self, sample):
            return prediction_factory(sample.sample_id, "physical", 4.0, method_id="D0_DIRECT_VLM")

    sample = sample_factory(index=1, physical=True, generator="g")
    with pytest.raises(ValueError, match="mismatched prediction key"):
        BenchmarkRunner(tmp_path / "predictions.jsonl").run((sample,), (WrongMethod(),))
```

Run: `\.venv\Scripts\python.exe -m pytest tests/benchmarking/test_runner.py -q`

Expected: PASS with tests proving corrupted JSONL stops the run rather than silently skipping data.

- [ ] **Step 5: Commit runner**

```powershell
git add src/pavg_critic/benchmarking/runner.py tests/benchmarking/test_runner.py
git commit -m "feat: add resumable benchmark runner"
```

### Task 7: Adapt PAVG B1 and M3 with a shared observation cache

**Files:**
- Create: `src/pavg_critic/benchmarking/pavg_methods.py`
- Test: `tests/benchmarking/test_pavg_methods.py`

- [ ] **Step 1: Write failing tests proving observations are reused**

```python
from pavg_critic.benchmarking.pavg_methods import CachedObservationProvider, PAVGMethod


def test_two_pavg_methods_share_one_observation_production(tmp_path, sample_factory, frame_state_factory):
    calls = []

    def producer(sample):
        calls.append(sample.sample_id)
        return (frame_state_factory(),)

    sample = sample_factory(index=1, physical=False, generator="g")
    provider = CachedObservationProvider(tmp_path / "observations", producer)
    PAVGMethod("B1_RULE", provider, model_id="fake-frontend").evaluate(sample)
    PAVGMethod("M3_MECHANICS", provider, model_id="fake-frontend").evaluate(sample)
    assert calls == [sample.sample_id]
```

- [ ] **Step 2: Run tests and verify failure**

Run: `\.venv\Scripts\python.exe -m pytest tests/benchmarking/test_pavg_methods.py -q`

Expected: FAIL importing `pavg_methods`.

- [ ] **Step 3: Implement atomic observation caching**

`CachedObservationProvider` must:

- use `<cache>/<sample_id>.json` after rejecting IDs containing path separators;
- serialize `FrameState.to_dict()` arrays;
- load with `FrameState.from_dict()`;
- write to a sibling `.tmp` and finish with `Path.replace()`;
- reject empty producer output instead of creating a misleading cache entry.

The producer interface is `Callable[[BenchmarkSample], tuple[FrameState, ...]]`.

```python
class CachedObservationProvider:
    def __init__(self, cache_dir: str | Path, producer):
        self.cache_dir = Path(cache_dir)
        self.producer = producer

    def _path(self, sample_id: str) -> Path:
        if not sample_id or Path(sample_id).name != sample_id or any(mark in sample_id for mark in ("/", "\\")):
            raise ValueError(f"unsafe sample ID for observation cache: {sample_id!r}")
        return self.cache_dir / f"{sample_id}.json"

    def get(self, sample: BenchmarkSample) -> tuple[FrameState, ...]:
        path = self._path(sample.sample_id)
        if path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
            states = tuple(FrameState.from_dict(item) for item in raw)
            if not states:
                raise ValueError(f"empty observation cache: {path}")
            return states
        states = tuple(self.producer(sample))
        if not states:
            raise ValueError(f"observation provider produced no states for {sample.sample_id}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps([state.to_dict() for state in states], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return states
```

- [ ] **Step 4: Implement the PAVG method adapter**

```python
class PAVGMethod:
    def __init__(self, mode: str, observations: CachedObservationProvider, *, model_id: str | None):
        if mode not in {"B1_RULE", "M1_GRAPH", "M2_CHECKLIST", "M3_MECHANICS"}:
            raise ValueError(f"Stage A PAVG mode is not supported: {mode}")
        self.method_id = mode
        self.observations = observations
        self.model_id = model_id

    def evaluate(self, sample: BenchmarkSample) -> BenchmarkPrediction:
        started = perf_counter()
        try:
            states = self.observations.get(sample)
            report = PhysicsCritic(build_ablation_config(self.method_id)).analyze(
                CriticRequest(video_path=sample.video_path, prompt=sample.prompt),
                observations=states,
            )
            categories = tuple(sorted({item.category for item in report.violations}))
            evidence = tuple(sorted({frame for item in report.violations for frame in item.critical_frames}))
            repairs = [item.repair_instruction for item in report.violations if item.repair_instruction]
            return BenchmarkPrediction(
                sample_id=sample.sample_id,
                method_id=self.method_id,
                model_id=self.model_id,
                semantic_score=None,
                physics_score=report.physics_score * 4 + 1,
                semantic_label="unknown",
                physics_label=report.decision,
                confidence=report.confidence,
                coverage=report.coverage,
                latency_sec=perf_counter() - started,
                visible_frame_count=len({state.frame for state in states}),
                violation_categories=categories,
                evidence_frames=evidence,
                repair_instruction="; ".join(repairs) or None,
            )
        except Exception as exc:
            return BenchmarkPrediction(
                sample_id=sample.sample_id,
                method_id=self.method_id,
                model_id=self.model_id,
                semantic_score=None,
                physics_score=None,
                semantic_label="unknown",
                physics_label="unknown",
                confidence=0.0,
                coverage=0.0,
                latency_sec=perf_counter() - started,
                visible_frame_count=0,
                failure={"type": type(exc).__name__, "message": str(exc)[:500]},
            )
```

Map `report.decision` only after validating it is one of `physical`, `violation`, `unknown`.

- [ ] **Step 5: Add an explicit observation-preparation command**

The Stage A CLI accepts `--observations-dir`. It does not invent trajectories. If a sample has no cached observations and no `--observation-provider vlm` is supplied, it writes an `ObservationUnavailable` failure record. Implement the optional producer as:

```python
def make_vlm_observation_producer(model, *, num_keyframes: int = 16):
    from pavg_critic.config import CriticConfig
    from pavg_critic.pipeline import PhysicsCritic
    from pavg_critic.schemas import CriticRequest
    from pavg_critic.vlm_detector import VLMObjectDetector

    def produce(sample: BenchmarkSample) -> tuple[FrameState, ...]:
        detector = VLMObjectDetector(
            model,
            sample.video_path,
            num_keyframes=num_keyframes,
        )
        artifacts = PhysicsCritic(CriticConfig(), detector=detector).analyze_detailed(
            CriticRequest(video_path=sample.video_path, prompt=sample.prompt)
        )
        states = tuple(
            state
            for track in artifacts.tracks
            for state in track.states
        )
        if not states:
            raise ValueError(f"observation provider produced no states for {sample.sample_id}")
        return states

    return produce
```

The same `CachedObservationProvider` instance wraps this producer for all requested PAVG ablations, so the VLM detector is called once per sample rather than once per method.

- [ ] **Step 6: Run PAVG adapter tests**

Run: `\.venv\Scripts\python.exe -m pytest tests/benchmarking/test_pavg_methods.py tests/test_evaluation.py -q`

Expected: PASS; existing six-sample frozen evaluation remains unchanged.

- [ ] **Step 7: Commit PAVG adapters**

```powershell
git add src/pavg_critic/benchmarking/pavg_methods.py tests/benchmarking/test_pavg_methods.py
git commit -m "feat: adapt PAVG ablations for video benchmarks"
```

### Task 8: Generate JSON and Markdown smoke reports

**Files:**
- Create: `src/pavg_critic/benchmarking/report.py`
- Test: `tests/benchmarking/test_report.py`

- [ ] **Step 1: Write a failing deterministic report test**

```python
from pavg_critic.benchmarking.report import build_smoke_report


def test_report_separates_methods_and_contains_claims_warning(sample_factory, prediction_factory):
    samples = (sample_factory(index=1, physical=False, generator="g"),)
    predictions = (
        prediction_factory("1", "violation", 2.0, method_id="D0_DIRECT_VLM"),
        prediction_factory("1", "violation", 2.0, method_id="B1_RULE"),
    )
    report = build_smoke_report(samples, predictions)
    assert set(report["methods"]) == {"D0_DIRECT_VLM", "B1_RULE"}
    assert report["claims_allowed"] is False
    assert "smoke" in report["warning"].lower()
```

- [ ] **Step 2: Run tests and verify failure**

Run: `\.venv\Scripts\python.exe -m pytest tests/benchmarking/test_report.py -q`

Expected: FAIL importing `report`.

- [ ] **Step 3: Implement report aggregation and rendering**

`build_smoke_report()` groups predictions by method, calls `compute_smoke_metrics`, records missing sample/method pairs, and returns:

```python
{
    "schema_version": "1.0",
    "stage": "A_SMOKE",
    "claims_allowed": False,
    "warning": "Stage A smoke results validate the pipeline only and are not benchmark performance claims.",
    "sample_count": 20,
    "methods": {
        "D0_DIRECT_VLM": {
            "metrics": {
                "count": 20,
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "macro_f1": 0.0,
                "violation_precision": 0.0,
                "violation_recall": 0.0,
                "unknown_rate": 0.0,
                "failure_rate": 0.0,
                "mean_latency_sec": 0.0,
                "mean_visible_frames": 16.0,
                "physics_spearman": None,
            },
            "missing_sample_ids": [],
            "failures": [],
        }
    },
}
```

`write_smoke_report()` writes `summary.json` and `summary.md`. Markdown contains the warning before any metric table, method-level latency/failure columns, and a list of failed sample IDs.

```python
# src/pavg_critic/benchmarking/report.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .contracts import BenchmarkPrediction, BenchmarkSample
from .metrics import compute_smoke_metrics

WARNING = "Stage A smoke results validate the pipeline only and are not benchmark performance claims."


def build_smoke_report(samples: Sequence[BenchmarkSample], predictions: Sequence[BenchmarkPrediction]) -> dict:
    by_method: dict[str, list[BenchmarkPrediction]] = {}
    for prediction in predictions:
        by_method.setdefault(prediction.method_id, []).append(prediction)
    sample_by_id = {item.sample_id: item for item in samples}
    methods = {}
    for method_id in sorted(by_method):
        records = by_method[method_id]
        predicted_ids = {item.sample_id for item in records}
        extra = sorted(predicted_ids - set(sample_by_id))
        if extra:
            raise ValueError(f"predictions contain unknown sample IDs for {method_id}: {extra}")
        matching_samples = tuple(sample_by_id[item_id] for item_id in sorted(predicted_ids) if item_id in sample_by_id)
        matching_predictions = tuple(sorted((item for item in records if item.sample_id in sample_by_id), key=lambda item: item.sample_id))
        missing = sorted(set(sample_by_id) - predicted_ids)
        metrics = compute_smoke_metrics(matching_samples, matching_predictions) if matching_samples else None
        methods[method_id] = {
            "metrics": metrics,
            "missing_sample_ids": missing,
            "failures": [
                {"sample_id": item.sample_id, **dict(item.failure)}
                for item in matching_predictions
                if item.failure is not None
            ],
        }
    return {
        "schema_version": "1.0",
        "stage": "A_SMOKE",
        "claims_allowed": False,
        "warning": WARNING,
        "sample_count": len(samples),
        "methods": methods,
    }


def write_smoke_report(samples, predictions, output_dir: str | Path) -> dict:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report = build_smoke_report(samples, predictions)
    (destination / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = ["# VideoPhy-2 Stage A Smoke", "", f"> **Warning:** {WARNING}", "", "| Method | Count | Macro-F1 | Unknown | Failures | Mean latency (s) |", "|---|---:|---:|---:|---:|---:|"]
    for method_id, item in report["methods"].items():
        metrics = item["metrics"]
        if metrics is None:
            lines.append(f"| {method_id} | 0 | N/A | N/A | 0 | N/A |")
        else:
            lines.append(
                f"| {method_id} | {metrics['count']} | {metrics['macro_f1']:.3f} | "
                f"{metrics['unknown_rate']:.3f} | {len(item['failures'])} | {metrics['mean_latency_sec']:.3f} |"
            )
    lines.extend(["", "## Failures", ""])
    failures = [(method_id, failure) for method_id, item in report["methods"].items() for failure in item["failures"]]
    lines.extend(
        [f"- `{method_id}` / `{failure['sample_id']}`: {failure['type']} — {failure['message']}" for method_id, failure in failures]
        or ["- None"]
    )
    (destination / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
```

- [ ] **Step 4: Run report tests**

Run: `\.venv\Scripts\python.exe -m pytest tests/benchmarking/test_report.py -q`

Expected: PASS with byte-stable Markdown output for the same input ordering.

- [ ] **Step 5: Commit reporting**

```powershell
git add src/pavg_critic/benchmarking/report.py tests/benchmarking/test_report.py
git commit -m "feat: render benchmark smoke reports"
```

### Task 9: Add the end-to-end benchmark CLI

**Files:**
- Create: `benchmarks/evaluate_video_benchmark.py`
- Test: `tests/benchmarking/test_cli.py`

- [ ] **Step 1: Write failing parser/config tests**

Test these exact behaviors:

- `--methods D0_DIRECT_VLM,D1_STRUCTURED_VLM,B1_RULE,M3_MECHANICS` parses in declared order;
- `--provider responses` requires `BENCH_API_KEY` and `BENCH_MODEL` but never includes the key in an error or config snapshot;
- `--provider chat` additionally accepts `BENCH_BASE_URL`, supporting a local vLLM OpenAI-compatible endpoint;
- unknown methods fail before any video/model call;
- `--max-samples` defaults to all manifest samples, while Stage A command explicitly passes `20`.

```python
import pytest

from benchmarks.evaluate_video_benchmark import build_benchmark_model, build_parser, parse_methods


def test_method_parser_preserves_declared_order():
    assert parse_methods("D0_DIRECT_VLM,D1_STRUCTURED_VLM,B1_RULE,M3_MECHANICS") == (
        "D0_DIRECT_VLM", "D1_STRUCTURED_VLM", "B1_RULE", "M3_MECHANICS"
    )


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown benchmark method"):
        parse_methods("D0_DIRECT_VLM,NOT_A_METHOD")


def test_missing_model_credentials_do_not_echo_secrets(monkeypatch):
    monkeypatch.delenv("BENCH_API_KEY", raising=False)
    monkeypatch.setenv("BENCH_MODEL", "gpt-test")
    with pytest.raises(ValueError) as error:
        build_benchmark_model("responses")
    assert "BENCH_API_KEY" in str(error.value)
    assert "gpt-test" not in str(error.value)


def test_max_samples_defaults_to_none():
    args = build_parser().parse_args(["--manifest", "m.json", "--run-dir", "run", "--methods", "B1_RULE"])
    assert args.max_samples is None
```

- [ ] **Step 2: Run tests and verify the CLI is absent**

Run: `\.venv\Scripts\python.exe -m pytest tests/benchmarking/test_cli.py -q`

Expected: FAIL importing `benchmarks.evaluate_video_benchmark`.

- [ ] **Step 3: Implement model construction without hard-coded secrets**

```python
def build_benchmark_model(provider: str):
    api_key = os.environ.get("BENCH_API_KEY", "")
    model = os.environ.get("BENCH_MODEL", "")
    if not api_key or not model:
        raise ValueError("Set BENCH_API_KEY and BENCH_MODEL before running model baselines")
    if provider == "responses":
        return OpenAIResponsesModel(
            api_key=api_key,
            model=model,
            base_url=os.environ.get("BENCH_BASE_URL", "https://api.openai.com/v1"),
        )
    if provider == "chat":
        return OpenAIChatModel(
            api_key=api_key,
            model=model,
            base_url=os.environ.get("BENCH_BASE_URL", "http://127.0.0.1:8000/v1"),
        )
    raise ValueError(f"unsupported provider: {provider}")
```

The CLI writes `resolved_config.json` containing provider, model, frame count, method list, manifest SHA-256 and code revision; it replaces any field whose name contains `key`, `token` or `secret` with `"REDACTED"`.

- [ ] **Step 4: Implement orchestration**

The CLI must:

1. load the immutable sample manifest;
2. cap to `--max-samples` only after stable sample-ID sorting;
3. build D0/D1 only when requested;
4. build B1/M3 with the same `CachedObservationProvider`;
5. append predictions to `<run-dir>/predictions.jsonl`;
6. rebuild reports from the complete JSONL, including records from previous invocations;
7. exit `2` when any requested method has zero successful predictions, otherwise `0` even when individual failures are recorded.

```python
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from pavg_critic.api_models import OpenAIChatModel, OpenAIResponsesModel
from pavg_critic.benchmarking.baselines import DirectVLMJudge
from pavg_critic.benchmarking.datasets import load_manifest
from pavg_critic.benchmarking.pavg_methods import CachedObservationProvider, PAVGMethod, make_vlm_observation_producer
from pavg_critic.benchmarking.report import write_smoke_report
from pavg_critic.benchmarking.runner import BenchmarkRunner, load_predictions

ALLOWED_METHODS = (
    "D0_DIRECT_VLM", "D1_STRUCTURED_VLM",
    "B1_RULE", "M1_GRAPH", "M2_CHECKLIST", "M3_MECHANICS",
)


def parse_methods(raw: str) -> tuple[str, ...]:
    methods = tuple(item.strip() for item in raw.split(",") if item.strip())
    unknown = [item for item in methods if item not in ALLOWED_METHODS]
    if not methods or unknown:
        raise ValueError(f"unknown benchmark method(s): {unknown}")
    return methods


def build_parser():
    parser = argparse.ArgumentParser(description="Run the PAVG Stage A video benchmark smoke")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--provider", choices=("responses", "chat"), default="responses")
    parser.add_argument("--frame-count", type=int, default=16)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--observations-dir", type=Path)
    parser.add_argument("--observation-provider", choices=("none", "vlm"), default="none")
    return parser


def _unavailable_observations(sample):
    raise ValueError(f"ObservationUnavailable: {sample.sample_id}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    method_ids = parse_methods(args.methods)
    samples = tuple(sorted(load_manifest(args.manifest), key=lambda item: item.sample_id))
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        samples = samples[:args.max_samples]
    requires_model = any(item.startswith("D") for item in method_ids) or args.observation_provider == "vlm"
    model = build_benchmark_model(args.provider) if requires_model else None
    model_id = os.environ.get("BENCH_MODEL") if model is not None else None
    methods = []
    if "D0_DIRECT_VLM" in method_ids:
        methods.append(DirectVLMJudge(model, model_id=model_id, structured=False, frame_count=args.frame_count))
    if "D1_STRUCTURED_VLM" in method_ids:
        methods.append(DirectVLMJudge(model, model_id=model_id, structured=True, frame_count=args.frame_count))
    pavg_ids = [item for item in method_ids if item in {"B1_RULE", "M1_GRAPH", "M2_CHECKLIST", "M3_MECHANICS"}]
    if pavg_ids:
        if args.observations_dir is None:
            raise ValueError("--observations-dir is required for PAVG methods")
        producer = make_vlm_observation_producer(model, num_keyframes=args.frame_count) if args.observation_provider == "vlm" else _unavailable_observations
        observations = CachedObservationProvider(args.observations_dir, producer)
        methods.extend(PAVGMethod(item, observations, model_id=model_id) for item in pavg_ids)
    by_id = {method.method_id: method for method in methods}
    ordered_methods = tuple(by_id[item] for item in method_ids)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "provider": args.provider,
        "model": model_id,
        "frame_count": args.frame_count,
        "methods": list(method_ids),
        "manifest_sha256": _sha256(args.manifest),
        "git_revision": _git_revision(),
    }
    (args.run_dir / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    prediction_path = args.run_dir / "predictions.jsonl"
    BenchmarkRunner(prediction_path).run(samples, ordered_methods)
    predictions = load_predictions(prediction_path)
    report = write_smoke_report(samples, predictions, args.run_dir)
    for method_id in method_ids:
        records = [item for item in predictions if item.method_id == method_id]
        if not records or all(item.failure is not None for item in records):
            return 2
    return 0
```

Import the concrete classes/functions shown above, and finish the file with `raise SystemExit(main())` under the standard `if __name__ == "__main__"` guard.

- [ ] **Step 5: Run CLI tests with scripted methods**

Run: `\.venv\Scripts\python.exe -m pytest tests/benchmarking/test_cli.py -q`

Expected: PASS without network or real API calls.

- [ ] **Step 6: Commit CLI**

```powershell
git add benchmarks/evaluate_video_benchmark.py tests/benchmarking/test_cli.py
git commit -m "feat: add video benchmark smoke CLI"
```

### Task 10: Prepare and run the 20-video VideoPhy-2 smoke

**Files:**
- Create external: `evaluation/external/videophy2/` (git-ignored)
- Create: `evaluation/manifests/videophy2_test.json`
- Create: `evaluation/manifests/videophy2_smoke20.json`
- Create output: `outputs/benchmarks/videophy2-smoke-<run-id>/`
- Modify: `README.md`

- [ ] **Step 1: Download only the public CSV snapshot**

Run:

```powershell
New-Item -ItemType Directory -Force evaluation/external/videophy2 | Out-Null
curl.exe -L --fail --output evaluation/external/videophy2/videophy2_test.csv https://huggingface.co/datasets/videophysics/videophy2_test/resolve/main/videophy2_test.csv
```

Expected: CSV exists and `python benchmarks/prepare_videophy_manifest.py inspect --csv evaluation/external/videophy2/videophy2_test.csv` prints a non-zero row count and the source columns.

- [ ] **Step 2: Materialize local videos and normalize paths**

Run the preparation CLI's download command:

```powershell
\.venv\Scripts\python.exe benchmarks/prepare_videophy_manifest.py download `
  --csv evaluation/external/videophy2/videophy2_test.csv `
  --video-dir evaluation/external/videophy2/videos `
  --output-csv evaluation/external/videophy2/videophy2_test_local.csv
```

Expected: every output row has an existing local video path; HTTP failures are written to `evaluation/external/videophy2/download_failures.jsonl` and cause a non-zero exit. Re-running skips existing non-empty local files, and the normalize command recomputes every video's SHA-256 into the immutable manifest.

- [ ] **Step 3: Build full and smoke manifests**

```powershell
\.venv\Scripts\python.exe benchmarks/prepare_videophy_manifest.py normalize `
  --csv evaluation/external/videophy2/videophy2_test_local.csv `
  --benchmark videophy2 --split test `
  --output evaluation/manifests/videophy2_test.json

\.venv\Scripts\python.exe benchmarks/prepare_videophy_manifest.py smoke `
  --manifest evaluation/manifests/videophy2_test.json `
  --count 20 --seed 20260715 `
  --output evaluation/manifests/videophy2_smoke20.json
```

Expected: smoke manifest contains exactly 20 unique sample IDs, both physical labels and at least two generators. If the accessible test snapshot cannot satisfy those constraints, the command fails with counts by label/generator instead of silently relaxing selection.

- [ ] **Step 4: Run deterministic PAVG smoke first**

Prepare observations with the configured shared VLM frontend, then run B1/M3:

```powershell
$env:BENCH_API_KEY = $env:OPENAI_API_KEY
$env:BENCH_BASE_URL = "https://api.openai.com/v1"
$env:BENCH_MODEL = "gpt-5.6-terra"
if (-not $env:BENCH_API_KEY) { throw "Set OPENAI_API_KEY in this terminal before the closed-model smoke run." }

\.venv\Scripts\python.exe benchmarks/evaluate_video_benchmark.py `
  --manifest evaluation/manifests/videophy2_smoke20.json `
  --run-dir outputs/benchmarks/videophy2-smoke-gpt `
  --methods B1_RULE,M3_MECHANICS `
  --provider responses `
  --observation-provider vlm `
  --observations-dir evaluation/external/videophy2/observations/gpt-5.6-terra `
  --max-samples 20
```

Expected: 40 sample/method rows or explicit failure rows, with one observation cache per successfully prepared sample.

- [ ] **Step 5: Run one matched VLM block**

For an official GPT block, set an official OpenAI key and fixed GPT model, then use `--provider responses`. For a Qwen block served through vLLM, set its OpenAI-compatible URL and use `--provider chat`. Run:

```powershell
\.venv\Scripts\python.exe benchmarks/evaluate_video_benchmark.py `
  --manifest evaluation/manifests/videophy2_smoke20.json `
  --run-dir outputs/benchmarks/videophy2-smoke-gpt `
  --methods D0_DIRECT_VLM,D1_STRUCTURED_VLM `
  --provider responses `
  --frame-count 16 `
  --max-samples 20
```

Expected: 40 sample/method rows or explicit failure rows; D0 and D1 records for each sample use the same 16 frame indices.

- [ ] **Step 6: Verify smoke artifacts**

Run:

```powershell
\.venv\Scripts\python.exe -m json.tool outputs/benchmarks/videophy2-smoke-gpt/summary.json > $null
Get-Content outputs/benchmarks/videophy2-smoke-gpt/summary.md -TotalCount 12
```

Expected: JSON validates; Markdown starts with the Stage A warning; prediction row counts, failures, latency and frame counts are visible. Do not interpret method ranking from this sample.

- [ ] **Step 7: Document exact commands and claims boundary**

Add a `Benchmark Stage A` README section containing:

- environment variables without real keys;
- manifest preparation and resume commands;
- output directory structure;
- the statement that Stage A cannot support performance claims;
- the gate for proceeding to the 200-300 sample pilot.

Add this warning verbatim before the commands:

```markdown
> Stage A is a 20-video pipeline smoke test. Its metric values must not be cited as evidence that one evaluator outperforms another. Proceed to the stratified pilot only after every requested sample/method pair has either a prediction or an explicit failure record and reruns are duplicate-free.
```

- [ ] **Step 8: Run the complete local verification suite**

Run:

```powershell
\.venv\Scripts\python.exe -m pytest -q
\.venv\Scripts\python.exe -m compileall -q src tests benchmarks
\.venv\Scripts\python.exe -m pip check
git diff --check
```

Expected: all tests pass, compileall and pip check exit 0, and no whitespace errors are reported.

- [ ] **Step 9: Commit Stage A docs and manifests**

Do not commit downloaded videos, API responses containing secrets, observation caches or output runs.

```powershell
git add README.md evaluation/manifests/videophy2_test.json evaluation/manifests/videophy2_smoke20.json
git commit -m "docs: add VideoPhy-2 smoke evaluation workflow"
```

## Completion checkpoint

Stage A is complete only when:

- the 20-sample manifest is immutable and checksummed;
- D0/D1 and B1/M3 each have either a prediction or explicit failure for all 20 samples;
- rerunning produces zero duplicate JSONL keys;
- D0/D1 visual frame indices are identical per sample;
- B1/M3 reuse one observation cache per sample;
- summary files clearly prohibit benchmark claims;
- the full test suite passes.

After this checkpoint, write a separate Stage B pilot plan covering the full matched GPT/Qwen blocks, VideoPhy AutoEval ingestion, prompt-cluster bootstrap confidence intervals, GPU rental execution and cost-based scaling.
