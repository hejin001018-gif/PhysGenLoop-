"""VideoPhy normalization and immutable manifest utilities."""

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
    "video_path": (
        "local_path",
        "video_path",
        "videopath",
        "path",
        "video_url",
        "video",
        "url",
    ),
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


def load_videophy_csv(
    path: str | Path,
    *,
    benchmark: str,
    split: str,
) -> tuple[BenchmarkSample, ...]:
    csv_path = Path(path)
    base = csv_path.resolve().parent
    result: list[BenchmarkSample] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            semantic = _score(row, "semantic_score")
            physics = _score(row, "physics_score")
            raw_path = _value(row, "video_path")
            video_path = Path(raw_path)
            if not video_path.is_absolute():
                video_path = base / video_path
            video_path = video_path.resolve()
            result.append(
                BenchmarkSample(
                    sample_id=_value(row, "sample_id"),
                    benchmark=benchmark,
                    split=split,
                    prompt=_value(row, "prompt"),
                    video_path=str(video_path),
                    prompt_group_id=_value(row, "prompt_group_id"),
                    generator=_value(row, "generator"),
                    semantic_label=(
                        "unknown"
                        if semantic is None
                        else ("adherent" if semantic >= 4 else "not_adherent")
                    ),
                    physics_label=(
                        "unknown"
                        if physics is None
                        else ("physical" if physics >= 4 else "violation")
                    ),
                    semantic_score=semantic,
                    physics_score=physics,
                    raw_labels={
                        key: value
                        for key, value in row.items()
                        if key
                        in {
                            "sa",
                            "pc",
                            "semantic_adherence",
                            "physical_commonsense",
                        }
                    },
                    sha256=_file_sha256(video_path),
                )
            )
    if not result:
        raise ValueError(f"no rows found in {path}")
    return tuple(result)


def select_smoke_samples(
    samples: Sequence[BenchmarkSample],
    *,
    count: int,
    seed: int,
) -> tuple[BenchmarkSample, ...]:
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
    payload = {
        "schema_version": "1.0",
        "samples": [item.to_dict() for item in samples],
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_manifest(path: str | Path) -> tuple[BenchmarkSample, ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != "1.0" or not isinstance(raw.get("samples"), list):
        raise ValueError("benchmark manifest must use schema 1.0 with a samples array")
    samples = tuple(BenchmarkSample.from_dict(item) for item in raw["samples"])
    ids = [item.sample_id for item in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("benchmark manifest contains duplicate sample IDs")
    return samples


def materialize_video_csv(
    input_csv: str | Path,
    *,
    video_dir: str | Path,
    output_csv: str | Path,
    timeout_sec: float = 60.0,
) -> tuple[dict[str, str | int], ...]:
    """Download or copy source videos and add a ``local_path`` column."""

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
    failures: list[dict[str, str | int]] = []
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
                request = urllib.request.Request(
                    source,
                    headers={"User-Agent": "PAVG-benchmark/1.0"},
                )
                with (
                    urllib.request.urlopen(request, timeout=timeout_sec) as response,
                    temporary.open("wb") as output,
                ):
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
            failures.append(
                {
                    "row": row_index,
                    "sample_id": sample_id,
                    "source": source,
                    "error": type(exc).__name__,
                    "message": str(exc)[:500],
                }
            )
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return tuple(failures)
