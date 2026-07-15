"""VideoPhy normalization and immutable manifest utilities."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence, TypeVar

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
    "generator": ("generator", "model", "video_model", "model_name"),
    "prompt_group_id": ("prompt_group_id", "action", "action_id", "prompt_id"),
}

T = TypeVar("T")


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


def _sample_id(
    row: Mapping[str, str],
    *,
    benchmark: str,
) -> str:
    for column in ALIASES["sample_id"]:
        value = row.get(column)
        if value is not None and str(value).strip():
            return str(value).strip()
    identity = "\0".join(
        (
            str(row.get("video_url") or row.get("url") or "").strip(),
            str(row.get("caption") or row.get("prompt") or "").strip(),
            str(row.get("model_name") or row.get("generator") or "").strip(),
        )
    )
    if not identity.replace("\0", ""):
        raise ValueError(
            f"missing sample_id and derivation fields; available columns: {sorted(row)}"
        )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return f"{benchmark}-{digest}"


def _source_url(row: Mapping[str, str]) -> str | None:
    for column in ("video_url", "url"):
        value = str(row.get(column) or "").strip()
        if value.startswith(("http://", "https://")):
            return value
    return None


def _physical_rules(row: Mapping[str, str]) -> tuple[str, ...]:
    result: list[str] = []
    for column in (
        "physics_rules_followed",
        "physics_rules_unfollowed",
        "physics_rules_cannot_be_determined",
        "human_violated_rules",
    ):
        raw = str(row.get(column) or "").strip()
        if not raw:
            continue
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"invalid {column} list: {raw[:100]}") from exc
        values = parsed if isinstance(parsed, (list, tuple)) else (parsed,)
        for value in values:
            text = str(value).strip()
            if text and text not in result:
                result.append(text)
    return tuple(result)


def _take_balanced(
    buckets: dict[tuple[str, str], list[T]],
    *,
    count: int,
) -> list[T]:
    """Balance labels first, then rotate generators within each label."""

    labels = sorted({label for label, _ in buckets})
    generators = {
        label: sorted(generator for candidate, generator in buckets if candidate == label)
        for label in labels
    }
    cursors = {label: 0 for label in labels}
    chosen: list[T] = []
    while len(chosen) < count and any(buckets.values()):
        for label in labels:
            order = generators[label]
            for offset in range(len(order)):
                index = (cursors[label] + offset) % len(order)
                key = (label, order[index])
                if buckets[key]:
                    chosen.append(buckets[key].pop())
                    cursors[label] = (index + 1) % len(order)
                    break
            if len(chosen) >= count:
                break
    return chosen


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
                    sample_id=_sample_id(row, benchmark=benchmark),
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
                    physical_rules=_physical_rules(row),
                    raw_labels={
                        key: value
                        for key, value in row.items()
                        if key
                        in {
                            "sa",
                            "pc",
                            "joint",
                            "is_hard",
                            "semantic_adherence",
                            "physical_commonsense",
                            "physics_rules_followed",
                            "physics_rules_unfollowed",
                            "physics_rules_cannot_be_determined",
                            "human_violated_rules",
                            "metadata_rules",
                        }
                    },
                    source_url=_source_url(row),
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
    chosen = _take_balanced(buckets, count=count)
    return tuple(sorted(chosen, key=lambda item: item.sample_id))


def split_diagnostic_samples(
    samples: Sequence[BenchmarkSample],
    *,
    dev_count: int,
    seed: int,
) -> tuple[tuple[BenchmarkSample, ...], tuple[BenchmarkSample, ...]]:
    """Create an exact, group-preserving split with label/generator balance."""

    if dev_count <= 0 or dev_count >= len(samples):
        raise ValueError("dev_count must leave at least one sample in each split")
    ids = [item.sample_id for item in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sample IDs are not allowed")
    grouped: dict[str, list[BenchmarkSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.prompt_group_id, []).append(sample)
    if len(grouped) > 30:
        raise ValueError("diagnostic exact splitting supports at most 30 prompt groups")

    group_ids = sorted(grouped)
    random.Random(seed).shuffle(group_ids)
    group_sizes = [len(grouped[group_id]) for group_id in group_ids]
    suffix_sizes = [0] * (len(group_ids) + 1)
    for index in range(len(group_ids) - 1, -1, -1):
        suffix_sizes[index] = suffix_sizes[index + 1] + group_sizes[index]

    label_totals = Counter(item.physics_label for item in samples)
    generator_totals = Counter(item.generator for item in samples)
    fraction = dev_count / len(samples)
    target_labels = {
        key: value * fraction for key, value in label_totals.items()
    }
    target_generators = {
        key: value * fraction for key, value in generator_totals.items()
    }
    best: tuple[float, tuple[str, ...], tuple[str, ...]] | None = None

    def consider(selected_group_ids: tuple[str, ...]) -> None:
        nonlocal best
        selected = [
            item
            for group_id in selected_group_ids
            for item in grouped[group_id]
        ]
        label_counts = Counter(item.physics_label for item in selected)
        generator_counts = Counter(item.generator for item in selected)
        score = sum(
            (label_counts[key] - target) ** 2 / max(target, 1.0)
            for key, target in target_labels.items()
        )
        score += sum(
            (generator_counts[key] - target) ** 2 / max(target, 1.0)
            for key, target in target_generators.items()
        )
        sample_ids = tuple(sorted(item.sample_id for item in selected))
        candidate = (score, sample_ids, tuple(sorted(selected_group_ids)))
        if best is None or candidate < best:
            best = candidate

    def search(
        index: int,
        selected_count: int,
        selected_groups: tuple[str, ...],
    ) -> None:
        if selected_count > dev_count:
            return
        if selected_count + suffix_sizes[index] < dev_count:
            return
        if index == len(group_ids):
            if selected_count == dev_count:
                consider(selected_groups)
            return
        group_id = group_ids[index]
        search(
            index + 1,
            selected_count + group_sizes[index],
            selected_groups + (group_id,),
        )
        search(index + 1, selected_count, selected_groups)

    search(0, 0, ())
    if best is None:
        raise ValueError(
            "cannot form an exact group-preserving diagnostic split for dev_count"
        )
    dev_ids = set(best[1])
    dev = tuple(sorted((item for item in samples if item.sample_id in dev_ids), key=lambda item: item.sample_id))
    evaluation = tuple(sorted((item for item in samples if item.sample_id not in dev_ids), key=lambda item: item.sample_id))
    return dev, evaluation


def write_source_smoke_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    count: int,
    seed: int,
) -> None:
    """Select a balanced source subset before downloading any videos."""

    with Path(input_csv).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or ())
    if count <= 0 or count > len(rows):
        raise ValueError("count must be between 1 and the number of source rows")
    buckets: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        physics = _score(row, "physics_score")
        label = (
            "unknown"
            if physics is None
            else ("physical" if physics >= 4 else "violation")
        )
        generator = _value(row, "generator")
        buckets.setdefault((label, generator), []).append(row)
    rng = random.Random(seed)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: _sample_id(row, benchmark="videophy2"))
        rng.shuffle(bucket)
    chosen = _take_balanced(buckets, count=count)
    chosen.sort(key=lambda row: _sample_id(row, benchmark="videophy2"))
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(chosen)


def write_manifest(samples: Iterable[BenchmarkSample], path: str | Path) -> None:
    destination = Path(path)
    serialized = []
    for item in samples:
        data = item.to_dict()
        try:
            data["video_path"] = os.path.relpath(
                item.video_path,
                start=destination.resolve().parent,
            ).replace(os.sep, "/")
        except ValueError:
            data["video_path"] = item.video_path
        serialized.append(data)
    payload = {
        "schema_version": "1.0",
        "samples": serialized,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_manifest(path: str | Path) -> tuple[BenchmarkSample, ...]:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "1.0" or not isinstance(raw.get("samples"), list):
        raise ValueError("benchmark manifest must use schema 1.0 with a samples array")
    normalized = []
    for item in raw["samples"]:
        data = dict(item)
        video_path = Path(str(data.get("video_path", "")))
        if not video_path.is_absolute():
            data["video_path"] = str((source.resolve().parent / video_path).resolve())
        normalized.append(data)
    samples = tuple(BenchmarkSample.from_dict(item) for item in normalized)
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
        sample_id = _sample_id(row, benchmark="videophy2")
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
