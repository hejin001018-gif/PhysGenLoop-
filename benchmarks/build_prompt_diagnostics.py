"""Freeze a deterministic cross-action shuffled-prompt diagnostic manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Sequence


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _load_manifest(path: Path) -> tuple[dict[str, Any], ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != "1.0":
        raise ValueError("manifest must use schema_version 1.0")
    samples = raw.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("manifest must contain a non-empty samples array")
    normalized: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, item in enumerate(samples):
        if not isinstance(item, dict):
            raise ValueError(f"manifest sample {index} must be an object")
        sample = dict(item)
        sample_id = str(sample.get("sample_id", ""))
        prompt = str(sample.get("prompt", ""))
        action = str(sample.get("prompt_group_id", ""))
        if not sample_id or not prompt or not action:
            raise ValueError(
                f"manifest sample {index} requires sample_id, prompt and prompt_group_id"
            )
        if sample_id in ids:
            raise ValueError(f"duplicate sample ID: {sample_id}")
        ids.add(sample_id)
        normalized.append(sample)
    return tuple(normalized)


def _allowed(recipient: dict[str, Any], donor: dict[str, Any]) -> bool:
    return (
        recipient["sample_id"] != donor["sample_id"]
        and recipient["prompt"] != donor["prompt"]
        and recipient["prompt_group_id"] != donor["prompt_group_id"]
    )


def _derange(
    samples: tuple[dict[str, Any], ...],
    *,
    seed: int,
) -> dict[str, str]:
    recipients = sorted(samples, key=lambda item: str(item["sample_id"]))
    recipient_by_id = {str(item["sample_id"]): item for item in recipients}
    donors = list(recipients)
    random.Random(seed).shuffle(donors)
    recipient_for_donor: dict[str, str] = {}

    def assign(recipient_id: str, seen_donors: set[str]) -> bool:
        recipient = recipient_by_id[recipient_id]
        for donor in donors:
            donor_id = str(donor["sample_id"])
            if donor_id in seen_donors or not _allowed(recipient, donor):
                continue
            seen_donors.add(donor_id)
            previous = recipient_for_donor.get(donor_id)
            if previous is None or assign(previous, seen_donors):
                recipient_for_donor[donor_id] = recipient_id
                return True
        return False

    for recipient in recipients:
        recipient_id = str(recipient["sample_id"])
        if not assign(recipient_id, set()):
            raise ValueError(
                "no complete cross-action prompt derangement exists for the manifest"
            )
    donor_for_recipient = {
        recipient_id: donor_id
        for donor_id, recipient_id in recipient_for_donor.items()
    }
    if len(donor_for_recipient) != len(samples):
        raise ValueError("prompt derangement is incomplete")
    return donor_for_recipient


def _write_immutable(path: Path, content: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise ValueError(f"refusing to replace different existing file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            raw_temporary = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(raw_temporary, path)
        raw_temporary = None
    finally:
        if raw_temporary is not None:
            Path(raw_temporary).unlink(missing_ok=True)


def build_prompt_diagnostics(
    manifest: str | Path,
    output_manifest: str | Path,
    donor_map: str | Path,
    *,
    seed: int = 20260717,
) -> None:
    samples = _load_manifest(Path(manifest))
    donor_for_recipient = _derange(samples, seed=seed)
    by_id = {str(item["sample_id"]): item for item in samples}
    shuffled_samples = []
    mappings = []
    for recipient in samples:
        recipient_id = str(recipient["sample_id"])
        donor = by_id[donor_for_recipient[recipient_id]]
        shuffled = dict(recipient)
        shuffled["prompt"] = donor["prompt"]
        shuffled_samples.append(shuffled)
        mappings.append(
            {
                "recipient_sample_id": recipient_id,
                "donor_sample_id": str(donor["sample_id"]),
                "recipient_action": str(recipient["prompt_group_id"]),
                "donor_action": str(donor["prompt_group_id"]),
                "recipient_prompt_sha256": _sha_text(str(recipient["prompt"])),
                "donor_prompt_sha256": _sha_text(str(donor["prompt"])),
            }
        )
    mappings.sort(key=lambda item: item["recipient_sample_id"])
    _write_immutable(
        Path(output_manifest),
        _json_bytes({"schema_version": "1.0", "samples": shuffled_samples}),
    )
    _write_immutable(
        Path(donor_map),
        _json_bytes(
            {
                "schema_version": "1.0",
                "seed": seed,
                "mapping_count": len(mappings),
                "mappings": mappings,
            }
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze a deterministic cross-action prompt diagnostic"
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--donor-map", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build_prompt_diagnostics(
        args.manifest,
        args.output_manifest,
        args.donor_map,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
