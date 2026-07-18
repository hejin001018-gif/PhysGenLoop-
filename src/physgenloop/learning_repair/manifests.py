"""Frozen Blender/Hunyuan rollout manifests with domain separation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


CAMPAIGN_SCHEMA_VERSION = "repair-campaign-manifest/1.0"


@dataclass(frozen=True)
class CampaignItem:
    sample_id: str
    group_id: str
    prompt: str
    source_video: str
    seed: int
    split: str
    source_sha256: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not all((self.sample_id.strip(), self.group_id.strip(), self.prompt.strip(), self.source_video.strip())):
            raise ValueError("campaign item identifiers, prompt, and source_video are required")
        if self.split not in {"train", "validation", "test", "calibration"}:
            raise ValueError(f"invalid campaign split: {self.split!r}")
        if self.source_sha256 is not None and len(self.source_sha256) != 64:
            raise ValueError("source_sha256 must be a SHA-256 hex digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "group_id": self.group_id,
            "prompt": self.prompt,
            "source_video": self.source_video,
            "seed": self.seed,
            "split": self.split,
            "source_sha256": self.source_sha256,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CampaignItem":
        return cls(
            sample_id=str(raw["sample_id"]),
            group_id=str(raw["group_id"]),
            prompt=str(raw["prompt"]),
            source_video=str(raw["source_video"]),
            seed=int(raw["seed"]),
            split=str(raw["split"]),
            source_sha256=None if raw.get("source_sha256") is None else str(raw["source_sha256"]),
            metadata=dict(raw.get("metadata", {})),
        )


@dataclass(frozen=True)
class FrozenCampaignManifest:
    campaign_id: str
    domain: str
    critic_model_id: str
    generator_model_id: str
    executor_version: str
    items: tuple[CampaignItem, ...]
    schema_version: str = CAMPAIGN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CAMPAIGN_SCHEMA_VERSION:
            raise ValueError("unsupported campaign manifest schema")
        if self.domain not in {"blender", "hunyuan"}:
            raise ValueError("campaign domain must be blender or hunyuan")
        if not self.items:
            raise ValueError("campaign manifest must contain items")
        ids = [item.sample_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("campaign sample_id values must be unique")
        if self.domain == "hunyuan" and any(
            item.split not in {"calibration", "test"} for item in self.items
        ):
            raise ValueError("Hunyuan campaign permits calibration/test splits only")
        group_splits: dict[str, set[str]] = {}
        for item in self.items:
            group_splits.setdefault(item.group_id, set()).add(item.split)
        leakage = {group: splits for group, splits in group_splits.items() if len(splits) > 1}
        if leakage:
            raise ValueError(f"campaign group leakage: {leakage}")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(include_fingerprint=False),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "domain": self.domain,
            "critic_model_id": self.critic_model_id,
            "generator_model_id": self.generator_model_id,
            "executor_version": self.executor_version,
            "items": [item.to_dict() for item in self.items],
        }
        if include_fingerprint:
            payload["manifest_sha256"] = self.fingerprint
        return payload

    @classmethod
    def load(cls, path: str | Path) -> "FrozenCampaignManifest":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        manifest = cls(
            schema_version=str(raw.get("schema_version", "")),
            campaign_id=str(raw["campaign_id"]),
            domain=str(raw["domain"]),
            critic_model_id=str(raw["critic_model_id"]),
            generator_model_id=str(raw["generator_model_id"]),
            executor_version=str(raw["executor_version"]),
            items=tuple(CampaignItem.from_dict(item) for item in raw["items"]),
        )
        recorded = raw.get("manifest_sha256")
        if recorded is not None and str(recorded) != manifest.fingerprint:
            raise ValueError("campaign manifest fingerprint mismatch")
        return manifest


def validate_campaign_artifacts(
    manifest: FrozenCampaignManifest,
    *,
    base_dir: str | Path,
) -> dict[str, Any]:
    root = Path(base_dir)
    failures = []
    if "REPLACE" in manifest.generator_model_id:
        failures.append("generator_model_id still contains a REPLACE placeholder")
    for item in manifest.items:
        if "REPLACE" in item.source_video:
            failures.append(f"placeholder source_video for {item.sample_id}")
            continue
        if "://" in item.source_video:
            continue
        path = root / item.source_video
        if not path.is_file():
            failures.append(f"missing {item.source_video}")
            continue
        if item.source_sha256:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != item.source_sha256:
                failures.append(f"checksum mismatch {item.source_video}")
    return {
        "valid": not failures,
        "campaign_id": manifest.campaign_id,
        "manifest_sha256": manifest.fingerprint,
        "domain": manifest.domain,
        "sample_count": len(manifest.items),
        "group_count": len({item.group_id for item in manifest.items}),
        "split_counts": {
            split: sum(item.split == split for item in manifest.items)
            for split in sorted({item.split for item in manifest.items})
        },
        "artifact_failures": failures,
    }
