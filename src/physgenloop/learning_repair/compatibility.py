"""Fail-fast Critic, feature, checkpoint, and executor compatibility gates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from physgenloop.learning_repair.contracts import ACTION_ORDER


COMPATIBILITY_SCHEMA_VERSION = "learning-repair-compatibility/1.0"


class CompatibilityError(RuntimeError):
    """Raised instead of silently running against an incompatible component."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CompatibilityManifest:
    critic_report_schema_version: str
    critic_model_id: str
    critic_config_sha256: str
    critic_schema_sha256: str
    feature_schema_sha256: str
    source_revision: str
    action_order: tuple[str, ...]
    data_domains: tuple[str, ...]
    schema_version: str = COMPATIBILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != COMPATIBILITY_SCHEMA_VERSION:
            raise CompatibilityError(
                f"unsupported compatibility manifest: {self.schema_version!r}"
            )
        expected = tuple(item.value for item in ACTION_ORDER)
        if self.action_order != expected:
            raise CompatibilityError(
                f"action order mismatch: expected {expected}, got {self.action_order}"
            )
        for name in (
            "critic_config_sha256",
            "critic_schema_sha256",
            "feature_schema_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise CompatibilityError(f"{name} is not a lowercase SHA-256")

    @property
    def compatibility_id(self) -> str:
        return f"lrcompat-{_canonical_sha(self.to_dict(include_id=False))[:16]}"

    @property
    def deployment_ready(self) -> bool:
        return self.source_revision not in {"", "unknown"}

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "critic_report_schema_version": self.critic_report_schema_version,
            "critic_model_id": self.critic_model_id,
            "critic_config_sha256": self.critic_config_sha256,
            "critic_schema_sha256": self.critic_schema_sha256,
            "feature_schema_sha256": self.feature_schema_sha256,
            "source_revision": self.source_revision,
            "action_order": list(self.action_order),
            "data_domains": list(self.data_domains),
            "deployment_ready": self.deployment_ready,
        }
        if include_id:
            payload["compatibility_id"] = self.compatibility_id
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CompatibilityManifest":
        return cls(
            schema_version=str(raw.get("schema_version", "")),
            critic_report_schema_version=str(raw["critic_report_schema_version"]),
            critic_model_id=str(raw["critic_model_id"]),
            critic_config_sha256=str(raw["critic_config_sha256"]),
            critic_schema_sha256=str(raw["critic_schema_sha256"]),
            feature_schema_sha256=str(raw["feature_schema_sha256"]),
            source_revision=str(raw.get("source_revision", "unknown")),
            action_order=tuple(str(item) for item in raw["action_order"]),
            data_domains=tuple(str(item) for item in raw.get("data_domains", ())),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CompatibilityManifest":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise CompatibilityError("compatibility manifest must be a JSON object")
        return cls.from_dict(raw)

    def verify_files(
        self,
        *,
        critic_config: str | Path,
        critic_schema: str | Path,
        feature_schema: str | Path,
    ) -> None:
        expected = {
            Path(critic_config): self.critic_config_sha256,
            Path(critic_schema): self.critic_schema_sha256,
            Path(feature_schema): self.feature_schema_sha256,
        }
        mismatches = []
        for path, digest in expected.items():
            if not path.is_file():
                mismatches.append(f"missing {path}")
                continue
            actual = sha256_file(path)
            if actual != digest:
                mismatches.append(f"{path}: expected {digest}, got {actual}")
        if mismatches:
            raise CompatibilityError("compatibility file check failed: " + "; ".join(mismatches))

    def assert_report(self, report: Any) -> None:
        if hasattr(report, "to_dict"):
            report = report.to_dict()
        if not isinstance(report, Mapping):
            raise CompatibilityError("CriticReport must be a mapping or expose to_dict()")
        version = str(report.get("schema_version", self.critic_report_schema_version))
        if version != self.critic_report_schema_version:
            raise CompatibilityError(
                f"CriticReport schema mismatch: expected {self.critic_report_schema_version}, got {version}"
            )
        missing = [
            name
            for name in ("decision", "physics_score", "confidence", "coverage", "violations")
            if name not in report
        ]
        if missing:
            raise CompatibilityError(f"CriticReport is missing required fields: {missing}")

    def assert_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        if tuple(checkpoint.get("action_order", ())) != self.action_order:
            raise CompatibilityError("checkpoint action order is incompatible")
        checkpoint_compat = str(checkpoint.get("compatibility_id", ""))
        if checkpoint_compat != self.compatibility_id:
            raise CompatibilityError(
                f"checkpoint compatibility mismatch: expected {self.compatibility_id}, got {checkpoint_compat or 'missing'}"
            )


def verify_proxy_baseline(
    manifest_path: str | Path,
    *,
    root: str | Path,
) -> dict[str, Any]:
    """Verify the archived 900g proxy release without changing it."""

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "proxy-baseline-archive/1.0":
        raise CompatibilityError("unsupported proxy baseline archive manifest")
    failures = []
    checked = []
    for item in manifest.get("files", ()):
        path = Path(root) / str(item["path"])
        if not path.is_file():
            failures.append(f"missing {path}")
            continue
        digest = sha256_file(path)
        size = path.stat().st_size
        checked.append(str(item["path"]))
        if digest != item["sha256"] or size != int(item["bytes"]):
            failures.append(
                f"{path}: expected {item['sha256']}/{item['bytes']}, got {digest}/{size}"
            )
    if failures:
        raise CompatibilityError("proxy baseline verification failed: " + "; ".join(failures))
    return {
        "valid": True,
        "baseline_id": manifest.get("baseline_id"),
        "checked_files": checked,
        "limitations": list(manifest.get("limitations", ())),
    }
