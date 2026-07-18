"""Immutable stage-keyed model response caching for benchmark runs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Callable, Mapping, Sequence

from pavg_critic.api_models import ModelAPIError
from pavg_critic.question_graph import QuestionGraphError
from pavg_critic.schemas import SchemaError


PROVIDER_ERRORS = (
    ModelAPIError,
    TimeoutError,
    ConnectionError,
    OSError,
    SchemaError,
    QuestionGraphError,
    KeyError,
    ValueError,
    TypeError,
)


@dataclass(frozen=True)
class ModelCallEvent:
    """Non-secret telemetry for one logical cached model call."""

    namespace: str
    model_id: str
    model_revision: str
    sample_id: str
    cache_key: str
    prompt_sha256: str
    schema_sha256: str
    input_evidence_sha256: str | None
    cache_hit: bool
    latency_sec: float
    error_type: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class AuditedCachedModel:
    """Implement text/multimodal protocols through one immutable cache."""

    def __init__(
        self,
        model,
        *,
        cache_dir: str | Path,
        namespace: str,
        model_id: str,
        model_revision: str,
        retries: int = 3,
        lock_timeout_sec: float = 300.0,
    ) -> None:
        if (
            not namespace
            or Path(namespace).name != namespace
            or any(mark in namespace for mark in ("/", "\\"))
        ):
            raise ValueError("namespace must be one safe path component")
        if not model_id:
            raise ValueError("model_id must not be empty")
        if not model_revision:
            raise ValueError("model_revision must not be empty")
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 1:
            raise ValueError("retries must be a positive integer")
        if lock_timeout_sec <= 0:
            raise ValueError("lock_timeout_sec must be positive")
        self.backend = model
        self.model = model_id
        self.cache_dir = Path(cache_dir)
        self.namespace = namespace
        self.model_id = model_id
        self.model_revision = model_revision
        self.sample_id: str | None = None
        self.retries = retries
        self.lock_timeout_sec = float(lock_timeout_sec)
        self._events: list[ModelCallEvent] = []

    def bind_sample(self, sample_id: str) -> None:
        if not sample_id or not sample_id.strip():
            raise ValueError("sample_id must not be empty")
        self.sample_id = sample_id

    @property
    def event_count(self) -> int:
        return len(self._events)

    def events_since(self, cursor: int) -> tuple[ModelCallEvent, ...]:
        if cursor < 0 or cursor > len(self._events):
            raise ValueError("event cursor is outside the event log")
        return tuple(self._events[cursor:])

    @staticmethod
    def _canonical_bytes(value: object) -> bytes:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def _sha(cls, value: object) -> str:
        return hashlib.sha256(cls._canonical_bytes(value)).hexdigest()

    @staticmethod
    def _pid_alive(pid: int | None) -> bool:
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

    @staticmethod
    def _lock_pid(path: Path) -> int | None:
        try:
            text = path.read_text(encoding="ascii").strip()
            return int(text.removeprefix("pid="))
        except (OSError, UnicodeError, ValueError):
            return None

    def _acquire_lock(self, path: Path, lock_path: Path) -> int | None:
        """Return an owned descriptor, or None when another writer filled cache."""

        started = perf_counter()
        while True:
            if path.is_file():
                return None
            try:
                descriptor = os.open(
                    lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            except FileExistsError:
                if not self._pid_alive(self._lock_pid(lock_path)):
                    try:
                        lock_path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                if perf_counter() - started >= self.lock_timeout_sec:
                    raise TimeoutError(f"model cache lock timed out: {lock_path}")
                sleep(0.05)
                continue
            os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
            os.fsync(descriptor)
            return descriptor

    def _read_cache(
        self,
        path: Path,
        *,
        cache_key: str,
    ) -> Mapping[str, Any]:
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeError, OSError) as exc:
            raise ValueError(f"invalid model cache JSON: {path}") from exc
        expected = {
            "cache_key": cache_key,
            "namespace": self.namespace,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "sample_id": self.sample_id,
        }
        if not isinstance(cached, dict) or any(
            cached.get(name) != value for name, value in expected.items()
        ):
            raise ValueError(f"model cache metadata mismatch: {path}")
        response = cached.get("response")
        if not isinstance(response, Mapping):
            raise ValueError(f"model cache response must be an object: {path}")
        return dict(response)

    @staticmethod
    def _write_cache(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw_temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                raw_temporary = handle.name
                json.dump(
                    payload,
                    handle,
                    allow_nan=False,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(raw_temporary, path)
            raw_temporary = None
        finally:
            if raw_temporary is not None:
                Path(raw_temporary).unlink(missing_ok=True)

    def _event(
        self,
        *,
        cache_key: str,
        prompt_sha256: str,
        schema_sha256: str,
        input_evidence_sha256: str | None,
        cache_hit: bool,
        started: float,
        error_type: str | None = None,
    ) -> None:
        self._events.append(
            ModelCallEvent(
                namespace=self.namespace,
                model_id=self.model_id,
                model_revision=self.model_revision,
                sample_id=self.sample_id or "",
                cache_key=cache_key,
                prompt_sha256=prompt_sha256,
                schema_sha256=schema_sha256,
                input_evidence_sha256=input_evidence_sha256,
                cache_hit=cache_hit,
                latency_sec=perf_counter() - started,
                error_type=error_type,
            )
        )

    def _invoke(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: Mapping[str, Any],
        image_data_urls: Sequence[str],
        provider_call: Callable[[], Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        if self.sample_id is None:
            raise RuntimeError("bind_sample() is required before a model call")
        prompt_hash = self._sha({"system": system_prompt, "user": user_prompt})
        schema_hash = self._sha(schema)
        image_hashes = tuple(
            hashlib.sha256(item.encode("utf-8")).hexdigest()
            for item in image_data_urls
        )
        evidence_hash = self._sha(image_hashes) if image_hashes else None
        key_payload = {
            "schema_version": "1.0",
            "namespace": self.namespace,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "sample_id": self.sample_id,
            "prompt_sha256": prompt_hash,
            "schema_sha256": schema_hash,
            "input_evidence_sha256": evidence_hash,
        }
        key = self._sha(key_payload)
        path = self.cache_dir / self.namespace / key[:2] / f"{key}.json"
        lock_path = path.with_suffix(".json.lock")
        started = perf_counter()
        if path.is_file():
            response = self._read_cache(path, cache_key=key)
            self._event(
                cache_key=key,
                prompt_sha256=prompt_hash,
                schema_sha256=schema_hash,
                input_evidence_sha256=evidence_hash,
                cache_hit=True,
                started=started,
            )
            return response

        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = self._acquire_lock(path, lock_path)
        if descriptor is None:
            response = self._read_cache(path, cache_key=key)
            self._event(
                cache_key=key,
                prompt_sha256=prompt_hash,
                schema_sha256=schema_hash,
                input_evidence_sha256=evidence_hash,
                cache_hit=True,
                started=started,
            )
            return response
        os.close(descriptor)
        try:
            response: Mapping[str, Any] | None = None
            for attempt in range(self.retries):
                try:
                    candidate = provider_call()
                    if not isinstance(candidate, Mapping):
                        raise TypeError("model response must be a JSON object")
                    response = dict(candidate)
                    break
                except PROVIDER_ERRORS as exc:
                    if attempt + 1 == self.retries:
                        self._event(
                            cache_key=key,
                            prompt_sha256=prompt_hash,
                            schema_sha256=schema_hash,
                            input_evidence_sha256=evidence_hash,
                            cache_hit=False,
                            started=started,
                            error_type=type(exc).__name__,
                        )
                        raise
                    sleep(2**attempt)
            if response is None:  # pragma: no cover - loop always returns or raises
                raise RuntimeError("model retry loop ended without a response")
            self._write_cache(
                path,
                {
                    **key_payload,
                    "cache_key": key,
                    "response": response,
                },
            )
            self._event(
                cache_key=key,
                prompt_sha256=prompt_hash,
                schema_sha256=schema_hash,
                input_evidence_sha256=evidence_hash,
                cache_hit=False,
                started=started,
            )
            return response
        finally:
            lock_path.unlink(missing_ok=True)

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return self._invoke(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=schema,
            image_data_urls=(),
            provider_call=lambda: self.backend.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=schema,
            ),
        )

    def generate_json_with_images(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_data_urls: Sequence[str],
        schema: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return self._invoke(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=schema,
            image_data_urls=image_data_urls,
            provider_call=lambda: self.backend.generate_json_with_images(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_data_urls=image_data_urls,
                schema=schema,
            ),
        )
