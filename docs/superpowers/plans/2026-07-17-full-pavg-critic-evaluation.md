# Full PAVG Critic Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate the complete prompt-conditioned PAVG Critic on all 3,397 frozen VideoPhy-2 videos, with module attribution and prompt-conditioning diagnostics, without recomputing SAM2 observations or modifying the accepted D0/B1 result.

**Architecture:** Extend the real-video adapter to inject one frozen Qwen3-VL snapshot independently into Planner, PQSG and candidate-verifier boundaries, with stage-keyed immutable response caches and paired prediction/diagnostic journaling. Run D1 and M1–M5 over two disjoint observation-owner shards, reuse accepted D0/B1 predictions, and publish a new immutable multi-method report plus three frozen 300-sample prompt views. Existing full-report defaults and accepted artifacts remain unchanged.

**Tech Stack:** Python 3.12, pytest, dataclasses, JSON/JSONL, OpenAI-compatible Chat Completions, Qwen3-VL-8B-Instruct, vLLM 0.11.0, PyTorch CUDA 12.8, official SAM2.1 Hiera B+, OpenCV, PowerShell/OpenSSH, tmux, NVIDIA A100 40 GB.

---

## Frozen execution rules

- Primary manifest: `evaluation/manifests/videophy2_test_full.json`, SHA-256 `d8be5fe97ddf6902515c09ccbb53f394b25230213db7c3058d61f84748624906`.
- Diagnostic manifest: `evaluation/manifests/videophy2_pilot300.json`, SHA-256 `a97762fe4033789eb14a82717c72c14e89bc75a7a67200d5890ff1647f72a670`.
- Model: the existing `Qwen/Qwen3-VL-8B-Instruct` snapshot and deterministic 16-frame policy; no model/prompt/threshold tuning after smoke starts.
- Accepted input report commit: `596663a010410eb3ca27ef9cdf060c3b9479a418`; its D0/B1 artifacts are read-only.
- No SAM2 propagation is allowed. A missing observation cache is a terminal preparation error.
- No `.env`, SSH password, API key, model weight, video, mask, image payload or raw provider response enters git.
- All new remote state stays below `/root/pavg-benchmark` or `/root/pavg-benchmark-shard2`.
- cloud2 owns 1,731 samples and cloud1 owns 1,666; ownership must be disjoint and cover 3,397 exactly.
- User-facing progress is checked at least every 15 minutes while work is active and summarized at least every 30 minutes. The remote heartbeat interval is 300 seconds.

## File responsibility map

| File | Responsibility |
|---|---|
| `src/pavg_critic/schemas.py` | Expose keyframes and VLM reviews in `CriticArtifacts` for audit only |
| `src/pavg_critic/pipeline.py` | Record pre-evidence-fusion state and return the complete typed artifacts |
| `src/pavg_critic/benchmarking/model_cache.py` | Stage-separated deterministic model response cache and non-secret call telemetry |
| `src/pavg_critic/benchmarking/pavg_diagnostics.py` | Build and validate one module-diagnostic sidecar record |
| `src/pavg_critic/benchmarking/audited_runner.py` | Crash-recoverable paired append of prediction and diagnostics |
| `src/pavg_critic/benchmarking/prompt_diagnostics.py` | Deterministic cross-action prompt derangement and oracle plan adapter |
| `src/pavg_critic/benchmarking/pavg_methods.py` | Real-video M5 construction and audited evaluation |
| `benchmarks/evaluate_video_benchmark.py` | Expose M5, stage caches, diagnostics and bounded failure policy |
| `benchmarks/build_prompt_diagnostics.py` | Freeze shuffled-prompt manifest and donor map |
| `src/pavg_critic/benchmarking/full_report.py` | Generalize paired helpers while preserving D0/B1 defaults |
| `src/pavg_critic/benchmarking/full_pavg_report.py` | Validate multi-method predictions/diagnostics and compute attribution |
| `benchmarks/report_full_pavg_critic.py` | Atomic report-bundle CLI for primary and diagnostic results |
| `benchmarks/monitor_video_benchmark.py` | Append non-secret 5-minute progress/GPU/endpoint heartbeats and flag stalls |
| `docs/results/criticbenchmark.md` | Publish the final Chinese interpretation and limitations |

### Task 1: Freeze approved inputs and local baseline

**Files:**
- Modify: `docs/superpowers/specs/2026-07-17-full-pavg-critic-evaluation-design.md`
- Create: `evaluation/manifests/videophy2_pilot300.json`
- Create: `docs/superpowers/plans/2026-07-17-full-pavg-critic-evaluation.md`
- Modify: this plan under `Execution results`

- [x] **Step 1: Verify the approved source state and immutable full manifest**

Run:

```powershell
git status --short
git rev-parse HEAD
Get-FileHash evaluation/manifests/videophy2_test_full.json -Algorithm SHA256
```

Expected: only the approved spec/plan documentation is changed before the plan commit; the manifest digest is `D8BE5FE97DDF6902515C09CCBB53F394B25230213DB7C3058D61F84748624906`.

- [x] **Step 2: Locate and synchronize the previously frozen pilot manifest by hash**

Run from PowerShell, using key-only authentication:

```powershell
$key = "$env:USERPROFILE\.ssh\pavg_benchmark_ed25519"
$known = "$env:LOCALAPPDATA\Temp\pavg-knownhosts-scan"
$hash = 'a97762fe4033789eb14a82717c72c14e89bc75a7a67200d5890ff1647f72a670'
$matches = ssh -i $key -o BatchMode=yes -o UserKnownHostsFile=$known -p 29848 root@px-cloud2.matpool.com "find /root/pavg-benchmark -type f -name '*.json' -exec sha256sum {} + 2>/dev/null | grep '^$hash '"
if (($matches | Measure-Object).Count -ne 1) { throw "expected exactly one pilot manifest: $matches" }
$remotePath = ($matches -split '\s+', 2)[1]
scp -i $key -o BatchMode=yes -o UserKnownHostsFile=$known -P 29848 "root@px-cloud2.matpool.com:$remotePath" evaluation/manifests/videophy2_pilot300.json
Get-FileHash evaluation/manifests/videophy2_pilot300.json -Algorithm SHA256
```

Expected: exactly one remote match and local SHA-256 `A97762FE4033789EB14A82717C72C14E89BC75A7A67200D5890FF1647F72A670`.

- [x] **Step 3: Run the existing complete test suite without touching code**

Run:

```powershell
New-Item -ItemType Directory -Force outputs/.pytest-full-pavg | Out-Null
.\.venv\Scripts\python.exe -m pytest --basetemp outputs/.pytest-full-pavg -q
```

Expected: the current full suite passes with zero failures; record the exact count and elapsed time.

- [x] **Step 4: Commit only the approved design, plan and frozen diagnostic manifest**

```powershell
git add docs/superpowers/specs/2026-07-17-full-pavg-critic-evaluation-design.md docs/superpowers/plans/2026-07-17-full-pavg-critic-evaluation.md evaluation/manifests/videophy2_pilot300.json
git diff --cached --check
git commit -m "docs: plan full PAVG critic evaluation"
```

Expected: one commit containing exactly those three paths.

### Task 2: Expose complete typed pipeline audit artifacts

**Files:**
- Modify: `src/pavg_critic/schemas.py`
- Modify: `src/pavg_critic/pipeline.py`
- Create: `tests/test_pipeline_artifacts.py`

- [x] **Step 1: Write failing tests for pre-fusion state, keyframes and reviews**

Add tests that construct a critic with fixed observations and a fake verifier, then assert:

```python
artifacts = critic.analyze_detailed(request, observations=states)
assert artifacts.keyframes
assert artifacts.reviews
assert artifacts.report.diagnostics["pre_evidence_fusion"]["decision"] in {
    "physical", "violation", "unknown"
}
assert artifacts.report.diagnostics["hard_violation_override"] is bool(
    artifacts.report.violations
)
```

Also round-trip with `json.dumps(artifacts.to_dict(), allow_nan=False)`.

- [x] **Step 2: Run the focused tests and verify the missing fields fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_pipeline_artifacts.py -q
```

Expected: FAIL because `CriticArtifacts` has no `keyframes` or `reviews` and the report lacks `pre_evidence_fusion`.

- [x] **Step 3: Add audit-only fields with backward-compatible defaults**

Add after `candidates` in `CriticArtifacts`:

```python
keyframes: dict[int, tuple[int, ...]] = field(default_factory=dict)
reviews: dict[int, VLMReview | None] = field(default_factory=dict)
```

Immediately before the existing `self.evidence_fusion.enrich` call, save only public decision fields:

```python
pre_evidence_fusion = {
    "decision": report.decision,
    "physics_score": report.physics_score,
    "confidence": report.confidence,
    "coverage": report.coverage,
}
report = self.evidence_fusion.enrich(
    report,
    tracks=tracks,
    candidates=candidates,
    reviews=reviews,
    checklist_summary=checklist_summary,
    mechanics_summary=mechanics_summary,
)
diagnostics = dict(report.diagnostics)
diagnostics["pre_evidence_fusion"] = pre_evidence_fusion
diagnostics["hard_violation_override"] = bool(report.violations)
report = replace(report, diagnostics=diagnostics)
```

Return immutable copies:

```python
keyframes={index: tuple(frames) for index, frames in keyframes.items()},
reviews=dict(reviews),
```

- [x] **Step 4: Run focused and complete tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_pipeline_artifacts.py -q
.\.venv\Scripts\python.exe -m pytest --basetemp outputs/.pytest-full-pavg -q
```

Expected: both commands PASS.

- [x] **Step 5: Commit the typed audit boundary**

```powershell
git add src/pavg_critic/schemas.py src/pavg_critic/pipeline.py tests/test_pipeline_artifacts.py docs/superpowers/plans/2026-07-17-full-pavg-critic-evaluation.md
git commit -m "feat: expose critic fusion audit artifacts"
```

### Task 3: Add deterministic stage-separated model caches

**Files:**
- Create: `src/pavg_critic/benchmarking/model_cache.py`
- Create: `tests/benchmarking/test_model_cache.py`

- [x] **Step 1: Write failing cache/telemetry tests**

Cover text and image calls, exact cache reuse, namespace separation, schema/prompt invalidation, corrupted-cache rejection and provider-error non-caching. The central assertion is:

```python
planner = AuditedCachedModel(fake, cache_dir=tmp_path, namespace="planner", model_id="qwen")
first = planner.generate_json(system_prompt="s", user_prompt="u", schema=SCHEMA)
second = planner.generate_json(system_prompt="s", user_prompt="u", schema=SCHEMA)
assert first == second
assert fake.text_calls == 1
events = planner.events_since(0)
assert [event.cache_hit for event in events] == [False, True]
assert all(len(event.cache_key) == 64 for event in events)
assert "u" not in json.dumps([event.to_dict() for event in events])
```

- [x] **Step 2: Run the new test and verify import failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/benchmarking/test_model_cache.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `model_cache`.

- [x] **Step 3: Implement canonical keying and atomic response caching**

Create these public types and methods; the implementation below is the required cache/retry boundary:

```python
@dataclass(frozen=True)
class ModelCallEvent:
    namespace: str
    model_id: str
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
    def __init__(self, model, *, cache_dir: str | Path, namespace: str,
                 model_id: str, retries: int = 3):
        if not namespace or "/" in namespace or "\\" in namespace:
            raise ValueError("namespace must be one safe path component")
        if retries < 1:
            raise ValueError("retries must be positive")
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.namespace = namespace
        self.model_id = model_id
        self.retries = retries
        self._events: list[ModelCallEvent] = []

    @property
    def event_count(self) -> int:
        return len(self._events)

    def events_since(self, cursor: int) -> tuple[ModelCallEvent, ...]:
        if cursor < 0 or cursor > len(self._events):
            raise ValueError("event cursor is outside the event log")
        return tuple(self._events[cursor:])

    @staticmethod
    def _sha(value: object) -> str:
        content = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def _invoke(self, *, system_prompt, user_prompt, schema,
                image_data_urls, provider_call):
        prompt_hash = self._sha({"system": system_prompt, "user": user_prompt})
        schema_hash = self._sha(schema)
        image_hashes = [hashlib.sha256(item.encode("utf-8")).hexdigest()
                        for item in image_data_urls]
        evidence_hash = self._sha(image_hashes) if image_hashes else None
        key_payload = {
            "schema_version": "1.0",
            "namespace": self.namespace,
            "model_id": self.model_id,
            "prompt_sha256": prompt_hash,
            "schema_sha256": schema_hash,
            "input_evidence_sha256": evidence_hash,
        }
        key = self._sha(key_payload)
        path = self.cache_dir / self.namespace / key[:2] / f"{key}.json"
        started = perf_counter()
        if path.is_file():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if any(cached.get(name) != value for name, value in {
                "cache_key": key, "namespace": self.namespace,
                "model_id": self.model_id,
            }.items()):
                raise ValueError(f"model cache metadata mismatch: {path}")
            response = cached["response"]
            self._events.append(ModelCallEvent(
                self.namespace, self.model_id, key, prompt_hash, schema_hash,
                evidence_hash, True, perf_counter() - started,
            ))
            return response
        error = None
        for attempt in range(self.retries):
            try:
                response = dict(provider_call())
                break
            except PROVIDER_ERRORS as exc:
                error = exc
                if attempt + 1 == self.retries:
                    self._events.append(ModelCallEvent(
                        self.namespace, self.model_id, key, prompt_hash, schema_hash,
                        evidence_hash, False, perf_counter() - started,
                        type(exc).__name__,
                    ))
                    raise
                sleep(2 ** attempt)
        if error is not None and "response" not in locals():
            raise error
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {**key_payload, "cache_key": key, "response": response}
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        self._events.append(ModelCallEvent(
            self.namespace, self.model_id, key, prompt_hash, schema_hash,
            evidence_hash, False, perf_counter() - started,
        ))
        return response

    def generate_json(self, *, system_prompt, user_prompt, schema):
        return self._invoke(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=schema,
            image_data_urls=(),
            provider_call=lambda: self.model.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=schema,
            ),
        )

    def generate_json_with_images(self, *, system_prompt, user_prompt,
                                  image_data_urls, schema):
        return self._invoke(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=schema,
            image_data_urls=image_data_urls,
            provider_call=lambda: self.model.generate_json_with_images(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_data_urls=image_data_urls,
                schema=schema,
            ),
        )
```

Import `asdict`, `dataclass`, `hashlib`, `json`, `Path`, `perf_counter` and `sleep`. Define `PROVIDER_ERRORS` as `(ModelAPIError, TimeoutError, ConnectionError, OSError, SchemaError, QuestionGraphError, KeyError, ValueError, TypeError)`. The cache key is SHA-256 of canonical JSON containing `schema_version=1.0`, namespace, model ID, prompt hashes, schema hash and ordered image-data SHA-256 values. Store only parsed response JSON and those hashes at `cache_dir / namespace / key[:2] / f"{key}.json"`; write via a sibling `.tmp` followed by `replace`. Never cache failures, and reject a cache file whose embedded key/namespace/model does not match.

- [x] **Step 4: Run cache tests and the complete suite**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/benchmarking/test_model_cache.py -q
.\.venv\Scripts\python.exe -m pytest --basetemp outputs/.pytest-full-pavg -q
```

Expected: PASS; the fake provider is called once across two identical calls.

- [x] **Step 5: Commit the model cache**

```powershell
git add src/pavg_critic/benchmarking/model_cache.py tests/benchmarking/test_model_cache.py
git commit -m "feat: cache audited benchmark model stages"
```

### Task 4: Build deterministic module diagnostics and M5/oracle adapters

**Files:**
- Create: `src/pavg_critic/benchmarking/pavg_diagnostics.py`
- Create: `src/pavg_critic/benchmarking/prompt_diagnostics.py`
- Modify: `src/pavg_critic/benchmarking/pavg_methods.py`
- Create: `tests/benchmarking/test_pavg_diagnostics.py`
- Modify: `tests/benchmarking/test_pavg_methods.py`
- Create: `tests/benchmarking/test_prompt_diagnostics.py`

- [x] **Step 1: Replace the old M5-rejection test with failing M5 injection tests**

The new constructor contract is:

```python
method = PAVGMethod(
    "M5_FULL",
    provider,
    model_id="Qwen/Qwen3-VL-8B-Instruct",
    planner_model=planner,
    question_model=pqsg,
    verifier_model=verifier,
    model_stages={"planner": planner, "pqsg": pqsg, "verifier": verifier},
)
prediction, diagnostics = method.evaluate_audited(sample)
assert prediction.method_id == "M5_FULL"
assert diagnostics["model_calls"]["planner"]["call_count"] == 1
assert diagnostics["model_calls"]["pqsg"]["call_count"] == 1
```

Assert M5 rejects any missing one of the three explicit models. Preserve the existing M4 tests.

- [x] **Step 2: Write failing oracle-plan tests**

Use an exact model plan and rules `("rule a", "rule b")`, then assert:

```python
plan = OracleRulePhysicsPlanner(model_planner, ("rule a", "rule b")).generate("prompt")
assert plan.objects == model_plan.objects
assert plan.expected_events == model_plan.expected_events
assert plan.relations == model_plan.relations
assert [item.id for item in plan.physics_constraints[-2:]] == [
    "oracle-rule-0", "oracle-rule-1"
]
assert plan.planner_metadata == PlannerMetadata(
    source="explicit", confidence=1.0, fallback_used=False, model=model_name
)
```

Also test the synthetic `scene` object and collision with an existing `oracle-rule-0` ID as a terminal `SchemaError`.

- [x] **Step 3: Write failing diagnostics-schema tests**

Build fixed `CriticArtifacts` and assert the record contains exactly the approved non-secret sections:

```python
assert diagnostics["key"] == {"sample_id": "1", "method_id": "M5_FULL"}
assert diagnostics["question_graph"]["status_counts"]["blocked"] >= 0
assert set(diagnostics["evidence_families"]) == {
    "rules", "pqsg", "checklist", "mechanics", "vlm"
}
assert diagnostics["hard_violation_override"] is True
assert "resolved_plan" not in json.dumps(diagnostics)
assert "image_data" not in json.dumps(diagnostics)
```

- [x] **Step 4: Implement the oracle adapter and diagnostic builder**

`OracleRulePhysicsPlanner.generate()` must call the cached normal `ModelPhysicsPlanner`, preserve objects/events/relations and existing constraints, append exact constraints, and use this implementation:

```python
class OracleRulePhysicsPlanner:
    def __init__(self, model_planner, rules, *, model_id):
        self.model_planner = model_planner
        self.rules = tuple(str(rule) for rule in rules if str(rule).strip())
        self.model_id = model_id

    def generate(self, prompt, partial_plan=None):
        plan = self.model_planner.generate(prompt, partial_plan)
        existing_ids = {item.id for item in plan.physics_constraints}
        oracle_ids = {f"oracle-rule-{index}" for index in range(len(self.rules))}
        collisions = sorted(existing_ids & oracle_ids)
        if collisions:
            raise SchemaError(f"oracle constraint ID collision: {collisions}")
        objects = plan.objects or ("scene",)
        constraints = plan.physics_constraints + tuple(
            PhysicsConstraint(
                id=f"oracle-rule-{index}",
                domain="oracle_natural_language",
                subjects=objects,
                expectation=rule,
            )
            for index, rule in enumerate(self.rules)
        )
        return PhysicsPlan(
            objects=objects,
            expected_events=plan.expected_events,
            relations=plan.relations,
            physics_constraints=constraints,
            planner_metadata=PlannerMetadata(
                source="explicit",
                confidence=1.0,
                fallback_used=False,
                model=self.model_id,
            ),
        )
```

`build_pavg_diagnostics` must emit `schema_version`, key, Planner counts/source, question-node status counts, VideoScience status counts, mechanics applicability counts, rule candidate/retained counts, VLM claim counts, five evidence-family records with configured/effective weights, pre/final decision, hard override, stage call events and stage/total latency. Serialize with `allow_nan=False` during tests.

- [x] **Step 5: Extend `PAVGMethod` without changing M1–M4 semantics**

Extend the constructor with `planner_model=None`, `question_model=None`, `model_stages: Mapping[str, AuditedCachedModel] | None = None`, `output_method_id: str | None = None`, and `oracle_plan: bool = False`. Use this mode/output contract:

```python
supported = {"B1_RULE", "M1_GRAPH", "M2_CHECKLIST", "M3_MECHANICS", "M4_VLM", "M5_FULL"}
if mode == "M5_FULL" and any(model is None for model in (
    planner_model, question_model, verifier_model
)):
    raise ValueError("M5_FULL requires planner, question, and verifier models")
self.mode = mode
self.method_id = output_method_id or mode
self.planner_model = planner_model
self.question_model = question_model
self.model_stages = dict(model_stages or {})
self.oracle_plan = oracle_plan
if oracle_plan and mode != "M5_FULL":
    raise ValueError("oracle_plan is valid only for M5_FULL")
```

Call `build_ablation_config(self.mode)` and `analyze_detailed`, not `analyze`. Capture `{stage: model.event_count}` before the call, then pass `{stage: model.events_since(cursor)}` to the diagnostic builder. Return `(BenchmarkPrediction, diagnostics)` from `evaluate_audited`; keep `evaluate` as `return self.evaluate_audited(sample)[0]` so existing runner callers remain compatible. M5 injects `planner_model`, `question_model` and grouped `vlm_verifier`; oracle M5 constructs `OracleRulePhysicsPlanner(ModelPhysicsPlanner(self.planner_model), sample.physical_rules, model_id=self.model_id)` and injects it as `physics_planner` instead of `planner_model`.

- [x] **Step 6: Run focused and complete tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/benchmarking/test_pavg_methods.py tests/benchmarking/test_pavg_diagnostics.py tests/benchmarking/test_prompt_diagnostics.py -q
.\.venv\Scripts\python.exe -m pytest --basetemp outputs/.pytest-full-pavg -q
```

Expected: PASS; existing B1–M4 assertions remain unchanged.

- [x] **Step 7: Commit M5 and diagnostics construction**

```powershell
git add src/pavg_critic/benchmarking/pavg_diagnostics.py src/pavg_critic/benchmarking/prompt_diagnostics.py src/pavg_critic/benchmarking/pavg_methods.py tests/benchmarking/test_pavg_diagnostics.py tests/benchmarking/test_pavg_methods.py tests/benchmarking/test_prompt_diagnostics.py
git commit -m "feat: add audited full PAVG benchmark method"
```

### Task 5: Add crash-recoverable paired prediction/diagnostic output

**Files:**
- Create: `src/pavg_critic/benchmarking/audited_runner.py`
- Create: `tests/benchmarking/test_audited_runner.py`
- Modify: `src/pavg_critic/benchmarking/runner.py`
- Modify: `tests/benchmarking/test_runner.py`

- [x] **Step 1: Write failing recovery and failure-budget tests**

Cover normal append, resume, duplicate rejection, a crash after the prediction append, a crash after the diagnostics append, stale pending recovery and failure-budget stop. Use this invariant:

```python
runner = AuditedBenchmarkRunner(predictions, diagnostics)
runner.run((sample,), (method,))
assert load_keys(predictions) == load_keys(diagnostics) == {("1", "M5_FULL")}
runner.run((sample,), (method,))
assert method.calls == 1
```

For ordinary `BenchmarkRunner`, add `max_new_failures=1` and assert it stops immediately after the first new failed prediction instead of consuming later samples.

- [x] **Step 2: Run focused tests and verify failures**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/benchmarking/test_audited_runner.py tests/benchmarking/test_runner.py -q
```

Expected: FAIL because paired output and failure budgets do not exist.

- [x] **Step 3: Implement the paired write-ahead record**

Create `AuditedBenchmarkRunner` with the exact public signatures below; `_recover_pending`, `_load_key_index`, `_exclusive_lock`, `_write_pending`, `_append_fsync` and `_validate_pair` are private helpers in the same file and are individually exercised by the crash-injection tests:

```python
class AuditedBenchmarkRunner:
    def __init__(self, prediction_path, diagnostics_path, *, max_new_failures=1):
        self.prediction_path = Path(prediction_path)
        self.diagnostics_path = Path(diagnostics_path)
        self.max_new_failures = max_new_failures
        if isinstance(max_new_failures, bool) or max_new_failures < 1:
            raise ValueError("max_new_failures must be a positive integer")

    def run(self, samples, methods) -> tuple[BenchmarkPrediction, ...]:
        new_records = []
        with self._exclusive_lock():
            self._recover_pending()
            completed = self._validate_pair()
            failures = 0
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
                    self.pending_path.unlink()
                    completed.add(key)
                    new_records.append(prediction)
                    failures += prediction.failure is not None
                    if failures >= self.max_new_failures:
                        raise RuntimeError("new failure budget reached")
        return tuple(new_records)
```

Use one exclusive `self.prediction_path.with_suffix(self.prediction_path.suffix + ".lock")`. Before either append, atomically write `self.prediction_path.with_suffix(self.prediction_path.suffix + ".pending.json")` containing both complete JSON objects. Append/fsync the missing prediction and diagnostic records, then delete pending. On startup, replay a valid pending record; without pending, any asymmetric key set is an error. Count newly appended `failure` records and raise `RuntimeError("new failure budget reached")` once the configured budget is reached. Never delete completed JSONL lines.

- [x] **Step 4: Add the same bounded failure policy to the ordinary runner**

Extend `BenchmarkRunner.run(self, samples, methods, *, max_new_failures: int | None = None)` while preserving `None` as the old unlimited default. Validate positive integers and stop only after fsyncing the terminal failure record.

- [x] **Step 5: Run focused and complete tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/benchmarking/test_audited_runner.py tests/benchmarking/test_runner.py -q
.\.venv\Scripts\python.exe -m pytest --basetemp outputs/.pytest-full-pavg -q
```

Expected: PASS, including injected crash recovery.

- [x] **Step 6: Commit paired resumability**

```powershell
git add src/pavg_critic/benchmarking/audited_runner.py src/pavg_critic/benchmarking/runner.py tests/benchmarking/test_audited_runner.py tests/benchmarking/test_runner.py
git commit -m "feat: journal benchmark diagnostics atomically"
```

### Task 6: Expose M5 and freeze prompt-diagnostic manifests

**Files:**
- Modify: `benchmarks/evaluate_video_benchmark.py`
- Create: `benchmarks/build_prompt_diagnostics.py`
- Modify: `tests/benchmarking/test_cli.py`
- Create: `tests/benchmarking/test_prompt_diagnostics_cli.py`
- Create: `evaluation/manifests/videophy2_pilot300_shuffled.json`
- Create: `evaluation/manifests/videophy2_pilot300_prompt_donors.json`

- [x] **Step 1: Write failing CLI-contract tests**

Assert accepted methods and exact new arguments:

```python
assert parse_methods("M5_FULL,M5_ORACLE_PLAN_300") == (
    "M5_FULL", "M5_ORACLE_PLAN_300"
)
args = build_parser().parse_args([
    "--manifest", "m.json", "--run-dir", "run", "--methods", "M5_FULL",
    "--model-cache-dir", "cache", "--max-new-failures", "1",
])
assert args.model_cache_dir == Path("cache")
assert args.max_new_failures == 1
```

Also assert M5 refuses a missing model cache and that resolved config contains stage namespaces/hashes but never the API key.

- [x] **Step 2: Write failing deterministic derangement tests**

Run the builder twice on a fixture and require byte-identical outputs. For every mapping:

```python
assert recipient.sample_id != donor.sample_id
assert recipient.prompt != donor.prompt
assert recipient.prompt_group_id != donor.prompt_group_id
```

Assert labels, video paths, generators and membership are unchanged, and prove the matcher never reads `physics_label` by changing all labels and obtaining the same donor map.

- [x] **Step 3: Implement the diagnostic builder**

Use a deterministic augmenting-path bipartite matcher. Recipients are sorted by sample ID; donor priority is a `random.Random(20260717)` shuffle. An edge exists only for a different sample ID, exact prompt and action group. Write the shuffled manifest and recipient→donor map with sorted keys, `allow_nan=False`, UTF-8 and a final newline; refuse to overwrite a different existing file.

- [x] **Step 4: Extend the evaluation CLI**

Add `M5_FULL`, `M5_SHUFFLED_PROMPT_300` and `M5_ORACLE_PLAN_300` to the method parser. Map the two diagnostic IDs to M5 configuration while preserving their output IDs. Require `--model-cache-dir` for M4/M5, create separate `AuditedCachedModel` wrappers for `planner`, `pqsg` and `verifier`, and use `AuditedBenchmarkRunner` for PAVG methods. Add `--max-new-failures` default `1`. The model is built once and shared only below the wrappers.

- [x] **Step 5: Freeze the real 300-sample shuffled artifacts**

Run:

```powershell
.\.venv\Scripts\python.exe -m benchmarks.build_prompt_diagnostics --manifest evaluation/manifests/videophy2_pilot300.json --output-manifest evaluation/manifests/videophy2_pilot300_shuffled.json --donor-map evaluation/manifests/videophy2_pilot300_prompt_donors.json --seed 20260717
Get-FileHash evaluation/manifests/videophy2_pilot300_shuffled.json,evaluation/manifests/videophy2_pilot300_prompt_donors.json -Algorithm SHA256
```

Expected: 300 recipients, 300 unique donors, zero forbidden matches; append both hashes to `Execution results` before any diagnostic inference.

- [x] **Step 6: Run focused and complete tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/benchmarking/test_cli.py tests/benchmarking/test_prompt_diagnostics_cli.py -q
.\.venv\Scripts\python.exe -m pytest --basetemp outputs/.pytest-full-pavg -q
```

Expected: PASS.

- [x] **Step 7: Commit CLI and frozen diagnostic inputs**

```powershell
git add benchmarks/evaluate_video_benchmark.py benchmarks/build_prompt_diagnostics.py tests/benchmarking/test_cli.py tests/benchmarking/test_prompt_diagnostics_cli.py evaluation/manifests/videophy2_pilot300_shuffled.json evaluation/manifests/videophy2_pilot300_prompt_donors.json
git commit -m "feat: expose frozen prompt-conditioned M5 runs"
```

### Task 7: Generalize paired statistics and add the full PAVG report

**Files:**
- Modify: `src/pavg_critic/benchmarking/full_report.py`
- Create: `src/pavg_critic/benchmarking/full_pavg_report.py`
- Create: `benchmarks/report_full_pavg_critic.py`
- Modify: `tests/benchmarking/test_full_report.py`
- Create: `tests/benchmarking/test_full_pavg_report.py`
- Create: `tests/benchmarking/test_full_pavg_report_cli.py`

- [x] **Step 1: Write backward-compatibility and generic-method tests**

Keep every existing D0/B1 assertion. Add explicit method parameters:

```python
paired_outcomes(samples, d0, m5,
                baseline_method="D0_DIRECT_VLM", candidate_method="M5_FULL")
action_group_bootstrap(samples, d0, m5,
                       baseline_method="D0_DIRECT_VLM", candidate_method="M5_FULL",
                       resamples=20, seed=20260717)
build_slices(samples, d0, m5,
             baseline_method="D0_DIRECT_VLM", candidate_method="M5_FULL")
```

Expected values must match a hand-computed four-sample fixture.

- [x] **Step 2: Write failing multi-method attribution tests**

Create exact predictions for D0, B1, M1–M5 and diagnostics for M1–M5. Assert:

```python
assert report["primary"]["candidate"] == "M5_FULL"
assert report["sequential_attribution"]["M2_CHECKLIST-M1_GRAPH"]["changed"] == 2
assert report["module_availability"]["video_science"]["available"] == 4
assert report["hard_override"]["forced_violation"] == 1
assert report["material_decision"]["gates"]["macro_f1_delta"]["threshold"] == 0.05
```

Reject missing/extra/duplicate prediction or diagnostic keys, non-finite numbers and diagnostics whose method/sample keys disagree with predictions.

- [x] **Step 3: Generalize helper signatures with old defaults**

Add keyword-only defaults to `paired_outcomes`, `action_group_bootstrap` and `build_slices`:

```python
*, baseline_method: str = _BASELINE_METHOD,
candidate_method: str = _CANDIDATE_METHOD
```

Use those names in `_prediction_index`; do not change default results or output shape.

- [x] **Step 4: Implement strict full-PAVG aggregation**

`build_full_pavg_report` validates exact full coverage for `D0_DIRECT_VLM`, `B1_RULE`, `D1_STRUCTURED_VLM`, `M1_GRAPH`, `M2_CHECKLIST`, `M3_MECHANICS`, `M4_VLM`, `M5_FULL`; diagnostics are exact for M1–M5 only. Compute per-method strict metrics, D0→M5 primary bootstrap/slices/gates, the five sequential transitions, module availability, model-call/cache-hit/latency summaries, hard overrides and provider failures. Prompt diagnostic300 validates `M5_FULL`, `M5_SHUFFLED_PROMPT_300`, `M5_ORACLE_PLAN_300` on the same 300 sample IDs and reports both paired differences as diagnostic only.

- [x] **Step 5: Implement immutable atomic report publication**

The CLI writes exactly:

```text
artifact_audit.json
merged_diagnostics.jsonl
merged_predictions.jsonl
module_attribution.json
paired_outcomes.json
prompt_diagnostics.json
slices.json
summary.json
summary.md
```

It captures input bytes before parsing, hashes every input/output, writes into a sibling staging directory, fsyncs files, atomically renames the directory, and refuses to replace a different existing bundle. Regenerating to a second path must be byte-identical.

- [x] **Step 6: Run focused, legacy and complete tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/benchmarking/test_full_report.py tests/benchmarking/test_full_pavg_report.py tests/benchmarking/test_full_pavg_report_cli.py -q
.\.venv\Scripts\python.exe -m pytest --basetemp outputs/.pytest-full-pavg -q
```

Expected: PASS; the accepted `report_full_video_benchmark` tests remain byte-compatible.

- [x] **Step 7: Commit reporting support**

```powershell
git add src/pavg_critic/benchmarking/full_report.py src/pavg_critic/benchmarking/full_pavg_report.py benchmarks/report_full_pavg_critic.py tests/benchmarking/test_full_report.py tests/benchmarking/test_full_pavg_report.py tests/benchmarking/test_full_pavg_report_cli.py
git commit -m "feat: report complete PAVG critic attribution"
```

### Task 8: Add non-secret run supervision and stall detection

**Files:**
- Create: `benchmarks/monitor_video_benchmark.py`
- Create: `tests/benchmarking/test_monitor_video_benchmark.py`

- [x] **Step 1: Write failing heartbeat/stall tests**

Use a fake `nvidia-smi` runner and fixed clock. Assert the monitor appends one JSON object containing timestamp, host, method counts, failure count, endpoint health, GPU utilization/memory and ETA; it must set `stalled=true` only when prediction count is unchanged for 15 minutes while expected keys remain.

```python
snapshot = build_snapshot(run_specs, expected_keys=300, previous=previous,
                          now=1_000.0, gpu_query=fake_gpu, endpoint_probe=fake_http)
assert snapshot["secrets_recorded"] is False
assert snapshot["prediction_count"] == 20
assert snapshot["stalled"] is False
```

- [x] **Step 2: Run the test and verify import failure**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/benchmarking/test_monitor_video_benchmark.py -q
```

Expected: FAIL with missing module.

- [x] **Step 3: Implement the read-only monitor**

CLI arguments are repeatable `--run METHOD=PATH`, `--expected-per-method`, `--endpoint`, `--heartbeat`, `--interval-sec` (default 300), `--stall-sec` (default 900) and `--once`. Read only JSONL line/key/failure counts; query `nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits`; probe `/health`; never inspect process command lines, environment or provider bodies. Append heartbeat via one UTF-8 JSON line and fsync.

- [x] **Step 4: Run focused and complete tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/benchmarking/test_monitor_video_benchmark.py -q
.\.venv\Scripts\python.exe -m pytest --basetemp outputs/.pytest-full-pavg -q
```

Expected: PASS.

- [x] **Step 5: Commit supervision support**

```powershell
git add benchmarks/monitor_video_benchmark.py tests/benchmarking/test_monitor_video_benchmark.py
git commit -m "feat: monitor benchmark progress without secrets"
```

### Task 9: Review, transfer and verify the implementation remotely

**Files:**
- Create remote: `/root/pavg-benchmark/artifacts/full-pavg-source-manifest.json`
- Create remote: `/root/pavg-benchmark-shard2/artifacts/full-pavg-source-manifest.json`
- Modify: this plan under `Execution results`

- [ ] **Step 1: Run implementation review and local verification**

Run `git diff --check`, unresolved-marker/secret scans over changed files, and the complete suite. Verify `git status --short` contains no unrelated paths.

- [ ] **Step 2: Create a clean git bundle and checksum**

```powershell
git bundle create outputs/full-pavg-eval.bundle HEAD
Get-FileHash outputs/full-pavg-eval.bundle -Algorithm SHA256 | Format-List
```

Expected: bundle verifies with `git bundle verify`; no overlay, `.env`, video, model or output artifact is included.

- [ ] **Step 3: Transfer to both existing project roots**

Use the dedicated ED25519 key and pinned known-host files. Transfer the bundle and checksum, fetch the bundle into each existing `src` checkout, and detach at the exact local HEAD. Do not delete or overwrite the accepted run directories.

- [ ] **Step 4: Verify both environments and model snapshots**

On each host record Python, package freeze hash, git tree, model snapshot hash file, endpoint health and GPU state. Require the intended `vLLM 0.11.0`, `transformers 4.57.0`, CUDA-working Torch and the same Qwen snapshot hashes.

- [ ] **Step 5: Run both complete remote test suites**

Run from the respective source roots with their existing Python 3.12 environments and ignored basetemp directories.

Expected: both hosts report the same all-pass test count as local.

### Task 10: Prove exact cache ownership and run smoke20

**Files:**
- Create remote: `/root/pavg-benchmark/runs/videophy2-full-pavg-qwen3vl8b/ownership-audit.json`
- Create remote: `/root/pavg-benchmark-shard2/runs/videophy2-full-pavg-qwen3vl8b/ownership-audit.json`
- Create remote: `/root/pavg-benchmark/runs/videophy2-full-pavg-qwen3vl8b/smoke20/`
- Create remote: `/root/pavg-benchmark-shard2/runs/videophy2-full-pavg-qwen3vl8b/smoke20/`
- Modify: this plan under `Execution results`

- [ ] **Step 1: Build owner manifests from existing cache IDs**

For each full-manifest sample, assign it to the host containing its accepted observation JSON. If both contain an identical cache, apply the frozen provenance rule (cloud2 owns the 32 recorded fallback IDs). Refuse missing or conflicting cache content.

- [ ] **Step 2: Audit exact ownership**

Require cloud2 `1,731`, cloud1 `1,666`, intersection `0`, union `3,397`. Write sorted sample IDs, source observation checksum and owner-manifest checksum to each `ownership-audit.json`.

- [ ] **Step 3: Start one existing Qwen endpoint per server and the 5-minute monitor**

Use the already verified 58% GPU-memory configuration, 16-image limit, 16,384-token context and strict JSON schema. Start under fixed tmux names `pavg-qwen` and `pavg-monitor`; refuse duplicate sessions. The monitor heartbeat remains below the new run root.

- [ ] **Step 4: Run smoke20 D1 and M1–M5**

Run CPU M1–M3 concurrently with three model clients D1/M4/M5. Use method-specific run directories, one prediction writer each, the shared stage cache and `--max-new-failures 1`.

- [ ] **Step 5: Apply smoke gates**

Require exact 20 predictions per method, exact 20 diagnostics per M1–M5, no OOM, no duplicate keys, no provider failure, all five M5 evidence families represented in diagnostics, and non-zero Planner/PQSG model calls. Run a secret scan before continuing.

### Task 11: Run and gate the frozen pilot300

**Files:**
- Create remote: `/root/pavg-benchmark/runs/videophy2-full-pavg-qwen3vl8b/pilot300/`
- Create remote: `/root/pavg-benchmark-shard2/runs/videophy2-full-pavg-qwen3vl8b/pilot300/`
- Modify: this plan under `Execution results`

- [ ] **Step 1: Launch owner-filtered pilot300 jobs**

Use the frozen pilot manifest intersected with each ownership manifest. Run D1, M1–M5 with the same commands/config/cache namespaces as smoke; do not change prompts or thresholds.

- [ ] **Step 2: Supervise every 15 minutes**

Check endpoint health, tmux sessions, heartbeat freshness, per-method prediction/diagnostic counts, failures, GPU utilization/memory, disk free space, throughput and ETA. If a client dies, inspect only non-secret logs and resume the exact command. If the endpoint dies, stop clients before restarting the exact frozen service command; never let clients accumulate terminal failures against a dead endpoint.

- [ ] **Step 3: Apply pilot gates**

Require M5 failure rate `< 5%`, no OOM, all stage call/cache telemetry present, exact owner-union coverage and no duplicate keys. Project full wall time: `<=24 h` proceed normally; `24–30 h` record extended ETA and proceed unchanged; `>30 h` stop before full launch.

- [ ] **Step 4: Freeze the primary configuration**

Hash both resolved configs, all prompt/schema source strings, model snapshot manifest, ownership manifests and smoke/pilot artifacts. Mark the hash set immutable; later diagnostic runs must reference it.

### Task 12: Run the full 3,397-sample two-server matrix

**Files:**
- Create remote: `/root/pavg-benchmark/runs/videophy2-full-pavg-qwen3vl8b/full/`
- Create remote: `/root/pavg-benchmark-shard2/runs/videophy2-full-pavg-qwen3vl8b/full/`
- Modify: this plan under `Execution results`

- [ ] **Step 1: Launch CPU and model work without duplicate computation**

On both hosts start M1–M3 CPU/cache jobs and D1/M4/M5 model clients in separate tmux sessions. Each process reads only its owner manifest, shares immutable observation/model stage caches, and writes its own predictions/diagnostics/lock files.

- [ ] **Step 2: Maintain the supervision contract**

Poll at least every 15 minutes and append a user-facing checkpoint at least every 30 minutes while this session is active. A healthy run requires heartbeat age `<10 min`, prediction progress within `15 min` unless an individual M5 sample is still producing model calls, endpoint HTTP health, free disk `>20 GiB`, and GPU memory below the measured no-OOM ceiling.

- [ ] **Step 3: Use finite recovery only**

For a dead evaluator, remove no lock until PID absence is proven, then resume the exact command. For endpoint failure, pause clients, restart the frozen endpoint at most twice, health-check it, then resume. For repeated endpoint failure or a third OOM, stop and report; do not reduce frames, context, model size, method membership or thresholds.

- [ ] **Step 4: Require complete terminal coverage**

Before diagnostics, require 3,397 predictions for each of D1/M1/M2/M3/M4/M5; require 3,397 matching diagnostics for each M1–M5; require union/intersection and duplicate audits to pass. Failed predictions remain in the denominator and are never deleted/retried under a changed config.

### Task 13: Run prompt diagnostics300 after primary freeze

**Files:**
- Create remote: `/root/pavg-benchmark/runs/videophy2-full-pavg-qwen3vl8b/diagnostic300/`
- Create remote: `/root/pavg-benchmark-shard2/runs/videophy2-full-pavg-qwen3vl8b/diagnostic300/`
- Modify: this plan under `Execution results`

- [ ] **Step 1: Materialize correct-prompt M5 subset**

Extract the 300 primary M5 predictions/diagnostics by exact frozen pilot IDs; do not infer them again. Require exact sample key alignment.

- [ ] **Step 2: Run shuffled-prompt M5**

Use `videophy2_pilot300_shuffled.json` and output method `M5_SHUFFLED_PROMPT_300`. Record the donor-map hash in resolved config and reuse only cache entries whose complete stage key matches.

- [ ] **Step 3: Run oracle-plan M5**

Use the normal pilot manifest and output method `M5_ORACLE_PLAN_300`. The Planner model call must be a hit on the primary `planner` cache for each sample; only the deterministic oracle append differs. Human rules never enter PQSG/verifier inputs except through this explicit plan.

- [ ] **Step 4: Audit diagnostic isolation**

Require 300 predictions and diagnostics for all three views, zero key differences, no label/rule leakage into correct/shuffled model inputs, and exact oracle constraint counts matching non-empty `physical_rules`.

### Task 14: Synchronize, report, audit and publish

**Files:**
- Create: `outputs/benchmarks/videophy2-full-pavg-qwen3vl8b/`
- Modify: `docs/results/criticbenchmark.md`
- Modify: this plan under `Execution results`

- [ ] **Step 1: Synchronize only non-secret artifacts**

Copy owner manifests, resolved configs, predictions, diagnostics, heartbeat summaries and non-secret logs. Do not copy videos, model weights, response-cache bodies, image data or `.env`.

- [ ] **Step 2: Generate the report twice**

Run `python -m benchmarks.report_full_pavg_critic` with the accepted D0/B1 inputs, new D1/M1–M5 shards, diagnostics and prompt diagnostic inputs. Generate once to the final path and once to a temporary sibling; require identical SHA-256 for all nine core files, then remove only the verified temporary sibling.

- [ ] **Step 3: Apply the frozen verdict arithmetic**

Report M5−D0 Macro-F1, 2,000 action-group bootstrap CI, both recalls, failure delta and positive-generator count. Report M1−B1 through M5−M4, prompt shuffle/oracle differences and hard-override counts as attribution only. Do not call the framework supported unless every primary gate passes.

- [ ] **Step 4: Run secret and artifact audits**

Scan synchronized artifacts for both SSH credential strings previously supplied by the user, API-key prefixes, `Authorization`, `BENCH_API_KEY`, `.env` values, `data:image`, raw provider bodies and model cache response fields. Require zero hits outside explicit redaction-test fixtures. Verify 3,397×8 primary prediction keys, 3,397×5 diagnostic keys and 300×3 prompt-view keys.

- [ ] **Step 5: Run clean-room local verification**

Run the complete pytest suite, regenerate the report from a fresh temporary directory and compare hashes. Check `git diff --check` and ensure accepted `outputs/benchmarks/videophy2-full-qwen3vl8b/` hashes are unchanged.

- [ ] **Step 6: Update the Chinese result narrative**

Append exact metrics, CIs, module availability, prompt attribution, runtime, GPU utilization, failures and limitations to `docs/results/criticbenchmark.md`. Explicitly distinguish “complete Critic on VideoPhy-2” from the unevaluated Generator/Repairer/Selector loop and deferred VideoPhy-1 OOD.

- [ ] **Step 7: Commit and push only approved source/results**

```powershell
git add src tests benchmarks evaluation/manifests docs/superpowers/plans/2026-07-17-full-pavg-critic-evaluation.md docs/results/criticbenchmark.md outputs/benchmarks/videophy2-full-pavg-qwen3vl8b
git diff --cached --check
git commit -m "results: evaluate complete PAVG critic on VideoPhy2"
git push origin sy
```

Expected: source/tests/manifests/documentation and the compact non-secret report bundle only; no unrelated user file is staged.

## Execution results

Append one immutable checkpoint sequentially named `E1` through `E14` after every task. Each entry records timestamp, source commit, exact command/config hashes, test counts, sample/method counts, failures, GPU/throughput/ETA and any finite recovery. Never rewrite a prior checkpoint.

### E1 — Approved inputs and local baseline

- Timestamp: `2026-07-17T15:17:51+08:00`; starting source commit `271499da038470f52ef6cbe15d23a59164a4d58b` on branch `sy`.
- The user previously selected direct execution in the current workspace. Worktree detection confirmed a normal checkout (`GIT_DIR == GIT_COMMON`) rather than main/master; only this turn's approved spec/plan files were dirty before synchronization.
- Full VideoPhy-2 manifest SHA-256 verified as `d8be5fe97ddf6902515c09ccbb53f394b25230213db7c3058d61f84748624906`.
- The frozen pilot manifest was uniquely located by hash at `/root/pavg-benchmark/runs/videophy2-pilot300-qwen3vl8b/manifest.json`, copied with key-only SSH, and verified locally as `a97762fe4033789eb14a82717c72c14e89bc75a7a67200d5890ff1647f72a670`. No sample was regenerated or replaced.
- Local Python 3.12 baseline: `281 passed in 12.13s` using ignored basetemp `outputs/.pytest-full-pavg`.

### E2 — Typed pipeline audit boundary

- Timestamp: `2026-07-17T15:25:39+08:00`; implementation started from commit `3443afa601c972308af700e2d5255d8314bc80e6`.
- Plan review found that the proposed `tests/test_pipeline.py` path did not exist. It was corrected before production edits to the focused new file `tests/test_pipeline_artifacts.py`; production scope remained `schemas.py` and `pipeline.py` only.
- RED evidence: the focused test failed with `AttributeError: 'CriticArtifacts' object has no attribute 'keyframes'`.
- GREEN evidence: the focused audit test passed `1/1`; the complete local suite passed `282/282` in 14.38 seconds.
- `CriticArtifacts` now exposes typed keyframe/review maps, and reports record the pre-evidence-fusion public decision fields plus the deterministic hard-violation override flag. No weights, thresholds or decisions were tuned.

### E3 — Stage-separated model response cache

- Timestamp: `2026-07-17T15:36:51+08:00`; implementation started from commit `3aa741d13eb9b8a83757dfb423139871f7c78d2a`.
- RED evidence: the new focused suite failed at collection with `ModuleNotFoundError: No module named 'pavg_critic.benchmarking.model_cache'` before production code existed.
- The first GREEN attempt without a repository basetemp hit the known Windows global pytest-temp ACL failure. With the correct repository basetemp, a second issue caused the test host to lose stdout. Systematic isolation traced this to POSIX-style `os.kill(pid, 0)` being unsafe for liveness checks on Windows.
- A single platform-boundary fix uses Windows `OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)` and retains `os.kill(pid, 0)` on POSIX. The existing concurrent-cache test is the regression reproducer.
- Focused cache tests passed `7/7` in 0.37 seconds; the complete local suite passed `289/289` in 15.25 seconds.
- Cache identity includes namespace, model ID, prompt/schema hashes and ordered image-evidence hashes. Provider failures are retried up to three attempts but never cached; per-key locks prevent duplicate concurrent provider calls; telemetry contains hashes/latency only and no prompt or image bytes.

### E4 — Real-video M5, oracle plan and module diagnostics

- Timestamp: `2026-07-17T15:53:24+08:00`; implementation started from commit `20336f4309b1c9bf2e9e51cbf3ce0e19ca2c4108`.
- RED evidence: the focused suite failed at collection because `pavg_diagnostics` and `prompt_diagnostics` did not exist. After the initial GREEN pass, an additional leakage test failed because the legacy prediction failure stored `Authorization: Bearer secret-value`; the minimal fix retains only the exception type.
- `M5_FULL` now requires and injects explicit Planner, question/PQSG and grouped-verifier models. The same cached backend exposes the frozen model ID to Planner metadata while keeping the provider object private.
- `OracleRulePhysicsPlanner` preserves the normal model plan, adds exact stable `oracle-rule-{index}` constraints, uses a synthetic `scene` only for an object-free plan, and terminates on ID collision.
- Each PAVG evaluation now returns a same-key diagnostic record with Planner/PQSG/VideoScience/mechanics/rule/VLM/evidence-family availability, configured/effective weights, pre/final fusion state, hard override, non-secret stage events and latency. Failures never store exception messages.
- Focused Task 4 tests passed `14/14` in 1.40 seconds; the complete local suite passed `296/296` in 10.20 seconds. No fusion weight, prompt, threshold or accepted prediction changed.

### E5 — Crash-recoverable paired output and bounded failures

- Timestamp: `2026-07-17T16:02:04+08:00`; implementation started from commit `12be12d440120eda7a3335b38c7e3e32af971901`.
- RED evidence: focused collection failed with `ModuleNotFoundError: No module named 'pavg_critic.benchmarking.audited_runner'` before implementation.
- `AuditedBenchmarkRunner` now writes one fsynced pending transaction before appending prediction and diagnostics. Tests inject crashes after the prediction append and after both appends; restart fills only the missing side, clears pending state and never re-invokes the method.
- Existing unrecoverable asymmetric/duplicate keys remain terminal. Prediction and diagnostics keys are validated before writing, and one exclusive writer lock protects the pair.
- Both ordinary and audited runners now stop after fsyncing the configured number of new terminal failures. The default legacy ordinary runner remains unlimited; the audited runner defaults to one failure.
- Focused runner tests passed `13/13` in 0.26 seconds; the complete local suite passed `303/303` in 10.46 seconds.

### E6 — M5 CLI and frozen prompt diagnostic inputs

- Timestamp: `2026-07-17T16:14:37+08:00`; implementation started from commit `2eb6f08953c77797a8c2612569c6b8d993d89210`.
- RED evidence: focused collection failed with `ModuleNotFoundError: No module named 'benchmarks.build_prompt_diagnostics'` before the builder existed.
- The CLI now exposes `M5_FULL`, `M5_SHUFFLED_PROMPT_300` and `M5_ORACLE_PLAN_300`, requires an explicit model-cache directory for M4/M5, records the stage namespaces and uses a one-new-failure default. Direct methods retain the ordinary runner; PAVG methods use paired audited output.
- The deterministic label-blind bipartite matcher produced 300 unique prompt donors with zero same-sample, same-exact-prompt or same-action matches. Shuffled manifest SHA-256: `5250aea3077f9360e42e20008ee8873a9d9a5f3284e7b52270cba33b098e5848`; donor-map SHA-256: `c43ae712a41513e0443233bf400d0f2d976846cc31beb6a858d0a91300f46049`.
- Focused Task 6 tests passed `13/13`. The first complete-suite attempt had one transient Windows `PermissionError` atomically renaming an old report-test directory; that exact test passed alone in a fresh basetemp, and the complete suite then passed `307/307` in 10.48 seconds in a fresh basetemp. No production change was made for the transient lock.

### E7 — Complete-PAVG statistics and immutable report bundle

- Timestamp: `2026-07-17T16:30:21+08:00`; implementation started from commit `f50d85c4892498a1aa826bda816c98e01deaebaa`.
- RED evidence: focused collection failed because both `full_pavg_report` and `report_full_pavg_critic` were absent.
- Existing paired/bootstrap/slice helpers now accept explicit method IDs but retain D0/B1 defaults. The complete legacy report suite remained green.
- The new aggregator requires exact coverage for eight methods and exact sidecars for M1–M5, computes D0→M5 strict metrics/gates, all five sequential transitions, M5 module availability, stage calls, provider failures, hard overrides and diagnostic-only correct/shuffled/oracle comparisons.
- The new CLI freezes and re-verifies input bytes, emits exactly nine core files through an atomic directory publication, refuses a different existing bundle, and reproduced byte-identical files in a second output directory.
- Focused report tests passed `84/84` in 0.78 seconds; the complete local suite passed `311/311` in 8.59 seconds.

### E8 — Non-secret progress supervision

- Timestamp: `2026-07-17T16:37:39+08:00`; implementation started from commit `ac9b1825c6fa9d66ecda7ca3ce80dda835106c2c`.
- RED evidence: focused collection failed with `ModuleNotFoundError: No module named 'benchmarks.monitor_video_benchmark'`.
- The monitor appends one fsynced heartbeat every configured interval (300 seconds remotely), counting exact prediction keys/failures per method and recording endpoint health, GPU utilization/memory, last progress time and short-window ETA.
- A run is marked stalled only when expected keys remain and no count progress occurs for 900 seconds. The monitor reads no process command lines, environment variables, prompts, images, diagnostics or provider bodies and writes `secrets_recorded=false` explicitly.
- Focused monitor tests passed `3/3` in 0.07 seconds; the complete local suite passed `314/314` in 8.64 seconds.

### E9 — Two-A100 open-model execution and cache recovery

- Timestamp: `2026-07-19T06:33:18+08:00`; source head on both new servers is the pushed `sy` commit `27ed54d6be15463eb473d705428b433f882f517a` (report-failure rendering fix); remote full suites passed `387/387` on each host before launch.
- The production run uses only local `Qwen/Qwen3-VL-8B-Instruct` through vLLM `0.11.0`, deterministic JSON-schema decoding, the frozen 16-frame policy, and official SAM2.1 Hiera B+ caches/propagation. Every row is from the frozen prompt-bearing VideoPhy-2 manifest; no closed model or GPT API is involved.
- cloud2 is evaluating its fixed 1,731-row owner (M1–M3 complete: `1,731` each; D1 in progress: `697` at this checkpoint) while restoring the fixed 833-row B-even observation recovery shard (`336` records). cloud1 is restoring its fixed 833-row B-odd recovery shard (`488` records); its three terminal failures are all `VLM produced no object seeds for SAM2 tracking` on blank-first-frame videos and remain explicitly journaled.
- Both vLLM endpoints returned HTTP `200`; GPU utilization was `91%` on cloud2 and `93%` on cloud1, with approximately `25.9GB` and `24.4GB` of `40GB` resident respectively. No OOM or process exit occurred. Recovery caches will be transferred only after the immutable recovery manifest reaches its terminal count, then cloud1 will launch the remaining fixed owner shard; no sample membership, prompt, threshold or weight is being changed.

### E10 — cloud1 recovery terminal and odd-owner launch

- Timestamp: `2026-07-19T08:39:59+08:00`; cloud1 recovery manifest reached its exact terminal count `833/833` with `5` explicit no-seed failures. The watcher launched the fixed odd-owner runs without changing the manifest: CPU M1/M2/M3 were at `557/833` each and open-model D1 was at `44/833` at the checkpoint.
- cloud2 D1 reached its exact owner count `1,731/1,731`; M4/M5 had started at `24/24` records each while the independent fixed B-even SAM2 recovery reached `550/833` with `4` explicit no-seed failures.
- Both endpoints remained HTTP `200`; GPU utilization was `93%` on each host and no OOM, duplicate key or process restart occurred. The next synchronization gate is cloud2 recovery terminal count, after which only successful immutable observation pairs will be relayed to cloud1 for the remaining fixed owner shard.

### E11 — Exact resume after method-level failure budget

- Timestamp: `2026-07-19T08:40:00+08:00`; the cloud1 odd CPU run stopped at M1/M2/M3 counts `557/556/556`. The terminal log showed `RuntimeError: new failure budget reached`; its 10 new failures were the expected missing-observation records repeated across methods, not a code, model or endpoint failure.
- The identical command was relaunched with the unchanged `--max-new-failures 10`. Because the runner budgets only failures newly appended by the current invocation, the restart completed the remaining immutable keys without changing any threshold or method setting.
- Terminal integrity: M1/M2/M3 each `833/833`; `2,499` predictions and `2,499` diagnostics; `2,499` unique keys on each side; exact key equality; 15 explicit failures (`5` no-seed samples × `3` methods); zero duplicate or asymmetric keys.
