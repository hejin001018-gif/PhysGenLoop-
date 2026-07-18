# Critic Execution Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add safe, real per-node Critic tracing, independent fusion validation, a trace CLI and four-video acceptance verification without changing evaluation decisions.

**Architecture:** A new `TraceRecorder` observes real pipeline and PQSG node boundaries through optional hooks that default to disabled. It stores only bounded domain summaries, exposes the complete weight arithmetic, writes an atomic versioned JSON document and feeds a standalone validator that independently checks graph, privacy and fusion invariants. The existing result JSON and all untraced callers retain their current behavior.

**Tech Stack:** Python 3.12, frozen dataclasses, `contextlib`, `hashlib`, `json`, `jsonschema`, `pytest`, existing PAVG Critic/SAM2/OpenAI-compatible adapters.

---

## File structure

- Create `src/pavg_critic/execution_trace.py`: recorder, safe bounded serialization, fusion audit, atomic writer and validator.
- Modify `src/pavg_critic/question_executor.py`: optional per-PQSG-node observer.
- Modify `src/pavg_critic/pipeline.py`: optional trace hooks around real stage boundaries.
- Modify `src/pavg_critic/__init__.py`: export stable tracing types used by examples.
- Modify `examples/evaluate_video.py`: trace flags, live renderer, recorder metadata and partial-trace persistence.
- Create `examples/validate_pipeline_trace.py`: independent validation CLI.
- Create `schemas/critic_trace.schema.json`: versioned structural schema.
- Create `tests/test_execution_trace.py`: recorder, safety, fusion arithmetic and validator unit tests.
- Create `tests/test_pipeline_trace.py`: deterministic pipeline and PQSG-node trace integration tests.
- Modify `tests/test_evaluate_video_example.py`: parser, output and failure persistence tests.
- Modify `schemas/README.md`: document the trace artifact and validation commands.

## Task 1: Establish the trace record and privacy contract

**Files:**
- Create: `src/pavg_critic/execution_trace.py`
- Create: `tests/test_execution_trace.py`

- [ ] **Step 1: Write failing recorder lifecycle tests**

Add tests that express the intended public API before the module exists:

```python
from pavg_critic.execution_trace import TraceRecorder, TraceSafetyError


def test_recorder_preserves_order_status_dependencies_and_elapsed_time():
    emitted = []
    recorder = TraceRecorder(on_record=emitted.append)
    recorder.record_completed(
        "request", label="输入请求", source_nodes=(),
        inputs={"prompt": "石头滚下坡"}, outputs={"accepted": True},
        elapsed_ms=0.0,
    )
    recorder.record_skipped(
        "mechanics", label="力学", source_nodes=("event_detection",),
        inputs={"event_count": 0}, reason="not_applicable",
    )

    document = recorder.to_dict()

    assert document["schema_version"] == "pavg-critic-trace/v1"
    assert [node["sequence"] for node in document["nodes"]] == [1, 2]
    assert [node["status"] for node in document["nodes"]] == ["completed", "skipped"]
    assert emitted == document["nodes"]


def test_node_context_records_sanitized_error_and_reraises():
    recorder = TraceRecorder()
    with pytest.raises(RuntimeError, match="provider unavailable"):
        with recorder.node(
            "physics_planner", label="Planner", source_nodes=("request",),
            inputs={"prompt": "ball falls"},
        ):
            raise RuntimeError("provider unavailable")
    node = recorder.to_dict()["nodes"][0]
    assert node["status"] == "error"
    assert node["error"] == {
        "type": "RuntimeError", "message": "provider unavailable"
    }


@pytest.mark.parametrize("key", ["api_key", "authorization", "headers", "raw_response"])
def test_forbidden_trace_keys_are_rejected(key):
    recorder = TraceRecorder()
    with pytest.raises(TraceSafetyError, match="forbidden trace key"):
        recorder.record_completed(
            "unsafe", label="unsafe", source_nodes=(),
            inputs={key: "secret"}, outputs={}, elapsed_ms=0.0,
        )
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_execution_trace.py -q
```

Expected: collection fails with `ModuleNotFoundError: pavg_critic.execution_trace`.

- [ ] **Step 3: Implement the minimal recorder**

Create `execution_trace.py` with these stable types and operations:

```python
TRACE_SCHEMA_VERSION = "pavg-critic-trace/v1"
TRACE_STATUSES = frozenset({"completed", "skipped", "degraded", "error"})
FORBIDDEN_TRACE_KEYS = frozenset({
    "api_key", "authorization", "headers", "raw_response",
    "image", "image_bytes", "mask", "masks", "base64",
})

class TraceSafetyError(ValueError):
    pass

@dataclass(frozen=True)
class TraceNodeRecord:
    sequence: int
    node_id: str
    label: str
    status: str
    source_nodes: tuple[str, ...]
    elapsed_ms: float
    inputs: Mapping[str, object]
    outputs: Mapping[str, object]
    parent_id: str | None = None
    warnings: tuple[str, ...] = ()
    error: Mapping[str, str] | None = None

class TraceRecorder:
    """Own ordered records and expose completed/degraded/skipped/error writes."""
```

Implement `TraceRecorder.__init__`, `update_metadata`, `record_completed`, `record_degraded`, `record_skipped`, `node`, `set_outcome` and `to_dict` with the arguments exercised by the tests above. `node()` uses `perf_counter()` and a small context object with `complete()` and `degrade()` methods. If the body exits without either method it records an empty completed output; if it raises it records an error and re-raises. Every record is passed through `_sanitize_trace_value()`, which rejects forbidden keys/bytes, bounds strings to 2,000 characters, stores at most 20 collection previews and adds canonical SHA-256/count metadata for truncated collections. Every duplicate `node_id`, including a duplicated `pqsg_node.<id>`, raises `ValueError`.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the Task 1 command. Expected: all new lifecycle and safety tests pass.

- [ ] **Step 5: Commit the recorder contract**

```powershell
git add src/pavg_critic/execution_trace.py tests/test_execution_trace.py
git commit -m "feat: add safe critic trace recorder"
```

## Task 2: Make fusion arithmetic independently reproducible

**Files:**
- Modify: `src/pavg_critic/execution_trace.py`
- Modify: `tests/test_execution_trace.py`

- [ ] **Step 1: Write failing fusion-audit tests**

Use real `EvidenceBundle`, `CriticReport` and `FusionConfig` values. The primary example must reproduce the accepted `2n.mp4` arithmetic:

```python
def test_fusion_audit_recomputes_effective_weights_and_score():
    report = CriticReport(
        is_physical=True, decision="physical", physics_score=0.914894,
        confidence=0.282, coverage=0.4,
        evidence_bundles=(
            EvidenceBundle(
                family="rules", source="deterministic_rules", status="available",
                score=1.0, coverage=0.8, confidence=0.75,
            ),
            EvidenceBundle(
                family="checklist", source="video_science_checklist",
                status="available", score=2 / 3, coverage=0.6, confidence=0.6,
            ),
            _missing_bundle("pqsg"), _missing_bundle("mechanics"),
            _missing_bundle("vlm"),
        ),
    )

    audit = build_fusion_audit(CriticConfig().fusion, report)

    families = {row["family"]: row for row in audit["families"]}
    assert families["rules"]["effective_weight"] == pytest.approx(0.21)
    assert families["checklist"]["effective_weight"] == pytest.approx(0.072)
    assert audit["score_before_hard_violation"] == pytest.approx(0.914893617)
    assert audit["final_score"] == pytest.approx(report.physics_score)


def test_validator_fails_when_effective_weight_is_tampered():
    trace = _valid_trace_document()
    trace["nodes"][-2]["outputs"]["families"][0]["effective_weight"] += 0.1
    validation = validate_trace(trace)
    assert not validation.passed
    assert any(check.code == "fusion.effective_weight" for check in validation.checks)
```

Add tests for zero effective weight on unavailable families, score capping when a retained hard violation exists, weighted coverage, final confidence, decision thresholds, and rejected/uncertain reviews leaking into final violations.

- [ ] **Step 2: Run the focused tests and verify RED**

Expected: imports or assertions fail because `build_fusion_audit()` and `validate_trace()` do not exist.

- [ ] **Step 3: Implement fusion audit and validation reports**

Add:

```python
@dataclass(frozen=True)
class TraceValidationPolicy:
    require_sam2: bool = False
    require_model_planner: bool = False
    fail_on_provider_fallback: bool = False

@dataclass(frozen=True)
class TraceValidationCheck:
    code: str
    level: str
    passed: bool
    message: str

@dataclass(frozen=True)
class TraceValidationReport:
    checks: tuple[TraceValidationCheck, ...]
    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks if check.level == "error")
```

Implement `build_fusion_audit(config: FusionConfig, report: CriticReport) -> dict[str, object]` to return every configured/effective weight and the final decision arithmetic. Implement `validate_trace(document: Mapping[str, object], *, policy: TraceValidationPolicy = TraceValidationPolicy(), tolerance: float = 1e-6) -> TraceValidationReport` to independently validate structure, dependencies, privacy and fusion math.

The audit uses exactly:

```python
effective_weight = configured_weight * coverage * confidence
weighted_contribution = score * effective_weight
score_before_hard = sum(contributions) / sum(effective_weights)
weighted_coverage = sum(configured_weight * coverage) / sum(configured_weights)
confidence = sum(effective_weights) / sum(configured_weights)
```

If `report.violations` is non-empty, record `hard_violation=True` and cap the pre-hard score to the final report score. Validation recalculates instead of trusting stored derived values and creates stable codes for every failure.

- [ ] **Step 4: Verify GREEN and commit**

Run the focused file, then:

```powershell
git add src/pavg_critic/execution_trace.py tests/test_execution_trace.py
git commit -m "feat: validate critic fusion traces"
```

## Task 3: Observe every PQSG node without changing graph semantics

**Files:**
- Modify: `src/pavg_critic/question_executor.py`
- Create: `tests/test_pipeline_trace.py`

- [ ] **Step 1: Write a failing observer test**

Build a two-node object→action graph and real `QuestionExecutionContext`, then assert the observer receives both results in topological order, their parent-result summaries and non-negative elapsed milliseconds. Add a failure case where a subclass raises from `_verify()` and the observer receives `result=None`, the exception type and elapsed time before the exception is re-raised.

The desired callback contract is:

```python
NodeExecutionObserver = Callable[
    [QuestionNode, Mapping[str, NodeResult], NodeResult | None,
     BaseException | None, float],
    None,
]

events = []

def observe(node, parent_results, result, error, elapsed_ms):
    events.append((node, parent_results, result, error, elapsed_ms))

results = executor.execute(graph, context, node_observer=observe)
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_pipeline_trace.py -q
```

Expected: `execute()` rejects `node_observer`.

- [ ] **Step 3: Add the optional observer**

Extend `execute()` with a keyword-only `node_observer=None`. For every node, collect only its declared parent results, measure `_verify()` or dependency blocking with `perf_counter()`, invoke the observer once on success and once with the exception on failure, then preserve the current return value and exception behavior. No observer means no callbacks and identical execution.

- [ ] **Step 4: Verify old and new executor behavior**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_pipeline_trace.py tests/test_pqsg.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add src/pavg_critic/question_executor.py tests/test_pipeline_trace.py
git commit -m "feat: expose pqsg node observations"
```

## Task 4: Instrument actual Critic stage boundaries

**Files:**
- Modify: `src/pavg_critic/pipeline.py`
- Modify: `src/pavg_critic/execution_trace.py`
- Modify: `src/pavg_critic/__init__.py`
- Modify: `tests/test_pipeline_trace.py`

- [ ] **Step 1: Write failing full-stage trace tests**

Use external deterministic `FrameState` observations and fixed Planner/question/VLM components so no network or video decoder is required. Assert:

```python
expected = [
    "request", "physics_planner", "question_graph", "video_observation",
    "trajectory", "event_detection", "mechanics", "rule_engine",
    "temporal_localization", "visual_evidence", "checklist",
    "keyframe_selection", "pqsg_execution", "vlm_verification",
    "candidate_fusion", "question_scoring", "evidence_fusion", "final_report",
]
assert [node["node_id"] for node in trace["nodes"] if not node["node_id"].startswith("pqsg_node.")] == expected
assert all(node["inputs"] is not None and node["outputs"] is not None for node in trace["nodes"])
assert trace["nodes"][-2]["outputs"]["families"]
```

Also assert disabled modules emit `skipped`, Planner/PQSG fallback emits `degraded`, a provider review failure is attached to `vlm_verification`, and tracing does not change `artifacts.report.to_dict()` compared with an untraced run.

- [ ] **Step 2: Run the integration tests and verify RED**

Expected: `PhysicsCritic.analyze_detailed()` rejects `trace=`.

- [ ] **Step 3: Add summary helpers**

In `execution_trace.py`, add deterministic bounded helpers for request, plan, graph, states, tracks, events, mechanics, candidates, checklist, keyframes, node results, reviews and final report. Every helper returns JSON-safe domain fields, counts and a bounded preview; no helper accepts images or masks.

- [ ] **Step 4: Add optional hooks in `analyze_detailed()`**

Extend the signature without breaking callers:

```python
def analyze_detailed(
    self,
    request: CriticRequest,
    *,
    observations: Iterable[FrameState] | None = None,
    floor_y: float | None = None,
    trace: TraceRecorder | None = None,
) -> CriticArtifacts:
```

Wrap each fixed stage in the Task 4 expected order. For external observations, `video_observation` completes with `backend="provided_observations"`; for real video it records the detector class and observation summary. Use the Task 3 observer to create `pqsg_node.<id>` child records with `parent_id="pqsg_execution"`. Record disabled stages explicitly as skipped. Build the `evidence_fusion` output with `build_fusion_audit()` after the real fusion call.

- [ ] **Step 5: Export `TraceRecorder` and validation types**

Add the stable tracing classes/functions to `pavg_critic.__all__`, leaving package version unchanged because this is an unreleased branch feature.

- [ ] **Step 6: Verify report identity and commit**

Run pipeline, evidence-fusion, question and trace tests. Commit only the listed files:

```powershell
git add src/pavg_critic/pipeline.py src/pavg_critic/execution_trace.py src/pavg_critic/question_executor.py src/pavg_critic/__init__.py tests/test_pipeline_trace.py
git commit -m "feat: trace critic pipeline stages"
```

## Task 5: Add CLI collection, atomic persistence and validation

**Files:**
- Modify: `examples/evaluate_video.py`
- Create: `examples/validate_pipeline_trace.py`
- Modify: `tests/test_evaluate_video_example.py`
- Modify: `tests/test_execution_trace.py`

- [ ] **Step 1: Write failing parser and atomic-writer tests**

Assert `build_parser()` accepts `--trace` and `--trace-output`, `evaluate_video(..., trace=recorder)` forwards the recorder, and `write_trace_atomic()` produces UTF-8 valid JSON without leaving its temporary sibling. Add a CLI test with a stubbed evaluation exception that verifies the partial trace file exists and the existing non-zero exit code remains unchanged.

- [ ] **Step 2: Verify RED**

Run the two focused files. Expected: missing CLI arguments and writer.

- [ ] **Step 3: Implement atomic persistence and live rendering**

Add to `execution_trace.py`:

```python
def write_trace_atomic(path: str | Path, document: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
```

Add the two CLI flags. `--trace` supplies a callback that prints sequence, node ID, Chinese label, status, elapsed time and compact count/score output to stderr. `--trace-output` enables the recorder even without live display. Populate recorder metadata with detector backend, SAM2 used flag, model names, module flags and safe video metadata. Write the trace in `finally` so error traces survive.

- [ ] **Step 4: Write and implement validator CLI tests**

Test `examples.validate_pipeline_trace.main(argv)` directly with a valid fixture, a tampered fusion trace and strict-policy failures. Implement arguments:

```text
trace_file
--require-sam2
--require-model-planner
--fail-on-provider-fallback
```

Print one `[PASS]`, `[WARN]` or `[FAIL]` line per check and a final Chinese summary. Return `0`, `1` or `2` exactly as frozen in the design.

- [ ] **Step 5: Verify CLI tests and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_execution_trace.py tests/test_pipeline_trace.py tests/test_evaluate_video_example.py -q
git add src/pavg_critic/execution_trace.py examples/evaluate_video.py examples/validate_pipeline_trace.py tests/test_execution_trace.py tests/test_evaluate_video_example.py
git commit -m "feat: add critic trace and validation cli"
```

## Task 6: Freeze the machine-readable schema and user documentation

**Files:**
- Create: `schemas/critic_trace.schema.json`
- Modify: `schemas/README.md`
- Modify: `tests/test_schemas.py`

- [ ] **Step 1: Write a failing schema-validation test**

Load a real `TraceRecorder.to_dict()` document, validate it with Draft 2020-12 `jsonschema`, then mutate the schema version, node status and sequence type to prove each is rejected.

- [ ] **Step 2: Verify RED**

Expected: schema file missing.

- [ ] **Step 3: Add the schema and commands**

Define required top-level fields `schema_version`, `metadata`, `nodes`, `outcome`, `warnings`; constrain the version constant and node terminal status enum; require all node fields from Section 4 of the design; permit domain-specific input/output objects but forbid unknown node-level fields. Document the evaluate/validate commands, exit codes and strict flags in `schemas/README.md`.

- [ ] **Step 4: Run schema and trace tests, then commit**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_schemas.py tests/test_execution_trace.py tests/test_pipeline_trace.py tests/test_evaluate_video_example.py -q
git add schemas/critic_trace.schema.json schemas/README.md tests/test_schemas.py
git commit -m "docs: define critic trace schema"
```

## Task 7: Run full regression and four-video acceptance

**Files:**
- Create ignored runtime artifacts under: `outputs/trace-validation/`
- Modify only if a regression is found: the source/test files already listed above

- [ ] **Step 1: Run static and complete automated verification**

```powershell
.\.venv\Scripts\python.exe -m compileall -q src examples tests
.\.venv\Scripts\python.exe -m pytest --basetemp outputs/.pytest-trace -q
git diff --check
```

Record exact pass/fail counts. Any defect first receives a failing regression test before source changes.

- [ ] **Step 2: Run four real videos**

Without editing `.env`, use a supported process-local model override if its configured model route is unavailable. Run these fixed video/prompt pairs with `--trace`, `--trace-output` and separate result files under `outputs/trace-validation/`:

- `2n.mp4` — `石头滚下坡`;
- `1n.mp4` — `A red rectangular block falls onto a white wooden plank and collides with it.`;
- `1y.mp4` — `A red rectangular block falls onto a white wooden plank and collides with it.`;
- `29932d12f47258c3b75f98e25e643d2c.mp4` — `An orange ball falls freely under gravity.`

- [ ] **Step 3: Validate all traces**

Run the validator on every trace with `--require-sam2 --require-model-planner --fail-on-provider-fallback`. Require exit code `0`, fusion recomputation tolerance `1e-6`, no error nodes, no fallback, both physical and violation outputs across the four videos, and no rejected/uncertain review published as a violation.

- [ ] **Step 4: Inspect trace usefulness**

Manually confirm every fixed node appears, each input/output summary identifies the relevant upstream evidence, configured/effective family weights are visible, and the final decision can be followed without reading source code. Record any ambiguity as a failing test and fix through a new red-green cycle.

## Task 8: Secret audit, final commit and GitHub synchronization

**Files:**
- Commit only Critic source, examples, tests, schema and approved documentation from this plan
- Never stage: `.env`, `*.mp4`, `*.pt`, model directories, `outputs/`, credentials or raw provider payloads

- [ ] **Step 1: Audit the exact diff and staged paths**

```powershell
git status --short
git diff --check
git diff --stat origin/sy...HEAD
git diff --name-only origin/sy...HEAD
```

Scan changed text for API-key prefixes, authorization headers, the known SSH credentials, `.env` content, base64 image prefixes and raw provider fields. Remove only generated/secret material; preserve unrelated user files.

- [ ] **Step 2: Run fresh final verification**

Repeat compileall, the complete pytest suite and all four strict trace validators. Completion claims use only this fresh output.

- [ ] **Step 3: Commit any final reviewed documentation**

If Task 7 produced no source changes, no synthetic empty commit is created. If acceptance documentation changed, stage only its exact path and use:

```powershell
git commit -m "docs: record critic trace verification"
```

- [ ] **Step 4: Push only the `sy` branch**

```powershell
git push origin sy
git rev-parse HEAD
git rev-parse origin/sy
git status --short
```

Require identical local/remote SHAs and a clean worktree. Return the commit SHA, test count, four-video result/trace paths, strict-validator outcomes and any remaining limitations.

## Execution results

### R1 — Design and implementation baseline

- Design was approved in conversation and frozen in `docs/superpowers/specs/2026-07-17-critic-execution-trace-design.md`.
- Implementation began from a clean `sy` checkout. The pre-change complete suite passed `340/340` in 8.79 seconds.
- Recorder, fusion validator, PQSG observer, pipeline hooks, CLI, JSON Schema and audit hardening were implemented through explicit RED→GREEN cycles and committed separately.
- Trace collection is optional and trace-only collection summaries are lazy; the untraced pipeline does not construct them.

### R2 — Automated verification

- Focused recorder/pipeline/CLI/schema tests passed after each task checkpoint.
- The final complete suite passed `382/382` in 9.29 seconds before real-video acceptance.
- `python -m compileall -q src examples tests` and `git diff --check` exited successfully.
- One earlier complete-suite run produced a Windows `PermissionError` while an unrelated report test atomically renamed a directory containing the literal `<script>`. The other 19 report publication tests and a minimal `os.replace` probe passed; the unchanged failing test passed on immediate rerun, and the next complete suite passed. No benchmark/report source was modified.

### R3 — Final-code real-video acceptance

All runs used the repository SAM2.1 checkpoint, the configured `gpt-5-mini` model route and the final implementation. `.env` was not modified. Runtime artifacts remain ignored under `outputs/trace-validation/`.

| Video | Final decision | Physics score | Coverage | Trace nodes | Planner constraints | Retained candidate indices |
|---|---:|---:|---:|---:|---:|---|
| `2n.mp4` | physical | 0.9463 | 0.6200 | 29 | 2 | none |
| `1n.mp4` | violation | 0.1360 | 0.7500 | 31 | 2 | 6 |
| `1y.mp4` | violation | 0.0769 | 0.7833 | 32 | 2 | 0, 3, 7, 8, 9, 12, 13 |
| `29932d12f47258c3b75f98e25e643d2c.mp4` | physical | 0.8786 | 0.7600 | 26 | 1 | none |

- Every trace reports `sam2_used=true`, Planner source `model` and provider fallback count `0`.
- Every trace passed all 22 checks with `--require-sam2 --require-model-planner --fail-on-provider-fallback`.
- The validator independently reproduced configured/effective weights, weighted contributions, score, coverage, confidence and final threshold/hard-violation decision.
- No rejected or uncertain VLM candidate was retained as a public violation.

### R4 — Auditability and safety decisions

- Each fixed pipeline stage and every executed PQSG node records bounded inputs, outputs, status and elapsed time.
- Planner output includes object, event, relation and physical-constraint semantics rather than counts alone.
- Public violation evidence retains `candidate_index`, making candidate→keyframe→review→violation alignment unambiguous even for duplicate categories/time windows.
- Forbidden trace fields cover exact and aliased API keys, authorization/request headers, access/refresh/bearer tokens, passwords, raw responses, images, masks and binary payloads.
- Live rendering callback failure is isolated as a trace warning and cannot change Critic execution.
