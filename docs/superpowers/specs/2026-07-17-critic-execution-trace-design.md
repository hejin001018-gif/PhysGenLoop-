# Critic Execution Trace and Validation Design

**Date:** 2026-07-17
**Status:** Approved in conversation; awaiting written-spec confirmation
**Primary scope:** Auditable, per-node execution tracing for the single-video Critic CLI
**Explicit non-scope:** Changing fusion weights, learning a fusion model, HTML reports, benchmark prediction rewrites, or storing raw model/image payloads

## 1. Goal

Make a single Critic evaluation understandable without reading implementation code. A user must be able to see, in execution order, what each pipeline node received, what it produced, whether it completed, skipped, degraded or failed, how long it took, and how its evidence affected the final decision. A separate validator must independently recompute the fusion arithmetic and report contract violations with a non-zero exit code.

Tracing is observational. Enabling or disabling it must not change Planner, SAM2, PQSG, rule, checklist, mechanics, VLM, fusion or final-report behavior.

## 2. Considered approaches

### A. Reconstruct a trace from final `CriticArtifacts`

This is the smallest change and can describe most outputs. It cannot reliably show actual timing, stage failures before artifacts exist, or the exact input seen by each component. It is retained only as a fallback for older saved artifacts.

### B. Instrument the core pipeline with a structured recorder

This is the selected approach. `PhysicsCritic.analyze_detailed()` accepts an optional recorder and emits bounded, sanitized records before and after each real stage. Existing callers remain unchanged because tracing defaults to `None`. The CLI can stream completed records to stderr and persist the complete trace as JSON.

### C. Dump raw Python objects/provider responses and build an HTML viewer

This would expose more data but risks API secrets, image payloads, masks, huge files and provider-specific raw content. It also adds a presentation layer before the audit contract is stable. It is rejected for this cycle.

## 3. Public interface

The existing result JSON remains unchanged unless tracing is explicitly requested.

```powershell
python examples/evaluate_video.py `
  --video 2n.mp4 `
  --prompt "石头滚下坡" `
  --trace `
  --trace-output outputs/2n.trace.json `
  --output outputs/2n.result.json

python examples/validate_pipeline_trace.py `
  outputs/2n.trace.json `
  --require-sam2 `
  --require-model-planner `
  --fail-on-provider-fallback
```

- `--trace` streams one concise Chinese line when each node reaches a terminal state.
- `--trace-output PATH` writes the full UTF-8 JSON trace and implicitly enables collection even when `--trace` is absent.
- A pipeline exception still writes the partial trace when `--trace-output` was supplied, then preserves the existing non-zero CLI exit behavior.
- The validator prints PASS/WARN/FAIL checks. Exit code `0` means all required checks passed, `1` means a semantic or arithmetic check failed, and `2` means the file or invocation is invalid.
- `--require-sam2`, `--require-model-planner` and `--fail-on-provider-fallback` turn normally informative warnings into validation failures.

## 4. Trace contract

The top-level schema version is `pavg-critic-trace/v1`. A trace contains safe request/config metadata, ordered node records, the terminal outcome and any trace-level warnings.

Each node record contains:

- monotonically increasing `sequence`;
- stable `node_id`, human-readable `label` and optional `parent_id`;
- `source_nodes` identifying upstream data dependencies;
- terminal `status`: `completed`, `skipped`, `degraded` or `error`;
- `elapsed_ms` measured with a monotonic clock;
- bounded `input` and `output` summaries;
- sanitized warnings or an error containing only exception type and a maximum 300-character message.

The fixed stage nodes are:

1. `request`
2. `physics_planner`
3. `question_graph`
4. `video_observation`
5. `trajectory`
6. `event_detection`
7. `mechanics`
8. `rule_engine`
9. `temporal_localization`
10. `visual_evidence`
11. `checklist`
12. `keyframe_selection`
13. `pqsg_execution`
14. zero or more `pqsg_node.<node-id>` child records
15. `vlm_verification`
16. `candidate_fusion`
17. `question_scoring`
18. `evidence_fusion`
19. `final_report`

Disabled or inapplicable stages remain visible with `status="skipped"` and a reason. Provider fallback produces `status="degraded"`, records the safe failure classification and identifies the fallback source. A terminal exception produces `status="error"`; no later stage may be reported as completed.

## 5. Input and output summaries

Summaries expose domain meaning while remaining bounded:

- request: video path, prompt, prompt hash, configured model identifiers and enabled module flags;
- Planner: prompt and existing plan source in, resolved objects/events/constraints, source, confidence and fallback status out;
- question graph: resolved plan counts in, graph source, node IDs/types/dependencies and sanitation/fallback status out;
- video observation: video metadata and detector backend in, frame/state/object/visibility counts, frame range and inferred floor out;
- trajectory/events: counts and bounded per-object/per-event summaries;
- mechanics/checklist: applicability, evaluator/dimension statuses, score, coverage and failures;
- rules/localization: candidate identity, category, scores, intervals and evidence-frame counts before and after localization;
- PQSG: graph and evidence counts in, each node's status, score, critical frames and reason out;
- VLM: candidate/keyframe metadata in, confirmed/rejected/uncertain/unavailable result per candidate out;
- fusion: all arithmetic fields required for independent recomputation;
- final report: decision, score, confidence, coverage and retained violation categories.

Large collections include total counts, a deterministic bounded preview and a SHA-256 digest of their canonical JSON summary. Image arrays, masks and full per-frame states are never serialized.

## 6. Fusion audit

The `evidence_fusion` node records, for every family:

```text
configured_weight
status
score
coverage
confidence
effective_weight = configured_weight × coverage × confidence
weighted_contribution = score × effective_weight
```

It also records:

- total configured and effective weights;
- recomputed score before hard-violation handling;
- weighted coverage;
- `physical_score_threshold` and `minimum_coverage`;
- retained hard-violation count;
- any score cap applied by candidate fusion;
- decision before and after the hard-violation rule;
- the final score, confidence and coverage.

The validator independently recalculates these values with absolute tolerance `1e-6`. It also verifies that unavailable families have zero effective weight and that VLM-rejected or uncertain candidates are not present as final violations.

This trace makes weight behavior visible but does not optimize weights. Weight search requires a separately frozen train/development protocol so VideoPhy labels cannot leak into the reported test result.

## 7. Validator checks

`examples/validate_pipeline_trace.py` validates:

1. schema version, required types and required top-level fields;
2. unique node IDs, monotonic sequence numbers and legal parent/source references;
3. stage order and the rule that no completed node follows a terminal error;
4. required-stage presence and explicit skipped reasons;
5. SAM2, model Planner and provider-fallback policies selected by CLI flags;
6. one PQSG child result for every executed question-graph node;
7. candidate-to-keyframe-to-review index alignment;
8. rejected/uncertain VLM candidates do not become final violations;
9. family effective weights, contributions, weighted score, coverage and confidence;
10. final decision consistency with thresholds, coverage and hard-violation state;
11. forbidden sensitive keys and suspicious base64/raw-payload fields.

Warnings do not change exit code unless a strict flag promotes them to failures. Arithmetic, schema, dependency, secret-safety and final-decision inconsistencies always fail.

## 8. Error and privacy behavior

The recorder must not receive or serialize API keys, authorization headers, `.env` values, request headers, raw provider payloads, image bytes, base64 images, SAM2 masks or chain-of-thought. Prompts and evidence reasons are allowed because they are required for user audit, but collection previews and messages are bounded.

The sanitizer recursively rejects forbidden key names and replaces unsupported or oversized values with a typed summary. Model/provider errors include stage, exception type and at most 300 characters of message. Trace writing uses a temporary sibling file followed by atomic replacement so interruption cannot leave a valid-looking truncated JSON file.

## 9. Code organization

- `src/pavg_critic/execution_trace.py`: trace schema objects, recorder, bounded sanitizers, fusion-audit construction and validation functions.
- `src/pavg_critic/pipeline.py`: optional recorder hooks at actual stage boundaries; no decision-logic changes.
- `examples/evaluate_video.py`: trace CLI flags, live stderr renderer and atomic trace persistence.
- `examples/validate_pipeline_trace.py`: standalone human-readable validator CLI.
- `schemas/critic_trace.schema.json`: machine-readable structural contract.
- `tests/test_execution_trace.py`: recorder, sanitization, arithmetic and validator unit tests.
- `tests/test_pipeline_trace.py`: real deterministic pipeline stage and PQSG-node tracing tests.
- `tests/test_evaluate_video_example.py`: CLI flag and partial-trace persistence tests.

## 10. Verification and acceptance

Implementation follows test-driven development. Acceptance requires:

- each new behavior first demonstrated by a failing test;
- focused trace and validator tests passing;
- the complete repository test suite passing;
- `compileall` and `git diff --check` passing;
- at least four real videos covering physical and violation decisions producing valid traces;
- at least one strict validation run requiring SAM2 and model Planner;
- independent fusion recomputation matching the Critic output within `1e-6`;
- a repository scan proving no `.env`, API secret, video, checkpoint, model weight or generated trace is staged;
- only Critic-related source, tests, schema and documentation committed and pushed to GitHub branch `sy`;
- final handoff reporting the exact commit SHA and verification commands/results.

