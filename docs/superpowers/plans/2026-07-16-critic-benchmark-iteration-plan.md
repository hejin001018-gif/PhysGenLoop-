# Critic Benchmark Iteration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an auditable, held-out VideoPhy-2 result where the SAM2-backed PAVG critic materially outperforms a matched direct-VLM baseline, while recording unsuccessful iterations instead of hiding them.

**Architecture:** Keep SAM2.1 Hiera B+ as the mandatory dense observation frontend. Separate observation quality, deterministic rule applicability, optional critical-frame VLM verification, and model-backbone effects so that an improvement can be attributed to a specific component. Tune architecture on a frozen diagnostic split only; evaluate each frozen candidate once on a held-out split.

**Tech Stack:** Python 3.12, PyTorch 2.13 CUDA 13.2, official Meta SAM2.1 Hiera B+, OpenCV, pytest, OpenAI-compatible Chat API, VideoPhy-2, optional remote NVIDIA GPU over SSH.

---

## Experimental contract

- Primary comparison is matched-backbone: the same model family is used for D0/D1 and every PAVG component that calls a VLM.
- SAM2 is mandatory for B1–M5. Sparse VLM observations are allowed only as an explicitly named frontend ablation.
- The 20-video smoke set remains a pipeline/diagnostic result, not a benchmark claim.
- The smoke set is split deterministically into `dev10` and `eval10`, stratified by physics label and generator where possible. Architecture and thresholds may inspect `dev10`; `eval10` is run once after an architecture is frozen.
- A result is “materially better” for this smoke gate only if held-out PAVG Macro-F1 is at least `D0 + 0.10`, balanced accuracy is at least `D0 + 0.10`, failure rate is no higher, and neither class has zero recall.
- No sample may be removed after predictions are observed. Unknown and failed predictions remain in the denominator.
- At most two architecture revisions may use `dev10`. Model escalation is evaluated after architecture freeze; it must not change sample membership or thresholds.
- If these gates are not met, report the negative result honestly and move the next decision to a larger pre-registered server run. Do not tune on `eval10` until a positive number appears.

## Current checkpoint (2026-07-16)

- Official SAM2.1 source commit: `2b90b9f5ceec907a1c18123530e92e794ad901a4`.
- SAM2.1 Hiera B+ checkpoint successfully loaded on RTX 5060 Laptop GPU.
- Real three-frame propagation returned one continuous track on all frames with exact moving-square boxes.
- VideoPhy-2 source commit reported by the dataset endpoint: `90b81ffa54f565d9e40e83b7a2c247ba1dccfa2b`.
- Frozen smoke manifest: 20 videos, 10 physical / 10 violation, 7 generators, 20/20 decodable, 1,660 total frames.
- Two-video preflight: 8/8 predictions, zero failures, SAM2 frame coverage 1.0 for both videos.
- Interrupted full run preserved 30 records: D0=8, D1=8, B1=7, M3=7.
- Common-seven diagnostic metrics:

| Method | Accuracy | Balanced accuracy | Macro-F1 | Violation recall | Unknown | Failure |
|---|---:|---:|---:|---:|---:|---:|
| D0 direct VLM | 0.429 | 0.500 | 0.300 | 0.000 | 0.000 | 0.000 |
| D1 checklist VLM | 0.286 | 0.333 | 0.222 | 0.000 | 0.000 | 0.000 |
| B1 rules + SAM2 | 0.571 | 0.500 | 0.364 | 1.000 | 0.000 | 0.000 |
| M3 mechanics + SAM2 | 0.571 | 0.500 | 0.364 | 1.000 | 0.000 | 0.000 |

Interpretation: the frontend is dense, but current PAVG predicts every common sample as a violation. The first architecture target is physical-sample false positives, not additional tracking density.

### Task 1: Freeze run metadata and complete the mini baseline

**Files:**
- Modify: `docs/superpowers/plans/2026-07-16-critic-benchmark-iteration-plan.md`
- Existing output: `outputs/benchmarks/videophy2-smoke20-gpt5mini-sam2bplus/`

- [x] **Step 1: Record immutable inputs**

Write manifest SHA-256, checkpoint SHA-256, git revision, model ID, provider type, frame count, and method order into `resolved_config.json`. Secret-bearing fields must be absent or `REDACTED`.

- [x] **Step 2: Resume the append-only run**

Run the existing Stage A CLI with the same run directory. Expected: the runner skips the 30 existing sample×method keys and appends only missing keys.

- [x] **Step 3: Verify completeness**

Run:

```powershell
.\.venv\Scripts\python.exe -c "from pavg_critic.benchmarking.runner import load_predictions; p=load_predictions('outputs/benchmarks/videophy2-smoke20-gpt5mini-sam2bplus/predictions.jsonl'); print(len(p))"
```

Expected: `80` records, with 20 records for each of D0, D1, B1 and M3.

- [x] **Step 4: Append results to this document**

Record the full metric table, runtime, failure list, SAM2 coverage distribution, and category prevalence under `Iteration results`.

### Task 2: Create a leakage-resistant dev/eval split

**Files:**
- Modify: `src/pavg_critic/benchmarking/datasets.py`
- Modify: `benchmarks/prepare_videophy_manifest.py`
- Test: `tests/benchmarking/test_datasets.py`
- Create: `evaluation/manifests/videophy2_smoke_dev10.json`
- Create: `evaluation/manifests/videophy2_smoke_eval10.json`

- [x] **Step 1: Write a failing split test**

The test must prove that a fixed seed creates disjoint 10/10 sample IDs, preserves both labels in each split, does not divide a `prompt_group_id` when a group has multiple samples, and is byte-stable.

- [x] **Step 2: Implement grouped deterministic splitting**

Add `split_diagnostic_manifest(samples, dev_count, seed)` that groups by `prompt_group_id`, balances physics label and generator greedily, and raises when an exact non-overlapping split cannot be formed.

- [x] **Step 3: Generate and hash both manifests**

Expected: disjoint IDs, union equal to smoke20, portable POSIX video paths, and 10 samples per file.

- [ ] **Step 4: Commit**

Commit only the adapter, tests, CLI change and the two manifests.

### Task 3: Export rule-level false-positive diagnostics

**Files:**
- Create: `src/pavg_critic/benchmarking/diagnostics.py`
- Create: `benchmarks/diagnose_pavg_predictions.py`
- Test: `tests/benchmarking/test_diagnostics.py`
- Create output: `outputs/benchmarks/diagnostics-gpt5mini-dev10/`

- [x] **Step 1: Write failing diagnostic tests**

Use frozen synthetic observations to require per-sample JSON containing track count, represented frames, events, raw candidates, fused violations, category counts, detector scores, critical frames and gold/predicted labels.

- [x] **Step 2: Implement diagnostic replay from observation cache**

Replay `PhysicsCritic.analyze_detailed()` without rerunning SAM2. Serialize only audit fields; do not include image bytes or credentials.

- [x] **Step 3: Run on dev10 only**

Produce `samples/*.json`, `category_summary.json`, and `false_positives.md`. Record which rule categories fire on physical videos and whether they originate from track discontinuity, missing applicability, floor inference, or VLM seed errors.

- [x] **Step 4: Append the diagnosis to this document**

Include exact counts and representative sample IDs. Do not use eval10 labels.

### Task 4: Add rule applicability and observation-quality gates

**Files:**
- Modify: `src/pavg_critic/physics_rules.py`
- Modify: `src/pavg_critic/event_detector.py`
- Modify: `src/pavg_critic/config.py`
- Test: `tests/test_physics_rules.py`
- Test: `tests/test_event_detector.py`
- Test: `tests/benchmarking/test_pavg_methods.py`

- [x] **Step 1: Write one failing test per diagnosed false-positive mechanism**

Examples must use explicit physics plans: ordinary hand/object motion must not trigger premature rebound; a planned rebound remains detectable; a one-frame mask jump with immediate recovery is treated as tracking uncertainty rather than a hard teleport; sustained discontinuity remains a violation.

- [x] **Step 2: Implement the smallest applicability changes**

Gate contact/rebound rules on relevant expected events or constraints. Add a configurable minimum persistence for hard continuity events. Do not change thresholds unrelated to a diagnosed dev10 failure.

- [x] **Step 3: Run the full 141+ unit test suite**

Expected: all existing synthetic violations remain detected and all new physical controls pass.

- [ ] **Step 4: Run architecture revision A on dev10**

Use the existing SAM2 cache. Compare against the frozen dev10 D0 predictions and append metrics/category counts below.

- [x] **Step 5: Stop or proceed by rule**

Proceed to Task 5 only if revision A still has a physical-class false-positive rate above 0.30 or misses the dev success gate. Otherwise freeze revision A.

### Task 5: Add critical-frame VLM verification as M4

**Files:**
- Modify: `src/pavg_critic/benchmarking/pavg_methods.py`
- Modify: `benchmarks/evaluate_video_benchmark.py`
- Modify: `src/pavg_critic/evaluation.py`
- Test: `tests/benchmarking/test_pavg_methods.py`
- Test: `tests/benchmarking/test_cli.py`

- [ ] **Step 1: Write failing M4 tests**

Require M4 to reuse the same cached SAM2 observations, invoke `EvidenceGroundedVLMVerifier` only for localized candidates, retain an explicit failure when verification fails, and preserve B1/M3 outputs.

- [ ] **Step 2: Implement M4 with a named verifier policy**

Inject `EvidenceGroundedVLMVerifier` into `PhysicsCritic`. Store verifier model ID, candidate detector score and VLM score in evidence. Use a separately named fusion configuration; do not silently alter B1/M3.

- [ ] **Step 3: Tune at most one verifier fusion policy on dev10**

The candidate policies are pre-registered as detector/VLM weights `0.5/0.5` and `0.4/0.6`, with violation threshold `0.5`. Evaluate both on dev10, select once, and record both results.

- [ ] **Step 4: Freeze architecture**

After this step no architecture, threshold, sample or prompt changes may use eval10 outcomes.

### Task 6: Run the held-out matched-backbone evaluation

**Files:**
- Create output: `outputs/benchmarks/videophy2-eval10-<architecture>-<model>/`
- Modify: `docs/superpowers/plans/2026-07-16-critic-benchmark-iteration-plan.md`

- [ ] **Step 1: Run gpt-5-mini once on eval10**

Run D0, D1 and the frozen PAVG candidate. Reuse immutable SAM2 observations but create a new prediction JSONL.

- [ ] **Step 2: Apply the material-improvement gate**

Report Macro-F1, balanced accuracy, both class recalls, Spearman, unknown/failure rates and paired sample outcomes. Do not change the frozen architecture after reading these results.

- [ ] **Step 3: Append results and decision**

Mark the gate pass/fail with arithmetic, not subjective language.

### Task 7: Escalate model backbones without changing architecture

**Files:**
- Create output: `outputs/benchmarks/videophy2-eval10-terra/`
- Create output: `outputs/benchmarks/videophy2-eval10-luna/`
- Modify: `docs/superpowers/plans/2026-07-16-critic-benchmark-iteration-plan.md`

- [x] **Step 1: Probe model availability without recording secrets**

Make one schema-only request to the configured endpoint for `gpt-5.6-terra` and `gpt-5.6-luna`. Record status, latency and returned model ID; never log authorization headers.

- [ ] **Step 2: Run Terra on eval10**

Use Terra for both the direct baseline and all VLM calls in the frozen PAVG architecture. Do not compare Terra-PAVG causally against mini-D0; always include Terra-D0.

- [ ] **Step 3: Run Luna only if needed**

Run the same locked matrix when Terra is unavailable, has excessive failure rate, or fails the success gate. Luna must not trigger a new architecture revision.

- [ ] **Step 4: Select the deployment candidate**

Select by the pre-registered gate, then lower failure rate, then lower cost/latency. Record all attempted models, including failures.

### Task 8: Use the remote GPU only when the local resource gate fires

**Files:**
- Modify: `docs/superpowers/plans/2026-07-16-critic-benchmark-iteration-plan.md`
- Create output: `outputs/benchmarks/server-audit/`

- [ ] **Step 1: Audit the server read-only**

Over SSH, record hostname, OS, GPU model/count/memory, driver, CUDA, disk free space, Python versions and running GPU processes. Redact credentials and unrelated process arguments.

- [ ] **Step 2: Decide migration**

Migrate only if the server offers materially greater usable VRAM/throughput, or the local run cannot finish. Closed-model API latency alone is not a GPU migration reason.

- [ ] **Step 3: Transfer reproducibly if selected**

Transfer a git bundle or clean source snapshot, portable manifests, 20 public videos and the verified SAM2 checkpoint. Do not transfer `.env`; create remote process environment variables for the run and delete the remote shell history entry if the provider records literal secrets.

- [ ] **Step 4: Verify parity**

Run unit tests and the three-frame SAM2 propagation test remotely before benchmark execution. Record commit and checkpoint hashes.

### Task 9: Final verification and result package

**Files:**
- Modify: `docs/superpowers/specs/2026-07-15-critic-benchmark-evaluation-design.md`
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-07-16-critic-benchmark-iteration-plan.md`

- [ ] **Step 1: Run all tests**

Run `.\.venv\Scripts\python.exe -m pytest -q`. Expected: all tests pass.

- [ ] **Step 2: Audit artifacts**

Verify every run has resolved config, predictions JSONL, summary JSON/Markdown, no duplicate sample×method keys, exact manifest alignment and no secret-shaped strings.

- [ ] **Step 3: Write the final comparison**

Include matched tables, confidence limitations, category-level diagnosis, runtime/cost, server decision and the exact next loop recommendation.

- [ ] **Step 4: Commit documentation and manifests**

Commit only project code, tests, portable manifests and documentation. Keep videos, model weights, credentials and raw API payloads git-ignored.

## Iteration results

Results are appended here immediately after each task checkpoint. Existing numbers are diagnostic and must not be rewritten when later iterations improve them.

### R0 — gpt-5-mini partial smoke20

- Status: interrupted after 30/80 append-only records; resumable.
- Common aligned samples: 7.
- Result: B1/M3 Macro-F1 0.364 versus D0 0.300, but B1/M3 predicted violation for all seven samples.
- Decision: not sufficient. Complete the run, split dev/eval, and diagnose hard-rule false positives before model escalation.

### R0-complete — gpt-5-mini smoke20

- Records: 80/80 unique sample×method keys; failures: 0.
- SAM2 caches: 20/20; mean and minimum represented-frame coverage: 1.0; propagation failures: 0.

| Method | Accuracy | Balanced accuracy | Macro-F1 | Violation precision | Violation recall | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| D0 direct VLM | 0.500 | 0.500 | 0.333 | 0.000 | 0.000 | 0.070 |
| D1 checklist VLM | 0.400 | 0.400 | 0.286 | 0.000 | 0.000 | 0.021 |
| B1 rules + SAM2 | 0.500 | 0.500 | 0.333 | 0.500 | 1.000 | N/A |
| M3 mechanics + SAM2 | 0.500 | 0.500 | 0.333 | 0.500 | 1.000 | 0.482 |

- Result: no improvement. D0 predicted every sample physical; B1/M3 predicted every sample violation.
- Decision: keep this as the frozen pre-fix baseline and evaluate revisions only through the dev/eval protocol.

### Reliability incident — concurrent resume recovery

- Two foreground wrappers were terminated while their Python child processes continued running.
- This created 64 physical JSONL lines with 12 duplicate sample×method keys. Every duplicate pair had identical label and score.
- The original 64-line file was archived as `predictions.concurrent-duplicate-archive.jsonl`; the working file retained the first record per key and was verified as 52/52 unique.
- `BenchmarkRunner` now acquires an atomic `.jsonl.lock` and rejects duplicate keys in an existing file. Tests cover both conditions.
- A UTF-8 BOM introduced by the PowerShell recovery was detected by the strict loader, removed without changing JSON content, and revalidated.

### Split S1 — frozen diagnostic and held-out manifests

- `dev10`: 10 samples, 5 physical / 5 violation, 7 generators, SHA-256 `ffba864b9351b8500da402dcb6f4ed04e437f6cbb1f75cf4d2bee978d5f02c89`.
- `eval10`: 10 samples, 5 physical / 5 violation, 7 generators, SHA-256 `d631e871a89f1db29fb1f22fad626553f84f2c470cca10e3ad07c65b135918bc`.
- Overlap: 0; union: all 20 frozen smoke samples.

### R1 — rule applicability revision A, dev10

- All 10 dev samples were replayed from the frozen R0 SAM2 cache; failures: 0.
- Root candidate counts on physical samples before the fix: surface penetration 19, premature rebound 30, disappearance 5, reverse gravity 3.
- Revision: surface penetration requires a planned contact/rebound constraint; premature rebound requires a planned fall/contact/rebound event.
- Physical false positives fell from 5/5 to 3/5 diagnosed physical samples.

| Method | Count | Accuracy | Balanced accuracy | Macro-F1 | Violation precision | Violation recall |
|---|---:|---:|---:|---:|---:|---:|
| D0 direct VLM | 10 | 0.500 | 0.500 | 0.333 | 0.000 | 0.000 |
| B1 before revision | 10 | 0.500 | 0.500 | 0.333 | 0.500 | 1.000 |
| B1 revision A | 10 | 0.700 | 0.700 | 0.670 | 0.625 | 1.000 |

- Remaining false-positive mechanisms: SAM2 stable identities were discarded by a second centroid tracker, and expected fall/rebound behavior was applied to unrelated tracked objects.
- Revision B implemented backend-provided stable track IDs and a raw-observation cache boundary. All 155 tests pass. A fresh dev10 SAM2 cache is required before measuring Revision B.

### Model availability probe P1

- `gpt-5.6-terra`: available; minimal structured-text latency 1.789 s; schema valid.
- `gpt-5.6-luna`: available; minimal structured-text latency 2.020 s; schema valid.
- No credential value, authorization header or raw provider payload was recorded.
- Decision: use Terra first after architecture freeze; Luna remains the pre-registered fallback/comparison.
