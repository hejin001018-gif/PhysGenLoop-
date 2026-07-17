# VideoPhy-2 Full Result Merge and Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Use test-driven development for every behavior change.

**Goal:** Merge the two existing append-only VideoPhy-2 prediction shards without rerunning inference, verify all 6,794 sample-by-method keys, and generate an auditable full-dataset Chinese report using the already frozen statistical protocol.

**Architecture:** Add a dependency-free full-report analysis layer beside the existing smoke reporter. It consumes the frozen 3,397-row manifest, the two completed prediction JSONL files and SAM2 observation metadata; it never calls a model, trains a model, changes predictions or edits benchmark membership. It rejects incomplete, duplicate or out-of-scope keys before producing deterministic merged artifacts, paired statistics, action-group bootstrap intervals and source-metadata slices.

**Tech Stack:** Python 3.12 standard library, existing `BenchmarkSample`/`BenchmarkPrediction` contracts, pytest, JSON/JSONL and Markdown.

---

## Frozen analysis contract

- Population: all 3,397 frozen VideoPhy-2 samples.
- Methods: baseline `D0_DIRECT_VLM`; candidate `B1_RULE` (SAM2-backed Revision B).
- Expected terminal keys: exactly 3,397 x 2 = 6,794. A failure record remains in the denominator and is not retried for reporting.
- Merge inputs: only canonical shard A and shard B JSONL files. The archived pre-split run is excluded.
- Classification: `physical` and `violation`; `unknown` and failures count as incorrect.
- Primary effect: `B1_RULE - D0_DIRECT_VLM` Macro-F1.
- Confidence interval: paired action-group bootstrap, 2,000 resamples, seed `20260717`, resampling the frozen `prompt_group_id` clusters with replacement, using linear 2.5th and 97.5th percentiles.
- Paired outcomes: both correct, D0 only correct, B1 only correct, both wrong.
- Slices: generator, action group and exact source `raw_labels.metadata_rules` family. Rule-family values are parsed only with `ast.literal_eval`, whitespace-normalized, multi-label, and otherwise assigned to `__unmapped__`; no prediction-derived family is used.
- Latency: prediction latency and SAM2 `production_latency_sec` are reported separately with mean/p50/p95. Duplicate observation metadata must agree exactly or fail audit.
- Material arithmetic: report each frozen gate separately. A VideoPhy-2-only support flag may be true only when delta is at least +0.05, the CI excludes zero, both recalls are nonzero, failure increase is at most +0.01 and more than one generator has positive Macro-F1 delta. The overall pre-registered material-improvement verdict remains `not_evaluable_ood_deferred` because VideoPhy-1 is outside the current scope.
- Output is deterministic: sorted keys, fixed JSON formatting, stable table ordering and cryptographic input/output hashes.

## Task 1: Extend complete classification and latency metrics

**Files:**
- Modify: `src/pavg_critic/benchmarking/metrics.py`
- Modify: `tests/benchmarking/test_metrics.py`

- [x] Add failing tests for physical recall and deterministic p50/p95 latency, including an even-length latency vector.
- [x] Implement a dependency-free linear percentile helper and expose `physical_recall`, `p50_latency_sec` and `p95_latency_sec` from `compute_smoke_metrics` without changing existing metric meanings.
- [x] Run the focused metric tests, then the existing benchmark report tests.
- [x] Commit only the tested metric change.

## Task 2: Implement strict shard merge and artifact audit

**Files:**
- Create: `src/pavg_critic/benchmarking/full_report.py`
- Create: `tests/benchmarking/test_full_report.py`

- [ ] Add failing tests proving that exact disjoint shards merge into a stable sample/method order.
- [ ] Add failing tests for duplicate keys across shards, missing keys, unknown sample IDs, unknown methods and malformed prediction records.
- [ ] Implement strict expected-key validation and a deterministic JSONL writer.
- [ ] Record per-input SHA-256, line count, method count, terminal/failure count, and merged-output SHA-256 in `artifact_audit.json`.
- [ ] Run focused tests and commit the strict merge layer.

## Task 3: Implement paired statistics and frozen slices

**Files:**
- Modify: `src/pavg_critic/benchmarking/full_report.py`
- Modify: `tests/benchmarking/test_full_report.py`

- [ ] Add failing tests for the four paired-outcome cells and for unknown/failure records counting as incorrect.
- [ ] Add a small hand-computed action-cluster example proving bootstrap resamples entire groups, is deterministic at seed `20260717`, uses exactly the requested resample count and returns ordered bounds.
- [ ] Add failing tests for generator/action slices and safe parsing of exact, multi-label, malformed and missing source rule-family metadata.
- [ ] Implement per-method full metrics, paired Macro-F1 delta/bootstrap, paired outcomes and deterministic generator/action/rule-family tables.
- [ ] Keep single-class or small slices visible with counts; do not silently filter inconvenient strata.
- [ ] Run focused tests and commit the statistics layer.

## Task 4: Add SAM2 end-to-end latency audit and decision arithmetic

**Files:**
- Modify: `src/pavg_critic/benchmarking/full_report.py`
- Modify: `tests/benchmarking/test_full_report.py`

- [ ] Add failing tests for mean/p50/p95 `production_latency_sec`, missing metadata accounting, identical duplicate acceptance and conflicting duplicate rejection.
- [ ] Add failing tests for every VideoPhy-2 decision gate and for the mandatory `not_evaluable_ood_deferred` overall verdict.
- [ ] Implement observation metadata aggregation without loading mask tensors or videos.
- [ ] Implement explicit arithmetic fields rather than a prose-only verdict.
- [ ] Run focused tests and commit the latency/decision layer.

## Task 5: Build the deterministic full-report CLI and Chinese renderer

**Files:**
- Create: `benchmarks/report_full_video_benchmark.py`
- Create: `tests/benchmarking/test_full_report_cli.py`
- Modify: `src/pavg_critic/benchmarking/full_report.py`

- [ ] Add a failing end-to-end CLI test using a temporary two-method manifest, two disjoint prediction shards and observation metadata.
- [ ] Expose required `--manifest`, repeatable `--predictions`, `--output-dir`, fixed/default method IDs, bootstrap count/seed and repeatable `--observation-meta-dir` arguments.
- [ ] Write `merged_predictions.jsonl`, `artifact_audit.json`, `summary.json`, `summary.md`, `paired_outcomes.json` and `slices.json` atomically after validation.
- [ ] Render a Chinese Markdown report that distinguishes prediction latency from SAM2 production latency, lists all failures, reports bootstrap settings and prominently states that VideoPhy-1 OOD is deferred.
- [ ] Run the CLI test, all benchmark tests and the complete local pytest suite using `outputs/.pytest-tmp`.
- [ ] Commit the tested reporting CLI.

## Task 6: Complete both existing inference shards without restart

**Remote inputs:**
- Shard A: `/root/pavg-benchmark/runs/videophy2-full-qwen3vl8b/shard-a/run/predictions.jsonl`
- Shard B: `/root/pavg-benchmark-shard2/shard-b/run/predictions.jsonl`

- [ ] Continue monitoring the existing evaluator PIDs and append-only files; do not relaunch while they are healthy.
- [ ] Require terminal counts 3,398 on shard A and 3,396 on shard B, with no duplicate keys and no method/sample outside each frozen shard manifest.
- [ ] Record final wall time, GPU state, failure counts and process exit status in the main execution plan.
- [ ] Freeze SHA-256 hashes of both completed prediction inputs before copying or analysis.

## Task 7: Synchronize shard B metadata and run the formal merge on cloud2

**Remote output:**
- `/root/pavg-benchmark/runs/videophy2-full-qwen3vl8b/final-report/`

- [ ] Copy only completed shard B predictions, its frozen manifest and observation `.meta.json` files to a separate cloud2 import directory; do not copy videos or model weights.
- [ ] Transfer the tested reporting commit/source to cloud2 and rerun the complete remote pytest suite.
- [ ] Run the full-report CLI against the frozen full manifest, shard A/B predictions and both observation metadata directories.
- [ ] Require exactly 6,794 merged keys, zero duplicates/extras/missing keys and deterministic rerun hashes before accepting metrics.
- [ ] Record the formal result and every negative outcome without tuning prompts, thresholds, rules or sample membership.

## Task 8: Synchronize, security-audit and publish the VideoPhy-2 report

**Files:**
- Create: `outputs/benchmarks/videophy2-full-qwen3vl8b/`
- Modify: `docs/results/criticbenchmark.md`
- Modify: `docs/superpowers/plans/2026-07-16-full-videophy-server-evaluation.md`

- [ ] Synchronize manifests, predictions, summaries, slices, resolved non-secret configs, observation latency metadata and logs required for audit; exclude videos, weights, `.env` and provider payloads.
- [ ] Recompute local hashes and key alignment, then scan artifacts for SSH passwords, API-key prefixes, authorization headers and `.env` content.
- [ ] Update the Chinese benchmark narrative with exact full-population metrics, confidence interval, paired outcomes, generator/action/rule-family evidence, runtime, failures and limitations.
- [ ] Mark Task 8 and VideoPhy-2 portions of Task 10 complete; leave VideoPhy-1 Task 9 explicitly deferred and unchecked.
- [ ] Run the complete local pytest suite and a clean-room report regeneration check before claiming completion.
- [ ] Commit only source, tests, non-secret result artifacts and documentation; preserve unrelated user files.

## Execution results

Append immutable checkpoints here as each task completes. Do not replace prior entries or rewrite prediction inputs.

### R1 — Complete classification and latency metrics

- Strict TDD red state: the two new tests failed because `physical_recall` and `p50_latency_sec` were absent; the remaining five metric tests passed.
- The dependency-free linear percentile uses `(n - 1)q` interpolation. The frozen even-length example `[1, 2, 3, 4]` yields p50 `2.5` and p95 `3.85`.
- `compute_smoke_metrics` now reports physical recall and p50/p95 prediction latency without changing any existing key or formula.
- Independent specification review passed; independent quality review found no correctness, typing, maintenance or regression issue.
- Final focused verification: `9 passed` across metric and smoke-report tests using isolated basetemp `outputs/.pytest-tmp-task1`.
- Commit: `89755d9` (`feat: add full benchmark recall and latency metrics`).
