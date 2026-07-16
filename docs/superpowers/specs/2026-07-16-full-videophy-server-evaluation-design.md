# Full VideoPhy Server Evaluation Design

**Date:** 2026-07-16  
**Status:** Approved for autonomous execution by the user  
**Scope:** VideoPhy-2 full public test set as the primary benchmark; VideoPhy-1 full accessible public test set as frozen OOD evaluation.

## 1. Goal

Measure whether the frozen SAM2-backed PAVG critic improves physical-commonsense classification over a matched direct video-language-model baseline on the full accessible benchmark, without tuning on full-test labels and without hiding failed samples.

The primary claim is a paired, same-backbone comparison. Every sample remains in the denominator. Unknowns, content-policy refusals, download failures, decode failures, OOMs and inference failures are reported explicitly.

## 2. Frozen data scope

### VideoPhy-2 primary

The official public test CSV is `videophysics/videophy2_test`, referenced by the official [VideoPhy repository](https://github.com/Hritikbansal/videophy/tree/main/VIDEOPHY2). The locally frozen source contains 3,397 unique video URLs, 198 action groups and seven generators:

- 1,785 samples with human physical-commonsense score at least 4 (`physical`);
- 1,612 samples with score below 4 (`violation`);
- generator counts: Wan 591, VideoCrafter 591, CogVideo 589, Hunyuan 587, Cosmos 585, Ray2 394 and Sora 60;
- estimated video storage is approximately 3.2 GiB from the measured smoke-video distribution.

All 3,397 rows are the primary full-test population. The earlier smoke20/dev10/eval10 samples remain included; their prior use is disclosed and the full result is not described as a pristine unseen test.

### VideoPhy-1 OOD

Use every downloadable, human-annotated row in the official `videophysics/videophy_test_public` test dataset referenced by the [official repository](https://github.com/Hritikbansal/videophy). No PAVG threshold, prompt, rule or model choice may change after VideoPhy-2 full predictions are read.

The exact row count, generator distribution, checksum and unavailable URL count are frozen after retrieval and recorded before inference.

## 3. Compared methods

### Primary full matched block

Use `Qwen/Qwen3-VL-8B-Instruct` served through vLLM on the remote A100 40GB:

- `D0_OPEN_DIRECT`: fixed 16-frame direct VLM physical judgment;
- `B1_OPEN_SAM2`: the frozen Revision B deterministic PAVG critic, with official SAM2.1 Hiera B+ dense tracks and the same Qwen model used for object-seed proposals.

The model, prompt hashes, frame count, SAM2 source/checkpoint hashes, vLLM version and decoding parameters are identical or explicitly separated in resolved configuration. Qwen3-VL-8B-Instruct replaces the originally proposed Qwen2.5-VL-7B after the user reported inadequate Qwen2.5-VL quality. The 8B dense model remains small enough to share an A100 40GB with SAM2 when vLLM is capped at 50% GPU-memory utilization. Official Qwen materials describe improved visual perception, spatial/video dynamics and reasoning; official vLLM support begins at v0.11.0.

### Secondary controls

- `D1_OPEN_STRUCTURED`: the existing structured-checklist direct baseline on the 300-sample pilot and, if runtime remains within the 72-hour gate, on the full set.
- `D0_QWEN25_WEAK`: Qwen2.5-VL-7B direct baseline on the frozen pilot only, retained to quantify the backbone upgrade rather than support the primary claim.
- `A0_VIDEOPHY_AUTO`: released official auto-evaluator scores aligned by source URL or sample identity when alignment is unambiguous.
- `D0_CLOSED` and `B1_CLOSED_SAM2`: matched `gpt-5-mini` audit on a frozen 300-sample stratified pilot. This anchors the open-model result without committing thousands of paid calls.

If Qwen3-VL cannot serve the existing image-data-URL contract after two documented compatibility fixes, the matched open block falls back first to `Qwen/Qwen2.5-VL-7B-Instruct`, then to `gpt-5-mini`. Terra and Luna are sensitivity models only and do not replace the primary matched block unless `gpt-5-mini` is unavailable.

## 4. Execution architecture

1. Establish key-based SSH access; retain password access only as break-glass recovery.
2. Audit GPU occupancy, CUDA, disk, RAM, Python environments and network access without exposing unrelated process arguments.
3. Transfer a clean git bundle plus a manifest of any required uncommitted benchmark files. Never transfer `.env`, raw API payloads or user credentials.
4. Create `/root/pavg-benchmark` with separate `src`, `data`, `models`, `runs` and `logs` directories.
5. Create a Python 3.12 environment, install the project, official SAM2 source at commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`, CUDA PyTorch and vLLM-compatible dependencies.
6. Run the full unit suite and a real three-frame SAM2 propagation test before downloading the full video corpus.
7. Materialize datasets with resumable, idempotent downloads and per-file SHA-256 checksums. Write failures append-only.
8. Run server smoke20, then a frozen 300-sample generator/action/label-stratified pilot.
9. Enter the full run only if pilot failure rate is below 5%, no GPU OOM occurs, dense frame coverage is at least 0.95 for at least 95% of samples, and projected wall time is at most 72 hours.
10. Run append-only full predictions under a process lock. A watchdog records progress, GPU memory, failures and estimated completion time without printing secrets.
11. Freeze VideoPhy-2 configuration and apply it unchanged to VideoPhy-1 OOD.
12. Synchronize manifests, resolved configurations, prediction JSONL, summaries and non-secret logs back to the local workspace.

## 5. Resource and failure policy

- vLLM initially reserves at most 50% of A100 memory; SAM2 uses the remaining memory. Reduce vLLM utilization once if the smoke produces OOM.
- The first open-model fallback is lower vLLM memory utilization and shorter maximum context, not a model or prompt change.
- The second compatibility fallback is `Qwen/Qwen2.5-VL-7B-Instruct` with the same prompts and frame sampling.
- After two failed compatibility fixes, use the closed-model matched block and record the open-model failure as a negative engineering result.
- Downloads, observation caches and predictions are resumable. Existing valid sample×method keys are skipped; duplicates are rejected.
- No sample is silently removed. Metrics report all-sample and successfully decoded subsets side by side.

## 6. Metrics and decision rule

Report accuracy, balanced accuracy, Macro-F1, physical recall, violation recall, violation precision, Spearman correlation, unknown rate, failure rate and mean/p50/p95 latency.

Compute paired 95% confidence intervals with 2,000 action-group bootstrap resamples. Report paired correctness deltas and generator/action/rule-family slices. The full result is considered materially better only if:

1. `B1_OPEN_SAM2 - D0_OPEN_DIRECT` Macro-F1 is at least +0.05;
2. the paired group-bootstrap 95% confidence interval for the Macro-F1 delta excludes zero;
3. neither class recall is zero and failure rate increases by no more than one percentage point;
4. the delta is positive on VideoPhy-1 OOD and is not supported by only one generator.

Failure to meet the rule is reported as a negative result; it does not trigger full-test threshold tuning.

## 7. Auditable outputs

Local final artifacts live under:

- `evaluation/manifests/videophy2_test_full.json`;
- `evaluation/manifests/videophy1_test_full.json`;
- `outputs/benchmarks/server-audit/`;
- `outputs/benchmarks/videophy2-full-qwen3vl8b/`;
- `outputs/benchmarks/videophy1-ood-qwen3vl8b/`;
- `docs/results/criticbenchmark.md`;
- `docs/superpowers/plans/2026-07-16-full-videophy-server-evaluation.md`.

Artifacts must contain source/checkpoint/manifest hashes, exact model IDs, command configuration, package versions, runtime/failure summaries and secret scans. Videos, model weights, `.env` files and credentials remain ignored.

## 8. Self-review

- No placeholder dataset scope: VideoPhy-2 is exactly 3,397 rows; VideoPhy-1 is all accessible official human-annotated test rows and must be counted before inference.
- The primary causal comparison is matched-backbone and full-population.
- The 300-sample pilot is a reliability/cost gate, not a substitute for the full run.
- Fallbacks are ordered and finite; failed open-model attempts remain documented.
- Full-test outcomes cannot alter architecture, rules, thresholds, prompts or sample membership.
