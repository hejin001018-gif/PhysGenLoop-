# Prompted Critic Server Validation Plan

**Date:** 2026-07-17
**Time box:** 8 hours
**Goal:** Validate the prompt-conditioned complete Critic on a frozen VideoPhy-2 subset using the main A100 server, fix reproducible code defects, and leave an auditable resumable result without changing the accepted evaluation configuration.

## Frozen contract

- Local source: `sy` at `c8d18109e65cbd83f15315a0d6f3daa75c2667f6`.
- Server: `qe74VL`, existing root `/root/benchmark/pavg-benchmark`; do not create a new top-level directory under `/root`.
- Model: the existing `Qwen/Qwen3-VL-8B-Instruct` snapshot served by vLLM 0.11.0 with deterministic decoding and the previously verified 0.58 GPU-memory fraction.
- Vision evidence: reuse the accepted official SAM2.1 Hiera B+ observation cache. Do not propagate the 3,397 videos through SAM2 again.
- Dataset: the previously frozen VideoPhy-2 smoke20 membership, 10 physical / 10 violation, seven generators, and a non-empty generation prompt for every sample. Resolve each member against the accepted full manifest and existing video/cache paths before inference.
- Primary method: `M5_FULL`, with model-backed Planner, hybrid PQSG, deterministic rules, VideoScience Checklist, mechanics, grouped VLM verification, and the frozen coverage-aware fusion configuration.
- The primary method may read video, prompt, identifiers, and cached observations only. Ground-truth labels remain in the metric layer and must not enter model prompts or Critic inputs.

## Execution tasks

### Task 1 — Audit before write

- Record GPU, disk, source revision/dirty state, active processes, model/venv versions, dataset counts, observation-cache counts, and manifest/checkpoint hashes.
- Preserve the legacy dirty server source and record its patch/status before installing the validated `sy` snapshot.

### Task 2 — Install a reproducible source snapshot

- Create a git bundle from the pushed `sy` commit, transfer it with a SHA-256 sidecar, and clone it into a versioned directory below `/root/benchmark/pavg-benchmark`.
- Verify the remote source commit and tree status, then run the complete test suite with an isolated basetemp.
- Use `PYTHONPATH`/working-directory isolation so the validation cannot import the legacy dirty source accidentally.

### Task 3 — Freeze the prompted subset

- Join smoke20 membership to the accepted full manifest by `sample_id`.
- Require exactly 20 unique rows, 20 non-empty prompts, 20 existing videos, 20 valid observation JSON files, 10/10 labels, and seven generators.
- Write the resolved manifest and checksum before any M5 prediction.

### Task 4 — Bring up and smoke the local model

- Start the existing Qwen3-VL snapshot on `127.0.0.1:8000` with request-body logging disabled.
- Verify endpoint health and one schema-constrained request before the benchmark.
- Run one M5 sample first. Inspect prediction, diagnostics, all three model-stage events, GPU peak memory, and provider failures.

### Task 5 — Run smoke20 and close defects

- Run `M5_FULL` through the crash-recoverable prediction/diagnostic journal and immutable Planner/PQSG/Verifier caches.
- Required operational gates: 20/20 terminal predictions and matching diagnostics, zero duplicate or asymmetric keys, zero provider failures, zero OOM, non-zero Planner/PQSG/Verifier calls, finite scores, and no label/rule leakage.
- For each defect: reproduce it, add a failing regression test locally, implement the smallest fix, run focused plus full tests, push the fix to `sy`, update the versioned server snapshot, and resume only from valid cache/journal keys.
- Do not tune prompts, thresholds, family weights, hard-violation policy, sample membership, or labels in response to smoke outcomes.

### Task 6 — Report and preserve evidence

- Record per-sample terminal status, decisions, class metrics as diagnostic-only, module availability, provider failure rate, cache hits, latency, GPU peak, and total wall time.
- Scan artifacts for credentials, authorization headers, `.env` content, image data, and raw provider payloads.
- Synchronize only manifests, resolved configuration, predictions, diagnostics, summaries, and non-secret logs to the local result directory.
- Append the exact commands, hashes, defects, fixes, and verification evidence to this plan under `Execution results`.

## Time budget

| Phase | Ceiling |
|---|---:|
| Read-only audit and plan freeze | 0.5 h |
| Source transfer and remote full tests | 1.0 h |
| vLLM start and one-sample smoke | 1.0 h |
| smoke20 inference | 2.0 h |
| Defect investigation/fixes and reruns | 2.5 h |
| Artifact audit, synchronization, report | 1.0 h |

If smoke20 passes early, use the remaining time only for a larger deterministic prompt subset with the same frozen configuration. Do not expand scope if doing so risks the required smoke20 report.

## Execution results

Results are appended after each checkpoint. Existing entries are immutable once recorded.

### E1 — Read-only main-server audit

- Server `qe74VL` is reachable through the dedicated SSH key. The accepted ED25519 host fingerprint for `[px-cloud2.matpool.com]:29848` is `SHA256:Lbqq8S4pvO7r+XsZdW3Ia1rsQMXVKGEu8snKKWvgjEA`.
- GPU: one idle NVIDIA A100-PCIE-40GB, 40,960 MiB; initial utilization and allocated memory were both zero.
- Storage: `/root/benchmark/pavg-benchmark` is on a 200GB filesystem with 125GB free.
- Existing assets: 3,397 VideoPhy-2 videos, 3,397 consolidated SAM2 observation caches, Qwen3-VL-8B weights, official SAM2.1 checkpoint, Python 3.12.13, PyTorch 2.8.0+cu128, and vLLM 0.11.0.
- The legacy server checkout is at `2210e16d5d0e5123a383e35bbc80da9e0c0b1a98` with historical tracked modifications. It will be preserved rather than reset or overwritten.

### E2 — Reproducible source and moved-environment repair

- Local bundle SHA-256 `b7ecdfac5b3673d263647cde7663d5868de016b3fca33806fa6051a29084b673` matched remotely. The clean versioned checkout is `/root/benchmark/pavg-benchmark/src-sy-c8d1810`, exact commit `c8d18109e65cbd83f15315a0d6f3daa75c2667f6`.
- The first remote suite passed 379 tests and failed three environment checks. The moved virtual environment still contained editable-install paths below the old `/root/pavg-benchmark`: one subprocess could not import `pavg_critic`, and two tests could not import official `sam2`.
- Rebinding only the `pavg-critic` and official `sam-2` editable installs to their new locations fixed the environment. The second remote suite passed `382/382` in 2.90 seconds and compileall passed.

### E3 — Frozen prompted smoke20

- Resolved manifest: `/root/benchmark/pavg-benchmark/runs/prompted-critic-smoke20-c8d1810/manifest.json`.
- SHA-256: `921239725003268e2fbd45b931a561e1bee486b3bb53dce674fa9e955fff762f`.
- Validation: 20 unique samples, 20 non-empty prompts, 20 readable videos, 20 non-empty existing SAM2 caches, 10 physical / 10 violation, and seven generators.
- The first inline vLLM launch attempt exited before model load because Windows-to-SSH quoting damaged JSON-valued CLI arguments. A transferred no-secret Bash script with SHA-256 `049abd64de759ff338c8d55b4f42f0ab98aca386aec44c38024ae39d269672a9` started the exact frozen service successfully. Idle residency was about 20.4GiB.

### E4 — One-sample functional gate and Planner defect

- The first M5 sample terminated in 7.31 seconds with no provider error and one successful call at each Planner, PQSG and Verifier boundary. Verifier rejected/softened an `object_disappearance` candidate and the prediction was physical.
- The functional gate correctly failed despite the terminal prediction: Planner returned a schema-valid all-empty plan and was recorded as `source=empty`; PQSG consequently had zero nodes. The cached response confirmed all four arrays were empty, so this was model output accepted too permissively rather than a parser or transport failure.
- A regression test was added before implementation and failed because the empty output was accepted after one call. The fix rejects an all-empty plan only when the generation prompt is non-empty and no authoritative partial plan exists, issues one repair request with bounded feedback, and explicitly instructs the model to extract at least one physically relevant entity. Empty prompts retain their previous valid empty-plan behavior.
- Planner tests passed `34/34`. The first local full run had one unrelated Windows atomic-directory rename denial; that exact test passed alone on a fresh basetemp, then the complete suite passed `383/383` in 8.79 seconds and compileall passed.
