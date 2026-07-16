# Full VideoPhy Server Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run an auditable full VideoPhy-2 matched open-model comparison on the rented A100 server, then apply the frozen configuration to VideoPhy-1 OOD and synchronize complete results locally.

**Architecture:** Serve Qwen2.5-VL-7B through a memory-bounded local vLLM endpoint while the frozen Revision B critic uses official SAM2.1 Hiera B+ for dense tracks. Use append-only manifests, observation caches and predictions so downloads and multi-day inference can resume safely. Closed `gpt-5-mini` is a frozen 300-sample audit anchor and a full-run fallback only after two documented open-model compatibility failures.

**Tech Stack:** Python 3.12, PyTorch CUDA, official Meta SAM2.1, vLLM, Qwen2.5-VL-7B-Instruct, OpenCV, pytest, Paramiko/OpenSSH, VideoPhy-2 and VideoPhy-1.

---

## Task 1: Freeze design, source state and local baseline

**Files:**
- Create: `docs/superpowers/specs/2026-07-16-full-videophy-server-evaluation-design.md`
- Create: `docs/superpowers/plans/2026-07-16-full-videophy-server-evaluation.md`
- Modify: this plan under `Execution results`

- [x] Record the local commit, dirty-file list, Python version, 159-test baseline, VideoPhy-2 CSV SHA-256 and exact row/label/generator counts.
- [x] Verify the new design has no `TBD`, `TODO` or unresolved sample/model choices.
- [ ] Commit only the new spec and plan; preserve unrelated user changes.

## Task 2: Establish secure remote access and complete server audit

**Files:**
- Create: `outputs/benchmarks/server-audit/server.json`
- Modify: this plan under `Execution results`

- [ ] Generate a dedicated local ED25519 key outside the repository if it does not exist.
- [ ] Append only that public key to remote `/root/.ssh/authorized_keys`, set SSH directory/file permissions, and verify key-only login in a fresh connection.
- [ ] Record hostname, OS, CPU count, RAM, disk, GPU name/memory/driver, CUDA runtime, Python/conda executables, GPU utilization and non-sensitive process names.
- [ ] Probe access to the official GitHub, Hugging Face and VideoPhy S3 endpoints from the server.
- [ ] Redact secrets and unrelated command lines from the saved audit.

## Task 3: Transfer a reproducible source snapshot

**Files:**
- Create remote: `/root/pavg-benchmark/src`
- Create remote: `/root/pavg-benchmark/artifacts/source-manifest.json`
- Modify: this plan under `Execution results`

- [ ] Create a git bundle at commit `ce004ff` plus an explicit tar archive containing only required uncommitted benchmark source/docs identified by a reviewed file list.
- [ ] Transfer the bundle/archive with SHA-256 sidecars; never include `.env`, videos, outputs, caches or credentials.
- [ ] Clone the bundle remotely, apply the explicit overlay, and record the resulting source-tree hash and `git status --short`.
- [ ] Verify remote imports use the intended source tree.

## Task 4: Build and verify the remote Python/SAM2 environment

**Files:**
- Create remote: `/root/pavg-benchmark/venv`
- Create remote: `/root/pavg-benchmark/logs/environment.txt`
- Modify: this plan under `Execution results`

- [ ] Create or locate Python 3.12; create an isolated virtual environment.
- [ ] Install CUDA PyTorch, the project benchmark/SAM2 extras, official SAM2 at the frozen commit and the pinned checkpoint.
- [ ] Record `pip freeze`, PyTorch/CUDA/cuDNN versions and checkpoint/source SHA-256 values.
- [ ] Run the full pytest suite and record exact pass/fail counts.
- [ ] Run the real three-frame SAM2 propagation test and require one continuous track across all frames.

## Task 5: Materialize and freeze the full VideoPhy-2 dataset

**Files:**
- Create remote: `/root/pavg-benchmark/data/videophy2/videophy2_test.csv`
- Create remote: `/root/pavg-benchmark/data/videophy2/videos/`
- Create local: `evaluation/manifests/videophy2_test_full.json`
- Modify: this plan under `Execution results`

- [ ] Transfer the already frozen 3,397-row official CSV and verify its SHA-256 remotely.
- [ ] Download every unique video URL with idempotent retries and append-only failure records.
- [ ] Decode-probe every video, record frame count/duration/size/checksum, and retry corrupt or partial files once.
- [ ] Normalize all rows into an immutable full manifest; retain failed rows with explicit status rather than dropping them.
- [ ] Freeze a deterministic 300-sample action/generator/label-stratified pilot manifest before model predictions.

## Task 6: Deploy the open model and pass server smoke

**Files:**
- Create remote: `/root/pavg-benchmark/models/Qwen2.5-VL-7B-Instruct/`
- Create remote: `/root/pavg-benchmark/runs/videophy2-server-smoke20/`
- Modify: this plan under `Execution results`

- [ ] Install a vLLM version compatible with the selected CUDA/PyTorch stack and download the official Qwen model snapshot.
- [ ] Start an OpenAI-compatible endpoint with GPU utilization at most 0.50, deterministic decoding, bounded context and no request-body logging.
- [ ] Run one schema-only image request and verify the existing chat adapter parses it.
- [ ] Run smoke20 `D0_OPEN_DIRECT,B1_OPEN_SAM2`; require no OOM, duplicate keys or credential-bearing logs.
- [ ] If compatibility fails, apply only the finite fallbacks in the design and document each attempt.

## Task 7: Run the frozen 300-sample pilot

**Files:**
- Create remote: `/root/pavg-benchmark/runs/videophy2-pilot300-qwen25vl7b/`
- Modify: this plan under `Execution results`

- [ ] Run `D0_OPEN_DIRECT,D1_OPEN_STRUCTURED,B1_OPEN_SAM2` with append-only predictions and SAM2 observation cache.
- [ ] Record download/decode/inference failure rates, frame coverage, GPU peak memory, throughput, p50/p95 latency and projected full-run time.
- [ ] Run the matched `gpt-5-mini` D0/B1 audit on the same frozen pilot only if a secret can be injected without repository or shell-history persistence.
- [ ] Enter the full run if failure rate is below 5%, no OOM occurs, coverage meets the design gate and projected wall time is at most 72 hours; otherwise apply the specified finite fallback and repeat smoke/pilot.

## Task 8: Run all 3,397 VideoPhy-2 samples

**Files:**
- Create remote: `/root/pavg-benchmark/runs/videophy2-full-qwen25vl7b/`
- Create local: `outputs/benchmarks/videophy2-full-qwen25vl7b/`
- Modify: this plan under `Execution results`

- [ ] Launch the full matched D0/B1 matrix in a resumable session with one process owning the prediction lock.
- [ ] Monitor progress, GPU state, failure count and ETA at least once per hour without altering prompts, thresholds or sample membership.
- [ ] Resume interrupted work from valid sample×method keys until every manifest row has a terminal prediction for both methods.
- [ ] Generate summaries, 2,000 action-group bootstrap confidence intervals, paired outcomes and generator/action/rule-family slices.
- [ ] Apply the material-improvement arithmetic exactly as frozen in the design.

## Task 9: Run frozen VideoPhy-1 OOD

**Files:**
- Create local: `evaluation/manifests/videophy1_test_full.json`
- Create local: `outputs/benchmarks/videophy1-ood-qwen25vl7b/`
- Modify: this plan under `Execution results`

- [ ] Retrieve the official public test metadata, record row count/schema/checksum and materialize every accessible video.
- [ ] Adapt the dataset without changing the VideoPhy-2-frozen PAVG/model configuration.
- [ ] Run the full matched D0/B1 matrix with the same failure and resume policy.
- [ ] Report OOD delta, group-bootstrap interval, material-interaction slices and whether the sign agrees with VideoPhy-2.

## Task 10: Synchronize, audit and report

**Files:**
- Modify: `docs/results/criticbenchmark.md`
- Modify: this plan under `Execution results`
- Create: `outputs/benchmarks/server-audit/artifact-audit.json`

- [ ] Synchronize manifests, predictions, summaries, resolved configs and non-secret logs; do not copy videos or model weights into git-tracked paths.
- [ ] Verify manifest/prediction key alignment, duplicate absence, checksums and terminal status for every sample×method pair.
- [ ] Scan synchronized artifacts for the SSH password, API key prefixes, authorization headers, `.env` contents and raw provider payloads.
- [ ] Run the local full pytest suite and regenerate the Chinese result narrative with exact tables, confidence intervals, runtime and negative results.
- [ ] Commit only source, tests, manifests and documentation; preserve user-owned dirty files.

## Execution results

Results are appended here after every task checkpoint. Existing results are immutable once recorded.

### E1 — Stage B contract and local baseline

- Source commit: `ce004ff9021e1266fefb00017a7a38e75cf94c87`; branch `sy`; normal checkout used because the user previously selected direct execution in the current workspace.
- Unrelated dirty files were inventoried and left untouched: `.env.example`, `README.md`, `src/pavg_critic/__init__.py`, `src/pavg_critic/api_models.py`, `test.py`, `docs/results/`, `src/pavg_critic/vlm_detector.py`, plus the prior iteration plan.
- Python: `3.12.10` from `.venv`.
- The first pytest run produced 37 setup errors because `C:\Users\sy\AppData\Local\Temp\pytest-of-sy` denied directory enumeration. This was an environment/ACL failure, not a test assertion failure. Re-running with ignored basetemp `outputs/.pytest-tmp` passed `159/159` in 3.64 seconds.
- Frozen VideoPhy-2 CSV SHA-256: `85a6690b9508b7e69c592f3cbcbc4113efd3a573eb5ec69d6ae030a8ffb8a4e7`.
- Population: 3,397 unique URLs, 198 actions, 1,785 physical / 1,612 violation; Wan 591, VideoCrafter 591, CogVideo 589, Hunyuan 587, Cosmos 585, Ray2 394 and Sora 60.
- Design self-review found no placeholders or unresolved primary method/data choices.
