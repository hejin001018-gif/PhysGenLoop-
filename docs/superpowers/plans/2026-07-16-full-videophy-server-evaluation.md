# Full VideoPhy Server Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run an auditable full VideoPhy-2 matched open-model comparison on the rented A100 server, then apply the frozen configuration to VideoPhy-1 OOD and synchronize complete results locally.

**Architecture:** Serve Qwen3-VL-8B-Instruct through a memory-bounded local vLLM endpoint while the frozen Revision B critic uses official SAM2.1 Hiera B+ for dense tracks. Use append-only manifests, observation caches and predictions so downloads and multi-day inference can resume safely. Qwen2.5-VL-7B is a weak pilot baseline only; closed `gpt-5-mini` is a frozen 300-sample audit anchor and a full-run fallback only after two documented Qwen3 compatibility failures.

**Tech Stack:** Python 3.12, PyTorch CUDA 12.8, official Meta SAM2.1, vLLM 0.11.0, Qwen3-VL-8B-Instruct, OpenCV, pytest, Paramiko/OpenSSH, VideoPhy-2 and VideoPhy-1.

---

## Task 1: Freeze design, source state and local baseline

**Files:**
- Create: `docs/superpowers/specs/2026-07-16-full-videophy-server-evaluation-design.md`
- Create: `docs/superpowers/plans/2026-07-16-full-videophy-server-evaluation.md`
- Modify: this plan under `Execution results`

- [x] Record the local commit, dirty-file list, Python version, 159-test baseline, VideoPhy-2 CSV SHA-256 and exact row/label/generator counts.
- [x] Verify the new design has no `TBD`, `TODO` or unresolved sample/model choices.
- [x] Commit only the new spec and plan; preserve unrelated user changes.

## Task 2: Establish secure remote access and complete server audit

**Files:**
- Create: `outputs/benchmarks/server-audit/server.json`
- Modify: this plan under `Execution results`

- [x] Generate a dedicated local ED25519 key outside the repository if it does not exist.
- [x] Append only that public key to remote `/root/.ssh/authorized_keys`, set SSH directory/file permissions, and verify key-only login in a fresh connection.
- [x] Record hostname, OS, CPU count, RAM, disk, GPU name/memory/driver, CUDA runtime, Python/conda executables, GPU utilization and non-sensitive process names.
- [x] Probe access to the official GitHub, Hugging Face and VideoPhy S3 endpoints from the server.
- [x] Redact secrets and unrelated command lines from the saved audit.

## Task 3: Transfer a reproducible source snapshot

**Files:**
- Create remote: `/root/pavg-benchmark/src`
- Create remote: `/root/pavg-benchmark/artifacts/source-manifest.json`
- Modify: this plan under `Execution results`

- [x] Create a git bundle at the committed Stage B head plus an explicit overlay containing only required uncommitted benchmark source identified by a reviewed file list.
- [x] Transfer the bundle/archive with SHA-256 sidecars; never include `.env`, videos, outputs, caches or credentials.
- [x] Clone the bundle remotely, apply the explicit overlay, and record the resulting source-tree hash and `git status --short`.
- [x] Verify remote imports use the intended source tree.

## Task 4: Build and verify the remote Python/SAM2 environment

**Files:**
- Create remote: `/root/pavg-benchmark/venv`
- Create remote: `/root/pavg-benchmark/logs/environment.txt`
- Modify: this plan under `Execution results`

- [x] Create or locate Python 3.12; create an isolated virtual environment.
- [x] Install CUDA PyTorch, the project benchmark/SAM2 extras, official SAM2 at the frozen commit and the pinned checkpoint.
- [x] Record `pip freeze`, PyTorch/CUDA/cuDNN versions and checkpoint/source SHA-256 values.
- [x] Run the full pytest suite and record exact pass/fail counts.
- [x] Run the real three-frame SAM2 propagation test and require one continuous track across all frames.

## Task 5: Materialize and freeze the full VideoPhy-2 dataset

**Files:**
- Create remote: `/root/pavg-benchmark/data/videophy2/videophy2_test.csv`
- Create remote: `/root/pavg-benchmark/data/videophy2/videos/`
- Create local: `evaluation/manifests/videophy2_test_full.json`
- Modify: this plan under `Execution results`

- [x] Transfer the already frozen 3,397-row official CSV and verify its SHA-256 remotely.
- [x] Download every unique video URL with idempotent retries and append-only failure records.
- [x] Decode-probe every video, record frame count/duration/size/checksum, and retry corrupt or partial files once.
- [x] Normalize all rows into an immutable full manifest; retain failed rows with explicit status rather than dropping them.
- [x] Freeze a deterministic 300-sample action/generator/label-stratified pilot manifest before model predictions.

## Task 6: Deploy the open model and pass server smoke

**Files:**
- Create remote: `/root/pavg-benchmark/models/Qwen3-VL-8B-Instruct/`
- Create remote: `/root/pavg-benchmark/runs/videophy2-server-smoke20/`
- Modify: this plan under `Execution results`

- [x] Install a vLLM version compatible with the selected CUDA/PyTorch stack and download the official Qwen model snapshot.
- [x] Start an OpenAI-compatible endpoint at the measured minimum 0.58 GPU-memory fraction, deterministic decoding, bounded context and no request-body logging; 0.50 cannot allocate any KV cache for this BF16 model.
- [x] Run one schema-only image request and verify the existing chat adapter parses it.
- [x] Run smoke20 `D0_OPEN_DIRECT,B1_OPEN_SAM2`; require no OOM, duplicate keys or credential-bearing logs.
- [x] If compatibility fails, apply only the finite fallbacks in the design and document each attempt.

## Task 7: Run the frozen 300-sample pilot

**Files:**
- Create remote: `/root/pavg-benchmark/runs/videophy2-pilot300-qwen3vl8b/`
- Modify: this plan under `Execution results`

- [x] Run `D0_DIRECT_VLM,D1_STRUCTURED_VLM,B1_RULE` with append-only predictions and SAM2 observation cache.
- [x] Record download/decode/inference failure rates, frame coverage, GPU peak memory, throughput, p50/p95 latency and projected full-run time.
- [ ] Run the matched `gpt-5-mini` D0/B1 audit on the same frozen pilot only if a secret can be injected without repository or shell-history persistence.
- [x] Enter the full run gate: failure rate was 0%, no OOM occurred, observation coverage was 300/300 and the pilot-derived full-run projection was approximately 37.5 hours, below the 72-hour limit.

## Task 8: Run all 3,397 VideoPhy-2 samples

**Files:**
- Create remote: `/root/pavg-benchmark/runs/videophy2-full-qwen3vl8b/`
- Create local: `outputs/benchmarks/videophy2-full-qwen3vl8b/`
- Modify: this plan under `Execution results`

- [x] Launch the full matched D0/B1 matrix in a resumable session with one process owning the prediction lock.
- [ ] Monitor progress, GPU state, failure count and ETA at least once per hour without altering prompts, thresholds or sample membership.
- [ ] Resume interrupted work from valid sample×method keys until every manifest row has a terminal prediction for both methods.
- [ ] Generate summaries, 2,000 action-group bootstrap confidence intervals, paired outcomes and generator/action/rule-family slices.
- [ ] Apply the material-improvement arithmetic exactly as frozen in the design.

## Task 9: Run frozen VideoPhy-1 OOD

> **Deferred on 2026-07-17 by user request.** Finish and report the complete
> VideoPhy-2 evaluation first; VideoPhy-1 OOD is outside the current execution
> scope and remains an explicit future task rather than being treated as done.

**Files:**
- Create local: `evaluation/manifests/videophy1_test_full.json`
- Create local: `outputs/benchmarks/videophy1-ood-qwen3vl8b/`
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
- Design/plan commit: `2210e16d5d0e5123a383e35bbc80da9e0c0b1a98`. An initially attempted commit inherited the user's already-staged `.env.example`; the commit was immediately amended to remove it, and `.env.example` was restored to its original staged state. Verification shows the final commit contains only the two new documentation files.

### E2 — SSH key and server audit

- Dedicated ED25519 key fingerprint: `SHA256:y6o4iKZ7CNkupn7RgRkoQyXBe3jE0HshPOfeOI/eVbo`; the private key is outside the repository. A fresh connection with password/agent lookup disabled authenticated as `root`.
- Host `qe74VL`: 12 CPUs, 90 GiB RAM (86 GiB available), 200 GiB root disk (191 GiB available).
- GPU: one idle NVIDIA A100-PCIE-40GB, 40,960 MiB, driver `570.211.01`; no compute processes; `nvcc` is absent, so the environment must use wheel-bundled CUDA runtime.
- Runtime: system Python 3.8.10; no conda. The isolated environment will be bootstrapped with `uv` and Python 3.12.
- Network: official GitHub raw and VideoPhy S3 are reachable; direct Hugging Face TLS is reset. `hf-mirror.com`, official Qwen ModelScope endpoints and the `uv` installer are reachable, so the frozen CSV/model can be obtained without changing the protocol.
- Audit artifact: `outputs/benchmarks/server-audit/server.json`; it contains no credential or unrelated process arguments.

### E3 — Reproducible remote source

- The transferred bundle uses source commit `2210e16d5d0e5123a383e35bbc80da9e0c0b1a98` rather than the older plan-header checkpoint `ce004ff`, because `2210e16` adds only the approved Stage B documents. Bundle SHA-256: `b785d199164f3067532c9701a64a9a2456fedd4cdec92580c875335475a53e0b`; committed tree: `4d44a036e9070a4aedb9f15606c9143263ec3f0d`.
- The clean snapshot failed test collection because the committed benchmark CLI imports `OpenAIChatModel`, while that required class exists only in the user's uncommitted `src/pavg_critic/api_models.py`. The exact file was added as the sole overlay after evidence of the import failure; overlay SHA-256: `b31206424bd68463b467900b859d3d958e9d1179f08268019d7be12164d1d154`.
- Remote `git status --porcelain` contains exactly `M src/pavg_critic/api_models.py`. The remote source manifest records the commit, bundle hash, overlay path/hash/reason and exclusions.
- No `.env`, videos, output caches, credentials, README/test.py changes or other uncommitted files were transferred.

### E4 — Remote CUDA/SAM2 verification

- Environment: uv `0.11.29`, CPython `3.12.13`, PyTorch `2.7.1+cu128`, torchvision `0.22.1+cu128`, CUDA runtime `12.8`, cuDNN `90701`, OpenCV `5.0.0`, NumPy `2.5.1`.
- Official SAM2 source: `2b90b9f5ceec907a1c18123530e92e794ad901a4`; checkpoint SHA-256: `a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5`.
- The first two editable-install attempts stalled because isolated build dependencies re-downloaded Torch and remote output was not drained. The successful installation used the existing verified Torch environment, `--no-build-isolation` and official `SAM2_BUILD_CUDA=0`; the optional post-processing extension is absent because `nvcc` is unavailable.
- Full remote test suite after the required overlay: `159 passed in 2.01s`.
- Real A100 propagation on a three-frame moving-square video produced exactly one stable `sam2:0` track per frame with boxes `(20,48,39,79)`, `(40,48,59,79)` and `(60,48,79,79)`. The missing optional `_C` post-processing warning was recorded; propagation itself passed.
- Frozen package inventory is stored at `/root/pavg-benchmark/logs/environment.txt`.

### E5 — Open-model upgrade and CUDA compatibility

- The initial latest `vLLM 0.25.1` environment resolved Torch `2.11.0+cu130`. Import succeeded, but the required real CUDA tensor operation failed because driver 570 exposes CUDA 12.8 and cannot execute CUDA 13.0 binaries.
- A CUDA 12.8 `vLLM 0.10.2` installation was started, then stopped after the user rejected Qwen2.5-VL quality and the official compatibility matrix showed that v0.10.2 does not support `Qwen3VLForConditionalGeneration`.
- Official vLLM matrices show Qwen3-VL support begins at v0.11.0. The selected environment is `vLLM 0.11.0`, Torch `2.8.0+cu128`, ModelScope `1.38.1`; a real A100 matrix multiplication returned `262144.0`.
- Primary open backbone is now `Qwen/Qwen3-VL-8B-Instruct` (Apache-2.0). Qwen2.5-VL is limited to a pilot weak baseline. Model weights are downloading from the official Qwen ModelScope repository because direct Hugging Face TLS is unavailable on the server.

### E6 — Qwen3-VL service bring-up

- The official ModelScope snapshot completed with 17 repository files and four complete safetensor shards. JSON configuration/index validation passed; nine SHA-256 entries covering the configuration, tokenizer/index metadata and all weight shards are frozen at `/root/pavg-benchmark/artifacts/qwen3vl8b-sha256.txt`.
- `vLLM 0.11.0` initially resolved the future `transformers 5.14.0`, which failed because vLLM accessed a tokenizer attribute removed in Transformers 5. The model's own README identifies `transformers 4.57.0`; pinning that exact release resolved tokenizer initialization without changing Torch or vLLM.
- A 50% GPU-memory limit was insufficient: after 16.64 GiB of weights and multimodal profiling it left `-0.67 GiB` for KV cache. At 55%, vLLM measured 1.31 GiB available versus 2.25 GiB required for one 16,384-token request. The minimum verified configuration is therefore 58%, which provides 2.49 GiB / 18,128 KV tokens and 1.11× concurrency at 16,384 tokens.
- The service accepts at most 16 image inputs and no native video input; frames remain deterministically decoded by the benchmark. The processor is capped at 1,003,520 pixels per image. Service idle residency is about 20.35 GiB.
- A real 480×720 VideoPhy-2 first-frame request through the project adapter returned four visible objects in 4.52 seconds. A real 16-frame request exercised all 16 requested frames without OOM.

### E7 — Strict structured output and joint SAM2 smoke

- The first 16-frame direct request returned the JSON Schema definition rather than a score instance, causing a correctly captured `KeyError('semantic_score')`. The raw response ended normally after 145 completion tokens, proving this was schema adherence rather than truncation.
- A test-first adapter change added an explicit `--chat-response-format json_schema` mode while leaving the compatible `json_object` default unchanged. Both new tests failed before implementation, passed after it, and the complete local suite passed `161/161` in 6.08 seconds.
- The same 16-frame request under strict schema returned semantic `5`, physics `5`, confidence `1.0`, all 16 evidence-frame indices and no failure in 1.87 seconds.
- A one-sample joint run completed Qwen3-VL object seeding, official SAM2 propagation over all 49 frames, direct judging and B1 critic scoring. Peak observed GPU memory was about 23.2 GiB; SAM2 propagation took about 16 seconds and no OOM occurred.
- This first sample is human-labelled physical. D0 predicted physical correctly, while B1 predicted violation from `object_disappearance`. This is recorded as an initial critic false positive, not evidence of improvement; the frozen balanced smoke20 run is required before architecture changes.

### E8 — Frozen Qwen3-VL smoke20 result

- The pre-existing frozen smoke manifest contains 20 checksum-verified videos: 10 physical / 10 violation, spanning CogVideo, Hunyuan, Cosmos, VideoCrafter, Sora, Ray2 and Wan. Remote manifest SHA-256: `8156d04b04c7f0966794ab3a99520a6abc28e639302eaf6f40f330b8fe174461`.
- The run completed 20/20 SAM2 observation caches and 140/140 sample×method predictions with zero failures, no duplicate keys and no OOM. Peak observed GPU memory was about 23.46 GiB.
- On all 20 samples, D0 direct and D1 structured each reached accuracy `0.55` and Macro-F1 `0.549`. B1, M1, M2 and M3 each reached accuracy `0.70` and Macro-F1 `0.697`, a smoke delta of `+0.15` accuracy and `+0.148` Macro-F1 over D0. B1 violation precision/recall were `0.667/0.800`; D0 was `0.545/0.600`.
- M4 VLM fusion collapsed to predicting no violations: accuracy `0.50`, Macro-F1 `0.333`, violation recall `0.0`. It is a recorded negative result and is not the primary critic method.
- The frozen dev10 split favoured D0 (accuracy `0.70`, Macro-F1 `0.697`) over B1 (`0.60`, `0.600`). The untouched eval10 split favoured B1 (`0.80`, `0.792`, violation recall `1.0`) over D0 (`0.40`, `0.400`). Because the aggregate and eval directions favour B1 but the sample is small, no rule tuning is justified; proceed to the frozen 300-sample pilot.

### E9 — Pilot300 freeze and launch

- The existing source selector balanced only label/generator, so a test-first deterministic pilot selector was added. It covers each available action once, then balances labels, generators and label×generator pairs before using repeat action count as a tie-breaker. A code-review counterexample exposed why action count must not remain the first priority after coverage; the new regression test fails under the old strategy and passes under the corrected one.
- Seed `20260716` froze a 300-row source CSV before any pilot prediction. Source CSV SHA-256: `7d29e0a6ba2cbd32daa1e58a7597e53d1f61a88e0f1d27a5f5d6cf670041fda6`. It contains 150 physical / 150 violation, all 198 action strata, and 42–43 samples from each of seven generators.
- Two rows have blank source actions. They are retained under the explicit `__missing_action__` group rather than dropped. A regression test covers both pilot selection and manifest normalization for this case.
- One selected S3 object contains Unicode `’` in its URL path. The old downloader raised `UnicodeEncodeError`; a failing test reproduced it, the path is now percent-encoded before requesting, and the same row downloaded successfully without changing pilot membership.
- Decode audit passed 300/300 videos with readable first and last frames, 32–150 frames per video and zero failures in 13.42 seconds. Frozen pilot manifest SHA-256: `a97762fe4033789eb14a82717c72c14e89bc75a7a67200d5890ff1647f72a670`.
- Review also found that the new dataclass field had shifted `OpenAIChatModel`'s legacy positional `transport` argument and that no supported CLI exposed the pilot selector. Both compatibility bugs now have regression tests; `source-pilot` is a first-class preparation command and the default chat response mode remains `json_object`.
- Re-running the corrected selector on all 3,397 source rows produced exactly the same 300 members and source SHA-256 as the pre-review freeze. No pilot membership changed.
- Pilot attempt 1 was archived after 41 predictions because four SAM2 seeds contained coordinates outside `[0,100]`. The fix adds schema bounds, rejects non-finite coordinates and projects finite coordinates onto valid image pixels. Attempt 2 verified 48 predictions with zero failures, then was archived solely for the independent selector review. Attempt 3 is the current canonical run and reuses only valid observation caches.
- The local and remote full regression suites after all review and SAM2 seed fixes passed `171/171`; the remote run completed in 2.00 seconds.

### E10 — Complete VideoPhy-2 materialization

- The first full download completed 3,390/3,397 rows; all seven failures were Unicode `’` or `—` in S3 object paths. Re-running with the tested percent-encoding fix completed 3,397/3,397 without changing URLs or row membership.
- Decode audit opened every video and read both first and last frames: 3,397 passed, zero failed, frame counts 32–150, elapsed 166.22 seconds. Resolution counts are recorded in `/root/pavg-benchmark/runs/videophy2-full-qwen3vl8b/decode-audit.json`.
- Three source rule cells contain the missing-value marker `[nan]`. A regression-tested normalization rule skips only `nan`/`[nan]` as missing metadata; it does not discard the samples or interpret the marker as a physics rule.
- Frozen full manifest SHA-256: `d8be5fe97ddf6902515c09ccbb53f394b25230213db7c3058d61f84748624906`. It contains 3,397 samples, 1,785 physical / 1,612 violation, 198 action strata plus 70 samples in the explicit missing-action group, and the exact official generator counts.

### E11 — M4 evidence-fusion repair and remote synchronization

- Commit `6f8b6db61362edfbf073fa903164f3521c67a71c` adds the approved M4-only repair. The local full suite passes `178/178`; the remote suite passes `178/178` in 2.01 seconds after creating the isolated pytest temp parent. A first remote attempt produced 47 setup errors solely because the requested basetemp parent did not exist; it was rerun with `mkdir -p` and is not counted as a code failure.
- SAM2 evidence is now used by M4 twice: when a VLM verifier is enabled, the pipeline attaches a bounded chronological `sam2_track` snapshot to each rule candidate, and the VLM prompt sends that snapshot alongside the selected keyframes. The snapshot is capped at 24 states per track and includes frame range, visibility, boxes, centers and motion fields. The NoOp rule baseline does not receive this payload, so its output path remains unchanged.
- M4 reviews are no longer shared across unrelated candidates with the same category. Calls are grouped by object, category and exact candidate time segment. The response schema accepts optional `claim_status` values `confirmed`, `rejected` or `uncertain`; the status is retained in violation evidence.
- M4's default fusion changed from detector/VLM `0.4/0.6` to `0.7/0.3`. The CLI still exposes `--m4-detector-weight` for the frozen dev sweep `0.7`, `0.6`, `0.5`; D0, D1 and B1 are unchanged.
- The remote source received only the M4 files from bundle `pavg-6f8b6db.bundle` (SHA-256 `92fb9cd0f58d552920e006d957ca2d3732b41ffff4b7f12bc5d5c5ceb30f3fe2`); the source manifest was updated to schema 1.2. The running pilot process was not restarted or modified.

### E12 — Pilot300 completion and M4 diagnostic result

- The canonical pilot completed all 300 samples with 300/300 SAM2 observation caches and 900/900 unique predictions for `D0_DIRECT_VLM`, `D1_STRUCTURED_VLM` and `B1_RULE`; duplicate count and failure count were both zero.
- Pilot metrics were D0 accuracy/Macro-F1 `0.503/0.503`, D1 `0.540/0.538`, and B1 `0.483/0.482`. D0/D1 mean model latency was about 3.28 seconds per sample; B1 mean latency was 0.007 seconds. The run occupied about 3 hours 18 minutes, or approximately 39.7 seconds per sample, projecting about 37.5 hours for all 3,397 rows. A sampled GPU residency was 25,605 MiB of 40,960 MiB; no OOM occurred.
- The matched closed-model audit was not run because no secret was injected into the remote shell or repository.
- M4 repair diagnostics are synchronized locally under `outputs/benchmarks/videophy2-m4-vlm-repair-smoke20/`. Dev10 weight 0.7 and 0.6 both scored `0.600/0.600` accuracy/Macro-F1; 0.5 scored `0.500/0.333`. The frozen tie-break selected 0.7. One eval10 run at 0.7 scored `0.800/0.792` with violation recall `1.0` and zero failures.
- Compared with the old smoke20 M4 result (`0.500/0.333`), the repair removes the all-physical collapse. It matches the previously measured B1 eval10 score but does not yet establish a clear improvement over B1, so M4 remains a diagnostic candidate rather than the primary method for the full 3,397-row run.

### E13 — Full VideoPhy-2 D0/B1 launch

- At 2026-07-16 16:22 Asia/Shanghai, the remote A100 launched PID `144429` in a detached session. The command uses the frozen full manifest `/root/pavg-benchmark/runs/videophy2-full-qwen3vl8b/manifest.json` (SHA-256 `d8be5fe97ddf6902515c09ccbb53f394b25230213db7c3058d61f84748624906`), methods `D0_DIRECT_VLM,B1_RULE`, provider `chat` with strict JSON schema, 16 frames, official SAM2.1 Hiera B+, and the resumable run directory `/root/pavg-benchmark/runs/videophy2-full-qwen3vl8b/run`.
- The full dataset is already materialized and decode-audited: 3,397/3,397 videos. The process owns `predictions.jsonl.lock`; initial progress after startup was 7 predictions and 3 SAM2 observation caches. Logs are `/root/pavg-benchmark/logs/videophy2-full-qwen3vl8b.stdout.log` and `.stderr.log`.
- The SSH launch command itself timed out after detaching, but an independent audit confirmed the intended process, lock, output file and SAM2 propagation log were active; no duplicate evaluator was started.

### E14 — Current-scope freeze

- On 2026-07-17 the user narrowed the active scope to completing VideoPhy-2, merging and auditing both disjoint A100 shards, and producing the final VideoPhy-2 Chinese summary report.
- VideoPhy-1 OOD is explicitly deferred. Its Task 9 checkboxes remain open and no OOD result will be implied by the current report.
- The active completion contract is therefore Task 8 plus the VideoPhy-2-relevant parts of Task 10: 6,794 terminal sample×method keys, exact merge/audit, frozen metrics and confidence intervals, synchronized non-secret artifacts, local tests and documentation.
