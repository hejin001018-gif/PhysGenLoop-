# Full VideoPhy Server Shard-2 Acceleration Plan

**Goal:** Add a second A100 worker without changing the frozen D0/B1 protocol, then merge its disjoint predictions with the first server's resumable run.

**Frozen contract:** The shard uses the same full manifest SHA-256 `d8be5fe97ddf6902515c09ccbb53f394b25230213db7c3058d61f84748624906`, Qwen3-VL-8B-Instruct, vLLM 0.11.0, Transformers 4.57.0, strict chat JSON schema, 16 frames, official SAM2.1 Hiera B+ checkpoint/config, and D0_DIRECT_VLM/B1_RULE methods. Only sample membership and output directory differ.

## Task 1: Audit the second server

- [x] Verify SSH access, OS, GPU/driver, memory, disk, Python and outbound connectivity.
- [x] Confirm no evaluation process is running and no user data is present.

## Task 2: Reproduce the verified runtime

- [x] Install isolated Python 3.12 with uv.
- [ ] Install Torch cu128, vLLM 0.11.0, Transformers 4.57.0, benchmark dependencies and official SAM2 at the frozen source/checkpoint.
- [ ] Run the full regression suite and a real three-frame SAM2 propagation test.

## Task 3: Materialize only the shard inputs

- [x] Transfer the reproducible source bundle and required overlay, excluding credentials and full output caches.
- [ ] Transfer Qwen3-VL weights and SAM2 checkpoint, verifying SHA-256.
- [ ] Create a stable shard manifest containing only rows not already terminal on server 1; download/decode-probe exactly those videos.

## Task 4: Launch and audit the second worker

- [ ] Start one detached D0/B1 evaluator with its own lock and output directory.
- [ ] Monitor progress, GPU state, failure count and ETA; never overlap sample×method keys with server 1.
- [ ] Record the shard command, source/runtime hashes and first checkpoint in this plan.

## Task 5: Merge and report

- [ ] Merge server-1 and shard-2 predictions after both reach terminal status.
- [ ] Verify 3,397 samples × 2 methods, duplicate absence, failure rate, manifest alignment and result checksums.
- [ ] Synchronize summaries and update the full evaluation plan without copying credentials or model/video binaries into git.

## Execution results

### E1 — Server audit

- Host `wXOGV9`, Ubuntu kernel `5.15.0-56-generic`, one NVIDIA A100-PCIE-40GB, 40,960 MiB, driver `570.86.10`.
- 90 GiB RAM, 200 GiB root filesystem with approximately 200 GiB free, system Python `3.10.12`, and official GitHub connectivity confirmed.
- No evaluation process or project data was present at audit time. Password was used only for this connection and was not written to the repository or logs.

### E2 — Disjoint shard freeze and first-worker restart

- The frozen 3,397-row manifest was split deterministically into shard A with 1,699 samples and shard B with 1,698 samples. The model, methods, prompts, frame count, thresholds, SAM2 configuration and source membership were not otherwise changed.
- The original resumable run contained 115 predictions at the split checkpoint. Exactly 48 keys belong to shard A and 67 keys belong to shard B; both subsets were copied into their own append-only prediction files and the original file was retained for audit.
- Server 1 resumed only shard A with `D0_DIRECT_VLM,B1_RULE`, strict JSON schema, 16 frames and the existing shared SAM2 observation cache. PID `151066` was live at the first checkpoint, GPU utilization was 88–100%, and shard-A predictions increased from 48 to 67.

### E3 — Second-worker staging

- Key-only SSH was verified on server 2. uv `0.11.29`, CPython `3.12.13`, Torch `2.8.0+cu128`, torchvision `0.23.0+cu128`, NumPy `2.5.1`, vLLM `0.11.0` and Transformers `4.57.0` import successfully; a real CUDA probe reports the NVIDIA A100-PCIE-40GB.
- The source bundle at commit `1989a81ce0c0ef849e5fa57480cc01c93ad5da94` and shard-B manifest/prediction checkpoint were transferred without `.env` or credentials. Official SAM2 source and the 323,606,802-byte SAM2.1 Hiera B+ checkpoint were transferred; checksum verification and propagation smoke remain pending.
- Server 2 is downloading the frozen Qwen3-VL snapshot and the 1,698 shard-B videos concurrently. At the first checkpoint the model directory was 2.1 GiB and 35 videos were complete. The evaluator will start only after model, video, checksum, decode and smoke gates pass.
