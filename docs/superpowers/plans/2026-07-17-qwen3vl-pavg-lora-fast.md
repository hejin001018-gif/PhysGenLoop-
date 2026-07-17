# Qwen3-VL-8B PAVG Fast LoRA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After the frozen VideoPhy-2 report is complete, train and validate two reversible Qwen3-VL-8B LoRA candidates within a measured 24-hour cycle, automatically falling back to judge-only SFT at hour 6 when the PAVG/M4 path cannot finish on time.

**Architecture:** Prepare deterministic VideoPhy-2 train/dev/holdout manifests and ms-swift JSONL locally, then materialize identical inputs inside each server's existing PAVG workspace. The two A100 servers train independent learning-rate candidates from byte-identical read-only base weights; both adapters are selected on one cloud2 validation runtime and deployed per request without changing SAM2 seed generation. No new top-level directory is created under `/root`.

**Tech Stack:** Python 3.12, pytest, CSV/JSONL, ms-swift 3.12.6, Transformers 4.57, PEFT LoRA, PyTorch 2.8 CUDA 12.8, Qwen3-VL-8B-Instruct, vLLM 0.11.0, official SAM2.1 Hiera B+.

---

## Task 1: Enforce completion and directory gates

**Files:**
- Modify: `docs/superpowers/plans/2026-07-16-full-videophy-server-evaluation.md`
- Modify: this plan under `Execution results`

- [ ] Verify shard A has exactly 3,398 terminal keys and shard B exactly 3,396, with two methods per sample, zero duplicates and absent `.lock` files.

```python
import collections, json
from pathlib import Path

path = Path("/root/pavg-benchmark/runs/videophy2-full-qwen3vl8b/shard-a/run/predictions.jsonl")
rows = [json.loads(line) for line in path.read_text().splitlines()]
keys = collections.Counter((r["sample_id"], r["method_id"]) for r in rows)
assert not [key for key, count in keys.items() if count != 1]
assert len(rows) == 3398
print(len(rows), collections.Counter(r["method_id"] for r in rows))
```

On cloud1 run:

```python
import collections, json
from pathlib import Path

path = Path("/root/pavg-benchmark-shard2/shard-b/run/predictions.jsonl")
rows = [json.loads(line) for line in path.read_text().splitlines()]
keys = collections.Counter((r["sample_id"], r["method_id"]) for r in rows)
assert not [key for key, count in keys.items() if count != 1]
assert len(rows) == 3396
print(len(rows), collections.Counter(r["method_id"] for r in rows))
```

- [ ] Complete the existing full-report plan through the exact 6,794-key merge, full metrics, 2,000 action-group bootstraps, slices, secret scan and synchronized Chinese report. Require `outputs/benchmarks/videophy2-full-qwen3vl8b/summary.json` before stopping either base vLLM.
- [ ] Set H0 to the timestamp when the report is accepted and both evaluator locks are released.
- [ ] Resolve each proposed remote path and reject it unless it starts with the existing `/root/pavg-benchmark/` on cloud2 or `/root/pavg-benchmark-shard2/` on cloud1. Do not create `/root/training`, `/root/data`, `/root/models`, `/root/qwen3-*` or another project root.

## Task 2: Add deterministic VideoPhy-2 splits

**Files:**
- Create: `src/pavg_critic/finetuning/__init__.py`
- Create: `src/pavg_critic/finetuning/videophy.py`
- Create: `tests/finetuning/test_videophy.py`

- [ ] Write a failing split test that requires exactly two dev and two holdout rows per action, deterministic membership, disjoint URLs and no test overlap.

```python
def test_freeze_split_assigns_two_per_action(rows):
    split = freeze_videophy2_split(rows, seed=20260717)
    assert Counter(r.action for r in split.dev) == {"a": 2, "b": 2}
    assert Counter(r.action for r in split.holdout) == {"a": 2, "b": 2}
    assert not ({r.video_url for r in split.train} & {r.video_url for r in split.dev})
```

- [ ] Run it and require missing-module/function failure.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/finetuning/test_videophy.py -q --basetemp outputs/.pytest-tmp-ft-split-red
```

- [ ] Implement immutable `VideoPhyTrainRow` and `FrozenSplit` dataclasses. Normalize blank action to `__missing_action__`; parse rule cells only with `ast.literal_eval`; reject scores outside 1–5, duplicate URLs, groups smaller than five and train/test URL or SHA overlap.
- [ ] Within every sorted action group, choose two dev and two holdout rows by `(current label count, current generator count, current hard count, sha256(seed|split|video_url))`; place all remaining rows in train.
- [ ] Require official counts 3,343 total, 2,551 train, 396 dev, 396 holdout, then run tests and commit.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/finetuning/test_videophy.py -q --basetemp outputs/.pytest-tmp-ft-split-green
git add src/pavg_critic/finetuning tests/finetuning/test_videophy.py
git commit -m "feat: freeze VideoPhy fine-tuning splits"
```

## Task 3: Build verifier and global-judge records

**Files:**
- Create: `src/pavg_critic/finetuning/records.py`
- Create: `tests/finetuning/test_records.py`

- [ ] Write failing tests for the exact ms-swift shape, followed→`rejected`, unfollowed/human-violated→`confirmed`, cannot-determine→`uncertain`, and empty evidence frames when no human frame annotation exists.
- [ ] Implement the documented custom record shape:

```python
def make_swift_record(video_path: str, prompt: str, answer: Mapping[str, object]) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<video>\n{prompt}"},
            {"role": "assistant", "content": json.dumps(answer, sort_keys=True)},
        ],
        "videos": [video_path],
    }
```

- [ ] Emit at most one rule from each status per video in deterministic rule-family-balanced order. Emit one global record with label `physical` iff `pc >= 4`. Never derive a training label from current critic or test predictions.
- [ ] Build the 70/30 train sampling stream without changing membership. Dev and holdout contain both task types for measurement only.
- [ ] Run tests and commit.

## Task 4: Add preparation and launch CLIs

**Files:**
- Create: `benchmarks/prepare_qwen3vl_finetune.py`
- Create: `benchmarks/run_qwen3vl_lora.py`
- Create: `configs/finetuning/qwen3vl8b-pavg-lora.json`
- Create: `tests/finetuning/test_cli.py`

- [ ] Add a failing end-to-end CLI test using temporary CSV, videos and a test manifest.
- [ ] Implement preparation arguments `--csv`, `--videos-dir`, `--test-manifest`, `--output-dir`, `--seed 20260717`, `--expected-rows 3343`, `--expected-train 2551`, `--expected-dev 396`, `--expected-holdout 396`.
- [ ] Atomically write three manifests, mixed Swift JSONL, `dataset-audit.json` and SHA-256 sidecars.
- [ ] Implement a dry-run command builder that returns an argv list rather than a shell string:

```python
def build_swift_argv(config: Mapping[str, object]) -> list[str]:
    return [
        "swift", "sft", "--model", str(config["model"]),
        "--dataset", str(config["dataset"]),
        "--val_dataset", str(config["val_dataset"]),
        "--train_type", "lora", "--torch_dtype", "bfloat16",
        "--num_train_epochs", "1", "--per_device_train_batch_size", "1",
        "--per_device_eval_batch_size", "1",
        "--learning_rate", str(config["learning_rate"]),
        "--lora_rank", "8", "--lora_alpha", "16",
        "--target_modules", "q_proj", "k_proj", "v_proj", "o_proj",
        "--freeze_vit", "true", "--freeze_aligner", "true",
        "--gradient_checkpointing", "true",
        "--gradient_accumulation_steps", "16",
        "--max_length", "4096", "--output_dir", str(config["output_dir"]),
    ]
```

- [ ] Freeze `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `VIDEO_MAX_TOKEN_NUM=128`, `FPS_MAX_FRAMES=8`, `SWIFT_PATCH_CONV3D=1`; require `--dry-run` before `--execute`.
- [ ] Run all finetuning tests and commit.

## Task 5: Create pinned environments inside existing workspaces

**Remote files:**
- Create: `/root/pavg-benchmark/training/qwen3vl8b-pavg-lora-v1/venv`
- Create: `/root/pavg-benchmark-shard2/training/qwen3vl8b-pavg-lora-v1/venv`
- Create under each run root: `logs/environment.txt`, `artifacts/base-sha256.txt`

- [ ] After Task 1 only, stop the now-idle vLLM services and require no GPU compute process.
- [ ] Construct each run root from its existing workspace and assert the resolved prefix before `mkdir`.
- [ ] Install in isolated venvs, never in benchmark/vLLM environments:

```bash
# cloud2
RUN_ROOT=/root/pavg-benchmark/training/qwen3vl8b-pavg-lora-v1
uv venv --python 3.12 "$RUN_ROOT/venv"
"$RUN_ROOT/venv/bin/uv" pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
"$RUN_ROOT/venv/bin/uv" pip install ms-swift==3.12.6 transformers==4.57.0 peft==0.17.1 qwen-vl-utils==0.0.14 torchcodec

# cloud1
RUN_ROOT=/root/pavg-benchmark-shard2/training/qwen3vl8b-pavg-lora-v1
uv venv --python 3.12 "$RUN_ROOT/venv"
"$RUN_ROOT/venv/bin/uv" pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
"$RUN_ROOT/venv/bin/uv" pip install ms-swift==3.12.6 transformers==4.57.0 peft==0.17.1 qwen-vl-utils==0.0.14 torchcodec
```

- [ ] Use Torch SDPA if no verified Flash Attention wheel exists; do not compile CUDA extensions because `nvcc` is absent.
- [ ] Verify real CUDA matmul, processor load, `swift sft --help`, identical package versions and all nine base hashes on both servers. Transfer committed source/config only and rerun the full project tests remotely.

## Task 6: Materialize and freeze official train data

**Remote files under each run root:**
- Create: `data/source/videophy2_training.csv`
- Create: `data/videos/`, `data/prepared/`
- Create: `logs/download.jsonl`, `logs/decode-audit.json`

- [ ] Require official CSV SHA-256 `076a03584da28a0622aa9f3bdad71f52c3d8afef2025d34f7ecd9e4ef081a42c` and 3,343 unique rows.
- [ ] Download every train video independently to both existing run roots with the tested percent-encoded downloader, temp files, retries and append-only failures.
- [ ] Decode first/last frames, retain failures in the audit and abort the preferred route above 2% failures.
- [ ] Run preparation on both servers and require byte-identical membership/hashes; only absolute local video paths may differ.

## Task 7: Execute the H0–H6 gate

**Remote files under each run root:**
- Create: `smoke128/`, `gate/measurements.json`, `gate/decision.json`

- [ ] Freeze the same 128 train and 64 dev sample IDs before output; build mixed smoke JSONL.
- [ ] Run A (`2e-7`) on cloud2 and B (`5e-7`) on cloud1 long enough to measure memory, throughput, finite loss and schema generation.
- [ ] Materialize disjoint SAM2 dev/holdout caches when GPU schedule permits; never read full-test labels.
- [ ] Compute measured ETA with a two-hour report/deployment buffer:

```python
projected_hours = (
    remaining_cache * cache_seconds
    + remaining_train * epochs * train_seconds
    + validation_seconds + full_test_seconds + 7200
) / 3600
```

- [ ] Continue PAVG only when peak ≤38GB, failures ≤2%, schema validity ≥98%, loss finite, no repeated OOM and ETA ≤24h. Otherwise atomically record `judge_only` plus reason and rebuild configs with global records only. The fallback is irreversible this cycle.

## Task 8: Train independent candidates

**Remote files:**
- Create: cloud2 run root `runs/candidate-a/`
- Create: cloud1 run root `runs/candidate-b/`

- [ ] Review dry-run argv; require identical base/data/config hashes except LR, output and local paths.
- [ ] Launch one resumable detached trainer per GPU with logs, PID and lock. Record memory, utilization, loss, throughput and ETA every 15 minutes.
- [ ] Abort only for non-finite loss, repeated OOM, corrupt checkpoint or frozen time gate. Require 0.5- and 1.0-epoch adapter hashes; never merge into base weights.
- [ ] If one candidate fails after H6, allow the other to finish without changing hyperparameters or using test labels.

## Task 9: Select on cloud2 and run holdout once

**Remote files under cloud2 run root:**
- Create: `selection/dev-results.json`, `selection/selected-adapter.json`, `selection/holdout-results.json`

- [ ] Copy cloud1 adapters and SHA sidecars to cloud2, excluding optimizer state; verify hashes.
- [ ] Evaluate four checkpoints on the same 396 dev samples and cloud2 runtime.
- [ ] Select by video Macro-F1, nonzero recalls, PAVG rule three-class Macro-F1, failure rate, latency, then earliest checkpoint.
- [ ] Freeze adapter/prompt/config/threshold hashes and run the 396 locked holdout once.
- [ ] Apply route-specific release arithmetic. On failure, retain a negative artifact and leave base deployment unchanged.

## Task 10: Run full comparison and deploy reversibly

**Remote files:**
- Create under existing run roots: `final-test/shard-a/`, `final-test/shard-b/`
- Create under cloud2 run root: `deployment/`

- [ ] Preferred matrix: current base D0/B1 plus fine-tuned D0/M4. Fallback matrix: base D0 plus fine-tuned D0. Reuse valid current test SAM2 caches.
- [ ] Merge exact keys and report bootstrap intervals, paired outcomes, generator/action/rule-family slices and failures in the denominator.
- [ ] Start isolated vLLM from unchanged base with `--enable-lora --max-lora-rank 8`. Seed requests use base; D0/M4 requests use adapter.
- [ ] Run schema, smoke20, GPU/SAM2 coexistence, latency, identity and rollback tests. Switch main routing only when holdout and full-test gates pass.
- [ ] Roll back by removing adapter routing and restarting the recorded base command; never overwrite base weights.
- [ ] Synchronize non-secret manifests, adapter hashes, predictions, metrics and logs; scan for credentials, update result/plan docs, run full local pytest and push tested commits to `origin/sy` without force.

## Execution results

Append immutable checkpoints after every task with timestamps relative to H0, commands, counts, hashes, failures, throughput, ETA decisions and negative results. Never rewrite a prior checkpoint.
