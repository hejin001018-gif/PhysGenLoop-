# Qwen3-VL-8B PAVG Fast LoRA Design

**Date:** 2026-07-17
**Status:** Approved in conversation; pending written-spec review
**Time objective:** Finish within 24 hours after both current VideoPhy-2 evaluators release their GPUs.
**Primary objective:** Improve PAVG/M4 VideoPhy-2 Macro-F1 with a verifier-focused LoRA. If the measured hour-6 projection cannot finish within 24 hours, switch immediately to a judge-only Qwen3-VL LoRA.

## 1. Non-interference and starting checkpoint

Training must not begin while either frozen VideoPhy-2 evaluator or its vLLM service is still required. The current full benchmark remains the first priority and is never restarted, truncated or mixed with fine-tuned predictions.

Both servers already contain the same `Qwen/Qwen3-VL-8B-Instruct` snapshot. The following nine SHA-256 values match exactly on cloud2 and cloud1:

| File | SHA-256 |
|---|---|
| `config.json` | `5cd452860dc1e9c29dd71cc3cef7f39b338b7a40793f7a260655c2d3568f3661` |
| `generation_config.json` | `8469742d1fce0de951c8909b26a2c0c0d8490837ce476efb114da9e0cefc4d44` |
| `preprocessor_config.json` | `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516` |
| `tokenizer_config.json` | `c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399` |
| `model.safetensors.index.json` | `520b2e05079402e9468a8701d03d1154d14b2599593afb6effa7fb60c1bff070` |
| `model-00001-of-00004.safetensors` | `d5d0aef0eb170fc7453a296c43c0849a56f510555d3588e4fd662bb35490aefa` |
| `model-00002-of-00004.safetensors` | `8be88fb5501e4d5719a6d4cc212e6a13480330e74f3e8c77daa1a68f199106b5` |
| `model-00003-of-00004.safetensors` | `83de00eafe6e0d57ccd009dbcf71c9974d74df2f016c27afb7e95aafd16b2192` |
| `model-00004-of-00004.safetensors` | `0a88b98e9f96270973f567e6a2c103ede6ccdf915ca3075e21c755604d0377a5` |

The base model directories are read-only inputs. Training writes only adapters, checkpoints, optimizer state and logs under new run directories. No safetensor in the base snapshot may be modified in place.

The GPUs are both A100 PCIe 40GB. cloud2 uses driver `570.211.01`; cloud1 uses `570.86.10`. Training therefore uses separately created but identically pinned Python 3.12, PyTorch CUDA 12.8, Transformers, PEFT, Flash Attention and training-framework environments. Exact package inventories and CUDA tests are recorded before the first optimizer step.

The two servers do not form a distributed job. Each trains one independent candidate from the same manifest and base hashes. This avoids WAN synchronization and makes interruption local. Both candidate adapters are evaluated on the same cloud2 validation runtime so driver or inference-environment differences cannot decide the winner.

## 2. Frozen data source and leakage boundary

Use only the official `videophysics/videophy2_train` CSV and videos for supervised training. The frozen metadata audit found:

- CSV SHA-256 `076a03584da28a0622aa9f3bdad71f52c3d8afef2025d34f7ecd9e4ef081a42c`;
- 3,343 unique rows and video URLs;
- 1,800 physical samples (`pc >= 4`) and 1,543 violations (`pc < 4`);
- 198 action groups including the explicit missing-action group;
- 1,013 hard and 2,330 non-hard samples;
- CogVideo 1,150, Hunyuan 1,105 and Cosmos 1,088;
- 3,097 samples with followed rules, 1,023 with unfollowed rules, 556 with cannot-determine rules and 1,744 with human-violated rules.

Freeze a deterministic split with seed `20260717`:

- train: 2,551;
- development: 396, exactly two samples per action group;
- locked internal holdout: 396, exactly two other samples per action group.

Within each action group, selection greedily balances physical label, source generator and hard status. The three manifests must be disjoint by sample ID, normalized URL and video SHA-256. They are also checked against all 3,397 VideoPhy-2 test URLs and hashes.

The current VideoPhy-2 smoke20, pilot300 and full-test labels, predictions, failures and diagnostics are prohibited training or checkpoint-selection inputs. The full test can be evaluated once only after the adapter, prompt, threshold and deployment configuration are frozen.

Download and decode failures remain in an audit file. They are not silently replaced after seeing labels. The training denominator may exclude a permanently unavailable video only through a pre-label availability rule recorded before optimization; dev and holdout failures remain failures.

## 3. Preferred PAVG/M4 training task

Train a language-backbone LoRA while freezing the vision tower and multimodal projector. The base Qwen3-VL remains responsible for SAM2 object-seed requests. The LoRA is activated only for direct-judge and M4 verifier requests, so a verifier improvement cannot corrupt the already stable seeding behavior.

The training sampler uses:

- 70% rule-verification examples;
- 30% global video-judgment examples.

### Rule-verification example

Input contains eight temporally sampled video frames, the caption, one expected physical rule, its source physics family, and—when the hour-6 PAVG gate succeeds—a bounded SAM2 trajectory summary and frozen critic candidate.

Output uses the production schema with `claim_status`, physics label, violation family and confidence. Supervision is derived without a teacher model:

- a rule listed as followed maps a violation claim to `rejected`;
- an unfollowed or human-violated rule maps to `confirmed`;
- a cannot-determine rule maps to `uncertain`.

Choose at most one rule from each status per video in deterministic rule-family-balanced order. This caps repeated video encoding and prevents abundant followed rules from dominating.

### Global-judgment example

Input contains the eight frames, caption and frozen physical-check prompt. Output matches the project's strict JSON schema for semantic/physics scores and labels, confidence and source-backed violation families. No evidence frame is fabricated when the dataset lacks frame-level human annotation.

## 4. Hour-6 automatic fallback

Hours are measured after both evaluation GPUs are released.

During hours 0–6, materialize data, create identical training environments, build the frozen manifests, prepare 128-example training smoke data, and build SAM2/critic validation caches in disjoint halves.

Continue the PAVG/M4 route only if all gates pass by hour 6:

1. measured peak training memory is at most 38GB;
2. loss is finite and at least 98% of sampled outputs conform to the schema;
3. download/decode failure rate is at most 2%;
4. measured cache and training throughput project both candidates, dev, holdout, full final evaluation and deployment to finish by hour 24;
5. no repeated CUDA OOM or compatibility failure occurs.

If any gate fails, preserve the failure record and immediately switch to the judge-only fallback. Do not spend additional time repairing the PAVG route within this cycle.

The fallback removes SAM2 summaries, critic candidates and rule-status loss. It trains only the global video judge on the same train/dev/holdout membership. It can improve Qwen3-VL but cannot by itself establish a critic-architecture gain; the final report states that limitation explicitly.

## 5. Candidate training configuration

Both candidates start from the verified base hashes and use the same manifest, prompt hashes, seed, frame sampling, batch schedule and one epoch:

- eight video frames, random temporal sampling for train and deterministic sampling for evaluation;
- model maximum length 4,096;
- BF16;
- per-device batch size 1;
- gradient accumulation 16;
- gradient checkpointing;
- LoRA rank 8, alpha 16, dropout 0;
- LoRA targets restricted to the supported language attention projections;
- frozen vision tower and multimodal projector;
- cosine schedule, warmup ratio 0.03 and weight decay 0.01;
- checkpoints at 0.5 and 1.0 epoch;
- candidate A learning rate `2e-7` on cloud2;
- candidate B learning rate `5e-7` on cloud1.

These values follow the official Qwen3-VL video/LoRA starting configuration. Actual smoke throughput, not an optimistic estimate, controls the hour-6 decision.

## 6. Validation and checkpoint selection

Only the 396-sample development manifest can select a candidate or checkpoint. Evaluate all four candidate checkpoints on the same cloud2 runtime.

Selection order is frozen:

1. highest video-level Macro-F1;
2. both physical and violation recall nonzero;
3. for the PAVG route, highest rule-level confirmed/rejected/uncertain Macro-F1;
4. lowest schema/failure rate;
5. lowest end-to-end latency;
6. earliest checkpoint if still tied.

After selection, freeze the adapter SHA-256, prompt/config hashes and thresholds. Run the 396-sample locked holdout once.

PAVG/M4 release gates:

- fine-tuned M4 minus frozen base B1 Macro-F1 at least `+0.05`;
- fine-tuned M4 minus matched fine-tuned D0 Macro-F1 at least `+0.03`;
- paired action-group bootstrap 95% interval lower bound above zero;
- both class recalls nonzero;
- failure-rate increase at most one percentage point.

Judge-only fallback release gates:

- fine-tuned D0 minus base D0 Macro-F1 at least `+0.05`;
- both class recalls nonzero;
- schema/failure rate at most 1%;
- the gain is positive on more than one source generator.

If the applicable holdout gates fail, do not deploy the adapter. A negative result is still reported and the original service remains unchanged.

## 7. Twenty-four-hour schedule

| Time | Preferred PAVG/M4 route | Judge-only fallback after hour 6 |
|---|---|---|
| H0–H3 | data, hashes, decode, split, pinned environments | same preparation |
| H3–H6 | SAM2/cache split plus 128-example train smoke; apply gate | gate failure triggers immediate fallback |
| H6–H15 | two independent one-epoch candidates | two independent candidates, expected H6–H14 |
| H14–H18 | shared cloud2 dev selection | dev H14–H16, holdout H16–H17 |
| H18–H20 | locked holdout | split full-test D0 H17–H20 |
| H20–H24 | split full test, adapter smoke and deployment if projection still holds | report, deploy or retain base H20–H24 |

The PAVG route proceeds beyond H6 only when measured ETA includes a deployment/report buffer. Otherwise the fallback is mandatory. The robust PAVG estimate remains 28–36 hours; a 24-hour result is a measured fast path, not a promise.

## 8. Final test and causal comparison

After checkpoint freeze, run disjoint two-server shards and merge by exact sample/method keys.

Preferred matrix:

- base D0 and base B1 from the current frozen run;
- fine-tuned D0;
- fine-tuned M4 using base Qwen for seeds, official SAM2 caches and the verifier LoRA.

Fallback matrix:

- base D0 from the current frozen run;
- fine-tuned D0.

The fallback comparison supports a model-improvement claim only. The preferred matrix separates backbone gain from critic gain because fine-tuned D0 and fine-tuned M4 use the same judging adapter.

All final metrics retain failures and unknowns in the denominator and include action-group bootstrap intervals, paired outcomes, generator/action/rule-family slices, runtime and schema failures.

## 9. Deployment and rollback

Keep the original snapshot and adapter separate:

```text
models/Qwen3-VL-8B-Instruct/
models/adapters/pavg-qwen3vl8b-v1/
```

The main vLLM service loads the frozen base and enables the language-backbone LoRA. Requests specify whether the adapter is active:

- object-seed requests: base model, no adapter;
- direct judge and M4 verifier: selected adapter.

Before switching production traffic, start an isolated endpoint, run schema smoke, smoke20, memory and latency tests, verify model/adapter IDs in responses and scan logs for secrets. Deployment changes only after applicable release gates pass.

Rollback removes the adapter from request routing and restarts the previously recorded base command. It does not require copying or restoring the base weights. Model files, videos, optimizer states and credentials remain outside git; manifests, hashes, resolved configs, metrics and non-secret logs are synchronized locally.

## 10. Self-review

- No test-label leakage: every training and selection input is from the official train set.
- No cross-server DDP ambiguity: the GPUs train independent candidates.
- No base-weight mutation: adapters are isolated and reversible.
- No hidden schedule expansion: hour 6 enforces the automatic fallback.
- No conflated claim: judge-only improvement is not described as PAVG architecture improvement.
- No unresolved dataset, model, seed, split, learning-rate, gate or deployment choice remains.
