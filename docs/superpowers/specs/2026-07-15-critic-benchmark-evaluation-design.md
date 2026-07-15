# PAVG Critic Benchmark Evaluation Design

**Date:** 2026-07-15
**Status:** Approved for implementation
**Primary objective:** Determine whether the structured PAVG Critic provides a statistically and practically meaningful improvement over direct VLM judging before investing in the generation-repair loop.

## 1. Decision to make

The evaluation must answer two separate questions:

1. Does PAVG classify physical plausibility more accurately and more reliably than a direct VLM judge given the same prompt, video and model backend?
2. Does PAVG produce sufficiently localized and actionable evidence to drive a later repair loop?

A better video-level score alone is not sufficient to justify the loop. The framework must also identify the violated physical rule, localize the relevant temporal evidence and produce a repair direction that humans prefer over the direct-VLM explanation.

## 2. Scope

### 2.1 In scope

- VideoPhy-2 as the primary in-distribution benchmark.
- VideoPhy-1 as the out-of-distribution benchmark.
- SAM2.1 as the required primary observation frontend for every headline PAVG method.
- A manually audited diagnostic subset drawn from both benchmarks.
- Direct-VLM, structured-direct-VLM, official auto-rater and PAVG ablation baselines.
- Human-label-aligned classification, ordinal scoring, calibration, localization, cost and robustness metrics.
- Prompt-clustered paired statistics and confidence intervals.
- Immutable manifests, resumable predictions and auditable run metadata.
- A go/no-go decision for implementing the generation-repair loop.

### 2.2 Deferred until the core evaluator is validated

- Generating a fresh PhyGenBench video suite.
- Training or fine-tuning a new evaluator model.
- Integrating a real video generator or optimizing prompts through the loop.
- Claiming publication-level improvements from the repository's six synthetic trajectory fixtures.
- Using an automatic evaluator as ground truth where human labels exist.

PhyGenBench, VBench and VideoScore remain planned extensions. PhyGenBench tests broader physical-law coverage; VBench and VideoScore provide general-quality guardrails after a generator is connected.

## 3. Benchmark stack

### 3.1 Primary: VideoPhy-2

[VideoPhy-2](https://github.com/Hritikbansal/videophy/tree/main/VIDEOPHY2) is the main benchmark because it provides action-centric generated videos, human semantic-adherence and physical-commonsense judgments, and physical-rule grounding labels. Its public train and test resources support a clean development/test separation.

Usage policy:

- Training data may be used only for schema development, threshold selection and fusion calibration.
- Test labels must not influence prompts, thresholds, fusion weights or rule selection.
- The final report uses the full accessible test split. A stratified pilot is run first only to validate the pipeline and estimate cost.
- Human annotations are the gold standard. VideoPhy-2-AutoEval is reported only as an external baseline.

### 3.2 OOD: VideoPhy-1

[VideoPhy-1](https://github.com/Hritikbansal/videophy) contains 688 material-interaction captions and human semantic-adherence and physical-commonsense judgments across multiple video generators. It stresses solid-solid, solid-fluid and fluid-fluid interactions and is used without additional calibration after VideoPhy-2 decisions are frozen.

### 3.3 Diagnostic subset

Public benchmark labels mainly measure whole-video correctness. A repair loop additionally needs temporal and causal evidence. A stratified sample of 150-200 violation videos will therefore receive a separate annotation pass.

Each item records:

- violation category and physical rule;
- involved objects;
- start, peak and end frame;
- up to five evidence frames;
- whether the rule is visually groundable;
- a concise correction target that does not prescribe a particular generator implementation;
- two independent annotator judgments and one adjudicated value for disagreements.

The subset is stratified by benchmark, generator, physical-rule family, video duration and violation severity. Annotators do not see method predictions.

### 3.4 Later extensions

- [PhyGenBench](https://github.com/OpenGVLab/PhyGenBench): 160 prompts across 27 physical laws and four domains, evaluated only after fresh videos are available.
- [VBench](https://github.com/Vchitect/VBench): temporal and perceptual quality guardrails.
- [VideoScore-Bench](https://github.com/TIGER-AI-Lab/VideoScore): human-aligned general video-quality correlation and pairwise preference guardrails.

## 4. Compared methods

All methods emit the same canonical prediction record. They differ only in allowed evidence and reasoning modules.

| Method ID | Description | Purpose |
|---|---|---|
| `D0_DIRECT_VLM` | Prompt plus uniformly sampled frames, one schema-constrained VLM judgment | User-requested direct VLM baseline |
| `D1_STRUCTURED_VLM` | Same visual input and VLM, with a fixed physical checklist and ordinal output schema | Strong prompt-engineering control |
| `A0_VIDEOPHY_AUTO` | Released VideoPhy auto-rater output when available | Specialized learned evaluator baseline |
| `B1_RULE` | PAVG deterministic rules only | Rule contribution |
| `M1_GRAPH` | Rules plus template question graph | Graph contribution |
| `M2_CHECKLIST` | M1 plus checklist evidence | Checklist contribution |
| `M3_MECHANICS` | M2 plus mechanics evidence | Mechanics contribution |
| `M4_VLM` | M3 plus evidence-grounded review using the same VLM as D0/D1 | Structured evidence versus direct judgment |
| `M5_FULL` | M4 plus model-generated PhysicsPlan/PQSG where configured | Full framework |

An optional `M3_ORACLE_FRONTEND` can be run only on controlled data with trusted tracks. It separates detector/tracker errors from reasoning errors and must never be mixed into the headline result.

All B1-M5 headline runs consume the same cached SAM2.1 frame-level observations for a sample/model block. Sparse VLM keyframe detections are retained as `F0_VLM_SPARSE` frontend ablations only; they are not an acceptable default because missing intermediate detections can create observation gaps and break trajectory/event reasoning. `F1_SAM2` is the production frontend and must report frame coverage, track count and propagation failures.

### 4.1 Model blocks and headline baselines

The first evaluation uses two matched model blocks. Results are reported within a block; a stronger model in one block is never compared against a weaker model in the other as evidence for framework improvement.

**Closed-model block**

- Full pilot and confirmatory default: `gpt-5.6-terra`, selected as the current quality/cost-balanced GPT-5.6 variant.
- Capability-ceiling subset: `gpt-5.6-sol` on a pre-registered subset only, unless pilot costs justify a full run.
- OpenAI calls use the Responses API and the repository's multimodal structured-output adapter. The exact model string is pinned in run metadata rather than using a moving alias.
- D0, D1, M4 and M5 all use the same GPT variant inside a paired comparison.

**Open-model block**

- Full pilot and confirmatory default: `Qwen/Qwen3-VL-8B-Instruct`.
- Optional pilot ceiling: `Qwen/Qwen3-VL-32B-Instruct` when the rented GPU can run the declared precision without changing the visual budget.
- `OpenGVLab/InternVL3.5-8B` is an optional robustness replication, not a required first-round baseline.
- The open model may be served through vLLM or another pinned inference server. The serving engine is infrastructure, not a separate method.
- D0, D1, M4 and M5 all use the same open model, precision, processor revision and serving configuration inside a paired comparison.

The headline claim is the within-backbone difference `M5 - D1`, supported by `M4 - D1` and the deterministic B1-M3 ablations. `M5 with GPT` versus `D1 with Qwen` is an invalid causal comparison and is excluded from headline tables.

## 5. Fairness controls

- D0, D1, M4 and M5 use the same VLM model identifier, provider revision and decoding settings within a comparison block.
- Video decoding, resize policy and maximum visual-input budget are fixed in the run manifest.
- D0/D1 use uniform sampling. PAVG may choose evidence frames, but it may not exceed the same maximum number of VLM-visible frames without reporting a separate cost-unconstrained result.
- Temperature is zero where supported. If the provider is nondeterministic, each model-based method is repeated three times and both mean and variance are reported.
- Videos are renamed to opaque sample IDs; generator names and benchmark labels are never included in model input.
- PAVG ablations reuse one immutable SAM2 observation cache per sample and frontend configuration. They must not rerun or change tracking between B1-M5.
- D0/D1 do not receive SAM2 masks, tracks or event evidence; this preserves their role as direct VLM baselines.
- Method prompts are versioned and frozen before test execution.
- All threshold selection occurs on development data. Test execution is one-way and produces append-only prediction artifacts.

## 6. Canonical data contracts

### 6.1 Evaluation sample

Each normalized sample contains:

- `sample_id`, `benchmark`, `split`, `prompt`, `video_path`;
- `prompt_group_id` and generator metadata used only for slicing/statistics;
- ordinal human `semantic_score` and `physics_score` when available;
- derived binary semantic/physics labels using the benchmark's published threshold;
- zero or more physical-rule labels;
- checksum, source URL and license metadata;
- optional diagnostic annotation.

The adapter must preserve the raw label alongside every normalized label so that conversions remain auditable.

### 6.2 Prediction record

Every method emits:

- method and run identifiers;
- physical and semantic scores;
- binary decision and explicit `unknown` state;
- confidence and evidence coverage;
- predicted violation categories/rules;
- predicted temporal interval and evidence frames;
- repair instruction when the method supports it;
- latency, model/API usage and failure metadata;
- hashes of input manifest, prompt template and code revision.

Provider failures are records, not silently dropped samples.

## 7. Metrics

### 7.1 Primary physical-plausibility metrics

- macro-F1 for the benchmark-defined physical/violation decision;
- AUROC and AUPRC using the continuous physical score;
- Spearman correlation and quadratic weighted kappa for ordinal human physical scores;
- balanced accuracy and Matthews correlation coefficient;
- unknown rate and selective-risk/coverage curve.

Macro-F1 is the headline metric. Raw accuracy is never reported alone.

### 7.2 Secondary metrics

- semantic-adherence macro-F1 and ordinal correlation;
- joint success rate using the benchmark's high-semantic/high-physics definition;
- physical-rule macro-F1, including the `not_groundable` class where present;
- per-rule, per-action, per-generator and difficulty-slice metrics;
- Brier score and expected calibration error;
- end-to-end latency, GPU time, API cost, VLM-visible frame count and provider failure rate.

### 7.3 Diagnostic/actionability metrics

- violation-category macro-F1;
- temporal intersection-over-union;
- normalized peak-frame error;
- evidence-frame Recall@1 and Recall@5;
- visually-groundable abstention accuracy;
- blinded human preference for repair instruction usefulness and correctness.

## 8. Statistical protocol

- All method comparisons are paired on identical sample IDs.
- Confidence intervals use at least 10,000 cluster-bootstrap resamples grouped by prompt, preserving multiple generator outputs from the same prompt as a cluster.
- Binary paired decisions use McNemar's test.
- Score/correlation differences use a paired permutation test at the prompt-cluster level.
- Holm correction controls family-wise error for the planned PAVG ablation comparisons.
- The report includes effect sizes and confidence intervals, not only p-values.
- A method is not declared better if the effect exists only on one generator or one physical-rule family.

## 9. Execution stages

### Stage A: infrastructure smoke test

- 20 samples spanning physical/violation labels and at least four rule families.
- Validate download manifests, decoding, baseline output schema, resumption and metrics.
- No performance conclusion is allowed.

### Stage B: stratified pilot

- 200-300 VideoPhy-2 test videos stratified by generator, action and label.
- Run D0, D1, B1, M3 and M4 first.
- Estimate API/GPU cost, failure rate, metric variance and likely full-run duration.
- Freeze final prompt templates and operational settings after reviewing only pilot diagnostics, not hidden/full-test aggregate outcomes.

### Stage C: confirmatory evaluation

- Run all planned methods on the full accessible VideoPhy-2 test set.
- Apply frozen settings to VideoPhy-1 without recalibration.
- Produce paired tables, confidence intervals, calibration plots and error slices.

### Stage D: actionability audit

- Build and adjudicate the diagnostic subset.
- Measure localization, evidence selection and repair-instruction utility.
- Produce the loop go/no-go recommendation.

## 10. Loop decision gates

Implementation of a real generator-repair loop is justified only if all conditions below are met:

1. M5 improves physical macro-F1 over D1 by at least 0.05 absolute and the paired 95% confidence interval excludes zero.
2. The improvement remains positive on VideoPhy-1 OOD and is not confined to one generator or rule family.
3. Violation-category macro-F1 on the diagnostic subset is at least 0.55.
4. Evidence-frame Recall@5 is at least 0.70, or an equivalent temporal-localization result is approved after inspecting duration-normalized errors.
5. Blinded annotators prefer PAVG repair instructions over D1 in at least 60% of non-tied comparisons.
6. Calibration improves or remains comparable, and PAVG abstains rather than confidently fabricating evidence on ungroundable cases.
7. PAVG remains on a defensible accuracy-cost Pareto frontier; the default operational target is no more than three times D1's per-video evaluation cost.
8. General temporal/perceptual quality guardrails do not show a material regression once generator outputs are compared.

Failure of a gate results in a targeted evaluator improvement plan, not immediate loop implementation.

## 11. Implementation boundaries

Evaluation functionality will be separated from production Critic code:

- `src/pavg_critic/benchmarking/datasets.py`: manifests and VideoPhy adapters;
- `src/pavg_critic/benchmarking/contracts.py`: normalized sample and prediction records;
- `src/pavg_critic/benchmarking/baselines.py`: D0/D1 and external prediction adapters;
- `src/pavg_critic/benchmarking/frontends.py`: SAM2 observation production, cache metadata and sparse-VLM frontend ablation;
- `src/pavg_critic/benchmarking/runner.py`: resumable paired execution;
- `src/pavg_critic/benchmarking/metrics.py`: classification, ordinal, calibration and diagnostic metrics;
- `src/pavg_critic/benchmarking/statistics.py`: cluster bootstrap and paired tests;
- `src/pavg_critic/benchmarking/report.py`: machine-readable and Markdown reports;
- `benchmarks/evaluate_video_benchmark.py`: CLI orchestration only.

The existing `src/pavg_critic/evaluation.py` remains backward compatible for frozen trajectory regression. It must not become a second incompatible source of metric definitions; shared metric functions will be migrated carefully with regression tests.

## 12. Artifacts and reproducibility

Each run directory contains:

- immutable dataset manifest and checksums;
- resolved configuration with secrets redacted;
- prompt-template hashes;
- one JSONL prediction file per method;
- provider failure records;
- aggregate metrics with bootstrap samples or seeds;
- slice tables and calibration data;
- environment/package snapshot and git revision;
- Markdown summary describing any deviations from the frozen protocol.

Downloaded benchmark assets remain evaluation-only and git-ignored. A manifest records their source, license, size and checksum. The runner never edits source data in place.

## 13. Testing and failure handling

- Unit tests cover label conversion, score direction, unknown handling, metric edge cases, clustered resampling and resumable writes.
- Contract tests use tiny local fake videos and scripted model outputs.
- Network/API tests are opt-in and never required for the default unit suite.
- Corrupt videos, missing labels and provider failures generate explicit failed records.
- A run resumes by sample/method key and never overwrites a completed prediction without an explicit flag.
- Test fixtures cannot be cited as benchmark performance.

## 14. Compute and rental strategy

GPU rental is stage-gated rather than a prerequisite for starting the evaluation.

- Stage A runs locally on the RTX 5060. Data adapters, deterministic ablations, decoding checks and API-backed D0/D1/M4 calls do not justify renting a server.
- Stage B moves to a rented GPU when an open-weight video VLM, VideoPhy AutoEval, repeated model runs or throughput requirements exceed the local 8 GB VRAM budget.
- The recommended pilot machine has 48 GB VRAM, at least 16 vCPUs, 64 GB system RAM and 500 GB of NVMe or equivalent persistent storage.
- A 24 GB GPU is acceptable for a quantized 7B-class evaluator and small batches, but every out-of-memory retry and effective batch size must be recorded.
- An 80 GB A100/H100-class GPU is reserved for unquantized larger evaluators, the full PhyGenEval stack or a demonstrated throughput bottleneck. It is not the default.
- Stage B begins with one 48 GB GPU for a metered pilot window. Full-test GPU count and rental duration are derived from measured per-video latency, peak VRAM and failure rate rather than estimated in advance.
- Benchmark data and prediction artifacts live on persistent storage. Ephemeral instances may be destroyed only after manifests, JSONL predictions, logs and environment snapshots have been synchronized.
- Rental cost is reported per method and per successfully evaluated video so that accuracy-cost Pareto comparisons remain possible.

## 15. Expected first deliverable

The first implementation milestone is not a leaderboard. It is a reproducible 20-sample smoke report containing D0, D1, B1 and M3 predictions, verified label mappings, decoded-video checks, failure accounting and a cost estimate for the pilot. Only after that report passes review does the project spend API/GPU budget on Stage B.
