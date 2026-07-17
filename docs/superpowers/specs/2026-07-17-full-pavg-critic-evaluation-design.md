# Full PAVG Critic Evaluation Design

**Date:** 2026-07-17
**Status:** Design approved in conversation; awaiting written-spec confirmation
**Primary scope:** Complete prompt-conditioned Critic pipeline on frozen VideoPhy-2
**Explicit non-scope:** Generator/Repairer/Selector loop, VideoPhy-1 OOD, fine-tuning, and a paid GPT full run

## 1. Goal

Evaluate the complete implemented PAVG Critic rather than treating the already accepted `B1_RULE` result as a verdict on the whole framework. The primary question is whether prompt-conditioned Planner/PQSG, SAM2 tracks, deterministic rules, VideoScience Checklist, Morpheus mechanics, candidate VLM verification and coverage-aware evidence fusion together improve over a matched direct Qwen3-VL judge.

The accepted D0/B1 report remains immutable. Its result only establishes that the current prompt-template + SAM2 + deterministic-rule B1 path did not outperform direct Qwen3-VL. It does not evaluate model-backed PQSG, VideoScience, mechanics, candidate VLM review or the full evidence-family fusion.

## 2. Dataset contract and leakage boundary

The frozen manifest is `evaluation/manifests/videophy2_test_full.json`, SHA-256 `d8be5fe97ddf6902515c09ccbb53f394b25230213db7c3058d61f84748624906`.

- Population: 3,397 videos, 1,785 physical / 1,612 violation.
- Prompt availability: 3,397/3,397 non-empty prompts, 599 unique prompt strings and 198 action groups.
- Each row also contains human-authored physical rules and label-oriented metadata.
- Primary evaluation may read only `sample_id`, `video_path`, `prompt`, generator/action identifiers and the ground-truth label used by the metric layer.
- The Critic must not receive `physics_label`, score fields, `human_violated_rules`, followed/unfollowed annotations or human physical rules in the primary run.
- Human `physical_rules` may enter only the separately named `ORACLE_PLAN_300` diagnostic. It must never be merged into the primary M5 predictions or presented as a deployable result.

The 300-sample diagnostic population is the previously frozen pilot manifest with SHA-256 `a97762fe4033789eb14a82717c72c14e89bc75a7a67200d5890ff1647f72a670`. It is synchronized without regeneration to `evaluation/manifests/videophy2_pilot300.json` before implementation; a checksum mismatch is terminal rather than a reason to select a replacement subset.

VideoPhy-2 can evaluate the complete Critic because it contains prompts and videos. It cannot evaluate the complete PhysGenLoop because it does not contain prompt-aligned initial/repaired video trajectories or a generator/repair/selection outcome contract.

## 3. Frozen method matrix

All model-backed methods use the same local `Qwen/Qwen3-VL-8B-Instruct` snapshot, deterministic decoding and the already frozen 16-frame image policy. All PAVG methods reuse the accepted SAM2.1 Hiera B+ observation caches; no video is propagated through SAM2 again.

| ID | Frozen definition | Purpose |
|---|---|---|
| `D0_DIRECT_VLM` | Existing 16-frame direct Qwen judge | Primary matched baseline; reuse accepted predictions |
| `D1_STRUCTURED_VLM` | Same frames/model with fixed physics checklist prompt | Prompt-structure baseline |
| `B1_RULE` | Template Planner + SAM2 tracks/events + deterministic rules | Existing rule-only result; reuse accepted predictions |
| `M1_GRAPH` | B1 + template question graph/PQSG-style graph scoring | Graph contribution without model-generated nodes |
| `M2_CHECKLIST` | M1 + five-dimensional VideoScience Checklist | Checklist contribution |
| `M3_MECHANICS` | M2 + freefall/projectile/rebound/collision mechanics | Mechanics contribution |
| `M4_VLM` | M3 + grouped keyframe Qwen candidate verification | False-positive rejection before final fusion |
| `M5_FULL` | Model Planner + hybrid template/model PQSG + SAM2 + rules + VideoScience + mechanics + Qwen verification + coverage-aware fusion | Complete implemented Critic |

`B0_PQSG` remains an external independent baseline that requires official PQSG repository outputs. It is not silently approximated by M1 and is not required for the primary complete-Critic claim.

## 4. M5 benchmark integration

The core `evaluation.py` configuration already recognizes `M5_FULL`, but the real-video `StageAPAVGMethod` and server CLI currently stop at M4. The benchmark adapter must be extended without changing the pipeline's frozen default weights or thresholds.

For M5, the same Qwen chat adapter is injected independently at three named boundaries:

1. `planner_model` for prompt → `PhysicsPlan`;
2. `question_model` for incremental PQSG nodes merged with the template graph;
3. `vlm_verifier` for grouped candidate/keyframe confirmation.

Provider failures remain explicit. Planner and PQSG provider failures may use the pipeline's existing template fallback but must be counted and sliced. VLM-review failure makes the M4/M5 method prediction a terminal failure rather than silently reverting to B1. Model IDs, prompt/schema hashes, cache keys, latency and non-secret usage fields are recorded.

The three boundaries share one frozen model snapshot and decoding configuration but have distinct prompt/schema versions and cache namespaces (`planner`, `pqsg`, `verifier`). A cache record is reusable only when the sample, model snapshot, stage, prompt/schema hash and input-evidence hash all match.

M5 uses the existing default family weights and thresholds. No full-result label, generator slice or prompt diagnostic may tune a family weight, hard-violation rule, coverage threshold or model prompt in this cycle.

## 5. Prompt-conditioning diagnostics

The primary full run always receives the correct generation prompt. Three matched M5 views are reported on the already frozen action/generator/label-stratified 300-sample diagnostic manifest:

- `M5_CORRECT_PROMPT_300`: subset of the primary full M5 result.
- `M5_SHUFFLED_PROMPT_300`: deterministic seed `20260717`; each sample receives a donor prompt whose sample ID, exact prompt string and action group all differ from its own, while videos, labels and membership stay unchanged. The complete recipient→donor mapping is frozen before inference and checksum-recorded.
- `M5_ORACLE_PLAN_300`: the correct prompt plus an explicitly audited plan adapter that appends only `physical_rules` as natural-language constraints. It does not read whether a rule was followed/violated or any score/label field. The adapter starts from the byte-cached normal M5 model plan for that same sample, preserves its objects/events/relations, then appends one constraint per rule with stable ID `oracle-rule-{index}`, domain `oracle_natural_language`, expectation equal to the exact rule text and subjects equal to the planned objects (or the single synthetic subject `scene` only when the normal plan has no object). The resulting plan is marked explicit with confidence 1.0.

The shuffled comparison measures whether the Critic is genuinely prompt-conditioned. The oracle comparison estimates the Planner/PQSG headroom. Neither diagnostic may be used to tune M5 or replace the full prompt-only primary result.

## 6. Module diagnostics

Predictions alone are insufficient because an enabled module can remain unavailable or be overridden. Each new PAVG prediction therefore has a deterministic diagnostics sidecar keyed by `sample_id × method_id` containing:

- Planner source, fallback status, confidence and resolved-plan counts;
- question-graph source, node count, answered/blocked/unknown counts and physics coverage;
- VideoScience dimension statuses and summary coverage;
- mechanics applicable/not-applicable/failed counts and per-evaluator score;
- rule candidate/retained-violation counts and categories;
- VLM review confirmed/rejected/uncertain/unavailable counts;
- all five evidence bundles, effective coverage/confidence/weight and family score;
- pre-evidence-fusion decision, final decision and `hard_violation` override flag;
- model-call count, stage latency, terminal provider failures and cache-hit status.

No image bytes, masks, raw provider payloads, authorization data or chain-of-thought are stored.

## 7. Metrics and attribution

The full-population table reports Accuracy, Balanced Accuracy, Macro-F1, both class recalls, violation precision, Spearman, unknown/failure rate, prediction latency and SAM2 production latency separately.

Primary comparison:

- `M5_FULL − D0_DIRECT_VLM` on all 3,397 rows.
- 2,000 action-group bootstrap resamples, seed `20260717`, candidate-minus-baseline Macro-F1.
- Paired correctness cells and generator/action/rule-family slices.

Sequential attribution:

- M1−B1, M2−M1, M3−M2, M4−M3 and M5−M4.
- Per transition, report changed predictions, gains/losses, failure changes and module availability.
- Count cases where PQSG/checklist/mechanics favor physical but a retained hard violation forces the final result to violation.

Prompt attribution on diagnostic300:

- correct minus shuffled prompt;
- oracle plan minus correct prompt;
- paired outcomes and action-group bootstrap intervals, labeled diagnostic rather than confirmatory.

## 8. Frozen success gates

The complete Critic earns VideoPhy-2 support only if all primary M5-vs-D0 gates pass:

1. Macro-F1 delta `>= +0.05`;
2. action-group bootstrap lower bound `> 0`;
3. M5 physical and violation recalls are both non-zero;
4. M5 failure rate minus D0 failure rate `<= 0.01`;
5. at least two generators have positive Macro-F1 delta.

Module-level improvements and oracle gains are explanatory and cannot override a failed primary gate. VideoPhy-1 remains deferred, so even a passing result retains the cross-dataset verdict `not_evaluable_ood_deferred`.

## 9. Execution architecture and resume policy

The two A100 servers keep separate existing project roots; no new top-level directory is created under `/root`.

- cloud2 run root: `/root/pavg-benchmark/runs/videophy2-full-pavg-qwen3vl8b/`;
- cloud1 run root: `/root/pavg-benchmark-shard2/runs/videophy2-full-pavg-qwen3vl8b/`.

Observation ownership follows the accepted provenance view rather than the old prediction split: cloud2 evaluates its 1,699 shard-A members plus the 32 pre-split shard-B fallback caches; cloud1 evaluates the remaining 1,666 shard-B members. The resulting sample sets are disjoint and cover 3,397 exactly.

M1–M3 are CPU/cache-bound and may run while the local Qwen endpoint serves D1/M4/M5. Each method has an append-only prediction file, diagnostics file and single-writer lock. Planner, PQSG and verifier outputs use stage-specific immutable caches, so interruption resumes from valid sample×method/stage keys without repeating paid or GPU model work.

The accepted D0/B1 artifacts and their directories are read-only inputs. New runs never append to or rewrite them.

## 10. Validation sequence

1. Test-first M5 adapter, diagnostics schema and CLI support; complete local suite.
2. Transfer a reproducible source snapshot and rerun the complete remote suite.
3. Run smoke20 for D1/M1–M5; require exact keys, no OOM and diagnostics coverage.
4. Run pilot300 and measure M5 provider failure, call counts, throughput and ETA; require failure rate `< 5%`. A projected wall time `<= 24 h` proceeds normally, `> 24 h` and `<= 30 h` proceeds with a documented extended ETA but no configuration change, and `> 30 h` stops before the full launch for a scope/time report.
5. Run full M1–M5/D1 over the two immutable ownership shards.
6. Run shuffled/oracle diagnostic300 only after the primary M5 configuration is frozen.
7. Strictly merge, run 2,000-resample statistics, regenerate twice and synchronize non-secret artifacts.
8. Run local clean-room regeneration, secret scan and the complete test suite before publication.

## 11. Time and resource estimate

With two A100 40 GB servers and reusable SAM2 caches:

| Phase | Expected time |
|---|---:|
| Design/plan freeze and source audit | 0.5–1 h |
| M5 adapter, diagnostics and tests | 3–5 h |
| smoke20 and compatibility fixes | 0.5–1.5 h |
| M1–M3 full cached inference | 0.25–0.75 h |
| D1 full matched baseline | 1.5–3 h |
| M4 full Qwen verification | 2–5 h |
| M5 full Planner/PQSG/verifier | 4–10 h |
| prompt/oracle diagnostic300 | 1–3 h |
| merge, statistics, audit and report | 1–2 h |

Parallel critical-path expectation is 12–22 hours. The contingency ceiling is 30 hours if M5 generates more candidate-review calls than the current B1 category distribution suggests or if one endpoint needs a documented restart. No SAM2 propagation rerun is budgeted.

## 12. Decision after this evaluation

- If M5 passes the primary gates, freeze it as the pre-loop Critic and move to OOD/loop evaluation before fine-tuning claims.
- If M5 improves over B1 but not D0, use module diagnostics to decide between learned fusion/LoRA and a stronger matched backbone audit.
- If M5 matches B1 and hard overrides dominate, redesign the hard-violation arbitration before spending on GPT or training.
- If oracle plan materially outperforms correct-prompt M5, prioritize Planner/PQSG supervision.
- If correct and shuffled prompts are indistinguishable, the current pipeline is not using prompt semantics effectively and cannot support a prompt-conditioned framework claim.

The previously approved LoRA cycle is paused until this complete, unfine-tuned M5 baseline is accepted, so training cannot erase the missing architectural baseline.
