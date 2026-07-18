# Learning Repair Agent 最终整合与团队交接

> 日期：2026-07-17
> 状态：两套本地实现已统一为一套 canonical package；v3 Blender proxy 训练完成；Actual Executor Trial 待采集

## 1. 最终结论

正式实现只有一个：

```text
src/physgenloop/learning_repair/
```

`src/physgenloop/learning_repair_pipeline/` 仅保留 deprecated compatibility namespace，
旧 import 会重定向到 canonical package，不再执行第二套业务实现。正式配置统一使用：

```text
configs/learning_repair/
```

独立 `LearningRepairLoopRunner` 是学习型 Repair 的正式闭环入口。团队原有
`LoopController`、共享 Protocol 和 `CriticReport` 契约没有被学习型动作侵入修改。

## 2. 最终运行链

```text
Frozen Physics Critic Report
            ↓
ReportFeatureEncoder
            ↓
Repair Policy
  ├─ action probabilities
  └─ per-action values
            ↓
RepairSelector
  ├─ Executor capability mask
  ├─ classification_proxy / action_value mode
  ├─ confidence abstention
  └─ heuristic fallback
            ↓
RepairDecision
  ├─ action + confidence
  ├─ action probabilities
  ├─ per-action values
  ├─ LocalEditTarget
  ├─ compatibility_id
  └─ provenance / fallback reason
            ↓
ExecutorRegistry
  ├─ PromptRepairExecutor
  ├─ GlobalRegenerationExecutor
  ├─ LocalEditingExecutor
  └─ RejectExecutor
            ↓
Critic re-evaluation + semantic/quality gate
            ↓
RepairTrialV1 → LearningTargetV1 → ActualTrialMemory → retraining
```

## 3. 契约与兼容策略

canonical `contracts.py` 是唯一权威定义，包含：

- `RepairAction`、`RepairContext`、`RepairExample`；
- 最终丰富版 `RepairDecision`；
- `CandidateRecord`、`LocalEditTarget`；
- `ExecutionRequest`、`ExecutionResult`；
- `ScoreBundle`、`RepairTrialV1`、`LearningTargetV1`、`RepairRunResult`。

`RepairDecisionV1` 仅是指向最终 `RepairDecision` 的兼容别名。旧分类模型产生的简化
decision 通过 `adapt_legacy_decision()` 显式升级，避免把旧 checkpoint 当成新
Action-Value checkpoint 静默加载。

兼容清单冻结 Critic config/schema、feature schema、action order 和 checkpoint
compatibility ID。任何 SHA、schema 或 action order 不匹配都会 fail fast。

## 4. Policy、Selector 与 Memory

当前支持两种明确选择模式：

| 模式 | 数据来源 | 选择依据 | 当前状态 |
|---|---|---|---|
| `classification_proxy` | Blender selected-action proxy label | capability mask 后的动作概率 | v3 当前使用 |
| `action_value` | 实际多动作 `RepairTrialV1` | 每动作 utility + 校准概率 | 代码就绪，待 Actual Trial |

proxy 数据只观察被选择动作的 reward。未执行动作必须保持 `null`，不能伪装成失败。
真实执行失败的 Trial 会以负 utility 进入 `ActualTrialMemory`，用于降低相似状态下的
风险动作价值。

## 5. v1–v3 训练结果

| 版本 | Groups | Samples | Held-out Macro-F1 | Gain/Value MAE |
|---|---:|---:|---:|---:|
| v1 | 600 | 7,800 | 1.000000 | 0.099762 |
| v2 | 900 | 11,700 | 1.000000 | 0.058520 |
| v3 | 1,200 | 22,200 | 1.000000 | 0.026106 |

v3 当前最佳模型：

```text
model_id:          repair-value-78304cfff2fa
policy_format:     repair-action-value-policy/2.1
selection_mode:    classification_proxy
compatibility_id:  lrcompat-bf0077c5081dafab
actual_trials:     0
deployment_ready:  false
```

指标来自 group-safe Blender proxy held-out test。它只能说明 proxy 映射已被模型稳定
学习，不能解释为 HunyuanVideo 修复成功率，也不能替代真实 Executor 闭环评测。

## 6. 单一 CLI

统一入口：

```text
pavg-repair
python -m physgenloop.learning_repair
```

| 阶段 | 命令 |
|---|---|
| proxy 数据 | `collect`、`validate`、`split`、`train` |
| proxy 迁移 | `adapt-proxy-targets`、`verify-baseline` |
| 兼容门禁 | `check-compatibility` |
| Actual Trial | `validate-campaign`、`run-campaign`、`build-targets`、`audit-targets` |
| Action-Value | `train-values`、`evaluate` |
| 推理交付 | `predict`、`export` |
| 评审报告 | `closed-loop-report`、`integration-review` |

`evaluate --manifest ...` 评估旧分类/proxy checkpoint；`evaluate --targets ...` 评估
Action-Value Policy、Memory 和 R0–R5。

## 7. 本地复核命令

```powershell
python -m physgenloop.learning_repair verify-baseline `
  --manifest configs/learning_repair/proxy_baseline_1200g_v3.json `
  --root .

python -m physgenloop.learning_repair check-compatibility `
  --manifest configs/learning_repair/critic_compatibility_v1.json `
  --critic-config configs/default.yaml `
  --critic-schema schemas/critic_output.schema.json `
  --feature-schema configs/learning_repair/feature_schema.json

python -m pytest -q tests/test_learning_repair.py tests/test_learning_repair_pipeline.py
```

## 8. 下一阶段

1. 团队冻结最终 Critic revision 后重新生成 compatibility manifest，消除
   `source_revision=unknown`。
2. 接入真实 Prompt、Global、Local Executor，按动作采集 `RepairTrialV1`。
3. 将 Blender Actual Trial 按 `group_id` 切分后训练 value-led Policy。
4. 构建严格分离的 Hunyuan calibration/test campaign。
5. 只有真实闭环 physics gain、semantic、quality、成本和失败率通过门禁后，才把
   `deployment_ready` 提升为 true。

## 9. 本地待人工清理项

由于项目约束禁止自动删除，原工作区中以下旧实现文件不会进入最终提交，但仍可能留在
本地 `src/physgenloop/learning_repair_pipeline/`：`baselines.py`、`campaign.py`、
`cli.py`、`cloud_campaign.py`、`compatibility.py`、`contracts.py`、`evaluation.py`、
`executors.py`、`manifests.py`、`memory_policy.py`、`proxy_adapter.py`、`recording.py`、
`review.py`、`runner.py`、`value_policy.py`、`value_training.py`。需要时由项目成员人工
确认后删除；必须保留 compatibility `__init__.py` 和 `__main__.py`。

---

署名：hejin
