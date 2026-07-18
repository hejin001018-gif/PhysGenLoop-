# Learning Repair Agent 路线与阶段工作记录

> 日期：2026-07-17  
> 作者：hejin  
> 状态：团队上下文已同步；Blender proxy v3 已完成；Actual RepairTrial 与 Hunyuan 校准尚未开始；主线合并未就绪  
> 适用范围：基于团队 Physics Critic 构建学习型 Repair Agent，并在不混淆 proxy 结果与真实闭环结果的前提下完成训练、校准和部署验证

---

## 一、上下文同步结论

团队项目的核心目标是提高生成视频的物理一致性，而不是只做视频物理错误分类。总体闭环为：

```text
User Prompt
  → PhysicsPlan Resolver
  → Video Generator
  → Physics Critic
  → Candidate Selector
  → Repair Agent
  → Repair Executor
  → New Candidate
  → Physics Critic re-evaluation
```

截至本次同步：

- 团队主仓库 `/root/PhysGenLoop-` 的 `main` 与 `origin/main` 一致，基线提交为 `e9b2eaf`；
- `pavg_critic` 已具备 Planner、SAM2/VLM 视觉前端、轨迹与事件、确定性规则、PQSG、Checklist、Mechanics、VLM 复核和 coverage-aware evidence fusion；
- `physgenloop` 主线已具备冻结契约、fake Generator、Prompt Repairer、Selector 和有界 Best-of-K `LoopController`；
- 主线真实 Generator、真实 Local Editor 和学习型 Repair Policy 尚未作为正式能力合入；
- 本人的 Learning Repair 成果目前主要存在于本地工作区和 release artifacts，不能仅依据本地未提交代码宣称团队主线已经具备同等能力。

因此，当前 Learning Repair Agent 的正确定位是：

> 消费冻结的 `CriticReport`，学习选择可执行修复动作并预测动作收益；在真实 Executor 与重新评估闭环完成前，只能称为 Blender proxy 预训练与工程基线。

---

## 二、与团队主线的职责边界

Learning Repair Agent 不替代 Critic，也不修改 Critic 的物理判断语义。输入至少包括：

- `CriticReport.decision / physics_score / confidence / coverage`；
- violation category、object、frame interval、critical frames；
- rule/PQSG/checklist/mechanics/VLM 证据；
- 当前轮次、剩余预算、历史动作；
- Executor capability；
- 可选 semantic preservation 与 visual quality。

输出四类互斥动作：

1. `prompt_repair`；
2. `global_regeneration`；
3. `local_editing`；
4. `reject`。

已经达到 `physical` 且通过接受门槛的候选由上游 Controller/Runner 接受，不把“无需修复”与 `reject` 混为同一动作。

共享接口的默认原则仍然是：

- Critic 只读消费；
- 动作决策与动作执行分离；
- 不支持的动作必须显式 mask；
- 真实 Trial 只追加、不覆盖；
- 共享 schema、Protocol 或 Controller 变更必须经团队批准。

---

## 三、本地训练演进

### 3.1 v1–v3 结果

| 版本 | Blender group | 监督样本 | Held-out Macro-F1 | Gain / Value MAE | 定位 |
|---|---:|---:|---:|---:|---|
| `repair_600g_v1` | 600 | 7,800 | 1.000000 | 0.099762 | MLP action/gain 工程基线 |
| `repair_900g_v2` | 900 | 11,700 | 1.000000 | 0.058520 | 数据扩容与 release 基线 |
| `repair_1200g_v3` | 1,200 | 22,200 | 1.000000 | 0.026106 | Action-Value Policy 当前最佳 proxy 版本 |

三轮结果说明结构化 Critic 特征对当前 proxy 标签高度可学习，不能据此推导真实视频修复成功率。

### 3.2 v3 数据与标签

`repair_1200g_v3` 已完成以下数据门禁：

- group 数：1,200；
- 样本数：22,200；
- train / validation / test：18,024 / 2,154 / 2,022；
- group leakage：0；
- proxy label：22,200；
- Actual `RepairTrialV1` label：0；
- target audit、compatibility、held-out test、release smoke、archive hash 和 cleanup receipt 均有效。

动作分布：

| Action | Count |
|---|---:|
| `prompt_repair` | 4,800 |
| `global_regeneration` | 4,200 |
| `local_editing` | 7,200 |
| `reject` | 6,000 |

数据标签类型固定为：

```text
blender_proxy_selected_action_only
```

只观测 selected action 的 proxy reward；其他未执行动作保持 unknown/null，没有伪造成负 Trial。

### 3.3 hard-v1 数据恢复事件

`hard-v1` 中两个 `multi_corrupt` 样本因为目标离开画面，只稳定暴露 disappearance，未满足目标证据门禁：

- `group_000957--multi_corrupt`；
- `group_001155--multi_corrupt`。

门禁没有放宽。使用 `hard-v1.1` 重新生成后，目标保持在画面内并稳定保留 penetration、disappearance 与 trajectory anomaly 证据，随后才纳入训练。

---

## 四、v3 模型、评测与 Release

### 4.1 模型架构

```text
Frozen CriticReport + RepairContext
            ↓
Versioned ReportFeatureEncoder
            ↓
Action-Value Policy
  ├── 四动作 classification head
  └── 四动作 per-action value head
            ↓
Capability Mask + provenance-aware selection
            ↓
Repair Memory
            ↓
RepairDecisionV1
            ↓
ExecutorRegistry
  ├── Prompt Repair
  ├── Global Regeneration
  ├── Local Editing
  └── Reject / historical selector
```

当前模型：

- policy format：`repair-action-value-policy/2.1`；
- model ID：`repair-value-78304cfff2fa`；
- compatibility ID：`lrcompat-bf0077c5081dafab`；
- selection mode：`classification_proxy`；
- memory：train-only curated proxy memory；
- value-led selection：等待真实多动作 Trial 后启用。

### 4.2 五随机种子验证

| Seed | Validation Macro-F1 | Balanced accuracy | Value MAE | Best epoch |
|---:|---:|---:|---:|---:|
| 17 | 1.000000 | 1.000000 | 0.046059 | 1 |
| 23 | 1.000000 | 1.000000 | 0.037880 | 1 |
| 42 | 1.000000 | 1.000000 | 0.036394 | 1 |
| 73 | 1.000000 | 1.000000 | 0.044934 | 1 |
| 101 | 1.000000 | 1.000000 | 0.026651 | 1 |

最终选用 seed 101。Held-out test：

- Accuracy：1.000000；
- Macro-F1：1.000000；
- Balanced accuracy：1.000000；
- Mean regret：0；
- Value MAE：0.026106。

四动作 test support：

| Action | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `prompt_repair` | 1.000000 | 1.000000 | 1.000000 | 480 |
| `global_regeneration` | 1.000000 | 1.000000 | 1.000000 | 330 |
| `local_editing` | 1.000000 | 1.000000 | 1.000000 | 720 |
| `reject` | 1.000000 | 1.000000 | 1.000000 | 492 |

### 4.3 R0–R4 proxy 对照

| Method | Accuracy | Macro-F1 | Balanced accuracy | Mean regret |
|---|---:|---:|---:|---:|
| R0 Category-only | 0.686944 | 0.587851 | 0.692424 | 0.096283 |
| R1 Heuristic | 0.686944 | 0.587851 | 0.692424 | 0.096283 |
| R2 Policy only | 1.000000 | 1.000000 | 1.000000 | 0 |
| R3 Memory only | 1.000000 | 1.000000 | 1.000000 | 0 |
| R4 Policy + Memory | 1.000000 | 1.000000 | 1.000000 | 0 |

这些指标衡量 proxy action mapping，而不是 Repair Executor 的实际收益。Memory 与 Policy 的满分也说明当前 proxy 标签规律较强，下一阶段应优先增加真实 action-specific reward，而不是继续无边界扩充同类 proxy 数据。

### 4.4 Release

- release manifest SHA-256：`b53cbfdb30fa7cabeb774a23c2687a61ec67e6e30eb50ea9c974d0078943f558`；
- archive SHA-256：`53c49359969a00555f49c7afad7c904a7bc28968395bb73e87db2ac823c2a4e2`；
- archive bytes：767,177；
- release files：10；
- 四动作 inference smoke：4/4；
- Blender shard cleanup receipt：2 份；
- 已清理 shard 数据约 648,483,445 bytes。

Release 已经不依赖 Blender 执行推理，但部署仍需要匹配版本的 Critic、PhysGenLoop 代码和真实 Executor backend。

---

## 五、本地两条 Learning Repair 实现路线

### 5.1 `src/physgenloop/learning_repair/`

这条路线是较早的 Learning Repair 核心实现，已经提供：

- `RepairAction / RepairContext / RepairDecision`；
- versioned CriticReport feature encoder；
- heuristic fallback；
- PyTorch MLP Policy；
- Repair Memory；
- train/evaluate/predict/export CLI；
- Prompt-compatible adapter。

本地工作区中它还存在对共享主线的未提交兼容改动：

- `src/physgenloop/__init__.py` 导出 Learning Repair API；
- `LoopRound` 增加可选 `repair_decision` 审计字段；
- `LoopController` 可选调用 `repair_with_decision`，并避免在最后一轮执行不会被消费的修复动作；
- `pyproject.toml` 增加 train extra 和 `pavg-repair` CLI。

这些改动当前尚未经过团队主线集成批准，不能描述为已合入能力。

### 5.2 `src/physgenloop/learning_repair_pipeline/`

这条路线按 7 月 17 日团队边界重新设计，核心目标是把结构化 Learning Repair 作为并行研究入口：

```text
CriticReport (read-only)
  → ReportFeatureEncoder
  → LearningRepairPolicy + RepairMemory
  → RepairDecisionV1
  → RepairExecutor
  → LearningRepairLoopRunner
  → Critic re-evaluation
```

新增能力包括：

- 私有版本化契约；
- `ExecutorRegistry` 与 capability mask；
- Prompt / Global / Local / Reject Executor adapter；
- `JsonlTrialRecorder`；
- `ActualTrialCampaign`；
- `LearningTargetV1`；
- `ActualTrialMemory`；
- per-action value training；
- frozen Blender/Hunyuan campaign manifest；
- R0–R5 分域评测；
- closed-loop report；
- versioned Memory；
- integration-review 门禁。

该路线不应直接替换现有 `LoopController`。团队批准统一结构化 Repair 契约之前，默认保持独立 `LearningRepairLoopRunner`。

---

## 六、Milestone 1–5 当前状态

| Milestone | 工程实现 | 真实数据/结果验收 | 当前结论 |
|---|---|---|---|
| M1 接口冻结与 proxy 归档 | 已完成 | v2/v3 proxy artifacts 已验证 | 通过实验级门禁 |
| M2 Executor 与独立 Runner | 已完成 fake/adapter 骨架 | 真实 Prompt/Global/Local 后端未全部接入 | 工程就绪，实际执行未验收 |
| M3 Blender Actual Trial | Campaign、Recorder、Target、Memory、Value Policy 已实现 | Actual Trial 数为 0 | 未通过数据验收 |
| M4 Hunyuan 校准与闭环 | manifest、backend factory、R0–R5、closed-loop report 已实现 | 无真实 Hunyuan rollout | 未通过端到端验收 |
| M5 团队主线集成 | integration-review 已实现 | 团队尚未批准共享契约修改 | 保持并行入口 |

Local Editor 不可用时必须从 Executor capability 中移除并显式 mask，不得用 fake Local Editor 生成正式 Trial。

---

## 七、本地验证证据

本地轻量验证记录：

- 新 `learning_repair_pipeline`：10 passed；
- Learning Repair + 原闭环回归：30 passed；
- 忽略 SAM2 optional test 后的仓库检查：173 passed，3 个 dependency-only failures；
- 没有在本轮本地实现中安装依赖、启动 GPU 训练或使用团队云服务器。

已知环境限制：

- 本地系统 Python 缺少现有 SAM2 测试需要的 `torch`；
- 缺少 `python-dotenv` 和 `opencv-python`，导致 3 个既有 optional-dependency 测试失败；
- 因此当前证据支持新模块的轻量逻辑与回归，不等同于完整 GPU/视频后端验收。

集成评审结果：

```text
research_entry_ready = true
mainline_merge_ready = false
recommendation = keep_parallel_learning_repair_loop_runner
```

未通过门禁：

- `source_revision_deployable = false`；
- `domain_separated_evaluation_present = false`。

---

## 八、Critic 全量评测同步及其影响

团队 benchmark 运行副本已完成 VideoPhy-2 全量评测，但结果和相关报告代码尚未正式同步回 `/root/PhysGenLoop-` 主线。

全量结果：

- 样本：3,397；
- prediction：6,794；
- D0 Direct VLM Macro-F1：0.548897；
- B1 Rule + SAM2 Macro-F1：0.544539；
- B1 - D0：-0.004359；
- action-group bootstrap 95% CI：[-0.031613, +0.020693]；
- B1 failure：5/3,397；
- VideoPhy-2 material support：false；
- VideoPhy-1 OOD：deferred。

这意味着：

1. 当前不能声称团队 Critic 已在全量开放基准上显著优于 Direct VLM；
2. Repair Agent 应继续冻结消费接口，但在真实 Trial 中保留人工/semantic/quality guardrail；
3. sy 的 Critic 改进、full-report 工具和 Qwen3-VL LoRA 计划仍位于运行/报告副本，未成为 main 的稳定依赖；
4. Qwen3-VL LoRA 目前只是已设计方案，fine-tuning 尚未开始；
5. Repair release 在团队 Critic 最终同步后必须重新冻结 compatibility，而不是沿用 `source_revision=unknown`。

---

## 九、兼容性门禁

当前兼容性：

- CriticReport schema：2.0；
- Critic model ID：`pavg-critic-0.3.0/configs-default`；
- Critic config SHA-256：`f8945bbb675f34215a8440793d25fd3f0df907a44aee5252e994fb666e91d4f4`；
- Critic schema SHA-256：`142af7ade55a5866bbe87c1a37f90b20b11959fa81302a5c957d5c8cce721518`；
- feature schema SHA-256：`b6e4a8cd71077b59d749dfba359e163cd4f7afd4623e401c13458d65386f5596`；
- action order：Prompt / Global / Local / Reject；
- source revision：`unknown`；
- deployment ready：false。

部署与训练必须 fail fast：

- Critic schema/hash 不兼容则拒绝；
- feature schema 与 checkpoint 不一致则拒绝；
- action order 不一致则拒绝；
- Executor 不支持的动作必须 mask；
- `source_revision=unknown` 只能用于实验；
- proxy 与 Actual Trial 指标必须分开；
- Hunyuan calibration/test 必须冻结且不重叠。

---

## 十、已完成与尚未完成

### 已完成

- Blender proxy v1/v2/v3 数据、训练、评测与 release；
- 1,200 group / 22,200 sample v3 Action-Value Policy；
- group-safe split 与 target audit；
- train-only proxy Memory；
- 四动作 release smoke；
- Blender-free inference bundle；
- capability mask、Trial recorder、campaign、per-action target/value、分域评测和 integration-review 工程骨架；
- 本地轻量测试证据；
- proxy 结果限制与兼容性清单。

### 尚未完成

- 真实 Prompt Repair backend；
- 真实 Global Regeneration backend；
- 真实 Local Editing backend；
- Actual `RepairTrialV1` 数据；
- action-specific counterfactual reward；
- Blender Actual Trial 训练；
- Hunyuan calibration/test rollout；
- value-led selection 的真实数据启用；
- deployable source revision；
- 团队主线集成批准；
- 端到端 physics gain、semantic preservation、quality、cost 与新增错误率结论。

---

## 十一、下一阶段执行顺序

1. 先由团队决定 Critic 稳定基线：同步或拒绝运行副本中的 Critic/report 改进，再冻结 clean source revision。
2. 默认保持 `LearningRepairLoopRunner` 为并行入口，不直接扩大共享 Controller 改动。
3. 优先接入真实 Prompt Repair 与 Global Regeneration；Local Editor 未就绪时保持 mask。
4. 在 Blender Actual Trial 中对同一 broken candidate 实际执行所有可用动作，并重新运行冻结 Critic。
5. 固定质量门禁和 reward，在读取结果后不调整：

```text
valid repair:
  after_physics_score >= 0.80
  semantic_score      >= 0.85
  quality_score       >= 0.75

reward = physics_gain
       + 0.30 * semantic_score
       + 0.20 * quality_score
       - 0.10 * normalized_cost
```

6. 将失败动作记录为真实负 utility；未执行动作继续保持 unknown。
7. 用 Actual Trial 重新训练 per-action value Policy，并与 R0–R4 做未见 group/template 对照。
8. 冻结 Hunyuan calibration/test，采集真实 Prompt/Global rollout；Local 可用后再加入。
9. 只有真实闭环在 physics gain、成功率或成本上产生预注册增益，同时 semantic/quality 通过门槛，才申请主线集成。

---

## 十二、主要本地证据路径

```text
artifacts/repair_training/repair_600g_v1_report.{md,json}
artifacts/repair_training/repair_900g_v2_report.{md,json}
artifacts/repair_training/repair_1200g_v3_report.{md,json}
artifacts/repair_training/final_v3/repair_agent/
artifacts/repair_training/repair_agent_repair_1200g_v3_action_value.tar.gz
artifacts/learning_repair_pipeline/local_test_evidence.json
artifacts/learning_repair_pipeline/proxy_baseline_verification.json
artifacts/learning_repair_pipeline/compatibility_verification.json
artifacts/learning_repair_pipeline/integration_review_local_v2/
docs/learning-repair-agent.md
docs/learning-repair-milestones-1-5.md
src/physgenloop/learning_repair/
src/physgenloop/learning_repair_pipeline/
```

这些路径当前属于本地工作成果；在团队评审、清理 dirty worktree 和确定提交边界之前，不应整体覆盖服务器主仓库。

---

## 十三、协作声明

后续工作继续遵守：

- 不覆盖他人未提交工作；
- 不修改 Critic 判断语义；
- 不把 proxy 结果包装为 Hunyuan 成功率；
- 不把运行副本能力描述为 main 已合并能力；
- 不把 fake backend 测试描述为真实 Executor 验收；
- 新增优先、共享契约冻结、结果可审计；
- 任何主线 Controller、Protocol、schema 变更先提交方案并获得团队批准。

---

**署名：hejin**  
**日期：2026-07-17**
