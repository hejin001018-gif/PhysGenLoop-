# PhysGenLoop 构想符合性审查

**审查日期：** 2026-07-19  
**审查对象：** `px-cloud1.matpool.com:27323:/root/PhysGenLoop-` 当前工作树  
**审查方式：** 只读查看 README、`worklog/` 设计与合并记录、源码、Git 状态、已有输出，并运行无 GPU 的构造/字段探针；未修改服务器代码、配置、权重或输出，未启动生成/训练任务。

## 结论摘要

当前仓库是“核心组件已合并、Learning Repair 研究框架已具备、Wan + Critic 首轮演示可运行”，但还不是 README/合并文档所描述的完整四动作闭环。

最关键的事实是：

```text
当前生产入口：run_loop.py
  → LoopController
  → WanSubprocessGenerator
  → Sam2VlmSubprocessCritic
  → EvidenceAwareSelector
  → ActionValueRepairer（改写 prompt）

构想中的真实闭环：
  CriticReport
  → RepairSelector
  → ExecutorRegistry
  → prompt/global/local/reject 对应执行
  → Critic 复评
  → RepairTrialV1 / Memory / value update
```

前者不能证明后者。当前状态应标记为：

| 能力 | 状态 |
|---|---|
| Wan2.2 单轮生成 | 已有运行产物证明 |
| SAM2 + Qwen Critic 首轮评分 | 已有运行产物证明 |
| Planner/PQSG/Checklist/Mechanics 代码 | 组件存在；完整模型注入未接入生产入口 |
| Proxy Repair Policy 加载 | 已有 checkpoint，可在 CPU 加载 |
| 四动作 Executor 真实执行 | 未验证；当前生产入口未调用 Registry |
| 第二轮 GPU 交接 | 未证明，当前实现存在显存生命周期风险 |
| `RepairTrialV1` 真实采集 | 未开始；`actual_trial_count=0` |
| Value-led / Memory 生产决策 | 未接入 |
| VideoSearch 模块 | 当前树中未发现实现或调用 |

## 已确认符合的部分

1. `src/physgenloop/learning_repair/` 中存在四动作契约、`RepairSelector`、`ExecutorRegistry`、`LearningRepairLoopRunner`、`ActualTrialCampaign`、Trial recorder、Memory 和训练代码。
2. `WanSubprocessGenerator`、`Sam2VlmSubprocessCritic` 和 `LoopController` 已被顶层入口组装，已有 `outputs/run_20260719_170012/loop_result.json` 证明一次 Wan 生成、一次 Critic 评估和首轮 `accepted` 可以完成。
3. 服务器主线无 GPU 测试通过 `405 passed in 3.28s`。这证明单元/契约测试基线通过，不等价于本次新增集成层已被真实闭环覆盖。
4. checkpoint 发布清楚标注 `selection_mode=classification_proxy`、`actual_trial_count=0`、`deployment_ready=false`、`proxy_label_count=22200`。README 也明确写出真实 Prompt/Global/Local rollout 和 Actual Repair Trial 尚未完成，这些声明是诚实的。
5. VideoScience 风格 Checklist、Morpheus 风格 mechanics、PQSG 启发的 graph 和 evidence fusion 在 `src/pavg_critic/` 中有实现与测试。

## 不符合项（按阻断程度排序）

### P0-1：生产入口没有使用四动作 ExecutorRegistry

服务器 `agents/wanphysics/run_loop.py:30-36, 104-110` 只构造 `LoopController`、`WanSubprocessGenerator`、`Sam2VlmSubprocessCritic`、`ActionValueRepairer` 和 `EvidenceAwareSelector`；没有导入或调用 `build_executor_registry()`。

`src/physgenloop/controller.py:18-38, 55-108` 的接口只接受 `PromptRepairer`，每轮失败后调用 `repairer.repair(...)` 并继续同一个 Generator。`generators/wanphysics/repairer.py:23-28, 48-79` 把四个动作都映射为 prompt 前缀：

- `prompt_repair` → `Physics correction`；
- `global_regeneration` → `Regeneration constraint`；
- `local_editing` → `Local-edit fallback constraint`；
- `reject` → `Replacement constraint`。

因此 `reject` 不终止，`local_editing` 不调用局部编辑器，`global_regeneration` 与普通下一轮生成没有独立执行语义。四动作标签被压扁成了“改 prompt 后重新生成”。

### P0-2：Executor 工厂当前不能构造

`generators/wanphysics/executor_factory.py:59-75` 直接调用 `RejectExecutor()`；但 `src/physgenloop/learning_repair/executors.py:171-177` 要求 keyword-only `selector`。

服务器无 GPU 探针结果：

```text
RejectExecutor() → TypeError: missing required keyword-only argument: 'selector'
```

工厂 docstring 还使用 `LearningRepairLoopRunner(registry=...)`，而真实构造器参数是 `executors=...`（`runner.py:46-75`）。所以即使把工厂接到入口，当前调用示例也不能直接运行。

### P0-3：Local Editor 与 LocalEditTarget 字段契约不一致

标准 `LocalEditTarget`（`src/physgenloop/learning_repair/contracts.py:111-153`）字段为：

```text
parent_candidate_id, objects, start_frame, end_frame,
critical_frames, mask_uri
```

`generators/wanphysics/local_editor.py:121-130` 却读取 `target.mask_path` 或 `target.bbox`。服务器字段探针确认：

```text
has_mask_path=False
has_bbox=False
```

因此标准契约即使提供 `mask_uri`，也会落入“缺少 mask_path 和 bbox”。此外，`local_editor.py:132-135` 将同一张 mask 写给所有帧，没有使用 `start_frame/end_frame/critical_frames` 控制编辑区间。ProPainter 是否能修复物理轨迹也尚无真实成功率证据。

### P0-4：跨子进程的 CriticReport 丢失 Repair Agent 特征

`src/pavg_critic/schemas.py:825-844` 的 `CriticReport` 包含 `violations`、`node_results`、`model_versions`、`evidence_bundles` 等字段。

`generators/wanphysics/sam2_vlm_critic.py:33-49` 的 `_report_from_dict()` 只恢复 `decision`、三个分数、`coverage`、`score_breakdown` 和 `diagnostics`，其注释也明确将嵌套对象保持为空。

而 `src/physgenloop/learning_repair/features.py:179-231` 读取 violation 的类别、对象、帧区间、repair instruction 和 evidence family。结果是线上 Policy 输入与训练输入不一致，Local Editing 所需对象/时间段也无法从这个报告恢复。

### P0-5：PhysicsPlan 在真实 Critic 子进程边界丢失

`LoopController` 确实在 `controller.py:63-72` 解析并传入 `physics_plan`，但 `Sam2VlmSubprocessCritic.evaluate()` 写入候选 JSON 时（`sam2_vlm_critic.py:142-167`）没有传入 plan；`agents/wanphysics/eval_step.py:68-73` 又固定构造空的 `PhysicsPlan()`。

因此“Planner → Critic”在真实子进程路径中没有保持，生成阶段的 resolved plan 不会成为 Critic 的检查条件。

### P0-6：完整 PAVG Critic 的模型化边界没有接入生产入口

`src/pavg_critic/pipeline.py:76-154` 支持注入 `planner_model`、`question_model`、visual evidence extractors 和 `vlm_verifier`；未注入时会使用模板 Planner、模板 graph 和 `NoOpVLMVerifier`。

但 `agents/wanphysics/eval_step.py:61-73` 只创建一个 `OpenAIChatModel`，再调用默认的 `PhysicsCritic(detector=detector)`，没有传入 `planner_model`、`question_model`、`vlm_verifier` 或 visual extractors。故生产演示并不能声称已经运行“模型 Planner + PQSG + Candidate VLM verification”的完整 Critic；Checklist 和 mechanics 代码虽启用，但其上游模型化证据并未完整接通。

### P0-7：40GB A100 的第二轮显存交接未实现

`Sam2VlmSubprocessCritic` 在 `sam2_vlm_critic.py:89-113` 拉起 vLLM，在 `:127-129` 仅由整个 `run_loop.py` 的 `finally` 关闭（`run_loop.py:112-115`）。

当第一轮判为 violation 时，`LoopController` 会进入下一轮并在 `controller.py:55-68` 再次调用 Wan Generator；此时 vLLM 仍常驻，占用约 16GB 级别显存。单张 40GB A100 上不能由“首轮 accepted”证明第二轮不会 OOM。`batch_run_loop.py:150-192` 对多个 prompt 也保持同一 vLLM 到整个 batch 结束。

### P0-8：没有真实 Trial 采集生产入口

仓库中的 `LearningRepairLoopRunner`、`ActualTrialCampaign` 和 `JsonlTrialRecorder` 只有框架实现、cloud campaign 支持和 fake 测试调用；`run_loop.py`/`batch_run_loop.py` 没有实例化它们，也没有调用 `build_executor_registry()`。

服务器 `outputs/` 未发现 Trial JSONL 或实际 repair campaign 产物；checkpoint `release_manifest.json` 明确记录 `actual_trial_count=0`。因此当前不能证明 before/action/after、physics gain、semantic/quality gate、失败动作负 utility 或 Actual Trial Memory 已产生。

## P1 风险与偏离

### P1-1：Memory 没有进入生产推理

checkpoint 内有约 9.9MB `memory/proxy_memory_train.jsonl`，但 `generators/wanphysics/repairer.py:82-92` 只加载 `TorchActionValuePolicy`，没有加载 Memory 或构造 blended predictor。当前生产是单一 classification proxy，不是“Policy + Memory”。

### P1-2：Prompt Executor 会再次调用 Policy

`PromptRepairExecutor.execute()`（`executors.py:43-65`）优先调用注入对象的 `.repair()`。如果注入 `ActionValueRepairer`，Runner 已经先由 Policy 生成 `request.decision`，Executor 又会重新运行 Policy 并推进自己的 attempt state，形成一次 Trial 两次决策，破坏 action/execute 分离。

### P1-3：生产接受条件没有 semantic/quality guardrail

`src/physgenloop/controller.py:87-90` 只检查 `decision == physical` 和 physics score 阈值。虽然 `LearningRepairLoopRunner` 支持 semantic/quality scorer 和 `RewardSpec`，生产入口没有使用它们，可能接受物理分数上升但语义或画质下降的候选。

### P1-4：最终产物缺少完整审计证据

`outputs/run_20260719_170012/loop_result.json` 只保存 run id、best candidate、physics score、decision 和轮数；没有完整 CriticReport、RepairDecision、Executor status、before/after、Trial record 或 detector provenance。`sam2_vlm_critic.py:177` 返回时也丢弃 `detector_backend`。

### P1-5：Critic 异常会静默降级且 provenance 不进入最终结果

`agents/wanphysics/eval_step.py:26-42` 捕获 SAM2 任何异常后退回默认 `PhysicsCritic()`。虽然临时 payload 写入了 `detector_backend`，但上层 `_report_from_dict()` 丢掉该字段，最终用户无法知道一轮实际使用了 `sam2+vlm` 还是 `rules_fallback`。

### P1-6：最后一轮失败后仍调用 Repairer

`src/physgenloop/controller.py:98-101` 在每一轮未接受后都调用 `repairer.repair()`；即使已经是最后一轮，更新后的 prompt 也不会再生成。未来接入 Trial recorder 时会产生“有动作、无执行”的记录歧义。

### P1-7：README 的相对路径约束与实现不一致

README §2 要求 pipeline 使用 `./data`、`./models`、`./outputs`，禁止硬编码绝对路径；但 `run_loop.py:23-28, 38`、`batch_run_loop.py:30-45`、`eval_step.py:14-19` 直接写入 `/root/PhysGenLoop-`，`configs/loop.yaml` 也全部使用绝对路径。代码只能在当前服务器目录布局下复现。

### P1-8：VideoSearch 未形成可执行模块

对当前树的文件名、源码和 worklog 搜索未发现 `VideoSearch`、`Video Search`、`video_search` 或对应调用。若 VideoSearch 是构想中需要参与 Critic/Repair 的模块，它目前既没有接口、配置、运行入口，也没有评测证据。

## Git 与可复现性

审查时服务器状态为：

```text
main...origin/main [ahead 133, behind 3]
HEAD d1d7595
```

`agents/`、`configs/loop.yaml`、`generators/wanphysics/adapter.py`、`executor_factory.py`、`local_editor.py`、`repairer.py`、`sam2_vlm_critic.py` 等关键集成文件仍是未跟踪文件；`wan_generator.py` 和 `requirements.txt` 还有未提交修改。`git ls-files` 对这些核心集成文件为空。

因此 405 个测试通过的是已检出的测试树，不能证明一个 clean checkout 能复现 7 月 19 日首轮演示。当前 dirty workspace 不应作为最终交付版本或训练数据采集基线。

## 最小整改顺序（本次不实施）

1. 明确唯一主路径：让生产入口直接使用 `LearningRepairLoopRunner + ExecutorRegistry`，或经团队批准扩展 `LoopController` 原生消费 `RepairDecision`；不能继续把四动作压成 PromptRepairer。
2. 修复 Executor 契约：注入 `RejectExecutor.selector`，让 Prompt Executor 执行已有 decision，Local Editor 消费 `mask_uri + frame interval + critical_frames`，并为能力 mask 做 fail-fast 检查。
3. 跨进程无损传递完整 `CriticReport` 和 `PhysicsPlan`，保存 detector/model/fallback provenance。
4. 在第二轮显存约束下跑一个确定 violation → repair → re-critic 的 GPU smoke，再覆盖四动作。
5. 引入 semantic/quality scorer、Trial JSONL 和兼容性文件校验；实际 Trial 达到目标数量后再训练 value-led policy。
6. 清理 dirty workspace，提交新增集成文件，用 clean checkout 重跑 `405` 测试及最小 GPU smoke。

## 最终判定

```text
Wan + Critic 单轮 E2E：已验证
Proxy Repair Policy 加载：已验证
PAVG 模块源码存在：已验证
完整模型化 PAVG Critic 生产链：未验证
四动作 Executor 闭环：未验证，且当前入口未接入
Actual RepairTrial 采集：未开始
Value-led Repair Agent：未开始
最终结果版：不符合，不能按完整闭环对外宣称
```

本审查文档没有修改服务器代码、配置、权重或运行产物。
