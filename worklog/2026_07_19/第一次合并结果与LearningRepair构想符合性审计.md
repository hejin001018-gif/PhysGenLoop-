# 第一次合并结果与 Learning Repair 构想符合性审计

> 日期：2026-07-19  
> 审计对象：服务器 `/root/PhysGenLoop-` 当前工作树及 `/root/PhysGenLoop-/worklog/2026_07_19/第一次合并结果.md`  
> 审计方式：只读文档、源码、Git 状态与已有运行产物核验；未修改服务器代码，未启动 GPU 生成或训练任务  
> 结论等级：**部分符合——研究架构骨架基本存在，但当前生产入口尚未实现构想中的四动作 Learning Repair 闭环，不能视为最终结果版**

---

## 1. 对照基准

本次审计以项目 README、Learning Repair 路线方案和此前训练边界为准。目标闭环应为：

```text
Prompt / PhysicsPlan
  → Video Generator
  → Frozen Physics Critic
  → CriticReport（完整诊断与证据）
  → Learning Repair Policy
  → RepairSelector（强制 capability mask）
  → Prompt / Global / Local / Reject Executor
  → 新 Candidate
  → 同一 Critic 复评
  → RepairTrialV1
  → Repair Memory / Action-Value 更新
```

关键边界：

1. Blender v3 当前是 proxy 预训练，不能包装成真实 Wan/Hunyuan 修复成功率。
2. Policy 的“动作选择”和 Executor 的“动作执行”必须分离。
3. 不可用动作必须被 capability mask，而不是退化成另一种动作。
4. 真实闭环必须保存 before/action/after、physics gain、semantic/quality、失败原因和成本。
5. 当前 Critic 原则上固定；Repair Agent 必须消费与训练时兼容的完整 `CriticReport`。
6. 部署阶段不依赖 Blender，但必须保留 Critic、Repair Agent、匹配版本的 Executor 和模型权重。

---

## 2. 总体判断

### 2.1 符合的部分

- `src/physgenloop/learning_repair/` 已成为 canonical Learning Repair 包。
- 四动作契约、`RepairSelector`、`ExecutorRegistry`、`LearningRepairLoopRunner`、`ActualTrialCampaign`、`JsonlTrialRecorder`、Memory 与 Action-Value 训练代码均存在。
- v3.1 checkpoint 和 proxy Memory 已部署到服务器；release manifest 如实标注：
  - `selection_mode=classification_proxy`；
  - `actual_trial_count=0`；
  - `deployment_ready=false`。
- `WanSubprocessGenerator` 与 `Sam2VlmSubprocessCritic` 已接入 `LoopController`，已有产物证明“一次真实 Wan 生成 + 一次 Critic 评分 + accepted”可以完成。
- README 仍正确声明真实 Prompt/Global/Local rollout 和 Hunyuan calibration/test 尚未完成，没有把 proxy 指标写成真实修复成功率。

### 2.2 不符合的核心

当前真正运行的入口为：

```text
run_loop.py
  → LoopController
  → WanSubprocessGenerator
  → Sam2VlmSubprocessCritic
  → EvidenceAwareSelector（候选选择）
  → ActionValueRepairer（统一改写 prompt）
  → 下一轮全局重新生成
```

它不是：

```text
CriticReport
  → RepairSelector
  → ExecutorRegistry
  → 四动作分别执行
  → Critic 复评
  → RepairTrialV1
```

因此，当前成果准确定位应是：

> **真实 Wan + Critic 的单轮生成评估入口，以及 proxy Repair Policy 的 prompt-feedback 适配演示；不是四动作 Learning Repair Agent 的真实闭环，也不是 Actual Trial 训练数据采集系统。**

---

## 3. 不符合项明细

### P0-1：生产入口没有执行四种 Repair Action

**构想要求**

Policy 输出 `prompt_repair / global_regeneration / local_editing / reject` 后，由匹配的 Executor 执行。

**实际代码**

- `agents/wanphysics/run_loop.py:96-110` 构造的是 `LoopController + ActionValueRepairer`，没有构造或注入 `ExecutorRegistry`。
- `agents/wanphysics/batch_run_loop.py:76-90` 同样只使用 `ActionValueRepairer`。
- `src/physgenloop/controller.py:98-101` 只接受 `repairer.repair(...) -> str`，下一轮始终调用同一个 Generator。
- `generators/wanphysics/repairer.py:71-75` 对四种动作的处理只有“在 prompt 后追加不同前缀”。
- 全仓调用点核验显示 `build_executor_registry()` 只有定义和文档示例，没有真实入口调用。

**直接后果**

- `global_regeneration` 没有独立 Executor 语义；
- `reject` 不会终止，而会被写成 `Replacement constraint` 后继续生成；
- `local_editing` 没有执行局部编辑；
- 四动作标签在生产路径中被压扁成同一种“改 prompt 后全局再生成”。

**结论**

这是与 Learning Repair 核心构想最直接的不符合项。

---

### P0-2：所谓“真实 Executor 工厂”当前无法成功构造

**实际代码与只读运行探针**

- `generators/wanphysics/executor_factory.py:74` 调用 `RejectExecutor()`。
- `src/physgenloop/learning_repair/executors.py:171-177` 要求必须传入 keyword-only `selector`。
- 只读 CPU 探针结果：

```text
RejectExecutor (*, selector, backend_id='candidate-selector')
RejectExecutor() → TypeError: missing required keyword-only argument: 'selector'
```

这意味着 `build_executor_registry()` 即使被入口调用，也会在构造阶段失败。

此外，工厂 docstring 中的 Runner 示例使用 `registry=...`，但 `LearningRepairLoopRunner` 构造参数实际名为 `executors`，且还需要 generator、policy、semantic scorer、quality scorer 等必填依赖；该示例也不能直接运行。

---

### P0-3：Local Editor 与 `LocalEditTarget` 契约不一致

**实际契约**

`src/physgenloop/learning_repair/contracts.py:111-153` 的 `LocalEditTarget` 字段是：

```text
parent_candidate_id
objects
start_frame / end_frame
critical_frames
mask_uri
```

**实际 Editor**

`generators/wanphysics/local_editor.py:121-130` 查找的是：

```text
target.mask_path
target.bbox
```

标准 `LocalEditTarget` 不包含这两个字段。只读探针确认：

```text
hasattr(target, "mask_path") == False
hasattr(target, "bbox") == False
```

所以即使 `mask_uri` 有值，当前 `_build_masks()` 仍会落入 `ValueError("缺少 mask_path 和 bbox")`。

另外，当前实现把同一个静态 mask 写到全部帧，未使用 `start_frame/end_frame/critical_frames` 控制编辑时段。ProPainter inpainting 本身主要用于移除/补背景，也尚无证据证明它能重建满足重力、接触、碰撞轨迹的物体运动。

**结论**

“Local Editing 已接入真实后端”的文档表述不成立；当前最多是未验收适配草稿。

---

### P0-4：CriticReport 在跨子进程返回时丢失 Repair Agent 的核心输入

**实际代码**

`generators/wanphysics/sam2_vlm_critic.py:33-49` 的 `_report_from_dict()` 只恢复：

```text
decision
physics_score
confidence
coverage
score_breakdown
diagnostics
```

没有恢复：

```text
violations
critical_frames
repair_instruction
evidence_bundles
node_results
model_versions
```

而 `src/physgenloop/learning_repair/features.py:179-235` 明确依赖 `violations` 的类别、对象、帧区间、repair instruction，以及五类 `evidence_bundles`。

**影响**

- 线上 Policy 输入分布与 Blender proxy 训练输入不一致；
- violation category、对象、时间段和 evidence 特征全部退化为默认值；
- `_target()` 无法为 Local Editing 生成对象、帧区间和 mask；
- 训练所得“错误类型 → 动作”映射在线上无法按设计发挥。

这比单纯少保存日志更严重，属于实际模型输入契约被破坏。

---

### P0-5：Controller 解析出的 PhysicsPlan 没有传给真实 Critic 子进程

**实际代码**

- `LoopController` 将 `resolved_plan` 传入 `Sam2VlmSubprocessCritic.evaluate(...)`。
- 但 `sam2_vlm_critic.py:142-167` 写给子进程的 JSON 中没有 `physics_plan`。
- `agents/wanphysics/eval_step.py:68-73` 固定使用新的空 `PhysicsPlan()`。

**影响**

Planner 与 Critic 的预期检查条件在进程边界处断开。系统图中宣称的“PhysicsPlan Resolver → Critic”没有在真实运行中保持。

---

### P0-6：40GB A100 的多轮 GPU 交接逻辑只支持第一轮

**实际控制流**

1. 第一轮 Wan 子进程退出；
2. Critic 启动 vLLM；
3. `Sam2VlmSubprocessCritic` 将 vLLM 保持常驻；
4. 若 Critic 判为 violation，`LoopController` 进入第二轮并再次启动 Wan；
5. vLLM 只在整个 `controller.run()` 结束后的 `finally` 中关闭。

因此第二轮 Wan 会在 vLLM 仍占约 16.6 GiB 显存时加载。`batch_run_loop.py` 也会在第二个 prompt 生成前保留同一 vLLM。

这与 `run_loop.py` 和合并文档所述“顺序化 GPU 交接”不一致。已有旧入口日志 `outputs/e2e_loop_full_v2.log` 也记录过 40GB 卡上 Wan 与其他进程并存导致 CUDA OOM。

**影响**

目前唯一成功演示恰好第一轮 accepted，没有覆盖真正需要 Repair 的第二轮。因此不能证明多轮闭环能在该硬件配置上运行。

---

### P0-7：没有真实 `RepairTrialV1`，也没有实际训练数据采集入口

**调用点核验**

- `LearningRepairLoopRunner(...)` 的真实调用只有 fake 测试；生产入口没有调用。
- `ActualTrialCampaign(...)` 由通用 `run-campaign` 框架支持，但仓库中没有把 Wan/Blender 实际后端组装成 `CloudBackendBundle` 的生产 factory。
- `JsonlTrialRecorder(...)` 只在通用 campaign 代码和 fake 测试中出现。
- 服务器项目目录中未发现任何文件名含 `trial` 的实际 Trial 产物。

**与文档的差异**

`第一次合并结果.md:109` 将职责写成“`LearningRepairLoopRunner` 负责训练数据采集”，但当前只有可复用框架，没有服务器侧真实调用与数据。

**影响**

- 无 before/action/after；
- 无真实 physics gain；
- 无 semantic/quality gate；
- 无失败动作负 utility；
- 无 Actual Trial Memory；
- 不能从 proxy 模式切换到 value-led 模式。

---

### P0-8：当前 checkpoint 与合并后的 Critic 哈希不兼容，但生产加载未阻止

服务器当前文件哈希：

| 文件 | 当前 SHA-256 |
|---|---|
| `configs/default.yaml` | `f2b1f65b9417faf96b78880e6fe3bc14344db54d84e9e323f2f47beb2c206b89` |
| `schemas/critic_output.schema.json` | `534e608d5230c39f6b7c812c6121e94e2c21d2c612d7b967d2b0cf21d7516052` |

生产入口加载的 checkpoint bundle 仍声明旧哈希：

| 项目 | checkpoint 声明 |
|---|---|
| Critic config | `f8945bbb675f34215a8440793d25fd3f0df907a44aee5252e994fb666e91d4f4` |
| Critic schema | `142af7ade55a5866bbe87c1a37f90b20b11959fa81302a5c957d5c8cce721518` |
| source revision | `unknown` |
| deployment ready | `false` |

`configs/learning_repair/critic_compatibility_v1.json` 虽已重新冻结为当前哈希，但 `load_action_value_repairer()` 使用的是 checkpoint bundle 内的旧 manifest，只校验 checkpoint 与该旧 manifest 相互一致，没有调用 `verify_files()` 对照当前 Critic 文件，也没有拒绝 `deployment_ready=false`。

**影响**

运行成功只能说明旧 proxy checkpoint 能被反序列化，不能说明它与当前 Critic 输入分布兼容。

---

### P1-1：Repair Memory 存在于发布包，但生产推理没有使用

checkpoint 中有约 9.9 MB 的 `memory/proxy_memory_train.jsonl`，代码也有 `ActualTrialMemory`、`MemoryValuePredictor` 和 `BlendedValuePredictor`。

但 `generators/wanphysics/repairer.py:82-92` 只加载 `TorchActionValuePolicy`，没有加载 Memory，也没有构造 blended predictor。

所以当前生产入口不是“Policy + Memory”的 Repair Agent，只是单一 proxy Policy。

---

### P1-2：Prompt Executor 会再次调用 Policy，破坏动作决策与执行分离

如果未来直接启用当前 `executor_factory.py`：

- Runner/Actual Campaign 已经先生成 `request.decision`；
- `PromptRepairExecutor.execute()` 却调用 `prompt_rewriter.repair(...)`；
- 注入的 `prompt_rewriter` 是 `ActionValueRepairer`，它会再次运行 Policy 并递增自己的 attempt state；
- Executor 没有直接执行第一次 `request.decision.instruction`。

这会导致“一次 Trial 两次决策”，forced-action campaign 也可能记录 Prompt Action，却实际按第二次选择生成不同前缀。

---

### P1-3：真实闭环没有 semantic/quality guardrail

`LearningRepairLoopRunner` 和 `ActualTrialCampaign` 设计了 physics、semantic、quality 三项门槛，但生产 `LoopController` 的接受条件只有：

```text
decision == physical
and physics_score >= acceptance_score
```

因此当前生产路径无法防止“物理分提高但语义或视觉质量损坏”的修复被接受。

---

### P1-4：产物不足以审计 Repair 行为

`outputs/run_20260719_170012/loop_result.json` 只有：

```json
{
  "stop_reason": "accepted",
  "best_physics_score": 1.0,
  "best_decision": "physical",
  "rounds": 1
}
```

该目录没有：

- 完整 `CriticReport`；
- RepairDecision；
- Executor action/status；
- before/after；
- Trial JSONL；
- detector backend provenance。

单 prompt 入口实际也没有写合并文档架构图中列出的 `critic.json`；只有批量入口会写一个缩减版 `critic.json`。

`vllm.log` 证明发生过一次 VLM 请求，但不能证明 Repair Agent 或任何 Executor 被调用。

---

### P1-5：自动降级 Critic 的 provenance 没有保留到最终结果

`eval_step.py` 对 SAM2 构造的任意异常都会降级为默认规则 Critic。`detector_backend` 虽写入临时 payload，但 `Sam2VlmSubprocessCritic` 返回时将其丢弃，最终 `loop_result.json` 也不保存。

这使团队无法从最终产物确认某轮到底使用 `sam2+vlm` 还是 `rules_fallback`，不利于固定 Critic 与 Trial 可审计性。

---

### P1-6：`LoopController` 在最后一轮失败后仍调用 Repairer，但结果不会被消费

`src/physgenloop/controller.py:98-101` 在每轮未 accepted 时都调用 Repairer，包括最后一轮。最后一次更新后的 prompt 不会再生成候选，却会改变 `ActionValueRepairer` 的状态。

这不会影响已生成视频，但会制造多余决策，且如果未来记录 Trial，会产生“有动作、无执行”的审计歧义。

---

## 4. “第一次合并结果”文档与实际证据的差异

| 文档声明 | 实际证据 | 审计结论 |
|---|---|---|
| 四个 Executor 已注入真实后端 | 工厂无调用点，且 `RejectExecutor()` 构造即报错 | 不成立 |
| ProPainter Local Editor 已完成局部修复 | target 字段不匹配，标准输入必然无法构造 mask | 不成立 |
| LearningRepairLoopRunner 负责训练数据采集 | 只有 fake 测试调用，无实际 Trial 文件 | 仅框架存在 |
| ActionValueRepairer 根据报告决定修复动作 | 能做动作分类，但所有动作最终都被压成 prompt 改写 | 部分成立 |
| GPU 顺序交接支持完整闭环 | vLLM 在第二轮 Wan 前不会关闭 | 只证明第一轮 |
| E2E 完整闭环跑通 | 结果为首轮 accepted，未触发 Repair | 只证明生成与评估 |
| 405 tests 证明合并结果 | 7 月 19 日新增入口和五个集成模块均无测试引用 | 不能覆盖本次集成层 |
| 代码已形成服务器 main 最终结果 | 关键集成文件全部未跟踪/未提交 | 不可复现 |

---

## 5. Git 与可复现性风险

审计时服务器状态：

```text
main...origin/main [ahead 133, behind 3]
HEAD = d1d7595b111ac2b07d184404af4c5604cd6d2817
integration = 968b701206d48b3e3c8309084736752e3d3ff4c3
origin/main = 13573b95afd4687f32439a47f82d2d538a320e66
```

关键新增项均未被 Git 跟踪，包括：

```text
agents/
configs/loop.yaml
generators/wanphysics/adapter.py
generators/wanphysics/executor_factory.py
generators/wanphysics/local_editor.py
generators/wanphysics/repairer.py
generators/wanphysics/sam2_vlm_critic.py
```

另有已修改未提交的 `wan_generator.py`、`requirements.txt`，以及 staged/modified 的 worklog。

`git ls-files` 对上述核心集成文件返回为空。这意味着：

- 当前成功演示不能从任一 commit clean checkout 复现；
- `第一次合并结果.md` 所称“推送到服务器 main”不能等价为“已形成可共享版本”；
- 405 tests 对应的是此前合并基线，不是这些未跟踪的 7 月 19 日集成代码。

---

## 6. 当前可以证明与不能证明的内容

### 可以证明

1. Wan2.2 可以生成一个 81 帧候选视频。
2. vLLM/Qwen3-VL 服务成功启动并收到一次请求。
3. 当前 Critic 对该候选给出 `physical / 1.0`。
4. `LoopController` 在首轮 accepted 时能正确停止。
5. proxy Repair checkpoint 能在 CPU 上加载，canonical Learning Repair 研究代码存在。

### 不能证明

1. violation 后第二轮能在单张 40GB A100 上完成。
2. Policy 选择的四动作会由对应 Executor 执行。
3. Prompt、Global、Local、Reject 四条路径均工作。
4. Local Editing 能生成可用视频。
5. 修复后 physics score 确实提高。
6. 修复保持原始语义和视觉质量。
7. `RepairTrialV1` 已被真实采集。
8. Repair Memory 已参与生产决策。
9. 当前 checkpoint 与合并后的 Critic 兼容。
10. proxy 满分指标能够迁移到 Wan/Hunyuan 实际视频。

---

## 7. 建议的最小整改优先级（本次不实施）

### 第一优先级：统一真实运行主路径

团队应明确二选一：

1. 生产和采集统一使用 `LearningRepairLoopRunner + ExecutorRegistry`；或
2. 经团队批准后扩展 `LoopController`，使其原生执行 `RepairDecision` 和 Executor。

无论选择哪条路线，都不能继续把四动作压成 PromptRepairer 字符串。

### 第二优先级：修复 Executor 契约

- 给 `RejectExecutor` 注入候选 Selector；
- 让 Local Editor 直接消费 `mask_uri + frame interval + critical_frames`；
- Prompt Executor 执行已有 decision，不得再次运行 Policy；
- 为每个真实 backend 生成 capability manifest，并在运行前 fail fast。

### 第三优先级：恢复完整 Critic 输入

- 跨子进程无损序列化/反序列化完整 `CriticReport 2.0`；
- 传递 Controller 的 resolved `PhysicsPlan`；
- 保存 detector backend、model versions 和 fallback provenance。

### 第四优先级：先跑一个确定会触发 Repair 的 E2E

至少覆盖：

```text
wrong candidate
  → violation
  → RepairSelector
  → 一个真实 Executor
  → 新 candidate
  → Critic re-evaluation
  → RepairTrialV1 落盘
```

并证明第二轮 GPU 生命周期不会 OOM。随后再覆盖四动作和 batch。

### 第五优先级：兼容性与 Actual Trial

- 对当前 Critic clean revision 重新冻结 manifest；
- 生产入口强制检查 `deployment_ready`、文件哈希、schema、action order；
- 引入 semantic/quality scorer；
- 采集真实多动作 Trial 后重新训练，再从 `classification_proxy` 切换到 `action_value`；
- Memory 只使用 train/calibration 数据，测试集保持隔离。

### 第六优先级：形成可复现提交

- 为新入口、GPU lifecycle、Critic round-trip、factory、四 Executor、Trial recorder 增加无 GPU 单元测试和最小 GPU smoke；
- 清理提交边界后提交到明确分支；
- 用 clean checkout 复验，不再以 dirty server workspace 作为最终交付物。

---

## 8. 最终结论

服务器代码在“组件是否存在”层面已覆盖 Learning Repair 构想的大部分名词和接口，但在“真实运行时是否按构想连接”层面仍有明显断层。

最关键的判断是：

> **当前有两个 Selector：生产使用 `EvidenceAwareSelector` 选择候选，Policy 内部使用 `RepairSelector` 选择动作；也有 Executor 框架。但生产入口没有使用 ExecutorRegistry，四动作没有被真实执行。因此不能称为完整 Learning Repair Agent 闭环。**

建议团队把本次状态标记为：

```text
Wan + Critic 单轮 E2E：已验证
Proxy Repair Policy 加载：已验证
四动作 Executor 闭环：未验证 / 当前不可用
Actual RepairTrial 采集：未开始
Value-led Repair Agent：未开始
最终结果版：尚未达到
```

本审计只生成此文档，没有修改服务器任何代码或删除任何文件。

---

**署名：hejin**  
**日期：2026-07-19**
