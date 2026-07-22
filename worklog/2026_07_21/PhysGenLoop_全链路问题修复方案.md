# PhysGenLoop 全链路问题修复与补全方案（当前服务器增量 V2 版）

> 文档版本：`2.0-draft`  
> 当前复核日期：2026-07-21（Asia/Shanghai）  
> 服务器基准：`/root/PhysGenLoop-`，`main@13c6289`，工作树干净  
> 证据范围：根 `README.md`、`worklog/2026_07_14` 至 `worklog/2026_07_20`、当前源码、迁移前历史运行产物  
> 状态：方案文档，仅供评审；当前不对服务器做任何修改  
> 原则：只新增 V2 sidecar，不删除、不移动、不重命名、不覆盖旧模块和旧产物，不把 proxy 或内部 Critic 分数包装成真实部署结果。

## 0. 本版方案的结论

当前项目不应立即扩大样本数或直接重训 Repair Policy。最先需要解决的是六个逻辑阻断点：

1. 迁移后 Wan2.2、Qwen3-VL、SAM2 和 `.env` 已于 2026-07-21 09:25–09:30 恢复，但 ProPainter 仍缺失，且尚未做迁移后测试/GPU smoke；当前只能说核心资产到位，不能说全链路已恢复。
2. `CriticReport` 跨子进程有损反序列化，导致 `violations`、`critical_frames` 和 mask 证据在 Repair 前丢失。
3. 现有生产入口未形成严格的 `Decision → Executor → Re-Critic` 四动作状态机。
4. 当前 `sam2+vlm` 命名会高估实际 Critic profile：Qwen3-VL 主要用于 SAM2 首帧目标种子，没有接入模型 Planner、PQSG 和 Evidence-Grounded VLM Verifier。
5. 现有 checkpoint 是 `classification_proxy`，`actual_trial_count=0`、`deployment_ready=false`，且与当前 Critic 文件哈希不一致。
6. 全量 VideoPhy2 的 B1 Rule+SAM2 未优于 Direct VLM，正式闭环不能只用同一 Critic 给自己发奖励。

因此，本方案选择新增一条完全隔离的 `WanPhysics V2 Sidecar`，先建立可无损、可审计、可回放、可显式降级的真实闭环，再进入 Actual Trial 和 value-led Policy 阶段。

## 1. 当前运行结论

下述数据来自迁移前历史运行 `videophy2_run_20260720_132415`，不是 2026-07-21 在当前新服务器上重跑得到。历史链路为：

~~~text
videophy2 manifest
  → Wan2.2 子进程生成
  → SAM2 + VLM Critic
  → CriticReport
  → LoopController / Repair Policy
  → Prompt 改写后二次生成
  → Critic 复评
~~~

20 条 smoke 的实际结果：

| 指标 | 结果 |
|---|---:|
| 样本数 | 20 |
| 候选视频数 | 30 |
| Critic 评估数 | 30 |
| accepted | 12 |
| max_rounds | 8 |
| 接受率 | 60% |
| 平均轮数 | 1.5 |
| Prompt Repair | 10 次 |
| Local Editing | 0 次 |
| Global Regeneration | 0 次 |
| Reject | 0 次 |
| OOM | 0 次 |

该历史运行只证明 Wan2.2 曾经可以与当时的 SAM2-seeded Critic 串行运行，并完成有限轮次的 Prompt 二次生成；没有证明四动作闭环、Local Editing、Actual Trial、Proxy Memory、完整模型化 PAVG Critic 或 deployment-ready 已完成。当前新服务器已恢复 Wan/Qwen/SAM2 核心资产，但本轮按用户要求未运行任何验证，所以仍不能将“文件存在”等同于“运行已恢复”；ProPainter/Local Editing 仍明确不就绪。

## 2. 已确认的核心问题

### 2.1 CriticReport 跨进程反序列化丢失违规证据

当前代码尝试调用 Violation.from_dict() 和 CriticReport.from_dict()，但 schema 中没有这些方法。异常被捕获后，violations 被静默置为空。

结果：

~~~text
eval_step.py 内部有 violations / critical_frames / mask_uri
→ 主进程反序列化后丢失
→ Repair Policy 看不到局部错误
→ error_scope 几乎全部变成 global
→ Local Editing 无法触发
~~~

### 2.2 SAM2 已生成 mask，但没有接入 Repair 决策

实际产物中存在 18 个 sam2_masks 目录和 216 张 mask PNG，但所有进入修复决策的记录都显示：

~~~json
{"has_local_evidence": false}
~~~

问题不是没有局部证据，而是感知层到 Repair Controller 的证据链断开。

### 2.3 ProPainter 后端实际不存在

服务器没有：

~~~text
/root/ProPainter
/root/ProPainter/inference_propainter.py
~~~

但 ExecutorRegistry 仍注册 Local Editing。当前只是适配器存在，不是真实后端可用。

### 2.4 ExecutorRegistry 名义接入，实际只执行 Prompt 路径

当前 Controller 中：

- Prompt Repair：直接修改 current_prompt；
- Global Regeneration：直接恢复原始 prompt；
- Reject：Controller 直接返回；
- 只有 Local Editing 才调用 executor_registry.execute()。

本次运行没有触发 Local Editing，因此 ExecutorRegistry 没有被真实执行过。

### 2.5 Prompt Repair 不具备物理修复内容

10 次修复全部追加同一句：

~~~text
Physics correction: Execute the highest calibrated, available repair action.
~~~

该文本没有描述违规对象、违规帧、物理错误类别或轨迹/接触/碰撞约束。当前更像“增加抽象系统提示后随机重生成”，而不是基于 Critic 证据的物理修复。

### 2.6 Proxy Memory 格式和加载器不兼容

日志中重复出现：

~~~text
failed to load proxy memory
invalid repair sample 0: repair example target must be an object
~~~

checkpoint 使用 target_action + action_rewards，加载器期待 target + outcome。本次实际运行退化为 Policy only。

### 2.7 接受门槛没有使用置信度、语义和质量

12 个 accepted 候选的 Critic confidence 全部低于 0.5：

~~~text
0.3025：10 个
0.3745：2 个
~~~

当前接受条件只有：

~~~text
decision == physical
and physics_score >= 0.8
~~~

没有使用 confidence、coverage、semantic_score、quality_score 或独立验证。因此 accepted 只能说明当前 Critic 判定 physical，不能说明视频被高置信度验证为物理正确。

### 2.8 Trial 产物不是 canonical RepairTrialV1

20 个 trials.jsonl 共 30 行，但只有 10 行含有真正的 before/after 分数，且缺少：

- action；
- action probability；
- per-action value；
- executor backend；
- execution status；
- failure reason；
- semantic / quality score；
- repair cost；
- mask 和 critical frames；
- PhysicsPlan；
- compatibility provenance。

这些文件只是运行摘要，不是可直接训练 Action-Value Policy 的真实 Trial 数据。

### 2.9 Critic 审计产物不完整

30 个 critic.json 都只有最小字段，reason 全部为 null，physics_score、violations、critical_frames、mask_uri 均缺失。完整 CriticReport 在主进程读取后被删除，任务完成后无法复盘具体判定理由。

### 2.10 SAM2 后处理处于降级状态

30 次评估都出现：

~~~text
cannot import name '_C' from 'sam2'
Skipping the post-processing step
~~~

但 provenance 仍统一写成 sam2+vlm，无法区分完整 SAM2 和缺少 C 扩展的降级 SAM2。

### 2.11 vLLM 显存交接有效，但效率较低

本次没有 OOM，但 30 个候选触发了 30 次 vLLM 启动；engine 初始化总耗时约 836 秒，整个 20 条 smoke 约 3 小时。

### 2.12 测试覆盖没有覆盖真实集成入口

已有 405 passed，但没有覆盖 run_videophy2_loop.py、build_executor_registry、Sam2VlmSubprocessCritic、实际 Proxy Memory、ProPainter、mask round-trip、vLLM 生命周期和真实 Trial 格式。因此 405 个测试通过不能作为全链路验收证据。

## 3. 目标修复架构

~~~text
V2 Preflight / Run Manifest
  → videophy2 manifest / CSV / prompt
  → PhysicsPlan Resolver
  → ResourceCoordinator
      ├─ dual_gpu: Wan@GPU0, Critic@GPU1
      └─ single_handoff: 单卡串行回退
  → Wan2.2 Generator
  → LosslessCriticBridge
      ├─ 完整 CriticReport
      ├─ mask_manifest
      ├─ Critic profile / fallback provenance
      └─ raw payload on failure
  → ScopeGuardedRepairPolicy（只决策一次）
  → capability mask + evidence validation
  → DecisionOnlyExecutorRegistry
      ├─ DecisionPromptRepairExecutor
      ├─ OriginalPromptGlobalRegenerationExecutor
      ├─ MaskSequenceLocalEditingExecutor
      └─ AuditedRejectExecutor
  → 同一 PhysicsPlan + 冻结 Critic 立即复评
  → physics / confidence / coverage / semantic / quality gate
  → Full RepairTrace + WanRepairTrialV2
  → 可选适配到 canonical RepairTrialV1
~~~

关键原则：

1. Policy 只决策一次；
2. Executor 不再次调用 Policy；
3. 每个 action 必须执行对应后端；
4. 每个 action 必须立即复评；
5. CriticReport 不允许静默丢字段；
6. 没有 mask 时必须禁用 Local Editing；
7. proxy 与 actual trial 分开；
8. 旧入口保留，新增 action-aware V2 入口先旁路验证。
9. V2 不修改现有 `LoopController`、canonical Learning Repair 契约或旧 `configs/loop.yaml`。
10. 生产 Critic 与正式效果评价分离，防止 Repair Policy 只学会提高内部打分。
11. `rules_fallback`、proxy checkpoint、SAM2 降级等状态必须显式写入产物，不允许用统一的 `sam2+vlm` 名称掩盖。

## 4. 建议新增文件

~~~text
generators/wanphysics/v2/__init__.py
generators/wanphysics/v2/preflight.py
generators/wanphysics/v2/resource_coordinator.py
generators/wanphysics/v2/critic_codec.py
generators/wanphysics/v2/critic_profiles.py
generators/wanphysics/v2/critic_backend.py
generators/wanphysics/v2/mask_manifest.py
generators/wanphysics/v2/policy_guard.py
generators/wanphysics/v2/prompt_renderer.py
generators/wanphysics/v2/executors.py
generators/wanphysics/v2/memory_adapter.py
generators/wanphysics/v2/guardrails.py
generators/wanphysics/v2/artifacts.py
generators/wanphysics/v2/runner.py
generators/wanphysics/v2/trials.py
agents/wanphysics/run_videophy2_loop_v2.py
agents/wanphysics/run_actual_trials_v2.py
configs/loop_v2.yaml
schemas/loop_trace_v2.schema.json
schemas/mask_manifest_v2.schema.json
schemas/wan_repair_trial_v2.schema.json
tests/wanphysics_v2/test_preflight.py
tests/wanphysics_v2/test_critic_codec.py
tests/wanphysics_v2/test_mask_manifest.py
tests/wanphysics_v2/test_policy_guard.py
tests/wanphysics_v2/test_decision_only_executors.py
tests/wanphysics_v2/test_memory_adapter.py
tests/wanphysics_v2/test_guardrails.py
tests/wanphysics_v2/test_runner.py
tests/wanphysics_v2/test_artifacts.py
tests/wanphysics_v2/test_trials.py
tests/wanphysics_v2/test_resource_coordinator.py
worklog/2026_07_21/PhysGenLoop_全链路问题修复方案.md
~~~

上述全部为新增文件。V2 codec 通过现有 dataclass 构造器恢复对象，不要求在 `src/pavg_critic/schemas.py` 中追加 `from_dict()`。现有 `run_loop.py`、`batch_run_loop.py`、`run_videophy2_loop.py`、`run_trial_campaign.py` 保留，旧行为作为 legacy/fallback。

V2 产物只写入新目录：

~~~text
outputs/v2_run_<timestamp>/
outputs/v2_trials_<timestamp>/
~~~

不得在旧 `videophy2_run_*` 目录中续写或补写文件。

## 5. 分阶段实施顺序

### Phase 0：只读冻结与迁移就绪门禁

先保存 Git SHA、配置 SHA、checkpoint SHA、Critic schema SHA、feature schema SHA、GPU 型号、Python 环境和资产清单。在获得实施授权之前，本阶段只生成报告，不下载模型、不创建 `.env`、不启动 GPU 任务。

获得授权后，迁移就绪门禁必须先确认以下资产均存在且哈希/版本可追溯：

| 资产 | 当前状态 | 正式运行要求 |
|---|---|---|
| Wan2.2-TI2V-5B Diffusers | 已恢复（33GB） | 记录 repo/revision/SHA inventory，未验证前不标 ready |
| Qwen3-VL-8B-Instruct | 已恢复（17GB） | 记录 repo/revision/SHA inventory，验证 vLLM 加载后才标 ready |
| SAM2 checkpoint | 已恢复（309MB） | 记录 SHA-256，与 config/source revision 绑定 |
| SAM2 source | 已恢复（64MB） | 固定 Git revision，后续验证 `_C` 扩展状态 |
| ProPainter | 缺失 | 固定 revision/权重，通过 preflight 才开放 local action |
| `.env` | 已恢复 | 只校验必需键是否存在，不输出值、不进 Git/日志 |
| VideoPhy2 | 已就绪 | 全量 3397 / pilot300 / smoke20 数量和文件校验 |
| Repair checkpoint | 已就绪 | 只允许 proxy research 模式，不得标记 deployable |

### Phase 1：修复 CriticReport round-trip

新增严格 codec，恢复：

- violations；
- critical_frames；
- evidence；
- mask_uri；
- repair_instruction；
- evidence_bundles；
- model_versions；
- diagnostics。

解析失败时保存 raw payload，标记 critic_roundtrip_failed，停止当前 repair，禁止静默清空 violations。

### Phase 2：修复 mask manifest

新增 mask_manifest.json，并把 mask_uri、mask_uris、mask_manifest_uri、mask_mapping_status 写回 Violation.evidence。

必须测试 object name 标准化、critical frame 对齐、mask 文件存在性、mask 尺寸、非零像素和 report round-trip。

### Phase 3：Local Editing 后端 preflight

检查 ProPainter 仓库、推理脚本、cv2、ffmpeg、输出目录和最小 dry-run。后端不存在时将 local_editing 从 capability mask 移除，不能注册一个运行时必然失败的 Executor。

### Phase 4：隔离的 V2 Action-aware Runner

复用 canonical Learning Repair 的 `RepairDecision`、`ExecutorRegistry`、feature encoder 和 policy，但新增独立 V2 Runner 补足原始 prompt、Best-of-K、资源交接、严格复评和完整轨迹语义。不直接替换现有 `LearningRepairLoopRunner`。

~~~text
当前 Candidate
  → Critic
  → Policy 一次
  → Executor 一次
  → 新 Candidate
  → Critic 立即复评
  → 更新当前 candidate
~~~

局部编辑结果必须进入下一轮当前状态，不能只加入 round_winners 后丢失。

### Phase 5：修复 Prompt Executor

新增 DecisionPromptRepairExecutor，只消费 request.decision.instruction，禁止在 Executor 内再次调用 Policy。

### Phase 6：具体化 Prompt Repair

根据 violation category、object、frame interval 和 repair instruction 生成具体自然语言约束，保留原对象、动作、场景和镜头语义。

### Phase 7：修复 Proxy Memory

新增格式识别和兼容适配，区分：

~~~text
RepairExample manifest
Proxy target manifest
Actual RepairTrial manifest
~~~

格式不兼容时只警告一次，并写入 memory_status，不得每个样本重复初始化刷屏。

### Phase 8：Semantic / Quality Gate

先使用：

~~~yaml
acceptance:
  mode: shadow
~~~

只记录 semantic 和 quality。验证 scorer 稳定后切换为 enforce。

### Phase 9：Canonical RepairTrial

每个 action 保存 before、decision、executor、after、分数、semantic、quality、cost、failure reason、mask、Critic backend 和 compatibility manifest。

当前轻量 trials.jsonl 保留，但标记为 lightweight_loop_summary，不能继续当作 RepairTrialV1。

### Phase 10：强制四动作 Trial

新增 `run_actual_trials_v2.py`，对同一个 broken candidate 分别执行 Prompt Repair、Global Regeneration、Local Editing、Reject。未执行动作不能伪造成失败；不可用的 action 记录为 unavailable，不参与该样本的 value target。

### Phase 11：pilot300

只有 20 条 action-aware smoke 通过后，才运行 pilot300，并统计各 action 触发率、成功率、physics gain、semantic/quality 通过率、失败率、mask mapping 失败率、round-trip 失败率、GPU 峰值和 p50/p95 时延。pilot300 仍属 calibration/research，不允许在读取结果后再改阈值并宣称同一批数据为 test。

## 6. V2 独立配置建议

新建 `configs/loop_v2.yaml`，不追加或修改现有 `configs/loop.yaml`：

~~~yaml
runtime:
  pipeline_version: v2
  enabled: false
  gpu_mode: dual_gpu
  generator_gpu: 0
  critic_gpu: 1
  strict_report_roundtrip: true
  persist_full_critic_report: true
  write_repair_trace: true
  resume: true
  fail_on_degraded_critic: true

critic:
  profile: sam2_seeded_rules
  formal_profile_required: false
  allow_rules_fallback: false
  persist_raw_payload_on_error: true

acceptance:
  mode: shadow
  physics_score: 0.80
  confidence: 0.60
  coverage: 0.50
  semantic_score: 0.85
  quality_score: 0.75

local_editing:
  enabled: false
  require_mask: true
  allow_full_frame_fallback: false
  propainter_repo: /root/ProPainter

memory:
  mode: disabled
  strict_schema: true
  allow_disabled: true

policy:
  mode: proxy_research
  allow_non_deployable_checkpoint: false
  require_explicit_proxy_override: true

trial:
  enabled: false
  canonical_format: repair-trial-v1
  min_per_action: 50

vllm:
  ownership_mode: pidfile
  stop_foreign_process: false
  host: 127.0.0.1
  port: 18000
  gpu_memory_utilization: 0.65

artifacts:
  persist_full_critic: true
  persist_raw_payload_on_error: true
  persist_mask_manifest: true
~~~

默认先保持：

~~~text
V2 enabled = false
acceptance.mode = shadow
local_editing.enabled = false
trial.enabled = false
memory.mode = disabled
allow_non_deployable_checkpoint = false
~~~

`confidence=0.60`、`coverage=0.50` 和 `vllm_gpu_memory_utilization=0.65` 只是预注册候选值，需要在独立 calibration 上校准；shadow 阶段不得把它们宣称为已验证阈值。

## 7. 测试门槛

迁移前文档记录的原有基线：

~~~text
405 passed
~~~

该 `405 passed` 尚未在当前迁移后主机上重跑，因此是历史基线，不是当前主机验收证据。实施后必须保持不回退，并新增覆盖：

- CriticReport round-trip；
- mask manifest；
- action capability；
- action override；
- no second Policy call；
- memory migration；
- strict gate；
- vLLM PID ownership；
- resume；
- RepairTrace；
- RepairTrialV1。

Mock 闭环必须强制覆盖：

~~~text
forced prompt_repair
forced global_regeneration
forced local_editing
forced reject
executor failure
critic round-trip failure
~~~

GPU 小型门禁必须用多个可控样本/强制动作分别覆盖首轮 accepted、Prompt Repair、Global Regeneration、Local Editing、Reject、显存交接和 semantic/quality shadow score。不应要求同一个样本同时证明五种终止/动作语义。

## 8. 最终验收标准

以下条件全部满足后，才能称为完整闭环：

~~~text
CriticReport 完整 round-trip
Violation 不丢失
critical_frames 不丢失
mask_uri 可追踪
LocalEditTarget 可构造
ProPainter preflight 通过
四种 Executor 均有真实执行记录
Prompt Executor 不二次调用 Policy
Global Regeneration 真实执行
Local Editing 真实执行
Reject 真实执行
RepairTrace 完整
WanRepairTrialV2 可解析，且 canonical adapter 映射经批准
semantic_score 可计算
quality_score 可计算
vLLM 只管理自己的进程
GPU 运行不发生 OOM
运行可以恢复
405 个旧测试不回退
新增集成测试通过
~~~

在此之前，项目状态继续标记为：

~~~text
research smoke / proxy integration
~~~

而不是：

~~~text
deployment-ready Learning Repair Agent
~~~

## 9. 最终优先级

~~~text
P0-1  CriticReport round-trip
P0-2  mask evidence 传递
P0-3  Local Editing 后端 preflight
P0-4  Action-aware Executor 闭环
P0-5  Prompt Executor 二次 Policy 问题
P1-1  Proxy Memory schema 兼容
P1-2  完整 RepairTrace / RepairTrial
P1-3  Semantic / Quality gate
P1-4  vLLM 进程所有权和资源审计
P2-1  resume / batch / pilot300
P2-2  性能优化和全量评测
~~~

> 核心判断：当前最重要的不是继续扩大样本量，而是先修复证据丢失、动作未执行、Trial 不规范和接受门禁缺失。否则继续运行更多样本，只会产生更多不可审计的“看起来有结果”的产物。

## 10. 文件级实施清单

以下清单按“只新增 V2 sidecar、旧入口保留”的原则设计。

| 文件 | 类型 | 具体工作 | 旧入口影响 |
|---|---|---|---|
| `generators/wanphysics/v2/preflight.py` | 新增 | 资产、环境、GPU、端口、checkpoint 就绪门禁 | 无 |
| `generators/wanphysics/v2/resource_coordinator.py` | 新增 | 双卡亲和性、单卡交接、PID 所有权和显存审计 | 无 |
| `generators/wanphysics/v2/critic_codec.py` | 新增 | 无损恢复 CriticReport/Violation/EvidenceBundle，失败保留 raw payload | 无 |
| `generators/wanphysics/v2/critic_profiles.py` | 新增 | 显式区分 seed-only、B1、M4/full PAVG profile | 无 |
| `generators/wanphysics/v2/critic_backend.py` | 新增 | V2 评分子进程、完整报告持久化和 strict fallback | 无 |
| `generators/wanphysics/v2/mask_manifest.py` | 新增 | mask 序列、SHA、尺寸、帧索引和覆盖率校验 | 无 |
| `generators/wanphysics/v2/policy_guard.py` | 新增 | action 规范化、scope guard、capability mask、override provenance | 无 |
| `generators/wanphysics/v2/prompt_renderer.py` | 新增 | 将 violation/PhysicsPlan 转为具体物理修正约束 | 无 |
| `generators/wanphysics/v2/executors.py` | 新增 | Decision-only Prompt、original-prompt Global、mask-sequence Local、audited Reject | 无 |
| `generators/wanphysics/v2/memory_adapter.py` | 新增 | 识别 proxy target、RepairExample、Actual Trial；默认禁用 | 无 |
| `generators/wanphysics/v2/guardrails.py` | 新增 | physics/confidence/coverage/semantic/quality 的 shadow/enforce gate | 无 |
| `generators/wanphysics/v2/artifacts.py` | 新增 | run/sample/candidate/action 级不可变审计产物 | 无 |
| `generators/wanphysics/v2/runner.py` | 新增 | Best-of-K + Decision → Executor → Re-Critic V2 状态机 | 无 |
| `generators/wanphysics/v2/trials.py` | 新增 | WanRepairTrialV2 和经批准的 canonical adapter | 无 |
| `agents/wanphysics/run_videophy2_loop_v2.py` | 新增 | V2 显式入口 | 无 |
| `agents/wanphysics/run_actual_trials_v2.py` | 新增 | 同源 candidate 强制多动作 Trial | 无 |
| `configs/loop_v2.yaml` | 新增 | V2 唯一运行配置，默认 disabled | 无 |
| `schemas/*_v2.schema.json` | 新增 | trace、mask、Wan Trial 版本化契约 | 无 |
| `tests/wanphysics_v2/` | 新增 | codec、四动作、资源、guardrail、产物和 Trial 测试 | 无 |

以下旧文件不删除、不覆盖：

~~~text
agents/wanphysics/run_loop.py
agents/wanphysics/batch_run_loop.py
agents/wanphysics/run_videophy2_loop.py
agents/wanphysics/run_trial_campaign.py
src/physgenloop/controller.py
src/physgenloop/learning_repair/executors.py
configs/loop.yaml
schemas/critic_output.schema.json
~~~

## 11. Action-Aware Runner 状态机

每个样本状态必须属于：

~~~text
CREATED
PREFLIGHT_FAILED
GENERATING
GENERATED
CRITIC_RUNNING
CRITIC_FAILED
CRITIC_COMPLETED
ACCEPTED
DECISION_READY
EXECUTING
EXECUTOR_FAILED
RE_EVALUATING
MAX_ROUNDS
REJECTED
COMPLETED
~~~

状态只能向前推进，不能覆盖历史状态。

~~~text
CREATED
  → PREFLIGHT_FAILED → COMPLETED
  → GENERATING → GENERATED
  → CRITIC_RUNNING → CRITIC_COMPLETED
  → ACCEPTED → COMPLETED
  → DECISION_READY → EXECUTING
  → RE_EVALUATING → CRITIC_COMPLETED
  → MAX_ROUNDS → COMPLETED
  → REJECTED → COMPLETED
~~~

每个状态必须有产物：

| 状态 | 必须产物 |
|---|---|
| CREATED | sample_status.json |
| GENERATED | video、prompt、metadata、candidate manifest |
| CRITIC_COMPLETED | critic_report.json、critic.json、mask manifest |
| DECISION_READY | repair_decision.json |
| EXECUTING | execution_started.json |
| RE_EVALUATING | 新 Candidate 和新的 CriticReport |
| COMPLETED | loop_result.json、repair_trace.jsonl、trials.jsonl |

## 12. 完整 CriticReport 契约

每个 candidate 目录新增完整报告：

~~~json
{
  "schema_version": "critic-report/2.0",
  "candidate_id": "wan-0242-314df6",
  "video_path": "wan-0242-314df6-v01.mp4",
  "prompt": "A baseball hits a brick wall.",
  "physics_plan": {},
  "report": {
    "decision": "violation",
    "is_physical": false,
    "physics_score": 0.07377,
    "confidence": 0.9,
    "coverage": 0.83,
    "violations": [
      {
        "object": "baseball",
        "category": "penetration",
        "start_frame": 12,
        "peak_frame": 15,
        "end_frame": 18,
        "critical_frames": [12, 13, 14, 15, 16, 17, 18],
        "reason": "The ball appears to pass through the wall.",
        "repair_instruction": "Keep the ball outside the wall and show a visible collision response.",
        "evidence": {
          "mask_uri": "sam2_masks/baseball_00015.png",
          "mask_uris": ["sam2_masks/baseball_00012.png"],
          "mask_manifest_uri": "mask_manifest.json"
        }
      }
    ],
    "diagnostics": {
      "requested_profile": "sam2_seeded_rules",
      "effective_profile": "sam2_seeded_rules",
      "detector_backend": "sam2",
      "vlm_usage": ["object_seed"],
      "sam2_postprocess": "disabled",
      "roundtrip_status": "ok"
    }
  },
  "provenance": {
    "critic_model_id": "pavg-critic-0.3.0",
    "critic_config_sha256": "...",
    "critic_schema_sha256": "...",
    "physics_plan_sha256": "...",
    "source_revision": "..."
  }
}
~~~

报告解析失败时必须保存：

~~~text
critic_payload_raw.json
critic_roundtrip_error.json
~~~

并禁止继续执行 Repair。不能静默把 violations 置为空。

## 13. Mask Manifest 和 Local Editing

新增 mask_manifest.json：

~~~json
{
  "schema_version": "mask-manifest/1.0",
  "candidate_id": "wan-0242-314df6",
  "video": "wan-0242-314df6-v01.mp4",
  "video_width": 1280,
  "video_height": 704,
  "video_frames": 81,
  "source": "sam2",
  "postprocess_enabled": false,
  "objects": [
    {
      "name": "baseball",
      "normalized_name": "baseball",
      "frames": [
        {
          "frame_index": 15,
          "path": "sam2_masks/baseball_00015.png",
          "sha256": "...",
          "nonzero_ratio": 0.032
        }
      ]
    }
  ]
}
~~~

Local Editing 必须拒绝以下输入：

- mask 不存在；
- mask 为空；
- mask 尺寸不匹配；
- critical frame 越界；
- object name 无法匹配；
- mask 覆盖超过整帧 95%；
- mask 非零区域低于 0.01%；
- manifest SHA 校验失败。

LocalEditTarget 应能还原：

~~~json
{
  "objects": ["baseball"],
  "start_frame": 12,
  "end_frame": 18,
  "critical_frames": [12, 13, 14, 15, 16, 17, 18],
  "mask_uri": "mask_manifest.json"
}
~~~

Local Editor 必须读取每帧 mask，不能把第一张 mask 复制到全部帧。没有有效 mask 时必须从 capability mask 中移除 local_editing，不得使用全帧白色 mask 伪装局部修复。

## 14. Repair Decision 和动作规范

每次决策保存：

~~~json
{
  "schema_version": "repair-decision/2.1",
  "action": "prompt_repair",
  "policy_action": "prompt_repair",
  "final_action": "prompt_repair",
  "confidence": 0.76,
  "instruction": "Keep the ball outside the wall and show a visible collision response.",
  "action_probabilities": {
    "prompt_repair": 0.61,
    "global_regeneration": 0.21,
    "local_editing": 0.14,
    "reject": 0.04
  },
  "per_action_values": {
    "prompt_repair": 0.32,
    "global_regeneration": 0.11,
    "local_editing": 0.08,
    "reject": -0.02
  },
  "scope": "global",
  "override_reason": null,
  "compatibility_id": "lrcompat-...",
  "source": "proxy-policy",
  "memory_status": "disabled"
}
~~~

动作名称统一为：

~~~text
prompt_repair
global_regeneration
local_editing
reject
~~~

禁止使用 repairaction.prompt_repair 或 RepairAction.PROMPT_REPAIR 作为最终审计名称。

## 15. Scope 判断算法

~~~text
critical = union(all violation.critical_frames)
ratio = len(critical) / total_frames

if no violations:
    scope = unknown
elif ratio == 0:
    scope = global
elif ratio >= local_threshold:
    scope = global
elif mask manifest invalid:
    scope = global
else:
    scope = local
~~~

动作覆盖规则：

~~~text
scope == local
  and local_editing capability == true
  and mask valid
    → local_editing allowed

scope == local
  and local_editing unavailable
    → local_editing masked
    → fallback prompt_repair / global_regeneration

scope == global
  and Policy selects local_editing
    → override to global_regeneration
    → record override_reason
~~~

## 16. Prompt Repair 具体化

根据 violation category 生成具体约束：

| 错误 | 生成约束 |
|---|---|
| gravity | 物体沿重力方向连续运动，不悬停、不反向漂移 |
| penetration | 两物体不能互相穿透，保持边界分离 |
| collision | 碰撞后出现合理速度或方向变化 |
| trajectory | 运动轨迹连续，不瞬移、不突然改变速度 |
| disappearance | 目标在关键帧范围内持续可见 |
| contact | 接触前后保持合理接触关系 |
| support | 被支撑时稳定，离开支撑后才运动 |
| floating | 物体不能无支撑悬浮 |

Prompt 模板：

~~~text
Preserve the original scene, objects, camera and intended action.
Correct the following physical issue:
{violation_specific_instruction}
The correction must remain visually plausible and temporally continuous.
Do not introduce new objects or change the scene semantics.
~~~

生成前检查：

- 原始 prompt 非空；
- 不包含内部 Policy、checkpoint、memory 等术语；
- 仍包含原始对象；
- 仍包含原始动作；
- 修复约束长度不超过配置上限；
- 保存 instruction source 和 SHA。

## 17. Proxy Memory 迁移

读取第一条记录后识别：

~~~text
if target and outcome:
    canonical_repair_example
elif target_action and action_rewards:
    proxy_target_memory
elif decision and execution and critic_before:
    actual_trial
else:
    incompatible
~~~

Proxy Memory 只能提供 proxy action distribution 和 selected action，不能伪造未执行动作的失败、真实执行 cost、semantic score、quality score 或 Wan2.2 成功率。

同一个 run 中 Memory 只初始化一次，日志只出现一次 schema 状态，不允许每个样本重复刷同一 warning。

## 18. Semantic / Quality Gate

第一阶段使用 shadow：

~~~yaml
acceptance:
  mode: shadow
~~~

记录 semantic 和 quality，但暂不改变 accepted 逻辑。稳定后切换：

~~~yaml
acceptance:
  mode: enforce
  physics_score: 0.80
  semantic_score: 0.85
  quality_score: 0.75
~~~

第一版 quality scorer 使用 CPU 指标：

- 黑帧比例；
- 编码读取成功率；
- 分辨率和 FPS；
- 模糊程度；
- 帧间突变；
- 平均亮度；
- 目标区域消失比例。

Semantic scorer 使用独立、结构化的 Prompt 保持判断，不能把 physics_score 直接当成 semantic_score。

## 19. GPU、vLLM 和恢复

当前服务器实际有两张 A100-PCIE-40GB，因此 V2 优先验证双卡隔离，而不是每个 candidate 反复杀死和重启 vLLM：

~~~text
GPU 0: Wan2.2 generation
       ProPainter（Wan 子进程退出后，经 preflight 确认）

GPU 1: Qwen3-VL vLLM
       SAM2 evaluation
~~~

`vllm_gpu_memory_utilization` 必须通过实测确定；不能因为 GPU 1 有 40GB 就假设 0.85 vLLM 与 SAM2 一定可共存。如发生不可共存，先降低 vLLM utilization 或在 GPU 1 内部串行，再考虑 `single_handoff`。

每次启动 vLLM 写入：

~~~text
vllm.pid
vllm.owner.json
~~~

停止时只停止当前 run 自己的 PID 和子进程，禁止宽泛使用 pkill -9 -f vllm。

V2 默认使用本次 run 专属端口，并在启动前检查端口所有者。已被其他用户/任务占用时必须换端口或停止当前 run，不得清理外部进程。

每个 candidate 记录：

~~~json
{
  "gpu_memory_before_mb": 0,
  "gpu_memory_peak_mb": 32517,
  "gpu_memory_after_mb": 0,
  "vllm_start_seconds": 28.4,
  "wan_generation_seconds": 245.7,
  "critic_seconds": 51.3
}
~~~

每个 sample 新增 sample_status.json，状态包括：

~~~text
created
running
accepted
max_rounds
rejected
critic_failed
executor_failed
completed
~~~

Resume 时扫描 sample_status.json，不依赖最终 summary.json 判断样本是否完成。

## 20. Canonical Trial 和强制四动作

每个正式 Trial 必须包含：

- trial_id；
- group_id；
- domain；
- source candidate；
- critic_before；
- decision；
- execution；
- critic_after；
- before/after scores；
- semantic / quality；
- successful；
- failure_reason；
- compatibility；
- reward spec fingerprint。

现有 canonical `RepairTrialV1` 的 domain 只允许 `blender/hunyuan/fake`，而当前真实 Generator 是 Wan2.2。V2 不得为了通过旧校验而把 Wan Trial 冒充为 Hunyuan Trial。

因此先新增 `WanRepairTrialV2`，显式保存：

~~~text
generator_family = wan
generator_model = Wan2.2-TI2V-5B
generator_revision
critic_profile
critic_revision
policy_mode
research_only
~~~

只有团队同意 domain/schema 映射后，才通过 adapter 转成 canonical training target。

新增 `run_actual_trials_v2.py`，对同一 broken candidate 分别执行：

~~~text
Prompt Repair
Global Regeneration
Local Editing
Reject
~~~

未执行动作保持 null，真实执行失败才写负 utility。

推荐起始数据门禁：

~~~text
每个可用 action 至少 50 条真实 Trial
每个 action 同时包含成功和失败样本
train / calibration / test 按 group_id 分离
~~~

门禁前保持：

~~~text
selection_mode = classification_proxy
deployment_ready = false
~~~

## 21. 推荐验证命令

> 本节是 V2 实施完成后的预期命令契约，当前服务器上尚无这些 V2 入口，不得在本文档评审阶段执行。

### 无 GPU 测试

~~~bash
/root/PhysGenLoop-/envs/main/bin/python -m pytest -q
~~~

要求原有测试不回退，并新增 round-trip、mask、memory、executor、trace 和 trial 测试。

### Mock 四动作

~~~bash
/root/PhysGenLoop-/envs/main/bin/python \
  agents/wanphysics/run_videophy2_loop_v2.py \
  --config configs/loop_v2.yaml \
  --manifest evaluation/manifests/videophy2_pilot300.json \
  --limit 1 \
  --enable \
  --dry-run \
  --force-action prompt_repair
~~~

分别替换 force-action 为：

~~~text
prompt_repair
global_regeneration
local_editing
reject
~~~

### 单样本 GPU smoke

~~~bash
/root/PhysGenLoop-/envs/main/bin/python \
  agents/wanphysics/run_videophy2_loop_v2.py \
  --config configs/loop_v2.yaml \
  --manifest evaluation/manifests/videophy2_pilot300.json \
  --limit 1 \
  --enable \
  --allow-proxy-policy \
  --trace-level full
~~~

### 20 条回归

~~~bash
/root/PhysGenLoop-/envs/main/bin/python \
  agents/wanphysics/run_videophy2_loop_v2.py \
  --config configs/loop_v2.yaml \
  --manifest evaluation/manifests/videophy2_pilot300.json \
  --limit 20 \
  --enable \
  --allow-proxy-policy \
  --resume \
  --trace-level full
~~~

## 22. Rollout 和回滚

### Rollout

1. 当前服务器只读 preflight，固化已知 MISS；
2. V2 CPU/mock codec + mask + state-machine 验证；
3. 获得资产/GPU 授权后恢复模型和后端，重跑 readiness preflight；
4. 资源协调器和单样本 GPU 动作门禁；
5. legacy 仅作历史/同计算量对照；
6. V2 action-aware shadow + strict report/mask/profile provenance；
7. semantic / quality enforce；
8. pilot300 calibration；
9. Actual Trial；
10. 独立 holdout 评价；
11. value-led Policy。

### 回滚条件

出现以下任一条件立即停止 V2 run，保留该 run 已写入的审计产物；旧入口因从未被修改，不需要回写旧配置：

- OOM；
- Critic round-trip 失败率超过 1%；
- mask mapping 失败率超过 5%；
- Executor 失败率超过 10%；
- 运行结果无法重建；
- 原有测试回退。

停止/回退配置：

~~~yaml
runtime:
  enabled: false
~~~

如需使用旧路径，显式调用旧 `run_videophy2_loop.py`；V2 不应修改旧入口的默认行为。

## 23. 最终完成定义

~~~text
[ ] CriticReport round-trip 完整
[ ] Violation / critical_frames 不丢失
[ ] mask_manifest 可验证
[ ] LocalEditTarget 可构造
[ ] ProPainter preflight 通过
[ ] Prompt Executor 不二次调用 Policy
[ ] Prompt Repair 使用具体物理约束
[ ] Global Regeneration 有真实执行记录
[ ] Local Editing 有真实执行记录
[ ] Reject 有真实执行记录
[ ] RepairTrace 完整
[ ] WanRepairTrialV2 可解析
[ ] canonical Trial adapter 经团队批准
[ ] semantic scorer 可运行
[ ] quality scorer 可运行
[ ] vLLM 只管理自己的进程
[ ] GPU smoke 无 OOM
[ ] resume 可用
[ ] 原有测试不回退
[ ] 新增集成测试通过
[ ] pilot300 完成审计
[ ] Actual Trial 达到 action 门禁
[ ] compatibility manifest 重新冻结
[ ] 独立 evaluator / human subset 不与内部 Critic 循环同源
[ ] 当前服务器资产和环境恢复可复现
[ ] deployment_ready 经过真实数据门禁
~~~

在以上条件完成之前，项目只能标记为 research smoke / proxy integration，不应描述为真实四动作物理修复系统或 deployment-ready Learning Repair Agent。

## 24. 2026-07-21 当前服务器就绪度清单

下表是本文档对“能否立即实施/运行”的权威判断。历史 outputs 存在不等于当前主机已恢复运行能力。

| 检查项 | 当前结果 | 影响 |
|---|---|---|
| Git 分支 | `main...origin/main` | 可作为新 V2 方案的代码基准 |
| Git 工作树 | tracked 文件无变化；`worklog/2026_07_21/` 未跟踪 | 两份新审计文档属于协作者未提交内容，不得覆盖/移动 |
| HEAD | `13c6289` | 必须写入 V2 run manifest |
| GPU | `2 × NVIDIA A100-PCIE-40GB` | 可设计 dual-GPU affinity，仍需实测 SAM2+vLLM 共存 |
| GPU 当前占用 | 0 MiB / 0% | 只表示复核时空闲，不是后续任务的资源锁 |
| `envs/main` | 已恢复 | torch/cv2/diffusers/transformers/project package 可 import |
| `envs/vllm-cu128` | 已恢复 | vLLM 0.11.0 可 import |
| VideoPhy2 全量 | 3397 条 | 数据资产就绪 |
| pilot300 / smoke20 | 300 / 20 | 输入集就绪 |
| Repair checkpoint | 9.8MB bundle 已恢复 | 只能用于 proxy research |
| Wan2.2 model | 已恢复（33GB） | 只证明文件到位，未做迁移后加载/生成验证 |
| Qwen3-VL model | 已恢复（17GB） | 只证明文件到位，未做迁移后 vLLM 加载验证 |
| SAM2 checkpoint/source | 已恢复（309MB/64MB） | 未做迁移后 predictor 和 `_C` 扩展验证 |
| ProPainter | 缺失 | Local Editing 必须 capability-mask |
| `.env` | 已恢复 | 未输出内容；后续只能做 key-existence 检查 |
| 当前可否 GPU smoke | 资产层面已具备 Wan+Critic 候选条件；本轮未授权/未执行 | Local action 仍不可；任何 GPU smoke 都必须另行授权 |

当前阶段允许进行的工作只有：

- 完善方案和评审契约；
- 在本地工作区编写未实施的设计文档；
- 服务器只读审计。

未获得明确批准前，不下载上述资产、不创建 V2 文件、不运行测试或 GPU 任务。

## 25. 根因—修复—证据闭环矩阵

| 问题 | 根因 | V2 修复 | 必须证据 |
|---|---|---|---|
| Local action 不触发 | 子进程报告反序列化后 `violations=()` | strict codec + mask manifest | raw/report 往返字段一致；历史 mask 可被重建引用 |
| Local 修复后不立即 accepted | 编辑结果只进入 `round_winners` | V2 state machine 把 edited candidate 设为 current 并 re-gate | state trace 显示 `EXECUTED → RE_EVALUATED → ACCEPTED/CONTINUED` |
| Prompt action 内容空泛 | proxy instruction 是通用占位文本 | evidence-grounded prompt renderer | prompt 含 object/category/frame/constraint，不含内部 policy 术语 |
| Prompt Executor 二次决策 | Executor 调用 `ActionValueRepairer.repair()` | decision-only executor | policy call count = 1；记录 instruction SHA |
| Global action 语义混淆 | 多轮后 `current_prompt` 可能已修改 | 显式保留 immutable original prompt | execution record 包含 input prompt/original prompt/hash |
| Reject 缺少 Executor 证据 | Controller 直接 return | audited reject executor | terminal execution result + historical best selection provenance |
| Memory 无法加载 | proxy target schema 不是 RepairExample schema | 显式 schema detector/adapter | memory type/count/hash/status 只写一次 |
| GPU 进程管理过宽 | `pkill -f vllm` 不区分所有者 | run-owned PID/process-group | 只终止 owner manifest 中的 PID |
| Critic profile 名不副实 | VLM 只参与 SAM2 seed 却记为完整 VLM Critic | named critic profiles | profile/modules/model IDs/fallback 逐轮写入 |
| Trial 不可训练 | 当前 JSONL 是轻量 round summary | WanRepairTrialV2 | before/decision/execution/after/guardrails/cost 完整 |
| 内部分数可被钻空子 | 生产和验收使用同一 Critic | independent evaluation lane | 内部 gain 与独立 evaluator/human 指标分开报告 |

## 26. Critic Profile 必须显式化

V2 不使用模糊的 `sam2+vlm` 作为唯一 backend 名称。建议至少定义以下 profile：

| Profile | 实际模块 | 定位 |
|---|---|---|
| `rules_color_blob` | 默认 HSV detector + rules | 仅用于无 GPU 基线，不得作为通用 fallback Trial |
| `sam2_seeded_rules` | Qwen object seed + SAM2 track + template Planner/graph + rules/checklist/mechanics | 对齐当前生产入口的真实能力 |
| `m4_evidence_vlm` | SAM2 + EvidenceGroundedVLMVerifier + 命名 fusion | 候选模型化 Critic，必须独立 benchmark |
| `full_pavg_model` | model Planner + Hybrid PQSG + SAM2 + rule/checklist/mechanics + VLM verifier | 研究目标 profile，未验证前不作默认 |

每轮 Critic 产物必须包含：

~~~json
{
  "requested_profile": "m4_evidence_vlm",
  "effective_profile": "m4_evidence_vlm",
  "modules": {
    "planner": "template",
    "question_graph": "template",
    "detector": "sam2",
    "vlm_object_seed": true,
    "vlm_verifier": "evidence_grounded",
    "checklist": true,
    "mechanics": true
  },
  "fallback_used": false,
  "degraded_reasons": []
}
~~~

正式 Trial 中 `requested_profile != effective_profile` 或 `fallback_used=true` 时，默认记录失败并停止该 action；只有明确的 exploratory 开关可继续，且该数据不进入正式训练集。

## 27. 接受条件与研究评价分离

### 27.1 运行时闭环接受

V2 内部接受条件分两阶段：

~~~text
shadow:
  legacy physical/score 决定是否停止
  confidence/coverage/semantic/quality 只记录

enforce:
  decision == physical
  physics_score >= calibrated threshold
  confidence >= calibrated threshold
  coverage >= calibrated threshold
  semantic_score >= calibrated threshold
  quality_score >= calibrated threshold
  critic_not_degraded
~~~

shadow 和 enforce 数据必须分开报告。不得读取 shadow 结果后在同一批样本上调阈值，然后把它称为 test。

### 27.2 正式效果验收

正式结论至少同时报告：

- 内部冻结 Critic 的 before/after physics gain；
- 独立 Direct VLM / 外部 benchmark 分数；
- 人工双盲复核子集；
- semantic preservation；
- visual quality；
- 新增错误率；
- 生成次数、GPU 时间、p50/p95 延迟和失败率。

对照组至少包含：

~~~text
S0  single generation
S1  compute-matched Best-of-K
S2  heuristic repair
S3  proxy Policy repair
S4  actual value-led Policy repair
~~~

当前 B1 全量 VideoPhy2 结果是 D0 `0.548897` vs B1 `0.544539`，因此在 Critic 未重新冻结/校准或未加入独立验证通道前，不应将内部 physics gain 作为唯一成功指标。

## 28. Checkpoint、Compatibility 和 Memory 门禁

当前文件状态：

| 项目 | 当前值 |
|---|---|
| policy format | `repair-action-value-policy/2.1` |
| selection mode | `classification_proxy` |
| actual trials | `0` |
| source revision | `unknown` |
| deployment ready | `false` |
| checkpoint Critic config SHA | `f8945b...` |
| 当前 Critic config SHA | `f2b1f6...` |
| checkpoint Critic schema SHA | `142af7...` |
| 当前 Critic schema SHA | `534e60...` |

V2 行为必须是：

1. 默认拒绝加载 non-deployable/mismatched checkpoint。
2. research smoke 需显式传入 `--allow-proxy-policy`。
3. 开启后 run/sample/trial 均写入 `research_only=true`。
4. proxy Memory 默认不混合；适配成功后也要用独立开关启用。
5. 新 Critic profile 冻结后重新生成 compatibility manifest，不篡改旧 checkpoint 内的 manifest。
6. Actual Trial 与 proxy Trial 保存在不同目录/清单，报告不合并成一个成功率。

## 29. 工作包和强制验收门

| 工作包 | 主要交付 | 强制验收门 | 未通过时 |
|---|---|---|---|
| W0a 只读库存 | 资产、环境、revision/hash 清单 | 已存在/缺失项全量记录；无 GPU 任务 | 不得宣称 ready |
| W1 codec | lossless Critic codec + raw failure artifact | 完整报告逐字段 round-trip；不静默丢字段 | 禁止 Repair |
| W2 mask | mask manifest + Local target builder | 帧对齐、文件存在、尺寸/覆盖率/SHA 通过 | mask local action |
| W3 action state machine | policy once + executor once + immediate re-critic | 四动作 mock 及失败分支完整 | 不进 GPU |
| W0b 迁移就绪 | 经授权恢复的模型/后端/.env | readiness preflight 零 MISS；版本/哈希完整 | 停止 W4+ |
| W4 resources | dual/single coordinator + PID ownership | 不杀外部进程；显存回收可审计 | 停止 GPU smoke |
| W5 single-sample GPU smoke | 每动作一个真实样本 | 无 OOM；报告/动作/复评产物完整 | 不进 smoke20 |
| W6 smoke20 | shadow guardrail report | 20/20 terminal；0 round-trip 静默降级；可 resume | 不进 pilot300 |
| W7 actual trials | 同 candidate 多动作 WanRepairTrialV2 | action 均有成功/失败；group-safe split | 不重训 |
| W8 evaluation | S0–S4 对照与独立 evaluator | 预注册指标/阈值；test 不回写 Memory | 不标 deployable |

每个工作包完成后必须先交付报告，再获得下一工作包的运行授权。“代码已存在”不等于“验收已通过”。

## 30. 风险登记表

| 风险 | 等级 | 影响 | 控制方法 |
|---|---|---|---|
| Critic reward hacking | P0 | physics score 上升但真实物理不改善 | independent evaluator + human subset |
| 报告 codec 静默丢字段 | P0 | Policy 输入分布与训练不同 | strict fail + raw payload |
| 双卡亲和性未生效 | P0 | Wan/vLLM 仍抢同一卡 | child env audit + nvidia-smi PID mapping |
| vLLM/SAM2 在 GPU1 共存 OOM | P0 | Critic 中断 | utilization calibration + serialized fallback |
| ProPainter 不能修复物理运动 | P0 | local action 只会移除/涂补物体 | 作为实验 backend；复评失败写负 utility |
| proxy Policy 过度自信 | P0 | 动作分布偏移 | explicit research mode + confidence calibration |
| semantic/quality scorer 同源偏置 | P1 | guardrail 形同虚设 | 规则指标 + 独立模型 + human audit |
| 二次 Policy 调用 | P0 | Trial 记录动作与真实执行不一致 | decision-only executor + call-count test |
| Trial domain 冒充 Hunyuan | P1 | 数据来源和训练结论错误 | WanRepairTrialV2 + approved adapter |
| 批量运行中断 | P1 | 重复生成和成本浪费 | sample state + atomic artifact + resume |
| 调参泄漏 | P0 | 指标无法解释 | calibration/test group freeze + immutable manifest |

## 31. 实施授权边界

本文档将后续工作拆分为三类授权，不互相包含：

### A. 文档授权

- 只完善本地 Markdown；
- 服务器只读；
- 不上传、不创建、不运行。

### B. 代码实施授权

- 只新增 V2 文件；
- 不修改/删除旧入口和公共契约；
- 只运行 CPU/mock 测试；
- 不自动包含模型下载和 GPU 运行。

### C. 资产/GPU 运行授权

- 恢复官方模型和 ProPainter；
- 创建服务器本地 `.env`；
- 启动 vLLM/Wan/SAM2/ProPainter；
- 写入新 `outputs/v2_*` 产物。

获得 B 不意味自动获得 C。大规模 smoke20、pilot300、Actual Trial 和 retraining 还应按 W5–W8 分别获得授权。

## 32. 推荐的第一步

当用户后续批准实施时，推荐不直接启动 GPU，而是先执行：

~~~text
W0a 只读 preflight 报告
  +
W1 Critic codec
  +
W2 mask manifest/target builder
  +
W3 CPU/mock 四动作状态机
~~~

这一步不要求官方大模型，可以先证明“数据不丢失、动作不串线、Trial 可审计”。只有 W1–W3 通过后，恢复约 50GB 官方模型并进入 GPU smoke 才具有明确工程价值。
