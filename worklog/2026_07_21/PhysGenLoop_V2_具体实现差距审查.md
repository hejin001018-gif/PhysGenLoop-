# PhysGenLoop V2 具体实现差距审查与全链路补全清单

> 文档性质：基于当前服务器只读审查、V2 单样本运行审计、现有全链路修复方案和 2026-07-21 新增代码的实现差距报告  
> 审查基线：服务器 main@13c6289；只读快照时间 2026-07-21 14:42:35 +08:00  
> 目标：回答“哪些代码只是出现了、哪些语义真正闭环、还需要补什么才能称为完整全链路实现”  
> 边界：本文件是独立文档，不替换、不覆盖既有修复方案、思路总结或运行问题审计

---

## 0. 结论先行

当前 V2 已经从“设计文档”推进到“有 Sidecar 代码、有单样本工程 smoke、有部分 artifact 和 scorer”的阶段，但尚未形成可以被严格验收的完整闭环。

最关键的判断不是“是否已经有 runner.py、trials.py、scorers_semantic.py”，而是以下因果链是否真实成立：

~~~text
同一个候选视频
  → 完整 PhysicsPlan
  → Critic 得到可回放证据
  → 严格 Gate 判定
  → Policy/force-action 选择一个可用动作
  → 对当前候选真实执行该动作
  → 得到明确的 edited/regenerated candidate
  → 对该 after candidate 重新 Critic
  → 对 after candidate 重新 Semantic/Quality Gate
  → 根据 before/after 的真实差值决定继续、接受或拒绝
  → 写出可追溯 RepairTrace
  → 写出可训练且不伪造反事实的 Actual Trial
~~~

截至审查快照，这条链仍在 Runner 因果关系、PhysicsPlan、Trial 配对、Local Editing、独立 Semantic scorer、进程所有权、checkpoint 硬门禁和降级传播等位置断裂。

因此，当前最准确的项目状态是：

> V2 engineering smoke + partial sidecar implementation；不是完整四动作修复闭环，不是可信 Actual Trial 采集系统，也不是 deployment-ready Learning Repair Agent。

---

## 1. 审查输入与不变约束

### 1.1 本报告依据

本报告整合以下材料：

1. 本地 PhysGenLoop_全链路问题修复方案.md；
2. 本地 PhysGenLoop_v2_run_20260721_033020_问题审计.md；
3. 本地 PhysGenLoop_全链路思路总结.md；
4. 服务器 /root/PhysGenLoop-/README.md 与 worklog 中的架构约定；
5. 服务器当前 main@13c6289 的 V2 文件只读审查；
6. 真实单样本输出 /root/PhysGenLoop-/outputs/v2_run_20260721_033020；
7. 2026-07-21 14:35 左右出现的新增 V2 文件和测试；
8. 用户确认的 ProPainter 仓库布局：models/ProPainter，与 models/sam2-src 同层。

### 1.2 代码协作约束

后续补全必须遵守：

- 保留已有 legacy 入口和产物；
- 新功能优先放入 generators/wanphysics/v2、agents/wanphysics 的 V2 入口和新 schema；
- 不删除其他成员代码；
- 不用破坏性重构替换现有 runner；
- 通过配置开关逐步开启新能力；
- 默认关闭尚未验收的 enforce、Actual Trial 和 Local Editing；
- proxy、mock、internal Critic score 不得写成真实修复成功率；
- collection 成功不得写成 tests passed；
- 文件存在不得写成后端可运行；
- shadow accepted 不得写成正式物理通过。

### 1.3 状态定义

| 状态 | 本报告中的含义 |
|---|---|
| 已完成 | 有真实代码路径、正确语义、自动测试和可审计产物，且相应验收已执行 |
| 部分完成 | 已有类、函数、回调或产物，但缺少关键语义、接线、门禁或运行验证 |
| 未完成 | 关键实现不存在，或现有实现不能满足修复方案 |
| 未验证 | 静态上可能存在，但没有足够执行证据证明有效 |
| 不可信 | 已产生数据，但 before/after 或标签语义不能证明来自真实对应执行，不能进入训练集 |
| 阻断 | 不修复就不能宣称完整全链路，通常对应 P0 |

---

## 2. 当前服务器与运行快照

### 2.1 代码与资源状态

截至 2026-07-21 14:42:35 +08:00：

| 项目 | 当前状态 | 解释 |
|---|---|---|
| Git | main@13c6289，工作树 dirty | 新增内容尚未形成可复现 commit |
| Untracked | 37 个 | 需在验收前冻结清单和 SHA |
| GPU | 2 × A100，0 MiB / 0% | 当时无运行任务，不代表代码已通过 GPU 验收 |
| 最新真实 V2 output | v2_run_20260721_033020 | 后续没有新的真实 V2 结果可覆盖该审计 |
| pytest cache | 460 total nodes，55 V2 nodes | 只能证明缓存/collection 信息 |
| 新 gap tests | 8 个 | 尚不能证明核心语义全部 passed |
| Repair checkpoint | deployment_ready=false | 当前只允许 proxy research |
| Actual Trial | actual_trial_count=0 | 尚无真实动作价值数据 |

### 2.2 最新新增文件代表的真实进展

已观察到以下文件：

~~~text
generators/wanphysics/v2/artifacts.py
generators/wanphysics/v2/build_backends.py
generators/wanphysics/v2/runner.py
generators/wanphysics/v2/scorers_semantic.py
agents/wanphysics/run_videophy2_loop_v2.py
agents/wanphysics/run_actual_trials_v2.py
configs/loop_v2.yaml
tests/wanphysics_v2/test_new_gaps.py
~~~

这些文件说明以下能力已有初版：

- Quality scorer 已有接入 Gate 的代码入口；
- Semantic scorer 已有 class；
- RepairTrace、trial/resource/memory/owner artifact 已有初版；
- force-action 参数和入口已出现；
- Actual Trial campaign 已有初版；
- 新增了 8 个 gap tests。

但“文件存在”不等于“全链路语义完成”。本报告后续逐项说明其不足。

### 2.3 单样本 Run 的有效证据

真实样本运行已经证明：

- Wan candidate 可以生成；
- vLLM 服务可以启动并完成请求；
- Critic report 可以写出；
- 状态机可以到达终态；
- 没有观察到 GPU OOM、vLLM 崩溃或请求失败。

同一运行也暴露：

- 非空 Prompt 得到空 PhysicsPlan；
- physics_score=1.0 不能代表关键物理事件已被检查；
- confidence 约 0.3025、coverage 约 0.43；
- semantic_score 和 quality_score 均为空；
- shadow gate 仍接受；
- Local Editing capability 为 false；
- mask manifest 为空；
- 没有真实 Repair、Executor、Re-Critic 和 Trial 证据；
- critic.json 与 critic_report.json 存在状态表达不一致。

所以该输出只应被标记为 engineering smoke。

---

## 3. Phase 0–11 实现差距总矩阵

| Phase | 修复方案目标 | 当前状态 | 具体差距 | 关闭条件 |
|---|---|---|---|---|
| Phase 0 | 冻结 revision、资产、环境、hash 与 readiness | 部分完成 | preflight 覆盖不全；all_ok=false 未阻断；工作树未冻结 | 完整 manifest、所有 required check 可机器判定、失败能阻断 |
| Phase 1 | CriticReport 无损 round-trip | 部分完成 | 单样本可写 report，但缺严格字段等价测试和降级传播验证 | raw/report 往返后 violation、evidence、backend、degraded 全等 |
| Phase 2 | mask manifest 与 Violation 证据接通 | 部分完成 | manifest 可出现但为空；未证明 Violation enrich；逐帧映射未验收 | 每帧索引、对象、SHA、尺寸、覆盖率可校验并被 Executor 消费 |
| Phase 3 | ProPainter preflight 与 capability mask | 正在补全 | 原配置路径错误；后端此前缺失；尚无 dry-run 与真实编辑证据 | 固定 models/ProPainter，依赖/权重/脚本/dry-run 全通过 |
| Phase 4 | Best-of-K Action-aware Runner | 部分完成但语义错误 | 每轮重生成；edited candidate 未 Re-Gate、未成为 current；Best-of-K 缺失 | 同一 current candidate 的完整 decision→execute→re-evaluate 链成立 |
| Phase 5 | Prompt Executor 真正执行二次 Policy | 未验证 | 不能证明 repaired prompt 作为下一次生成输入并被完整记录 | prompt diff、生成输入、after candidate 和 provenance 全可追踪 |
| Phase 6 | 物理约束具体化 Prompt Repair | 部分/未验证 | 需验证是否基于 violation 和 plan 生成具体约束，而非通用模板 | 指令引用对象、事件、时间窗、约束且不泄露内部术语 |
| Phase 7 | Memory schema 适配并接 Policy | 部分完成 | status artifact 存在，但 memory 未证明参与决策；proxy/actual 边界不完整 | 加载一次、类型明确、只提供合法信息、decision 有 memory provenance |
| Phase 8 | 独立 Semantic / Quality Gate | 部分完成 | semantic 不读视频；quality 丢 metrics/reason；shadow/enforce 语义未闭合 | 两 scorer 均对真实 after candidate 评分并进入 Re-Gate |
| Phase 9 | Canonical Actual Trial | 部分完成但当前不可信 | before/after 可能不是执行对应对；概率被重建；success 定义过弱 | trial 与同一 execution_id 强绑定且 schema/语义校验通过 |
| Phase 10 | 同源候选四动作强制 Trial | 部分完成但未成立 | 每动作重新跑 Runner；首轮 accepted 可绕过 force；local target 不完整 | 同一物化 broken candidate 上逐动作执行，不可用动作记 unavailable |
| Phase 11 | smoke20、pilot300 与正式评估 | 未完成 | 无新的可信 V2 smoke20/pilot300/Actual Trial | 前置门全部通过后按冻结配置运行并独立审计 |

---

## 4. P0 阻断差距

## P0-01 Runner 的修复因果链仍然错误

### 现有定位

文件：

- generators/wanphysics/v2/runner.py
- agents/wanphysics/run_videophy2_loop_v2.py

关键符号：

- Runner 主循环；
- generator.generate；
- executor 执行分支；
- RoundRecord；
- current candidate 更新；
- gate 与 re-critic 调用。

### 当前行为

当前实现仍可能在每一轮调用 generator.generate，形成：

~~~text
round 0: generate candidate A → critic A → decision → execute → edited A'
round 1: generate candidate B → critic B
~~~

这使 A' 没有成为 round 1 的 current candidate。即使 A' 被 Critic 一次并加入 history，也没有证明：

- A' 经过与 candidate A 相同的严格 Gate；
- A' 成为下一轮决策输入；
- A 与 A' 的 before/after 被写在同一个 RoundRecord；
- 下一轮 Policy 使用的是 A' 的 report；
- acceptance 针对的是 A'；
- trial 中的 after 真的是该 execution 的输出。

### 为什么这是 P0

如果 current candidate 的身份不连续，就无法回答“该动作是否修复了该视频”。后续 gain、success、action value 和 learning target 都失去因果意义。

### 必须实现的状态机

~~~text
materialize K initial candidates
  → critic + gate each candidate
  → selector chooses current candidate C0
  → if strict accepted: terminal accepted
  → otherwise decide action A0 on C0
  → execute A0 exactly once on C0
  → materialize output C1 with parent_candidate_id=C0
  → critic C1
  → semantic C1
  → quality C1
  → gate C1
  → write one complete RoundRecord(C0, A0, execution, C1)
  → if accepted: terminal accepted
  → else current=C1 and continue
~~~

Reject 不产生伪造的 after video；其 terminal reason 必须是 explicit reject。

### 建议新增或补全字段

Candidate：

- candidate_id；
- parent_candidate_id；
- source_action；
- source_execution_id；
- video_path；
- prompt；
- prompt_hash；
- physics_plan_id；
- artifact_hashes。

RoundRecord：

- round_index；
- before_candidate_id；
- decision_id；
- execution_id；
- after_candidate_id；
- before_report_id；
- after_report_id；
- before_gate；
- after_gate；
- score_delta；
- terminal_reason；
- timestamps。

### 必须补的测试

1. generator 只在初始候选或 Global/Prompt regeneration 的 Executor 内被调用；
2. Local Editing 后下一轮 current 等于 editor output；
3. after_gate 使用 after report，而不是下一轮无关 report；
4. after 被 accepted 时终止；
5. after 未通过时进入下一轮；
6. Reject 不生成 after；
7. max_rounds 不伪造 accepted；
8. candidate parent 链无环且 ID 唯一；
9. Best-of-K 选择结果可复现；
10. 同一 execution_id 在 trace、round 和 trial 中一致。

### 完成定义

只有在一个强制修复样本中，产物能明确还原 C0 → action → C1 → Re-Critic → Re-Gate，才可关闭此项。

---

## P0-02 非空 Prompt 仍可能得到空 PhysicsPlan 并被接受

### 现有定位

- V2 实际入口调用 Runner 时仍可传 physics_plan=None；
- Planner source 在真实输出中表现为 template/empty；
- eval_step 构造 Critic 时注入 detector，但未证明注入 Model Planner。

### 当前风险

空 plan 会让 interaction、appearance、contact、trajectory 等关键事件没有 expected event，规则检查因此落入 unknown 或 not_applicable。随后 physics_score=1.0 可能只是“没有可判定违规”，不是“物理正确”。

### 必须增加 Planner completeness gate

对非空 Prompt：

~~~text
if plan is None:
    unavailable(planner_missing)
elif plan.events is empty:
    unavailable(empty_physics_plan)
elif required entities are missing:
    degraded(incomplete_entities)
elif interaction verbs exist in prompt but no interaction event:
    unavailable(missing_interaction_event)
elif appearance/state-change verbs exist but no appearance event:
    unavailable(missing_appearance_event)
else:
    plan_ready
~~~

### Planner 需要覆盖的最小语义

- entities：对象、主体、环境；
- temporal events：发生时间或相对顺序；
- motion expectations：方向、速度趋势、轨迹；
- interaction events：接触、碰撞、支撑、抓取、释放；
- appearance/state events：破裂、融化、变形、液体状态变化；
- invariants：身份、数量、几何连续性；
- abstain reason：无法可靠解析时显式放弃。

### Gate 规则

- shadow 模式可以继续记录，但 summary 必须写 unavailable；
- enforce 模式下，非空 Prompt + 空 plan 必须拒绝进入正式 acceptance；
- 不得用 physics_score=1.0 覆盖 plan unavailable；
- Planner 的 model/revision/prompt/schema/hash 必须写入 provenance。

### 必须补的测试

- 非空运动 Prompt + 空 plan；
- 碰撞 Prompt 缺 interaction event；
- 外观变化 Prompt 缺 appearance event；
- 无物理事件的静态 Prompt；
- Planner abstain；
- schema invalid；
- 模型超时；
- template fallback 的 degraded 传播；
- complete plan 可正常进入 Critic。

### 完成定义

真实样本的 summary 同时报告 plan completeness、event coverage、abstain/degraded reason；正式 Gate 不再接受空 plan。

---

## P0-03 当前 Trial 的 before/after 和标签不可信

### 现有定位

- agents/wanphysics/run_actual_trials_v2.py
- V2 trial 组装函数 _assemble_trials
- generators/wanphysics/v2/artifacts.py
- generators/wanphysics/v2/trials.py

### 已发现的问题

1. after gate 可能取自下一 Round，而不是 execution output；
2. 如果没有下一轮，after 可能直接等于 before；
3. video_path 可能为空；
4. probabilities/value 可能被重建成均匀分布；
5. success 只依据 execution status；
6. 未绑定 semantic、quality 和 formal gate；
7. 未证明 before 与 after 共用同一 plan 和评估配置；
8. 不可用动作与真实执行失败可能混淆；
9. Reject 可能被错误当成有 after 的 Trial。

### 为什么不能进入训练集

Action-Value 学习需要观察：

~~~text
Q(s,a) ≈ outcome(after produced by executing a on state s)
~~~

如果 after 与 execution 没有唯一绑定，标签就不是 Q(s,a) 的观测，而是无关样本或自复制样本。将其用于训练会系统性污染 Policy。

### Trial 必须从 execution-first 生成

不要在运行结束后根据相邻 Round 猜 before/after。应在一次执行完成后，直接以 execution_id 为主键构造：

- before candidate；
- decision；
- actual execution request；
- execution output；
- after candidate；
- before/after Critic；
- before/after semantic；
- before/after quality；
- before/after Gate；
- cost；
- failure；
- artifact hash。

### success 的正式定义

建议区分：

- execution_succeeded：后端返回有效产物；
- physics_improved：after physics utility 高于 before；
- semantic_preserved：semantic 通过；
- quality_preserved：quality 通过；
- accepted_after：after formal Gate 通过；
- action_success：上述条件按冻结协议组合；
- unavailable：动作没有能力，不是失败；
- rejected_by_design：Reject 的策略结果，不伪造视频修复成功。

### Probability 与 value

- 只保存真实 Policy 原始 logits/probabilities；
- force-action 时保存 policy_proposed_action 与 forced_action；
- 如果没有真实概率，字段必须为 null，并标注 unavailable reason；
- 禁止均匀填充；
- value 只能来自 checkpoint 的真实输出或真实 outcome；
- proxy reward 与 actual reward 分字段存储。

### 必须补的测试

- after 必须由相同 execution_id 产生；
- 空 video_path schema fail；
- before=after 且 action 非 Reject 时 fail；
- 均匀伪概率检测；
- unavailable 不计 action failure；
- execution success 但 formal Gate fail；
- Reject 无 after；
- trial hash 可回放；
- schema version migration；
- 同一 candidate/action 重复记录的幂等性。

### 完成定义

随机抽取 Trial 可以从 trial_id 追溯到真实 before video、执行请求、真实 after video和两侧完整评估；任何字段都不依赖“下一条 Round 猜测”。

---

## P0-04 force-action 与 Actual Trial 没有保证执行语义

### 当前问题

- 首轮 shadow accepted 时，forced decider 可能根本不被调用；
- 每个 action 重新运行 Runner，会重新生成候选；
- 四动作未显式复用同一个物化 broken candidate；
- Local forced decision 缺少完整 LocalEditTarget；
- 入口中硬编码 local_editing capability=true 的做法会绕过真实 preflight；
- Actual Trial 中又可能以“ProPainter 缺失”硬编码 unavailable，两个入口语义不一致。

### 正确的 Campaign 模型

~~~text
prepare source sample
  → generate/materialize exactly one broken candidate C
  → freeze C, plan, critic config and hashes
  → for each action:
       clone immutable trial context
       capability check
       if unavailable: record unavailable
       else force decision after initial evaluation
       execute action on C
       evaluate action output
       write one trial
~~~

### force-action 必须优先于 acceptance short-circuit

应有明确模式：

- normal：strict accepted 可直接终止；
- force_trial：完成初评后必须进入指定动作，不受初始 accepted 短路；
- force_action 不是修改 Policy 输出；
- 同时保存 proposed 与 forced；
- Reject 仍要记录 before 和 terminal reason。

### LocalEditTarget 最小字段

- candidate_id；
- source_video_path；
- mask_manifest_uri；
- selected object IDs；
- selected frame indices；
- temporal range；
- violation IDs；
- prompt/plan context；
- output directory；
- expected frame count；
- mask mapping status。

### 完成定义

同一 campaign 的四条 trial 具有相同 before_candidate_id、before video hash、plan hash 和 before report hash；只有 action 与执行产物不同。

---

## P0-05 Local Editing 尚未形成逐帧闭环

### 当前实现风险

- target 可能只指向第一张 PNG；
- 旧 editor 可能把一张 mask 复制到所有 critical frames；
- 存在 full-white fallback；
- mask manifest 未完整 enrich 到 Violation；
- 没有真实 ProPainter output + Re-Critic 证据。

### ProPainter 固定目录

用户确认最终布局：

~~~text
/root/PhysGenLoop-/models/
├── sam2-src/
└── ProPainter/
    ├── inference_propainter.py
    ├── requirements.txt
    ├── scripts/download_models.py
    └── weights/
~~~

V2 配置必须统一为：

~~~yaml
local_editing:
  enabled: false
  propainter_repo: /root/PhysGenLoop-/models/ProPainter
  propainter_script: /root/PhysGenLoop-/models/ProPainter/inference_propainter.py
  propainter_weights: /root/PhysGenLoop-/models/ProPainter/weights
  python: /root/PhysGenLoop-/envs/main/bin/python
~~~

enabled 只能在完整 preflight 和 dry-run 通过后打开；不能因为目录存在自动变 true。

### 安装命令

~~~bash
cd /root/PhysGenLoop-
git clone https://github.com/sczhou/ProPainter.git models/ProPainter

envs/main/bin/pip install -r models/ProPainter/requirements.txt

cd models/ProPainter
/root/PhysGenLoop-/envs/main/bin/python scripts/download_models.py
cd /root/PhysGenLoop-

ls models/ProPainter/inference_propainter.py
ls models/ProPainter/weights/
~~~

### 安装之外还必须验证

- 记录 ProPainter Git commit；
- 记录 requirements 和权重文件 SHA-256；
- 使用 envs/main 的 Python import torch、cv2、mmcv 或仓库实际依赖；
- ffmpeg 与 ffprobe 可执行；
- 权重不是空文件或 HTML 错误页；
- inference_propainter.py --help 能启动；
- 最小短视频 + 逐帧 mask dry-run 成功；
- 输出视频帧数、fps、分辨率与预期一致；
- 输出可被 ffprobe 和 Critic 读取；
- GPU 亲和性与峰值显存写入 resource artifact。

### Mask 消费规范

- 每个视频帧都有明确 frame_index；
- 每帧 mask 与原视频尺寸一致；
- 允许显式 empty mask，但不能偷偷复用第一帧；
- 关键帧之外如何插值必须记录算法和版本；
- 禁止 full-white fallback；
- manifest 失败时 Local Editing 从 capability mask 移除；
- editor 必须读取 manifest 中逐帧路径；
- mask SHA 校验失败必须 fail closed；
- after video 必须回写 execution_id 和 manifest hash。

### 必须补的测试

- 多帧不同 mask 不会被复制成同一 mask；
- frame count mismatch；
- resolution mismatch；
- missing frame；
- corrupt mask；
- manifest SHA mismatch；
- empty object list；
- full-white fallback 被拒绝；
- ProPainter CLI failure；
- output missing；
- output frame count mismatch；
- Local Editing output 进入 Re-Critic/Re-Gate。

### 完成定义

至少一个真实 force-local 样本使用有效多帧 mask 执行 ProPainter，输出视频经 Re-Critic、Semantic、Quality 和 Gate，并生成完整 trace/trial。

---

## P0-06 Semantic scorer 目前不是真正的视频语义判定器

### 现有定位

- generators/wanphysics/v2/scorers_semantic.py
- build_backends 中 semantic scorer 构造；
- Gate 的 semantic_score 输入。

### 当前问题

- video_path 参数被忽略；
- 请求只包含 Prompt 文本；
- 没有视频、抽帧或 keyframes；
- 默认可能复用同一个 Qwen3-VL；
- 只返回 float；
- 没有结构化 reason、evidence 和 degraded 信息。

这最多是在评估 Prompt 文本，而不是“生成视频是否保持 Prompt 语义”。

### 最小正确输入

- original prompt；
- repaired prompt；
- PhysicsPlan 摘要；
- before/after video keyframes 或视频输入；
- entity/event checklist；
- scorer model/revision；
- sampling 参数；
- frame sampling provenance。

### 最小结构化输出

~~~json
{
  "score": 0.0,
  "passed": false,
  "entity_preservation": 0.0,
  "event_preservation": 0.0,
  "temporal_alignment": 0.0,
  "new_object_penalty": 0.0,
  "missing_event_ids": [],
  "evidence_frames": [],
  "reason": "",
  "backend": "",
  "model_revision": "",
  "degraded": false,
  "degraded_reasons": []
}
~~~

### 独立性要求

正式研究验收不能用同一 Critic 给自己发 physics reward 和 semantic acceptance。至少需要：

- 独立 prompt/template；
- 独立结构化 schema；
- 明确模型与 Critic 是否共享；
- 正式评估阶段增加 independent VLM 或 human subset；
- 报告 shared-model 与 independent-model 两种结果。

### 必须补的测试

- 相同 Prompt、不同视频得到不同结果；
- 视频路径缺失 fail；
- keyframe decode 失败；
- VLM timeout；
- invalid JSON；
- shared backend 标记；
- reason/evidence 保留；
- after candidate 路径实际进入请求。

### 完成定义

从 scorer 的原始 payload 可以证明它看到了对应 after candidate 的视觉内容；Gate 保存完整结构化结果而不只是 float。

---

## P0-07 vLLM 进程所有权仍不安全

### 当前问题

- 通过 pgrep -f 获取第一个 PID；
- 可能命中 foreign process；
- 没有从 Popen 保存 PID/PGID/start time；
- 缺少可靠 vllm.pid；
- ResourceCoordinator 未完整接入；
- legacy broad pkill -f vllm 仍存在；
- 资源统计可能读 GPU0，而 vLLM 实际在 GPU1。

### 正确所有权模型

启动时：

- 使用 Popen 返回的 pid；
- 创建独立 process group；
- 记录 pid、pgid、start_time、command hash、port、GPU、run_id；
- 写 vllm.pid 和 vllm.owner.json；
- readiness 连接必须验证目标 port；
- owner 文件使用原子写。

停止时：

- 校验 pid 仍属于同一 process start_time；
- 校验 command/port/run_id；
- 只向该 PGID 发送 TERM；
- 超时后只对该 PGID KILL；
- 禁止 broad pkill；
- foreign server 只可连接，不可停止，并标记 external ownership。

### ResourceCoordinator 接线

- Wan/ProPainter 与 vLLM/SAM2 的 GPU affinity 来自配置；
- 所有 subprocess 显式传 CUDA_VISIBLE_DEVICES；
- 指标读取实际逻辑 GPU/物理 GPU 映射；
- 记录 before/peak/after 显存；
- OOM 与 port conflict 机器可判定；
- cleanup 只处理当前 run 的 owner resources。

### 完成定义

并行启动一个 foreign vLLM 后，V2 能连接或避让，但绝不会停止 foreign PID；当前 run 的 vLLM 可以精确回收且 owner artifact 可核验。

---

## P0-08 Checkpoint compatibility hard gate 未执行

### 当前证据

当前 checkpoint 带有：

~~~text
deployment_ready=false
actual_trial_count=0
source_revision=unknown
compatibility/hash mismatch
~~~

配置又声明 allow_non_deployable_checkpoint=false，但实际行为仍可能只是 warning 后继续加载。

### 必须实现的分层模式

- disabled：不加载 checkpoint；
- proxy_research：可加载非 deployable checkpoint，但必须显式标记；
- actual_trial_research：要求 actual trial schema 与 feature 兼容；
- deployment：所有 compatibility 和 readiness hard gate 通过。

### Hard gate 项

- schema_version；
- feature_schema_hash；
- critic_schema_hash；
- action_space；
- source_revision；
- checkpoint SHA；
- training data type；
- actual_trial_count；
- deployment_ready；
- calibration version；
- supported generator domain。

### 行为要求

- allow_non_deployable_checkpoint=false 时立即阻断加载；
- 允许 proxy_research 时，run/trial/summary 均写 proxy_only；
- hash mismatch 不得只 warning；
- fallback policy 要明确记录；
- checkpoint 不可用时不得伪造 probability/value；
- compatibility manifest 与 checkpoint bundle 一起冻结。

### 完成定义

针对每种 mismatch 都有单测；正式模式无法加载当前 proxy checkpoint；research 模式可加载但所有产物明确标记不可部署。

---

## P0-09 SAM2 降级状态没有贯穿全链路

### 当前证据

运行日志出现：

~~~text
cannot import name '_C'
Skipping post-processing
~~~

但报告仍可能写：

~~~text
critic_degraded=false
degraded_reasons=[]
~~~

allow_rules_fallback=false 也未证明能阻断 fallback。

### 必须传播的字段

从 backend 初始化到最终 summary：

- requested_profile；
- effective_profile；
- detector_backend；
- planner_backend；
- verifier_backend；
- sam2_extension_available；
- sam2_postprocess；
- rules_fallback_used；
- critic_degraded；
- degraded_reasons；
- unavailable_reasons。

### Fail-open 与 fail-closed

- engineering shadow：可以降级继续，但必须显式；
- formal enforce：配置要求完整 SAM2 时，扩展缺失必须 fail closed；
- allow_rules_fallback=false：不得静默使用 rules fallback；
- 降级结果不得与完整 profile 共用同一名称；
- summary、report、trial 三处状态必须一致。

### 完成定义

模拟 SAM2 _C 缺失，shadow 输出 degraded；enforce 输出 preflight/critic unavailable；两者均有自动测试。

---

## 5. P1 重要实现差距

## P1-01 Preflight 覆盖不足，失败不阻断

### 当前仅覆盖或部分覆盖

- ProPainter repo；
- ffmpeg/cv2 的部分检查；
- port；
- 少量 local capability。

### 必须增加

- Wan model path、关键文件、revision/hash；
- Qwen3-VL model revision/hash；
- SAM2 source、checkpoint、_C extension；
- ProPainter repo/script/weights；
- Python executable 与关键 package versions；
- .env 必需项但不输出 secret；
- GPU count/model/free memory；
- CUDA/PyTorch compatibility；
- vLLM version；
- checkpoint compatibility；
- output 可写；
- schema 可加载；
- model sampling config；
- config dead/unknown fields。

### 阻断语义

- preflight.all_ok=false 时，不得进入要求完整能力的 run；
- partial smoke 需要显式 allow_partial_capability=true；
- require_local_editing=true 且 local false 时必须退出；
- 每个 check 标记 required/optional；
- exit code 与机器可读 report 一致。

---

## P1-02 配置字段没有完整消费

观察到的风险：

- local_editing.enabled 可能未被读取；
- capability=false 时仍注册 Local Executor；
- 部分阈值只写在 YAML；
- allow_non_deployable_checkpoint=false 未执行；
- allow_rules_fallback=false 未执行；
- GPU affinity、resume、trial、memory 等字段存在但行为不完整；
- build_backends 仍有旧 /root/ProPainter 默认值。

必须建立 Config Consumption Test：

1. 枚举 configs/loop_v2.yaml 所有叶子字段；
2. 为每个字段记录 consumer symbol；
3. 未消费字段在 preflight 报错；
4. unknown field 默认报错；
5. feature flag 开关必须改变可观测行为；
6. config snapshot 写入 run 目录；
7. 默认值只定义一次，不能在多文件中分叉。

---

## P1-03 Quality scorer 丢失结构化信息

当前已有 quality scorer 接 Gate 是进展，但只保存 float 不足以审计。

至少保留：

- overall score；
- sharpness；
- temporal consistency；
- flicker；
- compression/artifact 指标；
- resolution/fps/frame count；
- threshold；
- passed；
- reason；
- backend/version；
- degraded；
- raw metrics URI。

Gate 使用聚合值，artifact 保存完整结果。不得把 CPU proxy quality 宣传成独立感知质量模型。

---

## P1-04 Memory 只写状态，尚未证明参与 Policy

### 必须区分三类数据

- proxy_target_memory；
- repair_example_memory；
- actual_trial_memory。

### 接 Policy 的要求

- run 初始化时只加载一次；
- schema adapter 明确 source type；
- query 输入与 feature schema 一致；
- decision artifact 保存 memory_used、neighbors 和 influence；
- proxy memory 不能提供伪 actual cost/success；
- actual trial memory 只能使用通过数据门的 trial；
- 加载失败按配置 fail/disable，不得每样本重复 warning；
- 同一 run 内 memory snapshot 固定。

### 测试

- memory disabled；
- proxy research；
- incompatible schema；
- empty memory；
- deterministic neighbor retrieval；
- Policy with/without memory 可观测差异；
- actual trial leakage 检查。

---

## P1-05 RepairTrace 字段不足

当前初版 RepairTrace 仍需补：

- run/sample/round/candidate/decision/execution ID；
- before/after video URI 与 SHA；
- before/after Critic report；
- before/after Gate；
- semantic/quality 完整结果；
- Policy logits/probabilities；
- proposed/guarded/forced/final action；
- capability mask；
- mask manifest；
- executor request/response；
- artifact hashes；
- timestamps、latency、GPU metrics；
- checkpoint/memory/config provenance；
- degradation/unavailable；
- state transition；
- terminal reason。

Trace 应 append-only、JSONL 每条可独立校验，并有 run-level index。

---

## P1-06 状态转换未严格验证

必须将合法 transition 显式编码，例如：

~~~text
CREATED → PREFLIGHTED
PREFLIGHTED → GENERATED | PREFLIGHT_FAILED
GENERATED → CRITIQUED
CRITIQUED → GATED
GATED → ACCEPTED | DECIDED | REJECTED
DECIDED → EXECUTING
EXECUTING → EXECUTED | EXECUTION_FAILED
EXECUTED → RECRITIQUED
RECRITIQUED → REGATED
REGATED → ACCEPTED | DECIDED | REJECTED | MAX_ROUNDS
terminal → COMPLETED
~~~

要求：

- 非法跳转抛错；
- terminal 不可再执行；
- 每次 transition 记录前后状态、时间和理由；
- crash 恢复从最后完整 artifact 继续；
- artifact 不完整时回滚到最近安全状态，而不是猜完成。

---

## P1-07 Resume 当前不能保证回到同一 Run

### 风险

- 每次启动创建新 timestamp 目录；
- resume 扫描不到原 run；
- sample status 与 summary 混用；
- in-progress execution 幂等性不明确。

### 必须实现

- --run-dir 或 --run-id；
- config/hash 不兼容时拒绝 resume；
- 扫描 sample_status.json；
- completed 样本跳过；
- failed/in-progress 根据安全点恢复；
- 不重复生成已物化 candidate；
- 不重复追加相同 execution/trial；
- resume 次数写入 manifest；
- crash injection tests。

---

## P1-08 Run manifest 与 artifact 一致性不足

### 当前风险

- run manifest 可能在 sample loop 内被覆盖；
- critic.json 仍可停留 waiting；
- critic.json 与 critic_report.json 结论相反；
- summary 缺少 gate mode、preflight、capability、trial status；
- artifact write 非原子。

### 必须实现

- run manifest 只在 run 级创建并版本化更新；
- sample manifest 独立；
- artifact 使用临时文件 + atomic rename；
- 每个 artifact 有 schema_version 和 hash；
- final consistency checker；
- waiting artifact 在终态必须被替换或明确 failure；
- summary 从 canonical artifact 汇总，不重新猜测状态。

---

## P1-09 Resource metrics 可能记录错误 GPU

需要：

- 配置物理 GPU ID；
- subprocess 内逻辑 GPU 0 与宿主物理 ID 映射；
- NVML 读取指定物理设备；
- 每个后端记录 device；
- resource artifact 区分 Wan、vLLM、SAM2、ProPainter；
- 多进程峰值采样；
- 指标读取失败写 unavailable，不能默认 GPU0。

---

## P1-10 Schema 约束过弱

当前风险包括 additionalProperties=true、required 字段过少、URI/ID 未建立一致性约束。

建议：

- schema version 常量；
- additionalProperties=false，扩展放 extensions；
- 所有 ID、状态、action 使用 enum/pattern；
- before/after action 条件约束；
- unavailable 与 failure 互斥；
- score 范围；
- non-empty video URI；
- hash pattern；
- timestamp；
- provenance required；
- migration tests；
- invalid fixture tests。

---

## P1-11 新测试主要覆盖“存在”，未覆盖核心语义

当前 8 个 new-gap tests 和 55 个 V2 nodes 是进展，但 collection 不是 pass，callback 测试也不能替代真实因果语义。

测试必须分层：

### Unit

- codec；
- plan completeness；
- gate；
- policy guard；
- mask manifest；
- config consumption；
- checkpoint compatibility；
- owner PID；
- schema。

### CPU integration

- fake generator；
- fake critic；
- 四 fake executors；
- before/after pairing；
- resume；
- trace/trial consistency；
- artifact atomicity。

### GPU integration

- Wan generation；
- vLLM request；
- SAM2；
- ProPainter local editing；
- resource handoff；
- OOM/timeout handling。

### Dataset campaign

- forced four-action same-candidate；
- smoke20；
- pilot300；
- Actual Trial threshold；
- independent audit。

所有层必须保存测试命令、exit code、测试数和失败列表。

---

## 6. P2 完善项

## P2-01 vLLM 性能与复现

- 记录 vLLM、CUDA、PyTorch 版本；
- 记录 FlashInfer 是否可用；
- 记录 generation config 是否覆盖请求 sampling；
- 固定 temperature、top_p、seed；
- 避免每 candidate 重启 vLLM；
- 记录启动时延、请求时延和 p50/p95；
- 模型 revision 与 tokenizer hash 入 manifest。

## P2-02 Summary 防误读字段

summary 至少包含：

- run_mode；
- gate_mode；
- partial_capability_run；
- preflight_all_ok；
- plan_status；
- critic_profile requested/effective；
- critic_degraded；
- capability_mask；
- initial accepted；
- repair_attempted；
- executor_called；
- recritic_completed；
- regate_completed；
- semantic/quality status；
- trial_recorded；
- trial_eligible；
- terminal reason；
- proxy_only；
- deployment_ready。

## P2-03 依赖和资产可复现

- ProPainter revision/hash；
- 权重清单/hash；
- SAM2 revision/hash；
- Wan/Qwen revision/hash；
- pip freeze 或 lock；
- config hash；
- source Git SHA；
- 环境变量只记录 key present，不记录 secret；
- 第三方 license 清单。

## P2-04 错误类型与重试策略

- model timeout；
- decode error；
- OOM；
- port conflict；
- invalid scorer JSON；
- executor output missing；
- schema failure；
- retryable 与 terminal 分开；
- 重试次数与 backoff 入 trace；
- 重试不得产生重复 Trial。

---

## 7. 文件级具体差距矩阵

| 文件/模块 | 已有内容 | 仍需补全 | 关键验收 |
|---|---|---|---|
| configs/loop_v2.yaml | V2 开关与阈值初版 | ProPainter 新路径；字段消费；hard gate；GPU mapping | 配置叶子字段 100% 有 consumer |
| v2/preflight.py | ProPainter/端口等检查初版 | 全资产、环境、hash、_C、权重、checkpoint；阻断语义 | required fail 时进程非零退出 |
| v2/build_backends.py | 后端组装 | 移除旧默认路径；enabled/capability 一致；owner/resource 接线 | local false 不注册/不暴露动作 |
| v2/runner.py | Action-aware Runner 初版 | current candidate 因果链；Best-of-K；Re-Gate；RoundRecord | C0→action→C1 单链可回放 |
| v2/artifacts.py | 多类 artifact 初版 | 完整字段、atomic write、schema/hash/index | consistency checker 全通过 |
| v2/scorers_semantic.py | Semantic class | 真正读取视频/keyframes；结构化结果；独立性 | 不同视频可得到不同证据 |
| v2/quality scorer | float/Gate 接口 | metrics/reason/backend/degraded | Gate + artifact 信息一致 |
| v2/resource_coordinator.py | 资源策略初版 | 全后端接线；正确 GPU；PID ownership | foreign vLLM 不被停止 |
| v2/mask_manifest.py | manifest 初版 | Violation enrich；逐帧映射；SHA；fail closed | 多帧 mask 被逐帧消费 |
| local_editor.py | ProPainter 适配器 | 禁止首帧复制与白 mask；输出校验 | 真实多帧 local edit |
| v2/memory_adapter.py | schema/status 初版 | query→Policy 接线；source boundary | decision 显示 memory influence |
| v2/trials.py | Wan Trial schema 初版 | execution-first 构造；严格 success；canonical policy | trial 可追到真实执行 |
| run_videophy2_loop_v2.py | V2 入口/force 参数 | 不硬编码 capability；force 绕过初始短路；resume | forced action 必被调用 |
| run_actual_trials_v2.py | 四动作 campaign 初版 | 同一 frozen candidate；LocalEditTarget；真实 after | 四 trial before hash 相同 |
| schemas/*_v2 | 版本化 schema | required、enum、条件约束、additionalProperties | invalid fixtures 全拒绝 |
| test_new_gaps.py | 8 个补充测试 | 核心因果、字段、失败模式与 GPU tests | 不只检查存在/callback |
| legacy agents | 旧入口保留 | 标注 legacy；禁止 broad pkill 影响 V2 | V2 不调用破坏性 cleanup |

---

## 8. ProPainter 本次实施与配置检查表

本节用于记录服务器侧实际安装，不以“目录存在”作为完成标准。

### 8.1 目录与配置

- [ ] models/ProPainter 与 models/sam2-src 同层；
- [ ] inference_propainter.py 存在；
- [ ] requirements.txt 存在；
- [ ] weights 目录存在且非空；
- [ ] configs/loop_v2.yaml 指向 /root/PhysGenLoop-/models/ProPainter；
- [ ] build_backends.py 的 fallback 同步到新路径；
- [ ] 其他 V2/legacy 配置引用完成审计；
- [ ] Git revision 写入 preflight/run manifest。

### 8.2 环境验证

- [ ] 使用 /root/PhysGenLoop-/envs/main/bin/pip 安装；
- [ ] pip check 通过或冲突有隔离方案；
- [ ] Python 关键 import 通过；
- [ ] ffmpeg/ffprobe 可用；
- [ ] CUDA/PyTorch 与 ProPainter 兼容；
- [ ] inference_propainter.py --help 可运行；
- [ ] 权重清单和 SHA 写入 inventory。

### 8.3 功能验证

- [ ] 单帧/多帧 mask 输入 schema 正确；
- [ ] 最小 dry-run 生成视频；
- [ ] 输出帧数/fps/分辨率正确；
- [ ] LocalEditTarget 从 Violation + mask manifest 构造；
- [ ] editor 逐帧读取 mask；
- [ ] 不存在 full-white fallback；
- [ ] after candidate 进入 Re-Critic；
- [ ] Semantic/Quality/Re-Gate 完成；
- [ ] trace/trial 记录真实 before/after；
- [ ] GPU 资源释放完整。

### 8.4 开关策略

~~~yaml
# 安装完成但未 dry-run
local_editing:
  enabled: false

# preflight + dry-run + trace/re-gate 测试通过后，专用 force-local smoke
local_editing:
  enabled: true

trial:
  enabled: false
~~~

只有 force-local smoke 通过后，才可在 smoke20 中开放 Local Editing；只有 smoke20 审计通过后，才可进入 Actual Trial campaign。

---

## 9. 完整实现的分阶段关闭顺序

## Gate A：静态一致性

目标：

- 配置路径和 consumer 一致；
- schema 收紧；
- checkpoint hard gate；
- preflight 全覆盖；
- ProPainter 资产 ready；
- 无 broad PID cleanup 进入 V2。

退出条件：

- unit tests 全过；
- config consumption 100%；
- asset inventory/hash 完整。

## Gate B：CPU Mock 因果链

目标：

- Best-of-K；
- current candidate 更新；
- 四动作；
- Re-Critic/Re-Gate；
- trace/trial；
- resume。

退出条件：

- fake C0→action→C1 可完整回放；
- same-candidate forced campaign 通过；
- crash injection 不重复 Trial。

## Gate C：单模块 GPU Smoke

分别验证：

- Wan；
- vLLM；
- SAM2；
- ProPainter；
- Semantic scorer；
- GPU resource ownership。

退出条件：

- 每个模块有成功和失败路径；
- 资源可精确释放；
- 产物可被后续模块读取。

## Gate D：强制动作小型闭环

使用多个可控样本分别覆盖：

- initial accepted；
- Prompt Repair；
- Global Regeneration；
- Local Editing；
- Reject；
- max rounds；
- scorer unavailable；
- executor failure。

退出条件：

- 每个 action 至少有一条真实、可回放、schema-valid Trial；
- local action 有逐帧 ProPainter 证据；
- formal Gate 使用 after candidate。

## Gate E：V2 smoke20

前提：

- Gate A–D 全通过；
- 冻结 source/config/model/checkpoint hashes。

统计：

- action 分布；
- execution success；
- physics gain；
- semantic/quality pass；
- plan completeness；
- mask mapping；
- degraded/unavailable；
- p50/p95；
- GPU peak；
- artifact consistency；
- resume。

退出条件：

- 20 条均有终态和一致性审计；
- 不将 shadow 结果宣传为正式成功率。

## Gate F：pilot300

前提：

- smoke20 独立审计通过；
- 阈值提前冻结；
- calibration 与 test 分离。

退出条件：

- 300 条无 silent failure；
- action、scorer、plan、mask、resource 指标完整；
- 阈值变更只能进入下一批，不得回看后重算同批 test。

## Gate G：Actual Trial 数据门

- 同一 broken candidate 的多动作 Trial；
- 每个可用 action 达到预定样本数；
- unavailable 不伪造失败；
- proxy 与 actual 分离；
- semantic/quality/formal Gate 完整；
- human subset 或 independent VLM 审核。

## Gate H：Policy 更新与正式评估

- 只使用通过数据门的 Actual Trial；
- checkpoint compatibility 重新冻结；
- S0–S4 或预注册基线完成；
- Direct VLM、rules、SAM2-seeded、完整 Critic 分开报告；
- internal Critic 与 independent evaluation 分开；
- deployment_ready 只有全部门通过才可置 true。

---

## 10. 自动测试与验收命令建议

以下是实施后的命令模板，不代表当前已经执行或通过。

### 10.1 V2 CPU 测试

~~~bash
cd /root/PhysGenLoop-
envs/main/bin/python -m pytest tests/wanphysics_v2 -q
~~~

### 10.2 只收集测试

~~~bash
envs/main/bin/python -m pytest tests/wanphysics_v2 --collect-only -q
~~~

注意：collect-only 只证明测试可发现，不证明通过。

### 10.3 ProPainter 资产检查

~~~bash
test -f models/ProPainter/inference_propainter.py
test -f models/ProPainter/requirements.txt
test -d models/ProPainter/weights
find models/ProPainter/weights -maxdepth 2 -type f -size +0c
envs/main/bin/python models/ProPainter/inference_propainter.py --help
envs/main/bin/pip check
ffmpeg -version
ffprobe -version
~~~

### 10.4 强制 Local Editing

命令参数应以 V2 入口实际 CLI 为准。验收重点不是命令退出 0，而是：

- force action 确实进入 local executor；
- 输入是同一物化 candidate；
- mask 是逐帧 manifest；
- output 非空且可解码；
- Re-Critic、Semantic、Quality、Re-Gate 均完成；
- trace/trial ID 串联一致。

### 10.5 Artifact consistency

应新增统一校验入口，检查：

- schema；
- hash；
- ID reference；
- state transition；
- before/after；
- summary 与 canonical artifact 一致；
- trial eligibility；
- run completeness。

---

## 11. 最终验收清单

### 11.1 架构

- [ ] V2 Sidecar 不破坏 legacy；
- [ ] Best-of-K 已实现；
- [ ] current candidate 因果链正确；
- [ ] 四动作互斥且 capability-aware；
- [ ] Executor 后必有 Re-Critic/Re-Gate；
- [ ] Reject 语义明确；
- [ ] 状态 transition 有验证；
- [ ] resume 回到同一 run。

### 11.2 Planner 与 Critic

- [ ] 非空 Prompt 不再静默接受空 plan；
- [ ] interaction event；
- [ ] appearance/state event；
- [ ] plan completeness；
- [ ] Planner abstain；
- [ ] CriticReport round-trip 无损；
- [ ] SAM2 degraded 正确传播；
- [ ] rules fallback 受开关控制；
- [ ] requested/effective profile 分离。

### 11.3 Local Editing

- [ ] ProPainter 在 models/ProPainter；
- [ ] 依赖与权重 ready；
- [ ] config 指向新路径；
- [ ] preflight + dry-run；
- [ ] mask manifest enrich Violation；
- [ ] 逐帧 mask；
- [ ] 禁止 white-mask fallback；
- [ ] force-local 真实执行；
- [ ] after 进入完整评估。

### 11.4 Gate

- [ ] physics；
- [ ] confidence；
- [ ] coverage；
- [ ] semantic；
- [ ] quality；
- [ ] plan completeness；
- [ ] degraded/unavailable；
- [ ] shadow/enforce 分离；
- [ ] formal acceptance fail closed；
- [ ] 独立验证通道。

### 11.5 Policy、Memory 与 Checkpoint

- [ ] capability mask 来自真实 preflight；
- [ ] proposed/guarded/forced/final action 均保存；
- [ ] memory 真正接入 Policy；
- [ ] proxy/actual memory 分离；
- [ ] checkpoint hard gate；
- [ ] probability/value 不伪造；
- [ ] deployment_ready 语义真实。

### 11.6 Trace 与 Trial

- [ ] RepairTrace 字段完整；
- [ ] append-only + schema；
- [ ] artifact hashes；
- [ ] Trial execution-first；
- [ ] before/after 真实配对；
- [ ] success 是复合定义；
- [ ] unavailable 不算 failure；
- [ ] 同源四动作；
- [ ] Wan domain 不冒充 Hunyuan；
- [ ] canonical adapter 经团队批准。

### 11.7 资源与复现

- [ ] vLLM PID/PGID owner；
- [ ] 禁止 broad pkill；
- [ ] foreign process 安全；
- [ ] GPU ID 指标正确；
- [ ] model/source/config/hash 冻结；
- [ ] package versions；
- [ ] sampling 参数；
- [ ] resource p50/p95/peak；
- [ ] cleanup 可验证。

### 11.8 运行证据

- [ ] V2 unit tests passed；
- [ ] CPU mock integration passed；
- [ ] ProPainter dry-run passed；
- [ ] 单样本 GPU smoke passed；
- [ ] forced four-action passed；
- [ ] smoke20 passed + audited；
- [ ] pilot300 passed + audited；
- [ ] Actual Trial 数据门通过；
- [ ] S0–S4/基线评估；
- [ ] independent VLM/human subset；
- [ ] compatibility manifest 重新冻结。

---

## 12. 当前可关闭与不可关闭的表述

### 当前可以表述

- 已建立 V2 Sidecar 的主要文件骨架；
- 已完成 Wan2.2、vLLM 和 SAM2-seeded Critic 的单样本 engineering smoke；
- 已加入部分 Quality Gate、Semantic scorer、artifact、force-action 和 Trial campaign 初版；
- 已识别并开始补齐 ProPainter Local Editing 后端；
- 已明确完整全链路的验收门。

### 当前不能表述

- 完整四动作闭环已实现；
- Local Editing 已验证；
- PhysicsPlan 已模型化完成；
- Semantic scorer 已看视频；
- Actual Trial 数据可信；
- repair success rate 已获得；
- checkpoint deployment-ready；
- pilot300 已完成；
- Policy 已由真实 trial 学习；
- 全链路部署就绪。

---

## 13. 推荐的代码补全优先级

### P0 第一批：先修因果与数据真实性

1. 修 Runner current candidate + Re-Gate；
2. 增加 Planner completeness hard gate；
3. Trial 改为 execution-first；
4. force-action 改为 same-candidate campaign；
5. ProPainter + 逐帧 mask 闭环；
6. Semantic scorer 真正读取视频；
7. vLLM owner；
8. checkpoint hard gate；
9. SAM2 degradation 传播。

### P1 第二批：补可复现和运行韧性

1. 完整 preflight；
2. config consumption；
3. Quality structured artifact；
4. Memory→Policy；
5. RepairTrace；
6. state transition；
7. resume；
8. artifact consistency；
9. resource metrics；
10. schema；
11. 语义级 tests。

### P2 第三批：性能与研究评价

1. vLLM 常驻/复用；
2. sampling 和版本冻结；
3. smoke20；
4. pilot300；
5. Actual Trial；
6. independent VLM/human subset；
7. Policy 更新；
8. S0–S4 正式比较。

---

## 14. 最终完成定义

“PhysGenLoop V2 全链路完整实现”必须同时满足：

1. 非空 Prompt 产生完整或明确 abstain 的 PhysicsPlan；
2. 初始候选经可审计 Critic 和严格 Gate；
3. Policy 只在真实 capability mask 内选择；
4. 指定 Executor 对当前候选真实执行；
5. after candidate 身份连续、可追溯；
6. after 经 Re-Critic、Semantic、Quality 和 Re-Gate；
7. 每轮 before/action/execution/after 形成单一因果记录；
8. Trial 不通过相邻 Round 猜测，不伪造概率、失败或 reward；
9. Local Editing 使用 models/ProPainter 和逐帧 mask；
10. checkpoint、memory、degraded、resource ownership 均 fail-safe；
11. tests、smoke20、pilot300、Actual Trial 和独立评估按门禁完成；
12. 所有对外指标明确区分 proxy、internal、shadow、actual 与 independent。

在以上条件全部关闭前，项目应继续标记为 research/engineering 阶段。最优先的工作不是扩大样本量，而是修正 Runner 因果链和 Trial 数据真实性；否则新运行只会积累更多无法用于训练或学术结论的产物。

