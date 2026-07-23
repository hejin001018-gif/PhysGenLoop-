# PhysGenLoop 最终全链路总结介绍

> 版本：V2 Strict Enforce，2026-07-22  
> 项目位置：`/root/PhysGenLoop-`  
> 文档用途：项目总览、组会汇报、学术海报与 PPT 内容母稿  
> 实现基线：服务器 revision `569ffb7e8ec61e6af1355737e6a1acc346101830` 及其上尚未提交的 V2 最终对齐实现
署名：hj
---

## 1. 一句话介绍

PhysGenLoop 是一个面向文本生成视频的“生成—物理评价—定向修复—重新评价”闭环系统：它先由 Wan 根据文本生成视频，再用 SAM2、视觉语言模型和物理规则形成可审计的 Critic 证据，通过严格 Acceptance Gate 判断候选是否可接受；若不可接受，则由三动作 Repair Policy 在 Prompt Repair、ProPainter Local Editing 和 Reject 之间做一次真实决策，修复后的候选必须重新经过相同 Critic 与 Gate，最终输出可接受候选或一个有明确原因、可复现、可审计的失败终态。

---

## 2. 项目要解决的问题

### 2.1 文本生成视频的核心缺陷

当前视频生成模型可以产生视觉上逼真的短视频，但“看起来真实”不等于“物理上正确”。典型问题包括：

- 重力方向、加速度或落体过程不合理；
- 接触、碰撞、反弹、摩擦关系不一致；
- 物体轨迹突然跳变或运动不连续；
- 物体身份、形状或外观在时间上漂移；
- 局部遮挡、消失、穿透或重叠关系异常；
- 视频虽然流畅，但没有完成文本要求的事件；
- 修复后物理指标提高，却牺牲了语义一致性或画面质量。

仅依赖一次生成无法稳定解决这些问题；仅给出一个评分也不能自动产生更好的结果。因此，本项目把视频生成改造成一个受评价约束的迭代系统。

### 2.2 PhysGenLoop 的研究问题

PhysGenLoop 关注四个连续问题：

1. 如何从生成视频中提取足以支撑物理判断的时空证据？
2. 如何区分“视频确实不合格”和“评价链路自己不可用”？
3. 如何根据违背类型选择成本和风险不同的修复方式？
4. 如何证明修复真正改善了视频，而不是只完成了一次编辑操作？

### 2.3 核心回答

项目的回答不是“再生成一次”，而是建立严格闭环：

~~~text
Generate → Observe → Critique → Gate → Decide → Repair → Re-Critique → Re-Gate → Audit
~~~

其中评价前后使用同一套严格标准，任何修复只有在 Re-Gate 明确 `ACCEPTED` 时，才能记为严格成功。

---

## 3. 最终全链路总图

~~~text
用户 Prompt
    │
    ▼
Wan2.2 首轮视频生成
    │
    ▼
视频观测与证据构建
    ├── Qwen3-VL：对象语义与视觉种子
    ├── SAM2：对象分割、跨帧跟踪、逐帧 mask
    ├── 轨迹提取：位置、速度、连续性等时序信息
    └── 事件检测：接触、碰撞、消失、跳变等候选事件
    │
    ▼
Physics Critic
    ├── 规则证据
    ├── checklist / video-science 证据
    ├── mechanics 证据
    ├── VLM 证据
    ├── physics_score / confidence / coverage
    └── violation、关键帧、repair instruction、local target
    │
    ▼
Strict Enforce Acceptance Gate
    ├── ACCEPTED ───────────────────────────────► 最终候选
    │
    ├── REJECTED
    │      │
    │      ▼
    │   Three-Action Repair Policy
    │      ├── Prompt Repair ──► 改写 Prompt ──► Wan
    │      ├── Local Editing ──► SAM2 masks ──► ProPainter
    │      └── Reject ─────────► 审计式停止并保留最佳候选
    │
    └── UNAVAILABLE ───────────► EVALUATION_FAILED
                                   不执行 Policy，不伪装成视频不合格

Prompt Repair / Local Editing 产生新候选
    │
    ▼
Re-Critic
    │
    ▼
Strict Enforce Re-Gate
    ├── ACCEPTED ─► 严格成功，输出新候选
    ├── REJECTED ─► 下一轮或达到上限
    └── UNAVAILABLE ─► EVALUATION_FAILED
    │
    ▼
RoundRecord / Repair Trace / WanRepairTrialV3 / Run Audit
~~~

这不是两条互相独立的链路。首轮 Critic/Gate 和修复后的 Re-Critic/Re-Gate 是同一个评价闭环在两个时间点上的调用：第一次决定是否需要修复，第二次验证修复是否有效。

---

## 4. 最终架构的设计原则

### 4.1 严格失败关闭

评价工具失效不能被解释为视频物理不合格，也不能因为没有发现错误就默认通过。因此 Gate 使用三个状态：

- `ACCEPTED`：证据完整且所有硬性条件通过；
- `REJECTED`：评价有效，但视频没有达到接受条件；
- `UNAVAILABLE`：评价链路不完整、required scorer 缺失、provider failure 或禁止性 degraded。

只有 `REJECTED` 可以进入 Repair Policy。`UNAVAILABLE` 直接结束为 `EVALUATION_FAILED`。

### 4.2 修复必须被重新评价

Executor 成功产生文件不等于修复成功。Prompt Repair 或 ProPainter 产生的新视频必须经过 Re-Critic 和 Strict Re-Gate。

### 4.3 真实决策不可被测试参数覆盖

正式入口不提供 `--force-action`。Policy 每轮只决策一次，Guard 只检查动作能否执行，不能偷偷把一个动作替换成另一个动作。

### 4.4 评价、决策与执行分层

- Critic 回答“视频发生了什么、证据是什么”；
- Gate 回答“证据是否足够、候选是否达标”；
- Policy 回答“在当前证据和能力下采取什么动作”；
- Guard 回答“这个动作当前是否安全且可执行”；
- Executor 回答“如何完成选定动作”；
- Trial 回答“本轮真实发生了什么、结果是否改善”。

### 4.5 不依赖 PhysicsPlan 的在线主链路

最终 V2 主链路不在首轮前运行 PhysicsPlan，也不向 Runner、Generator、Executor、Local Editor 或 Critic subprocess 透传 PhysicsPlan。pavg_critic 中仍可能存在兼容历史数据的旧结构或文案，但在线 WanPhysics V2 显式设置 `use_physics_plan=False`，不会调用 Physics Planner。

### 4.6 不保留 Global Regen 与 Memory 在线路径

- Global Regeneration 已从正式动作空间删除；
- Memory 当前不参与在线读取、决策或写回；
- 动作空间固定为 Prompt Repair、Local Editing、Reject；
- 历史四动作 checkpoint 不允许进入三动作 runtime。

---

## 5. 阶段一：输入与运行契约

### 5.1 输入

每条样本至少包含：

- 稳定的 `sample_id`；
- 原始文本 Prompt；
- 生成 seed 或可追踪的默认 seed；
- manifest 中的样本元数据。

正式入口只有：

~~~text
agents/wanphysics/run_videophy2_loop_v2.py
~~~

### 5.2 Preflight

启用真实链路前，Preflight 检查：

- Wan 模型和推理环境；
- Qwen3-VL 模型和 vLLM 端口；
- SAM2 仓库、checkpoint 与 CUDA 扩展；
- ProPainter 仓库、入口脚本和三个必要权重；
- ffmpeg、OpenCV、GPU 与环境变量；
- 三动作 Executor capability；
- Strict Gate 所需配置。

关键能力不可用时应进入 `PREFLIGHT_FAILED`，而不是运行到中途再降级为“看似成功”。

### 5.3 RUN_ROOT

用户显式提供的 `--output-root` 就是最终 RUN_ROOT，不在其内部再创建时间戳子目录。所有日志、状态、样本、attempt 和 Trial 都归入该目录。

---

## 6. 阶段二：Wan 首轮视频生成

Wan2.2-TI2V-5B 根据原始 Prompt 生成首个视频候选。候选记录包括：

- `candidate_id`；
- 实际视频路径；
- 本次使用的 Prompt；
- seed；
- 帧数、分辨率、帧率；
- backend 和是否为真实视频等元数据。

生成失败是 `EXECUTION_FAILED`，不会进入 Critic。生成成功后，视频候选成为首轮 before candidate。

当前双卡配置中：

- GPU0 主要用于 Wan 生成和需要时的 ProPainter；
- GPU1 主要用于 vLLM/Qwen3-VL；
- 单卡条件下，Runner 必须在生成/编辑和 Critic 之间进行受控资源交接。

---

## 7. 阶段三：视频观测与证据构建

### 7.1 Qwen3-VL 对象语义种子

Qwen3-VL 的当前主要角色是为视频中与 Prompt 相关的对象提供识别和定位种子，帮助 SAM2 建立目标。它不应被夸大为独立完成全部物理判断的万能 verifier。

### 7.2 SAM2 分割与跨帧跟踪

SAM2 根据对象种子生成逐帧 mask，并在完整视频中传播对象身份。输出为：

- 对象在每一帧的可见性；
- 逐帧 mask；
- mask 边界框、中心或可推导的几何特征；
- 可用于 Local Editing 的严格 mask sequence。

SAM2 CUDA 扩展 `_C` 已在服务器编译成功：

~~~text
/root/PhysGenLoop-/models/sam2-src/sam2/_C.so
~~~

运行报告必须显式记录：

~~~text
sam2_cuda_extension = available | unavailable
sam2_postprocess = enabled | hole_filling_disabled
degraded = true | false
~~~

Strict Gate 不允许禁止性 degraded 的 Critic 被接受。

### 7.3 轨迹提取

由对象跨帧位置和 mask 推导时序轨迹，用于检查：

- 位移和速度变化；
- 轨迹连续性；
- 身份稳定性；
- 突然跳变、消失或非自然运动；
- 与关键事件相关的帧区间。

### 7.4 事件检测

事件检测将低层轨迹转成可解释的事件候选，例如接触、碰撞、反弹、遮挡、消失、轨迹突变。事件和关键帧随后进入 Critic 的证据融合。

### 7.5 Strict mask manifest

每帧 mask 必须与源视频帧一一对应。禁止：

- 用一张 mask 复制到所有帧；
- 在 mask 缺失时使用整帧 mask；
- 静默改变帧数或分辨率；
- 把不完整 mask 当成可执行 local target。

---

## 8. 阶段四：Physics Critic

### 8.1 Critic 的作用

Critic 不是一个简单二分类器，而是一个证据组织器。它需要输出：

- `decision`：physical、non-physical 或 unknown 语义；
- `physics_score`；
- `confidence`；
- `coverage`；
- violations 及类别；
- 关键帧；
- repair instruction；
- 可选 local target；
- score breakdown；
- diagnostics、provider failures 和 degraded 状态；
- 多来源 evidence bundles。

### 8.2 多来源证据

当前报告可以包含：

- deterministic rules；
- video-science checklist；
- mechanics evaluator；
- VLM candidate review；
- 对象轨迹与事件证据。

不同证据可能为 `available`、`not_applicable` 或不可用。Critic 必须保留这些区别，不能把“不适用”写成“通过”。

### 8.3 Critic 与 Gate 的区别

Critic 负责产生报告；Gate 负责按项目阈值解释报告。即使 `physics_score` 较高，如果 confidence、coverage 或 required scorer 不足，Gate 仍可拒绝或返回不可用。

---

## 9. 阶段五：辅助评分

除了 Critic 的物理评分，Gate 还关心：

- `semantic_score`：当前候选与当前生成 Prompt 的一致性；
- `original_prompt_semantic_score`：当前候选与用户原始 Prompt 的一致性；
- `quality_score`：画面质量或可用性评价。

双 Semantic 的必要性在于：Prompt Repair 会修改当前 Prompt。如果只比较修改后的 Prompt，系统可能通过“改变问题”获得高分，却偏离用户原始意图。因此必须同时保留与当前 Prompt、原始 Prompt 的一致性。

---

## 10. 阶段六：Strict Enforce Acceptance Gate

### 10.1 Gate 的作用

Acceptance Gate 是闭环的质量合同。它把 Critic 和辅助评分转换为明确、不可含糊的控制流状态。

### 10.2 ACCEPTED

只有在以下条件同时满足时才可接受：

- required scorers 都可用；
- Critic 没有禁止性 provider failure 或 degraded；
- 不存在 blocking violation；
- physics、confidence、coverage 达到阈值；
- semantic、original-prompt semantic、quality 达到阈值；
- 对修复后候选，还必须满足相对于 before candidate 的无退化限制。

### 10.3 REJECTED

评价链路有效，但候选未达到标准，包括：

- 存在 blocking violation；
- 物理分数不足；
- confidence 或 coverage 不足；
- Semantic 或 quality 不足。

只有这个状态进入 Repair Policy。

### 10.4 UNAVAILABLE

代表无法可靠评价，例如：

- required scorer 没有结果；
- Critic 报告 unknown 到无法做 Gate；
- provider failure；
- SAM2 等关键模块处于禁止性 degraded；
- 报告缺失必要字段。

它表示“评价失败”，不是“视频物理失败”。

---

## 11. 阶段七：Three-Action Repair Policy

### 11.1 Policy 输入

Policy 根据真实 Critic 报告、Gate 结果、历史动作、剩余轮数和当前 capability 构建 RepairContext，输出：

- 唯一动作；
- 三动作概率；
- per-action values；
- confidence；
- repair instruction；
- 可选 local target；
- decision source 和 compatibility 信息。

### 11.2 三动作

#### Prompt Repair

适用于可通过文字约束改善的全局生成问题，如重力、接触或摩擦表达不清。它修改 Prompt 文本，再调用 Wan 生成新候选。

#### Local Editing

适用于局部、时空范围明确且有完整 mask 的问题，如碰撞区域、轨迹局部、连续性或外观异常。它使用 SAM2 mask sequence 和 ProPainter 编辑原视频。

#### Reject

适用于未知违背、证据不足、没有安全 local target、修复能力不可用或继续修复风险高的情况。Reject 是显式、可审计的停止动作，不是异常。

### 11.3 Global Regen 为何删除

Global Regen 与 Prompt Repair 都会调用 Wan 生成全新视频，但前者不明确修改 Prompt，容易退化为仅换 seed 的盲目重试。最终架构只保留具有明确修复意图的 Prompt Repair。

---

## 12. 阶段八：Policy Guard

Guard 不负责重新决策，只验证 Policy 选出的动作：

- 动作是否属于三动作集合；
- 对应 backend 是否可用；
- Prompt Repair 是否有安全、非空的 instruction；
- Local Editing 是否有 local target；
- mask coverage、帧数和逐帧完整性是否达标；
- 当前轮数和上下文是否允许执行。

Guard 输出 `allowed` 或 blocked 原因。若动作不可执行，系统走 Audited Reject，而不是静默换成另一动作。

---

## 13. 阶段九：三类 Executor

### 13.1 Prompt Repair Executor

正式接线为：

~~~text
PromptRepairExecutor
  + InstructionPromptRepairer
  + WanSubprocessGenerator
~~~

流程：读取原 Prompt 和 Critic instruction，生成物理约束更明确的新 Prompt；若没有产生安全文本变化，则返回 `no_safe_prompt_change`；成功改写后由 Wan 生成新视频。

### 13.2 ProPainter Local Editing Executor

正式接线为：

~~~text
MaskSequenceLocalEditingExecutor
  + StrictProPainterLocalEditor
  + models/ProPainter/inference_propainter.py
~~~

流程：校验逐帧 mask manifest，整理 ProPainter 输入，执行局部视频修复，并验证输出视频存在、可解码、帧数和分辨率与输入契约一致。

### 13.3 Audited Reject Executor

Reject 不产生新视频。它记录选中的最佳历史候选、Policy 决策、Guard、停止原因和零编辑成本，使“为什么没有继续修复”可追踪。

---

## 14. 阶段十：Re-Critic 与 Re-Gate

Prompt Repair 或 Local Editing 成功产生 after candidate 后：

1. 使用相同 Critic 重新观察新视频；
2. 重新计算 physics、confidence、coverage；
3. 重新计算双 Semantic 和 quality；
4. 使用同一 Strict Gate 评价；
5. 比较 before/after 分数和副作用。

结果分支：

- Re-Gate `ACCEPTED`：最终接受；
- Re-Gate `REJECTED`：未达到严格标准，若还有轮数则进入下一轮；
- Re-Gate `UNAVAILABLE`：评价失败，不能记为修复失败或成功。

Runner 会缓存 after evaluation，若进入下一轮，它直接成为下一轮的 before evaluation，避免对同一候选重复调用 Critic。

---

## 15. Runner 状态机

### 15.1 样本终态

正式样本只能结束为：

- `ACCEPTED`；
- `REJECTED`；
- `MAX_ROUNDS`；
- `EVALUATION_FAILED`；
- `EXECUTION_FAILED`；
- `PREFLIGHT_FAILED`。

### 15.2 轮次逻辑

每轮的标准顺序是：

~~~text
before candidate
→ Critic/Gate（或复用上一轮 after evaluation）
→ Policy 一次
→ Guard 一次
→ Executor 一次
→ after candidate
→ Re-Critic/Re-Gate
→ 终止或进入下一轮
~~~

### 15.3 Best candidate

Runner 保留历史候选及评价。Reject 或达到轮数上限时可以返回可审计的最佳候选，但不能把“最佳候选”自动标记为 `ACCEPTED`。

---

## 16. Trial 与审计

### 16.1 WanRepairTrialV3

每次真实动作对应一个 `WanRepairTrialV3`，schema 为：

~~~text
wan-repair-trial/3.0
~~~

Trial 保存：

- source candidate、original Prompt、before Prompt；
- 完整 critic_before；
- 真实 RepairDecision 和三动作概率；
- Guard 结果；
- ExecutionResult；
- before scores；
- after candidate、critic_after 和 after scores；
- before/after Gate；
- before/after 视频路径；
- physics gain；
- `repair_improved` 与 `successful`；
- failure reason。

### 16.2 两种成功语义

- `repair_improved=true`：物理分数提高，且 Semantic 与 quality 没有超过允许的退化；
- `successful=true`：不仅 improved，而且 Strict Re-Gate 明确 `ACCEPTED`。

这避免将“局部指标有改善”夸大为“闭环修复成功”。

### 16.3 不伪造因果字段

Trial 从 Runner 的真实 RoundRecord 搬运字段，不重新构造 Decision，不把 Gate 当 Critic，不补写不存在的 after report，也不伪造动作概率。

---

## 17. 输出目录与可复现性

典型目录：

~~~text
RUN_ROOT/
├── run_manifest.json               # 不可变运行合同
├── run_status.json                 # 动态 run 状态
├── run.lock                        # 并发保护
├── summary.json                    # authoritative attempts 汇总
├── physgenloop_*.log
├── physgenloop_*.status
├── physgenloop_*.pid
├── vllm.log
└── <sample_id>/
    ├── active_attempt.json
    ├── sample_status.json
    ├── sample_status_history.jsonl
    └── attempts/
        └── attempt_XXXX/
            ├── loop_result.json
            ├── resource_metrics.jsonl
            ├── repair_trace.jsonl
            ├── trials.jsonl
            └── <candidate_id>/
                ├── video.mp4
                ├── prompt.txt
                ├── metadata.json
                ├── critic.json
                ├── critic_report.json
                ├── mask_manifest.json
                └── repair_decision.json
~~~

`run_manifest.json` 保存 source revision、工作区 fingerprint、配置和 manifest 哈希、样本集合、阈值模式和动作集合。Resume 时这些不可变字段必须匹配。

### 17.1 Resume

- `--resume` 在同一 RUN_ROOT 继续未完成样本；
- `--retry-failed` 只重试 `EVALUATION_FAILED`、`EXECUTION_FAILED`、`PREFLIGHT_FAILED`；
- 已接受、已拒绝或达到轮数上限的样本不会被无条件覆盖；
- 每次重试创建新 attempt；
- summary 只读取 `authoritative_attempt`。

---

## 18. GPU、vLLM 与进程所有权

每个 run 只管理自己启动的进程：

- vLLM 由本 run 的 Popen/process group 启动；
- owner 信息写入 `vllm.owner.json` 或相应审计状态；
- 结束时只终止自己的 PID；
- 禁止宽泛 `pkill -f` 影响其他组员任务；
- 运行结束后检查 GPU 进程是否释放。

资源指标写入 `resource_metrics.jsonl`，并与实际 GPU 对应。

---

## 19. 标准运行方式

### 19.1 单样本 smoke

~~~bash
cd /root/PhysGenLoop- || exit 1

RUN_ROOT=/root/PhysGenLoop-/outputs/v2_smoke_strict
LOG="$RUN_ROOT/physgenloop_v2_smoke_strict.log"
STATUS="$RUN_ROOT/physgenloop_v2_smoke_strict.status"
PIDFILE="$RUN_ROOT/physgenloop_v2_smoke_strict.pid"

mkdir -p "$RUN_ROOT"

nohup bash -c '
cd /root/PhysGenLoop- || exit 1

PYTHONPATH=/root/PhysGenLoop-:/root/PhysGenLoop-/src \
envs/main/bin/python \
agents/wanphysics/run_videophy2_loop_v2.py \
  --enable \
  --manifest evaluation/manifests/videophy2_smoke_dev10.json \
  --limit 1 \
  --max-rounds 2 \
  --output-root "$1"

status=$?
printf "%s\n" "$status" > "$2"
exit "$status"
' _ "$RUN_ROOT" "$STATUS" > "$LOG" 2>&1 &

PID=$!
printf "%s\n" "$PID" > "$PIDFILE"
~~~

### 19.2 监控

~~~bash
cat "$PIDFILE"
tail -f "$LOG"
cat "$STATUS"
cat "$RUN_ROOT/run_status.json"
cat "$RUN_ROOT/summary.json"
~~~

### 19.3 Resume

在原命令参数完全一致的基础上增加：

~~~text
--resume
~~~

只重试允许重试的失败状态时再增加：

~~~text
--retry-failed
~~~

---

## 20. 已完成验证

### 20.1 静态与单元验证

~~~text
git diff --check                         PASS
pytest tests/wanphysics_v2 -q            60 passed
SAM2 CUDA extension import               PASS
Wan / Qwen3-VL / SAM2 checkpoint         PASS
ProPainter repository and weights        PASS
ffmpeg / cv2 / GPU / port preflight      PASS
~~~

### 20.2 Strict dry-run

~~~text
RUN_ROOT = /root/PhysGenLoop-/outputs/v2_dryrun_strict_impl_20260722_095917
before Gate = REJECTED
Policy action = prompt_repair
Re-Gate = ACCEPTED
Trial successful = true
physics_gain = 0.7
~~~

### 20.3 最终真实 smoke

~~~text
RUN_ROOT = /root/PhysGenLoop-/outputs/v2_smoke_enforce_impl_final_20260722_181402
run state = COMPLETED_WITH_REJECTIONS
shell status = 0
Wan output = 81 frames, 832×480, 24 fps
SAM2 CUDA extension = available
Critic degraded = false
Gate = REJECTED
Policy action = reject
Guard = allowed
final state = REJECTED
terminal reason = policy_reject
Trial = wan-repair-trial/3.0
remaining GPU processes = none
~~~

真实 smoke 的 `REJECTED` 是业务终态，不是运行失败。该样本未达到 confidence、coverage 和双 Semantic 阈值，Policy 在没有可信 local target 时自然选择 Reject。

---

## 21. 当前验证边界

已经真实覆盖：

- Wan 真实生成；
- Qwen3-VL/vLLM 启动和释放；
- SAM2 81 帧跟踪与 CUDA 扩展；
- Critic 完整报告；
- Strict Gate；
- 自然 Policy 决策；
- Guard 和 Audited Reject；
- Trial V3 和 run 级审计。

尚需从自然 pilot 样本中继续获得：

- 自然选择 Prompt Repair 的真实 before/after 视频；
- 自然产生完整 local target 并触发 ProPainter；
- 多轮 Re-Critic/Re-Gate 的真实统计；
- Resume/retry-failed 长任务验证；
- limit=5、pilot300 的稳定性和成功率。

不应重新加入 `--force-action` 来制造动作覆盖，因为那会破坏“Policy 真实决策”的研究语义。

---

## 22. 项目的主要创新点

### 22.1 从一次生成变为闭环优化

系统不仅发现问题，还把评价结果转成可执行的修复策略，并用同一严格标准验证修复。

### 22.2 将不可用与不合格分开

三状态 Gate 防止评价系统自身故障污染物理失败统计，是研究可信度的重要保障。

### 22.3 多粒度修复

Prompt Repair 面向全局生成语义，ProPainter 面向局部时空区域，Reject 面向不可安全修复场景；三者形成成本、范围与风险不同的动作空间。

### 22.4 双 Semantic 防止目标漂移

同时约束当前 Prompt 和原始 Prompt，避免通过改写任务本身获得虚假成功。

### 22.5 因果可审计 Trial

每个 Trial 保存真实 before、Decision、Guard、Execution、after 和 Gate，使后续统计、学习和错误分析有可信数据基础。

### 22.6 工程与研究契约统一

RUN_ROOT、attempt、Resume、进程所有权和 schema 使长时间 GPU 实验可复现、可恢复且不干扰其他组员。

---

## 23. 可用于实验分析的指标

建议后续 pilot 汇总：

- 首轮 Gate 接受率；
- `REJECTED / UNAVAILABLE` 比例；
- 三动作自然选择分布；
- Prompt Repair 和 Local Editing 的执行成功率；
- Re-Gate 接受率；
- 平均 physics gain；
- Semantic 和 original-prompt semantic 变化；
- quality drop；
- 每类 violation 的修复成功率；
- 每个 accepted sample 的平均轮数和 GPU 时间；
- required scorer unavailable 和 provider failure 分布；
- Reject 的主要原因；
- SAM2 mask coverage 与 Local Editing 成功率关系。

---

## 24. PPT 制作建议

### 24.1 推荐叙事线

1. 文本生成视频“视觉逼真但物理不可信”的问题；
2. 一次生成和一次评分为什么不够；
3. PhysGenLoop 的闭环思想；
4. 视频观测与多证据 Critic；
5. Strict Gate 三状态；
6. 三动作 Repair Policy；
7. Prompt Repair 与 ProPainter 的互补性；
8. Re-Critic/Re-Gate 如何定义真正成功；
9. Trial V3 与工程审计；
10. 当前 smoke 结果、验证边界与下一步实验。

### 24.2 推荐 16 页结构

| 页码 | 标题 | 核心内容 | 推荐图形 |
|---|---|---|---|
| 1 | PhysGenLoop | 标题、项目一句话 | 视频帧 + 闭环箭头 |
| 2 | Problem | 物理违背类型与现有生成局限 | 错误视频关键帧 |
| 3 | Research Question | 发现、区分、修复、验证四个问题 | 四问框图 |
| 4 | Core Idea | Generate–Critique–Repair 闭环 | 总流程图 |
| 5 | System Overview | 最终十阶段架构 | 横向 pipeline |
| 6 | Video Observation | VLM、SAM2、轨迹、事件 | 四层证据图 |
| 7 | Physics Critic | 多来源 evidence fusion | 证据汇聚图 |
| 8 | Strict Gate | ACCEPTED/REJECTED/UNAVAILABLE | 三分支状态机 |
| 9 | Repair Policy | 三动作及适用范围 | 三列对比 |
| 10 | Prompt Repair | instruction 到新 Prompt 和 Wan | before/after Prompt |
| 11 | Local Editing | SAM2 mask 到 ProPainter | 帧-mask-结果图 |
| 12 | Re-Critic/Re-Gate | 为什么执行成功不等于修复成功 | 双 Gate 时间线 |
| 13 | Trial & Audit | before/action/after 因果记录 | Trial 数据结构 |
| 14 | Engineering | RUN_ROOT、Resume、GPU owner | 目录树 + 双卡图 |
| 15 | Validation | 60 tests、dry-run、真实 smoke | 结果表 |
| 16 | Outlook | natural branches、pilot300、学习型 Policy | roadmap |

### 24.3 PPT 中必须保留的严谨表述

- “真实 smoke 跑通”不等于“该样本被接受”；本次真实样本终态为正常 `REJECTED`。
- Prompt Repair 的严格闭环已由 dry-run 验证，但尚缺自然真实样本。
- ProPainter 的接口和 preflight 已验证，但尚缺自然 local target 触发的真实修复样本。
- 当前在线 Policy 是三动作 heuristic policy，不应表述为已经完成训练并 deployment-ready 的学习型策略。
- Qwen3-VL 当前主要提供对象语义和视觉种子，不应表述为独立完成全部物理推理。
- Memory 和 Global Regen 不属于最终在线主链路。

---

## 25. 术语表

| 术语 | 含义 |
|---|---|
| Candidate | 某轮生成或编辑得到的视频候选 |
| Critic | 产生物理评价、证据、置信度和覆盖度的模块 |
| Gate | 将评分和完整性转成三状态的严格质量合同 |
| Repair Policy | 在三个动作中选择唯一动作的决策模块 |
| Guard | 检查 Policy 动作是否安全、完整、可执行 |
| Prompt Repair | 修改文本约束后重新调用 Wan 的修复动作 |
| Local Editing | 使用逐帧 mask 和 ProPainter 编辑局部区域 |
| Reject | 有审计记录的停止动作 |
| Re-Critic | 对修复后候选重新执行 Critic |
| Re-Gate | 对修复后候选重新执行 Strict Gate |
| Repair-improved | 指标改善且副作用在限制内 |
| Successful | Repair-improved 且 Re-Gate 为 ACCEPTED |
| Trial | 一次 before/action/after 的真实因果审计记录 |
| Attempt | 同一样本在新运行或重试中的隔离执行实例 |
| RUN_ROOT | 一次运行所有日志、状态和产物的统一根目录 |

---

## 26. 总结

PhysGenLoop 的最终目标不是保证每个视频都被修好，而是保证每个结果都有可信含义：被接受的视频经过完整严格评价；被修复的视频经过同标准复评；被拒绝的视频有明确决策依据；评价失败不会被误记为视频失败；每一步都有真实、可复现的产物。最终 V2 因此形成了一个以严格 Gate 为控制中心、以三动作 Repair 为执行空间、以 Re-Critic/Re-Gate 为成功证明、以 Trial V3 为审计基础的物理一致性视频生成闭环。
