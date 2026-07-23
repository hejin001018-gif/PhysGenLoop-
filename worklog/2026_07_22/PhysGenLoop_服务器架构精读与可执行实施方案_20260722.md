# PhysGenLoop 服务器架构精读与可执行实施方案

日期：2026-07-22  
署名：hejin
适用代码：服务器 /root/PhysGenLoop-  
服务器基线：569ffb7 refactor: collapse wanphysics to v2 single chain

依据：

- /root/PhysGenLoop-/README.md
- /root/PhysGenLoop-/worklog/2026_07_22/重构v2方案.md
- /root/PhysGenLoop-/worklog/2026_07_22/v2重构后全链路架构.md
- 当前服务器源代码、配置、schema、测试和最新真实输出
- 最新决策：废弃 Memory 在线模块；Prompt Repair 使用旧版 PromptRepairExecutor；删除 Global Regeneration；在线运行只保留单入口、三动作 Policy 和 Strict Enforce Gate；退役 `--force-action` 与 forced-trial 运行路径

> 本文是新生成的实施文档，不修改既有 PhysGenLoop_全链路问题修复方案。本轮只完成服务器代码阅读和本地文档生成，没有修改服务器代码、配置、权重、输出或任务状态。

---

## 1. 目标与最终主链路

目标是基于服务器现有 V2 单链路继续补齐，不重新设计第二套 runner、executor 或 artifact 系统。

最终主链路：

~~~text
VideoPhy2 manifest / prompt
  -> Wan2.2 candidate generation
  -> Qwen3-VL object seed + SAM2 tracking + Physics Critic
  -> CriticReport V2 codec
  -> critic_report.json / mask_manifest.json
  -> Acceptance Gate
  -> Three-Action Repair Policy
  -> PolicyGuard
  -> Prompt Repair / ProPainter Local Editing / Reject
  -> Re-Critic / Re-Gate
  -> next round or final candidate
  -> WanRepairTrialV3 + audit artifacts
~~~

Memory 不再属于在线闭环。Trial、repair trace、critic report 和 loop result 继续保留，因为它们是实验与审计证据，不是 Memory。

架构决策：

1. 废弃 Memory 的运行时读取、混合和写回。
2. Prompt Repair 必须对齐旧版 PromptRepairExecutor + InstructionPromptRepairer。
3. 从在线主链路删除 PhysicsPlan 参数透传；Critic 内部依赖分阶段解耦，禁止直接删字段造成 Critic 失效。
4. 删除 Global Regeneration 动作；Repair Policy 严格收敛为 prompt_repair、local_editing、reject。
5. 不删除其他无关模块和历史审计产物；旧四动作 checkpoint/Trial 只标记 legacy incompatible，不再进入运行和训练。
6. 所有改动优先落在现有责任文件，避免新增同职责文件。
7. 服务器当前 V2 入口和 generators/wanphysics/v2/ 是权威运行边界。
8. 正式与真实 smoke 只允许 `agents/wanphysics/run_videophy2_loop_v2.py` 一个入口；动作必须来自真实 Three-Action Repair Policy。
9. `--force-action`、动作覆盖配置和 `run_actual_trials_v2.py` forced-trial 路径退出活跃架构；历史文件和产物保留只读，不再生成新版 Trial。
10. 显式 `--output-root "$RUN_ROOT"` 时，该路径就是最终运行目录；日志、PID、status、manifest、summary 与样本目录统一位于 `RUN_ROOT`。

三动作含义必须固定：

| 动作 | 实际执行 | 适用边界 |
|---|---|---|
| prompt_repair | InstructionPromptRepairer 修改当前 prompt，Wan 使用新 prompt 生成完整新视频 | 大范围或生成机制层面的明确物理违背 |
| local_editing | SAM2 严格逐帧 mask + StrictProPainterLocalEditor + ProPainter | 局部、短时、对象明确且 mask 有效 |
| reject | 停止继续修复并从候选历史选择最佳结果 | 证据不足、无安全修复、重复无收益或达到上限 |

Prompt Repair 虽然会重新生成完整视频，但它必须满足 repaired_prompt != input_prompt。使用相同 prompt 只换 seed 的动作已从新架构删除。

动作来源契约固定为：

~~~text
Critic -> Strict Gate(REJECTED) -> Three-Action Policy
       -> PolicyGuard capability validation
       -> ExecutorRegistry
~~~

CLI、配置、测试 manifest 和 Executor 均不得覆盖 Policy 选择；`ACCEPTED` 必须停止，`UNAVAILABLE` 必须进入 `EVALUATION_FAILED`，二者都不得调用 Policy。

---

## 2. 服务器当前代码结构精读

### 2.1 入口层

| 文件 | 当前职责 | 核心接口 |
|---|---|---|
| agents/wanphysics/run_videophy2_loop_v2.py | V2 主入口、样本循环、trial 组装 | main、_real_run、_assemble_trials |
| agents/wanphysics/run_actual_trials_v2.py | 历史 forced-trial 入口；目标状态为 retired/fail-fast，不属于活跃链路 | legacy _real_trials（只读审计） |
| agents/wanphysics/gen_step.py | 一次性 Wan2.2 生成子进程 | main |
| agents/wanphysics/eval_step.py | 一次性 SAM2/Critic 子进程 | main |

当前主入口对每个样本执行：

~~~python
runner, critic, artifacts, preflight = build_v2_runner(...)
result = runner.run(sample_id=sid, prompt=prompt, physics_plan=None)
~~~

关键事实：physics_plan 固定为 None，所以当前不是“先规划再生成”，而是“先生成，Critic 内部再尝试解析计划”。

目标入口删除无效参数：

~~~python
runner, critic, artifacts, preflight = build_v2_runner(...)
result = runner.run(sample_id=sid, prompt=prompt)
~~~

### 2.2 生成调用链

~~~text
ActionAwareRunnerV2
  -> WanSubprocessGenerator.generate()
  -> agents/wanphysics/gen_step.py
  -> WanGenerator.generate_video()
  -> Wan2.2-TI2V-5B
  -> candidate video + prompt.txt + metadata.json + critic.json
~~~

接口：

~~~python
generator.generate(
    prompt: str,
    physics_plan: PhysicsPlan,
    seed: int,
) -> GeneratedCandidate
~~~

当前 adapter 接收 physics_plan，但 gen_step 和 WanGenerator 实际只消费 prompt、seed、分辨率等，physics_plan 没有真正进入生成条件。

目标接口：

~~~python
generator.generate(
    prompt: str,
    seed: int,
) -> GeneratedCandidate
~~~

WanPhysicsGenerator 和 WanSubprocessGenerator 均删除 PhysicsPlan import 与形参，函数体不需要增加替代逻辑。

### 2.3 Critic 调用链

~~~text
_V2Critic.evaluate()
  -> V2SubprocessCritic.evaluate()
  -> agents/wanphysics/eval_step.py
  -> OpenAIChatModel
  -> SAM2ObjectDetector
  -> PhysicsCritic.analyze()
  -> CriticReport.to_dict()
  -> critic_codec.decode_report()
~~~

Qwen3-VL 当前主要用于首帧对象识别和 SAM2 初始化。PhysicsCritic 没有注入 model-based Physics Planner、VLM physics verifier 或 model-based question graph，因此准确能力画像是：

~~~text
sam2_seeded_rules
~~~

而不是完整的 VLM 物理判定模型。

### 2.4 Critic codec

critic_codec.py 当前负责从跨进程 JSON 恢复：

- violations
- critical_frames
- repair_instruction
- evidence
- mask_uri / mask_uris
- evidence_bundles
- diagnostics
- model_versions
- score_breakdown

解析失败时返回 roundtrip_failed 和 raw payload，runner 终止当前样本。这一边界应保留。



### 2.6 PolicyGuard

服务器基线中的 `policy_guard.resolve_action()` 会结合 Policy 输出、scope、mask、ProPainter capability 和 repair instruction 再次得到 final action，因而存在 Guard 替换 Policy Decision、以及在 Executor 前假设 rewriter 能修改 prompt 的接口问题。

目标架构必须改为：

~~~text
Strict Gate(REJECTED)
  -> Policy 消费 Critic/Gate/capability，每轮只产生一个真实三动作 Decision
  -> PolicyGuard 只验证 Decision 与 capability/必需字段
       -> allowed：原样交给对应 Executor
       -> blocked：保留原始 Decision 和 blocked_reason，显式执行 Audited Reject
~~~

Guard 不得把 `local_editing` 静默替换为 `prompt_repair`，也不得在 InstructionPromptRepairer 真正运行前判断 rewritten prompt 是否变化。Critic/provider/scorer failure 必须在 Gate 层成为 `UNAVAILABLE -> EVALUATION_FAILED`，不得进入 Guard 后转成 Reject。

动作集合的强约束由 Policy、contracts 和 Registry 共同保证：

~~~python
decision.action in {
    "prompt_repair",
    "local_editing",
    "reject",
}
~~~

### 2.7 当前 ExecutorRegistry

当前 build_backends.py 注册：

~~~text
DecisionPromptRepairExecutor
OriginalPromptGlobalRegenerationExecutor
MaskSequenceLocalEditingExecutor
AuditedRejectExecutor
~~~

其中第一项不符合最新决定。目标注册应是：

~~~text
PromptRepairExecutor
  + InstructionPromptRepairer
  + WanSubprocessGenerator

MaskSequenceLocalEditingExecutor
AuditedRejectExecutor
~~~

OriginalPromptGlobalRegenerationExecutor 从 import、实例化和 registry 中删除。DecisionPromptRepairExecutor 不再作为默认后端；如为读取历史产物暂时保留兼容定义，也不得进入 runtime registry。

#### 2.7.1 增强版旧 Prompt Repair：实施目标

保持现有职责边界，不把文本改写放进 Policy，也不使用 DecisionPromptRepairExecutor：

~~~text
CriticReport
  -> Repair Policy 只选择 prompt_repair
  -> PromptRepairExecutor
  -> InstructionPromptRepairer
  -> repaired prompt
  -> WanSubprocessGenerator
  -> after candidate
  -> Re-Critic / Re-Gate
~~~

增强目标：

- 原始对象、场景、动作、镜头和风格保持不变；
- 只根据 CriticReport.violations 增加物理行为约束；
- repair_instruction 为空时使用 category fallback；
- 把 frame evidence 转成事件阶段语言，不直接把帧号写给 Wan；
- 每轮最多使用 1～3 条关键约束；
- 多轮修复替换旧约束块，不无限追加；
- 不引入 Critic 没有依据的新对象、支撑物或事件；
- Policy 每轮只调用一次；
- 修改前后 prompt 和证据可审计；
- 无安全改写时显式失败，不使用同一 prompt 假装完成 Prompt Repair。

#### 2.7.2 只修改现有责任文件

实施时优先修改：

~~~text
src/physgenloop/repairer.py
src/physgenloop/learning_repair/executors.py
generators/wanphysics/v2/build_backends.py
configs/loop_v2.yaml
tests/wanphysics_v2/test_decision_only_executors.py
tests/wanphysics_v2/test_build_backends.py
agents/wanphysics/run_videophy2_loop_v2.py
~~~

说明：

- 当前提交已经删除 src/physgenloop/repairer.py，需要按原路径恢复 InstructionPromptRepairer；
- 不新增第二个 prompt renderer 文件；
- 不新增第二个 PromptRepairExecutor；
- generators/wanphysics/v2/prompt_renderer.py 可以继续保留，但不作为默认 Prompt Repair 路径；
- 原版 PromptRepairExecutor 的 action、registry 和 ExecutionRequest/ExecutionResult 契约保持不变。

#### 2.7.3 配置修改

在 configs/loop_v2.yaml 中增加：

~~~yaml
prompt_repair:
  backend: legacy
  backend_id: legacy-prompt-rewriter+video-generator

  # 每轮最多写入多少条物理约束
  max_constraints: 3

  # 包含原始 prompt 和物理约束块的总长度上限
  max_prompt_chars: 900

  # 多轮修复时替换上一轮约束块
  replace_existing_block: true

  # 不允许 repairer 自行创造新对象或因果装置
  allow_new_entities: false

  # 没有安全、有效的文本变化时不生成相同 prompt 的新视频
  fail_on_no_safe_change: true

  # 优先使用 violation.repair_instruction；为空时使用 category fallback
  category_fallback: true
~~~

默认必须是 legacy。只有显式实验时才能选择 v2 renderer，且实验结果不能混入旧版 Prompt Repair 的正式统计。

#### 2.7.4 恢复并增强 InstructionPromptRepairer

在 src/physgenloop/repairer.py 中恢复原类名和原接口：

~~~python
class InstructionPromptRepairer:
    def repair(self, *, prompt: str, report: CriticReport) -> str:
        ...
~~~

为审计增加兼容方法，但不改变 repair() 的返回类型：

~~~python
@dataclass(frozen=True)
class PromptRepairTrace:
    original_prompt: str
    input_prompt: str
    repaired_prompt: str
    changed: bool
    target_objects: tuple[str, ...]
    violation_categories: tuple[str, ...]
    constraints_used: tuple[str, ...]
    skipped_constraints: tuple[str, ...]
    instruction_source: str
    failure_reason: str | None = None


class InstructionPromptRepairer:
    def repair(self, *, prompt: str, report: CriticReport) -> str:
        repaired, _trace = self.repair_with_trace(
            prompt=prompt,
            report=report,
        )
        return repaired

    def repair_with_trace(
        self,
        *,
        prompt: str,
        report: CriticReport,
    ) -> tuple[str, PromptRepairTrace]:
        ...
~~~

这样：

- 旧调用方继续调用 repair() 并得到 str；
- PromptRepairExecutor 可以选择 repair_with_trace() 获得审计信息；
- 不需要新增文件或破坏现有 Protocol。

#### 2.7.5 约束块格式

统一使用一个可识别的尾部块：

~~~text
<原始或当前 Prompt>

Physical behavior requirements:
- <constraint 1>
- <constraint 2>
- <constraint 3>

Preserve the original objects, scene, camera view and intended action.
~~~

定义常量：

~~~python
_REPAIR_BLOCK_HEADER = "\n\nPhysical behavior requirements:\n"
_PRESERVE_SENTENCE = (
    "\n\nPreserve the original objects, scene, camera view "
    "and intended action."
)
~~~

多轮修复时，先从第一个 _REPAIR_BLOCK_HEADER 截断旧块：

~~~python
def _base_prompt(prompt: str) -> str:
    text = str(prompt).strip()
    if _REPAIR_BLOCK_HEADER in text:
        text = text.split(_REPAIR_BLOCK_HEADER, 1)[0].rstrip()
    return text
~~~

必须保证：

- base prompt 原文不被重新概括；
- 只替换由本 repairer 生成的尾部物理约束；
- 不删除用户原有的普通段落；
- 最终 repaired prompt 始终以完整 base prompt 开头。

#### 2.7.6 Violation 数据提取

对 report.violations 按报告顺序读取：

~~~text
object
category
reason
repair_instruction
start_frame
peak_frame
end_frame
critical_frames
evidence
~~~

内部建立候选结构：

~~~python
@dataclass(frozen=True)
class _ConstraintCandidate:
    object_name: str
    category: str
    reason: str
    instruction: str
    event_phase: str
    source_index: int
~~~

数据清理：

1. object/category/reason/instruction 全部 strip；
2. category 转成小写并统一连字符/空格；
3. 无 object 时使用 the target object；
4. critical_frames 只用于审计，不直接进入生成 prompt；
5. 非字符串 evidence 不进入文本；
6. 不把 Policy、checkpoint、score、frame index 等内部术语写入 Wan prompt。

#### 2.7.7 Category fallback

优先使用 violation.repair_instruction。为空、过短或过于抽象时，使用确定性 category fallback。

建议在同一个 repairer.py 中定义：

~~~python
_CATEGORY_FALLBACK = {
    "gravity": (
        "{object} must move continuously under gravity "
        "without unexplained hovering or upward drift."
    ),
    "gravity_violation": (
        "{object} must move continuously under gravity "
        "without unexplained hovering or upward drift."
    ),
    "collision": (
        "{object} may change direction or speed only after "
        "visible physical contact."
    ),
    "collision_violation": (
        "{object} may change direction or speed only after "
        "visible physical contact."
    ),
    "penetration": (
        "{object} must keep a visible boundary and must not "
        "pass through another object or surface."
    ),
    "surface_penetration": (
        "{object} must keep a visible boundary and must not "
        "pass through another object or surface."
    ),
    "trajectory": (
        "{object} must follow a continuous trajectory without "
        "teleportation or abrupt position jumps."
    ),
    "trajectory_violation": (
        "{object} must follow a continuous trajectory without "
        "teleportation or abrupt position jumps."
    ),
    "contact": (
        "{object} must maintain a plausible contact relationship "
        "before, during and immediately after contact."
    ),
    "contact_violation": (
        "{object} must maintain a plausible contact relationship "
        "before, during and immediately after contact."
    ),
    "disappearance": (
        "{object} must remain visible and temporally consistent "
        "throughout the intended event."
    ),
    "object_disappearance": (
        "{object} must remain visible and temporally consistent "
        "throughout the intended event."
    ),
    "support": (
        "{object} must remain stable while visibly supported and "
        "move only after losing that support."
    ),
    "friction": (
        "{object} must slow down plausibly while maintaining "
        "contact with the surface."
    ),
}
~~~

未知 category 的处理顺序：

~~~text
有效 repair_instruction
  -> 使用 instruction

没有 instruction，但 reason 能安全转成约束
  -> 使用受控的通用连续性约束

两者都不可用
  -> 跳过该 violation，并写 skipped_constraints
~~~

禁止使用空泛文本：

~~~text
Make the video better.
Execute the best repair action.
Keep everything physical.
~~~

#### 2.7.8 将帧区间转成事件阶段

帧号保存在 trace 中，但 prompt 使用事件阶段语言。

确定性映射：

| Category | Prompt 中的事件阶段 |
|---|---|
| gravity | after leaving support / while falling |
| collision | at visible impact / immediately after impact |
| penetration | during contact with the surface or object |
| trajectory | throughout the motion |
| contact | before, during and after contact |
| disappearance | throughout the intended event |
| support | while supported / after losing support |
| friction | while moving along the contact surface |

如果 repair_instruction 已经包含清晰阶段，不重复添加。

不生成：

~~~text
Fix frame 18 to frame 34.
~~~

生成：

~~~text
While falling, the glass must move continuously under gravity.
~~~

#### 2.7.9 Instruction 具体化

对已有 repair_instruction 做最小具体化，不使用新的 LLM：

原始 instruction：

~~~text
Keep position changes continuous across adjacent frames.
~~~

结合 violation.object=glass、category=trajectory 后变为：

~~~text
Throughout the motion, the glass must follow a continuous trajectory
without teleportation or abrupt position jumps.
~~~

处理规则：

1. 如果 instruction 已明确包含 object 和行为，保留原句；
2. 如果缺 object，在句首加入 object；
3. 如果是相邻帧、frame index 等内部表达，转换为 throughout the motion 等自然语言；
4. 保持英文生成 prompt，技术审计字段可保留原始 instruction；
5. 不改变原始 scene、camera 和 intended action。

#### 2.7.10 去重、优先级和数量上限

使用标准化 key 去重：

~~~python
def _constraint_key(text: str) -> str:
    return " ".join(
        text.lower()
        .replace(".", " ")
        .replace(",", " ")
        .split()
    )
~~~

优先级：

1. CriticReport 中出现顺序；
2. 有明确 repair_instruction 的项优先于 fallback；
3. 主体对象相关项优先于无对象项；
4. 相同 object/category 只保留表达最具体的一条；
5. 最多保留 max_constraints 条。

不要按不可追溯的随机顺序选择约束。

#### 2.7.11 原始语义保护

由于不重新总结 base prompt，原始语义保护首先由“原文完整保留”保证。

额外检查：

~~~python
assert repaired_prompt.startswith(base_prompt)
~~~

新增文本只能使用：

- violation.object；
- category fallback 中的物理关系词；
- repair_instruction 已有概念；
- 通用时序和连续性词。

默认不允许增加：

- 新主体；
- 新工具；
- 新支撑装置；
- 新场景；
- 新摄像机运动；
- 新的核心事件。

例如原始 Prompt 明确要求无支撑漂浮时，repairer 不能擅自加入 spring、rope、magnet 或 hidden force。

#### 2.7.12 原始 Prompt 与物理修复直接冲突

本方案不做生成前检查，只处理 CriticReport 已经发现的视频问题。

如果原始 Prompt 明确要求：

~~~text
without any force or support
defies gravity
passes through the wall
teleports instantly
disappears completely
~~~

而 Critic 又要求修正相同物理行为，则不能通过创造新机制改变用户语义。

在 repairer.py 中定义同文件异常：

~~~python
class PromptPhysicsConflictError(ValueError):
    pass
~~~

只在“Prompt 明确要求某种违反物理的行为，且当前 violation 正在否定该行为”时抛出。不能因为普通的 falls、bounces、flies 等词误判冲突。

PromptRepairExecutor 捕获后生成明确失败结果：

~~~text
status = failed
failure_reason = prompt_physics_conflict
candidate = None
~~~

runner 应把这种无安全改写的情况收口到历史最佳候选，并记录 Reject 原因，不能继续用原 prompt 重生成。

#### 2.7.13 无安全变化时禁止伪执行

以下情况 repaired prompt 可能等于 input prompt：

- violations 为空；
- 所有 repair_instruction 为空且 category 无 fallback；
- 所有约束被去重或安全检查过滤；
- 原始 Prompt 与修复要求冲突。

如果配置 fail_on_no_safe_change=true：

~~~python
if repaired_prompt.strip() == request.prompt.strip():
    return ExecutionResult(
        action=RepairAction.PROMPT_REPAIR,
        status="failed",
        backend_id=self.backend_id,
        failure_reason="prompt_repair_no_safe_change",
        metadata={
            "prompt_changed": False,
        },
    )
~~~

不得调用 WanSubprocessGenerator 生成一条相同 prompt、不同 seed 的视频；该行为不属于 Prompt Repair，并已从新三动作架构删除。

#### 2.7.14 长度控制

最终长度不能通过直接截断句子处理。

执行顺序：

1. 完整保留 base prompt；
2. 依次尝试加入约束；
3. 每加入一条后估算总长度；
4. 超过 max_prompt_chars 时停止加入后续约束；
5. Preserve 句保留；
6. 被跳过约束写入 trace。

如果 base prompt 本身已经超过上限，不截断用户原文；只限制新增修复块，并在 trace 中写 base_prompt_over_limit。

#### 2.7.15 repair_with_trace 伪代码

~~~python
def repair_with_trace(self, *, prompt, report):
    input_prompt = str(prompt).strip()
    base_prompt = self._strip_existing_repair_block(input_prompt)

    candidates = self._collect_candidates(report.violations)
    selected = []
    skipped = []
    seen = set()

    for item in candidates:
        if self._conflicts_with_explicit_prompt(base_prompt, item):
            raise PromptPhysicsConflictError(
                f"explicit prompt conflicts with {item.category}"
            )

        constraint = self._render_constraint(item)
        if not constraint:
            skipped.append(f"{item.category}:empty")
            continue

        key = self._constraint_key(constraint)
        if key in seen:
            skipped.append(f"{item.category}:duplicate")
            continue

        proposed = selected + [constraint]
        rendered = self._render_block(base_prompt, proposed)
        if len(rendered) > self.max_prompt_chars:
            skipped.append(f"{item.category}:length_limit")
            continue

        seen.add(key)
        selected.append(constraint)

        if len(selected) >= self.max_constraints:
            break

    repaired = (
        self._render_block(base_prompt, selected)
        if selected
        else input_prompt
    )

    trace = PromptRepairTrace(
        original_prompt=base_prompt,
        input_prompt=input_prompt,
        repaired_prompt=repaired,
        changed=repaired != input_prompt,
        target_objects=...,
        violation_categories=...,
        constraints_used=tuple(selected),
        skipped_constraints=tuple(skipped),
        instruction_source="critic_report.violations",
    )
    return repaired, trace
~~~

#### 2.7.16 PromptRepairExecutor 的兼容扩展

在原版 src/physgenloop/learning_repair/executors.py 中保持原 action 和 generate 调用，只增强可选 trace：

~~~python
if hasattr(self.prompt_rewriter, "repair_with_trace"):
    rewritten, rewrite_trace = self.prompt_rewriter.repair_with_trace(
        prompt=request.prompt,
        report=request.critic_report,
    )
else:
    rewritten = self.prompt_rewriter.repair(
        prompt=request.prompt,
        report=request.critic_report,
    )
    rewrite_trace = None
~~~

生成仍然使用：

~~~python
candidate = self.generator.generate(
    prompt=rewritten,
    seed=request.seed,
)
~~~

PromptRepairExecutor、WanSubprocessGenerator 和 ExecutionRequest 均不再声明或透传 physics_plan。rewritten 必须与 request.prompt 不同；否则返回显式 no_safe_prompt_change 并由 Guard/Runner 收口到 Reject，不能用相同 prompt 更换 seed 伪装成 Prompt Repair。

ExecutionResult 增加审计 metadata：

~~~python
metadata = {
    "prompt_changed": rewritten != request.prompt,
    "input_prompt": request.prompt,
    "repaired_prompt": rewritten,
    "instruction_source": "critic_report.violations",
    "target_objects": list(rewrite_trace.target_objects),
    "violation_categories": list(
        rewrite_trace.violation_categories
    ),
    "constraints_used": list(rewrite_trace.constraints_used),
    "skipped_constraints": list(
        rewrite_trace.skipped_constraints
    ),
}
~~~

注意：此处只调用 rewriter，不调用 Repair Policy。

#### 2.7.17 build_backends.py 的具体注册修改

恢复 import：

~~~python
from physgenloop.learning_repair.executors import PromptRepairExecutor
from physgenloop.repairer import InstructionPromptRepairer
~~~

读取配置：

~~~python
prompt_cfg = cfg.get("prompt_repair", {}) or {}
prompt_backend = str(
    prompt_cfg.get("backend", "legacy")
).strip().lower()
~~~

实例化：

~~~python
if prompt_backend == "legacy":
    prompt_rewriter = InstructionPromptRepairer(
        max_constraints=int(
            prompt_cfg.get("max_constraints", 3)
        ),
        max_prompt_chars=int(
            prompt_cfg.get("max_prompt_chars", 900)
        ),
        replace_existing_block=bool(
            prompt_cfg.get("replace_existing_block", True)
        ),
        allow_new_entities=bool(
            prompt_cfg.get("allow_new_entities", False)
        ),
    )
    prompt_repair_executor = PromptRepairExecutor(
        prompt_rewriter=prompt_rewriter,
        generator=generator,
        backend_id=str(
            prompt_cfg.get(
                "backend_id",
                "legacy-prompt-rewriter+video-generator",
            )
        ),
    )
else:
    raise ValueError(
        f"unsupported production prompt_repair.backend: "
        f"{prompt_backend!r}"
    )
~~~

注册到已有 registry：

~~~python
registry = ExecutorRegistry(
    executors=[
        prompt_repair_executor,
        MaskSequenceLocalEditingExecutor(editor=editor),
        AuditedRejectExecutor(
            selector=EvidenceAwareSelector()
        ),
    ]
)
~~~

不得同时注册两个 Prompt Repair executor；registry 的动作集合必须严格等于 prompt_repair、local_editing、reject。

#### 2.7.18 修改前后示例

原始 Prompt：

~~~text
A clear glass falls from the edge of a wooden table, hits the floor,
and bounces once. The camera remains fixed in a side view.
~~~

Critic violations：

~~~text
glass / gravity:
The glass briefly hovers after leaving the table.

glass / collision:
The glass changes direction before visible floor contact.

glass / trajectory:
The rebound height is larger than the original falling height.
~~~

repaired prompt：

~~~text
A clear glass falls from the edge of a wooden table, hits the floor,
and bounces once. The camera remains fixed in a side view.

Physical behavior requirements:
- After leaving the table, the glass must accelerate downward
  continuously under gravity without hovering or abrupt trajectory changes.
- The glass must visibly contact the floor before changing direction.
- After impact, the glass rebounds only once to a lower height,
  showing a plausible loss of energy.

Preserve the original objects, scene, camera view and intended action.
~~~

WanSubprocessGenerator 必须把该 repaired prompt 原样写入 after candidate 的 prompt.txt 和 GeneratedCandidate.prompt。

#### 2.7.19 多轮 Prompt Repair

第一轮约束：

~~~text
Physical behavior requirements:
- The glass must fall continuously under gravity without hovering.
~~~

Re-Critic 后发现 floor penetration，第二轮应替换旧块：

~~~text
Physical behavior requirements:
- The glass must fall continuously under gravity.
- The glass must visibly contact and remain above the floor surface
  without penetration.
~~~

不能变成：

~~~text
Physics correction: ...
Physics correction: ...
Physical behavior requirements: ...
~~~

每轮 repair trace 都要保存 input_prompt 和 repaired_prompt，才能证明多轮文本如何变化。

#### 2.7.20 单元测试

把测试补入现有 tests/wanphysics_v2/test_decision_only_executors.py，不新增重复测试文件。

必须覆盖：

1. 有 repair_instruction 时 prompt 被修改；
2. backend_id 为 legacy-prompt-rewriter+video-generator；
3. ExecutionResult.next_prompt 等于 repaired prompt；
4. Wan generator 收到 repaired prompt；
5. rewriter 不调用 Policy；
6. 相同 instruction 去重；
7. 空 instruction 使用 category fallback；
8. frame index 不进入 repaired prompt；
9. 原始 prompt 完整保留；
10. 第二轮替换旧 repair block；
11. 最多三条约束；
12. 长度超限时按整条约束跳过，不截断句子；
13. 无安全变化时不调用 generator；
14. 明确物理冲突时返回 prompt_physics_conflict；
15. 不允许引入 spring、rope、magnet 等无证据新对象。

核心断言：

~~~python
assert result.status == "succeeded"
assert result.backend_id == (
    "legacy-prompt-rewriter+video-generator"
)
assert result.next_prompt != request.prompt
assert result.next_prompt.startswith(request.prompt)
assert "Physical behavior requirements:" in result.next_prompt
assert generator.last_prompt == result.next_prompt
assert policy_call_count == 1
~~~

无安全变化：

~~~python
assert result.status == "failed"
assert result.failure_reason == "prompt_repair_no_safe_change"
assert generator.calls == 0
~~~

#### 2.7.21 build wiring 测试

在 tests/wanphysics_v2/test_build_backends.py 中断言：

~~~python
executor = runner.executor_registry._executors[
    RepairAction.PROMPT_REPAIR
]

assert executor.__class__.__name__ == "PromptRepairExecutor"
assert (
    executor.prompt_rewriter.__class__.__name__
    == "InstructionPromptRepairer"
)
assert executor.backend_id == (
    "legacy-prompt-rewriter+video-generator"
)
~~~

并确认 DecisionPromptRepairExecutor 不是默认 registry 中的 Prompt Repair executor。

#### 2.7.22 真实 smoke 验收

选择一条 Critic 能产生非空 `repair_instruction`，并且 Three-Action Policy 会自然选择 `prompt_repair` 的 violation 样本。禁止使用 CLI、配置或 manifest 覆盖动作；若 Policy 没有自然选择预期动作，smoke 应失败并暴露 Policy/capability 对齐问题。

命令：

命令统一使用第 16.2 节的固定 `RUN_ROOT + nohup` 形式，只调用 `run_videophy2_loop_v2.py`，且不得包含 `--allow-proxy-policy` 或 `--force-action`。Prompt Repair 专项 smoke 使用独立的最终目录，例如：

~~~text
/root/PhysGenLoop-/outputs/v2_smoke_natural_prompt_repair
~~~

需要核对：

~~~text
before candidate/prompt.txt
after candidate/prompt.txt
repair_decision.json
repair_trace.jsonl
loop_result.json
trials.jsonl
before critic_report.json
after critic_report.json
~~~

通过条件：

- before 和 after prompt 确实不同；
- after prompt 包含具体对象和物理约束；
- backend_id 是 legacy-prompt-rewriter+video-generator；
- Policy 没有在 Executor 中再次执行；
- after video 真实存在且可解码；
- after candidate 被 Re-Critic；
- Trial 保存真实 before/after prompt 和分数；
- 如果 physics 没有提升，Trial 必须标记 unsuccessful；即使出现部分改善，只要 Strict Re-Gate 未 `ACCEPTED`，也必须 `successful=false`；
- semantic/quality 下降必须保留，不能只看 physics gain。

#### 2.7.23 实施完成判据

只有同时满足以下条件，才能认定旧版 Prompt Repair 对齐完成：

- 默认 registry 使用 PromptRepairExecutor；
- rewriter 是 InstructionPromptRepairer；
- Wan 使用 repaired prompt；
- repair_instruction 为空时有确定性 fallback；
- 约束具体到 object/category/event phase；
- 不把帧号和内部术语写入生成 prompt；
- 原始对象、场景、动作和镜头原文保留；
- 多轮约束块可替换，不无限增长；
- 无安全变化时不伪装成 Prompt Repair；
- 明确物理冲突不引入新机制；
- prompt 修改过程可审计；
- 单元测试、build wiring 测试和一条真实 smoke 通过。

### 2.8 Local Editing

~~~text
SAM2 materialize_masks()
  -> sam2_masks/{object}_{frame:05d}.png
  -> build_manifest()
  -> verify_manifest(check_sha=True)
  -> build_local_edit_target()
  -> StrictProPainterLocalEditor
  -> ProPainter
~~~

Strict editor 已拒绝：

- 缺失 mask
- 空 mask
- 近全帧 mask
- 尺寸不一致
- SHA 不一致
- frame 越界
- object 不匹配

ProPainter 当前是 inpainting 后端，repair instruction 并不作为 ProPainter 条件输入。因此必须用 Re-Critic 证明其物理改善，不能仅以命令返回 0 认定修复成功。

### 2.9 Artifact 与 Trial

现有产物边界：

| 产物 | 当前职责 |
|---|---|
| run_manifest.json | run 配置和 preflight |
| sample_status.json | 当前样本状态 |
| sample_status_history.jsonl | 状态历史 |
| critic_report.json | 完整 Critic 文档 |
| mask_manifest.json | mask 校验清单 |
| repair_decision.json | Policy 与 Guard |
| repair_trace.jsonl | 动作执行轨迹 |
| resource_metrics.jsonl | GPU/耗时 |
| trials.jsonl | WanRepairTrialV2 |
| loop_result.json | 样本最终结果 |
| summary.json | run 汇总 |

当前 _assemble_trials() 会重新构造 RepairDecision，并把 gate 当成 critic_before/critic_after，真实 Policy 概率、instruction、local target、candidate path 和完整报告会丢失，必须修正。

#### 2.9.1 Local Editing 的 ProPainter 闭环必须显式进入 Trial

这里需要进一步明确：Local Editing 不是一个抽象的“局部编辑成功”状态，它在当前服务器架构中的真实执行后端就是 ProPainter。SAM2 负责提供对象在各帧上的 mask，ProPainter 消费原视频和逐帧 mask 生成修复候选，随后必须由同一套 Critic 和 Acceptance Gate 重新评价。正确因果链应固定为：

~~~text
RepairDecision(action=local_editing)
  -> LocalEditTarget
     - parent_candidate_id
     - objects
     - start_frame / end_frame
     - critical_frames
     - mask_uri = mask_manifest.json
  -> MaskSequenceLocalEditingExecutor
  -> StrictProPainterLocalEditor
  -> ProPainterLocalEditor
  -> /root/PhysGenLoop-/models/ProPainter/inference_propainter.py
  -> propainter-* 修复候选视频
  -> Re-Critic
  -> Re-Gate
  -> WanRepairTrialV3
~~~

因此，上一段所述 _assemble_trials() 修正不能只保存通用 action 和 physics 分数，还必须证明以下事实：

1. Policy 真实选择的动作是 local_editing，且 PolicyGuard 明确返回 allowed；
2. local target 确实来自本轮 before candidate，而不是其他样本或上一轮残留；
3. target 指向的 mask_manifest.json 已通过严格校验；
4. 真正被调用的是 StrictProPainterLocalEditor 和 ProPainter，而不是 mock、占位编辑器或全帧 fallback；
5. ProPainter 的输入源视频、逐帧 mask、脚本、权重目录和输出视频均可追溯；
6. 输出视频真实存在、可解码，并且被作为 after candidate 送回 Critic；
7. `repair_improved` 由 Re-Critic 的物理改善及双语义、画质约束决定；`successful` 还必须要求 Strict Re-Gate ACCEPTED，不能由 subprocess returncode == 0 单独决定；
8. ProPainter 调用失败、输出损坏或 Re-Critic 失败时，也必须生成 unsuccessful Trial，不能被静默丢弃。

#### 2.9.2 只扩展现有实现，不建立第二套 Local Editing

本修正应在现有文件内完成，不再新增另一套 editor、executor、runner 或 trial assembler：

~~~text
generators/wanphysics/v2/propainter_strict_editor.py
generators/wanphysics/local_editor.py
generators/wanphysics/v2/executors.py
generators/wanphysics/v2/runner.py
agents/wanphysics/run_videophy2_loop_v2.py
generators/wanphysics/v2/trials.py
schemas/wan_repair_trial_v3.schema.json
tests/wanphysics_v2/test_propainter_strict_editor.py
tests/wanphysics_v2/test_decision_only_executors.py
tests/wanphysics_v2/test_trials.py
~~~

职责保持单一：

| 现有组件 | 修改后的唯一职责 |
|---|---|
| StrictProPainterLocalEditor | 校验严格 mask manifest，调用原版 ProPainterLocalEditor，并补齐 V2 审计元数据 |
| ProPainterLocalEditor | 执行帧提取、ProPainter 子进程、视频编码和输出验证 |
| MaskSequenceLocalEditingExecutor | 把 RepairDecision 与 LocalEditTarget 传入 editor，将结果封装为 ExecutionResult |
| RoundRecord | 保存一次 before -> decision -> execution -> after 的完整因果快照 |
| _assemble_trials() | 只从 RoundRecord 还原真实 Trial，不重新猜测或伪造 Policy、Critic、候选路径 |
| WanRepairTrialV3 | 表达可训练、可审计的 before/action/after、双 Semantic 与 Strict Re-Gate 记录 |

不要把 ProPainter 调用逻辑复制进 runner 或 _assemble_trials()。runner 只编排，executor 只执行动作，editor 才持有后端细节，Trial 只记录事实。

#### 2.9.3 配置和 preflight 必须锁定真实 ProPainter

configs/loop_v2.yaml 应继续使用现有配置入口：

~~~yaml
local_editing:
  enabled: true
  require_mask: true
  allow_full_frame_fallback: false
  strict_manifest: true
  propainter_repo: models/ProPainter
  propainter_script: models/ProPainter/inference_propainter.py
  propainter_weights: models/ProPainter/weights
  python: envs/main/bin/python
~~~

解析后的绝对路径应为：

~~~text
/root/PhysGenLoop-/models/ProPainter
/root/PhysGenLoop-/models/ProPainter/inference_propainter.py
/root/PhysGenLoop-/models/ProPainter/weights
/root/PhysGenLoop-/envs/main/bin/python
~~~

build_backends.py 继续负责 preflight 和实例注入。当且仅当以下条件全部满足时，capability.local_editing 才能为 true：

- local_editing.enabled == true；
- strict_manifest == true；
- ProPainter repo 存在；
- inference_propainter.py 存在；
- weights 目录存在且包含实际权重文件；
- Python 解释器存在且可执行；
- SAM2 mask materialization 可用；
- allow_full_frame_fallback == false。

任何一项不满足，都应让 capability mask 禁用 local_editing，并在 run_manifest/preflight 中记录具体原因，不能悄悄换成 mock editor、单张 mask 复制或全帧白色 mask。

#### 2.9.4 StrictProPainterLocalEditor 要补齐的审计元数据

当前 StrictProPainterLocalEditor 只补写：

~~~text
backend = propainter-strict-local-edit
mask_manifest_uri
strict_mask_manifest = true
~~~

这不足以证明实际调用环境。应在现有 edit() 返回的 GeneratedCandidate.metadata 中增加一个结构化 propainter 块，至少包含：

~~~python
metadata["editor"] = "StrictProPainterLocalEditor"
metadata["editor_backend"] = "ProPainter"
metadata["repair_mode"] = "strict-mask-video-inpainting"
metadata["source_candidate_id"] = candidate.candidate_id
metadata["source_video"] = str(candidate.video_path)
metadata["output_video"] = str(edited.video_path)
metadata["mask_manifest_uri"] = target.mask_uri
metadata["target_objects"] = list(target.objects)
metadata["critical_frames"] = list(target.critical_frames)
metadata["propainter"] = {
    "repo": str(self._repo.resolve()),
    "script": str((self._repo / "inference_propainter.py").resolve()),
    "weights_dir": str((self._repo / "weights").resolve()),
    "repo_revision": repo_revision,
    "python": self._python,
    "fp16": True,
    "returncode": run_info["returncode"],
    "elapsed_seconds": run_info["elapsed_seconds"],
    "output_validation": validation,
}
~~~

权重 provenance 至少保存实际扫描到的文件名和文件大小；如果 preflight 已计算 SHA256，则直接复用其结果，不要每轮重复哈希大权重。建议结构为：

~~~json
{
  "weights": [
    {
      "name": "实际文件名",
      "path": "/root/PhysGenLoop-/models/ProPainter/weights/实际文件名",
      "size_bytes": 123,
      "sha256": "由 preflight 提供，无法获得时明确为 null"
    }
  ]
}
~~~

repo_revision 应取 models/ProPainter 仓库的 git commit SHA。若确实无法解析，可以保留 unknown，但必须同时记录 repo 路径和解析失败原因，不能只留下无上下文的 unknown。

metadata.json 写入失败不能再简单 pass 后完全无痕。候选视频可以继续返回，但应把 metadata_write_status 和 metadata_write_error 保存在 ExecutionResult.metadata；否则会出现“视频存在但审计证据缺失”的假完整状态。

#### 2.9.5 ProPainter 子进程必须返回结构化运行结果

generators/wanphysics/local_editor.py 中当前 _run_propainter() 返回 None，并依靠 check=True 抛异常。应在同一文件内把它改为返回结构化信息：

~~~python
{
    "returncode": int,
    "elapsed_seconds": float,
    "script": str,
    "repo": str,
    "python": str,
    "fp16": True,
    "stdout_tail": str,
    "stderr_tail": str,
}
~~~

实施逻辑：

1. 在调用前记录 perf_counter；
2. subprocess.run 使用明确的 argv 列表，不使用 shell=True；
3. 捕获 stdout/stderr，并只保存有长度上限的末尾文本，避免 Trial 膨胀；
4. 得到 CompletedProcess 后先构造 run_info；
5. returncode 非 0 时，抛出带 run_info 的异常，由 executor 转换为 failed ExecutionResult；
6. returncode 为 0 时返回 run_info，供候选 metadata 和 Trial 使用。

建议保持命令等价于：

~~~text
/root/PhysGenLoop-/envs/main/bin/python
  /root/PhysGenLoop-/models/ProPainter/inference_propainter.py
  --video <frames_dir>
  --mask <masks_dir>
  --output <result_dir>
  --fp16
~~~

不要把 repair instruction 当作 ProPainter 的文本条件。ProPainter 在这条链路中消费的是视频帧和逐帧 mask；instruction 用于审计“为什么编辑、编辑哪个对象和哪个事件阶段”，而编辑是否改善物理规律必须由 Re-Critic 证明。

#### 2.9.6 输出视频验证不能只看 returncode

ProPainter 返回 0 后，在构造 GeneratedCandidate 前必须验证输出。建议在现有 ProPainterLocalEditor 内增加私有验证函数，不新增独立模块：

~~~python
validation = {
    "exists": bool,
    "decode_ok": bool,
    "frame_count": int,
    "width": int,
    "height": int,
    "fps": float,
    "source_frame_count": int,
    "source_width": int,
    "source_height": int,
    "frame_count_match": bool,
    "size_match": bool,
    "candidate_prefix_ok": bool,
}
~~~

硬性失败条件：

- 输出文件不存在或大小为 0；
- OpenCV 无法打开；
- 首帧或任何必要检查帧无法解码；
- frame_count <= 0；
- 输出帧数与源帧数不一致；
- 输出宽高与源视频不一致；
- candidate_id 不以 propainter- 开头；
- 编码器未成功打开或编码后文件仍为空。

如后续确认某些编码器会造成帧数元数据误差，可在配置中增加极小且明确的 tolerance；默认应要求帧数一致，不能用“看起来合理”替代可复现规则。

成功候选必须满足：

~~~text
candidate_id = propainter-*
candidate.video_path = 实际绝对路径
candidate.prompt = before prompt
candidate.metadata.editor = StrictProPainterLocalEditor
candidate.metadata.editor_backend = ProPainter
candidate.metadata.output_validation.decode_ok = true
candidate.metadata.output_validation.frame_count_match = true
candidate.metadata.output_validation.size_match = true
~~~

#### 2.9.7 MaskSequenceLocalEditingExecutor 的成功与失败返回

成功时，现有 ExecutionResult 应明确保存：

~~~python
ExecutionResult(
    action=RepairAction.LOCAL_EDITING,
    status="succeeded",
    backend_id="v2-mask-sequence-local-editor",
    candidate=candidate,
    next_prompt=request.prompt,
    artifacts={
        "source_video": str(request.candidate.video_path),
        "repaired_video": str(candidate.video_path),
        "mask_manifest": str(target.mask_uri),
    },
    metadata={
        "executor": "MaskSequenceLocalEditingExecutor",
        "editor": "StrictProPainterLocalEditor",
        "editor_backend": "ProPainter",
        "repair_mode": "strict-mask-video-inpainting",
        "local_target": target.to_dict(),
        "target_objects": list(target.objects),
        "critical_frames": list(target.critical_frames),
        "propainter": candidate.metadata["propainter"],
        "output_validation": candidate.metadata["propainter"]["output_validation"],
    },
)
~~~

失败时不能让异常越过 runner，导致本轮没有 execution 记录。MaskSequenceLocalEditingExecutor.execute() 应捕获 ProPainter 调用、编码和验证异常，并返回：

~~~python
ExecutionResult(
    action=RepairAction.LOCAL_EDITING,
    status="failed",
    backend_id="v2-mask-sequence-local-editor",
    candidate=None,
    failure_reason="propainter_subprocess_failed",
    artifacts={
        "source_video": str(request.candidate.video_path),
        "mask_manifest": str(target.mask_uri),
    },
    metadata={
        "executor": "MaskSequenceLocalEditingExecutor",
        "editor": "StrictProPainterLocalEditor",
        "editor_backend": "ProPainter",
        "repair_mode": "strict-mask-video-inpainting",
        "local_target": target.to_dict(),
        "propainter": run_info,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    },
)
~~~

failure_reason 应使用有限、稳定的枚举，详细异常放 metadata：

~~~text
local_target_missing_or_invalid
mask_manifest_missing
mask_manifest_invalid
propainter_preflight_failed
propainter_subprocess_failed
propainter_no_output_frames
propainter_encode_failed
propainter_output_invalid
~~~

这样才能按原因统计失败率，避免把含路径和动态文本的异常直接当分类标签。

#### 2.9.8 RoundRecord 必须保存真实因果数据

当前 RoundRecord 只有 gate、execution、candidate id 和 physics 分数，无法恢复完整 Trial。应在 generators/wanphysics/v2/runner.py 的现有 RoundRecord 中增加：

~~~python
decision: dict[str, Any] | None = None
critic_before: dict[str, Any] | None = None
critic_after: dict[str, Any] | None = None
before_candidate: dict[str, Any] | None = None
after_candidate: dict[str, Any] | None = None
before_scores: dict[str, Any] | None = None
after_scores: dict[str, Any] | None = None
local_editing_backend: str | None = None
~~~

gate 和 after_gate 继续保留，但只表达 Acceptance Gate，不再冒充 CriticReport。

本轮 Critic 完成并创建 rec 时立即写入：

~~~python
rec.before_candidate = CandidateRecord.from_candidate(current_candidate).to_dict()
rec.critic_before = report.to_dict()
rec.before_scores = {
    "physics": phys,
    "semantic": side.get("semantic_score"),
    "original_prompt_semantic": side.get(
        "original_prompt_semantic_score"
    ),
    "quality": side.get("quality_score"),
}
~~~

Policy 决策和 Guard 校验完成后，分别保存真实 Decision 与 GuardResult；Guard 不修改 Decision。Executor 消费的仍是原始 Policy Decision，而不是另造 final_action：

~~~python
rec.decision = policy_decision.to_dict()
rec.guard = guard_result.to_dict()
~~~

Local Editing executor 返回后写入：

~~~python
rec.execution = exec_result.to_dict()
rec.local_editing_backend = exec_result.metadata.get("editor_backend")
~~~

Re-Critic 完成后写入：

~~~python
rec.after_candidate = CandidateRecord.from_candidate(after_candidate).to_dict()
rec.critic_after = after_report.to_dict()
rec.after_scores = {
    "physics": after_phys,
    "semantic": after_side.get("semantic_score"),
    "original_prompt_semantic": after_side.get(
        "original_prompt_semantic_score"
    ),
    "quality": after_side.get("quality_score"),
}
~~~

序列化 CriticReport 时应统一使用现有 to_dict()；如果对象不提供该方法，可以使用一个位于 runner.py 内的窄适配函数转换 Mapping。不要用 gate.to_dict() 代替 report，也不要把对象 repr 写入 JSON。

对于 local target precheck、ProPainter subprocess 或输出验证失败，before_candidate、critic_before、before_scores、decision 和 execution 仍然必须存在；after_candidate、critic_after、after_scores 可以为 null。这正是 unsuccessful Trial 所需的完整失败证据。

#### 2.9.9 _assemble_trials() 的具体修正

agents/wanphysics/run_videophy2_loop_v2.py 中 _assemble_trials() 应从“重新构造”改成“验证并搬运”。核心原则：

- decision 直接使用 RepairDecision.from_dict(rnd.decision)；
- source_candidate 直接使用 CandidateRecord.from_dict(rnd.before_candidate)；
- prompt 从 source_candidate.prompt 读取；
- critic_before 直接使用 rnd.critic_before；
- critic_after 直接使用 rnd.critic_after；
- before_scores/after_scores 使用 ScoreBundle.from_dict()；
- execution 原样保存真实 ExecutionResult.to_dict()；
- before gate 和 after gate 放在 Trial.metadata.gates，不能写入 critic_before/critic_after；
- 不再生成 one-hot 假概率；
- 不再写死 confidence=0.5、instruction=""、source="runner_round_record"；
- 不再拼接假的 candidate_id-v01.mp4 路径；
- 删除 except Exception: pass 这种静默丢 Trial 的逻辑。

推荐的组装骨架为：

~~~python
decision = RepairDecision.from_dict(rnd.decision)
source = CandidateRecord.from_dict(rnd.before_candidate)
before_scores = ScoreBundle.from_dict(rnd.before_scores)
after_scores = (
    None
    if rnd.after_scores is None
    else ScoreBundle.from_dict(rnd.after_scores)
)

execution = dict(rnd.execution)
execution_ok = execution.get("status") == "succeeded"
has_recritic = rnd.critic_after is not None and after_scores is not None
physics_improved = (
    has_recritic
    and after_scores.physics > before_scores.physics
)
semantic_ok = score_drop_within_limit(
    before_scores.semantic,
    after_scores.semantic if after_scores else None,
    semantic_drop_limit,
)
original_prompt_semantic_ok = score_drop_within_limit(
    before_scores.original_prompt_semantic,
    (
        after_scores.original_prompt_semantic
        if after_scores
        else None
    ),
    original_prompt_semantic_drop_limit,
)
quality_ok = score_drop_within_limit(
    before_scores.quality,
    after_scores.quality if after_scores else None,
    quality_drop_limit,
)
repair_improved = (
    execution_ok
    and has_recritic
    and physics_improved
    and semantic_ok
    and original_prompt_semantic_ok
    and quality_ok
)

successful = (
    repair_improved
    and after_gate.status == "ACCEPTED"
)
~~~

`repair_improved`、`successful` 与 `after_gate.status` 必须区分：

- `repair_improved=true` 表示动作产生了可验证的部分收益；
- `successful=true` 只表示候选已经通过 Strict Re-Gate，可以作为严格成功 Trial；
- physics 改善且副作用可控、但仍未达到最终阈值时，只能 `repair_improved=true, successful=false`，循环可以继续下一轮；
- subprocess 成功但物理未改善时，Trial 必须 unsuccessful；
- 物理改善但语义或画质下降超过阈值时，也必须 unsuccessful；
- 没有 Re-Critic 或 Re-Gate 结果时绝不能 successful。

针对 local_editing，_assemble_trials() 在构造 Trial 前必须额外校验：

~~~text
decision.action == local_editing
decision.local_target != null
decision.local_target.parent_candidate_id == source_candidate.candidate_id
execution.metadata.executor == MaskSequenceLocalEditingExecutor
execution.metadata.editor == StrictProPainterLocalEditor
execution.metadata.editor_backend == ProPainter
execution.metadata.repair_mode == strict-mask-video-inpainting
execution.artifacts.mask_manifest 存在
decision.local_target.mask_uri == execution.artifacts.mask_manifest
若 execution.status == succeeded：
  execution.candidate 存在
  execution.candidate.candidate_id 以 propainter- 开头
  execution.candidate.video_path 存在
  output_validation 全部通过
  critic_after 存在
  after_scores 存在
~~~

校验不通过时：

1. 如果 before、decision、execution 等最小事实仍足以构造 Trial，则生成 unsuccessful Trial，并把 failure_reason 设为 trial_evidence_incomplete，缺失项放 metadata.validation_errors；
2. 如果连真实 decision 或 source candidate 都缺失，不能伪造数据来凑 Trial，应让组装显式失败、将 sample 标记为 TRIAL_ASSEMBLY_FAILED，并在 summary 中增加计数；
3. 无论哪种情况，都不能 continue/pass 后让运行看起来完整。

#### 2.9.10 Trial 内 ProPainter 证据的唯一存放位置

为避免多份字段相互漂移，Trial 使用以下唯一位置：

| 事实 | Trial 中的唯一位置 |
|---|---|
| before candidate 与源视频 | source_candidate |
| 真实 Policy 概率、instruction、local target | decision |
| ProPainter after candidate 与输出路径 | execution.candidate |
| mask manifest | decision.local_target.mask_uri，同时由 execution.artifacts.mask_manifest 做一致性校验 |
| executor/editor/backend | execution.metadata |
| ProPainter repo/script/weights/revision | execution.metadata.propainter |
| 子进程 returncode/耗时/fp16 | execution.metadata.propainter |
| 输出视频验证 | execution.metadata.output_validation |
| 完整 before Critic | critic_before |
| 完整 after Critic | critic_after |
| physics/semantic/original_prompt_semantic/quality before/after | before_scores 与 after_scores |
| before/after Acceptance Gate | metadata.gates |
| 本次动作是否改善/严格成功 | repair_improved、successful 与 failure_reason |

不要再单独增加第二份 after_video、第二份 local_target 或第二份 ProPainter metadata 顶层字段。需要读取时按上述固定路径读取，并在 schema/test 中锁定一致性。

#### 2.9.11 Local Editing Trial 示例

下面展示一个真实字段结构。路径、分数和概率仅为格式示例，运行时必须写实际值：

~~~json
{
  "schema_version": "wan-repair-trial/3.0",
  "trial_id": "sample-0001-exec1",
  "group_id": "sample-0001",
  "generator": {
    "family": "wan",
    "model": "Wan2.2-TI2V-5B",
    "revision": "实际 revision"
  },
  "critic": {
    "profile": "sam2_seeded_rules",
    "revision": "实际 revision"
  },
  "policy_mode": "three_action_heuristic",
  "decision_source": "three_action_policy",
  "source_candidate": {
    "candidate_id": "wan-abc123",
    "video_path": "/root/PhysGenLoop-/outputs/run/sample-0001/wan-abc123-v01.mp4",
    "prompt": "A red ball falls onto a wooden floor.",
    "seed": 7,
    "metadata": {}
  },
  "prompt": "A red ball falls onto a wooden floor.",
  "critic_before": {
    "physics_score": 0.41,
    "violations": [
      {
        "object": "red ball",
        "type": "contact_or_collision",
        "critical_frames": [18, 19, 20, 21]
      }
    ]
  },
  "decision": {
    "schema_version": "learning-repair-decision/2.0",
    "action": "local_editing",
    "confidence": 0.82,
    "instruction": "Repair the red ball contact region around the impact event.",
    "action_probabilities": {
      "prompt_repair": 0.08,
      "local_editing": 0.88,
      "reject": 0.04
    },
    "per_action_values": {
      "prompt_repair": 0.10,
      "local_editing": 0.71,
      "reject": -0.20
    },
    "parameters": {},
    "local_target": {
      "parent_candidate_id": "wan-abc123",
      "objects": ["red ball"],
      "start_frame": 18,
      "end_frame": 21,
      "critical_frames": [18, 19, 20, 21],
      "mask_uri": "/root/PhysGenLoop-/outputs/run/sample-0001/sam2_masks/mask_manifest.json"
    },
    "source": "actual_policy",
    "abstained": false,
    "fallback_reason": null,
    "compatibility_id": "实际 compatibility id"
  },
  "execution": {
    "execution_id": "sample-0001-exec1",
    "action": "local_editing",
    "status": "succeeded",
    "backend_id": "v2-mask-sequence-local-editor",
    "candidate": {
      "candidate_id": "propainter-def456",
      "video_path": "/root/PhysGenLoop-/outputs/propainter-def456/propainter-def456-v01.mp4",
      "prompt": "A red ball falls onto a wooden floor.",
      "seed": 1007,
      "metadata": {
        "editor": "StrictProPainterLocalEditor",
        "editor_backend": "ProPainter"
      }
    },
    "next_prompt": "A red ball falls onto a wooden floor.",
    "cost": 0.0,
    "latency_seconds": 42.8,
    "terminal": false,
    "failure_reason": null,
    "artifacts": {
      "source_video": "/root/PhysGenLoop-/outputs/run/sample-0001/wan-abc123-v01.mp4",
      "repaired_video": "/root/PhysGenLoop-/outputs/propainter-def456/propainter-def456-v01.mp4",
      "mask_manifest": "/root/PhysGenLoop-/outputs/run/sample-0001/sam2_masks/mask_manifest.json"
    },
    "metadata": {
      "executor": "MaskSequenceLocalEditingExecutor",
      "editor": "StrictProPainterLocalEditor",
      "editor_backend": "ProPainter",
      "repair_mode": "strict-mask-video-inpainting",
      "target_objects": ["red ball"],
      "critical_frames": [18, 19, 20, 21],
      "propainter": {
        "repo": "/root/PhysGenLoop-/models/ProPainter",
        "script": "/root/PhysGenLoop-/models/ProPainter/inference_propainter.py",
        "weights_dir": "/root/PhysGenLoop-/models/ProPainter/weights",
        "repo_revision": "实际 git SHA",
        "python": "/root/PhysGenLoop-/envs/main/bin/python",
        "fp16": true,
        "returncode": 0,
        "elapsed_seconds": 41.9
      },
      "output_validation": {
        "exists": true,
        "decode_ok": true,
        "frame_count": 49,
        "source_frame_count": 49,
        "frame_count_match": true,
        "width": 832,
        "height": 480,
        "source_width": 832,
        "source_height": 480,
        "size_match": true,
        "candidate_prefix_ok": true
      }
    }
  },
  "before_scores": {
    "physics": 0.41,
    "semantic": 0.91,
    "original_prompt_semantic": 0.92,
    "quality": 0.86
  },
  "critic_after": {
    "physics_score": 0.68,
    "violations": []
  },
  "after_scores": {
    "physics": 0.68,
    "semantic": 0.90,
    "original_prompt_semantic": 0.90,
    "quality": 0.84
  },
  "physics_gain": 0.27,
  "repair_improved": true,
  "successful": true,
  "failure_reason": null,
  "metadata": {
    "gates": {
      "before": {
        "accepted": false
      },
      "after": {
        "status": "ACCEPTED",
        "accepted": true
      }
    },
    "score_deltas": {
      "physics": 0.27,
      "semantic": -0.01,
      "original_prompt_semantic": -0.02,
      "quality": -0.02
    },
    "validation_errors": []
  }
}
~~~

这个示例中 `successful=true`，因此 after gate 必须为 `ACCEPTED`。若 ProPainter 编辑只有物理收益且语义、画质下降在允许范围内，但仍未达到 Strict Gate 阈值，则示例必须改写为 `repair_improved=true, successful=false`，并以 `after_gate_rejected` 记录未通过原因。

#### 2.9.12 Schema 修正

`schemas/wan_repair_trial_v3.schema.json` 是此次不兼容契约升级所必需的唯一新增 schema；它不是第二套运行实现。旧 `wan_repair_trial_v2.schema.json` 保留只读，用于验证历史产物。V3 schema 不应只把 decision/execution 当作任意 object，至少应增加 local_editing 条件校验：

- decision.action 为 local_editing 时，decision.local_target 必须为非空 object；
- local_target 必须包含 parent_candidate_id、objects、critical_frames、mask_uri；
- execution 必须包含 status、backend_id、artifacts、metadata；
- execution.metadata 必须包含 executor、editor、editor_backend、repair_mode；
- editor_backend 必须等于 ProPainter；
- editor 必须等于 StrictProPainterLocalEditor；
- artifacts 必须包含 source_video 和 mask_manifest；
- status 为 succeeded 时，execution.candidate、repaired_video 和 output_validation 必须存在；
- successful 为 true 时，critic_after、after_scores 和 `after_gate.status=ACCEPTED` 必须同时成立；
- unsuccessful 时，failure_reason 必须为非空字符串。

这些要求应通过 JSON Schema 的 if/then 只约束 local_editing Trial，不应破坏 Prompt Repair 和 Reject 的合法结构。新 schema 的动作集合严格为 prompt_repair、local_editing、reject。

ProPainter 审计字段本身可以向后兼容，但本方案同时把动作空间从四项改为三项，属于不兼容语义变化，因此新运行统一使用 wan-repair-trial/3.0。旧 wan-repair-trial/2.0 仅通过显式 legacy adapter 读取，不能直接混入新训练集。

#### 2.9.13 必须补齐的测试

tests/wanphysics_v2/test_propainter_strict_editor.py：

- 每帧使用 manifest 中对应的 mask，不复制第一张 mask；
- 缺失、空、近全帧、尺寸不符、SHA 不符、越界 mask 均 fail closed；
- metadata.editor == StrictProPainterLocalEditor；
- metadata.editor_backend == ProPainter；
- metadata.mask_manifest_uri 等于 target.mask_uri；
- propainter repo/script/weights/revision 被记录；
- _run_propainter() 的 returncode、elapsed_seconds、script、fp16 被记录；
- 输出不存在、不可解码、零帧、帧数不一致、尺寸不一致均失败；
- 成功 candidate id 以 propainter- 开头。

tests/wanphysics_v2/test_decision_only_executors.py：

- local target 被原样传给 StrictProPainterLocalEditor；
- 成功 ExecutionResult 包含 executor/editor/editor_backend/repair_mode；
- source_video、repaired_video、mask_manifest 路径正确；
- ProPainter 异常被转换成 status=failed；
- 失败仍保留 local target、mask manifest、后端和错误元数据；
- executor 不重新运行 Policy，也不改变 prompt。

tests/wanphysics_v2/test_trials.py：

- _assemble_trials() 保留真实 action probabilities、confidence、instruction、source；
- 保留完整 local_target；
- source_candidate.video_path 是实际路径，不是字符串拼接出的假路径；
- critic_before/critic_after 是完整 CriticReport，不是 gate；
- before/after physics、semantic、quality 全部保留；
- ProPainter 成功但 physics 不提升时 unsuccessful；
- physics 提升但 semantic/quality 超限时 unsuccessful；
- 没有 Re-Critic 时 unsuccessful；
- ProPainter subprocess 失败仍产生 unsuccessful Trial；
- 缺失核心审计字段时显式失败，不允许 except/pass 静默跳过；
- JSON Schema 能验证成功与失败两类 Local Editing Trial。

另在现有 runner 测试中补一条完整因果断言：

~~~text
before candidate
  -> real RepairDecision(local_editing)
  -> StrictProPainterLocalEditor execution
  -> propainter-* after candidate
  -> full critic_after
  -> after_gate
  -> one and only one WanRepairTrialV3
~~~

#### 2.9.14 真实 smoke 验收

先确认配置与文件：

~~~bash
cd /root/PhysGenLoop-

test -f models/ProPainter/inference_propainter.py
test -d models/ProPainter/weights
test -x envs/main/bin/python

envs/main/bin/python -m pytest \
  tests/wanphysics_v2/test_propainter_strict_editor.py \
  tests/wanphysics_v2/test_decision_only_executors.py \
  tests/wanphysics_v2/test_trials.py \
  -q
~~~

真实 smoke 必须选一条能生成有效 SAM2 mask_manifest、Critic 能定位局部违背且 Three-Action Policy 会自然选择 `local_editing` 的样本。命令统一使用第 16.3 节的固定 `RUN_ROOT + nohup` 形式，只调用 `run_videophy2_loop_v2.py`，不得包含 `--allow-proxy-policy`、`--force-action` 或任何等价动作覆盖。

如果 Policy 没有自然选择 Local Editing，smoke 应失败并检查 Policy/capability；如果样本没有合法 mask，Guard 必须记录 blocked reason 并由 Runner 执行 Audited Reject，不能伪造 local target、全帧 mask 或成功 Trial。

smoke 后至少核对：

~~~text
run_manifest.json
preflight 中 local_editing capability
sample_status.json
critic_report.json
mask_manifest.json
repair_decision.json
repair_trace.jsonl
ProPainter 输出视频
loop_result.json
trials.jsonl
summary.json
~~~

并逐项验证：

1. repair_decision.action == local_editing；
2. repair_decision.local_target.mask_uri 指向本样本的 mask_manifest.json；
3. execution.metadata.editor == StrictProPainterLocalEditor；
4. execution.metadata.editor_backend == ProPainter；
5. ProPainter repo 是 /root/PhysGenLoop-/models/ProPainter；
6. returncode == 0 且输出验证通过；
7. after candidate id 以 propainter- 开头，视频实际存在且可解码；
8. after candidate 被 Re-Critic，critic_after 不是 after_gate；
9. Trial 保存真实 Policy 概率、instruction、local target 和 candidate 路径；
10. before/after physics、semantic、quality 均存在；
11. successful 判定与分数变化一致；
12. 本次 execution 在 trials.jsonl 中恰好对应一条 Trial，没有重复、没有丢失。

#### 2.9.15 ProPainter 实施完成判据

只有同时满足以下条件，才能认定 Local Editing 的 ProPainter 闭环完整：

- models/ProPainter 与 sam2-src 同层，配置解析到该真实目录；
- preflight 验证 repo、script、weights 和 Python；
- local_editing 默认注入 StrictProPainterLocalEditor；
- SAM2 逐帧 mask manifest 通过严格校验；
- 禁止单 mask 复制和全帧 fallback；
- ProPainter 子进程信息可审计；
- 输出视频经过存在性、解码、帧数和尺寸验证；
- candidate id 和 candidate path 均为真实值；
- ExecutionResult 显式标明 MaskSequenceLocalEditingExecutor、StrictProPainterLocalEditor 和 ProPainter；
- RoundRecord 保存真实 decision、before/after candidate、完整 Critic 和三类分数；
- _assemble_trials() 不再重建假 Decision，不再把 gate 当 Critic；
- ProPainter 失败也能生成 unsuccessful Trial；
- subprocess 返回 0 不等于修复成功；
- 每个成功候选都完成 Re-Critic/Re-Gate；
- physics、semantic、quality 的变化全部进入 Trial；
- 单元测试通过；
- 一条真实 Local Editing smoke 通过；
- execution 与 Trial 一一对应，summary 无 missing trial 或 trial assembly error。

完成以上修正后，Local Editing 的含义才是“由 SAM2 定位、由 ProPainter 实际修改、由 Critic 重新证明、由 Trial 完整留痕”的可验证闭环，而不是只记录一个 local_editing 标签。

---

## 3. 最新真实运行基线

最新服务器真实运行：

~~~text
/root/PhysGenLoop-/outputs/v2_run_20260722_033058
~~~

已经真实执行：

~~~text
Wan2.2 generation
-> vLLM Qwen3-VL
-> SAM2 propagation
-> Physics Critic
-> codec
-> Acceptance Gate
~~~

结果：

~~~text
stop_reason = accepted
trials_written = 0
physics_score = 1.0
confidence = 0.3025
coverage = 0.43
semantic_score = null
~~~

因为配置为 acceptance.mode=shadow，只有 decision=physical 和 physics_score 达标参与接受，低 confidence、低 coverage 和 semantic unavailable 没有阻止接受。

该 run 同时表明：

- physics_plan 为空。
- critic.json 仍为 waiting。
- SAM2 日志出现 _C 后处理不可用，但 critic_degraded=false。
- vllm.owner.json 没有生成。
- 没有执行任何 repair action。
- 没有真实 trial。

结论：目前证明了“生成—评价—接受”半链路，不证明修复闭环完整。

---

## 4. 实施边界

后续获得授权后允许：

- 修改现有 agents/wanphysics/ 文件。
- 修改现有 generators/wanphysics/ 和 generators/wanphysics/v2/ 文件。
- 修改现有 src/physgenloop/ 契约和兼容实现。
- 修改 configs/loop_v2.yaml 和现有 schema。
- 增加或修改 tests/wanphysics_v2/ 测试。
- 若旧版 src/physgenloop/repairer.py 已被重构删除，可按原路径恢复兼容实现。

禁止：

- git reset --hard、git clean、强制覆盖。
- 删除他人代码、worklog、输出或模型。
- 新增另一套 V2 runner 或 executor registry。
- 在 pipeline 中硬编码 /root/PhysGenLoop-。
- 伪造真实 trial、分数、概率或失败。
- 重新把 Memory 混入在线 Policy。

运行产物只进入：

~~~text
/root/PhysGenLoop-/outputs/v2_run_<timestamp>/
/root/PhysGenLoop-/outputs/v2_trials_<timestamp>/
~~~

设计与验证摘要进入：

~~~text
/root/PhysGenLoop-/worklog/YYYY_MM_DD/
~~~

---

## 5. Phase 0：冻结状态与基线检查

执行前只读检查：

~~~bash
cd /root/PhysGenLoop-
git status --short --branch
git log -1 --oneline
ps -eo pid,etimes,cmd | grep -E 'run_videophy2|gen_step|eval_step|vllm|ProPainter|pytest' | grep -v grep || true
nvidia-smi
~~~

验收：

- 确认没有覆盖其他成员未提交修改。
- 明确当前 HEAD 和 origin 差异。
- 明确当前 GPU、任务和最新输出。
- 不清理、不删除、不切换覆盖工作树。

---

## 6. Phase 1：废弃 Memory 在线路径

### 6.1 目标

在线链路不再调用：

~~~text
inspect_memory()
proxy_action_distribution()
memory mixing
memory write-back
~~~

主链路在 Trial 与 Artifact 结束。

### 6.2 修改位置

优先只改现有：

~~~text
generators/wanphysics/v2/build_backends.py
configs/loop_v2.yaml
tests/wanphysics_v2/test_memory_adapter.py
tests/wanphysics_v2/test_new_gaps.py
worklog/2026_07_22/v2重构后全链路架构.md
~~~

不删除 memory_adapter.py；把它保留为 deprecated historical compatibility。

从 build_v2_runner() 移除在线调用：

~~~python
mem_status = inspect_memory(...)
artifacts.write_memory_status(mem_status.to_dict())
~~~

repairer.py 当前已不加载 proxy memory，保持该行为。

兼容配置可以保留为：

~~~yaml
memory:
  mode: retired
  enable_proxy: false
~~~

但该配置不得影响 RepairDecision。

### 6.3 验收

~~~bash
grep -R -n "inspect_memory\|proxy_action_distribution\|memory_status\|enable_proxy" \
  agents generators src configs tests/wanphysics_v2
~~~

通过条件：

- build_v2_runner 不读取 memory 文件。
- 是否存在 proxy_memory_train.jsonl 不改变动作。
- 不再产生新 memory_status.json。
- Trial、repair trace 和 Critic artifacts 不受影响。
- 历史 memory 文件和模块没有被删除。

---

## 7. Phase 2：恢复旧版 Prompt Repair（不依赖 PhysicsPlan）

### 7.1 阶段结论

本阶段必须保留，不能因为取消 PhysicsPlan 而删除。Prompt Repair 的必要输入只有当前 prompt 和本轮 CriticReport：

~~~text
current prompt
  -> CriticReport.violations
  -> InstructionPromptRepairer
  -> rewritten prompt
  -> PromptRepairExecutor
  -> WanSubprocessGenerator.generate(prompt, seed)
  -> after candidate
  -> Re-Critic
  -> Re-Gate
~~~

服务器 revision 569ffb7 的实时审计事实：

- build_backends.py 仍注册 DecisionPromptRepairExecutor；
- 当前工作树没有 src/physgenloop/repairer.py；
- Git 历史 13e6602 和 b658c7c 中存在原版 InstructionPromptRepairer；
- src/physgenloop/learning_repair/executors.py 中仍存在原版 PromptRepairExecutor；
- 当前 PromptRepairExecutor 和 Generator 仍带无效 physics_plan 形参。

因此实施方式是恢复已有组件并修正接口，不是建立第二套 Prompt Repair。

### 7.2 权威接口

InstructionPromptRepairer 只消费 prompt 和 CriticReport：

~~~python
class InstructionPromptRepairer:
    def repair(
        self,
        *,
        prompt: str,
        report: CriticReport,
    ) -> str:
        ...
~~~

增强规则使用本方案 2.7.1 至 2.7.23 的约束：

- 保留原始对象、场景、动作、镜头和风格；
- 从 violation 中提取 object、category、event phase 和 repair_instruction；
- 同类指令去重；
- 约束按严重度与置信度排序；
- 多轮使用可替换 correction block，不无限追加；
- 不引入原 prompt 没有的新对象或新物理机制；
- 没有安全变化时显式返回 no_safe_prompt_change。

PromptRepairExecutor 的生成调用固定为：

~~~python
rewritten = prompt_rewriter.repair(
    prompt=request.prompt,
    report=request.critic_report,
)

if not rewritten.strip():
    return failed_result("empty_rewritten_prompt")

if rewritten.strip() == request.prompt.strip():
    return failed_result("no_safe_prompt_change")

candidate = generator.generate(
    prompt=rewritten,
    seed=request.seed,
)
~~~

不得出现 physics_plan，也不得在 rewritten 与输入相同时仅更换 seed 继续生成。

### 7.3 修改位置

~~~text
src/physgenloop/repairer.py
src/physgenloop/learning_repair/executors.py
src/physgenloop/learning_repair/contracts.py
generators/wanphysics/adapter.py
generators/wanphysics/v2/build_backends.py
configs/loop_v2.yaml
tests/wanphysics_v2/test_decision_only_executors.py
tests/wanphysics_v2/test_build_backends.py
~~~

优先从 Git 历史恢复原 src/physgenloop/repairer.py，再在同一文件内增强；这属于恢复被重构删除的原模块，不是新增平行实现。

配置：

~~~yaml
prompt_repair:
  backend: legacy
  backend_id: legacy-prompt-rewriter+video-generator
  max_constraints: 3
  max_prompt_chars: 900
  replace_existing_block: true
  allow_new_entities: false
~~~

### 7.4 三动作 Registry 接线

~~~python
from physgenloop.learning_repair.executors import PromptRepairExecutor
from physgenloop.repairer import InstructionPromptRepairer

prompt_repair_executor = PromptRepairExecutor(
    prompt_rewriter=InstructionPromptRepairer(
        max_constraints=prompt_cfg.get("max_constraints", 3),
        max_prompt_chars=prompt_cfg.get("max_prompt_chars", 900),
        replace_existing_block=prompt_cfg.get(
            "replace_existing_block",
            True,
        ),
        allow_new_entities=prompt_cfg.get(
            "allow_new_entities",
            False,
        ),
    ),
    generator=generator,
    backend_id=prompt_cfg.get(
        "backend_id",
        "legacy-prompt-rewriter+video-generator",
    ),
)

registry = ExecutorRegistry(
    executors=[
        prompt_repair_executor,
        MaskSequenceLocalEditingExecutor(editor=editor),
        AuditedRejectExecutor(
            selector=EvidenceAwareSelector(),
        ),
    ]
)
~~~

约束：

- 继续使用现有 ExecutorRegistry；
- runtime registry 中恰好三个动作；
- DecisionPromptRepairExecutor 不得成为默认后端；
- 不注册任何“原 prompt + 新 seed”执行器；
- Executor 不运行 Policy，Policy 每轮仍只调用一次。

### 7.5 ExecutionResult 与 Trial

Prompt Repair 成功时至少保存：

~~~text
backend_id=legacy-prompt-rewriter+video-generator
input_prompt
repaired_prompt
prompt_changed=true
instruction_source=critic_report.violations
target_objects
violation_categories
constraints_used
skipped_constraints
repaired_video
latency_seconds
~~~

失败时也必须产生 ExecutionResult 和 unsuccessful Trial：

~~~text
empty_rewritten_prompt
no_safe_prompt_change
prompt_rewriter_failed
wan_generation_failed
after_video_invalid
after_critic_failed
no_physics_gain
semantic_regression
quality_regression
~~~

### 7.6 测试要求

~~~python
assert result.status == "succeeded"
assert result.backend_id == (
    "legacy-prompt-rewriter+video-generator"
)
assert result.next_prompt != request.prompt
assert result.candidate.prompt == result.next_prompt
assert result.metadata["prompt_changed"] is True
~~~

还要验证：

- violation 没有可用 instruction 时不伪造 prompt 变化；
- 多条相同 instruction 去重；
- 多条不同 instruction 按优先级稳定输出；
- 多轮 correction block 被替换而非无限增长；
- Generator 调用参数只有 prompt 和 seed；
- Policy 只调用一次；
- after candidate 的 prompt 等于 rewritten prompt；
- after candidate 真实进入 Re-Critic/Re-Gate；
- Trial 保存修改前后 prompt 与三类分数；
- 没有 Re-Critic 或没有物理收益时不能 successful。

---

## 8. Phase 3：移除 PhysicsPlan 与 Global Regen，收敛为三动作闭环

### 8.1 本阶段目标和边界

目标动作空间：

~~~text
prompt_repair
local_editing
reject
~~~

目标在线链路：

~~~text
Prompt
  -> Wan
  -> Observation
  -> Physics Critic
  -> Acceptance Gate
  -> Three-Action Repair Policy
       -> Prompt Repair -> Wan
       -> Local Editing -> SAM2 + ProPainter
       -> Reject
  -> Re-Critic
  -> Re-Gate
  -> Trial / Final Result
~~~

删除动作的准确语义是：

~~~text
使用不变的原 prompt
+ 更换 seed
+ Wan 全视频重新生成
~~~

Prompt Repair 仍会调用 Wan 生成完整视频，但必须使用实际修改后的 prompt，因此不属于被删除动作。

PhysicsPlan 分两层处理：

1. 立即删除 Runner、Generator、Executor、ProPainter 和 Critic bridge 中的外部透传；
2. pavg_critic 内部先封装为实现细节，再按观测驱动方案解耦；禁止直接全局删类导致规则、力学、QuestionGraph 和 PQSG 崩溃。

### 8.2 第一层：删除主链路 PhysicsPlan

#### 8.2.1 Generator

文件：

~~~text
generators/wanphysics/adapter.py
~~~

删除 PhysicsPlan import，并将 WanPhysicsGenerator、WanSubprocessGenerator 从：

~~~python
def generate(
    self,
    *,
    prompt: str,
    physics_plan: PhysicsPlan,
    seed: int,
) -> GeneratedCandidate:
~~~

改为：

~~~python
def generate(
    self,
    *,
    prompt: str,
    seed: int,
) -> GeneratedCandidate:
~~~

两种 Generator 的函数体原本就未消费 plan，不需要增加替代逻辑。

#### 8.2.2 ExecutionRequest 与 Executor

文件：

~~~text
src/physgenloop/learning_repair/contracts.py
src/physgenloop/learning_repair/executors.py
generators/wanphysics/v2/executors.py
~~~

ExecutionRequest 删除：

~~~python
physics_plan: Any
~~~

所有生成调用统一为：

~~~python
candidate = generator.generate(
    prompt=target_prompt,
    seed=request.seed,
)
~~~

#### 8.2.3 ProPainter

文件：

~~~text
generators/wanphysics/local_editor.py
generators/wanphysics/v2/propainter_strict_editor.py
generators/wanphysics/v2/executors.py
~~~

ProPainterLocalEditor.edit() 和 StrictProPainterLocalEditor.edit() 删除 physics_plan 形参；MaskSequenceLocalEditingExecutor 删除对应实参。ProPainter 只消费 source video、逐帧 mask、local target、instruction 和 seed。

#### 8.2.4 Runner 与 Acceptance Gate

文件：

~~~text
generators/wanphysics/v2/runner.py
generators/wanphysics/v2/build_backends.py
agents/wanphysics/run_videophy2_loop_v2.py
~~~

Runner.run() 目标签名：

~~~python
def run(
    self,
    *,
    sample_id: str,
    prompt: str,
) -> V2RunResult:
~~~

删除：

~~~text
RunnerConfig.require_plan
ActionAwareRunnerV2._plan_ready()
plan_ready
plan_blocks
physics_plan=None 入口实参
ExecutionRequest.physics_plan
~~~

Acceptance Gate 恢复为：

~~~python
accepted = gate.accepted
~~~

Gate 只判断 physics、confidence、coverage、semantic、quality 和 critic degraded 状态。

#### 8.2.5 Critic bridge

公开接口改成：

~~~python
critic.evaluate(
    candidate,
    prompt=current_prompt,
)
~~~

涉及：

~~~text
generators/wanphysics/v2/build_backends.py
generators/wanphysics/v2/critic_backend.py
generators/wanphysics/sam2_vlm_critic.py
agents/wanphysics/eval_step.py
~~~

eval_step 第一阶段可以构造：

~~~python
CriticRequest(
    video_path=candidate.video_path,
    prompt=candidate.prompt,
)
~~~

Runner、Generator 和 Repair Executor 不再知道 PhysicsPlan。

### 8.3 第二层：pavg_critic 内部解耦

当前不能直接删除 planner.py，因为以下现有模块仍真实读取 request.physics_plan：

~~~text
src/pavg_critic/pipeline.py
src/pavg_critic/planner.py
src/pavg_critic/question_generator.py
src/pavg_critic/pqsg.py
src/pavg_critic/mechanics.py
src/pavg_critic/physics_rules.py
src/pavg_critic/checklist.py
src/pavg_critic/schemas.py
~~~

彻底删除时应把 Critic 改为观测驱动：

~~~text
Video
  -> Object Detection + SAM2 Tracking
  -> Trajectories
  -> Observed Events
  -> Question/Rule Construction
  -> Mechanics and Physics Evaluation
  -> CriticReport
~~~

CriticRequest 最终只保留：

~~~text
video_path
prompt
reference_simulation
schema_version
~~~

具体顺序：

1. 先完成视频观测、跟踪和事件检测；
2. question_generator.py 与 pqsg.py 消费 prompt、detected objects、tracks 和 events；
3. mechanics.py、physics_rules.py、checklist.py 根据检测到的 collision、rebound、falling、contact 等事件判断规则是否适用；
4. prompt fulfillment 交给 semantic scorer/VLM；
5. 物理规则只评价观测运动是否合理；
6. 移除 planner trace 与 resolved_plan diagnostics；
7. 全部 request.physics_plan 引用清零后，再删除 PhysicsPlanner、PhysicsPlanResolver、TemplatePhysicsPlanner、ModelPhysicsPlanner 和 PhysicsPlan schema。

这是 Critic 语义重构，必须独立升级 schema 并运行回归测试。第一层完成后即可得到干净在线接口，不应阻塞其他三动作闭环修复。

### 8.4 RepairAction 收敛为三动作

文件：

~~~text
src/physgenloop/learning_repair/base_contracts.py
~~~

目标：

~~~python
class RepairAction(str, Enum):
    PROMPT_REPAIR = "prompt_repair"
    LOCAL_EDITING = "local_editing"
    REJECT = "reject"

ACTION_ORDER = tuple(RepairAction)
~~~

RepairContext 删除 global_regeneration_available，并同步修改：

~~~text
action_available()
to_dict()
from_dict()
feature encoding
compatibility manifest
~~~

动作集合变化不向后兼容，建议至少升级：

~~~text
REPAIR_SCHEMA_VERSION=2.0
learning-repair-decision/2.0
repair-context/2.0
policy-guard/2.0
wan-repair-trial/3.0
~~~

### 8.5 删除动作 Executor 和 Registry 接线

删除活动代码中的：

~~~text
src/physgenloop/learning_repair/executors.py
  GlobalRegenerationExecutor

generators/wanphysics/v2/executors.py
  OriginalPromptGlobalRegenerationExecutor
~~~

build_backends.py 同步删除 import、实例化和 registry 项。最终 registry：

~~~python
ExecutorRegistry(
    executors=[
        PromptRepairExecutor(
            prompt_rewriter=InstructionPromptRepairer(),
            generator=generator,
            backend_id=(
                "legacy-prompt-rewriter+video-generator"
            ),
        ),
        MaskSequenceLocalEditingExecutor(editor=editor),
        AuditedRejectExecutor(
            selector=EvidenceAwareSelector(),
        ),
    ]
)
~~~

### 8.6 Policy 与 PolicyGuard 的三动作边界

`policy_guard.py` 的 scope 从 local/global/unknown 改为 local/broad/unknown；broad 只表示违背覆盖范围较大。动作选择与动作校验必须分离：

1. Three-Action Policy 消费 CriticReport、GateResult、capability mask、历史轮次和当前预算，且每轮只决策一次；
2. Policy 必须在可执行动作集合内归一化概率并选择 `prompt_repair`、`local_editing` 或 `reject`；
3. PolicyGuard 只验证该真实 Decision 是否满足 capability 与必要字段，不得替 Policy 改成另一动作；
4. Guard blocked 时保留原始 Policy Decision 和 blocked reason，并由 Runner 显式执行 Audited Reject；不得静默把 local_editing 改成 prompt_repair；
5. Prompt 文本是否真的变化只能由 Executor 调用 InstructionPromptRepairer 后得知，Guard 不得提前使用虚假的 `prompt_repair_ok`；
6. Critic/provider/scorer failure 在 Gate 层必须成为 `UNAVAILABLE -> EVALUATION_FAILED`，不得进入 Policy 后伪装成 Reject。

Guard 校验规则：

| Policy action | Guard 允许条件 | 不满足时 |
|---|---|---|
| reject | Decision 结构合法 | 执行 Audited Reject |
| local_editing | scope=local、local_target 完整、strict mask 合法、SAM2/ProPainter capability 可用 | 记录 blocked reason，执行 Audited Reject |
| prompt_repair | CriticReport 完整、有可靠 instruction、PromptRepairExecutor/rewriter/Wan capability 可用 | 记录 blocked reason，执行 Audited Reject |

真实执行顺序固定为：

~~~text
Policy -> PolicyGuard -> Executor
PromptRepairExecutor -> InstructionPromptRepairer -> rewritten prompt validation -> Wan
~~~

如果 rewritten prompt 为空或与输入相同，Executor 返回 `empty_rewritten_prompt` 或 `no_safe_prompt_change`，产生 unsuccessful Trial；不得仅更换 seed，也不得回到 Guard 重新伪造另一动作。

### 8.7 Capability、Preflight 和单入口 CLI

capability mask 目标：

~~~python
{
    "prompt_repair": True,
    "local_editing": True,
    "reject": True,
}
~~~

涉及：

~~~text
generators/wanphysics/v2/preflight.py
generators/wanphysics/v2/runner.py
generators/wanphysics/v2/build_backends.py
agents/wanphysics/run_videophy2_loop_v2.py
~~~

正式 CLI 删除并拒绝 `--force-action`、`--action-override` 等动作覆盖参数；配置、manifest 和环境变量也不得提供等价覆盖。唯一入口是 `agents/wanphysics/run_videophy2_loop_v2.py`。Preflight 只检查 Prompt Repair 后端、SAM2/ProPainter Local Editing 后端；Reject 恒可用，并把 capability mask 交给 Policy。

### 8.8 Policy checkpoint 不兼容处理

当前 checkpoint：

~~~text
checkpoints/repair_agent/
  repair-agent-v3.1-proxy-20260717/
~~~

其 action_order 为四项，action/value head 也是四维；删除一个动作后不能继续加载。

禁止：

~~~text
加载旧四维 checkpoint
  -> 丢弃一个 logit
  -> 把其概率并入 prompt_repair
~~~

这样会保留隐式旧动作语义，并破坏训练时的决策边界。

正确处理：

1. 旧 checkpoint 和训练报告保留为历史审计，不修改；
2. runtime 停止加载该 checkpoint；
3. compatibility gate 返回 action_order_mismatch 和 checkpoint_incompatible_with_three_action_runtime；
4. smoke 阶段使用现有 HeuristicDecisionPolicy 的三动作版本；
5. 三动作真实 Trial 足够后训练新的三输出 Policy；
6. 新 checkpoint action_order 固定为 prompt_repair、local_editing、reject。

旧 checkpoint 本身 deployment_ready=false、policy_mode=proxy_research，因此停止接入不会损失已部署模型。

### 8.9 Heuristic Policy 与 fallback

文件：

~~~text
src/physgenloop/learning_repair/policy.py
src/physgenloop/learning_repair/baselines.py
generators/wanphysics/repairer.py
~~~

推荐映射：

~~~python
_CATEGORY_ACTION = {
    "gravity_violation": RepairAction.PROMPT_REPAIR,
    "friction_violation": RepairAction.PROMPT_REPAIR,
    "contact_violation": RepairAction.PROMPT_REPAIR,
    "collision_violation": RepairAction.LOCAL_EDITING,
    "trajectory_violation": RepairAction.LOCAL_EDITING,
    "continuity_violation": RepairAction.LOCAL_EDITING,
    "appearance_violation": RepairAction.LOCAL_EDITING,
    "unknown_violation": RepairAction.REJECT,
}
~~~

低 coverage、decision=unknown 或 provider failure 直接 Reject。其他动作不可用时的安全顺序：

~~~text
首选动作可执行
  -> 执行首选
否则若有可靠 instruction
  -> Prompt Repair
否则
  -> Reject
~~~

Local Editing 只有在 strict mask 和 ProPainter 都可用时才进入候选集合。

### 8.10 Feature、Policy Head 和兼容性

删除：

~~~text
context.global_regeneration_available
previous_action.global_regeneration
~~~

涉及：

~~~text
src/physgenloop/learning_repair/features.py
src/physgenloop/learning_repair/policy.py
src/physgenloop/learning_repair/value_policy.py
src/physgenloop/learning_repair/selector.py
src/physgenloop/learning_repair/compatibility.py
~~~

新模型：

~~~python
action_head = nn.Linear(width, 3)
value_head = nn.Linear(width, 3)
~~~

Feature dimension、feature schema 和 action_order 必须同步更新；compatibility manifest 必须在加载 state_dict 前拒绝旧维度。

### 8.11 Trial、Schema 与历史数据

新 Decision：

~~~json
{
  "action": "prompt_repair",
  "action_probabilities": {
    "prompt_repair": 0.70,
    "local_editing": 0.20,
    "reject": 0.10
  },
  "per_action_values": {
    "prompt_repair": 0.50,
    "local_editing": 0.20,
    "reject": -0.10
  }
}
~~~

历史数据处理：

- 历史 prompt_repair、local_editing、reject 可以经过 schema adapter 后保留；
- 历史被删除动作 Trial 不自动改标签；
- 此类 Trial 标记 legacy_four_action_artifact、not_compatible_with_three_action_runtime、excluded_from_current_training；
- 历史 checkpoint、Trial 和报告不物理删除，以保留科研审计；
- 新 runtime、新 Trial 和新训练集不再产生或接受第四动作。

### 8.12 退役 Forced Trial 与第二入口

当前不再进行 forced-trial 研究采集。`agents/wanphysics/run_actual_trials_v2.py` 不物理删除历史实现，但必须退出活跃架构：

- 不再出现在正式命令、测试矩阵、README 入口和完成判据中；
- `main()` 入口应 fail fast，以非零退出码提示改用 `run_videophy2_loop_v2.py`；
- 不再写新版 `WanRepairTrialV3`，也不再维护独立 output-root/Resume 语义；
- 历史 forced Trial 保留只读，并标记 legacy、runtime incompatible、excluded from current statistics；
- 新 Decision、Trial、run manifest 与 repair trace 不再生成 `force_action`、`forced_action`、`action_override`、`research_only` 等字段。

三动作覆盖改由三层测试完成：Executor 单元测试直接构造 Decision；Policy 单元测试构造 Critic/Gate/capability；真实 smoke 使用精选样本让 Policy 自然选择动作。测试不得把 `expected_action` 反向注入 runtime。

### 8.13 测试与完成判据

动作空间：

~~~python
assert tuple(a.value for a in RepairAction) == (
    "prompt_repair",
    "local_editing",
    "reject",
)
~~~

Registry：

~~~python
assert registry.supports(RepairAction.PROMPT_REPAIR)
assert registry.supports(RepairAction.LOCAL_EDITING)
assert registry.supports(RepairAction.REJECT)
assert len(registry.actions()) == 3
~~~

Guard 必须覆盖：

- broad + instruction -> prompt_repair；
- broad + 无 instruction -> reject；
- local + valid mask -> local_editing；
- local + invalid mask + instruction -> prompt_repair；
- local + invalid mask + 无 instruction -> reject；
- unknown/provider failure -> reject；
- Reject 不被 Guard 改写；
- Guard 永远不产生第四动作。

Checkpoint 必须覆盖：

- 旧四动作 checkpoint 因 action_order mismatch 被拒绝；
- 三动作 checkpoint 可以加载；
- action/value head 输出维度均为 3。

全仓运行代码验收：

~~~bash
grep -RIn \
  --include='*.py' \
  --include='*.yaml' \
  --include='*.json' \
  'global_regeneration\|GLOBAL_REGENERATION' \
  agents generators src configs schemas
~~~

目标输出为空。历史 worklog、checkpoint 报告和旧 outputs 不纳入该零引用检查。

PhysicsPlan 第一层验收：

- Runner.run() 只有 sample_id 和 prompt；
- Generator.generate() 只有 prompt 和 seed；
- ExecutionRequest 没有 physics_plan；
- ProPainter editor 没有 physics_plan；
- V2 Critic bridge 不接收 physics_plan；
- Acceptance Gate 不再检查 require_plan；
- dry-run、CPU 测试和三条真实 smoke 全部通过。

---

## 9. Phase 4：Critic、fallback 与降级状态对齐

### 9.1 保持现有主路径

继续使用 eval_step.py、SAM2ObjectDetector、PhysicsCritic 和 critic codec，不新建 Critic。

### 9.2 准确标注能力

当前有效 profile 应保持：

~~~text
sam2_seeded_rules
~~~

只有真正接入 VLM physics verifier 后，才允许使用更强 profile 名称。

### 9.3 配置必须生效

当前配置：

~~~yaml
critic:
  allow_rules_fallback: false
  formal_profile_required: false
~~~

这是服务器基线事实，不是目标值。目标配置必须改为 `formal_profile_required: true`，同时保持能力画像名称 `profile: sam2_seeded_rules`；严格性由 `acceptance.mode=enforce` 和 `fail_on_degraded_critic=true` 表达，不另造 `sam2_seeded_rules_strict` profile。

需要真正传入 eval_step 或 Critic backend。`allow_rules_fallback=false` 时，SAM2 失败必须让 Gate 返回 `UNAVAILABLE`、样本终态写为 `EVALUATION_FAILED`，不能只打印 warning 并继续 rules fallback，也不再产生独立的 `CRITIC_FAILED` 样本终态。

### 9.4 SAM2 degraded

日志出现 _C 后处理不可用时，报告应写：

~~~json
{
  "diagnostics": {
    "sam2_postprocess": "disabled",
    "degraded": true,
    "degraded_reasons": ["sam2_postprocess_disabled"]
  }
}
~~~

fail_on_degraded_critic=true 时，enforce Gate 必须拒绝。

### 9.5 SAM2 CUDA 扩展 `_C` 修复方案

#### 9.5.1 问题性质

当前日志中的错误是：

~~~text
cannot import name '_C' from 'sam2'
Skipping the post-processing step
~~~

缺失的是 SAM2 的可选 CUDA 扩展 `sam2._C`。该扩展实现 connected components，供 `fill_holes_in_mask_scores()` 填补预测 mask 中面积较小的孔洞。

它不是 SAM2 checkpoint，也不是 SAM2 主网络。因此当前表现是：

~~~text
SAM2 主体仍可初始化和传播
但小孔填充后处理被跳过
mask 质量处于 degraded 状态
~~~

这不会必然导致整个 Critic 失败，但可能影响 strict ProPainter 消费的 mask，并且当前报告没有如实标记降级。

#### 9.5.2 服务器实测环境

只读检查得到：

~~~text
SAM2 安装方式       editable
SAM2 源码           /root/PhysGenLoop-/models/sam2-src
Python              3.12.13
PyTorch             2.7.1+cu128
torchvision         0.22.1+cu128
torch CUDA runtime  12.8
本机 CUDA Toolkit   /usr/local/cuda-12.2
nvcc                12.2.140
gcc/g++             9.4.0
GPU                 NVIDIA A100, compute capability 8.0
ninja               未安装
sam2._C spec         None
sam2/_C.so           不存在
~~~

SAM2 setup.py 默认配置：

~~~text
SAM2_BUILD_CUDA=1
SAM2_BUILD_ALLOW_ERRORS=1
~~~

`SAM2_BUILD_ALLOW_ERRORS=1` 会吞掉扩展编译失败并继续完成 editable 安装，所以 pip 显示 SAM-2 已安装，但没有生成 `_C.so`。这是当前问题的直接原因。

#### 9.5.3 首选修复：严格原地编译 `_C`

首选在现有 SAM2 源码和 main 环境中原地编译，不更换 PyTorch，不重装 SAM2 checkpoint。

编译会修改环境并在源码目录产生 `_C.so`，因此只有获得明确授权后才执行。编译日志写入 worklog：

~~~bash
cd /root/PhysGenLoop-

# 安装编译加速器；只安装到 main 环境
envs/main/bin/pip install ninja

cd models/sam2-src

export CUDA_HOME=/usr/local/cuda-12.2
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.0"
export SAM2_BUILD_CUDA=1

# 禁止 setup.py 吞掉编译错误
export SAM2_BUILD_ALLOW_ERRORS=0

/root/PhysGenLoop-/envs/main/bin/python setup.py build_ext --inplace -v \
  2>&1 | tee /root/PhysGenLoop-/worklog/2026_07_22/sam2_c_build.log

cd /root/PhysGenLoop-
~~~

选择 `build_ext --inplace` 而不是先 force-reinstall，原因是：

- 当前已经是 editable 安装，Python 直接从 models/sam2-src/sam2 导入；
- 原地生成 `sam2/_C.so` 后即可生效；
- 编译失败不会先卸载当前可运行的 SAM-2 包；
- `SAM2_BUILD_ALLOW_ERRORS=0` 能保留真实错误并返回非零状态。

预期新增文件：

~~~text
/root/PhysGenLoop-/models/sam2-src/sam2/_C.so
~~~

#### 9.5.4 CUDA 12.2 与 PyTorch cu128 的处理

当前 nvcc 是 12.2，而 PyTorch 编译目标是 CUDA 12.8。两者 major version 都是 12，PyTorch extension 通常会给出 minor-version warning 并允许编译，但必须以严格构建结果和导入测试为准。

处理顺序：

1. 先使用现有 CUDA Toolkit 12.2 严格编译。
2. 如果编译和 CUDA kernel 测试通过，不调整 PyTorch。
3. 如果出现明确的 CUDA 12.2/12.8 编译或链接不兼容，再经授权安装 CUDA Toolkit 12.8 到独立路径。
4. 之后只把 `CUDA_HOME` 指向 12.8 重新编译。
5. 不优先降级 PyTorch，因为 Wan、SAM2、torchvision 和其他 GPU 模块共享当前 main 环境。

CUDA 12.8 分支示意：

~~~bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.0"
export SAM2_BUILD_CUDA=1
export SAM2_BUILD_ALLOW_ERRORS=0

cd /root/PhysGenLoop-/models/sam2-src
/root/PhysGenLoop-/envs/main/bin/python setup.py build_ext --inplace -v \
  2>&1 | tee /root/PhysGenLoop-/worklog/2026_07_22/sam2_c_build_cuda128.log
~~~

该分支涉及系统 CUDA Toolkit，必须单独确认，不在普通代码补全中自动执行。

#### 9.5.5 一级验证：扩展文件与导入

~~~bash
cd /root/PhysGenLoop-

find models/sam2-src/sam2 -maxdepth 1 -type f -name '_C*.so' -ls

PYTHONDONTWRITEBYTECODE=1 envs/main/bin/python - <<'PY'
import importlib.util
print("spec:", importlib.util.find_spec("sam2._C"))
from sam2 import _C
print("SAM2 _C import OK:", _C)
PY
~~~

通过条件：

- `_C.so` 存在且非零；
- `find_spec("sam2._C")` 非 None；
- `from sam2 import _C` 成功；
- 没有 undefined symbol 或 libcudart 链接错误。

#### 9.5.6 二级验证：真实 CUDA kernel

只验证 import 不足以证明 kernel 可用，还要在 A100 上运行 connected components：

~~~bash
cd /root/PhysGenLoop-

CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 envs/main/bin/python - <<'PY'
import torch
from sam2.utils.misc import get_connected_components

mask = torch.zeros((1, 1, 32, 32), dtype=torch.bool, device="cuda")
mask[:, :, 4:12, 4:12] = True
labels, areas = get_connected_components(mask)

assert labels.shape == mask.shape
assert areas.shape == mask.shape
assert int(areas.max().item()) == 64
print("SAM2 connected-components CUDA kernel OK")
PY
~~~

通过条件：

- CUDA kernel 无异常；
- 输出 shape 与输入一致；
- 连通区域面积为预期值；
- 没有 illegal memory access、invalid device function 或 undefined symbol。

#### 9.5.7 三级验证：SAM2 propagation

使用现有单样本 smoke 或一个短视频运行 SAM2 propagation，检查：

- 日志不再出现 `cannot import name '_C'`；
- 日志不再出现 `Skipping the post-processing step`；
- propagation 完成全部帧；
- materialize_masks 能正常写出 critical frame mask；
- mask manifest 的尺寸、非零比例和 SHA 校验通过；
- Re-Critic 能读取同一批 mask。

只有三级验证通过，才能把 Critic diagnostics 标为：

~~~json
{
  "sam2_cuda_extension": "available",
  "sam2_postprocess": "enabled",
  "degraded": false
}
~~~

#### 9.5.8 代码层诊断接线

即使扩展修复成功，代码仍应主动探测能力，不能依靠日志文本推断。

在现有 eval_step.py 或 SAM2ObjectDetector 初始化路径中执行一次：

~~~python
try:
    from sam2 import _C  # noqa: F401
    sam2_cuda_extension = "available"
except Exception as exc:
    sam2_cuda_extension = "unavailable"
    sam2_cuda_extension_error = f"{type(exc).__name__}: {exc}"
~~~

然后通过 `dataclasses.replace()` 把结果写入 CriticReport.diagnostics。禁止把导入失败静默吞掉。

缺失 `_C` 不等于 rules fallback：SAM2 主体仍可能工作。因此建议状态区分为：

~~~text
sam2_cuda_extension=unavailable
sam2_postprocess=disabled
effective_profile=sam2_seeded_rules
degraded=true
~~~

active runtime 固定 enforce 且 fail_on_degraded_critic=true。出现 degraded 时不得 accepted；若 required evidence 因降级不可用则状态为 UNAVAILABLE，否则为 REJECTED，并完整保留 degraded 原因。

#### 9.5.9 编译暂不可用时的安全兜底

如果 `_C` 暂时无法编译，不应每一帧都触发异常后再跳过。可以在现有 sam2_detector.py 构造 predictor 后明确关闭依赖 `_C` 的 hole filling：

~~~python
predictor.fill_hole_area = 0
~~~

这只关闭小孔填充，尽量保留其他 SAM2 postprocessing。该方案必须同时写：

~~~json
{
  "sam2_cuda_extension": "unavailable",
  "sam2_postprocess": "hole_filling_disabled",
  "degraded": true,
  "degraded_reasons": ["sam2_cuda_extension_unavailable"]
}
~~~

该路径是显式降级，不是 `_C` 功能修复，不能将其标记为 postprocess enabled。

#### 9.5.10 回退原则

如果新 `_C.so` 可以导入但 kernel 不稳定：

1. 不删除 SAM2 源码、checkpoint 或现有环境。
2. 临时设置 `predictor.fill_hole_area=0`，避免调用扩展。
3. 保留 `_C.so` 和完整 build log 供定位。
4. 报告中标记 degraded。
5. 不运行大批量任务，直到 kernel smoke 和单样本 propagation 通过。

#### 9.5.11 `_C` 修复完成判据

必须同时满足：

- `_C.so` 存在；
- Python import 成功；
- A100 connected-components kernel 测试通过；
- 真实 SAM2 propagation 不再出现 `_C` warning；
- mask manifest 校验通过；
- CriticReport diagnostics 准确写入 extension/postprocess 状态；
- strict Local Editing 的 mask 输入没有回归；
- tests/wanphysics_v2 与一条真实 smoke 通过。

### 9.6 critic.json 同步

eval_step 完成后同步：

~~~json
{
  "status": "completed",
  "physics_violation": true,
  "confidence": 0.7,
  "detector_backend": "sam2+vlm"
}
~~~

避免 critic.json=waiting 与 critic_report.json=completed 冲突。

---

## 10. Phase 5：Runner 状态机修正

### 10.1 避免重复 Re-Critic

当前 after candidate 在执行后立即 Re-Critic；下一轮又重新 Critic 同一 candidate。

在现有 ActionAwareRunnerV2 内缓存：

~~~text
current_candidate
current_prompt
current_report
current_gate
~~~

下一轮直接消费已得到的 after_report/after_gate，不重复执行同一 Critic。

### 10.2 终态一致性

| 结果 | V2RunResult.final_state | sample_status.state |
|---|---|---|
| 接受 | ACCEPTED | ACCEPTED |
| Reject | REJECTED | REJECTED |
| Generator/Executor 失败 | EXECUTION_FAILED | EXECUTION_FAILED |
| Critic/Gate/required scorer 不可用 | EVALUATION_FAILED | EVALUATION_FAILED |
| Preflight 失败 | PREFLIGHT_FAILED | PREFLIGHT_FAILED |
| 达到上限 | MAX_ROUNDS | MAX_ROUNDS |

Gate 状态严格限定为 `ACCEPTED/REJECTED/UNAVAILABLE`；样本终态严格限定为上表六项。`CRITIC_FAILED` 统一并入 `EVALUATION_FAILED`；`EXECUTOR_FAILED` 统一为 `EXECUTION_FAILED`；`REJECTED_BEST_EFFORT` 改为 `MAX_ROUNDS + final_candidate_disposition=best_effort`。`COMPLETED` 只能是 run 级状态，runner 结尾不能用它覆盖样本真实终态。

### 10.3 Best-of-K

RunnerConfig 已有 candidates_per_round，但当前没有使用。

执行规则：

1. 初始轮按 candidates_per_round 生成 K 个候选。
2. 每个候选经过同一 Critic。
3. selector 选出 round winner。
4. Gate 只对 winner 做终止判定。
5. 所有候选报告仍需审计保存。
6. repair action 的 after candidate 成为下一轮 current，不重新生成无关候选，除非设计明确要求新一轮 K 候选。

若首轮暂时保持 K=1，必须在文档和 summary 中明确，不得宣称已经 Best-of-K。

### 10.4 异常闭环

Generator、Critic、Executor 任一异常必须：

1. 写 sample_status。
2. 写 failure_reason 和当前 command/backend。
3. 保存 raw payload。
4. 释放本 run 自己拥有的资源。
5. 尽量继续下一个样本。
6. 不把失败样本写成 completed。

---

## 11. Phase 6：Strict Enforce Acceptance Gate

### 11.1 阶段目标

在线运行只允许 strict enforce，不再允许 shadow 作为可选运行模式。Acceptance Gate 是整个无 PhysicsPlan、无 Global Regen、三动作闭环的唯一接受标准：

~~~text
所有必需评价真实可用
且所有指标全部达标
且 Critic 没有降级
且没有阻断性物理违背
才允许 accepted
~~~

严格模式禁止：

- 缺失分数由其他分数替代；
- scorer unavailable 被当成普通视频不合格；
- Critic degraded 后仍 accepted；
- Executor 返回 succeeded 就认定修复成功；
- best-effort 候选被标记为 accepted；
- Prompt Repair 通过改变用户任务来获得更高 semantic；
- ProPainter 通过擦除对象来获得更高 physics。

### 11.2 完整状态机

~~~text
Prompt
  -> Wan 首轮视频生成
  -> 视频观测
       -> VLM 对象识别
       -> SAM2 跟踪
       -> 轨迹提取
       -> 事件检测
  -> Physics Critic
  -> Strict Enforce Acceptance Gate
       -> ACCEPTED
            -> 输出最终候选
            -> 写 evaluation audit
            -> 结束

       -> REJECTED
            -> Three-Action Repair Policy
                 -> Prompt Repair
                      -> InstructionPromptRepairer
                      -> Wan 新候选
                      -> Re-Critic
                      -> Strict Enforce Re-Gate

                 -> Local Editing
                      -> SAM2 mask manifest
                      -> StrictProPainterLocalEditor
                      -> ProPainter 新候选
                      -> Re-Critic
                      -> Strict Enforce Re-Gate

                 -> Reject
                      -> 返回 rejected best-effort
                      -> accepted=false
                      -> 结束

       -> UNAVAILABLE
            -> 评价链路失败
            -> 不调用 Repair Policy
            -> 不伪装成视频物理不合格
            -> 结束

Re-Gate
  -> ACCEPTED
       -> 输出最终候选
  -> REJECTED
       -> 未到 max_rounds：回到 Three-Action Repair Policy
       -> 达到 max_rounds：MAX_ROUNDS，保留 best-effort 候选但 accepted=false
  -> UNAVAILABLE
       -> EVALUATION_FAILED
~~~

这不是两套独立 pipeline。上半部分评价当前候选；下半部分只处理 Gate 返回 REJECTED 后的修复分支。Prompt Repair 和 Local Editing 产生的新候选重新进入同一个 Critic 和同一个 Gate。

必须区分：

~~~text
Gate REJECTED
= 当前视频未达到严格标准，但评价链路完整

Policy Reject
= 当前没有值得继续执行的安全修复动作，终止循环

Gate UNAVAILABLE
= 系统无法可靠评价，不代表视频本身一定不合格
~~~

### 11.3 Gate 三状态

当前 GateResult 主要使用 accepted: bool，严格模式扩展为：

~~~text
ACCEPTED
REJECTED
UNAVAILABLE
~~~

| 状态 | 含义 | 后续 |
|---|---|---|
| ACCEPTED | 所有必需分数存在且全部达标 | 接受候选 |
| REJECTED | 评价链路完整，但至少一个视频质量条件不达标 | 进入三动作 Repair Policy |
| UNAVAILABLE | Critic、semantic、quality 或必要证据不可用 | 终止评价，不执行 Repair Policy |

状态约束：

~~~text
status=accepted
  -> accepted=true
  -> reasons=[]
  -> unavailable=[]

status=rejected
  -> accepted=false
  -> unavailable=[]
  -> reasons 非空

status=unavailable
  -> accepted=false
  -> unavailable 非空
~~~

例如 semantic scorer 服务崩溃时：

~~~text
错误：
semantic unavailable
  -> 当作视频 semantic 不达标
  -> 进入 Prompt Repair

正确：
semantic unavailable
  -> Gate UNAVAILABLE
  -> EVALUATION_FAILED
  -> 不执行修复
~~~

### 11.4 严格通过条件

ACCEPTED 必须同时满足：

~~~text
critic report roundtrip 成功
decision == physical
blocking violation count == 0
physics_score >= 0.80
confidence >= 0.60
coverage >= 0.50
semantic_score >= 0.85
original_prompt_semantic_score >= 0.85
quality_score >= 0.75
critic_degraded == false
required evidence 全部 available
没有 required scorer unavailable
~~~

伪代码：

~~~python
accepted = all(
    (
        report_roundtrip_ok,
        decision == "physical",
        blocking_violation_count == 0,
        physics_score >= thresholds.physics_score,
        confidence >= thresholds.confidence,
        coverage >= thresholds.coverage,
        semantic_score >= thresholds.semantic_score,
        original_prompt_semantic_score
        >= thresholds.original_prompt_semantic_score,
        quality_score >= thresholds.quality_score,
        not critic_degraded,
        not unavailable,
    )
)
~~~

这里不再包含：

~~~text
PhysicsPlan 完整
require_plan
plan_ready
plan_blocks
~~~

也不包含已删除动作的任何 fallback 或可用性条件。

严格不等于把阈值机械设为 1.0。严格的含义是所有已声明条件同时生效、证据缺失 fail closed、禁止分数替代、禁止降级通过。阈值仍需使用真实样本校准，但 runtime 从第一天起即使用 enforce 语义。

### 11.5 配置修改

configs/loop_v2.yaml：

~~~yaml
runtime:
  pipeline_version: v2
  strict_report_roundtrip: true
  persist_full_critic_report: true
  fail_on_degraded_critic: true
  fail_on_gate_unavailable: true

critic:
  profile: sam2_seeded_rules
  formal_profile_required: true
  allow_rules_fallback: false
  persist_raw_payload_on_error: true

acceptance:
  mode: enforce

  require_decision_physical: true
  require_no_blocking_violations: true
  require_all_scores: true
  require_non_degraded_critic: true

  physics_score: 0.80
  confidence: 0.60
  coverage: 0.50
  semantic_score: 0.85
  original_prompt_semantic_score: 0.85
  quality_score: 0.75

  max_physics_drop: 0.00
  max_semantic_drop: 0.03
  max_original_prompt_semantic_drop: 0.03
  max_quality_drop: 0.05

scorers:
  quality_enabled: true
  semantic_enabled: true
  original_prompt_semantic_enabled: true
  allow_score_substitution: false
~~~

build_backends.py 必须拒绝 active runtime 中的其他 mode：

~~~python
if acceptance_mode != "enforce":
    raise ValueError(
        "V2 active runtime requires acceptance.mode=enforce"
    )
~~~

历史 shadow artifacts 可以继续读取，但新运行不能选择 shadow。

### 11.6 双 Semantic Gate

Prompt Repair 会修改 prompt。只评价视频与 repaired prompt 的一致性，会产生“通过改变题目获得高分”的漏洞。

必须同时保存：

~~~text
semantic_score
= 视频与当前 generation prompt 的一致性

original_prompt_semantic_score
= 视频与用户原始 prompt 的一致性
~~~

两项都必须达到阈值。原 prompt 永久保存在：

~~~python
request.metadata["original_prompt"]
~~~

每次首轮 Gate 和 Re-Gate 都使用同一个 immutable original prompt。

Prompt Repair 后还要审计：

~~~text
original objects preserved
original scene preserved
original action preserved
camera/style preserved
prompt correction block 可追溯
~~~

示例：

~~~text
original:
A red ball falls onto the floor and bounces once.

错误 repaired prompt:
A red ball floats above the floor.
~~~

后者可能与新视频一致，但已经改变用户任务。即使 semantic_score 达标，只要 original_prompt_semantic_score 不达标，Gate 必须 REJECTED。

### 11.7 修复候选的无退化条件

Re-Gate 除绝对阈值外，还必须比较 before/after：

~~~python
physics_delta = after.physics - before.physics
semantic_delta = after.semantic - before.semantic
original_prompt_semantic_delta = (
    after.original_prompt_semantic
    - before.original_prompt_semantic
)
quality_delta = after.quality - before.quality
~~~

最低无退化要求：

~~~text
physics_delta >= 0
semantic_delta >= -0.03
original_prompt_semantic_delta >= -0.03
quality_delta >= -0.05
~~~

Trial successful 还要求：

~~~text
physics_delta > 0
且 after_gate.status == accepted
~~~

由此保证：

- ProPainter 不能通过擦除对象获得物理高分；
- Prompt Repair 不能通过改变原任务获得语义高分；
- Executor subprocess 成功不等于修复成功；
- 部分改善但仍未通过 Gate 的结果不能标 successful。

### 11.8 GateResult 结构

在现有 generators/wanphysics/v2/guardrails.py 内扩展，不新增第二套 Gate：

~~~python
@dataclass(frozen=True)
class GateResult:
    status: str
    accepted: bool

    physics_score: float | None
    confidence: float | None
    coverage: float | None
    semantic_score: float | None
    original_prompt_semantic_score: float | None
    quality_score: float | None

    blocking_violation_count: int
    critic_degraded: bool

    reasons: tuple[str, ...]
    unavailable: tuple[str, ...]
~~~

建议稳定 reason 枚举：

~~~text
decision_not_physical
blocking_violation_present
physics_below_threshold
confidence_below_threshold
coverage_below_threshold
semantic_below_threshold
original_prompt_semantic_below_threshold
quality_below_threshold
physics_regression
semantic_regression
quality_regression
critic_degraded
~~~

建议稳定 unavailable 枚举：

~~~text
critic_report_unavailable
critic_roundtrip_failed
confidence_unavailable
coverage_unavailable
semantic_score_unavailable
original_prompt_semantic_score_unavailable
quality_score_unavailable
required_evidence_unavailable
~~~

动态异常文本只进入 metadata/error，不作为分类标签。

### 11.9 Strict Enforce Runner 分支

#### 11.9.1 首轮 Gate

~~~python
before_gate = evaluate_gate(...)

if before_gate.status == "accepted":
    return accepted_result(current_candidate)

if before_gate.status == "unavailable":
    return evaluation_failed_result(
        reason=before_gate.reasons,
    )

# 只有 rejected 才能进入 Policy
decision = repair_policy.decide(...)
~~~

#### 11.9.2 Re-Gate

~~~python
after_gate = evaluate_gate(...)

if after_gate.status == "accepted":
    return accepted_result(after_candidate)

if after_gate.status == "unavailable":
    return evaluation_failed_result(
        reason=after_gate.reasons,
    )

if round_index + 1 < max_rounds:
    current_candidate = after_candidate
    continue

return max_rounds_result(
    candidate=selector.select(history),
    final_candidate_disposition="best_effort",
)
~~~

必须严格区分：

~~~text
评价失败 != 视频不合格
视频不合格 != Executor 失败
Executor 成功 != 修复成功
partial improvement != accepted
~~~

### 11.10 与三动作 Policy 的接口

只有 Gate 返回 REJECTED 时，才能进入 Repair Policy：

| Gate 结果、capability 与证据 | Policy 自然决策 |
|---|---|
| local violation + strict mask 有效 + ProPainter 可用 | Local Editing |
| broad violation + 有可靠 repair_instruction | Prompt Repair |
| local 但 mask 无效 + 有可靠 instruction，且 Policy 已接收 capability mask | Prompt Repair |
| 没有安全 prompt 修改方式 | Reject |
| 多轮修复但严格 Gate 始终不通过 | Reject |
| Critic/scorer unavailable | EVALUATION_FAILED，不进入 Policy |

Local Editing 不可用时，capability 必须在 Policy 决策前生效：

~~~text
有可靠 instruction
  -> Prompt Repair

无可靠 instruction
  -> Reject
~~~

禁止使用相同 prompt 更换 seed 作为隐式 fallback。

### 11.11 Preflight 必须 fail closed

当前服务器配置 semantic_enabled=false。如果只把 mode 改成 enforce，所有样本都会因为 semantic unavailable 无法接受。因此必须先接通 scorer，再开放 runtime。

build_backends/preflight 在首轮 Wan 生成前检查：

~~~text
acceptance.mode == enforce
Critic backend 可用
SAM2 strict backend 可用
strict report codec 可用
semantic scorer 可用
original-prompt semantic scorer 可用
quality scorer 可用
所需 vLLM 服务可用
所有 Gate 阈值已加载
三动作 Policy/heuristic compatibility 通过
~~~

任意必要组件不可用：

~~~text
preflight.failed=true
runtime 不启动首轮生成
~~~

禁止：

~~~text
semantic unavailable
  -> 用 physics_score 填充

quality unavailable
  -> 使用默认 1.0

confidence unavailable
  -> 使用默认 0.5

coverage unavailable
  -> 忽略该门
~~~

### 11.12 代码修改位置

核心文件：

~~~text
configs/loop_v2.yaml

generators/wanphysics/v2/guardrails.py
generators/wanphysics/v2/runner.py
generators/wanphysics/v2/build_backends.py
generators/wanphysics/v2/preflight.py
generators/wanphysics/v2/artifacts.py

agents/wanphysics/run_videophy2_loop_v2.py

generators/wanphysics/v2/trials.py
schemas/wan_repair_trial_v3.schema.json
~~~

按现有代码情况同步：

~~~text
generators/wanphysics/v2/critic_backend.py
generators/wanphysics/v2/scorers.py
~~~

所有改动扩展现有 evaluate_gate()、GateResult、Runner 和 Artifact，不新建第二套 Gate。

### 11.13 Trial 成功定义

严格 enforce 下 successful 定义为：

~~~text
execution.status == succeeded
且 critic_after 存在
且 after_scores 所有 required 字段存在
且 after_gate.status == accepted
且 physics_gain > 0
且 semantic_drop >= -0.03
且 original_prompt_semantic_drop >= -0.03
且 quality_drop >= -0.05
~~~

否则：

~~~python
successful = False
~~~

failure_reason：

~~~text
after_gate_rejected
after_gate_unavailable
physics_not_improved
semantic_regression
original_prompt_semantic_regression
quality_regression
critic_degraded
missing_required_score
~~~

可以额外保存：

~~~text
repair_improved=true/false
~~~

但 repair_improved=true 不等于 successful=true。它只表示出现部分收益，不能作为严格接受结论。

### 11.14 Reject 与 best candidate

达到 max_rounds 后仍没有候选通过 enforce：

~~~text
不能把 best candidate 标记为 accepted
~~~

结果：

~~~text
final_state=MAX_ROUNDS
accepted=false
final_candidate_disposition=best_effort
best_candidate_id=<实际候选>
~~~

最佳候选只是失败候选中的相对最优，不代表满足发布标准。

Selector 只能比较所有 required 分数完整的候选；Gate UNAVAILABLE 的候选不能成为 best candidate。排序和选择理由必须进入 audit。

### 11.15 审计产物

每次 before Gate 和 Re-Gate 保存：

~~~json
{
  "mode": "enforce",
  "status": "rejected",
  "accepted": false,
  "scores": {
    "physics_score": 0.76,
    "confidence": 0.72,
    "coverage": 0.68,
    "semantic_score": 0.91,
    "original_prompt_semantic_score": 0.90,
    "quality_score": 0.82
  },
  "thresholds": {
    "physics_score": 0.80,
    "confidence": 0.60,
    "coverage": 0.50,
    "semantic_score": 0.85,
    "original_prompt_semantic_score": 0.85,
    "quality_score": 0.75
  },
  "reasons": [
    "physics_below_threshold"
  ],
  "unavailable": [],
  "critic_degraded": false
}
~~~

进入：

~~~text
loop_result.json
repair_trace.jsonl
trials.jsonl
summary.json
~~~

summary 至少统计：

~~~text
accepted_count
rejected_count
max_rounds_count
gate_unavailable_count
critic_degraded_count
semantic_unavailable_count
original_prompt_semantic_unavailable_count
quality_unavailable_count
~~~

### 11.16 单元测试

Gate：

- 所有分数达标 -> ACCEPTED；
- physics 不达标 -> REJECTED；
- confidence 不达标 -> REJECTED；
- coverage 不达标 -> REJECTED；
- semantic 不达标 -> REJECTED；
- original-prompt semantic 不达标 -> REJECTED；
- quality 不达标 -> REJECTED；
- blocking violation 非空 -> REJECTED；
- critic degraded -> REJECTED；
- 任意 required 分数为 None -> UNAVAILABLE；
- score substitution 被禁止；
- active runtime 配置为 shadow 时构建失败。

Runner：

- ACCEPTED 不调用 Policy；
- REJECTED 调用 Policy 一次；
- UNAVAILABLE 不调用 Policy；
- after Gate 使用 after candidate；
- Re-Gate ACCEPTED 正常终止；
- Re-Gate REJECTED 继续或 Reject；
- Re-Gate UNAVAILABLE 标记 EVALUATION_FAILED；
- max rounds 后 best candidate 仍 accepted=false。

三动作：

- Gate REJECTED 后只能产生 Prompt Repair、Local Editing、Reject；
- Guard 永远不产生第四动作；
- Local mask 不可用时只允许 Prompt Repair 或 Reject；
- 旧四动作 checkpoint 被 compatibility gate 拒绝。

### 11.17 真实 smoke

正式视频 smoke 前：

~~~text
acceptance.mode == enforce
semantic_enabled == true
original_prompt_semantic_enabled == true
quality_enabled == true
fail_on_degraded_critic == true
所有 required scorer preflight 通过
三动作 Policy/heuristic compatibility 通过
~~~

依次验证：

~~~text
1. 首轮直接通过 Enforce 的样本
2. Prompt Repair 后通过 Re-Gate 的样本
3. ProPainter 后通过 Re-Gate 的样本
4. 始终不达标并 Reject 的样本
5. semantic scorer unavailable 的失败样本
6. original-prompt semantic unavailable 的失败样本
7. quality scorer unavailable 的失败样本
8. critic degraded 的失败样本
~~~

### 11.18 完成判据

只有同时满足以下条件，Strict Enforce Gate 才算完成：

- active runtime 不接受 shadow；
- 所有 accepted 候选均通过完整 enforce；
- 所有 required 分数均来自真实 scorer；
- 没有 unavailable 分数被替代；
- 没有 degraded Critic 被接受；
- blocking violation 非空时不能 accepted；
- Prompt Repair 同时通过 current/original prompt semantic；
- ProPainter 通过 Re-Critic 和无退化检查；
- 没有未通过 Gate 的 best candidate 被标 accepted；
- ACCEPTED、REJECTED、UNAVAILABLE 三状态审计完整；
- UNAVAILABLE 不进入 Repair Policy；
- Prompt Repair/ProPainter 都经过同一 Strict Enforce Re-Gate；
- 三动作链路中不存在第四动作；
- 在线接口中不存在 PhysicsPlan。

---

## 12. Phase 7：GPU 与 vLLM owner

### 12.1 双卡目标

~~~text
GPU0：Wan2.2 / ProPainter
GPU1：vLLM / Qwen3-VL
~~~

当前资源指标读取 GPU0 第一行，即使 vLLM 在 GPU1，也可能记录 0 MB。

### 12.2 owner 接线

使用现有 resource_coordinator.py：

~~~text
vllm.pid
vllm.owner.json
stop_owned_vllm(run_dir)
~~~

禁止继续使用：

~~~bash
pkill -9 -f vllm
~~~

当前 owner 缺失的代码原因：V2SubprocessCritic 构造时缓存了旧 start_vllm 回调，后续 monkey patch 没有改变 prepare_hook。应在构造 Critic backend 前完成包装，或显式更新 prepare hook。

owner 至少记录：

~~~text
run_id
pid
port
host
gpu_id
started_at
~~~

### 12.3 端口一致

以下位置必须使用同一个 host/port：

- vLLM Popen
- health URL
- .env BASE_URL
- preflight
- owner manifest
- semantic scorer

### 12.4 单卡回退

如果 GPU 数不足两张，必须真正使用现有 plan_gpu_assignment()：

~~~text
生成前 stop owned vLLM
-> Wan GPU0
-> Wan 子进程退出释放显存
-> start vLLM GPU0
-> Critic
~~~

critic.prepare_for_generation() 必须在每次生成前执行。

---

## 13. Phase 8：Strict ProPainter

当前服务器已有：

~~~text
models/ProPainter/inference_propainter.py
models/ProPainter/weights/raft-things.pth
models/ProPainter/weights/recurrent_flow_completion.pth
models/ProPainter/weights/ProPainter.pth
~~~

保持现有 strict manifest 规则，不允许：

- 单 mask 复制到全部帧。
- 全白 mask fallback。
- mask 缺失时继续编辑。
- SHA 失败时继续编辑。

真实 Local Editing 验收：

~~~text
输入视频存在
-> manifest 合法
-> ProPainter 输出存在
-> 输出视频可解码
-> 分辨率/帧数合理
-> Re-Critic 成功
-> before/after score 真实记录
~~~

如果 ProPainter 只移除对象而破坏 prompt 语义，即使 physics score 提高，也必须通过 semantic/quality 记录副作用。

---

## 14. Phase 9：Trial 与审计修正

### 14.1 RoundRecord 保存真实因果字段

在现有 runner.py 的 RoundRecord 中补充：

~~~text
before_candidate_path
after_candidate_path
before_prompt
after_prompt
critic_before_report
critic_after_report
decision_payload
guard_payload
execution_id
~~~

_assemble_trials() 直接使用这些真实字段，不能重新构造伪 decision。

### 14.2 Trial 内容

真实 WanRepairTrialV3 至少包含：

~~~text
trial_id
group_id
source_candidate
critic_before
decision
execution
before_scores
critic_after
after_scores
physics_gain
repair_improved
successful
failure_reason
generator/critic/policy provenance
~~~

`before_scores` 和 `after_scores` 必须同时包含 physics、semantic、original_prompt_semantic 和 quality。成功定义统一为：

~~~text
execution.status == succeeded
且 critic_after / after_scores 完整
且 after.physics > before.physics
且 semantic / original_prompt_semantic / quality 无超限退化
且 after_gate.status == ACCEPTED
~~~

部分改善只能写 `repair_improved=true, successful=false`，不得作为严格成功 Trial。

### 14.3 单一 Trial 来源与动作来源

所有新 Trial 只能由现有 ActionAwareRunnerV2 的真实 RoundRecord 组装：

~~~text
Strict Gate(REJECTED)
  -> Three-Action Policy 的真实 RepairDecision
  -> PolicyGuard 校验结果
  -> Executor 的真实 ExecutionResult
  -> Re-Critic / Re-Gate
  -> RoundRecord
  -> TrialAssembler
  -> WanRepairTrialV3
~~~

`_assemble_trials()` 不得重新构造 Decision，不得把 Gate 当 Critic，也不得接受 CLI/config/manifest 的动作覆盖。`run_actual_trials_v2.py` 不再产生新版 Trial。历史 forced Trial 只读兼容，不进入新训练和正式统计。

### 14.4 repair trace 与 schema

当前 repair_trace.jsonl 缺少 schema 要求的 round_index/state。每个动作行至少写：

~~~json
{
  "round_index": 0,
  "candidate_id": "before-id",
  "state": "EXECUTED",
  "decision_source": "three_action_policy",
  "policy_action": "prompt_repair",
  "guard_status": "allowed",
  "executed_action": "prompt_repair",
  "execution_status": "succeeded",
  "after_candidate_id": "after-id"
}
~~~

现有 schema 必须真正用于运行后验证：

~~~text
schemas/loop_trace_v2.schema.json
schemas/mask_manifest_v2.schema.json
schemas/wan_repair_trial_v3.schema.json
schemas/critic_output.schema.json
~~~

---

## 15. Phase 10：Resume 与 run 级产物

### 15.1 阶段目标

固定使用以下任务根目录：

~~~text
/root/PhysGenLoop-/outputs/v2_pilot300_real_limit5
~~~

当用户显式传入：

~~~bash
--output-root "$RUN_ROOT"
~~~

该路径直接作为最终 run_root，不再追加 v2_run_<timestamp>。日志、PID、shell 退出状态、run 级产物和所有样本子目录统一位于同一个 RUN_ROOT。

Phase 15 的核心：

~~~text
固定 RUN_ROOT
+ create-only run manifest
+ 样本终态扫描
+ 中断 attempt 隔离
+ 全量 summary 重建
+ 并发锁
~~~

### 15.2 当前代码问题

服务器 revision 569ffb7 当前实现：

~~~python
ts = datetime.now(timezone.utc).strftime(
    "%Y%m%d_%H%M%S"
)
run_root = (
    Path(args.output_root)
    / f"v2_run_{ts}"
)
~~~

因此即使 shell 设置：

~~~bash
RUN_ROOT=/root/PhysGenLoop-/outputs/v2_pilot300_real_limit5
~~~

Python 仍写入：

~~~text
RUN_ROOT/
  v2_run_20260722_HHMMSS/
~~~

当前 Resume 还存在：

1. 每次启动创建新时间戳目录，--resume 扫描的是新空目录；
2. run_manifest.json 在每条样本开始前重复覆盖；
3. summary.json 只包含本次进程实际处理的样本；
4. Resume 跳过的历史完成样本可能从 summary 消失；
5. 中断样本直接复用目录会产生重复 JSONL、候选文件和 execution_id；
6. PID 文件只能人工查看，不能阻止相同 RUN_ROOT 并发运行。

### 15.3 重新定义 --output-root

文件：

~~~text
agents/wanphysics/run_videophy2_loop_v2.py
~~~

参数改为：

~~~python
p.add_argument(
    "--output-root",
    default=None,
    help=(
        "Explicit final run directory. "
        "When omitted, create outputs/v2_run_<timestamp>."
    ),
)
~~~

目录构造：

~~~python
if args.output_root:
    run_root = Path(args.output_root).resolve()
else:
    ts = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )
    run_root = (
        _ROOT
        / "outputs"
        / f"v2_run_{ts}"
    )

run_root.mkdir(
    parents=True,
    exist_ok=True,
)
~~~

语义：

~~~text
显式传入 --output-root
  -> 参数值就是最终 run_root

没有传 --output-root
  -> 自动创建 outputs/v2_run_<timestamp>
~~~

固定启动命令无需增加 --run-dir。

### 15.4 统一目录结构

~~~text
/root/PhysGenLoop-/outputs/v2_pilot300_real_limit5/
├── physgenloop_v2_pilot300_real_limit5.log
├── physgenloop_v2_pilot300_real_limit5.pid
├── physgenloop_v2_pilot300_real_limit5.status
├── run.lock
├── run_manifest.json
├── run_status.json
├── summary.json
├── checkpoint_gate.json
├── vllm.owner.json
│
├── <sample_id_1>/
│   ├── sample_status.json
│   ├── sample_status_history.jsonl
│   ├── active_attempt.json
│   └── attempts/
│       ├── attempt_0001/  # 中断或历史 attempt，保留但不参与统计
│       └── attempt_0002/  # 完成后由 sample_status.authoritative_attempt 指向
│           ├── loop_result.json
│           ├── repair_trace.jsonl
│           ├── trials.jsonl
│           └── resource_metrics.jsonl
│
├── <sample_id_2>/
│   └── ...
└── <sample_id_5>/
    └── ...
~~~

所有文件统一在 RUN_ROOT 下，但样本和 attempt 继续使用子目录，禁止把所有候选、mask 和 Trial 平铺到第一层。新运行的权威执行产物只写入 attempt 目录；样本根目录只保存状态、历史和权威 attempt 指针。已有样本根目录产物保留为 legacy，不删除、不与新 attempt 双写成两套权威事实。

### 15.5 新运行与 Resume 目录保护

shell 会先 mkdir 并创建 log，因此不能以“目录存在”判断 run 是否初始化。唯一初始化标志是：

~~~text
RUN_ROOT/run_manifest.json
~~~

新运行：

~~~python
manifest_path = run_root / "run_manifest.json"

if not args.resume and manifest_path.exists():
    raise RuntimeError(
        "run directory is already initialized; "
        "use --resume to continue it"
    )
~~~

| 状态 | 无 --resume 的行为 |
|---|---|
| 目录不存在 | 创建并启动 |
| 目录存在但无 run_manifest.json，且目标 LOG/PIDFILE 不存在 | 允许初始化 |
| run_manifest.json 已存在 | 拒绝覆盖 |

Resume：

~~~python
if args.resume and not manifest_path.exists():
    raise RuntimeError(
        "--resume requires an initialized "
        "run_manifest.json"
    )
~~~

--resume 必须复用同一个 RUN_ROOT。

### 15.6 标准首跑命令

以下命令作为固定首跑命令：

~~~bash
cd /root/PhysGenLoop- || exit 1

RUN_ROOT=/root/PhysGenLoop-/outputs/v2_pilot300_real_limit5
LOG="$RUN_ROOT/physgenloop_v2_pilot300_real_limit5.log"
STATUS="$RUN_ROOT/physgenloop_v2_pilot300_real_limit5.status"
PIDFILE="$RUN_ROOT/physgenloop_v2_pilot300_real_limit5.pid"

if [ -e "$RUN_ROOT/run_manifest.json" ] \
   || [ -e "$LOG" ] \
   || [ -e "$PIDFILE" ]; then
  printf "refuse to overwrite initialized run: %s\n" "$RUN_ROOT" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT"

nohup bash -c '
cd /root/PhysGenLoop- || exit 1

PYTHONPATH=/root/PhysGenLoop-:/root/PhysGenLoop-/src \
envs/main/bin/python \
agents/wanphysics/run_videophy2_loop_v2.py \
  --enable \
  --manifest evaluation/manifests/videophy2_pilot300.json \
  --limit 5 \
  --max-rounds 2 \
  --output-root "$1"

status=$?
printf "%s\n" "$status" > "$2"
exit "$status"
' _ "$RUN_ROOT" "$STATUS" > "$LOG" 2>&1 &

PID=$!
printf "%s\n" "$PID" > "$PIDFILE"

printf "pid=%s\n" "$PID"
printf "log=%s\n" "$LOG"
printf "status=%s\n" "$STATUS"
printf "run_root=%s\n" "$RUN_ROOT"
~~~

完成 15.3 修改后，所有 Python 产物直接写入 RUN_ROOT，不再产生时间戳子目录。

### 15.7 标准 Resume 命令

Resume 使用同一命令，仅有两个必要变化：

1. 增加 --resume；
2. 日志重定向使用 >>，禁止截断首跑日志。

~~~bash
cd /root/PhysGenLoop- || exit 1

RUN_ROOT=/root/PhysGenLoop-/outputs/v2_pilot300_real_limit5
LOG="$RUN_ROOT/physgenloop_v2_pilot300_real_limit5.log"
STATUS="$RUN_ROOT/physgenloop_v2_pilot300_real_limit5.status"
PIDFILE="$RUN_ROOT/physgenloop_v2_pilot300_real_limit5.pid"

if [ ! -e "$RUN_ROOT/run_manifest.json" ]; then
  printf "resume requires run_manifest.json: %s\n" "$RUN_ROOT" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT"
printf "RUNNING\n" > "$STATUS"

nohup bash -c '
cd /root/PhysGenLoop- || exit 1

PYTHONPATH=/root/PhysGenLoop-:/root/PhysGenLoop-/src \
envs/main/bin/python \
agents/wanphysics/run_videophy2_loop_v2.py \
  --enable \
  --manifest evaluation/manifests/videophy2_pilot300.json \
  --limit 5 \
  --max-rounds 2 \
  --output-root "$1" \
  --resume

status=$?
printf "%s\n" "$status" > "$2"
exit "$status"
' _ "$RUN_ROOT" "$STATUS" >> "$LOG" 2>&1 &

PID=$!
printf "%s\n" "$PID" > "$PIDFILE"

printf "pid=%s\n" "$PID"
printf "log=%s\n" "$LOG"
printf "status=%s\n" "$STATUS"
printf "run_root=%s\n" "$RUN_ROOT"
~~~

Resume 继续使用 > "$LOG" 会清空原日志，必须禁止。

### 15.8 run_manifest 只创建一次

当前 artifacts.write_run_manifest() 位于样本循环内，应移动到循环前。

~~~python
root_artifacts = RunArtifacts(run_root)

if not args.resume:
    root_artifacts.create_run_manifest(
        {
            "schema_version": "v2-run-manifest/2.0",
            "run_id": run_root.name,
            "run_root": str(run_root),
            "created_at": utc_now(),
            "entrypoint": (
                "agents/wanphysics/run_videophy2_loop_v2.py"
            ),
            "decision_source": "three_action_policy",
            "project_root": str(_ROOT),
            "source_revision": git_revision,
            "config_path": str(args.config),
            "config_sha256": config_sha256,
            "manifest_path": str(args.manifest),
            "manifest_sha256": manifest_sha256,
            "sample_ids": sample_ids,
            "limit": args.limit,
            "max_rounds": max_rounds,
            "acceptance_mode": "enforce",
            "action_order": [
                "prompt_repair",
                "local_editing",
                "reject",
            ],
        }
    )
~~~

create_run_manifest() 使用 create-only：

~~~python
if path.exists():
    raise FileExistsError(
        f"run manifest already exists: {path}"
    )
~~~

run_manifest.json 是不可变实验身份，不允许每个样本覆盖，也不得保存 `state=RUNNING` 等动态字段。运行状态进入 `run_status.json`；样本 prompt、preflight 和 attempt 信息进入样本目录。新 manifest 不包含 force/override action 字段。

### 15.9 Resume 兼容性检查

Resume 前必须核对：

~~~text
manifest path/hash
sample_ids
limit
config path/hash
max_rounds
source revision
acceptance mode
action order
generator model/revision
critic profile/revision
~~~

~~~python
validate_resume_compatibility(
    saved_manifest,
    current={
        "manifest_sha256": manifest_sha256,
        "sample_ids": sample_ids,
        "limit": args.limit,
        "config_sha256": config_sha256,
        "max_rounds": max_rounds,
        "source_revision": git_revision,
        "acceptance_mode": "enforce",
        "action_order": [
            "prompt_repair",
            "local_editing",
            "reject",
        ],
    },
)
~~~

任何关键字段不一致：

~~~text
RESUME_COMPATIBILITY_FAILED
~~~

并立即停止，不能向原 run 混写。首跑 limit=5、max_rounds=2，Resume 必须保持一致。

### 15.10 样本终态与 Resume 规则

pending_samples() 继续读取：

~~~text
RUN_ROOT/<sample_id>/sample_status.json
~~~

终态集合对齐 Strict Enforce：

~~~text
ACCEPTED
REJECTED
MAX_ROUNDS
EVALUATION_FAILED
EXECUTION_FAILED
PREFLIGHT_FAILED
~~~

| sample 状态 | Resume 行为 |
|---|---|
| ACCEPTED | 跳过 |
| REJECTED | 跳过 |
| MAX_ROUNDS | 跳过；相对最佳候选只通过 final_candidate_disposition=best_effort 表达 |
| EVALUATION_FAILED | 默认跳过；显式 retry-failed 才重试 |
| EXECUTION_FAILED | 默认跳过；显式 retry-failed 才重试 |
| PREFLIGHT_FAILED | 默认跳过；显式 retry-failed 才重试 |
| GENERATING/CRITIC_RUNNING/EXECUTING/RE_EVALUATING | 中断样本，创建新 attempt 后重试 |
| 没有 sample_status.json | 执行 |

Resume 主逻辑：

~~~python
pending = set(
    pending_samples(
        run_root,
        sample_ids,
        retry_failed=args.retry_failed,
    )
)
done = set(sample_ids) - pending
~~~

### 15.11 中断 attempt 隔离

中断状态可能包括：

~~~text
GENERATING
CRITIC_RUNNING
EXECUTING
RE_EVALUATING
~~~

直接写回同一目录会导致：

~~~text
trials.jsonl 重复
repair_trace.jsonl 重复
candidate 文件混杂
execution_id 冲突
~~~

目标结构：

~~~text
RUN_ROOT/<sample_id>/
├── sample_status.json
├── sample_status_history.jsonl
├── active_attempt.json
└── attempts/
    ├── attempt_0001/
    │   └── 原中断产物
    └── attempt_0002/
        └── Resume 新产物
~~~

active_attempt.json：

~~~json
{
  "attempt_id": "attempt_0002",
  "resume_count": 1,
  "started_at": "...",
  "reason": "resume_after_interruption"
}
~~~

attempt 完成后必须原子更新 `sample_status.json`：

~~~json
{
  "state": "ACCEPTED",
  "active_attempt": "attempt_0002",
  "authoritative_attempt": "attempt_0002"
}
~~~

只有完整落盘且通过 schema 校验的 attempt 才能成为 authoritative；中断或半成品 attempt 永不参与 summary。summary 不按目录修改时间猜测，也不合并两个 attempt 的 JSONL。

Trial ID 和 execution ID 带 attempt：

~~~text
sample-id-attempt-0002-exec1
~~~

历史中断产物保留，不删除、不覆盖。

attempt 子目录和 authoritative pointer 是本方案的完成条件，不再接受直接复用样本根目录作为正式实现。历史根目录产物仅保留兼容读取。

#### 15.11.1 `--retry-failed` 明确定义

正式入口增加 `--retry-failed`，且只允许与 `--resume` 同时使用：

~~~text
默认 Resume：EVALUATION_FAILED、EXECUTION_FAILED、PREFLIGHT_FAILED 均跳过
--resume --retry-failed：只重试上述三种失败终态，并创建新 attempt
ACCEPTED、REJECTED、MAX_ROUNDS：始终跳过
~~~

重试不得覆盖旧失败证据；不增加其他隐式重跑开关。

### 15.12 summary 全量重建

当前 summaries 列表只包含本次进程处理的样本。Resume 结束时应扫描整个 RUN_ROOT：

~~~python
summary = rebuild_summary(
    run_root=run_root,
    sample_ids=sample_ids,
)
~~~

读取每个样本 `sample_status.json` 指向的 authoritative attempt：

~~~text
sample_status.json
loop_result.json
trials.jsonl
resource_metrics.jsonl
~~~

目标：

~~~json
{
  "run_id": "v2_pilot300_real_limit5",
  "total_samples": 5,
  "accepted": 2,
  "rejected": 1,
  "max_rounds": 1,
  "evaluation_failed": 0,
  "pending": 1,
  "completed_sample_ids": [],
  "pending_sample_ids": [],
  "resume_count": 1,
  "last_updated_at": "..."
}
~~~

summary.json 是可重建快照，使用原子写更新；run_manifest.json 是不可变事实，禁止更新。

### 15.13 并发运行保护

PIDFILE 只用于人工查看，不能作为可靠锁。Python 入口使用：

~~~text
RUN_ROOT/run.lock
~~~

通过 fcntl.flock() 获得排他锁：

~~~text
获得锁
  -> 允许启动或 Resume

锁已被其他进程持有
  -> 立即退出
  -> 不修改样本产物
~~~

锁应覆盖 manifest 检查、样本执行、summary 重建和 run_status 更新的整个进程生命周期。

### 15.14 run 级状态

结构化状态：

~~~text
RUN_ROOT/run_status.json
~~~

状态集合：

~~~text
INITIALIZING
RUNNING
RESUMING
COMPLETED
COMPLETED_WITH_REJECTIONS
FAILED
INTERRUPTED
~~~

示例：

~~~json
{
  "schema_version": "v2-run-status/1.0",
  "run_id": "v2_pilot300_real_limit5",
  "state": "RESUMING",
  "pid": 12345,
  "started_at": "...",
  "updated_at": "...",
  "resume_count": 1,
  "completed_samples": 3,
  "pending_samples": 2,
  "last_error": null
}
~~~

shell 文件：

~~~text
physgenloop_v2_pilot300_real_limit5.status
~~~

只保存最终进程退出码：

~~~text
0
1
...
~~~

职责：

~~~text
.status
  -> shell/process 退出码

run_status.json
  -> 业务运行状态
~~~

### 15.15 修改位置

只扩展现有责任文件：

~~~text
agents/wanphysics/run_videophy2_loop_v2.py
generators/wanphysics/v2/artifacts.py
generators/wanphysics/v2/runner.py
tests/wanphysics_v2/test_artifacts.py
tests/wanphysics_v2/test_runner.py
~~~

不新增第二套 run manager 或 resume runner。

### 15.16 测试要求

必须覆盖：

- 显式 --output-root 直接作为最终目录；
- 未传 --output-root 时仍自动创建时间戳目录；
- shell 预创建空目录不阻止首跑；
- fresh run 在启动 nohup 前检查 manifest/log/pid，不能先截断旧日志；
- 已有 run_manifest 且无 --resume 时拒绝覆盖；
- --resume 但无 run_manifest 时失败；
- run_manifest 只创建一次；
- Resume 参数/hash/action schema 不一致时失败；
- terminal 样本跳过；
- --retry-failed 只重试 EVALUATION_FAILED、EXECUTION_FAILED、PREFLIGHT_FAILED；
- 非终态样本创建新 attempt；
- summary 只读取 sample_status 指向的 authoritative attempt；
- execution_id/trial_id 不重复；
- summary 包含首跑和 Resume 的全部样本；
- 同一 RUN_ROOT 第二进程无法获得锁；
- run_status 状态转换合法；
- JSON 快照原子写；
- JSONL append-only；
- 历史中断 attempt 不删除。

### 15.17 最终验收

必须同时满足：

- 固定首跑命令无需增加 --run-dir；
- 显式 --output-root "$RUN_ROOT" 直接作为最终 run 目录；
- 不再创建 RUN_ROOT/v2_run_<timestamp>；
- log、pid、status、manifest、summary 和样本目录全部位于 RUN_ROOT；
- fresh run 的 shell guard 在日志重定向前拒绝覆盖已有 manifest/log/pid；
- 无 --resume 时拒绝覆盖已初始化 run；
- --resume 复用同一 RUN_ROOT；
- Resume 日志使用 >>，不截断原日志；
- run_manifest 只创建一次；
- Resume 前验证 manifest/config/code/action schema；
- terminal 样本不重跑；
- `--resume --retry-failed` 只重试三类失败终态；
- 中断样本安全重试；
- 每个完成样本只有一个 authoritative attempt 参与 summary；
- execution_id/trial_id 不重复；
- summary 根据所有样本全量重建；
- 相同 RUN_ROOT 不能并发运行；
- 所有 JSON 快照使用原子写；
- JSONL 保持 append-only；
- 历史中断产物不删除。

---

## 16. 测试矩阵

本节所有测试统一使用 Phase 15 的任务包装：

~~~text
一个测试任务
  -> 一个固定 RUN_ROOT
  -> 一个 LOG
  -> 一个 STATUS
  -> 一个 PIDFILE
  -> nohup bash -c
  -> 显式 PYTHONPATH
  -> 显式 --output-root "$RUN_ROOT"
~~~

所有真实运行只调用 `run_videophy2_loop_v2.py`：显式传入 `--output-root` 时，该路径就是最终 run 目录，不再追加 `v2_run_<timestamp>`。`run_actual_trials_v2.py` 已退役，不再属于测试矩阵或目录语义验收。

### 16.1 CPU 测试

~~~bash
cd /root/PhysGenLoop- || exit 1

RUN_ROOT=/root/PhysGenLoop-/outputs/v2_test_cpu
LOG="$RUN_ROOT/physgenloop_v2_test_cpu.log"
STATUS="$RUN_ROOT/physgenloop_v2_test_cpu.status"
PIDFILE="$RUN_ROOT/physgenloop_v2_test_cpu.pid"

if [ -e "$RUN_ROOT/run_manifest.json" ] \
   || [ -e "$LOG" ] \
   || [ -e "$PIDFILE" ]; then
  printf "refuse to overwrite initialized test run: %s\n" "$RUN_ROOT" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT"

nohup bash -c '
cd /root/PhysGenLoop- || exit 1

PYTHONPATH=/root/PhysGenLoop-:/root/PhysGenLoop-/src \
envs/main/bin/python -m pytest \
  tests/wanphysics_v2 \
  -q

status=$?

if [ "$status" -eq 0 ]; then
  PYTHONPATH=/root/PhysGenLoop-:/root/PhysGenLoop-/src \
  envs/main/bin/python \
  agents/wanphysics/run_videophy2_loop_v2.py \
    --dry-run \
    --output-root "$1"
  status=$?
fi

printf "%s\n" "$status" > "$2"
exit "$status"
' _ "$RUN_ROOT" "$STATUS" > "$LOG" 2>&1 &

PID=$!
printf "%s\n" "$PID" > "$PIDFILE"

printf "pid=%s\n" "$PID"
printf "log=%s\n" "$LOG"
printf "status=%s\n" "$STATUS"
printf "run_root=%s\n" "$RUN_ROOT"
~~~

必须覆盖：

- Strict Enforce Gate 三状态；active runtime 配置为 shadow 时必须构建失败。
- Runner/Generator/Executor 在线接口无 PhysicsPlan。
- 旧版 PromptRepairExecutor。
- Policy 每轮只调用一次。
- RepairAction、Policy head 和 capability mask 均为三动作。
- Policy 根据 Critic/Gate/capability 在可执行三动作内决策；Guard 只校验，不覆盖动作。
- Local mask strict failure。
- Reject terminal。
- Re-Critic 后不重复评估。
- Gate 的 ACCEPTED/REJECTED/UNAVAILABLE 与六类样本终态映射一致。
- Trial 保存真实 decision/report/path。
- 新 Trial 只来自真实 Policy Decision 和 RoundRecord，`successful` 必须要求 Strict Re-Gate ACCEPTED。
- 双 Semantic 分数进入 ScoreBundle、Gate、Trial、delta 和 schema。
- 正式 CLI 拒绝 `--force-action` 与 `--allow-proxy-policy`。
- resume 使用原 run dir。
- Memory 不影响 Policy。
- schema validation。
- 旧四动作 checkpoint 因 action_order mismatch 被拒绝。

查看结果：

~~~bash
RUN_ROOT=/root/PhysGenLoop-/outputs/v2_test_cpu
cat "$RUN_ROOT/physgenloop_v2_test_cpu.status"
tail -n 200 "$RUN_ROOT/physgenloop_v2_test_cpu.log"
find "$RUN_ROOT" -maxdepth 4 -type f | sort
~~~

通过条件：

~~~text
STATUS=0
pytest 无 failed/error
dry-run 完成
没有创建 RUN_ROOT/v2_run_<timestamp>
~~~

### 16.2 单样本真实 smoke

~~~bash
cd /root/PhysGenLoop- || exit 1

RUN_ROOT=/root/PhysGenLoop-/outputs/v2_smoke_enforce_limit1
LOG="$RUN_ROOT/physgenloop_v2_smoke_enforce_limit1.log"
STATUS="$RUN_ROOT/physgenloop_v2_smoke_enforce_limit1.status"
PIDFILE="$RUN_ROOT/physgenloop_v2_smoke_enforce_limit1.pid"

if [ -e "$RUN_ROOT/run_manifest.json" ] \
   || [ -e "$LOG" ] \
   || [ -e "$PIDFILE" ]; then
  printf "refuse to overwrite initialized run: %s\n" "$RUN_ROOT" >&2
  exit 2
fi

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

printf "pid=%s\n" "$PID"
printf "log=%s\n" "$LOG"
printf "status=%s\n" "$STATUS"
printf "run_root=%s\n" "$RUN_ROOT"
~~~

检查：

~~~bash
RUN_ROOT=/root/PhysGenLoop-/outputs/v2_smoke_enforce_limit1

cat "$RUN_ROOT/physgenloop_v2_smoke_enforce_limit1.status"
tail -n 200 "$RUN_ROOT/physgenloop_v2_smoke_enforce_limit1.log"
find "$RUN_ROOT" -maxdepth 6 -type f | sort
cat "$RUN_ROOT/run_manifest.json"
cat "$RUN_ROOT/run_status.json"
cat "$RUN_ROOT/summary.json"

find "$RUN_ROOT" -name loop_result.json -print -exec cat {} \;
find "$RUN_ROOT" -name repair_trace.jsonl -print -exec cat {} \;
find "$RUN_ROOT" -name trials.jsonl -print -exec cat {} \;
~~~

通过条件：

- shell STATUS 为 0；
- active acceptance mode 为 enforce；
- Strict Gate required scorers 全部 available；
- run_manifest、run_status、summary 位于 RUN_ROOT 第一层；
- 样本产物位于 RUN_ROOT/<sample_id>/；
- 不存在额外 v2_run_<timestamp> 子目录；
- 所有 accepted 候选有完整 Strict Enforce Gate 证据；
- REJECTED/UNAVAILABLE 分支状态与第 11 节一致。

### 16.3 自然三动作决策 smoke

不再运行 forced trial，也不再调用 `run_actual_trials_v2.py`。先用 Policy 单元测试分别构造三类 Critic/Gate/capability 输入；真实 smoke 再使用精选样本让 Policy 自然选择动作。测试 manifest 只能选择输入样本，不能把 `expected_action` 注入 runtime。

~~~bash
cd /root/PhysGenLoop- || exit 1

RUN_ROOT=/root/PhysGenLoop-/outputs/v2_natural_three_action_smoke
LOG="$RUN_ROOT/physgenloop_v2_natural_three_action_smoke.log"
STATUS="$RUN_ROOT/physgenloop_v2_natural_three_action_smoke.status"
PIDFILE="$RUN_ROOT/physgenloop_v2_natural_three_action_smoke.pid"

if [ -e "$RUN_ROOT/run_manifest.json" ] \
   || [ -e "$LOG" ] \
   || [ -e "$PIDFILE" ]; then
  printf "refuse to overwrite initialized run: %s\n" "$RUN_ROOT" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT"

nohup bash -c '
cd /root/PhysGenLoop- || exit 1

PYTHONPATH=/root/PhysGenLoop-:/root/PhysGenLoop-/src \
envs/main/bin/python \
agents/wanphysics/run_videophy2_loop_v2.py \
  --enable \
  --manifest evaluation/manifests/videophy2_smoke20.json \
  --limit 20 \
  --max-rounds 2 \
  --output-root "$1"

status=$?
printf "%s\n" "$status" > "$2"
exit "$status"
' _ "$RUN_ROOT" "$STATUS" > "$LOG" 2>&1 &

PID=$!
printf "%s\n" "$PID" > "$PIDFILE"

printf "pid=%s\n" "$PID"
printf "log=%s\n" "$LOG"
printf "status=%s\n" "$STATUS"
printf "run_root=%s\n" "$RUN_ROOT"
~~~

验收：

| 自然动作 | 必须观察到 |
|---|---|
| prompt_repair | Policy 真实选择；旧版 rewriter 修改 prompt；Wan 生成 after；Re-Critic/Re-Gate |
| local_editing | Policy 真实选择；strict SAM2 manifest；ProPainter after；Re-Critic/Re-Gate |
| reject | Policy 真实选择或 Guard blocked 后 Audited Reject；不产生 after；终态 REJECTED |

如果本批真实样本没有自然覆盖某一动作，该动作的真实 smoke 判为未完成，应补充经过只读筛选的输入样本后重新运行，不能用动作覆盖绕过。Local mask 不存在时必须记录 Guard blocked/Audited Reject，不能伪造 mask 或成功 Trial。

检查：

~~~bash
RUN_ROOT=/root/PhysGenLoop-/outputs/v2_natural_three_action_smoke

cat "$RUN_ROOT/physgenloop_v2_natural_three_action_smoke.status"
tail -n 300 "$RUN_ROOT/physgenloop_v2_natural_three_action_smoke.log"
find "$RUN_ROOT" -maxdepth 8 -type f | sort
find "$RUN_ROOT" -name repair_trace.jsonl -print -exec cat {} \;
find "$RUN_ROOT" -name trials.jsonl -print -exec cat {} \;
~~~

通过条件：

- STATUS=0；
- 所有 Decision 的 `decision_source=three_action_policy`；
- 不存在 force/override action 字段；
- runtime 产物只包含 prompt_repair、local_editing、reject；
- 每个 execution 恰好对应一条由 RoundRecord 组装的 Trial；
- 每个修复 after 都有真实 Re-Critic/Re-Gate；
- `successful=true` 的 Trial 全部满足 after gate ACCEPTED；
- 被删除动作和旧 proxy policy 不出现在 runtime 产物。

### 16.4 Resume smoke

#### 16.4.1 首跑

~~~bash
cd /root/PhysGenLoop- || exit 1

RUN_ROOT=/root/PhysGenLoop-/outputs/v2_resume_smoke
LOG="$RUN_ROOT/physgenloop_v2_resume_smoke.log"
STATUS="$RUN_ROOT/physgenloop_v2_resume_smoke.status"
PIDFILE="$RUN_ROOT/physgenloop_v2_resume_smoke.pid"

if [ -e "$RUN_ROOT/run_manifest.json" ] \
   || [ -e "$LOG" ] \
   || [ -e "$PIDFILE" ]; then
  printf "refuse to overwrite initialized run: %s\n" "$RUN_ROOT" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT"

nohup bash -c '
cd /root/PhysGenLoop- || exit 1

PYTHONPATH=/root/PhysGenLoop-:/root/PhysGenLoop-/src \
envs/main/bin/python \
agents/wanphysics/run_videophy2_loop_v2.py \
  --enable \
  --manifest evaluation/manifests/videophy2_smoke_dev10.json \
  --limit 2 \
  --max-rounds 2 \
  --output-root "$1"

status=$?
printf "%s\n" "$status" > "$2"
exit "$status"
' _ "$RUN_ROOT" "$STATUS" > "$LOG" 2>&1 &

PID=$!
printf "%s\n" "$PID" > "$PIDFILE"

printf "pid=%s\n" "$PID"
printf "log=%s\n" "$LOG"
printf "status=%s\n" "$STATUS"
printf "run_root=%s\n" "$RUN_ROOT"
~~~

在至少一个样本终态落盘、另一个样本仍处于非终态时进行受控中断。中断前保存：

~~~bash
RUN_ROOT=/root/PhysGenLoop-/outputs/v2_resume_smoke

find "$RUN_ROOT" -name sample_status.json -print -exec cat {} \;
cat "$RUN_ROOT/run_manifest.json"
cat "$RUN_ROOT/physgenloop_v2_resume_smoke.pid"
~~~

#### 16.4.2 Resume

Resume 必须使用相同 RUN_ROOT、相同 manifest、limit、max_rounds、配置和代码 revision；日志使用追加模式：

~~~bash
cd /root/PhysGenLoop- || exit 1

RUN_ROOT=/root/PhysGenLoop-/outputs/v2_resume_smoke
LOG="$RUN_ROOT/physgenloop_v2_resume_smoke.log"
STATUS="$RUN_ROOT/physgenloop_v2_resume_smoke.status"
PIDFILE="$RUN_ROOT/physgenloop_v2_resume_smoke.pid"

if [ ! -e "$RUN_ROOT/run_manifest.json" ]; then
  printf "resume requires run_manifest.json: %s\n" "$RUN_ROOT" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT"
printf "RUNNING\n" > "$STATUS"

nohup bash -c '
cd /root/PhysGenLoop- || exit 1

PYTHONPATH=/root/PhysGenLoop-:/root/PhysGenLoop-/src \
envs/main/bin/python \
agents/wanphysics/run_videophy2_loop_v2.py \
  --enable \
  --manifest evaluation/manifests/videophy2_smoke_dev10.json \
  --limit 2 \
  --max-rounds 2 \
  --output-root "$1" \
  --resume

status=$?
printf "%s\n" "$status" > "$2"
exit "$status"
' _ "$RUN_ROOT" "$STATUS" >> "$LOG" 2>&1 &

PID=$!
printf "%s\n" "$PID" > "$PIDFILE"

printf "pid=%s\n" "$PID"
printf "log=%s\n" "$LOG"
printf "status=%s\n" "$STATUS"
printf "run_root=%s\n" "$RUN_ROOT"
~~~

检查：

~~~bash
RUN_ROOT=/root/PhysGenLoop-/outputs/v2_resume_smoke

cat "$RUN_ROOT/physgenloop_v2_resume_smoke.status"
tail -n 300 "$RUN_ROOT/physgenloop_v2_resume_smoke.log"
cat "$RUN_ROOT/run_manifest.json"
cat "$RUN_ROOT/run_status.json"
cat "$RUN_ROOT/summary.json"
find "$RUN_ROOT" -name sample_status.json -print -exec cat {} \;
find "$RUN_ROOT" -path '*/attempts/*' -type f | sort
~~~

通过条件：

- Resume 未创建第二个 run root；
- run_manifest 未被覆盖；
- 已终止样本没有再次生成；
- 中断样本创建新 attempt 或按 execution_id 幂等恢复；
- trials.jsonl 和 repair_trace.jsonl 无重复 execution_id；
- Resume 使用 >> 后首跑日志仍存在；
- summary 包含首跑与恢复后的全部样本；
- resume_count 和 run_status 正确。

#### 16.4.3 Retry failed

先在测试 fixture 中保留至少一个 `EVALUATION_FAILED`、`EXECUTION_FAILED` 或 `PREFLIGHT_FAILED` 样本，然后继续使用同一 RUN_ROOT 和追加日志：

~~~bash
cd /root/PhysGenLoop- || exit 1

RUN_ROOT=/root/PhysGenLoop-/outputs/v2_resume_smoke
LOG="$RUN_ROOT/physgenloop_v2_resume_smoke.log"
STATUS="$RUN_ROOT/physgenloop_v2_resume_smoke.status"
PIDFILE="$RUN_ROOT/physgenloop_v2_resume_smoke.pid"

if [ ! -e "$RUN_ROOT/run_manifest.json" ]; then
  printf "retry-failed requires run_manifest.json: %s\n" "$RUN_ROOT" >&2
  exit 2
fi

printf "RUNNING\n" > "$STATUS"

nohup bash -c '
cd /root/PhysGenLoop- || exit 1

PYTHONPATH=/root/PhysGenLoop-:/root/PhysGenLoop-/src \
envs/main/bin/python \
agents/wanphysics/run_videophy2_loop_v2.py \
  --enable \
  --manifest evaluation/manifests/videophy2_smoke_dev10.json \
  --limit 2 \
  --max-rounds 2 \
  --output-root "$1" \
  --resume \
  --retry-failed

status=$?
printf "%s\n" "$status" > "$2"
exit "$status"
' _ "$RUN_ROOT" "$STATUS" >> "$LOG" 2>&1 &

PID=$!
printf "%s\n" "$PID" > "$PIDFILE"

printf "pid=%s\n" "$PID"
printf "log=%s\n" "$LOG"
printf "status=%s\n" "$STATUS"
printf "run_root=%s\n" "$RUN_ROOT"
~~~

通过条件：失败终态样本创建新 attempt；ACCEPTED、REJECTED、MAX_ROUNDS 不重跑；旧失败证据保留；summary 只统计新 authoritative attempt。

### 16.5 全量 pilot

所有 smoke 通过后：

~~~bash
cd /root/PhysGenLoop- || exit 1

RUN_ROOT=/root/PhysGenLoop-/outputs/v2_pilot300_real
LOG="$RUN_ROOT/physgenloop_v2_pilot300_real.log"
STATUS="$RUN_ROOT/physgenloop_v2_pilot300_real.status"
PIDFILE="$RUN_ROOT/physgenloop_v2_pilot300_real.pid"

if [ -e "$RUN_ROOT/run_manifest.json" ] \
   || [ -e "$LOG" ] \
   || [ -e "$PIDFILE" ]; then
  printf "refuse to overwrite initialized run: %s\n" "$RUN_ROOT" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT"

nohup bash -c '
cd /root/PhysGenLoop- || exit 1

PYTHONPATH=/root/PhysGenLoop-:/root/PhysGenLoop-/src \
envs/main/bin/python \
agents/wanphysics/run_videophy2_loop_v2.py \
  --enable \
  --manifest evaluation/manifests/videophy2_pilot300.json \
  --limit 300 \
  --max-rounds 2 \
  --output-root "$1"

status=$?
printf "%s\n" "$status" > "$2"
exit "$status"
' _ "$RUN_ROOT" "$STATUS" > "$LOG" 2>&1 &

PID=$!
printf "%s\n" "$PID" > "$PIDFILE"

printf "pid=%s\n" "$PID"
printf "log=%s\n" "$LOG"
printf "status=%s\n" "$STATUS"
printf "run_root=%s\n" "$RUN_ROOT"
~~~

监控：

~~~bash
RUN_ROOT=/root/PhysGenLoop-/outputs/v2_pilot300_real

cat "$RUN_ROOT/physgenloop_v2_pilot300_real.pid"
cat "$RUN_ROOT/physgenloop_v2_pilot300_real.status"
tail -f "$RUN_ROOT/physgenloop_v2_pilot300_real.log"
~~~

最终验收：

~~~bash
RUN_ROOT=/root/PhysGenLoop-/outputs/v2_pilot300_real

cat "$RUN_ROOT/physgenloop_v2_pilot300_real.status"
cat "$RUN_ROOT/run_status.json"
cat "$RUN_ROOT/summary.json"
find "$RUN_ROOT" -name sample_status.json | wc -l
find "$RUN_ROOT" -name trials.jsonl -print
~~~

pilot300 只有在 CPU、单样本、自然三动作和 Resume smoke 全部通过后才能启动。任何前置测试 STATUS 非 0、Strict Gate required scorer unavailable、目录语义不一致或 schema 校验失败，都必须阻止全量运行。

---

## 17. 全链路完成判据

只有同时满足以下条件，才能称为完整闭环。

### 17.1 功能

- Runner、Generator、Executor 和 ProPainter 在线接口不再透传 PhysicsPlan。
- Wan2.2 生成真实候选。
- SAM2/Critic 真实评价。
- CriticReport 无损恢复。
- Active runtime 只允许 Strict Enforce；历史 shadow artifacts 仅可读取。
- Policy 每轮只决策一次。
- 正式运行只有 `run_videophy2_loop_v2.py` 一个入口；CLI、配置和 manifest 均不能覆盖动作。
- Repair Policy 动作集合严格为 Prompt Repair、Local Editing、Reject。
- Prompt Repair 使用旧版 PromptRepairExecutor。
- 不存在相同 prompt 只换 seed 的修复动作或隐式 fallback。
- Local 使用 strict per-frame mask。
- Reject 返回历史最佳候选。
- after candidate 真实 Re-Critic/Re-Gate。
- 每个终态一致结束。

### 17.2 审计

- 每次 action 有 execution_id。
- before/action/after 一一对应。
- before/after CriticReport 真实保存。
- prompt、seed、candidate path、mask path 可追溯。
- Trial 不伪造概率、分数或状态。
- Trial 保存真实 Policy Decision、Guard 结果、双 Semantic、Re-Critic 和 Strict Re-Gate；部分改善与严格 successful 分开。
- schema 验证结果可查询。
- GPU metrics 对应实际 GPU。
- vLLM 只停止本 run 拥有的 PID。

### 17.3 研究边界

- Memory 不再是在线阶段。
- 旧四动作 proxy checkpoint 标记 research_only、legacy incompatible，且不进入三动作 runtime。
- actual_trial_count=0 时不宣称 production policy。
- ProPainter 效果由 Re-Critic 证明。
- 历史 shadow Gate 产物不作为严格质量结论，也不得混入新 accepted 统计。
- 当前 Qwen3-VL 角色准确描述为对象种子，不夸大为完整物理 VLM verifier。

---

## 18. 推荐执行顺序

~~~text
1. 冻结并记录服务器状态与 source revision
2. 固化共享 contracts、三动作 action_order 和 schema 版本
3. 废弃 Memory 在线路径
4. 删除主链路 PhysicsPlan 参数与 plan Gate
5. 解耦 pavg_critic 内部 PhysicsPlan 依赖
6. 将 RepairAction/Policy/Guard/Capability 收敛为三动作
7. 停止加载旧四动作 checkpoint，启用三动作 heuristic policy
8. 退役 `--force-action` 与 `run_actual_trials_v2.py` 活跃入口
9. 恢复无 PhysicsPlan 的旧版 PromptRepairExecutor 接线
10. 验证 strict SAM2 manifest + StrictProPainterLocalEditor + ProPainter
11. 补齐 semantic 与 original_prompt_semantic 双评分链路
12. 实现唯一 Strict Enforce Gate 和三状态
13. 修复 Runner 重复评估、六类终态和异常处理
14. 修复 RoundRecord、Trial、repair trace 和三动作 schema
15. 修复 SAM2 `_C`、Critic degraded/fallback 和 critic.json
16. 接通 vLLM owner、GPU 指标和单卡交接
17. 修复固定 RUN_ROOT、不可变 manifest、attempt、Resume 与 retry-failed
18. CPU/contracts/schema/Policy/Executor 测试
19. 单样本 Strict Gate 真实 smoke
20. 自然 Prompt Repair smoke
21. 自然 Local Editing smoke
22. 自然 Reject smoke
23. Resume/retry-failed smoke
24. limit=5 pilot
25. pilot300
~~~

每个阶段只修改现有责任模块，并先通过最小测试再进入下一阶段。未经用户授权，不在服务器执行上述代码修改和运行命令。

---

## 19. 服务器实施与统一验证结果（2026-07-22）

### 19.1 实施结论

本方案已在服务器 `/root/PhysGenLoop-` 的现有 V2 代码上实施。本次没有新增重复 Runner、Gate 或 Executor，也没有删除历史 worklog、既有输出或团队成员文件。正式运行契约已经收敛为：

~~~text
Prompt
  -> Wan 首轮生成
  -> SAM2 + Qwen3-VL Critic
  -> Strict Enforce Acceptance Gate
       ACCEPTED  -> 输出最终候选
       REJECTED  -> Three-Action Repair Policy
                    -> Prompt Repair / ProPainter Local Editing / Reject
       UNAVAILABLE -> EVALUATION_FAILED，不进入 Repair Policy
  -> 修复候选 Re-Critic
  -> Strict Enforce Re-Gate
  -> WanRepairTrialV3 / Audit
~~~

已经完成的代码对齐包括：

- 正式入口唯一保留 `agents/wanphysics/run_videophy2_loop_v2.py`；正式 CLI 移除 `--force-action` 和 `--allow-proxy-policy`。
- 在线动作集合严格收敛为 `prompt_repair`、`local_editing`、`reject`，Global Regen 与 Memory 不再进入 build/runtime。
- Prompt Repair 使用原责任链 `PromptRepairExecutor + InstructionPromptRepairer + WanSubprocessGenerator`；不允许“prompt 不变、只换 seed”伪修复。
- Local Editing 使用 `MaskSequenceLocalEditingExecutor + StrictProPainterLocalEditor`，ProPainter 固定为 `models/ProPainter/`，只接受逐帧 strict mask manifest，并校验输出视频可解码、帧数和分辨率。
- Runner、Generator、Executor、Local Editor 与 V2 Critic subprocess 的在线接口不再透传 PhysicsPlan；`PhysicsCritic(..., use_physics_plan=False)` 禁止在线 planner 解析 prompt。pavg 内部旧数据结构只保留兼容读取，不执行 Physics Planner。
- Acceptance Gate 只允许 `mode=enforce`，并明确区分 `ACCEPTED / REJECTED / UNAVAILABLE`；required scorer 缺失、Critic unknown/provider failure 或禁止性 degraded 不能伪装成物理不合格或 ACCEPTED。
- Runner 每轮只调用一次 Policy，Guard 只校验而不替换动作；修复候选必须经过真实 Re-Critic/Re-Gate。
- Trial 升级为 `wan-repair-trial/3.0`，直接搬运真实 RoundRecord，保留 Decision、Guard、双 Semantic、候选路径、before/after Critic 和 before/after Gate；`successful=true` 必须由 Strict Re-Gate ACCEPTED 支撑。
- RUN_ROOT 采用不可变 `run_manifest.json`、动态 `run_status.json`、`run.lock`、attempt 隔离、authoritative attempt、`--resume` 和 `--retry-failed`。
- vLLM 只终止本 run 持有的进程组；双卡运行时 Wan 使用 GPU0，vLLM 使用 GPU1，不再使用宽泛 `pkill -f`。

### 19.2 实施中发现并修复的问题

第一次真实 smoke 在 Wan 生成、vLLM 启动和 SAM2 81 帧跟踪完成后，Critic 报告组装触发：

~~~text
UnboundLocalError: cannot access local variable 'resolved_request'
~~~

根因是禁用 PhysicsPlan resolver 后，`resolved_request` 只在 resolver 启用分支赋值。已在原 `src/pavg_critic/pipeline.py` 内修正：进入可选 resolver 分支前令 `resolved_request = request`。这不会重新启用 PhysicsPlan，只保证关闭 planner 时 `CriticArtifacts` 仍保存原始请求。

第二次真实 smoke 已证明上述异常修复，但 Strict Gate 正确返回 `UNAVAILABLE`，原因是服务器 SAM2 CUDA 扩展尚未编译：

~~~text
sam2_cuda_extension = unavailable
sam2_postprocess = hole_filling_disabled
degraded = true
~~~

服务器已有 CUDA 12.2 编译器，但 `nvcc` 未在默认 PATH。已在主环境安装 `ninja`，并使用严格失败模式编译 SAM2 原生扩展：

~~~bash
cd /root/PhysGenLoop-/models/sam2-src

CUDA_HOME=/usr/local/cuda-12.2 \
PATH=/usr/local/cuda-12.2/bin:$PATH \
SAM2_BUILD_CUDA=1 \
SAM2_BUILD_ALLOW_ERRORS=0 \
/root/PhysGenLoop-/envs/main/bin/pip install -v -e . --no-build-isolation
~~~

最终产物为：

~~~text
/root/PhysGenLoop-/models/sam2-src/sam2/_C.so
sam2_cuda_extension = available
sam2_postprocess = enabled
degraded = false
~~~

### 19.3 统一验证结果

静态与单元验证：

~~~text
git diff --check                         PASS
pytest tests/wanphysics_v2 -q            60 passed
SAM2 import after torch                  PASS
ProPainter repo and three weights        PASS
Wan / Qwen3-VL / SAM2 checkpoint         PASS
GPU0 / GPU1 / ffmpeg / cv2 / port 8000  PASS
~~~

Strict dry-run 闭环：

~~~text
RUN_ROOT = /root/PhysGenLoop-/outputs/v2_dryrun_strict_impl_20260722_095917
before Gate = REJECTED
Policy action = prompt_repair
Re-Gate = ACCEPTED
Trial schema = wan-repair-trial/3.0
successful = true
physics_gain = 0.7
~~~

修复后的最终真实单样本 smoke：

~~~text
RUN_ROOT = /root/PhysGenLoop-/outputs/v2_smoke_enforce_impl_final_20260722_181402
shell status = 0
Wan real video = generated, 81 frames, 832x480, 24 fps
SAM2 CUDA extension = available
Critic degraded = false
before Gate = REJECTED
Policy action = reject
Guard status = allowed
executed action = reject
final state = REJECTED
terminal reason = policy_reject
Trial schema = wan-repair-trial/3.0
GPU processes after run = none
~~~

这里的 `REJECTED` 是有效业务终态，不是程序失败：该样本的 Strict Gate 因 confidence、coverage 和双 Semantic 未达到阈值而拒绝，三动作 Policy 在没有可信局部目标时自然选择 Audited Reject。由于正式入口已删除动作强制参数，真实 smoke 不应为了覆盖某个分支而改写 Policy 决策。对应的 `repair_decision.json`、`repair_trace.jsonl`、`trials.jsonl`、`loop_result.json`、`sample_status.json`、`run_status.json` 和 `summary.json` 均已生成。

### 19.4 验证边界与后续放量门槛

- 真实 smoke 已覆盖真实 Wan、真实 Critic、Strict Gate、自然三动作 Policy、Guard、Audited Reject、Trial V3、run 级产物和 vLLM 资源回收。
- Prompt Repair 的 before/action/after、Re-Critic/Re-Gate 已由 Strict dry-run 与单元测试覆盖；本次真实样本没有自然选择该动作。
- ProPainter strict manifest、执行接口和输出校验已由单元测试与 preflight 覆盖；本次真实样本没有可信 local target，因此没有自然触发 ProPainter。
- 不应恢复 `--force-action` 来制造 Prompt Repair 或 Local Editing 的成功记录。自然分支真实验证应从 pilot 样本中获得，并保留 Policy 原始概率与选择。
- 在启动 pilot300 前，仍应按测试矩阵完成自然 Prompt Repair、自然 Local Editing、Resume/retry-failed 和 limit=5 验证；任何 `UNAVAILABLE`、schema 失败或非预期进程残留都必须阻止放量。
