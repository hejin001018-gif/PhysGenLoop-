# PhysGenLoop 训练结果与全链路问题审计

**审计日期：** 2026-07-20  
**审计对象：** 远端 `/root/PhysGenLoop-`  以及其最近训练、闭环运行产物  
**远端连接：** `px-cloud1.matpool.com:27323`  只读检查  
**源码基线：** `03e96b2 feat: integrate wanphysics full-chain loop`  
**最新架构说明：** `worklog/2026_07_20/第一次全链路搭建框架.md`  
**限制：** 本次只生成审计文档，没有修改远端或本地代码；后续实施必须遵守“只新增、不删除、不覆盖他人代码”，并通过配置开关启用新增行为。

## 1. 结论摘要

当前系统已经具备“Wan 生成 → SAM2/VLM Critic → 证据选择 → Repairer → 再评估”的可运行骨架，但最近发布的 repair policy 仍是 **Blender proxy 预发布模型**，不能据此声称 Hunyuan 或真实视频修复质量提升。

最重要的结论有六项：

1. `repair-agent-v3.1-proxy-20260717` 使用 22,200 条 Blender proxy 标签训练，`actual_trial_count=0`。训练报告中验证集和测试集几乎全为 1.0，说明模型非常容易复现代理标签，不能证明它能选择真实修复动作。
2. 报告把 `hunyuan` 列为兼容域，但训练样本统计只有 `blender: 22,200`；没有 Hunyuan 的 group-disjoint 校准集或测试集，存在明显域迁移空缺。
3. 训练产生的 `LearningTargetV1` memory 与生产 Repairer 仍使用的旧 `RepairExample` loader 不兼容。远端实测加载失败：`ValueError invalid repair sample 0: repair example target must be an object`，生产代码捕获异常后继续运行，因此 memory mixing 实际没有生效。
4. 最近 20 条 pilot 闭环结果为 `12/20 accepted`，但接受条件只有 `decision == physical && physics_score >= 0.8`。语义保持和视觉质量 scorer 尚未接入，`accepted` 不能等同于修复成功。
5. 该 20 条运行中 `local_editing_count=0`，所有错误范围都是 global，且没有专门的 ProPainter/local-editing GPU smoke；局部修复分支尚未被真实验证。
6. 中间 hard-v1 数据生成的两个 multi-corrupt group 均通过文件完整性但未通过语义门：只观察到 `object_disappearance`，没有注入的第二个 violation family。后续 hard-v1.1 做了修复，但最终要求检查没有把“全部语义门有效”设成硬条件。

因此当前状态应标为：**架构可运行、训练指标不可外推、生产 memory 未启用、真实修复闭环尚未完成验收**。建议先补齐数据与评测契约，再训练或调整 policy；不应先用现有 1.0 proxy 指标调权重或宣称模型有效。

## 2. 已核对的对象和证据

### 2.1 代码与配置

- 主入口：`agents/wanphysics/run_videophy2_loop.py`
- 闭环控制：`src/physgenloop/controller.py`
- 生成器：`generators/wanphysics/adapter.py`、`agents/wanphysics/gen_step.py`
- Critic：`generators/wanphysics/sam2_vlm_critic.py`、`agents/wanphysics/eval_step.py`
- Repairer：`generators/wanphysics/repairer.py`
- policy 训练：`src/physgenloop/learning_repair/proxy_adapter.py`、`value_training.py`
- 新 memory 类型：`src/physgenloop/learning_repair/memory_policy.py`
- 旧 memory loader：`src/physgenloop/learning_repair/memory.py`
- 运行配置：`configs/loop.yaml`

远端 manifest 实际计数为：

| 文件 | 样本数 |
|---|---:|
| `evaluation/manifests/videophy2_pilot300.json` | 300 |
| `evaluation/manifests/videophy2_test_full.json` | 3,397 |

### 2.2 最近训练 checkpoint

目录：`checkpoints/repair_agent/repair-agent-v3.1-proxy-20260717/`

- `training_report.json`
- `evaluation_test.json`
- `proxy_adaptation.json`
- `model/best_action_value_policy.pt`
- `memory/proxy_memory_train.jsonl`
- `release_manifest.json`

关键字段：

| 字段 | 观测值 | 解释 |
|---|---|---|
| `selection_mode` | `classification_proxy` | 选择的是代理动作分类，不是实际修复价值学习 |
| `proxy_label_count` | 22,200 | 所有训练标签来自 Blender proxy |
| `actual_trial_label_count` | 0 | 没有真实生成/修复 trial 标签 |
| 训练/验证/测试 | 18,024 / 2,154 / 2,022 | group-safe split，但仍是同一代理标签来源 |
| 训练 domain | `blender: 22,200` | 没有 Hunyuan 训练或验证样本 |
| `source_revision` | `unknown` | checkpoint 与当前代码基线无法严格绑定 |
| `deployment_ready` | `false` | 发布清单明确是 pre-release |
| action order | 4 类 | `prompt_repair`, `global_regeneration`, `local_editing`, `reject` |

`evaluation_test.json` 的结果：

| 模式 | Accuracy | Macro-F1 | Balanced Acc. | Mean regret |
|---|---:|---:|---:|---:|
| `R0_category_only` | 0.686944 | 0.587851 | 0.692424 | 0.096283 |
| `R1_heuristic` | 0.686944 | 0.587851 | 0.692424 | 0.096283 |
| `R2_policy_only` | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| `R3_memory_only` | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| `R4_policy_plus_memory` | 1.000000 | 1.000000 | 1.000000 | 0.000000 |

四个完美结果都在同一份 proxy held-out test 上计算；它们不是“修复后物理正确率”。更重要的是，生产 memory 对当前 loader 的实测加载失败，所以离线 `R3/R4` 的完美结果不能代表线上 `Repairer` 会得到同样信息。

### 2.3 训练数据质量

`outputs/repair_training/diagnostics/node_a_shard_0011_finalize.json` 和 `node_b_shard_0021_finalize.json` 都记录了：

- `semantic_gate.valid=false`；
- 每个 shard 审查 65 个样本时发现 1 个失败；
- `group_000957--multi_corrupt` 和 `group_001155--multi_corrupt` 只暴露 `object_disappearance`，没有暴露预期的多违例组合。

后续报告说明 hard-v1.1 修正了这两个 group，但这只证明发生过数据语义事故并做过补救，不证明所有生成样本均已逐组完成语义校验。最终 release requirements 也没有把 `semantic_gate.valid == true` 作为不可绕过的发布条件。

### 2.4 最近 20 条闭环运行

产物：`outputs/videophy2_run_20260720_132415/summary.json`

- `samples=20`
- `accepted=12`
- `acceptance_rate=0.6`
- `average_rounds=1.5`
- `local_editing_count=0`
- `global_scope_count=10`
- 20/20 的 detector backend 为 `sam2+vlm`
- 其余未接受样本以 `max_rounds` 结束

这些数据证明当前脚本可以完成一轮小规模闭环，不证明 60% 是“真实修复成功率”。代码中的实际停止条件位于 `src/physgenloop/controller.py`，只检查物理分数和 Critic decision；`semantic_score`、`quality_score` 在当前运行结果中仍为 `null`。

## 3. 问题清单与根因

严重级别定义：

- **P0：** 会使训练/效果结论失效，必须在下一次训练前处理。
- **P1：** 会使线上行为或可审计性不可靠，应在正式全量运行前处理。
- **P2：** 不一定改变正确性，但影响可复现性、性能或维护安全。

### P0-1：训练目标是代理动作分类，不是实际修复价值

`proxy_adapter.py` 将 Blender `RepairExample` 转成 `LearningTargetV1`。每个样本只有“被选中的 action”带 reward，其他候选 action 的 reward 为 `None`；metadata 明确 `proxy_label=true`，`actual_trial_count=0`。因此 policy 学到的是现有规则/标签映射，而不是“执行某个动作后视频是否真的变好”。

**后果：** 1.0 Macro-F1 只能表示代理标签可预测，不能表示生成视频物理正确、语义保持或视觉质量提升。继续在这份数据上增加 epoch、调 loss 或调权重，很可能只会强化标签映射。

### P0-2：训练域与声明域不一致

release compatibility 声明 `['blender', 'hunyuan']`，但 training report 的 domain count 只有 `blender: 22,200`。没有 Hunyuan 的真实 trial、校准集或 group-disjoint test。

**后果：** policy 在 Wan/Hunyuan 输出上的动作价值、错误范围和成本估计没有证据；`hunyuan` 只能暂时标为“待验证兼容域”。

### P0-3：memory 训练格式与生产格式断裂

训练生成的是 `LearningTargetV1` proxy memory；生产 `generators/wanphysics/repairer.py` 仍调用旧 `RepairMemory.from_manifest`，该 loader 期望 `RepairExample`。远端实际加载得到：

```text
ValueError: invalid repair sample 0: repair example target must be an object
```

Repairer 捕获错误并继续，结果是线上 memory mixing 被静默关闭。当前系统的离线 R3/R4 完美指标与线上路径不一致。

### P0-4：接受条件不包含语义和视觉质量

闭环 controller 当前接受条件是：

```text
report.decision == "physical" and report.physics_score >= acceptance_score
```

而 `semantic_score`、`quality_score` 仅在 learning-repair campaign 接口中预留，没有注入 `run_videophy2_loop` 的主路径。

**后果：** Critic 误判或生成器改变主体/场景时也可能被 accepted；“accepted”无法作为实际修复成功标签。

### P1-1：数据语义门存在已知失败，最终发布门不够严格

hard-v1 的两个 multi-corrupt group 通过文件完整性但没通过语义门，说明“视频可读”不等于“注入了预期错误”。当前最终要求未强制所有 group 的 semantic gate 通过，也没有记录完整检查覆盖率。

### P1-2：local editing 分支没有真实 smoke

最新 20 条 run 中 `local_editing_count=0`，所以 ProPainter/掩码/critical frame 的真实执行链没有被验证。配置虽然允许 local editing，实际只覆盖了 global scope/prompt repair。

### P1-3：Critic 完整报告没有持久化

`Sam2VlmSubprocessCritic.evaluate()` 写出的 `critic.json` 只是摘要，包含 video、status、physics_violation、confidence、detector_backend 等字段，`reason` 常为 `null`；完整的 violations、repair instruction、evidence、critical frames、mask URI 只在内存中使用。

**后果：** 中断后无法仅凭磁盘产物重建决策，审计和重试会缺少修复依据。应采用新增 sidecar，而不是覆盖旧摘要文件。

### P1-4：训练评估存在“同标签系统内的完美闭环”风险

`R2/R3/R4` 都在同一 proxy held-out test 上得到 1.0；features 中包含 decision、physics score、violation category/count、coverage、evidence availability 等高度决定 action 的字段。即便 group split 没有泄漏，同一规则模板产生的标签也可能让任务近似确定性分类。

**后果：** 现有 metrics 没有反映候选动作的反事实效果，也没有反映修复成本、失败概率或画面保真度。

### P1-5：中间语义校验覆盖率不足

部分 shard 只审查 65/175 个样本就判定语义门；已发现的失败说明抽样校验不足以证明整个 shard 可靠。修复 hard-v1.1 后必须重新全量校验受影响 group，并记录 checked/total 和失败样本清单。

### P2-1：路径和环境不可移植

`configs/loop.yaml`、`run_videophy2_loop.py`、`eval_step.py` 中存在 `/root/PhysGenLoop-` 的绝对路径、绝对 `sys.path`、固定 `.env` 和 checkpoint 默认值，与 README 的“相对路径/可迁移环境”约定不一致。`source_revision=unknown` 进一步削弱可复现性。

### P2-2：vLLM 进程清理过宽

Critic 使用类似 `pkill -9 -f vllm` 的模式停止服务，可能误杀同机其他 vLLM 服务。应按启动 PID、端口或独立 process group 回收，并记录 owner/run_id。

### P2-3：旧入口仍有 GPU OOM 风险

`outputs/e2e_loop_full_v2.log` 记录了 generator `.to(device)` 时 A100 显存耗尽；新入口通过 subprocess handoff 已经在 20 条运行中避开该问题，但旧入口仍存在，且没有统一的资源回归测试。该问题应标为“新入口暂缓、旧入口未消除”。

### P2-4：运行告警未形成门禁

- SAM2 `_C` post-processing extension 缺失；当前 propagation 可运行，但后处理能力和性能需单独记录。
- vLLM 没有 FlashInfer 时回退到 PyTorch top-k/top-p，主要是性能影响。
- 生成配置覆盖 warning 可能改变预期的 deterministic decoding，应把最终 resolved config 写入结果。
- `Lossy conversion from float32 to uint8` 表明 mask/image 写盘前没有显式范围转换，应增加范围检查和测试。

### P2-5：可审计字段不足

`trials.jsonl` 记录 before/after physics score 和 stop reason，但没有稳定记录 policy action、memory hit、semantic/quality gate、候选失败原因和 provider/cache 状态。现在只能从 `error_scope_trace` 间接推断部分动作。

### P2-6：发布状态与源码状态未完全绑定

最新 worklog 处于 staged 但未提交状态，checkpoint 的 compatibility source revision 为 `unknown`。即使代码和模型内容本身正确，也无法可靠复原“哪个源码版本产生了哪个权重”。

## 4. 只增不删的修改方案（供审阅，暂不实施）

下面所有动作都采用“新增文件/新增字段/新增配置开关/新增 sidecar/新增测试”的方式，不删除、不覆盖已有代码和产物。旧路径保持默认兼容，新增路径必须显式打开。

### 阶段 A：冻结证据与发布门（P0，优先）

1. 新增 `reports/audits/training_contract_audit.json`，记录 checkpoint hash、源码 revision、proxy/actual trial 数、domain counts、semantic gate coverage 和 manifest hash。
2. 新增 `training.require_actual_trials`、`training.allow_proxy_selection` 两个开关：
   - 研究调试可允许 proxy；
   - 发布或线上 policy 默认要求 `actual_trial_count > 0`，否则状态为 `proxy_only`，不能标为 deployable。
3. 新增 domain gate：当 compatibility 声明 Hunyuan 时，必须有 Hunyuan 校准/测试 group，缺失则明确返回 `domain_unverified`。
4. 将 proxy 指标和真实 trial 指标分成两张表；禁止把 proxy held-out Macro-F1 写入“修复成功率”。

### 阶段 B：修复数据语义验证（P0/P1）

1. 新增 `src/physgenloop/learning_repair/semantic_gate_audit.py`，对 hard-v1.1 所有 group 检查：每个注入 violation family 是否在 Critic/evidence 中可观测、关键帧是否在画面内、目标物是否仍在场景中。
2. 新增 `semantic_gate_report.json`，写出 `checked`, `total`, `valid`, `failures`, `profile`, `generator_version`。
3. 将 release gate 扩展为 `semantic_gate_valid && checked == total`，但保留旧 requirements 字段；新增开关允许仅作诊断运行，不能让诊断结果误当发布。
4. 对两个已知失败 group 做回归 fixture，确保 multi-corrupt 不再退化成单一 disappearance。

### 阶段 C：新增 LearningTargetV1 memory 兼容层（P0）

1. 新增 `ProxyTargetMemory`/`TargetV1MemoryAdapter`，只读 `LearningTargetV1` JSONL；旧 `RepairMemory` 保留不动。
2. 在 Repairer 增加 `memory_backend: legacy|target_v1|auto` 配置开关，默认仍为 `legacy`，验证完成后才在实验配置中切换 `target_v1`。
3. memory load 失败不得静默吞掉：新增 diagnostics 字段 `memory_enabled`, `memory_format`, `memory_records`, `memory_error`。
4. 为同一条 memory fixture 增加 legacy 与 target_v1 两套单元测试，并增加真实 Repairer smoke，确认 memory hit 会影响 action score。

### 阶段 D：采集真实 trial，再训练 value policy（P0）

1. 使用当前完整 pipeline 先运行小规模、group-disjoint 的真实 trial：每个 sample 至少保留 before、候选 action、after 的物理/语义/质量/成本结果；不能只保存被选动作。
2. 首批建议：每类 action 至少 100 个真实 trial，至少覆盖 Wan/Hunyuan 两个 domain，并保留 20% 完全未见 group 做 test。
3. 真实 reward 采用现有 gate：physics、semantic、quality、cost；无法计算的字段保持 `null`，不得用 proxy 值填充。
4. 训练报告增加 `actual_trial_label_count`、每 action 的可观测 reward 比例、counterfactual coverage、domain slice 和置信区间；只有这些字段达标才允许生成 `deployment_candidate`。
5. proxy 数据继续作为 warm-start/对照组，但模型名称和报告必须带 `proxy_only`，不与真实 trial 结果合并。

### 阶段 E：补齐主闭环的独立接受门（P1）

1. 新增可插拔 `semantic_scorer` 和 `quality_scorer`，由配置开关 `loop.require_quality_metrics` 控制。
2. 当开关开启时，接受条件改为 physics、semantic、quality 三者均达阈值；当 scorer 缺失时返回明确的 `quality_gate_unavailable`，不伪装成 accepted。
3. 保留现有 `loop_result.json` 格式，新增 `acceptance_gate.json` sidecar，记录每个门的输入、阈值、结果、失败原因。
4. 先对 20 条已运行 pilot 做离线重放，比较“仅 physics”与“完整三门”的接受差异，再决定阈值；不在 full test 上调参。

### 阶段 F：让 Critic 结果可恢复、可审计（P1）

1. 保留当前摘要 `critic.json`，新增完整 `critic_report.json`；保存 violations、evidence family、critical frames、mask URI、provider/cache 状态、prompt/schema hash 和 resolved model id。
2. `trials.jsonl` 新增 policy action、memory hit、error scope、semantic/quality score、executor outcome 和 artifact checksum。
3. 增加 append-only 写入和 sample/action 唯一键，重启时只复用 schema/hash 完全匹配的 sidecar。

### 阶段 G：验证 local editing 和运行安全（P1/P2）

1. 新增最小 local-editing fixture：固定 mask、critical frames、输入输出视频，打开 `enable_local_editing=true` 运行一次 ProPainter/等价 executor，检查目标在画面内、帧数一致、mask URI 可读。
2. 增加 scoped vLLM manager：记录 PID/端口/run_id，只停止自己启动的进程；保留旧 stop 方法但默认不开启。
3. 增加 `paths.portable=true` 路径解析器和 `env_file`/checkpoint 显式配置；旧绝对路径默认保留，迁移测试通过后再切换。
4. 新增 GPU resource smoke，验证 generator 和 vLLM handoff 后显存峰值、进程归属和异常清理。

## 5. 建议的验证顺序和通过标准

1. **合同审计：** checkpoint、代码 revision、manifest、domain、semantic gate 全部可追溯；否则只标 `audit_incomplete`。
2. **memory 兼容 smoke：** target_v1 loader 成功读取固定 fixture，Repairer diagnostics 显示 `memory_enabled=true`。
3. **真实 trial pilot：** 两个 domain、四类 action 均有可观测 before/after；不得有 action 永远没有反事实样本。
4. **数据语义 gate：** hard-v1.1 相关 group 全部通过，`checked == total`。
5. **完整闭环 pilot：** 至少 20 条，分别报告 physics-only 和三门 acceptance；local-editing 至少成功执行一次。
6. **训练/评估：** 以 group 和 domain 双重隔离；proxy 只做对照，真实 trial test 只在最后一次读取；报告 action Macro-F1、value MAE、mean regret、repair success、false repair、语义保持、视觉质量、成本和失败率。
7. **发布判定：** `actual_trial_count > 0`、Hunyuan domain 有独立 test、semantic gate 全通过、memory 已启用、三门 acceptance 可用、源码 revision 非 `unknown`。

## 6. 时间与资源估计（不含用户确认后的实施）

在现有 A100 和已有模型缓存不变的前提下：

| 工作 | 预计耗时 |
|---|---:|
| 合同/数据审计与语义 gate | 1–2 小时 |
| TargetV1 memory 兼容层与测试 | 1–2 小时 |
| local editing 与资源 handoff smoke | 1–2 小时 |
| 真实 trial pilot（两 domain） | 3–8 小时 |
| 真实 trial policy 训练与隔离评估 | 1–3 小时 |
| 汇总、复现、发布审计 | 1–2 小时 |

其中最不确定的是真实 Hunyuan/Wan 生成与三门 scorer 的吞吐；在真实 trial 数据产生前，不应承诺 policy 效果提升。

## 7. 暂不建议的做法

- 不要继续在 22,200 条 proxy 标签上单纯增加 epoch、调大模型或调融合权重，然后把 1.0 指标当作真实修复提升。
- 不要在完整测试集上选择 acceptance 阈值、family 权重或 repair action；应使用独立校准集。
- 不要把 memory loader 的异常吞掉后继续宣称“使用了 memory”。
- 不要删除旧 loader、旧摘要产物或旧入口；新增兼容层和开关后再逐步迁移。
- 不要把当前 `12/20 accepted` 直接写成成功率；必须补独立 semantic/quality gate 和人工/自动质量验证。

## 8. 待审阅的决策点

请先审阅本文件中的 P0/P1 优先级和阶段 A–G 顺序。得到确认后，下一步只实施新增文件、字段和开关，并先运行合同审计、memory fixture、semantic gate fixture，再启动真实 trial；在此之前不修改现有代码行为。

