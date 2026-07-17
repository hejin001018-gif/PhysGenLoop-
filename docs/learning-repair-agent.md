# Learning Repair Agent：最终架构、Blender 数据与训练部署

## 1. 当前交付边界

当前正式实现统一位于 `src/physgenloop/learning_repair/`，把 Repair 拆成以下可审计闭环：

```text
Blender normal/broken/repair trials
              ↓
repair_sample.json → JSONL manifest → group-safe split
              ↓
CriticReport + RepairContext → Repair Policy（probability + per-action value）
              ↓
     RepairSelector（capability mask / abstention / fallback）
              ↓
     structured RepairDecision
              ↓
ExecutorRegistry → Prompt / Global / Local / Reject
              ↓
Critic re-evaluation → RepairTrialV1 → Memory / LearningTargetV1
```

当前已实现：

- `RepairExample`、`RepairContext`、丰富版 `RepairDecision` 和四动作空间；
- Blender 单样本记录收集、JSONL 清单、数据审计和按场景分组切分；
- 版本化 CriticReport 特征编码；
- PyTorch 分类 Policy 与四动作 Action-Value Policy；
- 独立 `RepairSelector`、能力遮罩、置信度 abstention 和规则 fallback；
- 四类 Executor、独立闭环 Runner、Actual Trial campaign 与 append-only recorder；
- proxy/actual provenance-aware Memory，失败 Trial 作为负 utility；
- 单一 `pavg-repair` CLI，覆盖数据、训练、推理、campaign、评估和交付。

当前已完成 1,200 个 Blender group、22,200 条 proxy 监督样本和 v3 Policy 训练。
当前尚未完成的是真实 Prompt/Global/Local Executor rollout、HunyuanVideo adapter 与
Actual Trial 校准；因此 v3 是研究基线，不是已验证的真实视频修复成功率。

## 2. Blender 应该输出什么

每个基础场景定义稳定的 `group_id`，例如 `ball-drop-layout-00017`。同一场景的正常
视频、不同错误强度、不同修复尝试都使用相同 group，防止它们跨训练/测试集合。

推荐目录：

```text
data/samples/ball-drop-layout-00017/gravity-severity-02/
├── normal_video.mp4
├── wrong_video.mp4
├── repaired_video.mp4
├── object_mask/
├── depth/
├── segmentation/
├── trajectory.json
├── physics_state.json
├── critic_report.json
└── repair_sample.json
```

`repair_sample.json` 遵循 `schemas/repair_sample.schema.json`。示例清单位于
`examples/repair_training_manifest.jsonl`。

一个训练样本不应只有“错误视频 + 正常视频”。要监督“选什么动作”，还必须实际
执行候选修复并记录结果：

1. 对同一 broken sample 尝试允许的 Prompt、全局重生成和局部编辑策略；
2. 分别计算修复前后物理分、语义保持、视觉质量和执行成本；
3. 通过预注册的 reward/门限选择 target action；
4. 所有动作都失败或修复预算耗尽时，target 才是 `reject`；
5. 失败尝试进入 Repair Memory 作为负经验，但默认不作为正确动作标签训练。

建议先固定选择准则，例如：

```text
valid repair:
  after_physics_score >= 0.80
  semantic_score      >= 0.85
  quality_score       >= 0.75

reward = physics_gain
       + 0.30 * semantic_score
       + 0.20 * quality_score
       - 0.10 * normalized_repair_cost
```

不要在数据生成中途改变这个规则，否则 target action 的语义会漂移。

### Blender 与 Hunyuan 的领域边界

Blender 能可靠监督物理错误类别、理想轨迹、可局部修复区间、局部/全局修复的可行性
以及修复后的物理增益。但是 Blender 无法单独证明“一段 Prompt 修改对 HunyuanVideo
是否有效”，因为这取决于具体生成模型的 prompt sensitivity。

因此推荐两阶段训练：

1. Blender 预训练：学习错误表征、物理纠正方向、local/global/reject 的基础策略；
2. Hunyuan 校准：用少量真实生成—修复 rollout 补充 Prompt Repair 和 Global
   Regeneration 的实际成功率，再微调 Policy 或更新 Memory。

## 3. 数据规模建议

先完成小规模门控，再扩大渲染：

| 阶段 | 基础场景 group | 每类错误 | 用途 |
|---|---:|---:|---|
| Pipeline smoke | 20–50 | 5–10 | 检查 schema、GT、修复标签 |
| Policy pilot | 500–1,000 | 500+ | 验证四动作是否可学习 |
| 正式预训练 | 5,000–20,000 | 5,000+ | 覆盖物体、相机、材质和错误强度 |
| Hunyuan calibration | 500–2,000 rollout | 按动作平衡 | 修正生成器领域差异 |

每个物理错误至少覆盖三个严重度，并随机化物体质量、材质、相机、光照、背景和运动
初值。训练/验证/测试只按 `group_id` 切分，不能按视频文件随机切分。

## 4. 本地与云端命令

安装训练依赖：

```powershell
python -m pip install -e ".[train,video,env,test]"
```

收集 Blender 作业的单样本记录：

```powershell
pavg-repair collect `
  --root data/samples `
  --output data/repair_manifest.jsonl
```

审计 schema、标签和文件是否齐全：

```powershell
pavg-repair validate `
  --manifest data/repair_manifest.jsonl `
  --check-artifacts `
  --base-dir .
```

生成无场景泄漏的数据切分：

```powershell
pavg-repair split `
  --manifest data/repair_manifest.jsonl `
  --output-dir data/repair_splits `
  --seed 42
```

训练：

```powershell
pavg-repair train `
  --manifest data/repair_manifest.jsonl `
  --config configs/repair_agent.yaml `
  --output-dir outputs/repair_agent/run_001
```

输出包括：

```text
outputs/repair_agent/run_001/
├── best_policy.pt
├── training_report.json
├── train.jsonl
├── validation.jsonl
└── test.jsonl
```

训练与独立测试完成后，导出不依赖 Blender 的部署包：

```powershell
pavg-repair export `
  --checkpoint outputs/repair_agent/run_001/best_policy.pt `
  --config configs/repair_agent.yaml `
  --memory data/repair_manifest.jsonl `
  --critic-config configs/default.yaml `
  --critic-model-id pavg-critic-0.3.0 `
  --output-dir outputs/repair_agent/release
```

```text
outputs/repair_agent/release/
├── model.pt
├── config.yaml
├── repair_memory.jsonl
├── feature_schema.json
├── critic_config.yaml
├── critic_snapshot.json
├── inference.py
├── requirements.txt
├── release_manifest.json
└── README.md
```

当前 Policy 是结构化 MLP，不使用自然语言 tokenizer。`feature_schema.json` 是对应的
版本化输入契约；部署时必须保留它和冻结 Critic 快照。只有未来改成文本 Transformer
策略时才需要导出 `tokenizer/`。

阶段 2 不需要 Blender，但仍需要匹配版本的 Physics Critic、PhysGenLoop 代码和动作
执行后端。Repair Agent 输出 Prompt Repair、Global Regeneration、Local Editing 或
Reject 决策；HunyuanVideo generator 和局部编辑器负责真正执行动作。

评估与单报告推理：

```powershell
pavg-repair evaluate `
  --manifest outputs/repair_agent/run_001/test.jsonl `
  --checkpoint outputs/repair_agent/run_001/best_policy.pt `
  --split test

pavg-repair predict `
  --critic-report result.json `
  --checkpoint outputs/repair_agent/run_001/best_policy.pt `
  --memory data/repair_manifest.jsonl
```

无 checkpoint 时 `evaluate/predict` 使用可解释规则策略，适合先验证数据流。

## 5. 云服务器应该租什么

### 当前最推荐的单机方案

如果一台机器同时承担 Blender Eevee/Cycles 渲染、SAM2、ProPainter 和 Repair Policy
训练，推荐：

```text
GPU:      1 × NVIDIA L40S 48 GB
CPU:      24–32 vCPU，高主频
RAM:      128 GB
本地盘:   2–4 TB NVMe SSD
对象存储: 按数据量准备 5–20 TB OSS/S3
系统:      Ubuntu 22.04/24.04 + NVIDIA Driver + CUDA 12.x
```

L40S 同时具备较好的 Blender OptiX 渲染和 AI 推理/训练能力，是这一阶段最均衡的
数据中心卡。若云平台提供 RTX 4090 24 GB 且价格明显更低，它通常更适合纯 Blender
渲染，但显存和数据中心稳定性不如 L40S。

### 按任务拆分

| 工作负载 | 最低可用 | 推荐 | 说明 |
|---|---|---|---|
| 当前结构化 MLP Policy 训练 | CPU 或 T4 16 GB | L4/A10 24 GB | 模型很小，通常几分钟到一小时，不需要多卡 |
| Learning Repair 推理 | CPU | L4 24 GB | 单报告推理几乎不占显存 |
| Blender Eevee 批量渲染 | RTX 3090/4090 24 GB | 4090 或 L40S | 物理模拟多在 CPU，GPU 主要负责渲染 |
| Blender Cycles/多通道高质量渲染 | 4090 24 GB | L40S 48 GB | depth/mask 本身不重，复杂光追和大场景更吃显存 |
| SAM2 + ProPainter | A10/4090 24 GB | L40S 48 GB | 48 GB 更适合长序列和并发批处理 |
| 视频 Critic/视频编码器 LoRA | L40S 48 GB | A100/H100 80 GB | 这是未来像素级学习，不是当前 MLP Policy |
| HunyuanVideo 推理 | A100 80 GB | H100/H200 80 GB+ | 具体取决于模型版本、量化和 CPU offload |
| HunyuanVideo/大型 Video-LLM LoRA | 2 × A100 80 GB | 4 × H100 80 GB | 开始前必须按最终模型实测显存 |

### 租卡结论

- 现在只训练本框架的 `CriticReport → Repair Action`：租 L4/A10 24 GB 即可，甚至
  不必租 GPU；不要为这个 MLP 付 H100 的费用。
- 现在要批量生成 Blender 数据并同时运行 SAM2/ProPainter：租 1 张 L40S 48 GB。
- Blender 渲染规模很大且不跑重型模型：优先多台 4090 渲染节点，而不是 H100。
- 等接入 HunyuanVideo 时，再单独租 A100/H100 80 GB；不要长期把昂贵推理卡用于
  Blender 物理模拟或小型 Policy 训练。

## 6. 存储和吞吐注意事项

视频本身通常不是最大部分；逐帧 PNG、EXR depth、segmentation、mask、修复候选和
中间缓存会迅速放大数据。正式规模建议：

- NVMe 只作为当前 shard 的 scratch，完成后上传对象存储；
- 每 500–2,000 个样本打成一个版本化 shard，避免数百万小文件；
- manifest 保存相对 URI、SHA-256、Blender 版本、场景 seed 和生成脚本版本；
- normal/broken/repaired 共用基础场景 ID，任何派生版本都不得跨 split；
- 先渲染 50 个 group 并实测“每 group 磁盘量 × 目标 group 数”，再购买存储。

## 7. 上云前验收门槛

- `pavg-repair validate --check-artifacts` 返回 `valid: true`；
- 四个动作均有足够样本，不存在只有一种标签的训练集；
- group leakage 为 0；
- 每种错误至少抽查 normal/wrong/repaired 三联视频和 Ground Truth；
- target action 来自实际修复结果，而不是仅由错误名称人工猜测；
- 规则 Policy 在 test 上形成可解释基线；
- 学习 Policy 至少报告 macro-F1、混淆矩阵、gain MAE、语义保持和视觉质量；
- Blender test 与后续 Hunyuan calibration/test 分开报告，不把合成域结果当成真实生成域结论。
