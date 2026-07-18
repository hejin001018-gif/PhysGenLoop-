# Repair Agent 训练最终总结

## 结果概览

- 数据：600 个场景组，7800 条监督样本
- 数据质量：所有纳入训练的 shard 均通过 artifact audit 与 Critic 语义门禁
- 最佳随机种子：101
- 最佳验证 macro-F1：1.0
- Held-out test macro-F1：1.0
- Held-out test accuracy：1.0
- Repair Memory：512 条，仅来自 train split
- 模型：repair-mlp-da12ae600236

## 数据集与可复现性

- Action 分布：`{"reject": 1200, "local_editing": 3600, "prompt_repair": 2400, "global_regeneration": 600}`
- Split 分布：`{"train": 6240, "validation": 780, "test": 780}`
- Group leakage：`{}`
- 固定 split seed：`20260716`
- Assigned manifest SHA256：`da12ae60023640188d6f1c8166f176ffeb76dc3a3dc1e5ba37d849a4b12888f3`
- Critic config SHA256：`f8945bbb675f34215a8440793d25fd3f0df907a44aee5252e994fb666e91d4f4`
- Critic model ID：`pavg-critic-0.3.0/configs-default`

Campaign 配置及生成器指纹：

```json
{
  "total_groups": 600,
  "groups_per_shard": 10,
  "start_group": 0,
  "seed": 20260716,
  "frames": 48,
  "width": 640,
  "height": 360,
  "samples": 8,
  "variants_per_group": 13,
  "source_fingerprint": {
    "Blender_video/scripts/generate_repair_shard.py": "3308daab89ad51e1fd80ea0a3950504c220c4fdc1ae844bc5506637a0a7ff806",
    "Blender_video/scripts/finalize_repair_shard.py": "59584f244c6debdaed766ae98fedd2122f1994424e30c40c109f160d3c7585cc",
    "Blender_video/run_cloud_shard.sh": "c7ff542132b82c7c3a25ac2dd1ba526c6bf9ae5de1d60667af981166a004f8e5",
    "configs/default.yaml": "f8945bbb675f34215a8440793d25fd3f0df907a44aee5252e994fb666e91d4f4"
  }
}
```

## Repair Agent 架构

- 输入：冻结 Physics Critic 输出的结构化 `CriticReport`
- 特征维度：53
- Policy：结构化特征编码器 + MLP action head + expected-gain head
- MLP hidden dims：`[128, 64]`；dropout：`0.15`
- 动作空间：`['prompt_repair', 'global_regeneration', 'local_editing', 'reject']`
- Memory：同一特征空间上的 cosine retrieval，推理融合权重 `0.25`
- Executor：不在本模型包中，由部署端四类执行适配器负责

## 五种子验证集对比

| Seed | Macro-F1 | Accuracy | Best epoch |
|---:|---:|---:|---:|
| 17 | 1.0 | 1.0 | 1 |
| 23 | 1.0 | 1.0 | 1 |
| 42 | 1.0 | 1.0 | 1 |
| 73 | 1.0 | 1.0 | 1 |
| 101 | 1.0 | 1.0 | 1 |

## Held-out test 分类结果

| Repair action | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| prompt_repair | 1.0 | 1.0 | 1.0 | 240 |
| global_regeneration | 1.0 | 1.0 | 1.0 | 60 |
| local_editing | 1.0 | 1.0 | 1.0 | 360 |
| reject | 1.0 | 1.0 | 1.0 | 120 |

混淆矩阵（action order: ['prompt_repair', 'global_regeneration', 'local_editing', 'reject']）：

```json
[[240, 0, 0, 0], [0, 60, 0, 0], [0, 0, 360, 0], [0, 0, 0, 120]]
```

## Release 与清理

- Release smoke：True
- Release 文件数：9
- Release manifest SHA256：`3441fc15899917533dceed2a39c160be1535bcd15f1387e5c28eacf0a5ba069a`
- 部署压缩包：`/workspace/pavg/artifacts/repair_agent_repair_600g_v1.tar.gz`
- 压缩包 SHA256：`b81b0a60364df60a1b14b9c229cd465abad95d2150b68e9e7591d9ec1fb32221`
- 已删除 Blender shards：868.49 MiB

Release 文件清单：

| File | Bytes | SHA256 |
|---|---:|---|
| `README.md` | 748 | `ba5b87d763415fd9ba8d67e43137ddd5f7474b5ea7f249b32b1a4d781a5172cd` |
| `config.yaml` | 444 | `bc7ddc68f8019e29c2c51b7dc69e16adf4770d1cc87b11d3f484d061575a2c74` |
| `critic_config.yaml` | 1528 | `f8945bbb675f34215a8440793d25fd3f0df907a44aee5252e994fb666e91d4f4` |
| `critic_snapshot.json` | 219 | `72f22af0e188e23eafec1c5b2af3a5627f27503216659507f2d87abe27fb3772` |
| `feature_schema.json` | 2223 | `a826a2aadf39902e2a389ac814cc6b6ffbe24871b1d9097c68d55269e87be2b5` |
| `inference.py` | 1988 | `89f14a981b74b014cc6620d2d69a4730142116f04f6bb4082d867b309b9e5365` |
| `model.pt` | 70791 | `50356f39ed696c889f3e2f22dff3b15fe4e7689bd42358c091742bb231bf5257` |
| `repair_memory.jsonl` | 5333407 | `8580f9e53ddd58777fa8133d12922dc38b88891876bed01709b47992dd62a25e` |
| `requirements.txt` | 45 | `2fd375a635e303760d41ba0efe8ea30792b8ee8434dc2d3787f2a0ac25971136` |

## 部署契约

```text
CriticReport + RepairContext
              ↓
LearningRepairAgent (Policy + Memory)
              ↓
Prompt Repair | Global Regeneration | Local Editing | Reject
              ↓
Deployment RepairExecutor adapter
```

推理入口：`python inference.py --critic-report critic_report.json --device cuda`

## 已知限制

- 当前监督标签来自 Blender 配对正常轨迹与策略代理映射，并非部署环境中 Executor 的真实修复试验回报。
- 当前 Release 提供 Selector/Policy 与 Repair Memory；Prompt、全局重生成、局部编辑和 Reject 的执行后端需由部署侧适配器实现。
- 训练输入绑定冻结的 Physics Critic 配置；CriticReport schema 或类别语义变化时必须重新做兼容性评估。
- Blender 场景覆盖受控刚体下落/接触族，迁移到 HunyuanVideo 后仍需采集真实生成分布上的闭环反馈。

## 团队下一步任务

1. 实现统一 RepairExecutor 接口及 prompt_generator、video_generator、local_video_editor、candidate_selector 四个适配器。
2. 把部署端每次尝试写成 RepairTrial，记录 before/after physics、semantic、quality 和 cost，替换代理标签。
3. 在 HunyuanVideo 小规模验证集上做 Critic→Selector→Executor→Critic 的端到端闭环回归。
4. 保持 critic_snapshot.json 与 feature_schema.json 的版本门禁，防止特征漂移和静默不兼容。
5. 依据 held-out test 的最低动作 F1 与混淆方向构建下一轮难例/类别重采样续训集。

最终部署包不依赖 Blender；保留 Physics Critic、Repair Agent 代码与该模型包即可进行策略推理。
