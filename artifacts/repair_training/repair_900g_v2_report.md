# Repair Agent 训练最终总结

## 结果概览

- 数据：900 个场景组，11700 条监督样本
- 数据质量：所有纳入训练的 shard 均通过 artifact audit 与 Critic 语义门禁
- 最佳随机种子：101
- 最佳验证 macro-F1：1.0
- Held-out test macro-F1：1.0
- Held-out test accuracy：1.0
- Repair Memory：768 条，仅来自 train split
- 模型：repair-mlp-08301442e861

## 数据集与可复现性

- Action 分布：`{"reject": 1800, "local_editing": 5400, "prompt_repair": 3600, "global_regeneration": 900}`
- Split 分布：`{"train": 9360, "validation": 1170, "test": 1170}`
- Group leakage：`{}`
- 固定 split seed：`20260716`
- Assigned manifest SHA256：`08301442e861d6139314a202b2cba91c65b4257bd5fd45201d02ef4171da73c9`
- Critic config SHA256：`f8945bbb675f34215a8440793d25fd3f0df907a44aee5252e994fb666e91d4f4`
- Critic model ID：`pavg-critic-0.3.0/configs-default`

Campaign 配置及生成器指纹：

```json
{
  "total_groups": 900,
  "groups_per_shard": 10,
  "variants_per_group": 13,
  "continuation": true,
  "source_campaigns": [
    "/workspace/pavg/campaigns/repair_600g_v1",
    "/workspace/pavg/campaigns/repair_300g_v2_data"
  ]
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
| prompt_repair | 1.0 | 1.0 | 1.0 | 360 |
| global_regeneration | 1.0 | 1.0 | 1.0 | 90 |
| local_editing | 1.0 | 1.0 | 1.0 | 540 |
| reject | 1.0 | 1.0 | 1.0 | 180 |

混淆矩阵（action order: ['prompt_repair', 'global_regeneration', 'local_editing', 'reject']）：

```json
[[360, 0, 0, 0], [0, 90, 0, 0], [0, 0, 540, 0], [0, 0, 0, 180]]
```

## Release 与清理

- Release smoke：True
- Release 文件数：9
- Release manifest SHA256：`5ec90d7d7313d6cb4e057de2eee74740edf09d1187c637dc17adea7282b4b6d9`
- 部署压缩包：`/workspace/pavg/artifacts/repair_agent_repair_900g_v2.tar.gz`
- 压缩包 SHA256：`589bebfdb9e9820252c23328101734462e195d11cc1e3e1d3bdca42b5820a8ea`
- 已删除 Blender shards：436.02 MiB

Release 文件清单：

| File | Bytes | SHA256 |
|---|---:|---|
| `README.md` | 748 | `ba5b87d763415fd9ba8d67e43137ddd5f7474b5ea7f249b32b1a4d781a5172cd` |
| `config.yaml` | 444 | `bc7ddc68f8019e29c2c51b7dc69e16adf4770d1cc87b11d3f484d061575a2c74` |
| `critic_config.yaml` | 1528 | `f8945bbb675f34215a8440793d25fd3f0df907a44aee5252e994fb666e91d4f4` |
| `critic_snapshot.json` | 219 | `72f22af0e188e23eafec1c5b2af3a5627f27503216659507f2d87abe27fb3772` |
| `feature_schema.json` | 2223 | `b6e4a8cd71077b59d749dfba359e163cd4f7afd4623e401c13458d65386f5596` |
| `inference.py` | 1988 | `89f14a981b74b014cc6620d2d69a4730142116f04f6bb4082d867b309b9e5365` |
| `model.pt` | 70791 | `5294eaf478624df5bed5770b8d12e965303b3ea3ea9bc76c3832b9d972e899ae` |
| `repair_memory.jsonl` | 8037475 | `70180170ed69bcc0204288eea186ee475044264d255cddf53b9e1b9d1f2fb3e9` |
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
