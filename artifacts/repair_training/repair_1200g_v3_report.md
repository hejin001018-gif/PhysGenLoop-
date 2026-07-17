# Repair Agent v3 Action-Value 训练与交付报告

## 结论

- 状态：**complete**
- 数据：1200 个场景组，22200 条监督样本
- 模型：`repair-value-78304cfff2fa`
- 最佳随机种子：101
- Held-out test macro-F1：1.0
- Held-out test balanced accuracy：1.0
- 选择模式：`classification_proxy`
- Actual Executor Trial：0；当前结果严格标记为 Blender proxy 训练

## 完成门禁

- ✅ `source_groups_1200`
- ✅ `source_samples_22200`
- ✅ `target_audit_valid`
- ✅ `no_group_leakage`
- ✅ `proxy_labels_explicit`
- ✅ `actual_trials_not_fabricated`
- ✅ `compatibility_valid`
- ✅ `held_out_test_present`
- ✅ `release_smoke_four_actions`
- ✅ `release_files_valid`
- ✅ `archive_sha256_valid`
- ✅ `cleanup_receipts_present`
- ✅ `cleanup_complete`

## 数据与来源

- 普通数据：900组、11,700条
- 困难数据：300组、10,500条
- 总计：1,200组、22,200条
- Group leakage：`{}`
- 动作分布：`{"reject": 6000, "local_editing": 7200, "prompt_repair": 4800, "global_regeneration": 4200}`
- Targets SHA256：`9700df6d963e1832ca8100afa783e3a3565208c679949e649703930ea27f0aa7`

`hard-v1` 曾有两个 multi-corrupt 样本因目标离开画面而只暴露消失错误。门禁没有放宽；使用
`hard-v1.1` 重新生成后，两者均稳定暴露穿透、消失和轨迹异常，再纳入训练。

## 架构合并

```text
Frozen CriticReport + RepairContext
            ↓
Action-Value Policy（四动作分类头 + 四动作 value 头）
            ↓
Capability Mask / provenance-aware selection
            ↓
ExecutorRegistry
            ↓
Prompt | Global | Local | Reject
```

当前只有 selected-action proxy reward；未执行动作保持 null，不伪造成失败 Trial。因此纯 proxy
checkpoint 使用分类概率选择；收集真实多动作 `RepairTrialV1` 后才启用 value 主导。

## 五种子验证

| Seed | Macro-F1 | Balanced accuracy | Value MAE | Best epoch |
|---:|---:|---:|---:|---:|
| 17 | 1.000000 | 1.000000 | 0.046059 | 1 |
| 23 | 1.000000 | 1.000000 | 0.037880 | 1 |
| 42 | 1.000000 | 1.000000 | 0.036394 | 1 |
| 73 | 1.000000 | 1.000000 | 0.044934 | 1 |
| 101 | 1.000000 | 1.000000 | 0.026651 | 1 |

## Held-out test 分动作结果

| Action | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| prompt_repair | 1.000000 | 1.000000 | 1.000000 | 480 |
| global_regeneration | 1.000000 | 1.000000 | 1.000000 | 330 |
| local_editing | 1.000000 | 1.000000 | 1.000000 | 720 |
| reject | 1.000000 | 1.000000 | 1.000000 | 492 |

## R0–R4 独立评估

| Method | Accuracy | Macro-F1 | Balanced accuracy | Mean regret |
|---|---:|---:|---:|---:|
| R0_category_only | 0.686944 | 0.587851 | 0.692424 | 0.096283 |
| R1_heuristic | 0.686944 | 0.587851 | 0.692424 | 0.096283 |
| R3_memory_only | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| R2_policy_only | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| R4_policy_plus_memory | 1.000000 | 1.000000 | 1.000000 | 0.000000 |

## Release

- Release manifest SHA256：`b53cbfdb30fa7cabeb774a23c2687a61ec67e6e30eb50ea9c974d0078943f558`
- 部署包：`/workspace/pavg/artifacts/repair_agent_repair_1200g_v3_action_value.tar.gz`
- 部署包 SHA256：`53c49359969a00555f49c7afad7c904a7bc28968395bb73e87db2ac823c2a4e2`
- 四动作 inference smoke：`True`
- 文件数：10

## Blender shards 清理

- /workspace/pavg/campaigns/repair_hard_150g_v3_node_a: 125417326 bytes
- /workspace/pavg/campaigns/repair_hard_95g_v3_node_a_v11: 199040242 bytes
- /workspace/pavg/campaigns/repair_hard_150g_v3_node_b: 229330475 bytes
- /workspace/pavg/campaigns/repair_hard_45g_v3_node_b_v11: 94695402 bytes

## 已知边界

- 当前数据为 Blender proxy 标签，不是实际 Executor 闭环回报。
- 当前结果不能解释为 HunyuanVideo 修复成功率。
- `source_revision=unknown`，在干净且经过团队评审的 revision 前不得提升为正式部署版本。
- Release 已包含 Executor-facing Action-Value Policy，但真实 Prompt/Global/Local 后端仍需部署侧注入。

## 团队下一步

1. 接入真实 Prompt、Global、Local Executor，生成 `RepairTrialV1`。
2. 将失败动作作为真实负 utility 纳入 Memory；未执行动作继续保持未知。
3. 构建严格分离的 Hunyuan calibration/test campaign。
4. 在真实 Trial 数据上重新训练，切换为 value-led selection。
