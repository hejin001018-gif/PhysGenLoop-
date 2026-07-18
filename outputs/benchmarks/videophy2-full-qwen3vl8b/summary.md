# VideoPhy-2 全量评测报告

样本数：3397；终止预测数：6794。

## 完整指标

| 方法 | Accuracy | Macro-F1 | Physical recall | Violation recall | Failure rate |
|---|---:|---:|---:|---:|---:|
| D0_DIRECT_VLM | 0.551663 | 0.548897 | 0.599440 | 0.498759 | 0.000000 |
| B1_RULE | 0.544598 | 0.544539 | 0.548459 | 0.540323 | 0.001472 |

候选减基线 Macro-F1：-0.004359；Accuracy：-0.007065。

## 配对 action-group bootstrap

重采样次数：2000；seed：20260717；cluster 数：198。
Macro-F1 差值点估计：-0.004359；95% CI：[-0.031613, +0.020693]。

## 配对结果

- 两者均正确：1036
- 仅基线正确：838
- 仅候选正确：814
- 两者均错误：709

## 延迟

预测延迟（模型/规则）与 SAM2 轨迹生产耗时分别统计，不可相加或混用。

### 预测延迟（模型/规则）

- D0_DIRECT_VLM: mean=3.297850s, p50=2.920166s, p95=5.354119s
- B1_RULE: mean=0.041080s, p50=0.003806s, p95=0.013224s

### SAM2 production latency

valid=3397, missing=0, mean=29.736575010940236, p50=23.946251904591918, p95=66.27397682890296 秒。

## 失败记录

预测失败 5 / 6794 (0.000736)。失败保留在分母内。

- videophy2&#45;1a4d8e4b16713ff507aa / B1&#95;RULE: VLM produced no object seeds for SAM2 tracking
- videophy2&#45;220b8130a3c7f8a6db0a / B1&#95;RULE: VLM produced no object seeds for SAM2 tracking
- videophy2&#45;99bd503a4ad4ccabdfc6 / B1&#95;RULE: VLM produced no object seeds for SAM2 tracking
- videophy2&#45;eada40f8e7e559114743 / B1&#95;RULE: VLM produced no object seeds for SAM2 tracking
- videophy2&#45;fbea2c13fc8bb62999c3 / B1&#95;RULE: Model API request failed: timed out

## 冻结门槛

- macro_f1_delta: value=-0.0043586177028469, operator=>=, threshold=0.05, pass=false
- bootstrap_lower: value=-0.031613432050651695, operator=>, threshold=0.0, pass=false
- candidate_nonzero_recalls: value={'physical_recall': 0.5484593837535015, 'violation_recall': 0.5403225806451613}, operator=>, threshold={'physical_recall': 0.0, 'violation_recall': 0.0}, pass=true
- failure_rate_increase: value=0.0014718869590815426, operator=<=, threshold=0.01, pass=true
- positive_generator_count: value=3, operator=>=, threshold=2, pass=true

VideoPhy-2-only support：false。

## OOD 限制（醒目）

> **VideoPhy-1 OOD：deferred。**
> **overall: not_evaluable_ood_deferred。**

本报告只覆盖冻结的 VideoPhy-2 全量比较；在 VideoPhy-1 OOD 完成前，不能据此声称架构已被证明。
