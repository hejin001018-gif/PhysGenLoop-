# JSON Schema

- `critic_output.schema.json`：PAVG Critic 2.0 的稳定输出，包括三态决策、覆盖率、违规、问题图结果和五类证据包。
- `sample.schema.json`：保留的评估样本/人工标注契约，继续兼容现有 1.0 数据。

运行 `python -m pytest tests/test_schemas.py -q` 可检查 schema 本身及示例输出。
