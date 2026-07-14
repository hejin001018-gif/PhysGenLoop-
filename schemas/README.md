# schemas/ — 骨骼契约

三份 JSON Schema 定义了整个 PAVG 阵法的数据契约。任何组件读写这些结构前，先用 `jsonschema` 校验。

| 文件 | 用途 | 消费方 |
|------|------|--------|
| `sample.schema.json` | 单样本目录的元数据 + 标注 | Kubric 适配器 · Physion/IntPhys 2 loader · 训练脚本 |
| `critic_output.schema.json` | Physics Critic 输出 | Critic · Repair Agent · Selector · Evaluation |
| `generator_request.schema.json` | HunyuanVideo 调用契约 | Planner · Repair Agent · `generators/hunyuan_probe.py` |

## 校验示例

```python
import json, jsonschema
schema = json.load(open("schemas/sample.schema.json"))
payload = json.load(open("data/samples/gravity_001/annotation.json"))
jsonschema.validate(payload, schema)
```

版本管理：`schema_version` 目前固定 `"1.0"`，破坏性改动必须升版号并在此 README 追加迁移说明。
