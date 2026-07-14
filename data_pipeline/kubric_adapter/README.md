# kubric_adapter — Kubric → PAVG sample schema

## 目标

Kubric（Google + Bullet + Blender 渲染）是本框架 L1 引擎层主管道。本模块把 Kubric 原生输出映射为 `schemas/sample.schema.json`。

## 输入

Kubric worker 生成的样本目录，典型结构：

```
kubric_out/scene_xxx/
├── metadata.json
├── rgba_00000.png ...
├── segmentation_00000.png ...
├── depth_00000.tiff ...
├── forward_flow_00000.tiff ...
└── video.mp4         # 由 rgba 序列 ffmpeg 合成
```

## 输出

对齐 `schemas/sample.schema.json` 的 dict，可直接 `json.dump` 到 `data/samples/<sample_id>/annotation.json`。

## 使用

```bash
python -m data_pipeline.kubric_adapter.converter path/to/kubric_out/scene_xxx gravity_001
```

Python API：

```python
from data_pipeline.kubric_adapter import convert_kubric_output
payload = convert_kubric_output(root, "gravity_001", is_physical=False, violations=[...])
```

## 待办

- [ ] `trajectory.json` / `contacts.json` 导出器（从 Kubric segmentation + depth 推轨迹 & 接触事件）
- [ ] 异常注入器（reverse gravity / premature rebound / penetration / vanish）
- [ ] 与 `configs/kubric_scenes/*.yaml` 联动
