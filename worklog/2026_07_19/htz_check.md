# WanPhysics 构想与合并后代码一致性审查

> **审查人：** htz（AI 辅助）  
> **审查日期：** 2026-07-19  
> **审查范围：** `/root/PhysGenLoop-/` 中与 WanPhysics 相关的代码  
> **审查依据：**
>
> - htz 原始 WanPhysics 架构
> - `run_wan.sh` 工作流
> - `generate_video.sh` 服务器脚本


---

# 一、审查结论

## 总体评价

**合并后的代码方向正确，但存在 6 项与原始 WanPhysics 构想不一致的问题。**

| 维度 | 是否符合 | 说明 |
|---|---|---|
| 模型选择 | ✅ 符合 | Wan2.2-TI2V-5B（50亿参数） |
| 推理参数 | ✅ 符合 | 1280×704、offload、bf16 |
| 输出目录结构 | ⚠️ 部分符合 | 字段名称及约定略有不同 |
| Mac 本地控制 | ❌ 不符合 | 完全丢弃 Mac → SSH → server 的远程调用模式 |
| 单 prompt 生成入口 | ❌ 不符合 | 丢失 run_wan.sh 简洁启动语义 |
| prompts/ 目录用法 | ❌ 不符合 | batch_run_loop.py 不再读取 prompts/*.txt |
| critic.json 接口 | ❌ 不符合 | 合并后的闭环不再写入 critic.json |
| 文件命名与种子策略 | ❌ 不符合 | candidate ID 不兼容原 output 目录约定 |

---

# 二、逐项差异分析


# 差异 1：Mac 本地 SSH / rsync 控制流被完全丢弃

## 原始构想

```
Mac 本地（控制层）                  服务器（生成层）

run_wan.sh ID PROMPT    --->      generate_video.sh

rsync 同步回本地        <---      outputs/{ID}/
```

设计原则：

- Mac 是**控制端**
- Server 是**纯生成端**
- 人在 Mac 上编写 prompt
- SSH 触发生成
- rsync 拉回结果

优势：

- 无需修改服务器生成代码
- 控制层和生成层解耦
- 方便本地管理实验任务


---

## 合并后结构

```
服务器（一体化运行）

agents/wanphysics/run_loop.py

        ↓

WanSubprocessGenerator

        ↓

gen_step.py
```

存在问题：

- Mac 控制层被完全移除
- 所有逻辑迁移到服务器
- 无法本地离线管理 prompt
- 降低实验灵活性


## 建议

不一定完全恢复 SSH 架构，但至少保留：

```
Mac
 |
 | prompts/*.txt
 |
同步
 |
Server
 |
生成
```

---

# 差异 2：单 Prompt 生成入口脚本丢失


## 原始构想

Mac 端唯一入口：

```bash
cd ~/Desktop/WanPhysics

./scripts/run_wan.sh 0015 \
"A car braking on wet road"
```


职责：

- 上传 prompt
- SSH 调用服务器
- 等待生成
- rsync 返回结果


---

## 合并后

```bash
python agents/wanphysics/run_loop.py \
--prompt "..."
```


问题：

`run_loop.py` 同时承担：

- 视频生成
- Critic 调用
- 循环控制
- 结果管理


导致：

调试单独生成效果时：

> 必须运行完整 Physical CoT Loop

缺少：

```
generate_only.py
```

这种纯生成入口。


---

# 差异 3：prompts/ 批量接口被绕过


## 原始设计


目录：

```
WanPhysics/

├── prompts/
│   ├── 0015.txt
│   ├── 0016.txt
│   └── 0017.txt
│
└── outputs/
```


运行：

```bash
bash scripts/batch_generate.sh
```


规则：

```
prompts/0015.txt

        ↓

outputs/0015/
```


其中：

文件名 = Task ID


---

## 合并后


运行：

```bash
python agents/wanphysics/batch_run_loop.py \
--prompts-dir ./prompts
```


虽然支持批量 prompt：

但是输出：

```
outputs/

└── run_20260719_170012/

    └── wan-71077cad56b8/
```


问题：

- Task ID 不再来自文件名
- 无法通过目录快速定位实验
- 不符合原始实验管理逻辑


---

# 差异 4：输出目录结构不兼容


## 原始结构


```
outputs/0015/

├── 0015-v01.mp4

├── prompt.txt

├── metadata.json

└── critic.json
```


---

## critic.json 初始接口


```json
{
    "video": "0015-v01.mp4",
    "status": "waiting",
    "physics_violation": null,
    "reason": null,
    "confidence": null
}
```


设计目的：

生成器和 Critic 解耦。


---

## 合并后结构


```
outputs/

└── run_20260719_170012/

    └── wan-71077cad56b8/

        ├── wan-71077cad56b8-v01.mp4

        ├── prompt.txt

        ├── metadata.json

        └── loop_result.json
```


---

## 差异表


| 字段 | 原始 WanPhysics | 合并后 |
|-|-|-|
| 目录命名 | outputs/{task_id}/ | outputs/run_时间/{candidate_id}/ |
| ID 来源 | prompt 文件名 | sha256 hash |
| 视频命名 | {ID}-v01.mp4 | {candidate_id}-v01.mp4 |
| critic.json | ✅ 存在 | ❌ 缺失 |
| loop_result.json | ❌ | ✅ 存在 |


---

# 差异 5：critic.json 文件接口被废弃


## 原始设计


生成阶段：

```
Generator

↓

video.mp4

↓

critic.json
```


生成器只负责：

```json
{
    "video": "0015-v01.mp4",
    "status": "waiting"
}
```


---

Critic 修改：


```json
{
    "video": "0015-v01.mp4",
    "status": "completed",
    "physics_violation": true,
    "reason": "物体在56-60帧消失",
    "confidence": 0.86
}
```


---

## 优点


生成器：

- 不关心 Critic
- 不依赖 VLM


Critic：

- 不关心生成过程
- 只读取视频


两者：

> 文件接口解耦


---

## 合并后


当前：

```
LoopController

    ↓

Python object

    ↓

Critic
```


问题：

- 模块耦合增加
- 难以替换 Critic
- 不利于未来接入 pavg_critic


---

# 差异 6：Candidate ID 策略不兼容


## 原始设计


```python
candidate_id = f"wan-{seed:04d}"
```


例如：

```
wan-0042
```


优点：

- 可读
- 与 seed 对应
- 方便实验追踪


---

## 合并后


```python
candidate_id = (
    f"wan-{sha256(prompt + seed)[:12]}"
)
```


例如：

```
wan-71077cad56b8
```


问题：

- 不可读
- 无法快速定位 seed
- 路径过长
- 不方便人工管理


---

# 三、总结


## 保持一致部分


| 项目 | 状态 |
|-|-|
| Wan2.2-TI2V-5B | ✅ |
| 1280×704 推理参数 | ✅ |
| 81 帧 / 24fps | ✅ |
| prompt.txt 保存 | ✅ |
| metadata.json 保存 | ✅ |
| 多轮生成能力 | ✅ |
| Batch prompt 支持 | ✅ |


---

# 四、建议修复方案


| 优先级 | 问题 | 修复方案 |
|-|-|-|
| 🔴 高 | critic.json 消失 | gen_step.py 创建 critic.json，初始 status=waiting |
| 🟡 中 | 输出目录不兼容 | 保留 outputs/{task_id}/ 结构 |
| 🟡 中 | 缺少纯生成入口 | 新增 generate_only.py |
| 🟢 低 | ID 不可读 | 使用 wan-{seed:04d} |
| 🟢 低 | batch 输出不兼容 | 使用 prompt 文件名作为 task_id |


---

# 五、最终建议架构


推荐恢复为：

```
Mac 控制层

WanPhysics/

├── prompts/
│
├── scripts/
│   ├── run_wan.sh
│   └── batch_generate.sh
│
└── outputs/


          SSH


Server 生成层

PhysGenLoop/

├── generate_video.sh

├── gen_step.py

└── critic interface


          ↓


outputs/{task_id}/

├── video.mp4

├── prompt.txt

├── metadata.json

└── critic.json
```


这样可以同时保留：

- Wan2.2 生成能力
- Physical CoT Loop
- pavg_critic 解耦接口
- Mac 实验管理能力


---

**审查结论：**

> 合并后的代码具备完整 Physical Loop 雏形，但偏向“服务器端一体化 Agent 系统”。  
> 若目标是构建长期可扩展的 WanPhysics 平台，建议恢复原始的 **Mac 控制层 + 文件接口 + critic.json 解耦设计**。