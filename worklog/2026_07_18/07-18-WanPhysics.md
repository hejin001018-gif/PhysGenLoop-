# WanPhysics：物理视频生成系统技术架构

---

## 1. 项目概述

WanPhysics 是一个基于 **Wan2.2-TI2V-5B** 文本引导图生视频模型的物理视频生成系统。

### 核心能力
- 接收文本 prompt，生成符合物理规律的短视频（1280×704）
- 支持批量化 prompt → 生成 → 自动同步 的完整流水线
- 输出目录结构已预留 critic 接口，为 Generate→Critic→Refine 闭环做好准备

### 技术栈
| 组件 | 技术 |
|------|------|
| 视频生成模型 | Wan2.2-TI2V-5B（50亿参数） |
| 运行平台 | Linux 服务器（px-cloud1） |
| GPU | NVIDIA（CUDA 12.x） |
| 模型加载 | 支持 offload 和 t5_cpu 以节省显存 |
| 本地控制器 | Mac 终端（bash + ssh + rsync） |

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     Mac 本地（控制层）                          │
│                                                                 │
│  prompts/            WanPhysics/outputs/                        │
│  ├── 0001.txt        ├── 0001/                                  │
│  ├── 0002.txt        │   ├── 0001-v01.mp4                       │
│  └── ...             │   ├── prompt.txt                         │
│                       │   ├── metadata.json                     │
│  scripts/             │   └── critic.json        ← 预留Critic接口│
│  ├── run_wan.sh      ├── 0002/                                  │
│  ├── batch_generate.sh   └── ...                                │
│  └── generate_video.sh                                          │
│                                                                 │
└────────────────────┬────────────────────────────────────────────┘
                     │ SSH + rsync
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                  服务器 px-cloud1（生成层）                      │
│                                                                 │
│  /root/WanPhysics/                                              │
│  ├── Wan2.2_code/          ← Wan2.2 推理代码                    │
│  │   └── generate.py                                            │
│  ├── models/                                                    │
│  │   └── Wan2.2-TI2V-5B/   ← 模型权重（约 10GB）               │
│  ├── outputs/              ← 生成结果目录                       │
│  │   └── {ID}/                                                   │
│  │       ├── {ID}-v01.mp4                                       │
│  │       ├── prompt.txt                                         │
│  │       ├── metadata.json                                      │
│  │       └── critic.json                                        │
│  └── scripts/                                                    │
│      └── generate_video.sh   ← 服务器端生成脚本                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 工作流详解

### 3.1 完整执行流程

```
Step 1: 写 prompt 文件
        Mac: prompts/0001.txt
        └── 内容: "A car braking on wet road..."

Step 2: 触发批量生成
        Mac: bash scripts/batch_generate.sh
        └── 逐个读取 prompts/*.txt

Step 3: 上传并远程执行（run_wan.sh）
        Mac: ssh -> px-cloud1
        └── ssh -p 27188 root@px-cloud1.matpool.com
            "cd /root/WanPhysics &&
             ./scripts/generate_video.sh \"0001\" \"A car braking...\""

Step 4: 服务器端生成（generate_video.sh）
        px-cloud1:
        ├── conda activate myconda
        ├── mkdir -p outputs/0001/
        ├── echo prompt > outputs/0001/prompt.txt
        ├── python Wan2.2_code/generate.py \
        │   --task ti2v-5B \
        │   --size 1280*704 \
        │   --ckpt_dir models/Wan2.2-TI2V-5B \
        │   --offload_model True \
        │   --convert_model_dtype \
        │   --t5_cpu \
        │   --save_file outputs/0001/0001-v01.mp4 \
        │   --prompt "A car braking..."
        ├── 写入 outputs/0001/metadata.json
        └── 写入 outputs/0001/critic.json（状态: waiting）

Step 5: 同步回本地
        Mac: rsync -avz -e "ssh -p 27188" \
             px-cloud1:/root/WanPhysics/outputs/0001/ \
             ~/Desktop/WanPhysics/outputs/0001/

Step 6: 本地查看
        Mac: ~/Desktop/WanPhysics/outputs/0001/0001-v01.mp4
```

### 3.2 各组件职责

| 文件 | 所在位置 | 职责 |
|------|---------|------|
| `run_wan.sh` | Mac 本地 | Mac 端的唯一入口：SSH 连接 + 调用远程生成 + rsync 同步回本地 |
| `batch_generate.sh` | Mac 本地 | 批量读取 prompts/*.txt，循环调用 run_wan.sh |
| `generate_video.sh` | 服务器 | conda 环境激活 + 调用 Wan2.2 推理 + 写 metadata/critic.json |
| `generate.py` | 服务器 | Wan2.2 官方推理脚本，加载模型并生成视频 |

---

## 4. 数据接口规范

### 4.1 输入接口

**Prompt 格式**（`prompts/{ID}.txt`）：
```
纯文本文件，一行或多行。
例如："A realistic physics simulation of a car braking on a wet road..."
```

### 4.2 输出接口

**`outputs/{ID}/` 目录结构：**

```
outputs/{ID}/
├── {ID}-v01.mp4          ← 生成的视频文件（1280×704）
├── prompt.txt            ← 原始 prompt
├── metadata.json         ← 元数据（id, model, prompt, video）
└── critic.json           ← 预留 Critic 接口（初始状态: waiting）
```

**`metadata.json` 格式：**
```json
{
    "id": "0001",
    "model": "Wan2.2-TI2V-5B",
    "prompt": "A car braking on wet road...",
    "video": "0001-v01.mp4"
}
```

**`critic.json` 格式（预留）：**
```json
{
    "video": "0001-v01.mp4",
    "status": "waiting",           // waiting | completed | failed
    "physics_violation": null,      // true | false | null
    "reason": null,                 // VLM 分析文本
    "confidence": null              // 0.0 ~ 1.0
}
```

---

## 5. Wan2.2 模型信息

### 5.1 使用模型

| 属性 | 值 |
|------|-----|
| 模型名称 | **Wan2.2-TI2V-5B** |
| 参数量 | 50亿 |
| 类型 | 文本引导图生视频（Text-Image-to-Video） |
| 仓库路径 | `/root/WanPhysics/models/Wan2.2-TI2V-5B` |

### 5.2 推理参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `--task` | `ti2v-5B` | TI2V 任务 + 5B 模型 |
| `--size` | `1280*704` | 输出视频分辨率 |
| `--offload_model` | `True` | 减少显存占用 |
| `--convert_model_dtype` | `True` | 模型精度转换 |
| `--t5_cpu` | `True` | T5 文本编码器在 CPU 上运行 |

---

## 6. 接入 Critic 的接口定义

### 6.1 当前已预留的接口

生成完成后 `generate_video.sh` 自动创建 `critic.json`，内容为空占位（`status: "waiting"`）。

### 6.2 需要暴露的最小接口

为了让 `pavg_critic` 作为独立模块接入 WanPhysics。

#### 接口 A：通过文件系统对接（推荐，零侵入）

```python
# WanPhysics 生成后
outputs/0001/
├── 0001-v01.mp4     ← pavg_critic 读取此文件
├── prompt.txt        ← pavg_critic 读取 prompt
└── metadata.json     ← pavg_critic 读取元数据

# pavg_critic 处理后写入
└── critic_v01.json   ← 写入评分结果
```

对接方式：生成完成后调用 pavg_critic CLI：
```bash
pavg-critic --request critic_request.json --output outputs/0001/critic_v01.json
```

#### 接口 B：通过 `physgenloop` 协议对接（完整闭环）

```python
from physgenloop.wan_generator import WanVideoGenerator
from physgenloop.controller import LoopController

# WanVideoGenerator 已实现 VideoGenerator 协议
# 直接接入 LoopController 实现 Generate→Critic→Refine→Generate 闭环

generator = WanVideoGenerator(
    remote_host="root@px-cloud1.matpool.com",
    remote_port=27188,
)
```

---

## 7. 部署要求

### 服务器要求

| 资源 | 要求 |
|------|------|
| GPU 显存 | ≥ 16GB（建议 24GB） |
| 磁盘空间 | ≥ 50GB（模型 10GB + 视频输出） |
| CUDA 版本 | 12.x |
| Python | 3.8+ |
| 推理框架 | Wan2.2 原生推理 |

### Mac 本地要求

| 工具 | 用途 |
|------|------|
| `ssh` | 远程连接 |
| `rsync` | 文件同步 |
| `sshpass`（可选）| 密码认证 |

---

## 8. 后续优化方向

### 短期
1. 生成完成后自动触发 Critic 分析
2. Critic 结果回写 `critic.json`
3. 支持多版本（v01, v02, ...）

### 中期
1. 接入 `physgenloop.LoopController` 实现自动闭环
2. VLM 修复 prompt 自动注入下一轮生成
3. 并行生成多个候选视频（Best-of-K）

### 长期
1. 本地 prompt 管理 + 实验结果追踪
2. 多模型支持（Wan2.1-T2V-14B 等）
3. Web 界面管理生成任务

---

## 附录 A：关键脚本内容

### `run_wan.sh`（Mac 端唯一入口）
```bash
ID=$1
PROMPT=$2
SERVER="root@px-cloud1.matpool.com"
PORT=27188

ssh -p $PORT $SERVER "cd /root/WanPhysics && ./scripts/generate_video.sh \"$ID\" \"$PROMPT\""
rsync -avz -e "ssh -p $PORT" $SERVER:/root/WanPhysics/outputs/$ID ~/Desktop/WanPhysics/outputs/
```

### `generate_video.sh`（服务器端核心脚本）
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myconda

python $BASE_DIR/Wan2.2_code/generate.py \
    --task ti2v-5B \
    --size 1280*704 \
    --ckpt_dir $BASE_DIR/models/Wan2.2-TI2V-5B \
    --offload_model True \
    --convert_model_dtype \
    --t5_cpu \
    --save_file "$TASK_DIR/$VIDEO_NAME.mp4" \
    --prompt "$PROMPT"
```