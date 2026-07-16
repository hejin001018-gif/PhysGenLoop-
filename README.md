# PAVG · PhysGenLoop

**Physics-Aware Agentic Video Generation**

通过构造物理真值数据、训练 Physics Critic、接入视频生成模型，形成「规划 → 生成 → 检测 → 修复 → 再生成」的物理一致性闭环。

---

## 项目约束

- **核心问题**：现有视频生成模型缺乏物理一致性，本项目提供可执行的检测与修复反馈。
- **两个包**：`src/pavg_critic`（Physics Planner + Critic，可独立运行）和 `src/physgenloop`（闭环编排层）。
- **数据路线**：Blender 合成带精确真值的正常/异常视频 → 训练/组合 Critic → 接入 HunyuanVideo 闭环。
- **当前阶段**：Critic 基线与 benchmark 评估，闭环编排契约已冻结，真实生成服务尚未接入。
- **不跟踪**：模型权重、生成视频、大型数据集（见 `.gitignore`）。

---

## 目录结构

```
PhysGenLoop-/
├── README.md                          # 本文件：项目约束与目录索引
├── pyproject.toml                     # 包定义（pavg_critic + physgenloop）
├── configs/
│   └── default.yaml                   # Critic 权重 / Repair 阈值 / 评测基准配置
├── schemas/                           # 跨模块契约（JSON Schema）
│   ├── sample.schema.json             #   单样本标注契约
│   ├── critic_output.schema.json      #   Critic 输出契约
│   └── README.md
├── src/
│   ├── pavg_critic/                   # Physics Planner & Critic 核心包
│   │   ├── planner.py                 #   Prompt → PhysicsPlan
│   │   ├── pipeline.py                #   主评估流水线
│   │   ├── detector.py / tracker.py   #   视觉检测与跟踪
│   │   ├── trajectory.py              #   轨迹提取
│   │   ├── event_detector.py          #   物理事件检测
│   │   ├── physics_rules.py           #   确定性规则引擎
│   │   ├── mechanics.py               #   力学验证
│   │   ├── pqsg.py                    #   物理问题场景图
│   │   ├── evidence_fusion.py         #   多路证据融合
│   │   ├── vlm_verifier.py            #   VLM 复核与解释
│   │   ├── keyframe_selector.py       #   关键帧选择
│   │   ├── temporal_localizer.py      #   异常时序定位
│   │   ├── sam2_detector.py           #   SAM2 可选后端
│   │   ├── checklist.py               #   VideoScience 证据清单
│   │   ├── benchmarking/              #   benchmark 评测子包
│   │   │   ├── runner.py              #     可恢复评测运行器
│   │   │   ├── metrics.py             #     指标计算
│   │   │   ├── datasets.py            #     数据集加载
│   │   │   ├── baselines.py           #     基线方法
│   │   │   └── diagnostics.py        #     诊断工具
│   │   └── cli.py / api_models.py     #   CLI 与 API 数据模型
│   └── physgenloop/                   # 闭环编排包
│       ├── contracts.py               #   闭环数据契约
│       ├── interfaces.py              #   Generator / Critic / Repairer 协议
│       ├── controller.py              #   Best-of-K 编排控制器
│       ├── critic_adapter.py          #   pavg_critic 适配层
│       ├── generator.py               #   视频生成器（当前为 fake）
│       ├── repairer.py                #   修复指令聚合
│       └── selector.py                #   候选视频选择器
├── agents/                            # Agentic 闭环（独立探索层）
│   ├── prompt_rewriter.py             #   LLM 改写层（Claude/OpenAI/Stub）
│   ├── repairer.py                    #   Repair Agent 决策器
│   ├── video_backend.py               #   视频后端抽象（local/replicate/fal/stub）
│   └── README.md
├── benchmarks/                        # 外域评测入口脚本
│   ├── evaluate_critic.py
│   ├── evaluate_video_benchmark.py
│   ├── prepare_videophy_manifest.py
│   └── diagnose_pavg_predictions.py
├── evaluation/                        # 评测数据与 manifests
│   ├── manifests/                     #   VideoPhy2 smoke manifests
│   └── fixtures/                      #   单元测试 fixture
├── examples/                          # 使用示例
│   ├── critic_request.json
│   ├── observations.json
│   └── evaluate_video.py
├── data/
│   └── samples/                       # 单样本目录（对齐 sample.schema.json）
├── data_pipeline/                     # 数据处理管道（骨架）
│   ├── kubric_adapter/
│   ├── physion_loader/
│   └── intphys2_loader/
├── models/                            # 模型层（骨架，待填充）
│   ├── critic/
│   └── vlm/
├── generators/                        # 视频生成探针（骨架）
├── experiments/                       # 消融实验脚本（骨架）
├── tests/                             # 测试
│   ├── benchmarking/                  #   benchmarking 子包测试
│   └── test_*.py                      #   各模块单元测试
├── docs/
│   ├── operation-guide.md             #   部署与运维指南
│   ├── results/                       #   实验结果记录
│   └── superpowers/                   #   AI 辅助生成的计划与设计文档
│       ├── plans/
│       └── specs/
├── outputs/                           # 生成产物（不入库）
└── worklog/                           # 迭代日志（按日期分卷）
    ├── 2026_7_14/
    └── 2026_07_15/
```
