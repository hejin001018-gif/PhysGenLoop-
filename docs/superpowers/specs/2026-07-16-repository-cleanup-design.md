# 仓库清理设计

## 目标

在保留现有功能、基准评测、结果文档、本机环境和本地数据的前提下，让仓库根目录和版本控制内的结构更清晰。完成后，将经过验证的改动以快进方式推送到远端 `main` 分支。

## 选定方案

采用最小化结构整理。不重新设计源码包边界，也不改写历史开发文档。保留当前的 VLM 检测器、API 适配器、benchmark 记录、结果报告和本地评测资源。

## 版本控制内的结构调整

- 将端到端视频命令从 `test.py` 移到 `examples/evaluate_video.py`。该文件是可执行示例，不是 pytest 测试模块，新路径能够准确表达其职责。
- 将根目录的 `项目总体思路(2).pdf` 移到 `docs/archive/project-overview.pdf`。文档继续纳入版本控制，但不再占用仓库根目录。
- 重新组织 README，使其成为整体框架入口，并更新其中的命令和仓库结构说明。
- 更新 `.gitignore`：明确允许提交 `.env.example`；继续忽略真实 `.env` 文件、生成结果、模型权重、视频、Python 缓存、覆盖率文件和工具缓存。
- 保持 `docs/results/`、`docs/superpowers/`、`src/`、`tests/`、benchmark 清单、schema、配置和其他示例的现有职责目录不变。

## README 整体框架

README 按读者理解项目所需的顺序组织，而不是按功能开发时间堆叠：

1. 项目定位与当前能力：说明 PhysGenLoop 解决的问题、已经实现的范围和当前限制。
2. 整体闭环：展示 Prompt、Generator、Physics Critic、Selector、Repairer 与下一轮 Prompt 之间的反馈关系。
3. 分层架构与数据流：区分 `physgenloop` 的编排层和 `pavg_critic` 的物理评估层，并解释 PhysicsPlan、Observation、ViolationCandidate、CriticReport 等核心数据如何流转。
4. 模块职责与仓库结构：用目录树对应源码、benchmark、评测清单、schema、示例、测试和文档的职责。
5. 安装、环境配置和快速开始：先给最短可运行路径，再提供 Planner、无 API Critic、视频评估和最小闭环用法。
6. 评测结果与复现：链接正式结果文档，说明公开数据只保存元数据、原始视频与模型权重不进入 Git。
7. 测试、已知限制、路线图与开发文档：将维护者信息集中放在 README 后部。

README 中保留一份简洁的整体架构图和一份 Physics Critic 内部数据流图。两图必须与当前代码一致，不把尚未实现的生成模型或学习型 Repairer 描述成已完成功能。

## 本机清理边界

只删除仓库内可重新生成的缓存目录：`.pytest_cache`、Python `__pycache__` 目录和 `*.egg-info` 元数据。保留 `.env`、`.venv`、`.idea`、`.vscode`、`.agents`、`.claude`、`evaluation/external`、`outputs`、本地视频，以及所有模型和数据集资源。

## 安全与集成方式

提交前逐项审查暂存和未暂存差异。扫描版本控制内的内容是否存在常见凭据模式，但不输出秘密值。使用项目的 Python 3.12 环境和仓库内可访问的 pytest 临时目录运行完整测试；同时编译 Python 源码，并执行迁移后示例的 `--help` 冒烟测试。

远端 `main` 是当前 `sy` 分支的祖先，因此集成必须保持快进。创建职责清晰的提交；发布前再次获取远端状态并核验祖先关系；不使用强推，将 `HEAD` 推送到 `main`；最后确认远端 `main` 与本地提交一致。

## 完成标准

- 仓库根目录只保留项目元数据和主要入口文档，不再放置名称含糊的可执行脚本或归档 PDF。
- README 能从系统闭环、分层边界和核心数据流三个层面准确呈现当前整体框架。
- 不丢失用户环境、benchmark 输出、视频、模型、数据集、IDE 设置或当前功能改动。
- `.env.example` 纳入版本控制，真实凭据继续被忽略。
- 完整自动化测试、源码编译和示例帮助命令均通过。
- `origin/main` 通过快进推送指向经过验证的清理提交。
