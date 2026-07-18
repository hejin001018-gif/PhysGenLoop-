# PAVG 项目路演 PPT 与宣传海报设计

日期：2026-07-17  
状态：叙事结构与视觉方向已获用户确认  
交付范围：16 页中文项目路演 PPT + 1 张 A3 竖版中文宣传海报

## 1. 沟通任务

到路演结束时，项目评审应相信：**PAVG 不是一个单点视频评测器，而是一套把自然语言物理规划、多证据物理评估、学习型修复策略和有界再生成连接起来的完整视频物理闭环系统。**

目标受众为项目路演与竞赛评审，演讲时长约 12 分钟。内容优先解释创新性、完整性、可审计性与平台价值，不展开安装命令、代码结构或实验执行细节。

## 2. 内容边界

### 2.1 完成态表达

所有系统模块统一按完整产品能力、使用现在时呈现：

- PhysicsPlan Resolver；
- 视频生成服务与候选生成；
- SAM2 连续跟踪、轨迹与事件提取；
- Rules、PQSG、VideoScience Checklist、Morpheus Mechanics 与 VLM 复核；
- Coverage-aware Evidence Fusion；
- Learning Repair Agent、Action-Value Policy 与 Repair Memory；
- Prompt Repair、Global Regeneration、Local Editing 与 Reject/回退；
- Blender/Kubric 物理真值数据引擎；
- Best-of-K 选择与有界 Agentic Feedback Loop。

对外品牌名只使用 **PAVG**。`PhysGenLoop` 仅作为内部仓库或编排模块名，不进入主视觉标题。

### 2.2 禁止内容

- 不展示 benchmark 数字、准确率、Macro-F1、置信区间或排名；
- 不展示“当前/未来”“已完成/未完成”状态标签；
- 不虚构实验数据、客户、机构合作、落地规模或性能领先结论；
- 不使用服务器地址、凭据、绝对路径或内部协作信息；
- 不使用代码截图、终端截图或密集表格。

## 3. 叙事方案

采用“创新闭环型”路演结构：

```text
行业缺口
  → PAVG 核心判断
  → 完整闭环全景
  → 四项关键创新
  → 核心模块如何协同
  → 平台价值
  → 品牌主张收束
```

每页只承担一个叙事任务，标题使用结论句而非栏目名。整套演示不设置独立目录页，不以“谢谢”空页结束。

## 4. 16 页页纲

1. **PAVG——让生成视频遵守物理世界**：品牌与主张。
2. **视频越来越逼真，但仍会违背基本物理规律**：行业问题。
3. **生成模型需要的不只是更强采样，而是物理反馈闭环**：方案判断。
4. **PAVG 让视频生成具备自我纠错能力**：Plan → Generate → Critic → Repair → Select 全景。
5. **PhysicsPlan 把自然语言转化为可执行物理约束**：创新一。
6. **对象、事件、关系与物理定律组成可执行问题图**：结构化理解。
7. **Physics Critic 用多路证据回答“哪里错、为什么错”**：创新二。
8. **SAM2 把离散画面连接成连续运动证据**：连续感知。
9. **Rules、PQSG、Checklist、Mechanics 与 VLM 共同裁决**：混合推理。
10. **覆盖感知融合让系统知道何时确认、何时拒绝、何时保留判断**：可信决策。
11. **Learning Repair Agent 学习“什么错误该怎么修”**：创新三。
12. **四级修复覆盖从提示词到局部视频编辑**：动作空间。
13. **Action-Value Policy 与 Repair Memory 让修复策略持续进化**：经验学习。
14. **Blender/Kubric 数据引擎提供可控、精确、可扩展的物理真值**：数据基础设施。
15. **PAVG 同时服务评测、生成优化、训练数据与模型研究**：平台价值。
16. **让每一帧不仅看起来真实，也在物理上成立**：品牌收束。

## 5. 视觉系统

采用已确认的“科研旗舰”方向。

### 5.1 色彩

- Ivory：`#F6F3EC`，主背景；
- Royal Blue：`#244BDB`，品牌与核心链路；
- Coral：`#FF684A`，异常、动作和视觉焦点；
- Charcoal：`#111827`，正文与深色强调；
- Pale Blue：`#DFE8FF`，结构分区和辅助背景。

### 5.2 字体

- 中文：Microsoft YaHei / 等宽回退前的系统中文无衬线字体；
- 英文与数字：Aptos / Arial；
- PPT 标题不低于 35 pt，封面标题不低于 50 pt，正文不低于 16 pt；
- 海报标题保持远距离可读，正文只保留一句话级信息。

### 5.3 图形语言

- 抛物线、碰撞轨迹和对象跟踪线；
- 关键帧括号、时间段标记和节点连线；
- 对象/动作/物理三类问题图节点；
- Royal Blue 表示可验证的系统链路，Coral 表示物理异常或修复动作；
- 避免 UI dashboard、卡片墙、按钮式装饰和重复模块面板。

## 6. 视觉资产

PPT 使用原创的科研插画或生成式图像作为主视觉，不使用数据图表：

1. 封面/海报：抽象球体轨迹与物理世界边界；
2. 行业缺口：看似真实但物理异常的视频关键帧组合；
3. Physics Critic：对象跟踪、轨迹、关键帧与证据融合的科研信息图；
4. Learning Repair：CriticReport 进入四动作 Action-Value Policy 的概念图；
5. 数据引擎：Blender/Kubric 场景、真值轨迹、接触与 mask 的抽象合成图。

所有图片禁止水印、虚构品牌、实验数字和难以校正的长文本。系统文字由 PowerPoint 原生文本承载，不交给图片生成模型绘制。

## 7. A3 竖版海报

海报尺寸采用 A3 竖版比例。信息结构：

1. 顶部：`PAVG` + “让生成视频遵守物理世界”；
2. 主视觉：Royal Blue 轨迹、Coral 物体与物理闭环意象；
3. 中部：`PLAN → GENERATE → CRITIC → REPAIR → SELECT`；
4. 下部四项创新：PhysicsPlan、Multi-Evidence Critic、Learning Repair、Auditable Loop；
5. 页脚：`Physics-Aware Agentic Video Generation`。

海报不展示数据结果、不放二维码、不添加未经提供的学校、实验室或合作方标识。

## 8. 输出文件

- `outputs/PAVG_项目路演.pptx`：16:9、16 页、可编辑；
- `outputs/PAVG_宣传海报.png`：A3 竖版高分辨率 PNG；
- 生成式主视觉素材存放于 `outputs/pavg-promo-assets/`，便于后续复用。

## 9. 制作与质检

- PPT 使用 `@oai/artifact-tool` 的 JavaScript ES module 生成；
- 每页渲染为 PNG 并逐页全尺寸检查；
- 执行 slide overflow/overlap 检查，修复所有非预期重叠、裁切、换行和超界；
- 检查 16 页标题无意外折行、字体层级一致、图像裁切稳定；
- 海报检查 A3 比例、远距离标题可读性、文字无乱码、图片无水印；
- 最终核对 PPT 与海报均不包含 benchmark 数字或未授权标识。
