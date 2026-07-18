# 基于真实视频批量生成物理异常视频：对话总结与可复制 Prompt

## 一、当前对话与实践结果总结

### 1. 方法边界

SAM2 本身不负责生成或修改视频，它负责通过点、框或掩膜提示，在真实视频中分割并持续跟踪目标。完整异常注入流程是：

```text
真实视频
  -> SAM2 目标分割与时序跟踪
  -> 目标移除掩膜和阴影处理
  -> ProPainter 时序背景修复
  -> 修改目标轨迹或状态
  -> 重新合成目标、遮挡与阴影
  -> Original / Sham-edit / Anomaly 同规格编码
  -> 元数据、标签和严格 QA
```

### 2. 该方法适合怎样使用

- 适合合成异常训练数据、预训练、数据增强和可控消融实验。
- 可以作为 synthetic benchmark 子集，但不能单独证明真实世界泛化能力。
- 最终 benchmark 应额外使用来源独立、没有参与生成的真实异常测试集。
- 同一源视频产生的 Original、Sham 和所有异常变体必须属于同一个 `split_group`，不得跨 train/validation/test。

### 3. 已完成的示范数据

本次从 DAVIS 真实视频中生成了三组结果：

- `soccerball`：足球空中悬停；
- `drift-straight`：赛车瞬移；
- `car-turn`：汽车重力反转。

每组均包含：

- 原始正常视频；
- Sham-edit 正常编辑对照；
- 物理异常视频；
- Original / Sham / Anomaly 三栏同步对比视频；
- SAM2 掩膜、修复掩膜、异常位置掩膜、阴影掩膜；
- 逐帧异常轨迹、异常开始帧和完整元数据；
- 编码、掩膜、视觉和局部性审计报告。

最终非提示帧 SAM2 mIoU 分别为 0.9229、0.9653、0.9753；Sham 平均 PSNR 分别为 45.42、34.73、31.66 dB；三组均通过严格 QA。

### 4. 已验证的重要工程结论

1. SAM2 应使用固定、非自适应的稀疏关键帧提示，例如视频的 0%、33%、67%。如果有参考掩膜，应单独报告非提示帧 IoU，避免提示帧抬高结果。
2. OpenCV 单帧修复容易产生时序闪烁，正式数据优先使用 ProPainter 等时序修复模型。
3. 必须生成 Sham-edit 负对照，让模型不能仅依赖修复边缘、模糊或编码指纹识别异常。
4. 异常开始前，Anomaly 和 Sham 的无损中间帧必须逐像素一致。
5. ProPainter 会把尺寸调整到 8 的倍数。本次 854 像素宽的视频被内部缩放为 848 再放回 854，曾导致整帧发生轻微插值变化。正确做法是只采用修复掩膜内的 ProPainter 结果，掩膜外强制恢复源像素。
6. 汽车、人物等目标移动后必须处理原阴影。瞬移时应移除原阴影并在新位置生成柔化地面阴影；重力反转时阴影应留在地面，并随高度增加而减弱、变软。
7. 多对象强交互场景风险很高。本次 `soapbox` 中车辆、驾驶员和推动者是独立对象，单独移动汽车会留下人物和修复拖影，因此被拒绝。
8. 目标出画、重新入画或参考掩膜中断的序列不适合作为最严格的连续像素级样本。本次 `bmx-bumps` 因多帧无目标标注而被拒绝。
9. 不能通过降低质量阈值让样本通过。失败时应修正提示、掩膜、阴影或修复方法；仍不合格就拒绝样本并保留拒绝原因。
10. 正常、Sham 和异常视频必须使用完全相同的分辨率、帧率、编码器、CRF、GOP 和音频策略。

### 5. 本次采用的最低质量门槛

- 如果有参考掩膜：非提示帧 SAM2 mean IoU >= 0.75；
- Sham mean PSNR >= 30 dB；
- Sham mean SSIM >= 0.95；
- Sham minimum per-frame SSIM >= 0.95；
- Anomaly 与 Sham 在异常开始前最大像素误差为 0；
- 异常开始后必须存在可测的视觉变化；
- 最大异常位移至少达到画面对角线的 3%，除非异常类型本身不依赖空间位移；
- 无损中间帧中，声明编辑区域外最大像素误差 <= 1；
- Original、Sham、Anomaly 的帧数、fps、宽高和 codec 必须一致；
- 必须对最终编码视频重新解码并检查关键帧，而不是只检查编码前 PNG；
- 每个失败样本必须进入拒绝清单，不得静默删除或混入最终数据集。

---

## 二、可直接复制到新对话的批量生成 Prompt

复制下面整个代码块到新对话，只需先替换最顶部的路径和批量参数。若真实视频以附件形式提供，将 `INPUT_VIDEO_DIR` 改为附件或挂载目录。

```text
你是一名负责构建严格、可复现物理异常视频数据集的计算机视觉工程师。请直接执行任务，不要只给方案或示例代码；持续工作直到合格视频、全部中间文件、元数据和审计报告实际生成完毕。不要降低质量阈值来掩盖失败，无法修复的样本应明确拒绝并记录原因。

====================
一、用户参数
====================

INPUT_VIDEO_DIR = "请替换为已有真实视频目录或附件目录的绝对路径"
OUTPUT_ROOT = "请替换为本次批量生成任务的唯一输出目录绝对路径"
TARGET_SPEC = "可选：目标与异常配置 JSON/JSONL 的绝对路径；没有则填 AUTO"
VARIANTS_PER_SOURCE = 1
AUDIO_POLICY = "strip"  # 推荐 strip；也可用 preserve，但所有变体必须完全一致
DEFAULT_PROMPT_SCHEDULE = [0.0, 0.33, 0.67]
MIN_ANOMALY_CONTEXT_SECONDS = 0.8
RANDOM_SEED = 20260715

优先复用用户现有的 SAM2、ProPainter、PyTorch 和 FFmpeg。如果缺失，允许下载官方代码和权重，但所有新环境、下载文件、缓存、临时文件、日志和生成数据都必须位于 OUTPUT_ROOT 内。不得修改或覆盖原始视频；先复制或以只读方式读取，并记录源文件 SHA-256。

====================
二、任务目标
====================

对 INPUT_VIDEO_DIR 中的真实视频批量注入物理异常。每个合格样本至少生成：

1. original.mp4：统一编码后的原始正常视频；
2. sham.mp4：经过相同分割、修复、合成和编码路径，但不改变物理运动的正常编辑负对照；
3. anomaly.mp4：物理异常视频；
4. comparison.mp4：Original / Sham-edit / Anomaly 三栏逐帧同步对比视频；
5. metadata.json：源身份、split_group、异常类型、异常开始帧、逐帧参数、路径和版本；
6. SAM2 掩膜、修复掩膜、异常位置掩膜、阴影掩膜和逐帧轨迹；
7. 自动 QA、接触表、编码后关键帧和拒绝样本报告；
8. 整批数据集 manifest.jsonl、文件清单、依赖版本和 SHA-256 清单。

输出定位为 synthetic training/development data，而不是独立真实异常测试集。最终报告必须明确说明这一点。

====================
三、输入盘点与候选筛选
====================

1. 首先递归列出所有支持的视频，记录路径、SHA-256、帧数、分辨率、fps、时长、codec、是否含音频和许可/来源信息。
2. 所有衍生变体使用 source_video_id 和 split_group 绑定到源视频。同一源视频的任何版本不得跨数据集 split。
3. 如果 TARGET_SPEC 存在，严格按其中的目标、提示帧、框/点、异常类型和参数执行。
4. 如果 TARGET_SPEC=AUTO：
   - 为每个视频生成均匀采样接触表；
   - 自动选择一个清晰、连续、具有明确物理运动的主要目标；
   - 优先选择球、车辆、单人运动目标或其他轮廓完整、遮挡较少的刚性目标；
   - 避免多人紧密交互、目标长期出画、镜头切换、严重遮挡、强反射、水面、透明物体和无法可靠修复的超大目标；
   - 在正式推理前写出 candidate_selection.json，记录选择依据和风险；
   - 若同一视频存在多个同等合理目标且自动选择会明显改变任务含义，先展示接触表并只询问一个简短问题；否则按最稳妥目标继续。
5. 异常发生前必须保留至少 MIN_ANOMALY_CONTEXT_SECONDS 的正常运动上下文，使观察者能够建立物理预期。

====================
四、异常类型选择
====================

从下列异常中选择与场景匹配、视觉上明确且可高质量合成的一种。整批数据尽量平衡异常类别和强度：

- midair_hover：运动物体突然悬停；
- instant_teleport：目标瞬时发生不连续空间位移；
- gravity_reversal：目标向上加速，违反重力；
- trajectory_reversal：无合理外力时突然反向运动；
- impossible_acceleration：无合理作用力时异常加速；
- collision_passthrough：目标穿过障碍物，但只有在遮挡顺序可以可靠建模时使用；
- freeze_during_motion：目标在运动中冻结，而场景继续；
- disappearance：目标无因消失，仅在背景可以完整修复时使用；
- duplication：无因复制，仅在两个目标的遮挡、阴影都能可靠处理时使用。

第一批优先使用 midair_hover、instant_teleport 和 gravity_reversal。避免仅靠夸张形变或生成式纹理变化造成异常，因为这容易引入生成器指纹。

====================
五、SAM2 跟踪规则
====================

1. SAM2 只负责目标分割和视频传播，不要把它描述成视频生成模型。
2. 默认使用固定、非自适应关键帧提示：0%、33%、67%。可使用框、正负点或已有掩膜。
3. 所有提示必须记录 frame_index、box/points、object_id 和产生方式。
4. 掩膜保存为逐帧无损 PNG，文件名与源帧严格对齐。
5. 检查空掩膜、突然面积跳变、目标泄漏、遮挡漂移、出画和重新入画。
6. 若存在人工或数据集参考掩膜：分别报告 overall IoU 和 non-prompt-frame IoU，后者必须 >= 0.75。
7. 若没有参考掩膜：
   - 至少检查首帧、异常前一帧、异常开始帧、中间帧和末帧的叠加接触表；
   - 对掩膜面积、质心速度、边界变化进行时序异常检测；
   - 明显跟踪失败必须追加固定计划中的人工修正提示或拒绝样本；
   - 不得伪造 IoU 指标，在报告中写明“无参考掩膜”。

====================
六、背景修复、阴影和遮挡
====================

1. 从 SAM2 掩膜产生 repair mask，使用适当膨胀并填补孔洞。汽车或接地物体应向下扩展掩膜以覆盖原阴影。
2. 使用 ProPainter 或同等级时序视频修复模型生成无目标背景，保存无损背景帧。
3. 不要用逐帧 OpenCV inpaint 作为最终正式修复方案，除非只作为失败诊断或后备基线。
4. ProPainter 可能因 8 的倍数约束缩放整帧。合成前必须执行：
   background[outside_repair_mask] = original[outside_repair_mask]
   确保修复区域外保持源像素不变。
5. Sham-edit 应使用相同的 SAM2、repair mask、ProPainter、alpha 合成和编码路径，但物体保持原运动。保留部分修复痕迹作为编辑负对照，同时保证其与原视频具有高 PSNR/SSIM。
6. Anomaly 在异常开始前必须直接等于 Sham 的无损中间帧，保证逐像素一致。
7. 接地目标需要显式阴影策略：
   - 从 repair mask 中去除原阴影；
   - 瞬移时在新位置生成柔化地面阴影；
   - 重力反转时阴影留在路面，随高度增加降低不透明度并增加模糊；
   - 保存 source_shadow 和 anomaly_shadow 掩膜。
8. 存在前景障碍物时，必须另外分割障碍物或使用可靠深度/遮挡排序。不能正确恢复遮挡关系就拒绝 collision_passthrough 或相应样本。

====================
七、轨迹和合成要求
====================

1. 目标层使用每帧原始纹理，避免长时间复用单帧导致外观冻结，除非异常定义本身要求冻结。
2. 保存每帧 dx、dy、旋转、缩放、active 状态和时间戳。
3. 异常开始帧之前变换为零；悬停和重力反转应从零位移连续开始；瞬移允许在开始帧发生预期的不连续跳变。
4. 目标不得无意越界；若异常定义不要求出画，应对变换做边界约束。
5. alpha 边缘使用小范围羽化，防止黑边、锯齿和原位置残影。
6. 保存 anomaly object mask 和完整 declared edit support，后者至少包含 repair mask、异常目标 mask 和阴影 mask。

====================
八、编码与音频
====================

1. Original、Sham、Anomaly 使用同一条确定性编码路径。
2. 保持相同的分辨率、fps、codec、pixel format、CRF、preset、GOP 和 keyframe 设置。
3. 推荐 H.264/libx264、yuv420p、CRF 18、preset medium、GOP 约 2 秒。
4. comparison.mp4 只用于人工查看，可添加英文栏标题、异常状态和红色边框；单独的 original/sham/anomaly 视频不得烧录标签。
5. AUDIO_POLICY=strip 时三者全部无音频。AUDIO_POLICY=preserve 时三者必须复制完全相同的源音频；若音频会泄漏异常或与画面矛盾，则改为全部移除并记录。

====================
九、目录结构
====================

所有产生的数据必须位于 OUTPUT_ROOT。至少采用：

OUTPUT_ROOT/
  README.md
  config.json
  data/
    sources/<source_video_id>/
    provenance/
  downloads/
  external/
    sam2/
    ProPainter/
  runtime/
    env/
    cache/
    tmp/
  work/<video_id>/
    source_frames/
    masks/sam2/
    masks/repair/
    masks/alpha/
    masks/anomaly/
    masks/shadow_source/
    masks/shadow_anomaly/
    background_frames/
    original_frames/
    sham_frames/
    anomaly_frames/
    comparison_frames/
  outputs/<video_id>/
    original.mp4
    sham.mp4
    anomaly.mp4
    comparison.mp4
    metadata.json
  reports/
    quality_audit.json
    rejected_candidates.json
    encoded_keyframes/
    contact_sheets/
    reproducibility_manifest.json
  logs/
  scripts/

不得把模型缓存、pip 临时文件或中间帧写到 OUTPUT_ROOT 外。运行外部工具时显式设置 TEMP、TMP、HOME、HF_HOME、TORCH_HOME 和 XDG_CACHE_HOME 到 OUTPUT_ROOT/runtime 下。

====================
十、必须执行的严格 QA
====================

每个样本都必须执行并记录：

1. 所有中间帧、掩膜和编码视频帧数完全一致；
2. Original、Sham、Anomaly 的 fps、宽高、codec 和音频策略一致；
3. 异常开始前 Anomaly 与 Sham 最大像素误差 = 0；
4. Sham mean PSNR >= 30 dB；
5. Sham mean SSIM >= 0.95；
6. Sham minimum per-frame SSIM >= 0.95；
7. 有参考掩膜时，SAM2 non-prompt mean IoU >= 0.75；
8. 异常开始后具有非零、可测的目标变化；
9. 空间异常的最大位移通常应 >= 画面对角线 3%，但不得以越界或明显贴图为代价；
10. declared edit support 外最大像素误差 <= 1；
11. 检查原位置残影、边缘黑边、掩膜泄漏、阴影错位、反射不一致、遮挡错误和时序闪烁；
12. 对最终 MP4 重新解码，生成异常开始帧、中间帧和末帧接触表并进行视觉复核；
13. 检查是否存在未完成的 .part、.aria2、临时下载或损坏输出；
14. 对所有关键输入、权重、配置、脚本和交付视频计算 SHA-256；
15. 输出整批 pass/fail 汇总，不通过的样本不得进入最终 manifest.jsonl。

如果某项失败：先修正提示、掩膜、修复区域、阴影、轨迹或合成，再重新生成和审计。禁止直接降低阈值。仍无法达到标准则保留失败中间文件，在 rejected_candidates.json 中记录具体原因和失败指标。

当批量样本数足够时，额外训练一个只区分 Original 与 Sham 的编辑伪影分类器。如果其准确率明显高于随机水平，说明存在生成器捷径，应修正数据生成流程后重新生成。还应进行跨修复器测试，避免模型只学习 ProPainter 指纹。

====================
十一、metadata.json 最低字段
====================

至少包含：

schema_version、video_id、source_video_id、split_group、source_sha256、label、anomaly_type、severity、start_frame、end_frame、start_seconds、fps、frame_count、width、height、audio_policy、object_id、prompt_records、per_frame_transform、shadow_policy、generation_version、random_seed、SAM2 commit/checkpoint、ProPainter commit/checkpoints、torch/CUDA/GPU、所有中间目录、所有输出文件及 SHA-256、QA 指标、pass/fail、recommended_role。

====================
十二、最终交付回复
====================

最终回复必须：

1. 先说明实际生成数量、通过数量、拒绝数量和总体 QA 状态；
2. 给出每个 comparison.mp4 的可点击绝对路径；
3. 给出 quality_audit.json、manifest.jsonl、reproducibility_manifest.json 和 README.md；
4. 用简短表格报告每个样本的异常类型、开始帧、SAM2 指标或“无参考掩膜”、Sham PSNR/SSIM 和失败原因；
5. 明确说明全部生成文件是否均位于 OUTPUT_ROOT；
6. 报告占用空间和文件数量；
7. 提醒许可、隐私和“合成训练数据不能替代真实异常测试集”的限制。

现在先检查 INPUT_VIDEO_DIR 和 OUTPUT_ROOT，建立执行计划，然后直接开始批量生成。除非目标选择确实存在无法安全推断的歧义，否则不要停下来等待用户确认。
```

## 三、建议的 TARGET_SPEC 示例

如协作者已经知道每段视频要修改哪个目标，最好同时提供下面形式的 JSONL，以减少自动选择歧义：

```json
{"input_video":"E:/real_videos/ball_001.mp4","video_id":"ball_001_hover","object":{"description":"the moving football","prompts":[{"frame_fraction":0.0,"box_xyxy":[620,260,780,420]},{"frame_fraction":0.33,"box_xyxy":[390,250,540,405]},{"frame_fraction":0.67,"box_xyxy":[160,240,300,390]}]},"anomaly":{"type":"midair_hover","onset_fraction":0.42,"severity":0.7},"split_group":"ball_001"}
{"input_video":"E:/real_videos/car_002.mp4","video_id":"car_002_teleport","object":{"description":"the red moving car","prompts":[{"frame_fraction":0.0,"box_xyxy":[570,180,710,280]},{"frame_fraction":0.33,"box_xyxy":[480,80,690,235]},{"frame_fraction":0.67,"box_xyxy":[100,20,850,320]}]},"anomaly":{"type":"instant_teleport","onset_fraction":0.48,"dx_fraction":-0.16,"dy_fraction":-0.04,"severity":0.8},"shadow_mode":"synthetic_ground","split_group":"car_002"}
```

所有坐标必须基于原始视频分辨率，`box_xyxy` 格式为 `[x1, y1, x2, y2]`。如果只提供首帧框，也可让新对话先运行 SAM2，再自动建议后两个固定关键帧修正框并继续。
