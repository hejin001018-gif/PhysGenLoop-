# Blender 物理异常视频数据集

本目录包含根据三段真实参考视频重新搭建的三套完整 Blender 3D 场景、逐帧渲染、异常视频和真值数据。所有新生成的数据都位于 `data_pipeline/blender` 内；`references/` 中保存了原视频副本及 SHA-256，以保证来源可追溯。

## 最终视频

| 场景 | 物理异常 | 起始帧（0-based） | 规格 | 成片 |
|---|---|---:|---|---|
| 山路银色越野车 | 重力反转：车辆持续向上加速，路面阴影继续保留 | 32 | 80 帧，3.333 s | `videos/car-turn_anomaly.mp4` |
| 赛道红色漂移车 | 瞬时传送：相邻帧发生不连续位移，并出现短暂空间残影 | 24 | 50 帧，2.083 s | `videos/drift-straight_anomaly.mp4` |
| 庭院蓝白足球 | 空中悬停：平移突然冻结，球仍旋转，落叶与镜头继续运动 | 20 | 48 帧，2.000 s | `videos/soccerball_anomaly.mp4` |

三段成片均为 854×480、24 fps、H.264、`yuv420p`。接触表位于 `previews/*_contact_sheet.png`。

## 目录结构

```text
data_pipeline/blender/
├─ references/          原视频副本
├─ scenes/              可编辑的 .blend 场景
├─ renders/<scene>/     逐帧无损 PNG（frame_0001.png 起）
├─ videos/              最终 H.264 MP4
├─ previews/            关键帧与 2×2 接触表
├─ metadata/            每帧位置、反事实位置和异常状态真值
├─ scripts/             场景生成、接触表和验证脚本
├─ tools/ffmpeg/        本地便携编码器
├─ logs/                构建、渲染与编码日志
├─ manifest.json        输出规格、探测结果与 SHA-256
└─ run_all.ps1          一键复现入口
```

## 场景细节

- `car-turn.blend`：程序化弯道、双侧标线、护栏/界桩、碎石、分层针叶林、岩质山体与积雪材质；车辆包含独立车轮、轮毂、灯组、格栅、玻璃、保险杠、后视镜和行李架。
- `drift-straight.blend`：柏油噪声/凹凸材质、轮胎印、红白轮胎墙、维修区、帐篷、看台结构和观众；漂移车包含赛车涂装、宽体车身、尾翼、扰流部件、灯组和蓝色轮毂。
- `soccerball.blend`：链网围栏、混凝土立柱、果树枝干、分层树叶/灌木和独立草叶；足球使用独立蓝色五边形面片与接缝。悬停后持续旋转，另有贯穿异常时刻的落叶作为“时间仍在继续”的内部参照。

渲染采用 Blender 5.1 Eevee、32 个渲染采样、光线追踪阴影、运动模糊和 AgX 中高对比色彩管理。编码使用 `libx264 -preset slow -crf 16 -pix_fmt yuv420p -movflags +faststart`。

## 复现

在 PowerShell 中执行：

```powershell
cd <repository>/data_pipeline/blender
.\run_all.ps1 -Rebuild -Samples 32
python .\scripts\verify_outputs.py
```

- 不传 `-Rebuild` 时复用现有 `.blend` 文件，但会重新渲染和编码。
- `metadata/*.json` 使用 0-based `frame_index`，同时提供对应的 1-based `blender_frame`。
- `manifest.json` 和每个元数据文件中的 `verification` 记录最终视频解码帧数与校验值。

## 编码说明

本机 Blender 5.1.2 后台模式虽然带有 FFmpeg 库，但无法选择 `FFMPEG` 输出枚举。因此工作流先由 Blender 输出无损 PNG，再使用目录内的 FFmpeg 8.1.2 进行无损帧序列到高质量 H.264 的封装；这不影响场景、物理动画或像素生成均由 Blender 完成。
