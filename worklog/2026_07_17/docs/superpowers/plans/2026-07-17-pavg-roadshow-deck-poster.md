# PAVG Roadshow Deck and Poster Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a polished 12-minute, 16-slide Chinese PAVG roadshow deck and one high-resolution A3 portrait poster that present the complete physics-aware generation loop, emphasize four innovations, and contain no benchmark results.

**Architecture:** Build both artifacts from one frozen content manifest and one visual-token module. Extract authentic stills from repository videos for image-led slides, use editable PowerPoint text and a small number of native shapes/connectors for system diagrams, and generate the PPTX/poster with `@oai/artifact-tool`. Render every slide and the poster, inspect layout JSON, run overflow tests, and iterate until all visual QA gates pass.

**Tech Stack:** JavaScript ES modules, `@oai/artifact-tool`, Node.js 24, OpenCV frame extraction, PowerPoint/PPTX export, Poppler/LibreOffice-backed rendering helpers, PNG output.

---

## File map

Final deliverables:

- Create: `outputs/PAVG_项目路演.pptx` — 16:9, 16 editable slides.
- Create: `outputs/PAVG_宣传海报.png` — A3 portrait ratio, high-resolution PNG.
- Create: `outputs/pavg-promo-assets/` — selected source frames and final reusable visual assets.

External scratch workspace, retained after delivery:

- Create: `%TEMP%/codex-presentations/<thread-id>/pavg-roadshow/tmp/source-notes.txt` — evidence and provenance notes.
- Create: `%TEMP%/codex-presentations/<thread-id>/pavg-roadshow/tmp/slide-brief.txt` — frozen audience-facing copy.
- Create: `%TEMP%/codex-presentations/<thread-id>/pavg-roadshow/tmp/extract-promo-frames.py` — frame extractor.
- Create: `%TEMP%/codex-presentations/<thread-id>/pavg-roadshow/tmp/build-pavg-roadshow.mjs` — PPTX and poster builder.
- Create: `%TEMP%/codex-presentations/<thread-id>/pavg-roadshow/tmp/preview/` — rendered slide PNGs.
- Create: `%TEMP%/codex-presentations/<thread-id>/pavg-roadshow/tmp/layout/` — slide layout JSON.
- Create: `%TEMP%/codex-presentations/<thread-id>/pavg-roadshow/tmp/qa/` — montage, QA ledger, and test output.

Source references:

- Read: `docs/superpowers/specs/2026-07-17-pavg-roadshow-deck-poster-design.md`.
- Read: `README.md`.
- Read: `docs/superpowers/specs/2026-07-17-full-pavg-critic-evaluation-design.md`.
- Read: `worklog/2026_07_17/learning_repair_agent_路线方案.md` when present locally; otherwise use the already synchronized server worklog notes.
- Read: `1n.mp4`, `2n.mp4`, `1y.mp4`, `29932d12f47258c3b75f98e25e643d2c.mp4`.

## Frozen content manifest

The builder must use this exact slide sequence and must not add performance data:

```js
const SLIDES = [
  { n: 1, title: "PAVG——让生成视频遵守物理世界", role: "brand_open" },
  { n: 2, title: "视频越来越逼真，但仍会违背基本物理规律", role: "problem" },
  { n: 3, title: "生成模型需要的不只是更强采样，而是物理反馈闭环", role: "thesis" },
  { n: 4, title: "PAVG 让视频生成具备自我纠错能力", role: "system_overview" },
  { n: 5, title: "PhysicsPlan 把自然语言转化为可执行物理约束", role: "planner" },
  { n: 6, title: "对象、事件、关系与物理定律组成可执行问题图", role: "question_graph" },
  { n: 7, title: "Physics Critic 用多路证据回答‘哪里错、为什么错’", role: "critic" },
  { n: 8, title: "SAM2 把离散画面连接成连续运动证据", role: "tracking" },
  { n: 9, title: "Rules、PQSG、Checklist、Mechanics 与 VLM 共同裁决", role: "reasoning" },
  { n: 10, title: "覆盖感知融合让系统知道何时确认、何时拒绝、何时保留判断", role: "fusion" },
  { n: 11, title: "Learning Repair Agent 学习‘什么错误该怎么修’", role: "repair_policy" },
  { n: 12, title: "四级修复覆盖从提示词到局部视频编辑", role: "repair_actions" },
  { n: 13, title: "Action-Value Policy 与 Repair Memory 让修复策略持续进化", role: "repair_memory" },
  { n: 14, title: "Blender/Kubric 数据引擎提供可控、精确、可扩展的物理真值", role: "data_engine" },
  { n: 15, title: "PAVG 同时服务评测、生成优化、训练数据与模型研究", role: "applications" },
  { n: 16, title: "让每一帧不仅看起来真实，也在物理上成立", role: "brand_close" },
];

const PACE_SECONDS = [35, 45, 45, 65, 55, 45, 60, 45, 60, 45, 55, 45, 40, 40, 25, 15];
if (PACE_SECONDS.reduce((sum, seconds) => sum + seconds, 0) !== 720) {
  throw new Error("Roadshow pacing must total 720 seconds");
}

const FORBIDDEN_COPY = [
  "Macro-F1", "accuracy", "准确率", "benchmark", "置信区间", "排名",
  "尚未完成", "未来接入", "TODO", "TBD", "px-cloud", "root@",
];
```

### Task 1: Freeze workspace, sources, and copy

**Files:**
- Create: `%TEMP%/codex-presentations/<thread-id>/pavg-roadshow/tmp/source-notes.txt`
- Create: `%TEMP%/codex-presentations/<thread-id>/pavg-roadshow/tmp/slide-brief.txt`

- [ ] **Step 1: Initialize the presentation workspace**

Run in PowerShell:

```powershell
$env:SKILL_DIR = "C:\Users\sy\.codex\plugins\cache\openai-primary-runtime\presentations\26.715.12143\skills\presentations"
$thread = if ($env:CODEX_THREAD_ID) { $env:CODEX_THREAD_ID } else { "manual-pavg-20260717" }
$env:WORKSPACE = Join-Path $env:TEMP "codex-presentations\$thread\pavg-roadshow"
$env:TMP_DIR = Join-Path $env:WORKSPACE "tmp"
$env:ASSET_DIR = Join-Path $env:TMP_DIR "assets"
$env:PREVIEW_DIR = Join-Path $env:TMP_DIR "preview"
$env:LAYOUT_DIR = Join-Path $env:TMP_DIR "layout"
$env:QA_DIR = Join-Path $env:TMP_DIR "qa"
New-Item -ItemType Directory -Force $env:ASSET_DIR,$env:PREVIEW_DIR,$env:LAYOUT_DIR,$env:QA_DIR | Out-Null
node "$env:SKILL_DIR\container_tools\setup_artifact_tool_workspace.mjs" --workspace "$env:TMP_DIR"
```

Expected: `$TMP_DIR/node_modules/@oai/artifact-tool` exists, and no repository source file changes.

- [ ] **Step 2: Record evidence sources and exclusions**

Write `source-notes.txt` with the exact repository commit, the five source documents above, the four video SHA-256 values, and these exclusions:

```text
No benchmark metrics or ranking claims.
No external institutions, partners, users, or deployment-scale claims.
Complete-state product language is required by the approved promotional design.
All visible brand copy uses PAVG; PhysGenLoop is internal-only.
```

Run:

```powershell
Get-FileHash -Algorithm SHA256 1n.mp4,2n.mp4,1y.mp4,29932d12f47258c3b75f98e25e643d2c.mp4
git rev-parse HEAD
```

Expected: four non-empty hashes and one source revision are recorded.

- [ ] **Step 3: Freeze the full slide copy**

Write `slide-brief.txt` with, for every slide, the exact title from `SLIDES`, its target duration from `PACE_SECONDS`, one speaking cue of no more than 100 Chinese characters, one visible supporting sentence of no more than 34 Chinese characters, and at most five visible labels of no more than 12 Chinese characters each. The 16 durations must total exactly 720 seconds. Use these required labels:

```text
Slide 4: PLAN / GENERATE / CRITIC / REPAIR / SELECT
Slide 6: Objects / Actions / Physics
Slide 9: Rules / PQSG / VideoScience / Morpheus / VLM
Slide 12: Prompt Repair / Global Regeneration / Local Editing / Reject
Slide 15: 物理评测 / 生成优化 / 数据生产 / 模型研究
```

- [ ] **Step 4: Validate the copy gate**

Run:

```powershell
$text = Get-Content -Raw -Encoding UTF8 "$env:TMP_DIR\slide-brief.txt"
$forbidden = @("Macro-F1","accuracy","准确率","benchmark","置信区间","排名","尚未完成","未来接入","TODO","TBD","px-cloud","root@")
$hits = $forbidden | Where-Object { $text.Contains($_) }
if ($hits) { throw "Forbidden copy: $($hits -join ', ')" }
```

Expected: exit code 0 and no forbidden copy.

Also validate the pacing total:

```powershell
$pace = @(35,45,45,65,55,45,60,45,60,45,55,45,40,40,25,15)
if (($pace | Measure-Object -Sum).Sum -ne 720) { throw "Roadshow pacing must total 720 seconds" }
```

### Task 2: Extract authentic visual assets from project videos

**Files:**
- Create: `%TEMP%/codex-presentations/<thread-id>/pavg-roadshow/tmp/extract-promo-frames.py`
- Create: `outputs/pavg-promo-assets/*.png`

- [ ] **Step 1: Write the frame extractor**

Create `extract-promo-frames.py` with this complete implementation:

```python
from pathlib import Path
import cv2

ROOT = Path(r"C:\Users\sy\Desktop\PAVG")
OUT = ROOT / "outputs" / "pavg-promo-assets"
VIDEOS = {
    "normal-a": ROOT / "1y.mp4",
    "violation-a": ROOT / "1n.mp4",
    "violation-b": ROOT / "2n.mp4",
    "generated-scene": ROOT / "29932d12f47258c3b75f98e25e643d2c.mp4",
}
FRACTIONS = (0.12, 0.42, 0.72)

OUT.mkdir(parents=True, exist_ok=True)
for stem, video in VIDEOS.items():
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 3:
        raise RuntimeError(f"Too few frames in {video}: {total}")
    for index, fraction in enumerate(FRACTIONS, start=1):
        frame_index = min(total - 1, round((total - 1) * fraction))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot read {video} frame {frame_index}")
        destination = OUT / f"{stem}-{index:02d}.png"
        if not cv2.imwrite(str(destination), frame):
            raise RuntimeError(f"Cannot write {destination}")
    capture.release()
```

- [ ] **Step 2: Run extraction**

Run:

```powershell
.\.venv\Scripts\python.exe "$env:TMP_DIR\extract-promo-frames.py"
```

Expected: 12 PNG files under `outputs/pavg-promo-assets/`.

- [ ] **Step 3: Validate dimensions and montage**

Run:

```powershell
.\.venv\Scripts\python.exe -c "from pathlib import Path; import cv2; p=Path('outputs/pavg-promo-assets'); files=sorted(p.glob('*.png')); assert len(files)==12; dims=[(f.name,cv2.imread(str(f)).shape[:2]) for f in files]; assert all(h>0 and w>0 for _,(h,w) in dims); print(dims)"
```

Expected: 12 readable raster images, each with non-zero width and height.

- [ ] **Step 4: Select asset roles**

Visually inspect all 12 frames and record the selected filenames in `source-notes.txt`:

```text
problem_triptych = three frames showing distinct physical situations
sam2_strip = three chronological frames from violation-b
data_engine_scene = one clean synthetic-looking scene
cover_reference = one frame with the clearest negative space
```

### Task 3: Implement the shared visual system and deck scaffolding

**Files:**
- Create: `%TEMP%/codex-presentations/<thread-id>/pavg-roadshow/tmp/build-pavg-roadshow.mjs`

- [ ] **Step 1: Create the artifact-tool builder shell**

The module must import only Node built-ins and `@oai/artifact-tool`:

```js
import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const ROOT = "C:/Users/sy/Desktop/PAVG";
const TMP = process.env.TMP_DIR;
const OUT = path.join(ROOT, "outputs");
const ASSETS = path.join(OUT, "pavg-promo-assets");
const PREVIEW = path.join(TMP, "preview");
const LAYOUT = path.join(TMP, "layout");
const QA = path.join(TMP, "qa");

const C = {
  ivory: "#F6F3EC", blue: "#244BDB", coral: "#FF684A",
  charcoal: "#111827", paleBlue: "#DFE8FF", white: "#FFFFFF",
  gray: "#667085", line: "#CBD5E1",
};
const FONT_CN = "Microsoft YaHei";
const FONT_EN = "Aptos";
const SLIDE = { width: 1280, height: 720 };
const FRAME = { left: 76, top: 58, width: 1128, height: 604 };

async function writeBlob(file, blob) {
  await fs.writeFile(file, new Uint8Array(await blob.arrayBuffer()));
}
async function imageBytes(file) {
  const bytes = await fs.readFile(file);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}
function addText(slide, text, position, style = {}) {
  const box = slide.shapes.add({
    geometry: "textbox", position, fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  box.text = text;
  box.text.style = { typeface: FONT_CN, fontSize: 22, color: C.charcoal, ...style };
  return box;
}
function addTitle(slide, text, n) {
  addText(slide, `PAVG / ${String(n).padStart(2, "0")}`, { left: 76, top: 38, width: 230, height: 24 },
    { typeface: FONT_EN, fontSize: 14, bold: true, color: C.blue });
  return addText(slide, text, { left: 76, top: 76, width: 1040, height: 90 },
    { fontSize: 38, bold: true, color: C.charcoal });
}
function addFooter(slide) {
  addText(slide, "Physics-Aware Agentic Video Generation", { left: 76, top: 676, width: 390, height: 18 },
    { typeface: FONT_EN, fontSize: 10, color: C.gray });
}
function addAnchor(slide, x, y) {
  return slide.shapes.add({
    geometry: "ellipse", position: { left: x, top: y, width: 2, height: 2 },
    fill: "none", line: { style: "solid", fill: "none", width: 0 },
  });
}
function connectAnchors(slide, from, to, color = C.coral) {
  return slide.shapes.connect(from, to, {
    kind: "straight", fromSide: "right", toSide: "left",
    line: { style: "solid", fill: color, width: 2 },
    head: { type: "arrow", width: "med", length: "med" },
  });
}
```

- [ ] **Step 2: Add content and safety assertions**

Add the exact `SLIDES` and `FORBIDDEN_COPY` constants from this plan plus:

```js
function assertDeckCopy(slides) {
  const all = JSON.stringify(slides);
  for (const token of FORBIDDEN_COPY) {
    if (all.includes(token)) throw new Error(`Forbidden deck copy: ${token}`);
  }
  if (slides.length !== 16) throw new Error(`Expected 16 slides, got ${slides.length}`);
}
```

- [ ] **Step 3: Create the presentation and fixed chrome**

Create a 1280×720 presentation. Use `C.ivory` or `C.white` backgrounds, one headline, one visual composition, and a footer per slide. Do not create dashboard grids, decorative buttons, or more than six content boxes on any slide.

- [ ] **Step 4: Smoke-render the cover**

Run the builder with a temporary `ONLY_SLIDE=1` environment switch.

Expected: `preview/slide-01.png` exists; title is at least 50 px and does not wrap unexpectedly.

### Task 4: Author slides 1–8

**Files:**
- Modify: `%TEMP%/codex-presentations/<thread-id>/pavg-roadshow/tmp/build-pavg-roadshow.mjs`

- [ ] **Step 1: Build slides 1–4**

Use these layouts:

```text
1 cover: left title; right pale-blue trajectory field; coral object; no body paragraph.
2 problem: three authentic video frames across the canvas; labels 重力 / 碰撞 / 恒存; one bottom takeaway.
3 thesis: large “更强采样 ≠ 更懂物理”; one Royal Blue feedback-loop line.
4 overview: five horizontally aligned nodes PLAN / GENERATE / CRITIC / REPAIR / SELECT; label SELECT as bounded Best-of-K and close the Agentic Feedback Loop with a connector behind nodes.
```

Connectors on slide 4 must be created before node shapes so arrows never cross node text.

- [ ] **Step 2: Build slides 5–8**

Use these layouts:

```text
5 planner: prompt on left; structured objects/events/relations/constraints on right; one conversion arrow.
6 graph: three node families Objects / Actions / Physics with forward-only connectors.
7 critic: central candidate-video strip; five evidence sources converge into one CriticReport.
8 tracking: three chronological frames from violation-b; Royal Blue track line; Coral critical-frame bracket.
```

- [ ] **Step 3: Render slides 1–8**

Run:

```powershell
$env:SLIDE_RANGE = "1-8"
node "$env:TMP_DIR\build-pavg-roadshow.mjs"
```

Expected: eight PNG previews and eight layout JSON files.

- [ ] **Step 4: Inspect slides 1–8 at full size**

Check each PNG separately for title wrapping, frame crops, connector order, visual balance, and minimum text size. Record every issue and fix in `qa/qa-ledger.txt` before continuing.

### Task 5: Author slides 9–16

**Files:**
- Modify: `%TEMP%/codex-presentations/<thread-id>/pavg-roadshow/tmp/build-pavg-roadshow.mjs`

- [ ] **Step 1: Build slides 9–12**

Use these layouts:

```text
9 reasoning: one vertical evidence stack ending in a single decision; no five-card dashboard.
10 fusion: evidence coverage on the left; physical / violation / unknown decision field on the right.
11 repair policy: CriticReport input → Action-Value Policy → selected action; Repair Memory below.
12 actions: one continuous severity spectrum from Prompt Repair to Reject, with Local Editing centered.
```

- [ ] **Step 2: Build slides 13–16**

Use these layouts:

```text
13 memory: circular experience loop; action, outcome, utility, memory update.
14 data engine: one large synthetic-scene frame plus trajectory / contact / mask overlays.
15 applications: four horizontal lanes — 物理评测 / 生成优化 / 数据生产 / 模型研究.
16 close: minimal brand close with trajectory motif and the line “让每一帧不仅看起来真实，也在物理上成立”.
```

- [ ] **Step 3: Render slides 9–16**

Run:

```powershell
$env:SLIDE_RANGE = "9-16"
node "$env:TMP_DIR\build-pavg-roadshow.mjs"
```

Expected: eight new PNG previews and layout JSON files.

- [ ] **Step 4: Inspect slides 9–16 at full size**

Check every PNG separately. Fix any repeated silhouette, dense copy, weak contrast, unintended overlap, or title wrap. Append fixes to `qa/qa-ledger.txt`.

### Task 6: Build the A3 poster and export final files

**Files:**
- Modify: `%TEMP%/codex-presentations/<thread-id>/pavg-roadshow/tmp/build-pavg-roadshow.mjs`
- Create: `outputs/PAVG_项目路演.pptx`
- Create: `outputs/PAVG_宣传海报.png`

- [ ] **Step 1: Implement the A3 poster canvas**

Create a separate `Presentation` with `slideSize: { width: 1123, height: 1587 }`. Use this exact hierarchy:

```text
Top 22%: PAVG + 让生成视频遵守物理世界 + trajectory hero.
Middle 27%: PLAN → GENERATE → CRITIC → REPAIR → SELECT, with Best-of-K and bounded feedback visibly encoded.
Lower 38%: PhysicsPlan / Multi-Evidence Critic / Learning Repair / Auditable Loop in a 2×2 editorial composition.
Bottom 7%: Physics-Aware Agentic Video Generation.
```

Use at least 64 px for the poster headline, at least 25 px for innovation headers, and at least 17 px for supporting text. Do not add a QR code or organization logo.

- [ ] **Step 2: Export the poster at high resolution**

```js
const posterPng = await poster.export({ slide: poster.slides.items[0], format: "png", scale: 2 });
await writeBlob(path.join(OUT, "PAVG_宣传海报.png"), posterPng);
```

Expected: PNG dimensions are approximately 2246×3174 and preserve the A3 ratio.

- [ ] **Step 3: Export the final PPTX**

```js
const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(path.join(OUT, "PAVG_项目路演.pptx"));
```

Expected: the PPTX exists, contains exactly 16 slides, and remains editable.

- [ ] **Step 4: Export final previews and montage**

Export every slide to PNG, every slide layout to JSON, and a deck montage to `qa/deck-montage.webp`.

### Task 7: Full QA and delivery gate

**Files:**
- Verify: `outputs/PAVG_项目路演.pptx`
- Verify: `outputs/PAVG_宣传海报.png`
- Create: `%TEMP%/codex-presentations/<thread-id>/pavg-roadshow/tmp/qa/final-qa.txt`

- [ ] **Step 1: Run structural slide tests**

Run:

```powershell
python "$env:SKILL_DIR\container_tools\slides_test.py" "outputs\PAVG_项目路演.pptx"
```

Expected: no slide elements overflow the slide canvas.

- [ ] **Step 2: Render the exported PPTX independently**

Run:

```powershell
python "$env:SKILL_DIR\container_tools\render_slides.py" "outputs\PAVG_项目路演.pptx"
python "$env:SKILL_DIR\container_tools\create_montage.py" --input_dir "outputs\PAVG_项目路演" --output_file "$env:QA_DIR\exported-deck-montage.png"
```

Expected: 16 slide PNGs and one montage are created from the actual PPTX, not only from the in-memory presentation.

- [ ] **Step 3: Inspect every rendered slide**

Open all 16 rendered PNGs individually at full size. Check:

```text
No unintended overlap or clipping.
No one-line title wraps to two lines unexpectedly.
No body text below 16 pt equivalent.
No distorted or repeated image crop.
No connector crosses a label.
No benchmark number or forbidden status wording.
Slide-to-slide silhouettes vary while typography and palette stay consistent.
The speaking cues and pacing remain deliverable in 12 minutes.
```

- [ ] **Step 4: Inspect poster at original resolution**

Confirm A3 ratio, readable headline, balanced lower innovation section, no watermark, no garbled Chinese, and no forbidden copy.

- [ ] **Step 5: Run final content scan**

Use artifact-tool inspection or PPTX text extraction to scan all visible text against `FORBIDDEN_COPY`. Scan the poster source text from the builder constants with the same list.

Expected: zero forbidden matches.

- [ ] **Step 6: Record final QA**

Write `final-qa.txt` with:

```text
PPTX slide count: 16
Roadshow pacing: 720 seconds
PPTX overflow errors: 0
Full-size slide inspection: 16/16
Poster A3 ratio: pass
Forbidden-copy matches: 0
Unexpected title wraps: 0
Unintended overlaps: 0
```

- [ ] **Step 7: Verify repository scope**

Run:

```powershell
git status --short
```

Expected: only final files under ignored `outputs/` plus pre-existing user-owned changes; no source, test, config, README, or schema file is modified by presentation production.
