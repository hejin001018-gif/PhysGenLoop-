# SAM2 physical-anomaly counterfactual videos

The original local run is self-contained.  This Git snapshot intentionally retains
only reusable scripts, configuration, and audit metadata; source archives, cloned
third-party repositories, checkpoints, the local Python environment, intermediate
frames, and generated videos stay outside Git under the data/weight separation rule.

For applying the same workflow to a new batch of existing real videos, copy the master
prompt in `BATCH_GENERATION_PROMPT.md` into a new conversation and replace its input/output
path variables.

## Deliverables

Final QA status: **PASS**. See `reports/quality_audit.json` and
`reports/QA_SUMMARY.md` for the machine-readable checks and concise results.

The three primary comparison videos are:

- `outputs/01_soccerball_comparison.mp4` - mid-air hover.
- `outputs/02_drift-straight_comparison.mp4` - instantaneous teleportation.
- `outputs/03_car-turn_comparison.mp4` - gravity reversal.

Each comparison has three synchronized panels: original, Sham-edit normal control,
and the physical-anomaly counterfactual. The per-sequence directory also contains
separate clean videos and machine-readable metadata.

All deliverables are silent H.264 video at 24 fps. Individual Original, Sham, and
Anomaly videos are 854x480; comparison videos are 2562x524. Top-level comparison
copies are SHA-256-identical to their per-sequence counterparts.

## Scientific controls

- The three examples use three different DAVIS source sequences and `split_group`
  values. Derived clips from one source must never be split across train/test.
- SAM2 receives a fixed, non-adaptive sparse box schedule at 0%, 33%, and 67% of each
  sequence. DAVIS masks provide those boxes and audit all frames; non-prompt-frame IoU
  is reported separately so prompt frames cannot inflate the tracking gate.
- Sham-edit traverses the same SAM2, ProPainter, compositing, and encoder path without
  anomalous displacement. It is a negative control for editing artifacts.
- Before anomaly onset, Sham-edit and anomaly intermediate frames are pixel-identical.
- Normal, Sham, and anomaly videos use the same H.264 encoder settings.
- Frame-level onset, per-frame displacement, SAM2 mask, repair mask, anomaly mask,
  source identity, and generator versions are retained.
- Pixels outside the declared repair, anomaly-object, and shadow supports are unchanged
  in lossless intermediate frames (maximum measured error 0-1 intensity level).

These clips are suitable as synthetic training/development examples. They are not by
themselves a real-world benchmark test set; final claims still require an independently
collected real-anomaly test set.

## Reproduction

Run from PowerShell:

```powershell
Set-Location <repository>/data_pipeline/sam2
.\run_pipeline.ps1
```

Before running, provision the paths declared in `config.json` (`runtime/`,
`downloads/`, `external/sam2`, `external/ProPainter`, and source data).  These large
runtime dependencies are not included in the repository snapshot.

The pipeline reuses complete existing stages. Reports are written to `reports/`, with
`quality_audit.json` as the gating result and `reproducibility_manifest.json` as the
input/output inventory.

The retained run occupies approximately 13.89 GiB and 62,000 files because the local
CUDA environment, downloaded archives/checkpoints, lossless intermediate PNGs, rejected
candidate audit trail, and final videos are intentionally kept together. Rejected source
decisions are documented in `reports/rejected_candidates.json`.

## Data license

The DAVIS archive states CC BY-NC 4.0 and includes source-specific terms. The original
archive notices are extracted verbatim to `data/provenance/README.md` and
`data/provenance/SOURCES.md`. Verify the non-commercial restriction before redistributing
or using these files beyond research.
