#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/workspace/pavg/.venv/bin/python}"
BASE="${BASE:-/workspace/pavg/campaigns/repair_600g_v1}"
DATA="${DATA:-/workspace/pavg/campaigns/repair_300g_v2_data}"
COMBINED="${COMBINED:-/workspace/pavg/campaigns/repair_900g_v2}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/workspace/pavg/artifacts}"
TEAM_DIR="${TEAM_DIR:-/workspace/pavg/team_handoff}"

mkdir -p "$TEAM_DIR" "$ARTIFACT_DIR"
while true; do
  if grep -q '"status": "failed"' "$BASE/campaign_state.json"; then
    echo "Base campaign failed; continuation will not start" >&2
    exit 2
  fi
  if grep -q '"status": "cleaned"' "$BASE/campaign_state.json" \
      && test -f "$ARTIFACT_DIR/repair_agent_$(basename "$BASE").tar.gz"; then
    break
  fi
  sleep 60
done

# Force the detailed v1 report before any continuation data is generated.
cd "$ROOT"
"$PYTHON" Blender_video/scripts/monitor_repair_campaign.py \
  --campaign-root "$BASE" --once
cp "$BASE/monitoring/final_training_summary.md" "$TEAM_DIR/repair_600g_v1_report.md"
cp "$BASE/monitoring/final_training_summary.json" "$TEAM_DIR/repair_600g_v1_report.json"

# Generate a second, independent random campaign.  The cumulative campaign below
# trains on both manifests, so this is continuation with new evidence, not a
# replacement model trained only on the second batch.
"$PYTHON" Blender_video/scripts/run_repair_campaign.py \
  --campaign-root "$DATA" \
  --total-groups 300 \
  --groups-per-shard 10 \
  --start-group 600 \
  --seed 20260717 \
  --frames 48 --width 640 --height 360 --samples 8

"$PYTHON" Blender_video/scripts/prepare_continued_campaign.py \
  --first-campaign "$BASE" \
  --second-campaign "$DATA" \
  --output-campaign "$COMBINED"

"$PYTHON" Blender_video/scripts/train_repair_campaign.py \
  --campaign-root "$COMBINED" \
  --config configs/repair_agent.yaml \
  --critic-config configs/default.yaml \
  --seeds 17,23,42,73,101 \
  --memory-size 768

"$PYTHON" Blender_video/scripts/cleanup_continued_campaign.py \
  --combined-campaign "$COMBINED" \
  --data-campaign "$DATA" \
  --confirm delete-verified-continuation-shards

archive="$ARTIFACT_DIR/repair_agent_$(basename "$COMBINED").tar.gz"
tar -C "$COMBINED" -czf "$archive.tmp" repair_agent
mv -f "$archive.tmp" "$archive"
sha256sum "$archive" > "$archive.sha256"

"$PYTHON" Blender_video/scripts/monitor_repair_campaign.py \
  --campaign-root "$COMBINED" --once
cp "$COMBINED/monitoring/final_training_summary.md" "$TEAM_DIR/repair_900g_v2_report.md"
cp "$COMBINED/monitoring/final_training_summary.json" "$TEAM_DIR/repair_900g_v2_report.json"
echo "Continuation stage complete: $archive"
