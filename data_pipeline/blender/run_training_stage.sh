#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/workspace/pavg/.venv/bin/python}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/workspace/pavg/campaigns/repair_600g_v1}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/workspace/pavg/artifacts}"
TOTAL_GROUPS="${TOTAL_GROUPS:-600}"
GROUPS_PER_SHARD="${GROUPS_PER_SHARD:-10}"
START_GROUP="${START_GROUP:-0}"
SEEDS="${SEEDS:-17,23,42,73,101}"

cd "$ROOT"

"$PYTHON" data_pipeline/blender/scripts/run_repair_campaign.py \
  --campaign-root "$CAMPAIGN_ROOT" \
  --total-groups "$TOTAL_GROUPS" \
  --groups-per-shard "$GROUPS_PER_SHARD" \
  --start-group "$START_GROUP" \
  --frames 48 \
  --width 640 \
  --height 360 \
  --samples 8

"$PYTHON" data_pipeline/blender/scripts/train_repair_campaign.py \
  --campaign-root "$CAMPAIGN_ROOT" \
  --config configs/repair_agent.yaml \
  --critic-config configs/default.yaml \
  --seeds "$SEEDS" \
  --memory-size 512

"$PYTHON" data_pipeline/blender/scripts/cleanup_repair_campaign.py \
  --campaign-root "$CAMPAIGN_ROOT" \
  --confirm delete-verified-blender-shards

mkdir -p "$ARTIFACT_DIR"
campaign_name="$(basename "$CAMPAIGN_ROOT")"
archive="$ARTIFACT_DIR/repair_agent_${campaign_name}.tar.gz"
temporary="$archive.tmp"
tar -C "$CAMPAIGN_ROOT" -czf "$temporary" repair_agent
mv -f "$temporary" "$archive"
sha256sum "$archive" > "$archive.sha256"

echo "Training stage complete: $archive"
