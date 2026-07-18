#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BLENDER="${BLENDER:-/workspace/pavg/tools/blender/blender}"
PYTHON="${PYTHON:-/workspace/pavg/.venv/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/pavg/scratch/shards/shard_0000}"
SHARD_ID="${SHARD_ID:-$(basename "$OUTPUT_ROOT")}"
START_GROUP="${START_GROUP:-0}"
GROUP_COUNT="${GROUP_COUNT:-10}"
SEED="${SEED:-20260716}"
FRAMES="${FRAMES:-48}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-360}"
SAMPLES="${SAMPLES:-8}"
DIFFICULTY_PROFILE="${DIFFICULTY_PROFILE:-standard}"

mkdir -p "$OUTPUT_ROOT"

"$BLENDER" --background --factory-startup \
  --python "$ROOT/data_pipeline/blender/scripts/generate_repair_shard.py" -- \
  --output-root "$OUTPUT_ROOT" \
  --shard-id "$SHARD_ID" \
  --start-group "$START_GROUP" \
  --groups "$GROUP_COUNT" \
  --seed "$SEED" \
  --frames "$FRAMES" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --samples "$SAMPLES" \
  --difficulty-profile "$DIFFICULTY_PROFILE"

cd "$ROOT"
"$PYTHON" data_pipeline/blender/scripts/finalize_repair_shard.py \
  --shard-root "$OUTPUT_ROOT" \
  --config configs/default.yaml

"$PYTHON" -m physgenloop.learning_repair.cli validate \
  --manifest "$OUTPUT_ROOT/repair_manifest.jsonl" \
  --check-artifacts \
  --base-dir "$OUTPUT_ROOT"

echo "Cloud shard complete: $OUTPUT_ROOT"
