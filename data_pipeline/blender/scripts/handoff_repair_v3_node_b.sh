#!/usr/bin/env bash
set -euo pipefail

CAMPAIGNS_ROOT="${CAMPAIGNS_ROOT:-/workspace/pavg/campaigns}"
LEGACY_CAMPAIGN="${LEGACY_CAMPAIGN:-repair_hard_150g_v3_node_b}"
V11_CAMPAIGN="${V11_CAMPAIGN:-repair_hard_45g_v3_node_b_v11}"
SCRATCH="${SCRATCH:-/workspace/pavg/scratch}"
NODE_A_HOST="${NODE_A_HOST:?set NODE_A_HOST to the source training node}"
NODE_A_PORT="${NODE_A_PORT:?set NODE_A_PORT to the source SSH port}"
NODE_A_KEY="${NODE_A_KEY:?set NODE_A_KEY to the SSH private-key path}"
REMOTE_IMPORT_ROOT="${REMOTE_IMPORT_ROOT:-/workspace/pavg/imports/node_b_repair_v3}"
ARCHIVE="$SCRATCH/repair_v3_node_b_handoff.tar.gz"
CHECKSUM="$ARCHIVE.sha256"

while true; do
  status="$(python3 - "$CAMPAIGNS_ROOT/$V11_CAMPAIGN/campaign_state.json" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
print(json.loads(path.read_text(encoding="utf-8")).get("status") if path.is_file() else "missing")
PY
)"
  if [[ "$status" == "generated" ]]; then
    break
  fi
  if [[ "$status" == "failed" ]]; then
    echo "v1.1 campaign failed; refusing handoff" >&2
    exit 2
  fi
  sleep 60
done

python3 - "$CAMPAIGNS_ROOT" "$LEGACY_CAMPAIGN" "$V11_CAMPAIGN" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
legacy = root / sys.argv[2]
v11 = root / sys.argv[3]
snapshots = sorted((legacy / "manifests").glob("shard_*.jsonl"))
if len(snapshots) != 21:
    raise SystemExit(f"expected 21 admitted legacy snapshots, found {len(snapshots)}")
legacy_records = sum(sum(bool(line.strip()) for line in item.read_text(encoding="utf-8").splitlines()) for item in snapshots)
if legacy_records != 3675:
    raise SystemExit(f"expected 3675 legacy records, found {legacy_records}")
state = json.loads((v11 / "campaign_state.json").read_text(encoding="utf-8"))
if state.get("status") != "generated" or state.get("group_count") != 45 or state.get("record_count") != 1575:
    raise SystemExit(f"unexpected v1.1 terminal state: {state}")
PY

mkdir -p "$SCRATCH"
tar -C "$CAMPAIGNS_ROOT" -czf "$ARCHIVE" \
  "$LEGACY_CAMPAIGN/manifests" \
  "$LEGACY_CAMPAIGN/campaign_state.json" \
  "$V11_CAMPAIGN/campaign_manifest.jsonl" \
  "$V11_CAMPAIGN/campaign_state.json" \
  "$V11_CAMPAIGN/monitoring"
(cd "$SCRATCH" && sha256sum "$(basename "$ARCHIVE")" > "$(basename "$CHECKSUM")")

ssh -i "$NODE_A_KEY" -p "$NODE_A_PORT" -o BatchMode=yes "$NODE_A_HOST" \
  "mkdir -p '$REMOTE_IMPORT_ROOT'"
scp -i "$NODE_A_KEY" -P "$NODE_A_PORT" -q "$ARCHIVE" "$CHECKSUM" \
  "$NODE_A_HOST:$REMOTE_IMPORT_ROOT/"
ssh -i "$NODE_A_KEY" -p "$NODE_A_PORT" -o BatchMode=yes "$NODE_A_HOST" \
  "cd '$REMOTE_IMPORT_ROOT' && sha256sum -c '$(basename "$CHECKSUM")' && \
   test ! -e extracted.ok && tar -xzf '$(basename "$ARCHIVE")' && \
   date --iso-8601=seconds > extracted.ok"

echo "Node-B Repair v3 handoff complete: $REMOTE_IMPORT_ROOT"
