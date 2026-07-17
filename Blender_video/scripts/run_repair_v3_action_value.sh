#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/pavg/src}"
CAMPAIGNS="${CAMPAIGNS:-/workspace/pavg/campaigns}"
IMPORT="${IMPORT:-/workspace/pavg/imports/node_b_repair_v3}"
OUT="${OUT:-/workspace/pavg/campaigns/repair_1200g_v3_action_value}"
PY="${PY:-/workspace/pavg/.venv/bin/python}"
cd "$ROOT"

if [[ -e "$OUT" ]]; then
  echo "refusing to overwrite existing v3 output: $OUT" >&2
  exit 2
fi

while true; do
  status="$($PY - "$CAMPAIGNS/repair_hard_95g_v3_node_a_v11/campaign_state.json" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
print(json.loads(path.read_text(encoding="utf-8")).get("status") if path.is_file() else "missing")
PY
)"
  if [[ "$status" == "generated" ]]; then break; fi
  if [[ "$status" == "failed" ]]; then echo "node-a v1.1 campaign failed" >&2; exit 2; fi
  sleep 60
done

mkdir -p "$OUT/sources"
manifest_args=(
  --manifest "$CAMPAIGNS/repair_900g_v2/campaign_manifest.jsonl"
)
for path in "$CAMPAIGNS/repair_hard_150g_v3_node_a/manifests"/shard_*.jsonl; do
  manifest_args+=(--manifest "$path")
done
manifest_args+=(--manifest "$CAMPAIGNS/repair_hard_95g_v3_node_a_v11/campaign_manifest.jsonl")
for path in "$IMPORT/repair_hard_150g_v3_node_b/manifests"/shard_*.jsonl; do
  manifest_args+=(--manifest "$path")
done
manifest_args+=(--manifest "$IMPORT/repair_hard_45g_v3_node_b_v11/campaign_manifest.jsonl")

TARGETS="$OUT/proxy_targets.jsonl"
ADAPTATION="$OUT/proxy_adaptation.json"
SOURCE_INDEX="$OUT/sources/source_index.json"

$PY - "$SOURCE_INDEX" "${manifest_args[@]}" <<'PY'
import hashlib, json, pathlib, sys
index_path = pathlib.Path(sys.argv[1])
raw = sys.argv[2:]
paths = [pathlib.Path(raw[i + 1]).resolve() for i in range(0, len(raw), 2) if raw[i] == "--manifest"]
if len(paths) != 35:
    raise SystemExit(f"expected 35 source manifests, found {len(paths)}")
sample_ids = set()
groups = {}
source_rows = []
for path in paths:
    count = 0
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        sample = str(record["sample_id"])
        group = str(record["group_id"])
        if sample in sample_ids:
            raise SystemExit(f"duplicate sample_id {sample} from {path}:{line_number}")
        sample_ids.add(sample)
        groups[group] = groups.get(group, 0) + 1
        count += 1
    source_rows.append({"path": str(path), "sha256": digest, "record_count": count})
expected = {f"group_{index:06d}": (13 if index < 900 else 35) for index in range(1200)}
if set(groups) != set(expected):
    missing = sorted(set(expected) - set(groups))[:10]
    extra = sorted(set(groups) - set(expected))[:10]
    raise SystemExit(f"group coverage mismatch: missing={missing}, extra={extra}")
bad = {key: value for key, value in groups.items() if value != expected[key]}
if bad:
    raise SystemExit(f"per-group record count mismatch: {list(bad.items())[:10]}")
index = {
    "schema_version": "repair-v3-source-index/1.0",
    "source_count": len(source_rows),
    "source_rows": source_rows,
    "sample_count": len(sample_ids),
    "group_count": len(groups),
    "expected_counts": {"groups": 1200, "samples": 22200, "normal_records_per_group": 13, "hard_records_per_group": 35},
}
index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"groups": len(groups), "samples": len(sample_ids), "sources": len(source_rows)}))
PY

$PY -m physgenloop.learning_repair adapt-proxy-targets \
  "${manifest_args[@]}" \
  --output "$TARGETS" \
  --report "$ADAPTATION" \
  --reward-config configs/learning_repair/reward_v1.yaml
$PY -m physgenloop.learning_repair audit-targets \
  --targets "$TARGETS" --output "$OUT/target_audit.json"
$PY -m physgenloop.learning_repair check-compatibility \
  --manifest configs/learning_repair/critic_compatibility_v1.json \
  --critic-config configs/default.yaml \
  --critic-schema schemas/critic_output.schema.json \
  --feature-schema "$CAMPAIGNS/repair_900g_v2/repair_agent/feature_schema.json" \
  --output "$OUT/compatibility.json"

TRAINING="$OUT/training"
$PY -m physgenloop.learning_repair train-values \
  --targets "$TARGETS" \
  --output-dir "$TRAINING" \
  --compatibility configs/learning_repair/critic_compatibility_v1.json \
  --config configs/learning_repair/action_value_training_v1.yaml

MEMORY="$OUT/proxy_memory_train.jsonl"
$PY - "$TRAINING/train.jsonl" "$MEMORY" <<'PY'
import json, pathlib, sys
source, destination = map(pathlib.Path, sys.argv[1:])
records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
buckets = {}
for item in records:
    meta = item.get("metadata", {})
    key = (
        item["target_action"],
        str(meta.get("category", "unknown")),
        str(meta.get("context_case", "standard")),
    )
    buckets.setdefault(key, []).append(item)
for bucket in buckets.values():
    bucket.sort(key=lambda item: item["sample_id"])
selected = []
cursor = {key: 0 for key in sorted(buckets, key=str)}
while len(selected) < min(1024, len(records)):
    advanced = False
    for key in sorted(buckets, key=str):
        index = cursor[key]
        if index >= len(buckets[key]):
            continue
        selected.append(buckets[key][index])
        cursor[key] += 1
        advanced = True
        if len(selected) >= min(1024, len(records)):
            break
    if not advanced:
        break
destination.write_text(
    "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in selected),
    encoding="utf-8",
)
print(json.dumps({"source": len(records), "selected": len(selected), "path": str(destination)}))
PY

$PY -m physgenloop.learning_repair evaluate \
  --targets "$TRAINING/test.jsonl" \
  --memory-targets "$MEMORY" \
  --checkpoint "$TRAINING/best_action_value_policy.pt" \
  --compatibility configs/learning_repair/critic_compatibility_v1.json \
  --device cpu --output "$OUT/evaluation_test.json"

RELEASE="$OUT/repair_agent"
$PY Blender_video/scripts/build_action_value_release.py \
  --training-dir "$TRAINING" --targets "$TARGETS" --memory-targets "$MEMORY" \
  --compatibility configs/learning_repair/critic_compatibility_v1.json \
  --critic-config configs/default.yaml \
  --feature-schema "$CAMPAIGNS/repair_900g_v2/repair_agent/feature_schema.json" \
  --adaptation-report "$ADAPTATION" --evaluation "$OUT/evaluation_test.json" \
  --output "$RELEASE"

$PY - "$RELEASE" "$TRAINING/test.jsonl" "$OUT/release_smoke.json" <<'PY'
import json, pathlib, subprocess, sys, tempfile
release, test_path, report_path = map(pathlib.Path, sys.argv[1:])
records = [json.loads(line) for line in test_path.read_text(encoding="utf-8").splitlines() if line.strip()]
representatives = {}
for item in records:
    representatives.setdefault(item["target_action"], item)
results = []
with tempfile.TemporaryDirectory(dir=release.parent) as raw:
    work = pathlib.Path(raw)
    for action, item in sorted(representatives.items()):
        report = work / f"{action}.critic.json"
        context = work / f"{action}.context.json"
        output = work / f"{action}.decision.json"
        report.write_text(json.dumps(item["critic_report"], ensure_ascii=False), encoding="utf-8")
        context.write_text(json.dumps(item["context"], ensure_ascii=False), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(release / "inference.py"), "--critic-report", str(report), "--context", str(context), "--device", "cpu", "--output", str(output)],
            cwd=release.parent,
            capture_output=True,
            text=True,
        )
        predicted = None
        if output.is_file():
            predicted = json.loads(output.read_text(encoding="utf-8"))["decision"]["action"]
        results.append({"sample_id": item["sample_id"], "expected": action, "predicted": predicted, "returncode": result.returncode, "matches": predicted == action, "stderr": result.stderr[-500:]})
valid = len(results) == 4 and all(item["returncode"] == 0 and item["matches"] for item in results)
payload = {"valid": valid, "checked": len(results), "results": results}
report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False))
if not valid:
    raise SystemExit(2)
PY

$PY - "$RELEASE" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / "release_manifest.json").read_text(encoding="utf-8"))
for relative, expected in manifest["files"].items():
    path = root / relative
    if not path.is_file(): raise SystemExit(f"missing release file: {relative}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected["sha256"] or path.stat().st_size != expected["bytes"]:
        raise SystemExit(f"release hash/size mismatch: {relative}")
print(json.dumps({"valid": True, "file_count": len(manifest["files"]), "model_id": manifest.get("model_id")}))
PY

ARCHIVE="/workspace/pavg/artifacts/repair_agent_repair_1200g_v3_action_value.tar.gz"
mkdir -p /workspace/pavg/artifacts
tar -C "$RELEASE" -czf "$ARCHIVE" .
(cd /workspace/pavg/artifacts && sha256sum "$(basename "$ARCHIVE")" > "$(basename "$ARCHIVE").sha256")

echo "REPAIR_V3_ACTION_VALUE_COMPLETE release=$RELEASE archive=$ARCHIVE"
