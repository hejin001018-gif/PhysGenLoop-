from __future__ import annotations

import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from common import BASE, load_config, numbered_files, sha256_file, write_json


def extract_member(archive: zipfile.ZipFile, member: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member) as source, output.open("wb") as target:
        shutil.copyfileobj(source, target)


def main() -> None:
    config = load_config()
    archive_path = BASE / config["dataset"]["archive"]
    if not archive_path.is_file():
        raise FileNotFoundError(f"DAVIS archive is missing: {archive_path}")

    manifest: dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": config["dataset"],
        "archive_sha256": sha256_file(archive_path),
        "sequences": [],
    }

    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        provenance_dir = BASE / "data" / "provenance"
        for member in ("DAVIS/README.md", "DAVIS/SOURCES.md"):
            if member in names:
                extract_member(archive, member, provenance_dir / Path(member).name)
        for sequence in config["sequences"]:
            sequence_id = sequence["id"]
            source_root = BASE / "data" / "sources" / sequence_id
            frames_dir = source_root / "frames"
            gt_dir = source_root / "davis_gt_masks"
            frame_prefix = f"DAVIS/JPEGImages/480p/{sequence_id}/"
            mask_prefix = f"DAVIS/Annotations/480p/{sequence_id}/"
            frame_members = sorted(
                name for name in names if name.startswith(frame_prefix) and name.endswith(".jpg")
            )
            mask_members = sorted(
                name for name in names if name.startswith(mask_prefix) and name.endswith(".png")
            )
            if not frame_members or len(frame_members) != len(mask_members):
                raise RuntimeError(
                    f"Sequence {sequence_id!r} not found or frame/mask counts differ: "
                    f"{len(frame_members)} frames, {len(mask_members)} masks"
                )
            for member in frame_members:
                extract_member(archive, member, frames_dir / Path(member).name)
            for member in mask_members:
                extract_member(archive, member, gt_dir / Path(member).name)

            frames = numbered_files(frames_dir, (".jpg", ".jpeg"))
            masks = numbered_files(gt_dir, (".png",))
            manifest["sequences"].append(
                {
                    **sequence,
                    "frame_count": len(frames),
                    "first_frame_sha256": sha256_file(frames[0]),
                    "last_frame_sha256": sha256_file(frames[-1]),
                    "frame_directory": str(frames_dir.relative_to(BASE)),
                    "ground_truth_mask_directory": str(gt_dir.relative_to(BASE)),
                }
            )

    write_json(BASE / "data" / "source_manifest.json", manifest)
    print(f"Prepared {len(manifest['sequences'])} source sequences")


if __name__ == "__main__":
    main()
