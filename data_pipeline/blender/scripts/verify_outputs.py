"""Verify rendered sequences/videos and write a machine-readable manifest."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "car-turn": {"frames": 80, "onset": 32, "anomaly": "gravity_reversal"},
    "drift-straight": {"frames": 50, "onset": 24, "anomaly": "instant_teleport"},
    "soccerball": {"frames": 48, "onset": 20, "anomaly": "midair_hover"},
}


def find_tool(name):
    matches = sorted((ROOT / "tools" / "ffmpeg").rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find {name} below data_pipeline/blender/tools/ffmpeg")
    return matches[0]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def png_size(path):
    with path.open("rb") as stream:
        header = stream.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG: {path}")
    return struct.unpack(">II", header[16:24])


def main():
    ffmpeg = find_tool("ffmpeg.exe")
    ffprobe = find_tool("ffprobe.exe")
    ffmpeg_version = subprocess.check_output([str(ffmpeg), "-version"], text=True, encoding="utf-8").splitlines()[0]
    manifest = {
        "schema_version": "1.0",
        "verified_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "workspace": ".",
        "blender": {
            "version": "5.1.2",
            "executable": "C:/Program Files/Blender Foundation/Blender 5.1/blender.exe",
            "render_engine": "BLENDER_EEVEE",
        },
        "encoder": {
            "version": ffmpeg_version,
            "relative_path": ffmpeg.relative_to(ROOT).as_posix(),
            "settings": "libx264 preset=slow crf=16 pix_fmt=yuv420p faststart",
        },
        "scenes": {},
    }

    for name, spec in SPECS.items():
        frame_dir = ROOT / "renders" / name
        frames = sorted(frame_dir.glob("frame_*.png"))
        if len(frames) != spec["frames"]:
            raise RuntimeError(f"{name}: expected {spec['frames']} PNGs, got {len(frames)}")
        dimensions = {png_size(path) for path in frames}
        if dimensions != {(854, 480)}:
            raise RuntimeError(f"{name}: unexpected PNG dimensions: {dimensions}")
        sequence_digest = hashlib.sha256()
        for path in frames:
            sequence_digest.update(bytes.fromhex(sha256(path)))

        video = ROOT / "videos" / f"{name}_anomaly.mp4"
        probe = json.loads(
            subprocess.check_output(
                [
                    str(ffprobe), "-v", "error", "-count_frames", "-select_streams", "v:0",
                    "-show_entries", "stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames:format=duration,size,format_name",
                    "-of", "json", str(video),
                ],
                text=True,
                encoding="utf-8",
            )
        )
        stream = probe["streams"][0]
        if (stream["codec_name"], stream["width"], stream["height"], stream["r_frame_rate"], int(stream["nb_read_frames"])) != (
            "h264", 854, 480, "24/1", spec["frames"]
        ):
            raise RuntimeError(f"{name}: video probe mismatch: {stream}")
        subprocess.run(
            [str(ffmpeg), "-v", "error", "-i", str(video), "-f", "null", os.devnull],
            check=True,
        )

        metadata_path = ROOT / "metadata" / f"{name}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        gt = metadata["ground_truth"]
        if len(gt) != spec["frames"] or not gt[spec["onset"]]["active"] or any(row["active"] for row in gt[: spec["onset"]]):
            raise RuntimeError(f"{name}: anomaly timing ground truth failed validation")
        metadata["encoding"] = {
            "codec": "H.264 / libx264",
            "container": "MP4",
            "crf": 16,
            "preset": "slow",
            "pixel_format": "yuv420p",
            "faststart": True,
            "encoder": ffmpeg_version,
        }
        metadata["verification"] = {
            "status": "passed",
            "decoded_frame_count": int(stream["nb_read_frames"]),
            "video_sha256": sha256(video),
            "png_sequence_sha256": sequence_digest.hexdigest(),
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        blend = ROOT / "scenes" / f"{name}.blend"
        reference = ROOT / "references" / f"{name}_original.mp4"
        contact = ROOT / "previews" / f"{name}_contact_sheet.png"
        manifest["scenes"][name] = {
            "anomaly": spec["anomaly"],
            "anomaly_onset_zero_based": spec["onset"],
            "rendered_frames": len(frames),
            "png_dimensions": [854, 480],
            "png_sequence_sha256": sequence_digest.hexdigest(),
            "video_probe": probe,
            "files": {
                "reference": {"path": reference.relative_to(ROOT).as_posix(), "sha256": sha256(reference)},
                "blend": {"path": blend.relative_to(ROOT).as_posix(), "sha256": sha256(blend)},
                "metadata": {"path": metadata_path.relative_to(ROOT).as_posix(), "sha256": sha256(metadata_path)},
                "video": {"path": video.relative_to(ROOT).as_posix(), "sha256": sha256(video)},
                "contact_sheet": {"path": contact.relative_to(ROOT).as_posix(), "sha256": sha256(contact)},
            },
        }

    output = ROOT / "manifest.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"VERIFIED: {output}")


if __name__ == "__main__":
    main()
