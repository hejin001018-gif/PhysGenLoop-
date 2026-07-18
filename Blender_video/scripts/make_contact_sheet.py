"""Build a lossless 2x2 contact sheet using Blender image I/O and NumPy."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import bpy
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FRAMES = {
    "car-turn": [1, 32, 48, 80],
    "drift-straight": [1, 24, 25, 50],
    "soccerball": [1, 20, 21, 48],
}


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, choices=sorted(FRAMES))
    argv = os.sys.argv[os.sys.argv.index("--") + 1 :] if "--" in os.sys.argv else []
    return parser.parse_args(argv)


def main():
    name = args().scene
    frame_dir = ROOT / "renders" / name
    source_paths = [frame_dir / f"frame_{frame:04d}.png" for frame in FRAMES[name]]
    for path in source_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    arrays = []
    width = height = None
    for path in source_paths:
        image = bpy.data.images.load(str(path), check_existing=False)
        width, height = image.size
        pixels = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        arrays.append(pixels.reshape((height, width, 4)))
    canvas_pixels = np.zeros((height * 2, width * 2, 4), dtype=np.float32)
    canvas_pixels[..., 3] = 1.0
    canvas_pixels[height:, :width] = arrays[0]
    canvas_pixels[height:, width:] = arrays[1]
    canvas_pixels[:height, :width] = arrays[2]
    canvas_pixels[:height, width:] = arrays[3]
    canvas = bpy.data.images.new(name + " contact sheet", width * 2, height * 2, alpha=False)
    canvas.pixels.foreach_set(canvas_pixels.ravel())
    canvas.file_format = "PNG"
    canvas.filepath_raw = str(ROOT / "previews" / f"{name}_contact_sheet.png")
    canvas.save()
    print(f"CONTACT_SHEET {name}: {canvas.filepath_raw}")


if __name__ == "__main__":
    main()
