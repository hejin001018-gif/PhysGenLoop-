"""Generate a resumable shard of paired Blender videos for Repair Agent training.

The generator deliberately targets the violation families understood by the frozen
PAVG Critic: premature rebound, surface penetration, disappearance, reverse gravity,
and teleportation.  Each group shares geometry, camera, materials and a normal clip;
only the target trajectory/visibility changes between variants.

Run with Blender, for example::

    blender --background --factory-startup --python generate_repair_shard.py -- \
      --output-root /workspace/pavg/scratch/shards/shard_0000 --groups 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


FPS = 24


DIFFICULTY_PROFILES = ("standard", "hard-v1", "hard-v1.1")


VARIANTS = (
    ("normal", "physical", 0),
    ("premature_rebound_mild", "premature_rebound", 1),
    ("premature_rebound_severe", "premature_rebound", 2),
    ("surface_penetration_mild", "surface_penetration", 1),
    ("surface_penetration_severe", "surface_penetration", 2),
    ("object_disappearance_mild", "object_disappearance", 1),
    ("object_disappearance_severe", "object_disappearance", 2),
    ("reverse_gravity_mild", "reverse_gravity", 1),
    ("reverse_gravity_severe", "reverse_gravity", 2),
    ("teleportation_mild", "teleportation", 1),
    ("teleportation_severe", "teleportation", 2),
    ("unknown_occluded", "unknown", 1),
    ("multi_corrupt", "multi_violation", 2),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--shard-id", default="shard_0000")
    parser.add_argument("--start-group", type=int, default=0)
    parser.add_argument("--groups", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument(
        "--difficulty-profile",
        choices=DIFFICULTY_PROFILES,
        default="standard",
        help="Scene/data profile. hard-v1 adds visual ambiguity and contextual policy cases.",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--keep-frames", action="store_true")
    argv = os.sys.argv[os.sys.argv.index("--") + 1 :] if "--" in os.sys.argv else []
    args = parser.parse_args(argv)
    if args.groups < 1 or args.frames < 24:
        parser.error("groups must be positive and frames must be at least 24")
    return args


def clean_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def material(name, color, roughness=0.55, metallic=0.0):
    item = bpy.data.materials.new(name)
    item.diffuse_color = (*color, 1.0)
    item.use_nodes = True
    bsdf = item.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic
    return item


def assign(obj, mat):
    obj.data.materials.append(mat)


def look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def configure_scene(args, rng):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.render.fps = FPS
    scene.frame_start = 1
    scene.frame_end = args.frames
    scene.render.use_file_extension = True
    scene.render.image_settings.color_depth = "8"
    scene.render.resolution_percentage = 100
    scene.view_settings.look = "AgX - Medium High Contrast"
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = args.samples

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (
            rng.uniform(0.025, 0.10),
            rng.uniform(0.04, 0.14),
            rng.uniform(0.08, 0.20),
            1.0,
        )
        background.inputs["Strength"].default_value = rng.uniform(0.25, 0.5)
    return scene


def build_group_scene(args, rng, group_id):
    scene = configure_scene(args, rng)
    floor_mat = material(
        f"Floor_{group_id}",
        (rng.uniform(0.12, 0.35), rng.uniform(0.12, 0.35), rng.uniform(0.12, 0.35)),
        0.82,
    )
    target_mat = material(
        f"Target_{group_id}",
        (rng.uniform(0.55, 0.95), rng.uniform(0.04, 0.35), rng.uniform(0.04, 0.35)),
        0.38,
    )
    accent_mat = material(
        f"Accent_{group_id}",
        (rng.uniform(0.05, 0.25), rng.uniform(0.35, 0.75), rng.uniform(0.4, 0.9)),
        0.58,
    )

    bpy.ops.mesh.primitive_cube_add(location=(0, 0, -0.25), scale=(8.0, 6.0, 0.25))
    floor = bpy.context.object
    floor.name = "Ground"
    assign(floor, floor_mat)

    hard = args.difficulty_profile in {"hard-v1", "hard-v1.1"}
    shape = rng.choice(("sphere", "cube", "cylinder"))
    radius = rng.uniform(0.26, 0.46) if hard else rng.uniform(0.38, 0.58)
    if shape == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=radius)
    elif shape == "cube":
        bpy.ops.mesh.primitive_cube_add(scale=(radius, radius, radius))
    else:
        bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=radius, depth=radius * 2.0)
    target = bpy.context.object
    target.name = f"target_{shape}"
    assign(target, target_mat)

    # Static distractors add appearance diversity without entering the tracked GT.
    distractor_count = rng.randint(8, 13) if hard else rng.randint(3, 7)
    for index in range(distractor_count):
        x = rng.uniform(-5.5, 5.5)
        y = rng.uniform(0.8, 4.5)
        scale = rng.uniform(0.15, 0.55)
        bpy.ops.mesh.primitive_cube_add(
            location=(x, y, scale), scale=(scale, scale, scale)
        )
        distractor = bpy.context.object
        distractor.name = f"distractor_{index:02d}"
        assign(distractor, accent_mat)

    # The hard profile deliberately introduces partial foreground occlusion.  The
    # paired GT observations remain exact, so these scenes stress downstream visual
    # adapters without corrupting the frozen-Critic supervision contract.
    if hard:
        for index in range(2):
            x = rng.choice((-1.0, 1.0)) * rng.uniform(0.75, 2.2)
            y = rng.uniform(-2.2, -0.8)
            half_width = rng.uniform(0.12, 0.28)
            half_height = rng.uniform(0.65, 1.45)
            bpy.ops.mesh.primitive_cube_add(
                location=(x, y, half_height),
                scale=(half_width, rng.uniform(0.18, 0.42), half_height),
            )
            occluder = bpy.context.object
            occluder.name = f"foreground_occluder_{index:02d}"
            assign(occluder, accent_mat)

    bpy.ops.object.light_add(type="AREA", location=(-4.5, -3.0, 7.5))
    key = bpy.context.object
    key.data.energy = rng.uniform(700.0, 1100.0)
    key.data.shape = "DISK"
    key.data.size = 5.0
    look_at(key, (0, 0, 0))
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 6))
    bpy.context.object.rotation_euler = (
        math.radians(rng.uniform(20, 55)),
        math.radians(rng.uniform(-25, 25)),
        math.radians(rng.uniform(-50, 50)),
    )
    bpy.context.object.data.energy = rng.uniform(1.5, 3.0)

    camera_location = (
        (
            rng.uniform(6.8, 9.2),
            rng.uniform(-13.2, -10.8),
            rng.uniform(3.8, 6.4),
        )
        if hard
        else (rng.uniform(6.5, 8.0), -12.0, rng.uniform(4.8, 5.8))
    )
    bpy.ops.object.camera_add(location=camera_location)
    camera = bpy.context.object
    camera.data.lens = rng.uniform(38.0, 58.0) if hard else rng.uniform(45.0, 52.0)
    look_at(camera, (0, 0.5, 1.65))
    scene.camera = camera
    return scene, target, camera, shape, radius


def normal_path(frame_index, frames, radius, start_z, rebound_height):
    """Return one point on a deterministic fall/contact/rebound trajectory.

    ``start_z`` and ``rebound_height`` are sampled once per scene group.  Sampling
    either value inside this per-frame function would turn the clean trajectory into
    jitter and create false rebound/teleport events in the frozen Critic.
    """

    contact = int(frames * 0.60)
    t = frame_index / max(contact, 1)
    x = -2.2 + 4.2 * frame_index / max(frames - 1, 1)
    y = 0.15 * math.sin(frame_index / max(frames - 1, 1) * math.pi)
    if frame_index <= contact:
        z = radius + (start_z - radius) * (1.0 - t * t)
    else:
        u = (frame_index - contact) / max(frames - 1 - contact, 1)
        # A single ballistic arc: contact at u=0 and u=1, one upward/downward
        # velocity reversal at the apex, and no frame-wise random perturbation.
        z = radius + 4.0 * rebound_height * u * (1.0 - u)
    return Vector((x, y, max(radius, z)))


def make_variant(
    variant_name,
    category,
    severity,
    base_positions,
    radius,
    frames,
    difficulty_profile="standard",
):
    positions = [position.copy() for position in base_positions]
    visible = [True] * frames
    onset = int(frames * (0.42 if severity == 2 else 0.52))
    strength = 1.0 if severity == 1 else 1.8

    if category == "premature_rebound":
        anchor = positions[onset].copy()
        for frame in range(onset, frames):
            u = (frame - onset) / max(frames - 1 - onset, 1)
            positions[frame].z = anchor.z + strength * 1.15 * math.sin(math.pi * u)
    elif category == "surface_penetration":
        contact = int(frames * 0.60)
        for frame in range(contact, frames):
            u = (frame - contact) / max(frames - 1 - contact, 1)
            positions[frame].z = radius - strength * (0.22 + 0.55 * u)
    elif category == "object_disappearance":
        length = 6 if severity == 1 else frames - onset
        for frame in range(onset, min(frames, onset + length)):
            visible[frame] = False
    elif category == "reverse_gravity":
        # Start unsupported and move upward from the beginning.  A fall followed by
        # upward motion is deliberately interpreted by the frozen Critic as a
        # rebound; reverse gravity must therefore have no preceding positive
        # (downward image-axis) velocity.  Holding x/y fixed also prevents perspective
        # motion from manufacturing such a transition on oblique cameras.
        anchor = positions[0].copy()
        anchor.z = radius + 0.75
        for frame in range(frames):
            u = frame / max(frames - 1, 1)
            positions[frame].x = anchor.x
            positions[frame].y = anchor.y
            positions[frame].z = anchor.z + strength * 1.8 * u * u
        onset = 0
    elif category == "teleportation":
        jump = strength * 2.2
        for frame in range(onset, frames):
            positions[frame].x += jump
    elif category == "unknown":
        visible = [False] * frames
    elif category == "multi_violation":
        for frame in range(onset, frames):
            # hard-v1.1 keeps the target in frame while combining penetration and
            # disappearance.  The original x jump could push an oblique-camera
            # target out of frame, collapsing the intended multi-error case into a
            # single disappearance violation.
            if difficulty_profile != "hard-v1.1":
                positions[frame].x += 3.2
            positions[frame].z = radius - 0.65
        for frame in range(min(frames - 4, onset + 8), frames):
            visible[frame] = False
    return positions, visible, onset


def set_animation(target, positions, visible):
    target.animation_data_clear()
    for frame_index, (position, is_visible) in enumerate(zip(positions, visible), start=1):
        target.location = position
        target.hide_render = not is_visible
        target.hide_viewport = not is_visible
        target.keyframe_insert("location", frame=frame_index)
        target.keyframe_insert("hide_render", frame=frame_index)
        target.keyframe_insert("hide_viewport", frame=frame_index)
    if target.animation_data and target.animation_data.action:
        try:
            curves = target.animation_data.action.fcurves
        except AttributeError:
            curves = ()
        for curve in curves:
            for point in curve.keyframe_points:
                point.interpolation = "LINEAR" if curve.data_path == "location" else "CONSTANT"


def project_observations(scene, camera, target_name, positions, visible, radius, args):
    records = []
    world_records = []
    for frame_index, (position, is_visible) in enumerate(zip(positions, visible)):
        center_ndc = world_to_camera_view(scene, camera, position)
        top_ndc = world_to_camera_view(scene, camera, position + Vector((0, 0, radius)))
        side_ndc = world_to_camera_view(scene, camera, position + Vector((radius, 0, 0)))
        floor_ndc = world_to_camera_view(scene, camera, Vector((position.x, position.y, 0)))
        center_x = center_ndc.x * args.width
        center_y = (1.0 - center_ndc.y) * args.height
        half_w = max(4.0, abs(side_ndc.x - center_ndc.x) * args.width)
        half_h = max(4.0, abs(top_ndc.y - center_ndc.y) * args.height)
        bbox = [center_x - half_w, center_y - half_h, center_x + half_w, center_y + half_h]
        local_floor_y = (1.0 - floor_ndc.y) * args.height
        distance = local_floor_y - bbox[3]
        overlap = min(1.0, max(0.0, -distance / max(2.0 * half_h, 1.0)))
        in_frame = 0 <= center_x < args.width and 0 <= center_y < args.height and center_ndc.z > 0
        records.append(
            {
                "frame": frame_index,
                "timestamp_sec": frame_index / FPS,
                "object": target_name,
                "track_id": f"{target_name}-1",
                "center": [round(center_x, 5), round(center_y, 5)],
                "bbox": [round(value, 5) for value in bbox],
                "visible": bool(is_visible and in_frame),
                "confidence": 1.0,
                "distance_to_floor": round(distance, 5),
                "overlap_with_floor": round(overlap, 6),
            }
        )
        world_records.append(
            {
                "frame": frame_index,
                "timestamp_sec": frame_index / FPS,
                "position_m": [round(value, 6) for value in position],
                "visible": bool(is_visible),
            }
        )
    for index, record in enumerate(records):
        if index == 0 or not record["visible"] or not records[index - 1]["visible"]:
            record["velocity"] = None
            record["acceleration"] = None
            continue
        dt = 1.0 / FPS
        previous = records[index - 1]["center"]
        velocity = [
            (record["center"][0] - previous[0]) / dt,
            (record["center"][1] - previous[1]) / dt,
        ]
        record["velocity"] = [round(value, 5) for value in velocity]
        prior_velocity = records[index - 1].get("velocity")
        record["acceleration"] = (
            [round((velocity[axis] - prior_velocity[axis]) / dt, 5) for axis in (0, 1)]
            if prior_velocity is not None
            else None
        )
    return records, world_records


def encode_video(frames_dir, video_path, ffmpeg):
    video_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(FPS),
        "-start_number",
        "1",
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video_path),
    ]
    subprocess.run(command, check=True)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render_group(args, group_index):
    group_id = f"group_{group_index:06d}"
    group_dir = args.output_root / "groups" / group_id
    complete_path = group_dir / "group_complete.json"
    if complete_path.is_file():
        print(f"SKIP complete {group_id}")
        return
    started = time.monotonic()
    rng = random.Random(args.seed + group_index * 1009)
    clean_scene()
    scene, target, camera, shape, radius = build_group_scene(args, rng, group_id)
    start_z = rng.uniform(2.8, 3.5)
    rebound_height = rng.uniform(0.65, 1.0)
    base_positions = [
        normal_path(
            frame,
            args.frames,
            radius,
            start_z=start_z,
            rebound_height=rebound_height,
        )
        for frame in range(args.frames)
    ]
    prompt = f"A {shape} falls under gravity, contacts the ground, and rebounds plausibly."

    for variant_name, category, severity in VARIANTS:
        variant_dir = group_dir / variant_name
        video_path = variant_dir / "video.mp4"
        metadata_path = variant_dir / "metadata.json"
        if video_path.is_file() and metadata_path.is_file():
            print(f"SKIP variant {group_id}/{variant_name}")
            continue
        positions, visible, onset = make_variant(
            variant_name,
            category,
            severity,
            base_positions,
            radius,
            args.frames,
            args.difficulty_profile,
        )
        set_animation(target, positions, visible)
        frames_dir = variant_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        scene.render.filepath = str(frames_dir / "frame_")
        scene.frame_start = 1
        scene.frame_end = args.frames
        bpy.ops.render.render(animation=True)
        encode_video(frames_dir, video_path, args.ffmpeg)
        projected_observations, world_states = project_observations(
            scene, camera, target.name, positions, visible, radius, args
        )
        contact_index = min(int(args.frames * 0.60), len(projected_observations) - 1)
        contact_observation = projected_observations[contact_index]
        floor_y = contact_observation["bbox"][3] + contact_observation["distance_to_floor"]
        # An entirely occluded clip gives the Critic no track observations.  Passing
        # synthetic invisible states would instead assert that a previously tracked
        # object disappeared, which is positive violation evidence rather than an
        # epistemic ``unknown`` case.
        observations = [] if category == "unknown" else projected_observations
        write_json(
            metadata_path,
            {
                "schema_version": "1.0",
                "difficulty_profile": args.difficulty_profile,
                "shard_id": args.shard_id,
                "group_id": group_id,
                "variant": variant_name,
                "category": category,
                "severity": severity,
                "seed": args.seed + group_index * 1009,
                "prompt": prompt,
                "object": target.name,
                "shape": shape,
                "radius_m": radius,
                "fps": FPS,
                "frame_count": args.frames,
                "resolution": [args.width, args.height],
                "anomaly_onset": onset,
                "floor_y": floor_y,
                "trajectory_parameters": {
                    "start_z_m": start_z,
                    "rebound_height_m": rebound_height,
                    "contact_frame": contact_index,
                },
                "video": str(video_path.relative_to(args.output_root).as_posix()),
                "observations": observations,
                "world_states": world_states,
                "generator": {
                    "blender_version": bpy.app.version_string,
                    "script": Path(__file__).name,
                },
            },
        )
        if not args.keep_frames:
            shutil.rmtree(frames_dir)
        print(f"DONE variant {group_id}/{variant_name}")

    write_json(
        complete_path,
        {
            "group_id": group_id,
            "variants": [item[0] for item in VARIANTS],
            "elapsed_sec": round(time.monotonic() - started, 3),
        },
    )
    print(f"DONE group {group_id} elapsed={time.monotonic() - started:.2f}s")


def main():
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_root / "shard_config.json",
        {
            "shard_id": args.shard_id,
            "start_group": args.start_group,
            "groups": args.groups,
            "seed": args.seed,
            "frames": args.frames,
            "resolution": [args.width, args.height],
            "samples": args.samples,
            "difficulty_profile": args.difficulty_profile,
            "variants": [item[0] for item in VARIANTS],
        },
    )
    for group_index in range(args.start_group, args.start_group + args.groups):
        render_group(args, group_index)
    write_json(
        args.output_root / "render_complete.json",
        {"shard_id": args.shard_id, "groups": args.groups, "status": "complete"},
    )


if __name__ == "__main__":
    main()
