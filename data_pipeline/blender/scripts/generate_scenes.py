"""Build three detailed, reproducible Blender anomaly scenes.

Run with Blender, for example:
  blender --background --factory-startup --python generate_scenes.py -- --scene car-turn

All generated paths are rooted in data_pipeline/blender.  Source footage is used only
as a visual reference; every rendered pixel in the anomaly videos is 3D.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path(__file__).resolve().parents[1]
FPS = 24
WIDTH = 854
HEIGHT = 480
RNG = random.Random(240715)

SPECS = {
    "car-turn": {
        "frames": 80,
        "onset": 32,
        "anomaly": "gravity_reversal",
        "title": "Mountain road / gravity reversal",
        "reference": "references/car-turn_original.mp4",
    },
    "drift-straight": {
        "frames": 50,
        "onset": 24,
        "anomaly": "instant_teleport",
        "title": "Drift circuit / discontinuous teleportation",
        "reference": "references/drift-straight_original.mp4",
    },
    "soccerball": {
        "frames": 48,
        "onset": 20,
        "anomaly": "midair_hover",
        "title": "Backyard football / mid-air hover",
        "reference": "references/soccerball_original.mp4",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, choices=sorted(SPECS))
    parser.add_argument("--samples", type=int, default=32)
    argv = []
    if "--" in os.sys.argv:
        argv = os.sys.argv[os.sys.argv.index("--") + 1 :]
    return parser.parse_args(argv)


def clean_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for block in list(datablocks):
            if block.users == 0:
                datablocks.remove(block)


def set_input(node, name, value):
    socket = node.inputs.get(name)
    if socket is not None:
        socket.default_value = value


def material(name, color, roughness=0.5, metallic=0.0, emission=None, alpha=1.0):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.diffuse_color = (*color[:3], alpha)
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    set_input(bsdf, "Base Color", (*color[:3], 1.0))
    set_input(bsdf, "Roughness", roughness)
    set_input(bsdf, "Metallic", metallic)
    set_input(bsdf, "Alpha", alpha)
    if emission is not None:
        set_input(bsdf, "Emission Color", (*emission[:3], 1.0))
        set_input(bsdf, "Emission Strength", emission[3] if len(emission) > 3 else 2.0)
    if alpha < 1.0 and hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "DITHERED"
    return mat


def noise_material(name, color_a, color_b, scale, roughness=0.8, bump=0.15, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    tex = nodes.new("ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value = scale
    tex.inputs["Detail"].default_value = 5.0
    tex.inputs["Roughness"].default_value = 0.65
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*color_a, 1.0)
    ramp.color_ramp.elements[1].color = (*color_b, 1.0)
    bump_node = nodes.new("ShaderNodeBump")
    bump_node.inputs["Strength"].default_value = bump
    bump_node.inputs["Distance"].default_value = 0.12
    links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(tex.outputs["Fac"], bump_node.inputs["Height"])
    links.new(bump_node.outputs["Normal"], bsdf.inputs["Normal"])
    set_input(bsdf, "Roughness", roughness)
    set_input(bsdf, "Metallic", metallic)
    return mat


def assign(obj, mat):
    if obj.data and hasattr(obj.data, "materials"):
        obj.data.materials.append(mat)
    return obj


def smooth(obj):
    if obj.type == "MESH":
        for p in obj.data.polygons:
            p.use_smooth = True
    return obj


def bevel(obj, width=0.08, segments=3):
    mod = obj.modifiers.new("Edge softness", "BEVEL")
    mod.width = width
    mod.segments = segments
    return obj


def box(name, location, scale, mat, rotation=(0, 0, 0), bevel_width=0.0, parent=None):
    bpy.ops.mesh.primitive_cube_add(location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    assign(obj, mat)
    if bevel_width:
        bevel(obj, bevel_width)
    if parent:
        obj.parent = parent
    return obj


def cylinder(name, location, radius, depth, mat, vertices=24, rotation=(0, 0, 0), parent=None):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth, location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    assign(obj, mat)
    if parent:
        obj.parent = parent
    return obj


def sphere(name, location, scale, mat, segments=24, rings=12, parent=None):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=rings, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    assign(obj, mat)
    smooth(obj)
    if parent:
        obj.parent = parent
    return obj


def torus(name, location, major_radius, minor_radius, mat, rotation=(0, 0, 0), parent=None):
    bpy.ops.mesh.primitive_torus_add(
        major_radius=major_radius,
        minor_radius=minor_radius,
        major_segments=32,
        minor_segments=10,
        location=location,
        rotation=rotation,
    )
    obj = bpy.context.object
    obj.name = name
    assign(obj, mat)
    smooth(obj)
    if parent:
        obj.parent = parent
    return obj


def curve_object(name, points, mat, bevel_depth=0.03, cyclic=False, resolution=2, parent=None):
    data = bpy.data.curves.new(name + "_Curve", "CURVE")
    data.dimensions = "3D"
    data.resolution_u = resolution
    data.bevel_depth = bevel_depth
    data.bevel_resolution = 2
    spline = data.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for p, co in zip(spline.points, points):
        p.co = (*co, 1.0)
    spline.use_cyclic_u = cyclic
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    assign(obj, mat)
    if parent:
        obj.parent = parent
    return obj


def empty(name, location=(0, 0, 0), parent=None):
    obj = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    if parent:
        obj.parent = parent
    return obj


def look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def create_camera(location, target, lens=50, focus=None, fstop=5.6):
    data = bpy.data.cameras.new("Camera")
    data.lens = lens
    data.sensor_width = 36
    cam = bpy.data.objects.new("Camera", data)
    bpy.context.collection.objects.link(cam)
    cam.location = location
    look_at(cam, target)
    if focus is not None:
        data.dof.use_dof = True
        data.dof.focus_object = focus
        data.dof.aperture_fstop = fstop
    bpy.context.scene.camera = cam
    return cam


def add_sun(rotation=(math.radians(32), math.radians(-18), math.radians(-32)), energy=3.0):
    data = bpy.data.lights.new("Sun", "SUN")
    data.energy = energy
    data.angle = math.radians(4.0)
    obj = bpy.data.objects.new("Sun", data)
    bpy.context.collection.objects.link(obj)
    obj.rotation_euler = rotation
    return obj


def add_area(name, location, energy, size, color=(1, 1, 1), target=(0, 0, 0)):
    data = bpy.data.lights.new(name, "AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    data.color = color
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    look_at(obj, target)
    return obj


def configure_scene(scene_name, samples, frames):
    scene = bpy.context.scene
    scene.name = scene_name
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = WIDTH
    scene.render.resolution_y = HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.fps = FPS
    scene.render.fps_base = 1.0
    scene.render.use_file_extension = True
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 0.28
    scene.frame_start = 1
    scene.frame_end = frames
    scene.render.film_transparent = False
    scene.render.filepath = str(ROOT / "renders" / scene_name / "frame_")
    scene.eevee.taa_render_samples = samples
    scene.eevee.use_fast_gi = True
    scene.eevee.fast_gi_quality = 1.0
    scene.eevee.fast_gi_ray_count = 4
    scene.eevee.use_raytracing = True
    scene.eevee.shadow_ray_count = 2
    scene.eevee.shadow_step_count = 8
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.look = "AgX - Medium High Contrast"
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputWorld")
    bg = nodes.new("ShaderNodeBackground")
    sky = nodes.new("ShaderNodeTexSky")
    # Blender 5.1 renamed the physically based Nishita modes.
    sky.sky_type = "MULTIPLE_SCATTERING"
    sky.sun_elevation = math.radians(35)
    sky.sun_rotation = math.radians(125)
    sky.altitude = 0.8
    bg.inputs["Strength"].default_value = 0.32
    links.new(sky.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], out.inputs["Surface"])
    return scene


def create_road(name, centers, width, road_mat, shoulder_mat=None):
    verts = []
    faces = []
    n = len(centers)
    for i, p in enumerate(centers):
        prev = Vector(centers[max(0, i - 1)])
        nxt = Vector(centers[min(n - 1, i + 1)])
        tangent = (nxt - prev).normalized()
        side = Vector((-tangent.y, tangent.x, 0)).normalized()
        c = Vector(p)
        verts.extend([tuple(c - side * width * 0.5), tuple(c + side * width * 0.5)])
    for i in range(n - 1):
        a = i * 2
        faces.append((a, a + 1, a + 3, a + 2))
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    assign(obj, road_mat)
    return obj


def create_mountain(name, location, radius, height, rock_mat, snow_mat=None, seed=1):
    rng = random.Random(seed)
    rings = 7
    segments = 28
    verts = []
    for r in range(rings):
        rr = radius * (r / (rings - 1))
        base_z = height * (1.0 - (r / (rings - 1)) ** 0.78)
        for s in range(segments):
            ang = 2 * math.pi * s / segments
            jag = 1.0 + rng.uniform(-0.11, 0.11) * (r / (rings - 1))
            z = base_z + rng.uniform(-0.06, 0.06) * height * (r / (rings - 1))
            verts.append((rr * jag * math.cos(ang), rr * jag * math.sin(ang), z))
    faces = []
    for r in range(rings - 1):
        for s in range(segments):
            a = r * segments + s
            b = r * segments + (s + 1) % segments
            c = (r + 1) * segments + (s + 1) % segments
            d = (r + 1) * segments + s
            faces.append((a, b, c, d))
    mesh = bpy.data.meshes.new(name + "Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    obj.data.materials.append(rock_mat)
    if snow_mat:
        obj.data.materials.append(snow_mat)
        for poly in obj.data.polygons:
            z_avg = sum(mesh.vertices[i].co.z for i in poly.vertices) / len(poly.vertices)
            poly.material_index = 1 if z_avg > height * 0.72 else 0
    return obj


def create_pine(name, location, height, trunk_mat, needle_mat, parent=None):
    root = empty(name, location, parent)
    cylinder(name + "_trunk", (0, 0, height * 0.30), height * 0.055, height * 0.60, trunk_mat, vertices=8, parent=root)
    for i, zf in enumerate((0.42, 0.60, 0.76)):
        bpy.ops.mesh.primitive_cone_add(
            vertices=12,
            radius1=height * (0.23 - i * 0.045),
            radius2=0.02,
            depth=height * 0.45,
            location=(0, 0, height * zf),
        )
        crown = bpy.context.object
        crown.name = f"{name}_crown_{i}"
        assign(crown, needle_mat)
        crown.parent = root
    return root


def create_wheel(name, location, radius, width, parent, tire_mat, rim_mat, accent_mat=None):
    hub = empty(name, location, parent)
    torus(name + "_tire", (0, 0, 0), radius * 0.74, radius * 0.26, tire_mat, rotation=(math.pi / 2, 0, 0), parent=hub)
    cylinder(name + "_rim", (0, 0, 0), radius * 0.55, width * 1.04, rim_mat, vertices=24, rotation=(math.pi / 2, 0, 0), parent=hub)
    cylinder(name + "_hubcap", (0, -width * 0.53, 0), radius * 0.15, 0.035, accent_mat or rim_mat, vertices=16, rotation=(math.pi / 2, 0, 0), parent=hub)
    spoke_mat = accent_mat or tire_mat
    for a in range(0, 360, 60):
        rad = math.radians(a)
        box(
            name + f"_spoke_{a}",
            (math.cos(rad) * radius * 0.22, -width * 0.55, math.sin(rad) * radius * 0.22),
            (radius * 0.28, 0.018, 0.035),
            spoke_mat,
            rotation=(0, rad, 0),
            bevel_width=0.01,
            parent=hub,
        )
    return hub


def create_vehicle(name, style="suv"):
    root = empty(name)
    rubber = material(name + " Rubber", (0.015, 0.018, 0.02), 0.32)
    dark = material(name + " Trim", (0.025, 0.035, 0.045), 0.26, 0.05)
    glass = material(name + " Glass", (0.035, 0.12, 0.19), 0.12, 0.15)
    chrome = material(name + " Chrome", (0.42, 0.46, 0.50), 0.16, 0.92)
    white = material(name + " Lamps", (0.92, 0.97, 1.0), 0.18, emission=(0.75, 0.9, 1.0, 0.55))
    redlamp = material(name + " Rear lamps", (0.55, 0.012, 0.008), 0.2, emission=(1.0, 0.01, 0.005, 0.6))
    wheels = []
    if style == "suv":
        paint = material(name + " Silver paint", (0.43, 0.49, 0.54), 0.22, 0.68)
        box(name + "_lower_body", (0, 0, 0.72), (1.95, 0.86, 0.42), paint, bevel_width=0.16, parent=root)
        box(name + "_hood", (1.18, 0, 1.18), (0.78, 0.83, 0.20), paint, rotation=(0, -0.06, 0), bevel_width=0.12, parent=root)
        box(name + "_cabin", (-0.28, 0, 1.42), (1.05, 0.77, 0.62), paint, bevel_width=0.18, parent=root)
        box(name + "_front_glass", (0.46, -0.005, 1.56), (0.025, 0.70, 0.42), glass, rotation=(0, -0.28, 0), bevel_width=0.03, parent=root)
        box(name + "_side_glass_L", (-0.33, -0.785, 1.56), (0.68, 0.018, 0.35), glass, bevel_width=0.05, parent=root)
        box(name + "_side_glass_R", (-0.33, 0.785, 1.56), (0.68, 0.018, 0.35), glass, bevel_width=0.05, parent=root)
        for sx in (-1.27, 0.03):
            box(name + "_window_divider", (sx, -0.81, 1.56), (0.035, 0.025, 0.39), dark, parent=root)
        box(name + "_front_bumper", (1.92, 0, 0.60), (0.10, 0.80, 0.16), dark, bevel_width=0.05, parent=root)
        box(name + "_rear_bumper", (-1.93, 0, 0.58), (0.09, 0.78, 0.14), dark, bevel_width=0.04, parent=root)
        for y in (-0.48, -0.24, 0, 0.24, 0.48):
            box(name + "_grille_" + str(y), (2.03, y, 0.93), (0.035, 0.055, 0.20), dark, bevel_width=0.018, parent=root)
        for y in (-0.62, 0.62):
            cylinder(name + "_headlamp", (2.02, y, 1.05), 0.17, 0.05, white, vertices=24, rotation=(0, math.pi / 2, 0), parent=root)
            box(name + "_mirror", (0.16, y * 1.36, 1.46), (0.18, 0.08, 0.09), paint, bevel_width=0.05, parent=root)
        for y in (-0.55, 0.55):
            curve_object(name + "_roof_rail", [(-1.15, y, 2.05), (0.62, y, 2.05)], dark, 0.035, parent=root)
        box(name + "_license", (2.075, 0, 0.66), (0.015, 0.22, 0.065), material(name + " Plate", (0.92, 0.94, 0.88), 0.4), parent=root)
        wheel_x = (-1.22, 1.22)
        wheel_r = 0.47
        wheel_z = 0.48
        rim = chrome
    else:
        paint = material(name + " Red paint", (0.52, 0.012, 0.018), 0.18, 0.72)
        cyan = material(name + " Blue rims", (0.02, 0.24, 0.52), 0.19, 0.72)
        box(name + "_lower_body", (0, 0, 0.55), (2.25, 0.96, 0.34), paint, bevel_width=0.15, parent=root)
        box(name + "_hood", (1.30, 0, 0.91), (0.92, 0.90, 0.16), paint, rotation=(0, -0.04, 0), bevel_width=0.10, parent=root)
        box(name + "_cabin", (-0.30, 0, 1.12), (0.95, 0.82, 0.48), paint, bevel_width=0.16, parent=root)
        box(name + "_windscreen", (0.45, 0, 1.26), (0.025, 0.74, 0.36), glass, rotation=(0, -0.36, 0), bevel_width=0.03, parent=root)
        box(name + "_side_window", (-0.34, -0.835, 1.27), (0.62, 0.018, 0.28), glass, bevel_width=0.04, parent=root)
        box(name + "_splitter", (2.28, 0, 0.32), (0.22, 1.03, 0.07), dark, bevel_width=0.03, parent=root)
        box(name + "_rear_diffuser", (-2.27, 0, 0.37), (0.12, 1.0, 0.09), dark, bevel_width=0.03, parent=root)
        box(name + "_wing", (-1.82, 0, 1.42), (0.38, 1.24, 0.055), dark, rotation=(0, 0.05, 0), bevel_width=0.025, parent=root)
        for y in (-0.72, 0.72):
            box(name + "_wing_mount", (-1.70, y, 1.12), (0.045, 0.045, 0.28), dark, parent=root)
            box(name + "_headlamp", (2.22, y, 0.82), (0.06, 0.23, 0.10), white, bevel_width=0.03, parent=root)
        for y in (-0.55, 0.55):
            box(name + "_taillamp", (-2.24, y, 0.78), (0.05, 0.22, 0.09), redlamp, bevel_width=0.025, parent=root)
        # White racing livery echoes the reference drift car.
        decal = material(name + " White livery", (0.93, 0.93, 0.90), 0.34)
        box(name + "_door_decal", (-0.38, -0.982, 0.76), (0.78, 0.018, 0.16), decal, rotation=(0, 0, -0.12), bevel_width=0.025, parent=root)
        box(name + "_hood_stripe", (1.22, -0.01, 1.082), (0.77, 0.21, 0.018), decal, rotation=(0, -0.04, 0), parent=root)
        wheel_x = (-1.38, 1.42)
        wheel_r = 0.49
        wheel_z = 0.46
        rim = cyan
    for x in wheel_x:
        for y in (-0.91 if style == "suv" else -1.00, 0.91 if style == "suv" else 1.00):
            wheels.append(create_wheel(f"{name}_wheel_{x}_{y}", (x, y, wheel_z), wheel_r, 0.23, root, rubber, rim, chrome))
    return root, wheels


def insert_transform(obj, frame, location=None, rotation=None):
    if location is not None:
        obj.location = location
        obj.keyframe_insert("location", frame=frame)
    if rotation is not None:
        obj.rotation_euler = rotation
        obj.keyframe_insert("rotation_euler", frame=frame)


def set_visibility(obj, frame, visible):
    obj.hide_render = not visible
    obj.hide_viewport = not visible
    obj.keyframe_insert("hide_render", frame=frame)
    obj.keyframe_insert("hide_viewport", frame=frame)


def make_interpolation_constant(obj):
    if obj.animation_data and obj.animation_data.action:
        action = obj.animation_data.action
        try:
            curves = action.fcurves
        except AttributeError:
            curves = []
        for fc in curves:
            for kp in fc.keyframe_points:
                kp.interpolation = "CONSTANT"


def car_path(u):
    # The quadratic depth term keeps the airborne vehicle readable through
    # the final frame while retaining the reference's approaching turn.
    return Vector((7.5 - 13.0 * u + 3.5 * u * u, 24.0 - 32.0 * u + 6.0 * u * u, 0.07))


def car_path_heading(u):
    dx = -13.0 + 7.0 * u
    dy = -32.0 + 12.0 * u
    return math.atan2(dy, dx)


def build_car_turn(spec):
    scene = bpy.context.scene
    grass = noise_material("Alpine grass", (0.055, 0.18, 0.025), (0.28, 0.47, 0.09), 5.0, 0.9, 0.18)
    asphalt = noise_material("Mountain asphalt", (0.075, 0.082, 0.09), (0.23, 0.24, 0.25), 24.0, 0.88, 0.13)
    line_mat = material("Road edge paint", (0.82, 0.84, 0.80), 0.72)
    bark = material("Pine bark", (0.12, 0.055, 0.025), 0.92)
    needles = noise_material("Pine needles", (0.015, 0.075, 0.016), (0.04, 0.20, 0.035), 4.0, 0.9, 0.05)
    rock = noise_material("Dolomite rock", (0.22, 0.23, 0.22), (0.52, 0.50, 0.46), 3.0, 0.93, 0.32)
    snow = material("Mountain snow", (0.84, 0.88, 0.88), 0.78)
    fence_mat = material("Fence wire", (0.32, 0.34, 0.32), 0.38, 0.58)
    marker_mat = material("Road marker", (0.88, 0.88, 0.82), 0.72)
    black = material("Marker black", (0.025, 0.025, 0.022), 0.65)

    box("Alpine meadow", (0, 12, -0.45), (42, 48, 0.5), grass)
    centers = [tuple(car_path(i / 89.0)) for i in range(90)]
    create_road("Winding road", centers, 6.0, asphalt)
    left_edge, right_edge = [], []
    for i, p in enumerate(centers):
        prev = Vector(centers[max(0, i - 1)])
        nxt = Vector(centers[min(len(centers) - 1, i + 1)])
        tangent = (nxt - prev).normalized()
        side = Vector((-tangent.y, tangent.x, 0)).normalized()
        c = Vector(p)
        left_edge.append(tuple(c - side * 2.86 + Vector((0, 0, 0.035))))
        right_edge.append(tuple(c + side * 2.86 + Vector((0, 0, 0.035))))
    curve_object("Road edge left", left_edge, line_mat, 0.055)
    curve_object("Road edge right", right_edge, line_mat, 0.055)

    # Layered Dolomite-style backdrop.
    create_mountain("Mountain_A", (-14, 55, 1), 22, 25, rock, snow, 11)
    create_mountain("Mountain_B", (8, 58, 0), 27, 30, rock, snow, 17)
    create_mountain("Mountain_C", (28, 62, -2), 24, 24, rock, snow, 23)

    # Forest wall plus scattered foreground trees, seeded for repeatability.
    for i in range(52):
        y = RNG.uniform(18, 45)
        x = RNG.uniform(-27, 27)
        u = max(0.0, min(1.0, (24 - y) / 32.0))
        road_x = car_path(u).x
        if abs(x - road_x) < 5.5:
            x += 8.0 if x >= road_x else -8.0
        h = RNG.uniform(2.8, 7.0)
        create_pine(f"Pine_{i:02d}", (x, y, 0), h, bark, needles)
    for i in range(18):
        x = RNG.uniform(-16, 15)
        y = RNG.uniform(-4, 17)
        # Keep camera sightline and road clear.
        u = max(0.0, min(1.0, (24 - y) / 32.0))
        if abs(x - car_path(u).x) < 5.4:
            x += 7.5 * (-1 if x < car_path(u).x else 1)
        create_pine(f"Near_pine_{i:02d}", (x, y, 0), RNG.uniform(2.2, 5.2), bark, needles)

    # Fence follows the outer bend, with thin cable strands.
    fence_pts_top, fence_pts_mid = [], []
    for i in range(8, 72, 6):
        p = Vector(centers[i])
        prev = Vector(centers[i - 1])
        nxt = Vector(centers[i + 1])
        tangent = (nxt - prev).normalized()
        side = Vector((-tangent.y, tangent.x, 0)).normalized()
        loc = p + side * 4.25
        cylinder(f"Fence_post_{i}", (loc.x, loc.y, 0.62), 0.045, 1.24, fence_mat, vertices=8)
        fence_pts_top.append((loc.x, loc.y, 1.08))
        fence_pts_mid.append((loc.x, loc.y, 0.60))
    curve_object("Fence top wire", fence_pts_top, fence_mat, 0.018)
    curve_object("Fence mid wire", fence_pts_mid, fence_mat, 0.014)

    # Roadside delineators and rock scatter add readable scale.
    for i in range(7, 83, 9):
        p = Vector(centers[i])
        prev = Vector(centers[max(0, i - 1)])
        nxt = Vector(centers[min(len(centers) - 1, i + 1)])
        t = (nxt - prev).normalized()
        side = Vector((-t.y, t.x, 0)).normalized()
        for sign in (-1, 1):
            loc = p + side * 3.55 * sign
            box(f"Delineator_{i}_{sign}", (loc.x, loc.y, 0.50), (0.06, 0.06, 0.50), marker_mat, bevel_width=0.018)
            box(f"Delineator_band_{i}_{sign}", (loc.x, loc.y, 0.72), (0.067, 0.067, 0.075), black, bevel_width=0.01)
    for i in range(42):
        x, y = RNG.uniform(-25, 25), RNG.uniform(-2, 36)
        u = max(0.0, min(1.0, (24 - y) / 32.0))
        if abs(x - car_path(u).x) < 4.4:
            continue
        sphere(f"Meadow_rock_{i:02d}", (x, y, 0.12), (RNG.uniform(0.12, 0.42), RNG.uniform(0.12, 0.35), RNG.uniform(0.08, 0.24)), rock, 12, 6)

    vehicle, wheels = create_vehicle("Silver compact SUV", "suv")
    ground_truth = []
    for frame_idx in range(spec["frames"]):
        u = frame_idx / (spec["frames"] - 1)
        base = car_path(u)
        anomaly_t = max(0.0, (frame_idx - spec["onset"]) / max(1, spec["frames"] - 1 - spec["onset"]))
        lift = 3.6 * anomaly_t * anomaly_t
        loc = base + Vector((0, 0, lift))
        yaw = car_path_heading(u)
        pitch = -0.08 * anomaly_t + 0.22 * anomaly_t * anomaly_t
        roll = 0.16 * anomaly_t * math.sin(anomaly_t * math.pi)
        insert_transform(vehicle, frame_idx + 1, tuple(loc), (pitch, roll, yaw))
        for wheel in wheels:
            wheel.rotation_euler[1] = -u * 37.0
            wheel.keyframe_insert("rotation_euler", frame=frame_idx + 1)
        ground_truth.append(
            {
                "frame_index": frame_idx,
                "blender_frame": frame_idx + 1,
                "time_seconds": frame_idx / FPS,
                "active": frame_idx >= spec["onset"],
                "position_m": [round(v, 5) for v in loc],
                "counterfactual_position_m": [round(v, 5) for v in base],
                "gravity_reversal_lift_m": round(lift, 5),
            }
        )
    focus = empty("Camera focus", (1.2, 7.0, 1.35))
    # A longer camera-to-road distance preserves the complete airborne SUV at
    # the end while keeping the opening scale close to the source shot.
    create_camera((0.5, -22.0, 7.0), (0.8, 6.8, 1.8), 55, focus, 7.1)
    add_sun(energy=3.2)
    add_area("Soft sky fill", (-8, -5, 13), 520, 9, (0.72, 0.82, 1.0), (0, 6, 0))
    return ground_truth


def drift_path(u):
    return Vector((8.2 - 15.2 * u, 14.8 - 16.6 * u, 0.05))


def build_drift(spec):
    asphalt = noise_material("Circuit asphalt", (0.045, 0.047, 0.052), (0.22, 0.22, 0.21), 31.0, 0.92, 0.22)
    grass = noise_material("Circuit verge", (0.025, 0.12, 0.015), (0.20, 0.36, 0.035), 7.0, 0.92, 0.18)
    white = material("Track white", (0.88, 0.88, 0.82), 0.78)
    red = material("Barrier red", (0.56, 0.018, 0.012), 0.43)
    rubber = material("Tire barrier rubber", (0.012, 0.014, 0.016), 0.37)
    wall = material("Paddock white", (0.70, 0.72, 0.70), 0.78)
    roof = material("Paddock roof", (0.055, 0.065, 0.07), 0.68)
    steel = material("Grandstand steel", (0.24, 0.27, 0.28), 0.32, 0.66)
    cyan = material("Teleport plasma", (0.005, 0.30, 0.74), 0.12, 0.32, emission=(0.0, 0.55, 1.0, 8.0))
    skid = material("Skid marks", (0.006, 0.006, 0.007), 0.34)
    tent = material("Tent fabric", (0.90, 0.90, 0.86), 0.72)

    box("Circuit grass base", (0, 7, -0.42), (29, 33, 0.45), grass)
    # Keep the asphalt top above the grass base to avoid coplanar z-fighting.
    box("Main tarmac", (0, 7, -0.02), (17, 23, 0.10), asphalt)
    # Track seams, painted edges and accumulated drift arcs.
    for x in (-9.5, 9.5):
        curve_object(f"Track edge {x}", [(x, -10, 0.045), (x, 25, 0.045)], white, 0.07)
    for i in range(16):
        y = -6 + i * 2.1
        curve_object(f"Concrete seam {i}", [(-10, y, 0.04), (10, y + RNG.uniform(-0.15, 0.15), 0.04)], wall, 0.018)
    for i in range(13):
        pts = []
        offset = RNG.uniform(-0.28, 0.28)
        for k in range(26):
            u = k / 25
            p = drift_path(u)
            pts.append((p.x + offset + math.sin(u * math.pi) * i * 0.025, p.y - i * 0.06, 0.055))
        curve_object(f"Drift skid {i:02d}", pts, skid, RNG.uniform(0.025, 0.055))

    # Red/white tire wall across the paddock edge.
    for i in range(30):
        x = -15 + i * 1.04
        mat = red if i % 2 == 0 else white
        for level in range(2):
            torus(f"Barrier_{i}_{level}", (x, 20.0, 0.28 + level * 0.36), 0.27, 0.115, mat, rotation=(0, 0, 0))
    box("Paddock building", (1.5, 24.0, 2.2), (13.5, 2.2, 2.2), wall, bevel_width=0.12)
    box("Paddock roof", (1.5, 24.0, 4.52), (14.0, 2.45, 0.16), roof, bevel_width=0.08)
    for i in range(9):
        box(f"Garage_{i}", (-10.0 + i * 2.9, 21.73, 1.55), (1.18, 0.07, 1.38), roof, bevel_width=0.025)
        box(f"Garage_header_{i}", (-10.0 + i * 2.9, 21.58, 3.2), (1.2, 0.09, 0.18), red if i % 3 == 0 else wall)

    # Grandstand scaffold and a few spectators as colored silhouettes.
    for x in (-12.5, -8.5, -4.5, 6.5, 10.5):
        cylinder("Stand post", (x, 22.3, 5.6), 0.055, 4.4, steel, 8)
        curve_object("Stand brace", [(x - 1.7, 22.2, 3.5), (x + 1.7, 22.2, 7.0)], steel, 0.035)
    palette = [
        material("Spectator orange", (0.95, 0.17, 0.015), 0.55),
        material("Spectator blue", (0.02, 0.16, 0.55), 0.55),
        material("Spectator yellow", (0.86, 0.55, 0.02), 0.55),
        material("Spectator dark", (0.04, 0.045, 0.05), 0.55),
    ]
    for i in range(38):
        x = RNG.uniform(-13, 13)
        y = 22.0 + RNG.uniform(-0.25, 0.25)
        z = RNG.uniform(4.7, 6.7)
        cylinder(f"Spectator_body_{i}", (x, y, z), 0.105, 0.38, palette[i % len(palette)], 10)
        sphere(f"Spectator_head_{i}", (x, y, z + 0.29), (0.105, 0.105, 0.115), material("Skin", (0.52, 0.30, 0.18), 0.75), 12, 6)

    # Two event tents on the left match the white canopy silhouettes.
    for tx in (-11.8, -7.8):
        for sx in (-1, 1):
            for sy in (-1, 1):
                cylinder("Tent leg", (tx + sx * 1.3, 17.2 + sy * 0.9, 1.3), 0.035, 2.6, steel, 8)
        bpy.ops.mesh.primitive_cone_add(vertices=4, radius1=2.05, radius2=0.05, depth=1.55, location=(tx, 17.2, 3.15), rotation=(0, 0, math.pi / 4))
        assign(bpy.context.object, tent)

    vehicle, wheels = create_vehicle("Red drift coupe", "drift")
    onset = spec["onset"]
    jump_u = 0.245
    ground_truth = []
    pre_portal_position = drift_path((onset - 1) / (spec["frames"] - 1) * 0.78)
    post_portal_position = drift_path(onset / (spec["frames"] - 1) * 0.78 + jump_u)
    portal_parts = []
    for idx, pos in enumerate((pre_portal_position, post_portal_position)):
        for j, radius in enumerate((0.82, 1.08, 1.33)):
            portal_parts.append(torus(f"Teleport_ring_{idx}_{j}", (pos.x, pos.y, 1.15), radius, 0.025 + j * 0.009, cyan, rotation=(math.pi / 2, 0, 0)))
        for j in range(10):
            ang = 2 * math.pi * j / 10
            portal_parts.append(
                curve_object(
                    f"Teleport_streak_{idx}_{j}",
                    [
                        (pos.x + math.cos(ang) * 0.35, pos.y, 1.15 + math.sin(ang) * 0.35),
                        (pos.x + math.cos(ang) * 1.55, pos.y, 1.15 + math.sin(ang) * 1.55),
                    ],
                    cyan,
                    0.018,
                )
            )
    for part in portal_parts:
        set_visibility(part, 1, False)
        set_visibility(part, onset + 1, True)
        set_visibility(part, onset + 3, False)

    for frame_idx in range(spec["frames"]):
        continuous_u = frame_idx / (spec["frames"] - 1) * 0.78
        active = frame_idx >= onset
        actual_u = continuous_u + (jump_u if active else 0.0)
        base = drift_path(continuous_u)
        loc = drift_path(actual_u)
        heading = math.atan2(-16.6, -15.2)
        slip = math.radians(10.0) * math.sin(actual_u * math.pi * 1.35)
        insert_transform(vehicle, frame_idx + 1, tuple(loc), (0.0, math.radians(-1.2), heading + slip))
        for wheel in wheels:
            wheel.rotation_euler[1] = -actual_u * 46.0
            wheel.keyframe_insert("rotation_euler", frame=frame_idx + 1)
        ground_truth.append(
            {
                "frame_index": frame_idx,
                "blender_frame": frame_idx + 1,
                "time_seconds": frame_idx / FPS,
                "active": active,
                "position_m": [round(v, 5) for v in loc],
                "counterfactual_position_m": [round(v, 5) for v in base],
                "teleport_offset_along_path_fraction": jump_u if active else 0.0,
                "teleport_distance_m": round((drift_path(continuous_u + jump_u) - base).length, 5) if active else 0.0,
            }
        )
    focus = empty("Camera focus", (0, 6.2, 0.8))
    create_camera((0.2, -17.4, 5.2), (0.3, 6.3, 0.95), 54, focus, 8.0)
    add_sun((math.radians(28), math.radians(-22), math.radians(-42)), 3.4)
    add_area("Track fill", (-9, -4, 12), 650, 10, (0.76, 0.86, 1.0), (0, 6, 0))
    return ground_truth


def create_soccer_ball():
    root = empty("Blue-white football")
    white = noise_material("Football leather white", (0.58, 0.60, 0.62), (0.93, 0.94, 0.91), 18.0, 0.52, 0.08)
    blue = material("Football blue patches", (0.012, 0.075, 0.52), 0.40)
    seam = material("Football seams", (0.02, 0.025, 0.03), 0.56)
    sphere("Football shell", (0, 0, 0), (0.48, 0.48, 0.48), white, 48, 24, root)
    phi = (1 + math.sqrt(5)) / 2
    dirs = []
    for a, b in ((1, phi), (-1, phi), (1, -phi), (-1, -phi)):
        dirs.extend((Vector((0, a, b)), Vector((a, b, 0)), Vector((b, 0, a))))
    for i, direction in enumerate(dirs):
        direction.normalize()
        loc = direction * 0.479
        patch = cylinder(f"Blue pentagon {i:02d}", tuple(loc), 0.145, 0.016, blue, 5, parent=root)
        patch.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
        torus_patch = torus(f"Patch seam {i:02d}", tuple(direction * 0.488), 0.145, 0.009, seam, parent=root)
        torus_patch.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
    return root


def build_soccer(spec):
    grass = noise_material("Backyard grass", (0.025, 0.10, 0.012), (0.22, 0.38, 0.055), 16.0, 0.96, 0.32)
    bark = noise_material("Fruit tree bark", (0.10, 0.035, 0.012), (0.28, 0.13, 0.045), 5.0, 0.98, 0.28)
    leaf_green = material("Leaves green", (0.025, 0.21, 0.028), 0.75)
    leaf_light = material("Leaves light", (0.10, 0.38, 0.055), 0.72)
    shrub_dark = material("Shrub dark", (0.012, 0.085, 0.018), 0.86)
    shrub_light = material("Shrub light", (0.04, 0.24, 0.035), 0.82)
    fence = material("Chain link galvanized", (0.29, 0.33, 0.31), 0.27, 0.78)
    concrete = noise_material("Concrete post", (0.24, 0.23, 0.21), (0.50, 0.47, 0.42), 6.0, 0.95, 0.18)
    house = material("Warm house wall", (0.54, 0.24, 0.14), 0.90)
    leaf_fall = material("Falling autumn leaf", (0.62, 0.16, 0.015), 0.78)

    box("Yard ground", (0, 2, -0.22), (12, 13, 0.25), grass)
    box("House background", (0, 8.7, 3.2), (12, 0.3, 3.2), house, bevel_width=0.08)
    # Fence posts and chain-link diamonds.
    for x in (-6, -3, 0, 3, 6):
        cylinder(f"Fence post {x}", (x, 4.4, 2.05), 0.075, 4.1, fence, 12)
    for x in (-3.5,):
        box("Concrete fence pillar", (x, 4.25, 2.05), (0.22, 0.22, 2.05), concrete, bevel_width=0.035)
    for k in range(-16, 17):
        x0 = k * 0.52
        curve_object(f"Mesh up {k}", [(x0 - 2.2, 4.36, 0.0), (x0 + 2.2, 4.36, 4.0)], fence, 0.010)
        curve_object(f"Mesh down {k}", [(x0 - 2.2, 4.37, 4.0), (x0 + 2.2, 4.37, 0.0)], fence, 0.010)
    curve_object("Fence top rail", [(-7.5, 4.4, 4.0), (7.5, 4.4, 4.0)], fence, 0.045)

    # Mature fruit trees with curved branches and layered foliage.
    tree_specs = [(2.9, 3.55, 0.38, 5.6), (-5.0, 4.0, 0.30, 5.0)]
    for ti, (tx, ty, rad, h) in enumerate(tree_specs):
        cylinder(f"Tree trunk {ti}", (tx, ty, h * 0.43), rad, h * 0.86, bark, 14, rotation=(0.04, -0.07, 0.02))
        for bi in range(7):
            ang = 2 * math.pi * bi / 7 + 0.3 * ti
            start = Vector((tx, ty, h * (0.54 + bi * 0.035)))
            end = start + Vector((math.cos(ang) * (1.0 + 0.18 * bi), math.sin(ang) * 0.65, 0.9 + 0.16 * (bi % 3)))
            curve_object(f"Tree {ti} branch {bi}", [tuple(start), tuple((start + end) * 0.5 + Vector((0, 0, 0.25))), tuple(end)], bark, 0.07 - bi * 0.004)
            for li in range(3):
                jitter = Vector((RNG.uniform(-0.45, 0.45), RNG.uniform(-0.32, 0.32), RNG.uniform(-0.25, 0.4)))
                sphere(
                    f"Tree {ti} foliage {bi}_{li}",
                    tuple(end + jitter),
                    (0.48, 0.33, 0.42),
                    leaf_green if (bi + li) % 2 else leaf_light,
                    12,
                    6,
                )
    # Dense understory behind the mesh.
    for i in range(58):
        x = RNG.uniform(-8, 8)
        y = RNG.uniform(4.65, 7.5)
        z = RNG.uniform(0.25, 1.5)
        sc = RNG.uniform(0.22, 0.65)
        sphere(f"Shrub_{i:02d}", (x, y, z), (sc, sc * 0.75, sc * 0.9), shrub_dark if i % 3 else shrub_light, 12, 6)
    # Individual blades in the foreground catch highlights and imply continuing wind.
    for i in range(150):
        x, y = RNG.uniform(-7, 7), RNG.uniform(-0.5, 4.0)
        h = RNG.uniform(0.08, 0.42)
        lean = RNG.uniform(-0.10, 0.10)
        curve_object(f"Grass blade {i:03d}", [(x, y, 0.01), (x + lean, y, h)], leaf_light if i % 5 == 0 else leaf_green, 0.008)

    ball = create_soccer_ball()
    onset = spec["onset"]
    hover_pos = None
    ground_truth = []
    for frame_idx in range(spec["frames"]):
        if frame_idx <= onset:
            u = frame_idx / max(1, onset)
            pos = Vector((4.2 - 4.35 * u, 0.65 + 0.22 * u, 0.56 + 1.12 * (u ** 0.72)))
            hover_pos = pos.copy()
        else:
            pos = hover_pos.copy()
        # Rotation persists after translation freezes: the environment and time continue.
        rot_t = frame_idx / FPS
        insert_transform(ball, frame_idx + 1, tuple(pos), (rot_t * 3.7, rot_t * 7.2, rot_t * 1.1))
        ground_truth.append(
            {
                "frame_index": frame_idx,
                "blender_frame": frame_idx + 1,
                "time_seconds": frame_idx / FPS,
                "active": frame_idx >= onset,
                "position_m": [round(v, 5) for v in pos],
                "translation_frozen": frame_idx >= onset,
                "angular_velocity_rad_s": [3.7, 7.2, 1.1],
            }
        )

    # A falling leaf crosses the anomaly onset, providing an internal clock.
    bpy.ops.mesh.primitive_circle_add(vertices=7, radius=0.17, fill_type="TRIFAN", location=(-1.7, 0.4, 3.5))
    falling = bpy.context.object
    falling.name = "Continuing-time falling leaf"
    assign(falling, leaf_fall)
    for frame_idx in range(spec["frames"]):
        t = frame_idx / (spec["frames"] - 1)
        insert_transform(falling, frame_idx + 1, (-1.7 + 1.1 * t + 0.16 * math.sin(t * 12), 0.35, 3.5 - 3.1 * t), (t * 9, t * 5, t * 14))

    focus = empty("Camera focus", (0.1, 1.0, 1.3))
    cam = create_camera((0.0, -10.8, 3.15), (0.05, 2.1, 1.55), 53, focus, 6.3)
    # A subtle hand-held drift keeps the plate alive while the ball remains fixed.
    for frame_idx in (1, spec["frames"]):
        t = (frame_idx - 1) / (spec["frames"] - 1)
        loc = Vector((0.0 + 0.16 * t, -10.8, 3.15 + 0.05 * t))
        cam.location = loc
        look_at(cam, (0.05, 2.1, 1.55))
        cam.keyframe_insert("location", frame=frame_idx)
        cam.keyframe_insert("rotation_euler", frame=frame_idx)
    add_sun((math.radians(42), math.radians(-20), math.radians(-36)), 2.6)
    add_area("Garden sky fill", (-4, -4, 9), 430, 8, (0.72, 0.84, 1.0), (0, 2, 1))
    return ground_truth


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_metadata(scene_name, spec, ground_truth, samples):
    ref = ROOT / spec["reference"]
    payload = {
        "schema_version": "1.0",
        "generator": "Blender 5.1 procedural scene generator",
        "blender_version": bpy.app.version_string,
        "scene_id": scene_name,
        "title": spec["title"],
        "anomaly": spec["anomaly"],
        "reference_video": spec["reference"],
        "reference_sha256": sha256(ref) if ref.exists() else None,
        "render": {
            "engine": bpy.context.scene.render.engine,
            "resolution": [WIDTH, HEIGHT],
            "fps": FPS,
            "frame_start": 1,
            "frame_end": spec["frames"],
            "frame_count": spec["frames"],
            "samples": samples,
            "color_management": bpy.context.scene.view_settings.look,
            "motion_blur_shutter": bpy.context.scene.render.motion_blur_shutter,
        },
        "anomaly_onset": {
            "zero_based_frame": spec["onset"],
            "blender_frame": spec["onset"] + 1,
            "time_seconds": spec["onset"] / FPS,
        },
        "paths": {
            "blend": f"scenes/{scene_name}.blend",
            "frames": f"renders/{scene_name}/frame_####.png",
            "video": f"videos/{scene_name}_anomaly.mp4",
            "preview": f"previews/{scene_name}_contact_sheet.png",
        },
        "ground_truth": ground_truth,
    }
    path = ROOT / "metadata" / f"{scene_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    spec = SPECS[args.scene]
    clean_scene()
    scene = configure_scene(args.scene, args.samples, spec["frames"])
    if args.scene == "car-turn":
        ground_truth = build_car_turn(spec)
    elif args.scene == "drift-straight":
        ground_truth = build_drift(spec)
    else:
        ground_truth = build_soccer(spec)
    scene["scene_id"] = args.scene
    scene["anomaly"] = spec["anomaly"]
    scene["anomaly_onset_zero_based"] = spec["onset"]
    scene["reference_video"] = spec["reference"]
    write_metadata(args.scene, spec, ground_truth, args.samples)
    blend_path = ROOT / "scenes" / f"{args.scene}.blend"
    blend_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), compress=True)
    print(f"BUILT {args.scene}: {blend_path}")


if __name__ == "__main__":
    main()
