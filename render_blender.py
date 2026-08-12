#!/usr/bin/env python3
"""
render_blender.py -- path-traced version of the E3-shrink shot.

Runs against `bpy`, Blender's Python module, so it needs no Blender install and
no GUI:

    pip install bpy                     # ~370 MB wheel, cp311
    python3 render_blender.py --quality preview

ChimeraX (render_shrink.cxc) remains the portable deliverable. This exists
because Cycles gives real path-traced ambient occlusion, depth of field and
translucency, which is the difference between "a molecular figure" and "a
shot", and because it renders on any headless box with no display.

Design notes
------------
Geometry is rebuilt from scratch every frame rather than keyframed. Molecular
trajectories are already explicit coordinate lists, so a keyframe layer buys
nothing and costs a lot of fragile bpy state; a frame is a pure function of
its index, which also means --start/--end resume works for free.

Representations, cheapest first:
  spheres   one merged mesh, per-vertex colour + glow attributes, so 700 atoms
            cost one object and one material rather than 700 of each
  tube      Catmull-Rom interpolated Ca trace, poly curve with a bevel
  surface   metaballs -- Blender's native blobby-molecule surface, which is
            Blinn's Gaussian density surface, i.e. the real thing
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np

import structlib as S

try:
    import bpy
    import mathutils
except ImportError:
    sys.exit("bpy not importable. Install with:  pip install bpy")


# ---------------------------------------------------------------------------
# palette -- same hexes as generated.cxc so the two renderers agree
# ---------------------------------------------------------------------------

PALETTE = {
    "e3":       "#2E6FA8",
    "e3dark":   "#1B4C78",
    "motif":    "#E8663C",
    "hotspot":  "#FF8C42",
    "ligand":   "#14B8A6",
    "binder":   "#F2B441",
    "binderhi": "#FFD98A",
    "ghost":    "#C8CCD2",
    "zinc":     "#9AA7B4",
    "bg":       "#0B1017",
}

# Scene units are NANOMETRES, not Angstroms. Blender's lights are physical
# (Watts) and its depth of field is a real thin-lens model, so a scene built
# at 1 unit = 1 A behaves like a 40-metre-wide protein: lights 50 m away read
# as pitch black and no f-stop produces visible bokeh. At 1 unit = 1 nm the
# defaults land where they were designed to.
NM = 0.1                      # multiply Angstroms by this

VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
       "ZN": 1.39, "FE": 1.40, "MG": 1.73, "H": 1.10}


def ang(x):
    """Angstroms -> scene units. (Not named `A`: that is the Assets bundle.)"""
    return x * NM

MOTIF_RANGE = (312, 325)
HOTSPOTS = [312, 313, 314, 325]


def hex_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def srgb_to_linear(c: tuple) -> tuple:
    def f(u):
        return u / 12.92 if u <= 0.04045 else ((u + 0.055) / 1.055) ** 2.4
    return tuple(f(v) for v in c)


def col(name: str) -> tuple:
    return srgb_to_linear(hex_rgb(PALETTE[name]))


def shade(rgb: tuple, factor: float) -> tuple:
    """Lighten (>1) or darken (<1) in HSV, keeping hue. Used to give the
    dissolving fragments some variation so they don't read as one slab."""
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    return colorsys.hsv_to_rgb(h, s, max(0.0, min(1.0, v * factor)))


def ease(t: float, power: float = 3.0) -> float:
    t = max(0.0, min(1.0, t))
    return 0.5 * (2 * t) ** power if t < 0.5 else 1 - 0.5 * (2 * (1 - t)) ** power


def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


# ---------------------------------------------------------------------------
# scene primitives
# ---------------------------------------------------------------------------

def purge() -> None:
    """Wipe everything the previous frame built. Orphan meshes and metaballs
    accumulate fast enough to exhaust memory over a few hundred frames."""
    for coll in (bpy.data.objects, bpy.data.meshes, bpy.data.curves,
                 bpy.data.metaballs):
        for item in list(coll):
            coll.remove(item, do_unlink=True)


def icosphere_template(subdiv: int = 1):
    """One unit icosphere as (verts, faces), reused for every atom."""
    t = (1.0 + 5.0 ** 0.5) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=float)
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    for _ in range(subdiv):
        vlist = [tuple(v) for v in verts]
        index = {v: i for i, v in enumerate(vlist)}
        new_faces = []
        for (a, b, c) in faces:
            mids = []
            for (i, j) in ((a, b), (b, c), (c, a)):
                m = tuple((verts[i] + verts[j]) / 2.0)
                if m not in index:
                    index[m] = len(vlist)
                    vlist.append(m)
                mids.append(index[m])
            ab, bc, ca = mids
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        verts = np.array(vlist, dtype=float)
        faces = new_faces
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    return verts, faces


_ICO_CACHE: dict = {}


def get_ico(subdiv: int):
    if subdiv not in _ICO_CACHE:
        _ICO_CACHE[subdiv] = icosphere_template(subdiv)
    return _ICO_CACHE[subdiv]


def make_spheres(name: str, centres: np.ndarray, radii: np.ndarray,
                 colours: np.ndarray, glow: np.ndarray | None = None,
                 subdiv: int = 1, material=None):
    """
    All spheres merged into one mesh, with per-vertex `col` and `glow`
    attributes. One object, one material, one draw -- which is what makes a
    few hundred atoms per frame affordable.
    """
    if len(centres) == 0:
        return None
    tv, tf = get_ico(subdiv)
    nv, nf = len(tv), len(tf)
    n = len(centres)

    verts = (tv[None, :, :] * radii[:, None, None]) + centres[:, None, :]
    verts = verts.reshape(-1, 3)

    base = (np.arange(n) * nv)[:, None, None]
    faces = (np.array(tf)[None, :, :] + base).reshape(-1, 3)

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts.tolist(), [], faces.tolist())
    mesh.validate()
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))

    cattr = mesh.color_attributes.new(name="col", type="FLOAT_COLOR", domain="POINT")
    cols = np.repeat(colours, nv, axis=0)
    if cols.shape[1] == 3:
        cols = np.concatenate([cols, np.ones((len(cols), 1))], axis=1)
    cattr.data.foreach_set("color", cols.reshape(-1).tolist())

    g = np.zeros(n) if glow is None else np.asarray(glow, dtype=float)
    gattr = mesh.attributes.new(name="glow", type="FLOAT", domain="POINT")
    gattr.data.foreach_set("value", np.repeat(g, nv).tolist())

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    if material:
        obj.data.materials.append(material)
    return obj


def catmull_rom(points: np.ndarray, samples_per_seg: int = 6) -> np.ndarray:
    """Smooth an ordered point list. A poly curve straight through Ca positions
    looks like a polyline; this makes it look like a backbone."""
    p = np.asarray(points, dtype=float)
    if len(p) < 3:
        return p
    ext = np.vstack([p[0] + (p[0] - p[1]), p, p[-1] + (p[-1] - p[-2])])
    out = []
    for i in range(len(ext) - 3):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        for s in range(samples_per_seg):
            t = s / samples_per_seg
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t
                              + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(p[-1])
    return np.array(out)


def make_tube(name: str, points: np.ndarray, radius: float, material,
              smooth: int = 6, break_at: float = ang(5.0)):
    """
    Ca trace as a bevelled poly curve, one spline per continuous segment.

    Splitting at gaps matters. A Ca trace drawn straight through a chain break
    lays a rod across the whole molecule, and crystal structures have breaks
    wherever a loop was too disordered to model -- so the artefact appears
    exactly where the protein is most interesting. Consecutive Ca are 3.8 A
    apart, so anything past ~5 A is a break rather than a bond.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) < 2:
        return None

    gaps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cuts = np.flatnonzero(gaps > break_at) + 1
    segments = np.split(points, cuts) if len(cuts) else [points]

    cu = bpy.data.curves.new(name, type="CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = 3
    cu.use_fill_caps = True

    made = 0
    for seg in segments:
        if len(seg) < 2:
            continue                     # a lone residue has no tube to draw
        pts = catmull_rom(seg, smooth) if len(seg) >= 3 else seg
        sp = cu.splines.new("POLY")
        sp.points.add(len(pts) - 1)
        flat = np.concatenate([pts, np.ones((len(pts), 1))], axis=1).reshape(-1)
        sp.points.foreach_set("co", flat.tolist())
        made += 1
    if made == 0:
        bpy.data.curves.remove(cu)
        return None

    obj = bpy.data.objects.new(name, cu)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


def make_metaball_surface(name: str, centres: np.ndarray, radii: np.ndarray,
                          material, resolution: float = 0.55):
    """
    Blender metaballs are Blinn blobby-molecule surfaces -- a Gaussian density
    field thresholded at an isovalue -- so this is a genuine molecular surface
    rather than a shrink-wrap approximation. Resolution is the tessellation
    step in Blender units; below ~0.4 it gets expensive fast.
    """
    if len(centres) == 0:
        return None
    # Metaball tessellation cost climbs steeply with element count. BIR3 is
    # ~700 atoms and fine; a full-length AlphaFold model is ~5000 and would
    # crawl at the same resolution, so coarsen rather than stall.
    if len(centres) > 1500:
        resolution *= (len(centres) / 1500.0) ** 0.34
    mb = bpy.data.metaballs.new(name)
    mb.resolution = resolution
    mb.render_resolution = resolution
    mb.threshold = 0.6
    obj = bpy.data.objects.new(name, mb)
    bpy.context.scene.collection.objects.link(obj)
    for c, r in zip(centres, radii):
        el = mb.elements.new()
        el.co = tuple(float(v) for v in c)
        el.radius = float(r)
        el.stiffness = 2.0
    obj.data.materials.append(material)
    return obj


# ---------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------

def _principled(mat):
    return next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")


def _set(node, key, value):
    """bpy renames Principled sockets between versions; skip what's absent
    rather than dying on a socket name."""
    if key in node.inputs:
        node.inputs[key].default_value = value
        return True
    return False


def mat_attribute(name: str, roughness: float = 0.42, glow_boost: float = 2.2):
    """Per-vertex colour + per-vertex emission, for the sphere meshes."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = _principled(mat)
    _set(bsdf, "Roughness", roughness)
    _set(bsdf, "Metallic", 0.0)
    _set(bsdf, "Specular IOR Level", 0.5)

    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "col"
    nt.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    if "Emission Color" in bsdf.inputs:
        nt.links.new(attr.outputs["Color"], bsdf.inputs["Emission Color"])

    glow = nt.nodes.new("ShaderNodeAttribute")
    glow.attribute_name = "glow"
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = glow_boost
    nt.links.new(glow.outputs["Fac"], mul.inputs[0])
    if "Emission Strength" in bsdf.inputs:
        nt.links.new(mul.outputs[0], bsdf.inputs["Emission Strength"])
    return mat


def mat_solid(name: str, rgb: tuple, roughness: float = 0.4,
              alpha: float = 1.0, emission: float = 0.0,
              transmission: float = 0.0, metallic: float = 0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = _principled(mat)
    _set(bsdf, "Base Color", (*rgb, 1.0))
    _set(bsdf, "Roughness", roughness)
    _set(bsdf, "Metallic", metallic)
    if emission > 0:
        _set(bsdf, "Emission Color", (*rgb, 1.0))
        _set(bsdf, "Emission Strength", emission)
    if transmission > 0:
        _set(bsdf, "Transmission Weight", transmission)
        _set(bsdf, "IOR", 1.36)
    if alpha < 1.0:
        _set(bsdf, "Alpha", alpha)
        mat.blend_method = "BLEND" if hasattr(mat, "blend_method") else mat.blend_method
    return mat


# ---------------------------------------------------------------------------
# scene setup
# ---------------------------------------------------------------------------

# Each tier: resolution, sample ceiling, light bounces, icosphere subdiv,
# metaball step (nm), adaptive-noise threshold, and a per-frame wall-clock cap
# in seconds (0 = uncapped).
#
# The cap is what makes 1080p tractable on CPU. Cycles + adaptive sampling +
# OpenImageDenoise converges the easy regions fast and spends its remaining
# budget on the noisy ones; bounding the wall clock means one awkward frame
# (deep transmission through the translucent surface, say) cannot blow the
# whole-sequence estimate. The denoiser cleans up whatever the cap left
# unconverged, which for smooth molecular surfaces with no caustics is plenty.
QUALITY = {
    #             res_x  res_y  samp  bnce ico  meta   athr   tcap(s)
    "thumb":     (480,   270,    24,   4,   1,   0.075, 0.02,   0),
    "preview":   (960,   540,    64,   6,   1,   0.055, 0.015,  0),
    "hd":        (1920,  1080,  192,   8,   2,   0.035, 0.010, 40),
    "final":     (1920,  1080,  384,  10,   2,   0.030, 0.005,  0),
}


def enable_gpu(kind: str) -> str:
    """
    Turn on GPU rendering. `kind` is auto | metal | cuda | optix | hip | cpu.

    On Apple Silicon this is the difference between minutes and hours: the
    same shot that takes ~55 s/frame on four CPU cores runs several times
    faster on the Metal GPU. Returns the backend that actually engaged, so
    the caller can print it and the user knows CPU didn't silently win.
    """
    if kind == "cpu":
        return "CPU"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except (KeyError, AttributeError):
        return "CPU"

    order = {
        "auto": ["METAL", "OPTIX", "CUDA", "HIP", "ONEAPI"],
        "metal": ["METAL"], "cuda": ["CUDA"], "optix": ["OPTIX"],
        "hip": ["HIP"], "oneapi": ["ONEAPI"],
    }.get(kind, ["METAL", "OPTIX", "CUDA", "HIP", "ONEAPI"])

    available = {opt.identifier for opt in
                 prefs.bl_rna.properties["compute_device_type"].enum_items}
    for backend in order:
        if backend not in available:
            continue
        prefs.compute_device_type = backend
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type == backend]
        if not gpus:
            continue
        for d in prefs.devices:
            d.use = (d.type == backend) or (d.type == "CPU" and backend == "OPTIX")
        return backend
    return "CPU"


def setup_world(quality: str, transparent: bool, device: str = "cpu") -> None:
    sc = bpy.context.scene
    rx, ry, samples, bounces, _, _, athr, tcap = QUALITY[quality]
    sc.render.engine = "CYCLES"
    backend = enable_gpu(device)
    sc.cycles.device = "CPU" if backend == "CPU" else "GPU"
    setup_world._backend = backend                   # for the driver to report
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    try:
        sc.cycles.denoiser = "OPENIMAGEDENOISE"      # CPU denoiser, no GPU needed
        sc.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    except (TypeError, AttributeError):
        pass
    sc.cycles.max_bounces = bounces
    sc.cycles.diffuse_bounces = bounces
    sc.cycles.glossy_bounces = bounces
    sc.cycles.transmission_bounces = bounces
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = athr
    sc.cycles.time_limit = float(tcap)               # 0 = no per-frame cap
    sc.render.resolution_x = rx
    sc.render.resolution_y = ry
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA" if transparent else "RGB"
    sc.render.film_transparent = transparent
    sc.view_settings.view_transform = "Filmic" if "Filmic" in [
        t.name for t in bpy.types.ColorManagedViewSettings.bl_rna
        .properties["view_transform"].enum_items] else "Standard"

    world = bpy.data.worlds.new("W")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (*col("bg"), 1.0)
    bg.inputs[1].default_value = 0.35 if not transparent else 0.15


def setup_lights(centre: np.ndarray, scale: float) -> None:
    """Three-point rig scaled to the molecule. Big area lights because soft
    shadows are most of what makes a molecular render read as solid."""
    # Energies are per unit of (distance^2) so the rig stays correctly exposed
    # whatever the molecule's size; `scale` is the framing radius in nm.
    specs = [
        ("key",  (1.0, -1.3, 0.9),  0.85, 900.0, (1.00, 0.97, 0.92)),
        ("fill", (-1.4, -0.6, 0.2), 1.10, 260.0, (0.80, 0.88, 1.00)),
        ("rim",  (-0.3, 1.5, 0.8),  0.75, 520.0, (0.85, 0.93, 1.00)),
    ]
    for name, d, size, energy, tint in specs:
        data = bpy.data.lights.new(name, type="AREA")
        data.energy = energy * (scale ** 2)
        data.size = size * scale
        data.color = tint
        obj = bpy.data.objects.new(name, data)
        bpy.context.scene.collection.objects.link(obj)
        pos = centre + np.array(d) * scale * 1.9
        obj.location = tuple(pos)
        direction = mathutils.Vector(tuple(centre - pos))
        obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def place_camera(eye: np.ndarray, target: np.ndarray, lens: float = 52.0,
                 dof_dist: float | None = None, fstop: float = 2.4):
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = lens
    if dof_dist:
        cam_data.dof.use_dof = True
        cam_data.dof.focus_distance = dof_dist
        cam_data.dof.aperture_fstop = fstop
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = tuple(eye)
    d = mathutils.Vector(tuple(np.asarray(target) - np.asarray(eye)))
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    return cam


SENSOR_MM = 36.0          # Blender's default sensor width
ASPECT = 9.0 / 16.0


def orbit_dir(az_deg: float, el_deg: float) -> np.ndarray:
    """Unit vector from the subject towards the camera."""
    az, el = math.radians(az_deg), math.radians(el_deg)
    return np.array([math.cos(el) * math.cos(az),
                     math.cos(el) * math.sin(az),
                     math.sin(el)])


def camera_basis(d_hat: np.ndarray) -> tuple:
    """(right, up) for a camera at +d_hat looking back at the origin, matching
    Blender's -Z forward / Y up track-quat convention."""
    fwd = -d_hat
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    up /= np.linalg.norm(up)
    return right, up


def fit_distance(coords: np.ndarray, centre: np.ndarray, az_deg: float,
                 el_deg: float, lens_mm: float, margin: float = 1.10,
                 percentile: float = 97.0) -> float:
    """
    Camera distance that frames `coords` from the given orbit angle.

    Solves the real projection rather than fitting a bounding sphere. For a
    point at depth z towards the camera and lateral offset (x, y), it is
    inside the frame when the camera sits at least
    `z + max(x*f/half_w, y*f/half_h)` away; the answer is a percentile over
    the points of that per-point requirement.

    Sphere fitting -- which this replaces -- is isotropic, so an elongated
    molecule viewed side-on gets framed as if it were as tall as it is long,
    and ends up a speck in the middle of an empty frame. Full-length cIAP1 is
    four BIR domains on linkers and is very elongated indeed, so this is the
    difference between a usable opening shot and an unusable one.

    The percentile (rather than the max) lets a few floppy linker residues
    clip off the edge instead of dictating the whole shot.
    """
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    if len(coords) == 0:
        return 10.0
    d_hat = orbit_dir(az_deg, el_deg)
    right, up = camera_basis(d_hat)
    rel = coords - np.asarray(centre)

    half_w = SENSOR_MM / 2.0
    half_h = SENSOR_MM * ASPECT / 2.0
    x = np.abs(rel @ right)
    y = np.abs(rel @ up)
    z = rel @ d_hat                      # +z is towards the camera

    need = z + np.maximum(x * lens_mm / half_w, y * lens_mm / half_h)
    return float(np.percentile(need, percentile)) * margin


def bound_of(coords: np.ndarray, centre: np.ndarray,
             percentile: float = 96.0) -> float:
    """Framing radius about `centre`, ignoring the outermost few percent."""
    if len(coords) == 0:
        return 1.0
    d = np.linalg.norm(np.asarray(coords) - np.asarray(centre), axis=1)
    return float(np.percentile(d, percentile))


def orbit(centre: np.ndarray, radius: float, az_deg: float,
          el_deg: float) -> np.ndarray:
    az, el = math.radians(az_deg), math.radians(el_deg)
    return centre + radius * np.array([
        math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])


# ---------------------------------------------------------------------------
# assets
# ---------------------------------------------------------------------------

class Assets:
    """Everything prep_shrink.py wrote, loaded once."""

    def __init__(self, out_dir: str):
        def path(n):
            return os.path.join(out_dir, n)

        def opt(n):
            p = path(n)
            return S.read_pdb(p) if os.path.exists(p) else None

        def rescale(st):
            """Angstroms -> nm, once, at load."""
            if st is not None:
                st.set_coords(st.coords() * NM)
            return st

        self.full = rescale(opt("01_ciap1_full.pdb"))
        self.ref = rescale(S.read_pdb(path("02_bir3_ref.pdb")))
        self.binder = rescale(S.read_pdb(path("03_binder_final.pdb")))
        self.seed = rescale(S.read_pdb(path("04_motif_seed.pdb")))
        self.dissolve = [rescale(m) for m in S.read_pdb_models(path("05_dissolve.pdb"))]
        self.condense = [rescale(m) for m in S.read_pdb_models(path("06_condense.pdb"))]
        self.breathe = [rescale(m) for m in S.read_pdb_models(path("07_complex_breathe.pdb"))]
        with open(path("metrics.json")) as fh:
            self.metrics = json.load(fh)

        self.warhead = self.metrics.get("warhead")
        self.opening = self.full if self.full is not None else self.ref
        self.pocket = np.array([[a.x, a.y, a.z] for a in self.ref.atoms
                                if a.resname == self.warhead]).mean(0) \
            if self.warhead else self.ref.centre()

        # Everything is framed on the crystal receptor, so one centre and one
        # scale drive every camera move and every light rig.
        self.centre = self.ref.centre()
        self.scale = max(self.ref.radius_of_gyration(),
                         self.binder.radius_of_gyration()) * 2.2
        self.full_centre = (self.full.centre() if self.full is not None
                            else self.centre)

        # Coordinate sets each shot frames on. fit_distance() projects these
        # through the camera, which is what makes the choreography survive
        # swapping the 87-residue BIR3 for the 618-residue full-length model.
        self.x_open = self.opening.coords()
        self.x_ref = self.ref.coords()
        self.x_seed = self.seed.coords()
        self.x_cond0 = self.condense[0].coords()
        self.x_cond1 = self.condense[-1].coords()
        self.x_final = np.vstack([self.breathe[0].coords(), self.x_seed])
        # Shot 5 ends wide enough to hold the ghost of the original beside the
        # binder -- that side-by-side comparison is the point of the shot.
        self.x_reveal = np.vstack([self.x_final, self.x_open])


def atom_arrays(atoms, radius_scale=1.0):
    """Coordinates already arrive in nm (Assets rescales on load); van der
    Waals radii still need converting."""
    xyz = np.array([[a.x, a.y, a.z] for a in atoms], dtype=float)
    rad = np.array([VDW.get(a.element.strip().upper(), 1.6)
                    for a in atoms]) * radius_scale * NM
    return xyz, rad


def colour_protein(atoms, base, motif_rgb, hotspot_rgb, hotspot_mix=0.0):
    out = np.zeros((len(atoms), 3))
    glow = np.zeros(len(atoms))
    for i, a in enumerate(atoms):
        if a.resseq in HOTSPOTS and a.record == "ATOM":
            out[i] = lerp(np.array(motif_rgb), np.array(hotspot_rgb), hotspot_mix)
            glow[i] = hotspot_mix
        elif MOTIF_RANGE[0] <= a.resseq <= MOTIF_RANGE[1] and a.record == "ATOM":
            out[i] = motif_rgb
        else:
            out[i] = base
    return out, glow


# ---------------------------------------------------------------------------
# the shots
# ---------------------------------------------------------------------------

def build_frame(A: Assets, shot: str, u: float, quality: str) -> tuple:
    """
    Build the scene for one frame. `u` runs 0->1 within the shot.
    Returns (camera_eye, camera_target, lens, dof_distance).
    """
    ico = QUALITY[quality][4]
    meta_res = QUALITY[quality][5]  # tuple layout unchanged for indices 4,5

    m_atom = mat_attribute("atoms")
    m_e3 = mat_solid("e3", col("e3"), roughness=0.45)
    m_motif = mat_solid("motif", col("motif"), roughness=0.35)
    m_binder = mat_solid("binder", col("binder"), roughness=0.38)
    m_ghost = mat_solid("ghost", col("ghost"), roughness=0.6, alpha=0.12,
                        transmission=0.85)
    m_surf = mat_solid("surf", col("e3"), roughness=0.25, alpha=0.30,
                       transmission=0.75)

    prot = [a for a in A.ref.atoms if a.record == "ATOM"]
    lig = [a for a in A.ref.atoms if a.resname == A.warhead] if A.warhead else []
    zn = [a for a in A.ref.atoms if a.resname == "ZN"]

    # ---------------------------------------------------------- shot 1
    if shot == "establish":
        src = A.opening
        satoms = [a for a in src.atoms if a.record == "ATOM"]
        ca, _ = src.ca_trace()
        if len(ca) > 2:
            make_tube("bb", ca, ang(1.15), m_e3)
        cx, rr = atom_arrays(satoms, 1.0)
        make_metaball_surface("surf", cx, rr * 1.55, m_surf, meta_res)
        if lig:
            lx, lr = atom_arrays(lig, 0.62)
            make_spheres("lig", lx, lr,
                         np.tile(col("ligand"), (len(lx), 1)),
                         np.full(len(lx), 0.18), ico, m_atom)
        if zn:
            zx, zr = atom_arrays(zn, 0.95)
            make_spheres("zn", zx, zr, np.tile(col("zinc"), (len(zx), 1)),
                         None, ico + 1, m_atom)

        lens, el = 48.0, 14.0
        az = lerp(-35.0, 55.0, ease(u, 2.0))
        d = fit_distance(A.x_open, A.full_centre, az, el, lens, 1.12)
        eye = orbit(A.full_centre, d, az, el)
        return eye, A.full_centre, lens, d

    # ---------------------------------------------------------- shot 2
    if shot == "pocket":
        ignite = smoothstep((u - 0.35) / 0.5)
        ca, _ = A.ref.ca_trace()
        make_tube("bb", ca, ang(1.10), m_e3)
        cx, rr = atom_arrays(prot, 1.0)
        cols, glow = colour_protein(prot, col("e3dark"), col("motif"),
                                    col("hotspot"), ignite)
        keep = np.array([MOTIF_RANGE[0] <= a.resseq <= MOTIF_RANGE[1]
                         for a in prot])
        if keep.any():
            make_spheres("motifatoms", cx[keep], rr[keep] * 0.85,
                         cols[keep], glow[keep] * 1.6, ico, m_atom)
        make_metaball_surface("surf", cx, rr * 1.50, m_surf, meta_res * 1.15)
        if lig:
            lx, lr = atom_arrays(lig, 0.72)
            make_spheres("lig", lx, lr, np.tile(col("ligand"), (len(lx), 1)),
                         np.full(len(lx), 0.20 + 0.55 * ignite), ico + 1, m_atom)
        if zn:
            zx, zr = atom_arrays(zn, 1.0)
            make_spheres("zn", zx, zr, np.tile(col("zinc"), (len(zx), 1)),
                         None, ico + 1, m_atom)

        t = ease(u, 2.2)
        lens = lerp(48.0, 62.0, t)
        az, el = lerp(55.0, 88.0, t), lerp(14.0, 6.0, t)
        # push from "the whole domain" to "the pocket and its shoulders"
        d = lerp(fit_distance(A.x_open, A.full_centre, az, el, lens, 1.12),
                 fit_distance(A.x_seed, A.pocket, az, el, lens, 1.05), t)
        centre = A.full_centre + (A.pocket - A.full_centre) * t
        eye = orbit(centre, d, az, el)
        return eye, centre, lens, d

    # ---------------------------------------------------------- shot 3
    if shot == "dissolve":
        idx = min(int(u * (len(A.dissolve) - 1) + 0.5), len(A.dissolve) - 1)
        frame = A.dissolve[idx]
        fa = frame.atoms
        held = np.array([(a.record == "HETATM")
                         or (MOTIF_RANGE[0] <= a.resseq <= MOTIF_RANGE[1])
                         for a in fa])
        cx, rr = atom_arrays(fa, 1.0)

        # bulk flies apart as beads that shrink and dim as they go
        bulk = ~held & np.array([a.record == "ATOM" for a in fa])
        if bulk.any():
            d = np.linalg.norm(cx[bulk] - A.pocket, axis=1)
            fade = np.clip(1.0 - (d - A.scale * 0.5) / (A.scale * 1.6), 0.06, 1.0)
            rng = np.random.default_rng(4)
            cols = np.array([shade(col("e3"), 0.75 + 0.5 * v)
                             for v in rng.random(int(bulk.sum()))])
            make_spheres("bulk", cx[bulk], rr[bulk] * 0.52 * fade,
                         cols * fade[:, None], None, max(ico - 1, 0), m_atom)

        keep = held & np.array([a.record == "ATOM" for a in fa])
        if keep.any():
            ca = np.array([[a.x, a.y, a.z] for a, k in zip(fa, keep)
                           if k and a.name.strip() == "CA"])
            if len(ca) > 2:
                make_tube("motifbb", ca, ang(1.25), m_motif)
            make_spheres("motif", cx[keep], rr[keep] * 0.80,
                         np.tile(col("motif"), (int(keep.sum()), 1)),
                         np.full(int(keep.sum()), 0.35), ico, m_atom)
        het = np.array([a.record == "HETATM" for a in fa])
        if het.any():
            hc = np.array([col("zinc") if a.resname == "ZN" else col("ligand")
                           for a, h in zip(fa, het) if h])
            make_spheres("het", cx[het], rr[het] * 0.75, hc,
                         np.full(int(het.sum()), 0.55), ico + 1, m_atom)

        lens = 58.0
        az, el = lerp(88.0, 118.0, u), lerp(6.0, 18.0, u)
        # Hold wide enough to watch the bulk leave, then settle on what is left.
        d = lerp(fit_distance(A.x_ref, A.pocket, az, el, lens, 1.30),
                 fit_distance(A.x_seed, A.pocket, az, el, lens, 1.60),
                 ease(u, 2.0))
        eye = orbit(A.pocket, d, az, el)
        return eye, A.pocket, lens, d

    # ---------------------------------------------------------- shot 4
    if shot == "condense":
        idx = min(int(u * (len(A.condense) - 1) + 0.5), len(A.condense) - 1)
        frame = A.condense[idx]
        cx = frame.coords()
        n = len(cx)
        # Beads, not ribbon: mid-diffusion there is no backbone geometry to
        # assign secondary structure from, and a cartoon degenerates into
        # spikes. Beads are honest at every timestep.
        order = np.linspace(0.0, 1.0, n)
        base = np.array(col("binder"))
        hi = np.array(col("binderhi"))
        cols = base[None, :] * (1 - order[:, None]) + hi[None, :] * order[:, None]
        settle = smoothstep((u - 0.25) / 0.75)
        rr = np.full(n, lerp(ang(2.1), ang(1.5), settle))
        glow = np.full(n, lerp(0.55, 0.06, settle))
        make_spheres("cond", cx, rr, cols, glow, ico, m_atom)
        if settle > 0.45 and n > 3:
            make_tube("condbb", cx, ang(1.05) * (settle - 0.45) / 0.55, m_binder)

        sx, sr = atom_arrays([a for a in A.seed.atoms if a.record == "ATOM"], 0.80)
        if len(sx):
            make_spheres("seed", sx, sr,
                         np.tile(col("motif"), (len(sx), 1)),
                         np.full(len(sx), 0.30), ico, m_atom)
        seed_het = [a for a in A.seed.atoms if a.record == "HETATM"]
        if seed_het:
            hx, hr = atom_arrays(seed_het, 0.75)
            hc = np.array([col("zinc") if a.resname == "ZN" else col("ligand")
                           for a in seed_het])
            make_spheres("seedhet", hx, hr, hc, np.full(len(hx), 0.55),
                         ico + 1, m_atom)

        lens = 56.0
        az, el = lerp(118.0, 168.0, u), lerp(18.0, 10.0, u)
        # Framed off the cloud's own extent at both ends, so the camera
        # closes in exactly as fast as the binder condenses.
        d = lerp(fit_distance(A.x_cond0, A.pocket, az, el, lens, 1.02),
                 fit_distance(A.x_cond1, A.pocket, az, el, lens, 1.14),
                 smoothstep(u))
        eye = orbit(A.pocket, d, az, el)
        return eye, A.pocket, lens, d

    # ---------------------------------------------------------- shot 5
    if shot == "reveal":
        idx = int(u * (len(A.breathe) - 1) * 1.0) % max(len(A.breathe), 1)
        frame = A.breathe[idx]
        fa = frame.atoms
        cx, rr = atom_arrays(fa, 1.0)
        binder_mask = np.array([a.chain == "B" for a in fa])
        motif_mask = np.array([a.record == "ATOM" and a.chain != "B" for a in fa])
        het_mask = np.array([a.record == "HETATM" for a in fa])

        if binder_mask.any():
            ca = np.array([[a.x, a.y, a.z] for a, m in zip(fa, binder_mask)
                           if m and a.name.strip() == "CA"])
            if len(ca) > 2:
                make_tube("binderbb", ca, ang(1.25), m_binder)
            make_metaball_surface("bsurf", cx[binder_mask],
                                  rr[binder_mask] * 1.50,
                                  mat_solid("bs", col("binder"), 0.3, 0.22, 0.0, 0.8),
                                  meta_res)
        if motif_mask.any():
            ca = np.array([[a.x, a.y, a.z] for a, m in zip(fa, motif_mask)
                           if m and a.name.strip() == "CA"])
            if len(ca) > 2:
                make_tube("motifbb", ca, ang(1.35), m_motif)
            make_spheres("motif", cx[motif_mask], rr[motif_mask] * 0.80,
                         np.tile(col("motif"), (int(motif_mask.sum()), 1)),
                         np.full(int(motif_mask.sum()), 0.25), ico, m_atom)
        if het_mask.any():
            hc = np.array([col("zinc") if a.resname == "ZN" else col("ligand")
                           for a, m in zip(fa, het_mask) if m])
            make_spheres("het", cx[het_mask], rr[het_mask] * 0.78, hc,
                         np.full(int(het_mask.sum()), 0.5), ico + 1, m_atom)

        # ghost of what we started from, fading up behind
        gu = smoothstep((u - 0.25) / 0.6)
        if gu > 0.02:
            gca, _ = A.opening.ca_trace()
            if len(gca) > 2:
                make_tube("ghost", gca, ang(0.85) * gu, m_ghost)

        lens = lerp(56.0, 46.0, u)
        az, el = lerp(168.0, 205.0, u), lerp(10.0, 20.0, u)
        # Pull back far enough that the ghost of the original fits too.
        d = lerp(fit_distance(A.x_final, A.pocket, az, el, lens, 1.14),
                 fit_distance(A.x_reveal, A.pocket, az, el, lens, 1.10),
                 ease(u, 2.0))
        eye = orbit(A.pocket, d, az, el)
        return eye, A.pocket, lens, d

    raise ValueError(f"unknown shot {shot}")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def build_shotlist(fps: int, assets: Assets) -> list[tuple]:
    """Shot lengths follow the trajectory lengths where one exists, so the
    playback never resamples and never stalls on a held frame."""
    return [
        ("establish", int(2.0 * fps)),
        ("pocket",    int(1.5 * fps)),
        ("dissolve",  len(assets.dissolve)),
        ("condense",  len(assets.condense)),
        ("reveal",    len(assets.breathe)),
    ]


def main() -> int:
    p = argparse.ArgumentParser(description="Path-trace the E3-shrink shot.")
    p.add_argument("--out-dir", default="output")
    p.add_argument("--frames-dir", default="output/frames")
    p.add_argument("--movie", default="output/e3_shrink_blender.mp4")
    p.add_argument("--quality", default="preview", choices=sorted(QUALITY))
    p.add_argument("--device", default="cpu",
                   choices=["cpu", "auto", "metal", "cuda", "optix", "hip", "oneapi"],
                   help="render device. 'metal' on Apple Silicon, 'cuda'/'optix' "
                        "on NVIDIA; 'auto' picks the first GPU backend present. "
                        "Default cpu.")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument("--transparent", action="store_true",
                   help="alpha background; writes PNGs only, mp4 cannot carry alpha")
    p.add_argument("--no-encode", action="store_true")
    p.add_argument("--list-shots", action="store_true")
    args = p.parse_args()

    A = Assets(args.out_dir)
    shots = build_shotlist(args.fps, A)
    total = sum(n for _, n in shots)

    print(f"\n[shots]  {total} frames @ {args.fps} fps = {total/args.fps:.1f} s")
    f0 = 0
    for name, n in shots:
        print(f"  {name:10s} {n:4d} frames  {f0:4d}-{f0+n-1:4d}  {n/args.fps:.2f}s")
        f0 += n
    if args.list_shots:
        return 0

    rx, ry, samples = QUALITY[args.quality][:3]
    print(f"[quality] {args.quality}: {rx}x{ry}, {samples} samples; device request={args.device}")

    os.makedirs(args.frames_dir, exist_ok=True)
    end = total - 1 if args.end < 0 else min(args.end, total - 1)

    # index -> (shot, u)
    timeline = []
    for name, n in shots:
        for i in range(n):
            timeline.append((name, i / max(n - 1, 1)))

    import time
    t_start = time.time()
    done = 0
    for gi in range(args.start, end + 1):
        shot, u = timeline[gi]
        bpy.ops.wm.read_factory_settings(use_empty=True)
        purge()
        setup_world(args.quality, args.transparent, args.device)
        eye, target, lens, dof = build_frame(A, shot, u, args.quality)
        setup_lights(np.asarray(target), float(A.scale))
        place_camera(np.asarray(eye), np.asarray(target), lens, dof, 3.2)

        if done == 0:
            print(f"[device] Cycles backend engaged: "
                  f"{getattr(setup_world, '_backend', 'CPU')}")
        out = os.path.join(args.frames_dir, f"f{gi:05d}.png")
        bpy.context.scene.render.filepath = out
        bpy.ops.render.render(write_still=True)
        done += 1
        el = time.time() - t_start
        rate = el / done
        left = (end - gi) * rate
        print(f"  [{gi+1:4d}/{end+1}] {shot:10s} u={u:4.2f}  "
              f"{rate:5.1f}s/frame  eta {left/60:5.1f} min", flush=True)

    if args.no_encode or args.transparent:
        print(f"\nframes in {args.frames_dir}")
        return 0

    encode(args.frames_dir, args.movie, args.fps)
    return 0


def encode(frames_dir: str, movie: str, fps: int) -> None:
    exe = shutil.which("ffmpeg")
    if exe is None:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            print("no ffmpeg; frames left in", frames_dir)
            return
    os.makedirs(os.path.dirname(os.path.abspath(movie)) or ".", exist_ok=True)
    cmd = [exe, "-y", "-framerate", str(fps),
           "-pattern_type", "glob", "-i", os.path.join(frames_dir, "f*.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17",
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", movie]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        print("encode failed; frames are still in", frames_dir)
    else:
        size = os.path.getsize(movie) / 1e6
        print(f"\nwrote {movie} ({size:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
