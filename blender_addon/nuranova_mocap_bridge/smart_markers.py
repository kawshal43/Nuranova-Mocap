"""Smart Landmark Marker System for Nuranova Character Auto-Rigging in Blender 4.5+ / 5.x.

Lets the user place guided joint markers directly onto ANY character mesh (realistic, stylized,
anime, cartoon, A-pose, or T-pose) via interactive click-to-place with automatic through-mesh
depth centering + X-symmetry mirroring, or draggable preset markers, plus optional finger markers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import bpy
from mathutils import Vector

MARKER_COLLECTION_NAME = "Nuranova_Rig_Markers"
MARKER_PREFIX = "NRV_MK_"

FINGER_NAMES: List[str] = ["thumb", "index", "middle", "ring", "pinky"]


@dataclass(frozen=True)
class MarkerSpec:
    key: str
    label: str
    hint: str
    symmetric: bool  # True -> has _L and _R pair; False -> single center marker
    # Default normalized position in mesh bbox: (x_from_center_in_half_span, y_offset, z_frac_0_to_1)
    default_nx: float
    default_ny: float
    default_nz: float
    color_rgba: Tuple[float, float, float, float]


# 12 intuitive landmarks -> fully specifies all 25 body bones + 30 proportional finger bones!
BODY_MARKER_SPECS: List[MarkerSpec] = [
    MarkerSpec(
        "chin",
        "Chin / Neck Base",
        "Click right under the chin where the neck meets the head",
        False,
        0.0, 0.0, 0.78,
        (0.20, 0.95, 0.90, 1.0),
    ),
    MarkerSpec(
        "head_top",
        "Top of Head",
        "Click the very top (crown) of the head",
        False,
        0.0, 0.0, 0.99,
        (0.20, 0.95, 0.90, 1.0),
    ),
    MarkerSpec(
        "chest",
        "Chest / Sternum",
        "Click the center of the upper chest (between shoulders & ribcage)",
        False,
        0.0, 0.0, 0.68,
        (0.25, 0.85, 1.00, 1.0),
    ),
    MarkerSpec(
        "pelvis",
        "Pelvis / Hip Center",
        "Click the center of the pelvis / groin",
        False,
        0.0, 0.0, 0.50,
        (1.00, 0.82, 0.20, 1.0),
    ),
    MarkerSpec(
        "shoulder",
        "Shoulder Joint",
        "Click ONE shoulder joint (opposite side auto-mirrors when Mirror is ON)",
        True,
        0.22, 0.0, 0.74,
        (0.25, 0.95, 0.45, 1.0),
    ),
    MarkerSpec(
        "elbow",
        "Elbow Joint",
        "Click the SAME side's elbow joint",
        True,
        0.52, 0.02, 0.73,
        (0.25, 0.95, 0.45, 1.0),
    ),
    MarkerSpec(
        "wrist",
        "Wrist",
        "Click the SAME side's wrist",
        True,
        0.80, 0.0, 0.73,
        (0.25, 0.95, 0.45, 1.0),
    ),
    MarkerSpec(
        "hand_tip",
        "Middle Finger Tip",
        "Click the tip of the middle finger on that hand",
        True,
        0.96, 0.0, 0.73,
        (0.45, 1.00, 0.65, 1.0),
    ),
    MarkerSpec(
        "hip",
        "Upper Thigh / Hip Joint",
        "Click where ONE thigh meets the pelvis (hip socket)",
        True,
        0.13, 0.0, 0.47,
        (1.00, 0.58, 0.20, 1.0),
    ),
    MarkerSpec(
        "knee",
        "Knee",
        "Click the SAME side's knee joint",
        True,
        0.13, -0.02, 0.26,
        (1.00, 0.58, 0.20, 1.0),
    ),
    MarkerSpec(
        "ankle",
        "Ankle",
        "Click the SAME side's ankle",
        True,
        0.13, 0.02, 0.05,
        (1.00, 0.58, 0.20, 1.0),
    ),
    MarkerSpec(
        "toe",
        "Toe Tip",
        "Click the front tip of the toes on that foot (also detects front facing!)",
        True,
        0.13, -0.14, 0.015,
        (1.00, 0.42, 0.42, 1.0),
    ),
]

SPEC_BY_KEY: Dict[str, MarkerSpec] = {s.key: s for s in BODY_MARKER_SPECS}

# Preview skeleton edges drawn in the 3D viewport between placed markers
PREVIEW_MARKER_EDGES: List[Tuple[str, str]] = [
    ("pelvis", "chest"),
    ("chest", "chin"),
    ("chin", "head_top"),
    ("chest", "shoulder_L"),
    ("shoulder_L", "elbow_L"),
    ("elbow_L", "wrist_L"),
    ("wrist_L", "hand_tip_L"),
    ("chest", "shoulder_R"),
    ("shoulder_R", "elbow_R"),
    ("elbow_R", "wrist_R"),
    ("wrist_R", "hand_tip_R"),
    ("pelvis", "hip_L"),
    ("hip_L", "knee_L"),
    ("knee_L", "ankle_L"),
    ("ankle_L", "toe_L"),
    ("pelvis", "hip_R"),
    ("hip_R", "knee_R"),
    ("knee_R", "ankle_R"),
    ("ankle_R", "toe_R"),
]


def marker_obj_name(key: str, side_suffix: str = "") -> str:
    if side_suffix:
        return f"{MARKER_PREFIX}{key}_{side_suffix}"
    return f"{MARKER_PREFIX}{key}"


def get_or_create_marker_collection(context: Optional[bpy.types.Context] = None) -> bpy.types.Collection:
    ctx = context or bpy.context
    col = bpy.data.collections.get(MARKER_COLLECTION_NAME)
    if col is None:
        col = bpy.data.collections.new(MARKER_COLLECTION_NAME)
        ctx.scene.collection.children.link(col)
    elif col.name not in ctx.scene.collection.children:
        try:
            ctx.scene.collection.children.link(col)
        except RuntimeError:
            pass
    return col


def compute_meshes_world_bbox(mesh_objs: Sequence[bpy.types.Object]) -> Tuple[Vector, Vector]:
    mins = Vector((1e9, 1e9, 1e9))
    maxs = Vector((-1e9, -1e9, -1e9))
    for obj in mesh_objs:
        if obj is None or obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            wc = obj.matrix_world @ Vector(corner)
            mins.x = min(mins.x, wc.x)
            mins.y = min(mins.y, wc.y)
            mins.z = min(mins.z, wc.z)
            maxs.x = max(maxs.x, wc.x)
            maxs.y = max(maxs.y, wc.y)
            maxs.z = max(maxs.z, wc.z)
    if mins.x > maxs.x:
        return Vector((-0.75, -0.20, 0.0)), Vector((0.75, 0.20, 1.72))
    return mins, maxs


def infer_facing_sign_y(
    markers: Dict[str, Vector],
    override: str = "AUTO",
) -> float:
    """Return -1.0 if character faces -Y (standard Blender Front View), or +1.0 if character faces +Y."""
    mode = (override or "AUTO").upper()
    if mode == "NEG_Y":
        return -1.0
    if mode == "POS_Y":
        return 1.0

    # 1. Most reliable cue: toes are in front of ankles!
    dy_samples: List[float] = []
    for suf in ("L", "R"):
        toe = markers.get(f"toe_{suf}")
        ankle = markers.get(f"ankle_{suf}")
        if toe is not None and ankle is not None:
            dy = float(toe.y - ankle.y)
            if abs(dy) > 0.012:
                dy_samples.append(dy)
    if dy_samples:
        avg_dy = sum(dy_samples) / len(dy_samples)
        return -1.0 if avg_dy < 0.0 else 1.0

    # 2. Secondary cue: knees sit slightly forward of the hip-ankle line
    knee_dy: List[float] = []
    for suf in ("L", "R"):
        hip = markers.get(f"hip_{suf}")
        knee = markers.get(f"knee_{suf}")
        ankle = markers.get(f"ankle_{suf}")
        if hip is not None and knee is not None and ankle is not None:
            mid_y = 0.5 * (hip.y + ankle.y)
            dy = float(knee.y - mid_y)
            if abs(dy) > 0.008:
                knee_dy.append(dy)
    if knee_dy:
        return -1.0 if (sum(knee_dy) / len(knee_dy)) < 0.0 else 1.0

    # Standard Blender Front View convention: characters face -Y
    return -1.0


def left_side_x_sign(facing_sign_y: float) -> float:
    """Return +1.0 when Character-Left is at +X (facing -Y), or -1.0 when Character-Left is at -X (facing +Y)."""
    return 1.0 if facing_sign_y < 0.0 else -1.0


def place_or_update_marker(
    full_key: str,
    world_pos: Vector,
    radius: float = 0.025,
    color_rgba: Tuple[float, float, float, float] = (0.2, 0.9, 1.0, 1.0),
    context: Optional[bpy.types.Context] = None,
) -> bpy.types.Object:
    """Create or reposition a visible always-in-front marker Empty inside `Nuranova_Rig_Markers`."""
    ctx = context or bpy.context
    col = get_or_create_marker_collection(ctx)
    obj_name = f"{MARKER_PREFIX}{full_key}"
    obj = bpy.data.objects.get(obj_name)
    if obj is None:
        obj = bpy.data.objects.new(obj_name, None)
        obj.empty_display_type = "SPHERE"
        col.objects.link(obj)
    elif obj.name not in col.objects:
        col.objects.link(obj)

    obj.empty_display_size = max(0.006, float(radius))
    obj.location = Vector(world_pos)
    obj.show_in_front = True
    obj.show_name = True
    obj.color = color_rgba
    obj.hide_viewport = False
    obj.hide_set(False)
    obj["nuranova_marker_key"] = full_key
    return obj


def raycast_mesh_center(
    context: bpy.types.Context,
    mesh_objs: Sequence[bpy.types.Object],
    ray_origin: Vector,
    ray_dir: Vector,
    fallback_plane_point: Optional[Vector] = None,
    max_thickness_m: float = 0.55,
) -> Tuple[Vector, bool]:
    """Cast a 3D ray through `mesh_objs` and return the internal midpoint (front + back) / 2.

    This is the secret to Mixamo / Auto-Rig Pro Smart precision: clicking on the front of a
    shoulder, elbow, wrist, knee, or chin automatically places the joint inside the center
    of the limb/torso rather than stuck on the outer skin surface!

    Returns (world_position, hit_mesh_bool).
    """
    ray_dir = ray_dir.normalized()
    depsgraph = context.evaluated_depsgraph_get()

    best_front_dist = 1e18
    best_front_pos: Optional[Vector] = None
    best_obj: Optional[bpy.types.Object] = None

    valid_evals: List[Tuple[bpy.types.Object, bpy.types.Object]] = []
    for obj in mesh_objs:
        if obj is None or obj.type != "MESH" or obj.hide_viewport:
            continue
        eval_obj = obj.evaluated_get(depsgraph)
        valid_evals.append((obj, eval_obj))
        try:
            mat_inv = obj.matrix_world.inverted()
            l_orig = mat_inv @ ray_origin
            l_dir = (mat_inv.to_3x3() @ ray_dir).normalized()
            hit, loc, _norm, _idx = eval_obj.ray_cast(l_orig, l_dir)
            if hit:
                w_hit = obj.matrix_world @ loc
                dist = float((w_hit - ray_origin).dot(ray_dir))
                if dist > 0.0 and dist < best_front_dist:
                    best_front_dist = dist
                    best_front_pos = w_hit
                    best_obj = eval_obj
        except Exception:
            continue

    if best_front_pos is not None:
        # Shoot a second ray from just past the front hit to locate the back wall of the limb/torso
        eps = 0.004
        back_origin = best_front_pos + ray_dir * eps
        best_back_dist = 1e18
        best_back_pos: Optional[Vector] = None
        for orig_obj, eval_obj in valid_evals:
            # Prefer exiting the same mesh first, but allow any part of the character
            try:
                mat_inv = orig_obj.matrix_world.inverted()
                l_orig = mat_inv @ back_origin
                l_dir = (mat_inv.to_3x3() @ ray_dir).normalized()
                hit, loc, _norm, _idx = eval_obj.ray_cast(l_orig, l_dir)
                if hit:
                    w_back = orig_obj.matrix_world @ loc
                    thickness = float((w_back - best_front_pos).dot(ray_dir))
                    if 0.006 <= thickness <= max_thickness_m:
                        # Prefer exit hit on the same object if distances are similar
                        bonus = 0.0 if eval_obj == best_obj else 0.005
                        if thickness + bonus < best_back_dist:
                            best_back_dist = thickness + bonus
                            best_back_pos = w_back
            except Exception:
                continue

        if best_back_pos is not None:
            return 0.5 * (best_front_pos + best_back_pos), True
        return best_front_pos, True

    # Fallback: intersect ray with plane through character center facing the camera ray
    plane_pt = fallback_plane_point if fallback_plane_point is not None else Vector((0.0, 0.0, 0.9))
    denom = float(ray_dir.dot(ray_dir))
    if denom > 1e-8:
        t = float((plane_pt - ray_origin).dot(ray_dir)) / denom
        if t > 0.0:
            return ray_origin + ray_dir * t, False
    return plane_pt.copy(), False


def get_symmetry_center_x(
    mesh_objs: Optional[Sequence[bpy.types.Object]] = None,
    existing_markers: Optional[Dict[str, Vector]] = None,
) -> float:
    """Find the character's horizontal X symmetry center from center markers or mesh bounding box."""
    if existing_markers:
        center_xs = [
            float(existing_markers[k].x)
            for k in ("pelvis", "chest", "chin", "head_top")
            if k in existing_markers
        ]
        if center_xs:
            return sum(center_xs) / len(center_xs)
    if mesh_objs:
        bmin, bmax = compute_meshes_world_bbox(mesh_objs)
        return 0.5 * float(bmin.x + bmax.x)
    return 0.0


def place_body_landmark(
    spec_key: str,
    world_pos: Vector,
    center_x: float,
    char_height: float = 1.72,
    mirror: bool = True,
    facing_override: str = "AUTO",
    explicit_side: Optional[str] = None,
    context: Optional[bpy.types.Context] = None,
) -> List[bpy.types.Object]:
    """Place a center or symmetric (L/R) body landmark marker, auto-mirroring across `center_x` when enabled."""
    spec = SPEC_BY_KEY[spec_key]
    radius = max(0.010, min(0.045, char_height * 0.016))
    existing = collect_marker_positions()
    facing_y = infer_facing_sign_y(existing, override=facing_override)
    l_sign = left_side_x_sign(facing_y)

    if not spec.symmetric:
        pos = Vector(world_pos)
        if mirror:
            # Keep spine/head center markers neatly centered on symmetry axis if clicked close to center
            if abs(pos.x - center_x) < max(0.08, char_height * 0.06):
                pos.x = center_x
        return [place_or_update_marker(spec.key, pos, radius=radius, color_rgba=spec.color_rgba, context=context)]

    # Determine whether the clicked point is Character-Left ("L") or Character-Right ("R")
    if explicit_side in ("L", "R"):
        side = explicit_side
    else:
        dx = float(world_pos.x - center_x)
        side = "L" if (dx * l_sign >= 0.0) else "R"

    opp_side = "R" if side == "L" else "L"
    pos_primary = Vector(world_pos)
    objs = [
        place_or_update_marker(
            f"{spec.key}_{side}",
            pos_primary,
            radius=radius,
            color_rgba=spec.color_rgba,
            context=context,
        )
    ]

    if mirror:
        pos_mirror = Vector((2.0 * center_x - pos_primary.x, pos_primary.y, pos_primary.z))
        objs.append(
            place_or_update_marker(
                f"{spec.key}_{opp_side}",
                pos_mirror,
                radius=radius,
                color_rgba=spec.color_rgba,
                context=context,
            )
        )
    return objs


def spawn_preset_markers(
    mesh_objs: Sequence[bpy.types.Object],
    facing_override: str = "AUTO",
    context: Optional[bpy.types.Context] = None,
) -> Dict[str, bpy.types.Object]:
    """Spawn all 20 draggable body landmark markers pre-fitted to the character's world bounding box."""
    ctx = context or bpy.context
    bmin, bmax = compute_meshes_world_bbox(mesh_objs)
    height = max(0.25, float(bmax.z - bmin.z))
    half_x = max(0.15, 0.5 * float(bmax.x - bmin.x))
    cx = 0.5 * float(bmin.x + bmax.x)
    cy = 0.5 * float(bmin.y + bmax.y)
    facing_y = -1.0 if facing_override.upper() != "POS_Y" else 1.0
    l_sign = left_side_x_sign(facing_y)
    radius = max(0.010, min(0.045, height * 0.016))

    created: Dict[str, bpy.types.Object] = {}
    for spec in BODY_MARKER_SPECS:
        z = bmin.z + spec.default_nz * height
        # default_ny is expressed assuming -Y facing; flip if facing +Y
        y = cy + spec.default_ny * height * (-facing_y)
        if not spec.symmetric:
            obj = place_or_update_marker(
                spec.key,
                Vector((cx, y, z)),
                radius=radius,
                color_rgba=spec.color_rgba,
                context=ctx,
            )
            created[spec.key] = obj
        else:
            dx = spec.default_nx * half_x
            obj_l = place_or_update_marker(
                f"{spec.key}_L",
                Vector((cx + l_sign * dx, y, z)),
                radius=radius,
                color_rgba=spec.color_rgba,
                context=ctx,
            )
            obj_r = place_or_update_marker(
                f"{spec.key}_R",
                Vector((cx - l_sign * dx, y, z)),
                radius=radius,
                color_rgba=spec.color_rgba,
                context=ctx,
            )
            created[f"{spec.key}_L"] = obj_l
            created[f"{spec.key}_R"] = obj_r
    return created


def spawn_finger_markers(
    facing_override: str = "AUTO",
    context: Optional[bpy.types.Context] = None,
) -> Dict[str, bpy.types.Object]:
    """Spawn 20 optional draggable finger knuckle/tip markers (10 per hand) aligned to Wrist -> HandTip."""
    ctx = context or bpy.context
    markers = collect_marker_positions()
    facing_y = infer_facing_sign_y(markers, override=facing_override)
    created: Dict[str, bpy.types.Object] = {}

    # Relative offsets in units of hand_len: (out_base, fwd_base, out_tip, fwd_tip, z_off)
    finger_fracs = [
        ("thumb", 0.22, 0.24, 0.62, 0.42, -0.04),
        ("index", 0.48, 0.16, 0.96, 0.18, 0.01),
        ("middle", 0.50, 0.02, 1.00, 0.02, 0.015),
        ("ring", 0.48, -0.10, 0.94, -0.11, 0.01),
        ("pinky", 0.44, -0.22, 0.82, -0.24, -0.005),
    ]

    for suf in ("L", "R"):
        wrist = markers.get(f"wrist_{suf}")
        hand_tip = markers.get(f"hand_tip_{suf}")
        if wrist is None or hand_tip is None:
            continue
        vec = hand_tip - wrist
        hand_len = max(0.05, float(vec.length))
        out_dir = vec.normalized() if vec.length > 1e-4 else Vector((1.0 if suf == "L" else -1.0, 0.0, 0.0))

        # Forward direction points along character facing (+Y or -Y), orthogonalized against hand direction
        raw_fwd = Vector((0.0, facing_y, 0.0))
        fwd_dir = raw_fwd - out_dir * raw_fwd.dot(out_dir)
        if fwd_dir.length < 1e-4:
            fwd_dir = Vector((0.0, 0.0, 1.0)).cross(out_dir)
        fwd_dir.normalize()
        up_dir = out_dir.cross(fwd_dir)
        if up_dir.z < 0.0:
            up_dir = -up_dir
        up_dir.normalize()

        r = max(0.004, min(0.012, hand_len * 0.045))
        for fname, ob, fb, ot, ft, zo in finger_fracs:
            p_base = wrist + out_dir * (ob * hand_len) + fwd_dir * (fb * hand_len) + up_dir * (zo * hand_len)
            p_tip = wrist + out_dir * (ot * hand_len) + fwd_dir * (ft * hand_len) + up_dir * (zo * hand_len)
            k_b = f"f_{fname}_base_{suf}"
            k_t = f"f_{fname}_tip_{suf}"
            created[k_b] = place_or_update_marker(k_b, p_base, radius=r, color_rgba=(0.95, 0.45, 1.0, 1.0), context=ctx)
            created[k_t] = place_or_update_marker(k_t, p_tip, radius=r * 0.85, color_rgba=(1.0, 0.70, 1.0, 1.0), context=ctx)
    return created


def collect_marker_positions() -> Dict[str, Vector]:
    """Return a dict mapping marker_key (e.g. 'chin', 'shoulder_L', 'f_index_tip_L') -> world Vector."""
    positions: Dict[str, Vector] = {}
    col = bpy.data.collections.get(MARKER_COLLECTION_NAME)
    objs = list(col.objects) if col is not None else [o for o in bpy.data.objects if o.name.startswith(MARKER_PREFIX)]
    for obj in objs:
        key = obj.get("nuranova_marker_key")
        if not key and obj.name.startswith(MARKER_PREFIX):
            key = obj.name[len(MARKER_PREFIX):]
        if key:
            positions[str(key)] = obj.matrix_world.translation.copy()
    return positions


def count_placed_body_landmarks() -> Tuple[int, int]:
    """Return (placed_steps_count, total_steps_count=12)."""
    pos = collect_marker_positions()
    placed = 0
    for spec in BODY_MARKER_SPECS:
        if not spec.symmetric:
            if spec.key in pos:
                placed += 1
        else:
            if f"{spec.key}_L" in pos and f"{spec.key}_R" in pos:
                placed += 1
    return placed, len(BODY_MARKER_SPECS)


def mirror_markers_lr(
    mesh_objs: Optional[Sequence[bpy.types.Object]] = None,
    source_side: str = "AUTO",
    context: Optional[bpy.types.Context] = None,
) -> int:
    """Mirror all symmetric L/R body & finger markers across the character's X center."""
    ctx = context or bpy.context
    markers = collect_marker_positions()
    center_x = get_symmetry_center_x(mesh_objs=mesh_objs, existing_markers=markers)

    # If user currently selected a _R marker, mirror R -> L; otherwise L -> R
    side = source_side.upper()
    if side == "AUTO":
        active = ctx.active_object
        if active is not None and active.name.startswith(MARKER_PREFIX) and active.name.endswith("_R"):
            side = "R"
        else:
            side = "L"
    dst_side = "R" if side == "L" else "L"

    mirrored = 0
    for key, pos in list(markers.items()):
        if not key.endswith(f"_{side}"):
            continue
        base_key = key[:-2]
        dst_key = f"{base_key}_{dst_side}"
        src_obj = bpy.data.objects.get(f"{MARKER_PREFIX}{key}")
        radius = src_obj.empty_display_size if src_obj else 0.02
        col_rgba = tuple(src_obj.color) if src_obj else (0.2, 0.9, 1.0, 1.0)
        mirrored_pos = Vector((2.0 * center_x - pos.x, pos.y, pos.z))
        place_or_update_marker(dst_key, mirrored_pos, radius=radius, color_rgba=col_rgba, context=ctx)
        mirrored += 1
    return mirrored


def set_markers_visibility(visible: bool) -> bool:
    col = bpy.data.collections.get(MARKER_COLLECTION_NAME)
    if col is None:
        return False
    for obj in col.objects:
        obj.hide_viewport = not visible
        obj.hide_set(not visible)
    return True


def clear_markers() -> int:
    """Delete all Nuranova Smart Rig Markers and their helper collection."""
    col = bpy.data.collections.get(MARKER_COLLECTION_NAME)
    removed = 0
    if col is not None:
        for obj in list(col.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
        bpy.data.collections.remove(col)
    for obj in list(bpy.data.objects):
        if obj.name.startswith(MARKER_PREFIX):
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
    return removed
