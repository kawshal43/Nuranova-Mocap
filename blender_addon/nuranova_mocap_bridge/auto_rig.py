"""Nuranova Auto-Rig & Humanoid Rig Generator (25 Body + 30 Articulated Finger Bones) for Blender 4.5+ / 5.x.

Provides:
- Smart Marker-Driven Auto-Rigging (`create_rig_from_markers` / `auto_rig_from_markers`):
  builds an anatomically accurate 55-bone rig directly from user-placed landmark markers
  (works on stylized, anime, chibi, cartoon, realistic, A-pose, or T-pose characters of ANY
  proportion, automatically handling standard Blender -Y facing and +Y facing!).
- 1-Click Bounding-Box Auto-Rig fallback (`auto_rig_character_mesh`) and standalone humanoid rig
  generator (`create_nuranova_humanoid_rig`).
- Automatic Weights binding with clean Envelope / Empty Groups fallbacks and iterative re-binding
  without duplicating armature modifiers.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import bpy
from mathutils import Vector

from .smart_markers import (
    FINGER_NAMES,
    collect_marker_positions,
    compute_meshes_world_bbox,
    get_symmetry_center_x,
    infer_facing_sign_y,
    left_side_x_sign,
    set_markers_visibility,
)

# Default 25-bone Z-Up rest offsets (meters) matching engine/skeleton/rig_definition.py
DEFAULT_BODY_BONES: List[Tuple[str, Optional[str], Tuple[float, float, float]]] = [
    ("hips", None, (0.00, 0.00, 0.96)),
    ("spine", "hips", (0.00, 0.00, 0.11)),
    ("spine1", "spine", (0.00, 0.00, 0.14)),
    ("chest", "spine1", (0.00, 0.00, 0.16)),
    ("neck", "chest", (0.00, 0.00, 0.11)),
    ("head", "neck", (0.00, 0.00, 0.10)),
    ("head_end", "head", (0.00, 0.00, 0.14)),
    ("left_clavicle", "chest", (-0.05, 0.00, 0.08)),
    ("left_upper_arm", "left_clavicle", (-0.13, 0.00, 0.00)),
    ("left_forearm", "left_upper_arm", (-0.28, 0.00, 0.00)),
    ("left_hand", "left_forearm", (-0.25, 0.00, 0.00)),
    ("left_hand_end", "left_hand", (-0.09, 0.00, 0.00)),
    ("right_clavicle", "chest", (0.05, 0.00, 0.08)),
    ("right_upper_arm", "right_clavicle", (0.13, 0.00, 0.00)),
    ("right_forearm", "right_upper_arm", (0.28, 0.00, 0.00)),
    ("right_hand", "right_forearm", (0.25, 0.00, 0.00)),
    ("right_hand_end", "right_hand", (0.09, 0.00, 0.00)),
    ("left_upper_leg", "hips", (-0.095, 0.00, -0.04)),
    ("left_lower_leg", "left_upper_leg", (0.00, 0.00, -0.43)),
    ("left_foot", "left_lower_leg", (0.00, 0.00, -0.41)),
    ("left_toe", "left_foot", (0.00, 0.15, -0.07)),
    ("right_upper_leg", "hips", (0.095, 0.00, -0.04)),
    ("right_lower_leg", "right_upper_leg", (0.00, 0.00, -0.43)),
    ("right_foot", "right_lower_leg", (0.00, 0.00, -0.41)),
    ("right_toe", "right_foot", (0.00, 0.15, -0.07)),
]

# Relative finger layout from wrist (along hand outward direction `out`, forward `fwd`, vertical `up`)
# (finger_name, (out_start, fwd_start, z_start), (seg1_len, seg2_len, seg3_len), fwd_splay)
_FINGER_LAYOUT: List[Tuple[str, Tuple[float, float, float], Tuple[float, float, float], float]] = [
    ("thumb", (0.032, 0.030, -0.012), (0.032, 0.026, 0.022), 0.55),
    ("index", (0.078, 0.026, 0.002), (0.034, 0.024, 0.018), 0.08),
    ("middle", (0.082, 0.005, 0.004), (0.038, 0.026, 0.019), 0.00),
    ("ring", (0.078, -0.015, 0.002), (0.035, 0.024, 0.018), -0.06),
    ("pinky", (0.072, -0.034, -0.002), (0.028, 0.020, 0.016), -0.14),
]


def get_canonical_finger_bone_names() -> List[str]:
    names: List[str] = []
    for side in ("left", "right"):
        for fname in FINGER_NAMES:
            for seg in (1, 2, 3):
                names.append(f"{side}_{fname}_{seg:02d}")
    return names


def append_finger_edit_bones(
    arm_data: bpy.types.Armature,
    hand_bone_name: str,
    side: str,
    scale: float = 1.0,
    facing_sign_y: float = 1.0,
    finger_markers_local: Optional[Dict[str, Vector]] = None,
) -> List[str]:
    """Append 15 articulated finger EditBones (thumb/index/middle/ring/pinky 01..03) onto `hand_bone_name`.

    - If `finger_markers_local` provides explicit `f_{finger}_base_{L/R}` and `f_{finger}_tip_{L/R}`
      positions, each finger's 3 phalanges are placed directly along the user's markers.
    - Otherwise, fingers are automatically scaled and aligned to the hand's exact direction &
      `facing_sign_y` (-1.0 for Blender -Y front view, +1.0 for +Y) so thumbs always sit on the front side.

    Must be called while `arm_data` is in EDIT mode.
    """
    hand_eb = arm_data.edit_bones.get(hand_bone_name)
    if hand_eb is None:
        return []

    out_dir = hand_eb.tail - hand_eb.head
    if out_dir.length < 1e-4:
        out_dir = Vector((-1.0, 0.0, 0.0) if side == "left" else (1.0, 0.0, 0.0))
    else:
        out_dir = out_dir.normalized()

    raw_fwd = Vector((0.0, float(facing_sign_y), 0.0))
    fwd_dir = raw_fwd - out_dir * raw_fwd.dot(out_dir)
    if fwd_dir.length < 1e-4:
        fwd_dir = Vector((0.0, 0.0, 1.0)).cross(out_dir)
    fwd_dir.normalize()

    up_dir = out_dir.cross(fwd_dir)
    if up_dir.z < 0.0:
        up_dir = -up_dir
    if up_dir.length < 1e-4:
        up_dir = Vector((0.0, 0.0, 1.0))
    else:
        up_dir.normalize()

    created: List[str] = []
    wrist_pos = hand_eb.head.copy()
    suf = "L" if side == "left" else "R"

    for fname, (o_s, f_s, z_s), (l1, l2, l3), splay in _FINGER_LAYOUT:
        m_base = finger_markers_local.get(f"f_{fname}_base_{suf}") if finger_markers_local else None
        m_tip = finger_markers_local.get(f"f_{fname}_tip_{suf}") if finger_markers_local else None

        if m_base is not None and m_tip is not None and (m_tip - m_base).length > 1e-4:
            # Build 3 anatomical phalanges directly from user's custom finger knuckle -> tip markers!
            vec = m_tip - m_base
            ratios = (0.42, 0.33, 0.25)
            parent_eb = hand_eb
            cur_head = m_base.copy()
            for seg_idx, frac in enumerate(ratios, start=1):
                bname = f"{side}_{fname}_{seg_idx:02d}"
                if bname in arm_data.edit_bones:
                    parent_eb = arm_data.edit_bones[bname]
                    cur_head = parent_eb.tail.copy()
                    continue
                eb = arm_data.edit_bones.new(bname)
                eb.head = cur_head
                eb.tail = cur_head + vec * frac
                eb.align_roll(up_dir)
                eb.parent = parent_eb
                eb.use_connect = seg_idx > 1
                parent_eb = eb
                cur_head = eb.tail.copy()
                created.append(bname)
            continue

        base_pos = wrist_pos + out_dir * (o_s * scale) + fwd_dir * (f_s * scale) + up_dir * (z_s * scale)
        f_dir = (out_dir + fwd_dir * splay).normalized()
        parent_eb = hand_eb
        cur_head = base_pos
        for seg_idx, seg_len in enumerate((l1, l2, l3), start=1):
            bname = f"{side}_{fname}_{seg_idx:02d}"
            if bname in arm_data.edit_bones:
                parent_eb = arm_data.edit_bones[bname]
                cur_head = parent_eb.tail.copy()
                continue
            eb = arm_data.edit_bones.new(bname)
            eb.head = cur_head
            eb.tail = cur_head + f_dir * (seg_len * scale)
            eb.align_roll(up_dir)
            eb.parent = parent_eb
            eb.use_connect = seg_idx > 1
            parent_eb = eb
            cur_head = eb.tail.copy()
            created.append(bname)

    return created


def _safe_tail(head: Vector, tail: Vector, fallback_dir: Vector, min_len: float = 0.012) -> Vector:
    if (tail - head).length < min_len:
        d = fallback_dir.normalized() if fallback_dir.length > 1e-6 else Vector((0.0, 0.0, 1.0))
        return head + d * max(min_len, 0.03)
    return tail


def _resolve_lr_pair(
    markers: Dict[str, Vector],
    base_key: str,
    center_x: float,
    l_sign: float,
) -> Tuple[Vector, Vector]:
    """Return (left_world, right_world) properly oriented to the character's actual Left/Right side."""
    p_l = markers.get(f"{base_key}_L")
    p_r = markers.get(f"{base_key}_R")
    if p_l is not None and p_r is None:
        p_r = Vector((2.0 * center_x - p_l.x, p_l.y, p_l.z))
    elif p_r is not None and p_l is None:
        p_l = Vector((2.0 * center_x - p_r.x, p_r.y, p_r.z))
    elif p_l is None and p_r is None:
        raise ValueError(f"Missing required landmark marker '{base_key}_L' / '{base_key}_R'.")

    # Ensure Character-Left is actually on the l_sign side of center_x!
    # (Prevents left/right arm swap if user placed markers on the opposite side of screen)
    dx_l = float(p_l.x - center_x) * l_sign
    dx_r = float(p_r.x - center_x) * l_sign
    if dx_l < dx_r:
        return p_r.copy(), p_l.copy()
    return p_l.copy(), p_r.copy()


def create_rig_from_markers(
    markers_world: Dict[str, Vector],
    name: str = "Nuranova_SmartRig",
    include_fingers: bool = True,
    facing_override: str = "AUTO",
    context: Optional[bpy.types.Context] = None,
) -> bpy.types.Object:
    """Build an accurate 55-bone (25 body + 30 finger) humanoid armature directly from placed markers."""
    ctx = context or bpy.context
    required_center = ("pelvis", "chest", "chin", "head_top")
    for k in required_center:
        if k not in markers_world:
            raise ValueError(f"Missing required center marker '{k}'. Place all 12 body landmarks first.")

    facing_y = infer_facing_sign_y(markers_world, override=facing_override)
    l_sign = left_side_x_sign(facing_y)
    center_x = get_symmetry_center_x(existing_markers=markers_world)

    pelvis_w = markers_world["pelvis"].copy()
    chest_w = markers_world["chest"].copy()
    chin_w = markers_world["chin"].copy()
    head_top_w = markers_world["head_top"].copy()

    shoulder_l, shoulder_r = _resolve_lr_pair(markers_world, "shoulder", center_x, l_sign)
    elbow_l, elbow_r = _resolve_lr_pair(markers_world, "elbow", center_x, l_sign)
    wrist_l, wrist_r = _resolve_lr_pair(markers_world, "wrist", center_x, l_sign)
    hand_tip_l, hand_tip_r = _resolve_lr_pair(markers_world, "hand_tip", center_x, l_sign)
    hip_l, hip_r = _resolve_lr_pair(markers_world, "hip", center_x, l_sign)
    knee_l, knee_r = _resolve_lr_pair(markers_world, "knee", center_x, l_sign)
    ankle_l, ankle_r = _resolve_lr_pair(markers_world, "ankle", center_x, l_sign)

    # Toe is optional (auto-estimated forward from ankle if missing)
    if "toe_L" in markers_world or "toe_R" in markers_world:
        toe_l, toe_r = _resolve_lr_pair(markers_world, "toe", center_x, l_sign)
    else:
        leg_len = max(0.25, float((hip_l - ankle_l).length))
        foot_fwd = Vector((0.0, facing_y * leg_len * 0.20, -leg_len * 0.07))
        toe_l = ankle_l + foot_fwd
        toe_r = ankle_r + foot_fwd

    # Put armature origin cleanly on the floor directly beneath pelvis
    floor_z = min(float(ankle_l.z), float(ankle_r.z), float(toe_l.z), float(toe_r.z))
    origin = Vector((center_x, float(pelvis_w.y), floor_z))

    def to_local(p: Vector) -> Vector:
        return p - origin

    pelvis = to_local(pelvis_w)
    chest = to_local(chest_w)
    chin = to_local(chin_w)
    head_top = to_local(head_top_w)

    # Neck base sits between chest and chin, aligned close to shoulder height
    avg_shoulder_z = 0.5 * float(to_local(shoulder_l).z + to_local(shoulder_r).z)
    neck_base = chest.lerp(chin, 0.72)
    # Blend neck_base Z cleanly toward shoulder line while staying strictly between chest and chin
    lo_z = min(chest.z, chin.z) + 0.02
    hi_z = max(chest.z, chin.z) - 0.015
    if hi_z > lo_z:
        neck_base.z = max(lo_z, min(hi_z, 0.5 * (neck_base.z + avg_shoulder_z)))

    arm_data = bpy.data.armatures.new(name + "_Data")
    arm_obj = bpy.data.objects.new(name, arm_data)
    arm_obj.location = origin
    arm_obj.show_in_front = True
    arm_data.display_type = "OCTAHEDRAL"

    ctx.collection.objects.link(arm_obj)
    bpy.ops.object.select_all(action="DESELECT")
    arm_obj.select_set(True)
    ctx.view_layer.objects.active = arm_obj

    bpy.ops.object.mode_set(mode="EDIT")
    ebs: Dict[str, bpy.types.EditBone] = {}

    def add_bone(
        bname: str,
        head: Vector,
        tail: Vector,
        parent_name: Optional[str],
        connect: bool = False,
        fallback_dir: Vector = Vector((0.0, 0.0, 1.0)),
    ) -> bpy.types.EditBone:
        eb = arm_data.edit_bones.new(bname)
        eb.head = head.copy()
        eb.tail = _safe_tail(head, tail, fallback_dir)
        if parent_name is not None:
            eb.parent = ebs[parent_name]
            eb.use_connect = connect
        ebs[bname] = eb
        return eb

    # 1. Spine + Neck + Head chain (fully connected, zero gaps!)
    spine_base = pelvis.lerp(chest, 0.30)
    spine1_base = pelvis.lerp(chest, 0.65)
    head_mid = chin.lerp(head_top, 0.76)

    add_bone("hips", pelvis, spine_base, None, connect=False)
    add_bone("spine", spine_base, spine1_base, "hips", connect=True)
    add_bone("spine1", spine1_base, chest, "spine", connect=True)
    add_bone("chest", chest, neck_base, "spine1", connect=True)
    add_bone("neck", neck_base, chin, "chest", connect=True)
    add_bone("head", chin, head_mid, "neck", connect=True)
    add_bone("head_end", head_mid, head_top, "head", connect=True)

    # 2. Left & Right Arms + Legs (connected limb chains!)
    side_data = [
        ("left", "L", to_local(shoulder_l), to_local(elbow_l), to_local(wrist_l), to_local(hand_tip_l),
         to_local(hip_l), to_local(knee_l), to_local(ankle_l), to_local(toe_l)),
        ("right", "R", to_local(shoulder_r), to_local(elbow_r), to_local(wrist_r), to_local(hand_tip_r),
         to_local(hip_r), to_local(knee_r), to_local(ankle_r), to_local(toe_r)),
    ]

    finger_markers_local = {
        k: to_local(v)
        for k, v in markers_world.items()
        if k.startswith("f_")
    }

    for side, _suf, sh, el, wr, ht, hp, kn, an, to in side_data:
        out_fallback = Vector((l_sign if side == "left" else -l_sign, 0.0, 0.0))
        clav_head = neck_base.lerp(sh, 0.22)
        hand_mid = wr.lerp(ht, 0.46)

        add_bone(f"{side}_clavicle", clav_head, sh, "chest", connect=False, fallback_dir=out_fallback)
        add_bone(f"{side}_upper_arm", sh, el, f"{side}_clavicle", connect=True, fallback_dir=out_fallback)
        add_bone(f"{side}_forearm", el, wr, f"{side}_upper_arm", connect=True, fallback_dir=out_fallback)
        add_bone(f"{side}_hand", wr, hand_mid, f"{side}_forearm", connect=True, fallback_dir=out_fallback)
        eb_hand_end = add_bone(f"{side}_hand_end", hand_mid, ht, f"{side}_hand", connect=True, fallback_dir=out_fallback)
        if include_fingers:
            # Non-deforming leaf tip so Automatic Weights never steals middle/index/ring finger vertices!
            eb_hand_end.use_deform = False

        # Legs
        ball = an.lerp(to, 0.65)
        ball.z = min(ball.z, to.z + 0.02 * max(0.3, float((hp - an).length)))
        add_bone(f"{side}_upper_leg", hp, kn, "hips", connect=False, fallback_dir=Vector((0.0, 0.0, -1.0)))
        add_bone(f"{side}_lower_leg", kn, an, f"{side}_upper_leg", connect=True, fallback_dir=Vector((0.0, 0.0, -1.0)))
        add_bone(f"{side}_foot", an, ball, f"{side}_lower_leg", connect=True, fallback_dir=Vector((0.0, facing_y, 0.0)))
        add_bone(f"{side}_toe", ball, to, f"{side}_foot", connect=True, fallback_dir=Vector((0.0, facing_y, 0.0)))

        # 3. 15 Articulated Finger Bones per hand (scaled precisely to actual Wrist -> HandTip length!)
        if include_fingers:
            hand_len = max(0.05, float((ht - wr).length))
            # Default finger template spans ~0.165 m from wrist to middle finger tip
            finger_scale = hand_len / 0.165
            append_finger_edit_bones(
                arm_data,
                f"{side}_hand",
                side,
                scale=finger_scale,
                facing_sign_y=facing_y,
                finger_markers_local=finger_markers_local,
            )

    # Clean bone roll alignment for smooth skinning & retargeting
    try:
        bpy.ops.armature.select_all(action="SELECT")
        bpy.ops.armature.calculate_roll(type="GLOBAL_POS_Z")
    except Exception:
        pass

    bpy.ops.object.mode_set(mode="OBJECT")

    for pb in arm_obj.pose.bones:
        pb.rotation_mode = "QUATERNION"

    arm_obj["nuranova_auto_rig"] = True
    arm_obj["nuranova_smart_marker_rig"] = True
    arm_obj["nuranova_has_fingers"] = bool(include_fingers)
    arm_obj["nuranova_facing_y"] = float(facing_y)
    arm_obj["nuranova_rest_hip_height"] = float(pelvis.z)
    return arm_obj


def _bind_meshes_to_armature(
    mesh_objs: Sequence[bpy.types.Object],
    arm_obj: bpy.types.Object,
    context: bpy.types.Context,
) -> str:
    """Bind one or more character meshes to `arm_obj` cleanly (removing stale Nuranova modifiers first)."""
    valid_meshes = [o for o in mesh_objs if o is not None and o.type == "MESH"]
    if not valid_meshes:
        raise ValueError("Select at least one character Mesh object to bind.")

    # Clean up any previous Nuranova auto-rig parenting/modifiers so iterative re-rigging stays spotless!
    for m in valid_meshes:
        if m.parent is not None and m.parent.type == "ARMATURE" and m.parent.get("nuranova_auto_rig"):
            old_world = m.matrix_world.copy()
            m.parent = None
            m.matrix_world = old_world
        for mod in list(m.modifiers):
            if mod.type == "ARMATURE" and (mod.object is None or mod.object.get("nuranova_auto_rig")):
                m.modifiers.remove(mod)

    bpy.ops.object.select_all(action="DESELECT")
    for m in valid_meshes:
        m.select_set(True)
    arm_obj.select_set(True)
    context.view_layer.objects.active = arm_obj

    weight_method = "ARMATURE_AUTO"
    try:
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    except Exception:
        weight_method = "ARMATURE_ENVELOPE"
        try:
            bpy.ops.object.parent_set(type="ARMATURE_ENVELOPE")
        except Exception:
            weight_method = "ARMATURE_NAME"
            bpy.ops.object.parent_set(type="ARMATURE_NAME")
    return weight_method


def auto_rig_from_markers(
    mesh_objs: Sequence[bpy.types.Object],
    include_fingers: bool = True,
    facing_override: str = "AUTO",
    hide_markers_after: bool = True,
    rig_name: Optional[str] = None,
    context: Optional[bpy.types.Context] = None,
) -> Tuple[bpy.types.Object, str]:
    """Generate a 55-bone rig directly from placed Smart Markers and bind character mesh(es)."""
    ctx = context or bpy.context
    valid_meshes = [o for o in mesh_objs if o is not None and o.type == "MESH"]
    if not valid_meshes:
        raise ValueError("Pick a Character Mesh or select at least one Mesh object before generating the rig.")

    markers = collect_marker_positions()
    if not markers:
        raise ValueError("No Smart Markers found! Click 'Start Guided Marker Placement' or 'Place Preset Markers' first.")

    base_name = rig_name or f"{valid_meshes[0].name}_NuranovaRig"
    # Replace an older Smart Rig with the exact same name if user is refining markers iteratively
    old_arm = bpy.data.objects.get(base_name)
    if old_arm is not None and old_arm.type == "ARMATURE" and old_arm.get("nuranova_auto_rig"):
        for m in valid_meshes:
            if m.parent == old_arm:
                old_world = m.matrix_world.copy()
                m.parent = None
                m.matrix_world = old_world
            for mod in list(m.modifiers):
                if mod.type == "ARMATURE" and mod.object == old_arm:
                    m.modifiers.remove(mod)
        bpy.data.objects.remove(old_arm, do_unlink=True)

    arm_obj = create_rig_from_markers(
        markers_world=markers,
        name=base_name,
        include_fingers=include_fingers,
        facing_override=facing_override,
        context=ctx,
    )
    method = _bind_meshes_to_armature(valid_meshes, arm_obj, ctx)
    if hide_markers_after:
        set_markers_visibility(False)
    return arm_obj, method


def create_nuranova_humanoid_rig(
    name: str = "Nuranova_Humanoid_Rig",
    height_m: float = 1.72,
    include_fingers: bool = True,
    location: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    fit_bbox: Optional[Tuple[Vector, Vector]] = None,
    context: Optional[bpy.types.Context] = None,
) -> bpy.types.Object:
    """Create a complete Z-Up Nuranova Humanoid Armature (25 body + optional 30 finger bones = 55 bones)."""
    ctx = context or bpy.context
    base_height = 1.72
    if fit_bbox is not None:
        bmin, bmax = fit_bbox
        mesh_h = max(0.25, float(bmax.z - bmin.z))
        scale_z = mesh_h / base_height
        mesh_span_x = max(0.20, float(bmax.x - bmin.x))
        scale_x = max(0.45 * scale_z, min(1.35 * scale_z, mesh_span_x / 1.60))
        origin = Vector((0.5 * (bmin.x + bmax.x), 0.5 * (bmin.y + bmax.y), bmin.z))
    else:
        scale_z = max(0.2, float(height_m)) / base_height
        scale_x = scale_z
        origin = Vector(location)

    arm_data = bpy.data.armatures.new(name + "_Data")
    arm_obj = bpy.data.objects.new(name, arm_data)
    arm_obj.location = origin
    arm_obj.show_in_front = True
    arm_data.display_type = "OCTAHEDRAL"

    ctx.collection.objects.link(arm_obj)
    bpy.ops.object.select_all(action="DESELECT")
    arm_obj.select_set(True)
    ctx.view_layer.objects.active = arm_obj

    heads: Dict[str, Vector] = {}
    children: Dict[str, List[str]] = {b[0]: [] for b in DEFAULT_BODY_BONES}
    parent_map: Dict[str, Optional[str]] = {}

    for bname, pname, (ox, oy, oz) in DEFAULT_BODY_BONES:
        sx = scale_x if ("arm" in bname or "clavicle" in bname or "hand" in bname) else scale_z
        off = Vector((ox * sx, oy * scale_z, oz * scale_z))
        if pname is None:
            heads[bname] = off
        else:
            heads[bname] = heads[pname] + off
            children[pname].append(bname)
        parent_map[bname] = pname

    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones: Dict[str, bpy.types.EditBone] = {}
    for bname, pname, _ in DEFAULT_BODY_BONES:
        eb = arm_data.edit_bones.new(bname)
        eb.head = heads[bname]
        kids = children[bname]
        if kids:
            tail = sum((heads[k] for k in kids), Vector((0.0, 0.0, 0.0))) / len(kids)
            if (tail - eb.head).length < 0.008:
                tail = eb.head + Vector((0.0, 0.0, 0.06 * scale_z))
            eb.tail = tail
        else:
            pvec = (heads[bname] - heads[pname]) if pname else Vector((0.0, 0.0, 0.1))
            if pvec.length < 0.008:
                pvec = Vector((0.0, 0.0, 0.06 * scale_z))
            eb.tail = eb.head + pvec.normalized() * (0.07 * scale_z)
        edit_bones[bname] = eb

    for bname, pname in parent_map.items():
        if pname is not None:
            edit_bones[bname].parent = edit_bones[pname]
            edit_bones[bname].use_connect = False

    if include_fingers:
        for end_bn in ("left_hand_end", "right_hand_end"):
            if end_bn in edit_bones:
                # Non-deforming leaf tip so Automatic Weights never steals finger vertices!
                edit_bones[end_bn].use_deform = False
        append_finger_edit_bones(arm_data, "left_hand", "left", scale=scale_z, facing_sign_y=1.0)
        append_finger_edit_bones(arm_data, "right_hand", "right", scale=scale_z, facing_sign_y=1.0)

    bpy.ops.object.mode_set(mode="OBJECT")

    for pb in arm_obj.pose.bones:
        pb.rotation_mode = "QUATERNION"

    arm_obj["nuranova_auto_rig"] = True
    arm_obj["nuranova_has_fingers"] = bool(include_fingers)
    arm_obj["nuranova_rest_hip_height"] = float(heads["hips"].z)
    return arm_obj


def auto_rig_character_mesh(
    mesh_objs: List[bpy.types.Object],
    include_fingers: bool = True,
    rig_name: Optional[str] = None,
    context: Optional[bpy.types.Context] = None,
) -> Tuple[bpy.types.Object, str]:
    """Quick 1-click Auto-Rig using bounding box (for realistic adult meshes without placing markers)."""
    ctx = context or bpy.context
    valid_meshes = [o for o in mesh_objs if o is not None and o.type == "MESH"]
    if not valid_meshes:
        raise ValueError("Select at least one character Mesh object to Auto-Rig.")

    bbox = compute_meshes_world_bbox(valid_meshes)
    base_name = rig_name or f"{valid_meshes[0].name}_NuranovaRig"
    arm_obj = create_nuranova_humanoid_rig(
        name=base_name,
        include_fingers=include_fingers,
        fit_bbox=bbox,
        context=ctx,
    )
    weight_method = _bind_meshes_to_armature(valid_meshes, arm_obj, ctx)
    return arm_obj, weight_method
