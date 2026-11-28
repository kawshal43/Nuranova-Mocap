"""Character-agnostic Rigify, Mixamo, Unreal, VRM & Auto-Rig retargeting (25 Body + 30 Finger Bones)
with rest-pose delta & scale compensation, real-time live retargeting, editable keyframe baking, and mapping profiles.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import bpy
from mathutils import Matrix, Vector

from .facial_hands import HAND_BONE_SUFFIXES, _finger_bone_candidates
from .source_armature import _ensure_action_on_obj, push_action_to_nla, write_fcurve_batch

# Fuzzy synonym patterns supporting Nuranova, Rigify, Mixamo, Unreal Mannequin, VRM/Unity, and custom rigs
BONE_SYNONYMS: Dict[str, List[str]] = {
    "hips": ["hips", "pelvis", "root", "torso", "def-spine", "mixamorig:hips"],
    "spine": ["spine", "def-spine.001", "spine_01", "mixamorig:spine"],
    "spine1": ["spine1", "spine.001", "def-spine.002", "spine_02", "mixamorig:spine1"],
    "chest": ["chest", "upper_chest", "spine2", "def-spine.003", "spine_03", "mixamorig:spine2"],
    "neck": ["neck", "def-spine.004", "neck_01", "mixamorig:neck"],
    "head": ["head", "def-spine.006", "mixamorig:head"],
    "left_clavicle": ["shoulder.l", "clavicle.l", "def-shoulder.l", "clavicle_l", "leftshoulder", "mixamorig:leftshoulder"],
    "left_upper_arm": ["upper_arm.l", "def-upper_arm.l", "upperarm_l", "leftarm", "arm.l", "mixamorig:leftarm"],
    "left_forearm": ["forearm.l", "def-forearm.l", "lowerarm_l", "leftforearm", "mixamorig:leftforearm"],
    "left_hand": ["hand.l", "def-hand.l", "hand_l", "lefthand", "wrist.l", "mixamorig:lefthand"],
    "right_clavicle": ["shoulder.r", "clavicle.r", "def-shoulder.r", "clavicle_r", "rightshoulder", "mixamorig:rightshoulder"],
    "right_upper_arm": ["upper_arm.r", "def-upper_arm.r", "upperarm_r", "rightarm", "arm.r", "mixamorig:rightarm"],
    "right_forearm": ["forearm.r", "def-forearm.r", "lowerarm_r", "rightforearm", "mixamorig:rightforearm"],
    "right_hand": ["hand.r", "def-hand.r", "hand_r", "righthand", "wrist.r", "mixamorig:righthand"],
    "left_upper_leg": ["thigh.l", "def-thigh.l", "thigh_l", "leftupleg", "upper_leg.l", "mixamorig:leftupleg"],
    "left_lower_leg": ["shin.l", "def-shin.l", "calf_l", "leftleg", "lower_leg.l", "mixamorig:leftleg"],
    "left_foot": ["foot.l", "def-foot.l", "foot_l", "leftfoot", "mixamorig:leftfoot"],
    "left_toe": ["toe.l", "def-toe.l", "ball_l", "lefttoebase", "mixamorig:lefttoebase"],
    "right_upper_leg": ["thigh.r", "def-thigh.r", "thigh_r", "rightupleg", "upper_leg.r", "mixamorig:rightupleg"],
    "right_lower_leg": ["shin.r", "def-shin.r", "calf_r", "rightleg", "lower_leg.r", "mixamorig:rightleg"],
    "right_foot": ["foot.r", "def-foot.r", "foot_r", "rightfoot", "mixamorig:rightfoot"],
    "right_toe": ["toe.r", "def-toe.r", "ball_r", "righttoebase", "mixamorig:righttoebase"],
}

# Populate all 30 finger bone synonyms automatically!
for _side_word, _side_tag in [("left", "l"), ("right", "r")]:
    for _suf in HAND_BONE_SUFFIXES[1:]:
        _src = f"{_side_word}_{_suf}"
        BONE_SYNONYMS[_src] = _finger_bone_candidates(_side_word, _side_tag, _suf)

# Priority prefix hints per rig preset
_PRESET_PRIORITY_HINTS: Dict[str, List[str]] = {
    "RIGIFY": ["def-", "f_", ".l", ".r"],
    "MIXAMO": ["mixamorig:", "lefthand", "righthand"],
    "UNREAL": ["_l", "_r", "spine_0"],
    "VRM": ["proximal", "intermediate", "distal"],
    "NURANOVA": ["left_", "right_"],
}


def auto_match_bones(
    source_arm: bpy.types.Object,
    target_arm: bpy.types.Object,
    rig_preset: str = "AUTO",
) -> Dict[str, str]:
    """Automatically map Nuranova source bones (body + 30 fingers) to any target rig."""
    target_names = [b.name for b in target_arm.data.bones]
    target_lower = {name.lower(): name for name in target_names}
    mapping: Dict[str, str] = {}
    used_targets = set()
    preset_hints = _PRESET_PRIORITY_HINTS.get(rig_preset.upper(), [])

    def _ordered_syns(syns: List[str]) -> List[str]:
        if not preset_hints:
            return syns
        prioritized = [s for s in syns if any(h in s for h in preset_hints)]
        rest = [s for s in syns if s not in prioritized]
        return prioritized + rest

    for src_bone in source_arm.data.bones:
        s_name = src_bone.name
        s_low = s_name.lower()
        # 1. Exact match
        if s_low in target_lower and target_lower[s_low] not in used_targets:
            mapping[s_name] = target_lower[s_low]
            used_targets.add(target_lower[s_low])
            continue
        # 2. Synonym exact or substring match (respecting rig preset hints)
        syns = _ordered_syns(BONE_SYNONYMS.get(s_low, []))
        matched = None
        for syn in syns:
            if syn in target_lower and target_lower[syn] not in used_targets:
                matched = target_lower[syn]
                break
        if matched is None:
            for syn in syns:
                for t_low, t_orig in target_lower.items():
                    if syn in t_low and t_orig not in used_targets:
                        matched = t_orig
                        break
                if matched:
                    break
        if matched is not None:
            mapping[s_name] = matched
            used_targets.add(matched)

    return mapping


def save_mapping_profile(mapping: Dict[str, str], filepath: str | Path, rig_name: str = "") -> Path:
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"schema": "nuranova-rig-map-v1", "rig_name": rig_name, "bone_map": mapping}, indent=2),
        encoding="utf-8",
    )
    return p


def load_mapping_profile(filepath: str | Path) -> Dict[str, str]:
    data = json.loads(Path(filepath).read_text(encoding="utf-8"))
    return dict(data.get("bone_map", {}))


def _compute_facing_yaw_matrix(
    source_arm: bpy.types.Object,
    target_arm: bpy.types.Object,
    bone_map: Dict[str, str],
) -> Matrix:
    """Compute the Z-axis yaw rotation aligning `source_arm`'s horizontal facing to `target_arm`'s facing.

    Nuranova mocap source armatures face +Y (left arm along -X, right arm along +X), while most
    Blender character meshes (Auto-Rig, Mixamo, Rigify Front View) face -Y (left arm along +X).
    Aligning the horizontal Left->Right axis guarantees forward bending, arm raising, and root
    walking always move in the target character's true forward direction!
    """
    lr_candidates = [
        ("left_upper_arm", "right_upper_arm"),
        ("left_upper_leg", "right_upper_leg"),
        ("left_clavicle", "right_clavicle"),
        ("left_foot", "right_foot"),
    ]
    for src_l, src_r in lr_candidates:
        tgt_l = bone_map.get(src_l)
        tgt_r = bone_map.get(src_r)
        if not tgt_l or not tgt_r:
            continue
        if (
            src_l not in source_arm.data.bones
            or src_r not in source_arm.data.bones
            or tgt_l not in target_arm.data.bones
            or tgt_r not in target_arm.data.bones
        ):
            continue
        p_sl = source_arm.matrix_world @ source_arm.data.bones[src_l].head_local
        p_sr = source_arm.matrix_world @ source_arm.data.bones[src_r].head_local
        p_tl = target_arm.matrix_world @ target_arm.data.bones[tgt_l].head_local
        p_tr = target_arm.matrix_world @ target_arm.data.bones[tgt_r].head_local
        v_src = Vector((p_sr.x - p_sl.x, p_sr.y - p_sl.y, 0.0))
        v_tgt = Vector((p_tr.x - p_tl.x, p_tr.y - p_tl.y, 0.0))
        if v_src.length > 0.02 and v_tgt.length > 0.02:
            yaw = math.atan2(v_tgt.y, v_tgt.x) - math.atan2(v_src.y, v_src.x)
            return Matrix.Rotation(yaw, 3, "Z")
    return Matrix.Identity(3)


def _prepare_retarget_pairs(
    source_arm: bpy.types.Object,
    target_arm: bpy.types.Object,
    bone_map: Dict[str, str],
) -> Tuple[List[Tuple[str, str]], Dict[str, Matrix], Tuple[str, str], float, Vector, Vector, Matrix]:
    src_world_R = source_arm.matrix_world.to_3x3().normalized()
    tgt_world_R = target_arm.matrix_world.to_3x3().normalized()
    R_yaw = _compute_facing_yaw_matrix(source_arm, target_arm, bone_map)

    rest_deltas: Dict[str, Matrix] = {}
    valid_pairs: List[Tuple[str, str]] = []
    for src_name, tgt_name in bone_map.items():
        if src_name in source_arm.pose.bones and tgt_name in target_arm.pose.bones:
            R_src_rest = R_yaw @ (src_world_R @ source_arm.data.bones[src_name].matrix_local.to_3x3())
            R_tgt_rest = tgt_world_R @ target_arm.data.bones[tgt_name].matrix_local.to_3x3()
            rest_deltas[tgt_name] = R_src_rest.inverted() @ R_tgt_rest
            valid_pairs.append((src_name, tgt_name))

    if not valid_pairs:
        raise ValueError("No valid matching bones found between Source Rig and Target Character Rig.")

    def bone_depth(b: bpy.types.Bone) -> int:
        d = 0
        cur = b.parent
        while cur is not None:
            d += 1
            cur = cur.parent
        return d

    valid_pairs.sort(key=lambda pair: bone_depth(target_arm.data.bones[pair[1]]))

    root_pair = next(((s, t) for s, t in valid_pairs if s.lower() == "hips"), valid_pairs[0])
    src_hip_h = max((source_arm.matrix_world @ source_arm.data.bones[root_pair[0]].head_local).z, 0.1)
    tgt_hip_h = max((target_arm.matrix_world @ target_arm.data.bones[root_pair[1]].head_local).z, 0.1)
    scale_ratio = tgt_hip_h / src_hip_h

    src_root_rest_world = source_arm.matrix_world @ source_arm.data.bones[root_pair[0]].head_local
    tgt_root_rest_world = target_arm.matrix_world @ target_arm.data.bones[root_pair[1]].head_local
    return valid_pairs, rest_deltas, root_pair, scale_ratio, src_root_rest_world, tgt_root_rest_world, R_yaw


def _collect_foot_bone_names(
    arm: bpy.types.Object,
    names: List[str],
) -> List[str]:
    return [n for n in names if n and n in arm.data.bones and n in arm.pose.bones]


def _compute_rest_floor_z(arm: bpy.types.Object, foot_names: List[str]) -> float:
    zs: List[float] = []
    for n in foot_names:
        b = arm.data.bones[n]
        zs.append((arm.matrix_world @ b.head_local).z)
        zs.append((arm.matrix_world @ b.tail_local).z)
    return min(zs) if zs else 0.0


def _compute_pose_min_foot_z(arm: bpy.types.Object, foot_names: List[str]) -> Optional[float]:
    zs: List[float] = []
    for n in foot_names:
        pb = arm.pose.bones[n]
        zs.append((arm.matrix_world @ pb.head).z)
        zs.append((arm.matrix_world @ pb.tail).z)
    return min(zs) if zs else None


def _apply_retarget_single_step(
    source_arm: bpy.types.Object,
    target_arm: bpy.types.Object,
    valid_pairs: List[Tuple[str, str]],
    rest_deltas: Dict[str, Matrix],
    root_pair: Tuple[str, str],
    scale_ratio: float,
    src_root_rest_world: Vector,
    tgt_root_rest_world: Vector,
    facing_yaw_R: Matrix,
    extract_root_motion: bool,
    in_place: bool,
    insert_frame: Optional[int] = None,
    ground_to_floor: bool = False,
    src_foot_names: Optional[List[str]] = None,
    tgt_foot_names: Optional[List[str]] = None,
    src_rest_floor_z: float = 0.0,
    tgt_rest_floor_z: float = 0.0,
    update_fn=None,
) -> None:
    # Read the complete source pose before writing any target bone. Interleaving
    # evaluated matrix reads with pose writes forces repeated dependency updates.
    source_world = source_arm.matrix_world.copy()
    target_world = target_arm.matrix_world.copy()
    source_rotations = {
        src: facing_yaw_R @ (source_world @ source_arm.pose.bones[src].matrix).to_3x3().normalized()
        for src, _ in valid_pairs
    }
    mapped_targets = {target for _, target in valid_pairs}
    fallback_parent_rotations = {
        bone.parent.name: (target_world @ target_arm.pose.bones[bone.parent.name].matrix).to_3x3().normalized()
        for _, name in valid_pairs for bone in [target_arm.data.bones[name]]
        if bone.parent is not None and bone.parent.name not in mapped_targets
    }
    tgt_root_pb = target_arm.pose.bones[root_pair[1]]
    tgt_rest_mat = target_arm.matrix_world @ target_arm.data.bones[root_pair[1]].matrix_local
    desired_world = tgt_root_rest_world.copy()

    if extract_root_motion:
        src_root_pb = source_arm.pose.bones[root_pair[0]]
        src_root_world = source_arm.matrix_world @ src_root_pb.head
        delta_world = (facing_yaw_R @ (src_root_world - src_root_rest_world)) * scale_ratio
        if in_place:
            delta_world.x = 0.0
            delta_world.y = 0.0
        desired_world = tgt_root_rest_world + delta_world
        tgt_root_pb.location = tgt_rest_mat.inverted() @ desired_world

    computed_tgt_world_R: Dict[str, Matrix] = {}
    for src_name, tgt_name in valid_pairs:
        tgt_pb = target_arm.pose.bones[tgt_name]
        tgt_bone = target_arm.data.bones[tgt_name]

        R_src_world = source_rotations[src_name]
        R_tgt_desired_world = R_src_world @ rest_deltas[tgt_name]

        if tgt_bone.parent is not None:
            p_name = tgt_bone.parent.name
            if p_name in computed_tgt_world_R:
                R_parent_world = computed_tgt_world_R[p_name]
            else:
                R_parent_world = fallback_parent_rotations[p_name]
            R_parent_rest = (target_arm.matrix_world @ tgt_bone.parent.matrix_local).to_3x3().normalized()
            R_bone_rest = (target_arm.matrix_world @ tgt_bone.matrix_local).to_3x3().normalized()
            R_local_rest = R_parent_rest.inverted() @ R_bone_rest
            R_basis = R_local_rest.inverted() @ R_parent_world.inverted() @ R_tgt_desired_world
        else:
            R_bone_rest = (target_arm.matrix_world @ tgt_bone.matrix_local).to_3x3().normalized()
            R_basis = R_bone_rest.inverted() @ R_tgt_desired_world

        tgt_pb.rotation_mode = "QUATERNION"
        tgt_pb.rotation_quaternion = R_basis.to_quaternion().normalized()
        if insert_frame is not None:
            tgt_pb.keyframe_insert(data_path="rotation_quaternion", frame=insert_frame)
        computed_tgt_world_R[tgt_name] = R_tgt_desired_world

    # Automatic Target-Character Floor Contact Grounding (eliminates floating & floor-sinking on ANY rig proportions!)
    if extract_root_motion and ground_to_floor and tgt_foot_names:
        if update_fn is not None:
            update_fn()
        tgt_min_z = _compute_pose_min_foot_z(target_arm, tgt_foot_names)
        if tgt_min_z is not None:
            src_clearance = 0.0
            if src_foot_names:
                src_min_z = _compute_pose_min_foot_z(source_arm, src_foot_names)
                if src_min_z is not None:
                    raw_clr = max(0.0, src_min_z - src_rest_floor_z) * scale_ratio
                    # Snap small <4.5cm stance clearance directly to floor Z=0; preserve real jumps!
                    src_clearance = 0.0 if raw_clr < 0.045 else (raw_clr - 0.045)
            z_corr = (tgt_rest_floor_z + src_clearance) - tgt_min_z
            desired_world.z += z_corr
            tgt_root_pb.location = tgt_rest_mat.inverted() @ desired_world

    if extract_root_motion and insert_frame is not None:
        tgt_root_pb.keyframe_insert(data_path="location", frame=insert_frame)


def retarget_live_frame(
    source_arm: bpy.types.Object,
    target_arm: bpy.types.Object,
    bone_map: Dict[str, str],
    scale_compensation: bool = True,
    extract_root_motion: bool = True,
    in_place: bool = False,
    ground_to_floor: bool = True,
    context: Optional[bpy.types.Context] = None,
) -> int:
    """Retarget a single real-time live frame from `source_arm` onto `target_arm` (no keyframes inserted)."""
    ctx = context or bpy.context
    if not bone_map:
        return 0
    ctx.view_layer.update()
    valid_pairs, rest_deltas, root_pair, ratio, src_rest, tgt_rest, R_yaw = _prepare_retarget_pairs(
        source_arm, target_arm, bone_map
    )
    src_foot_keys = ["left_toe", "right_toe", "left_foot", "right_foot"]
    src_feet = _collect_foot_bone_names(source_arm, src_foot_keys)
    tgt_feet = _collect_foot_bone_names(target_arm, [bone_map.get(k, "") for k in src_foot_keys])
    _apply_retarget_single_step(
        source_arm=source_arm,
        target_arm=target_arm,
        valid_pairs=valid_pairs,
        rest_deltas=rest_deltas,
        root_pair=root_pair,
        scale_ratio=ratio if scale_compensation else 1.0,
        src_root_rest_world=src_rest,
        tgt_root_rest_world=tgt_rest,
        facing_yaw_R=R_yaw,
        extract_root_motion=extract_root_motion,
        in_place=in_place,
        insert_frame=None,
        ground_to_floor=ground_to_floor,
        src_foot_names=src_feet,
        tgt_foot_names=tgt_feet,
        src_rest_floor_z=_compute_rest_floor_z(source_arm, src_feet),
        tgt_rest_floor_z=_compute_rest_floor_z(target_arm, tgt_feet),
        update_fn=ctx.view_layer.update,
    )
    return len(valid_pairs)


def retarget_and_bake(
    source_arm: bpy.types.Object,
    target_arm: bpy.types.Object,
    bone_map: Dict[str, str],
    frame_start: int,
    frame_end: int,
    scale_compensation: bool = True,
    extract_root_motion: bool = True,
    in_place: bool = False,
    ground_to_floor: bool = True,
    action_name: Optional[str] = None,
    push_nla: bool = True,
    context: Optional[bpy.types.Context] = None,
) -> bpy.types.Action:
    """Retarget motion (body + fingers) from `source_arm` onto `target_arm` and bake editable keyframes."""
    ctx = context or bpy.context
    if not bone_map:
        raise ValueError("Bone mapping is empty; run Auto-Match or map bones manually first.")

    valid_pairs, rest_deltas, root_pair, ratio, src_rest, tgt_rest, R_yaw = _prepare_retarget_pairs(
        source_arm, target_arm, bone_map
    )
    scale_ratio = ratio if scale_compensation else 1.0

    src_foot_keys = ["left_toe", "right_toe", "left_foot", "right_foot"]
    src_feet = _collect_foot_bone_names(source_arm, src_foot_keys)
    tgt_feet = _collect_foot_bone_names(target_arm, [bone_map.get(k, "") for k in src_foot_keys])
    src_floor_z = _compute_rest_floor_z(source_arm, src_feet)
    tgt_floor_z = _compute_rest_floor_z(target_arm, tgt_feet)

    act_name = action_name or f"Retargeted_{source_arm.name}_to_{target_arm.name}"
    action = _ensure_action_on_obj(target_arm, act_name)

    for _, tgt_name in valid_pairs:
        target_arm.pose.bones[tgt_name].rotation_mode = "QUATERNION"

    # Temporarily pause viewport skinning on high-poly meshes driven by target_arm during bake (10x-15x faster!)
    paused_mods: List[Tuple[bpy.types.Modifier, bool]] = []
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            for mod in obj.modifiers:
                if mod.type == "ARMATURE" and mod.object == target_arm and mod.show_viewport:
                    paused_mods.append((mod, mod.show_viewport))
                    mod.show_viewport = False

    n_frames = max(0, frame_end - frame_start + 1)
    frame_nums = [float(f) for f in range(frame_start, frame_end + 1)]
    root_loc_buf = [[0.0] * n_frames for _ in range(3)]
    rot_bufs: Dict[str, List[List[float]]] = {
        tgt_name: [[0.0] * n_frames for _ in range(4)] for _, tgt_name in valid_pairs
    }
    tgt_root_pb = target_arm.pose.bones[root_pair[1]]

    depsgraph = ctx.evaluated_depsgraph_get()
    try:
        for idx_f, frame in enumerate(range(frame_start, frame_end + 1)):
            ctx.scene.frame_set(frame)
            depsgraph.update()
            _apply_retarget_single_step(
                source_arm=source_arm,
                target_arm=target_arm,
                valid_pairs=valid_pairs,
                rest_deltas=rest_deltas,
                root_pair=root_pair,
                scale_ratio=scale_ratio,
                src_root_rest_world=src_rest,
                tgt_root_rest_world=tgt_rest,
                facing_yaw_R=R_yaw,
                extract_root_motion=extract_root_motion,
                in_place=in_place,
                insert_frame=None,  # Batch-written below via C-level foreach_set (~60x faster!)
                ground_to_floor=ground_to_floor,
                src_foot_names=src_feet,
                tgt_foot_names=tgt_feet,
                src_rest_floor_z=src_floor_z,
                tgt_rest_floor_z=tgt_floor_z,
                update_fn=depsgraph.update,
            )
            if extract_root_motion:
                loc = tgt_root_pb.location
                root_loc_buf[0][idx_f] = loc.x
                root_loc_buf[1][idx_f] = loc.y
                root_loc_buf[2][idx_f] = loc.z
            for _, tgt_name in valid_pairs:
                q = target_arm.pose.bones[tgt_name].rotation_quaternion
                rb = rot_bufs[tgt_name]
                rb[0][idx_f] = q.w
                rb[1][idx_f] = q.x
                rb[2][idx_f] = q.y
                rb[3][idx_f] = q.z

        if extract_root_motion:
            rdp = tgt_root_pb.path_from_id("location")
            for ax in range(3):
                write_fcurve_batch(action, target_arm, rdp, ax, frame_nums, root_loc_buf[ax])
        for _, tgt_name in valid_pairs:
            qdp = target_arm.pose.bones[tgt_name].path_from_id("rotation_quaternion")
            rb = rot_bufs[tgt_name]
            # Matrix extraction may choose q or -q on different frames. Preserve a
            # continuous hemisphere so fractional-frame evaluation cannot pass through zero.
            for sample in range(1, len(frame_nums)):
                if sum(rb[axis][sample - 1] * rb[axis][sample] for axis in range(4)) < 0:
                    for axis in range(4):
                        rb[axis][sample] = -rb[axis][sample]
            for ax in range(4):
                write_fcurve_batch(action, target_arm, qdp, ax, frame_nums, rb[ax])
    finally:
        for mod, was_vis in paused_mods:
            mod.show_viewport = was_vis
        depsgraph.update()

    if push_nla:
        push_action_to_nla(target_arm, track_name=f"Retarget_{source_arm.name}")

    return action
