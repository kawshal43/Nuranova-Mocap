"""Character-agnostic Rigify & Custom Armature retargeting with rest-pose delta & scale compensation,
editable keyframe baking, and reusable mapping profiles.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import bpy
from mathutils import Matrix, Quaternion, Vector

from .source_armature import _ensure_action_on_obj, push_action_to_nla

# Fuzzy synonym patterns supporting Rigify, Mixamo, Unreal Mannequin, VRM, and custom rigs
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


def auto_match_bones(source_arm: bpy.types.Object, target_arm: bpy.types.Object) -> Dict[str, str]:
    """Automatically map Nuranova source bones to any target rig (Rigify, Mixamo, custom)."""
    target_names = [b.name for b in target_arm.data.bones]
    target_lower = {name.lower(): name for name in target_names}
    mapping: Dict[str, str] = {}
    used_targets = set()

    for src_bone in source_arm.data.bones:
        s_name = src_bone.name
        s_low = s_name.lower()
        # 1. Exact match
        if s_low in target_lower and target_lower[s_low] not in used_targets:
            mapping[s_name] = target_lower[s_low]
            used_targets.add(target_lower[s_low])
            continue
        # 2. Synonym exact or substring match
        syns = BONE_SYNONYMS.get(s_low, [])
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


def retarget_and_bake(
    source_arm: bpy.types.Object,
    target_arm: bpy.types.Object,
    bone_map: Dict[str, str],
    frame_start: int,
    frame_end: int,
    scale_compensation: bool = True,
    extract_root_motion: bool = True,
    in_place: bool = False,
    action_name: Optional[str] = None,
    push_nla: bool = True,
    context: Optional[bpy.types.Context] = None,
) -> bpy.types.Action:
    """Retarget motion from `source_arm` onto arbitrary `target_arm` with rest-pose delta compensation,
    scale compensation, and direct baking into pure editable quaternion keyframes!

    Handles rigs with completely different local bone roll/axes and different stylized proportions.
    """
    ctx = context or bpy.context
    if not bone_map:
        raise ValueError("Bone mapping is empty; run Auto-Match or map bones manually first.")

    # Precompute rest-pose world rotation matrices for every mapped pair
    # Rest-pose delta: R_delta = R_src_rest^-1 @ R_tgt_rest
    # On frame t: R_tgt_world(t) = R_src_world(t) @ R_delta
    src_world_R = source_arm.matrix_world.to_3x3().normalized()
    tgt_world_R = target_arm.matrix_world.to_3x3().normalized()

    rest_deltas: Dict[str, Matrix] = {}
    valid_pairs: List[Tuple[str, str]] = []
    for src_name, tgt_name in bone_map.items():
        if src_name in source_arm.pose.bones and tgt_name in target_arm.pose.bones:
            R_src_rest = src_world_R @ source_arm.data.bones[src_name].matrix_local.to_3x3()
            R_tgt_rest = tgt_world_R @ target_arm.data.bones[tgt_name].matrix_local.to_3x3()
            rest_deltas[tgt_name] = R_src_rest.inverted() @ R_tgt_rest
            valid_pairs.append((src_name, tgt_name))

    # Order target bones top-down by hierarchy depth so parents update before children
    def bone_depth(b: bpy.types.Bone) -> int:
        d = 0
        cur = b.parent
        while cur is not None:
            d += 1
            cur = cur.parent
        return d

    valid_pairs.sort(key=lambda pair: bone_depth(target_arm.data.bones[pair[1]]))

    # Compute proportion scale factor from hip heights
    root_pair = next(((s, t) for s, t in valid_pairs if s.lower() == "hips"), valid_pairs[0])
    src_hip_h = max((source_arm.matrix_world @ source_arm.data.bones[root_pair[0]].head_local).z, 0.1)
    tgt_hip_h = max((target_arm.matrix_world @ target_arm.data.bones[root_pair[1]].head_local).z, 0.1)
    scale_ratio = (tgt_hip_h / src_hip_h) if scale_compensation else 1.0

    src_root_rest_world = source_arm.matrix_world @ source_arm.data.bones[root_pair[0]].head_local
    tgt_root_rest_world = target_arm.matrix_world @ target_arm.data.bones[root_pair[1]].head_local

    act_name = action_name or f"Retargeted_{source_arm.name}_to_{target_arm.name}"
    action = _ensure_action_on_obj(target_arm, act_name)

    for _, tgt_name in valid_pairs:
        target_arm.pose.bones[tgt_name].rotation_mode = "QUATERNION"

    first_src_root_world: Optional[Vector] = None
    depsgraph = ctx.evaluated_depsgraph_get()

    for frame in range(frame_start, frame_end + 1):
        ctx.scene.frame_set(frame)
        depsgraph.update()

        # Root translation with scale compensation & optional in-place mode
        if extract_root_motion:
            src_root_pb = source_arm.pose.bones[root_pair[0]]
            tgt_root_pb = target_arm.pose.bones[root_pair[1]]
            src_root_world = source_arm.matrix_world @ src_root_pb.head
            if first_src_root_world is None:
                first_src_root_world = src_root_world.copy()
            delta_world = (src_root_world - src_root_rest_world) * scale_ratio
            if in_place:
                delta_world.x = 0.0
                delta_world.y = 0.0
            desired_world = tgt_root_rest_world + delta_world
            # Convert world translation delta to target root pose bone local space
            tgt_rest_mat = target_arm.matrix_world @ target_arm.data.bones[root_pair[1]].matrix_local
            tgt_root_pb.location = tgt_rest_mat.inverted() @ desired_world
            tgt_root_pb.keyframe_insert(data_path="location", frame=frame)

        # Solve rotations hierarchically (top-down) so parent world matrices are exact
        computed_tgt_world_R: Dict[str, Matrix] = {}
        for src_name, tgt_name in valid_pairs:
            src_pb = source_arm.pose.bones[src_name]
            tgt_pb = target_arm.pose.bones[tgt_name]
            tgt_bone = target_arm.data.bones[tgt_name]

            R_src_world = (source_arm.matrix_world @ src_pb.matrix).to_3x3().normalized()
            R_tgt_desired_world = R_src_world @ rest_deltas[tgt_name]

            # Compute parent world rotation (from already-computed mapped parent or rest parent)
            if tgt_bone.parent is not None:
                p_name = tgt_bone.parent.name
                if p_name in computed_tgt_world_R:
                    R_parent_world = computed_tgt_world_R[p_name]
                else:
                    R_parent_world = (target_arm.matrix_world @ target_arm.pose.bones[p_name].matrix).to_3x3().normalized()
                R_parent_rest = (target_arm.matrix_world @ tgt_bone.parent.matrix_local).to_3x3().normalized()
                R_bone_rest = (target_arm.matrix_world @ tgt_bone.matrix_local).to_3x3().normalized()
                R_local_rest = R_parent_rest.inverted() @ R_bone_rest
                R_basis = R_local_rest.inverted() @ R_parent_world.inverted() @ R_tgt_desired_world
            else:
                R_bone_rest = (target_arm.matrix_world @ tgt_bone.matrix_local).to_3x3().normalized()
                R_basis = R_bone_rest.inverted() @ R_tgt_desired_world

            q_basis = R_basis.to_quaternion().normalized()
            tgt_pb.rotation_quaternion = q_basis
            tgt_pb.keyframe_insert(data_path="rotation_quaternion", frame=frame)
            computed_tgt_world_R[tgt_name] = R_tgt_desired_world

    if push_nla:
        push_action_to_nla(target_arm, track_name=f"Retarget_{source_arm.name}")

    return action
