"""Facial 52-blendshape shape-key retargeting and 30-bone hand/finger animation (baked & real-time live)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import bpy
from mathutils import Quaternion

# 16 canonical hand/finger indices matching engine/hands/hand_tracker.py HAND_BONE_NAMES
HAND_BONE_SUFFIXES: List[str] = [
    "hand",
    "thumb_01", "thumb_02", "thumb_03",
    "index_01", "index_02", "index_03",
    "middle_01", "middle_02", "middle_03",
    "ring_01", "ring_02", "ring_03",
    "pinky_01", "pinky_02", "pinky_03",
]

_MIXAMO_SEG = {"01": "1", "02": "2", "03": "3"}
_VRM_SEG = {"01": "proximal", "02": "intermediate", "03": "distal"}


def _finger_bone_candidates(side_word: str, side_tag: str, suffix: str) -> List[str]:
    """Generate comprehensive bone-name candidates across Nuranova, Rigify, Mixamo, Unreal, and VRM/Unity rigs."""
    if suffix == "hand":
        return [
            f"{side_word}_hand",
            f"hand.{side_tag}",
            f"def-hand.{side_tag}",
            f"hand_{side_tag}",
            f"{side_word}hand",
            f"mixamorig:{side_word}hand",
        ]
    finger, seg = suffix.split("_")  # e.g. "index", "01"
    m_num = _MIXAMO_SEG[seg]
    v_seg = _VRM_SEG[seg]
    # Rigify uses `thumb.01.L` for thumb and `f_index.01.L` for the other 4 fingers
    rigify_base = f"{finger}.{seg}.{side_tag}" if finger == "thumb" else f"f_{finger}.{seg}.{side_tag}"
    return [
        f"{side_word}_{suffix}",                    # Nuranova: left_index_01
        f"{side_word}_{finger}_{m_num}",            # left_index_1
        rigify_base,                                # Rigify: f_index.01.l / thumb.01.l
        f"def-{rigify_base}",                       # Rigify deform: def-f_index.01.l
        f"f_{suffix}.{side_tag}",                   # Legacy Rigify underscore: f_index_01.l
        f"def-f_{suffix}.{side_tag}",               # Legacy test rig: def-f_index_01.l
        f"{suffix}.{side_tag}",                     # index_01.l
        f"{suffix}_{side_tag}",                     # Unreal: index_01_l
        f"{finger}_{m_num}_{side_tag}",             # index_1_l
        f"mixamorig:{side_word}hand{finger}{m_num}",# Mixamo: mixamorig:lefthandindex1
        f"{side_word}hand{finger}{m_num}",          # Mixamo no prefix: lefthandindex1
        f"{side_word}{finger}{v_seg}",              # VRM/Unity: leftindexproximal
        f"{finger}{v_seg}.{side_tag}",              # Unity dot: indexproximal.l
        f"{finger}_{v_seg}.{side_tag}",             # index_proximal.l
    ]


def _resolve_hand_pose_bones(target_arm: bpy.types.Object) -> Dict[str, List[Optional[bpy.types.PoseBone]]]:
    """Map left/right 16 canonical hand/finger indices to matching PoseBones on `target_arm`."""
    pb_lower = {pb.name.lower(): pb for pb in target_arm.pose.bones}
    resolved: Dict[str, List[Optional[bpy.types.PoseBone]]] = {}
    for side_key, side_tag, side_word in [
        ("left_local_rotations_xyzw", "l", "left"),
        ("right_local_rotations_xyzw", "r", "right"),
    ]:
        bones_list: List[Optional[bpy.types.PoseBone]] = []
        for suffix in HAND_BONE_SUFFIXES:
            matched: Optional[bpy.types.PoseBone] = None
            for cand in _finger_bone_candidates(side_word, side_tag, suffix):
                if cand in pb_lower:
                    matched = pb_lower[cand]
                    break
            bones_list.append(matched)
        resolved[side_key] = bones_list
    return resolved


def apply_facial_blendshapes_to_mesh(
    mesh_obj: bpy.types.Object,
    payload: Dict[str, Any],
    gain: float = 1.0,
    custom_map: Optional[Dict[str, str]] = None,
) -> int:
    """Apply MediaPipe 52 blendshape trajectories as editable keyframes on a Blender mesh's Shape Keys."""
    if "face" not in payload or payload["face"] is None:
        return 0
    if mesh_obj is None or mesh_obj.type != "MESH" or mesh_obj.data.shape_keys is None:
        return 0

    face_data = payload["face"]
    names: List[str] = face_data.get("blendshape_names", [])
    values: List[List[float]] = face_data.get("blendshapes", [])
    key_blocks = mesh_obj.data.shape_keys.key_blocks
    kb_lower = {kb.name.lower(): kb for kb in key_blocks}

    mapped_indices: List[tuple[int, bpy.types.ShapeKey]] = []
    for idx, bs_name in enumerate(names):
        target_name = custom_map.get(bs_name, bs_name) if custom_map else bs_name
        kb = key_blocks.get(target_name) or kb_lower.get(target_name.lower())
        if kb is not None and kb.name.lower() != "basis":
            mapped_indices.append((idx, kb))

    if not mapped_indices or not values:
        return 0
    # Capture samples use linear curves. Bezier handles can introduce facial
    # overshoot that disappears or changes when FBX/glTF resample the take.
    from .source_armature import write_fcurve_batch
    keys = mesh_obj.data.shape_keys
    keys.animation_data_create()
    previous = keys.animation_data.action
    if previous is not None:
        previous.use_fake_user = True
    action = bpy.data.actions.new(mesh_obj.name + "_FaceCapture")
    keys.animation_data.action = action
    keys.animation_data.use_nla = False
    if hasattr(action,"slots"):
        keys.animation_data.action_slot = action.slots.new(id_type="KEY",name=keys.name)
    frames = [float(t+1) for t in range(len(values))]
    for idx, key in mapped_indices:
        channel = [max(0.,min(1.,float(frame[idx])*gain)) for frame in values]
        write_fcurve_batch(action,keys,key.path_from_id("value"),0,frames,channel)

    return len(mapped_indices)


def apply_hand_rotations_to_armature(
    target_arm: bpy.types.Object,
    payload: Dict[str, Any],
    include_wrist: bool = False,
) -> int:
    """Apply left/right 15 finger (and optional wrist) quaternion animations onto matching target armature bones."""
    if "hands" not in payload or payload["hands"] is None or target_arm is None:
        return 0

    resolved = _resolve_hand_pose_bones(target_arm)
    animated = 0
    start_idx = 0 if include_wrist else 1

    # Ensure target_arm has an Action so we can batch-write FCurves (~60x faster than keyframe_insert loops!)
    if target_arm.animation_data is None:
        target_arm.animation_data_create()
    action = target_arm.animation_data.action
    if action is None:
        action = bpy.data.actions.new(name=f"{target_arm.name}_HandsAction")
        target_arm.animation_data.action = action
    from .source_armature import write_fcurve_batch

    for side_key, bones_list in resolved.items():
        seq = payload["hands"].get(side_key)
        if not seq:
            continue
        T = len(seq)
        frame_nums = [float(t + 1) for t in range(T)]
        for b_idx in range(start_idx, len(bones_list)):
            pb = bones_list[b_idx]
            if pb is None:
                continue
            pb.rotation_mode = "QUATERNION"
            ws, xs, ys, zs = [0.0] * T, [0.0] * T, [0.0] * T, [0.0] * T
            for t, frame_rots in enumerate(seq):
                xyzw = frame_rots[b_idx]
                q = Quaternion((xyzw[3], xyzw[0], xyzw[1], xyzw[2])).normalized()
                ws[t], xs[t], ys[t], zs[t] = q.w, q.x, q.y, q.z
            dp = pb.path_from_id("rotation_quaternion")
            write_fcurve_batch(action, target_arm, dp, 0, frame_nums, ws)
            write_fcurve_batch(action, target_arm, dp, 1, frame_nums, xs)
            write_fcurve_batch(action, target_arm, dp, 2, frame_nums, ys)
            write_fcurve_batch(action, target_arm, dp, 3, frame_nums, zs)
            animated += 1

    return animated


def apply_live_hand_rotations(
    target_arm: bpy.types.Object,
    hands_payload: Optional[Dict[str, Any]],
    include_wrist: bool = False,
) -> int:
    """Apply a single live frame of left/right 30-finger rotations onto `target_arm` without inserting keyframes."""
    if not hands_payload or target_arm is None:
        return 0
    resolved = _resolve_hand_pose_bones(target_arm)
    updated = 0
    start_idx = 0 if include_wrist else 1

    for side_key, bones_list in resolved.items():
        frame_rots = hands_payload.get(side_key)
        if not frame_rots or len(frame_rots) < 16:
            continue
        for b_idx in range(start_idx, len(bones_list)):
            pb = bones_list[b_idx]
            if pb is None:
                continue
            xyzw = frame_rots[b_idx]
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = Quaternion((xyzw[3], xyzw[0], xyzw[1], xyzw[2])).normalized()
            updated += 1

    return updated
