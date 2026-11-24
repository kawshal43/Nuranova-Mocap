"""Facial 52-blendshape shape-key retargeting and 16-bone hand/finger animation import."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import bpy
from mathutils import Quaternion


def apply_facial_blendshapes_to_mesh(
    mesh_obj: bpy.types.Object,
    payload: Dict[str, Any],
    gain: float = 1.0,
    custom_map: Optional[Dict[str, str]] = None,
) -> int:
    """Apply MediaPipe 52 blendshape trajectories as editable keyframes on a Blender mesh's Shape Keys.

    Returns number of shape keys successfully mapped and animated.
    """
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

    for t, frame_vals in enumerate(values):
        frame_num = t + 1
        for idx, kb in mapped_indices:
            kb.value = max(0.0, min(1.0, float(frame_vals[idx]) * gain))
            kb.keyframe_insert(data_path="value", frame=frame_num)

    return len(mapped_indices)


def apply_hand_rotations_to_armature(
    target_arm: bpy.types.Object,
    payload: Dict[str, Any],
) -> int:
    """Apply left/right 16-bone finger quaternion animations onto matching target armature finger bones."""
    if "hands" not in payload or payload["hands"] is None or target_arm is None:
        return 0

    hand_bone_suffixes = [
        "hand",
        "thumb_01", "thumb_02", "thumb_03",
        "index_01", "index_02", "index_03",
        "middle_01", "middle_02", "middle_03",
        "ring_01", "ring_02", "ring_03",
        "pinky_01", "pinky_02", "pinky_03",
    ]

    pb_lower = {pb.name.lower(): pb for pb in target_arm.pose.bones}
    animated = 0

    for side_key, side_tag in [("left_local_rotations_xyzw", "l"), ("right_local_rotations_xyzw", "r")]:
        seq = payload["hands"].get(side_key)
        if not seq:
            continue
        side_word = "left" if side_tag == "l" else "right"
        for b_idx, suffix in enumerate(hand_bone_suffixes):
            candidates = [
                f"{side_word}_{suffix}",
                f"{suffix}.{side_tag}",
                f"f_{suffix}.{side_tag}",
                f"def-f_{suffix}.{side_tag}",
            ]
            pb = None
            for cand in candidates:
                if cand in pb_lower:
                    pb = pb_lower[cand]
                    break
            if pb is None:
                continue
            pb.rotation_mode = "QUATERNION"
            for t, frame_rots in enumerate(seq):
                xyzw = frame_rots[b_idx]
                pb.rotation_quaternion = Quaternion((xyzw[3], xyzw[0], xyzw[1], xyzw[2])).normalized()
                pb.keyframe_insert(data_path="rotation_quaternion", frame=t + 1)
            animated += 1

    return animated
