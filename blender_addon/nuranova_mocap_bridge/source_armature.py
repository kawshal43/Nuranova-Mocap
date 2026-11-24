"""Source armature creation and editable keyframe animation import for Blender 4.5 LTS & 5.0+."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bpy
from mathutils import Matrix, Quaternion, Vector


def _ensure_action_on_obj(obj: bpy.types.Object, action_name: str) -> bpy.types.Action:
    if obj.animation_data is None:
        obj.animation_data_create()
    action = bpy.data.actions.new(name=action_name)
    obj.animation_data.action = action
    # Blender 4.4+/5.0 slotted actions compatibility
    if hasattr(action, "slots") and hasattr(obj.animation_data, "action_slot"):
        if len(action.slots) == 0:
            try:
                slot = action.slots.new(id_type="OBJECT", name=obj.name)
                obj.animation_data.action_slot = slot
            except Exception:
                pass
    return action


def push_action_to_nla(obj: bpy.types.Object, track_name: str = "NuranovaMocap") -> Optional[bpy.types.NlaTrack]:
    """Push the active Action on `obj` to a non-destructive NLA track strip."""
    if obj.animation_data is None or obj.animation_data.action is None:
        return None
    action = obj.animation_data.action
    track = obj.animation_data.nla_tracks.new()
    track.name = track_name
    start = int(action.frame_range[0]) if action.frame_range else 1
    try:
        track.strips.new(action.name, start, action)
    except Exception:
        pass
    return track


def load_nmocap_json(filepath: str | Path) -> Dict[str, Any]:
    return json.loads(Path(filepath).read_text(encoding="utf-8"))


def create_source_armature_from_payload(
    payload: Dict[str, Any],
    armature_name: Optional[str] = None,
    in_place: bool = False,
    push_nla: bool = True,
    context: Optional[bpy.types.Context] = None,
) -> bpy.types.Object:
    """Build a calibrated Z-Up source armature and populate editable quaternion keyframes."""
    ctx = context or bpy.context
    skel = payload["skeleton"]
    bone_names: List[str] = list(skel["bone_names"])
    parent_indices: List[int] = list(skel["parent_indices"])
    rest_offsets: List[List[float]] = skel["rest_offsets"]
    frames = payload["frames"]
    fps = float(payload.get("fps", 60.0))
    take_id = str(payload.get("take_id", "take_01"))

    name = armature_name or f"Nuranova_Source_{take_id}"
    arm_data = bpy.data.armatures.new(name + "_Data")
    arm_obj = bpy.data.objects.new(name, arm_data)
    ctx.collection.objects.link(arm_obj)
    ctx.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)

    # Compute rest world head positions in Z-Up meters
    B = len(bone_names)
    rest_heads: List[Vector] = [Vector((0.0, 0.0, 0.0)) for _ in range(B)]
    children: Dict[int, List[int]] = {i: [] for i in range(B)}
    for i, p in enumerate(parent_indices):
        off = Vector(rest_offsets[i])
        if p < 0:
            rest_heads[i] = off
        else:
            rest_heads[i] = rest_heads[p] + off
            children[p].append(i)

    # Enter EDIT mode to construct exact bone hierarchy
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones: List[bpy.types.EditBone] = []
    for i, bname in enumerate(bone_names):
        eb = arm_data.edit_bones.new(bname)
        eb.head = rest_heads[i]
        kids = children[i]
        if kids:
            # Point tail toward primary child (or average of children if branching)
            tail = sum((rest_heads[k] for k in kids), Vector((0.0, 0.0, 0.0))) / len(kids)
            if (tail - eb.head).length < 0.01:
                tail = eb.head + Vector((0.0, 0.0, 0.08))
            eb.tail = tail
        else:
            parent_vec = (rest_heads[i] - rest_heads[parent_indices[i]]) if parent_indices[i] >= 0 else Vector((0.0, 0.0, 0.1))
            if parent_vec.length < 0.01:
                parent_vec = Vector((0.0, 0.0, 0.08))
            eb.tail = eb.head + parent_vec.normalized() * 0.08
        edit_bones.append(eb)

    for i, p in enumerate(parent_indices):
        if p >= 0:
            edit_bones[i].parent = edit_bones[p]
            edit_bones[i].use_connect = False

    bpy.ops.object.mode_set(mode="OBJECT")

    # Configure scene FPS & frame range
    ctx.scene.render.fps = int(round(fps))
    root_trans = frames["root_translation"]
    local_rots = frames["local_rotations_xyzw"]
    T = len(root_trans)
    ctx.scene.frame_start = 1
    ctx.scene.frame_end = max(1, T)

    _ensure_action_on_obj(arm_obj, f"Nuranova_{take_id}_RawAction")

    # Set PoseBone quaternion mode and animate every frame cleanly
    # Note: In Nuranova space, local_rotations_xyzw[t][b] is the world-axis-aligned bone delta
    # relative to rest pose. Convert from canonical [x, y, z, w] -> Blender (w, x, y, z),
    # then conjugate by rest bone matrix R_rest to convert into Blender PoseBone local space:
    # q_pose_local = R_rest^-1 * q_delta_world * R_rest
    rest_mats: List[Matrix] = [arm_data.bones[bname].matrix_local.to_3x3() for bname in bone_names]
    rest_quats: List[Quaternion] = [m.to_quaternion() for m in rest_mats]
    rest_quats_inv: List[Quaternion] = [q.inverted() for q in rest_quats]

    rest_root_pos = rest_heads[0].copy()
    first_root = Vector(root_trans[0]) if T > 0 else rest_root_pos.copy()

    for pb in arm_obj.pose.bones:
        pb.rotation_mode = "QUATERNION"

    for t in range(T):
        frame_num = t + 1
        rt = Vector(root_trans[t])
        if in_place:
            rt = Vector((first_root.x, first_root.y, rt.z))
        # Root pose bone location is delta in root bone local rest space
        delta_world = rt - rest_root_pos
        root_pb = arm_obj.pose.bones[bone_names[0]]
        root_pb.location = rest_mats[0].inverted() @ delta_world
        root_pb.keyframe_insert(data_path="location", frame=frame_num)

        for b_idx, bname in enumerate(bone_names):
            xyzw = local_rots[t][b_idx]
            q_delta = Quaternion((xyzw[3], xyzw[0], xyzw[1], xyzw[2])).normalized()
            # Transform canonical local delta into Blender pose-bone local axes
            q_blender = rest_quats_inv[b_idx] @ q_delta @ rest_quats[b_idx]
            pb = arm_obj.pose.bones[bname]
            pb.rotation_quaternion = q_blender
            pb.keyframe_insert(data_path="rotation_quaternion", frame=frame_num)

    # Store custom metadata on armature for downstream retargeting & facial/hand tools
    arm_obj["nuranova_take_id"] = take_id
    arm_obj["nuranova_fps"] = fps
    arm_obj["nuranova_rest_hip_height"] = float(rest_root_pos.z)
    arm_obj["nuranova_source_json"] = json.dumps({
        "has_face": "face" in payload,
        "has_hands": "hands" in payload,
    })

    if push_nla:
        push_action_to_nla(arm_obj, track_name=f"Source_{take_id}")

    return arm_obj
