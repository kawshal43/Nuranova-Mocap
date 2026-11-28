"""Source armature creation, editable keyframe animation import (body + 30 fingers), and live pose driver."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import bpy
from mathutils import Matrix, Quaternion, Vector

from .auto_rig import append_finger_edit_bones
from .facial_hands import apply_hand_rotations_to_armature, apply_live_hand_rotations
from .animation_utils import continuous_quaternions


def _get_action_fcurves(action: bpy.types.Action, obj: bpy.types.Object):
    """Return the writable FCurves collection for Blender 4.x and Blender 5.x slotted Actions."""
    if hasattr(action, "layers") and hasattr(action, "slots"):
        slot = getattr(obj.animation_data, "action_slot", None) if obj.animation_data else None
        if slot is None:
            slot = action.slots[0] if len(action.slots) > 0 else action.slots.new(id_type="OBJECT", name=obj.name)
            if obj.animation_data is not None and hasattr(obj.animation_data, "action_slot"):
                try:
                    obj.animation_data.action_slot = slot
                except Exception:
                    pass
        layer = action.layers[0] if len(action.layers) > 0 else action.layers.new("Layer")
        strip = layer.strips[0] if len(layer.strips) > 0 else layer.strips.new(type="KEYFRAME")
        cbag = strip.channelbag(slot, ensure=True)
        return cbag.fcurves
    return action.fcurves


def write_fcurve_batch(
    action: bpy.types.Action,
    obj: bpy.types.Object,
    data_path: str,
    index: int,
    frames: List[float],
    values: List[float],
) -> None:
    """Write an entire animation curve in one C-level `foreach_set` call (~60x faster than keyframe_insert loops!)."""
    n = len(frames)
    if n == 0:
        return
    fcurves = _get_action_fcurves(action, obj)
    fc = fcurves.find(data_path, index=index)
    if fc is None:
        fc = fcurves.new(data_path=data_path, index=index)
    else:
        while len(fc.keyframe_points) > 0:
            fc.keyframe_points.remove(fc.keyframe_points[-1], fast=True)
    fc.keyframe_points.add(n)
    flat = [0.0] * (2 * n)
    flat[0::2] = frames
    flat[1::2] = values
    fc.keyframe_points.foreach_set("co", flat)
    # Dense capture samples must not gain Bezier overshoot between integer frames.
    for point in fc.keyframe_points:
        point.interpolation = "LINEAR"
    fc.update()


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
        # Keep the editable active action; the NLA copy is an archived alternative.
        # Evaluating both copies can double additive transforms or surprise later edits.
        track.mute = True
    except Exception:
        pass
    return track


def load_nmocap_json(filepath: str | Path) -> Dict[str, Any]:
    return json.loads(Path(filepath).read_text(encoding="utf-8"))


def _build_armature_skeleton(
    name: str,
    bone_names: List[str],
    parent_indices: List[int],
    rest_offsets: List[List[float]],
    include_fingers: bool,
    ctx: bpy.types.Context,
) -> tuple[bpy.types.Object, List[Vector]]:
    arm_data = bpy.data.armatures.new(name + "_Data")
    arm_obj = bpy.data.objects.new(name, arm_data)
    arm_obj.show_in_front = True
    ctx.collection.objects.link(arm_obj)
    ctx.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)

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

    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones: List[bpy.types.EditBone] = []
    for i, bname in enumerate(bone_names):
        eb = arm_data.edit_bones.new(bname)
        eb.head = rest_heads[i]
        kids = children[i]
        if kids:
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

    if include_fingers:
        scale_h = max(0.3, float(rest_heads[0].z) / 0.96)
        append_finger_edit_bones(arm_data, "left_hand", "left", scale=scale_h)
        append_finger_edit_bones(arm_data, "right_hand", "right", scale=scale_h)

    bpy.ops.object.mode_set(mode="OBJECT")
    for pb in arm_obj.pose.bones:
        pb.rotation_mode = "QUATERNION"
    return arm_obj, rest_heads


def create_source_armature_from_payload(
    payload: Dict[str, Any],
    armature_name: Optional[str] = None,
    in_place: bool = False,
    push_nla: bool = True,
    include_fingers: bool = True,
    context: Optional[bpy.types.Context] = None,
) -> bpy.types.Object:
    """Build a calibrated Z-Up source armature (25 body + optional 30 fingers) and populate editable keyframes."""
    ctx = context or bpy.context
    skel = payload["skeleton"]
    bone_names: List[str] = list(skel["bone_names"])
    parent_indices: List[int] = list(skel["parent_indices"])
    rest_offsets: List[List[float]] = skel["rest_offsets"]
    frames = payload["frames"]
    fps = float(payload.get("fps", 60.0))
    take_id = str(payload.get("take_id", "take_01"))
    has_hands = payload.get("hands") is not None

    name = armature_name or f"Nuranova_Source_{take_id}"
    arm_obj, rest_heads = _build_armature_skeleton(
        name=name,
        bone_names=bone_names,
        parent_indices=parent_indices,
        rest_offsets=rest_offsets,
        include_fingers=bool(include_fingers and has_hands),
        ctx=ctx,
    )

    ctx.scene.render.fps = int(round(fps))
    ctx.scene.render.fps_base = int(round(fps)) / fps
    root_trans = frames["root_translation"]
    local_rots = continuous_quaternions(frames["local_rotations_xyzw"])
    T = len(root_trans)
    ctx.scene.frame_start = 1
    ctx.scene.frame_end = max(1, T)

    action = _ensure_action_on_obj(arm_obj, f"Nuranova_{take_id}_RawAction")

    arm_data = arm_obj.data
    rest_mats: List[Matrix] = [arm_data.bones[bname].matrix_local.to_3x3() for bname in bone_names]
    rest_mats_inv: List[Matrix] = [m.inverted() for m in rest_mats]
    rest_quats: List[Quaternion] = [m.to_quaternion() for m in rest_mats]
    rest_quats_inv: List[Quaternion] = [q.inverted() for q in rest_quats]

    rest_root_pos = rest_heads[0].copy()
    first_root = Vector(root_trans[0]) if T > 0 else rest_root_pos.copy()

    frame_nums = [float(t + 1) for t in range(T)]
    root_loc_ch = [[0.0] * T for _ in range(3)]
    rot_ch: List[List[List[float]]] = [[[0.0] * T for _ in range(4)] for _ in range(len(bone_names))]

    for t in range(T):
        rt = Vector(root_trans[t])
        if in_place:
            rt = Vector((first_root.x, first_root.y, rt.z))
        loc = rest_mats_inv[0] @ (rt - rest_root_pos)
        root_loc_ch[0][t] = loc.x
        root_loc_ch[1][t] = loc.y
        root_loc_ch[2][t] = loc.z

        frame_rots = local_rots[t]
        for b_idx in range(len(bone_names)):
            xyzw = frame_rots[b_idx]
            q_delta = Quaternion((xyzw[3], xyzw[0], xyzw[1], xyzw[2])).normalized()
            q_b = rest_quats_inv[b_idx] @ q_delta @ rest_quats[b_idx]
            rot_ch[b_idx][0][t] = q_b.w
            rot_ch[b_idx][1][t] = q_b.x
            rot_ch[b_idx][2][t] = q_b.y
            rot_ch[b_idx][3][t] = q_b.z

    root_dp = f'pose.bones["{bone_names[0]}"].location'
    for ax in range(3):
        write_fcurve_batch(action, arm_obj, root_dp, ax, frame_nums, root_loc_ch[ax])
    for b_idx, bname in enumerate(bone_names):
        rot_dp = f'pose.bones["{bname}"].rotation_quaternion'
        for ax in range(4):
            write_fcurve_batch(action, arm_obj, rot_dp, ax, frame_nums, rot_ch[b_idx][ax])

    # Animate all 30 finger bones on the Source Armature too when hand data is present!
    if include_fingers and has_hands:
        apply_hand_rotations_to_armature(arm_obj, payload, include_wrist=False)

    arm_obj["nuranova_take_id"] = take_id
    arm_obj["nuranova_fps"] = fps
    arm_obj["nuranova_rest_hip_height"] = float(rest_root_pos.z)
    arm_obj["nuranova_source_json"] = json.dumps({
        "has_face": payload.get("face") is not None,
        "has_hands": has_hands,
    })

    if push_nla:
        push_action_to_nla(arm_obj, track_name=f"Source_{take_id}")

    return arm_obj


def create_live_preview_armature(
    live_payload: Dict[str, Any],
    armature_name: str = "Nuranova_Live_Preview",
    include_fingers: bool = True,
    context: Optional[bpy.types.Context] = None,
) -> bpy.types.Object:
    """Create a real-time 55-bone (25 body + 30 finger) Live Preview Source Armature."""
    ctx = context or bpy.context
    skel = live_payload["skeleton"]
    arm_obj, rest_heads = _build_armature_skeleton(
        name=armature_name,
        bone_names=list(skel["bone_names"]),
        parent_indices=list(skel["parent_indices"]),
        rest_offsets=skel["rest_offsets"],
        include_fingers=include_fingers,
        ctx=ctx,
    )
    arm_obj["nuranova_live_rig"] = True
    arm_obj["nuranova_rest_hip_height"] = float(rest_heads[0].z)
    return arm_obj


def apply_live_frame_to_source_armature(
    arm_obj: bpy.types.Object,
    live_payload: Dict[str, Any],
    in_place: bool = False,
    include_fingers: bool = True,
) -> None:
    """Drive a Blender source armature in real time (zero keyframes!) from `/api/v1/bridge/live`."""
    if arm_obj is None or arm_obj.type != "ARMATURE":
        return
    skel = live_payload.get("skeleton", {})
    bone_names: List[str] = list(skel.get("bone_names", []))
    root_trans = live_payload.get("root_translation")
    local_rots = live_payload.get("local_rotations_xyzw")
    if not bone_names or not root_trans or not local_rots:
        return

    arm_data = arm_obj.data
    root_bname = bone_names[0]
    if root_bname in arm_data.bones and root_bname in arm_obj.pose.bones:
        rest_root_pos = arm_data.bones[root_bname].head_local.copy()
        rt = Vector(root_trans)
        if in_place:
            rt = Vector((rest_root_pos.x, rest_root_pos.y, rt.z))
        root_mat = arm_data.bones[root_bname].matrix_local.to_3x3()
        arm_obj.pose.bones[root_bname].location = root_mat.inverted() @ (rt - rest_root_pos)

    for b_idx, bname in enumerate(bone_names):
        if b_idx >= len(local_rots) or bname not in arm_data.bones or bname not in arm_obj.pose.bones:
            continue
        xyzw = local_rots[b_idx]
        q_delta = Quaternion((xyzw[3], xyzw[0], xyzw[1], xyzw[2])).normalized()
        q_rest = arm_data.bones[bname].matrix_local.to_3x3().to_quaternion()
        pb = arm_obj.pose.bones[bname]
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = q_rest.inverted() @ q_delta @ q_rest

    if include_fingers and live_payload.get("hands"):
        apply_live_hand_rotations(arm_obj, live_payload["hands"], include_wrist=False)
