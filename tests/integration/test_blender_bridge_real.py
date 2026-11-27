"""Real headless Blender 5.0 integration test for Nuranova Mocap Bridge extension.

Verifies inside actual installed Blender:
1. Extension registration & unregistration (`nuranova_mocap_bridge`).
2. Importing `.nmocap.json` with full-body + 52 facial blendshapes + 16 finger bones.
3. Generating calibrated Z-Up source armature with editable quaternion keyframes & NLA track.
4. Auto-mapping & retargeting onto a stylized 0.72x-scale secondary character rig with rotated bone rolls.
5. Applying facial blendshapes onto mesh shape keys & finger rotations onto hand bones.
6. Exporting retargeted character animation to both `.fbx` and `.bvh` files.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from desktop.settings.app_settings import detect_blender_executable
from engine.data_format.motion_io import (
    FaceMotionTrack,
    HandMotionTrack,
    MotionCaptureDocument,
    export_blender_json,
)
from engine.face.face_tracker import MEDIAPIPE_52_BLENDSHAPES
from engine.math_core.transforms import quat_identity
from engine.skeleton.rig_definition import BONE_NAMES, PARENT_INDICES
from engine.skeleton.solver import ArticulatedSkeletonSolver
from tests.unit.test_skeleton_hybrid_cleanup import _make_synthetic_walk_sequence


def test_real_blender_import_retarget_bake_and_export(tmp_path: Path):
    blender_exe = detect_blender_executable()
    if not blender_exe or not Path(blender_exe).exists():
        pytest.skip("Blender executable not installed on this machine.")

    # Generate real solved synthetic motion with body + 52 face blendshapes + hands
    actor, _, _, _, _, body26 = _make_synthetic_walk_sequence(T=20)
    conf = np.ones((20, 26), dtype=np.float32) * 0.95
    raw_track = ArticulatedSkeletonSolver(actor).solve_sequence(body26, conf)

    bs = np.zeros((20, 52), dtype=np.float32)
    bs[:, MEDIAPIPE_52_BLENDSHAPES.index("jawOpen")] = np.linspace(0.0, 0.85, 20)
    bs[:, MEDIAPIPE_52_BLENDSHAPES.index("eyeBlinkLeft")] = np.sin(np.linspace(0, np.pi, 20))
    face = FaceMotionTrack(
        blendshapes=bs,
        blendshape_names=list(MEDIAPIPE_52_BLENDSHAPES),
        head_rotation_xyzw=np.tile(quat_identity().astype(np.float32), (20, 1)),
        confidence=np.ones(20, dtype=np.float32),
    )
    hands = HandMotionTrack(
        left_local_rotations_xyzw=np.tile(quat_identity().astype(np.float32), (20, 16, 1)),
        right_local_rotations_xyzw=np.tile(quat_identity().astype(np.float32), (20, 16, 1)),
    )

    doc = MotionCaptureDocument(
        project_id="proj_blender_test",
        session_id="sess_1",
        take_id="blender_walk_01",
        actor_id=actor.actor_id,
        fps=59.94,
        frame_timestamps_ns=np.arange(20, dtype=np.int64) * int(1e9 / 60),
        bone_names=list(BONE_NAMES),
        parent_indices=list(PARENT_INDICES),
        rest_offsets=actor.rest_offsets_m.astype(np.float32),
        raw_motion=raw_track,
        face=face,
        hands=hands,
    )
    motion_json = export_blender_json(doc, tmp_path / "blender_walk_01.nmocap.json", use_corrected=False)
    fbx_out = tmp_path / "retargeted.fbx"
    bvh_out = tmp_path / "retargeted.bvh"
    report_out = tmp_path / "blender_result.json"

    addon_root = Path(__file__).resolve().parents[2] / "blender_addon"
    script_path = tmp_path / "run_blender_test.py"
    script_path.write_text(
        f"""
import json, sys, math
from pathlib import Path
import bpy
from mathutils import Vector

sys.path.insert(0, r"{addon_root}")
import nuranova_mocap_bridge as bridge
from nuranova_mocap_bridge.auto_rig import (
    auto_rig_character_mesh,
    auto_rig_from_markers,
    create_nuranova_humanoid_rig,
)
from nuranova_mocap_bridge.smart_markers import (
    clear_markers,
    count_placed_body_landmarks,
    place_body_landmark,
    raycast_mesh_center,
    spawn_finger_markers,
)
from nuranova_mocap_bridge.source_armature import (
    _get_action_fcurves,
    apply_live_frame_to_source_armature,
    create_live_preview_armature,
    create_source_armature_from_payload,
    load_nmocap_json,
)
from nuranova_mocap_bridge.retargeting import (
    _compute_facing_yaw_matrix,
    auto_match_bones,
    retarget_and_bake,
    retarget_live_frame,
)
from nuranova_mocap_bridge.facial_hands import (
    apply_facial_blendshapes_to_mesh,
    apply_hand_rotations_to_armature,
    apply_live_hand_rotations,
)

# Reset scene
bpy.ops.wm.read_factory_settings(use_empty=True)
bridge.register()

payload = load_nmocap_json(r"{motion_json}")
src_arm = create_source_armature_from_payload(payload, push_nla=True, include_fingers=True)
assert abs(bpy.context.scene.render.fps / bpy.context.scene.render.fps_base - payload["fps"]) < 0.001
assert all(track.mute for track in src_arm.animation_data.nla_tracks)
assert all(point.interpolation == 'LINEAR' for curve in _get_action_fcurves(src_arm.animation_data.action, src_arm) for point in curve.keyframe_points)

# 1. Test Smart Landmark Marker Auto-Rig on a STYLIZED -Y-facing character mesh (big head, low shoulders)!
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(2.0, 0.0, 0.78))
char_mesh = bpy.context.active_object
char_mesh.name = "StylizedHeroMesh"
char_mesh.scale = (1.10, 0.24, 1.56)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
bpy.context.view_layer.update()

# 1a. Verify through-mesh depth centering raycast hits internal center Y ~= 0.0 (not front face Y = -0.12)
ray_hit, ray_ok = raycast_mesh_center(
    bpy.context,
    [char_mesh],
    ray_origin=Vector((2.0, -3.0, 0.90)),
    ray_dir=Vector((0.0, 1.0, 0.0)),
)
raycast_center_y = float(ray_hit.y)

# 1b. Place all 12 stylized landmarks (shoulders at Z=1.12 on a 1.56m character, facing -Y!)
cx = 2.0
stylized_clicks = [
    ("chin", Vector((cx, 0.0, 1.24))),
    ("head_top", Vector((cx, 0.0, 1.56))),
    ("chest", Vector((cx, 0.0, 1.04))),
    ("pelvis", Vector((cx, 0.0, 0.78))),
    ("shoulder", Vector((cx + 0.16, 0.0, 1.12))),   # +X = Character-Left when facing -Y!
    ("elbow", Vector((cx + 0.36, 0.02, 1.11))),
    ("wrist", Vector((cx + 0.54, 0.0, 1.11))),
    ("hand_tip", Vector((cx + 0.67, 0.0, 1.11))),
    ("hip", Vector((cx + 0.09, 0.0, 0.74))),
    ("knee", Vector((cx + 0.09, -0.02, 0.42))),
    ("ankle", Vector((cx + 0.09, 0.02, 0.08))),
    ("toe", Vector((cx + 0.09, -0.14, 0.02))),      # Toe Y < Ankle Y => auto-detects -Y facing!
]
for k, pt in stylized_clicks:
    place_body_landmark(k, pt, center_x=cx, char_height=1.56, mirror=True)
placed_cnt, total_cnt = count_placed_body_landmarks()
finger_mks = spawn_finger_markers()

# Generate Smart Marker Rig (and verify iterative re-generation cleans up modifiers cleanly!)
smart_arm_1, _ = auto_rig_from_markers([char_mesh], include_fingers=True, hide_markers_after=False)
auto_arm, weight_method = auto_rig_from_markers([char_mesh], include_fingers=True, hide_markers_after=True)
char_arm_mods = sum(1 for m in char_mesh.modifiers if m.type == "ARMATURE")
left_sh_world = auto_arm.matrix_world @ auto_arm.data.bones["left_upper_arm"].head_local
right_sh_world = auto_arm.matrix_world @ auto_arm.data.bones["right_upper_arm"].head_local
auto_map = auto_match_bones(src_arm, auto_arm, rig_preset="NURANOVA")
yaw_mat = _compute_facing_yaw_matrix(src_arm, auto_arm, auto_map)
facing_yaw_deg = abs(math.degrees(yaw_mat.to_euler().z))
cleared_cnt = clear_markers()

# 2. Test Real-Time Live Preview (Body + 30 Fingers) retargeted live onto -Y Smart-Rigged character!
live_payload = {{
    "active": True,
    "seq": 1,
    "fps": 18.0,
    "source": "Live Test + Fingers",
    "skeleton": payload["skeleton"],
    "root_translation": payload["frames"]["root_translation"][5],
    "local_rotations_xyzw": payload["frames"]["local_rotations_xyzw"][5],
    "hands": {{
        "left_local_rotations_xyzw": payload["hands"]["left_local_rotations_xyzw"][5],
        "right_local_rotations_xyzw": payload["hands"]["right_local_rotations_xyzw"][5],
    }},
}}
live_arm = create_live_preview_armature(live_payload, include_fingers=True)
apply_live_frame_to_source_armature(live_arm, live_payload, include_fingers=True)
live_map = auto_match_bones(live_arm, auto_arm, rig_preset="NURANOVA")
live_retargeted = retarget_live_frame(live_arm, auto_arm, live_map)
live_fingers = apply_live_hand_rotations(auto_arm, live_payload["hands"])

# 3. Create a stylized secondary Rigify-named target rig (0.72x scale, non-zero bone rolls + Rigify fingers!)
tgt_data = bpy.data.armatures.new("StylizedCharacter_Data")
tgt_arm = bpy.data.objects.new("StylizedCharacter", tgt_data)
bpy.context.collection.objects.link(tgt_arm)
bpy.context.view_layer.objects.active = tgt_arm
bpy.ops.object.mode_set(mode="EDIT")
s = 0.72
rig_def = [
    ("DEF-spine", None, (0, 0, 0.96*s), (0, 0, 1.07*s), 0.3),
    ("DEF-spine.001", "DEF-spine", (0, 0, 1.07*s), (0, 0, 1.21*s), 0.1),
    ("DEF-spine.003", "DEF-spine.001", (0, 0, 1.21*s), (0, 0, 1.38*s), -0.2),
    ("DEF-spine.004", "DEF-spine.003", (0, 0, 1.38*s), (0, 0, 1.48*s), 0.0),
    ("DEF-spine.006", "DEF-spine.004", (0, 0, 1.48*s), (0, 0, 1.65*s), 0.0),
    ("DEF-upper_arm.L", "DEF-spine.003", (-0.16*s, 0, 1.40*s), (-0.42*s, 0, 1.40*s), 0.7),
    ("DEF-forearm.L", "DEF-upper_arm.L", (-0.42*s, 0, 1.40*s), (-0.65*s, 0, 1.40*s), 0.7),
    ("DEF-hand.L", "DEF-forearm.L", (-0.65*s, 0, 1.40*s), (-0.75*s, 0, 1.40*s), 0.0),
    ("DEF-f_index.01.L", "DEF-hand.L", (-0.75*s, 0, 1.40*s), (-0.80*s, 0, 1.40*s), 0.0),
    ("DEF-thumb.01.L", "DEF-hand.L", (-0.72*s, 0.03*s, 1.39*s), (-0.76*s, 0.05*s, 1.39*s), 0.0),
    ("DEF-upper_arm.R", "DEF-spine.003", (0.16*s, 0, 1.40*s), (0.42*s, 0, 1.40*s), -0.7),
    ("DEF-forearm.R", "DEF-upper_arm.R", (0.42*s, 0, 1.40*s), (0.65*s, 0, 1.40*s), -0.7),
    ("DEF-thigh.L", "DEF-spine", (-0.09*s, 0, 0.92*s), (-0.09*s, 0, 0.50*s), 0.5),
    ("DEF-shin.L", "DEF-thigh.L", (-0.09*s, 0, 0.50*s), (-0.09*s, 0, 0.09*s), 0.5),
    ("DEF-foot.L", "DEF-shin.L", (-0.09*s, 0, 0.09*s), (-0.09*s, 0.14*s, 0.02*s), 0.0),
    ("DEF-thigh.R", "DEF-spine", (0.09*s, 0, 0.92*s), (0.09*s, 0, 0.50*s), -0.5),
    ("DEF-shin.R", "DEF-thigh.R", (0.09*s, 0, 0.50*s), (0.09*s, 0, 0.09*s), -0.5),
    ("DEF-foot.R", "DEF-shin.R", (0.09*s, 0, 0.09*s), (0.09*s, 0.14*s, 0.02*s), 0.0),
]
ebs = {{}}
for bname, pname, head, tail, roll in rig_def:
    eb = tgt_data.edit_bones.new(bname)
    eb.head = Vector(head)
    eb.tail = Vector(tail)
    eb.roll = roll
    if pname:
        eb.parent = ebs[pname]
    ebs[bname] = eb
bpy.ops.object.mode_set(mode="OBJECT")

# Create a face mesh with Basis + jawOpen + eyeBlinkLeft shape keys
bpy.ops.mesh.primitive_ico_sphere_add(radius=0.12, location=(0, 0, 1.55*s))
face_mesh = bpy.context.active_object
face_mesh.shape_key_add(name="Basis")
face_mesh.shape_key_add(name="jawOpen")
face_mesh.shape_key_add(name="eyeBlinkLeft")

mapping = auto_match_bones(src_arm, tgt_arm, rig_preset="RIGIFY")
action = retarget_and_bake(src_arm, tgt_arm, mapping, frame_start=1, frame_end=20, scale_compensation=True, push_nla=True)
assert all(track.mute for track in tgt_arm.animation_data.nla_tracks)
for bone in tgt_arm.pose.bones:
    previous = None
    for frame in range(1, 21):
        bpy.context.scene.frame_set(frame)
        current = bone.rotation_quaternion.copy().normalized()
        if previous is not None:
            assert previous.dot(current) >= -1e-6, 'Quaternion sign discontinuity'
            expected = previous.slerp(current, 0.5).normalized()
            bpy.context.scene.frame_set(frame - 1, subframe=0.5)
            actual = bone.rotation_quaternion.copy().normalized()
            assert abs(expected.dot(actual)) > 0.99999, ('Fractional-frame interpolation overshot', bone.name, frame, tuple(previous), tuple(current), tuple(actual), expected.dot(actual))
        previous = current
mapped_shapes = apply_facial_blendshapes_to_mesh(face_mesh, payload, gain=1.0)
mapped_hands = apply_hand_rotations_to_armature(tgt_arm, payload)

# Export FBX and BVH
bpy.ops.export_scene.fbx(filepath=r"{fbx_out}", use_selection=False, bake_anim=True, add_leaf_bones=False)
bpy.ops.object.select_all(action="DESELECT")
tgt_arm.select_set(True)
bpy.context.view_layer.objects.active = tgt_arm
bpy.ops.export_anim.bvh(filepath=r"{bvh_out}", frame_start=1, frame_end=20)

# Verify keyframes changed on frame 20
bpy.context.scene.frame_set(20)
jaw_val_f20 = float(face_mesh.data.shape_keys.key_blocks["jawOpen"].value)

res = {{
    "blender_version": list(bpy.app.version),
    "source_bones": len(src_arm.data.bones),
    "ray_ok": ray_ok,
    "raycast_center_y": raycast_center_y,
    "placed_landmarks": placed_cnt,
    "total_landmarks": total_cnt,
    "finger_markers_count": len(finger_mks),
    "cleared_markers": cleared_cnt,
    "char_arm_mods": char_arm_mods,
    "smart_facing_y": float(auto_arm.get("nuranova_facing_y", 0.0)),
    "facing_yaw_deg": facing_yaw_deg,
    "left_sh_world": [float(left_sh_world.x), float(left_sh_world.y), float(left_sh_world.z)],
    "right_sh_world": [float(right_sh_world.x), float(right_sh_world.y), float(right_sh_world.z)],
    "auto_rig_bones": len(auto_arm.data.bones),
    "auto_rig_matched": len(auto_map),
    "auto_rig_weight_method": weight_method,
    "live_retargeted": live_retargeted,
    "live_fingers": live_fingers,
    "matched_bones": len(mapping),
    "action_name": action.name,
    "nla_tracks": len(tgt_arm.animation_data.nla_tracks),
    "mapped_shapes": mapped_shapes,
    "mapped_hands": mapped_hands,
    "jaw_val_f20": jaw_val_f20,
}}
bridge.unregister()
Path(r"{report_out}").write_text(json.dumps(res), encoding="utf-8")
print("BLENDER_BRIDGE_TEST_OK", json.dumps(res))
""",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [blender_exe, "--background", "--python-exit-code", "1", "--python", str(script_path)],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert proc.returncode == 0, f"Blender failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert report_out.exists(), f"Blender report missing.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    res = json.loads(report_out.read_text(encoding="utf-8"))
    assert res["source_bones"] == 55  # 25 body + 30 articulated fingers!
    assert res["ray_ok"] is True
    assert abs(res["raycast_center_y"]) < 0.01  # Centered INSIDE the mesh, not on front face -0.12!
    assert res["placed_landmarks"] == 12 and res["total_landmarks"] == 12
    assert res["finger_markers_count"] == 20
    assert res["cleared_markers"] >= 40
    assert res["char_arm_mods"] == 1  # Iterative re-rigging leaves exactly 1 clean Armature modifier
    assert res["smart_facing_y"] == -1.0  # Auto-detected standard Blender -Y facing!
    assert abs(res["facing_yaw_deg"] - 180.0) < 2.0  # Retargeter auto-aligned +Y source -> -Y character!
    # Left shoulder sits precisely at our stylized marker (2.16, 0.0, 1.12), Right mirrored at (1.84, 0.0, 1.12)
    assert abs(res["left_sh_world"][0] - 2.16) < 0.005
    assert abs(res["left_sh_world"][2] - 1.12) < 0.005
    assert abs(res["right_sh_world"][0] - 1.84) < 0.005
    assert abs(res["right_sh_world"][2] - 1.12) < 0.005
    assert res["auto_rig_bones"] == 55
    assert res["auto_rig_matched"] == 55
    assert res["live_retargeted"] == 55
    assert res["live_fingers"] == 30
    assert res["matched_bones"] >= 16
    assert res["nla_tracks"] >= 1
    assert res["mapped_shapes"] == 2
    assert res["mapped_hands"] >= 2
    assert res["jaw_val_f20"] > 0.75
    assert fbx_out.exists() and fbx_out.stat().st_size > 1000
    assert bvh_out.exists() and bvh_out.stat().st_size > 500
