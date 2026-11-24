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
        fps=60.0,
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

    addon_root = Path("e:/Backup/KAWSHAL/Nuranova-Mocap/blender_addon").resolve()
    script_path = tmp_path / "run_blender_test.py"
    script_path.write_text(
        f"""
import json, sys, math
from pathlib import Path
import bpy
from mathutils import Vector

sys.path.insert(0, r"{addon_root}")
import nuranova_mocap_bridge as bridge
from nuranova_mocap_bridge.source_armature import create_source_armature_from_payload, load_nmocap_json
from nuranova_mocap_bridge.retargeting import auto_match_bones, retarget_and_bake
from nuranova_mocap_bridge.facial_hands import apply_facial_blendshapes_to_mesh, apply_hand_rotations_to_armature

# Reset scene
bpy.ops.wm.read_factory_settings(use_empty=True)
bridge.register()

payload = load_nmocap_json(r"{motion_json}")
src_arm = create_source_armature_from_payload(payload, push_nla=True)

# Create a stylized secondary Rigify-named target rig (0.72x scale, non-zero bone rolls!)
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
    ("DEF-f_index_01.L", "DEF-hand.L", (-0.75*s, 0, 1.40*s), (-0.80*s, 0, 1.40*s), 0.0),
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

mapping = auto_match_bones(src_arm, tgt_arm)
action = retarget_and_bake(src_arm, tgt_arm, mapping, frame_start=1, frame_end=20, scale_compensation=True, push_nla=True)
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
        [blender_exe, "--background", "--python", str(script_path)],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert proc.returncode == 0, f"Blender failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert report_out.exists(), f"Blender report missing.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    res = json.loads(report_out.read_text(encoding="utf-8"))
    assert res["source_bones"] == 25
    assert res["matched_bones"] >= 14
    assert res["nla_tracks"] >= 1
    assert res["mapped_shapes"] == 2
    assert res["mapped_hands"] >= 1
    assert res["jaw_val_f20"] > 0.75
    assert fbx_out.exists() and fbx_out.stat().st_size > 1000
    assert bvh_out.exists() and bvh_out.stat().st_size > 500
