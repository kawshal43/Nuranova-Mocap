"""Real Blender checks for mesh contacts, reversible correction, and scoped 3D exports."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from desktop.settings.app_settings import detect_blender_executable
from engine.data_format.motion_io import MotionCaptureDocument, SkeletalMotionTrack, export_blender_json
from engine.skeleton.rig_definition import BONE_NAMES, PARENT_INDICES, DEFAULT_REST_OFFSETS_M, compute_forward_kinematics_sequence


def test_real_character_contacts_and_scoped_export_roundtrips(tmp_path):
    executable = detect_blender_executable()
    if not executable:
        pytest.skip("Blender is not installed.")
    n = 8
    root = np.tile([0., 0., .96], (n, 1))
    rotations = np.tile([0., 0., 0., 1.], (n, 25, 1))
    for bone, sign in ((8, -1), (12, 1)):
        rotations[:, bone, 2] = np.sin(sign * np.radians(30) / 2)
        rotations[:, bone, 3] = np.cos(sign * np.radians(30) / 2)
    positions, matrices = compute_forward_kinematics_sequence(root, rotations, DEFAULT_REST_OFFSETS_M)
    track = SkeletalMotionTrack(root, rotations[:, 0], rotations, matrices, positions, np.ones((n, 25)))
    doc = MotionCaptureDocument("qa", "qa", "contacts", "qa", 30., (np.arange(n) / 30 * 1e9).astype(np.int64),
        BONE_NAMES, PARENT_INDICES, DEFAULT_REST_OFFSETS_M, track)
    payload = export_blender_json(doc, tmp_path / "motion.json")
    addon = Path(__file__).resolve().parents[2] / "blender_addon"
    script = tmp_path / "verify.py"
    script.write_text(f'''
import sys, json
from pathlib import Path
import bpy
from mathutils import Vector
sys.path.insert(0, {str(addon)!r})
import nuranova_mocap_bridge as bridge
from nuranova_mocap_bridge.source_armature import create_source_armature_from_payload, load_nmocap_json
from nuranova_mocap_bridge.character_quality import build_character_profile, CharacterGeometry, fix_character_frame, duplicate_character
from nuranova_mocap_bridge.character_quality import visual_bake_copy, regularize_contact_corrections
from nuranova_mocap_bridge.character_export import export_character
bridge.register()
assert hasattr(bpy.context.scene.nuranova_mocap, 'quality_margin_mm')
bpy.ops.wm.read_factory_settings(use_empty=True)
arm = create_source_armature_from_payload(load_nmocap_json({str(payload)!r}), include_fingers=False, push_nla=False)
arm.name = "ChosenActor"
for name, bone, location, scale in [
    ('Torso', 'chest', (0,0,1.2), (.80,.30,.80)),
    ('LeftHand', 'left_hand', (-.73,0,1.43), (.045,.045,.045)),
    ('RightHand', 'right_hand', (.73,0,1.43), (.045,.045,.045))]:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=10, location=location)
    mesh = bpy.context.object
    mesh.name = name
    mesh.scale = scale
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    group = mesh.vertex_groups.new(name=bone)
    group.add(list(range(len(mesh.data.vertices))), 1., 'REPLACE')
    mod = mesh.modifiers.new('Skin', 'ARMATURE')
    mod.object = arm
    if name == 'Torso':
        mesh.shape_key_add(name='Basis')
        key = mesh.shape_key_add(name='jawOpen')
        for point in key.data:
            if point.co.z > 1.5:
                point.co.y += .025
        key.value = 0
        key.keyframe_insert(data_path='value', frame=1)
        key.value = 1
        key.keyframe_insert(data_path='value', frame=8)
        constant=mesh.shape_key_add(name='StaticCorrective')
        constant.value=.04
        for point in constant.data:
            if point.co.z>1.5: point.co.y+=.02
mapping = {{name:name for name in arm.data.bones.keys()}}
profile = build_character_profile(arm, mapping)
geometry = CharacterGeometry(arm, profile)
bpy.context.scene.frame_set(1)
before = geometry.analyze(1)
assert before['contacts'], before
original_action = arm.animation_data.action
duplicate, mesh_map = duplicate_character(arm)
profile['meshes'] = {{mesh_map[k]:v for k,v in profile['meshes'].items()}}
profile['armature'] = duplicate.name
corrected = fix_character_frame(CharacterGeometry(duplicate, profile), 1, iterations=2)
assert corrected['after']['max_depth_m'] <= corrected['before']['max_depth_m'] + .0001, corrected
assert arm.animation_data.action == original_action
assert duplicate.animation_data.action != original_action
# An isolated contact adjustment must not introduce a new flicker into the base motion.
from mathutils import Quaternion
test_bone=duplicate.pose.bones['left_forearm']
bpy.context.scene.frame_set(4)
test_bone.rotation_quaternion = arm.pose.bones['left_forearm'].rotation_quaternion @ Quaternion((0,1,0),.8)
test_bone.keyframe_insert(data_path='rotation_quaternion',frame=4)
continuity=list(regularize_contact_corrections(arm,duplicate,2,7,30.))
assert continuity[-1]['max_correction_step_deg']<=6.01,continuity[-1]
for frame in (2,7):
    bpy.context.scene.frame_set(frame)
    assert abs(duplicate.pose.bones['left_forearm'].rotation_quaternion.dot(arm.pose.bones['left_forearm'].rotation_quaternion))>.99999
assert arm.animation_data.action == original_action
static_original,_=duplicate_character(arm)
static_original.animation_data_clear()
for bone in static_original.pose.bones:
    basis=bone.matrix_basis.copy()
    bone.rotation_mode='XYZ'
    bone.matrix_basis=basis
static_working,_=duplicate_character(static_original)
static_profile=build_character_profile(static_working,mapping)
fix_character_frame(CharacterGeometry(static_working,static_profile),1,iterations=1)
list(regularize_contact_corrections(static_original,static_working,1,2,30.))
assert static_original.animation_data is None
bpy.ops.object.armature_add()
bpy.context.object.name = 'UnrelatedRigThatMustNotExport'
bpy.context.scene.frame_start, bpy.context.scene.frame_end = 1,8
# A second, non-neutral static facial-style armature must be baked into a
# temporary mesh, including every morph target, rather than silently removed.
bpy.ops.object.armature_add()
auxiliary=bpy.context.object
auxiliary.name='StaticFacialAuxiliary'
auxiliary.data.bones[0].name='AuxFace'
auxiliary.pose.bones[0].location.x=.035
torso=bpy.data.objects['Torso']
torso.vertex_groups.new(name='AuxFace').add(list(range(len(torso.data.vertices))),.4,'REPLACE')
auxmod=torso.modifiers.new('FacialPrefix','ARMATURE')
auxmod.object=auxiliary
with bpy.context.temp_override(object=torso):
    bpy.ops.object.modifier_move_up(modifier=auxmod.name)
reports = {{}}
for suffix in ('.glb', '.fbx', '.gltf'):
    path = Path({str(tmp_path)!r}) / ("actor's_සිංහල" + suffix)
    reports[suffix] = export_character(arm, path, frame_start=1, frame_end=8, validate=True)
    assert reports[suffix]['passed'] and reports[suffix]['bones'] == 25, reports[suffix]
    assert reports[suffix]['meshes'] == 3
    assert reports[suffix]['static_auxiliary_meshes_baked']==1
    assert reports[suffix]['shape_channels_verified'], reports[suffix]
    assert abs(bpy.data.objects['Torso'].data.shape_keys.key_blocks['StaticCorrective'].value-.04)<.00001
static = export_character(arm,Path({str(tmp_path)!r})/'static.glb',variant='static',frame_start=4)
assert static['passed'] and static['bones']==0 and static['meshes']==3
motion = export_character(arm,Path({str(tmp_path)!r})/'motion.glb',variant='motion',frame_start=1,frame_end=8)
assert motion['passed'] and motion['bones']==25 and motion['meshes']==0
# An animated prefix needs a richer bake; the simple exporter must reject it.
auxiliary.pose.bones[0].keyframe_insert(data_path='location',frame=1)
auxiliary.pose.bones[0].location.x=.08
auxiliary.pose.bones[0].keyframe_insert(data_path='location',frame=8)
try:
    export_character(arm,Path({str(tmp_path)!r})/'unsupported.glb',frame_start=1,frame_end=8)
    raise AssertionError('Animated auxiliary deformation was silently lost')
except ValueError as exc:
    assert 'additional deformation rig moves' in str(exc), str(exc)
auxiliary.animation_data_clear()
auxiliary.pose.bones[0].location.x=.035
bpy.ops.wm.save_as_mainfile(filepath=str(Path({str(tmp_path)!r})/'character.blend'))
# An evaluated control constraint must survive baking to a separate deform rig.
bpy.ops.object.empty_add()
controller=bpy.context.object
controller.rotation_euler.z=.2
constraint=arm.pose.bones['left_forearm'].constraints.new('COPY_ROTATION')
constraint.target=controller
arm.scale=(1.1,.9,1.2)
complete=None
for step in visual_bake_copy(arm,1,8):
    if step.get('complete'): complete=step
assert complete
baked=complete['working']
for frame in [1,4,8]:
    bpy.context.scene.frame_set(frame)
    graph=bpy.context.evaluated_depsgraph_get()
    a=arm.evaluated_get(graph)
    b=baked.evaluated_get(graph)
    for name in arm.pose.bones.keys():
        assert ((a.matrix_world@a.pose.bones[name].head)-(b.matrix_world@b.pose.bones[name].head)).length < .0001, name
Path({str(tmp_path / 'report.json')!r}).write_text(json.dumps({{'contacts':before, 'correction':corrected, 'exports':reports}}))
bridge.unregister()
''', encoding="utf-8")
    process = subprocess.run([executable, "--background", "--factory-startup", "--python", str(script)],
                             capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    assert process.returncode == 0, process.stdout[-10000:] + process.stderr[-5000:]
    report_path = tmp_path / "report.json"
    assert report_path.exists(), process.stdout[-10000:] + process.stderr[-5000:]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["exports"][".glb"]["max_joint_error_m"] < .003
    from engine.export.blender_job import run_blender_export
    import threading
    destination=tmp_path/"worker.glb"
    worker=run_blender_export(executable,{"scene_file":str(tmp_path/"character.blend"),
        "target_armature":"ChosenActor", "variant":"character", "output":str(destination),
        "frame_start":1,"frame_end":8})
    assert worker["passed"] and destination.exists()
    separate=tmp_path/"worker's_සිංහල.gltf"
    run_blender_export(executable,{"scene_file":str(tmp_path/"character.blend"),
        "target_armature":"ChosenActor","variant":"character","output":str(separate),"frame_start":1,"frame_end":8})
    from urllib.parse import unquote
    payload=json.loads(separate.read_text(encoding="utf-8"))
    assert all((separate.parent/unquote(item["uri"])).exists() for item in payload["buffers"])
    previous=destination.read_bytes()
    cancelled=threading.Event()
    cancelled.set()
    with pytest.raises(RuntimeError,match="cancelled"):
        run_blender_export(executable,{"output":str(destination)},cancelled)
    assert destination.read_bytes()==previous
