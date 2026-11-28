"""Actual skinned mesh improvement while an observed wrist stays fixed."""
import json
from pathlib import Path
import subprocess

import pytest

from desktop.settings.app_settings import detect_blender_executable


def test_surface_rotation_reduces_contact_without_moving_wrist(tmp_path):
    blender = detect_blender_executable()
    if not blender:
        pytest.skip("Blender is not installed.")
    addon = Path(__file__).resolve().parents[2]/"blender_addon"
    script = tmp_path/"surface.py"
    result = tmp_path/"surface.json"
    script.write_text(f'''
import bpy,sys,json
from pathlib import Path
from mathutils import Vector
sys.path.insert(0,{str(addon)!r})
from nuranova_mocap_bridge.auto_rig import create_nuranova_humanoid_rig
from nuranova_mocap_bridge.character_quality import CharacterGeometry,build_character_profile,_adjust_limb_surface,capture_stance_anchors
bpy.ops.wm.read_factory_settings(use_empty=True)
arm=create_nuranova_humanoid_rig("SurfaceTest")
mapping={{name:name for name in arm.data.bones.keys()}}
hand=arm.pose.bones['left_hand']
world=arm.matrix_world@hand.matrix
head=world.translation.copy()
center=head+Vector((0,.12,0))
for name,bone,position,radius in [('BlockingTorso','chest',center+Vector((.025,0,0)),.04),
                                 ('HandSurface','left_hand',center,.012)]:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24,ring_count=16,radius=radius,location=position)
    obj=bpy.context.object
    obj.name=name
    bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
    group=obj.vertex_groups.new(name=bone)
    group.add(list(range(len(obj.data.vertices))),1.,'REPLACE')
    obj.modifiers.new('Skin','ARMATURE').object=arm
try:
    build_character_profile(arm,{{}})
    raise AssertionError('Missing mapping was accepted')
except ValueError as exc:
    assert 'Map the character' in str(exc)
profile=build_character_profile(arm,mapping)
geometry=CharacterGeometry(arm,profile)
before=geometry.analyze(1)
contact=next(c for c in before['contacts'] if c['moving_region']=='left_hand' and c['depth_m']>.001)
anchors={{side+'_'+part:arm.matrix_world@arm.pose.bones[side+'_'+part].head
         for side in ('left','right') for part in ('hand','foot')}}
changed=_adjust_limb_surface(geometry,contact,anchors,.001,.001)
after=geometry.analyze(1)
assert changed, (before,after)
assert after['max_depth_m']<before['max_depth_m']*.95,(before,after)
assert ((arm.matrix_world@hand.head)-head).length<1e-7
anchors_report=list(capture_stance_anchors(arm,profile,1,12,30.))
assert anchors_report[-1]['anchored_frames']==12
assert all(set(value)=={{'left','right'}} for value in profile['flat_stance_anchors'].values())
Path({str(result)!r}).write_text(json.dumps({{'before':before['max_depth_m'],'after':after['max_depth_m'],'changed':changed}}))
''', encoding="utf-8")
    process = subprocess.run([blender,"--background","--python",str(script)], capture_output=True, text=True, timeout=120)
    assert result.exists(), process.stdout[-5000:]+process.stderr[-5000:]
    report = json.loads(result.read_text())
    assert report["after"] < report["before"]*.95


def test_partial_contact_corrections_preserve_keyed_quaternion_hemispheres(tmp_path):
    blender = detect_blender_executable()
    if not blender:
        pytest.skip("Blender is not installed.")
    addon = Path(__file__).resolve().parents[2]/"blender_addon"
    script, result = tmp_path/"boundary.py", tmp_path/"boundary.json"
    script.write_text(f'''
import bpy,sys,json,math
from pathlib import Path
from mathutils import Quaternion
sys.path.insert(0,{str(addon)!r})
from nuranova_mocap_bridge.auto_rig import create_nuranova_humanoid_rig
from nuranova_mocap_bridge.character_quality import duplicate_character,regularize_contact_corrections
from nuranova_mocap_bridge.source_armature import _get_action_fcurves
bpy.ops.wm.read_factory_settings(use_empty=True)
arm=create_nuranova_humanoid_rig('BoundarySource')
bone=arm.pose.bones['left_upper_arm']
for frame,angle in enumerate(range(150,230,10),1):
    half=math.radians(angle)*.5
    bone.rotation_quaternion=(math.cos(half),0,0,math.sin(half))
    bone.keyframe_insert(data_path='rotation_quaternion',frame=frame)
for curve in _get_action_fcurves(arm.animation_data.action,arm):
    for key in curve.keyframe_points:key.interpolation='LINEAR'
for start,end in [(3,6),(5,7)]:
    working,_=duplicate_character(arm)
    for frame in range(start,end+1):
        bpy.context.scene.frame_set(frame)
        edited=working.pose.bones[bone.name]
        edited.rotation_quaternion=arm.pose.bones[bone.name].rotation_quaternion@Quaternion((1,0,0),.1)
        edited.keyframe_insert(data_path='rotation_quaternion',frame=frame)
    list(regularize_contact_corrections(arm,working,start,end,30.))
    rotations=[]
    for frame in range(1,9):
        bpy.context.scene.frame_set(frame)
        rotations.append(working.pose.bones[bone.name].rotation_quaternion.copy())
    assert all(a.dot(b)>0 for a,b in zip(rotations,rotations[1:])),(start,end,[tuple(q) for q in rotations])
    for frame in range(1,8):
        bpy.context.scene.frame_set(frame,subframe=.5)
        actual=working.pose.bones[bone.name].matrix_basis.to_quaternion()
        expected=rotations[frame-1].slerp(rotations[frame],.5)
        assert abs(actual.dot(expected))>1-1e-6,(start,end,frame)
    for frame in (start-1,end):
        bpy.context.scene.frame_set(frame,subframe=.5)
        a=arm.pose.bones[bone.name].matrix_basis.to_quaternion()
        b=working.pose.bones[bone.name].matrix_basis.to_quaternion()
        assert abs(a.dot(b))>1-1e-6,(start,end,frame)
Path({str(result)!r}).write_text(json.dumps({{'passed':True}}))
''', encoding="utf-8")
    process = subprocess.run([blender,"--background","--python",str(script)],capture_output=True,text=True,timeout=120)
    assert result.exists(), process.stdout[-5000:]+process.stderr[-5000:]
