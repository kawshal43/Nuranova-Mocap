"""An actor shared with another Blender scene must export its active-scene motion."""
from pathlib import Path
import subprocess

import pytest

from desktop.settings.app_settings import detect_blender_executable


def test_glb_from_secondary_scene_keeps_animation(tmp_path):
    blender = detect_blender_executable()
    if not blender:
        pytest.skip('Blender is not installed.')
    addon = Path(__file__).resolve().parents[2]/'blender_addon'
    script = tmp_path/'secondary_scene.py'
    script.write_text(f'''
import sys,bpy
from pathlib import Path
sys.path.insert(0,{str(addon)!r})
from nuranova_mocap_bridge.character_export import export_character
from nuranova_mocap_bridge.source_armature import _get_action_fcurves
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.object.armature_add()
arm=bpy.context.object
arm.name='SharedActor'
arm.rotation_mode='QUATERNION'
for frame,x in [(1,0.),(2,.1),(3,.2)]:
 arm.pose.bones[0].location.x=x
 arm.pose.bones[0].keyframe_insert(data_path='location',frame=frame)
arm.animation_data.use_nla=False
for curve in _get_action_fcurves(arm.animation_data.action,arm):
 for key in curve.keyframe_points:key.interpolation='LINEAR'
bpy.ops.mesh.primitive_uv_sphere_add(segments=12,ring_count=8,location=(0,0,.5))
mesh=bpy.context.object
mesh.scale=(.1,.1,.5)
bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
mesh.vertex_groups.new(name=arm.data.bones[0].name).add(list(range(len(mesh.data.vertices))),1.,'REPLACE')
mesh.modifiers.new('Skin','ARMATURE').object=arm
mesh.parent=arm
secondary=bpy.data.scenes.new('Z_ChosenSecondaryScene')
for obj in (arm,mesh):secondary.collection.objects.link(obj)
bpy.context.window.scene=secondary
secondary.frame_start,secondary.frame_end=1,3
bpy.context.view_layer.objects.active=arm
arm.select_set(True)
report=export_character(arm,{str(tmp_path/'secondary.glb')!r},variant='character',frame_start=1,frame_end=3)
assert report['passed'] and report['samples']==5,report
static=export_character(arm,{str(tmp_path/'secondary_static.glb')!r},variant='static',frame_start=2)
assert static['passed'] and static['bones']==0 and static['meshes']==1,static
assert bpy.context.window.scene==secondary
assert bpy.data.scenes[0]!=secondary
''',encoding='utf-8')
    result = subprocess.run([blender,'--background','--factory-startup','--python-exit-code','1',
        '--python',str(script)],capture_output=True,timeout=120)
    assert result.returncode == 0, result.stdout.decode(errors='replace')[-3500:]+result.stderr.decode(errors='replace')[-1500:]
