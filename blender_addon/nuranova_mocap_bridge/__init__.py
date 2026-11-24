"""Nuranova Mocap Bridge — Blender 4.5 LTS / 5.0+ Extension Entry Point.

License: GPL-3.0-or-later (per Blender Python API `bpy` licensing requirements).
Communicates with Nuranova Mocap Studio strictly over local HTTP/JSON and open files.
All Blender scene mutations execute safely on the Blender main thread via `bpy.app.timers`.
"""

from __future__ import annotations

import json
import queue
import threading
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)

from .facial_hands import apply_facial_blendshapes_to_mesh, apply_hand_rotations_to_armature
from .retargeting import auto_match_bones, load_mapping_profile, retarget_and_bake, save_mapping_profile
from .source_armature import create_source_armature_from_payload, load_nmocap_json

bl_info = {
    "name": "Nuranova Mocap Bridge",
    "author": "Nuranova Solutions (Kawshal, Shanal, Shehan)",
    "version": (1, 0, 0),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar > Nuranova Mocap",
    "description": "Import, retarget, bake, and export full-body, facial, and hand motion capture from Nuranova Mocap Studio",
    "category": "Animation",
}

# Thread-safe mailbox for main-thread scene updates via bpy.app.timers
_BRIDGE_QUEUE: "queue.Queue[Dict[str, Any]]" = queue.Queue()
_LAST_PAYLOAD_CACHE: Dict[str, Any] = {}


def _fetch_json_async(url: str, tag: str) -> None:
    def worker() -> None:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NuranovaMocapBridge/1.0"})
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            _BRIDGE_QUEUE.put({"tag": tag, "ok": True, "data": data})
        except Exception as exc:
            _BRIDGE_QUEUE.put({"tag": tag, "ok": False, "error": str(exc)})

    threading.Thread(target=worker, daemon=True).start()


def _bridge_main_thread_timer() -> float:
    """Runs safely on Blender's main thread to consume network responses from background threads."""
    try:
        while not _BRIDGE_QUEUE.empty():
            msg = _BRIDGE_QUEUE.get_nowait()
            scene = bpy.context.scene
            if scene is None or not hasattr(scene, "nuranova_mocap"):
                continue
            props = scene.nuranova_mocap
            if msg["tag"] == "status":
                if msg["ok"]:
                    d = msg["data"]
                    props.studio_connected = True
                    props.studio_status_text = f"Connected ({d.get('active_project_name') or 'No project'}, {len(d.get('available_takes', []))} takes)"
                    takes = d.get("available_takes", [])
                    if takes:
                        props.selected_take_id = str(takes[-1]["take_id"])
                else:
                    props.studio_connected = False
                    props.studio_status_text = f"Disconnected: {msg['error']}"
            elif msg["tag"] == "import_take":
                if msg["ok"]:
                    payload = msg["data"]
                    _LAST_PAYLOAD_CACHE["latest"] = payload
                    arm = create_source_armature_from_payload(
                        payload,
                        in_place=props.in_place,
                        push_nla=props.push_to_nla,
                        context=bpy.context,
                    )
                    props.source_armature = arm
                    props.studio_status_text = f"Imported take '{payload.get('take_id')}' ({payload.get('num_frames')} frames)"
                else:
                    props.studio_status_text = f"Import failed: {msg['error']}"
    except Exception:
        pass
    return 0.4


class NuranovaBoneMapItem(bpy.types.PropertyGroup):
    source_bone: StringProperty(name="Source Bone")
    target_bone: StringProperty(name="Target Bone")


class NuranovaMocapProperties(bpy.types.PropertyGroup):
    studio_url: StringProperty(name="Studio URL", default="http://127.0.0.1:8765")
    studio_connected: BoolProperty(name="Connected", default=False)
    studio_status_text: StringProperty(name="Status", default="Not connected")
    selected_take_id: StringProperty(name="Take ID", default="")

    motion_filepath: StringProperty(name="Motion File", subtype="FILE_PATH", default="")
    mapping_filepath: StringProperty(name="Mapping Profile", subtype="FILE_PATH", default="")

    source_armature: PointerProperty(
        name="Source Rig",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == "ARMATURE",
    )
    target_armature: PointerProperty(
        name="Target Character Rig",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == "ARMATURE",
    )
    target_face_mesh: PointerProperty(
        name="Target Face Mesh",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == "MESH",
    )

    bone_map_items: CollectionProperty(type=NuranovaBoneMapItem)
    bone_map_index: IntProperty(default=0)

    scale_compensation: BoolProperty(name="Scale Compensation", default=True)
    extract_root_motion: BoolProperty(name="Extract Root Motion", default=True)
    in_place: BoolProperty(name="In-Place Animation", default=False)
    push_to_nla: BoolProperty(name="Push to NLA Track", default=True)
    face_gain: FloatProperty(name="Facial Blendshape Gain", default=1.0, min=0.0, max=3.0)
    export_filepath: StringProperty(name="Export Path", subtype="FILE_PATH", default="//nuranova_export.fbx")


class NURANOVA_OT_connect_studio(bpy.types.Operator):
    bl_idname = "nuranova.connect_studio"
    bl_label = "Check Studio Connection"

    def execute(self, context):
        props = context.scene.nuranova_mocap
        props.studio_status_text = "Connecting..."
        _fetch_json_async(f"{props.studio_url.rstrip('/')}/api/v1/bridge/status", "status")
        return {"FINISHED"}


class NURANOVA_OT_pull_take(bpy.types.Operator):
    bl_idname = "nuranova.pull_take"
    bl_label = "Pull Take from Studio"

    def execute(self, context):
        props = context.scene.nuranova_mocap
        if not props.selected_take_id:
            self.report({"WARNING"}, "Enter or discover a Take ID first.")
            return {"CANCELLED"}
        props.studio_status_text = f"Pulling take '{props.selected_take_id}'..."
        _fetch_json_async(
            f"{props.studio_url.rstrip('/')}/api/v1/bridge/takes/{props.selected_take_id}",
            "import_take",
        )
        return {"FINISHED"}


class NURANOVA_OT_import_offline(bpy.types.Operator):
    bl_idname = "nuranova.import_offline"
    bl_label = "Import Offline Motion (.nmocap.json / .bvh)"

    def execute(self, context):
        props = context.scene.nuranova_mocap
        fp = bpy.path.abspath(props.motion_filepath)
        if not fp or not Path(fp).exists():
            self.report({"ERROR"}, f"Motion file not found: {fp}")
            return {"CANCELLED"}

        if fp.lower().endswith(".bvh"):
            bpy.ops.import_anim.bvh(filepath=fp)
            if context.active_object and context.active_object.type == "ARMATURE":
                props.source_armature = context.active_object
            self.report({"INFO"}, f"Imported BVH: {Path(fp).name}")
            return {"FINISHED"}

        payload = load_nmocap_json(fp)
        _LAST_PAYLOAD_CACHE["latest"] = payload
        arm = create_source_armature_from_payload(
            payload,
            in_place=props.in_place,
            push_nla=props.push_to_nla,
            context=context,
        )
        props.source_armature = arm
        self.report({"INFO"}, f"Imported Nuranova take '{payload.get('take_id')}' ({payload.get('num_frames')} frames)")
        return {"FINISHED"}


class NURANOVA_OT_auto_map(bpy.types.Operator):
    bl_idname = "nuranova.auto_map_bones"
    bl_label = "Auto-Match Rig Bones"

    def execute(self, context):
        props = context.scene.nuranova_mocap
        if not props.source_armature or not props.target_armature:
            self.report({"ERROR"}, "Select both Source Rig and Target Character Rig first.")
            return {"CANCELLED"}
        mapping = auto_match_bones(props.source_armature, props.target_armature)
        props.bone_map_items.clear()
        for s, t in mapping.items():
            item = props.bone_map_items.add()
            item.source_bone = s
            item.target_bone = t
        self.report({"INFO"}, f"Matched {len(mapping)} bones automatically.")
        return {"FINISHED"}


class NURANOVA_OT_save_mapping(bpy.types.Operator):
    bl_idname = "nuranova.save_mapping"
    bl_label = "Save Mapping Profile"

    def execute(self, context):
        props = context.scene.nuranova_mocap
        fp = bpy.path.abspath(props.mapping_filepath)
        if not fp:
            self.report({"ERROR"}, "Specify a Mapping Profile file path first.")
            return {"CANCELLED"}
        mapping = {it.source_bone: it.target_bone for it in props.bone_map_items if it.source_bone and it.target_bone}
        rig_name = props.target_armature.name if props.target_armature else ""
        save_mapping_profile(mapping, fp, rig_name=rig_name)
        self.report({"INFO"}, f"Saved mapping profile ({len(mapping)} bones) to {fp}")
        return {"FINISHED"}


class NURANOVA_OT_load_mapping(bpy.types.Operator):
    bl_idname = "nuranova.load_mapping"
    bl_label = "Load Mapping Profile"

    def execute(self, context):
        props = context.scene.nuranova_mocap
        fp = bpy.path.abspath(props.mapping_filepath)
        if not fp or not Path(fp).exists():
            self.report({"ERROR"}, f"Mapping file not found: {fp}")
            return {"CANCELLED"}
        mapping = load_mapping_profile(fp)
        props.bone_map_items.clear()
        for s, t in mapping.items():
            item = props.bone_map_items.add()
            item.source_bone = s
            item.target_bone = t
        self.report({"INFO"}, f"Loaded {len(mapping)} bone mappings.")
        return {"FINISHED"}


class NURANOVA_OT_retarget_bake(bpy.types.Operator):
    bl_idname = "nuranova.retarget_bake"
    bl_label = "Retarget & Bake Editable Keyframes"

    def execute(self, context):
        props = context.scene.nuranova_mocap
        if not props.source_armature or not props.target_armature:
            self.report({"ERROR"}, "Select both Source Rig and Target Character Rig.")
            return {"CANCELLED"}
        mapping = {it.source_bone: it.target_bone for it in props.bone_map_items if it.source_bone and it.target_bone}
        if not mapping:
            mapping = auto_match_bones(props.source_armature, props.target_armature)
        act = retarget_and_bake(
            source_arm=props.source_armature,
            target_arm=props.target_armature,
            bone_map=mapping,
            frame_start=context.scene.frame_start,
            frame_end=context.scene.frame_end,
            scale_compensation=props.scale_compensation,
            extract_root_motion=props.extract_root_motion,
            in_place=props.in_place,
            push_nla=props.push_to_nla,
            context=context,
        )
        # Apply cached facial & hand tracks if present
        payload = _LAST_PAYLOAD_CACHE.get("latest")
        if payload:
            if props.target_face_mesh:
                apply_facial_blendshapes_to_mesh(props.target_face_mesh, payload, gain=props.face_gain)
            apply_hand_rotations_to_armature(props.target_armature, payload)
        self.report({"INFO"}, f"Baked editable Action '{act.name}' ({len(mapping)} bones).")
        return {"FINISHED"}


class NURANOVA_OT_export_fbx(bpy.types.Operator):
    bl_idname = "nuranova.export_fbx"
    bl_label = "Export FBX"

    def execute(self, context):
        props = context.scene.nuranova_mocap
        fp = bpy.path.abspath(props.export_filepath)
        if not fp.lower().endswith(".fbx"):
            fp = str(Path(fp).with_suffix(".fbx"))
        Path(fp).parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.export_scene.fbx(filepath=fp, use_selection=False, bake_anim=True, add_leaf_bones=False)
        self.report({"INFO"}, f"Exported FBX to {fp}")
        return {"FINISHED"}


class NURANOVA_OT_export_bvh(bpy.types.Operator):
    bl_idname = "nuranova.export_bvh"
    bl_label = "Export BVH"

    def execute(self, context):
        props = context.scene.nuranova_mocap
        fp = bpy.path.abspath(props.export_filepath)
        if not fp.lower().endswith(".bvh"):
            fp = str(Path(fp).with_suffix(".bvh"))
        Path(fp).parent.mkdir(parents=True, exist_ok=True)
        arm = props.target_armature or props.source_armature
        if arm is None:
            self.report({"ERROR"}, "Select an armature to export BVH.")
            return {"CANCELLED"}
        bpy.ops.object.select_all(action="DESELECT")
        arm.select_set(True)
        context.view_layer.objects.active = arm
        bpy.ops.export_anim.bvh(
            filepath=fp,
            frame_start=context.scene.frame_start,
            frame_end=context.scene.frame_end,
            root_transform_only=False,
        )
        self.report({"INFO"}, f"Exported BVH to {fp}")
        return {"FINISHED"}


class NURANOVA_UL_bone_map(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "source_bone", text="", emboss=False)
        row.label(text="->")
        row.prop(item, "target_bone", text="")


class NURANOVA_PT_main_panel(bpy.types.Panel):
    bl_label = "Nuranova Mocap Bridge"
    bl_idname = "NURANOVA_PT_main_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Nuranova Mocap"

    def draw(self, context):
        layout = self.layout
        props = context.scene.nuranova_mocap

        box = layout.box()
        box.label(text="1. Mocap Studio Connection", icon="LINKED")
        box.prop(props, "studio_url")
        row = box.row(align=True)
        row.operator("nuranova.connect_studio", icon="FILE_REFRESH")
        row.operator("nuranova.pull_take", icon="IMPORT")
        box.prop(props, "selected_take_id")
        box.label(text=props.studio_status_text)

        box2 = layout.box()
        box2.label(text="2. Offline Motion Import", icon="FILE_FOLDER")
        box2.prop(props, "motion_filepath")
        box2.operator("nuranova.import_offline", icon="ARMATURE_DATA")

        box3 = layout.box()
        box3.label(text="3. Character Rig Retargeting", icon="POSE_HLT")
        box3.prop(props, "source_armature")
        box3.prop(props, "target_armature")
        box3.prop(props, "target_face_mesh")
        box3.operator("nuranova.auto_map_bones", icon="AUTO")
        box3.template_list("NURANOVA_UL_bone_map", "", props, "bone_map_items", props, "bone_map_index", rows=5)
        box3.prop(props, "mapping_filepath")
        row_m = box3.row(align=True)
        row_m.operator("nuranova.save_mapping", icon="FILE_TICK")
        row_m.operator("nuranova.load_mapping", icon="FILE_FOLDER")
        box3.prop(props, "scale_compensation")
        box3.prop(props, "extract_root_motion")
        box3.prop(props, "in_place")
        box3.prop(props, "push_to_nla")
        box3.prop(props, "face_gain")
        box3.operator("nuranova.retarget_bake", icon="ACTION")

        box4 = layout.box()
        box4.label(text="4. Animation Export", icon="EXPORT")
        box4.prop(props, "export_filepath")
        row_e = box4.row(align=True)
        row_e.operator("nuranova.export_fbx", icon="EXPORT")
        row_e.operator("nuranova.export_bvh", icon="EXPORT")


_CLASSES = (
    NuranovaBoneMapItem,
    NuranovaMocapProperties,
    NURANOVA_OT_connect_studio,
    NURANOVA_OT_pull_take,
    NURANOVA_OT_import_offline,
    NURANOVA_OT_auto_map,
    NURANOVA_OT_save_mapping,
    NURANOVA_OT_load_mapping,
    NURANOVA_OT_retarget_bake,
    NURANOVA_OT_export_fbx,
    NURANOVA_OT_export_bvh,
    NURANOVA_UL_bone_map,
    NURANOVA_PT_main_panel,
)


def register() -> None:
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.nuranova_mocap = PointerProperty(type=NuranovaMocapProperties)
    if not bpy.app.timers.is_registered(_bridge_main_thread_timer):
        bpy.app.timers.register(_bridge_main_thread_timer, persistent=True)


def unregister() -> None:
    if bpy.app.timers.is_registered(_bridge_main_thread_timer):
        bpy.app.timers.unregister(_bridge_main_thread_timer)
    if hasattr(bpy.types.Scene, "nuranova_mocap"):
        del bpy.types.Scene.nuranova_mocap
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
