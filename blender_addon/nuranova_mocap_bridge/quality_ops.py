"""Responsive Blender UI operators for character contact analysis and repair."""

from __future__ import annotations

import json
import time

import bpy
from bpy.props import EnumProperty

from .character_export import bound_meshes
from .character_quality import (CharacterGeometry, build_character_profile, duplicate_character,
    fix_character_frame, load_profile, regularize_contact_corrections, save_profile, visual_bake_copy)
from .retargeting import auto_match_bones


def _mapping(props, target):
    mapping = {item.source_bone: item.target_bone for item in props.bone_map_items if item.source_bone and item.target_bone}
    return mapping or auto_match_bones(props.source_armature or target, target)


class NURANOVA_OT_character_quality(bpy.types.Operator):
    bl_idname = "nuranova.character_quality"
    bl_label = "Character contact review"
    bl_options = {"REGISTER"}

    operation: EnumProperty(items=[("ANALYZE", "Analyze", "Inspect deformed geometry"),
                                  ("FIX", "Fix range", "Make a separate corrected action"),
                                  ("BAKE", "Visual bake copy", "Bake a control rig into a separate deform copy")])

    def invoke(self, context, event):
        props = context.scene.nuranova_mocap
        self.original = props.target_armature
        if self.original is None:
            self.report({"ERROR"}, "Choose the target character armature.")
            return {"CANCELLED"}
        self.restore_frame = context.scene.frame_current
        self.visibility = [(obj, obj.hide_get()) for obj in [self.original, *bound_meshes(self.original)]]
        self.working = None
        start = props.quality_start or context.scene.frame_start
        end = props.quality_end or context.scene.frame_end
        if end < start:
            self.report({"ERROR"}, "The review range ends before it starts.")
            return {"CANCELLED"}
        self.report_data = {"operation": self.operation, "original": self.original.name,
                            "start": start, "end": end, "frames": [], "status": "running"}
        try:
            if self.operation == "BAKE":
                self.iterator = iter(visual_bake_copy(self.original, start, end))
            else:
                mapping = _mapping(props, self.original)
                profile = build_character_profile(self.original, mapping, props.quality_margin_mm / 1000.)
                try:
                    profile["contact_pins"] = load_profile(self.original).get("contact_pins", [])
                except ValueError:
                    pass
                self.report_data["profile_warnings"] = profile["warnings"]
                if self.operation == "FIX":
                    from .source_armature import _get_action_fcurves
                    constrained = [pb.name for pb in self.original.pose.bones if any(not c.mute and c.influence > 0 for c in pb.constraints)]
                    if constrained:
                        raise ValueError("Use Visual bake copy before correcting this control rig.")
                    action = self.original.animation_data.action if self.original.animation_data else None
                    if self.original.animation_data and self.original.animation_data.use_nla and any(
                            not track.mute for track in self.original.animation_data.nla_tracks):
                        raise ValueError("Use Visual bake copy before correcting layered NLA animation.")
                    if action and any("rotation_euler" in curve.data_path for curve in _get_action_fcurves(action,self.original)):
                        raise ValueError("Use Visual bake copy before correcting Euler-keyed rig animation.")
                    if action and any("rotation_quaternion" in curve.data_path and
                            any(key.interpolation != "LINEAR" for key in curve.keyframe_points)
                            for curve in _get_action_fcurves(action,self.original)):
                        raise ValueError("Use Visual bake copy before correcting curved quaternion animation.")
                    self.working, mesh_map = duplicate_character(self.original)
                    profile["armature"] = self.working.name
                    profile["meshes"] = {mesh_map[name]: values for name, values in profile["meshes"].items()}
                    for obj, _ in self.visibility:
                        obj.hide_set(True)
                save_profile(self.working or self.original, profile)
                self.geometry = CharacterGeometry(self.working or self.original, profile)
                self.iterator = self._steps(start, end)
        except (ValueError, RuntimeError) as exc:
            self._restore(context, cancelled=True)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.timer = context.window_manager.event_timer_add(.02, window=context.window)
        context.window_manager.modal_handler_add(self)
        context.window_manager.progress_begin(0, max(1, end - start + 1))
        return {"RUNNING_MODAL"}

    def _steps(self, start, end):
        for frame in range(start, end + 1):
            bpy.context.scene.frame_set(frame)
            if self.operation == "FIX":
                yield fix_character_frame(self.geometry, frame)
            else:
                yield self.geometry.analyze(frame)
        # Validate integer and intermediate poses after all corrections have been keyed.
        if self.operation == "FIX":
            yield from regularize_contact_corrections(self.original,self.working,start,end,
                bpy.context.scene.render.fps / bpy.context.scene.render.fps_base)
        for i in range(start * 2, end * 2 + 1):
            frame = i / 2
            if self.operation == "ANALYZE" and frame.is_integer():
                continue
            bpy.context.scene.frame_set(int(frame), subframe=frame % 1)
            report = self.geometry.analyze(frame)
            report["validation"] = True
            yield report
            if report["contacts"]:
                for value in (frame + .25,):
                    if value < end:
                        bpy.context.scene.frame_set(int(value), subframe=value % 1)
                        extra = self.geometry.analyze(value)
                        extra["validation"] = True
                        yield extra

    def modal(self, context, event):
        if event.type == "ESC":
            self.report_data["status"] = "cancelled"
            self._finish(context, cancelled=True)
            return {"CANCELLED"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        deadline = time.monotonic() + .025
        try:
            while time.monotonic() < deadline:
                item = next(self.iterator)
                if "working" in item:
                    self.working = item["working"]
                serializable = {k: v for k, v in item.items() if k not in {"working", "mesh_map"}}
                self.report_data["frames"].append(serializable)
                context.window_manager.progress_update(item.get("frame", 0) - self.report_data["start"])
        except StopIteration:
            self.report_data["status"] = "completed"
            self._finish(context)
            return {"FINISHED"}
        except Exception as exc:
            self.report_data.update(status="failed", error=str(exc))
            self._finish(context, cancelled=True)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"RUNNING_MODAL"}

    def _restore(self, context, cancelled=False):
        for obj, hidden in self.visibility:
            obj.hide_set(hidden if cancelled or not self.working else True)
        if cancelled and self.working:
            for obj in [self.working, *bound_meshes(self.working)]:
                obj.hide_set(True)
            self.working["nuranova_incomplete"] = True
        if not cancelled and self.working:
            props = context.scene.nuranova_mocap
            props.quality_original = self.original
            props.quality_working = self.working
            props.target_armature = self.working
        context.scene.frame_set(self.restore_frame)

    def _finish(self, context, cancelled=False):
        context.window_manager.event_timer_remove(self.timer)
        context.window_manager.progress_end()
        self._restore(context, cancelled)
        validation = [r for r in self.report_data["frames"] if r.get("validation")]
        contacts = sum(bool(r.get("contacts")) for r in validation)
        self.report_data["remaining_contact_samples"] = contacts
        text = bpy.data.texts.get("Nuranova_Character_Quality.json") or bpy.data.texts.new("Nuranova_Character_Quality.json")
        text.clear()
        text.write(json.dumps(self.report_data))
        props = context.scene.nuranova_mocap
        props.quality_status = f"{self.report_data['status']} · {len(validation)} validation poses · {contacts} need contact review"
        self.report({"INFO"}, props.quality_status)


class NURANOVA_OT_compare_character(bpy.types.Operator):
    bl_idname = "nuranova.compare_character"
    bl_label = "Compare original / corrected"

    def execute(self, context):
        props = context.scene.nuranova_mocap
        if props.target_armature and not props.quality_working:
            original_name = props.target_armature.get("nuranova_original_rig")
            if original_name and original_name in bpy.data.objects:
                props.quality_original = bpy.data.objects[original_name]
                props.quality_working = props.target_armature
        if not props.quality_original or not props.quality_working:
            self.report({"ERROR"}, "Fix or visually bake a copy first.")
            return {"CANCELLED"}
        show_original = props.quality_original.hide_get()
        for armature, hidden in ((props.quality_original, not show_original), (props.quality_working, show_original)):
            for obj in [armature, *bound_meshes(armature)]:
                obj.hide_set(hidden)
        return {"FINISHED"}


class NURANOVA_OT_add_contact_pin(bpy.types.Operator):
    bl_idname = "nuranova.add_contact_pin"
    bl_label = "Add contact pin / touch"

    def execute(self, context):
        props = context.scene.nuranova_mocap
        if props.target_armature is None:
            self.report({"ERROR"},"Choose the target character armature first.")
            return {"CANCELLED"}
        try:
            profile = load_profile(props.target_armature)
            pair = [value.strip() for value in props.quality_contact_pair.split(",")]
            if len(pair) != 2 or not all(pair):
                raise ValueError("Enter two regions, for example left_hand,torso.")
            profile["contact_pins"].append({"a": pair[0], "b": pair[1], "kind": props.quality_contact_kind,
                "start": props.quality_start or context.scene.frame_start, "end": props.quality_end or context.scene.frame_end})
            save_profile(props.target_armature, profile)
            props.quality_status = "Contact intent saved. Analyze/fix the affected range again."
        except (ValueError, TypeError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


QUALITY_CLASSES = (NURANOVA_OT_character_quality, NURANOVA_OT_compare_character, NURANOVA_OT_add_contact_pin)


def draw_quality_panel(layout, props):
    box = layout.box()
    box.label(text="Character contacts & correction", icon="MOD_PHYSICS")
    row = box.row(align=True)
    row.prop(props, "quality_start")
    row.prop(props, "quality_end")
    box.prop(props, "quality_margin_mm")
    row = box.row(align=True)
    row.operator("nuranova.character_quality", text="Analyze").operation = "ANALYZE"
    row.operator("nuranova.character_quality", text="Fix range (copy)").operation = "FIX"
    box.operator("nuranova.character_quality", text="Visual bake control rig to copy").operation = "BAKE"
    box.operator("nuranova.compare_character")
    box.prop(props, "quality_contact_pair")
    box.prop(props, "quality_contact_kind")
    box.operator("nuranova.add_contact_pin")
    box.label(text=props.quality_status)
    box.label(text="ESC cancels. Full report: Nuranova_Character_Quality.json")
