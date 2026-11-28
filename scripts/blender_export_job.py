"""Blender-side structured export worker. Invoked by engine.export.blender_job."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import traceback

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "blender_addon"))

from nuranova_mocap_bridge.character_export import bound_meshes, export_character
from nuranova_mocap_bridge.facial_hands import apply_facial_blendshapes_to_mesh
from nuranova_mocap_bridge.retargeting import auto_match_bones, retarget_and_bake
from nuranova_mocap_bridge.source_armature import create_source_armature_from_payload, load_nmocap_json


def execute(job):
    variant = job.get("variant", "motion")
    if job.get("scene_file"):
        if job.get("in_place"):
            raise ValueError("Saved-scene export retains existing root motion. Edit it in Blender or export a generated take in place.")
        bpy.ops.wm.open_mainfile(filepath=job["scene_file"])
        target = bpy.context.scene.objects.get(job.get("target_armature", ""))
        if target is None or target.type != "ARMATURE":
            raise ValueError("Choose the corrected armature by name in the saved Blender scene.")
        return export_character(target, job["output"], variant=variant,
            frame_start=job.get("frame_start"), frame_end=job.get("frame_end"),
            embed_textures=job.get("embed_textures", True), validate=True)
    character = job.get("character_file")
    target = None
    if variant != "motion":
        if not character or not Path(character).is_file():
            raise ValueError("Choose the actual rigged character file for character exports.")
        suffix = Path(character).suffix.lower()
        if suffix == ".blend":
            bpy.ops.wm.open_mainfile(filepath=character)
        elif suffix in {".glb", ".gltf"}:
            bpy.ops.import_scene.gltf(filepath=character)
        elif suffix == ".fbx":
            bpy.ops.import_scene.fbx(filepath=character)
        else:
            raise ValueError("Rigged character inputs must be BLEND, FBX, GLB or glTF.")
        arms = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE" and obj.data.bones]
        name = job.get("target_armature")
        target = bpy.context.scene.objects.get(name) if name else (arms[0] if len(arms) == 1 else None)
        if target is None or target.type != "ARMATURE":
            raise ValueError("Choose the target armature by name. Available: " + ", ".join(o.name for o in arms))
    payload = load_nmocap_json(job["motion_json"])
    frame_count = len(payload["frames"]["root_translation"])
    source = create_source_armature_from_payload(payload, in_place=job.get("in_place", False), push_nla=False)
    if target is None:
        target = source
    else:
        constraints = [f"{pb.name}: {c.type}" for pb in target.pose.bones for c in pb.constraints if not c.mute and c.influence > 0]
        if constraints:
            raise ValueError("This character has active control constraints. Bake/map its deform rig in Blender before export: " + ", ".join(constraints[:5]))
        mapping = job.get("bone_mapping") or auto_match_bones(source, target)
        required = {"hips", "left_upper_arm", "left_forearm", "left_hand", "right_upper_arm", "right_forearm", "right_hand",
                    "left_upper_leg", "left_lower_leg", "left_foot", "right_upper_leg", "right_lower_leg", "right_foot"}
        missing = sorted(required - set(mapping))
        if missing:
            raise ValueError("Incomplete character mapping: " + ", ".join(missing))
        retarget_and_bake(source, target, mapping, 1, frame_count, in_place=job.get("in_place", False),
                          push_nla=False, action_name="Nuranova_Export_Corrected")
        for mesh in bound_meshes(target):
            apply_facial_blendshapes_to_mesh(mesh, payload)
    fps = float(job.get("fps") or payload.get("fps", 30))
    bpy.context.scene.render.fps = max(1, round(fps))
    bpy.context.scene.render.fps_base = bpy.context.scene.render.fps / fps
    report = export_character(target, job["output"], variant=variant, frame_start=job.get("frame_start", 1),
        frame_end=job.get("frame_end") or frame_count, fps=fps,
        embed_textures=job.get("embed_textures", True), validate=True)
    report.update(input_frame_count=frame_count)
    return report


if __name__ == "__main__":
    job = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text(encoding="utf-8-sig"))
    try:
        report = execute(job)
        report.update(success=True, run_id=job["run_id"])
    except Exception as error:
        traceback.print_exc()
        report = {"success": False, "run_id": job["run_id"], "error": str(error)}
    Path(job["receipt"]).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["success"]:
        raise SystemExit(1)
