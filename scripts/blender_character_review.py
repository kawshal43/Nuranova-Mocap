"""Blender batch contact review; retains the original rig and writes a separate corrected copy."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "blender_addon"))

from nuranova_mocap_bridge.character_quality import (
    CharacterGeometry, build_character_profile, duplicate_character, fix_character_frame, regularize_contact_corrections, save_profile,
)
from nuranova_mocap_bridge.character_export import bound_meshes
from nuranova_mocap_bridge.retargeting import auto_match_bones


def _anchors(positions, contacts):
    targets = {}
    for side, pair in (("left", (0, 1)), ("right", (2, 3))):
        flat = np.all(contacts[:, pair] > 0, axis=1)
        t = 0
        while t < len(flat):
            if not flat[t]:
                t += 1
                continue
            end = t + 1
            while end < len(flat) and flat[end]:
                end += 1
            if end - t >= 4:
                goal = np.median(positions[side][t:min(t + 5, end)], axis=0)
                for frame in range(t, end):
                    # Blend contact transitions, leaving heel/toe pivots free.
                    weight = min(1., (frame - t + 1) / 3, (end - frame) / 3)
                    target = positions[side][frame] * (1 - weight) + goal * weight
                    targets.setdefault(str(frame + 1), {})[side] = target.tolist()
            t = end
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion-json", type=Path)
    parser.add_argument("--fix", action="store_true")
    parser.add_argument("--reuse-before", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    args.output.mkdir(parents=True, exist_ok=True)
    target, source = bpy.data.objects[args.target], bpy.data.objects[args.source]
    mapping = auto_match_bones(source, target)
    profile = build_character_profile(target, mapping)
    # A separate view layer avoids evaluating archived rigs on every QA frame.
    qa_collection = bpy.data.collections.new("Nuranova_QA_Scope")
    bpy.context.scene.collection.children.link(qa_collection)
    for obj in [target, *bound_meshes(target)]:
        qa_collection.objects.link(obj)
    layer = bpy.context.scene.view_layers.new("Nuranova_QA")
    for child in layer.layer_collection.children:
        child.exclude = child.collection != qa_collection
    if bpy.context.window:
        bpy.context.window.view_layer = layer
    start, end = [int(v) for v in target.animation_data.action.frame_range]
    geometry = CharacterGeometry(target, profile)
    identity = hashlib.sha256()
    for path in (Path(bpy.data.filepath), Path(__file__),
                 Path(__file__).resolve().parents[1] / "blender_addon/nuranova_mocap_bridge/character_quality.py"):
        if path.is_file():
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    identity.update(chunk)
    identity.update(json.dumps([target.name, start, end, profile["mapping"]], sort_keys=True).encode())
    scan_identity = identity.hexdigest()
    positions = {side: [] for side in ("left", "right")}
    before_path = args.output / "contacts_before.json"
    cached = json.loads(before_path.read_text(encoding="utf-8")) if args.reuse_before and before_path.exists() else None
    before = cached["before"] if cached and cached.get("scan_identity") == scan_identity and len(cached["before"]) == end-start+1 else []
    reused = bool(before)
    started = time.monotonic()
    for frame in range(start, end + 1):
        bpy.context.scene.frame_set(frame)
        if not reused:
            before.append(geometry.analyze(frame))
        for side in positions:
            positions[side].append(tuple(target.matrix_world @ target.pose.bones[mapping[side + "_foot"]].head))
        if frame % 100 == 0:
            print(f"SCAN {frame}/{end} {time.monotonic() - started:.1f}s", flush=True)
    report = {"target": target.name, "scan_identity":scan_identity, "profile_warnings": profile["warnings"], "before": before}
    (args.output / "contacts_before.json").write_text(json.dumps(report), encoding="utf-8")
    if args.fix:
        original = target
        target, mesh_map = duplicate_character(original)
        for obj in [original, *bound_meshes(original)]:
            obj.hide_set(True)
        target.name = original.name + "_SurfaceCorrected"
        fixed_profile = deepcopy(profile)
        fixed_profile["meshes"] = {mesh_map[name]: values for name, values in profile["meshes"].items()}
        fixed_profile["armature"] = target.name
        if args.motion_json:
            payload = json.loads(args.motion_json.read_text(encoding="utf-8"))
            contacts = np.asarray(payload["frames"].get("foot_contacts"))
            if contacts.shape == (end - start + 1, 4):
                fixed_profile["flat_stance_anchors"] = _anchors({s: np.asarray(v) for s, v in positions.items()}, contacts)
        geometry = CharacterGeometry(target, fixed_profile)
        results = []
        for frame in range(start, end + 1):
            bpy.context.scene.frame_set(frame)
            results.append(fix_character_frame(geometry, frame, iterations=3))
            if frame % 25 == 0:
                print(f"FIX {frame}/{end} {time.monotonic() - started:.1f}s", flush=True)
            if frame % 250 == 0:
                (args.output / "correction_progress.json").write_text(json.dumps({"completed":frame,"total":end}), encoding="utf-8")
        for obj in [original, *bound_meshes(original)]:
            obj.hide_set(True)
        for item in regularize_contact_corrections(original,target,start,end,
                bpy.context.scene.render.fps/bpy.context.scene.render.fps_base):
            if item.get("regularized_bones") is not None:
                report["temporal_correction"] = item
        save_profile(target, fixed_profile)
        report["corrections"] = results
        report["status"] = "correction complete; validation incomplete"
        (args.output / "correction_checkpoint.json").write_text(json.dumps(report), encoding="utf-8")
        target["nuranova_validation_status"] = report["status"]
        bpy.ops.wm.save_as_mainfile(filepath=str(args.output / "Mcapn_Character_CorrectionCheckpoint.blend"), copy=True)
    final_geometry = CharacterGeometry(target, fixed_profile if args.fix else profile)
    validation = []
    with (args.output / "validation_progress.jsonl").open("w", encoding="utf-8") as journal:
        for tick in range(start * 2, end * 2 + 1):
            frame = tick / 2
            bpy.context.scene.frame_set(int(frame), subframe=frame % 1)
            sample = final_geometry.analyze(frame)
            validation.append(sample)
            journal.write(json.dumps(sample) + "\n")
            if sample["contacts"] and frame + .25 < end:
                quarter = frame + .25
                bpy.context.scene.frame_set(int(quarter), subframe=quarter % 1)
                extra = final_geometry.analyze(quarter)
                validation.append(extra)
                journal.write(json.dumps(extra) + "\n")
            if tick % 200 == 0:
                journal.flush()
                print(f"VALIDATE {frame}/{end} {time.monotonic() - started:.1f}s", flush=True)
    report.update(validation=validation, final_target=target.name,
        validation_samples=len(validation), samples_with_contacts=sum(bool(r["contacts"]) for r in validation),
        max_depth_m=max(r["max_depth_m"] for r in validation), elapsed_s=time.monotonic() - started,
        status="review required" if any(r["contacts"] for r in validation) else "clear")
    (args.output / "character_contact_report.json").write_text(json.dumps(report), encoding="utf-8")
    target["nuranova_validation_status"] = report["status"]
    if hasattr(bpy.context.scene, "nuranova_mocap"):
        bpy.context.scene.nuranova_mocap.target_armature = target
    bpy.context.scene.frame_set(start)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output / "Mcapn_Character_Reviewed.blend"))
    print(json.dumps({k:report[k] for k in ("status","validation_samples","samples_with_contacts","max_depth_m","elapsed_s")}), flush=True)


if __name__ == "__main__":
    main()
