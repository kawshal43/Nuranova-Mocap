"""Evaluate every Blender pose and midpoint; import from Blender's Python or MCP."""

from __future__ import annotations

import numpy as np
import importlib.util
from pathlib import Path


def audit_rigs(scene, rigs, first, last, output_path):
    """Audit evaluated local quaternions, world positions, scaling, and subframe continuity."""
    # Blender ships NumPy but need not have the desktop app's SciPy dependencies.
    module_path = Path(__file__).resolve().parents[1] / "engine" / "motion_cleanup" / "quality_audit.py"
    spec = importlib.util.spec_from_file_location("nuranova_standalone_quality_audit", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    audit_quaternions = module.audit_quaternions

    buffers = {rig.name: {"q": [], "positions": [], "mid_q": [], "scales": []} for rig in rigs}
    for frame in range(first, last + 1):
        scene.frame_set(frame)
        for rig in rigs:
            data = buffers[rig.name]
            data["q"].append(
                [
                    list(b.matrix_basis.to_quaternion())[1:] + [b.matrix_basis.to_quaternion().w]
                    for b in rig.pose.bones
                ]
            )
            data["positions"].append([list(rig.matrix_world @ b.head) for b in rig.pose.bones])
            data["scales"].append([list(b.scale) for b in rig.pose.bones])
        if frame < last:
            scene.frame_set(frame, subframe=0.5)
            for rig in rigs:
                buffers[rig.name]["mid_q"].append(
                    [
                        list(b.matrix_basis.to_quaternion())[1:]
                        + [b.matrix_basis.to_quaternion().w]
                        for b in rig.pose.bones
                    ]
                )
    report = {}
    arrays = {}
    fps = scene.render.fps / scene.render.fps_base
    for index, rig in enumerate(rigs):
        data = buffers[rig.name]
        q, mid = np.asarray(data["q"]), np.asarray(data["mid_q"])
        positions, scales = np.asarray(data["positions"]), np.asarray(data["scales"])
        names = [b.name for b in rig.pose.bones]
        audit = audit_quaternions(q, names, fps)
        mid_normalized = mid / np.linalg.norm(mid, axis=-1, keepdims=True)
        signed_next = np.where(np.sum(q[:-1] * q[1:], axis=-1, keepdims=True) < 0, -q[1:], q[1:])
        expected = q[:-1] + signed_next
        expected /= np.maximum(np.linalg.norm(expected, axis=-1, keepdims=True), 1e-12)
        midpoint_error = np.degrees(
            2 * np.arccos(np.clip(np.abs(np.sum(expected * mid_normalized, axis=-1)), 0, 1))
        )
        audit["positions_finite"] = bool(np.isfinite(positions).all())
        audit["scales_finite"] = bool(np.isfinite(scales).all())
        audit["max_midpoint_error_deg"] = float(midpoint_error.max()) if midpoint_error.size else 0
        audit["midpoint_frames_tested"] = len(mid)
        audit["active_nla_tracks"] = (
            sum(not t.mute for t in rig.animation_data.nla_tracks) if rig.animation_data else 0
        )
        body = [
            i
            for i, n in enumerate(names)
            if not any(f in n for f in ("thumb", "index", "middle", "ring", "pinky"))
        ]
        audit["body_max_step_deg"] = max(audit["per_bone"][i]["max_step_deg"] for i in body)
        report[rig.name] = audit
        arrays[f"rig_{index}_rotations"] = q
        arrays[f"rig_{index}_positions"] = positions
        arrays[f"rig_{index}_midpoints"] = mid
        arrays[f"rig_{index}_names"] = np.asarray(names)
    np.savez_compressed(output_path, **arrays)
    return report
