"""Rotation and keyframe continuity checks; these do not establish capture accuracy."""

from __future__ import annotations

import numpy as np


def audit_landmark_fit(joint_positions, bone_names, body26_targets, confidence):
    """Measure IK fitting error before floor leveling; targets are model estimates."""
    mapping = {
        "left_upper_arm": 5,
        "right_upper_arm": 6,
        "left_forearm": 7,
        "right_forearm": 8,
        "left_hand": 9,
        "right_hand": 10,
        "left_upper_leg": 11,
        "right_upper_leg": 12,
        "left_lower_leg": 13,
        "right_lower_leg": 14,
        "left_foot": 15,
        "right_foot": 16,
    }
    positions = np.asarray(joint_positions)
    targets = np.asarray(body26_targets)
    conf = np.asarray(confidence)
    errors, per_joint = [], []
    for bone, target_index in mapping.items():
        index = bone_names.index(bone)
        distances = np.linalg.norm(positions[:, index] - targets[:, target_index], axis=-1) * 1000
        valid = np.isfinite(distances) & (conf[:, target_index] >= 0.35)
        values = distances[valid]
        if len(values):
            errors.append(values)
            per_joint.append(
                {
                    "bone": bone,
                    "samples": len(values),
                    "median_mm": float(np.median(values)),
                    "p95_mm": float(np.percentile(values, 95)),
                    "max_mm": float(values.max()),
                }
            )
    values = np.concatenate(errors) if errors else np.array([])
    return {
        "reference": "Optimized model landmarks before floor leveling; not ground truth",
        "samples": len(values),
        "median_mm": float(np.median(values)) if len(values) else None,
        "p95_mm": float(np.percentile(values, 95)) if len(values) else None,
        "per_joint": per_joint,
    }


def audit_quaternions(quaternions, names, fps):
    q = np.asarray(quaternions, dtype=np.float64)
    if q.ndim != 3 or q.shape[-1] != 4 or q.shape[1] != len(names):
        raise ValueError("Quaternion channels must match the bone names.")
    norms = np.linalg.norm(q, axis=-1)
    if not np.all(np.isfinite(q)) or np.any(norms < 1e-8):
        return {"finite": False, "frames": len(q), "per_bone": []}
    q = q / norms[..., None]
    steps = np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(q[1:] * q[:-1], axis=-1)), 0, 1)))
    # Use duration-independent angular velocities alongside per-frame jump counts.
    per_bone = []
    for i, name in enumerate(names):
        values = steps[:, i]
        per_bone.append(
            {
                "bone": name,
                "max_step_deg": float(values.max()) if len(values) else 0,
                "p99_step_deg": float(np.percentile(values, 99)) if len(values) else 0,
                "steps_over_30": int(np.sum(values > 30)),
                "max_velocity_deg_s": float(values.max() * fps) if len(values) else 0,
                "max_step_frame": int(np.argmax(values) + 2) if len(values) else 1,
            }
        )
    return {
        "finite": True,
        "frames": len(q),
        "bones": len(names),
        "max_norm_error": float(np.max(np.abs(norms - 1))) if q.size else 0,
        "max_step_deg": float(steps.max()) if steps.size else 0,
        "steps_over_30": int(np.sum(steps > 30)),
        "antipodal_steps": int(np.sum(np.sum(q[1:] * q[:-1], axis=-1) < 0)),
        "per_bone": per_bone,
    }
