"""Calibrated palm frames and constant-length 21-point hand forward kinematics."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def default_hand_template(side="right", scale=1.0):
    sign = -1.0 if side == "left" else 1.0
    points = np.zeros((21, 3), float)
    bases = [(-.038, .018, 0), (-.030, .072, 0), (0, .080, 0), (.025, .075, 0), (.046, .063, 0)]
    lengths = [( .035, .027, .024), (.041, .026, .021), (.046, .030, .022), (.043, .028, .021), (.034, .022, .018)]
    for finger, (base, lengths_f) in enumerate(zip(bases, lengths)):
        start = 1 + 4 * finger
        points[start] = np.array(base) * [sign, 1, 1]
        direction = np.array([-.65 * sign, .76, 0]) if finger == 0 else np.array([0, 1., 0])
        for segment, length in enumerate(lengths_f):
            points[start + segment + 1] = points[start + segment] + direction * length
    return points * scale


def palm_frame(points, side="right", confidence=None):
    pts = np.asarray(points, float)
    anchors = [0, 5, 9, 17]
    if not np.isfinite(pts[anchors]).all():
        return None
    if confidence is not None and np.min(np.asarray(confidence)[anchors]) < .25:
        return None
    y = pts[9] - pts[0]
    across = pts[17] - pts[5] if side == "right" else pts[5] - pts[17]
    if np.linalg.norm(y) < .025 or np.linalg.norm(across) < .025:
        return None
    y = y / np.linalg.norm(y)
    z = np.cross(across, y)
    if np.linalg.norm(z) < .015:
        return None
    z /= np.linalg.norm(z)
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def hand_forward_kinematics(rotations, side="right", template=None, palm_rotation=None, wrist=None):
    """Finger rotations are local deltas; palm placement is an explicit separate transform."""
    template = default_hand_template(side) if template is None else np.asarray(template, float)
    q = np.asarray(rotations, float).reshape(16, 4)
    points = template.copy()
    for finger in range(5):
        lm, bone = 1 + 4 * finger, 1 + 3 * finger
        parent = np.eye(3)
        for segment in range(3):
            parent = parent @ Rotation.from_quat(q[bone + segment]).as_matrix()
            offset = template[lm + segment + 1] - template[lm + segment]
            points[lm + segment + 1] = points[lm + segment] + parent @ offset
    if palm_rotation is not None:
        matrix = np.asarray(palm_rotation)
        if matrix.shape == (4,):
            matrix = Rotation.from_quat(matrix).as_matrix()
        points = points @ matrix.T
    if wrist is not None:
        points += np.asarray(wrist)
    return points


def calibrate_hand_template(points, confidence, side="right"):
    """Estimate fixed palm/segment dimensions only from observed finite landmarks."""
    points, confidence = np.asarray(points), np.asarray(confidence)
    local, masks = [], []
    for pts, conf in zip(points, confidence):
        frame = palm_frame(pts, side, conf)
        if frame is not None:
            local.append((pts - pts[0]) @ frame)
            masks.append((conf >= .35) & np.isfinite(pts).all(axis=1))
    template = default_hand_template(side)
    if not local:
        return template
    local, masks = np.asarray(local), np.asarray(masks)
    palm_widths = np.linalg.norm(local[:, 17] - local[:, 5], axis=1)
    scale = float(np.clip(np.median(palm_widths) / np.linalg.norm(template[17] - template[5]), .65, 1.5))
    template *= scale
    for finger in range(5):
        start = 1 + finger * 4
        if masks[:, start].sum() >= 8:
            base = np.median(local[masks[:, start], start], axis=0)
            if np.linalg.norm(base) < .16 * scale:
                template[start] = base
        for segment in range(3):
            a, b = start + segment, start + segment + 1
            valid = masks[:, a] & masks[:, b]
            rest_offset = default_hand_template(side)[b] - default_hand_template(side)[a]
            nominal = np.linalg.norm(rest_offset) * scale
            length = nominal
            if valid.sum() >= 8:
                measured = np.median(np.linalg.norm(local[valid, b] - local[valid, a], axis=1))
                length = float(np.clip(measured, .75 * nominal, 1.25 * nominal))
            template[b] = template[a] + rest_offset / np.linalg.norm(rest_offset) * length
    return template
