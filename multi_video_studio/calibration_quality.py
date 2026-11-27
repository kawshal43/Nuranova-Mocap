"""Camera consistency checks using a view that was excluded from each triangulation."""

from __future__ import annotations

import itertools

import cv2
import numpy as np


def assess_calibration(calibration, observations, image_sizes=None, max_frames=180):
    cameras = calibration.cameras
    uids = [u for u in observations if u in cameras]
    if len(uids) < 2:
        return {"score_px": float("inf"), "per_camera": {}}
    count = min(len(observations[u]) for u in uids)
    if count < 3:
        return {"score_px": float("inf"), "per_camera": {}}
    frames = np.unique(np.linspace(1, max(1, count - 2), min(count, max_frames)).astype(int))
    selected = {}
    for uid in uids:
        arr = np.asarray(observations[uid][frames], dtype=float).copy()
        intr = cameras[uid].intrinsics
        w, h = (image_sizes or {}).get(uid, (intr.image_width, intr.image_height))
        arr[:, :, 0] *= intr.image_width / w
        arr[:, :, 1] *= intr.image_height / h
        selected[uid] = arr
    per_camera = {}
    scores = []
    for held in uids:
        other = [u for u in uids if u != held]
        pairs = list(itertools.combinations(other, 2)) if len(other) >= 2 else [(uids[0], uids[1])]
        errors = []
        for first, second in pairs:
            a, b, target = selected[first], selected[second], selected[held]
            valid = (a[:, :, 2] >= 0.45) & (b[:, :, 2] >= 0.45) & (target[:, :, 2] >= 0.45)
            valid[:, [0, 1, 2, 3, 4, 17, 18, 19]] = False
            if valid.sum() < 20:
                continue
            ca, cb, ch = cameras[first], cameras[second], cameras[held]
            ua = ca.intrinsics.undistort_points(a[:, :, :2][valid])
            ub = cb.intrinsics.undistort_points(b[:, :, :2][valid])
            x4 = cv2.triangulatePoints(ca.projection_matrix, cb.projection_matrix, ua.T, ub.T)
            xyz = (x4[:3] / np.where(np.abs(x4[3:]) > 1e-12, x4[3:], 1e-12)).T
            predicted = ch.project_world_points(xyz, True)
            depth = (xyz @ ch.extrinsics.R_cw.T + ch.extrinsics.t_cw)[:, 2]
            error = (
                np.linalg.norm(predicted - target[:, :, :2][valid], axis=-1)
                * 1920
                / ch.intrinsics.image_width
            )
            errors.extend(
                np.where((depth > 0.1) & np.isfinite(error), np.minimum(error, 1000), 1000).tolist()
            )
        if errors:
            median = float(np.median(errors))
            per_camera[held] = {
                "median_px": median,
                "p75_px": float(np.percentile(errors, 75)),
                "samples": len(errors),
            }
            scores.append(median)
    return {
        "score_px": float(np.mean(scores)) if scores else float("inf"),
        "per_camera": per_camera,
    }
