"""Conservative geometric visibility weights for calibrated image evidence.

The seed body is an estimate, so blocked views are downweighted, never declared
missing. These capsules model sightlines, not character collision geometry.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def _segment_distance(origin, target, a, b):
    """Closest finite sightline/capsule-axis distance and sightline fraction."""
    u, v, w = target - origin, b - a, origin - a
    uu, uv, vv = np.sum(u*u, -1), np.sum(u*v, -1), np.sum(v*v, -1)
    uw, vw = np.sum(u*w, -1), np.sum(v*w, -1)
    denominator = uu*vv - uv*uv
    s = (uv*vw - vv*uw) / np.maximum(denominator, 1e-12)
    t = (uu*vw - uv*uw) / np.maximum(denominator, 1e-12)
    candidates = [(s, t, (denominator > 1e-12) & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)),
                  (np.zeros_like(s), np.clip(vw/np.maximum(vv, 1e-12), 0, 1), True),
                  (np.ones_like(s), np.clip((uv+vw)/np.maximum(vv, 1e-12), 0, 1), True),
                  (np.clip(-uw/np.maximum(uu, 1e-12), 0, 1), np.zeros_like(s), True),
                  (np.clip((uv-uw)/np.maximum(uu, 1e-12), 0, 1), np.ones_like(s), True)]
    distances, fractions = [], []
    for rs, cs, valid in candidates:
        delta = w + rs[..., None]*u - cs[..., None]*v
        distances.append(np.where(valid, np.sum(delta*delta, -1), np.inf))
        fractions.append(rs)
    distances = np.stack(distances)
    nearest = np.argmin(distances, axis=0)
    return np.sqrt(np.take_along_axis(distances, nearest[None], axis=0)[0]), np.take_along_axis(
        np.stack(fractions), nearest[None], axis=0)[0]


def visibility_weights(cameras, points, confidence, fps=30.):
    points, confidence = np.asarray(points), np.asarray(confidence)
    count, joints = points.shape[:2]
    output = {}
    # Only thick, independently observed segments can veto a sightline. Radius
    # scales with the actor instead of assuming every performer is an adult.
    shoulder_width = np.linalg.norm(points[:, 5]-points[:, 6], axis=1)
    valid_scale = (np.minimum(confidence[:, 5], confidence[:, 6]) >= .5) & np.isfinite(shoulder_width)
    scale = np.clip(np.median(shoulder_width[valid_scale])/.38, .5, 1.7) if valid_scale.any() else 1.
    capsules = [(19, 18, .135), (11, 13, .065), (12, 14, .065), (5, 7, .045), (6, 8, .045)]
    for uid, camera in cameras.items():
        weights = np.ones((count, joints), float)
        origin = np.asarray(camera.extrinsics.camera_center_world)[None, None, :]
        target = points
        ray_length = np.linalg.norm(target-origin, axis=-1)
        for a, b, nominal in capsules:
            radius = nominal*scale
            reliable = (np.minimum(confidence[:, a], confidence[:, b]) >= .5)
            reliable &= np.isfinite(points[:, [a, b]]).all(axis=(1, 2))
            reliable &= np.linalg.norm(points[:, a]-points[:, b], axis=1) >= .04*scale
            distance, fraction = _segment_distance(origin, target, points[:, a, None], points[:, b, None])
            blocked = reliable[:, None] & (distance < radius*.8)
            blocked &= (1-fraction)*ray_length > max(.06*scale, radius*1.2)
            blocked[:, [a, b]] = False
            weights[blocked] = np.minimum(weights[blocked], .25)
        if count > 2:
            weights = gaussian_filter1d(weights, max(.5, fps*.035), axis=0, mode="nearest")
        output[uid] = weights.astype(np.float32)
    return output


def weight_body_observations(observations, weights):
    """Copy only the confidence channel; retain native pixel coordinates."""
    result = {}
    for uid, values in observations.items():
        copy = np.asarray(values).copy()
        if uid in weights:
            copy[..., 2] *= weights[uid]
        result[uid] = copy
    return result


def protect_manual_views(weights, review, canonical_times, marker_mask=None):
    """Manual points and measured marker constraints override estimated occlusion."""
    if marker_mask is not None:
        for values in weights.values():
            values[np.asarray(marker_mask, bool)] = 1.
    times = np.asarray(canonical_times)
    tolerance = float(np.median(np.diff(times)))*.55 if len(times) > 1 else .02
    for point in review.points:
        uid, key = point["camera_uid"], point["key"]
        if uid not in weights or not point["valid"]:
            continue
        parts = key.split(":")
        joint = int(parts[1]) if len(parts) == 2 and parts[0] == "body" else (
            9 if key.startswith("hand:left:") else 10 if key.startswith("hand:right:") else None)
        if joint is None or not 0 <= joint < weights[uid].shape[1]:
            continue
        stamp = review.global_time(uid, point["pts_s"])
        weights[uid][abs(times-stamp) <= tolerance, joint] = 1.
