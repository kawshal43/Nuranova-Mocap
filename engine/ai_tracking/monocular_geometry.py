"""Explicit camera units and sequence-level scale for single-view estimates.

RTMLib RTMPose3d returns crop XY pixels and root-relative Z meters, not XYZ
meters. Native image XY must be unprojected before any skeletal fitting.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

# Proper rotation (det +1), matching the existing body solver's Z-up convention.
CAMERA_TO_STUDIO = np.array([[-1., 0., 0.], [0., 0., -1.], [0., -1., 0.]])


def unproject_native(uv, relative_depth, root_depth, width, height, focal_px):
    uv = np.asarray(uv, float)
    depth = np.asarray(relative_depth, float) + np.asarray(root_depth)[..., None]
    camera = np.stack(((uv[..., 0] - width / 2) * depth / focal_px,
                       (uv[..., 1] - height / 2) * depth / focal_px, depth), axis=-1)
    return camera @ CAMERA_TO_STUDIO.T


def map_coco_xyz_to_body26(points):
    """Map anatomical identities without exchanging left and right."""
    p = np.asarray(points)
    out = np.full(p.shape[:-2] + (26, 3), np.nan)
    out[..., :17, :] = p[..., :17, :]
    out[..., 18, :] = (p[..., 5, :] + p[..., 6, :]) / 2
    out[..., 19, :] = (p[..., 11, :] + p[..., 12, :]) / 2
    ears = (p[..., 3, :] + p[..., 4, :]) / 2
    out[..., 17, :] = ears + .35 * (ears - out[..., 18, :])
    for dest, src in ((20, 17), (22, 18), (24, 19), (21, 20), (23, 21), (25, 22)):
        out[..., dest, :] = p[..., src, :]
    return out.astype(np.float32)


def _smooth(values, valid, sigma):
    """Confidence-weighted filtering; gaps never become observed confidence."""
    x = np.asarray(values, float)
    w = np.asarray(valid, float) * np.isfinite(x)
    numerator = gaussian_filter1d(np.nan_to_num(x) * w, sigma, axis=0, mode="nearest")
    denominator = gaussian_filter1d(w, sigma, axis=0, mode="nearest")
    return np.divide(numerator, denominator, out=np.zeros_like(x), where=denominator > 1e-6)


def fit_monocular_sequence(relative, observations, times, width, height, actor_height_m,
                           root_mode="estimated", focal_px=None, pins=None, depth_hints=(), camera_moving=False):
    """Estimate one scale/floor and camera-space root; retain crouches and flight.

    Perspective root is a robust fit to model-relative geometry. Its scale and
    trajectory remain estimates, even when the image residual is small.
    """
    p = np.asarray(relative, float).copy()
    uv = np.asarray(observations, float)
    confidence = uv[..., 2].copy()
    dt = float(np.median(np.diff(times)))
    fps = 1 / max(dt, .001)
    focal = float(focal_px or .84 * max(width, height))
    pelvis = (p[:, 11] + p[:, 12]) / 2
    p -= pelvis[:, None]
    # A chain length survives crouches; an axis-aligned bounding box does not.
    chains = ((17, 18), (18, 19), (11, 13), (13, 15), (12, 14), (14, 16))
    lengths = []
    for a, b in chains:
        valid = (confidence[:, a] > .3) & (confidence[:, b] > .3)
        distance = np.linalg.norm(p[:, a] - p[:, b], axis=-1)
        good = distance[valid & np.isfinite(distance) & (distance > .01)]
        lengths.append(float(np.median(good)) if len(good) else np.nan)
    reference_height = lengths[0] + lengths[1] + .5 * sum(lengths[2:]) + .10
    if not np.isfinite(reference_height) or reference_height < .4:
        raise RuntimeError("Not enough reliable full-body geometry to estimate actor scale.")
    scale = float(actor_height_m / reference_height)
    p *= scale
    camera_relative = p @ CAMERA_TO_STUDIO
    rays = (uv[..., :2] - [width / 2, height / 2]) / focal
    roots = np.full((len(p), 3), np.nan)
    fit_ids = [0, 5, 6, 11, 12, 13, 14, 15, 16, 18, 19]
    for i in range(len(p)):
        ids = [j for j in fit_ids if confidence[i, j] > .3
               and np.isfinite(camera_relative[i, j]).all() and np.isfinite(rays[i, j]).all()]
        if len(ids) < 4:
            continue
        r, q = rays[i, ids], camera_relative[i, ids]
        a = np.zeros((len(ids), 2, 3))
        a[:, 0, 0], a[:, 1, 1] = 1, 1
        a[:, :, 2] = -r
        b = r * q[:, 2, None] - q[:, :2]
        weights = np.repeat(np.sqrt(confidence[i, ids]), 2)
        a, b = a.reshape(-1, 3), b.ravel()
        for _ in range(2):
            solution = np.linalg.lstsq(a * weights[:, None], b * weights, rcond=None)[0]
            residual = abs(a @ solution - b)
            weights *= np.minimum(1., .08 / np.maximum(residual, 1e-6)) ** .5
        if .3 < solution[2] < 30:
            roots[i] = solution
    valid_roots = np.isfinite(roots)
    if np.count_nonzero(valid_roots[:, 2]) < 2:
        raise RuntimeError("Cannot estimate a stable camera distance. Check performer visibility.")
    roots = _smooth(roots, valid_roots, max(.5, fps * .10))
    studio_roots = roots @ CAMERA_TO_STUDIO.T
    if camera_moving:
        studio_roots[:] = studio_roots[0]
        root_mode = "in_place"
    if root_mode == "in_place":
        studio_roots[:, :2] = 0
    else:
        studio_roots[:, :2] -= studio_roots[0, :2]
    fitted = p + studio_roots[:, None]
    # Corrections change camera rays while preserving estimated relative depth.
    # They are applied before the anatomical solver, rather than editing metadata.
    if pins is not None:
        corrected = unproject_native(uv[..., :2], camera_relative[..., 2], roots[:, 2],
                                     width, height, focal)
        corrected -= (roots @ CAMERA_TO_STUDIO.T)[:, None]
        corrected += studio_roots[:, None]
        valid_pin = np.asarray(pins) & (confidence > 0) & np.isfinite(corrected).all(axis=-1)
        fitted[valid_pin] = corrected[valid_pin]
    # Optional user depth order: an interval, joint, and signed relative offset.
    for hint in depth_hints:
        mask = (np.asarray(times) >= hint["start_s"]) & (np.asarray(times) <= hint["end_s"])
        joint = int(hint["joint"])
        if not 0 <= joint < 26:
            raise ValueError("Depth hint joint must be in Body26.")
        fitted[mask, joint, 1] -= float(hint["offset_m"])
    feet = fitted[:, [20, 21, 24, 25], 2]
    valid = (confidence[:, [20, 21, 24, 25]] > .3) & np.isfinite(feet)
    floor = float(np.percentile(feet[valid], 5)) if valid.any() else 0.
    fitted[..., 2] -= floor
    fitted[confidence <= .05] = np.nan
    return fitted.astype(np.float32), confidence.astype(np.float32), {
        "actor_height_m": actor_height_m, "relative_scale": scale,
        "focal_length_px": focal, "intrinsics_status": "supplied" if focal_px else "estimated",
        "root_mode": root_mode, "trajectory_status": "estimated", "floor_z_before_leveling": floor,
        "moving_camera_fallback": bool(camera_moving),
        "root_camera_m": roots.tolist(), "depth_status": "single-view model estimate",
    }
