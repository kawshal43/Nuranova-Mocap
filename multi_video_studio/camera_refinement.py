"""Refine one camera against trusted views without moving the world reference."""

from __future__ import annotations

from copy import deepcopy

import cv2
import numpy as np

from engine.math_core.camera_model import MultiCameraCalibration
from engine.triangulation.triangulator import triangulate_sequence


def refine_camera_from_views(calibration, camera_uid, aligned_observations, max_error_px=12):
    trusted = {u: c for u, c in calibration.cameras.items()
               if u != camera_uid and u in aligned_observations}
    if len(trusted) < 2:
        raise ValueError("Camera refinement needs two other trusted views, or known 3D calibration points.")
    points, confidence, _, _ = triangulate_sequence(
        trusted, {u: aligned_observations[u] for u in trusted},
        min_confidence=.45, max_reproj_error_px=max_error_px, refine_nonlinear=False,
    )
    target = aligned_observations[camera_uid]
    core = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    frame_indices = np.linspace(0, len(points) - 1, min(120, len(points))).astype(int)
    xyz, uv, validation = [], [], []
    for frame in frame_indices:
        for k in core:
            if confidence[frame, k] >= .45 and target[frame, k, 2] >= .45 and np.isfinite(points[frame, k]).all():
                xyz.append(points[frame, k])
                uv.append(target[frame, k, :2])
                validation.append(frame % 5 == 0)
    return refine_camera_from_points(calibration, camera_uid, xyz, uv, validation, max_error_px)


def refine_camera_from_points(calibration, camera_uid, world_points, image_points,
                              validation_mask=None, max_error_px=12):
    xyz, uv = np.asarray(world_points, float).reshape(-1, 3), np.asarray(image_points, float).reshape(-1, 2)
    if len(xyz) != len(uv) or len(xyz) < 12 or not np.isfinite(xyz).all() or not np.isfinite(uv).all():
        raise ValueError("At least 12 finite 3D/image point correspondences are required.")
    if np.linalg.matrix_rank(xyz - xyz.mean(axis=0), tol=1e-5) < 2:
        raise ValueError("Calibration points are collinear; use spatially distributed points.")
    holdout = np.asarray(validation_mask, bool) if validation_mask is not None else np.arange(len(xyz)) % 5 == 0
    if holdout.sum() < 3 or (~holdout).sum() < 6:
        holdout = np.arange(len(xyz)) % 5 == 0
    camera = calibration.cameras[camera_uid]
    intr = camera.intrinsics
    okay, rvec, tvec, inliers = cv2.solvePnPRansac(
        xyz[~holdout], uv[~holdout], intr.camera_matrix, intr.dist_coeffs,
        iterationsCount=300, reprojectionError=max_error_px, confidence=.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not okay or inliers is None or len(inliers) < 6:
        raise ValueError("Camera pose could not be verified against these correspondences.")
    ii = inliers.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(xyz[~holdout][ii], uv[~holdout][ii],
                                     intr.camera_matrix, intr.dist_coeffs, rvec, tvec)
    projected, _ = cv2.projectPoints(xyz, rvec, tvec, intr.camera_matrix, intr.dist_coeffs)
    errors = np.linalg.norm(projected.reshape(-1, 2) - uv, axis=1)
    rotation = cv2.Rodrigues(rvec)[0]
    depth = (xyz @ rotation.T + tvec.reshape(3))[:, 2]
    if np.mean(depth > .05) < .95 or np.median(errors[holdout]) > max_error_px:
        raise ValueError("Refined pose failed held-out reprojection or positive-depth validation.")
    candidate = MultiCameraCalibration.from_dict(deepcopy(calibration.to_dict()))
    candidate.cameras[camera_uid].extrinsics.R_cw = rotation
    candidate.cameras[camera_uid].extrinsics.t_cw = tvec.reshape(3)
    candidate.cameras[camera_uid].extrinsics.reprojection_error_px = float(np.sqrt(np.mean(errors**2)))
    before = np.linalg.norm(camera.project_world_points(xyz, True) - uv, axis=1)
    if np.median(errors[holdout]) > max(.25, np.median(before[holdout]) * 1.02):
        raise ValueError("Refinement would worsen held-out reprojection; the trusted pose was retained.")
    return candidate, {"camera_uid": camera_uid, "correspondences": len(xyz),
                       "training_inliers": len(ii), "held_out": int(holdout.sum()),
                       "before_median_px": float(np.median(before[holdout])),
                       "after_median_px": float(np.median(errors[holdout])),
                       "after_p95_px": float(np.percentile(errors[holdout], 95))}
