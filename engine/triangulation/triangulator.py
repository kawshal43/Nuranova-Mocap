"""Multi-view triangulation engine: Weighted DLT, RANSAC / subset outlier rejection, and nonlinear refinement."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

from engine.data_format.motion_io import ProvenanceCode
from engine.math_core.camera_model import CalibratedCamera


@dataclass
class Observation2D:
    camera_uid: str
    u: float
    v: float
    confidence: float = 1.0


@dataclass
class TriangulationResult:
    valid: bool
    point_3d: np.ndarray  # (3,) float64 meters (NaN if invalid)
    reprojection_error_px: float
    per_camera_errors_px: Dict[str, float] = field(default_factory=dict)
    inlier_camera_uids: List[str] = field(default_factory=list)
    outlier_camera_uids: List[str] = field(default_factory=list)
    confidence: float = 0.0
    uncertainty_m: float = float("inf")
    min_ray_angle_deg: float = 0.0
    provenance: ProvenanceCode = ProvenanceCode.MISSING


def weighted_dlt_triangulate(
    cameras: Sequence[CalibratedCamera],
    undistorted_uvs: Sequence[np.ndarray],
    weights: Sequence[float],
) -> np.ndarray:
    """Solve confidence-weighted Direct Linear Transform (DLT) for a single 3D point.

    Requires at least 2 camera observations with undistorted pixel coordinates.
    """
    n = len(cameras)
    if n < 2:
        raise ValueError("Triangulation requires at least 2 calibrated camera views.")

    A = np.zeros((2 * n, 4), dtype=np.float64)
    for i, (cam, uv, w) in enumerate(zip(cameras, undistorted_uvs, weights)):
        P = cam.projection_matrix
        sw = float(np.sqrt(max(w, 1e-6)))
        A[2 * i] = sw * (uv[0] * P[2] - P[0])
        A[2 * i + 1] = sw * (uv[1] * P[2] - P[1])

    _, _, Vt = np.linalg.svd(A)
    X_hom = Vt[-1]
    if abs(X_hom[3]) < 1e-12:
        return np.full(3, np.nan, dtype=np.float64)
    return X_hom[:3] / X_hom[3]


def _ray_geometry_quality(
    cameras: Sequence[CalibratedCamera],
    point_3d: np.ndarray,
    reproj_err_px: float,
) -> Tuple[float, float]:
    """Estimate maximum inter-ray convergence angle (deg) and approximate 3D uncertainty (meters)."""
    if len(cameras) < 2 or not np.all(np.isfinite(point_3d)):
        return 0.0, float("inf")

    dirs = []
    dists = []
    f_vals = []
    for cam in cameras:
        c_pos = cam.extrinsics.camera_center_world
        vec = point_3d - c_pos
        d = float(np.linalg.norm(vec))
        if d > 1e-6:
            dirs.append(vec / d)
            dists.append(d)
            f_vals.append(0.5 * (cam.intrinsics.fx + cam.intrinsics.fy))

    if len(dirs) < 2:
        return 0.0, float("inf")

    max_angle = 0.0
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            cos_a = float(np.clip(np.dot(dirs[i], dirs[j]), -1.0, 1.0))
            ang = float(np.degrees(np.arccos(cos_a)))
            if ang > max_angle:
                max_angle = ang

    mean_dist = float(np.mean(dists))
    mean_f = max(float(np.mean(f_vals)), 1.0)
    sin_ang = max(float(np.sin(np.radians(min(max_angle, 90.0)))), 0.05)
    # Approximate spatial uncertainty = (reproj_px / f) * distance / sin(angle) / sqrt(n_views - 1)
    pixel_err = max(reproj_err_px, 0.25)
    unc_m = (pixel_err / mean_f) * mean_dist / (sin_ang * np.sqrt(len(dirs) - 1))
    return max_angle, float(unc_m)


def refine_point_nonlinear(
    point_init: np.ndarray,
    cameras: Sequence[CalibratedCamera],
    distorted_uvs: Sequence[np.ndarray],
    weights: Sequence[float],
) -> np.ndarray:
    """Refine 3D point via weighted Levenberg-Marquardt minimizing distorted reprojection error."""
    if not np.all(np.isfinite(point_init)):
        return point_init

    rvecs = [cv2.Rodrigues(c.extrinsics.R_cw)[0] for c in cameras]
    tvecs = [c.extrinsics.t_cw.reshape(3, 1) for c in cameras]
    Ks = [c.intrinsics.camera_matrix for c in cameras]
    dists = [c.intrinsics.dist_coeffs for c in cameras]
    sws = [float(np.sqrt(max(w, 1e-6))) for w in weights]
    uvs = [np.asarray(uv, dtype=np.float64).reshape(2) for uv in distorted_uvs]

    def res_fn(x: np.ndarray) -> np.ndarray:
        pt = x.reshape(1, 3)
        errs = np.empty(2 * len(cameras), dtype=np.float64)
        for i in range(len(cameras)):
            proj, _ = cv2.projectPoints(pt, rvecs[i], tvecs[i], Ks[i], dists[i])
            diff = (proj.reshape(2) - uvs[i]) * sws[i]
            errs[2 * i] = diff[0]
            errs[2 * i + 1] = diff[1]
        return errs

    opt = least_squares(res_fn, np.asarray(point_init, dtype=np.float64), method="lm", max_nfev=30)
    return opt.x if opt.success else point_init


def triangulate_point_robust(
    cameras_map: Dict[str, CalibratedCamera],
    observations: Sequence[Observation2D],
    min_confidence: float = 0.2,
    max_reproj_error_px: float = 12.0,
    refine_nonlinear: bool = True,
    provenance_if_valid: ProvenanceCode = ProvenanceCode.MEASURED_AI,
) -> TriangulationResult:
    """Robustly triangulate a single 3D point from 2..12 cameras.

    - Dynamically accepts any subset of >= 2 valid observations.
    - Undistorts observations before DLT.
    - When >= 3 views are present and reprojection error exceeds `max_reproj_error_px`,
      performs outlier view rejection (RANSAC / best-subset pruning).
    - Optionally refines final 3D coordinates in distorted pixel space via LM optimization.
    """
    valid_obs = [
        obs
        for obs in observations
        if obs.camera_uid in cameras_map
        and obs.confidence >= min_confidence
        and np.isfinite(obs.u)
        and np.isfinite(obs.v)
    ]

    if len(valid_obs) < 2:
        return TriangulationResult(
            valid=False,
            point_3d=np.full(3, np.nan, dtype=np.float64),
            reprojection_error_px=float("inf"),
            provenance=ProvenanceCode.MISSING,
        )

    cams = [cameras_map[o.camera_uid] for o in valid_obs]
    dist_uvs = [np.array([o.u, o.v], dtype=np.float64) for o in valid_obs]
    undist_uvs = [c.intrinsics.undistort_points(uv.reshape(1, 2))[0] for c, uv in zip(cams, dist_uvs)]
    weights = [o.confidence for o in valid_obs]

    def evaluate_subset(indices: Sequence[int]) -> Tuple[np.ndarray, Dict[str, float], float, bool]:
        sub_cams = [cams[i] for i in indices]
        sub_undist = [undist_uvs[i] for i in indices]
        sub_w = [weights[i] for i in indices]
        pt = weighted_dlt_triangulate(sub_cams, sub_undist, sub_w)
        if not np.all(np.isfinite(pt)):
            return pt, {}, float("inf"), False
        in_front = all(bool(c.is_in_front(pt.reshape(1, 3))[0]) for c in sub_cams)
        errs: Dict[str, float] = {}
        for i in indices:
            proj = cams[i].project_world_points(pt.reshape(1, 3), apply_distortion=True)[0]
            errs[valid_obs[i].camera_uid] = float(np.linalg.norm(proj - dist_uvs[i]))
        mean_err = float(np.mean(list(errs.values()))) if errs else float("inf")
        return pt, errs, mean_err, in_front

    all_indices = list(range(len(valid_obs)))
    best_pt, best_errs, best_mean_err, best_in_front = evaluate_subset(all_indices)
    inlier_indices = all_indices

    # Outlier rejection if error is high and we have >= 3 cameras
    if (not best_in_front or max(best_errs.values(), default=float("inf")) > max_reproj_error_px) and len(valid_obs) >= 3:
        # Try pairwise RANSAC consensus: triangulate each pair, score inliers among all views
        best_consensus: List[int] = []
        best_consensus_err = float("inf")
        best_consensus_pt = best_pt

        for i, j in combinations(all_indices, 2):
            pair_pt, _, _, pair_front = evaluate_subset([i, j])
            if not pair_front or not np.all(np.isfinite(pair_pt)):
                continue
            consensus = []
            errs_c = []
            for k in all_indices:
                if not bool(cams[k].is_in_front(pair_pt.reshape(1, 3))[0]):
                    continue
                proj = cams[k].project_world_points(pair_pt.reshape(1, 3), apply_distortion=True)[0]
                e = float(np.linalg.norm(proj - dist_uvs[k]))
                if e <= max_reproj_error_px:
                    consensus.append(k)
                    errs_c.append(e)
            if len(consensus) >= 2:
                m_err = float(np.mean(errs_c))
                if len(consensus) > len(best_consensus) or (
                    len(consensus) == len(best_consensus) and m_err < best_consensus_err
                ):
                    best_consensus = consensus
                    best_consensus_err = m_err
                    best_consensus_pt = pair_pt

        if len(best_consensus) >= 2:
            inlier_indices = best_consensus
            best_pt, best_errs, best_mean_err, best_in_front = evaluate_subset(inlier_indices)
        else:
            best_pt = best_consensus_pt

    if not best_in_front or not np.all(np.isfinite(best_pt)):
        return TriangulationResult(
            valid=False,
            point_3d=np.full(3, np.nan, dtype=np.float64),
            reprojection_error_px=float("inf"),
            provenance=ProvenanceCode.MISSING,
        )

    inlier_cams = [cams[i] for i in inlier_indices]
    inlier_dist_uvs = [dist_uvs[i] for i in inlier_indices]
    inlier_weights = [weights[i] for i in inlier_indices]

    if refine_nonlinear:
        refined = refine_point_nonlinear(best_pt, inlier_cams, inlier_dist_uvs, inlier_weights)
        if np.all(np.isfinite(refined)) and all(bool(c.is_in_front(refined.reshape(1, 3))[0]) for c in inlier_cams):
            best_pt = refined

    # Recompute exact per-camera reprojection errors on final point
    final_errs: Dict[str, float] = {}
    for i in inlier_indices:
        proj = cams[i].project_world_points(best_pt.reshape(1, 3), apply_distortion=True)[0]
        final_errs[valid_obs[i].camera_uid] = float(np.linalg.norm(proj - dist_uvs[i]))
    mean_err = float(np.mean(list(final_errs.values())))

    if mean_err > max_reproj_error_px:
        return TriangulationResult(
            valid=False,
            point_3d=np.full(3, np.nan, dtype=np.float64),
            reprojection_error_px=mean_err,
            per_camera_errors_px=final_errs,
            provenance=ProvenanceCode.MISSING,
        )

    ray_angle_deg, unc_m = _ray_geometry_quality(inlier_cams, best_pt, mean_err)
    inlier_uids = [valid_obs[i].camera_uid for i in inlier_indices]
    outlier_uids = [valid_obs[i].camera_uid for i in all_indices if i not in inlier_indices]

    # Combined confidence: detection conf * view factor * reprojection quality * geometry angle
    mean_det_conf = float(np.mean(inlier_weights))
    view_bonus = min(1.0, 0.75 + 0.125 * (len(inlier_indices) - 2))
    err_factor = float(np.exp(-0.5 * (mean_err / max(max_reproj_error_px * 0.4, 1.0)) ** 2))
    ang_factor = min(1.0, ray_angle_deg / 20.0)
    final_conf = float(np.clip(mean_det_conf * view_bonus * err_factor * max(ang_factor, 0.25), 0.0, 1.0))

    return TriangulationResult(
        valid=True,
        point_3d=best_pt,
        reprojection_error_px=mean_err,
        per_camera_errors_px=final_errs,
        inlier_camera_uids=inlier_uids,
        outlier_camera_uids=outlier_uids,
        confidence=final_conf,
        uncertainty_m=unc_m,
        min_ray_angle_deg=ray_angle_deg,
        provenance=provenance_if_valid,
    )


def triangulate_sequence(
    cameras_map: Dict[str, CalibratedCamera],
    observations_per_camera: Dict[str, np.ndarray],
    min_confidence: float = 0.2,
    max_reproj_error_px: float = 12.0,
    refine_nonlinear: bool = True,
    provenance_if_valid: ProvenanceCode = ProvenanceCode.MEASURED_AI,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Triangulate a full (T, K) sequence across cameras.

    Args:
        cameras_map: Calibrated cameras dict.
        observations_per_camera: Mapping camera_uid -> float array (T, K, 3) of [u, v, conf].

    Returns:
        points_3d (T, K, 3), confidences (T, K), reproj_errors_px (T, K), provenance (T, K) uint8
    """
    valid_cam_uids = [u for u in observations_per_camera if u in cameras_map]
    if len(valid_cam_uids) < 2:
        raise ValueError(
            f"Multi-view reconstruction requires at least 2 calibrated cameras with observations (got {len(valid_cam_uids)})."
        )

    sample_arr = observations_per_camera[valid_cam_uids[0]]
    T, K = sample_arr.shape[0], sample_arr.shape[1]

    points_3d = np.full((T, K, 3), np.nan, dtype=np.float32)
    confs = np.zeros((T, K), dtype=np.float32)
    reproj = np.full((T, K), np.nan, dtype=np.float32)
    prov = np.full((T, K), int(ProvenanceCode.MISSING), dtype=np.uint8)

    for t in range(T):
        for k in range(K):
            obs_list = []
            for uid in valid_cam_uids:
                arr = observations_per_camera[uid]
                if t < arr.shape[0]:
                    u, v, c = float(arr[t, k, 0]), float(arr[t, k, 1]), float(arr[t, k, 2])
                    if c >= min_confidence and np.isfinite(u) and np.isfinite(v):
                        obs_list.append(Observation2D(camera_uid=uid, u=u, v=v, confidence=c))
            if len(obs_list) < 2:
                continue
            res = triangulate_point_robust(
                cameras_map=cameras_map,
                observations=obs_list,
                min_confidence=min_confidence,
                max_reproj_error_px=max_reproj_error_px,
                refine_nonlinear=refine_nonlinear,
                provenance_if_valid=provenance_if_valid,
            )
            if res.valid:
                points_3d[t, k] = res.point_3d.astype(np.float32)
                confs[t, k] = float(res.confidence)
                reproj[t, k] = float(res.reprojection_error_px)
                prov[t, k] = int(res.provenance)

    return points_3d, confs, reproj, prov
