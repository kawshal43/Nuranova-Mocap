"""Multi-camera intrinsic, extrinsic, global bundle adjustment, and world Z-up calibration."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

from desktop.calibration.charuco_board import ChArUcoBoardConfig, ChArUcoDetection
from engine.math_core.camera_model import (
    CalibratedCamera,
    CameraExtrinsics,
    CameraIntrinsics,
    MultiCameraCalibration,
)
from engine.math_core.transforms import invert_transform, make_transform


@dataclass
class CalibrationReport:
    success: bool
    global_reprojection_rmse_px: float
    per_camera_reprojection_rmse_px: Dict[str, float]
    num_cameras: int
    num_observations: int
    known_distance_error_mm: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


def calibrate_camera_intrinsics(
    camera_uid: str,
    detections: Sequence[ChArUcoDetection],
    board_config: ChArUcoBoardConfig,
    image_size: Tuple[int, int],
    min_corners_per_view: int = 6,
) -> CameraIntrinsics:
    """Calibrate single-camera intrinsics (K, dist_coeffs) from multiple ChArUco views."""
    corners_3d_all = board_config.get_chessboard_corners_3d()
    obj_points: List[np.ndarray] = []
    img_points: List[np.ndarray] = []

    for det in detections:
        if det.num_corners < min_corners_per_view:
            continue
        obj_pts = corners_3d_all[det.corner_ids].astype(np.float32)
        img_pts = det.corners_2d.astype(np.float32)
        obj_points.append(obj_pts)
        img_points.append(img_pts)

    if len(obj_points) < 3:
        raise ValueError(
            f"Camera '{camera_uid}' needs at least 3 valid ChArUco views (found {len(obj_points)})."
        )

    w, h = int(image_size[0]), int(image_size[1])
    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_points,
        img_points,
        (w, h),
        None,
        None,
    )
    return CameraIntrinsics(
        camera_uid=camera_uid,
        image_width=w,
        image_height=h,
        camera_matrix=np.asarray(K, dtype=np.float64),
        dist_coeffs=np.asarray(dist, dtype=np.float64).reshape(-1),
        reprojection_error_px=float(rms),
    )


def estimate_board_pose_pnp(
    detection: ChArUcoDetection,
    board_config: ChArUcoBoardConfig,
    intrinsics: CameraIntrinsics,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Estimate board-to-camera pose (R_cb, t_cb) from detected ChArUco corners."""
    if detection.num_corners < 4:
        return None
    obj_pts = board_config.get_chessboard_corners_3d()[detection.corner_ids].astype(np.float64)
    img_pts = detection.corners_2d.astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts,
        img_pts,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    R_cb, _ = cv2.Rodrigues(rvec)
    return R_cb, tvec.reshape(3)


def initialize_extrinsics_from_board_frames(
    intrinsics_map: Dict[str, CameraIntrinsics],
    synced_detections: Sequence[Dict[str, ChArUcoDetection]],
    board_config: ChArUcoBoardConfig,
    reference_camera_uid: Optional[str] = None,
) -> Tuple[Dict[str, CameraExtrinsics], List[Optional[Tuple[np.ndarray, np.ndarray]]]]:
    """Initialize relative extrinsics for 2..12 cameras via pairwise board PnP + BFS pose graph.

    Args:
        intrinsics_map: Per-camera calibrated intrinsics.
        synced_detections: List of synchronized frames, each mapping camera_uid -> ChArUcoDetection.
        board_config: ChArUco physical board specification.
        reference_camera_uid: Optional anchor camera (defaults to first key).

    Returns:
        (extrinsics_map, board_poses_in_ref_cam)
    """
    cam_uids = list(intrinsics_map.keys())
    if len(cam_uids) < 2:
        raise ValueError("Multi-camera extrinsic calibration requires at least 2 cameras.")
    ref_uid = reference_camera_uid if reference_camera_uid in intrinsics_map else cam_uids[0]

    # Step 1: Estimate board pose in each camera for each frame
    frame_cam_board: List[Dict[str, np.ndarray]] = []
    for frame_dets in synced_detections:
        poses: Dict[str, np.ndarray] = {}
        for uid, det in frame_dets.items():
            if uid not in intrinsics_map:
                continue
            res = estimate_board_pose_pnp(det, board_config, intrinsics_map[uid])
            if res is not None:
                poses[uid] = make_transform(res[0], res[1])
        frame_cam_board.append(poses)

    # Step 2: Collect relative transforms T_{j <- i} = T_{j <- board} @ inv(T_{i <- board})
    pair_transforms: Dict[Tuple[str, str], List[np.ndarray]] = {}
    for poses in frame_cam_board:
        present = list(poses.keys())
        for idx_i in range(len(present)):
            for idx_j in range(len(present)):
                if idx_i == idx_j:
                    continue
                ci, cj = present[idx_i], present[idx_j]
                T_ji = poses[cj] @ invert_transform(poses[ci])
                pair_transforms.setdefault((ci, cj), []).append(T_ji)

    # Average pairwise transforms robustly (median translation + chordal L2 mean rotation)
    avg_edges: Dict[Tuple[str, str], Tuple[np.ndarray, int]] = {}
    for (ci, cj), mats in pair_transforms.items():
        R_sum = np.sum([m[:3, :3] for m in mats], axis=0)
        U, _, Vt = np.linalg.svd(R_sum)
        R_avg = U @ Vt
        if np.linalg.det(R_avg) < 0:
            U[:, -1] *= -1.0
            R_avg = U @ Vt
        t_med = np.median([m[:3, 3] for m in mats], axis=0)
        avg_edges[(ci, cj)] = (make_transform(R_avg, t_med), len(mats))

    # Step 3: BFS spanning tree from reference camera (prioritizing edges with highest co-visibility)
    T_cam_from_ref: Dict[str, np.ndarray] = {ref_uid: np.eye(4, dtype=np.float64)}
    queue: deque[str] = deque([ref_uid])
    while queue:
        curr = queue.popleft()
        candidates = [
            (cj, edge_mat, count)
            for (ci, cj), (edge_mat, count) in avg_edges.items()
            if ci == curr and cj not in T_cam_from_ref
        ]
        candidates.sort(key=lambda x: x[2], reverse=True)
        for nxt, T_nxt_curr, _ in candidates:
            T_cam_from_ref[nxt] = T_nxt_curr @ T_cam_from_ref[curr]
            queue.append(nxt)

    missing = [u for u in cam_uids if u not in T_cam_from_ref]
    if missing:
        raise ValueError(
            f"Extrinsic pose graph disconnected: cameras {missing} share no co-visible calibration frames with '{ref_uid}'."
        )

    extrinsics_map = {
        uid: CameraExtrinsics(
            camera_uid=uid,
            R_cw=T_cam_from_ref[uid][:3, :3].copy(),
            t_cw=T_cam_from_ref[uid][:3, 3].copy(),
        )
        for uid in cam_uids
    }

    # Estimate initial board poses in reference world frame
    board_poses: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []
    for poses in frame_cam_board:
        est_list = []
        for uid, T_cb in poses.items():
            if uid in T_cam_from_ref:
                T_wb = invert_transform(T_cam_from_ref[uid]) @ T_cb
                est_list.append(T_wb)
        if not est_list:
            board_poses.append(None)
        else:
            R_sum = np.sum([m[:3, :3] for m in est_list], axis=0)
            U, _, Vt = np.linalg.svd(R_sum)
            R_avg = U @ Vt
            if np.linalg.det(R_avg) < 0:
                U[:, -1] *= -1.0
                R_avg = U @ Vt
            t_med = np.median([m[:3, 3] for m in est_list], axis=0)
            board_poses.append((R_avg, t_med))

    return extrinsics_map, board_poses


def run_bundle_adjustment(
    intrinsics_map: Dict[str, CameraIntrinsics],
    extrinsics_map: Dict[str, CameraExtrinsics],
    observations_3d_init: np.ndarray,
    observations_2d: Sequence[Tuple[int, str, np.ndarray]],
    anchor_camera_uid: Optional[str] = None,
    max_nfev: int = 120,
) -> Tuple[Dict[str, CameraExtrinsics], np.ndarray, CalibrationReport]:
    """Global nonlinear multi-camera bundle adjustment (Levenberg-Marquardt / TRF with Soft-L1 loss).

    Optimizes camera extrinsics (keeping anchor camera fixed) and 3D scene points simultaneously.

    Args:
        intrinsics_map: Calibrated camera intrinsics (kept fixed during extrinsic BA).
        extrinsics_map: Initial extrinsics per camera.
        observations_3d_init: (P, 3) initial 3D point coordinates in meters.
        observations_2d: List of (point_idx, camera_uid, uv_pixel_2d) observations.
        anchor_camera_uid: Camera whose extrinsics remain fixed (gauge freedom).
    """
    cam_uids = list(extrinsics_map.keys())
    anchor = anchor_camera_uid if anchor_camera_uid in extrinsics_map else cam_uids[0]
    free_cams = [u for u in cam_uids if u != anchor]
    cam_to_free_idx = {u: idx for idx, u in enumerate(free_cams)}

    P_init = np.asarray(observations_3d_init, dtype=np.float64).reshape(-1, 3)
    num_pts = P_init.shape[0]
    num_free_cams = len(free_cams)

    # Pack parameter vector: [rvec_0, tvec_0, ..., rvec_{C-2}, tvec_{C-2}, X_0, ..., X_{P-1}]
    x0 = np.zeros(6 * num_free_cams + 3 * num_pts, dtype=np.float64)
    for i, uid in enumerate(free_cams):
        x0[6 * i : 6 * i + 3] = extrinsics_map[uid].rvec
        x0[6 * i + 3 : 6 * i + 6] = extrinsics_map[uid].t_cw
    x0[6 * num_free_cams :] = P_init.reshape(-1)

    # Group observations by camera for fast vectorized cv2.projectPoints
    obs_by_cam: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    temp_idx: Dict[str, List[int]] = {u: [] for u in cam_uids}
    temp_uv: Dict[str, List[np.ndarray]] = {u: [] for u in cam_uids}
    for pt_idx, uid, uv in observations_2d:
        if uid in temp_idx:
            temp_idx[uid].append(int(pt_idx))
            temp_uv[uid].append(np.asarray(uv, dtype=np.float64).reshape(2))
    for u in cam_uids:
        if temp_idx[u]:
            obs_by_cam[u] = (np.array(temp_idx[u], dtype=np.int32), np.vstack(temp_uv[u]))

    anchor_rvec = extrinsics_map[anchor].rvec
    anchor_tvec = extrinsics_map[anchor].t_cw.reshape(3)

    def residuals_fn(params: np.ndarray) -> np.ndarray:
        pts_3d = params[6 * num_free_cams :].reshape(num_pts, 3)
        res_chunks: List[np.ndarray] = []
        for uid, (p_indices, uv_meas) in obs_by_cam.items():
            intr = intrinsics_map[uid]
            if uid == anchor:
                rv, tv = anchor_rvec, anchor_tvec
            else:
                ci = cam_to_free_idx[uid]
                rv = params[6 * ci : 6 * ci + 3]
                tv = params[6 * ci + 3 : 6 * ci + 6]
            proj, _ = cv2.projectPoints(
                pts_3d[p_indices],
                rv.reshape(3, 1),
                tv.reshape(3, 1),
                intr.camera_matrix,
                intr.dist_coeffs,
            )
            diff = (proj.reshape(-1, 2) - uv_meas).reshape(-1)
            res_chunks.append(diff)
        return np.concatenate(res_chunks) if res_chunks else np.zeros((0,), dtype=np.float64)

    result = least_squares(
        residuals_fn,
        x0,
        method="trf",
        loss="soft_l1",
        f_scale=1.5,
        max_nfev=max_nfev,
    )

    opt_params = result.x
    opt_pts_3d = opt_params[6 * num_free_cams :].reshape(num_pts, 3)
    opt_extrinsics: Dict[str, CameraExtrinsics] = {}
    per_cam_rmse: Dict[str, float] = {}
    all_sq_errors: List[np.ndarray] = []

    for uid in cam_uids:
        if uid == anchor:
            R_cw = extrinsics_map[anchor].R_cw.copy()
            t_cw = extrinsics_map[anchor].t_cw.copy()
        else:
            ci = cam_to_free_idx[uid]
            R_cw, _ = cv2.Rodrigues(opt_params[6 * ci : 6 * ci + 3])
            t_cw = opt_params[6 * ci + 3 : 6 * ci + 6].copy()

        rmse = 0.0
        if uid in obs_by_cam:
            p_indices, uv_meas = obs_by_cam[uid]
            intr = intrinsics_map[uid]
            rv, _ = cv2.Rodrigues(R_cw)
            proj, _ = cv2.projectPoints(
                opt_pts_3d[p_indices],
                rv,
                t_cw.reshape(3, 1),
                intr.camera_matrix,
                intr.dist_coeffs,
            )
            sq = np.sum((proj.reshape(-1, 2) - uv_meas) ** 2, axis=1)
            rmse = float(np.sqrt(np.mean(sq)))
            all_sq_errors.append(sq)

        per_cam_rmse[uid] = rmse
        opt_extrinsics[uid] = CameraExtrinsics(
            camera_uid=uid,
            R_cw=R_cw,
            t_cw=t_cw,
            reprojection_error_px=rmse,
        )

    global_rmse = float(np.sqrt(np.mean(np.concatenate(all_sq_errors)))) if all_sq_errors else 0.0
    warnings: List[str] = []
    for uid, err in per_cam_rmse.items():
        if err > 1.5:
            warnings.append(f"Camera '{uid}' reprojection RMSE ({err:.2f} px) exceeds 1.5 px target.")

    report = CalibrationReport(
        success=bool(result.success or global_rmse < 2.0),
        global_reprojection_rmse_px=global_rmse,
        per_camera_reprojection_rmse_px=per_cam_rmse,
        num_cameras=len(cam_uids),
        num_observations=len(observations_2d),
        warnings=warnings,
    )
    return opt_extrinsics, opt_pts_3d, report


def run_board_bundle_adjustment(
    intrinsics_map: Dict[str, CameraIntrinsics],
    extrinsics_map: Dict[str, CameraExtrinsics],
    synced_detections: Sequence[Dict[str, ChArUcoDetection]],
    board_poses_init: Sequence[Optional[Tuple[np.ndarray, np.ndarray]]],
    board_config: ChArUcoBoardConfig,
    anchor_camera_uid: Optional[str] = None,
    max_nfev: int = 120,
) -> Tuple[Dict[str, CameraExtrinsics], List[Tuple[np.ndarray, np.ndarray]], CalibrationReport]:
    """Metric-scale-preserving ChArUco board bundle adjustment.

    Optimizes camera extrinsics (keeping anchor camera fixed) and 6-DoF rigid board poses per frame,
    locking exact physical metric scale from the measured ChArUco board corner geometry.
    """
    cam_uids = list(extrinsics_map.keys())
    anchor = anchor_camera_uid if anchor_camera_uid in extrinsics_map else cam_uids[0]
    free_cams = [u for u in cam_uids if u != anchor]
    cam_to_free_idx = {u: idx for idx, u in enumerate(free_cams)}

    corners_local = board_config.get_chessboard_corners_3d()
    valid_frames = [
        (f_idx, bp)
        for f_idx, bp in enumerate(board_poses_init)
        if bp is not None and len(synced_detections[f_idx]) >= 2
    ]
    if not valid_frames:
        raise ValueError("No valid multi-view board frames available for bundle adjustment.")

    num_free_cams = len(free_cams)
    num_frames = len(valid_frames)
    x0 = np.zeros(6 * num_free_cams + 6 * num_frames, dtype=np.float64)
    for i, uid in enumerate(free_cams):
        x0[6 * i : 6 * i + 3] = extrinsics_map[uid].rvec
        x0[6 * i + 3 : 6 * i + 6] = extrinsics_map[uid].t_cw
    for j, (_, (R_wb, t_wb)) in enumerate(valid_frames):
        rv_b, _ = cv2.Rodrigues(R_wb)
        base = 6 * num_free_cams + 6 * j
        x0[base : base + 3] = rv_b.reshape(3)
        x0[base + 3 : base + 6] = np.asarray(t_wb, dtype=np.float64).reshape(3)

    anchor_rvec = extrinsics_map[anchor].rvec
    anchor_tvec = extrinsics_map[anchor].t_cw.reshape(3)

    # Pre-pack observations per (frame_slot, camera_uid)
    packed_obs: List[Tuple[int, str, np.ndarray, np.ndarray]] = []
    total_obs_count = 0
    for slot, (f_idx, _) in enumerate(valid_frames):
        for uid, det in synced_detections[f_idx].items():
            if uid in intrinsics_map and det.num_corners >= 4:
                packed_obs.append((slot, uid, det.corner_ids, det.corners_2d))
                total_obs_count += det.num_corners

    def residuals_fn(params: np.ndarray) -> np.ndarray:
        board_R_t: List[Tuple[np.ndarray, np.ndarray]] = []
        for slot in range(num_frames):
            base = 6 * num_free_cams + 6 * slot
            Rb, _ = cv2.Rodrigues(params[base : base + 3])
            tb = params[base + 3 : base + 6]
            board_R_t.append((Rb, tb))

        chunks: List[np.ndarray] = []
        for slot, uid, c_ids, uv_meas in packed_obs:
            Rb, tb = board_R_t[slot]
            pts_w = (corners_local[c_ids] @ Rb.T) + tb
            intr = intrinsics_map[uid]
            if uid == anchor:
                rv, tv = anchor_rvec, anchor_tvec
            else:
                ci = cam_to_free_idx[uid]
                rv = params[6 * ci : 6 * ci + 3]
                tv = params[6 * ci + 3 : 6 * ci + 6]
            proj, _ = cv2.projectPoints(
                pts_w, rv.reshape(3, 1), tv.reshape(3, 1), intr.camera_matrix, intr.dist_coeffs
            )
            chunks.append((proj.reshape(-1, 2) - uv_meas).reshape(-1))
        return np.concatenate(chunks)

    result = least_squares(residuals_fn, x0, method="trf", loss="soft_l1", f_scale=1.5, max_nfev=max_nfev)
    opt = result.x

    opt_boards: List[Tuple[np.ndarray, np.ndarray]] = []
    for slot in range(num_frames):
        base = 6 * num_free_cams + 6 * slot
        Rb, _ = cv2.Rodrigues(opt[base : base + 3])
        tb = opt[base + 3 : base + 6].copy()
        opt_boards.append((Rb, tb))

    opt_extrinsics: Dict[str, CameraExtrinsics] = {}
    per_cam_sq: Dict[str, List[np.ndarray]] = {u: [] for u in cam_uids}

    for uid in cam_uids:
        if uid == anchor:
            R_cw = extrinsics_map[anchor].R_cw.copy()
            t_cw = extrinsics_map[anchor].t_cw.copy()
        else:
            ci = cam_to_free_idx[uid]
            R_cw, _ = cv2.Rodrigues(opt[6 * ci : 6 * ci + 3])
            t_cw = opt[6 * ci + 3 : 6 * ci + 6].copy()
        opt_extrinsics[uid] = CameraExtrinsics(camera_uid=uid, R_cw=R_cw, t_cw=t_cw)

    for slot, uid, c_ids, uv_meas in packed_obs:
        Rb, tb = opt_boards[slot]
        pts_w = (corners_local[c_ids] @ Rb.T) + tb
        intr = intrinsics_map[uid]
        rv, _ = cv2.Rodrigues(opt_extrinsics[uid].R_cw)
        tv = opt_extrinsics[uid].t_cw
        proj, _ = cv2.projectPoints(pts_w, rv, tv.reshape(3, 1), intr.camera_matrix, intr.dist_coeffs)
        sq = np.sum((proj.reshape(-1, 2) - uv_meas) ** 2, axis=1)
        per_cam_sq[uid].append(sq)

    per_cam_rmse: Dict[str, float] = {}
    all_sq: List[np.ndarray] = []
    for uid in cam_uids:
        if per_cam_sq[uid]:
            cat = np.concatenate(per_cam_sq[uid])
            per_cam_rmse[uid] = float(np.sqrt(np.mean(cat)))
            all_sq.append(cat)
        else:
            per_cam_rmse[uid] = 0.0
        opt_extrinsics[uid].reprojection_error_px = per_cam_rmse[uid]

    global_rmse = float(np.sqrt(np.mean(np.concatenate(all_sq)))) if all_sq else 0.0
    warnings = [
        f"Camera '{uid}' reprojection RMSE ({err:.2f} px) exceeds 1.0 px target."
        for uid, err in per_cam_rmse.items()
        if err > 1.0
    ]
    report = CalibrationReport(
        success=bool(result.success or global_rmse < 1.5),
        global_reprojection_rmse_px=global_rmse,
        per_camera_reprojection_rmse_px=per_cam_rmse,
        num_cameras=len(cam_uids),
        num_observations=total_obs_count,
        warnings=warnings,
    )
    return opt_extrinsics, opt_boards, report


def align_calibration_to_world_z_up(
    calibration: MultiCameraCalibration,
    floor_points_world: np.ndarray,
    origin_world: Optional[np.ndarray] = None,
    forward_hint_world: Optional[np.ndarray] = None,
    camera_up_hint: bool = True,
) -> MultiCameraCalibration:
    """Transform an entire MultiCameraCalibration into Blender-compatible Right-Handed Z-Up world space.

    - Fits floor plane from `floor_points_world` (>= 3 points on floor).
    - Sets +Z normal pointing toward the side where cameras reside (if `camera_up_hint=True`).
    - Sets world origin at `origin_world` (or centroid of `floor_points_world`).
    - Sets +Y along `forward_hint_world` projected onto floor plane, and +X = Y x Z.
    """
    pts = np.asarray(floor_points_world, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 3:
        raise ValueError("Floor plane calibration requires at least 3 non-collinear floor points.")

    centroid = np.mean(pts, axis=0)
    _, _, Vt = np.linalg.svd(pts - centroid)
    z_axis = Vt[-1]
    z_axis /= np.linalg.norm(z_axis)

    if camera_up_hint and calibration.cameras:
        cam_centers = np.array([c.extrinsics.camera_center_world for c in calibration.cameras.values()])
        mean_cam_offset = np.mean(cam_centers, axis=0) - centroid
        if np.dot(z_axis, mean_cam_offset) < 0:
            z_axis = -z_axis

    if forward_hint_world is not None:
        fwd = np.asarray(forward_hint_world, dtype=np.float64).reshape(3)
        fwd = fwd - np.dot(fwd, z_axis) * z_axis
        if np.linalg.norm(fwd) < 1e-8:
            fwd = np.array([0.0, 1.0, 0.0]) - np.dot([0.0, 1.0, 0.0], z_axis) * z_axis
    else:
        basis = np.array([0.0, 1.0, 0.0]) if abs(z_axis[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        fwd = basis - np.dot(basis, z_axis) * z_axis
    y_axis = fwd / np.linalg.norm(fwd)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    # Rotation matrix R_new_old maps old world vectors -> new Z-up world vectors
    R_new_old = np.vstack([x_axis, y_axis, z_axis])
    orig = centroid if origin_world is None else np.asarray(origin_world, dtype=np.float64).reshape(3)
    # p_new = R_new_old @ (p_old - orig) => p_old = R_new_old.T @ p_new + orig
    # X_cam = R_cw @ p_old + t_cw = (R_cw @ R_new_old.T) @ p_new + (R_cw @ orig + t_cw)
    new_cams: Dict[str, CalibratedCamera] = {}
    for uid, cam in calibration.cameras.items():
        R_new = cam.extrinsics.R_cw @ R_new_old.T
        t_new = cam.extrinsics.R_cw @ orig + cam.extrinsics.t_cw
        new_cams[uid] = CalibratedCamera(
            intrinsics=cam.intrinsics,
            extrinsics=CameraExtrinsics(
                camera_uid=uid,
                R_cw=R_new,
                t_cw=t_new,
                reprojection_error_px=cam.extrinsics.reprojection_error_px,
            ),
        )

    return MultiCameraCalibration(
        profile_name=calibration.profile_name,
        cameras=new_cams,
        world_up_axis="Z",
        units="meters",
        global_reprojection_rmse_px=calibration.global_reprojection_rmse_px,
        known_distance_measured_m=calibration.known_distance_measured_m,
        known_distance_expected_m=calibration.known_distance_expected_m,
        known_distance_error_mm=calibration.known_distance_error_mm,
        notes=calibration.notes,
    )


def validate_known_distance(
    point_a_world: np.ndarray,
    point_b_world: np.ndarray,
    expected_distance_m: float,
) -> Tuple[float, float]:
    """Measure Euclidean distance between two reconstructed 3D points and return (measured_m, abs_error_mm)."""
    measured = float(np.linalg.norm(np.asarray(point_a_world, dtype=np.float64) - np.asarray(point_b_world, dtype=np.float64)))
    error_mm = abs(measured - float(expected_distance_m)) * 1000.0
    return measured, error_mm
