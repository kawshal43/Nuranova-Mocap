"""Dual-Mode Studio Multi-Video Calibrator:
Mode A (Highest Accuracy < 0.5 px): Sharp-Frame ChArUco Video Calibration + Huber Bundle Adjustment + Upright Z-Up
Mode B (Zero-Board Fallback): Boardless Human-Motion Self-Calibration (HumanCalib-style Essential Matrix + Joint BA + Stature Scale)
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from desktop.calibration.calibrator import (
    align_calibration_from_upright_board,
    calibrate_camera_intrinsics,
    initialize_extrinsics_from_board_frames,
    run_board_bundle_adjustment,
)
from desktop.calibration.charuco_board import (
    ChArUcoBoardConfig,
    ChArUcoDetection,
    detect_charuco_corners,
)
from engine.math_core.camera_model import (
    CalibratedCamera,
    CameraExtrinsics,
    CameraIntrinsics,
    MultiCameraCalibration,
)
from engine.math_core.transforms import invert_transform, make_transform
from multi_video_studio.pyav_sync import estimate_video_sync_offsets


def _frame_sharpness_laplacian(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(
        gray,
        (480, max(1, int(round(gray.shape[0] * 480.0 / max(gray.shape[1], 1))))),
        interpolation=cv2.INTER_AREA,
    )
    return float(cv2.Laplacian(small, cv2.CV_64F).var())


def calibrate_from_charuco_videos(
    video_paths_by_cam: Dict[str, Path],
    board_cfg: Optional[ChArUcoBoardConfig] = None,
    candidate_samples: int = 75,
    max_selected_views: int = 42,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> MultiCameraCalibration:
    """Solve studio-grade (<0.5 px) multi-camera calibration from synchronized ChArUco board videos.

    Uses PyAV audio clap sync + Laplacian motion-blur filtering to select only the sharpest shared frames!
    """
    if len(video_paths_by_cam) < 2:
        raise ValueError("ChArUco multi-video calibration requires at least 2 camera videos.")

    cfg = board_cfg or ChArUcoBoardConfig(
        squares_x=9,
        squares_y=6,
        square_length_mm=40.0,
        marker_length_mm=30.0,
        dictionary_name="DICT_5X5_250",
    )
    cam_uids = list(video_paths_by_cam.keys())
    sync_est = estimate_video_sync_offsets(
        video_paths_by_cam, enable_audio=True, enable_flash=True, enable_motion=False
    )

    meta: Dict[str, Tuple[int, int, float, int]] = {}
    for uid, vp in video_paths_by_cam.items():
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open calibration video: {vp}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if w <= 0 or h <= 0 or fc < 5:
            raise RuntimeError(f"Calibration video unreadable or too short: {vp}")
        meta[uid] = (w, h, fps if 5.0 <= fps <= 240.0 else 30.0, fc)

    offsets = {u: sync_est[u].offset_s for u in cam_uids}
    durs = {u: meta[u][3] / meta[u][2] for u in cam_uids}
    t0 = max(offsets.values())
    t1 = min(offsets[u] + durs[u] for u in cam_uids)
    if t1 <= t0 + 0.5:
        offsets = {u: 0.0 for u in cam_uids}
        t0, t1 = 0.0, min(durs.values())

    sample_times = np.linspace(
        t0 + 0.15, max(t0 + 0.25, t1 - 0.15), max(16, int(candidate_samples))
    )
    scored_frames: List[Tuple[float, Dict[str, ChArUcoDetection]]] = []
    per_cam_all: Dict[str, List[Tuple[float, ChArUcoDetection]]] = {u: [] for u in cam_uids}

    caps = {u: cv2.VideoCapture(str(video_paths_by_cam[u])) for u in cam_uids}
    try:
        for s_idx, t_glob in enumerate(sample_times):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Cancelled by user.")
            if progress_cb and s_idx % 4 == 0:
                progress_cb(
                    0.05 + 0.55 * (s_idx / len(sample_times)),
                    f"Scanning sharp ChArUco frames ({s_idx + 1}/{len(sample_times)})...",
                )
            frame_dets: Dict[str, ChArUcoDetection] = {}
            sharp_scores: List[float] = []
            for uid in cam_uids:
                w, h, fps, fc = meta[uid]
                local_t = max(0.0, float(t_glob - offsets[uid]))
                fno = int(min(fc - 1, max(0, round(local_t * fps))))
                caps[uid].set(cv2.CAP_PROP_POS_FRAMES, fno)
                ok, frame = caps[uid].read()
                if not ok or frame is None:
                    continue
                det = detect_charuco_corners(frame, cfg)
                if det.num_corners >= 6:
                    sharp = _frame_sharpness_laplacian(frame)
                    score = det.num_corners * (1.0 + min(sharp / 200.0, 2.0))
                    per_cam_all[uid].append((score, det))
                    frame_dets[uid] = det
                    sharp_scores.append(score)
            if len(frame_dets) >= 2:
                scored_frames.append((float(np.mean(sharp_scores)), frame_dets))
    finally:
        for c in caps.values():
            c.release()

    if len(scored_frames) < 4:
        counts = ", ".join(f"{u}: {len(per_cam_all[u])}" for u in cam_uids)
        raise RuntimeError(
            f"Only {len(scored_frames)} sharp shared ChArUco frames found ({counts}). "
            "Check A3/A4 board preset and make sure the board is seen by all cameras together."
        )

    # Keep the sharpest, highest-corner-count shared frames (eliminates motion-blurred outliers!)
    scored_frames.sort(key=lambda x: x[0], reverse=True)
    shared_frames = [fd for _, fd in scored_frames[:max_selected_views]]

    if progress_cb:
        progress_cb(
            0.66,
            f"Solving lens intrinsics & multi-view extrinsics ({len(shared_frames)} sharp views)...",
        )

    intr_map: Dict[str, CameraIntrinsics] = {}
    for uid in cam_uids:
        w, h, _, _ = meta[uid]
        sorted_dets = [
            d
            for _, d in sorted(per_cam_all[uid], key=lambda x: x[0], reverse=True)[
                :max_selected_views
            ]
        ]
        if len(sorted_dets) < 3:
            raise RuntimeError(
                f"Camera {uid} saw the board in only {len(sorted_dets)} frames (need >= 3)."
            )
        intr_map[uid] = calibrate_camera_intrinsics(uid, sorted_dets, cfg, (w, h))

    extr_init, board_poses = initialize_extrinsics_from_board_frames(intr_map, shared_frames, cfg)
    if progress_cb:
        progress_cb(0.84, "Running Huber-loss multi-camera bundle adjustment & Z-Up alignment...")
    extr_opt, opt_boards, ba_rep = run_board_bundle_adjustment(
        intr_map, extr_init, shared_frames, board_poses, cfg
    )

    calib = MultiCameraCalibration(
        profile_name=f"CinemaVideoCalib_{len(cam_uids)}cams",
        cameras={u: CalibratedCamera(intr_map[u], extr_opt[u]) for u in cam_uids},
        global_reprojection_rmse_px=ba_rep.global_reprojection_rmse_px,
        notes=f"Sharp-frame ChArUco video calibration ({len(shared_frames)} views; sync: {', '.join(f'{u}={sync_est[u].method}' for u in cam_uids)})",
    )

    upright_idx = next((i for i, bp in enumerate(opt_boards) if bp is not None), None)
    if upright_idx is not None and opt_boards[upright_idx] is not None:
        Rb, tb = opt_boards[upright_idx]
        calib = align_calibration_from_upright_board(
            calib, R_wb=Rb, t_wb=tb, board_config=cfg, board_center_height_m=1.15
        )

    return calib


def _validate_calibration_pair(
    calib: MultiCameraCalibration,
    aligned_body26_by_cam: Dict[str, np.ndarray],
    image_sizes_by_cam: Dict[str, Tuple[int, int]],
    max_norm_rmse_px_1080: float = 14.0,
    min_inlier_fraction: float = 0.65,
) -> Tuple[bool, str]:
    """Verify that a candidate calibration truly matches the loaded multi-camera videos.

    Prevents stale / unrelated room calibrations (e.g. a saved desk calibration from an earlier session)
    from corrupting external `.zip` or folder mocap takes!
    """
    cam_uids = [u for u in aligned_body26_by_cam.keys() if u in calib.cameras]
    if len(cam_uids) < 2:
        return False, f"Only {len(cam_uids)} camera(s) match the calibration profile."

    for u in cam_uids:
        dc = np.asarray(calib.cameras[u].intrinsics.dist_coeffs, dtype=np.float64).reshape(-1)
        if np.any(np.abs(dc[:2]) > 0.9) or np.any(np.abs(dc) > 4.0):
            return (
                False,
                f"Camera '{u}' has extreme lens distortion coefficients ({np.round(dc[:3], 2).tolist()}).",
            )
        ci = calib.cameras[u].intrinsics
        vw, vh = image_sizes_by_cam.get(u, (ci.image_width, ci.image_height))
        if ci.image_width > 0 and ci.image_height > 0 and vw > 0 and vh > 0:
            ar_c = float(ci.image_width) / float(ci.image_height)
            ar_v = float(vw) / float(vh)
            if abs(ar_c - ar_v) / max(ar_v, 1e-6) > 0.12:
                return (
                    False,
                    f"Camera '{u}' aspect ratio mismatch: calib {ci.image_width}x{ci.image_height} vs video {vw}x{vh}.",
                )

    core_kpts = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19]
    T = min(aligned_body26_by_cam[u].shape[0] for u in cam_uids)
    if T < 3:
        return False, "Too few frames to validate calibration."
    sample_f = np.linspace(0, T - 1, min(T, 40), dtype=np.int32)

    u0, u1 = cam_uids[0], cam_uids[1]
    c0, c1 = calib.cameras[u0], calib.cameras[u1]
    w0, h0 = image_sizes_by_cam.get(u0, (c0.intrinsics.image_width, c0.intrinsics.image_height))
    w1, h1 = image_sizes_by_cam.get(u1, (c1.intrinsics.image_width, c1.intrinsics.image_height))
    sx0 = float(c0.intrinsics.image_width) / max(float(w0), 1.0)
    sy0 = float(c0.intrinsics.image_height) / max(float(h0), 1.0)
    sx1 = float(c1.intrinsics.image_width) / max(float(w1), 1.0)
    sy1 = float(c1.intrinsics.image_height) / max(float(h1), 1.0)

    P0 = c0.projection_matrix
    P1 = c1.projection_matrix
    errs_1080: List[float] = []
    valid_cheirality = 0
    total_pairs = 0
    neck_ankle_m: List[float] = []

    for f in sample_f:
        pts_frame: Dict[int, np.ndarray] = {}
        for k in core_kpts:
            o0 = aligned_body26_by_cam[u0][f, k]
            o1 = aligned_body26_by_cam[u1][f, k]
            if float(o0[2]) < 0.45 or float(o1[2]) < 0.45:
                continue
            total_pairs += 1
            uv0 = np.array([o0[0] * sx0, o0[1] * sy0], dtype=np.float64)
            uv1 = np.array([o1[0] * sx1, o1[1] * sy1], dtype=np.float64)
            un0 = c0.intrinsics.undistort_points(uv0.reshape(1, 2))[0]
            un1 = c1.intrinsics.undistort_points(uv1.reshape(1, 2))[0]
            x4 = cv2.triangulatePoints(P0, P1, un0.reshape(2, 1), un1.reshape(2, 1))
            if abs(float(x4[3, 0])) < 1e-8:
                continue
            X = x4[:3, 0] / x4[3, 0]
            z0 = float((c0.extrinsics.R_cw @ X + c0.extrinsics.t_cw)[2])
            z1 = float((c1.extrinsics.R_cw @ X + c1.extrinsics.t_cw)[2])
            if z0 <= 0.15 or z1 <= 0.15 or z0 > 25.0 or z1 > 25.0:
                continue
            p0 = c0.project_world_points(X.reshape(1, 3), True)[0]
            p1 = c1.project_world_points(X.reshape(1, 3), True)[0]
            e0 = float(np.linalg.norm(p0 - uv0)) * (
                1920.0 / max(float(c0.intrinsics.image_width), 1.0)
            )
            e1 = float(np.linalg.norm(p1 - uv1)) * (
                1920.0 / max(float(c1.intrinsics.image_width), 1.0)
            )
            err = 0.5 * (e0 + e1)
            errs_1080.append(err)
            if err <= max_norm_rmse_px_1080 * 1.6:
                valid_cheirality += 1
                pts_frame[k] = X
        if 18 in pts_frame and (15 in pts_frame or 16 in pts_frame):
            ak = pts_frame[15] if 15 in pts_frame else pts_frame[16]
            neck_ankle_m.append(float(np.linalg.norm(pts_frame[18] - ak)))

    if total_pairs < 15:
        return False, "Insufficient shared keypoints to verify calibration."

    inlier_frac = float(valid_cheirality) / float(total_pairs)
    med_err = float(np.median(errs_1080)) if errs_1080 else 999.0
    if inlier_frac < min_inlier_fraction or med_err > max_norm_rmse_px_1080:
        return (
            False,
            f"Calibration does not match these videos (median reproj={med_err:.1f}px@1080p, inliers={inlier_frac*100:.0f}%).",
        )
    if neck_ankle_m:
        med_h = float(np.median(neck_ankle_m))
        if not (0.75 <= med_h <= 2.20):
            return False, f"Calibration produces unrealistic human scale (neck-ankle={med_h:.2f}m)."
    return (
        True,
        f"Calibration verified (median reproj={med_err:.1f}px@1080p, inliers={inlier_frac*100:.0f}%).",
    )


def validate_calibration_against_observations(
    calib,
    aligned_body26_by_cam,
    image_sizes_by_cam,
    max_norm_rmse_px_1080=14.0,
    min_inlier_fraction=0.65,
):
    """Check every camera using its best-conditioned partner, including third/rear views."""
    missing = set(aligned_body26_by_cam) - set(calib.cameras)
    if missing:
        return False, "Calibration is missing cameras: " + ", ".join(sorted(missing))
    uids = [u for u in aligned_body26_by_cam if u in calib.cameras]
    if len(uids) < 2:
        return False, "Fewer than two matching cameras."
    pairs = set()
    for uid in uids:
        forward = calib.cameras[uid].extrinsics.R_cw[2]
        partner = min(
            (v for v in uids if v != uid),
            key=lambda v: abs(float(np.dot(forward, calib.cameras[v].extrinsics.R_cw[2]))),
        )
        pairs.add(tuple(sorted((uid, partner))))
    messages = []
    for pair in sorted(pairs):
        okay, reason = _validate_calibration_pair(
            calib,
            {u: aligned_body26_by_cam[u] for u in pair},
            image_sizes_by_cam,
            max_norm_rmse_px_1080,
            min_inlier_fraction,
        )
        if not okay:
            return False, f"{pair[0]}/{pair[1]}: {reason}"
        messages.append(reason)
    return True, f"All {len(uids)} cameras verified. " + " ".join(messages)


def _calibrate_boardless_once(
    aligned_body26_by_cam: Dict[str, np.ndarray],  # uid -> (T, 26, 3) [u, v, conf]
    image_sizes_by_cam: Dict[str, Tuple[int, int]],
    actor_height_m: float = 1.72,
    approx_hfov_deg: float = 65.0,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> MultiCameraCalibration:
    """Boardless Human-Motion Self-Calibration using synchronized 2D human keypoints across 2+ cameras.

    1. Seeds camera intrinsics via multi-FOV epipolar check (handles mixed 4K/1080p phones!).
    2. Essential Matrix RANSAC + `cv2.recoverPose` for initial relative geometry (PnP for cams 3..N).
    3. Fast Sparse Vectorized Huber Bundle Adjustment jointly refining camera extrinsics AND focal lengths!
    4. Upright-frame stature scaling (`actor_height_m`) + upright Z-Up + studio floor alignment.
    """
    cam_uids = list(aligned_body26_by_cam.keys())
    if len(cam_uids) < 2:
        raise ValueError("Boardless multi-view self-calibration requires at least 2 cameras.")

    if progress_cb:
        progress_cb(
            0.10,
            "Running deterministic boardless self-calibration (Essential Matrix + Radial BA)...",
        )

    # Guarantee deterministic RANSAC across runs
    cv2.setRNGSeed(42)

    core_kpts = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19]
    ref_uid = cam_uids[0]
    ref_obs = aligned_body26_by_cam[ref_uid]
    T = min(arr.shape[0] for arr in aligned_body26_by_cam.values())

    # Trim extreme 3% walk-in / walk-out edges on longer takes when actor walks up to start/stop phones
    f_lo = int(0.03 * T) if T >= 120 else 0
    f_hi = int(0.97 * T) - 1 if T >= 120 else (T - 1)
    frame_indices = np.linspace(
        f_lo, max(f_lo, f_hi), min(max(1, f_hi - f_lo + 1), 180), dtype=np.int32
    )

    # Adaptive calibrated-confidence threshold: prefer crisp unobstructed joints (>=0.55), gracefully fallback to 0.40
    all_core_c = np.concatenate(
        [aligned_body26_by_cam[u][frame_indices][:, core_kpts, 2].reshape(-1) for u in cam_uids]
    )
    hi_frac = float(np.mean(all_core_c >= 0.55)) if all_core_c.size else 0.0
    conf_thresh = 0.55 if hi_frac >= 0.35 else 0.40

    def _make_K(uid: str, hfov_deg: float) -> np.ndarray:
        w, h = image_sizes_by_cam[uid]
        fx = float(w) / (2.0 * np.tan(np.radians(hfov_deg) * 0.5))
        return np.array([[fx, 0.0, 0.5 * w], [0.0, fx, 0.5 * h], [0.0, 0.0, 1.0]], dtype=np.float64)

    # 1. Seed pinhole intrinsics & pick best HFOV via Essential Matrix inliers
    first_other = cam_uids[1]
    best_hfov = approx_hfov_deg
    best_e_score = -1e9
    best_rel_init: Optional[Tuple[np.ndarray, np.ndarray]] = None

    p1_raw, p2_raw = [], []
    tgt0_obs = aligned_body26_by_cam[first_other]
    for f in frame_indices:
        for k in core_kpts:
            if ref_obs[f, k, 2] >= conf_thresh and tgt0_obs[f, k, 2] >= conf_thresh:
                p1_raw.append(ref_obs[f, k, :2])
                p2_raw.append(tgt0_obs[f, k, :2])
    if len(p1_raw) < 16 and conf_thresh > 0.35:
        p1_raw, p2_raw = [], []
        for f in frame_indices:
            for k in core_kpts:
                if ref_obs[f, k, 2] >= 0.35 and tgt0_obs[f, k, 2] >= 0.35:
                    p1_raw.append(ref_obs[f, k, :2])
                    p2_raw.append(tgt0_obs[f, k, :2])
    if len(p1_raw) < 16:
        raise RuntimeError(
            f"Boardless self-calibration could not find enough shared human keypoints between {ref_uid} and {first_other} ({len(p1_raw)})."
        )
    p1_arr = np.asarray(p1_raw, dtype=np.float64)
    p2_arr = np.asarray(p2_raw, dtype=np.float64)

    for cand_fov in (approx_hfov_deg,):
        K0 = _make_K(ref_uid, cand_fov)
        K1 = _make_K(first_other, cand_fov)
        p1_n = (np.linalg.inv(K0) @ np.column_stack([p1_arr, np.ones(len(p1_arr))]).T)[:2].T
        p2_n = (np.linalg.inv(K1) @ np.column_stack([p2_arr, np.ones(len(p2_arr))]).T)[:2].T
        E, mask = cv2.findEssentialMat(
            p1_n, p2_n, focal=1.0, pp=(0.0, 0.0), method=cv2.RANSAC, prob=0.999, threshold=0.010
        )
        if E is None or E.shape[0] < 3:
            continue
        E = E[:3, :3]
        n_pos, R_rel, t_rel, _ = cv2.recoverPose(E, p1_n, p2_n, focal=1.0, pp=(0.0, 0.0), mask=mask)
        if n_pos > best_e_score:
            best_e_score = float(n_pos)
            best_hfov = cand_fov
            best_rel_init = (
                np.asarray(R_rel, dtype=np.float64),
                np.asarray(t_rel, dtype=np.float64).reshape(3),
            )

    if best_rel_init is None:
        raise RuntimeError(
            f"Essential Matrix failed between {ref_uid} and {first_other}; use a ChArUco board calibration video."
        )

    intr_map: Dict[str, CameraIntrinsics] = {}
    for uid in cam_uids:
        w, h = image_sizes_by_cam[uid]
        intr_map[uid] = CameraIntrinsics(
            camera_uid=uid,
            image_width=w,
            image_height=h,
            camera_matrix=_make_K(uid, best_hfov),
            dist_coeffs=np.zeros(5, dtype=np.float64),
            reprojection_error_px=0.0,
        )

    extr_map: Dict[str, CameraExtrinsics] = {
        ref_uid: CameraExtrinsics(
            camera_uid=ref_uid, R_cw=np.eye(3, dtype=np.float64), t_cw=np.zeros(3, dtype=np.float64)
        ),
        first_other: CameraExtrinsics(
            camera_uid=first_other, R_cw=best_rel_init[0], t_cw=best_rel_init[1]
        ),
    }

    K_ref = intr_map[ref_uid].camera_matrix
    K_ref_inv = np.linalg.inv(K_ref)
    for uid in cam_uids[2:]:
        tgt_obs = aligned_body26_by_cam[uid]
        K_tgt = intr_map[uid].camera_matrix
        K_tgt_inv = np.linalg.inv(K_tgt)
        pts_r, pts_t = [], []
        for f in frame_indices:
            for k in core_kpts:
                if ref_obs[f, k, 2] >= conf_thresh and tgt_obs[f, k, 2] >= conf_thresh:
                    pts_r.append(ref_obs[f, k, :2])
                    pts_t.append(tgt_obs[f, k, :2])
        if len(pts_r) < 16 and conf_thresh > 0.35:
            pts_r, pts_t = [], []
            for f in frame_indices:
                for k in core_kpts:
                    if ref_obs[f, k, 2] >= 0.35 and tgt_obs[f, k, 2] >= 0.35:
                        pts_r.append(ref_obs[f, k, :2])
                        pts_t.append(tgt_obs[f, k, :2])
        if len(pts_r) < 16:
            raise RuntimeError(f"Insufficient shared keypoints between {ref_uid} and {uid}.")
        p1_n = (K_ref_inv @ np.column_stack([np.asarray(pts_r), np.ones(len(pts_r))]).T)[:2].T
        p2_n = (K_tgt_inv @ np.column_stack([np.asarray(pts_t), np.ones(len(pts_t))]).T)[:2].T
        E, mask = cv2.findEssentialMat(
            p1_n, p2_n, focal=1.0, pp=(0.0, 0.0), method=cv2.RANSAC, prob=0.999, threshold=0.012
        )
        if E is None or E.shape[0] < 3:
            raise RuntimeError(f"Essential Matrix failed between {ref_uid} and {uid}.")
        _, R_rel, t_rel, _ = cv2.recoverPose(
            E[:3, :3], p1_n, p2_n, focal=1.0, pp=(0.0, 0.0), mask=mask
        )
        extr_map[uid] = CameraExtrinsics(
            camera_uid=uid,
            R_cw=np.asarray(R_rel, dtype=np.float64),
            t_cw=np.asarray(t_rel, dtype=np.float64).reshape(3),
        )

    def _triangulate_sample_cloud() -> np.ndarray:
        P0 = intr_map[ref_uid].camera_matrix @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P1 = intr_map[first_other].camera_matrix @ np.hstack(
            [extr_map[first_other].R_cw, extr_map[first_other].t_cw.reshape(3, 1)]
        )
        R1, t1 = extr_map[first_other].R_cw, extr_map[first_other].t_cw
        c0_intr = intr_map[ref_uid]
        c1_intr = intr_map[first_other]
        cloud = np.full((len(frame_indices), 26, 3), np.nan, dtype=np.float64)
        o1_all = aligned_body26_by_cam[first_other]
        for fi, f in enumerate(frame_indices):
            for k in range(26):
                if ref_obs[f, k, 2] >= 0.35 and o1_all[f, k, 2] >= 0.35:
                    u0 = c0_intr.undistort_points(
                        ref_obs[f, k, :2].astype(np.float64).reshape(1, 2)
                    )[0]
                    u1 = c1_intr.undistort_points(
                        o1_all[f, k, :2].astype(np.float64).reshape(1, 2)
                    )[0]
                    x4 = cv2.triangulatePoints(
                        P0,
                        P1,
                        u0.reshape(2, 1),
                        u1.reshape(2, 1),
                    )
                    if abs(float(x4[3, 0])) > 1e-8:
                        X = x4[:3, 0] / x4[3, 0]
                        z0 = float(X[2])
                        z1 = float((R1 @ X + t1)[2])
                        if 0.1 < z0 < 60.0 and 0.1 < z1 < 60.0:
                            cloud[fi, k] = X
        return cloud

    pts3d_samples = _triangulate_sample_cloud()

    for uid in cam_uids[2:]:
        obj_p, img_p = [], []
        o_obs = aligned_body26_by_cam[uid]
        for fi, f in enumerate(frame_indices):
            for k in core_kpts:
                if o_obs[f, k, 2] >= 0.45 and np.all(np.isfinite(pts3d_samples[fi, k])):
                    obj_p.append(pts3d_samples[fi, k])
                    img_p.append(o_obs[f, k, :2])
        if len(obj_p) >= 8:
            ok_pnp, rvec, tvec, _ = cv2.solvePnPRansac(
                np.asarray(obj_p, dtype=np.float64),
                np.asarray(img_p, dtype=np.float64),
                intr_map[uid].camera_matrix,
                intr_map[uid].dist_coeffs,
            )
            if ok_pnp:
                R_p, _ = cv2.Rodrigues(rvec)
                extr_map[uid] = CameraExtrinsics(camera_uid=uid, R_cw=R_p, t_cw=tvec.reshape(3))

    # 3. Sparse Vectorized Huber Bundle Adjustment (jointly refining extrinsics + focal lengths + k1 distortion + human bone lengths!)
    if progress_cb:
        progress_cb(
            0.55,
            "Refining boardless camera poses, focal lengths & radial distortion via sparse Huber BA...",
        )

    ba_pts_init: List[np.ndarray] = []
    obs_pid: List[int] = []
    obs_cid: List[int] = []
    obs_uv: List[np.ndarray] = []
    obs_scale: List[float] = []
    obs_conf_w: List[float] = []
    frame_joint_pid: Dict[Tuple[int, int], int] = {}

    uid_to_cid = {u: i for i, u in enumerate(cam_uids)}
    ba_step = max(1, len(frame_indices) // 110)
    ba_min_c = min(conf_thresh, 0.48)
    for fi in range(0, len(frame_indices), ba_step):
        f = int(frame_indices[fi])
        for k in core_kpts:
            if not np.all(np.isfinite(pts3d_samples[fi, k])):
                continue
            vis_cams = [u for u in cam_uids if aligned_body26_by_cam[u][f, k, 2] >= ba_min_c]
            if len(vis_cams) >= 2:
                pid = len(ba_pts_init)
                ba_pts_init.append(pts3d_samples[fi, k].copy())
                frame_joint_pid[(fi, k)] = pid
                for u in vis_cams:
                    c_val = float(aligned_body26_by_cam[u][f, k, 2])
                    obs_pid.append(pid)
                    obs_cid.append(uid_to_cid[u])
                    obs_uv.append(aligned_body26_by_cam[u][f, k, :2].astype(np.float64))
                    obs_scale.append(1920.0 / max(float(image_sizes_by_cam[u][0]), 1.0))
                    obs_conf_w.append(float(np.sqrt(np.clip(c_val, 0.25, 1.0))))

    # Build symmetric human bone-length consistency pairs across BA frames!
    _BA_BONE_GROUPS: List[Tuple[int, int, int]] = [
        (5, 7, 0),
        (6, 8, 0),  # upper arms (L == R)
        (7, 9, 1),
        (8, 10, 1),  # forearms (L == R)
        (11, 13, 2),
        (12, 14, 2),  # thighs (L == R)
        (13, 15, 3),
        (14, 16, 3),  # shins (L == R)
        (19, 18, 4),  # torso
    ]
    bone_pa: List[int] = []
    bone_pb: List[int] = []
    bone_grp: List[int] = []
    grp_init_lens: Dict[int, List[float]] = {g: [] for g in range(5)}
    for fi in range(0, len(frame_indices), ba_step):
        for a_k, b_k, g_id in _BA_BONE_GROUPS:
            pa = frame_joint_pid.get((fi, a_k))
            pb = frame_joint_pid.get((fi, b_k))
            if pa is not None and pb is not None:
                d_ab = float(np.linalg.norm(ba_pts_init[pb] - ba_pts_init[pa]))
                if 1e-3 < d_ab < 20.0:
                    bone_pa.append(pa)
                    bone_pb.append(pb)
                    bone_grp.append(g_id)
                    grp_init_lens[g_id].append(d_ab)

    use_bone_prior = len(bone_pa) >= 20 and all(len(v) >= 3 for v in grp_init_lens.values())
    n_bone_res = len(bone_pa) if use_bone_prior else 0
    n_bone_vars = 5 if use_bone_prior else 0

    ba_rmse = 2.0
    n_cams = len(cam_uids)
    n_free = n_cams - 1
    n_pts = len(ba_pts_init)
    if n_pts >= 20:
        f0_arr = np.array([intr_map[u].fx for u in cam_uids], dtype=np.float64)
        cx_arr = np.array([intr_map[u].cx for u in cam_uids], dtype=np.float64)
        cy_arr = np.array([intr_map[u].cy for u in cam_uids], dtype=np.float64)
        k1_offset = n_cams
        cam_param_offset = 2 * n_cams
        pts_offset = 2 * n_cams + 6 * n_free
        len_offset = pts_offset + 3 * n_pts

        x0_parts = [np.zeros(n_cams, dtype=np.float64), np.zeros(n_cams, dtype=np.float64)]
        lb_parts = [
            np.full(n_cams, -0.45, dtype=np.float64),
            np.full(n_cams, -0.35, dtype=np.float64),
        ]
        ub_parts = [
            np.full(n_cams, 0.45, dtype=np.float64),
            np.full(n_cams, 0.35, dtype=np.float64),
        ]
        for u in cam_uids[1:]:
            rv, _ = cv2.Rodrigues(extr_map[u].R_cw)
            x0_parts.extend([rv.reshape(3), extr_map[u].t_cw.reshape(3)])
            lb_parts.extend([np.full(3, -np.pi), np.full(3, -50.0)])
            ub_parts.extend([np.full(3, np.pi), np.full(3, 50.0)])
        pts_init_arr = np.asarray(ba_pts_init, dtype=np.float64)
        x0_parts.append(pts_init_arr.reshape(-1))
        lb_parts.append(np.full(3 * n_pts, -100.0))
        ub_parts.append(np.full(3 * n_pts, 100.0))
        bone_w_scale = 0.0
        if use_bone_prior:
            init_ls = np.array(
                [float(np.median(grp_init_lens[g])) for g in range(5)], dtype=np.float64
            )
            x0_parts.append(init_ls)
            lb_parts.append(init_ls * 0.45)
            ub_parts.append(init_ls * 1.85)
            # Scale 1% bone length deviation to ~2.5 px-equivalent residual
            bone_w_scale = 80.0 / max(float(init_ls[4]), 0.1)

        x0 = np.concatenate(x0_parts)
        lb = np.concatenate(lb_parts)
        ub = np.concatenate(ub_parts)

        obs_pid_arr = np.asarray(obs_pid, dtype=np.int32)
        obs_cid_arr = np.asarray(obs_cid, dtype=np.int32)
        obs_uv_arr = np.asarray(obs_uv, dtype=np.float64)
        obs_sc_arr = np.asarray(obs_scale, dtype=np.float64)
        obs_cw_arr = np.asarray(obs_conf_w, dtype=np.float64)
        bone_pa_arr = (
            np.asarray(bone_pa, dtype=np.int32) if use_bone_prior else np.zeros(0, dtype=np.int32)
        )
        bone_pb_arr = (
            np.asarray(bone_pb, dtype=np.int32) if use_bone_prior else np.zeros(0, dtype=np.int32)
        )
        bone_grp_arr = (
            np.asarray(bone_grp, dtype=np.int32) if use_bone_prior else np.zeros(0, dtype=np.int32)
        )
        n_obs = len(obs_pid_arr)
        t1_norm_target = max(float(np.linalg.norm(extr_map[first_other].t_cw)), 1e-4)

        bone_res_off = 2 * n_obs + 1 + 2 * n_cams
        n_res = bone_res_off + n_bone_res
        A_sp = lil_matrix((n_res, len(x0)), dtype=np.int8)
        for o_i in range(n_obs):
            c_i = int(obs_cid_arr[o_i])
            p_i = int(obs_pid_arr[o_i])
            r0 = 2 * o_i
            A_sp[r0 : r0 + 2, c_i] = 1
            A_sp[r0 : r0 + 2, k1_offset + c_i] = 1
            if c_i > 0:
                col_c = cam_param_offset + 6 * (c_i - 1)
                A_sp[r0 : r0 + 2, col_c : col_c + 6] = 1
            col_p = pts_offset + 3 * p_i
            A_sp[r0 : r0 + 2, col_p : col_p + 3] = 1
        A_sp[2 * n_obs, cam_param_offset + 3 : cam_param_offset + 6] = 1
        for c_i in range(n_cams):
            A_sp[2 * n_obs + 1 + c_i, c_i] = 1
            A_sp[2 * n_obs + 1 + n_cams + c_i, k1_offset + c_i] = 1
        if use_bone_prior:
            for b_i in range(n_bone_res):
                row_b = bone_res_off + b_i
                ca = pts_offset + 3 * int(bone_pa_arr[b_i])
                cb = pts_offset + 3 * int(bone_pb_arr[b_i])
                A_sp[row_b, ca : ca + 3] = 1
                A_sp[row_b, cb : cb + 3] = 1
                A_sp[row_b, len_offset + int(bone_grp_arr[b_i])] = 1

        def _res_vec(x: np.ndarray) -> np.ndarray:
            f_vec = f0_arr * np.exp(x[:n_cams])
            k1_vec = x[k1_offset : k1_offset + n_cams]
            Rs = np.zeros((n_cams, 3, 3), dtype=np.float64)
            ts = np.zeros((n_cams, 3), dtype=np.float64)
            Rs[0] = np.eye(3, dtype=np.float64)
            for i in range(n_free):
                base = cam_param_offset + 6 * i
                R_i, _ = cv2.Rodrigues(x[base : base + 3])
                Rs[i + 1] = R_i
                ts[i + 1] = x[base + 3 : base + 6]
            pts3 = x[pts_offset:len_offset].reshape(n_pts, 3)

            P_obs = pts3[obs_pid_arr]
            R_obs = Rs[obs_cid_arr]
            t_obs = ts[obs_cid_arr]
            pc = np.einsum("nij,nj->ni", R_obs, P_obs) + t_obs
            z = np.where(np.abs(pc[:, 2]) > 1e-4, pc[:, 2], 1e-4)
            xn = pc[:, 0] / z
            yn = pc[:, 1] / z
            r2 = np.minimum(xn * xn + yn * yn, 4.0)
            rad = 1.0 + k1_vec[obs_cid_arr] * r2
            f_o = f_vec[obs_cid_arr]
            pu = f_o * (xn * rad) + cx_arr[obs_cid_arr]
            pv = f_o * (yn * rad) + cy_arr[obs_cid_arr]

            out = np.empty(n_res, dtype=np.float64)
            w_pix = obs_sc_arr * obs_cw_arr
            out[0 : 2 * n_obs : 2] = (pu - obs_uv_arr[:, 0]) * w_pix
            out[1 : 2 * n_obs : 2] = (pv - obs_uv_arr[:, 1]) * w_pix
            out[2 * n_obs] = (float(np.linalg.norm(ts[1])) - t1_norm_target) * 50.0
            out[2 * n_obs + 1 : 2 * n_obs + 1 + n_cams] = x[:n_cams] * 4.0
            out[2 * n_obs + 1 + n_cams : bone_res_off] = k1_vec * 8.0
            if use_bone_prior:
                ls = x[len_offset : len_offset + n_bone_vars]
                d_cur = np.linalg.norm(pts3[bone_pb_arr] - pts3[bone_pa_arr], axis=1)
                out[bone_res_off:] = (d_cur - ls[bone_grp_arr]) * bone_w_scale
            return out

        opt = least_squares(
            _res_vec,
            x0,
            bounds=(lb, ub),
            jac_sparsity=A_sp,
            method="trf",
            loss="huber",
            f_scale=4.0,
            max_nfev=45,
        )
        # Pass 2: Outlier-pruned tight Huber BA (f_scale=1.8) removes occluded single-view outliers cleanly!
        r_pass1 = opt.fun[: 2 * n_obs].reshape(n_obs, 2) / np.maximum(obs_cw_arr[:, None], 0.25)
        err_per_obs = np.linalg.norm(r_pass1, axis=1)
        inlier_obs_w = np.where(err_per_obs <= 24.0, 1.0, 0.03)
        if float(np.mean(err_per_obs <= 24.0)) >= 0.55:

            def _res_vec_p2(x: np.ndarray) -> np.ndarray:
                out = _res_vec(x)
                out[0 : 2 * n_obs : 2] *= inlier_obs_w
                out[1 : 2 * n_obs : 2] *= inlier_obs_w
                return out

            opt = least_squares(
                _res_vec_p2,
                opt.x,
                bounds=(lb, ub),
                jac_sparsity=A_sp,
                method="trf",
                loss="huber",
                f_scale=1.8,
                max_nfev=45,
            )
        r_pix_raw = _res_vec(opt.x)[: 2 * n_obs].reshape(n_obs, 2) / np.maximum(
            obs_cw_arr[:, None], 0.25
        )
        e_in = np.linalg.norm(r_pix_raw, axis=1)
        keep_e = e_in[e_in <= 28.0]
        ba_rmse = (
            float(np.sqrt(np.mean(keep_e**2)))
            if len(keep_e) >= 20
            else float(np.sqrt(np.mean(e_in**2)))
        )

        f_opt = f0_arr * np.exp(opt.x[:n_cams])
        k1_opt = opt.x[k1_offset : k1_offset + n_cams]
        for i, u in enumerate(cam_uids):
            w, h = image_sizes_by_cam[u]
            K_new = np.array(
                [[f_opt[i], 0.0, 0.5 * w], [0.0, f_opt[i], 0.5 * h], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            dc_new = np.array([float(k1_opt[i]), 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
            intr_map[u] = CameraIntrinsics(
                camera_uid=u,
                image_width=w,
                image_height=h,
                camera_matrix=K_new,
                dist_coeffs=dc_new,
                reprojection_error_px=ba_rmse,
            )
        for i, u in enumerate(cam_uids[1:]):
            base = cam_param_offset + 6 * i
            R_i, _ = cv2.Rodrigues(opt.x[base : base + 3])
            t_i = opt.x[base + 3 : base + 6]
            extr_map[u] = CameraExtrinsics(
                camera_uid=u, R_cw=R_i, t_cw=t_i, reprojection_error_px=ba_rmse
            )

        pts3d_samples = _triangulate_sample_cloud()

    # 4. Recover metric scale & Z-Up ONLY from upright standing frames (so squats/crouches NEVER tilt the world!)
    frame_spans: List[Tuple[float, int]] = []
    for fi in range(len(frame_indices)):
        neck = pts3d_samples[fi, 18]
        hips = pts3d_samples[fi, 19]
        if not (np.all(np.isfinite(neck)) and np.all(np.isfinite(hips))):
            continue
        ankles = [
            pts3d_samples[fi, ak] for ak in (15, 16) if np.all(np.isfinite(pts3d_samples[fi, ak]))
        ]
        if not ankles:
            continue
        ank_mid = np.mean(ankles, axis=0)
        span = float(np.linalg.norm(neck - ank_mid))
        if span > 1e-4:
            frame_spans.append((span, fi))

    if frame_spans:
        spans_only = np.array([s for s, _ in frame_spans], dtype=np.float64)
        upright_thresh = float(np.percentile(spans_only, 70))
        upright_fis = [fi for s, fi in frame_spans if s >= upright_thresh]
        upright_spans = [s for s, fi in frame_spans if s >= upright_thresh]
        expected_neck_ankle = 0.805 * float(actor_height_m)
        scale_factor = expected_neck_ankle / float(np.median(upright_spans))
    else:
        upright_fis = list(range(len(frame_indices)))
        scale_factor = 1.0

    # 5. Align World Z-Up from ankle-mid -> neck on upright standing frames
    up_vecs = []
    for fi in upright_fis:
        neck = pts3d_samples[fi, 18]
        ankles = [
            pts3d_samples[fi, ak] for ak in (15, 16) if np.all(np.isfinite(pts3d_samples[fi, ak]))
        ]
        if np.all(np.isfinite(neck)) and ankles:
            v_up = neck - np.mean(ankles, axis=0)
            n_up = float(np.linalg.norm(v_up))
            if n_up > 1e-4:
                up_vecs.append(v_up / n_up)
    if up_vecs:
        z_up = np.median(np.vstack(up_vecs), axis=0)
        z_up /= max(float(np.linalg.norm(z_up)), 1e-8)
    else:
        z_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)

    lr_vecs = []
    for fi in upright_fis:
        if np.all(np.isfinite(pts3d_samples[fi, 11])) and np.all(
            np.isfinite(pts3d_samples[fi, 12])
        ):
            v_lr = pts3d_samples[fi, 12] - pts3d_samples[fi, 11]
            v_lr = v_lr - np.dot(v_lr, z_up) * z_up
            if np.linalg.norm(v_lr) > 1e-4:
                lr_vecs.append(v_lr / np.linalg.norm(v_lr))
    if lr_vecs:
        x_w = np.median(np.vstack(lr_vecs), axis=0)
        x_w = x_w - np.dot(x_w, z_up) * z_up
        x_w /= max(float(np.linalg.norm(x_w)), 1e-8)
    else:
        tmp = np.array([1.0, 0.0, 0.0]) if abs(z_up[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        x_w = np.cross(tmp, z_up)
        x_w /= max(float(np.linalg.norm(x_w)), 1e-8)
    y_w = np.cross(z_up, x_w)
    y_w /= max(float(np.linalg.norm(y_w)), 1e-8)
    x_w = np.cross(y_w, z_up)

    R_no = np.vstack([x_w, y_w, z_up])

    scaled_pts = pts3d_samples * scale_factor
    pelvis_pts = [
        R_no @ scaled_pts[fi, 19] for fi in upright_fis if np.all(np.isfinite(scaled_pts[fi, 19]))
    ]
    foot_pts = [
        R_no @ scaled_pts[fi, k]
        for fi in range(len(frame_indices))
        for k in (15, 16, 20, 21, 24, 25)
        if np.all(np.isfinite(scaled_pts[fi, k]))
    ]
    cx = float(np.median([p[0] for p in pelvis_pts])) if pelvis_pts else 0.0
    cy = float(np.median([p[1] for p in pelvis_pts])) if pelvis_pts else 0.0
    floor_z = float(np.percentile([p[2] for p in foot_pts], 5)) if foot_pts else 0.0
    t_no = np.array([-cx, -cy, -floor_z], dtype=np.float64)

    T_new_from_old = make_transform(R_no, t_no)
    T_old_from_new = invert_transform(T_new_from_old)

    final_cams: Dict[str, CalibratedCamera] = {}
    for uid in cam_uids:
        R_old = extr_map[uid].R_cw
        t_old_scaled = extr_map[uid].t_cw * scale_factor
        T_c_old = make_transform(R_old, t_old_scaled)
        T_c_new = T_c_old @ T_old_from_new
        final_cams[uid] = CalibratedCamera(
            intrinsics=intr_map[uid],
            extrinsics=CameraExtrinsics(
                camera_uid=uid,
                R_cw=T_c_new[:3, :3],
                t_cw=T_c_new[:3, 3],
                reprojection_error_px=ba_rmse,
            ),
        )

    if progress_cb:
        progress_cb(
            1.0,
            f"Boardless human-motion calibration solved ({len(cam_uids)} cams · RMSE {ba_rmse:.2f} px)!",
        )

    return MultiCameraCalibration(
        profile_name=f"BoardlessHumanCalib_{len(cam_uids)}cams",
        cameras=final_cams,
        global_reprojection_rmse_px=ba_rmse,
        world_up_axis="Z",
        notes=f"Boardless Human-Motion Self-Calibration (sparse focal-length BA) scaled to {actor_height_m:.2f}m.",
    )


def calibrate_boardless_from_human_motion(
    aligned_body26_by_cam,
    image_sizes_by_cam,
    actor_height_m=1.72,
    approx_hfov_deg=65.0,
    progress_cb=None,
):
    """Choose a camera solution by independent cross-view prediction, not RANSAC count alone."""
    from multi_video_studio.calibration_quality import assess_calibration

    candidates = []
    failures = []
    seeds = (58.0, float(approx_hfov_deg), 76.0)
    for i, fov in enumerate(seeds):
        try:
            cv2.setRNGSeed(42)
            calibration = _calibrate_boardless_once(
                aligned_body26_by_cam, image_sizes_by_cam, actor_height_m, fov
            )
            report = assess_calibration(calibration, aligned_body26_by_cam, image_sizes_by_cam)
            if np.isfinite(report["score_px"]):
                candidates.append((report["score_px"], calibration, fov))
            if progress_cb:
                progress_cb(
                    (i + 1) / len(seeds),
                    f"Camera candidate {i + 1}: cross-view error {report['score_px']:.1f}px",
                )
        except (RuntimeError, ValueError, cv2.error) as exc:
            failures.append(str(exc))
    if not candidates:
        raise RuntimeError(
            "Camera estimation could not find a consistent solution. " + "; ".join(failures)
        )
    candidates.sort(key=lambda item: item[0])
    score, best, _ = candidates[0]
    summary = ", ".join(f"{fov:.0f}deg={error:.1f}px" for error, _, fov in candidates)
    best.notes += f" Candidate cross-view medians: {summary}. Selected {score:.1f}px."
    return best
