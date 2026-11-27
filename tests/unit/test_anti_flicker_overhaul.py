"""Unit & regression tests for the Anti-Flicker Overhaul (SimCC calibration, 2D cleaner, radial DLT, global 3D optimizer, hands)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from engine.ai_tracking.pose_tracker import calibrate_rtm_scores
from engine.hands.hand_tracker import HandCaptureEngine, HandObservation
from engine.math_core.camera_model import (
    CalibratedCamera,
    CameraExtrinsics,
    CameraIntrinsics,
)
from engine.math_core.transforms import quat_identity
from engine.triangulation.triangulator import (
    Observation2D,
    triangulate_point_robust,
    triangulate_sequence,
)
from multi_video_studio.multiview_3d_optimizer import optimize_multiview_3d_global
from multi_video_studio.obs2d_cleaner import clean_synchronized_2d_tracks


def _make_dist_cam(uid: str, pos: np.ndarray, target: np.ndarray, k1: float = -0.08) -> CalibratedCamera:
    fwd = target - pos
    fwd /= np.linalg.norm(fwd)
    up_w = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(fwd, up_w)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R_cw = np.vstack([right, down, fwd])
    t_cw = -R_cw @ pos
    K = np.array([[1650.0, 0.0, 960.0], [0.0, 1650.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return CalibratedCamera(
        intrinsics=CameraIntrinsics(
            camera_uid=uid,
            image_width=1920,
            image_height=1080,
            camera_matrix=K,
            dist_coeffs=np.array([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        ),
        extrinsics=CameraExtrinsics(camera_uid=uid, R_cw=R_cw, t_cw=t_cw),
    )


def test_simcc_score_calibration_separates_occluded_from_visible() -> None:
    raw_simcc = np.array([1.8, 2.5, 3.5, 4.9, 6.8], dtype=np.float32)
    cal = calibrate_rtm_scores(raw_simcc)
    assert cal[0] <= 0.06  # clearly occluded
    assert cal[1] <= 0.15  # weak guess
    assert 0.55 <= cal[3] <= 0.70  # median visible
    assert cal[4] >= 0.98  # crisp visible

    # Standard [0, 1] models pass through untouched
    raw_prob = np.array([0.0, 0.35, 0.82, 1.0], dtype=np.float32)
    assert np.allclose(calibrate_rtm_scores(raw_prob), raw_prob)


def test_obs2d_cleaner_repairs_lr_swaps_and_spikes() -> None:
    T = 40
    t = np.linspace(0.0, 1.0, T)
    uvc = np.zeros((T, 26, 3), dtype=np.float32)
    uvc[:, :, 2] = 0.85
    # Smoothly moving left wrist (9) and right wrist (10)
    uvc[:, 9, 0] = 600.0 + 40.0 * t
    uvc[:, 9, 1] = 500.0
    uvc[:, 10, 0] = 1100.0 + 40.0 * t
    uvc[:, 10, 1] = 500.0
    # Inject a single-frame L/R wrist swap at frame 15
    uvc[15, 9, :2], uvc[15, 10, :2] = uvc[15, 10, :2].copy(), uvc[15, 9, :2].copy()
    # Inject a 300px single-frame occlusion spike on left elbow (7) at frame 25
    uvc[:, 7, 0] = 680.0
    uvc[:, 7, 1] = 420.0
    uvc[25, 7, 0] += 320.0

    cleaned, n_rep = clean_synchronized_2d_tracks({"cam_01": uvc}, {"cam_01": (1920, 1080)}, fps=30.0)
    assert n_rep >= 2
    out = cleaned["cam_01"]
    # L/R swap at frame 15 restored within a few pixels
    assert abs(float(out[15, 9, 0]) - (600.0 + 40.0 * t[15])) < 8.0
    assert abs(float(out[15, 10, 0]) - (1100.0 + 40.0 * t[15])) < 8.0
    # 320px spike at frame 25 repaired cleanly
    assert abs(float(out[25, 7, 0]) - 680.0) < 10.0


def test_vectorized_triangulate_sequence_matches_scalar_with_radial_distortion() -> None:
    rng = np.random.default_rng(7)
    target = np.array([0.0, 0.0, 1.0])
    cams = {
        "c1": _make_dist_cam("c1", np.array([3.0, 0.2, 1.5]), target, k1=-0.10),
        "c2": _make_dist_cam("c2", np.array([0.2, 3.0, 1.5]), target, k1=-0.06),
        "c3": _make_dist_cam("c3", np.array([-2.6, 1.8, 1.3]), target, k1=-0.12),
    }
    T, K = 10, 6
    pts_gt = rng.uniform([-0.7, -0.7, 0.3], [0.7, 0.7, 1.8], size=(T, K, 3))
    obs = {}
    for uid, cam in cams.items():
        arr = np.zeros((T, K, 3), dtype=np.float32)
        for t in range(T):
            uv = cam.project_world_points(pts_gt[t], apply_distortion=True)
            arr[t, :, :2] = uv + rng.normal(scale=1.2, size=(K, 2))
            arr[t, :, 2] = 0.85
        obs[uid] = arr
    obs["c3"][3, 2, :2] += 160.0  # outlier

    p_vec, _, e_vec, _ = triangulate_sequence(cams, obs, min_confidence=0.25, max_reproj_error_px=24.0, refine_nonlinear=False)
    for t in range(T):
        for k in range(K):
            o_list = [Observation2D(u, float(obs[u][t, k, 0]), float(obs[u][t, k, 1]), float(obs[u][t, k, 2])) for u in cams]
            res = triangulate_point_robust(cams, o_list, min_confidence=0.25, max_reproj_error_px=24.0, refine_nonlinear=False)
            assert np.all(np.isfinite(p_vec[t, k])) == res.valid
            if res.valid:
                assert float(np.linalg.norm(p_vec[t, k] - res.point_3d)) < 1e-4
                assert abs(float(e_vec[t, k]) - float(res.reprojection_error_px)) < 1e-2


def test_global_multiview_3d_optimizer_eliminates_teleports_and_keeps_rigid_torso() -> None:
    target = np.array([0.0, 0.0, 1.0])
    cams = {
        "c1": _make_dist_cam("c1", np.array([3.2, 0.0, 1.5]), target, k1=-0.05),
        "c2": _make_dist_cam("c2", np.array([0.0, 3.2, 1.5]), target, k1=0.0),
        "c3": _make_dist_cam("c3", np.array([-2.6, 2.2, 1.4]), target, k1=-0.04),
    }
    T, K = 30, 26
    gt = np.zeros((T, K, 3), dtype=np.float64)
    # Build simple standing humanoid moving smoothly along +X
    base = np.zeros((K, 3), dtype=np.float64)
    base[19] = [0.0, 0.0, 0.95]   # pelvis
    base[18] = [0.0, 0.0, 1.45]   # neck
    base[5], base[6] = [-0.18, 0.0, 1.42], [0.18, 0.0, 1.42]
    base[7], base[8] = [-0.46, 0.0, 1.42], [0.46, 0.0, 1.42]
    base[9], base[10] = [-0.72, 0.0, 1.42], [0.72, 0.0, 1.42]
    base[11], base[12] = [-0.10, 0.0, 0.92], [0.10, 0.0, 0.92]
    base[13], base[14] = [-0.10, 0.0, 0.48], [0.10, 0.0, 0.48]
    base[15], base[16] = [-0.10, 0.0, 0.06], [0.10, 0.0, 0.06]
    base[17] = [0.0, 0.0, 1.70]
    for t in range(T):
        gt[t] = base + np.array([0.015 * t, 0.0, 0.0])

    obs = {}
    for uid, cam in cams.items():
        arr = np.zeros((T, K, 3), dtype=np.float32)
        for t in range(T):
            uv = cam.project_world_points(gt[t], apply_distortion=True)
            arr[t, :, :2] = uv
            arr[t, :, 2] = 0.85
        obs[uid] = arr

    # Corrupt initial 3D with a massive 0.5m teleport on right wrist at frame 15
    init_3d = gt.astype(np.float32).copy()
    init_3d[15, 10] += np.array([0.0, 0.50, 0.0], dtype=np.float32)
    init_conf = np.full((T, K), 0.80, dtype=np.float32)

    opt_3d, opt_conf, reproj = optimize_multiview_3d_global(cams, obs, init_3d, init_conf, fps=30.0, iterations=8)
    # Teleport must be completely pulled back to ground truth!
    assert float(np.linalg.norm(opt_3d[15, 10] - gt[15, 10])) < 0.02
    # Shoulder width stays rock-solid rigid
    sh_w = np.linalg.norm(opt_3d[:, 6] - opt_3d[:, 5], axis=-1)
    assert float(np.std(sh_w)) < 0.005


def test_flicker_free_hand_smoother_kills_170_deg_spikes() -> None:
    T = 40
    q_id = np.tile(quat_identity().astype(np.float32), (16, 1))
    frames = []
    for t in range(T):
        q = q_id.copy()
        # Smooth curl on index_02 (joint 5)
        q[5] = Rotation.from_euler("x", -40.0 * np.sin(np.pi * t / T), degrees=True).as_quat().astype(np.float32)
        if t == 20:
            # Inject massive 165 deg palm flip + 90 deg finger glitch!
            q[0] = Rotation.from_euler("z", 165.0, degrees=True).as_quat().astype(np.float32)
            q[5] = Rotation.from_euler("x", -140.0, degrees=True).as_quat().astype(np.float32)
        frames.append({"left": HandObservation("left", np.zeros((21, 3), dtype=np.float32), 0.85, q)})
    track = HandCaptureEngine().build_track(frames, fps=30.0, smooth=True)
    hq = track.left_local_rotations_xyzw
    ang = np.degrees(2.0 * np.arccos(np.clip(np.abs(np.sum(hq[1:] * hq[:-1], axis=-1)), 0.0, 1.0)))
    assert float(np.max(ang)) < 18.0
