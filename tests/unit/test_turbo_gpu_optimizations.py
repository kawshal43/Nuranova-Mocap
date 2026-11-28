"""Regression tests for GPU acceleration, vectorized triangulation, fast quaternion math, and overlap pre-sync."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from engine.hands.hand_tracker import HandObservation
from engine.math_core.camera_model import (
    CalibratedCamera,
    CameraExtrinsics,
    CameraIntrinsics,
)
from engine.math_core.transforms import matrix_to_quat, quat_identity, quat_to_matrix
from engine.skeleton.rig_definition import (
    ANATOMICAL_LIMITS_DEG,
    BONE_NAMES,
    ActorProfile,
    clamp_bone_rotation,
    compute_forward_kinematics_frame,
    compute_forward_kinematics_sequence,
)
from engine.triangulation.triangulator import (
    Observation2D,
    triangulate_point_robust,
    triangulate_sequence,
)
from multi_video_studio.pyav_sync import CameraSyncEstimate, map_camera_pts_to_global_s
from multi_video_studio.wholebody_cuda import _AsyncHandFacePool, _bbox_from_keypoints


def _make_cam(uid: str, pos: np.ndarray, target: np.ndarray, w: int = 1920, h: int = 1080) -> CalibratedCamera:
    fwd = target - pos
    fwd /= np.linalg.norm(fwd)
    up_w = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(fwd, up_w)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R_cw = np.vstack([right, down, fwd])
    t_cw = -R_cw @ pos
    K = np.array([[1600.0, 0.0, w * 0.5], [0.0, 1600.0, h * 0.5], [0.0, 0.0, 1.0]], dtype=np.float64)
    return CalibratedCamera(
        intrinsics=CameraIntrinsics(camera_uid=uid, image_width=w, image_height=h, camera_matrix=K, dist_coeffs=np.zeros(5)),
        extrinsics=CameraExtrinsics(camera_uid=uid, R_cw=R_cw, t_cw=t_cw),
    )


def test_fast_quat_matrix_roundtrip_matches_scipy() -> None:
    rng = np.random.default_rng(123)
    rots = Rotation.random(250, random_state=rng)
    qs_ref = rots.as_quat()
    Ms_ref = rots.as_matrix()

    for i in range(len(qs_ref)):
        M = quat_to_matrix(qs_ref[i])
        assert np.max(np.abs(M - Ms_ref[i])) < 1e-12
        q = matrix_to_quat(Ms_ref[i])
        assert abs(float(np.dot(q, qs_ref[i]))) > 1.0 - 1e-12


def test_vectorized_fk_and_analytic_clamp_match_reference() -> None:
    rng = np.random.default_rng(42)
    actor = ActorProfile(actor_id="test", name="Test")
    T, B = 15, len(BONE_NAMES)
    root_t = rng.normal(scale=0.5, size=(T, 3))
    local_q = np.zeros((T, B, 4), dtype=np.float64)
    for t in range(T):
        local_q[t] = Rotation.random(B, random_state=rng).as_quat()

    pos_vec, mats_vec = compute_forward_kinematics_sequence(root_t, local_q, actor.rest_offsets_m)
    for t in range(T):
        p_ref, m_ref = compute_forward_kinematics_frame(root_t[t], local_q[t], actor.rest_offsets_m)
        assert np.max(np.abs(pos_vec[t] - p_ref)) < 1e-6
        assert np.max(np.abs(mats_vec[t] - m_ref)) < 1e-6

    for bone_name, lims in ANATOMICAL_LIMITS_DEG.items():
        q_in = Rotation.random(random_state=rng).as_quat()
        q_clamped = clamp_bone_rotation(bone_name, q_in, strength=1.0)
        eul = Rotation.from_quat(q_clamped).as_euler("xyz", degrees=True)
        for ax in range(3):
            lo, hi = lims[ax]
            assert lo - 1e-3 <= eul[ax] <= hi + 1e-3


def test_vectorized_triangulate_sequence_matches_scalar_3cam_and_2cam() -> None:
    rng = np.random.default_rng(99)
    target = np.array([0.0, 0.0, 1.0])
    cams = {
        "c1": _make_cam("c1", np.array([3.0, 0.0, 1.5]), target),
        "c2": _make_cam("c2", np.array([0.0, 3.0, 1.6]), target),
        "c3": _make_cam("c3", np.array([-2.5, 2.0, 1.2]), target),
    }
    T, K = 12, 8
    pts_gt = rng.uniform([-0.8, -0.8, 0.2], [0.8, 0.8, 1.9], size=(T, K, 3))
    obs_by_cam = {}
    for uid, cam in cams.items():
        P = cam.projection_matrix
        arr = np.zeros((T, K, 3), dtype=np.float32)
        for t in range(T):
            hom = np.hstack([pts_gt[t], np.ones((K, 1))])
            proj = hom @ P.T
            uv = proj[:, :2] / proj[:, 2:3]
            arr[t, :, :2] = uv + rng.normal(scale=1.5, size=(K, 2))
            arr[t, :, 2] = rng.uniform(0.6, 0.99, size=K)
        obs_by_cam[uid] = arr
    # Inject a strong 1-camera outlier and a 1-camera dropout
    obs_by_cam["c3"][2, 3, :2] += 180.0
    obs_by_cam["c2"][5, 4, 2] = 0.05

    p_vec, c_vec, e_vec, _ = triangulate_sequence(
        cams, obs_by_cam, min_confidence=0.25, max_reproj_error_px=24.0, refine_nonlinear=False
    )
    for t in range(T):
        for k in range(K):
            o_list = [
                Observation2D(u, float(obs_by_cam[u][t, k, 0]), float(obs_by_cam[u][t, k, 1]), float(obs_by_cam[u][t, k, 2]))
                for u in cams
                if obs_by_cam[u][t, k, 2] >= 0.25
            ]
            res = triangulate_point_robust(
                cams, o_list, min_confidence=0.25, max_reproj_error_px=24.0, refine_nonlinear=False
            )
            assert np.all(np.isfinite(p_vec[t, k])) == res.valid
            if res.valid:
                assert float(np.linalg.norm(p_vec[t, k] - res.point_3d)) < 1e-4
                assert abs(float(e_vec[t, k]) - float(res.reprojection_error_px)) < 1e-3


def test_smart_bbox_and_overlap_pts_preservation() -> None:
    kpts = np.zeros((133, 2), dtype=np.float32)
    scores = np.zeros(133, dtype=np.float32)
    # Standing actor (narrow width 200px, tall height 1000px) with toes at 20..22
    kpts[0] = [960, 120]
    kpts[5], kpts[6] = [880, 300], [1040, 300]
    kpts[11], kpts[12] = [900, 620], [1020, 620]
    kpts[15], kpts[16] = [910, 1020], [1010, 1020]
    kpts[20], kpts[21] = [860, 1120], [1060, 1120]
    scores[[0, 5, 6, 11, 12, 15, 16, 20, 21]] = 0.9
    bb = _bbox_from_keypoints(kpts, scores, 1920, 1200)
    assert bb is not None
    # Height-scaled horizontal padding gives ample arm-swing room (>= 180px each side) and includes toes
    assert bb[0] <= 860 - 180
    assert bb[2] >= 1060 + 180
    assert bb[3] >= 1120

    # Overlap-trimmed PTS starting at 12.5s must preserve original 0.0s origin via base_pts_s
    est = CameraSyncEstimate(
        camera_uid="cam_03",
        offset_s=-7.2,
        method="audio_clap",
        confidence=200.0,
        summary="Audio Clap",
        audio_start_s=0.0,
        base_pts_s=0.0,
    )
    g_pts = map_camera_pts_to_global_s([12.5, 12.6, 12.7], est)
    assert np.allclose(g_pts, [5.3, 5.4, 5.5], atol=1e-6)


def test_async_hand_pool_stride_slerp_fill() -> None:
    pool = _AsyncHandFacePool(enable_hands=True, enable_face=False, num_workers=1, crop_stride=3)
    q0 = np.tile(quat_identity().astype(np.float32), (16, 1))
    q1 = q0.copy()
    q1[4] = Rotation.from_euler("x", -60.0, degrees=True).as_quat().astype(np.float32)
    lm0 = np.zeros((21, 3), dtype=np.float32)
    lm1 = np.ones((21, 3), dtype=np.float32)
    pool._hands_out[0] = {"left": HandObservation("left", lm0, 0.9, q0)}
    pool._hands_out[3] = {"left": HandObservation("left", lm1, 0.9, q1)}
    hands, _ = pool.finish(n_frames=5)
    assert len(hands) == 5
    # Frames 1 & 2 interpolated via SLERP between 0 and 3 (~-20 deg and ~-40 deg)
    ang1 = abs(Rotation.from_quat(hands[1]["left"].local_rotations_xyzw[4]).as_euler("xyz", degrees=True)[0])
    ang2 = abs(Rotation.from_quat(hands[2]["left"].local_rotations_xyzw[4]).as_euler("xyz", degrees=True)[0])
    assert 18.0 <= ang1 <= 22.0
    assert 38.0 <= ang2 <= 42.0
    assert hands[1]["left"].source == "interpolated"
    assert "left" not in hands[4]  # No new measurement after the last sampled frame.
    from engine.hands.hand_tracker import HandCaptureEngine
    track = HandCaptureEngine().build_track(hands, fps=30)
    assert not np.any(track.left_confidence[1:3])
