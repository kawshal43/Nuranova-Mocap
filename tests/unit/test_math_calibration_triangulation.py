"""Unit and synthetic ground-truth validation tests for Phase 1: Math, Calibration, Bundle Adjustment, and Triangulation.

NOTE: All synthetic tests are explicitly labeled SYNTHETIC_GROUND_TRUTH.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from desktop.calibration.calibrator import (
    align_calibration_to_world_z_up,
    calibrate_camera_intrinsics,
    initialize_extrinsics_from_board_frames,
    run_board_bundle_adjustment,
    validate_known_distance,
)
from desktop.calibration.charuco_board import (
    ChArUcoBoardConfig,
    ChArUcoDetection,
    detect_charuco_corners,
    generate_printable_charuco_pdf,
    render_charuco_png,
)
from engine.math_core.camera_model import (
    CalibratedCamera,
    CameraExtrinsics,
    CameraIntrinsics,
    MultiCameraCalibration,
    epipolar_line_distance_px,
)
from engine.math_core.transforms import (
    enforce_quaternion_continuity,
    matrix_to_quat,
    quat_angle_between,
    quat_from_two_vectors,
    quat_multiply,
    quat_rotate_vector,
    quat_slerp,
    quat_to_matrix,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
    solve_rigid_transform_kabsch,
    swing_twist_decomposition,
)
from engine.triangulation.triangulator import (
    Observation2D,
    triangulate_point_robust,
    triangulate_sequence,
)


def make_synthetic_ring_cameras(num_cams: int = 4, radius_m: float = 3.2, height_m: float = 1.8) -> dict[str, CalibratedCamera]:
    """SYNTHETIC_GROUND_TRUTH: Create N calibrated cameras arranged in a ring looking at (0, 0, 1.0)."""
    target = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    up_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    K = np.array([[1400.0, 0.0, 960.0], [0.0, 1400.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array([-0.12, 0.04, 0.0005, -0.0003, -0.01], dtype=np.float64)

    cams: dict[str, CalibratedCamera] = {}
    for i in range(num_cams):
        angle = 2.0 * np.pi * i / num_cams + 0.25
        pos = np.array([radius_m * np.cos(angle), radius_m * np.sin(angle), height_m], dtype=np.float64)
        z_cam = target - pos
        z_cam /= np.linalg.norm(z_cam)
        x_cam = np.cross(z_cam, up_world)
        x_cam /= np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)  # OpenCV +Y points down
        R_cw = np.vstack([x_cam, y_cam, z_cam])
        t_cw = -R_cw @ pos
        uid = f"cam_{i+1:02d}"
        cams[uid] = CalibratedCamera(
            intrinsics=CameraIntrinsics(
                camera_uid=uid,
                image_width=1920,
                image_height=1080,
                camera_matrix=K.copy(),
                dist_coeffs=dist.copy(),
            ),
            extrinsics=CameraExtrinsics(camera_uid=uid, R_cw=R_cw, t_cw=t_cw),
        )
    return cams


def test_quaternion_and_kabsch_math():
    q1 = Rotation.from_euler("xyz", [25, -40, 70], degrees=True).as_quat()
    q2 = Rotation.from_euler("xyz", [-15, 30, -20], degrees=True).as_quat()
    q_prod = quat_multiply(q1, q2)
    R_prod = quat_to_matrix(q1) @ quat_to_matrix(q2)
    assert np.allclose(quat_to_matrix(q_prod), R_prod, atol=1e-9)
    assert quat_angle_between(matrix_to_quat(R_prod), q_prod) < 1e-9

    # Shortest arc
    v1 = np.array([1.0, 0.2, -0.3])
    v2 = np.array([-0.4, 0.9, 0.5])
    q_arc = quat_from_two_vectors(v1, v2)
    v1_rot = quat_rotate_vector(q_arc, v1 / np.linalg.norm(v1))
    assert np.allclose(v1_rot, v2 / np.linalg.norm(v2), atol=1e-9)

    # Swing-twist decomposition reconstructs original rotation
    q_swing, q_twist = swing_twist_decomposition(q1, np.array([0.0, 0.0, 1.0]))
    assert quat_angle_between(quat_multiply(q_swing, q_twist), q1) < 1e-9

    # Blender WXYZ conversion round-trip
    assert np.allclose(quat_wxyz_to_xyzw(quat_xyzw_to_wxyz(q1)), q1)

    # Continuity removes sign flip
    seq = np.stack([q1, -q1, quat_slerp(q1, q2, 0.5), -q2], axis=0)
    cont = enforce_quaternion_continuity(seq)
    for t in range(1, len(cont)):
        assert np.dot(cont[t], cont[t - 1]) >= 0.0

    # Kabsch rigid body 6-DoF recovery
    rng = np.random.default_rng(42)
    local_pts = np.array([[0.0, 0.0, 0.0], [0.08, 0.0, 0.0], [0.02, 0.07, 0.01], [-0.03, 0.02, 0.06]])
    R_gt = Rotation.from_euler("zyx", [55, -22, 118], degrees=True).as_matrix()
    t_gt = np.array([0.45, -0.32, 1.12])
    world_pts = (local_pts @ R_gt.T) + t_gt + rng.normal(0.0, 0.0005, size=local_pts.shape)
    R_est, t_est, rms = solve_rigid_transform_kabsch(local_pts, world_pts)
    assert rms < 0.002
    assert np.linalg.norm(t_est - t_gt) < 0.002
    assert quat_angle_between(matrix_to_quat(R_est), matrix_to_quat(R_gt)) < np.radians(1.0)


def test_charuco_generation_pdf_and_detection(tmp_path: Path):
    cfg = ChArUcoBoardConfig(squares_x=7, squares_y=5, square_length_mm=35.0, marker_length_mm=26.0)
    png_path = render_charuco_png(cfg, tmp_path / "charuco.png", dpi=150)
    pdf_path = generate_printable_charuco_pdf(cfg, tmp_path / "charuco_a4.pdf", paper_size="A4", dpi=150)
    assert png_path.exists() and png_path.stat().st_size > 1000
    assert pdf_path.exists() and pdf_path.stat().st_size > 2000

    img = cv2.imread(str(png_path))
    det = detect_charuco_corners(img, cfg)
    expected_internal = (cfg.squares_x - 1) * (cfg.squares_y - 1)
    assert det.num_corners == expected_internal


def test_synthetic_triangulation_and_outlier_rejection():
    """SYNTHETIC_GROUND_TRUTH: 4 cameras + 1 corrupted outlier view."""
    cams = make_synthetic_ring_cameras(num_cams=4)
    gt_pt = np.array([0.23, -0.18, 1.15], dtype=np.float64)

    obs_clean = []
    for uid, cam in cams.items():
        uv = cam.project_world_points(gt_pt.reshape(1, 3), apply_distortion=True)[0]
        obs_clean.append(Observation2D(camera_uid=uid, u=float(uv[0]), v=float(uv[1]), confidence=0.95))

    res_clean = triangulate_point_robust(cams, obs_clean, refine_nonlinear=True)
    assert res_clean.valid
    assert np.linalg.norm(res_clean.point_3d - gt_pt) < 1e-4  # < 0.1 mm on exact observations
    assert res_clean.reprojection_error_px < 0.05
    assert len(res_clean.inlier_camera_uids) == 4

    # Corrupt cam_04 by +85 pixels (severe outlier)
    obs_corrupt = list(obs_clean[:3]) + [
        Observation2D(camera_uid="cam_04", u=obs_clean[3].u + 85.0, v=obs_clean[3].v - 60.0, confidence=0.9)
    ]
    res_robust = triangulate_point_robust(cams, obs_corrupt, max_reproj_error_px=8.0, refine_nonlinear=True)
    assert res_robust.valid
    assert "cam_04" in res_robust.outlier_camera_uids
    assert len(res_robust.inlier_camera_uids) == 3
    assert np.linalg.norm(res_robust.point_3d - gt_pt) < 1e-3  # < 1 mm after rejecting outlier

    # Check epipolar distance for consistent vs inconsistent pair
    d_ok = epipolar_line_distance_px(cams["cam_01"], [obs_clean[0].u, obs_clean[0].v], cams["cam_02"], [obs_clean[1].u, obs_clean[1].v])
    d_bad = epipolar_line_distance_px(cams["cam_01"], [obs_clean[0].u, obs_clean[0].v], cams["cam_04"], [obs_corrupt[3].u, obs_corrupt[3].v])
    assert d_ok < 0.2
    assert d_bad > 20.0

    # Batch sequence triangulation with mixed visibility (some occluded cameras)
    T, K = 15, 5
    rng = np.random.default_rng(7)
    gt_seq = rng.uniform([-0.8, -0.8, 0.2], [0.8, 0.8, 1.8], size=(T, K, 3))
    obs_seq = {}
    for uid, cam in cams.items():
        arr = np.zeros((T, K, 3), dtype=np.float32)
        for t in range(T):
            uvs = cam.project_world_points(gt_seq[t], apply_distortion=True)
            uvs += rng.normal(0.0, 0.35, size=uvs.shape)
            arr[t, :, :2] = uvs
            arr[t, :, 2] = 0.9
        obs_seq[uid] = arr
    # Occlude cam_01 on first 5 frames for joint 0
    obs_seq["cam_01"][:5, 0, 2] = 0.0
    pts3d, confs, reproj, prov = triangulate_sequence(cams, obs_seq)
    err_mm = np.linalg.norm(pts3d - gt_seq, axis=-1) * 1000.0
    assert np.nanmean(err_mm) < 2.0
    assert np.all(confs > 0.5)


def test_synthetic_extrinsics_bundle_adjustment_and_world_alignment(tmp_path: Path):
    """SYNTHETIC_GROUND_TRUTH: End-to-end 4-camera extrinsic init + bundle adjustment + Z-up alignment."""
    gt_cams = make_synthetic_ring_cameras(num_cams=4)
    board_cfg = ChArUcoBoardConfig(squares_x=6, squares_y=4, square_length_mm=80.0, marker_length_mm=60.0)
    board_corners_local = board_cfg.get_chessboard_corners_3d()
    n_corners = board_corners_local.shape[0]
    rng = np.random.default_rng(123)

    # Generate 12 synthetic board poses inside capture volume
    synced_dets: list[dict[str, ChArUcoDetection]] = []
    gt_board_corners_world = []
    for f in range(12):
        R_wb = Rotation.from_euler(
            "xyz",
            [rng.uniform(-35, 35), rng.uniform(-35, 35), rng.uniform(-180, 180)],
            degrees=True,
        ).as_matrix()
        t_wb = np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(0.6, 1.5)])
        pts_w = (board_corners_local @ R_wb.T) + t_wb
        gt_board_corners_world.append(pts_w)

        frame_det: dict[str, ChArUcoDetection] = {}
        for uid, cam in gt_cams.items():
            uvs = cam.project_world_points(pts_w, apply_distortion=True)
            uvs += rng.normal(0.0, 0.25, size=uvs.shape)
            frame_det[uid] = ChArUcoDetection(
                corner_ids=np.arange(n_corners, dtype=np.int32),
                corners_2d=uvs,
                marker_ids=np.zeros((0,), dtype=np.int32),
                marker_corners=[],
            )
        synced_dets.append(frame_det)

    intr_map = {uid: cam.intrinsics for uid, cam in gt_cams.items()}
    # Verify single-camera intrinsic calibration also succeeds on synthetic board detections
    intr_est = calibrate_camera_intrinsics("cam_01", [fd["cam_01"] for fd in synced_dets], board_cfg, (1920, 1080))
    assert intr_est.reprojection_error_px < 1.0

    # Extrinsic pose graph init
    extr_init, board_poses = initialize_extrinsics_from_board_frames(intr_map, synced_dets, board_cfg, reference_camera_uid="cam_01")

    # Run metric-scale-preserving board bundle adjustment
    extr_opt, opt_boards, report = run_board_bundle_adjustment(
        intr_map, extr_init, synced_dets, board_poses, board_cfg, anchor_camera_uid="cam_01"
    )
    assert report.success
    assert report.global_reprojection_rmse_px < 0.8

    # Align to Z-up floor using 4 known floor points projected from ground truth then triangulated
    calib = MultiCameraCalibration(
        profile_name="synthetic_4cam",
        cameras={uid: CalibratedCamera(intr_map[uid], extr_opt[uid]) for uid in gt_cams},
        global_reprojection_rmse_px=report.global_reprojection_rmse_px,
    )
    floor_gt = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    floor_est = []
    for p in floor_gt:
        obs = [
            Observation2D(u, *gt_cams[u].project_world_points(p.reshape(1, 3))[0], 1.0)
            for u in gt_cams
        ]
        floor_est.append(triangulate_point_robust(calib.cameras, obs).point_3d)
    floor_est = np.vstack(floor_est)

    aligned = align_calibration_to_world_z_up(
        calib,
        floor_points_world=floor_est,
        origin_world=floor_est[0],
        forward_hint_world=floor_est[2] - floor_est[0],
    )
    # Re-triangulate 1.0m calibration wand in aligned Z-up coordinates
    wand_gt_a = np.array([-0.5, 0.0, 1.0])
    wand_gt_b = np.array([0.5, 0.0, 1.0])
    obs_a = [Observation2D(u, *gt_cams[u].project_world_points(wand_gt_a.reshape(1, 3))[0], 1.0) for u in gt_cams]
    obs_b = [Observation2D(u, *gt_cams[u].project_world_points(wand_gt_b.reshape(1, 3))[0], 1.0) for u in gt_cams]
    pa = triangulate_point_robust(aligned.cameras, obs_a).point_3d
    pb = triangulate_point_robust(aligned.cameras, obs_b).point_3d
    measured_m, err_mm = validate_known_distance(pa, pb, expected_distance_m=1.0)
    assert err_mm < 2.0  # Sub-2mm metric scale accuracy with board-constrained BA

    # Save / reload round-trip + invalidation warning check
    saved = aligned.save(tmp_path / "calib.json")
    reloaded = MultiCameraCalibration.load(saved)
    assert len(reloaded.cameras) == 4
    valid, reason = reloaded.cameras["cam_01"].check_config_validity(1280, 720)
    assert not valid and "Resolution mismatch" in str(reason)
