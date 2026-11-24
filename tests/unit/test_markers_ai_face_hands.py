"""Unit & integration tests for Physical Marker Tracking, AI Tracking, Face, and Hand engines."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from engine.ai_tracking.pose_tracker import AIPoseTracker2D
from engine.ai_tracking.runtime import detect_hardware
from engine.face.face_tracker import MEDIAPIPE_52_BLENDSHAPES, FacialCaptureEngine
from engine.hands.hand_tracker import solve_hand_rotations_from_landmarks21
from engine.marker_tracking.detector import (
    FiducialMarkerDetector,
    TemporalBlobTracker2D,
    VisibleBlobMarkerDetector,
    match_and_triangulate_unlabeled_blobs,
    solve_rigid_cluster_pose,
)
from engine.marker_tracking.marker_kit import (
    create_default_cinema_marker_kit,
    generate_printable_marker_sheet_pdf,
    generate_valid_marker_image,
)
from engine.math_core.transforms import matrix_to_quat, quat_angle_between
from engine.video.video_io import probe_video_metadata
from tests.unit.test_math_calibration_triangulation import make_synthetic_ring_cameras


def test_valid_apriltag_generation_pdf_and_detection(tmp_path: Path):
    kit = create_default_cinema_marker_kit()
    assert len(kit.markers) >= 35
    assert len(kit.clusters) == 7

    # Save & reload marker kit
    saved = kit.save(tmp_path / "kit.json")
    assert saved.exists()

    # Generate printable PDF of actual AprilTag 36h11 markers
    pdf = generate_printable_marker_sheet_pdf([0, 4, 8, 12], tmp_path / "tags.pdf", dictionary_name="DICT_APRILTAG_36h11")
    assert pdf.exists() and pdf.stat().st_size > 2000

    # Compose a synthetic matte-dark scene containing 2 valid AprilTags and detect them
    canvas = np.full((600, 800, 3), 35, dtype=np.uint8)
    for tag_id, (ox, oy) in [(0, (120, 150)), (12, (460, 220))]:
        raw_tag = generate_valid_marker_image(tag_id, "DICT_APRILTAG_36h11", size_px=140, border_bits=1)
        padded = cv2.copyMakeBorder(raw_tag, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        canvas[oy : oy + 180, ox : ox + 180] = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)

    det = FiducialMarkerDetector("DICT_APRILTAG_36h11")
    found = det.detect(canvas)
    found_ids = {t.tag_id for t in found}
    assert {0, 12}.issubset(found_ids)


def test_visible_blob_subpixel_tracking_and_rigid_cluster_6dof():
    """SYNTHETIC_GROUND_TRUTH: Render high-contrast white spheres into 4 cameras, detect, triangulate, and solve 6-DoF rigid cluster."""
    cams = make_synthetic_ring_cameras(num_cams=4)
    kit = create_default_cinema_marker_kit()
    chest_cluster = next(c for c in kit.clusters if c.cluster_id == "cl_chest")

    R_gt = Rotation.from_euler("xyz", [18.0, -24.0, 42.0], degrees=True).as_matrix()
    t_gt = np.array([0.12, -0.05, 1.30], dtype=np.float64)
    local_pts = np.asarray(chest_cluster.local_points_m, dtype=np.float64)
    world_pts_gt = (local_pts @ R_gt.T) + t_gt

    blob_det = VisibleBlobMarkerDetector(brightness_threshold=190, min_area_px=12.0, min_circularity=0.6)
    blobs_per_cam = {}
    for uid, cam in cams.items():
        # Render matte-black suit background (intensity ~20) with anti-aliased white circular markers
        img = np.full((1080, 1920), 20, dtype=np.uint8)
        uvs = cam.project_world_points(world_pts_gt, apply_distortion=True)
        for uv in uvs:
            cv2.circle(img, (int(round(uv[0])), int(round(uv[1]))), 4, 250, -1, lineType=cv2.LINE_AA)
        detected = blob_det.detect(img)
        assert len(detected) == 4
        blobs_per_cam[uid] = detected

    # Cross-camera epipolar matching & triangulation (unlabeled!)
    tri_pts = match_and_triangulate_unlabeled_blobs(cams, blobs_per_cam, max_epipolar_px=4.0, max_reproj_px=4.0)
    assert len(tri_pts) == 4

    # Match triangulated points to GT markers by nearest neighbor (simulating labeled initialization)
    obs_3d = {}
    for mid, gt_p in zip(chest_cluster.marker_ids, world_pts_gt):
        best = min(tri_pts, key=lambda r: float(np.linalg.norm(r.point_3d - gt_p)))
        obs_3d[mid] = (best.point_3d, best.confidence)

    cluster_pose = solve_rigid_cluster_pose(chest_cluster, obs_3d)
    assert cluster_pose.valid
    assert not cluster_pose.possible_identity_swap
    assert np.linalg.norm(cluster_pose.position_world - t_gt) < 0.005  # < 5 mm
    assert quat_angle_between(cluster_pose.rotation_quat_xyzw, matrix_to_quat(R_gt)) < np.radians(2.0)

    # Verify temporal blob tracker maintains persistent IDs across smooth motion
    tracker = TemporalBlobTracker2D()
    t1, _ = tracker.update(blobs_per_cam["cam_01"])
    ids_1 = [b.track_id for b in t1]
    t2, _ = tracker.update(blobs_per_cam["cam_01"])
    ids_2 = [b.track_id for b in t2]
    assert ids_1 == ids_2


def test_hardware_diagnostics_and_real_video_mediapipe_inference():
    hw = detect_hardware(prefer_gpu=True)
    assert hw.selected_device in ("cuda", "cpu")
    assert "CPUExecutionProvider" in hw.selected_providers

    # Real 4-camera BSD-3-Clause Pose2Sim demo video (cited reference dataset, NOT smartphone validation)
    demo_vid = Path("e:/Backup/KAWSHAL/Nuranova-Mocap/references/pose2sim-main/Pose2Sim/Demo_SinglePerson/videos/cam01.mp4")
    if demo_vid.exists():
        meta = probe_video_metadata(demo_vid, scan_pts=True)
        assert meta.width > 0 and meta.height > 0 and meta.frame_count >= 10

        tracker = AIPoseTracker2D(backend="mediapipe_pose", mode="lightweight", prefer_gpu=False)
        arr, pts, stats = tracker.process_video(demo_vid, start_frame=0, end_frame=10)
        tracker.close()
        assert arr.shape == (10, 26, 3)
        assert stats.frames_processed == 10
        assert stats.detection_rate >= 0.7
        assert stats.measured_fps > 1.0


def test_hand_solver_and_face_calibration():
    # Build synthetic open hand with slightly bent index finger
    lms = np.zeros((21, 3), dtype=np.float64)
    lms[0] = [0.0, 0.0, 0.0]
    lms[5] = [-0.03, 0.08, 0.0]
    lms[9] = [0.0, 0.085, 0.0]
    lms[13] = [0.025, 0.08, 0.0]
    lms[17] = [0.05, 0.072, 0.0]
    # Thumb
    lms[1], lms[2], lms[3], lms[4] = [-0.035, 0.02, 0.0], [-0.055, 0.045, 0.0], [-0.065, 0.07, 0.0], [-0.07, 0.09, 0.0]
    # Index bent forward (+Z)
    lms[6] = [-0.03, 0.115, 0.01]
    lms[7] = [-0.03, 0.135, 0.03]
    lms[8] = [-0.03, 0.145, 0.055]
    # Middle, ring, pinky straight
    for base, x in [(9, 0.0), (13, 0.025), (17, 0.05)]:
        lms[base + 1] = [x, 0.12, 0.0]
        lms[base + 2] = [x, 0.15, 0.0]
        lms[base + 3] = [x, 0.175, 0.0]

    quats = solve_hand_rotations_from_landmarks21(lms, handedness="right")
    assert quats.shape == (16, 4)
    assert np.allclose(np.linalg.norm(quats, axis=-1), 1.0, atol=1e-4)

    # Verify facial calibration baseline subtraction
    face_eng = FacialCaptureEngine()
    assert len(MEDIAPIPE_52_BLENDSHAPES) == 52
    raw = np.full(52, 0.25, dtype=np.float32)
    face_eng.calibration.neutral_baseline = np.full(52, 0.10, dtype=np.float32)
    cal = face_eng.calibration.apply(raw)
    assert np.allclose(cal, 0.15, atol=1e-5)
