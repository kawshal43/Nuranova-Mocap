"""Unit tests for real-time live calibration coaching, directional feedback, board mismatch detection, and HUD drawing."""

from __future__ import annotations

import cv2
import numpy as np

from desktop.calibration.charuco_board import ChArUcoBoardConfig, ChArUcoDetection
from desktop.calibration.live_coach import (
    LiveVoiceCoach,
    analyze_camera_detection,
    detect_wrong_board_preset,
    draw_live_calibration_hud,
    evaluate_multi_cam_coaching,
    pose_spoken_instruction,
    render_board_thumbnail_bgr,
)


def _fake_det(x1: float, y1: float, x2: float, y2: float, n: int = 12) -> ChArUcoDetection:
    xs = np.linspace(x1, x2, 4)
    ys = np.linspace(y1, y2, 3)
    pts = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)[:n]
    return ChArUcoDetection(
        corner_ids=np.arange(len(pts), dtype=np.int32),
        corners_2d=pts,
        marker_ids=np.zeros((0,), dtype=np.int32),
        marker_corners=[],
    )


def test_camera_coaching_distance_and_direction_feedback():
    w, h = 640, 360

    # Too far / small
    fb_far = analyze_camera_detection("cam_01", _fake_det(300, 170, 345, 200), w, h, pose_index=1)
    assert fb_far.status == "too_far"

    # Clipped against left/top edge
    fb_clip = analyze_camera_detection("cam_01", _fake_det(3, 50, 260, 220), w, h, pose_index=1)
    assert fb_clip.status == "clipped"

    # Too close (fills > 78% of width)
    fb_close = analyze_camera_detection("cam_01", _fake_det(40, 25, 600, 330), w, h, pose_index=1)
    assert fb_close.status == "too_close"

    # Pose 5 (Low Angle) held too high in frame -> move_down
    fb_high_low = analyze_camera_detection("cam_01", _fake_det(240, 40, 400, 120), w, h, pose_index=4)
    assert fb_high_low.status == "move_down"

    # Pose 2 (Center) held too far left -> move_right
    fb_left = analyze_camera_detection("cam_01", _fake_det(20, 120, 130, 220), w, h, pose_index=1)
    assert fb_left.status == "move_right"

    # Pose 2 (Center) centered & steady -> ready
    det_center = _fake_det(210, 110, 430, 250)
    fb_moving = analyze_camera_detection("cam_01", det_center, w, h, pose_index=1, prev_center_norm=(0.20, 0.50))
    assert fb_moving.status == "moving"
    fb_ready = analyze_camera_detection("cam_01", det_center, w, h, pose_index=1, prev_center_norm=(0.50, 0.50))
    assert fb_ready.status == "ready"


def test_multi_cam_coaching_aggregation_and_hud_rendering():
    w, h = 640, 360
    det_ok = _fake_det(210, 110, 430, 250)
    det_none = ChArUcoDetection(
        np.zeros((0,), dtype=np.int32),
        np.zeros((0, 2), dtype=np.float64),
        np.zeros((0,), dtype=np.int32),
        [],
    )

    fb1 = analyze_camera_detection("cam_01", det_ok, w, h, pose_index=1)
    fb2_miss = analyze_camera_detection("cam_02", det_none, w, h, pose_index=1)
    dec_miss = evaluate_multi_cam_coaching([fb1, fb2_miss], pose_index=1)
    assert dec_miss.all_ready_to_capture is False
    assert "cam_02" in dec_miss.headline

    fb2_ok = analyze_camera_detection("cam_02", det_ok, w, h, pose_index=1)
    dec_ok = evaluate_multi_cam_coaching([fb1, fb2_ok], pose_index=1)
    assert dec_ok.all_ready_to_capture is True
    assert dec_ok.severity == "ok"

    # Verify HUD rendering & board thumbnail work cleanly
    frame = np.full((h, w, 3), 24, dtype=np.uint8)
    vis = draw_live_calibration_hud(frame, det_ok, fb1, pose_index=1, hold_progress=0.6, flash_text="CAPTURED!")
    assert vis.shape == (h, w, 3)

    thumb = render_board_thumbnail_bgr(ChArUcoBoardConfig(9, 6, 40.0, 30.0, "DICT_5X5_250"), 120, 84)
    assert thumb.shape == (84, 120, 3)


def test_wrong_board_preset_detection_a3_shown_while_a4_selected():
    cfg_a4 = ChArUcoBoardConfig(7, 5, 35.0, 26.0, "DICT_5X5_250")
    cfg_a3 = ChArUcoBoardConfig(9, 6, 40.0, 30.0, "DICT_5X5_250")

    # Render an A3 (9x6) board onto a clean camera frame
    board_a3, _ = cfg_a3.create_opencv_board()
    img_gray = board_a3.generateImage((540, 360), marginSize=40, borderBits=1)
    frame_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    alt = detect_wrong_board_preset(frame_bgr, active_cfg=cfg_a4, current_corners=0)
    assert alt == "A3"


def test_enhanced_charuco_detect_recovers_small_dim_preview_board():
    from desktop.calibration.charuco_board import detect_charuco_corners

    cfg_a4 = ChArUcoBoardConfig(7, 5, 35.0, 26.0, "DICT_5X5_250")
    board, _ = cfg_a4.create_opencv_board()
    raw = board.generateImage((700, 500), marginSize=20, borderBits=1)
    small = cv2.resize(raw, (96, 68), interpolation=cv2.INTER_AREA)
    canvas = np.full((360, 640), 115, dtype=np.uint8)
    canvas[146:214, 272:368] = small
    dim = np.clip(canvas.astype(np.float32) * 0.55 + 10, 0, 255).astype(np.uint8)
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(dim, cv2.COLOR_GRAY2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 65])
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    det = detect_charuco_corners(dec, cfg_a4)
    assert det.num_corners >= 12


def test_live_voice_coach_continuous_pacing_and_natural_prompts():
    w, h = 640, 360
    det_ok = _fake_det(210, 110, 430, 250)
    det_none = ChArUcoDetection(
        np.zeros((0,), dtype=np.int32),
        np.zeros((0, 2), dtype=np.float64),
        np.zeros((0,), dtype=np.int32),
        [],
    )
    fb1 = analyze_camera_detection("cam_01", det_ok, w, h, pose_index=1)
    fb2 = analyze_camera_detection("cam_02", det_none, w, h, pose_index=1)
    dec = evaluate_multi_cam_coaching([fb1, fb2], pose_index=1)
    assert "Camera 2" in dec.voice_prompt

    for idx in range(6):
        txt = pose_spoken_instruction(idx, "A3")
        assert f"Pose {idx + 1}" in txt and "A3" in txt

    coach = LiveVoiceCoach()
    # First instruction speaks right away
    assert coach.coach("come_forward", "Come forward.", now=10.0) is True
    # Same instruction suppressed during repeat window
    assert coach.coach("come_forward", "Come forward.", now=11.5) is False
    # Same instruction repeats after repeat_interval_s (2.6s) until user fixes it!
    assert coach.coach("come_forward", "Come forward.", now=12.8) is True
    # Different instruction speaks quickly after change_interval_s (0.95s)
    assert coach.coach("move_left", "Move left.", now=13.2) is False
    assert coach.coach("move_left", "Move left.", now=13.9) is True
    coach.stop()


def test_align_calibration_from_upright_board_produces_z_up_cameras():
    from desktop.calibration.calibrator import align_calibration_from_upright_board
    from engine.math_core.camera_model import (
        CalibratedCamera,
        CameraExtrinsics,
        CameraIntrinsics,
        MultiCameraCalibration,
    )

    cfg = ChArUcoBoardConfig(9, 6, 40.0, 30.0, "DICT_5X5_250")
    K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
    intr = CameraIntrinsics("cam_01", 1920, 1080, K, np.zeros(5))
    # Camera 1 at world origin looking along +Z (OpenCV camera convention: +Y down, +Z forward)
    extr1 = CameraExtrinsics("cam_01", np.eye(3), np.zeros(3))
    extr2 = CameraExtrinsics("cam_02", np.eye(3), np.array([-1.5, 0.0, 0.0]))
    calib = MultiCameraCalibration(
        profile_name="test_upright",
        cameras={"cam_01": CalibratedCamera(intr, extr1), "cam_02": CalibratedCamera(intr, extr2)},
    )
    # Upright board facing camera at z=+2.5m (local +Y down == camera +Y down, so -Rb[:, 1] == camera up!)
    Rb = np.eye(3)
    tb = np.array([-0.18, -0.12, 2.5])
    aligned = align_calibration_from_upright_board(calib, Rb, tb, cfg, board_center_height_m=1.15)
    assert aligned.world_up_axis == "Z"
    # Cameras should sit around Z ~= 1.15 m above the studio floor Z = 0
    for cam in aligned.cameras.values():
        assert 0.8 <= cam.extrinsics.camera_center_world[2] <= 1.6


def test_single_cam_root_tracker_walks_left_right_forward_back_and_jumps():
    from desktop.workers.live_preview import SingleCamRootTracker

    w, h = 640, 360
    # Synthetic standing hip-centered Body26 (Z-up meters: hips z=0.95, shoulders z=1.45, ankles z=0.08)
    base_3d = np.zeros((26, 3), dtype=np.float64)
    base_3d[5] = [0.18, 0.0, 1.45]
    base_3d[6] = [-0.18, 0.0, 1.45]
    base_3d[18] = [0.0, 0.0, 1.45]
    base_3d[11] = [0.10, 0.0, 0.95]
    base_3d[12] = [-0.10, 0.0, 0.95]
    base_3d[19] = [0.0, 0.0, 0.95]
    base_3d[13] = [0.10, 0.0, 0.50]
    base_3d[14] = [-0.10, 0.0, 0.50]
    base_3d[15] = [0.10, 0.0, 0.08]
    base_3d[16] = [-0.10, 0.0, 0.08]
    conf = np.full(26, 0.95, dtype=np.float64)

    def make_2d(cx_px: float, torso_px: float, foot_y_px: float) -> np.ndarray:
        uvc = np.zeros((26, 3), dtype=np.float64)
        hip_y = foot_y_px - 1.75 * torso_px
        sh_y = hip_y - torso_px
        knee_y = hip_y + 0.9 * torso_px
        uvc[5] = [cx_px - 22, sh_y, 0.95]
        uvc[6] = [cx_px + 22, sh_y, 0.95]
        uvc[18] = [cx_px, sh_y, 0.95]
        uvc[11] = [cx_px - 14, hip_y, 0.95]
        uvc[12] = [cx_px + 14, hip_y, 0.95]
        uvc[19] = [cx_px, hip_y, 0.95]
        uvc[13] = [cx_px - 14, knee_y, 0.95]
        uvc[14] = [cx_px + 14, knee_y, 0.95]
        uvc[15] = [cx_px - 14, foot_y_px, 0.95]
        uvc[16] = [cx_px + 14, foot_y_px, 0.95]
        return uvc

    tr = SingleCamRootTracker(enabled=True)
    # 1. Centered baseline frame
    p0 = tr.apply(base_3d, conf, make_2d(320.0, 90.0, 320.0), w, h)
    assert abs(p0[19, 0]) < 0.05 and abs(p0[19, 1]) < 0.05

    # 2. Walk right in image (cx_px increases) & step closer (torso_px larger -> smaller depth -> negative Y)
    p1 = base_3d
    for _ in range(8):
        p1 = tr.apply(base_3d, conf, make_2d(440.0, 125.0, 320.0), w, h)
    assert p1[19, 0] < -0.25  # moved laterally
    assert p1[19, 1] < -0.20  # stepped closer to camera

    # 3. Jump upward (both feet lift well above baseline 320px -> 250px)
    p2 = base_3d
    for _ in range(5):
        p2 = tr.apply(base_3d, conf, make_2d(440.0, 125.0, 250.0), w, h)
    assert p2[19, 2] > base_3d[19, 2] + 0.10

    # 4. In-place mode (disabled) keeps original coordinates unchanged
    tr.enabled = False
    p_pin = tr.apply(base_3d, conf, make_2d(440.0, 125.0, 250.0), w, h)
    assert np.allclose(p_pin, base_3d)


def test_calibration_wizard_auto_captures_and_saves_when_both_cameras_lock(tmp_path):
    """When both cameras lock onto the ChArUco board, CalibrationWizardPage must auto-capture, save JPEGs + manifest, and advance."""
    import os
    from PySide6.QtWidgets import QApplication
    from desktop.ui.app_controller import StudioController
    from desktop.ui.pages.pages_cameras import CalibrationWizardPage

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    assert app is not None

    ctrl = StudioController(settings_path=tmp_path / "settings.json")
    page = CalibrationWizardPage(ctrl)
    page.chk_voice_coach.setChecked(False)
    try:
        # Register 2 phones seeing the A3 board + 1 idle USB desk webcam seeing nothing
        c1 = ctrl.camera_manager.register_imported_or_local_camera("p1", "Phone 1", 960, 540, 30.0, adapter_type="android_wifi")
        c2 = ctrl.camera_manager.register_imported_or_local_camera("p2", "Phone 2", 960, 540, 30.0, adapter_type="android_wifi")
        c3 = ctrl.camera_manager.register_usb_camera("usb_0", "Idle Desk Cam", 0, 640, 480, 30.0)

        cfg = page._board_cfg()  # Default A3 9x6
        board, _ = cfg.create_opencv_board()
        board_img = board.generateImage((420, 280), marginSize=16, borderBits=1)
        canvas = np.full((540, 960, 3), 210, dtype=np.uint8)
        canvas[130:410, 270:690] = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)
        ok_b, buf_board = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        assert ok_b
        blank = np.full((480, 640, 3), 80, dtype=np.uint8)
        ok_n, buf_blank = cv2.imencode(".jpg", blank, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        assert ok_n

        ctrl.camera_manager.update_preview_frame(c1.device_id, buf_board.tobytes())
        ctrl.camera_manager.update_preview_frame(c2.device_id, buf_board.tobytes())
        ctrl.camera_manager.update_preview_frame(c3.device_id, buf_blank.tobytes())

        page.show()
        page.tabs.setCurrentIndex(2)
        page._refresh_camera_tiles()

        # Run 2 live ticks (~0.35s) -> must auto-arm, auto-capture, save to disk, and advance from Pose 0 -> Pose 1!
        page._refresh_live_charuco_feeds()
        assert page._auto_capture_enabled is True
        page._refresh_live_charuco_feeds()

        assert len(page._captured_frames) == 1
        assert 0 in page._completed_pose_indices
        assert page.guide.canvas.pose_index == 1  # Automatically advanced to Pose 2!
        # Saved snapshot contains ONLY the 2 valid cameras (not the idle desk cam!)
        _pidx, saved_snap = page._captured_frames[0]
        assert set(saved_snap.keys()) == {c1.camera_uid, c2.camera_uid}
        assert page._capture_session_dir is not None
        assert (page._capture_session_dir / "captures_manifest.json").exists()
        assert (page._capture_session_dir / c1.camera_uid / "pose_01_shot_01.jpg").exists()
        assert (page._capture_session_dir / c2.camera_uid / "pose_01_shot_01.jpg").exists()
    finally:
        page._live_timer.stop()
        page.shutdown_voice()
        page.close()
        ctrl.shutdown()
