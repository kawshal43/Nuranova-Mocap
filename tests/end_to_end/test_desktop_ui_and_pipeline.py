"""End-to-end Desktop UI (PySide6) test verifying all 20 pages, Light/Dark themes, 3D viewport,
Motion Editor playback/cleanup/undo-redo, and project creation.
"""

from __future__ import annotations

from pathlib import Path


from desktop.ui.app_controller import StudioController
from desktop.ui.main_window import NuranovaMainWindow
from desktop.ui.pages.pages_cameras import find_camera_image_subfolders
from engine.motion_cleanup.cleanup import MotionCleanupConfig


def test_desktop_studio_20_pages_theme_viewport_and_editor(qtbot, tmp_path: Path):
    ctrl = StudioController(settings_path=tmp_path / "settings.json")
    win = NuranovaMainWindow(controller=ctrl)
    qtbot.addWidget(win)
    win.show()

    # Verify all 21 studio pages (including Multi-Video Cinema Lab) exist and navigate cleanly
    expected_pages = [
        "Welcome", "Dashboard", "Projects", "Camera Manager", "Camera Configuration",
        "Calibration Wizard", "Calibration Profiles", "Capture Studio",
        "Multi-Video Cinema Lab", "Recording Library",
        "Processing Queue", "Motion Editor", "Marker Editor", "Actor Manager",
        "Facial Capture", "Hand Capture", "Character Retargeting", "Animation Exports",
        "Diagnostics", "Settings", "About",
    ]
    assert list(win.pages.keys()) == expected_pages
    assert win.nav_list.count() == 21

    # Create project & generate synthetic demo take
    proj = ctrl.create_project(tmp_path / "projects", "UI_E2E_Project")
    assert proj.manifest_path.exists()
    doc = ctrl.generate_synthetic_demo_take("ui_demo_walk")
    assert doc.num_frames == 90

    # Visit all 20 pages and verify theme toggle (Light <-> Dark)
    for page_name in expected_pages:
        win.navigate_to(page_name)
        qtbot.wait(5)

    win.apply_theme("dark")
    assert ctrl.settings.general.theme == "dark"
    win.apply_theme("light")
    assert ctrl.settings.general.theme == "light"

    # Exercise Motion Editor scrubbing, cleanup, Undo & Redo
    win.navigate_to("Motion Editor")
    me = win.pages["Motion Editor"]
    me.set_frame(25)
    assert me._frame == 25
    ctrl.apply_motion_cleanup(MotionCleanupConfig(strength=0.8, filter_type="one_euro", enable_foot_locking=True))
    assert ctrl.active_motion.corrected_motion is not None
    assert ctrl.undo_motion_edit() is True
    assert ctrl.redo_motion_edit() is True

    # Exercise 3D viewport view presets, Character/Skeleton display modes & live HUD repaint cleanly
    for mode in ("character", "skeleton", "both"):
        me.viewport.canvas.set_display_mode(mode)
        me.viewport.canvas.set_live_status(f"● LIVE · {mode}", "live")
        me.viewport.canvas.repaint()
    for preset in ("perspective", "front", "rear", "left", "right", "top"):
        me.viewport.canvas.set_view_preset(preset)
        me.viewport.canvas.repaint()

    # Exercise Card-Type Home UI & Capture Studio Live 3D Character Preview worker
    win.navigate_to("Welcome")
    welcome = win.pages["Welcome"]
    assert len(welcome._kpi_labels) == 4
    assert len(welcome._card_badges) == 6

    win.navigate_to("Capture Studio")
    cap = win.pages["Capture Studio"]
    received_poses = []
    ctrl.live_preview.pose_updated.connect(lambda pos, conf, src, fps: received_poses.append((pos, src)))
    cap._toggle_live_preview()
    assert ctrl.live_preview.running is True
    qtbot.waitUntil(lambda: len(received_poses) >= 1, timeout=3000)
    pos0, src0 = received_poses[0]
    assert pos0.shape == (25, 3)
    assert "Demo" in src0
    cap.viewport.canvas.repaint()
    cap._toggle_live_preview()
    assert ctrl.live_preview.running is False

    # Exercise Calibration Wizard visual Camera & Board Placement Guide, A3/A4 presets, and folder filtering
    win.navigate_to("Calibration Wizard")
    cal_pg = win.pages["Calibration Wizard"]
    assert cal_pg.tabs.count() == 3
    for n_cams in (2, 3, 4, 6):
        cal_pg.guide.set_num_cameras(n_cams)
        for p_idx in range(6):
            cal_pg.guide.set_pose_index(p_idx)
            cal_pg.guide.canvas.repaint()
    cal_pg._apply_a3_preset()
    assert cal_pg.spin_sx.value() == 9 and cal_pg.spin_sy.value() == 6 and cal_pg.spin_sq.value() == 40.0
    assert "A3" in cal_pg.lbl_board_spec.text()
    assert cal_pg.lbl_board_thumb.pixmap() is not None and not cal_pg.lbl_board_thumb.pixmap().isNull()
    cal_pg._apply_a4_preset()
    assert cal_pg.spin_sx.value() == 7 and cal_pg.spin_sy.value() == 5 and cal_pg.spin_sq.value() == 35.0
    assert "A4" in cal_pg.lbl_board_spec.text()
    assert len(cal_pg._live_pose_pills) == 6
    cal_pg.tabs.setCurrentIndex(2)
    cal_pg._set_auto_capture(True)
    assert cal_pg._auto_capture_enabled is True
    cal_pg._set_auto_capture(False)
    assert cal_pg._auto_capture_enabled is False

    # Verify folder calibration ignores .git / __pycache__ / empty directories
    calib_test_root = tmp_path / "calib_scan"
    (calib_test_root / ".git").mkdir(parents=True)
    (calib_test_root / "__pycache__").mkdir(parents=True)
    (calib_test_root / "empty_dir").mkdir(parents=True)
    (calib_test_root / "cam_01").mkdir(parents=True)
    (calib_test_root / "cam_01" / "001.jpg").write_bytes(b"fake")
    (calib_test_root / "cam_02").mkdir(parents=True)
    (calib_test_root / "cam_02" / "001.png").write_bytes(b"fake")
    found_dirs = [d.name for d in find_camera_image_subfolders(calib_test_root)]
    assert found_dirs == ["cam_01", "cam_02"]

    # Exercise Single-Cam Mode (Skip Calibration) + 3D Root Walk toggle
    ctrl.set_single_camera_mode(True)
    assert ctrl.settings.general.single_camera_mode is True
    assert "Skip Calib" in win._step_pills["Calibration Wizard"].text()
    assert "1-Cam" in welcome._kpi_labels["calib"][0].text()
    ctrl.set_single_cam_root_motion(False)
    assert ctrl.settings.general.single_cam_root_motion is False
    ctrl.set_single_cam_root_motion(True)
    ctrl.set_single_camera_mode(False)
    assert ctrl.settings.general.single_camera_mode is False

    # Verify QR pairing URI & pixmap generation + CameraManager QR hub
    uri = ctrl.get_pairing_uri()
    assert uri.startswith("nuranova://pair?") and "pin=" in uri
    pm = ctrl.generate_pairing_qr_pixmap(160)
    assert pm is not None and not pm.isNull()

    win.close()


def test_calibration_wizard_auto_arms_captures_and_saves_without_project(qtbot, tmp_path: Path):
    """Verify Step 3 Live Guided Calibration auto-arms with 2 streaming cameras, auto-captures all 6 poses,
    saves per-pose JPEGs + manifest immediately (even with no open project), and auto-solves & saves JSON.
    """
    import cv2
    import numpy as np
    from desktop.camera_manager.manager import ManagedCameraRecord

    ctrl = StudioController(settings_path=tmp_path / "settings.json")
    # Match user's exact starting state: NO active project + single_camera_mode=True + 2 connected phones!
    ctrl.project_manager.active_project = None
    ctrl.set_single_camera_mode(True)

    win = NuranovaMainWindow(controller=ctrl)
    qtbot.addWidget(win)
    win.show()

    # Register two streaming cameras (cam_02 and cam_03)
    cam2 = ManagedCameraRecord(camera_uid="cam_02", device_id="dev_cam_02", name="Phone A", auth_token="tok2")
    cam3 = ManagedCameraRecord(camera_uid="cam_03", device_id="dev_cam_03", name="Phone B", auth_token="tok3")
    ctrl.camera_manager._cameras[cam2.device_id] = cam2
    ctrl.camera_manager._cameras[cam3.device_id] = cam3

    win.navigate_to("Calibration Wizard")
    cal_pg = win.pages["Calibration Wizard"]
    cal_pg.chk_voice_coach.setChecked(False)
    cal_pg._apply_a4_preset()
    cfg = cal_pg._board_cfg()
    board, _ = cfg.create_opencv_board()

    # Render synthetic perspective-projected views of the A4 ChArUco board from 2 stereo cameras
    w, h = 640, 360
    board_img = board.generateImage((700, 500), marginSize=20, borderBits=1)
    board_bgr = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)
    bw_m = cfg.squares_x * cfg.effective_square_length_m
    bh_m = cfg.squares_y * cfg.effective_square_length_m
    margin_m = bw_m * (20.0 / 660.0)
    src_pts = np.array(
        [[0.0, 0.0], [700.0, 0.0], [700.0, 500.0], [0.0, 500.0]],
        dtype=np.float32,
    )
    board_corners_3d = np.array(
        [
            [-margin_m, -margin_m, 0.0],
            [bw_m + margin_m, -margin_m, 0.0],
            [bw_m + margin_m, bh_m + margin_m, 0.0],
            [-margin_m, bh_m + margin_m, 0.0],
        ],
        dtype=np.float64,
    )
    K = np.array([[620.0, 0.0, w * 0.5], [0.0, 620.0, h * 0.5], [0.0, 0.0, 1.0]], dtype=np.float64)
    # Two converged stereo cameras ~0.9m apart looking at board center around z=1.1m
    R_c2, _ = cv2.Rodrigues(np.array([0.0, 0.22, 0.0], dtype=np.float64))
    t_c2 = np.array([-0.38, 0.0, 0.0], dtype=np.float64)
    R_c3, _ = cv2.Rodrigues(np.array([0.0, -0.22, 0.0], dtype=np.float64))
    t_c3 = np.array([0.38, 0.0, 0.0], dtype=np.float64)

    def _render_cam_jpeg(R_cam, t_cam, rvec_b, tvec_b) -> bytes:
        R_b, _ = cv2.Rodrigues(np.asarray(rvec_b, dtype=np.float64))
        pts_w = (board_corners_3d @ R_b.T) + np.asarray(tvec_b, dtype=np.float64)
        pts_c = (pts_w @ R_cam.T) + t_cam
        uv = (pts_c @ K.T)
        uv = (uv[:, :2] / uv[:, 2:3]).astype(np.float32)
        H_mat = cv2.getPerspectiveTransform(src_pts, uv)
        canvas = np.full((h, w, 3), 185, dtype=np.uint8)
        warped = cv2.warpPerspective(board_bgr, H_mat, (w, h), borderValue=(185, 185, 185))
        mask = cv2.warpPerspective(np.full((500, 700), 255, dtype=np.uint8), H_mat, (w, h))
        canvas[mask > 127] = warped[mask > 127]
        ok, enc = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        assert ok
        return enc.tobytes()

    # 6 distinct 3D board poses matching POSE_TARGET_ZONES 0..5 for both cameras
    pose_specs = [
        ([0.0, 0.0, 0.0], [-0.12, -0.09, 1.05]),        # 0: Center Upright
        ([0.28, 0.14, 0.04], [-0.12, -0.09, 1.02]),     # 1: Center Tilted
        ([-0.15, 0.25, -0.06], [-0.24, -0.09, 1.08]),   # 2: Left Side
        ([0.15, -0.25, 0.06], [0.02, -0.09, 1.08]),     # 3: Right Side
        ([-0.30, 0.08, 0.03], [-0.12, 0.05, 1.04]),     # 4: Low Angle
        ([0.30, -0.08, -0.03], [-0.12, -0.22, 1.04]),   # 5: High Angle
    ]

    # Seed Pose 0 live frames BEFORE switching to Step 3 tab (never clicking 'Start Guided Auto-Capture'!)
    cam2.latest_preview_jpeg = _render_cam_jpeg(R_c2, t_c2, *pose_specs[0])
    cam3.latest_preview_jpeg = _render_cam_jpeg(R_c3, t_c3, *pose_specs[0])
    cal_pg._refresh_camera_tiles()
    cal_pg.tabs.setCurrentIndex(2)

    # Verify Guided Auto-Capture AUTO-ARMED without manual click!
    assert cal_pg._auto_capture_enabled is True

    # Feed all 6 poses through live detection loop -> must auto-capture & save every single pose!
    for p_idx, (rv, tv) in enumerate(pose_specs):
        cal_pg._cooldown_until = 0.0
        cal_pg.guide.set_pose_index(p_idx)
        cam2.latest_preview_jpeg = _render_cam_jpeg(R_c2, t_c2, rv, tv)
        cam3.latest_preview_jpeg = _render_cam_jpeg(R_c3, t_c3, rv, tv)
        # Steady lock-on ticks (target = 3)
        for _ in range(4):
            cal_pg._refresh_live_charuco_feeds()
            if p_idx in cal_pg._completed_pose_indices:
                break
        assert p_idx in cal_pg._completed_pose_indices, f"Pose {p_idx} failed to auto-capture! Log:\n{cal_pg.log.toPlainText()}"

    # Verify per-pose JPEGs + manifest were immediately saved on disk even without an active project!
    sess_dir = cal_pg._capture_session_dir
    assert sess_dir is not None and sess_dir.exists()
    assert (sess_dir / "captures_manifest.json").exists()
    assert len(list((sess_dir / "cam_02").glob("*.jpg"))) >= 6
    assert len(list((sess_dir / "cam_03").glob("*.jpg"))) >= 6

    # Wait for the auto-scheduled calibration solve + save to finish!
    qtbot.waitUntil(lambda: ctrl.active_calibration is not None, timeout=10000)
    assert ctrl.active_calibration is not None
    assert len(ctrl.active_calibration.cameras) == 2
    assert ctrl._last_calib_path.exists()
    archived = list((ctrl._last_calib_path.parent / "calibrations").glob("LiveCalib_2cams_*.json"))
    assert len(archived) >= 1
    # Multi-camera calibration automatically disabled 1-Cam AI Mode!
    assert ctrl.settings.general.single_camera_mode is False
    assert "SOLVED & SAVED" in cal_pg.lbl_coach_headline.text()

    # Verify explicit user pause is respected and never claims "CAPTURING POSE..."
    cal_pg._clear_captured_poses()
    cal_pg._set_auto_capture(False, user_triggered=True)
    cal_pg._refresh_live_charuco_feeds()
    assert cal_pg._auto_capture_enabled is False
    assert "CAPTURING POSE" not in cal_pg.lbl_coach_headline.text()

    win.close()
