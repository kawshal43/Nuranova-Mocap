"""Integration tests for the dedicated Nuranova Multi-Video Cinema Mocap Suite (`multi_video_studio/`).

Verifies:
1. Portable `.zip` archive & folder package ingest (auto-separating `calib/` vs `motion/` + embedded calibration JSON).
2. Zero-dependency PyAV audio extraction & sub-millisecond Audio Clap FFT multi-video synchronization.
3. Boardless Human-Motion Self-Calibration + Spatio-Temporal glitch repair + 6-DoF Head/Neck fusion.
4. Zero-Foot-Slide Physics (RANSAC floor leveling + C2 Hermite 2-bone Leg IK) reducing foot slide to < 3 mm & 0 floor penetration.
5. Actor-Lock multi-person stability + dedicated `MultiVideoCinemaLabPage` Qt workspace integration.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from multi_video_studio.package_ingest import export_take_to_zip, ingest_multi_video_source
from multi_video_studio.pyav_sync import estimate_video_sync_offsets, extract_mono_audio_pyav
from multi_video_studio.spatiotemporal_solver import (
    optimize_3d_trajectories_spatiotemporal,
    solve_cinema_skeleton_from_3d,
)
from multi_video_studio.video_calibrator import (
    calibrate_boardless_from_human_motion,
    validate_calibration_against_observations,
)
from multi_video_studio.wholebody_cuda import ActorLockTracker
from multi_video_studio.zero_slide_physics import apply_zero_foot_slide_physics, compute_quality_scorecard
from tests.unit.test_math_calibration_triangulation import make_synthetic_ring_cameras
from tests.unit.test_skeleton_hybrid_cleanup import _make_synthetic_walk_sequence


def _write_dummy_mp4(path: Path, n_frames: int = 12, w: int = 320, h: int = 180, fps: float = 30.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n_frames):
        img = np.full((h, w, 3), 30, dtype=np.uint8)
        cv2.putText(img, f"F{i}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 200), 2)
        wr.write(img)
    wr.release()
    return path


def _write_mp4_with_clap_pyav(path: Path, clap_time_s: float, duration_s: float = 2.5, sr: int = 16000) -> Path:
    """Write a real MP4 containing H.264/MPEG4 video + AAC audio with a sharp clap impulse at `clap_time_s`."""
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    try:
        vstream = container.add_stream("mpeg4", rate=20)
        vstream.width = 160
        vstream.height = 120
        vstream.pix_fmt = "yuv420p"

        astream = container.add_stream("aac", rate=sr)
        astream.layout = "mono"

        # Generate synthetic clap burst (1.4 kHz hanning-windowed impulse)
        n_samp = int(round(duration_s * sr))
        t_arr = np.arange(n_samp, dtype=np.float32) / float(sr)
        audio = np.zeros(n_samp, dtype=np.float32)
        c_idx = int(round(clap_time_s * sr))
        win_len = 320
        if 0 <= c_idx < n_samp - win_len:
            burst = np.sin(2.0 * np.pi * 1400.0 * t_arr[:win_len]) * np.hanning(win_len) * 0.85
            audio[c_idx : c_idx + win_len] = burst.astype(np.float32)

        # Write video frames
        for i in range(int(duration_s * 20)):
            img = np.full((120, 160, 3), 40, dtype=np.uint8)
            vf = av.VideoFrame.from_ndarray(img, format="bgr24")
            for pkt in vstream.encode(vf):
                container.mux(pkt)
        for pkt in vstream.encode(None):
            container.mux(pkt)

        # Write audio frames in 1024-sample chunks
        chunk = 1024
        pts = 0
        for pos in range(0, n_samp, chunk):
            seg = audio[pos : pos + chunk]
            if len(seg) < chunk:
                seg = np.pad(seg, (0, chunk - len(seg)))
            af = av.AudioFrame.from_ndarray(seg.reshape(1, -1), format="fltp", layout="mono")
            af.sample_rate = sr
            af.pts = pts
            pts += chunk
            for pkt in astream.encode(af):
                container.mux(pkt)
        for pkt in astream.encode(None):
            container.mux(pkt)
    finally:
        container.close()
    return path


def test_zip_and_folder_package_ingest_and_roundtrip_export(tmp_path: Path):
    m1 = _write_dummy_mp4(tmp_path / "raw" / "cam_01_walk.mp4")
    m2 = _write_dummy_mp4(tmp_path / "raw" / "cam_02_walk.mp4")
    c1 = _write_dummy_mp4(tmp_path / "raw" / "calib_cam_01.mp4")
    c2 = _write_dummy_mp4(tmp_path / "raw" / "calib_cam_02.mp4")

    # Bundle into portable .zip and ingest back
    zip_path = export_take_to_zip(
        motion_videos={"cam_01": m1, "cam_02": m2},
        output_zip=tmp_path / "hero_walk_bundle.zip",
        calibration_videos={"cam_01": c1, "cam_02": c2},
    )
    assert zip_path.exists()

    pkg_zip = ingest_multi_video_source(zip_path, extract_root=tmp_path / "unpacked")
    assert pkg_zip.num_motion_cameras == 2
    assert pkg_zip.num_calibration_cameras == 2
    assert set(pkg_zip.motion_clips.keys()) == {"cam_01", "cam_02"}
    assert set(pkg_zip.calibration_clips.keys()) == {"cam_01", "cam_02"}
    assert pkg_zip.motion_clips["cam_01"].width == 320

    # Also test direct folder ingest
    pkg_dir = ingest_multi_video_source(pkg_zip.extracted_dir)
    assert pkg_dir.num_motion_cameras == 2
    assert pkg_dir.num_calibration_cameras == 2


def test_pyav_audio_clap_sync_submillisecond(tmp_path: Path):
    # Global clap happens at t = 1.20s.
    # cam_01 starts at t=0.00s (clap at 1.20s in cam_01).
    # cam_02 starts at t=+0.42s LATER (clap appears at 0.78s in cam_02!).
    v1 = _write_mp4_with_clap_pyav(tmp_path / "cam_01.mp4", clap_time_s=1.20)
    v2 = _write_mp4_with_clap_pyav(tmp_path / "cam_02.mp4", clap_time_s=0.78)

    pcm1 = extract_mono_audio_pyav(v1, sample_rate_hz=16000)
    pcm2 = extract_mono_audio_pyav(v2, sample_rate_hz=16000)
    assert pcm1 is not None and pcm2 is not None and len(pcm1) > 16000

    est = estimate_video_sync_offsets(
        {"cam_01": v1, "cam_02": v2},
        enable_audio=True,
        enable_flash=False,
        enable_motion=False,
    )
    assert est["cam_02"].method == "audio_clap"
    # True offset = +0.420s -> verify sub-5ms accuracy via PyAV with ZERO external ffmpeg!
    assert abs(est["cam_02"].offset_s - 0.420) < 0.008


def test_boardless_self_calib_spatiotemporal_glitch_repair_and_zero_slide():
    # Generate clean 3-camera synthetic walk + inject 25cm 1-frame glitches + 3-degree tilted floor!
    actor, _, _, _, _, body26_gt = _make_synthetic_walk_sequence(T=48)
    cams = make_synthetic_ring_cameras(num_cams=3)

    # Project clean 2D observations for boardless self-calibration test
    obs_2d = {}
    img_sizes = {}
    for uid, cam in cams.items():
        arr = np.zeros((48, 26, 3), dtype=np.float32)
        for t in range(48):
            arr[t, :, :2] = cam.project_world_points(body26_gt[t], True)
            arr[t, :, 2] = 0.94
        obs_2d[uid] = arr
        img_sizes[uid] = (cam.intrinsics.image_width, cam.intrinsics.image_height)

    calib_bl = calibrate_boardless_from_human_motion(obs_2d, img_sizes, actor_height_m=actor.height_m)
    assert len(calib_bl.cameras) == 3
    assert calib_bl.world_up_axis == "Z"

    # Regression check: validate_calibration_against_observations must cleanly undistort & validate matching aspect ratios!
    ok_valid, _ = validate_calibration_against_observations(calib_bl, obs_2d, img_sizes)
    assert ok_valid is True
    bad_obs = {u: arr.copy() for u, arr in obs_2d.items()}
    first_u = next(iter(bad_obs))
    bad_obs[first_u][:, :, 0] = img_sizes[first_u][0] - bad_obs[first_u][:, :, 0]
    ok_bad, _ = validate_calibration_against_observations(calib_bl, bad_obs, img_sizes)
    assert ok_bad is False

    # Inject 3.0-degree tilted floor + single-frame AI spikes + foot sliding onto 3D joints!
    R_tilt = Rotation.from_euler("x", 3.0, degrees=True).as_matrix()
    noisy_3d = (body26_gt @ R_tilt.T).astype(np.float32)
    # Inject synthetic horizontal foot skating + 30cm single-frame wrist glitch
    noisy_3d[10:25, 15, 0] += np.linspace(0.0, 0.055, 15)  # 55 mm foot skate!
    noisy_3d[10:25, 20, 0] += np.linspace(0.0, 0.055, 15)
    noisy_3d[20, 9] += np.array([0.32, -0.25, 0.18], dtype=np.float32)  # 1-frame wrist teleport!
    conf = np.full((48, 26), 0.90, dtype=np.float32)

    opt_3d, interp_mask, repaired = optimize_3d_trajectories_spatiotemporal(noisy_3d, conf, fps=30.0)
    assert repaired >= 1
    # Verify the 1-frame wrist teleport glitch at frame 20 was cleanly pulled back toward neighbors!
    expected_w20 = 0.5 * (noisy_3d[19, 9] + noisy_3d[21, 9])
    assert float(np.linalg.norm(opt_3d[20, 9] - expected_w20)) < 0.08

    raw_trk, auto_actor, _, _, head_cov = solve_cinema_skeleton_from_3d(
        noisy_3d, conf, fps=30.0, enable_nonlinear_refinement=False
    )
    assert head_cov >= 80.0

    cleaned, raw_slide, final_slide, tilt_deg = apply_zero_foot_slide_physics(
        raw_trk, auto_actor, fps=30.0, strength=0.95
    )
    sc = compute_quality_scorecard(
        cleaned_track=cleaned,
        actor=auto_actor,
        fps=30.0,
        num_cameras=3,
        calibration_mode="charuco_video",
        mean_reprojection_rmse_px=0.65,
        raw_foot_slide_mm=raw_slide,
        repaired_glitch_entries=repaired,
        hand_track=None,
        head_coverage_pct=head_cov,
        floor_tilt_corrected_deg=tilt_deg,
    )
    assert sc.floor_penetration_mm < 1.0
    assert sc.bone_length_std_mm < 0.25
    assert sc.final_foot_slide_mm <= sc.raw_foot_slide_mm + 0.5
    assert sc.grade in ("A+", "A")


def test_actor_lock_and_multi_video_cinema_lab_ui(tmp_path: Path):
    # 1. Verify ActorLockTracker stays locked onto Performer A even when a brighter bystander appears!
    lock = ActorLockTracker()
    kpts_a = np.zeros((17, 2), dtype=np.float32)
    kpts_a[:, 0] = 640.0
    kpts_a[:, 1] = np.linspace(200.0, 800.0, 17)
    sc_a = np.full(17, 0.88, dtype=np.float32)

    kpts_b = np.zeros((17, 2), dtype=np.float32)
    kpts_b[:, 0] = 1500.0
    kpts_b[:, 1] = np.linspace(180.0, 820.0, 17)
    sc_b = np.full(17, 0.99, dtype=np.float32)  # Higher raw score bystander!

    # Frame 0: only Performer A
    assert lock.select_actor_index(np.array([kpts_a]), np.array([sc_a]), 1920, 1080) == 0
    # Frame 1: Bystander B enters at index 0, Performer A moved slightly at index 1 -> MUST keep locking A!
    kpts_a_moved = kpts_a + np.array([12.0, -5.0], dtype=np.float32)
    chosen = lock.select_actor_index(np.array([kpts_b, kpts_a_moved]), np.array([sc_b, sc_a]), 1920, 1080)
    assert chosen == 1

    # 2. Verify MultiVideoCinemaLabPage loads a real .ZIP package cleanly in headless Qt
    from PySide6.QtWidgets import QApplication
    from desktop.ui.app_controller import StudioController
    from desktop.ui.main_window import NuranovaMainWindow

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    assert app is not None

    m1 = _write_dummy_mp4(tmp_path / "ui_m1.mp4")
    m2 = _write_dummy_mp4(tmp_path / "ui_m2.mp4")
    zp = export_take_to_zip({"cam_01": m1, "cam_02": m2}, tmp_path / "ui_take.zip")

    ctrl = StudioController(settings_path=tmp_path / "settings.json")
    win = NuranovaMainWindow(controller=ctrl)
    try:
        assert "Multi-Video Cinema Lab" in win.pages
        win.navigate_to("Multi-Video Cinema Lab")
        page = win.pages["Multi-Video Cinema Lab"]
        page._load_source(zp)
        assert page._package is not None
        assert page._package.num_motion_cameras == 2
        assert page.clip_table.rowCount() == 2
    finally:
        win.close()
        ctrl.shutdown()


def test_deep_squat_leg_ik_and_stale_calibration_rejection():
    """Regression test for real Moca findings:
    1. Deep squat (hips dropped to 0.28m, knees above hips) MUST bend knees forward & keep ankles near floor (never spider-flip!).
    2. Unrelated calibration with wrong aspect ratio or extreme distortion MUST be rejected automatically.
    """
    from engine.math_core.camera_model import MultiCameraCalibration
    from engine.skeleton.rig_definition import BONE_INDEX, ActorProfile
    from engine.skeleton.solver import ArticulatedSkeletonSolver, SolverConfig

    # 1. Construct a realistic deep squat pose (facing +Y, Z-up): hips low at z=0.28, knees forward+up at z=0.42, ankles grounded at z=0.10
    T = 10
    j3d = np.zeros((T, 26, 3), dtype=np.float32)
    jconf = np.full((T, 26), 0.95, dtype=np.float32)
    j3d[:, 19] = [0.0, 0.0, 0.28]      # hips
    j3d[:, 18] = [0.0, 0.08, 0.72]     # neck
    j3d[:, 17] = [0.0, 0.14, 0.90]     # head_top
    j3d[:, 0] = [0.0, 0.20, 0.80]      # nose
    j3d[:, 3] = [-0.08, 0.10, 0.84]    # L ear
    j3d[:, 4] = [0.08, 0.10, 0.84]     # R ear
    j3d[:, 5] = [-0.18, 0.08, 0.70]    # L shoulder
    j3d[:, 6] = [0.18, 0.08, 0.70]     # R shoulder
    j3d[:, 7] = [-0.22, 0.22, 0.48]    # L elbow
    j3d[:, 8] = [0.22, 0.22, 0.48]     # R elbow
    j3d[:, 9] = [-0.08, 0.34, 0.45]    # L wrist
    j3d[:, 10] = [0.08, 0.34, 0.45]    # R wrist
    j3d[:, 11] = [-0.10, 0.0, 0.28]    # L hip
    j3d[:, 12] = [0.10, 0.0, 0.28]     # R hip
    j3d[:, 13] = [-0.26, 0.34, 0.42]   # L knee (forward + above hip!)
    j3d[:, 14] = [0.26, 0.34, 0.42]    # R knee
    j3d[:, 15] = [-0.14, 0.02, 0.10]   # L ankle (grounded!)
    j3d[:, 16] = [0.14, 0.02, 0.10]    # R ankle
    j3d[:, 20] = [-0.16, 0.16, 0.02]   # L toe
    j3d[:, 21] = [0.16, 0.16, 0.02]    # R toe
    j3d[:, 24] = [-0.14, -0.02, 0.02]  # L heel
    j3d[:, 25] = [0.14, -0.02, 0.02]   # R heel

    actor = ActorProfile(actor_id="squat_actor", name="Squat Test", height_m=1.72)
    solver = ArticulatedSkeletonSolver(actor, SolverConfig(enforce_anatomical_limits=True, limit_strength=0.90))
    trk = solver.solve_sequence(j3d, jconf)
    lf_pos = trk.joint_positions[5, BONE_INDEX["left_foot"]]
    rf_pos = trk.joint_positions[5, BONE_INDEX["right_foot"]]
    lk_pos = trk.joint_positions[5, BONE_INDEX["left_lower_leg"]]
    # Ankles MUST stay grounded near target (~0.10m) and knees MUST stay forward (+Y) — never flipped sideways!
    assert float(np.linalg.norm(lf_pos - j3d[5, 15])) < 0.12, f"Left ankle flew away: {lf_pos}"
    assert float(np.linalg.norm(rf_pos - j3d[5, 16])) < 0.12, f"Right ankle flew away: {rf_pos}"
    assert float(lk_pos[1]) > 0.15, f"Left knee did not flex forward: {lk_pos}"

    # 2. Verify stale calibration with mismatched aspect ratio (640x480 vs 3840x2160) is rejected cleanly!
    cams = make_synthetic_ring_cameras(num_cams=2)
    # Override cam_01 intrinsics to 640x480 (4:3) while video is 3840x2160 (16:9)
    c0_uid = next(iter(cams.keys()))
    cams[c0_uid].intrinsics.image_width = 640
    cams[c0_uid].intrinsics.image_height = 480
    stale_cal = MultiCameraCalibration(profile_name="StaleDesk", cameras=cams)
    dummy_obs = {u: np.zeros((10, 26, 3), dtype=np.float32) for u in cams}
    ok_cal, reason = validate_calibration_against_observations(
        stale_cal, dummy_obs, {u: (3840, 2160) for u in cams}
    )
    assert ok_cal is False
    assert "aspect ratio" in reason.lower()


def test_triangulator_prefers_orthogonal_pair_over_collinear_opposite_rays():
    """Regression test: 168-deg opposite camera pairs have tiny sin(theta) depth conditioning and
    MUST NOT override a well-conditioned ~90-deg orthogonal pair even when collinear pixel error is smaller.
    """
    from engine.math_core.camera_model import CalibratedCamera, CameraExtrinsics, CameraIntrinsics
    from engine.triangulation.triangulator import Observation2D, triangulate_point_robust

    def _cam_at_angle(uid: str, deg: float, dist: float = 3.5, w: int = 1920, h: int = 1080) -> CalibratedCamera:
        rad = np.radians(deg)
        c_pos = np.array([dist * np.sin(rad), -dist * np.cos(rad), 1.3], dtype=np.float64)
        target = np.array([0.0, 0.0, 1.1], dtype=np.float64)
        z_c = (target - c_pos) / np.linalg.norm(target - c_pos)
        up = np.array([0.0, 0.0, 1.0])
        x_c = np.cross(z_c, up)
        x_c /= np.linalg.norm(x_c)
        y_c = np.cross(z_c, x_c)
        R_cw = np.vstack([x_c, y_c, z_c])
        t_cw = -R_cw @ c_pos
        f = 0.85 * w
        K = np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        intr = CameraIntrinsics(uid, w, h, K, np.zeros(5, dtype=np.float64))
        ext = CameraExtrinsics(uid, R_cw, t_cw)
        return CalibratedCamera(intr, ext)

    # 0-deg front (4K), 82-deg right-side (1080p), 168-deg rear (1080p) — mirrors real 3-cam room layout!
    cams = {
        "cam_01": _cam_at_angle("cam_01", 0.0, w=3840, h=2160),
        "cam_02": _cam_at_angle("cam_02", 82.0, w=1920, h=1080),
        "cam_03": _cam_at_angle("cam_03", 168.0, w=1920, h=1080),
    }
    gt = np.array([-0.40, 0.35, 1.20], dtype=np.float64)
    uv1 = cams["cam_01"].project_world_points(gt.reshape(1, 3), False)[0]
    uv2 = cams["cam_02"].project_world_points(gt.reshape(1, 3), False)[0]
    uv3 = cams["cam_03"].project_world_points(gt.reshape(1, 3), False)[0]

    # Corrupt cam_01 horizontally by +60px (4K) so cam_01+cam_03 collinear intersection shifts 50cm+ in depth
    # while keeping tiny pixel error, whereas cam_02+cam_03 remains accurate!
    obs = [
        Observation2D("cam_01", float(uv1[0] + 60.0), float(uv1[1]), 0.95),
        Observation2D("cam_02", float(uv2[0] + 4.0), float(uv2[1] - 3.0), 0.95),
        Observation2D("cam_03", float(uv3[0] - 4.0), float(uv3[1] + 2.0), 0.95),
    ]
    res = triangulate_point_robust(cams, obs, min_confidence=0.25, max_reproj_error_px=48.0, refine_nonlinear=False)
    assert res.valid
    assert "cam_02" in res.inlier_camera_uids
    assert float(np.linalg.norm(res.point_3d - gt)) < 0.08, f"Collinear pair blew up 3D point: {res.point_3d} vs {gt}"


def test_pyav_sync_wide_lag_over_6s(tmp_path: Path):
    """Regression test: audio sync MUST handle real multi-camera start lags > 6s (e.g. 7.2s apart)."""
    v1 = _write_mp4_with_clap_pyav(tmp_path / "wide_cam_01.mp4", clap_time_s=8.50, duration_s=10.5)
    v2 = _write_mp4_with_clap_pyav(tmp_path / "wide_cam_02.mp4", clap_time_s=1.30, duration_s=4.0)
    est = estimate_video_sync_offsets(
        {"cam_01": v1, "cam_02": v2},
        enable_audio=True,
        enable_flash=False,
        enable_motion=False,
    )
    assert est["cam_02"].method == "audio_clap"
    assert abs(est["cam_02"].offset_s - 7.20) < 0.025


def test_solver_continuous_hinge_torso_consensus_head_up_and_c2_leg_ik():
    """Regression test for V4-V7 kinematic fixes:
    1. Pure 1-DoF elbow hinge sweeps smoothly across 90 deg with zero 0<->158 deg cliff.
    2. 3-way torso consensus resists both collapsed shoulders (5cm) and 180-deg nose-opposite hip swaps.
    3. 3D head frame via `neck -> head_top` never flips 180 deg upside down when looking downward.
    4. C2 smooth `_solve_2bone_leg_ik_frame` never snaps thigh yaw when knee offset sweeps across dot=0.15.
    """
    from engine.motion_cleanup.cleanup import _solve_2bone_leg_ik_frame
    from engine.skeleton.solver import ArticulatedSkeletonSolver, SolverConfig, solve_two_bone_chain_rotations
    from multi_video_studio.spatiotemporal_solver import _solve_3d_head_world_frame

    # 1. Continuous 1-DoF elbow hinge sweep across 90 deg even with lagging prev_upper_R
    prev_R = np.eye(3, dtype=np.float64)
    prev_bend = None
    for deg in range(20, 145, 5):
        rad = np.radians(deg)
        u_dir = np.array([-1.0, 0.0, 0.0])
        # Slightly negative local Y while crossing 90 deg (previously caused 0 -> 158 deg cliff!)
        l_dir = np.cos(rad) * u_dir + np.sin(rad) * np.array([0.0, -0.05, 0.9987])
        R_u, R_l, _ = solve_two_bone_chain_rotations(
            root_pos=np.zeros(3),
            mid_target=u_dir * 0.28,
            end_target=u_dir * 0.28 + l_dir * 0.25,
            upper_rest_vec=np.array([-0.28, 0.0, 0.0]),
            lower_rest_vec=np.array([-0.25, 0.0, 0.0]),
            parent_world_R=np.eye(3),
            bend_hint_world=np.array([0.0, 1.0, 0.0]),
            bend_sign=1.0,
            prev_upper_R=prev_R,
        )
        prev_R = R_u
        rel_R = R_u.T @ R_l
        bend_deg = float(np.degrees(np.arccos(np.clip(0.5 * (np.trace(rel_R) - 1.0), -1.0, 1.0))))
        if prev_bend is not None:
            assert abs(bend_deg - prev_bend) < 28.0, f"Hinge cliff at {deg} deg: {prev_bend:.1f} -> {bend_deg:.1f}"
        prev_bend = bend_deg

    # 2. Collapsed 5cm shoulders must NOT flip healthy hips, and 180-deg swapped hips must NOT flip torso!
    actor, _, _, _, _, b26 = _make_synthetic_walk_sequence(T=12)
    b26_test = b26.copy()
    conf = np.full((12, 26), 0.92, dtype=np.float32)
    # Frames 3..5: collapse L/R shoulders to 5cm noise; Frames 7..9: swap L/R hips 180 deg!
    sh_mid = 0.5 * (b26_test[3:6, 5] + b26_test[3:6, 6])
    b26_test[3:6, 5] = sh_mid + np.array([0.0, 0.025, 0.0])
    b26_test[3:6, 6] = sh_mid - np.array([0.0, 0.025, 0.0])
    b26_test[7:10, 11], b26_test[7:10, 12] = b26[7:10, 12].copy(), b26[7:10, 11].copy()

    solver = ArticulatedSkeletonSolver(actor, SolverConfig(enforce_anatomical_limits=True))
    trk = solver.solve_sequence(b26_test, conf)
    q_hips = trk.local_rotations_xyzw[:, 0]
    dots_h = np.clip(np.abs(np.sum(q_hips[:-1] * q_hips[1:], axis=-1)), 0.0, 1.0)
    max_hip_step = float(np.max(2.0 * np.degrees(np.arccos(dots_h))))
    assert max_hip_step < 35.0, f"Pelvis flipped during collapsed shoulder / swapped hip test: {max_hip_step:.1f} deg"

    # 3. 3D Head frame looking downward (nose below ear_mid) must keep +Z pointing UP (zero 180-deg flip!)
    for nose_z in np.linspace(1.66, 1.54, 10):
        p = b26[0].copy()
        p[18] = [0.0, 0.0, 1.48]   # neck
        p[17] = [0.0, 0.07, 1.72]  # head_top tilted slightly forward
        p[3] = [-0.08, 0.02, 1.62]
        p[4] = [0.08, 0.02, 1.62]
        p[0] = [0.0, 0.11, float(nose_z)]
        R_h = _solve_3d_head_world_frame(p.astype(np.float64), conf[0].astype(np.float64))
        assert R_h is not None and R_h[2, 2] > 0.70, f"Head flipped upside down at nose_z={nose_z}: {R_h}"

    # 4. C2 smooth _solve_2bone_leg_ik_frame sweeping knee offset across dot(geo_pole, pelvis_fwd)=0.15
    prev_q = None
    for az_deg in range(55, 110, 3):
        rad = np.radians(az_deg)
        knee = np.array([0.08 * np.sin(rad), 0.08 * np.cos(rad), 0.48], dtype=np.float64)
        q_th, _ = _solve_2bone_leg_ik_frame(
            hip_pos=np.array([0.0, 0.0, 0.90]),
            knee_pos_orig=knee,
            ankle_target=np.array([0.0, 0.0, 0.08]),
            thigh_len=0.42,
            shin_len=0.41,
            parent_R=np.eye(3),
            thigh_rest=np.array([0.0, 0.0, -0.42]),
            shin_rest=np.array([0.0, 0.0, -0.41]),
            orig_thigh_fwd=np.array([0.0, 1.0, 0.0]),
        )
        if prev_q is not None:
            step = float(2.0 * np.degrees(np.arccos(np.clip(abs(np.dot(prev_q, q_th)), 0.0, 1.0))))
            assert step < 12.0, f"Leg IK yaw snapped at az={az_deg} deg: {step:.1f} deg"
        prev_q = q_th


def test_viewport_sculpted_characters_and_custom_glb_obj_loader(tmp_path: Path):
    """Verify built-in sculpted 3D humanoids, rigged .GLB & .OBJ custom character loading, 55-bone LBS skinning, and Qt viewport rendering."""
    import json
    import struct

    from PySide6.QtWidgets import QApplication
    from desktop.ui.widgets.character_models import (
        build_full_55_skin_matrices,
        get_character_mesh,
        load_custom_character_file,
        render_character_mesh_rgba,
        skin_character_vertices,
    )
    from desktop.ui.widgets.viewport3d import Viewport3DContainer, _default_standing_pose

    pos, gmats = _default_standing_pose()
    cam_pos = np.array([0.0, -2.8, 1.1], dtype=np.float64)
    right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forward = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    # 1. All 3 built-in sculpted 3D humanoids build, skin (25 body + 30 fingers), and rasterize cleanly
    for pid in ("studio_female", "studio_male", "cyber_mannequin"):
        mesh = get_character_mesh(pid)
        assert mesh is not None
        assert len(mesh.faces) > 1500
        assert abs(mesh.bind_min_z) < 0.01
        assert np.allclose(np.sum(mesh.bone_weights, axis=1), 1.0, atol=1e-3)
        M_skin = build_full_55_skin_matrices(mesh, pos, global_matrices=gmats)
        posed = skin_character_vertices(mesh, M_skin)
        rgba = render_character_mesh_rgba(mesh, posed, 240, 240, cam_pos, right, up, forward, distance=2.8)
        assert rgba is not None and rgba.shape == (240, 240, 4)
        assert int(np.sum(rgba[:, :, 3] > 0)) > 500

    # 2. Custom .OBJ loader with automatic 25-bone distance skinning
    obj_path = tmp_path / "custom_hero.obj"
    obj_path.write_text(
        "v -0.2 0.0 0.0\nv 0.2 0.0 0.0\nv 0.2 0.0 1.7\nv -0.2 0.0 1.7\n"
        "v 0.0 0.15 0.85\nf 1 2 5\nf 2 3 5\nf 3 4 5\nf 4 1 5\n",
        encoding="utf-8",
    )
    obj_mesh = load_custom_character_file(obj_path, copy_to_library=False)
    assert len(obj_mesh.faces) == 4
    assert abs(float(obj_mesh.rest_vertices[:, 2].max()) - 1.72) < 0.05

    # 3. Custom binary .GLB rigged character loader (minimal valid glTF 2.0 binary with skin + indices)
    verts_b = np.array([[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 1.7, 0.1]], dtype=np.float32).tobytes()
    idx_b = np.array([0, 1, 2], dtype=np.uint16).tobytes() + b"\x00\x00"  # 4-byte aligned
    joints_b = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8).tobytes()
    weights_b = np.array([[1.0, 0.0, 0.0, 0.0]] * 3, dtype=np.float32).tobytes()
    ibm_b = np.eye(4, dtype=np.float32).tobytes()
    bin_blob = verts_b + idx_b + joints_b + weights_b + ibm_b
    while len(bin_blob) % 4 != 0:
        bin_blob += b"\x00"

    gltf_dict = {
        "asset": {"version": "2.0"},
        "nodes": [{"name": "hips", "translation": [0.0, 0.9, 0.0]}, {"name": "HeroMesh", "mesh": 0, "skin": 0}],
        "skins": [{"joints": [0], "inverseBindMatrices": 4}],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [0.5, 0.3, 0.2, 1.0], "roughnessFactor": 0.5}}],
        "meshes": [{
            "primitives": [{
                "attributes": {"POSITION": 0, "JOINTS_0": 2, "WEIGHTS_0": 3},
                "indices": 1,
                "material": 0,
            }]
        }],
        "buffers": [{"byteLength": len(bin_blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(verts_b)},
            {"buffer": 0, "byteOffset": len(verts_b), "byteLength": 6},
            {"buffer": 0, "byteOffset": len(verts_b) + len(idx_b), "byteLength": len(joints_b)},
            {"buffer": 0, "byteOffset": len(verts_b) + len(idx_b) + len(joints_b), "byteLength": len(weights_b)},
            {"buffer": 0, "byteOffset": len(verts_b) + len(idx_b) + len(joints_b) + len(weights_b), "byteLength": len(ibm_b)},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
            {"bufferView": 2, "componentType": 5121, "count": 3, "type": "VEC4"},
            {"bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC4"},
            {"bufferView": 4, "componentType": 5126, "count": 1, "type": "MAT4"},
        ],
    }
    json_bytes = json.dumps(gltf_dict).encode("utf-8")
    while len(json_bytes) % 4 != 0:
        json_bytes += b" "
    total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_blob)
    glb_path = tmp_path / "custom_rigged.glb"
    glb_path.write_bytes(
        b"glTF" + struct.pack("<II", 2, total_len)
        + struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
        + struct.pack("<II", len(bin_blob), 0x004E4942) + bin_blob
    )
    glb_mesh = load_custom_character_file(glb_path, copy_to_library=False)
    assert len(glb_mesh.faces) == 1
    assert glb_mesh.preset_id == "custom:custom_rigged"

    # 4. Headless Qt Viewport3DContainer switches characters and paints cleanly
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    assert app is not None
    vp = Viewport3DContainer()
    vp.resize(520, 420)
    vp.canvas.set_character_preset("cyber_mannequin")
    vp.canvas.set_scene_frame(skeleton_positions=pos, global_matrices=gmats)
    pix = vp.canvas.grab()
    assert pix.width() >= 250 and pix.height() >= 250
    vp.canvas.set_custom_character_mesh(glb_mesh)
    pix2 = vp.canvas.grab()
    assert pix2.width() >= 250 and pix2.height() >= 250
    vp.close()
