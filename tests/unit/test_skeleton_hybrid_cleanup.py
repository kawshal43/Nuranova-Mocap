"""Synthetic ground-truth & animation validation tests for Skeleton Solver, Hybrid Cinema Mode, Motion Cleanup, Foot Locking, and BVH Export.

NOTE: All synthetic test datasets are explicitly labeled SYNTHETIC_GROUND_TRUTH.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from engine.export.bvh_exporter import export_bvh
from engine.marker_tracking.marker_kit import create_default_cinema_marker_kit
from engine.math_core.camera_model import MultiCameraCalibration
from engine.math_core.transforms import quat_identity
from engine.motion_cleanup.cleanup import (
    MotionCleanupConfig,
    clean_skeletal_motion,
    despike_and_fill_3d_points,
)
from engine.optimization.hybrid_engine import HybridCinemaReconstructor
from engine.skeleton.rig_definition import (
    BONE_INDEX,
    BONE_NAMES,
    DEFAULT_REST_OFFSETS_M,
    PARENT_INDICES,
    ActorProfile,
    calibrate_actor_from_body26_frames,
    compute_forward_kinematics_sequence,
)
from engine.skeleton.solver import ArticulatedSkeletonSolver, SolverConfig
from tests.unit.test_math_calibration_triangulation import make_synthetic_ring_cameras


def _make_synthetic_walk_sequence(T: int = 60, fps: float = 60.0):
    """SYNTHETIC_GROUND_TRUTH: Generate an articulated walking + arm-swinging sequence with planted stance phases."""
    B = len(BONE_NAMES)
    actor = ActorProfile(actor_id="synth_actor", name="Synthetic Actor", rest_offsets_m=DEFAULT_REST_OFFSETS_M.copy())
    root_t = np.zeros((T, 3), dtype=np.float64)
    local_q = np.tile(quat_identity(), (T, B, 1))

    for t in range(T):
        phase = 2.0 * np.pi * (t / T)
        root_t[t] = [0.0, 0.35 * (t / T), 0.96 + 0.015 * np.sin(2.0 * phase)]
        # Mild torso yaw
        local_q[t, BONE_INDEX["chest"]] = Rotation.from_euler("z", 8.0 * np.sin(phase), degrees=True).as_quat()
        # Arms swing & bend elbows naturally
        local_q[t, BONE_INDEX["left_upper_arm"]] = Rotation.from_euler("yz", [55.0, 20.0 * np.sin(phase)], degrees=True).as_quat()
        local_q[t, BONE_INDEX["left_forearm"]] = Rotation.from_euler("z", -40.0 - 15.0 * np.sin(phase), degrees=True).as_quat()
        local_q[t, BONE_INDEX["right_upper_arm"]] = Rotation.from_euler("yz", [-55.0, -20.0 * np.sin(phase)], degrees=True).as_quat()
        local_q[t, BONE_INDEX["right_forearm"]] = Rotation.from_euler("z", 40.0 + 15.0 * np.sin(phase), degrees=True).as_quat()
        # Legs walk cycle with knees bending anatomically backward (-X)
        l_hip_x = 18.0 * np.sin(phase)
        l_knee_x = -max(5.0, 35.0 * np.sin(phase))
        r_hip_x = -18.0 * np.sin(phase)
        r_knee_x = -max(5.0, -35.0 * np.sin(phase))
        local_q[t, BONE_INDEX["left_upper_leg"]] = Rotation.from_euler("x", l_hip_x, degrees=True).as_quat()
        local_q[t, BONE_INDEX["left_lower_leg"]] = Rotation.from_euler("x", l_knee_x, degrees=True).as_quat()
        local_q[t, BONE_INDEX["right_upper_leg"]] = Rotation.from_euler("x", r_hip_x, degrees=True).as_quat()
        local_q[t, BONE_INDEX["right_lower_leg"]] = Rotation.from_euler("x", r_knee_x, degrees=True).as_quat()

    fk_pos, fk_mats = compute_forward_kinematics_sequence(root_t, local_q, actor.rest_offsets_m)

    # Map FK joints into canonical Body26 positions
    body26 = np.zeros((T, 26, 3), dtype=np.float64)
    mapping = {
        19: "hips", 18: "neck", 17: "head_end",
        5: "left_upper_arm", 7: "left_forearm", 9: "left_hand",
        6: "right_upper_arm", 8: "right_forearm", 10: "right_hand",
        11: "left_upper_leg", 13: "left_lower_leg", 15: "left_foot", 20: "left_toe", 22: "left_toe", 24: "left_foot",
        12: "right_upper_leg", 14: "right_lower_leg", 16: "right_foot", 21: "right_toe", 23: "right_toe", 25: "right_foot",
    }
    for k26, bname in mapping.items():
        body26[:, k26] = fk_pos[:, BONE_INDEX[bname]]
    # Head face features
    head_pos = fk_pos[:, BONE_INDEX["head"]]
    body26[:, 0] = head_pos + np.array([0.0, 0.09, 0.02])   # nose
    body26[:, 3] = head_pos + np.array([-0.08, 0.0, 0.02])  # left_ear
    body26[:, 4] = head_pos + np.array([0.08, 0.0, 0.02])   # right_ear
    body26[:, 1] = head_pos + np.array([-0.03, 0.08, 0.04])
    body26[:, 2] = head_pos + np.array([0.03, 0.08, 0.04])

    return actor, root_t, local_q, fk_pos, fk_mats, body26


def test_actor_calibration_and_constant_bone_lengths():
    actor_gt, root_t, local_q, fk_pos, _, body26 = _make_synthetic_walk_sequence(T=40)
    conf = np.ones((40, 26), dtype=np.float32) * 0.95

    # Actor calibration accurately recovers limb lengths
    cal_actor = calibrate_actor_from_body26_frames("a1", "Test Actor", body26, conf)
    assert abs(cal_actor.bone_length("left_lower_leg") - actor_gt.bone_length("left_lower_leg")) < 0.015
    assert abs(cal_actor.bone_length("left_forearm") - actor_gt.bone_length("left_forearm")) < 0.015

    # Solve skeleton & verify invariant bone lengths across all frames
    solver = ArticulatedSkeletonSolver(actor_gt, SolverConfig(enforce_anatomical_limits=True))
    track = solver.solve_sequence(body26, conf)

    for b_idx in range(1, len(BONE_NAMES)):
        p_idx = PARENT_INDICES[b_idx]
        expected_len = float(np.linalg.norm(actor_gt.rest_offsets_m[b_idx]))
        actual_lens = np.linalg.norm(track.joint_positions[:, b_idx] - track.joint_positions[:, p_idx], axis=-1)
        assert np.max(np.abs(actual_lens - expected_len)) < 1e-4

    # Major joint position accuracy on synthetic walk
    eval_bones = [BONE_INDEX[n] for n in ("hips", "left_forearm", "right_forearm", "left_hand", "right_hand", "left_lower_leg", "right_lower_leg", "left_foot", "right_foot")]
    pos_err_mm = np.linalg.norm(track.joint_positions[:, eval_bones] - fk_pos[:, eval_bones], axis=-1) * 1000.0
    assert float(np.mean(pos_err_mm)) < 20.0  # Well below 20-30 mm engineering target


def test_hybrid_cinema_mode_and_bvh_export(tmp_path: Path):
    """SYNTHETIC_GROUND_TRUTH: Full 4-camera Hybrid Cinema reconstruction with calibrated surface markers."""
    cams = make_synthetic_ring_cameras(num_cams=4)
    calib = MultiCameraCalibration(profile_name="synth_4cam", cameras=cams)
    kit = create_default_cinema_marker_kit()
    actor, _, _, fk_pos, fk_mats, body26 = _make_synthetic_walk_sequence(T=30)

    # Register calibrated surface marker offsets onto actor (chest cluster + pelvis cluster)
    target_clusters = [c for c in kit.clusters if c.cluster_id in ("cl_chest", "cl_pelvis")]
    marker_names = []
    marker_world_seq = []
    for cl in target_clusters:
        b_idx = BONE_INDEX[cl.attached_bone]
        for mid, lpt in zip(cl.marker_ids, cl.local_points_m):
            off = np.array(lpt) + np.array(cl.cluster_to_bone_offset_m)
            actor.calibrated_marker_offsets_m[mid] = (cl.attached_bone, off.tolist())
            marker_names.append(mid)
            pts_t = np.einsum("tij,j->ti", fk_mats[:, b_idx, :3, :3], off) + fk_mats[:, b_idx, :3, 3]
            marker_world_seq.append(pts_t)
    markers_gt = np.stack(marker_world_seq, axis=1)  # (T, M, 3)

    # Project noisy AI 2D observations (sigma=3.0 px) and accurate physical marker observations (sigma=0.3 px)
    rng = np.random.default_rng(99)
    ai_obs_2d = {}
    marker_obs_2d = {}
    for uid, cam in cams.items():
        ai_arr = np.zeros((30, 26, 3), dtype=np.float32)
        m_arr = np.zeros((30, len(marker_names), 3), dtype=np.float32)
        for t in range(30):
            uv_ai = cam.project_world_points(body26[t], apply_distortion=True) + rng.normal(0.0, 2.5, size=(26, 2))
            ai_arr[t, :, :2] = uv_ai
            ai_arr[t, :, 2] = 0.85
            uv_m = cam.project_world_points(markers_gt[t], apply_distortion=True) + rng.normal(0.0, 0.3, size=(len(marker_names), 2))
            m_arr[t, :, :2] = uv_m
            m_arr[t, :, 2] = 0.98
        ai_obs_2d[uid] = ai_arr
        marker_obs_2d[uid] = m_arr

    reconstructor = HybridCinemaReconstructor(calibration=calib, actor=actor, marker_kit=kit)
    ts_ns = np.arange(30, dtype=np.int64) * int(1e9 / 60)

    doc, summary = reconstructor.reconstruct_from_2d_observations(
        project_id="p_synth",
        session_id="s_synth",
        take_id="take_hybrid_01",
        fps=60.0,
        frame_timestamps_ns=ts_ns,
        ai_obs_2d_per_cam=ai_obs_2d,
        marker_obs_2d_per_cam=marker_obs_2d,
        marker_names=marker_names,
        capture_mode="hybrid",
        profile_name="cinema",
    )
    assert doc.corrected_motion is not None
    assert summary.mean_reprojection_error_px < 4.0
    assert summary.mean_joint_confidence > 0.6

    # Export BVH and verify structure
    bvh_path = export_bvh(doc, tmp_path / "take_hybrid_01.bvh", use_corrected=True)
    text = bvh_path.read_text(encoding="utf-8")
    assert "HIERARCHY" in text and "ROOT hips" in text and "Frames: 30" in text


def test_motion_cleanup_gap_fill_and_foot_locking():
    """SYNTHETIC_GROUND_TRUTH: Verify gap interpolation, Butterworth smoothing, and planted-foot locking (<10mm drift)."""
    actor = ActorProfile(actor_id="fl_actor", name="FootLock Actor")
    T = 60
    fps = 60.0
    rng = np.random.default_rng(7)

    # Create a standing-then-swaying clip where left foot is planted on floor (z ~ 0.08m ankle height)
    # but has artificial 25 mm footskate jitter added to the raw track
    root_t = np.zeros((T, 3), dtype=np.float64)
    local_q = np.tile(quat_identity(), (T, len(BONE_NAMES), 1))
    for t in range(T):
        # Realistic standing sway + slow 22mm footskate drift (~0.14 m/s) + high-freq jitter
        root_t[t] = [0.024 * np.sin(2.0 * np.pi * t / T) + rng.normal(0, 0.002), 0.0, 0.955]
        local_q[t, BONE_INDEX["left_upper_leg"]] = quat_identity()
        local_q[t, BONE_INDEX["left_lower_leg"]] = Rotation.from_euler("x", -10.0, degrees=True).as_quat()
        local_q[t, BONE_INDEX["right_lower_leg"]] = Rotation.from_euler("x", -10.0, degrees=True).as_quat()

    fk_pos, fk_mats = compute_forward_kinematics_sequence(root_t, local_q, actor.rest_offsets_m)
    # Shift root Z so left ankle starts near floor contact height (0.08 m)
    min_ankle_z = float(np.min(fk_pos[:, BONE_INDEX["left_foot"], 2]))
    root_t[:, 2] -= (min_ankle_z - 0.08)
    fk_pos, fk_mats = compute_forward_kinematics_sequence(root_t, local_q, actor.rest_offsets_m)

    from engine.data_format.motion_io import SkeletalMotionTrack

    raw = SkeletalMotionTrack(
        root_translation=root_t.astype(np.float32),
        root_rotation_xyzw=local_q[:, 0].astype(np.float32),
        local_rotations_xyzw=local_q.astype(np.float32),
        global_matrices=fk_mats,
        joint_positions=fk_pos,
        joint_confidence=np.ones((T, len(BONE_NAMES)), dtype=np.float32),
    )

    raw_foot_xy = raw.joint_positions[10:50, BONE_INDEX["left_foot"], :2]
    raw_drift_mm = float(np.max(np.linalg.norm(raw_foot_xy - np.median(raw_foot_xy, axis=0), axis=-1))) * 1000.0
    assert raw_drift_mm > 15.0  # Confirm substantial footskate before cleanup

    cleaned = clean_skeletal_motion(
        raw,
        actor=actor,
        fps=fps,
        config=MotionCleanupConfig(strength=1.0, filter_type="butterworth", enable_foot_locking=True),
    )
    assert cleaned.foot_contacts is not None
    assert np.sum(cleaned.foot_contacts[:, 0]) >= 35

    # Verify planted foot horizontal drift reduced below 10 mm target
    clean_foot_xy = cleaned.joint_positions[15:45, BONE_INDEX["left_foot"], :2]
    clean_drift_mm = float(np.max(np.linalg.norm(clean_foot_xy - np.median(clean_foot_xy, axis=0), axis=-1))) * 1000.0
    assert clean_drift_mm < 10.0

    # Verify thigh & shin bone lengths remained invariant after IK foot locking!
    thigh_exp = float(np.linalg.norm(actor.rest_offsets_m[BONE_INDEX["left_lower_leg"]]))
    shin_exp = float(np.linalg.norm(actor.rest_offsets_m[BONE_INDEX["left_foot"]]))
    thigh_act = np.linalg.norm(
        cleaned.joint_positions[:, BONE_INDEX["left_lower_leg"]] - cleaned.joint_positions[:, BONE_INDEX["left_upper_leg"]],
        axis=-1,
    )
    shin_act = np.linalg.norm(
        cleaned.joint_positions[:, BONE_INDEX["left_foot"]] - cleaned.joint_positions[:, BONE_INDEX["left_lower_leg"]],
        axis=-1,
    )
    assert np.max(np.abs(thigh_act - thigh_exp)) < 1e-4
    assert np.max(np.abs(shin_act - shin_exp)) < 1e-4

    # Verify 3D outlier despike & short-gap interpolation
    pts = np.linspace([0, 0, 1], [1, 0, 1], 40)[:, None, :].astype(np.float32)
    confs = np.ones((40, 1), dtype=np.float32)
    pts[10:15, 0] = np.nan
    confs[10:15, 0] = 0.0
    pts[25, 0] = [0.6, 5.0, 1.0]  # Huge 5m spike!
    repaired, interp_mask = despike_and_fill_3d_points(pts, confs, max_gap_frames=8)
    assert np.all(np.isfinite(repaired[10:15, 0]))
    assert bool(np.all(interp_mask[10:15, 0]))
    assert abs(float(repaired[25, 0, 1])) < 0.05  # Spike removed!
