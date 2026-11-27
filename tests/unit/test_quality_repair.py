"""Temporal quality regressions beyond file validity and synthetic reprojection grades."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from engine.motion_cleanup.quality_audit import audit_quaternions
from engine.motion_cleanup.rotation_filter import smooth_rotations, stabilize_upper_body
from multi_video_studio.audio_alignment import fit_clock_consensus
from multi_video_studio.tracking_cache import (
    load_tracking_cache,
    save_tracking_cache,
    tracking_cache_key,
)
from multi_video_studio.wholebody_cuda import CameraTrackingResult


def test_camera_validation_rejects_mismatched_or_missing_third_view():
    from engine.math_core.camera_model import MultiCameraCalibration
    from multi_video_studio.video_calibrator import validate_calibration_against_observations
    from tests.unit.test_anti_flicker_overhaul import _make_dist_cam
    from tests.unit.test_skeleton_hybrid_cleanup import _make_synthetic_walk_sequence

    _, _, _, _, _, points = _make_synthetic_walk_sequence(T=20)
    cameras = {
        uid: _make_dist_cam(uid, np.array(position), np.array([0.0, 0.0, 1.0]), k1=0)
        for uid, position in [
            ("c1", [3.0, 0.0, 1.5]),
            ("c2", [0.0, 3.0, 1.5]),
            ("c3", [-3.0, 0.0, 1.5]),
        ]
    }
    observations = {}
    for uid, camera in cameras.items():
        uv = camera.project_world_points(points.reshape(-1, 3), True).reshape(20, 26, 2)
        observations[uid] = np.concatenate([uv, np.full((20, 26, 1), 0.9)], axis=-1)
    sizes = {uid: (1920, 1080) for uid in cameras}
    calibration = MultiCameraCalibration("test", cameras)
    okay, reason = validate_calibration_against_observations(calibration, observations, sizes)
    assert okay, reason
    observations["c3"][:, :, 1] += 250
    okay, reason = validate_calibration_against_observations(calibration, observations, sizes)
    assert not okay and "c3" in reason
    del calibration.cameras["c3"]
    okay, reason = validate_calibration_against_observations(calibration, observations, sizes)
    assert not okay and "missing cameras: c3" in reason


@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("bend_deg", [2, 10, 35, 95, 150])
def test_limb_twist_stabilization_preserves_measured_endpoints(side, bend_deg):
    from engine.skeleton.solver import solve_two_bone_chain_rotations

    upper_rest = np.array([side * 0.29, 0.0, 0.0])
    lower_rest = np.array([side * 0.26, 0.0, 0.0])
    upper_dir = np.array([side, 0.0, 0.0])
    previous = np.eye(3)
    # Abrupt changes of bend-plane azimuth exposed a 40cm wrist error in the
    # capture. Twist regularization must never change the observed directions.
    for azimuth in [0, 135, -60, 179, -179, 45]:
        pole = Rotation.from_rotvec(upper_dir * np.radians(azimuth)).apply([0, 1, 0])
        direction = np.cos(np.radians(bend_deg)) * upper_dir + np.sin(np.radians(bend_deg)) * pole
        middle = upper_rest.copy()
        end = middle + np.linalg.norm(lower_rest) * direction
        upper, lower, _ = solve_two_bone_chain_rotations(
            np.zeros(3),
            middle,
            end,
            np.eye(3),
            upper_rest,
            lower_rest,
            np.array([0, 1, 0]),
            prev_upper_R=previous,
        )
        np.testing.assert_allclose(upper @ upper_rest, middle, atol=1e-7)
        np.testing.assert_allclose(middle + lower @ lower_rest, end, atol=1e-7)
        np.testing.assert_allclose(upper.T @ upper, np.eye(3), atol=1e-7)
        previous = upper


def test_rotation_filter_preserves_full_turn_and_ignores_quaternion_sign():
    angles = np.linspace(0, 360, 181)
    q = Rotation.from_euler("z", angles, degrees=True).as_quat()[:, None]
    signed = q.copy()
    signed[::3] *= -1
    out = smooth_rotations(signed, 30, 4)
    error = (Rotation.from_quat(out[:, 0]).inv() * Rotation.from_quat(q[:, 0])).magnitude()
    assert np.max(np.degrees(error[8:-8])) < 0.1
    assert np.max(np.degrees(error)) < 3
    np.testing.assert_allclose(
        np.abs(np.sum(smooth_rotations(q, 30, 4) * out, axis=-1)), 1, atol=1e-6
    )
    assert audit_quaternions(out, ["turn"], 30)["steps_over_30"] == 0


def test_rotation_filter_rejects_isolated_spike_without_freezing_motion():
    angles = np.linspace(-40, 60, 90)
    angles[45] += 110
    q = Rotation.from_euler("x", angles, degrees=True).as_quat()[:, None]
    confidence = np.ones((90, 1))
    confidence[45] = 0.1
    out = smooth_rotations(q, 30, 3, confidence)
    clean = np.degrees(Rotation.from_quat(out[:, 0]).as_rotvec()[:, 0])
    assert abs(clean[45] - np.linspace(-40, 60, 90)[45]) < 4
    assert clean[-1] - clean[0] > 95
    assert audit_quaternions(out, ["elbow"], 30)["max_step_deg"] < 8


def test_upper_body_filter_keeps_planted_leg_chain_and_raw_track_unchanged():
    from tests.unit.test_skeleton_hybrid_cleanup import _make_synthetic_walk_sequence
    from engine.skeleton.solver import ArticulatedSkeletonSolver
    from engine.skeleton.rig_definition import BONE_INDEX

    actor, _, _, _, _, points = _make_synthetic_walk_sequence(T=30)
    track = ArticulatedSkeletonSolver(actor).solve_sequence(points, np.full((30, 26), 0.9))
    original_q = track.local_rotations_xyzw.copy()
    out = stabilize_upper_body(track, actor, 30)
    for name in ("left_foot", "right_foot", "left_toe", "right_toe"):
        np.testing.assert_allclose(
            out.joint_positions[:, BONE_INDEX[name]],
            track.joint_positions[:, BONE_INDEX[name]],
            atol=1e-6,
        )
    np.testing.assert_array_equal(track.local_rotations_xyzw, original_q)


def test_clock_drift_requires_consensus_and_rejects_false_transients():
    t = np.linspace(5, 85, 15)
    offset = 5.77 + 0.0006 * (t - 45)
    offset[[2, 7, 12]] += [0.12, -0.15, 0.09]
    off, slope, pivot, info = fit_clock_consensus(t, offset, np.full(15, 25), 5.77)
    assert abs((off + (slope - 1) * (45 - pivot)) - 5.77) < 0.001
    assert abs(slope - 1.0006) < 1e-5
    assert info["inliers"] == 12
    assert info["drift_verified"]
    _, slope, _, info = fit_clock_consensus([0, 50], [0, 0.05], [100, 100], 0)
    assert slope == 1 and not info.get("drift_verified", False)


def test_clock_does_not_invent_drift_from_small_acoustic_changes():
    times = np.linspace(0, 80, 15)
    offsets = 2 + times * 0.00006
    _, slope, _, info = fit_clock_consensus(times, offsets, np.full(15, 20), 2)
    assert slope == 1
    assert not info["drift_verified"]


def test_tracking_cache_roundtrip_preserves_hand_and_face_observations(tmp_path: Path):
    from engine.face.face_tracker import FaceFrameResult
    from engine.hands.hand_tracker import HandObservation

    q = np.tile([0, 0, 0, 1], (16, 1)).astype(np.float32)
    hand = HandObservation("left", np.ones((21, 3), np.float32), 0.8, q)
    face = FaceFrameResult(True, np.ones(52), np.array([0, 0, 0, 1]), np.zeros(4), confidence=0.7)
    original = CameraTrackingResult(
        "cam",
        tmp_path / "source.mp4",
        320,
        240,
        30,
        [0, 0.1],
        np.ones((2, 26, 3)),
        wrist_crop_hands=[{"left": hand}, {}],
        head_crop_faces=[face, None],
    )
    cache = tmp_path / "cache.npz"
    save_tracking_cache(cache, original)
    loaded = load_tracking_cache(cache)
    np.testing.assert_array_equal(loaded.body26_uvc, original.body26_uvc)
    np.testing.assert_array_equal(loaded.wrist_crop_hands[0]["left"].local_rotations_xyzw, q)
    assert loaded.head_crop_faces[0].confidence == 0.7
    assert loaded.head_crop_faces[1] is None
    source = tmp_path / "source.mp4"
    source.write_bytes(b"first")
    first = tracking_cache_key(source, {"stride": 1})
    source.write_bytes(b"second video")
    assert first != tracking_cache_key(source, {"stride": 1})


def test_rotation_filter_rejects_invalid_values():
    with pytest.raises(ValueError):
        smooth_rotations(np.zeros((3, 1, 4)), 30)
