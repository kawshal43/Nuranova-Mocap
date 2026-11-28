"""Cleanup must distinguish coherent fast gestures from isolated rotation spikes."""
import numpy as np
from scipy.spatial.transform import Rotation

from engine.data_format.motion_io import SkeletalMotionTrack
from engine.skeleton.rig_definition import ActorProfile, DEFAULT_REST_OFFSETS_M, BONE_INDEX, compute_forward_kinematics_sequence
from multi_video_studio.zero_slide_physics import apply_zero_foot_slide_physics


def _clean(angles, final_filter=False):
    count = len(angles)
    actor = ActorProfile('fast', 'Fast movement', rest_offsets_m=DEFAULT_REST_OFFSETS_M.copy())
    root = np.tile([0., 0., .96], (count, 1))
    rotations = np.tile([0., 0., 0., 1.], (count, 25, 1))
    bone = BONE_INDEX['left_forearm']
    rotations[:, bone] = Rotation.from_euler('z', angles, degrees=True).as_quat()
    positions, matrices = compute_forward_kinematics_sequence(root, rotations, actor.rest_offsets_m)
    track = SkeletalMotionTrack(root, rotations[:, 0], rotations, matrices, positions,
        np.full((count, 25), .95), rotation_observable=np.ones((count, 25), bool))
    cleaned, *_ = apply_zero_foot_slide_physics(track, actor, 30., contacts_override=np.zeros((count, 4)),
        allow_floor_tilt=False)
    if final_filter:
        from engine.motion_cleanup.rotation_filter import stabilize_upper_body
        cleaned = stabilize_upper_body(cleaned, actor, 30.)
    return Rotation.from_quat(cleaned.local_rotations_xyzw[:, bone]).magnitude()*180/np.pi


def test_coherent_fast_gesture_is_not_erased_as_a_glitch():
    angles = np.zeros(90)
    angles[30:35] = np.linspace(0., 100., 5)
    angles[35:39] = np.linspace(75., 0., 4)
    result = _clean(angles)
    assert result[34] > 85., f'Observed fast gesture was flattened to {result[34]:.1f} degrees'
    assert abs(int(np.argmax(result))-34) <= 1


def test_isolated_rotation_spike_is_still_rejected():
    angles = np.zeros(90)
    angles[34] = 120.
    result = _clean(angles)
    assert result.max() < 20., f'Isolated tracking spike survived cleanup: {result.max():.1f} degrees'


def test_final_upper_body_filter_keeps_supported_fast_gesture():
    angles = np.zeros(90)
    angles[30:35] = np.linspace(0., 100., 5)
    angles[35:39] = np.linspace(75., 0., 4)
    result = _clean(angles, final_filter=True)
    assert result[34] > 85., f'Final filter flattened the recovered gesture to {result[34]:.1f} degrees'
