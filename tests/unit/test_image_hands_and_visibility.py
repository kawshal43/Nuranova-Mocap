"""Known image poses and calibrated sightlines, independent of detector guesses."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from engine.hands.hand_fusion import _parameters_to_quaternions, select_crop_views
from engine.hands.hand_tracker import HandObservation
from engine.hands.image_fit import fit_hand_image, _points
from engine.hands.kinematics import default_hand_template, hand_forward_kinematics
from multi_video_studio.limb_visibility import (_segment_distance, visibility_weights,
    weight_body_observations, protect_manual_views)
from tests.unit.test_math_calibration_triangulation import make_synthetic_ring_cameras


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("source", ["single_view", "rtmw3d_estimate"])
def test_image_fit_recovers_visible_curl_without_changing_observations(side, source):
    template = default_hand_template(side)
    target = np.r_[np.tile([.85, 1.1, .72], 5), np.zeros(5)]
    initial = target.copy()
    initial[:15] *= .30
    q = _parameters_to_quaternions(initial)
    palm = Rotation.from_euler("x", 85, degrees=True)
    q[0] = palm.as_quat()
    points = hand_forward_kinematics(q, side, template, palm.as_matrix())
    expected = hand_forward_kinematics(_parameters_to_quaternions(target), side, template, palm.as_matrix())
    projection = np.array([[-1 if source.startswith("rtmw3d") else 1, 0, 0], [0, 0, -1]])
    uv = expected @ projection.T * 1000 + [320, 240]
    obs = HandObservation(side, points.copy(), .9, q.copy(), np.full(21, .9), uv.copy(), source)
    result, report = fit_hand_image(obs, template)
    assert report["accepted"], report
    assert report["after_px"] < report["before_px"] * .25, report
    actual = hand_forward_kinematics(result.local_rotations_xyzw, side, template,
        Rotation.from_quat(result.local_rotations_xyzw[0]).as_matrix())
    assert np.sqrt(np.mean((actual @ projection.T * 1000 + [320, 240] - uv)**2)) < 5
    np.testing.assert_array_equal(obs.landmarks_2d, uv)
    np.testing.assert_array_equal(obs.local_rotations_xyzw, q)
    np.testing.assert_array_equal(result.landmark_confidence, obs.landmark_confidence)
    assert result.confidence == obs.confidence
    for start in (1, 5, 9, 13, 17):
        np.testing.assert_allclose(np.linalg.norm(np.diff(actual[start:start+4], axis=0), axis=1),
            np.linalg.norm(np.diff(template[start:start+4], axis=0), axis=1), atol=1e-8)


def test_tiny_missing_and_interpolated_hands_keep_the_prior():
    q = np.tile([0., 0., 0., 1.], (16, 1))
    points = default_hand_template()
    obs = HandObservation("right", points, .9, q, landmarks_2d=points[:, :2]*50 + 100)
    assert fit_hand_image(obs)[0] is obs
    assert fit_hand_image(replace(obs, landmarks_2d=None))[1]["accepted"] is False
    assert fit_hand_image(replace(obs, source="interpolated"))[1]["accepted"] is False
    bad = obs.landmarks_2d.copy()
    bad[[0, 5, 9, 17]] = np.nan
    assert fit_hand_image(replace(obs, landmarks_2d=bad))[0].source == obs.source


def test_batched_hand_kinematics_matches_existing_fixed_length_solver():
    parameters = np.r_[np.linspace(.05, 1., 15), [.3, -.2, .1, .1, -.1]]
    for side in ("left", "right"):
        template = default_hand_template(side)
        np.testing.assert_allclose(_points(parameters, template),
            hand_forward_kinematics(_parameters_to_quaternions(parameters), side, template), atol=1e-10)


def test_segment_visibility_handles_parallel_and_endpoint_cases():
    distance, fraction = _segment_distance(np.array([0., 0, 0]), np.array([0., 0, 4]),
        np.array([-.1, 0, 2]), np.array([.1, 0, 2]))
    assert distance == pytest.approx(0)
    assert fraction == pytest.approx(.5)
    distance, _ = _segment_distance(np.array([0., 0, 0]), np.array([0., 0, 4]),
        np.array([1., 0, 2]), np.array([1., 0, 3]))
    assert distance == pytest.approx(1)


def test_hidden_wrist_loses_weight_only_in_the_obstructed_camera():
    cameras = make_synthetic_ring_cameras(num_cams=3)
    uid, camera = next(iter(cameras.items()))
    center = camera.extrinsics.camera_center_world
    points = np.zeros((8, 26, 3))
    points[:] = [0, 0, 1.]
    toward = center - [0, 0, 1.]
    toward /= np.linalg.norm(toward)
    points[:, 19] = [0, 0, .7]
    points[:, 18] = [0, 0, 1.3]
    points[:, 5], points[:, 6] = [-.19, 0, 1.3], [.19, 0, 1.3]
    points[:, 9] = np.array([0, 0, 1.]) - toward*.4
    confidence = np.zeros((8, 26))
    confidence[:, [19, 18, 5, 6, 9]] = .9
    weights = visibility_weights(cameras, points, confidence)
    assert weights[uid][:, 9].max() < .4
    assert max(w[:, 9].min() for u,w in weights.items() if u != uid) > .9
    observations = {u: np.dstack([np.zeros((8, 26, 2)), np.full((8, 26), .9)]) for u in cameras}
    corrected = weight_body_observations(observations, weights)
    np.testing.assert_array_equal(observations[uid][..., 2], .9)
    np.testing.assert_array_equal(corrected[uid][..., :2], observations[uid][..., :2])
    confidence[:, [18, 19]] = 0
    assert visibility_weights(cameras, points, confidence)[uid][:, 9].min() == 1


def test_hidden_crop_cannot_win_using_detector_confidence_alone():
    q = np.tile([0., 0., 0., 1.], (16, 1))
    hidden = HandObservation("right", default_hand_template(), .95, q)
    visible = replace(hidden, confidence=.7)
    weights = {"a":np.ones((3, 26)), "b":np.ones((3, 26))}
    weights["a"][:, 10] = .25
    chosen = select_crop_views({"a":[{"right":hidden}]*3, "b":[{"right":visible}]*3}, 3, weights)
    assert all(frame["right"].confidence == .7 for frame in chosen)
    assert hidden.confidence == .95


def test_manual_point_overrides_estimated_occlusion_at_its_native_time():
    from multi_video_studio.review_state import ReviewState
    state = ReviewState(sources={"a":{"asset_id":"a","width":640,"height":480}})
    state.edit_point("a", .1, "body:9", [200, 150])
    state.edit_point("a", .2, "hand:right:8", [220, 130])
    state.set_clocks({"a":{"offset_s":2.,"drift_slope":1.,"base_pts_s":0.}})
    weights = {"a":np.full((4,26), .25)}
    protect_manual_views(weights, state, np.arange(4)*.1+2.)
    assert weights["a"][1,9] == 1 and weights["a"][2,10] == 1
    assert weights["a"][0,9] == .25 and weights["a"][3,10] == .25


def test_hybrid_marker_constraints_override_estimated_body_occlusion():
    from multi_video_studio.review_state import ReviewState
    weights = {uid:np.full((4,26),.25) for uid in ("a","b")}
    measured = np.zeros((4,26),bool)
    measured[1,9] = True
    protect_manual_views(weights,ReviewState(),np.arange(4)*.1,measured)
    assert all(value[1,9] == 1 for value in weights.values())
    assert all(value[1,10] == .25 for value in weights.values())


def test_visibility_weights_improve_a_calibrated_solve_with_a_hidden_wrong_wrist():
    from multi_video_studio.multiview_3d_optimizer import optimize_multiview_3d_global
    from tests.unit.test_skeleton_hybrid_cleanup import _make_synthetic_walk_sequence
    cameras = make_synthetic_ring_cameras(num_cams=3)
    points = np.repeat(_make_synthetic_walk_sequence(T=16)[-1][:1],16,axis=0)
    for left, right, position in ((5,6,[-.18,0,1.5]), (7,8,[-.30,.15,1.4]), (9,10,[-.32,0,1.15])):
        points[:,left] = position
        points[:,right] = np.asarray(position)*[-1,1,1]
    confidence = np.full((16,26), .9)
    observations = {u:np.dstack([c.project_world_points(points.reshape(-1,3), True).reshape(16,26,2),
                                  np.full((16,26), .9)]) for u,c in cameras.items()}
    hidden = next(iter(cameras))
    observations[hidden][:,9,:2] += [30,25]
    weights = visibility_weights(cameras, points, confidence)
    before = optimize_multiview_3d_global(cameras, observations, points, confidence, 30., iterations=4)[0]
    after = optimize_multiview_3d_global(cameras, weight_body_observations(observations,weights),
                                      points, confidence, 30., iterations=4)[0]
    old_error = np.mean(np.linalg.norm(before[:,9]-points[:,9],axis=1))
    new_error = np.mean(np.linalg.norm(after[:,9]-points[:,9],axis=1))
    assert new_error < old_error*.8, (old_error,new_error)
