from copy import deepcopy
from fractions import Fraction
from types import SimpleNamespace

import av
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from engine.hands.hand_fusion import fit_hand_frame
from engine.hands.kinematics import default_hand_template, hand_forward_kinematics
from engine.math_core.camera_model import MultiCameraCalibration
from multi_video_studio.camera_refinement import refine_camera_from_points
from multi_video_studio.marker_constraints import apply_tag_constraints
from multi_video_studio.review_observations import ReviewObservations
from multi_video_studio.review_state import ReviewState
from multi_video_studio.review_video import ReviewVideoReader, hand_crop
from tests.unit.test_math_calibration_triangulation import make_synthetic_ring_cameras


def test_native_annotations_survive_retime_reorder_portable_history(tmp_path):
    clips = {}
    for uid, content in (("a", b"source A"), ("b", b"source B")):
        path = tmp_path / (uid + ".mp4")
        path.write_bytes(content)
        clips[uid] = SimpleNamespace(path=path, width=640, height=480)
    review = ReviewState()
    review.bind_sources(clips)
    review.edit_point("a", 2.03, "hand:left:8", [123., 234.])
    review.set_clocks({"a": {"offset_s": 4.5, "drift_slope": 1.001, "base_pts_s": .03}})
    expected_time = review.global_time("a", 2.03)
    assert review.local_time("a", expected_time) == pytest.approx(2.03)
    path = tmp_path / "review.json"
    review.save(path)
    reopened = ReviewState.load(path)
    reopened.bind_sources({"front": clips["a"], "side": clips["b"]})
    point = reopened.points[0]
    assert (point["camera_uid"], point["pts_s"], point["xy"]) == ("front", 2.03, [123., 234.])
    assert reopened.undo()  # Undo clocks; the preceding point edit still follows the source.
    assert reopened.points[0]["camera_uid"] == "front"
    assert reopened.redo()
    values = np.zeros((3, 21, 3), np.float32)
    edited, pins = reopened.apply_points("front", [2., 2.03, 2.07], values, "hand:left")
    np.testing.assert_array_equal(values, 0)
    np.testing.assert_allclose(edited[1, 8], [123, 234, 1])
    assert pins.sum() == 1


def test_missing_annotation_and_channel_exclusion_do_not_create_origin_measurement():
    review = ReviewState(sources={"a": {"asset_id": "test", "width": 640, "height": 480}})
    review.edit_point("a", 1., "body:9", valid=False)
    raw = np.full((2, 26, 3), 123., np.float32)
    raw[..., 2] = .8
    edited, pins = review.apply_points("a", [1., 1.04], raw)
    assert edited[0, 9, 2] == 0 and edited[0, 9, 0] == 123 and pins[0, 9]
    review.camera_policy = {"a": {"hands": False}}
    assert review.apply_exclusions("a", [1, 1.04], edited, "body")[1, 5, 2] > 0
    assert review.apply_exclusions("a", [1, 1.04], edited, "hands")[1, 5, 2] == 0


def test_replacing_a_video_cannot_reuse_the_old_slots_edits_or_geometry(tmp_path):
    clips = {}
    for uid in ("front","side"):
        path = tmp_path/(uid+".mp4")
        path.write_bytes(uid.encode())
        clips[uid] = SimpleNamespace(path=path,width=640,height=480)
    state = ReviewState()
    state.bind_sources(clips)
    for uid in clips:
        state.edit_point(uid,1.,"body:9",[50.,60.])
    state.clocks = {u:{"camera_uid":u,"offset_s":0.} for u in clips}
    state.calibration = {"cameras":{u:{"camera_uid":u} for u in clips}}
    state.alignment_accepted = True
    state.solved_revision = state.revision
    clips["front"].path.write_bytes(b"a different performance in the same camera slot")
    remap = state.bind_sources(clips)
    assert remap=={"side":"side"}
    assert all(p["camera_uid"]=="side" for p in state.points)
    assert list(state.calibration["cameras"])==["side"]
    assert list(state.clocks)==["side"] and state.stale and not state.alignment_accepted
    while state.undo():
        assert all(p["camera_uid"]!="front" for p in state.points)


def test_review_decoder_uses_actual_vfr_pts_and_has_bounded_cache(tmp_path):
    path = tmp_path / "vfr.mp4"
    container = av.open(str(path), "w")
    stream = container.add_stream("mpeg4", rate=30)
    stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
    stream.time_base = Fraction(1, 1000)
    times = [0, 33, 70, 135, 169, 220, 250, 310]
    for i, pts in enumerate(times):
        frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 25, np.uint8), format="bgr24")
        frame.pts, frame.time_base = pts, Fraction(1, 1000)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    reader = ReviewVideoReader(path, cache_frames=2)
    try:
        for seconds, expected in ((.077, .07), (.13, .135), (.02, .033), (.249, .25)):
            actual, image, dimensions = reader.at(seconds, max_width=None)
            assert actual == pytest.approx(expected, abs=.002)
            assert dimensions == (64, 48) and image.shape == (48, 64, 3)
        assert len(reader.cache) <= 2
        assert reader.at(3.) is None
    finally:
        reader.close()


def test_hand_crop_transform_does_not_mutate_world_or_source_points():
    points = np.c_[default_hand_template()[:, :2] * 500 + [200, 150], np.ones(21)]
    original = points.copy()
    image = np.zeros((240, 320, 3), np.uint8)
    crop, transform = hand_crop(image, points, (640, 480))
    assert crop.shape == (384, 384, 3)
    projected = np.c_[points[:, :2], np.ones(21)] @ transform.T
    inverse = np.linalg.inv(np.vstack([transform, [0, 0, 1]]))
    restored = np.c_[projected, np.ones(21)] @ inverse.T
    np.testing.assert_allclose(restored[:, :2], points[:, :2], atol=1e-8)
    np.testing.assert_array_equal(points, original)


def test_camera_refinement_keeps_trusted_world_and_rejects_bad_holdout():
    cameras = make_synthetic_ring_cameras(num_cams=3)
    calibration = MultiCameraCalibration("test", cameras)
    uid = list(cameras)[1]
    rng = np.random.default_rng(17)
    xyz = rng.uniform([-.5, -.5, .4], [.5, .5, 1.7], (60, 3))
    measured = cameras[uid].project_world_points(xyz, True)
    before = deepcopy(calibration.to_dict())
    calibration.cameras[uid].extrinsics.t_cw += [.04, -.08, .03]
    fixed, report = refine_camera_from_points(calibration, uid, xyz, measured)
    assert report["after_median_px"] < .01
    for other in cameras:
        if other != uid:
            assert fixed.to_dict()["cameras"][other] == before["cameras"][other]
    assert calibration.to_dict()["cameras"][uid] != fixed.to_dict()["cameras"][uid]
    bad = measured.copy()
    bad[np.arange(60) % 5 == 0] += 100
    with pytest.raises(ValueError, match="held-out"):
        refine_camera_from_points(calibration, uid, xyz, bad)


def test_partial_hand_fit_has_fixed_lengths_and_does_not_fit_nan_tip():
    template = default_hand_template("right")
    q = np.tile([0., 0., 0., 1.], (16, 1))
    q[4:7] = Rotation.from_euler("x", [-.6, -.8, -.5]).as_quat()
    target = hand_forward_kinematics(q, template=template)
    target[20] = np.nan
    confidence = np.ones(21)
    confidence[20] = 0
    fitted = fit_hand_frame(target, confidence, "right", template, initial=q)
    actual = hand_forward_kinematics(fitted, template=template)
    assert np.isfinite(actual).all()
    assert np.linalg.norm(actual[8] - target[8]) < .004
    for start in (1, 5, 9, 13, 17):
        np.testing.assert_allclose(np.linalg.norm(np.diff(actual[start:start + 4], axis=0), axis=1),
                                   np.linalg.norm(np.diff(template[start:start + 4], axis=0), axis=1), atol=1e-7)


def test_calibrated_tag_offsets_constrain_joint_and_center_only_does_not():
    cameras = make_synthetic_ring_cameras(num_cams=3)
    calibration = MultiCameraCalibration("tags", cameras)
    square = np.array([[-.03, .03, 0], [.03, .03, 0], [.03, -.03, 0], [-.03, -.03, 0]])
    rotation = Rotation.from_euler("xyz", [.4, -.2, .3]).as_matrix()
    center, offset = np.array([.1, .2, 1.1]), np.array([.02, -.01, .015])
    corners = square @ rotation.T + center
    aligned, body = {}, {}
    for uid, camera in cameras.items():
        uv = camera.project_world_points(corners, True)
        aligned[uid] = np.tile(np.c_[uv, np.ones(4)][None], (2, 1, 1))
        body[uid] = np.zeros((2, 26, 3), np.float32)
    state = ReviewState(marker_assignments={"12": {"size_m": .06, "body_index": 9}})
    mask, report = apply_tag_constraints(calibration, {"tag:12": aligned}, state, body, [0, .033])
    assert not mask.any() and report["warnings"]
    state.marker_assignments["12"].update(joint_offset_tag_m=offset.tolist(), offset_calibrated=True)
    mask, report = apply_tag_constraints(calibration, {"tag:12": aligned}, state, body, [0, .033])
    assert mask[:, 9].all() and report["constrained_samples"] == 2
    for uid, camera in cameras.items():
        np.testing.assert_allclose(body[uid][0, 9, :2], camera.project_world_points((rotation @ offset + center)[None], True)[0], atol=.01)


def test_native_review_arrays_and_tag_geometry_roundtrip(tmp_path):
    observations = ReviewObservations()
    body = np.zeros((26, 3), np.float32)
    tags = [{"id": 1, "corners": [[1, 1], [5, 1], [5, 5], [1, 5]]}]
    observations.put_frame("front", .07, body, tags=tags)
    observations.put_frame("front", .03, body + 1)
    path = tmp_path / "observations.npz"
    observations.save(path)
    loaded = ReviewObservations.load(path)
    assert loaded.cameras["front"]["tags"]["0.070000"] == tags
    np.testing.assert_allclose(loaded.frame("front", .03)["body"], body + 1)


def test_world_palm_roll_fit_preserves_elbow_wrist_and_handles_pi_boundary():
    from engine.data_format.motion_io import HandMotionTrack, SkeletalMotionTrack
    from engine.hands.hand_fusion import apply_observed_wrist_orientation
    from engine.skeleton.rig_definition import ActorProfile, BONE_INDEX, compute_forward_kinematics_sequence
    actor = ActorProfile("qa", "qa")
    count = 60
    root = np.tile([0, 0, .96], (count, 1))
    local = np.tile([0., 0., 0., 1.], (count, 25, 1))
    positions, matrices = compute_forward_kinematics_sequence(root, local, actor.rest_offsets_m)
    body = SkeletalMotionTrack(root, local[:, 0], local, matrices, positions, np.ones((count, 25)))
    palm_rest = np.column_stack([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    # A continuous rotation through 180 degrees must not produce a clipped +/-pi jump.
    rotation = Rotation.from_euler("x", np.linspace(150, 200, count), degrees=True)
    palms = Rotation.from_matrix(rotation.as_matrix() @ palm_rest).as_quat()
    palms[::2] *= -1
    hand = HandMotionTrack(right_palm_world_xyzw=palms, right_confidence=np.ones((count, 21)))
    fixed = apply_observed_wrist_orientation(body, hand, actor, 30.)
    ids = [BONE_INDEX[n] for n in ("right_upper_arm", "right_forearm", "right_hand")]
    np.testing.assert_allclose(fixed.joint_positions[:, ids], body.joint_positions[:, ids], atol=1e-6)
    q = fixed.local_rotations_xyzw[:, BONE_INDEX["right_hand"]]
    steps = 2 * np.arccos(np.clip(abs(np.sum(q[1:] * q[:-1], axis=1)), 0, 1))
    assert np.degrees(steps).max() < 13
    np.testing.assert_array_equal(body.local_rotations_xyzw, local)


def test_hand_roi_follows_corrected_native_wrist_without_changing_observations():
    from multi_video_studio.roi_tracking import roi_for_frame, crop_override
    state = ReviewState(hand_rois=[{"camera_uid":"front", "side":"left", "start_s":1., "end_s":2.,
        "rect":[100.,80.,80.,60.], "wrist_offset":[10.,-5.]}])
    body = np.zeros((26,3))
    body[9] = [200.,150.,.9]
    original = body.copy()
    roi = roi_for_frame(state, "front", "left", 1.5, body)
    np.testing.assert_allclose(roi, [170.,115.,80.,60.])
    assert roi_for_frame(state, "front", "right", 1.5, body) is None
    assert roi_for_frame(state, "front", "left", 3., body) is None
    crop, _, native = crop_override(np.zeros((480,640,3),np.uint8), roi)
    assert crop.shape == (224,224,3) and native[2] == native[3]
    np.testing.assert_array_equal(body, original)
