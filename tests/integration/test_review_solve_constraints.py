"""Cross-stage test: a disabled bad view cannot veto calibrated solving or overwrite pins."""

import numpy as np
import cv2
import pytest

from engine.math_core.camera_model import MultiCameraCalibration
from multi_video_studio.package_ingest import CameraVideoClip, MultiVideoPackage
from multi_video_studio.pipeline import CinemaPipelineConfig, MultiVideoCinemaPipeline
from multi_video_studio.pyav_sync import CameraSyncEstimate
from multi_video_studio.review_state import ReviewState
from multi_video_studio.wholebody_cuda import CameraTrackingResult
from tests.unit.test_math_calibration_triangulation import make_synthetic_ring_cameras
from tests.unit.test_skeleton_hybrid_cleanup import _make_synthetic_walk_sequence


@pytest.mark.parametrize("reviewed_geometry", [False, True])
def test_review_pins_and_excluded_camera_reach_the_actual_solver(tmp_path, monkeypatch, reviewed_geometry):
    cameras = make_synthetic_ring_cameras(num_cams=3)
    _, _, _, _, _, points = _make_synthetic_walk_sequence(T=20)
    pts = np.arange(20) / 30.
    package = MultiVideoPackage("review", tmp_path)
    tracks = {}
    state = ReviewState(alignment_accepted=True)
    for uid, camera in cameras.items():
        intrinsic = camera.intrinsics
        uv = camera.project_world_points(points.reshape(-1,3),True).reshape(20,26,2)
        observations = np.concatenate([uv,np.full((20,26,1),.95)],axis=2).astype(np.float32)
        path = tmp_path/(uid+".mp4")
        package.motion_clips[uid] = CameraVideoClip(uid,path,intrinsic.image_width,intrinsic.image_height,30.,20,20/30,False)
        tracks[uid] = CameraTrackingResult(uid,path,intrinsic.image_width,intrinsic.image_height,30.,pts.tolist(),observations,detection_rate=1.)
        state.sources[uid] = {"asset_id":uid,"width":intrinsic.image_width,"height":intrinsic.image_height}
        state.clocks[uid] = CameraSyncEstimate(uid,0.,"manual",1.,"test").__dict__
    uids = list(cameras)
    tracks[uids[2]].body26_uvc[...,0] += 400  # A visibly broken body track.
    tracks[uids[2]].pts_s=tracks[uids[2]].pts_s[:6]
    tracks[uids[2]].body26_uvc=tracks[uids[2]].body26_uvc[:6]
    tracks[uids[2]].detection_rate=0.
    state.camera_policy[uids[2]] = {"body":False,"hands":False}
    point = tracks[uids[0]].body26_uvc[6,9,:2] + [2.,1.]
    raw = tracks[uids[0]].body26_uvc.copy()
    state.edit_point(uids[0],pts[6],"body:9",point)
    if reviewed_geometry:
        state.calibration = MultiCameraCalibration("Reviewed geometry", cameras).to_dict()
        package.calibration_json_path = MultiCameraCalibration("Archived geometry", cameras).save(tmp_path / "archived.json")

    class RecordedTracker:
        def __init__(self,**kwargs):
            pass
        def track_videos_parallel(self,**kwargs):
            return {spec[0]:tracks[spec[0]] for spec in kwargs["camera_specs"]}
        def close(self):
            pass
    monkeypatch.setattr("multi_video_studio.pipeline.CinemaWholeBodyVideoTracker",RecordedTracker)
    config = CinemaPipelineConfig(tracking_cache_dir=None,review_state=state.to_dict(),enable_auto_sync=False,
        enable_fingers=False,enable_head_face=False,calibration_source="active_studio")
    result = MultiVideoCinemaPipeline(config).run_from_source(package,tmp_path/"exports",
        existing_calibration=MultiCameraCalibration("trusted",cameras))
    np.testing.assert_allclose(result.document.observations_2d[uids[0]]["keypoints_2d"][6,9,:2],point,atol=.002)
    np.testing.assert_array_equal(tracks[uids[0]].body26_uvc,raw)
    assert result.document.processing_config["review_state"]["camera_policy"][uids[2]]["body"] is False
    assert np.isfinite(result.document.corrected_motion.joint_positions).all()
    assert result.document.num_frames==20
    if reviewed_geometry:
        assert result.document.calibration["profile_name"] == "Reviewed geometry"
    assert uids[2] not in result.document.observations_2d
    assert uids[2] in result.document.processing_config["review_state"]["clocks"]
    from multi_video_studio.review_state import source_fingerprint
    first_path = package.motion_clips[uids[0]].path
    first_path.write_bytes(b"reviewed source")
    state.sources[uids[0]]["asset_id"] = source_fingerprint(first_path)
    config.review_state = state.to_dict()
    first_path.write_bytes(b"source replaced after review")
    with pytest.raises(ValueError,match="source video changed"):
        MultiVideoCinemaPipeline(config).run_from_source(package,tmp_path/"changed_source",
            existing_calibration=MultiCameraCalibration("trusted",cameras))


def test_rendered_wearable_tags_reconstruct_without_an_ai_tracker(tmp_path, monkeypatch):
    """Synthetic printed-tag footage checks detection through the actual solver."""
    from engine.math_core.camera_model import CalibratedCamera, CameraExtrinsics, CameraIntrinsics
    from multi_video_studio.review_observations import ReviewObservations
    package = MultiVideoPackage("synthetic_tags", tmp_path)
    state = ReviewState(capture_mode="markers", alignment_accepted=True)
    camera_models = {}
    count = 12
    body = np.array([[-.2,0,1.5],[.2,0,1.5],[-.5,0,1.3],[.5,0,1.3],[-.7,0,1.0],[.7,0,1.0],
        [-.1,0,.9],[.1,0,.9],[-.15,0,.5],[.15,0,.5],[-.15,0,.12],[.15,0,.12]])
    size = .09
    half = size/2
    # Tag local +Y is world +Z; its printed face points toward the cameras.
    square = np.array([[-half,0,half],[half,0,half],[half,0,-half],[-half,0,-half]])
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    printed = {i:cv2.aruco.generateImageMarker(dictionary,i,80) for i in range(5,17)}
    for index, x in enumerate((-.4,.4)):
        uid = f"tag_cam_{index}"
        position = np.array([x,-2.8,1.])
        z = np.array([0.,0.,1.])-position
        z /= np.linalg.norm(z)
        right = np.cross(z,[0,0,1.])
        right /= np.linalg.norm(right)
        down = np.cross(z,right)
        rotation = np.vstack([right,down,z])
        intrinsic = CameraIntrinsics(uid,960,960,np.array([[1100.,0,480],[0,1100.,480],[0,0,1]]),np.zeros(5))
        camera = CalibratedCamera(intrinsic,CameraExtrinsics(uid,rotation,-rotation@position))
        camera_models[uid] = camera
        path = tmp_path/(uid+".avi")
        writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"MJPG"),30.,(960,960))
        assert writer.isOpened()
        try:
            for frame in range(count):
                image = np.full((960,960),255,np.uint8)
                for tag_id, point in zip(range(5,17),body):
                    corners = camera.project_world_points(square+point+np.array([frame*.001,0,0]),True).astype(np.float32)
                    homography = cv2.getPerspectiveTransform(np.array([[0,0],[79,0],[79,79],[0,79]],np.float32),corners)
                    projected = cv2.warpPerspective(printed[tag_id],homography,(960,960),borderValue=255)
                    image = np.minimum(image,projected)
                writer.write(cv2.cvtColor(image,cv2.COLOR_GRAY2BGR))
        finally:
            writer.release()
        package.motion_clips[uid] = CameraVideoClip(uid,path,960,960,30.,count,count/30,False)
        state.sources[uid] = {"asset_id":uid,"width":960,"height":960}
        state.clocks[uid] = CameraSyncEstimate(uid,0.,"manual",1.,"synthetic timing").__dict__
    for tag_id in range(5,17):
        state.marker_assignments[str(tag_id)] = {"body_index":tag_id,"size_m":size,"joint_offset_tag_m":[0,0,0],
            "offset_calibrated":True,"offset_method":"synthetic measured offset"}
    def forbidden_tracker(*args, **kwargs):
        raise AssertionError("Marker-only capture constructed an AI tracker")
    monkeypatch.setattr("multi_video_studio.pipeline.CinemaWholeBodyVideoTracker",forbidden_tracker)
    state.hand_rois=[{"camera_uid":next(iter(camera_models)),"side":"left","start_s":0.,"end_s":1.,"rect":[0,0,100,100]}]
    monkeypatch.setattr("multi_video_studio.roi_tracking.refine_tracks_with_rois",forbidden_tracker)
    config = CinemaPipelineConfig(review_state=state.to_dict(),enable_auto_sync=False,enable_fingers=True,
        enable_head_face=False,calibration_source="active_studio",tracking_cache_dir=None)
    result = MultiVideoCinemaPipeline(config).run_from_source(package,tmp_path/"marker_exports",
        existing_calibration=MultiCameraCalibration("synthetic measured geometry",camera_models))
    report = result.document.processing_config["marker_constraints"]
    assert report["calibrated"] == 12 and report["constrained_samples"] >= count*10
    expected = body[None]+np.arange(count)[:,None,None]*np.array([.001,0,0])
    error = np.linalg.norm(result.document.joints_3d_raw[:,5:17]-expected,axis=-1)
    assert np.nanpercentile(error,95)<.02
    from pathlib import Path
    observations = ReviewObservations.load(Path(result.document.processing_config["review_observations_path"]))
    assert min(len(v["tags"]) for v in observations.cameras.values())==count
    # One view cannot silently switch marker-only operation back to AI.
    package.motion_clips.pop(next(iter(package.motion_clips)))
    with pytest.raises(ValueError,match="at least two calibrated cameras"):
        MultiVideoCinemaPipeline(config).run_from_source(package,tmp_path/"single_marker")
