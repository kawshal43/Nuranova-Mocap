"""Single-camera coordinate, persistence, correction and uncertainty regressions."""
from pathlib import Path

import numpy as np
import pytest

from engine.ai_tracking.monocular_geometry import CAMERA_TO_STUDIO, unproject_native, map_coco_xyz_to_body26, fit_monocular_sequence
from engine.data_format.motion_io import load_motion_file
from multi_video_studio.package_ingest import MultiVideoPackage, CameraVideoClip
from multi_video_studio.pipeline import CinemaPipelineConfig, MultiVideoCinemaPipeline
from multi_video_studio.review_state import ReviewState
from multi_video_studio.review_observations import ReviewObservations
from multi_video_studio.wholebody_cuda import CameraTrackingResult
from tests.unit.test_skeleton_hybrid_cleanup import _make_synthetic_walk_sequence


def observations(count=50):
    *_, body = _make_synthetic_walk_sequence(T=count)
    relative = body - body[:, 19:20]
    roots = np.c_[np.linspace(0, .25, count), -.1-.32*np.sin(np.linspace(0, np.pi, count))**4,
                  np.full(count, 4.)]
    camera = relative @ CAMERA_TO_STUDIO + roots[:, None]
    native = np.zeros((count, 26, 3), np.float32)
    native[..., :2] = camera[..., :2] / camera[..., 2:3] * 1400 + [960, 540]
    native[..., 2] = .95
    return relative.astype(np.float32), native, 20 + np.arange(count)/30


def test_rtmw_depth_units_and_axes_roundtrip():
    camera = np.array([[.2, -.3, 4.1], [-.4, .5, 3.8]])
    uv = camera[:, :2]/camera[:, 2:3]*1400 + [960, 540]
    studio = unproject_native(uv, camera[:, 2]-4., 4., 1920, 1080, 1400)
    np.testing.assert_allclose(studio @ CAMERA_TO_STUDIO, camera, atol=1e-7)
    assert np.linalg.det(CAMERA_TO_STUDIO) == pytest.approx(1)
    assert studio[0, 0] < 0 and studio[0, 2] > 0
    xyz = np.arange(133*3).reshape(133, 3)
    mapped = map_coco_xyz_to_body26(xyz)
    np.testing.assert_equal(mapped[9], xyz[9])
    np.testing.assert_equal(mapped[10], xyz[10])
    np.testing.assert_equal(mapped[24], xyz[19])


def test_sequence_floor_keeps_jump_scale_and_in_place():
    relative, uv, times = observations(80)
    fitted, _, metadata = fit_monocular_sequence(relative, uv, times, 1920, 1080, 1.72, focal_px=1400)
    still, _, _ = fit_monocular_sequence(relative, uv, times, 1920, 1080, 1.72, "in_place", 1400)
    assert np.ptp(fitted[:, 19, 0]) > .2
    assert np.ptp(still[:, 19, 0]) < 1e-5
    assert np.ptp(fitted[:, 19, 2]) > .25
    # A single floor offset preserves flight; per-frame foot grounding would erase it.
    assert np.ptp(np.min(fitted[:, [20, 21], 2], axis=1)) > .2
    taller, _, _ = fit_monocular_sequence(relative, uv, times, 1920, 1080, 2.1, focal_px=1400)
    def thigh(x):
        return np.median(np.linalg.norm(x[:, 11]-x[:, 13], axis=-1))
    assert thigh(taller)/thigh(fitted) == pytest.approx(2.1/1.72, rel=.002)
    assert metadata["trajectory_status"] == "estimated"
    fallback, _, flags = fit_monocular_sequence(relative, uv, times, 1920, 1080, 1.72,
                                                focal_px=1400, camera_moving=True)
    assert flags["moving_camera_fallback"] and flags["root_mode"] == "in_place"
    assert np.ptp(fallback[:, 19, :2], axis=0).max() < 1e-5


def test_manual_targets_survive_temporal_despiking():
    from multi_video_studio.spatiotemporal_solver import optimize_3d_trajectories_spatiotemporal
    relative, uv, _ = observations()
    target = relative.copy()
    target[25, 9, 0] += .17
    pins = np.zeros(target.shape[:2], bool)
    pins[25, 9] = True
    corrected, _, _ = optimize_3d_trajectories_spatiotemporal(target, uv[..., 2], pin_mask=pins)
    np.testing.assert_allclose(corrected[25, 9], target[25, 9], atol=1e-6)


def test_single_video_review_save_cache_and_correction(tmp_path, monkeypatch):
    relative, uv, times = observations(24)
    source = tmp_path/"original.mp4"
    source.write_bytes(b"synthetic source identity")
    clip = CameraVideoClip("cam01", source, 1920, 1080, 30., 24, .8, False)
    package = MultiVideoPackage("mono", tmp_path, motion_clips={"cam01": clip})
    calls = []
    def track(*args):
        calls.append(True)
        return CameraTrackingResult("cam01", source, 1920, 1080, 30., times.tolist(), uv,
            np.zeros((24, 21, 3)), np.zeros((24, 21, 3)),
            wrist_crop_hands=[{} for _ in times], head_crop_faces=[None for _ in times],
            backend_tag="mediapipe", relative_body26=relative,
            runtime_metadata={"providers": ["CPUExecutionProvider"]})
    monkeypatch.setattr("multi_video_studio.monocular_pipeline.track_single_camera", track)
    state = ReviewState()
    state.bind_sources(package.motion_clips)
    cfg = CinemaPipelineConfig(single_camera_backend="mediapipe", enable_fingers=False, enable_head_face=False,
        tracking_cache_dir=str(tmp_path/"cache"), review_state=state.to_dict())
    first = MultiVideoCinemaPipeline(cfg).run_from_source(package, tmp_path/"exports", take_name="mono")
    saved = load_motion_file(first.h5_path)
    assert saved.processing_config["scorecard"]["mean_reprojection_rmse_px"] is None
    assert "N/A" in first.scorecard.summary_line()
    np.testing.assert_equal(saved.observations_2d["cam01"]["native_pts_s"], times)
    raw_observations = ReviewObservations.load(Path(saved.processing_config["review_observations_path"]))
    np.testing.assert_equal(raw_observations.cameras["cam01"]["body"], uv)
    np.testing.assert_allclose(saved.frame_timestamps_ns*1e-9, times, atol=1e-8)
    state.edit_point("cam01", times[10], "body:9", uv[10, 9, :2]+[30, 0])
    cfg.review_state = state.to_dict()
    second = MultiVideoCinemaPipeline(cfg).run_from_source(package, tmp_path/"exports", take_name="mono_corrected")
    assert len(calls) == 1
    assert second.document.processing_config["single_camera"]["cache_hit"]
    assert second.document.processing_config["single_camera"]["manual_body_points"] == 1
    assert np.linalg.norm(first.document.joints_3d_raw[10, 9]-second.document.joints_3d_raw[10, 9]) > .04
    np.testing.assert_equal(raw_observations.cameras["cam01"]["body"], uv)
    assert "cam01" in saved.processing_config["review_state"]["sources"]
    reopened = ReviewObservations()
    reopened.from_document(saved, state)
    np.testing.assert_equal(reopened.cameras["cam01"]["pts"], times)
    np.testing.assert_equal(reopened.cameras["cam01"]["body"], uv)
    cfg.single_start_s, cfg.single_end_s = times[8], times[16]
    cfg.single_root_mode = "in_place"
    state.hand_rois = [{"camera_uid":"cam01", "side":"left", "start_s":20., "end_s":21., "rect":[50, 50, 100, 100]}]
    cfg.review_state = state.to_dict()
    def forbidden_roi(*args, **kwargs):
        raise AssertionError("Finger-disabled solve tried to run hand inference")
    monkeypatch.setattr("multi_video_studio.monocular_pipeline.refine_tracks_with_rois", forbidden_roi)
    cropped = MultiVideoCinemaPipeline(cfg).run_from_source(package, tmp_path/"range", take_name="mono_range")
    assert len(calls) == 1
    assert cropped.document.num_frames == 9
    np.testing.assert_allclose(cropped.document.frame_timestamps_ns*1e-9, times[8:17], atol=1e-8)
    assert np.ptp(cropped.document.corrected_motion.root_translation[:, :2], axis=0).max() < 1e-6


def test_wrong_wrist_and_tiny_palm_are_not_counted():
    from engine.hands.hand_tracker import HandObservation
    from multi_video_studio.monocular_tracker import gate_native_hands
    body = np.zeros((26, 3))
    body[9], body[10] = [100, 100, .95], [300, 100, .95]
    points = np.tile([300., 100.], (21, 1))
    points[5, 0], points[17, 0] = 290, 310
    obs = HandObservation("left", np.zeros((21, 3)), .95,
        np.tile([0., 0., 0., 1.], (16, 1)), landmarks_2d=points)
    assert not gate_native_hands({"left": obs}, body)
    points[:] = [100, 100]
    assert not gate_native_hands({"left": obs}, body)


@pytest.mark.parametrize("options", [{"single_hand_interval": 0}, {"single_camera_backend":"unknown"},
    {"single_root_mode":"measured"}, {"single_start_s":3,"single_end_s":2}, {"single_focal_length_px":float('nan')}])
def test_invalid_single_camera_controls(options):
    with pytest.raises(ValueError):
        CinemaPipelineConfig(**options)


def test_bidirectional_quaternion_projection_does_not_create_pi_boundary_snap():
    from scipy.spatial.transform import Rotation
    from multi_video_studio.zero_slide_physics import _geodesic_rate_limit_bidirectional
    from engine.motion_cleanup.quality_audit import audit_quaternions
    q = Rotation.random(80, random_state=23).as_quat()
    limited = _geodesic_rate_limit_bidirectional(q, 18.)
    audit = audit_quaternions(limited[:, None], ["arm"], 30.)
    assert audit["max_step_deg"] <= 18.001
    np.testing.assert_allclose(np.linalg.norm(limited, axis=1), 1., atol=1e-6)


def test_cancel_during_physics_keeps_original_motion_and_profile():
    import threading
    from engine.data_format.motion_io import SkeletalMotionTrack
    from multi_video_studio.zero_slide_physics import apply_zero_foot_slide_physics
    actor, root, rotations, positions, matrices, _ = _make_synthetic_walk_sequence(T=24)
    track = SkeletalMotionTrack(root, rotations[:, 0], rotations, matrices, positions, np.ones((24, 25)))
    before = actor.rest_offsets_m.copy()
    original_positions = positions.copy()
    cancel = threading.Event()
    def progress(fraction, message):
        if fraction >= .25:
            cancel.set()
    with pytest.raises(RuntimeError, match="Cancelled"):
        apply_zero_foot_slide_physics(track, actor, 30., cancel_event=cancel, progress_cb=progress)
    np.testing.assert_equal(track.joint_positions, original_positions)
    np.testing.assert_equal(actor.rest_offsets_m, before)


def test_background_motion_diagnostic_excludes_performer():
    import cv2
    from multi_video_studio.monocular_tracker import BackgroundMotionMonitor
    random = np.random.default_rng(7)
    gray = random.integers(0, 255, (360, 640), np.uint8)
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    body = np.zeros((26, 3))
    body[:, :2] = [320, 180]
    body[:, 2] = .95
    monitor = BackgroundMotionMonitor()
    monitor.sample(frame, body)
    shifted = cv2.warpAffine(frame, np.float32([[1, 0, 12], [0, 1, 0]]), (640, 360))
    monitor.sample(shifted, body)
    assert monitor.report()["detected"]
    assert monitor.report()["valid_pairs"] == 1


def test_cache_decompresses_each_member_once(tmp_path, monkeypatch):
    from multi_video_studio.tracking_cache import save_tracking_cache, load_tracking_cache
    relative, uv, times = observations(16)
    t = CameraTrackingResult("cam01", tmp_path/"source.mp4", 1920, 1080, 30., times.tolist(), uv,
        wrist_crop_hands=[{} for _ in times], head_crop_faces=[None for _ in times], relative_body26=relative)
    path = tmp_path/"cache.npz"
    save_tracking_cache(path, t)
    original = np.lib.npyio.NpzFile.__getitem__
    counts = {}
    def counted(self, name):
        counts[name] = counts.get(name, 0)+1
        return original(self, name)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", counted)
    restored = load_tracking_cache(path)
    assert max(counts.values()) == 1
    np.testing.assert_equal(restored.relative_body26, relative)


def test_model_url_query_does_not_enter_windows_filename(tmp_path):
    import json
    import threading
    from model_registry.registry import ModelRegistry
    p = tmp_path/"models.json"
    p.write_text(json.dumps({"models":[{"id":"pose", "family":"test", "task":"pose", "version":"test",
        "url":"https://example.invalid/model.onnx?download=true", "license":"test"}]}))
    registry = ModelRegistry(p, tmp_path/"weights")
    assert registry.get_local_path("pose").name == "pose_model.onnx"
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        registry.ensure_downloaded("pose", cancel=cancel)


def test_review_timeline_includes_native_end_after_nonzero_stream_origin(tmp_path):
    from types import SimpleNamespace
    from desktop.ui.widgets.camera_review import CameraReviewWorkspace
    class Slider:
        def setRange(self, low, high):
            self.low, self.high = low, high
        def setValue(self, value):
            self.current = value
        def value(self):
            return self.current
    clip = CameraVideoClip("cam01", tmp_path/"video.mp4", 3840, 2160, 30., 3019, 100.607, False)
    observations = ReviewObservations()
    observations.set_track("cam01", [.1255, 100.699877], np.zeros((2, 26, 3)))
    slider = Slider()
    workspace = SimpleNamespace(package=SimpleNamespace(motion_clips={"cam01":clip}),
        state=SimpleNamespace(global_time=lambda uid, t:t), observations=observations,
        time_slider=slider, range_start=SimpleNamespace(setValue=lambda value:None),
        range_end=SimpleNamespace(setValue=lambda value:None), _seek=lambda value:None)
    CameraReviewWorkspace._update_timeline(workspace)
    assert slider.high >= 100699


def test_raw_preview_does_not_override_new_model_or_new_hand_roi(tmp_path):
    from multi_video_studio.monocular_pipeline import correction_frame
    source = tmp_path/"source.mp4"
    source.write_bytes(b"current source")
    clip = CameraVideoClip("cam01", source, 1920, 1080, 30., 16, .5, False)
    state = ReviewState()
    state.bind_sources({"cam01":clip})
    _, uv, pts = observations(16)
    saved = ReviewObservations()
    saved.set_track("cam01", pts, uv, source="native single-camera observations",
                    asset_id=state.sources["cam01"]["asset_id"])
    assert correction_frame(saved, state, "cam01", pts[4]) is None
    saved.cameras["cam01"]["metadata"]["correction_pts"] = [float(pts[4])]
    assert correction_frame(saved, state, "cam01", pts[4]) is not None
    saved.cameras["cam01"]["metadata"]["asset_id"] = "replaced source"
    assert correction_frame(saved, state, "cam01", pts[4]) is None


def test_blender_halfway_poses_and_fractional_fps_motion_exports(tmp_path):
    import json
    import subprocess
    from scipy.spatial.transform import Rotation
    from desktop.settings.app_settings import detect_blender_executable
    from engine.data_format.motion_io import MotionCaptureDocument, SkeletalMotionTrack, export_blender_json
    from engine.skeleton.rig_definition import BONE_NAMES, PARENT_INDICES, DEFAULT_REST_OFFSETS_M, compute_forward_kinematics_sequence
    executable = detect_blender_executable()
    if not executable:
        pytest.skip("Blender is not installed.")
    q = np.tile([0., 0., 0., 1.], (8, 25, 1))
    q[:, 0] = Rotation.from_euler("z", np.arange(8)*2+35, degrees=True).as_quat()
    q[3:] *= -1  # Equivalent key poses, disastrous linear interpolation without a hemisphere guard.
    root = np.tile([0., 0., .96], (8, 1))
    positions, matrices = compute_forward_kinematics_sequence(root, q, DEFAULT_REST_OFFSETS_M)
    track = SkeletalMotionTrack(root, q[:, 0], q, matrices, positions, np.ones((8, 25)))
    doc = MotionCaptureDocument("test", "test", "signs", "actor", 29.97,
        np.round(np.arange(8)/29.97*1e9).astype(np.int64), BONE_NAMES, PARENT_INDICES,
        DEFAULT_REST_OFFSETS_M, track)
    payload = export_blender_json(doc, tmp_path/"motion.json")
    repo = Path(__file__).resolve().parents[2]
    script = tmp_path/"blender_signs.py"
    script.write_text(f'''
import bpy,sys,json,importlib.util
from pathlib import Path
sys.path.insert(0,{str(repo/'blender_addon')!r})
from nuranova_mocap_bridge.source_armature import create_source_armature_from_payload,load_nmocap_json
from nuranova_mocap_bridge.character_export import export_character
spec=importlib.util.spec_from_file_location('pose_audit',{str(repo/'scripts/blender_motion_audit.py')!r})
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
rig=create_source_armature_from_payload(load_nmocap_json({str(payload)!r}),push_nla=False)
r=audit.audit_rigs(bpy.context.scene,[rig],1,8,Path({str(tmp_path/'poses.npz')!r}))[rig.name]
assert r['max_midpoint_error_deg']<.1,r
reports=[]
for suffix in ['glb','fbx']:
 result=export_character(rig,Path({str(tmp_path)!r})/('motion.'+suffix),variant='motion',frame_start=1,frame_end=8)
 assert result['passed'] and result['max_deformation_matrix_error']<.001,result
 assert abs(result['duration_frames']-7)<.001,result
 reports.append(result)
from types import SimpleNamespace
from nuranova_mocap_bridge.character_quality import duplicate_character,fix_character_frame
working,_=duplicate_character(rig)
calls=[0]
def staged_contacts(frame):
 calls[0]+=1
 depth=[.03,.02,.04][min(calls[0]-1,2)]
 contact=dict(a='torso',b='left_hand',moving_region='left_hand',push_normal=[0,1,0],depth_m=depth)
 return dict(frame=frame,contacts=[contact],max_depth_m=depth)
profile=dict(mapping=dict((name,name) for name in working.pose.bones.keys()),unit_scale=1.,margin_m=.003)
geometry=SimpleNamespace(armature=working,profile=profile,analyze=staged_contacts)
bpy.context.scene.frame_set(1)
fixed=fix_character_frame(geometry,1,iterations=3)
assert fixed['after']['max_depth_m']==.02,fixed
assert fixed['changed_bones'],fixed
Path({str(tmp_path/'report.json')!r}).write_text(json.dumps(reports))
''', encoding="utf-8")
    result = subprocess.run([executable, "--background", "--factory-startup", "--python", str(script)],
                            capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout.decode(errors="replace")[-5000:]
    assert (tmp_path/"report.json").exists(), result.stdout.decode(errors="replace")[-5000:]
    assert all(v["passed"] for v in json.loads((tmp_path/"report.json").read_text()))
