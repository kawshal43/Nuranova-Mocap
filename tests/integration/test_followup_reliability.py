"""Regressions for native clock bounds, airborne poses and contact rotation wraps."""
from fractions import Fraction
from pathlib import Path
import subprocess

import numpy as np
import pytest


def test_video_probe_preserves_nonzero_source_clock(tmp_path):
    import av
    from multi_video_studio.package_ingest import probe_camera_clip
    path = tmp_path / "starts_at_two_seconds.mp4"
    with av.open(str(path), "w", options={"avoid_negative_ts": "disabled"}) as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = 160, 120, "yuv420p"
        stream.codec_context.time_base = Fraction(1, 30)
        for i in range(30):
            frame = av.VideoFrame.from_ndarray(np.full((120, 160, 3), i*5, np.uint8), format="bgr24")
            frame.pts, frame.time_base = 60+i, Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(path)) as container:
        actual = [float(f.pts*f.time_base) for f in container.decode(video=0)]
    assert actual[0] >= 1.9, "The fixture must retain a nonzero stream origin."
    clip = probe_camera_clip(path, "cam01")
    assert clip is not None
    assert clip.start_pts_s == pytest.approx(actual[0], abs=.001)
    assert clip.start_pts_s+clip.duration_s >= actual[-1]
    # Review must have correct bounds before any source-frame index is loaded.
    from types import SimpleNamespace
    from desktop.ui.widgets.camera_review import CameraReviewWorkspace
    from multi_video_studio.review_state import ReviewState

    class Slider:
        def setRange(self, start, end):
            self.bounds = start, end

        def setValue(self, value):
            self.current = value

        def value(self):
            return self.current

    slider = Slider()
    workspace = SimpleNamespace(package=SimpleNamespace(motion_clips={"cam01": clip}),
        observations=SimpleNamespace(cameras={}), state=ReviewState(), time_slider=slider,
        range_start=SimpleNamespace(setValue=lambda value: None),
        range_end=SimpleNamespace(setValue=lambda value: None), _seek=lambda value: None,
        decoder=SimpleNamespace(frame_times={}, frame_epoch=-1), preview_epoch=1, pause=lambda: None)
    CameraReviewWorkspace._update_timeline(workspace)
    assert slider.bounds[0] == round(actual[0]*1000)
    assert slider.bounds[1] >= round(actual[-1]*1000)
    CameraReviewWorkspace.seek_source_edge(workspace, "cam01")
    assert slider.value() == round(actual[0]*1000)
    CameraReviewWorkspace.seek_source_edge(workspace, "cam01", last=True)
    assert slider.value() == pytest.approx(actual[-1]*1000, abs=1)


def test_foot_cleanup_respects_airborne_contact_evidence():
    from engine.data_format.motion_io import SkeletalMotionTrack
    from engine.skeleton.rig_definition import ActorProfile, DEFAULT_REST_OFFSETS_M, BONE_INDEX, compute_forward_kinematics_sequence
    from multi_video_studio.zero_slide_physics import apply_zero_foot_slide_physics
    count = 120
    actor = ActorProfile("flight", "Flight test", rest_offsets_m=DEFAULT_REST_OFFSETS_M.copy())
    rotations = np.tile([0., 0., 0., 1.], (count, 25, 1))
    root = np.zeros((count, 3))
    rest_positions, _ = compute_forward_kinematics_sequence(root, rotations, actor.rest_offsets_m)
    feet = [BONE_INDEX[n] for n in ("left_foot", "left_toe", "right_foot", "right_toe")]
    root[:, 2] = -rest_positions[0, feet, 2].min()
    # A slow-motion/assisted raised pose has no foot contact even at zero vertical velocity.
    root[30:40, 2] += np.linspace(0., .25, 10)
    root[40:80, 2] += .25
    root[80:90, 2] += np.linspace(.25, 0., 10)
    positions, matrices = compute_forward_kinematics_sequence(root, rotations, actor.rest_offsets_m)
    track = SkeletalMotionTrack(root, rotations[:, 0], rotations, matrices, positions, np.ones((count, 25)))
    contacts = np.ones((count, 4), np.uint8)
    contacts[30:90] = 0
    cleaned, *_ = apply_zero_foot_slide_physics(track, actor, 30., contacts_override=contacts,
                                               allow_floor_tilt=False)
    raised_height = np.median(np.min(cleaned.joint_positions[50:70, feet, 2], axis=1))
    assert raised_height > .20, f"Cleanup incorrectly grounded the airborne feet: {raised_height:.3f} m"


def test_repeated_preview_refreshes_accept_the_frame_already_decoding(tmp_path, monkeypatch):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QObject, Signal
    from PySide6.QtWidgets import QApplication
    from desktop.ui.widgets import camera_review
    from multi_video_studio.package_ingest import ingest_multi_video_source
    from multi_video_studio.review_video import ReviewVideoFrame
    from tests.integration.test_multi_video_cinema_studio import _write_dummy_mp4

    class DelayedDecoder(QObject):
        ready = Signal(object, object)

        def __init__(self):
            super().__init__()
            self.frame_times = {}

        def request(self, generation, requests):
            self.latest = generation, requests

        def close(self):
            pass

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(camera_review, "PreviewDecoder", DelayedDecoder)
    workspace = camera_review.CameraReviewWorkspace(tmp_path/"review")
    try:
        workspace.load_package(ingest_multi_video_source([_write_dummy_mp4(tmp_path/"cam_01.mp4")]))
        workspace.time_slider.setValue(100)
        generation, requests = workspace.decoder.latest
        uid = requests[0][0]
        # Layout, tab and selection events refresh the same source time while
        # its decode is still pending. They must not starve the finished image.
        for _ in range(10):
            workspace.request_frames()
        frame = ReviewVideoFrame(.1, np.zeros((180, 320, 3), np.uint8), (320, 180), 3)
        workspace.decoder.ready.emit(generation, {uid: frame})
        assert workspace.last_frames.get(uid) is frame
        assert workspace.tiles[("cameras", uid, None)].overlay._has_frame
        app.processEvents()
    finally:
        workspace.shutdown()


def test_character_corrections_follow_short_path_across_180_degrees(tmp_path):
    from desktop.settings.app_settings import detect_blender_executable
    from engine.data_format.motion_io import MotionCaptureDocument, SkeletalMotionTrack, export_blender_json
    from engine.skeleton.rig_definition import BONE_NAMES, PARENT_INDICES, DEFAULT_REST_OFFSETS_M, compute_forward_kinematics_sequence
    blender = detect_blender_executable()
    if not blender:
        pytest.skip("Blender is not installed.")
    count = 120
    root = np.tile([0., 0., .96], (count, 1))
    rotations = np.tile([0., 0., 0., 1.], (count, 25, 1))
    positions, matrices = compute_forward_kinematics_sequence(root, rotations, DEFAULT_REST_OFFSETS_M)
    track = SkeletalMotionTrack(root, rotations[:, 0], rotations, matrices, positions, np.ones((count, 25)))
    doc = MotionCaptureDocument("test", "test", "correction_wrap", "actor", 30.,
        np.round(np.arange(count)/30*1e9).astype(np.int64), BONE_NAMES, PARENT_INDICES,
        DEFAULT_REST_OFFSETS_M, track)
    payload = export_blender_json(doc, tmp_path/"motion.json")
    addon = Path(__file__).resolve().parents[2]/"blender_addon"
    script = tmp_path/"wrap.py"
    script.write_text(f'''
import bpy,sys,math
from mathutils import Quaternion
sys.path.insert(0,{str(addon)!r})
from nuranova_mocap_bridge.source_armature import create_source_armature_from_payload,load_nmocap_json,_get_action_fcurves
from nuranova_mocap_bridge.character_quality import duplicate_character,regularize_contact_corrections
original=create_source_armature_from_payload(load_nmocap_json({str(payload)!r}),push_nla=False)
working,_=duplicate_character(original)
for frame in range(1,121):
 bpy.context.scene.frame_set(frame)
 bone=working.pose.bones['left_hand']
 bone.rotation_quaternion=original.pose.bones['left_hand'].rotation_quaternion @ Quaternion((1,0,0),math.radians(170+20*(frame-1)/119))
 bone.keyframe_insert(data_path='rotation_quaternion',frame=frame)
for curve in _get_action_fcurves(working.animation_data.action,working):
 for key in curve.keyframe_points:key.interpolation='LINEAR'
report=list(regularize_contact_corrections(original,working,1,120,30.))[-1]
assert report['max_correction_step_deg']<=6.001,report
bpy.context.scene.frame_set(90)
actual=original.pose.bones['left_hand'].rotation_quaternion.conjugated() @ working.pose.bones['left_hand'].rotation_quaternion
expected=Quaternion((1,0,0),math.radians(170+20*89/119))
error=math.degrees(actual.rotation_difference(expected).angle)
error=min(error,360-error)
assert error<3,('A smooth correction took the long rotation path',error)
''', encoding="utf-8")
    result = subprocess.run([blender, "--background", "--factory-startup", "--python-exit-code", "1",
                             "--python", str(script)], capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout.decode(errors="replace")[-4000:]+result.stderr.decode(errors="replace")[-2000:]
