"""Moving-camera fidelity, guarded background matching and frozen cache regressions."""
from pathlib import Path

import cv2
import numpy as np

from engine.ai_tracking.monocular_geometry import CAMERA_TO_STUDIO, fit_monocular_sequence
from multi_video_studio.camera_stabilization import BackgroundReference, interpolate_alignment, assess_alignment_support
from multi_video_studio.tracking_cache import tracking_cache_key
from tests.integration.test_monocular_studio import observations


def _mask_points(x, y, width, height):
    body = np.zeros((26, 3), np.float32)
    body[[5, 6, 11, 12]] = [[x, y, .95], [x+width, y, .95],
                            [x, y+height, .95], [x+width, y+height, .95]]
    return body


def test_background_reference_tracks_camera_without_tracking_the_person():
    rng = np.random.default_rng(42)
    background = cv2.GaussianBlur(rng.integers(0, 256, (640, 480, 3), np.uint8), (3, 3), .5)
    reference = background.copy()
    reference[180:560, 100:220] = rng.integers(0, 256, (380, 120, 3), np.uint8)
    camera = cv2.getRotationMatrix2D((240, 320), 5., 1.07)
    camera[:, 2] += [14., -11.]
    current = cv2.warpAffine(background, camera, (480, 640))
    current[190:570, 240:360] = rng.integers(0, 256, (380, 120, 3), np.uint8)
    matcher = BackgroundReference()
    first, _ = matcher.estimate(reference, _mask_points(100, 180, 120, 380))
    assert first is not None
    aligned, report = matcher.estimate(current, _mask_points(240, 190, 120, 380))
    assert aligned is not None, report
    samples = np.array([[30., 30., 1.], [450., 30., 1.], [30., 600., 1.]])
    moved = samples@np.vstack([camera, [0, 0, 1]]).T
    recovered = moved@aligned.T
    assert np.max(np.linalg.norm(recovered[:, :2]-samples[:, :2], axis=1)) < 1.
    assert report['inliers'] >= 10


def test_blank_background_does_not_invent_a_camera_transform():
    aligned, report = BackgroundReference().estimate(np.zeros((640, 480, 3), np.uint8),
        _mask_points(100, 180, 120, 380))
    assert aligned is None and 'texture' in report['reason']


def test_camera_alignment_preserves_roll_wrap_scale_and_marks_long_gaps():
    matrices = np.tile(np.eye(3), (2, 1, 1))
    for i, angle in enumerate(np.radians([179., -179.])):
        matrices[i, :2, :2] = (1+i)*np.array([[np.cos(angle), -np.sin(angle)],
                                             [np.sin(angle), np.cos(angle)]])
    times = np.linspace(0, 1, 11)
    aligned, supported = interpolate_alignment([0, 1], matrices, times)
    angle = np.unwrap(np.arctan2(aligned[:, 1, 0], aligned[:, 0, 0]))
    assert np.max(abs(np.degrees(np.diff(angle)))) < .21
    np.testing.assert_allclose(np.linalg.det(aligned[5, :2, :2]), 2., atol=1e-8)
    assert supported[0] and supported[-1] and not supported[5]


def test_short_fade_is_flagged_without_discarding_all_supported_movement():
    times = np.arange(150)/30
    supported = np.ones(len(times), bool)
    supported[-10:] = False
    accepted, uncertain = assess_alignment_support(times, supported)
    assert accepted and len(uncertain) == 1
    assert uncertain[0]['start_s'] == times[-10]
    supported[50:70] = False
    assert not assess_alignment_support(times, supported)[0]


def test_background_compensation_retains_travel_and_flight_without_scaling_bones():
    relative, native, times = observations(80)
    expected, _, _ = fit_monocular_sequence(relative, native, times, 1920, 1080, 1.72, focal_px=1400)
    transforms = np.tile(np.eye(3), (len(times), 1, 1))
    rolled = relative.copy()
    moved = native.copy()
    for i, phase in enumerate(np.linspace(0, 2*np.pi, len(times))):
        angle = .06*np.sin(phase)
        scale = 1.08+.04*np.cos(phase)
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                             [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        transform = rotation.copy()
        transform[:2, :2] *= scale
        transform[:2, 2] = [45*np.sin(phase), 30*np.cos(phase)]
        uv = np.c_[native[i, :, :2], np.ones(26)]@transform.T
        moved[i, :, :2] = uv[:, :2]
        basis = CAMERA_TO_STUDIO@rotation@CAMERA_TO_STUDIO.T
        rolled[i] = relative[i]@basis.T
        transforms[i] = np.linalg.inv(transform)
    before = moved.copy()
    actual, _, metadata = fit_monocular_sequence(rolled, moved, times, 1920, 1080, 1.72,
        focal_px=1400, camera_moving=True, camera_transforms=transforms)
    np.testing.assert_allclose(actual, expected, atol=2e-6)
    np.testing.assert_array_equal(moved, before)
    assert np.ptp(actual[:, 19, 0]) > .2
    assert np.ptp(actual[:, 19, 2]) > .25
    assert metadata['camera_motion_compensated'] and not metadata['moving_camera_fallback']


def test_missing_source_interval_never_resets_camera_distance_to_zero():
    relative, native, times = observations(120)
    relative[40:80] = np.nan
    native[40:80, :, 2] = 0.
    _, _, metadata = fit_monocular_sequence(relative, native, times, 1920, 1080, 1.72, focal_px=1400)
    roots = np.asarray(metadata['root_camera_m'])
    assert np.isfinite(roots).all()
    assert np.min(roots[40:80, 2]) > 3.


def test_uncertain_camera_tail_holds_travel_but_keeps_visible_limb_evidence():
    relative, native, times = observations(80)
    supported = np.ones(80, bool)
    supported[-8:] = False
    points, confidence, _ = fit_monocular_sequence(relative, native, times, 1920, 1080, 1.72,
        focal_px=1400, camera_moving=True, camera_transforms=np.tile(np.eye(3), (80, 1, 1)),
        camera_supported=supported)
    assert np.isfinite(points[-8:]).all()
    np.testing.assert_array_equal(confidence, native[..., 2])


def test_frozen_tracking_cache_does_not_require_loose_python_files(tmp_path, monkeypatch):
    from multi_video_studio import tracking_cache
    video = tmp_path/'video.mp4'
    video.write_bytes(b'source identity')
    bundle = tmp_path/'studio.exe'
    bundle.write_bytes(b'version one bundled bytecode')
    monkeypatch.setattr(tracking_cache, '__file__', str(tmp_path/'no_sources/tracking_cache.py'))
    monkeypatch.setattr(tracking_cache.sys, 'frozen', True, raising=False)
    monkeypatch.setattr(tracking_cache.sys, 'executable', str(bundle))
    first = tracking_cache_key(video, {'kind':'monocular_v3'})
    assert first == tracking_cache_key(video, {'kind':'monocular_v3'})
    replacement = tmp_path/'studio_new.exe'
    replacement.write_bytes(b'version two bundled bytecode')
    monkeypatch.setattr(tracking_cache.sys, 'executable', str(replacement))
    assert first != tracking_cache_key(video, {'kind':'monocular_v3'})


def test_monocular_adapter_change_invalidates_its_native_cache(tmp_path, monkeypatch):
    video = tmp_path/'video.mp4'
    video.write_bytes(b'source identity')
    first = tracking_cache_key(video, {'kind':'monocular_v3'})
    read_bytes = Path.read_bytes
    monkeypatch.setattr(Path, 'read_bytes', lambda path: read_bytes(path)+b'\n# adapter changed\n'
        if path.name == 'monocular_tracker.py' else read_bytes(path))
    assert first != tracking_cache_key(video, {'kind':'monocular_v3'})
