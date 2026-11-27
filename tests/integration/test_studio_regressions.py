"""Regressions from inspecting the desktop and mixed-frame-rate phone workflow."""

import threading
import zipfile
from pathlib import Path

import numpy as np
import pytest

from desktop.project_manager.manager import ProjectManager
from desktop.synchronization.sync_engine import align_observations_to_common_timeline
from desktop.ui.app_controller import StudioController
from desktop.ui.main_window import NuranovaMainWindow
from multi_video_studio.package_ingest import _safe_extract_zip
from multi_video_studio.pipeline import CinemaPipelineConfig, MultiVideoCinemaPipeline
from multi_video_studio.pyav_sync import CameraSyncEstimate, resample_synchronized_tracks


@pytest.mark.parametrize(
    "member",
    [
        "../extract_sibling/probe.txt",
        "..\\extract_sibling\\probe.txt",
        "C:/probe.txt",
        "C:probe.txt",
        "/probe.txt",
    ],
)
def test_archive_importers_reject_escapes_before_writing(tmp_path: Path, member: str):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("valid.txt", "must not be written before validation finishes")
        zf.writestr(member, "outside")
    with pytest.raises(ValueError, match="Unsafe"):
        _safe_extract_zip(archive, tmp_path / "extract")
    assert not (tmp_path / "extract" / "valid.txt").exists()
    manager = ProjectManager(tmp_path / "recent.json")
    with pytest.raises(ValueError, match="Unsafe"):
        manager.import_or_restore_project_zip(archive, tmp_path / "restore")
    assert not (tmp_path / "restore" / "valid.txt").exists()


def test_project_switch_clears_motion_calibration_bridge_and_editor(qtbot, tmp_path):
    ctrl = StudioController(tmp_path / "settings.json")
    win = NuranovaMainWindow(ctrl)
    qtbot.addWidget(win)
    try:
        first = ctrl.create_project(tmp_path, "First")
        ctrl.generate_synthetic_demo_take("first_take")
        ctrl.imported_videos["old"] = tmp_path / "old.mp4"
        assert ctrl.active_calibration is not None
        assert ctrl.network_server.bridge_takes
        ctrl.create_project(tmp_path, "Second")
        assert ctrl.active_motion is None
        assert ctrl.active_calibration is None
        assert not ctrl.network_server.bridge_takes
        assert not ctrl.imported_videos
        ctrl.open_project(first.root_dir)
        assert ctrl.active_motion.take_id == "first_take"
        ctrl.offline_processing = True
        with pytest.raises(RuntimeError, match="video solve"):
            ctrl.create_project(tmp_path, "Blocked")
    finally:
        ctrl.offline_processing = False
        win.close()


def test_alignment_preserves_exact_samples_next_to_missing_detections():
    times = np.array([0, 100_000_000, 200_000_000], dtype=np.int64)
    obs = np.ones((3, 1, 3), dtype=np.float32)
    obs[0, :, 2] = 0
    obs[1, :, :2] = [42, 23]
    _, aligned = align_observations_to_common_timeline(
        {"a": times, "b": times},
        {"a": obs, "b": obs},
        target_fps=10,
        max_interp_gap_s=0.2,
    )
    np.testing.assert_array_equal(aligned["a"][1], obs[1])


def test_editor_accepts_a_take_with_only_one_hand(qtbot, tmp_path):
    from engine.data_format.motion_io import HandMotionTrack

    ctrl = StudioController(tmp_path / "settings.json")
    win = NuranovaMainWindow(ctrl)
    qtbot.addWidget(win)
    try:
        doc = ctrl.generate_synthetic_demo_take("one_hand")
        doc.hands = HandMotionTrack(
            right_local_rotations_xyzw=np.tile([0, 0, 0, 1], (doc.num_frames, 16, 1)).astype(
                np.float32
            )
        )
        ctrl.set_active_motion(doc)
        editor = win.pages["Motion Editor"]
        editor.set_frame(10)
        assert editor.viewport.canvas.left_hand_quats is None
        assert editor.viewport.canvas.right_hand_quats.shape == (16, 4)
    finally:
        win.close()


def test_motion_save_preserves_actor_proportions_and_cleanup(qtbot, tmp_path):
    from engine.data_format.motion_io import load_motion_file
    from engine.motion_cleanup.cleanup import MotionCleanupConfig

    ctrl = StudioController(tmp_path / "settings.json")
    try:
        project = ctrl.create_project(tmp_path, "Save Test")
        doc = ctrl.generate_synthetic_demo_take("save_take")
        doc.rest_offsets = doc.rest_offsets * 0.85
        ctrl.set_active_motion(doc)
        np.testing.assert_allclose(ctrl.active_actor.rest_offsets_m, doc.rest_offsets)
        ctrl.apply_motion_cleanup(MotionCleanupConfig(strength=0.5))
        ctrl.save_project()
        loaded = load_motion_file(project.subdir("corrected_motion") / "save_take.nmocap.h5")
        np.testing.assert_allclose(
            loaded.corrected_motion.joint_positions, doc.corrected_motion.joint_positions
        )
        np.testing.assert_allclose(loaded.rest_offsets, doc.rest_offsets)
    finally:
        ctrl.shutdown()


def test_model_checksums_are_saved_in_user_cache_without_rewriting_registry(tmp_path, monkeypatch):
    import io
    import json
    from model_registry.registry import ModelRegistry

    source = tmp_path / "packaged_models.json"
    text = json.dumps(
        {
            "models": [
                {
                    "id": "pose",
                    "family": "test",
                    "task": "pose",
                    "version": "1",
                    "url": "https://example.test/pose.onnx",
                    "license": "test",
                }
            ]
        }
    )
    source.write_text(text)
    payload = b"model bytes" * 200
    monkeypatch.setattr(
        "model_registry.registry.urllib.request.urlopen", lambda *a, **kw: io.BytesIO(payload)
    )
    registry = ModelRegistry(source, tmp_path / "cache")
    registry.ensure_downloaded("pose")
    assert source.read_text() == text
    assert (tmp_path / "cache" / "checksums.json").exists()
    assert (
        ModelRegistry(source, tmp_path / "cache").models["pose"].sha256
        == registry.models["pose"].sha256
    )


def test_nonoverlapping_sync_never_silently_resets_offsets():
    obs = np.ones((20, 26, 3), dtype=np.float32)
    pts = np.arange(20) / 10
    estimates = {
        "a": CameraSyncEstimate("a", 0, "reference", 1, "Reference"),
        "b": CameraSyncEstimate("b", 10, "manual", 1, "10 seconds"),
    }
    with pytest.raises(ValueError, match="shared timeline"):
        resample_synchronized_tracks(estimates, {"a": pts, "b": pts}, {"a": obs, "b": obs}, 10)


def test_single_video_vfr_resampling_preserves_elapsed_time():
    from multi_video_studio.pipeline import _resample_single_view

    times = np.array([0.0, 0.1, 0.18, 0.3, 0.4])
    points = np.broadcast_to(times[:, None, None], (5, 26, 3)).copy()
    conf = np.ones((5, 26))
    out, _, uniform, _ = _resample_single_view(points, conf, times, 10)
    np.testing.assert_allclose(uniform, np.arange(5) / 10)
    np.testing.assert_allclose(out[:, 0, 0], uniform, atol=1e-6)


def test_upload_rejects_holes_oversized_data_and_changed_resume_metadata(tmp_path):
    import hashlib
    from fastapi.testclient import TestClient
    from desktop.camera_manager.manager import CameraManager
    from desktop.network.server import StudioNetworkServer

    manager = CameraManager()
    pin = manager.generate_pairing_pin()
    with TestClient(StudioNetworkServer(manager, tmp_path / "storage").app) as client:
        paired = client.post(
            "/api/v1/pair",
            json={"device_id": "test", "device_name": "Test", "pin": pin, "capabilities": {}},
        )
        assert paired.status_code == 200
        headers = {"Authorization": "Bearer " + paired.json()["auth_token"], "X-Device-ID": "test"}
        payload = b"1234"
        request = {
            "device_id": "test",
            "session_id": "session",
            "take_id": "take",
            "filename": "test.mp4",
            "file_kind": "video",
            "total_size_bytes": 4,
            "sha256_hex": hashlib.sha256(payload).hexdigest(),
        }
        initial = client.post("/api/v1/transfers/init", json=request, headers=headers)
        assert initial.status_code == 200
        endpoint = "/api/v1/transfers/" + initial.json()["transfer_id"] + "/chunk"
        assert (
            client.put(
                endpoint, content=b"x", headers={**headers, "X-Chunk-Offset": "2"}
            ).status_code
            == 409
        )
        assert (
            client.put(
                endpoint, content=b"12345", headers={**headers, "X-Chunk-Offset": "0"}
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/api/v1/transfers/init", json={**request, "total_size_bytes": 5}, headers=headers
            ).status_code
            == 409
        )
        assert (
            client.put(
                endpoint, content=payload, headers={**headers, "X-Chunk-Offset": "0"}
            ).status_code
            == 200
        )


def test_cancelling_running_job_is_reported_as_cancelled(qtbot):
    from desktop.workers.job_queue import ProcessingQueueManager, JobStatus

    queue = ProcessingQueueManager(1)
    entered = threading.Event()

    def work(progress, cancel):
        entered.set()
        cancel.wait(5)
        raise RuntimeError("Cancelled by user")

    try:
        job = queue.submit("Cancel test", "test", work)
        qtbot.waitUntil(entered.is_set)
        queue.cancel_job(job.job_id)
        qtbot.waitUntil(lambda: job.finished_at is not None)
        assert job.status == JobStatus.CANCELLED
        assert job.error is None
    finally:
        queue.shutdown()


def test_hand_tracks_follow_three_camera_body_overlap():
    pts = {"a": np.arange(100) / 10, "b": np.arange(100) / 10, "c": np.arange(30) / 10 + 2}
    estimates = {u: CameraSyncEstimate(u, 0, "reference", 1, "") for u in pts}
    body = {u: np.ones((len(t), 26, 3), np.float32) for u, t in pts.items()}
    hands = {u: np.ones((len(pts[u]), 21, 3), np.float32) for u in ("a", "b")}
    canonical, _, extra = resample_synchronized_tracks(estimates, pts, body, 10, {"left": hands})
    assert len(canonical) == 30
    assert all(len(track) == len(canonical) for track in extra["left"].values())


def test_recording_library_keeps_all_imported_views_without_studio_calibration(
    qtbot, tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from tests.integration.test_multi_video_cinema_studio import _write_dummy_mp4

    ctrl = StudioController(tmp_path / "settings.json")
    try:
        doc = ctrl.generate_synthetic_demo_take("recorded")
        ctrl.active_calibration = None
        videos = {f"cam_{i}": _write_dummy_mp4(tmp_path / f"view_{i}.mp4") for i in range(3)}
        captured = []

        def run(pipeline, source, output, **kwargs):
            captured.append(source.num_motion_cameras)
            return SimpleNamespace(document=doc, calibration=None, blender_json_path=None)

        monkeypatch.setattr(MultiVideoCinemaPipeline, "run_from_source", run)
        ctrl.reconstruct_from_video_files(videos)
        assert captured == [3]
    finally:
        ctrl.shutdown()


def test_video_workspace_fits_laptop_and_keeps_solve_visible(qtbot, tmp_path):
    ctrl = StudioController(tmp_path / "settings.json")
    win = NuranovaMainWindow(ctrl)
    qtbot.addWidget(win)
    try:
        win.resize(1366, 768)
        win.show()
        win.navigate_to("Multi-Video Cinema Lab")
        qtbot.wait(20)
        page = win.pages["Multi-Video Cinema Lab"]
        assert win.width() == 1366
        assert win.minimumSizeHint().width() < 1366
        assert page.workflow_tabs.count() == 2
        scroll = win.page_scrolls["Multi-Video Cinema Lab"]
        assert (
            scroll.viewport()
            .rect()
            .contains(page.btn_solve.mapTo(scroll.viewport(), page.btn_solve.rect().center()))
        )
    finally:
        win.close()


def test_cancelled_pipeline_produces_no_exports(tmp_path):
    from multi_video_studio.package_ingest import MultiVideoPackage, CameraVideoClip

    clip = CameraVideoClip("cam_01", tmp_path / "video.mp4", 320, 240, 30, 30, 1, False)
    package = MultiVideoPackage("take", tmp_path, motion_clips={"cam_01": clip})
    event = threading.Event()
    event.set()
    with pytest.raises(RuntimeError, match="Cancelled"):
        MultiVideoCinemaPipeline().run_from_source(package, tmp_path / "out", cancel_event=event)
    assert not list((tmp_path / "out").glob("*"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frame_stride": 0},
        {"target_fps": float("nan")},
        {"actor_height_m": -1},
        {"manual_offsets_ms": {"cam": float("inf")}},
    ],
)
def test_processing_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        CinemaPipelineConfig(**kwargs)
