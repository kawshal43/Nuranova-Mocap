"""Recovery at publication boundaries and project/review UI guards."""

from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest
from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

from desktop.settings.app_settings import AppSettings
from desktop.ui.app_controller import StudioController
from desktop.ui.pages.pages_workspace import ProjectsPage
from desktop.ui.widgets.camera_review import CameraReviewWorkspace
from engine.math_core.camera_model import (
    CameraIntrinsics, CameraExtrinsics, CalibratedCamera, MultiCameraCalibration,
)
from multi_video_studio.package_ingest import ingest_multi_video_source
from tests.integration.test_multi_video_cinema_studio import _write_dummy_mp4
from tests.integration.test_startup_and_motion_ui import _cpu_hardware


def _calibration(uids, name):
    cameras = {}
    for index, uid in enumerate(uids):
        cameras[uid] = CalibratedCamera(
            CameraIntrinsics(uid, 320, 180, np.array([[300.,0,160],[0,300,90],[0,0,1]]), np.zeros(5)),
            CameraExtrinsics(uid, np.eye(3), np.array([index*.5, 0., 3.])),
        )
    return MultiCameraCalibration(name, cameras)


def test_mid_publication_failure_rolls_back_every_completed_replacement(tmp_path, monkeypatch):
    import shared_protocol.atomic_io as atomic
    staged = tmp_path / "staged"
    staged.mkdir()
    pairs = []
    for name in ("body.h5", "body.json", "body.bvh"):
        (tmp_path / name).write_bytes(b"Previous complete " + name.encode())
        (staged / name).write_bytes(b"New output " + name.encode())
        pairs.append((staged / name, tmp_path / name))
    original = atomic.os.replace

    def replace(source, destination):
        if Path(source) == staged / "body.json":
            raise OSError("Injected publication failure")
        return original(source, destination)

    monkeypatch.setattr(atomic.os, "replace", replace)
    with pytest.raises(OSError, match="publication failure"):
        atomic.publish_files(pairs)
    for _, destination in pairs:
        assert destination.read_bytes() == b"Previous complete " + destination.name.encode()
    assert not list(tmp_path.glob(".nuranova-recovery-*"))


def test_cancellation_between_file_replacements_restores_the_entire_delivery(tmp_path, monkeypatch):
    import shared_protocol.atomic_io as atomic
    cancelled = threading.Event()
    staged = tmp_path / "staged"
    staged.mkdir()
    pairs = []
    for name in ("body.h5", "body.json"):
        (tmp_path / name).write_bytes(b"Old " + name.encode())
        (staged / name).write_bytes(b"New " + name.encode())
        pairs.append((staged / name, tmp_path / name))
    original = atomic.os.replace

    def replace(source, destination):
        original(source, destination)
        if Path(source).parent == staged:
            cancelled.set()

    monkeypatch.setattr(atomic.os, "replace", replace)
    with pytest.raises(RuntimeError, match="Cancelled"):
        atomic.publish_files(pairs, cancel=cancelled)
    for _, target in pairs:
        assert target.read_bytes() == b"Old " + target.name.encode()


def test_cancel_during_serialization_does_not_publish_new_motion(tmp_path, monkeypatch):
    import multi_video_studio.pipeline as pipeline
    from tests.integration.test_bug_audit_failure_safety import _document
    processor = pipeline.MultiVideoCinemaPipeline(pipeline.CinemaPipelineConfig())
    processor._save_exports(_document(), tmp_path, "take")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    cancelled = threading.Event()
    original = pipeline.export_blender_json

    def export(*args, **kwargs):
        result = original(*args, **kwargs)
        cancelled.set()
        return result

    monkeypatch.setattr(pipeline, "export_blender_json", export)
    with pytest.raises(RuntimeError, match="Cancelled"):
        processor._save_exports(_document(), tmp_path, "take", cancel_event=cancelled)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()} == before


@pytest.mark.parametrize("operation", ["duplicate", "import_zip", "delete"])
def test_project_ui_cannot_bypass_processing_guard(qtbot, tmp_path, monkeypatch, operation):
    controller = StudioController(tmp_path / "settings.json", settings=AppSettings(), hardware=_cpu_hardware())
    page = ProjectsPage(controller)
    qtbot.addWidget(page)
    try:
        original = controller.create_project(tmp_path, "Original")
        archive = controller.project_manager.backup_or_export_project_zip(tmp_path / "saved.zip")
        warnings = []
        monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
        monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
        monkeypatch.setattr(QInputDialog, "getText", lambda *args, **kwargs: ("Copy", True))
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(archive), ""))
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(tmp_path / "imports"))
        controller.offline_processing = True
        getattr(page, "_" + operation)()
        assert controller.project_manager.active_project is original
        assert original.manifest_path.exists()
        assert warnings and any(word in warnings[0].lower() for word in ("processing", "solve"))
    finally:
        controller.offline_processing = False
        controller.shutdown()


def test_deleting_project_clears_motion_and_network_destination(qtbot, tmp_path, monkeypatch):
    settings = AppSettings()
    settings.storage.cache_dir = str(tmp_path / "cache")
    controller = StudioController(tmp_path / "settings.json", settings=settings, hardware=_cpu_hardware())
    page = ProjectsPage(controller)
    qtbot.addWidget(page)
    try:
        project = controller.create_project(tmp_path, "Disposable")
        assert project.root_dir.resolve().is_relative_to(tmp_path.resolve())
        controller.generate_synthetic_demo_take("disposable")
        monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
        page._delete()
        assert controller.active_motion is None and controller.active_calibration is None
        assert controller.project_manager.active_project is None
        assert controller.network_server.storage_root == Path(settings.storage.cache_dir) / "incoming"
        assert not controller.network_server.bridge_project_path
        assert not controller.network_server.bridge_project_name
        assert not project.root_dir.exists()
    finally:
        controller.shutdown()


def test_model_download_keeps_the_ui_responsive_without_reprobing_hardware(qtbot, qapp, tmp_path, monkeypatch):
    from PySide6.QtCore import QThread, QTimer
    import desktop.ui.pages.pages_export_system as pages
    controller = StudioController(tmp_path / "settings.json", settings=AppSettings(), hardware=_cpu_hardware())
    page = pages.DiagnosticsPage(controller)
    qtbot.addWidget(page)
    gate = threading.Event()
    workers, ticks = [], []
    result = tmp_path / "download.onnx"
    result.write_bytes(b"Verified test result")

    def download(model_id):
        workers.append(QThread.currentThread())
        gate.wait(5)
        return result

    monkeypatch.setattr(controller.model_registry, "ensure_downloaded", download)
    monkeypatch.setattr(pages, "detect_hardware", lambda **kwargs: pytest.fail("Download must not run a hardware probe"))
    timer = QTimer()
    timer.setInterval(5)
    timer.timeout.connect(lambda: ticks.append(1))
    try:
        page.model_table.setCurrentCell(0, 0)
        timer.start()
        page._download_selected_model()
        qtbot.waitUntil(lambda: bool(workers))
        qtbot.waitUntil(lambda: len(ticks) >= 6)
        assert workers[0] != qapp.thread() and not page.btn_download.isEnabled()
        gate.set()
        qtbot.waitUntil(lambda: not page._download_busy)
        assert page.btn_download.isEnabled() and "SHA-256 verified" in page.operation_status.text()
    finally:
        gate.set()
        timer.stop()
        page.shutdown()
        controller.shutdown()


def test_review_save_is_a_pair_and_reloading_retains_unsaved_manual_edits(qtbot, tmp_path, monkeypatch):
    package = ingest_multi_video_source([_write_dummy_mp4(tmp_path / "cam.mp4")])
    workspace = CameraReviewWorkspace(tmp_path / "review")
    qtbot.addWidget(workspace)
    try:
        workspace.load_package(package)
        workspace.state.marker_assignments["4"] = {"name":"Unsaved manual name"}
        workspace.load_package(package)
        assert workspace.state.marker_assignments["4"]["name"] == "Unsaved manual name"
        before = {path: path.read_bytes() for path in (workspace.state_path, workspace.state_path.with_suffix(".npz"))}
        workspace.state.marker_assignments["4"]["name"] = "New unsaved correction"
        with monkeypatch.context() as patch:
            patch.setattr(workspace.observations, "save", lambda path: (_ for _ in ()).throw(OSError("Disk write interrupted")))
            assert workspace.save() is False
            assert all(path.read_bytes() == value for path, value in before.items())
            with pytest.raises(OSError, match="current review"):
                workspace.load_package(package)
        assert workspace.state.marker_assignments["4"]["name"] == "New unsaved correction"
    finally:
        workspace.shutdown()


def test_imported_calibration_invalidates_old_result_and_undo_restores_geometry(qtbot, tmp_path, monkeypatch):
    from multi_video_studio.ui_page import MultiVideoCinemaLabPage
    controller = StudioController(tmp_path / "settings.json", settings=AppSettings(), hardware=_cpu_hardware())
    page = MultiVideoCinemaLabPage(controller, lambda _: None)
    qtbot.addWidget(page)
    try:
        videos = [_write_dummy_mp4(tmp_path / "cam_01.mp4"), _write_dummy_mp4(tmp_path / "cam_02.mp4", n_frames=15)]
        page._load_source(videos)
        original = _calibration(page._package.motion_clips, "Original")
        replacement = _calibration(page._package.motion_clips, "Replacement")
        replacement.cameras["cam_02"].extrinsics.t_cw[0] += .1
        page.review.state.calibration = original.to_dict()
        page.review.calibration = original
        page.review.state.solved_revision = page.review.state.revision
        path = replacement.save(tmp_path / "replacement.json")
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(path), ""))
        page._import_calibration()
        assert page.review.state.stale
        assert page.review.state.calibration["profile_name"] == "Replacement"
        assert page.review.calibration.profile_name == "Replacement"
        page.review.undo()
        assert page.review.state.calibration["profile_name"] == "Original"
        assert page.review.calibration.profile_name == "Original"
    finally:
        page.shutdown()
        controller.shutdown()


def test_camera_refinement_does_not_start_during_alignment(qtbot, tmp_path, monkeypatch):
    from multi_video_studio.ui_page import MultiVideoCinemaLabPage
    controller = StudioController(tmp_path / "settings.json", settings=AppSettings(), hardware=_cpu_hardware())
    page = MultiVideoCinemaLabPage(controller, lambda _: None)
    qtbot.addWidget(page)
    started = threading.Event()
    release = threading.Event()

    def refine(*args):
        started.set()
        release.wait(5)
        raise ValueError("Synthetic worker finished")

    monkeypatch.setattr("multi_video_studio.review_jobs.refine_review_camera", refine)
    controller.active_calibration = SimpleNamespace()
    try:
        page._sync_busy = True
        page._refine_review_camera("cam_02")
        # Waiting on a thread event allows the worker to run without needing UI events.
        assert not started.wait(.12)
        assert not page._review_busy
    finally:
        release.set()
        page.shutdown()
        controller.shutdown()


@pytest.mark.parametrize("options", [["--height", "0.2"], ["--fps", "0"], ["--calibration", "missing.json"]])
def test_cli_invalid_configuration_returns_a_useful_error_without_a_traceback(capsys, options):
    from multi_video_studio.cli import main
    assert main(["process", "input.mp4", *options]) == 1
    assert "Processing failed:" in capsys.readouterr().err


def test_cli_explicit_calibration_cannot_silently_fall_back(tmp_path, monkeypatch):
    import multi_video_studio.cli as cli
    path = _calibration(["cam_01", "cam_02"], "Explicit").save(tmp_path / "profile.json")
    configurations = []

    class Pipeline:
        def __init__(self, config):
            configurations.append(config)

        def run_from_source(self, *args, **kwargs):
            raise RuntimeError("Stopped before inference")

    monkeypatch.setattr(cli, "MultiVideoCinemaPipeline", Pipeline)
    assert cli.main(["process", "input.mp4", "--calibration", str(path)]) == 1
    assert configurations[0].calibration_source == "active_studio"
