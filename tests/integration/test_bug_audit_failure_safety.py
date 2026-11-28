"""Reproductions from the failure-path audit; completed work must survive failures."""

import hashlib
import json
from pathlib import Path
import threading
import zipfile

import numpy as np
import pytest
from PySide6.QtCore import QThread, QTimer

from desktop.project_manager.manager import ProjectManager
from desktop.settings.app_settings import AppSettings
from engine.data_format.motion_io import (
    MotionCaptureDocument, SkeletalMotionTrack, load_motion_file, save_motion_file,
)
from engine.skeleton.rig_definition import (
    BONE_NAMES, PARENT_INDICES, DEFAULT_REST_OFFSETS_M, compute_forward_kinematics_sequence,
)
from model_registry.registry import ModelRegistry
from multi_video_studio.package_ingest import ingest_multi_video_source
from tests.integration.test_multi_video_cinema_studio import _write_dummy_mp4
from tests.integration.test_startup_and_motion_ui import _cpu_hardware


def _document():
    root = np.tile([0., 0., .96], (8, 1))
    rotations = np.tile([0., 0., 0., 1.], (8, len(BONE_NAMES), 1))
    positions, matrices = compute_forward_kinematics_sequence(root, rotations, DEFAULT_REST_OFFSETS_M)
    track = SkeletalMotionTrack(root, rotations[:, 0], rotations, matrices, positions,
                                np.ones((8, len(BONE_NAMES))))
    return MotionCaptureDocument("audit", "audit", "audit", "audit", 30.,
        (np.arange(8) / 30 * 1e9).astype(np.int64), BONE_NAMES, PARENT_INDICES,
        DEFAULT_REST_OFFSETS_M, track)


def _interrupt_text_write(monkeypatch, filename):
    original = Path.open

    class Interrupted:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def write(self, value):
            self.stream.write(value[:10])
            raise OSError("Injected interrupted disk write")

        def __getattr__(self, name):
            return getattr(self.stream, name)

    def open_file(path, mode="r", *args, **kwargs):
        stream = original(path, mode, *args, **kwargs)
        return Interrupted(stream) if "w" in mode and filename in path.name else stream

    monkeypatch.setattr(Path, "open", open_file)


@pytest.mark.parametrize("kind", ["settings", "project"])
def test_interrupted_metadata_write_preserves_the_last_complete_file(tmp_path, monkeypatch, kind):
    manager = ProjectManager(tmp_path / "recent.json")
    try:
        if kind == "settings":
            settings = AppSettings()
            path = settings.save(tmp_path / "settings.json")
            settings.general.theme = "light"
            def save():
                return settings.save(path)
        else:
            project = manager.create_project(tmp_path, "Original")
            path = project.manifest_path
            project.name = "Changed"
            save = manager.save_project
        before = path.read_bytes()
        _interrupt_text_write(monkeypatch, path.name)
        with pytest.raises(OSError, match="interrupted"):
            save()
        assert path.read_bytes() == before
        json.loads(path.read_text())
    finally:
        manager.close_project()


def test_failed_motion_serialization_preserves_hdf5_and_manifest(tmp_path):
    document = _document()
    path = save_motion_file(document, tmp_path / "last.nmocap.h5")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    document.processing_config["invalid"] = object()
    with pytest.raises(TypeError):
        save_motion_file(document, path)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()} == before
    assert load_motion_file(path).num_frames == 8


def test_failed_later_export_preserves_the_entire_previous_delivery(tmp_path, monkeypatch):
    import multi_video_studio.pipeline as pipeline
    processor = pipeline.MultiVideoCinemaPipeline(pipeline.CinemaPipelineConfig())
    document = _document()
    processor._save_exports(document, tmp_path, "take")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    document.processing_config["new_run"] = True

    def fail(*args, **kwargs):
        raise OSError("Injected JSON export failure")

    monkeypatch.setattr(pipeline, "export_blender_json", fail)
    with pytest.raises(OSError, match="export failure"):
        processor._save_exports(document, tmp_path, "take")
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()} == before


def test_failed_duplicate_retains_active_project_and_descendant_is_rejected(tmp_path, monkeypatch):
    manager = ProjectManager(tmp_path / "recent.json")
    try:
        source = manager.create_project(tmp_path, "Source")
        (tmp_path / "Existing").mkdir()
        with pytest.raises(FileExistsError):
            manager.duplicate_project(tmp_path, "Existing")
        assert manager.active_project is source
        called = []

        def prevent_recursive_copy(*args, **kwargs):
            called.append(args)
            raise ValueError("Recursive copy must never start")

        monkeypatch.setattr("desktop.project_manager.manager.shutil.copytree", prevent_recursive_copy)
        with pytest.raises(ValueError):
            manager.duplicate_project(source.root_dir, "Nested")
        assert not called and manager.active_project is source
    finally:
        manager.close_project()


@pytest.mark.parametrize("existing_destination", [False, True])
def test_restore_validates_before_touching_destination_or_active_project(tmp_path, existing_destination):
    manager = ProjectManager(tmp_path / "recent.json")
    try:
        source = manager.create_project(tmp_path, "Source")
        archive = tmp_path / "restore.zip"
        destination = tmp_path / "Restored"
        destination.mkdir()
        if existing_destination:
            (destination / "keep.txt").write_text("Existing user content")
        before = {p.name: p.read_bytes() for p in destination.iterdir()}
        with zipfile.ZipFile(archive, "w") as stream:
            if existing_destination:
                stream.writestr("project.nmocap", source.manifest_path.read_bytes())
            stream.writestr("new.txt", "Must not leak from a failed restore")
        with pytest.raises((FileExistsError, FileNotFoundError, ValueError)):
            manager.import_or_restore_project_zip(archive, destination)
        assert {p.name: p.read_bytes() for p in destination.iterdir()} == before
        assert manager.active_project is source
    finally:
        manager.close_project()


@pytest.mark.parametrize("fail_write", [False, True])
def test_project_archive_excludes_itself_and_failed_export_keeps_old_archive(tmp_path, monkeypatch, fail_write):
    manager = ProjectManager(tmp_path / "recent.json")
    try:
        source = manager.create_project(tmp_path, "Source")
        output = source.root_dir / "portable.zip"
        output.write_bytes(b"Previous completed archive")
        original = zipfile.ZipFile.write
        self_included = []

        def write(stream, filename, *args, **kwargs):
            if Path(filename).resolve() == output.resolve():
                self_included.append(filename)
                return  # Prevent reading a ZIP file while it is growing.
            if fail_write:
                raise OSError("Injected archive write failure")
            return original(stream, filename, *args, **kwargs)

        monkeypatch.setattr(zipfile.ZipFile, "write", write)
        if fail_write:
            with pytest.raises(OSError, match="archive write failure"):
                manager.backup_or_export_project_zip(output)
            assert output.read_bytes() == b"Previous completed archive"
        else:
            manager.backup_or_export_project_zip(output)
            assert not self_included
            with zipfile.ZipFile(output) as stream:
                assert stream.testzip() is None and "project.nmocap" in stream.namelist()
    finally:
        manager.close_project()


@pytest.mark.parametrize("bad_cache", [False, True])
def test_model_download_checks_hash_before_publishing_and_recovers_bad_cache(tmp_path, bad_cache):
    source = tmp_path / "source.onnx"
    source.write_bytes(b"known good model" * 150)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    registry_path = tmp_path / "models.json"
    registry_path.write_text(json.dumps({"models": [{"id":"pose", "family":"test", "task":"pose",
        "version":"test", "url":source.as_uri(), "license":"test", "sha256":digest}]}))
    registry = ModelRegistry(registry_path, tmp_path / "cache")
    destination = registry.get_local_path("pose")
    if bad_cache:
        destination.write_bytes(b"corrupt cached model" * 150)
        assert registry.ensure_downloaded("pose").read_bytes() == source.read_bytes()
    else:
        source.write_bytes(b"wrong server response" * 150)
        with pytest.raises(ValueError, match="Checksum mismatch"):
            registry.ensure_downloaded("pose")
        assert not destination.exists()


def test_malformed_checksum_cache_does_not_prevent_startup(tmp_path):
    registry_path = tmp_path / "models.json"
    registry_path.write_text(json.dumps({"models": [{"id":"pose", "family":"test", "task":"pose",
        "version":"test", "url":"https://example.invalid/pose.onnx", "license":"test"}]}))
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "checksums.json").write_text("[]")
    assert ModelRegistry(registry_path, cache).list_models()[0].id == "pose"


def test_invalid_review_import_preserves_previous_source_views_and_edits(qtbot, tmp_path):
    from desktop.ui.widgets.camera_review import CameraReviewWorkspace
    from multi_video_studio.review_state import source_fingerprint
    first = ingest_multi_video_source([_write_dummy_mp4(tmp_path / "first.mp4")])
    second = ingest_multi_video_source([_write_dummy_mp4(tmp_path / "second.mp4", w=352)])
    workspace = CameraReviewWorkspace(tmp_path / "review")
    qtbot.addWidget(workspace)
    try:
        workspace.load_package(first)
        workspace.state.marker_assignments["3"] = {"name":"Saved manual assignment"}
        workspace.save()
        state, old_path = workspace.state, workspace.state_path
        old_view = next(iter(workspace.tiles.values())).overlay
        identity = "_".join(sorted(source_fingerprint(c.path)[:12] for c in second.motion_clips.values()))
        corrupt_path = workspace.state_root / (identity + ".review.json")
        corrupt_path.write_text("{incomplete JSON")
        with pytest.raises(ValueError):
            workspace.load_package(second)
        assert workspace.package is first and workspace.state is state
        assert workspace.state_path == old_path
        assert next(iter(workspace.tiles.values())).overlay is old_view
        assert corrupt_path.read_text() == "{incomplete JSON"
    finally:
        workspace.shutdown()


def test_diagnostics_reuses_startup_probe_and_background_refresh_stays_responsive(qtbot, qapp, tmp_path, monkeypatch):
    import desktop.ui.pages.pages_export_system as pages
    from desktop.ui.app_controller import StudioController
    controller = StudioController(tmp_path / "settings.json", settings=AppSettings(), hardware=_cpu_hardware())
    gate = threading.Event()
    probes = []

    def probe(**kwargs):
        probes.append(QThread.currentThread())
        gate.wait(5)
        return _cpu_hardware()

    monkeypatch.setattr(pages, "detect_hardware", lambda **kwargs: probes.append(QThread.currentThread()) or _cpu_hardware())
    page = pages.DiagnosticsPage(controller)
    qtbot.addWidget(page)
    heartbeat = QTimer()
    ticks = []
    heartbeat.setInterval(5)
    heartbeat.timeout.connect(lambda: ticks.append(1))
    try:
        assert not probes, "Startup already detected the hardware; page construction must not probe again."
        monkeypatch.setattr(pages, "detect_hardware", probe)
        heartbeat.start()
        page.refresh()
        qtbot.waitUntil(lambda: bool(probes))
        qtbot.waitUntil(lambda: len(ticks) >= 6)
        assert probes[0] != qapp.thread()
        gate.set()
        qtbot.waitUntil(lambda: not page._probe_busy)
    finally:
        gate.set()
        heartbeat.stop()
        if hasattr(page, "shutdown"):
            page.shutdown()
        controller.shutdown()
