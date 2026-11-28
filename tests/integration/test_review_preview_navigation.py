"""Preview navigation preserves source coordinates, timing and capture annotations."""

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog, QTabWidget, QVBoxLayout, QWidget
import shiboken6

from desktop.ui.widgets.camera_review import CameraReviewWorkspace, CameraTile
from desktop.ui.widgets.review_views import PointOverlay
from multi_video_studio.package_ingest import ingest_multi_video_source
from multi_video_studio.pyav_sync import CameraSyncEstimate
from tests.integration.test_multi_video_cinema_studio import _write_dummy_mp4


def _move_drag(widget, start, end):
    QTest.mousePress(widget, Qt.MouseButton.LeftButton, pos=start)
    QApplication.sendEvent(
        widget,
        QMouseEvent(
            QEvent.Type.MouseMove,
            QPointF(end),
            QPointF(widget.mapToGlobal(end)),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        ),
    )
    QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, pos=end)


@pytest.fixture
def review(qtbot, tmp_path):
    videos = [
        _write_dummy_mp4(tmp_path / "a.mp4", n_frames=48),
        _write_dummy_mp4(tmp_path / "b.mp4", n_frames=18, fps=15.0),
    ]
    root = QWidget()
    box = QVBoxLayout(root)
    workspace = CameraReviewWorkspace(tmp_path / "review")
    tabs = QTabWidget()
    for kind in ("cameras", "body", "hands"):
        tabs.addTab(workspace.pages[kind], kind)
    tabs.currentChanged.connect(
        lambda index: workspace.set_active_page(("cameras", "body", "hands")[index])
    )
    box.addWidget(workspace)
    box.addWidget(tabs, 1)
    root.resize(1280, 850)
    qtbot.addWidget(root)
    root.show()
    package = ingest_multi_video_source(videos)
    workspace.load_package(package)
    workspace.set_clocks(
        {
            u: CameraSyncEstimate(u, 0.0, "reference" if i == 0 else "manual", 1.0, "test")
            for i, u in enumerate(package.motion_clips)
        }
    )
    for index, uid in enumerate(package.motion_clips):
        n, fps = (48, 30.0) if index == 0 else (18, 15.0)
        body = np.tile([160.0, 90.0, 0.9], (n, 26, 1)).astype(np.float32)
        hands = np.tile([100.0, 70.0, 0.9], (n, 21, 1)).astype(np.float32)
        hands[:, :, 0] += np.arange(21) * 3
        hands[:, :, 1] += (np.arange(21) % 4) * 8
        model = np.zeros((n, 21, 3), np.float32)
        model[:, :, 0] = np.arange(21) * 0.003
        model[:, :, 1] = (np.arange(21) % 4) * 0.01
        workspace.observations.set_track(uid, np.arange(n) / fps, body, hands, hands, model, model)
    tabs.setCurrentIndex(2)
    workspace.time_slider.setValue(133)
    qtbot.waitUntil(
        lambda: len(workspace.last_frames) == 2 and all(workspace.last_frames.values()),
        timeout=5000,
    )
    yield workspace, tabs, package
    workspace.shutdown()


def test_video_zoom_survives_frames_and_native_crop_mapping(qtbot):
    view = PointOverlay()
    qtbot.addWidget(view)
    view.resize(600, 400)
    view.show()
    image = np.zeros((180, 320, 3), np.uint8)
    view.frame(image, (640, 360), {})
    view.set_zoom_percent(200)
    view.centerOn(350, 190)
    center = view.mapToScene(view.viewport().rect().center())
    for _ in range(3):
        view.frame(image, (640, 360), {})
    assert view.zoom_percent() == pytest.approx(200)
    assert (view.mapToScene(view.viewport().rect().center()) - center).manhattanLength() < 2
    view.resize(720, 460)
    QApplication.processEvents()
    assert view.zoom_percent() == pytest.approx(200)
    transform = np.array([[4.0, 0.0, -100.0], [0.0, 4.0, -200.0]])
    view.frame(image, (640, 360), {}, transform=transform)
    view.set_zoom_percent(100)
    assert view.transform().m11() * view.native_to_scene.m11() == pytest.approx(1.0)
    edits = []
    view.edited.connect(lambda key, xy: edits.append(xy))
    view._point_event("hand:left:8", view.native_to_scene.map(QPointF(85.0, 90.0)))
    np.testing.assert_allclose(edits[-1], [85.0, 90.0])
    view.placeholder("Outside this recording")
    view.frame(image, (640, 360), {}, transform=transform)
    assert view.zoom_percent() == pytest.approx(100)


def test_navigation_does_not_edit_and_drag_keeps_original_source_frame(qtbot):
    tile = CameraTile("cam_01")
    qtbot.addWidget(tile)
    tile.resize(640, 450)
    tile.show()
    tile.pts_s = 0.1
    overlay = tile.overlay
    image = np.zeros((360, 640, 3), np.uint8)
    points = {"body:9": {"xy": [320.0, 180.0], "name": "Left wrist", "source": "detected"}}
    overlay.frame(image, (640, 360), points)
    overlay.set_zoom_percent(200)
    edits = []
    tile.point_edited.connect(lambda uid, key, pts, xy: edits.append((pts, xy)))
    start = overlay.mapFromScene(QPointF(320, 180))
    _move_drag(overlay.viewport(), start, start + QPoint(25, 8))
    assert edits == []
    overlay.set_editing(True)
    overlay.centerOn(320, 180)
    start = overlay.mapFromScene(QPointF(320, 180))
    QTest.mousePress(overlay.viewport(), Qt.MouseButton.LeftButton, pos=start)
    mark = overlay._points["body:9"]
    tile.pts_s = 0.2
    overlay.frame(image, (640, 360), {})
    assert overlay._points["body:9"] is mark
    end = start + QPoint(20, 0)
    QApplication.sendEvent(
        overlay.viewport(),
        QMouseEvent(
            QEvent.Type.MouseMove,
            QPointF(end),
            QPointF(overlay.viewport().mapToGlobal(end)),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        ),
    )
    QTest.mouseRelease(overlay.viewport(), Qt.MouseButton.LeftButton, pos=end)
    assert len(edits) == 1 and edits[0][0] == pytest.approx(0.1)
    np.testing.assert_allclose(edits[0][1], [330.0, 180.0], atol=1.0)


def test_independent_windows_selection_transport_and_snapshot(review, qtbot, tmp_path, monkeypatch):
    workspace, tabs, package = review
    uid = next(iter(package.motion_clips))
    tile = workspace.tiles[("hands", uid, "left")]
    original = tile.overlay
    revision = workspace.state.revision
    raw = workspace.observations.cameras[uid]["hand:left"].copy()
    QTest.mouseClick(tile.video_panel.title, Qt.MouseButton.LeftButton)
    assert original.hasFocus()
    tile.video_panel.pop_out()
    video_window = tile.video_panel.floating_window
    tile.hand_panel.pop_out()
    model_window = tile.hand_panel.floating_window
    assert video_window is not model_window and original is tile.video_panel.content
    tile.video_panel.activate()
    assert tile.video_panel.surface.property("previewSelected")
    assert not tile.hand_panel.surface.property("previewSelected")
    original.set_zoom_percent(250)
    tile.hand_view.set_zoom_percent(175)
    assert tile.hand_view.zoom_percent() == pytest.approx(175)
    assert workspace.tiles[("hands", uid, "right")].overlay.zoom_percent() != pytest.approx(250)
    tile.video_panel.transport.slider.setValue(300)
    assert workspace.time_slider.value() == 300
    qtbot.waitUntil(
        lambda: workspace.last_frames[uid] and abs(workspace.last_frames[uid].pts_s - 0.3) < 0.001
    )
    tile.video_panel.transport.play.click()
    assert workspace.timer.isActive()
    workspace.pause()
    QTest.keyClick(original, Qt.Key.Key_Space)
    assert workspace.timer.isActive()
    workspace.pause()
    assert tile.hand_panel.transport.play.text() == "Play"
    tile.video_panel.transport.edit_toggle.setChecked(True)
    assert workspace.edit_points.isChecked() and original.editing
    workspace.place_hand_point.setChecked(True)
    tile.video_panel.transport.edit_toggle.setChecked(False)
    assert not original.editing and not workspace.place_hand_point.isChecked()
    tabs.setCurrentIndex(0)
    assert video_window.isVisible() and model_window.isVisible()
    workspace.time_slider.setValue(400)
    qtbot.waitUntil(lambda: abs(tile.pts_s - 0.4) < 0.001)
    target = tmp_path / "view.png"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(target), "PNG image")
    )
    tile.video_panel.save_snapshot()
    assert target.stat().st_size > 100
    video_window.close()
    model_window.close()
    assert tile.video_panel.floating_window is None and tile.hand_panel.floating_window is None
    assert original.parentWidget() is tile.video_panel.surface
    assert original.zoom_percent() == pytest.approx(250)
    assert workspace.state.revision == revision and workspace.state.camera_policy == {}
    np.testing.assert_array_equal(workspace.observations.cameras[uid]["hand:left"], raw)


def test_fullscreen_returns_the_same_view_and_filtering_keeps_zoom(review, qtbot):
    workspace, tabs, package = review
    uid = next(iter(package.motion_clips))
    tile = workspace.tiles[("hands", uid, "left")]
    tile.overlay.set_zoom_percent(300)
    tile.video_panel.toggle_fullscreen()
    qtbot.waitUntil(lambda: tile.video_panel.floating_window.isFullScreen())
    QTest.keyClick(tile.video_panel.floating_window, Qt.Key.Key_Escape)
    assert tile.video_panel.floating_window is None
    assert tile.overlay.zoom_percent() == pytest.approx(300)
    workspace.preview_mode.setCurrentText("One camera")
    workspace.preview_mode.setCurrentText("All cameras")
    assert workspace.tiles[("hands", uid, "left")] is tile
    assert tile.overlay.zoom_percent() == pytest.approx(300)
    tile.video_panel.pop_out()
    window = tile.video_panel.floating_window
    tile.video_panel.toggle_fullscreen()
    QTest.keyClick(window, Qt.Key.Key_Escape)
    assert tile.video_panel.floating_window is window and not window.isFullScreen()
    workspace.return_all_views()


def test_floating_camera_stays_live_outside_the_preview_subset(review, qtbot):
    workspace, tabs, package = review
    first, second = package.motion_clips
    tile = workspace.tiles[("hands", second, "left")]
    tile.video_panel.pop_out()
    workspace.camera_pick.setCurrentIndex(workspace.camera_pick.findData(first))
    workspace.preview_mode.setCurrentText("One camera")
    assert workspace.visible_cameras() == [first]
    assert second in workspace._requested_cameras()
    workspace.time_slider.setValue(600)
    qtbot.waitUntil(lambda: abs(tile.pts_s - 0.6) < 0.001)
    assert tile.video_panel.floating_window.isVisible()
    assert workspace.state.camera_policy == {}


def test_native_frame_steps_and_missing_clip_feedback(review, qtbot):
    workspace, tabs, package = review
    second = list(package.motion_clips)[1]
    tile = workspace.tiles[("hands", second, "left")]
    tile.video_panel.activate()
    workspace.time_slider.setValue(133)
    workspace.step_frame(1)
    assert workspace.global_s == pytest.approx(0.2, abs=0.001)
    workspace.step_frame(-1)
    assert workspace.global_s == pytest.approx(2 / 15, abs=0.001)
    workspace.time_slider.setValue(1400)
    qtbot.waitUntil(lambda: "Outside" in tile.overlay.message.text())
    assert "unavailable" in tile.status.text() or "no source frame" in tile.status.text()
    workspace.step_frame(-1)
    assert workspace.global_s < 1.2


def test_layout_restores_and_replacing_take_closes_old_windows(review, qtbot, tmp_path):
    workspace, tabs, package = review
    uid = next(iter(package.motion_clips))
    tile = workspace.tiles[("hands", uid, "left")]
    tile.overlay.set_zoom_percent(225)
    tile.video_panel.pop_out()
    tile.video_panel.floating_window.resize(850, 600)
    workspace.save_preview_layout()
    stored = json.loads((workspace.state_root / "preview_layout.json").read_text())
    state = stored["sources"][workspace._source_key]["tiles"][
        workspace._tile_key(("hands", uid, "left"))
    ]
    assert state["video"]["floating"] and state["video"]["navigation"]["zoom"] == pytest.approx(225)
    workspace.load_package(package)
    replacement = workspace.tiles[("hands", uid, "left")]
    qtbot.waitUntil(lambda: replacement.video_panel.floating_window is not None)
    assert replacement.overlay.zoom_percent() == pytest.approx(225)
    old_window = replacement.video_panel.floating_window
    new_video = _write_dummy_mp4(tmp_path / "new.mp4", n_frames=24)
    workspace.load_package(ingest_multi_video_source([new_video]))
    qtbot.waitUntil(lambda: not shiboken6.isValid(old_window))
    assert not any(tile.is_floating() for tile in workspace.tiles.values())


def test_grid_rows_resize_at_small_window_and_save_separately(review, qtbot):
    workspace, tabs, package = review
    workspace.parentWidget().resize(1100, 600)
    workspace.layout_columns.setCurrentIndex(workspace.layout_columns.findData(2))
    qtbot.wait(80)
    grid = workspace.grids["hands"]
    before = grid.splitter.sizes()
    assert len(before) >= 2
    target = before.copy()
    target[0] += 30
    target[1] -= 30
    grid.splitter.setSizes(target)
    after = grid.splitter.sizes()
    assert after[0] > before[0] + 10
    revision = workspace.state.revision
    workspace.save_preview_layout()
    saved = json.loads((workspace.state_root / "preview_layout.json").read_text())
    assert saved["sources"][workspace._source_key]["grids"]["hands"]["sizes"]["vertical"] == after
    assert workspace.state.revision == revision
