import os

import numpy as np

from multi_video_studio.package_ingest import ingest_multi_video_source, export_take_to_zip
from multi_video_studio.pyav_sync import CameraSyncEstimate
from tests.integration.test_multi_video_cinema_studio import _write_dummy_mp4


def test_review_edit_crop_portable_package_and_selection_are_independent(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtTest import QTest
    from desktop.ui.widgets.camera_review import CameraReviewWorkspace
    app = QApplication.instance() or QApplication([])
    videos = [_write_dummy_mp4(tmp_path / "cam_01.mp4"), _write_dummy_mp4(tmp_path / "cam_02.mp4", n_frames=15)]
    package = ingest_multi_video_source(videos)
    workspace = CameraReviewWorkspace(tmp_path / "state")
    try:
        workspace.load_package(package)
        workspace.set_clocks({u:CameraSyncEstimate(u,0.,"reference" if i==0 else "manual",1.,"test")
                              for i,u in enumerate(package.motion_clips)})
        uid = next(iter(package.motion_clips))
        pts = np.arange(12) / 30
        body = np.zeros((12,26,3),np.float32)
        hand = np.tile([160.,90.,.9],(12,21,1)).astype(np.float32)
        hand[:,:,0] += np.arange(21)
        workspace.observations.set_track(uid,pts,body,hand,hand)
        before = hand.copy()
        workspace.preview_mode.setCurrentText("One camera")
        assert workspace.visible_cameras() == [uid]
        assert workspace.state.camera_policy == {}
        workspace.selected_cameras.item(1).setCheckState(Qt.CheckState.Unchecked)
        assert workspace.preview_mode.currentText() == "Selected cameras"
        assert workspace.visible_cameras() == [uid]
        assert workspace.state.camera_policy == {}
        workspace.set_active_page("hands")
        workspace.time_slider.setValue(100)
        for _ in range(15):
            app.processEvents()
            QTest.qWait(40)
            if workspace.last_frames:
                break
        tile = workspace.tiles[("hands",uid,"left")]
        assert tile.overlay._has_frame
        tile.overlay._point_event("hand:left:8",tile.overlay.native_to_scene.map(QPointF(185.,95.)))
        point = workspace.state.points[-1]
        assert point["key"] == "hand:left:8"
        np.testing.assert_allclose(point["xy"],[185.,95.],atol=.01)
        assert point["frame_index"] == 3
        np.testing.assert_array_equal(workspace.observations.cameras[uid]["hand:left"],before)
        workspace.use_markers.setChecked(True)
        workspace.marker_only.setChecked(True)
        assert workspace.state.capture_mode == "markers"
        workspace.marker_only.setChecked(False)
        assert workspace.state.capture_mode == "hybrid"
        workspace.save()
        archive = export_take_to_zip(package.motion_paths_by_cam(),tmp_path/"portable.zip",
            review_json=workspace.state_path, review_observations=workspace.state_path.with_suffix(".npz"))
        restored_package = ingest_multi_video_source(archive)
        restored = CameraReviewWorkspace(tmp_path/"another_machine")
        try:
            restored.load_package(restored_package)
            assert restored.state.capture_mode == "hybrid"
            np.testing.assert_allclose(restored.state.points[-1]["xy"],[185.,95.],atol=.01)
            assert restored.observations.cameras
        finally:
            restored.shutdown()
    finally:
        workspace.shutdown()
