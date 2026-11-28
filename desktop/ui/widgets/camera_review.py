"""Synchronized camera/point/hand review using source PTS and reversible annotations."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import hashlib
import json
import queue
import threading
import tempfile
import time

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtGui import QActionGroup
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout,
    QInputDialog, QLabel, QListWidget, QListWidgetItem, QMenu, QMessageBox, QPushButton,
    QScrollArea, QSlider, QVBoxLayout, QWidget, QWidgetAction,
    QSplitter, QToolButton,
    QFrame, QSizePolicy,
    QStyle,
)

from desktop.ui.widgets.review_panels import ElidedLabel, PreviewPanel, ReviewGrid
from desktop.ui.widgets.review_views import PointOverlay, HandPoseView

from engine.hands.kinematics import hand_forward_kinematics
from multi_video_studio.review_observations import ReviewObservations
from multi_video_studio.review_state import BODY_POINT_NAMES, HAND_POINT_NAMES, ReviewState, point_name
from multi_video_studio.review_video import ReviewVideoReader, hand_crop
from shared_protocol.atomic_io import publish_files

HAND_EDGES = [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17)] + [
    (i, i + 1) for start in (1, 5, 9, 13, 17) for i in range(start, start + 3)
]
BODY_EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
              (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (15, 20), (16, 21)]


class PreviewDecoder(QObject):
    ready = Signal(object, object)

    def __init__(self):
        super().__init__()
        self.jobs = queue.Queue(maxsize=1)
        self.stopped = threading.Event()
        self.worker = threading.Thread(target=self._run, name="camera-review-decoder", daemon=True)
        self.frame_times = {}
        self.worker.start()

    def request(self, generation, requests):
        try:
            self.jobs.get_nowait()
        except queue.Empty:
            pass
        self.jobs.put_nowait((generation, requests))

    def _run(self):
        readers = OrderedDict()
        epoch = -1
        try:
            while not self.stopped.is_set():
                try:
                    generation, requests = self.jobs.get(timeout=.1)
                except queue.Empty:
                    continue
                if generation >> 32 != epoch:
                    for reader in readers.values():
                        reader.close()
                    readers.clear()
                    self.frame_times = {}
                    epoch = generation >> 32
                    self.frame_epoch = epoch
                output = {}
                for uid, path, pts, max_width in requests:
                    if self.stopped.is_set():
                        break
                    try:
                        key = str(path)
                        if key not in readers:
                            readers[key] = ReviewVideoReader(Path(path), cache_frames=2)
                        self.frame_times[uid] = readers[key].presentation_times
                        readers.move_to_end(key)
                        output[uid] = readers[key].at(pts, max_width=max_width)
                        while len(readers) > 12:
                            _, old = readers.popitem(last=False)
                            old.close()
                    except (OSError, ValueError, RuntimeError, StopIteration):
                        output[uid] = None
                if not self.stopped.is_set():
                    self.ready.emit(generation, output)
        finally:
            for reader in readers.values():
                reader.close()

    def close(self):
        self.stopped.set()
        self.worker.join(timeout=2)


class CameraTile(QWidget):
    point_selected = Signal(str, str, float)
    point_edited = Signal(str, str, float, object)

    def __init__(self, uid, side=None):
        super().__init__()
        self.uid, self.side, self.pts_s = uid, side, 0.
        self.full_frame = False
        self.model_source = -1
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.overlay = PointOverlay()
        title = uid + (" · " + side.capitalize() + " hand" if side else "")
        self.video_panel = PreviewPanel(self.overlay, title + " · Video")
        compact = uid + (" · "+side[0].upper() if side else "")
        self.video_panel.title.setText(compact+" · Video")
        self.status = self.video_panel.status
        self.video_panel.set_status("Waiting for frame")
        self.hand_view = HandPoseView() if side else None
        self.hand_panel = None
        if self.hand_view:
            self.hand_panel = PreviewPanel(self.hand_view, title + " · 3D")
            self.hand_panel.title.setText(compact+" · 3D")
            self.views = QSplitter(Qt.Orientation.Horizontal)
            self.views.setChildrenCollapsible(False)
            self.views.setHandleWidth(6)
            self.views.addWidget(self.video_panel)
            self.views.addWidget(self.hand_panel)
            self.views.setStretchFactor(0, 3)
            self.views.setStretchFactor(1, 2)
            layout.addWidget(self.views, 1)
        else:
            self.views = None
            layout.addWidget(self.video_panel, 1)
        self._selected_pts = None
        self.overlay.selected.connect(self._select_overlay_point)
        self.overlay.edited.connect(lambda key,xy:self.point_edited.emit(self.uid,key,
            self._selected_pts if self.overlay._dragging_point and self._selected_pts is not None else self.pts_s,xy))

    def _select_overlay_point(self, key):
        self._selected_pts = self.pts_s
        self.point_selected.emit(self.uid,key,self.pts_s)

    def panels(self):
        return [self.video_panel] + ([self.hand_panel] if self.hand_panel else [])

    def is_floating(self):
        return any(panel.floating_window is not None for panel in self.panels())


class ReviewTransport(QWidget):
    """A floating window controls the same clock as the main review page."""

    def __init__(self, workspace, parent=None):
        super().__init__(parent)
        self.workspace = workspace
        row = QHBoxLayout(self)
        row.setContentsMargins(3, 2, 3, 2)
        self.play = QPushButton("Pause" if workspace.timer.isActive() else "Play")
        self.play.clicked.connect(workspace._toggle_play)
        previous, following = QToolButton(), QToolButton()
        previous.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipBackward))
        following.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipForward))
        previous.setAccessibleName("Previous source frame")
        following.setAccessibleName("Next source frame")
        previous.setToolTip("Previous source frame (Left)")
        following.setToolTip("Next source frame (Right)")
        previous.clicked.connect(lambda: workspace.step_frame(-1))
        following.clicked.connect(lambda: workspace.step_frame(1))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(workspace.time_slider.minimum(), workspace.time_slider.maximum())
        self.slider.setValue(workspace.time_slider.value())
        self.slider.valueChanged.connect(workspace.time_slider.setValue)
        self.slider.sliderPressed.connect(workspace.pause)
        self.stamp = QLabel(f"{workspace.global_s:.3f}s")
        for widget in (self.play, previous, following):
            row.addWidget(widget)
        row.addWidget(self.slider, 1)
        row.addWidget(self.stamp)
        self.edit_toggle = None
        if getattr(parent.panel.content,"supports_actual_size",False):
            self.edit_toggle = QCheckBox("Edit points")
            self.edit_toggle.setChecked(workspace.edit_points.isChecked())
            self.edit_toggle.toggled.connect(workspace.edit_points.setChecked)
            workspace.edit_points.toggled.connect(self.edit_toggle.setChecked)
            row.addWidget(self.edit_toggle)
        return_button = QPushButton("Return to grid")
        return_button.clicked.connect(lambda: parent.panel.return_to_grid())
        row.addWidget(return_button)
        workspace.time_changed.connect(self._time_changed)
        workspace.playback_changed.connect(self._playing_changed)
        workspace.time_slider.rangeChanged.connect(self.slider.setRange)

    def _time_changed(self, seconds):
        self.slider.blockSignals(True)
        self.slider.setValue(round(seconds*1000))
        self.slider.blockSignals(False)
        self.stamp.setText(f"{seconds:.3f}s")

    def _playing_changed(self, playing):
        self.play.setText("Pause" if playing else "Play")


class CameraReviewWorkspace(QWidget):
    changed = Signal()
    time_changed = Signal(float)
    playback_changed = Signal(bool)
    align_requested = Signal()
    detection_requested = Signal(object, float, float, bool)
    recalibrate_requested = Signal(str)

    def __init__(self, state_root: Path):
        super().__init__()
        self.state_root = state_root
        self.state = ReviewState()
        self.observations = ReviewObservations()
        self.package = self.document = self.calibration = None
        self.state_path = None
        self.global_s = 0.
        self.last_frames = {}
        self.selected_point = None
        self.generation = 0
        self.last_generation = -1
        self.preview_epoch = 0
        self.active_page = "cameras"
        self.crop_sizes = {}
        self.pages, self.grids, self.tiles = {}, {}, {}
        self._grid_scrolls = {}
        self._detection_scopes = {}
        self._source_key = None
        self._active_view = None
        self._layout_loading = False
        self._shutting_down = False
        self._preview_state = {}
        self._restored_grids = set()
        self._external_panels = {}
        self._preview_store = self._read_preview_preferences()
        self._native_requests = {}
        self._layout_timer = QTimer(self)
        self._layout_timer.setSingleShot(True)
        self._layout_timer.setInterval(400)
        self._layout_timer.timeout.connect(self.save_preview_layout)
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(30)
        self._resize_timer.timeout.connect(self._refresh_grid_sizes)
        self.decoder = PreviewDecoder()
        self.decoder.ready.connect(self._frames_ready)
        self.timer = QTimer(self)
        self.timer.setInterval(70)
        self.timer.timeout.connect(self._tick)
        self.play_origin = 0.
        self.play_started = 0.
        self._loading = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.preview_mode = QComboBox()
        for name in ("All cameras", "One camera", "Selected cameras"):
            self.preview_mode.addItem(name)
        self.camera_pick = QComboBox()
        self.camera_pick.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.camera_pick.setMinimumContentsLength(12)
        self.camera_pick.setMaximumWidth(240)
        self.selected_cameras = QListWidget()
        self.selected_cameras.setFixedSize(230, 160)
        self.camera_selector = QPushButton("Select cameras")
        self.camera_selector.setToolTip("Choose a preview subset; solve inclusion stays unchanged.")
        camera_menu = QMenu(self.camera_selector)
        camera_action = QWidgetAction(camera_menu)
        camera_action.setDefaultWidget(self.selected_cameras)
        camera_menu.addAction(camera_action)
        self.camera_selector.setMenu(camera_menu)
        self.use_markers = QCheckBox("Use physical markers")
        self.use_markers.setToolTip("Use printed AprilTag 36h11 squares with unique IDs and measured sizes on the performer.")
        self.marker_only = QCheckBox("Markers only")
        self.marker_only.setToolTip("Requires at least two calibrated cameras and enough visible tags with calibrated joint offsets.")
        self.mode_note = ElidedLabel("AI tracking")
        self.mode_note.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Preferred)
        self.mode_note.setMinimumWidth(50)
        self.mode_note.setMaximumWidth(150)
        row.addWidget(QLabel("Preview"))
        row.addWidget(self.preview_mode)
        row.addWidget(self.camera_pick)
        row.addWidget(self.camera_selector)
        row.addStretch()
        root.addLayout(row)
        row.addWidget(self.use_markers)
        row.addWidget(self.marker_only)
        row.addWidget(self.mode_note,1)
        self.save_button = QPushButton("Save review")
        row.addWidget(self.save_button)
        timeline = QHBoxLayout()
        self.play_button = QPushButton("Play cameras")
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, 0)
        self.time_label = QLabel("Add videos")
        self.status = QLabel("Timing and calibration review")
        timeline.addWidget(self.play_button)
        previous, following = QToolButton(), QToolButton()
        previous.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipBackward))
        following.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipForward))
        previous.setAccessibleName("Previous source frame")
        following.setAccessibleName("Next source frame")
        previous.setToolTip("Previous source frame (Left; Shift+Left for ten)")
        following.setToolTip("Next source frame (Right; Shift+Right for ten)")
        previous.clicked.connect(lambda: self.step_frame(-1))
        following.clicked.connect(lambda: self.step_frame(1))
        timeline.addWidget(previous)
        timeline.addWidget(following)
        timeline.addWidget(self.time_slider, 1)
        timeline.addWidget(self.time_label)
        root.addLayout(timeline)
        root.addWidget(self.status)
        tools = QHBoxLayout()
        tools.setSpacing(8)
        self.active_view_label = ElidedLabel("Select a preview")
        self.active_view_label.setMinimumWidth(65)
        self.active_view_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        tools.addWidget(self.active_view_label, 1)
        self.layout_columns = QComboBox()
        for label, count in (("Auto layout",0),("1 column",1),("2 columns",2),("3 columns",3)):
            self.layout_columns.addItem(label,count)
        self.layout_columns.setToolTip("Arrange previews. Drag the dividers to resize rows and columns.")
        self.layout_columns.currentIndexChanged.connect(self._refresh_grid_sizes)
        tools.addWidget(self.layout_columns)
        self.overlays_visible = QCheckBox("Overlays")
        self.overlays_visible.setChecked(True)
        self.labels_visible = QCheckBox("Names")
        self.edit_points = QCheckBox("Edit points (E)")
        self.edit_points.setToolTip("Off: drag to pan. On: drag detected/manual points to correct them. Middle or Alt-drag always pans.")
        for widget in (self.overlays_visible,self.labels_visible):
            widget.toggled.connect(self._presentation_options_changed)
            tools.addWidget(widget)
        self.edit_points.toggled.connect(self._update_edit_keys)
        tools.addWidget(self.edit_points)
        for label, callback in (("Fit all",self.fit_all_views),("Return all",self.return_all_views),("Reset views",self.reset_views)):
            button = QToolButton()
            button.setText(label)
            button.clicked.connect(callback)
            tools.addWidget(button)
        root.addLayout(tools)
        self.preview_mode.currentIndexChanged.connect(self._selection_changed)
        self.camera_pick.currentIndexChanged.connect(self._selection_changed)
        self.selected_cameras.itemChanged.connect(self._subset_changed)
        self.time_slider.valueChanged.connect(self._seek)
        self.time_slider.sliderPressed.connect(self.pause)
        self.play_button.clicked.connect(self._toggle_play)
        self.save_button.clicked.connect(self.save)
        self.use_markers.toggled.connect(self._capture_changed)
        self.marker_only.toggled.connect(self._capture_changed)
        self._make_pages()

    def _subset_changed(self):
        if self._loading:
            return
        if self.preview_mode.currentText() != "Selected cameras":
            self.preview_mode.setCurrentText("Selected cameras")
        else:
            self._selection_changed()

    def _make_pages(self):
        for kind, title in (("cameras", "Cameras & Calibration"), ("body", "Markers & Body"), ("hands", "Hands & Fingers")):
            page = QWidget()
            layout = QVBoxLayout(page)
            bar = QHBoxLayout()
            if kind == "cameras":
                align = QPushButton("Align videos")
                align.clicked.connect(self.align_requested)
                accept = QPushButton("Accept displayed timing")
                accept.clicked.connect(self._accept_timing)
                recal = QPushButton("Recalibrate this camera")
                recal.clicked.connect(lambda: self.recalibrate_requested.emit(self.camera_pick.currentData() or ""))
                self.body_enabled = QCheckBox("Use selected camera for body")
                self.hands_enabled = QCheckBox("Use selected camera for hands")
                self.body_enabled.setChecked(True)
                self.hands_enabled.setChecked(True)
                self.body_enabled.toggled.connect(lambda v: self._policy_changed("body", v))
                self.hands_enabled.toggled.connect(lambda v: self._policy_changed("hands", v))
                for widget in (align, accept, recal):
                    bar.addWidget(widget)
                solve_inputs = QHBoxLayout()
                solve_inputs.addWidget(self.body_enabled)
                solve_inputs.addWidget(self.hands_enabled)
            else:
                scope = QComboBox()
                scope.addItem("Visible cameras","visible")
                scope.addItem("Selected camera","selected")
                scope.setToolTip("Choose which cameras detection and retracking will update.")
                self._detection_scopes[kind] = scope
                bar.addWidget(scope)
                detect = QPushButton("Detect current frame")
                detect.clicked.connect(lambda _=False: self._detect(True))
                track = QPushButton("Track selected range")
                track.clicked.connect(lambda _=False: self._detect(False))
                bar.addWidget(detect)
                bar.addWidget(track)
                if kind == "hands":
                    self.hand_source = QComboBox()
                    self.hand_source.addItems(["Per-camera 3D estimate", "Fused hand result"])
                    self.hand_source.currentIndexChanged.connect(self._render_frames)
                    self.hand_side = QComboBox()
                    self.hand_side.addItems(["Both hands", "Left hand", "Right hand"])
                    self.center_hand = QCheckBox("Center hand (preview only)")
                    self.center_hand.setChecked(True)
                    self.full_hand_frame = QCheckBox("Full frame")
                    self.full_hand_frame.toggled.connect(self._full_frame_changed)
                    roi_button = QPushButton("Draw crop for range")
                    roi_button.clicked.connect(self._draw_hand_roi)
                    reset_roi = QPushButton("Reset crop")
                    reset_roi.clicked.connect(self._reset_hand_roi)
                    self.hand_side.currentIndexChanged.connect(self._rebuild_grids)
                    self.center_hand.toggled.connect(lambda: self._render_frames())
                    hand_options = QHBoxLayout()
                    for widget in (self.hand_side, self.hand_source, self.center_hand):
                        hand_options.addWidget(widget)
                    crop_options = QHBoxLayout()
                    for widget in (self.full_hand_frame, roi_button, reset_roi):
                        crop_options.addWidget(widget)
                else:
                    self.point_choice = QComboBox()
                    for i, name in enumerate(BODY_POINT_NAMES):
                        self.point_choice.addItem(name, f"body:{i}")
                    for side in ("left", "right"):
                        for i, name in enumerate(HAND_POINT_NAMES):
                            self.point_choice.addItem(side.capitalize() + " " + name.lower(), f"hand:{side}:{i}")
                    self.place_point = QCheckBox("Place selected point")
                    self.place_point.toggled.connect(self._update_edit_keys)
                    self.point_choice.currentIndexChanged.connect(self._update_edit_keys)
                    bar.addWidget(self.point_choice)
                    bar.addWidget(self.place_point)
            if kind!="hands":
                layout.addLayout(bar)
            if kind == "hands":
                layout.addLayout(hand_options)
                self.hand_side.setMaximumWidth(160)
                self.hand_source.setMaximumWidth(235)
                self.hand_tracking_tools = QWidget()
                self.hand_tracking_tools.setLayout(bar)
                self.hand_tracking_tools.hide()
                self.toggle_tracking_tools = QToolButton()
                self.toggle_tracking_tools.setText("Detect / track")
                self.toggle_tracking_tools.setToolTip("Show detection scope, current-frame detection and range retracking.")
                self.toggle_tracking_tools.setCheckable(True)
                self.toggle_tracking_tools.toggled.connect(self.hand_tracking_tools.setVisible)
                hand_options.addWidget(self.toggle_tracking_tools)
                layout.addWidget(self.hand_tracking_tools)
                self.hand_edit_tools = QWidget()
                tools_layout = QVBoxLayout(self.hand_edit_tools)
                tools_layout.setContentsMargins(0, 0, 0, 0)
                tools_layout.addLayout(crop_options)
                self.toggle_hand_tools = QPushButton("Point / crop tools")
                self.toggle_hand_tools.setCheckable(True)
                self.toggle_hand_tools.toggled.connect(self.hand_edit_tools.setVisible)
                hand_options.addWidget(self.toggle_hand_tools)
                edits = QHBoxLayout()
                self.hand_point_choice = QComboBox()
                for side in ("left","right"):
                    for i,name in enumerate(HAND_POINT_NAMES):
                        self.hand_point_choice.addItem(side.capitalize()+" "+name.lower(),f"hand:{side}:{i}")
                self.place_hand_point = QCheckBox("Place missing point")
                self.hand_point_choice.currentIndexChanged.connect(self._update_edit_keys)
                self.place_hand_point.toggled.connect(self._update_edit_keys)
                edits.addWidget(self.hand_point_choice)
                edits.addWidget(self.place_hand_point)
                for label,callback in (("Missing",self._mark_missing),("Restore",self._restore_point),("Undo",self.undo),("Redo",self.redo)):
                    button=QPushButton(label)
                    button.clicked.connect(callback)
                    edits.addWidget(button)
                tools_layout.addLayout(edits)
                layout.addWidget(self.hand_edit_tools)
                self.hand_edit_tools.hide()
            elif kind == "cameras":
                layout.addLayout(solve_inputs)
            if kind == "body":
                actions = QHBoxLayout()
                for label, callback in (("Mark missing", self._mark_missing), ("Restore detection", self._restore_point),
                                        ("Assign selected tag", self._assign_tag), ("Calibrate tag offsets", self._calibrate_tags),
                                        ("Undo", self.undo), ("Redo", self.redo)):
                    button = QPushButton(label)
                    button.clicked.connect(callback)
                    actions.addWidget(button)
                self.point_details = QLabel("Hover for names. Select a point to inspect it; enable Edit points to drag corrections.")
                self.point_details.setWordWrap(True)
                layout.addLayout(actions)
                manual_tags = QHBoxLayout()
                for label, callback in (("Add physical tag point…", self._add_tag_point), ("Set measured tag offset…", self._set_tag_offset)):
                    button = QPushButton(label)
                    button.clicked.connect(callback)
                    manual_tags.addWidget(button)
                manual_tags.addStretch()
                layout.addLayout(manual_tags)
                layout.addWidget(self.point_details)
            if kind == "cameras":
                note = QLabel("Viewing selection does not change solve inputs. For a bad view: fix timing, edit points, recalibrate geometry, or exclude the affected channel/range.")
                note.setWordWrap(True)
                layout.addWidget(note)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            grid = ReviewGrid()
            grid.layout_changed.connect(self._preview_changed)
            scroll.setWidget(grid)
            scroll.viewport().installEventFilter(self)
            layout.addWidget(scroll, 1)
            self.pages[kind] = page
            self.grids[kind] = grid
            self._grid_scrolls[kind] = scroll
        self.range_start = QDoubleSpinBox()
        self.range_end = QDoubleSpinBox()
        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Global range (seconds)"))
        for spin in (self.range_start, self.range_end):
            spin.setRange(-3600, 86400)
            spin.setDecimals(3)
            range_row.addWidget(spin)
        exclude = QPushButton("Exclude selected camera/range")
        exclude.clicked.connect(self._exclude_range)
        range_row.addWidget(exclude)
        reference = QPushButton("Use camera as reference")
        reference.clicked.connect(self._set_reference)
        range_row.addWidget(reference)
        self.pages["cameras"].layout().insertLayout(1, range_row)

    def load_package(self, package, project=None):
        if not self.save():
            raise OSError("Save the current review successfully before replacing its source.")
        # Validate the incoming take completely before detaching the current views
        # or changing where their annotations will be saved.
        from multi_video_studio.review_state import source_fingerprint
        if not package.motion_clips:
            raise ValueError("The selected source contains no readable camera videos.")
        identity = "_".join(sorted(source_fingerprint(c.path)[:12] for c in package.motion_clips.values()))
        state_path = (project.subdir("tracking") if project else self.state_root) / (identity + ".review.json")
        embedded = getattr(package, "review_path", None)
        load_path = Path(embedded) if embedded and Path(embedded).exists() else state_path
        state = ReviewState.load(load_path) if load_path.exists() else ReviewState()
        remap = state.bind_sources(package.motion_clips)
        calibration = None
        if state.calibration:
            from engine.math_core.camera_model import MultiCameraCalibration
            calibration = MultiCameraCalibration.from_dict(state.calibration)
        observation_path = Path(getattr(package, "review_observations_path", None) or state_path.with_suffix(".npz"))
        observations = ReviewObservations.load(observation_path) if observation_path.exists() else ReviewObservations()
        observations.cameras = {remap[u]: v for u, v in observations.cameras.items() if u in remap}
        self.save_preview_layout()
        self._layout_loading = True
        self._clear_preview_tiles()
        self.edit_points.setChecked(False)
        self.timer.stop()
        self.play_button.setText("Play cameras")
        self.playback_changed.emit(False)
        self.preview_epoch += 1
        self.last_frames.clear()
        self.package = package
        self.state_path, self.state = state_path, state
        self._source_key = hashlib.sha256("|".join(sorted(v["asset_id"] for v in self.state.sources.values())).encode()).hexdigest()
        self._preview_state = self._preview_store.get("sources",{}).get(self._source_key,{})
        self._restored_grids.clear()
        self.layout_columns.blockSignals(True)
        self.layout_columns.setCurrentIndex(max(0,self.layout_columns.findData(self._preview_state.get("columns",0))))
        self.layout_columns.blockSignals(False)
        for name,widget,default in (("overlays",self.overlays_visible,True),("labels",self.labels_visible,False)):
            widget.blockSignals(True)
            widget.setChecked(bool(self._preview_state.get(name,default)))
            widget.blockSignals(False)
        self.document = None
        self.calibration, self.observations = calibration, observations
        self._loading = True
        self.camera_pick.clear()
        self.selected_cameras.clear()
        for uid, clip in package.motion_clips.items():
            self.camera_pick.addItem(uid + " · " + clip.path.name, uid)
            item = QListWidgetItem(uid)
            item.setData(Qt.ItemDataRole.UserRole, uid)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.selected_cameras.addItem(item)
        self.use_markers.setChecked(self.state.capture_mode != "markerless")
        self.marker_only.setChecked(self.state.capture_mode == "markers")
        self.marker_only.setEnabled(self.use_markers.isChecked())
        self._loading = False
        self._update_timeline()
        self._rebuild_grids()
        for name,panel in self._external_panels.items():
            panel.restore_presentation(self._preview_state.get("external",{}).get(name,{}),floating=True)
        self._layout_loading = False
        saved = self._preview_state.get("active")
        if isinstance(saved,list) and len(saved)==4 and tuple(saved[:3]) in self.tiles:
            self._select_view(tuple(saved))
        else:
            self._select_first_view()
        self.save()

    def _update_timeline(self):
        if self.package is None:
            return
        starts = [self.state.global_time(u, 0) for u in self.package.motion_clips]
        ends = [self.state.global_time(u, c.duration_s) for u, c in self.package.motion_clips.items()]
        start, end = min(starts), max(ends)
        self.time_slider.setRange(round(start * 1000), round(end * 1000))
        overlap_start, overlap_end = max(starts), min(ends)
        self.range_start.setValue(overlap_start)
        self.range_end.setValue(max(overlap_start, overlap_end))
        self.time_slider.setValue(round(overlap_start * 1000))
        self._seek(self.time_slider.value())

    def set_clocks(self, estimates):
        if estimates:
            self.state.set_clocks(estimates)
            self._update_timeline()
            self._changed()

    def _accept_timing(self):
        if self.package is None:
            return
        if not self.state.clocks:
            from multi_video_studio.pyav_sync import CameraSyncEstimate
            estimates = {uid: CameraSyncEstimate(uid, self.state.manual_offsets_ms.get(uid, 0.) / 1000., "manual", 0., "Manually aligned native timeline")
                         for uid in self.package.motion_clips}
            self.state.set_clocks(estimates)
            self.state.reference_camera = next(iter(estimates))
        else:
            self.state.checkpoint()
        self.state.alignment_accepted = True
        self._changed()

    def _set_reference(self):
        uid = self.camera_pick.currentData()
        if uid not in self.state.clocks:
            self.status.setText("Check or manually accept timing before changing the reference camera.")
            return
        self.state.checkpoint()
        reference = self.state.clocks[uid]
        slope = float(reference.get("drift_slope", 1.))
        intercept = reference.get("offset_s", 0.) - (slope - 1) * reference.get("drift_pivot_s", 0.)
        for key, clock in self.state.clocks.items():
            old_slope = float(clock.get("drift_slope", 1.))
            old_intercept = clock.get("offset_s", 0.) - (old_slope - 1) * clock.get("drift_pivot_s", 0.)
            clock.update(drift_slope=old_slope / slope, drift_pivot_s=0., offset_s=(old_intercept - intercept) / slope)
            clock["method"] = "reference" if key == uid else "manual"
            clock["summary"] = "Reference camera" if key == uid else f"Timing relative to {uid}"
        self.state.reference_camera = uid
        self._update_timeline()
        self._changed()

    def set_document(self, doc, calibration=None):
        self.document, self.calibration = doc, calibration
        revision = doc.processing_config.get("review_state", {}).get("revision", self.state.revision)
        if revision == self.state.revision:
            self.state.clocks.update(doc.sync_info or {})
            self.state.alignment_accepted = True
        self.state.solved_revision = revision
        raw_path = doc.processing_config.get("review_observations_path")
        if raw_path and Path(raw_path).exists():
            self.observations.merge(ReviewObservations.load(Path(raw_path)))
        elif not self.observations.cameras:
            self.observations.from_document(doc, self.state)
        if calibration is not None and not self.state.calibration:
            self.state.calibration = calibration.to_dict()
        self._update_timeline()
        self._changed()

    def visible_cameras(self):
        if self.package is None:
            return []
        mode = self.preview_mode.currentText()
        if mode == "One camera":
            return [self.camera_pick.currentData()] if self.camera_pick.currentData() else []
        if mode == "Selected cameras":
            return [self.selected_cameras.item(i).data(Qt.ItemDataRole.UserRole)
                    for i in range(self.selected_cameras.count())
                    if self.selected_cameras.item(i).checkState() == Qt.CheckState.Checked]
        return list(self.package.motion_clips)

    def _selection_changed(self, *args):
        if self._loading:
            return
        uid = self.camera_pick.currentData()
        policy = self.state.camera_policy.get(uid, {})
        for widget, channel in ((self.body_enabled, "body"), (self.hands_enabled, "hands")):
            widget.blockSignals(True)
            widget.setChecked(policy.get(channel, True))
            widget.blockSignals(False)
        self._rebuild_grids()

    def _rebuild_grids(self):
        for kind, grid in self.grids.items():
            visible = []
            for uid in self.visible_cameras():
                sides = [None]
                if kind == "hands":
                    sides = ["left", "right"] if self.hand_side.currentIndex() == 0 else ["left" if self.hand_side.currentIndex() == 1 else "right"]
                for side in sides:
                    key = kind,uid,side
                    if key not in self.tiles:
                        self.tiles[key] = self._create_tile(key)
                    visible.append(self.tiles[key])
            grid.set_tiles(visible,self.layout_columns.currentData(),self._grid_scrolls[kind].viewport().width())
            if kind not in self._restored_grids:
                saved = self._preview_state.get("grids",{}).get(kind,{})
                if saved.get("views")==[tile.objectName() for tile in visible] and saved.get("columns")==grid.columns:
                    QTimer.singleShot(0,self,lambda g=grid,s=saved,e=self.preview_epoch:
                        g.restore_sizes(s.get("sizes",{})) if self.preview_epoch==e and not self._shutting_down else None)
                self._restored_grids.add(kind)
        self._update_edit_keys()
        self._presentation_options_changed()
        self.request_frames()

    def _create_tile(self, key):
        kind,uid,side = key
        tile = CameraTile(uid,side)
        tile.setObjectName("_".join(str(v) for v in key))
        tile.point_selected.connect(self._select_point)
        tile.point_edited.connect(self._edit_point)
        tile.video_panel.menu.addAction("First source frame (Home)",lambda:self.seek_source_edge(uid,False))
        tile.video_panel.menu.addAction("Last source frame (End)",lambda:self.seek_source_edge(uid,True))
        if kind in ("body","hands"):
            tile.video_panel.menu.addAction("Detect this camera frame",lambda:self._detect_camera_frame(tile))
        state = self._preview_state.get("tiles",{}).get(self._tile_key(key),{})
        tile.full_frame = bool(state.get("full_frame",False))
        tile.model_source = int(state.get("model_source",-1))
        if side:
            tile.overlay.roi_edited.connect(lambda rect,u=uid,s=side:self._store_hand_roi(u,s,rect))
            full = tile.video_panel.menu.addAction("Show full camera frame")
            full.setCheckable(True)
            full.setChecked(tile.full_frame)
            full.toggled.connect(lambda checked,t=tile:self._set_full_frame(t,checked))
            choices = QMenu("3D hand source",tile.hand_panel.menu)
            group = QActionGroup(choices)
            group.setExclusive(True)
            for label,value in (("Follow Hands tab",-1),("Camera-local estimate",0),("Fused hand result",1)):
                action = choices.addAction(label)
                action.setCheckable(True)
                action.setChecked(tile.model_source==value)
                group.addAction(action)
                action.triggered.connect(lambda checked,v=value,t=tile:self._set_model_source(t,v))
            tile.hand_panel.menu.addMenu(choices)
            if tile.views and len(state.get("split",[]))==2:
                tile.views.setSizes([max(1,min(10000,int(v))) for v in state["split"]])
            tile.views.splitterMoved.connect(lambda:self._preview_changed())
        for role,panel in (("video",tile.video_panel),("hand",tile.hand_panel)):
            if panel is None:
                continue
            view_key = (*key,role)
            panel.transport_factory = lambda parent:ReviewTransport(self,parent)
            panel.activated.connect(lambda k=view_key:self._select_view(k))
            panel.presentation_changed.connect(self._preview_changed)
            panel.floating_changed.connect(lambda _=False:self.request_frames())
            panel.command.connect(self._view_command)
            panel.restore_presentation(state.get(role,{}))
            if state.get(role,{}).get("floating"):
                QTimer.singleShot(0,self,lambda p=panel,e=self.preview_epoch,k=key,t=tile:
                    p.pop_out() if not self._shutting_down and self.preview_epoch==e and self.tiles.get(k)==t else None)
        return tile

    @staticmethod
    def _tile_key(key):
        return "/".join(str(v or "body") for v in key)

    def _set_full_frame(self, tile, enabled):
        tile.full_frame = enabled
        tile.overlay.fit_view()
        self._render_frames()
        self._preview_changed()

    def _full_frame_changed(self):
        for key,tile in self.tiles.items():
            if key[0]=="hands":
                tile.overlay.fit_view()
        self._render_frames()

    def _set_model_source(self, tile, value):
        tile.model_source = value
        self._render_frames()
        self._preview_changed()

    def _select_view(self, key):
        if key[:3] not in self.tiles:
            return
        self._active_view = key
        for panel in self._external_panels.values():
            panel.set_selected(False)
        for tile_key,tile in self.tiles.items():
            for role,panel in (("video",tile.video_panel),("hand",tile.hand_panel)):
                if panel:
                    panel.set_selected((*tile_key,role)==key)
        text = "Selected: " + key[1] + (" · "+key[2].capitalize()+" hand" if key[2] else "") + (" · Video" if key[3]=="video" else " · 3D")
        self.active_view_label.setText(text)
        self.active_view_label.setToolTip(text)
        if not self._loading and not self._layout_loading:
            index = self.camera_pick.findData(key[1])
            if index>=0 and index!=self.camera_pick.currentIndex():
                self.camera_pick.setCurrentIndex(index)
        self._preview_changed()

    def _select_first_view(self):
        preferred = self.camera_pick.currentData()
        keys = [k for k in self.tiles if k[0]==self.active_page and k[1] in self.visible_cameras()]
        key = next((k for k in keys if k[1]==preferred),next(iter(keys),None))
        if key:
            self._select_view((*key,"video"))

    def _view_command(self, command):
        if command=="play":
            self._toggle_play()
        elif command=="edit":
            self.edit_points.toggle()
        elif command.startswith(("previous","next")):
            self.step_frame(-1 if command.startswith("previous") else 1,10 if command.endswith("10") else 1)
        elif command in ("first","last"):
            uid = self._active_view[1] if self._active_view else self.camera_pick.currentData()
            self.seek_source_edge(uid,command=="last")

    def _presentation_options_changed(self):
        for tile in self.tiles.values():
            tile.overlay.set_overlays(self.overlays_visible.isChecked(),self.labels_visible.isChecked())
        self._preview_changed()

    def fit_all_views(self):
        for key,tile in self.tiles.items():
            if key[0]==self.active_page or tile.is_floating():
                for panel in tile.panels():
                    panel.fit_view()

    def return_all_views(self):
        for tile in self.tiles.values():
            for panel in tile.panels():
                panel.return_to_grid()
        for panel in self._external_panels.values():
            panel.return_to_grid()

    def register_external_panel(self, name, panel):
        self._external_panels[name] = panel
        panel.presentation_changed.connect(self._preview_changed)
        panel.activated.connect(lambda:self._select_external_panel(panel))

    def _select_external_panel(self, panel):
        self._active_view = None
        for tile in self.tiles.values():
            for item in tile.panels():
                item.set_selected(False)
        for item in self._external_panels.values():
            item.set_selected(item==panel)
        self.active_view_label.setText("Selected: "+panel.window_title)
        self.active_view_label.setToolTip(panel.window_title)
        self._preview_changed()

    def reset_views(self):
        self.return_all_views()
        for tile in self.tiles.values():
            tile.overlay.fit_view()
            if tile.hand_view:
                tile.hand_view.reset_view()
                tile.views.setSizes([600,400])
        for grid in self.grids.values():
            grid._layouts.clear()
            grid._signature = None
        self.layout_columns.setCurrentIndex(0)
        self._refresh_grid_sizes()

    def eventFilter(self, watched, event):
        if event.type()==QEvent.Type.Resize:
            self._resize_timer.start()
        return super().eventFilter(watched,event)

    def _refresh_grid_sizes(self):
        for kind,grid in self.grids.items():
            grid.set_tiles(grid.tiles,self.layout_columns.currentData(),self._grid_scrolls[kind].viewport().width())
        self._preview_changed()

    def _read_preview_preferences(self):
        try:
            value = json.loads((self.state_root/"preview_layout.json").read_text(encoding="utf-8"))
            return value if isinstance(value,dict) and value.get("version")==1 and isinstance(value.get("sources"),dict) else {"version":1,"sources":{}}
        except (OSError,ValueError):
            return {"version":1,"sources":{}}

    def _preview_changed(self):
        if self._layout_loading or self._shutting_down:
            return
        self._layout_timer.start()
        if self.package and self._needed_native_requests()!=self._native_requests:
            self.request_frames()

    def save_preview_layout(self):
        if self._layout_loading or not self._source_key:
            return
        value = {"columns":self.layout_columns.currentData(),"active":list(self._active_view) if self._active_view else None,
                 "overlays":self.overlays_visible.isChecked(),"labels":self.labels_visible.isChecked(),"tiles":{},
                 "external":{name:panel.presentation_state() for name,panel in self._external_panels.items()},
                 "grids":{kind:{"views":[tile.objectName() for tile in grid.tiles],"columns":grid.columns,"sizes":grid.split_sizes()}
                          for kind,grid in self.grids.items()}}
        for key,tile in self.tiles.items():
            state = {"video":tile.video_panel.presentation_state(),"full_frame":tile.full_frame,"model_source":tile.model_source}
            if tile.hand_panel:
                state.update(hand=tile.hand_panel.presentation_state(),split=tile.views.sizes())
            value["tiles"][self._tile_key(key)] = state
        sources = self._preview_store["sources"]
        sources.pop(self._source_key,None)
        sources[self._source_key] = value
        while len(sources)>12:
            del sources[next(iter(sources))]
        try:
            self.state_root.mkdir(parents=True,exist_ok=True)
            path = self.state_root/"preview_layout.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self._preview_store),encoding="utf-8")
            temporary.replace(path)
        except OSError:
            self.active_view_label.setToolTip("Preview preferences could not be saved. Capture annotations are saved separately.")

    def _clear_preview_tiles(self):
        for panel in self._external_panels.values():
            panel.return_to_grid()
        for tile in self.tiles.values():
            for panel in tile.panels():
                panel.return_to_grid()
        for grid in self.grids.values():
            grid.set_tiles([])
        for tile in self.tiles.values():
            tile.deleteLater()
        self.tiles.clear()
        self._active_view = None
        self._native_requests.clear()

    def _update_edit_keys(self):
        if self.sender()==self.edit_points and not self.edit_points.isChecked():
            for checkbox in (self.place_point,self.place_hand_point):
                checkbox.blockSignals(True)
                checkbox.setChecked(False)
                checkbox.blockSignals(False)
        elif (self.place_point.isChecked() or self.place_hand_point.isChecked()) and not self.edit_points.isChecked():
            self.edit_points.setChecked(True)
        key = self.point_choice.currentData() if self.place_point.isChecked() else None
        for (kind, uid, side), tile in self.tiles.items():
            selected = self.hand_point_choice.currentData() if kind=="hands" and self.place_hand_point.isChecked() else key
            tile.overlay.edit_key = selected if kind != "hands" or (selected and selected.startswith("hand:" + side)) else None
            tile.overlay.set_editing(self.edit_points.isChecked() or tile.overlay.edit_key is not None)
        if self.edit_points.isChecked() or self.place_point.isChecked() or self.place_hand_point.isChecked():
            self.pause()

    def _capture_changed(self):
        if self._loading:
            return
        self.state.checkpoint()
        self.state.capture_mode = "markers" if self.use_markers.isChecked() and self.marker_only.isChecked() else ("hybrid" if self.use_markers.isChecked() else "markerless")
        self.marker_only.setEnabled(self.use_markers.isChecked())
        self._changed()

    def _draw_hand_roi(self):
        uid = self.camera_pick.currentData()
        sides = ["left", "right"] if self.hand_side.currentIndex() == 0 else ["left" if self.hand_side.currentIndex() == 1 else "right"]
        side = sides[0]
        if len(sides) > 1:
            side, okay = QInputDialog.getItem(self, "Adjust hand crop", "Hand", sides, 0, False)
            if not okay:
                return
        tile = self.tiles.get(("hands", uid, side))
        result = self.last_frames.get(uid)
        if tile is None or result is None:
            self.status.setText("Show the selected camera's hand before drawing its crop.")
            return
        if self.timer.isActive():
            self._toggle_play()
        self.full_hand_frame.setChecked(True)
        tile.overlay.frame(result[1], result[2], {})
        tile.overlay.select_roi = True
        self.status.setText("Drag a box around the hand and fingertips. The override follows the wrist through the selected time range.")

    def _store_hand_roi(self, uid, side, rect):
        source = self.state.sources[uid]
        x, y, width, height = rect
        x0, y0 = max(0., x), max(0., y)
        x1, y1 = min(source["width"], x + width), min(source["height"], y + height)
        if x1 - x0 < 8 or y1 - y0 < 8:
            self.status.setText("Draw a crop at least eight source pixels wide and tall.")
            return
        frame = self.last_frames[uid]
        pts = frame[0]
        observations = self.observations.frame(uid, pts)
        body = observations["body"] if observations else np.zeros((26, 3))
        body = self.state.apply_points(uid, [pts], body[None])[0][0]
        wrist = body[9 if side == "left" else 10]
        offset = ([((x0 + x1) / 2) - wrist[0], ((y0 + y1) / 2) - wrist[1]] if wrist[2] >= .2 else None)
        self.state.checkpoint()
        self.state.hand_rois.append({"camera_uid":uid, "side":side, "reference_pts":pts,
            "start_s": self.state.local_time(uid, self.range_start.value()),
            "end_s": self.state.local_time(uid, self.range_end.value()),
            "rect":[x0, y0, x1 - x0, y1 - y0], "wrist_offset":offset})
        self.full_hand_frame.setChecked(False)
        self._changed()
        self._render_frames()

    def _reset_hand_roi(self):
        uid = self.camera_pick.currentData()
        self.state.checkpoint()
        self.state.hand_rois = [r for r in self.state.hand_rois if r["camera_uid"] != uid]
        self._changed()
        self._render_frames()

    def _policy_changed(self, channel, enabled):
        uid = self.camera_pick.currentData()
        if not uid or self._loading:
            return
        self.state.checkpoint()
        self.state.camera_policy.setdefault(uid, {})[channel] = bool(enabled)
        self._changed()

    def _exclude_range(self):
        uid = self.camera_pick.currentData()
        if not uid:
            return
        channel, okay = QInputDialog.getItem(self, "Exclude observations", "Channel", ["body", "hands", "all"], 0, False)
        if not okay:
            return
        start, end = self.range_start.value(), self.range_end.value()
        if end < start:
            QMessageBox.warning(self, "Range", "End time must be after start time.")
            return
        self.state.checkpoint()
        self.state.exclusions.append({"camera_uid": uid, "channel": channel,
                                      "start_s": self.state.local_time(uid, start),
                                      "end_s": self.state.local_time(uid, end)})
        self._changed()

    def _select_point(self, uid, key, pts):
        if self.timer.isActive():
            self._toggle_play()
        source_time = self.state.global_time(uid,pts)
        if abs(source_time-self.global_s)>.001:
            self.time_slider.setValue(round(source_time*1000))
        self.selected_point = (uid, key, pts)
        self.point_details.setText(f"{point_name(key, self.state.marker_assignments)} · {uid} · source {pts:.3f}s · {key}")

    def _edit_point(self, uid, key, pts, xy):
        try:
            frame = self.last_frames.get(uid)
            self.state.edit_point(uid, pts, key, xy, frame_index=frame.frame_index if frame else None)
            self._select_point(uid, key, pts)
            self._changed()
            self._render_frames()
        except ValueError as exc:
            self.status.setText(str(exc))

    def _mark_missing(self):
        if self.selected_point:
            uid, key, pts = self.selected_point
            if key.startswith("fused:"):
                self.status.setText("Correct the camera's detected points; fused result points are read-only.")
                return
            self.state.edit_point(uid, pts, key, valid=False)
            self._changed()
            self._render_frames()

    def _restore_point(self):
        if self.selected_point:
            uid, key, pts = self.selected_point
            self.state.reset_point(uid, pts, key)
            self._changed()
            self._render_frames()

    def _assign_tag(self):
        if not self.selected_point or not self.selected_point[1].startswith("tag:"):
            self.status.setText("Select a detected tag first, then assign its anatomical point.")
            return
        tag_id = self.selected_point[1].split(":")[1]
        name, okay = QInputDialog.getItem(self, "Assign physical tag", "Anatomical point", BODY_POINT_NAMES, 9, False)
        if not okay:
            return
        size, okay = QInputDialog.getDouble(self, "Tag geometry", "Black square side length (metres)", .05, .005, .4, 4)
        if okay:
            self.state.checkpoint()
            self.state.marker_assignments[tag_id] = {"name": name, "body_index": BODY_POINT_NAMES.index(name), "size_m": size}
            self._changed()
            self._render_frames()

    def _add_tag_point(self):
        tag_id, okay = QInputDialog.getInt(self, "Physical tag", "AprilTag 36h11 ID", 0, 0, 586)
        if not okay:
            return
        labels = ["Center (position only)", "Corner 0 (printed top-left)", "Corner 1 (printed top-right)",
                  "Corner 2 (printed bottom-right)", "Corner 3 (printed bottom-left)"]
        label, okay = QInputDialog.getItem(self, "Tag point", "Use the printed marker orientation", labels, 0, False)
        if not okay:
            return
        index = labels.index(label)
        key = f"tag:{tag_id}:" + ("center" if index == 0 else "corner" + str(index - 1))
        item = self.point_choice.findData(key)
        if item < 0:
            self.point_choice.addItem(f"Tag {tag_id} · {label}", key)
            item = self.point_choice.count() - 1
        self.point_choice.setCurrentIndex(item)
        self.place_point.setChecked(True)
        self.status.setText("Click this tag point in each selected camera. Four corners are required for orientation.")

    def _set_tag_offset(self):
        if not self.selected_point or not self.selected_point[1].startswith("tag:"):
            self.status.setText("Select and assign a physical tag first.")
            return
        tag_id = self.selected_point[1].split(":")[1]
        if tag_id not in self.state.marker_assignments:
            self._assign_tag()
        if tag_id not in self.state.marker_assignments:
            return
        value, okay = QInputDialog.getText(self, "Measured marker offset",
            "Tag center → joint in printed-tag coordinates (metres), X,Y,Z. +X right, +Y up, +Z out of the marker.",
            text=",".join(str(v) for v in self.state.marker_assignments[tag_id].get("joint_offset_tag_m", [0, 0, 0])))
        if not okay:
            return
        try:
            offset = np.array([float(v.strip()) for v in value.split(",")])
            if offset.shape != (3,) or not np.isfinite(offset).all() or np.linalg.norm(offset) > .3:
                raise ValueError("Enter three finite measurements within 30 cm of the joint.")
            self.state.checkpoint()
            self.state.marker_assignments[tag_id].update(joint_offset_tag_m=offset.tolist(),
                offset_calibrated=True, offset_method="user measured")
            self._changed()
        except ValueError as exc:
            self.status.setText(str(exc))

    def undo(self):
        if self.state.undo():
            self._changed()
            self._render_frames()

    def _calibrate_tags(self):
        if self.package is None or self.calibration is None:
            self.status.setText("Solve or load the camera calibration first.")
            return
        from copy import deepcopy
        from multi_video_studio.marker_constraints import calibrate_review_marker_offsets
        assignments = deepcopy(self.state.marker_assignments)
        try:
            count = calibrate_review_marker_offsets(self.package, self.state, self.observations,
                                                   self.calibration, self.document, self.global_s)
            calibrated = self.state.marker_assignments
            self.state.marker_assignments = assignments
            self.state.checkpoint()
            self.state.marker_assignments = calibrated
            self._changed()
            self.status.setText(f"Calibrated {count} tag offsets from the body estimate. Review a neutral pose; this is not measured ground truth.")
        except (ValueError, OSError) as exc:
            self.state.marker_assignments = assignments
            self.status.setText(str(exc))

    def redo(self):
        if self.state.redo():
            self._changed()
            self._render_frames()

    def _detect(self, current):
        if self.package is not None:
            scope = self._detection_scopes.get(self.active_page)
            uids = [self.camera_pick.currentData()] if scope and scope.currentData()=="selected" else self.visible_cameras()
            self.detection_requested.emit(uids, self.global_s if current else self.range_start.value(),
                                          self.global_s if current else self.range_end.value(), current)

    def _detect_camera_frame(self, tile):
        if self.package and tile.uid in self.package.motion_clips:
            self.pause()
            stamp = self.state.global_time(tile.uid,tile.pts_s)
            self.time_slider.setValue(round(stamp*1000))
            self.detection_requested.emit([tile.uid],stamp,stamp,True)

    def _seek(self, milliseconds):
        self.global_s = milliseconds / 1000
        self.time_label.setText(f"Global {self.global_s:.3f}s")
        self.time_changed.emit(self.global_s)
        self.request_frames()

    def _toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Play cameras")
        else:
            self.play_started, self.play_origin = time.monotonic(), self.global_s
            self.timer.start()
            self.play_button.setText("Pause cameras")
        self.playback_changed.emit(self.timer.isActive())

    def pause(self):
        if self.timer.isActive():
            self._toggle_play()

    def step_frame(self, direction, count=1):
        if self.package is None:
            return
        self.pause()
        uid = self._active_view[1] if self._active_view else self.camera_pick.currentData()
        if uid not in self.package.motion_clips:
            return
        times = self.decoder.frame_times.get(uid) if getattr(self.decoder,"frame_epoch",-1)==self.preview_epoch else None
        if times is not None and len(times):
            local = self.state.local_time(uid,self.global_s)
            index = int(np.argmin(abs(times-local)))
            if times[0]-.001 <= local <= times[-1]+.001:
                index = int(np.clip(index+direction*count,0,len(times)-1))
            target = self.state.global_time(uid,float(times[index]))
        else:
            fps = self.document.fps if self.document else 30.
            target = self.global_s+direction*count/fps
        self.time_slider.setValue(round(target*1000))

    def seek_source_edge(self, uid, last=False):
        if not self.package or uid not in self.package.motion_clips:
            return
        self.pause()
        times = self.decoder.frame_times.get(uid) if getattr(self.decoder,"frame_epoch",-1)==self.preview_epoch else None
        stamp = float(times[-1 if last else 0]) if times is not None and len(times) else max(0.,self.package.motion_clips[uid].duration_s-1/30) if last else 0.
        self.time_slider.setValue(round(self.state.global_time(uid,stamp)*1000))

    def _requested_cameras(self):
        uids = set(self.visible_cameras())
        uids.update(key[1] for key,tile in self.tiles.items() if tile.is_floating())
        return [uid for uid in self.package.motion_clips if uid in uids] if self.package else []

    def _needed_native_requests(self):
        result = {}
        for uid in self._requested_cameras():
            frame = self.last_frames.get(uid)
            width = frame.native_size[0] if frame else 1920
            candidates = [(key,tile) for key,tile in self.tiles.items() if key[1]==uid and (key[0]==self.active_page or tile.is_floating())]
            result[uid] = any(key[0]=="hands" or tile.overlay.zoom_percent()>960/width*100+.5
                              for key,tile in candidates)
        return result

    def _tick(self):
        value = round((self.play_origin + time.monotonic() - self.play_started) * 1000)
        if value > self.time_slider.maximum():
            self.timer.stop()
            self.play_button.setText("Play cameras")
            self.playback_changed.emit(False)
        else:
            self.time_slider.setValue(value)

    def request_frames(self):
        if self.package is None:
            return
        self.generation += 1
        self._native_requests = self._needed_native_requests()
        requests = [(u, self.package.motion_clips[u].path, self.state.local_time(u, self.global_s),
                     None if self._native_requests.get(u) else 960)
                    for u in self._requested_cameras()]
        self.decoder.request((self.preview_epoch << 32) + self.generation, requests)

    def _frames_ready(self, generation, frames):
        if generation >> 32 != self.preview_epoch or generation <= self.last_generation:
            return
        if not self.timer.isActive() and generation != (self.preview_epoch << 32) + self.generation:
            return
        self.last_generation = generation
        self.last_frames = frames
        self._render_frames()

    def _render_frames(self):
        for (kind, uid, side), tile in self.tiles.items():
            if (kind != self.active_page or uid not in self.visible_cameras()) and not tile.is_floating():
                continue
            result = self.last_frames.get(uid)
            if result is None:
                clip = self.package.motion_clips.get(uid) if self.package else None
                local = self.state.local_time(uid,self.global_s)
                if uid not in self.last_frames:
                    message = "Loading source frame…"
                    tile.video_panel.set_status(f"{uid} · decoding source frame")
                elif clip and (local<0 or local>clip.duration_s):
                    start,end = self.state.global_time(uid,0.),self.state.global_time(uid,clip.duration_s)
                    message = f"Outside this camera’s recording\nRecorded global time: {start:.3f}–{end:.3f}s\nHome / End jumps to a source frame"
                    tile.video_panel.set_status(f"{uid} · no source frame at global {self.global_s:.3f}s")
                else:
                    message = "No source frame at this time"
                    tile.video_panel.set_status(f"{uid} · source frame unavailable")
                tile.overlay.placeholder(message)
                if tile.hand_view:
                    tile.hand_view.set_hand(None,message)
                continue
            pts, image, native = result
            tile.pts_s = pts
            delta = self.state.global_time(uid, pts) - self.global_s
            tile.video_panel.set_status(f"{uid} · source {pts:.3f}s · sampling Δ {delta * 1000:+.1f}ms")
            frame = self.observations.frame(uid, pts)
            channel = "hand:" + side if side else "body"
            values = frame[channel] if frame else np.zeros((21 if side else 26, 3), np.float32)
            values, manual = self.state.apply_points(uid, [pts], values[None], channel)
            values, manual = values[0], manual[0]
            points = {f"{channel}:{i}": {"xy": xy[:2], "name": point_name(f"{channel}:{i}"),
                       "source": "manual" if manual[i] else "detected", "valid": xy[2] > .1}
                      for i, xy in enumerate(values)}
            if not side:
                camera = self.observations.cameras.get(uid, {})
                tags = camera.get("tags", {}).get(f"{pts:.6f}", [])
                for tag in tags:
                    key = f"tag:{tag['id']}:center"
                    points[key] = {"xy": np.mean(tag["corners"], axis=0),
                                   "name": point_name(key, self.state.marker_assignments), "source": "tag"}
                    for j, corner in enumerate(tag["corners"]):
                        corner_key = f"tag:{tag['id']}:corner{j}"
                        points[corner_key] = {"xy": corner, "name": point_name(key, self.state.marker_assignments) + f" corner {j}", "source": "tag"}
                for edit in self.state.points:
                    if edit["camera_uid"] == uid and abs(edit["pts_s"] - pts) < .005 and edit["key"].startswith("tag:"):
                        points[edit["key"]] = {"xy": edit["xy"] or [0, 0], "name": point_name(edit["key"], self.state.marker_assignments), "source": "manual", "valid": edit["valid"]}
            edges = [(f"{channel}:{a}", f"{channel}:{b}") for a, b in (HAND_EDGES if side else BODY_EDGES)
                     if values[a, 2] > .1 and values[b, 2] > .1]
            if side:
                from multi_video_studio.roi_tracking import roi_for_frame
                body = frame["body"] if frame else np.zeros((26, 3))
                roi = roi_for_frame(self.state, uid, side, pts, body)
                key = uid, side
                valid = values[:, 2] >= .2
                needed = max(48., float(np.max(np.ptp(values[valid, :2], axis=0))) * 1.4) if valid.sum() >= 3 else 48.
                previous = self.crop_sizes.get(key, (pts, needed))
                size = max(needed, .85 * previous[1] + .15 * needed) if abs(pts - previous[0]) < .3 else needed
                self.crop_sizes[key] = pts, size
                palm_pixels = float(np.linalg.norm(values[17,:2]-values[5,:2])) if values[17,2]>.2 and values[5,2]>.2 else 0.
                if palm_pixels:
                    tile.status.setText(tile.status.text()+f" · palm {palm_pixels:.0f} source px"+(" · small / uncertain" if palm_pixels<25 else ""))
                if frame is not None:
                    other = frame["hand:right" if side=="left" else "hand:left"]
                    if values[0,2]>.2 and other[0,2]>.2 and np.linalg.norm(values[0,:2]-other[0,:2]) < palm_pixels:
                        tile.status.setText(tile.status.text()+" · hands overlap in this view")
                crop = (image, None) if self.full_hand_frame.isChecked() or tile.full_frame else hand_crop(image, values, native,
                    center=self.center_hand.isChecked(), roi=roi, size_override=size)
                if crop is None:
                    tile.overlay.placeholder("Hand not visible / tracking uncertain")
                else:
                    crop_image, transform = crop
                    tile.overlay.frame(crop_image, native, points, edges, transform)
                estimate = frame.get("local:" + side) if frame else None
                source_mode = tile.model_source if tile.model_source>=0 else self.hand_source.currentIndex()
                if source_mode == 0:
                    tile.hand_view.set_hand(estimate if estimate is not None and np.isfinite(estimate).all() else None,
                        "Camera-local hand · "+frame.get("local_source:"+side,"model estimate") if estimate is not None and np.isfinite(estimate).all() else "No camera-local hand estimate at this source frame",
                        self.center_hand.isChecked())
                else:
                    self._render_hand_motion(tile)
            else:
                projected, fused_edges = self._project_fused(uid, native)
                points.update(projected)
                edges.extend(fused_edges)
                tile.overlay.frame(image, native, points, edges)

    def set_active_page(self, kind):
        self.active_page = kind
        if self._active_view is None or self._active_view[0]!=kind:
            self._select_first_view()
        self.request_frames()
        self._render_frames()

    def _project_fused(self, uid, native):
        doc = self.document
        if doc is None or uid not in doc.calibration.get("cameras", {}):
            return {}, []
        times = doc.frame_timestamps_ns * 1e-9
        if self.global_s < times[0] - .04 or self.global_s > times[-1] + .04:
            return {}, []
        from engine.math_core.camera_model import MultiCameraCalibration
        calibration = MultiCameraCalibration.from_dict(doc.calibration)
        camera = calibration.cameras[uid]
        index = int(np.argmin(abs(times - self.global_s)))
        track = doc.corrected_motion or doc.raw_motion
        positions = track.joint_positions[index].copy()
        transform = np.asarray(track.cleanup_config.get("world_from_capture", np.eye(4)))
        positions = (positions - transform[:3, 3]) @ transform[:3, :3]
        uv = camera.project_world_points(positions, True)
        uv *= [native[0] / camera.intrinsics.image_width, native[1] / camera.intrinsics.image_height]
        points = {f"fused:{i}": {"xy": xy, "name": "Fused corrected " + doc.bone_names[i],
                  "source": "fused", "valid": track.joint_confidence[index, i] > .15}
                  for i, xy in enumerate(uv)}
        edges = [(f"fused:{parent}", f"fused:{i}") for i, parent in enumerate(doc.parent_indices)
                 if parent >= 0 and points[f"fused:{i}"]["valid"] and points[f"fused:{parent}"]["valid"]]
        return points, edges

    def _render_hand_motion(self, tile):
        doc, side = self.document, tile.side
        if doc is None or doc.hands is None:
            tile.hand_view.set_hand(None, "3D hand available after solving")
            return
        times = doc.frame_timestamps_ns * 1e-9
        tolerance = .5 / doc.fps + .001
        if not len(times) or self.global_s < times[0] - tolerance or self.global_s > times[-1] + tolerance:
            tile.hand_view.set_hand(None, "No solved hand at this time")
            return
        index = int(np.argmin(abs(times - self.global_s)))
        rotations = getattr(doc.hands, side + "_local_rotations_xyzw")
        confidence = getattr(doc.hands, side + "_confidence")
        if rotations is None or (confidence is not None and np.max(confidence[index]) < .15):
            tile.hand_view.set_hand(None, "Hand uncertain · estimated, not observed")
            return
        track = doc.corrected_motion or doc.raw_motion
        wrist_index = doc.bone_names.index(side + "_hand")
        sign = -1 if side == "left" else 1
        palm_rest = np.column_stack([[0, -sign, 0], [sign, 0, 0], [0, 0, 1]])
        points = hand_forward_kinematics(rotations[index], side, template=getattr(doc.hands, side + "_template"),
            palm_rotation=track.global_matrices[index, wrist_index, :3, :3] @ palm_rest,
            wrist=track.joint_positions[index, wrist_index])
        tile.hand_view.set_hand(points, "Fused hand articulation · model estimate", self.center_hand.isChecked())

    def _changed(self):
        from engine.math_core.camera_model import MultiCameraCalibration
        self.calibration = (MultiCameraCalibration.from_dict(self.state.calibration)
                            if self.state.calibration else None)
        loading=self._loading
        self._loading=True
        self.use_markers.setChecked(self.state.capture_mode!="markerless")
        self.marker_only.setChecked(self.state.capture_mode=="markers")
        self.marker_only.setEnabled(self.use_markers.isChecked())
        self._loading=loading
        self.mode_note.setText({"markerless": "AI tracking", "hybrid": "Markers + AI", "markers": "Markers only"}[self.state.capture_mode])
        if self.document is not None and self.state.capture_mode=="hybrid":
            marker_report=self.document.processing_config.get("marker_constraints",{})
            if marker_report.get("constrained_samples",0)==0:
                self.mode_note.setText("AI result · no usable calibrated tag constraints")
        self.mode_note.setToolTip(self.mode_note.text())
        self.status.setText((f"Reference: {self.state.reference_camera or 'pending'} · " )
                           + ("Timing aligned" if self.state.alignment_accepted else "Align videos before multi-view solving")
                           + (" · Changes need a new solve" if self.state.stale else " · Solve is current"))
        self.save()
        self.changed.emit()

    def save(self):
        if self.state_path is None:
            return True
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".nuranova-review-", dir=self.state_path.parent) as folder:
                staged = Path(folder) / self.state_path.name
                self.state.save(staged)
                self.observations.save(staged.with_suffix(".npz"))
                publish_files([(staged, self.state_path),
                               (staged.with_suffix(".npz"), self.state_path.with_suffix(".npz"))])
            self.save_preview_layout()
            return True
        except (ValueError, OSError) as exc:
            self.status.setText("Review could not be saved: " + str(exc))
            return False

    def shutdown(self):
        self.timer.stop()
        self.save()
        self.save_preview_layout()
        self._layout_timer.stop()
        self._resize_timer.stop()
        self._shutting_down = True
        self._layout_loading = True
        self.return_all_views()
        self.decoder.close()

    def clear_source(self):
        self.save_preview_layout()
        self._layout_loading = True
        self._clear_preview_tiles()
        self._layout_loading = False
        self._source_key = None
        self._preview_state = {}
