"""Synchronized camera/point/hand review using source PTS and reversible annotations."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import queue
import threading
import time

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap, QTransform
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QGraphicsEllipseItem,
    QGraphicsItem, QGraphicsLineItem, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene,
    QGraphicsSimpleTextItem, QGraphicsView, QGridLayout, QGroupBox, QHBoxLayout,
    QInputDialog, QLabel, QListWidget, QListWidgetItem, QMenu, QMessageBox, QPushButton,
    QScrollArea, QSlider, QVBoxLayout, QWidget, QWidgetAction,
)

from engine.hands.kinematics import hand_forward_kinematics
from multi_video_studio.review_observations import ReviewObservations
from multi_video_studio.review_state import BODY_POINT_NAMES, HAND_POINT_NAMES, ReviewState, point_name
from multi_video_studio.review_video import ReviewVideoReader, hand_crop

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
        self.worker.start()

    def request(self, generation, requests):
        try:
            self.jobs.get_nowait()
        except queue.Empty:
            pass
        self.jobs.put_nowait((generation, requests))

    def _run(self):
        readers = OrderedDict()
        try:
            while not self.stopped.is_set():
                try:
                    generation, requests = self.jobs.get(timeout=.1)
                except queue.Empty:
                    continue
                output = {}
                for uid, path, pts, max_width in requests:
                    if self.stopped.is_set():
                        break
                    try:
                        key = str(path)
                        if key not in readers:
                            readers[key] = ReviewVideoReader(Path(path), cache_frames=2)
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


class EditablePoint(QGraphicsEllipseItem):
    def __init__(self, key, xy, name, source, callback):
        super().__init__(-5, -5, 10, 10)
        self.key, self.callback = key, callback
        self.original = QPointF(*xy)
        self.setPos(self.original)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        if source == "fused":
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setAcceptHoverEvents(True)
        self.setZValue(5)
        color = "#38bdf8" if source == "manual" else "#34d399"
        self.setBrush(QBrush(QColor(color)))
        self.setPen(QPen(QColor("#ffffff"), 1))
        self.setToolTip(name + "\n" + source)
        self.label = QGraphicsSimpleTextItem(name, self)
        self.label.setBrush(QBrush(QColor("#ffffff")))
        self.label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.label.setPos(9, -18)
        self.label.setVisible(False)

    def hoverEnterEvent(self, event):
        self.label.setVisible(True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.label.setVisible(self.isSelected())
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        self.original = self.pos()
        self.callback(self.key, None)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if (self.pos() - self.original).manhattanLength() > .01:
            self.callback(self.key, self.pos())


class PointOverlay(QGraphicsView):
    selected = Signal(str)
    edited = Signal(str, object)
    roi_edited = Signal(object)

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setMinimumSize(250, 170)
        self.setMouseTracking(True)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.native_to_scene = QTransform()
        self.edit_key = None
        self._has_frame = False
        self.select_roi = False
        self.roi_start = self.roi_item = None
        self.setBackgroundBrush(QBrush(QColor("#111827")))

    def frame(self, image, native_size, points, edges=(), transform=None):
        self.scene().clear()
        self._has_frame = True
        image = np.ascontiguousarray(image)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimage = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        item = QGraphicsPixmapItem(QPixmap.fromImage(qimage))
        self.scene().addItem(item)
        if transform is None:
            nw, nh = native_size
            item.setTransform(QTransform.fromScale(nw / w, nh / h))
            self.native_to_scene = QTransform()
        else:
            nw, nh = w, h
            self.native_to_scene = QTransform(float(transform[0, 0]), float(transform[1, 0]),
                                              float(transform[0, 1]), float(transform[1, 1]),
                                              float(transform[0, 2]), float(transform[1, 2]))
        for a, b in edges:
            if a in points and b in points:
                p, q = [self.native_to_scene.map(QPointF(*points[k]["xy"])) for k in (a, b)]
                line = QGraphicsLineItem(p.x(), p.y(), q.x(), q.y())
                pen = QPen(QColor("#22c55e"), 1.5)
                pen.setCosmetic(True)
                line.setPen(pen)
                self.scene().addItem(line)
        for key, point in points.items():
            if not point.get("valid", True) or not np.isfinite(point["xy"]).all():
                continue
            xy = self.native_to_scene.map(QPointF(*point["xy"]))
            mark = EditablePoint(key, (xy.x(), xy.y()), point["name"], point.get("source", "detected"), self._point_event)
            self.scene().addItem(mark)
        self.scene().setSceneRect(0, 0, nw, nh)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def placeholder(self, text):
        self.scene().clear()
        self._has_frame = False
        label = self.scene().addText(text)
        label.setDefaultTextColor(QColor("#cbd5e1"))
        self.scene().setSceneRect(0, 0, 350, 200)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _point_event(self, key, xy):
        self.selected.emit(key)
        if xy is not None:
            inverse, okay = self.native_to_scene.inverted()
            if okay:
                native = inverse.map(xy)
                self.edited.emit(key, [native.x(), native.y()])

    def mousePressEvent(self, event):
        if self.select_roi and self._has_frame and event.button() == Qt.MouseButton.LeftButton:
            self.roi_start = self.mapToScene(event.pos())
            self.roi_item = QGraphicsRectItem(QRectF(self.roi_start, self.roi_start))
            self.roi_item.setPen(QPen(QColor("#38bdf8"), 3))
            self.scene().addItem(self.roi_item)
            return
        if self._has_frame and self.edit_key and event.button() == Qt.MouseButton.LeftButton:
            if not isinstance(self.itemAt(event.pos()), EditablePoint):
                self._point_event(self.edit_key, self.mapToScene(event.pos()))
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.select_roi and self.roi_item is not None:
            self.roi_item.setRect(QRectF(self.roi_start, self.mapToScene(event.pos())).normalized())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.select_roi and self.roi_item is not None:
            rectangle = self.roi_item.rect()
            self.roi_item, self.select_roi = None, False
            inverse, okay = self.native_to_scene.inverted()
            if okay:
                a, b = inverse.map(rectangle.topLeft()), inverse.map(rectangle.bottomRight())
                self.roi_edited.emit([a.x(), a.y(), b.x() - a.x(), b.y() - a.y()])
            return
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        self.scale(1.2 if event.angleDelta().y() > 0 else 1 / 1.2,
                   1.2 if event.angleDelta().y() > 0 else 1 / 1.2)


class HandPoseView(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(150)
        self.points = None
        self.reference = None
        self.center = True
        self.caption = "3D hand available after solving"

    def set_hand(self, points, caption, center=True):
        self.points = np.asarray(points, float) if points is not None else None
        self.caption, self.center = caption, center
        if self.reference is None and self.points is not None:
            self.reference = self.points[0].copy()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0c1422"))
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(QRectF(6,4,self.width()-12,36), Qt.TextFlag.TextWordWrap, self.caption)
        if self.points is None or not np.isfinite(self.points).all():
            return
        pts = self.points - (self.points[0] if self.center else self.reference)
        # An oblique local view reveals articulation without inventing a camera depth measurement.
        x = pts[:, 0] + .3 * pts[:, 2]
        y = -pts[:, 1] + .3 * pts[:, 2]
        coordinates = np.column_stack([x,y])
        if self.center:
            coordinates -= (coordinates.min(axis=0) + coordinates.max(axis=0)) / 2
        scale = min((self.width()-24)/.28,(self.height()-42)/.28)
        xy = coordinates * scale + [self.width()/2,(self.height()+24)/2]
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#34d399"), 2))
        for a, b in HAND_EDGES:
            painter.drawLine(QPointF(*xy[a]), QPointF(*xy[b]))
        painter.setBrush(QColor("#e2e8f0"))
        for pt in xy:
            painter.drawEllipse(QPointF(*pt), 3, 3)


class CameraTile(QGroupBox):
    point_selected = Signal(str, str, float)
    point_edited = Signal(str, str, float, object)

    def __init__(self, uid, side=None):
        super().__init__(uid + (" · " + side + " hand" if side else ""))
        self.uid, self.side, self.pts_s = uid, side, 0.
        layout = QVBoxLayout(self)
        self.status = QLabel("Waiting for frame")
        self.status.setWordWrap(True)
        self.overlay = PointOverlay()
        layout.addWidget(self.status)
        self.hand_view = HandPoseView() if side else None
        if self.hand_view:
            views = QHBoxLayout()
            views.addWidget(self.overlay,2)
            self.hand_view.setMinimumWidth(140)
            views.addWidget(self.hand_view,1)
            layout.addLayout(views,1)
        else:
            layout.addWidget(self.overlay,1)
        self.overlay.selected.connect(lambda key: self.point_selected.emit(self.uid, key, self.pts_s))
        self.overlay.edited.connect(lambda key, xy: self.point_edited.emit(self.uid, key, self.pts_s, xy))


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
        self.mode_note = QLabel("AI tracking")
        row.addWidget(QLabel("Preview"))
        row.addWidget(self.preview_mode)
        row.addWidget(self.camera_pick)
        row.addWidget(self.camera_selector)
        row.addStretch()
        root.addLayout(row)
        modes = QHBoxLayout()
        modes.addWidget(self.use_markers)
        modes.addWidget(self.marker_only)
        modes.addWidget(self.mode_note, 1)
        self.save_button = QPushButton("Save review")
        modes.addWidget(self.save_button)
        root.addLayout(modes)
        timeline = QHBoxLayout()
        self.play_button = QPushButton("Play cameras")
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, 0)
        self.time_label = QLabel("Add videos")
        self.status = QLabel("Timing and calibration review")
        timeline.addWidget(self.play_button)
        timeline.addWidget(self.time_slider, 1)
        timeline.addWidget(self.time_label)
        root.addLayout(timeline)
        root.addWidget(self.status)
        self.preview_mode.currentIndexChanged.connect(self._selection_changed)
        self.camera_pick.currentIndexChanged.connect(self._selection_changed)
        self.selected_cameras.itemChanged.connect(self._subset_changed)
        self.time_slider.valueChanged.connect(self._seek)
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
                    self.full_hand_frame.toggled.connect(self._render_frames)
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
            layout.addLayout(bar)
            if kind == "hands":
                layout.addLayout(hand_options)
                self.hand_edit_tools = QWidget()
                tools_layout = QVBoxLayout(self.hand_edit_tools)
                tools_layout.setContentsMargins(0, 0, 0, 0)
                tools_layout.addLayout(crop_options)
                self.toggle_hand_tools = QPushButton("Point / crop tools")
                self.toggle_hand_tools.setCheckable(True)
                self.toggle_hand_tools.toggled.connect(self.hand_edit_tools.setVisible)
                bar.addWidget(self.toggle_hand_tools)
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
                self.point_details = QLabel("Hover a point for its name; select/drag it to inspect/correct.")
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
            content = QWidget()
            grid = QGridLayout(content)
            scroll.setWidget(content)
            layout.addWidget(scroll, 1)
            self.pages[kind] = page
            self.grids[kind] = grid
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
        self.timer.stop()
        self.play_button.setText("Play cameras")
        self.playback_changed.emit(False)
        self.save()
        self.preview_epoch += 1
        self.last_frames.clear()
        self.package = package
        from multi_video_studio.review_state import source_fingerprint
        identity = "_".join(sorted(source_fingerprint(c.path)[:12] for c in package.motion_clips.values()))
        self.state_path = (project.subdir("tracking") if project else self.state_root) / (identity + ".review.json")
        embedded = getattr(package, "review_path", None)
        load_path = embedded if embedded and Path(embedded).exists() else self.state_path
        self.state = ReviewState.load(load_path) if load_path.exists() else ReviewState()
        remap = self.state.bind_sources(package.motion_clips)
        self.document = None
        self.calibration = None
        if self.state.calibration:
            from engine.math_core.camera_model import MultiCameraCalibration
            self.calibration = MultiCameraCalibration.from_dict(self.state.calibration)
        observation_path = getattr(package, "review_observations_path", None) or self.state_path.with_suffix(".npz")
        self.observations = ReviewObservations.load(observation_path) if observation_path.exists() else ReviewObservations()
        self.observations.cameras = {remap[u]: v for u, v in self.observations.cameras.items() if u in remap}
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
        self.tiles.clear()
        for kind, grid in self.grids.items():
            while grid.count():
                item = grid.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            index = 0
            for uid in self.visible_cameras():
                sides = [None]
                if kind == "hands":
                    sides = ["left", "right"] if self.hand_side.currentIndex() == 0 else ["left" if self.hand_side.currentIndex() == 1 else "right"]
                for side in sides:
                    tile = CameraTile(uid, side)
                    tile.point_selected.connect(self._select_point)
                    tile.point_edited.connect(self._edit_point)
                    if side:
                        tile.overlay.roi_edited.connect(lambda rect, u=uid, s=side: self._store_hand_roi(u, s, rect))
                    grid.addWidget(tile, index // 2, index % 2)
                    self.tiles[(kind, uid, side)] = tile
                    index += 1
        self._update_edit_keys()
        self.request_frames()

    def _update_edit_keys(self):
        key = self.point_choice.currentData() if self.place_point.isChecked() else None
        for (kind, uid, side), tile in self.tiles.items():
            selected = self.hand_point_choice.currentData() if kind=="hands" and self.place_hand_point.isChecked() else key
            tile.overlay.edit_key = selected if kind != "hands" or (selected and selected.startswith("hand:" + side)) else None

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
            self.detection_requested.emit(self.visible_cameras(), self.global_s if current else self.range_start.value(),
                                          self.global_s if current else self.range_end.value(), current)

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
        requests = [(u, self.package.motion_clips[u].path, self.state.local_time(u, self.global_s),
                     None if self.active_page == "hands" else 960)
                    for u in self.visible_cameras()]
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
            if kind != self.active_page:
                continue
            result = self.last_frames.get(uid)
            if result is None:
                tile.overlay.placeholder("No corresponding source frame")
                if tile.hand_view:
                    tile.hand_view.set_hand(None, "Hand not visible")
                continue
            pts, image, native = result
            tile.pts_s = pts
            delta = self.state.global_time(uid, pts) - self.global_s
            tile.status.setText(f"{uid} · source {pts:.3f}s · sampling Δ {delta * 1000:+.1f}ms")
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
                crop = (image, None) if self.full_hand_frame.isChecked() else hand_crop(image, values, native,
                    center=self.center_hand.isChecked(), roi=roi, size_override=size)
                if crop is None:
                    tile.overlay.placeholder("Hand not visible / tracking uncertain")
                else:
                    crop_image, transform = crop
                    tile.overlay.frame(crop_image, native, points, edges, transform)
                estimate = frame.get("local:" + side) if frame else None
                if self.hand_source.currentIndex() == 0:
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
        self.status.setText((f"Reference: {self.state.reference_camera or 'pending'} · " )
                           + ("Timing aligned" if self.state.alignment_accepted else "Align videos before multi-view solving")
                           + (" · Changes need a new solve" if self.state.stale else " · Solve is current"))
        self.save()
        self.changed.emit()

    def save(self):
        if self.state_path is None:
            return
        try:
            self.state.save(self.state_path)
            self.observations.save(self.state_path.with_suffix(".npz"))
        except (ValueError, OSError) as exc:
            self.status.setText("Review could not be saved: " + str(exc))

    def shutdown(self):
        self.timer.stop()
        self.save()
        self.decoder.close()
