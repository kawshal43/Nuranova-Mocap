"""Dedicated 'Multi-Video Cinema Lab' PySide6 Workspace Page.

A completely separate studio workspace inside Nuranova Mocap Studio dedicated to achieving
maximum-accuracy offline multi-video mocap:
- Import a `.ZIP` archive, a Folder of synchronized videos, OR Record Multi-Cam directly inside the app!
- Inspect camera clips, audio availability, auto-detected PyAV sync offsets (with manual ±ms nudge!).
- Choose ChArUco Video Calibration, Active Studio Calibration, or Boardless Human-Motion Self-Calibration.
- Run GPU-accelerated RTMW-x (133-pt) + Full-Res 30 Fingers + 6-DoF Head/52 Face + Zero-Slide Physics.
- Review the objective Studio Quality Scorecard + interactive 3D Character Preview + 1-click Blender export!
"""

from __future__ import annotations

import subprocess
import threading
import math
import numpy as np
from pathlib import Path
from typing import Callable, Dict, Optional

from PySide6.QtCore import Qt, QTimer, Signal, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QTabWidget,
    QMenu,
    QToolButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from desktop.ui.app_controller import StudioController
from desktop.ui.pages.pages_workspace import _header
from desktop.ui.widgets.viewport3d import Viewport3DContainer
from desktop.ui.widgets.camera_review import CameraReviewWorkspace
from multi_video_studio.package_ingest import (
    MultiVideoPackage,
    export_take_to_zip,
    ingest_multi_video_source,
)
from multi_video_studio.pipeline import (
    CinemaPipelineConfig,
    CinemaPipelineResult,
    MultiVideoCinemaPipeline,
)
from multi_video_studio.pyav_sync import estimate_video_sync_offsets


class MultiVideoCinemaLabPage(QWidget):
    """Dedicated standalone workspace for Cinema-Grade Multi-Video Offline Mocap (Body + 30 Fingers + Head/Face)."""

    _progress_sig = Signal(float, str)
    _finished_sig = Signal(object)
    _failed_sig = Signal(str)
    _gpu_stat_sig = Signal(str)
    _sync_finished_sig = Signal(object)
    _sync_failed_sig = Signal(str)
    _review_finished_sig = Signal(object)
    _review_failed_sig = Signal(str)
    _camera_refined_sig = Signal(object)
    workflow_changed = Signal()

    def __init__(self, ctrl: StudioController, navigate: Callable[[str], None]) -> None:
        super().__init__()
        self.ctrl = ctrl
        self.navigate = navigate
        self._package: Optional[MultiVideoPackage] = None
        self._last_result: Optional[CinemaPipelineResult] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._cancel_evt = threading.Event()
        self._play_frame: int = 0
        self._playing: bool = False
        self._syncing_timeline = False
        self._closing = False
        self._sync_busy = False
        self._review_busy = False
        project = ctrl.project_manager.active_project
        self._project_id = project.project_id if project else None

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)
        root.addLayout(
            _header(
                "Motion capture from videos",
                "1. Add videos of the same performance.  2. Check timing and solve.  3. Review the motion and export.",
            )
        )

        # ---------------- TOP BAR: 3 Input Sources + GPU Badge ----------------
        src_card = QFrame()
        self.source_controls = src_card
        src_card.setObjectName("Card")
        src_l = QHBoxLayout(src_card)
        src_l.setContentsMargins(12, 9, 12, 9)
        src_l.setSpacing(8)

        btn_zip = QPushButton("Open ZIP")
        btn_zip.setObjectName("PrimaryButton")
        btn_zip.setMinimumHeight(34)
        btn_zip.setToolTip(
            "Import a .zip containing multi-camera MP4/MOV videos (optional calib/ + motion/ folders inside!)"
        )
        btn_zip.clicked.connect(self._import_zip)

        btn_folder = QPushButton("Add folder")
        btn_folder.setObjectName("PrimaryButton")
        btn_folder.setMinimumHeight(34)
        btn_folder.setToolTip(
            "Import a folder of synchronized camera videos recorded on phones, USB cameras, GoPros, or DSLRs"
        )
        btn_folder.clicked.connect(self._import_folder)

        btn_files = QPushButton("Add videos")
        btn_files.setObjectName("PrimaryButton")
        btn_files.clicked.connect(self._import_files)

        btn_load_rec = QPushButton("🎥  Load In-App Recorded Take")
        btn_load_rec.setMinimumHeight(34)
        btn_load_rec.setToolTip(
            "Load the latest multi-camera take recorded inside Capture Studio (phones + USB)"
        )
        btn_load_rec.clicked.connect(self._load_latest_in_app_take)

        btn_record_now = QPushButton("●  Record New Take In-App →")
        btn_record_now.setMinimumHeight(34)
        btn_record_now.clicked.connect(lambda: self.navigate("Capture Studio"))

        btn_pack_zip = QPushButton("💾  Save Package as .ZIP")
        btn_pack_zip.setToolTip(
            "Bundle the current camera videos + calibration into one portable .zip archive"
        )
        btn_pack_zip.clicked.connect(self._export_package_zip)

        hw = ctrl.hardware
        gpu_txt = f"⚡ GPU CUDA: {hw.gpu_name}" if hw.cuda_usable else f"CPU ({hw.cpu_count} cores)"
        self.lbl_gpu = QLabel(gpu_txt)
        self.lbl_gpu.setObjectName("BadgeOk" if hw.cuda_usable else "BadgeWarn")

        more = QToolButton()
        more.setText("More")
        more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(more)
        for text, handler in (
            ("Open ZIP package…", self._import_zip),
            ("Load camera calibration…", self._import_calibration),
            ("Load recorded take", self._load_latest_in_app_take),
            ("Record a new take", lambda: self.navigate("Capture Studio")),
            ("Save video package…", self._export_package_zip),
        ):
            menu.addAction(text, handler)
        more.setMenu(menu)
        # These actions are available from More without crowding the import row.
        for unused in (btn_zip, btn_load_rec, btn_record_now, btn_pack_zip):
            unused.deleteLater()
        for w in (btn_files, btn_folder, more):
            src_l.addWidget(w)
        src_l.addStretch()
        src_l.addWidget(self.lbl_gpu)
        root.addWidget(src_card)
        self.review = CameraReviewWorkspace(ctrl._base_state_dir / "take_review")
        root.addWidget(self.review)
        self.review.align_requested.connect(self._precheck_sync)
        self.review.detection_requested.connect(self._detect_review_points)
        self.review.recalibrate_requested.connect(self._refine_review_camera)

        # ---------------- MAIN SPLITTER: Left (Clips + Settings + Solve) | Right (3D Preview + Scorecard) ----------------
        split = QTabWidget()
        split.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        split.setMinimumHeight(120)
        self.workflow_tabs = split
        split.currentChanged.connect(lambda index: self.review.set_active_page(
            {1: "cameras", 2: "body", 3: "hands"}.get(index, "cameras")))
        split.currentChanged.connect(lambda index: self.source_controls.setVisible(index == 0))

        # LEFT WORKBENCH
        left_w = QWidget()
        left_l = QVBoxLayout(left_w)
        left_l.setContentsMargins(0, 0, 6, 0)
        left_l.setSpacing(8)

        # 1. Loaded Package & Sync Inspector
        pkg_box = QGroupBox("Videos")
        pl = QVBoxLayout(pkg_box)
        pkg_top = QHBoxLayout()
        self.lbl_pkg_summary = QLabel(
            "No multi-video package loaded yet — import a .ZIP, Folder, or In-App Recorded Take above."
        )
        self.lbl_pkg_summary.setObjectName("Subtitle")
        self.lbl_pkg_summary.setWordWrap(True)
        btn_pre_sync = QPushButton("Check timing")
        self.btn_pre_sync = btn_pre_sync
        btn_pre_sync.clicked.connect(self._precheck_sync)
        pkg_top.addWidget(self.lbl_pkg_summary, 1)
        pkg_top.addWidget(btn_pre_sync)
        pl.addLayout(pkg_top)

        self.clip_table = QTableWidget(0, 6)
        self.clip_table.setHorizontalHeaderLabels(
            ["Camera", "Role", "File", "Video Info", "Detected Sync", "Manual Nudge (ms)"]
        )
        self.clip_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.clip_table.setMaximumHeight(150)
        self.clip_table.itemChanged.connect(self._manual_timing_changed)
        pl.addWidget(self.clip_table)
        left_l.addWidget(pkg_box)

        # 2. Calibration & Studio Accuracy Settings
        cfg_box = QGroupBox("Processing settings")
        self.processing_controls = cfg_box
        cl = QVBoxLayout(cfg_box)
        cl.setSpacing(6)

        row_c1 = QGridLayout()
        self.combo_backend = QComboBox()
        self.combo_backend.addItem("RTMW-x 133-pt WholeBody (Best Accuracy)", "rtmlib_wholebody")
        self.combo_backend.addItem("HALPE-26 Body+Feet (Fast)", "rtmlib_halpe26")
        self.combo_backend.addItem("MediaPipe Pose Heavy (Offline Safe)", "mediapipe_pose")

        self.combo_board = QComboBox()
        self.combo_board.addItem("A3 ChArUco (9×6 · 40mm)", "A3")
        self.combo_board.addItem("A4 ChArUco (7×5 · 35mm)", "A4")

        self.spin_height = QDoubleSpinBox()
        self.spin_height.setRange(1.00, 2.30)
        self.spin_height.setSingleStep(0.02)
        self.spin_height.setValue(1.72)
        self.spin_height.setSuffix(" m")
        self.spin_height.setMinimumWidth(90)
        self.spin_height.setToolTip(
            "Actor height for Boardless Human-Motion Self-Calibration (when no ChArUco video is included)"
        )

        row_c1.addWidget(QLabel("Tracking model"), 0, 0)
        row_c1.addWidget(self.combo_backend, 0, 1, 1, 3)
        row_c1.addWidget(QLabel("Calibration board"), 1, 0)
        row_c1.addWidget(self.combo_board, 1, 1)
        row_c1.addWidget(QLabel("Actor height"), 1, 2)
        row_c1.addWidget(self.spin_height, 1, 3)
        cl.addLayout(row_c1)

        row_c2 = QGridLayout()
        self.chk_sync = QCheckBox("Auto-Sync (Audio Clap / Motion)")
        self.chk_sync.setChecked(True)
        self.chk_fingers = QCheckBox("✋ 30 Fingers (3D + Wrist Crop)")
        self.chk_fingers.setChecked(True)
        self.chk_head_face = QCheckBox("🙂 6-DoF Head + 52 Face")
        self.chk_head_face.setChecked(True)
        self.chk_boardless = QCheckBox("Boardless Calib Fallback")
        self.chk_boardless.setChecked(True)
        self.chk_boardless.setToolTip(
            "If no ChArUco board video is present, solve camera positions directly from walking human motion!"
        )
        self.chk_sync.setText("Align video timing automatically")
        self.chk_fingers.setText("Track fingers")
        self.chk_head_face.setText("Track head and face")
        self.chk_boardless.setText("Estimate cameras without a board")
        for i, w in enumerate(
            (self.chk_sync, self.chk_fingers, self.chk_head_face, self.chk_boardless)
        ):
            row_c2.addWidget(w, i // 2, i % 2)
        cl.addLayout(row_c2)

        row_c3 = QHBoxLayout()
        self.spin_zero_slide = QDoubleSpinBox()
        self.spin_zero_slide.setRange(0.0, 1.0)
        self.spin_zero_slide.setSingleStep(0.05)
        self.spin_zero_slide.setValue(0.92)
        self.spin_zero_slide.setMinimumWidth(86)
        self.spin_cutoff = QDoubleSpinBox()
        self.spin_cutoff.setRange(2.0, 15.0)
        self.spin_cutoff.setSingleStep(0.5)
        self.spin_cutoff.setValue(5.5)
        self.spin_cutoff.setSuffix(" Hz")
        self.spin_cutoff.setMinimumWidth(96)
        row_c3.addWidget(QLabel("Foot locking:"))
        row_c3.addWidget(self.spin_zero_slide)
        row_c3.addWidget(QLabel("Smoothing:"))
        row_c3.addWidget(self.spin_cutoff)
        row_c3.addStretch()
        cl.addLayout(row_c3)
        left_l.addWidget(cfg_box)

        # 3. Run Cinema Solve + Progress + Log
        run_row = QHBoxLayout()
        self.btn_solve = QPushButton("Solve motion")
        self.btn_solve.setEnabled(False)
        self.btn_solve.setObjectName("PrimaryButton")
        self.btn_solve.setMinimumHeight(40)
        self.btn_solve.clicked.connect(self._start_cinema_solve)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setObjectName("DangerButton")
        self.btn_cancel.setMinimumHeight(40)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_solve)
        run_row.addWidget(self.btn_solve, 1)
        run_row.addWidget(self.btn_cancel)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Idle — load videos & click Analyze")

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(165)
        self.log.setPlainText(
            "Multi-Video Cinema Lab Ready.\n"
            "• Import a .ZIP archive (optional calib/ + motion/ folders inside) OR a folder of 2–8 camera videos.\n"
            "• Or record synchronized videos directly inside Capture Studio and click 'Load In-App Recorded Take'.\n"
            "• Produces zero-foot-slide 25-bone body + 30 fingers + 6-DoF head + 52 facial expressions ready for Blender!"
        )
        details = QToolButton()
        details.setText("Processing details")
        details.setCheckable(True)
        details.toggled.connect(self.log.setVisible)
        self.log.hide()
        left_l.addWidget(details)
        left_l.addWidget(self.log)
        left_l.addStretch()
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        left_scroll.setWidget(left_w)
        split.addTab(left_scroll, "1. Videos & timing")
        split.addTab(self.review.pages["cameras"], "2. Cameras & calibration")
        split.addTab(self.review.pages["body"], "3. Markers & body")
        split.addTab(self.review.pages["hands"], "4. Hands & fingers")

        # RIGHT WORKBENCH: Studio Quality Scorecard + 3D Character Preview + Blender Handoff
        right_w = QWidget()
        right_l = QVBoxLayout(right_w)
        right_l.setContentsMargins(6, 0, 0, 0)
        right_l.setSpacing(8)

        score_box = QGroupBox("Tracking report")
        sl = QVBoxLayout(score_box)
        self.lbl_grade = QLabel("Run solve to inspect tracking")
        self.lbl_grade.setObjectName("BadgeWarn")
        self.lbl_grade.setWordWrap(True)
        self.lbl_grade.setStyleSheet("font-size: 14px; font-weight: 800; padding: 4px 8px;")
        self.lbl_metrics = QLabel(
            "Foot Slide: --  |  Reproj Error: --  |  30 Fingers: --  |  6-DoF Head: --"
        )
        self.lbl_metrics.setWordWrap(True)
        sl.addWidget(self.lbl_grade)
        sl.addWidget(self.lbl_metrics)
        right_l.addWidget(score_box)

        vp_frame = QFrame()
        vp_frame.setObjectName("Card")
        vpl = QVBoxLayout(vp_frame)
        vpl.setContentsMargins(8, 8, 8, 8)
        self.viewport = Viewport3DContainer()
        self.viewport.canvas.setMinimumHeight(200)
        self.viewport.canvas.set_display_mode("character")
        vpl.addWidget(self.viewport, 1)

        play_row = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.clicked.connect(self._toggle_play)
        self.slider_frame = QSlider(Qt.Orientation.Horizontal)
        self.slider_frame.setRange(0, 0)
        self.slider_frame.valueChanged.connect(self._seek_frame)
        self.lbl_frame = QLabel("0 / 0")
        play_row.addWidget(self.btn_play)
        play_row.addWidget(self.slider_frame, 1)
        play_row.addWidget(self.lbl_frame)
        vpl.addLayout(play_row)
        right_l.addWidget(vp_frame, 1)

        exp_row = QHBoxLayout()
        btn_edit = QPushButton("Open in Motion Editor")
        btn_edit.clicked.connect(lambda: self.navigate("Motion Editor"))
        btn_blender = QPushButton("Send to Blender & Export →")
        btn_blender.setObjectName("PrimaryButton")
        btn_blender.clicked.connect(lambda: self.navigate("Character Retargeting"))
        self.btn_exports = QPushButton("Open exports")
        self.btn_exports.setEnabled(False)
        self.btn_exports.clicked.connect(self._open_exports)
        exp_row.addWidget(btn_edit)
        exp_row.addWidget(btn_blender)
        run_row.addWidget(self.btn_exports)
        right_l.addLayout(exp_row)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        right_scroll.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        right_scroll.setWidget(right_w)
        split.addTab(right_scroll, "5. Character & export")
        root.addWidget(split, 1)
        root.addLayout(run_row)
        root.addWidget(self.progress_bar)

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._tick_playback)
        self._gpu_timer = QTimer(self)
        self._gpu_timer.setInterval(1500)
        self._gpu_timer.timeout.connect(self._poll_live_gpu_stat)
        self._gpu_poll_busy = False

        self._progress_sig.connect(self._on_progress)
        self._finished_sig.connect(self._on_solve_finished)
        self._failed_sig.connect(self._on_solve_failed)
        self._gpu_stat_sig.connect(self.lbl_gpu.setText)
        self._sync_finished_sig.connect(self._on_sync_finished)
        self._sync_failed_sig.connect(self._on_sync_failed)
        self._review_finished_sig.connect(self._on_review_detected)
        self._review_failed_sig.connect(self._on_review_failed)
        self._camera_refined_sig.connect(self._on_camera_refined)
        self.review.time_changed.connect(self._review_time_changed)
        self.review.playback_changed.connect(self._review_playback_changed)
        ctrl.project_changed.connect(self._on_project_changed)
        self._review_options = {"backend":self.combo_backend, "board":self.combo_board, "height":self.spin_height,
            "foot_strength":self.spin_zero_slide, "cutoff":self.spin_cutoff, "sync":self.chk_sync,
            "fingers":self.chk_fingers, "head_face":self.chk_head_face, "boardless":self.chk_boardless}
        for widget in self._review_options.values():
            signal = widget.currentIndexChanged if isinstance(widget,QComboBox) else widget.toggled if isinstance(widget,QCheckBox) else widget.valueChanged
            signal.connect(self._review_option_changed)

    def _review_option_changed(self, *args):
        if self.review.package is None:
            return
        options = {key:(w.currentData() if isinstance(w,QComboBox) else w.isChecked() if isinstance(w,QCheckBox) else w.value())
                   for key,w in self._review_options.items()}
        if options != self.review.state.processing_settings:
            self.review.state.checkpoint()
            self.review.state.processing_settings = options
            self.review._changed()

    # ---------------- Source Import Handlers ----------------
    def _open_exports(self) -> None:
        if self._last_result is not None and self._last_result.h5_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_result.h5_path.parent)))

    def _on_project_changed(self) -> None:
        project = self.ctrl.project_manager.active_project
        project_id = project.project_id if project else None
        if project_id == self._project_id:
            return
        self._project_id = project_id
        if self._review_busy or self.ctrl.offline_processing:
            self._cancel_evt.set()
        self._package = None
        self.review.save()
        self.review.package = self.review.document = None
        self.review.preview_epoch += 1
        self.review.last_frames.clear()
        self.review.timer.stop()
        self.review.play_button.setText("Play cameras")
        self.review._rebuild_grids()
        self._last_result = None
        self._play_timer.stop()
        self._playing = False
        self.btn_play.setText("Play")
        self.clip_table.setRowCount(0)
        self.slider_frame.setRange(0, 0)
        self.btn_solve.setEnabled(False)
        self.btn_exports.setEnabled(False)
        self.lbl_pkg_summary.setText("Add videos to start a new take.")
        self.lbl_grade.setText("No motion loaded")
        self.lbl_metrics.setText("Results will appear after processing.")
        self.viewport.canvas.hide()
        self.viewport.canvas.set_scene_frame(skeleton_positions=None)
        self.workflow_changed.emit()

    def _import_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select camera videos", "", "Videos (*.mp4 *.mov *.mkv *.avi *.m4v *.webm)"
        )
        if paths:
            self._load_source(paths)

    def _import_calibration(self) -> None:
        if self._package is None:
            QMessageBox.information(
                self,
                "Add videos first",
                "Add the camera videos before selecting a calibration profile.",
            )
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Select camera calibration", "", "Calibration JSON (*.json)"
        )
        if path:
            from engine.math_core.camera_model import MultiCameraCalibration

            try:
                MultiCameraCalibration.load(Path(path))
            except (ValueError, OSError, KeyError) as exc:
                QMessageBox.warning(self, "Invalid calibration", str(exc))
                return
            self._package.calibration_json_path = Path(path)
            self.lbl_pkg_summary.setText(
                f"{self._package.num_motion_cameras} videos · Calibration: {Path(path).name} (verified during solve)"
            )

    def _import_zip(self) -> None:
        fp, _ = QFileDialog.getOpenFileName(
            self, "Import Multi-Camera Video Package (.ZIP)", "", "ZIP Archives (*.zip)"
        )
        if not fp:
            return
        self._load_source(Path(fp))

    def _import_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder Containing Multi-Camera Videos"
        )
        if not folder:
            return
        self._load_source(Path(folder))

    def _load_latest_in_app_take(self) -> None:
        takes = self.ctrl.list_recorded_take_folders()
        if not takes:
            QMessageBox.information(
                self,
                "No In-App Recorded Takes",
                "Record a multi-camera take in '4. Capture Studio' first (or import a .ZIP / Folder above).",
            )
            return
        take_id, vids = takes[0]
        self._load_source(list(vids.values()), package_name=take_id)

    def _load_source(self, source, package_name: Optional[str] = None) -> None:
        if self.ctrl.offline_processing or self._sync_busy or self._review_busy:
            return
        try:
            pkg = ingest_multi_video_source(source, package_name=package_name)
            self.review.load_package(pkg, self.ctrl.project_manager.active_project)
            for key, value in self.review.state.processing_settings.items():
                widget = self._review_options.get(key)
                if widget is None:
                    continue
                widget.blockSignals(True)
                if isinstance(widget,QComboBox):
                    widget.setCurrentIndex(max(0,widget.findData(value)))
                elif isinstance(widget,QCheckBox):
                    widget.setChecked(bool(value))
                else:
                    widget.setValue(value)
                widget.blockSignals(False)
        except Exception as exc:
            QMessageBox.warning(self, "Package Import Error", str(exc))
            return
        self._package = pkg
        for uid, clip in pkg.motion_clips.items():
            clip.manual_offset_ms = self.review.state.manual_offsets_ms.get(uid, 0.)
        self._last_result = None
        self.viewport.canvas.hide()
        self.viewport.canvas.set_scene_frame(skeleton_positions=None)
        self.btn_exports.setEnabled(False)
        self._play_timer.stop()
        self._playing = False
        self.btn_play.setText("Play")
        self.slider_frame.setRange(0, 0)
        self.lbl_grade.setText("Ready to solve")
        self.lbl_metrics.setText("Results will appear after processing.")
        self.progress_bar.setValue(0)
        self.btn_solve.setEnabled(pkg.num_motion_cameras > 0)
        self._populate_clip_table()
        cal_txt = (
            f"{pkg.num_calibration_cameras} calib videos"
            if pkg.num_calibration_cameras >= 2
            else (
                "Embedded calib JSON"
                if pkg.calibration_json_path
                else (
                    "Auto-Validated / Boardless"
                    if self.ctrl.active_calibration
                    else "Boardless Auto-Calib"
                )
            )
        )
        self.lbl_pkg_summary.setText(
            f"Loaded '{pkg.package_name}': {pkg.num_motion_cameras} motion camera(s) · Calibration: {cal_txt}"
        )
        self.log.append(
            f"[LOAD] '{pkg.package_name}' ({pkg.num_motion_cameras} motion cams, {pkg.num_calibration_cameras} calib vids)"
        )
        for w in pkg.warnings:
            self.log.append(f"[WARN] {w}")
        self._restore_saved_result()
        self.workflow_changed.emit()

    def _restore_saved_result(self):
        saved = self.review.state.result_path
        if not saved or not Path(saved).is_file():
            return
        try:
            from engine.data_format.motion_io import load_motion_file
            from engine.math_core.camera_model import MultiCameraCalibration
            from multi_video_studio.zero_slide_physics import QualityScorecard
            document = load_motion_file(Path(saved))
            previous = document.processing_config.get("review_state",{}).get("sources",{})
            current = self.review.state.sources
            if not previous or {u:v["asset_id"] for u,v in previous.items()} != {u:v["asset_id"] for u,v in current.items()}:
                raise ValueError("Saved result camera identities differ; re-solve to use the current camera ordering.")
            calibration = MultiCameraCalibration.from_dict(document.calibration) if document.calibration else None
            score = QualityScorecard(**document.processing_config["scorecard"])
            self._last_result = CinemaPipelineResult(document,score,calibration,{},h5_path=Path(saved))
            self.slider_frame.setRange(0,document.num_frames-1)
            self.review.set_document(document,calibration)
            self.viewport.canvas.set_calibration(calibration)
            self.btn_exports.setEnabled(True)
            self.lbl_grade.setText(f"Saved result ({score.grade})")
            self.lbl_metrics.setText(score.summary_line())
            self.ctrl.set_active_motion(document)
            self._review_time_changed(self.review.global_s)
        except (ValueError,KeyError,OSError,TypeError) as exc:
            self._last_result = None
            self.log.append("[REVIEW] Saved result not restored: "+str(exc))

    def _populate_clip_table(self, sync_summaries: Optional[Dict[str, str]] = None) -> None:
        self.clip_table.blockSignals(True)
        self.clip_table.setRowCount(0)
        if self._package is None:
            self.clip_table.blockSignals(False)
            return
        rows = []
        for uid, clip in self._package.motion_clips.items():
            rows.append((uid, "Motion", clip))
        for uid, clip in self._package.calibration_clips.items():
            rows.append((uid, "Calib Board", clip))
        self.clip_table.setRowCount(len(rows))
        for r, (uid, role, clip) in enumerate(rows):
            sync_s = (sync_summaries or {}).get(uid, "Pending (Auto-Sync)")
            vals = [
                uid,
                role,
                clip.path.name,
                clip.summary_label,
                sync_s,
                f"{clip.manual_offset_ms:.1f}",
            ]
            for c, txt in enumerate(vals):
                item = QTableWidgetItem(txt)
                if c != 5:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.clip_table.setItem(r, c, item)
        self.clip_table.blockSignals(False)

    def _manual_timing_changed(self, item):
        if item.column() != 5 or self._package is None:
            return
        try:
            offsets = self._read_manual_offsets_ms()
        except ValueError as exc:
            self.review.status.setText(str(exc))
            return
        uid = self.clip_table.item(item.row(), 0).text()
        previous = self.review.state.manual_offsets_ms.get(uid, 0.)
        current = offsets.get(uid, 0.)
        self.review.state.checkpoint()
        if uid in self.review.state.clocks:
            clock = self.review.state.clocks[uid]
            clock["offset_s"] += (current - previous) / 1000.
            clock["summary"] = f"Manual timing correction: {current:+.1f} ms"
        self.review.state.manual_offsets_ms[uid] = current
        self._package.motion_clips[uid].manual_offset_ms = current
        self.review.state.alignment_accepted = False
        self.review._update_timeline()
        self.review._changed()

    def _read_manual_offsets_ms(self) -> Dict[str, float]:
        offsets: Dict[str, float] = {}
        for r in range(self.clip_table.rowCount()):
            uid_it = self.clip_table.item(r, 0)
            role_it = self.clip_table.item(r, 1)
            off_it = self.clip_table.item(r, 5)
            if uid_it and role_it and role_it.text() == "Motion" and off_it:
                try:
                    val = float(off_it.text().strip())
                    if not math.isfinite(val):
                        raise ValueError("Timing offsets must be finite numbers.")
                    if abs(val) > 1e-6:
                        offsets[uid_it.text()] = val
                except ValueError as exc:
                    raise ValueError(f"Invalid timing offset for {uid_it.text()}.") from exc
        return offsets

    def _precheck_sync(self) -> None:
        if self._package is None or self._package.num_motion_cameras < 2:
            QMessageBox.information(
                self,
                "Need 2+ Videos",
                "Load at least 2 motion videos first to pre-check audio/flash sync.",
            )
            return
        if self._sync_busy or self.ctrl.offline_processing:
            return
        try:
            offsets = self._read_manual_offsets_ms()
        except ValueError as exc:
            QMessageBox.warning(self, "Check timing", str(exc))
            return
        paths = self._package.motion_paths_by_cam()
        self._sync_busy = True
        self.source_controls.setEnabled(False)
        self.btn_solve.setEnabled(False)
        self.btn_pre_sync.setEnabled(False)
        self.clip_table.setEnabled(False)
        self.btn_pre_sync.setText("Checking…")

        def worker():
            try:
                est = estimate_video_sync_offsets(
                    paths, manual_offsets_ms=offsets, enable_motion=False
                )
                if not self._closing:
                    self._sync_finished_sig.emit(est)
            except Exception as exc:
                if not self._closing:
                    self._sync_failed_sig.emit(str(exc))

        threading.Thread(target=worker, name="nuranova-sync-check", daemon=True).start()

    def _on_sync_finished(self, est) -> None:
        self._sync_busy = False
        self.source_controls.setEnabled(True)
        self.btn_solve.setEnabled(True)
        self.btn_pre_sync.setEnabled(True)
        self.clip_table.setEnabled(True)
        self.btn_pre_sync.setText("Check timing")
        if self._package is None:
            self.btn_solve.setEnabled(False)
            return
        summaries = {u: e.summary for u, e in est.items()}
        self.review.set_clocks(est)
        self.review.state.manual_offsets_ms = self._read_manual_offsets_ms()
        self.review.save()
        offsets = self._read_manual_offsets_ms()
        for uid, clip in self._package.motion_clips.items():
            clip.manual_offset_ms = offsets.get(uid, 0.0)
        self._populate_clip_table(sync_summaries=summaries)
        self.log.append(
            "[SYNC] Pre-check: " + " | ".join(f"{u}: {e.summary}" for u, e in est.items())
        )

    def _on_sync_failed(self, err: str) -> None:
        self._on_sync_finished({})
        self.log.append(f"Timing check failed: {err}")
        QMessageBox.warning(self, "Timing check failed", err)

    def _export_package_zip(self) -> None:
        if self._package is None:
            QMessageBox.information(self, "No Package Loaded", "Load a take or folder first.")
            return
        fp, _ = QFileDialog.getSaveFileName(
            self,
            "Save Portable Multi-Video Package (.ZIP)",
            f"{self._package.package_name}.zip",
            "ZIP Archive (*.zip)",
        )
        if not fp:
            return
        cal_json = self._package.calibration_json_path
        self.review.save()
        out = export_take_to_zip(
            motion_videos=self._package.motion_paths_by_cam(),
            output_zip=fp,
            calibration_videos=self._package.calibration_paths_by_cam() or None,
            calibration_json=cal_json,
            review_json=self.review.state_path,
            review_observations=self.review.state_path.with_suffix(".npz") if self.review.state_path else None,
        )
        self.log.append(f"[ZIP] Saved portable multi-video package to {out}")

    # ---------------- Cinema Solve Execution ----------------
    def _start_cinema_solve(self) -> None:
        if self._package is None:
            QMessageBox.information(
                self,
                "No Videos Loaded",
                "Import a .ZIP package, Video Folder, or In-App Recorded Take first.",
            )
            return
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        if self._review_busy or self._sync_busy:
            return
        try:
            manual_offsets = self._read_manual_offsets_ms()
        except ValueError as exc:
            QMessageBox.warning(self, "Check timing offsets", str(exc))
            return

        cfg = CinemaPipelineConfig(
            ai_backend=str(self.combo_backend.currentData() or "rtmlib_wholebody"),
            ai_mode="performance",
            prefer_gpu=self.ctrl.settings.processing.compute_backend != "cpu",
            enable_auto_sync=self.chk_sync.isChecked(),
            manual_offsets_ms=manual_offsets,
            enable_fingers=self.chk_fingers.isChecked(),
            enable_head_face=self.chk_head_face.isChecked(),
            zero_slide_strength=float(self.spin_zero_slide.value()),
            smooth_cutoff_hz=float(self.spin_cutoff.value()),
            board_preset=str(self.combo_board.currentData() or "A3"),
            actor_height_m=float(self.spin_height.value()),
            allow_boardless_calibration=self.chk_boardless.isChecked(),
            review_state=self.review.state.to_dict(),
            review_observations_path=str(self.review.state_path.with_suffix(".npz")) if self.review.state_path else None,
        )
        for uid, clip in self._package.motion_clips.items():
            clip.manual_offset_ms = manual_offsets.get(uid, 0.0)

        if self.ctrl.project_manager.active_project is not None:
            out_dir = self.ctrl.project_manager.active_project.subdir("exports")
            proj_id = self.ctrl.project_manager.active_project.project_id
        else:
            out_dir = self.ctrl._base_state_dir / "cinema_exports"
            proj_id = "multi_video_cinema"

        self._cancel_evt.clear()
        self.ctrl.offline_processing = True
        self.source_controls.setEnabled(False)
        self.processing_controls.setEnabled(False)
        self.btn_pre_sync.setEnabled(False)
        self.clip_table.setEnabled(False)
        self.btn_solve.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setValue(2)
        self.progress_bar.setFormat("Starting Multi-Video Cinema Pipeline...")
        self.log.append(
            f"[SOLVE] Starting Cinema pipeline on '{self._package.package_name}' ({cfg.ai_backend})..."
        )

        pkg = self._package
        existing_cal = self.ctrl.active_calibration
        if self.review.state.calibration:
            from engine.math_core.camera_model import MultiCameraCalibration
            existing_cal = MultiCameraCalibration.from_dict(self.review.state.calibration)

        def _worker() -> None:
            def progress(frac, message):
                if not self._closing:
                    self._progress_sig.emit(float(frac), str(message))

            try:
                pipe = MultiVideoCinemaPipeline(cfg)
                res = pipe.run_from_source(
                    source=pkg,
                    output_dir=out_dir,
                    existing_calibration=existing_cal,
                    project_id=proj_id,
                    take_name=pkg.package_name,
                    progress_cb=progress,
                    cancel_event=self._cancel_evt,
                )
                if not self._closing:
                    self._finished_sig.emit(res)
            except Exception as exc:
                if not self._closing:
                    self._failed_sig.emit(str(exc))

        self._worker_thread = threading.Thread(
            target=_worker, name="nuranova-cinema-solve", daemon=True
        )
        self._worker_thread.start()
        if self.ctrl.hardware.cuda_usable:
            self._gpu_timer.start()

    def _poll_live_gpu_stat(self) -> None:
        if self._gpu_poll_busy:
            return
        self._gpu_poll_busy = True

        def _query() -> None:
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=1.2,
                ).strip()
                if out and not self._closing:
                    u, used, tot = [p.strip() for p in out.splitlines()[0].split(",")[:3]]
                    gb_u, gb_t = float(used) / 1024.0, float(tot) / 1024.0
                    self._gpu_stat_sig.emit(
                        f"⚡ GPU CUDA: {self.ctrl.hardware.gpu_name} · {u}% · {gb_u:.1f}/{gb_t:.1f} GB"
                    )
            except Exception:
                pass
            finally:
                self._gpu_poll_busy = False

        threading.Thread(target=_query, daemon=True).start()

    def _cancel_solve(self) -> None:
        self._cancel_evt.set()
        self.progress_bar.setFormat("Cancelling...")

    def _on_progress(self, frac: float, msg: str) -> None:
        self.progress_bar.setValue(int(max(0, min(100, round(frac * 100.0)))))
        self.progress_bar.setFormat(f"{int(frac * 100)}% — {msg}")
        if any(
            tag in msg for tag in ("Loading ", "model ready", "[FALLBACK]", "Skipped unrelated")
        ):
            self.log.append(f"[INFO] {msg}")

    def _on_solve_finished(self, res: CinemaPipelineResult) -> None:
        if self._cancel_evt.is_set():
            self._on_solve_failed("Cancelled by user.")
            return
        self._restore_controls()
        self._gpu_timer.stop()
        if self.ctrl.hardware.cuda_usable:
            self.lbl_gpu.setText(f"⚡ GPU CUDA: {self.ctrl.hardware.gpu_name}")
        self._last_result = res
        self.review.set_document(res.document, res.calibration)
        self.review.state.result_path = str(res.h5_path) if res.h5_path else None
        self.review.save()
        self.btn_exports.setEnabled(True)
        self.btn_solve.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat(
            f"✓ Complete in {res.elapsed_s:.1f}s — {res.scorecard.summary_line()}"
        )

        # Sync table with actual measured offsets
        self._populate_clip_table(
            sync_summaries={u: e.summary for u, e in res.sync_estimates.items()}
        )

        sc = res.scorecard
        self.lbl_grade.setText(
            f"Tracking grade: {sc.grade}  ({sc.num_frames} frames @ {sc.fps:.1f} FPS · {sc.num_cameras} cams)"
        )
        self.lbl_grade.setObjectName("BadgeOk" if sc.grade in ("A+", "A") else "BadgeWarn")
        self.lbl_grade.style().unpolish(self.lbl_grade)
        self.lbl_grade.style().polish(self.lbl_grade)
        self.lbl_metrics.setText(
            f"Foot Slide: {sc.raw_foot_slide_mm:.1f} mm → {sc.final_foot_slide_mm:.1f} mm (P95: {sc.max_foot_slide_mm:.1f} mm)  |  "
            f"Floor Pen: {sc.floor_penetration_mm:.1f} mm  |  Bone Std: {sc.bone_length_std_mm:.3f} mm  |  "
            f"Reproj: {sc.mean_reprojection_rmse_px:.2f} px  |  30 Fingers: {sc.finger_coverage_pct:.0f}%  |  6-DoF Head: {sc.head_coverage_pct:.0f}%"
        )
        audit = res.document.processing_config.get("keyframe_audit", {})
        if audit:
            body = audit.get("body", {})
            hand_jump = max(
                (
                    audit.get(side + "_hand", {}).get("max_step_deg", 0)
                    for side in ("left", "right")
                ),
                default=0,
            )
            self.lbl_metrics.setText(
                self.lbl_metrics.text()
                + f"\nLargest adjacent-frame turn: body {body.get('max_step_deg', 0):.1f}° · fingers {hand_jump:.1f}°. Review hidden joints; detection coverage is not motion accuracy."
            )

        # Register into Studio Controller & Blender Bridge automatically!
        if res.calibration is not None:
            self.ctrl.set_active_calibration(res.calibration)
        self.ctrl.set_active_motion(res.document)
        if res.blender_json_path is not None:
            self.ctrl.network_server.register_bridge_take(
                res.document.take_id, res.blender_json_path
            )
        if self.ctrl.project_manager.active_project is not None:
            self.ctrl.project_manager.register_take(
                take_id=res.document.take_id,
                session_id="multi_video_cinema_lab",
                name=f"{res.document.take_id} [Cinema {sc.grade}]",
                capture_mode=res.document.capture_mode,
                fps=res.document.fps,
                num_frames=res.document.num_frames,
                motion_h5_path=res.h5_path,
                blender_json_path=res.blender_json_path,
            )
            self.ctrl.project_changed.emit()

        self.log.append(f"[✓ DONE in {res.elapsed_s:.1f}s] {sc.summary_line()}")
        for note in sc.notes:
            self.log.append(f"  • {note}")
        if res.blender_json_path:
            self.log.append(f"  • Ready for Blender Bridge: {res.blender_json_path}")

        # Load into 3D viewport & start preview playback
        self.viewport.canvas.set_calibration(res.calibration)
        self.slider_frame.setRange(0, max(0, res.document.num_frames - 1))
        self._seek_frame(0)
        self.workflow_tabs.setCurrentIndex(4)
        self.workflow_changed.emit()
        self.ctrl.status_message.emit(
            f"Multi-Video Cinema solve finished ({sc.grade}) — ready for Blender!"
        )

    def _on_solve_failed(self, err: str) -> None:
        self._restore_controls()
        self._gpu_timer.stop()
        if self.ctrl.hardware.cuda_usable:
            self.lbl_gpu.setText(f"⚡ GPU CUDA: {self.ctrl.hardware.gpu_name}")
        self.btn_solve.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setFormat(f"Failed: {err}")
        self.log.append(f"[ERROR] {err}")
        if not self._cancel_evt.is_set():
            QMessageBox.warning(self, "Multi-Video Cinema Error", err)

    def _restore_controls(self) -> None:
        self.ctrl.offline_processing = False
        self.source_controls.setEnabled(True)
        self.processing_controls.setEnabled(True)
        self.btn_pre_sync.setEnabled(True)
        self.clip_table.setEnabled(True)

    def shutdown(self) -> None:
        self._closing = True
        self._cancel_evt.set()
        self._play_timer.stop()
        self._gpu_timer.stop()
        self.review.shutdown()

    def _detect_review_points(self, cameras, start_s, end_s, current):
        if self._package is None or self._review_busy or self.ctrl.offline_processing or self._sync_busy:
            return
        from multi_video_studio.review_state import ReviewState
        from multi_video_studio.review_jobs import detect_review_points
        package, state = self._package, ReviewState.from_dict(self.review.state.to_dict())
        backend = str(self.combo_backend.currentData())
        prefer_gpu = self.ctrl.settings.processing.compute_backend != "cpu"
        self._review_busy = True
        self.source_controls.setEnabled(False)
        self.btn_solve.setEnabled(False)
        self._cancel_evt.clear()
        self.btn_cancel.setEnabled(True)

        def work():
            try:
                result = detect_review_points(package, state, cameras, start_s, end_s, current,
                    backend, prefer_gpu, lambda p, m: self._progress_sig.emit(p, m), self._cancel_evt)
                if not self._closing:
                    self._review_finished_sig.emit(result)
            except Exception as exc:
                if not self._closing:
                    self._review_failed_sig.emit(str(exc))
        threading.Thread(target=work, name="nuranova-point-review", daemon=True).start()

    def _on_review_detected(self, result):
        if self._package is None or self._cancel_evt.is_set():
            self._finish_review_job()
            return
        self.review.observations.merge(result)
        self.review.state.checkpoint()
        self.review._changed()
        self.review._render_frames()
        self._finish_review_job()
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Detection complete · manual corrections preserved")

    def _finish_review_job(self):
        self._review_busy = False
        self.source_controls.setEnabled(True)
        self.btn_solve.setEnabled(self._package is not None)
        self.btn_cancel.setEnabled(False)

    def _on_review_failed(self, message):
        self._finish_review_job()
        self.review.status.setText(message)
        self.log.append("[REVIEW] " + message)
        self.progress_bar.setFormat(message)

    def _refine_review_camera(self, uid):
        if self._review_busy or self.ctrl.offline_processing or not uid:
            return
        from engine.math_core.camera_model import MultiCameraCalibration
        from multi_video_studio.review_jobs import refine_review_camera
        from multi_video_studio.review_state import ReviewState
        calibration = MultiCameraCalibration.from_dict(self.review.state.calibration) if self.review.state.calibration else self.ctrl.active_calibration
        if calibration is None:
            self.review.status.setText("Solve or load an initial calibration, then refine this camera.")
            return
        self._review_busy = True
        state = ReviewState.from_dict(self.review.state.to_dict())
        self._cancel_evt.clear()
        # Freeze inference arrays while the background job reads them.
        from copy import deepcopy
        observations = deepcopy(self.review.observations)
        self.btn_solve.setEnabled(False)
        self.source_controls.setEnabled(False)

        def work():
            try:
                result = refine_review_camera(calibration, uid, observations, state)
                if not self._closing:
                    self._camera_refined_sig.emit(result)
            except Exception as exc:
                if not self._closing:
                    self._review_failed_sig.emit(str(exc))
        threading.Thread(target=work, name="nuranova-camera-refinement", daemon=True).start()

    def _on_camera_refined(self, result):
        if self._package is None or self._cancel_evt.is_set():
            self._finish_review_job()
            return
        calibration, report = result
        self.review.state.checkpoint()
        self.review.state.calibration = calibration.to_dict()
        self.review.calibration = calibration
        self.review._changed()
        self._finish_review_job()
        message = (f"{report['camera_uid']} refined · held-out median {report['before_median_px']:.2f} → "
                   f"{report['after_median_px']:.2f}px · P95 {report['after_p95_px']:.2f}px. Undo restores the previous calibration.")
        self.review.status.setText(message)
        self.log.append("[CALIBRATION] " + message)

    # ---------------- 3D Preview Playback ----------------
    def _toggle_play(self) -> None:
        if self._last_result is None or self._last_result.document.num_frames < 2:
            return
        self.review._toggle_play()

    def _review_playback_changed(self, playing):
        self._playing = playing
        self.btn_play.setText("Pause" if playing else "Play")

    def _review_time_changed(self, seconds):
        if self._syncing_timeline or self._last_result is None:
            return
        doc = self._last_result.document
        times = doc.frame_timestamps_ns * 1e-9
        if not len(times):
            return
        tolerance = .5 / doc.fps + .001
        if seconds < times[0]-tolerance or seconds > times[-1]+tolerance:
            self.viewport.canvas.hide()
            self.lbl_frame.setText("No solved pose at this time")
            return
        index = int(np.argmin(abs(times-seconds)))
        self._syncing_timeline = True
        try:
            self.slider_frame.blockSignals(True)
            self.slider_frame.setValue(index)
            self.slider_frame.blockSignals(False)
            self._seek_frame(index)
        finally:
            self._syncing_timeline = False

    def _tick_playback(self) -> None:
        if self._last_result is None:
            self._play_timer.stop()
            return
        n = self._last_result.document.num_frames
        if n <= 1:
            return
        nxt = (self._play_frame + 1) % n
        self.slider_frame.setValue(nxt)

    def _seek_frame(self, idx: int) -> None:
        if self._last_result is None:
            return
        doc = self._last_result.document
        trk = doc.corrected_motion or doc.raw_motion
        if trk is None or trk.num_frames == 0:
            return
        f = max(0, min(trk.num_frames - 1, int(idx)))
        self._play_frame = f
        self.viewport.canvas.show()
        self.lbl_frame.setText(f"{f + 1} / {trk.num_frames}")
        lh_q = (
            doc.hands.left_local_rotations_xyzw[f]
            if (
                doc.hands is not None
                and doc.hands.left_local_rotations_xyzw is not None
                and f < len(doc.hands.left_local_rotations_xyzw)
            )
            else None
        )
        rh_q = (
            doc.hands.right_local_rotations_xyzw[f]
            if (
                doc.hands is not None
                and doc.hands.right_local_rotations_xyzw is not None
                and f < len(doc.hands.right_local_rotations_xyzw)
            )
            else None
        )
        self.viewport.canvas.set_scene_frame(
            skeleton_positions=trk.joint_positions[f],
            skeleton_confidence=trk.joint_confidence[f],
            global_matrices=trk.global_matrices[f] if trk.global_matrices is not None else None,
            left_hand_quats=lh_q,
            right_hand_quats=rh_q,
        )
        if not self._syncing_timeline:
            self._syncing_timeline = True
            try:
                self.review.time_slider.setValue(round(float(doc.frame_timestamps_ns[f])*1e-6))
            finally:
                self._syncing_timeline = False
