"""Pages 8-11: Capture Studio, Recording Library, Processing Queue, and Motion Editor."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from desktop.ui.app_controller import StudioController
from desktop.ui.pages.pages_workspace import _header
from desktop.ui.widgets.curve_editor import CurveInspectorWidget
from desktop.ui.widgets.viewport3d import Viewport3DContainer
from engine.data_format.motion_io import load_motion_file
from engine.motion_cleanup.cleanup import MotionCleanupConfig
from engine.skeleton.rig_definition import BONE_NAMES
from engine.video.video_io import probe_video_metadata


class CaptureStudioPage(QWidget):
    """Page 8: Capture Studio with central 3D viewport, multi-camera preview grid, and synchronized recording controls."""

    def __init__(self, ctrl: StudioController, navigate: Callable[[str], None]) -> None:
        super().__init__()
        self.ctrl = ctrl
        self.navigate = navigate
        self._is_recording = False

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(8)
        root.addLayout(_header("Capture Studio", "Multi-camera live preview grid, 3D capture volume monitor, and synchronized take recording"))

        # Top control ribbon
        ctrl_bar = QHBoxLayout()
        self.btn_rec = QPushButton("Start Synchronized Recording")
        self.btn_rec.setObjectName("PrimaryButton")
        self.btn_rec.clicked.connect(self._toggle_recording)
        btn_import = QPushButton("Import Multi-Cam Video Take...")
        btn_import.clicked.connect(self._import_videos)
        btn_demo = QPushButton("Run Synthetic Demo Capture")
        btn_demo.clicked.connect(self._run_demo)
        self.lbl_status = QLabel("Idle | Calib: None | Sync: --")
        for w in (self.btn_rec, btn_import, btn_demo, self.lbl_status):
            ctrl_bar.addWidget(w)
        ctrl_bar.addStretch()
        root.addLayout(ctrl_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.viewport = Viewport3DContainer()
        splitter.addWidget(self.viewport)

        # Right side: configurable 2-12 camera preview grid
        cam_panel = QGroupBox("Camera Preview Grid (Low-Bandwidth Previews)")
        self.grid_layout = QGridLayout(cam_panel)
        self._preview_labels: dict[str, QLabel] = {}
        splitter.addWidget(cam_panel)
        splitter.setSizes([720, 420])
        root.addWidget(splitter, 1)

        self._preview_timer = QTimer(self)
        self._preview_timer.timeout.connect(self._refresh_previews)
        self._preview_timer.start(250)

        ctrl.cameras_changed.connect(self.refresh)
        ctrl.calibration_changed.connect(self.refresh)
        ctrl.motion_changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        self.viewport.canvas.set_calibration(self.ctrl.active_calibration)
        doc = self.ctrl.active_motion
        if doc is not None and doc.num_frames > 0:
            trk = doc.corrected_motion or doc.raw_motion
            self.viewport.canvas.set_scene_frame(
                skeleton_positions=trk.joint_positions[0],
                skeleton_confidence=trk.joint_confidence[0],
                markers_3d=doc.markers_3d[0] if doc.markers_3d is not None else None,
            )

        # Rebuild camera tiles cleanly
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._preview_labels.clear()

        cams = self.ctrl.camera_manager.list_cameras()
        if not cams:
            empty = QLabel("No cameras connected yet.\nStart Local Server in Camera Manager or click 'Run Synthetic Demo Capture'.")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.grid_layout.addWidget(empty, 0, 0)
        else:
            cols = 2 if len(cams) <= 4 else 3
            for idx, cam in enumerate(cams):
                lbl = QLabel(f"{cam.camera_uid}: {cam.name}\n[{cam.state.value}] {cam.config.width}x{cam.config.height}@{cam.config.target_fps:.0f}fps")
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl.setMinimumSize(160, 105)
                lbl.setStyleSheet("border: 1px solid #3A4B5C; border-radius: 6px; padding: 4px;")
                self.grid_layout.addWidget(lbl, idx // cols, idx % cols)
                self._preview_labels[cam.device_id] = lbl

        cal_s = f"Calibrated ({len(self.ctrl.active_calibration.cameras)} cams)" if self.ctrl.active_calibration else "Uncalibrated"
        active_n = len(self.ctrl.camera_manager.get_active_body_cameras())
        self.lbl_status.setText(f"{'RECORDING' if self._is_recording else 'Ready'} | Active Cams: {active_n} | {cal_s}")

    def _refresh_previews(self) -> None:
        for did, lbl in self._preview_labels.items():
            cam = self.ctrl.camera_manager.get_camera(did)
            if cam and cam.latest_preview_jpeg:
                img = QImage.fromData(cam.latest_preview_jpeg)
                if not img.isNull():
                    lbl.setPixmap(QPixmap.fromImage(img).scaled(lbl.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def _toggle_recording(self) -> None:
        ok, msg = self.ctrl.camera_manager.can_record_multi_view()
        if not self._is_recording and not ok:
            QMessageBox.warning(self, "Cannot Start Multi-View Recording", msg)
            return
        self._is_recording = not self._is_recording
        self.btn_rec.setText("STOP RECORDING" if self._is_recording else "Start Synchronized Recording")
        self.btn_rec.setObjectName("DangerButton" if self._is_recording else "PrimaryButton")
        self.btn_rec.style().unpolish(self.btn_rec)
        self.btn_rec.style().polish(self.btn_rec)
        if self._is_recording:
            self.ctrl.status_message.emit("Armed & dispatched synchronized SCHEDULE_RECORD to active cameras.")
        else:
            self.ctrl.status_message.emit("Dispatched STOP_RECORD; awaiting verified file transfers.")
        self.refresh()

    def _import_videos(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Import Synchronized Multi-Camera Videos", "", "Video Files (*.mp4 *.mov *.avi *.mkv)")
        if not files:
            return
        summaries = []
        for f in files:
            m = probe_video_metadata(f, scan_pts=False)
            self.ctrl.camera_manager.register_imported_or_local_camera(
                device_id=f"import_{Path(f).stem}",
                name=Path(f).name,
                width=m.width,
                height=m.height,
                fps=m.nominal_fps,
                adapter_type="imported_video",
            )
            summaries.append(f"{Path(f).name}: {m.width}x{m.height} @ {m.nominal_fps:.1f}fps ({m.frame_count}f)")
        QMessageBox.information(self, "Imported Videos", "Registered imported camera sources:\n" + "\n".join(summaries))

    def _run_demo(self) -> None:
        self.ctrl.generate_synthetic_demo_take("synthetic_capture_demo")
        self.refresh()


class RecordingLibraryPage(QWidget):
    """Page 9: Recording Library."""

    def __init__(self, ctrl: StudioController, navigate: Callable[[str], None]) -> None:
        super().__init__()
        self.ctrl = ctrl
        self.navigate = navigate
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(10)
        root.addLayout(_header("Recording Library", "Browse indexed project takes, load .nmocap.h5 files, and send takes to Motion Editor"))

        bar = QHBoxLayout()
        btn_open_h5 = QPushButton("Open .nmocap.h5 File...")
        btn_open_h5.setObjectName("PrimaryButton")
        btn_open_h5.clicked.connect(self._open_h5)
        btn_edit = QPushButton("Load Selected into Motion Editor")
        btn_edit.clicked.connect(self._load_selected)
        bar.addWidget(btn_open_h5)
        bar.addWidget(btn_edit)
        bar.addStretch()
        root.addLayout(bar)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Take ID", "Name", "Mode", "FPS", "Frames", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        root.addWidget(self.table, 1)

        ctrl.project_changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        takes = self.ctrl.project_manager.list_takes()
        self.table.setRowCount(len(takes))
        for r, t in enumerate(takes):
            for c, k in enumerate(["take_id", "name", "capture_mode", "fps", "num_frames", "status"]):
                self.table.setItem(r, c, QTableWidgetItem(str(t[k])))

    def _open_h5(self) -> None:
        fp, _ = QFileDialog.getOpenFileName(self, "Open Nuranova Motion File", "", "Nuranova HDF5 (*.nmocap.h5 *.h5)")
        if fp:
            doc = load_motion_file(fp)
            self.ctrl.set_active_motion(doc)
            self.navigate("Motion Editor")

    def _load_selected(self) -> None:
        row = self.table.currentRow()
        takes = self.ctrl.project_manager.list_takes()
        if 0 <= row < len(takes):
            p = takes[row].get("motion_h5_path")
            if p and Path(p).exists():
                self.ctrl.set_active_motion(load_motion_file(p))
                self.navigate("Motion Editor")


class ProcessingQueuePage(QWidget):
    """Page 10: Background Processing Queue & Reconstruction Profile Launcher."""

    def __init__(self, ctrl: StudioController) -> None:
        super().__init__()
        self.ctrl = ctrl
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(10)
        root.addLayout(_header("Processing Queue", "Non-blocking offline reconstruction pipeline (Fast Preview / Standard / Cinema)"))

        launch_box = QGroupBox("Launch Offline Reconstruction Job")
        l_row = QHBoxLayout(launch_box)
        self.combo_mode = QComboBox(); self.combo_mode.addItems(["hybrid", "markerless_ai", "physical_marker"])
        self.combo_profile = QComboBox(); self.combo_profile.addItems(["cinema", "standard", "fast_preview"])
        btn_run = QPushButton("Queue Demo Reconstruction Job")
        btn_run.setObjectName("PrimaryButton")
        btn_run.clicked.connect(self._queue_job)
        btn_cancel = QPushButton("Cancel Selected Job")
        btn_cancel.setObjectName("DangerButton")
        btn_cancel.clicked.connect(self._cancel_selected)
        l_row.addWidget(QLabel("Tracking Mode:")); l_row.addWidget(self.combo_mode)
        l_row.addWidget(QLabel("Profile:")); l_row.addWidget(self.combo_profile)
        l_row.addWidget(btn_run); l_row.addWidget(btn_cancel)
        root.addWidget(launch_box)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Job ID", "Title", "Status", "Progress", "Message"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        root.addWidget(self.table, 1)

        ctrl.jobs_changed.connect(self.refresh)
        self.refresh()

    def _queue_job(self) -> None:
        mode = self.combo_mode.currentText()
        prof = self.combo_profile.currentText()

        def task(prog, cancel):
            prog(0.2, f"Initializing {prof} ({mode}) pipeline...")
            if cancel.is_set():
                return None
            doc = self.ctrl.generate_synthetic_demo_take(f"queued_{prof}_{mode}")
            prog(1.0, f"Completed {doc.num_frames} frames")
            return doc.take_id

        self.ctrl.job_queue.submit(f"Reconstruct ({prof} / {mode})", "reconstruct", task)

    def _cancel_selected(self) -> None:
        row = self.table.currentRow()
        jobs = self.ctrl.job_queue.list_jobs()
        if 0 <= row < len(jobs):
            self.ctrl.job_queue.cancel_job(jobs[row].job_id)

    def refresh(self) -> None:
        jobs = self.ctrl.job_queue.list_jobs()
        self.table.setRowCount(len(jobs))
        for r, j in enumerate(jobs):
            self.table.setItem(r, 0, QTableWidgetItem(j.job_id))
            self.table.setItem(r, 1, QTableWidgetItem(j.title))
            self.table.setItem(r, 2, QTableWidgetItem(j.status.value))
            self.table.setItem(r, 3, QTableWidgetItem(f"{int(j.progress * 100)}%"))
            self.table.setItem(r, 4, QTableWidgetItem(j.message))


class MotionEditorPage(QWidget):
    """Page 11: Interactive Motion Editor with 3D Viewport, Timeline, Curves, Cleanup Controls, and Undo/Redo."""

    def __init__(self, ctrl: StudioController) -> None:
        super().__init__()
        self.ctrl = ctrl
        self._frame = 0
        self._playing = False
        self._playback_timer = QTimer(self)
        self._playback_timer.timeout.connect(self._advance_frame)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(8)
        root.addLayout(_header("Motion Editor", "Inspect 3D skeletal motion, raw vs corrected overlays, animation curves, foot-locking, and non-destructive cleanup"))

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left/Center: 3D Viewport + Timeline + Curve Inspector
        center_w = QWidget()
        center_l = QVBoxLayout(center_w)
        center_l.setContentsMargins(0, 0, 0, 0)
        self.viewport = Viewport3DContainer()
        center_l.addWidget(self.viewport, 3)

        # Playback controls
        pb_row = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.setObjectName("PrimaryButton")
        self.btn_play.clicked.connect(self._toggle_play)
        btn_prev = QPushButton("< Frame")
        btn_prev.clicked.connect(lambda: self.set_frame(self._frame - 1))
        btn_next = QPushButton("Frame >")
        btn_next.clicked.connect(lambda: self.set_frame(self._frame + 1))
        self.chk_loop = QCheckBox("Loop"); self.chk_loop.setChecked(True)
        self.combo_speed = QComboBox(); self.combo_speed.addItems(["0.25x", "0.5x", "1.0x", "1.5x", "2.0x"]); self.combo_speed.setCurrentText("1.0x")
        self.lbl_frame = QLabel("Frame 0 / 0")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.set_frame)
        for w in (self.btn_play, btn_prev, btn_next, self.chk_loop, self.combo_speed, self.lbl_frame):
            pb_row.addWidget(w)
        center_l.addLayout(pb_row)
        center_l.addWidget(self.slider)

        self.curve_widget = CurveInspectorWidget()
        center_l.addWidget(self.curve_widget, 1)
        splitter.addWidget(center_w)

        # Right panel: Joint selector + Cleanup controls + Undo/Redo
        right_w = QWidget()
        right_l = QVBoxLayout(right_w)
        right_l.setContentsMargins(6, 0, 0, 0)

        insp_box = QGroupBox("Joint & Comparison")
        insp_f = QFormLayout(insp_box)
        self.combo_bone = QComboBox(); self.combo_bone.addItems(BONE_NAMES)
        self.combo_bone.currentIndexChanged.connect(self._update_curve_and_viewport)
        self.chk_use_corrected = QCheckBox("Show Corrected Motion"); self.chk_use_corrected.setChecked(True)
        self.chk_use_corrected.toggled.connect(self._update_curve_and_viewport)
        insp_f.addRow("Inspect Bone:", self.combo_bone)
        insp_f.addRow("", self.chk_use_corrected)
        right_l.addWidget(insp_box)

        clean_box = QGroupBox("Non-Destructive Motion Cleanup")
        cf = QFormLayout(clean_box)
        self.spin_strength = QDoubleSpinBox(); self.spin_strength.setRange(0.0, 1.0); self.spin_strength.setSingleStep(0.1); self.spin_strength.setValue(0.75)
        self.combo_filter = QComboBox(); self.combo_filter.addItems(["butterworth", "one_euro", "none"])
        self.spin_cutoff = QDoubleSpinBox(); self.spin_cutoff.setRange(1.0, 25.0); self.spin_cutoff.setValue(7.5); self.spin_cutoff.setSuffix(" Hz")
        self.chk_footlock = QCheckBox("2-Bone IK Foot Locking"); self.chk_footlock.setChecked(True)
        self.chk_ground = QCheckBox("Ground Penetration Clamp"); self.chk_ground.setChecked(True)
        self.chk_limits = QCheckBox("Anatomical Joint Limits"); self.chk_limits.setChecked(True)
        self.spin_trim_in = QSpinBox(); self.spin_trim_in.setRange(0, 100000)
        self.spin_trim_out = QSpinBox(); self.spin_trim_out.setRange(0, 100000)

        cf.addRow("Strength:", self.spin_strength)
        cf.addRow("Filter:", self.combo_filter)
        cf.addRow("Cutoff Hz:", self.spin_cutoff)
        cf.addRow("", self.chk_footlock)
        cf.addRow("", self.chk_ground)
        cf.addRow("", self.chk_limits)
        cf.addRow("Range Start:", self.spin_trim_in)
        cf.addRow("Range End:", self.spin_trim_out)

        btn_apply_clean = QPushButton("Apply Cleanup (Reprocess)")
        btn_apply_clean.setObjectName("PrimaryButton")
        btn_apply_clean.clicked.connect(self._apply_cleanup)
        ur_row = QHBoxLayout()
        btn_undo = QPushButton("Undo"); btn_undo.clicked.connect(self.ctrl.undo_motion_edit)
        btn_redo = QPushButton("Redo"); btn_redo.clicked.connect(self.ctrl.redo_motion_edit)
        ur_row.addWidget(btn_undo); ur_row.addWidget(btn_redo)
        cf.addRow(btn_apply_clean)
        cf.addRow(ur_row)
        right_l.addWidget(clean_box)
        right_l.addStretch()

        splitter.addWidget(right_w)
        splitter.setSizes([820, 310])
        root.addWidget(splitter, 1)

        ctrl.motion_changed.connect(self.refresh)
        ctrl.calibration_changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        self.viewport.canvas.set_calibration(self.ctrl.active_calibration)
        doc = self.ctrl.active_motion
        if doc is None or doc.num_frames == 0:
            self.slider.setRange(0, 0)
            self.lbl_frame.setText("No take loaded")
            return
        self.slider.setRange(0, doc.num_frames - 1)
        self.spin_trim_out.setValue(doc.num_frames)
        self._frame = min(self._frame, doc.num_frames - 1)
        self._update_curve_and_viewport()

    def set_frame(self, frame: int) -> None:
        doc = self.ctrl.active_motion
        if doc is None or doc.num_frames == 0:
            return
        self._frame = max(0, min(doc.num_frames - 1, int(frame)))
        self.slider.blockSignals(True)
        self.slider.setValue(self._frame)
        self.slider.blockSignals(False)
        self.lbl_frame.setText(f"Frame {self._frame + 1} / {doc.num_frames} ({doc.fps:.0f} FPS)")
        self._update_curve_and_viewport()

    def _update_curve_and_viewport(self) -> None:
        doc = self.ctrl.active_motion
        if doc is None or doc.num_frames == 0:
            return
        use_corr = self.chk_use_corrected.isChecked() and doc.corrected_motion is not None
        primary = doc.corrected_motion if use_corr else doc.raw_motion
        b_idx = self.combo_bone.currentIndex()
        self.viewport.canvas.selected_bone_idx = b_idx
        self.viewport.canvas.set_scene_frame(
            skeleton_positions=primary.joint_positions[self._frame],
            skeleton_confidence=primary.joint_confidence[self._frame],
            raw_skeleton_positions=doc.raw_motion.joint_positions[self._frame] if use_corr else None,
            markers_3d=doc.markers_3d[self._frame] if doc.markers_3d is not None else None,
        )
        # Update Curve Inspector with selected bone's local quaternion (or root XYZ for hips)
        if b_idx == 0:
            raw_c = doc.raw_motion.root_translation
            cor_c = doc.corrected_motion.root_translation if doc.corrected_motion else None
            title = "Root Translation (X=Red, Y=Green, Z=Blue) — Dashed: Raw | Solid: Corrected"
        else:
            raw_c = doc.raw_motion.local_rotations_xyzw[:, b_idx, :]
            cor_c = doc.corrected_motion.local_rotations_xyzw[:, b_idx, :] if doc.corrected_motion else None
            title = f"Bone '{BONE_NAMES[b_idx]}' Local Quaternion [x,y,z,w] — Dashed: Raw | Solid: Corrected"
        self.curve_widget.set_data(title, raw_c, cor_c, current_frame=self._frame)

    def _toggle_play(self) -> None:
        self._playing = not self._playing
        if self._playing:
            fps = self.ctrl.active_motion.fps if self.ctrl.active_motion else 60.0
            spd = float(self.combo_speed.currentText().replace("x", ""))
            interval_ms = max(8, int(1000.0 / max(fps * spd, 1.0)))
            self._playback_timer.start(interval_ms)
            self.btn_play.setText("Pause")
        else:
            self._playback_timer.stop()
            self.btn_play.setText("Play")

    def _advance_frame(self) -> None:
        doc = self.ctrl.active_motion
        if doc is None or doc.num_frames == 0:
            self._toggle_play()
            return
        nxt = self._frame + 1
        if nxt >= doc.num_frames:
            if self.chk_loop.isChecked():
                nxt = 0
            else:
                self._toggle_play()
                return
        self.set_frame(nxt)

    def _apply_cleanup(self) -> None:
        if self.ctrl.active_motion is None:
            QMessageBox.information(self, "No Motion Loaded", "Load or generate a motion take first.")
            return
        cfg = MotionCleanupConfig(
            strength=self.spin_strength.value(),
            filter_type=self.combo_filter.currentText(),
            butterworth_cutoff_hz=self.spin_cutoff.value(),
            enable_foot_locking=self.chk_footlock.isChecked(),
            enable_ground_clamp=self.chk_ground.isChecked(),
            enable_anatomical_limits=self.chk_limits.isChecked(),
        )
        f0 = self.spin_trim_in.value()
        f1 = self.spin_trim_out.value()
        rng = (f0, f1) if f1 > f0 else None
        self.ctrl.apply_motion_cleanup(cfg, frame_range=rng)
