"""Pages 8-11: Live Capture Studio (with Real-Time 3D Character/Skeleton Preview & 2D Skeleton Overlays),
Recording Library, Processing Queue, and Motion Editor.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QKeyEvent, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
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
    """Page 8: Capture Studio with Real-Time 3D Character/Skeleton Viewport, 2D Skeleton Overlays, and WebSocket REC."""

    def __init__(self, ctrl: StudioController, navigate: Callable[[str], None]) -> None:
        super().__init__()
        self.ctrl = ctrl
        self.navigate = navigate
        self._rec_started_t: float = 0.0
        self._live_source_label: str = ""
        self._live_fps: float = 0.0

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)
        root.addLayout(
            _header(
                "Capture Studio — Live Mocap Preview",
                "Real-time 3D Character & Skeleton viewport + live smartphone camera feeds with 2D skeleton tracking overlay",
            )
        )

        # Top control ribbon
        ctrl_card = QFrame()
        ctrl_card.setObjectName("Card")
        ctrl_bar = QHBoxLayout(ctrl_card)
        ctrl_bar.setContentsMargins(12, 10, 12, 10)
        ctrl_bar.setSpacing(8)

        self.btn_rec = QPushButton("●  Start Mocap Recording")
        self.btn_rec.setObjectName("PrimaryButton")
        self.btn_rec.setMinimumHeight(36)
        self.btn_rec.clicked.connect(self._toggle_recording)

        self.btn_live = QPushButton("▶  Live Character Preview")
        self.btn_live.setMinimumHeight(36)
        self.btn_live.clicked.connect(self._toggle_live_preview)

        self.chk_overlay = QCheckBox("2D Skeleton Overlay")
        self.chk_overlay.setChecked(True)

        btn_pair = QPushButton("Pair Phones (QR)")
        btn_pair.clicked.connect(lambda: self.navigate("Camera Manager"))
        btn_import = QPushButton("Import Videos...")
        btn_import.clicked.connect(self._import_videos)
        btn_process = QPushButton("Process Takes →")
        btn_process.clicked.connect(lambda: self.navigate("Processing Queue"))

        self.lbl_status = QLabel("Ready · 3D Character Preview")
        self.lbl_status.setObjectName("BadgeWarn")

        for w in (self.btn_rec, self.btn_live, self.chk_overlay, btn_pair, btn_import, btn_process):
            ctrl_bar.addWidget(w)
        ctrl_bar.addStretch()
        ctrl_bar.addWidget(self.lbl_status)
        root.addWidget(ctrl_card)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left/Center: Large 3D Character / Skeleton Viewport
        vp_frame = QFrame()
        vp_frame.setObjectName("Card")
        vp_l = QVBoxLayout(vp_frame)
        vp_l.setContentsMargins(10, 10, 10, 10)
        self.viewport = Viewport3DContainer()
        self.viewport.canvas.set_display_mode("character")
        vp_l.addWidget(self.viewport, 1)
        splitter.addWidget(vp_frame)

        # Right side: Live Camera Previews with 2D Skeleton Overlay & Telemetry
        cam_panel = QGroupBox("Live Smartphone Feeds (2D Skeleton Overlay)")
        self.grid_layout = QGridLayout(cam_panel)
        self._preview_img_labels: dict[str, QLabel] = {}
        self._preview_meta_labels: dict[str, QLabel] = {}
        splitter.addWidget(cam_panel)
        splitter.setSizes([780, 400])
        root.addWidget(splitter, 1)

        self._preview_timer = QTimer(self)
        self._preview_timer.timeout.connect(self._refresh_previews_and_hud)
        self._preview_timer.start(180)

        ctrl.cameras_changed.connect(self.refresh)
        ctrl.calibration_changed.connect(self.refresh)
        ctrl.motion_changed.connect(self.refresh)
        ctrl.recording_state_changed.connect(self._on_recording_state_changed)
        ctrl.live_preview.pose_updated.connect(self._on_live_pose)
        self.refresh()

    def _on_live_pose(self, positions, confidence, source_label: str, fps: float) -> None:
        self._live_source_label = source_label
        self._live_fps = fps
        self.viewport.canvas.set_scene_frame(
            skeleton_positions=positions,
            skeleton_confidence=confidence,
        )
        self._update_viewport_live_badge()

    def _update_viewport_live_badge(self) -> None:
        if self.ctrl.is_recording:
            elapsed = max(0.0, time.perf_counter() - self._rec_started_t)
            src = f" · {self._live_source_label}" if self._live_source_label else ""
            fps_s = f" · {self._live_fps:.0f} FPS" if self._live_fps > 0.5 else ""
            self.viewport.canvas.set_live_status(f"● LIVE REC  {elapsed:05.1f}s{src}{fps_s}", "recording")
        elif self.ctrl.live_preview.running:
            is_demo = "Demo" in self._live_source_label
            src = self._live_source_label or "Waiting for phone stream..."
            fps_s = f" · {self._live_fps:.0f} FPS" if self._live_fps > 0.5 else ""
            self.viewport.canvas.set_live_status(
                f"● {src}{fps_s}",
                "demo" if is_demo else "live",
            )
        else:
            self.viewport.canvas.set_live_status("Standby · Click 'Start Mocap' or 'Live Character Preview'", "idle")

    def _toggle_live_preview(self) -> None:
        if self.ctrl.live_preview.running:
            self.ctrl.live_preview.stop()
            self._live_source_label = ""
            self._live_fps = 0.0
            self.btn_live.setText("▶  Live Character Preview")
            self._update_viewport_live_badge()
        else:
            self.ctrl.live_preview.start(simulate_when_no_frames=True)
            self.btn_live.setText("■  Stop Live Preview")
            self._update_viewport_live_badge()

    def _on_recording_state_changed(self, recording: bool, msg: str) -> None:
        if recording:
            self._rec_started_t = time.perf_counter()
            self.ctrl.live_preview.start(simulate_when_no_frames=True)
            self.btn_live.setText("■  Stop Live Preview")
        self.refresh()

    def refresh(self) -> None:
        self.viewport.canvas.set_calibration(self.ctrl.active_calibration)
        doc = self.ctrl.active_motion
        if doc is not None and doc.num_frames > 0 and not self.ctrl.live_preview.running:
            trk = doc.corrected_motion or doc.raw_motion
            self.viewport.canvas.set_scene_frame(
                skeleton_positions=trk.joint_positions[0],
                skeleton_confidence=trk.joint_confidence[0],
                markers_3d=doc.markers_3d[0] if doc.markers_3d is not None else None,
            )

        rec = self.ctrl.is_recording
        self.btn_rec.setText("■  STOP MOCAP RECORDING" if rec else "●  Start Mocap Recording")
        self.btn_rec.setObjectName("DangerButton" if rec else "PrimaryButton")
        self.btn_rec.style().unpolish(self.btn_rec)
        self.btn_rec.style().polish(self.btn_rec)

        self.btn_live.setText("■  Stop Live Preview" if self.ctrl.live_preview.running else "▶  Live Character Preview")

        # Rebuild camera tiles cleanly
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._preview_img_labels.clear()
        self._preview_meta_labels.clear()

        cams = self.ctrl.camera_manager.list_cameras()
        if not cams:
            empty = QLabel(
                "No smartphones connected yet.\n\n"
                "• Click 'Pair Phones (QR)' to connect your Android phone\n"
                "• Or click '▶ Live Character Preview' to see the live 3D Character & Skeleton!"
            )
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setObjectName("Subtitle")
            self.grid_layout.addWidget(empty, 0, 0)
        else:
            cols = 1 if len(cams) == 1 else (2 if len(cams) <= 4 else 3)
            for idx, cam in enumerate(cams):
                tile = QFrame()
                tile.setObjectName("Card")
                tl = QVBoxLayout(tile)
                tl.setContentsMargins(6, 6, 6, 6)
                tl.setSpacing(4)

                img_lbl = QLabel(f"{cam.camera_uid} · Waiting for stream...")
                img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                img_lbl.setMinimumSize(170, 120)
                img_lbl.setStyleSheet("background: #080C12; border-radius: 7px; color: #8EA0B5;")
                tl.addWidget(img_lbl, 1)

                sync_s = f"±{cam.clock_model.residual_rmse_ms:.1f}ms" if cam.clock_model else "unsynced"
                batt_s = f"{cam.telemetry.battery_percent:.0f}%" if cam.telemetry else "--"
                fps_s = (
                    f"{cam.telemetry.measured_fps:.1f}fps"
                    if (cam.telemetry and cam.telemetry.measured_fps > 0)
                    else f"{cam.config.target_fps:.0f}fps"
                )
                meta_lbl = QLabel(f"{cam.camera_uid} ({cam.name}) · {cam.state.value.upper()} · {fps_s} · {sync_s} · {batt_s}")
                meta_lbl.setStyleSheet("font-size: 11px; font-weight: 600;")
                tl.addWidget(meta_lbl)

                self.grid_layout.addWidget(tile, idx // cols, idx % cols)
                self._preview_img_labels[cam.device_id] = img_lbl
                self._preview_meta_labels[cam.device_id] = meta_lbl

        cal_s = f"Calib ({len(self.ctrl.active_calibration.cameras)}c)" if self.ctrl.active_calibration else "Uncalibrated"
        active_n = len(self.ctrl.camera_manager.get_active_recording_cameras())
        self.lbl_status.setText(f"{'● REC' if rec else 'Ready'} · {active_n} Online · {cal_s}")
        self.lbl_status.setObjectName("BadgeErr" if rec else ("BadgeOk" if active_n >= 1 else "BadgeWarn"))
        self.lbl_status.style().unpolish(self.lbl_status)
        self.lbl_status.style().polish(self.lbl_status)
        self._update_viewport_live_badge()

    def _refresh_previews_and_hud(self) -> None:
        # Auto-activate live skeleton tracking whenever a real phone starts streaming frames
        cams = self.ctrl.camera_manager.list_cameras()
        has_live_frames = any(c.latest_preview_jpeg for c in cams)
        if has_live_frames and not self.ctrl.live_preview.running:
            self.ctrl.live_preview.start(simulate_when_no_frames=False)
            self.btn_live.setText("■  Stop Live Preview")

        for did, lbl in self._preview_img_labels.items():
            cam = self.ctrl.camera_manager.get_camera(did)
            if not cam:
                continue
            raw_jpeg = (
                self.ctrl.live_overlay_jpegs.get(did)
                if self.chk_overlay.isChecked() and did in self.ctrl.live_overlay_jpegs
                else cam.latest_preview_jpeg
            )
            if raw_jpeg:
                img = QImage.fromData(raw_jpeg)
                if not img.isNull():
                    lbl.setPixmap(QPixmap.fromImage(img).scaled(lbl.size(), Qt.AspectRatioMode.KeepAspectRatio))

        if self.ctrl.is_recording or self.ctrl.live_preview.running:
            if self.ctrl.live_preview.running and "Stop" not in self.btn_live.text():
                self.btn_live.setText("■  Stop Live Preview")
            self._update_viewport_live_badge()

    def _toggle_recording(self) -> None:
        if not self.ctrl.is_recording:
            ok, _ = self.ctrl.camera_manager.can_record_any()
            if not ok:
                # Start live demo character preview immediately so user still gets live feedback + prompt to pair phone!
                self.ctrl.live_preview.start(simulate_when_no_frames=True)
                self._update_viewport_live_badge()
                QMessageBox.information(
                    self,
                    "Live Preview Active — No Phone Paired Yet",
                    "Started Live 3D Character Preview in demo mode.\n"
                    "To record real motion capture video, pair at least 1 Android phone in '2. Cameras & Pair'.",
                )
                return
            try:
                self._rec_started_t = time.perf_counter()
                self.ctrl.live_preview.start(simulate_when_no_frames=True)
                res = self.ctrl.start_synchronized_recording(allow_single_camera=True)
                if not res.get("dispatched_devices"):
                    self.ctrl.status_message.emit(
                        "Live 3D Character Preview running (simulated/offline cameras — connect live phone WebSocket to record MP4)."
                    )
            except Exception as exc:
                QMessageBox.warning(self, "Recording Error", str(exc))
        else:
            try:
                self.ctrl.stop_synchronized_recording()
            except Exception as exc:
                QMessageBox.warning(self, "Stop Recording Error", str(exc))
        self.refresh()

    def _import_videos(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Import Synchronized Multi-Camera Videos", "", "Video Files (*.mp4 *.mov *.avi *.mkv)"
        )
        if not files:
            return
        summaries = []
        self.ctrl.imported_videos.clear()
        for f in files:
            m = probe_video_metadata(f, scan_pts=False)
            rec = self.ctrl.camera_manager.register_imported_or_local_camera(
                device_id=f"import_{Path(f).stem}",
                name=Path(f).name,
                width=m.width,
                height=m.height,
                fps=m.nominal_fps,
                adapter_type="imported_video",
            )
            self.ctrl.imported_videos[rec.camera_uid] = Path(f)
            summaries.append(f"{rec.camera_uid} ({Path(f).name}): {m.width}x{m.height} @ {m.nominal_fps:.1f}fps ({m.frame_count}f)")
        self.ctrl.project_changed.emit()
        QMessageBox.information(
            self,
            "Imported Multi-Camera Videos",
            "Registered imported camera sources (ready in Processing Queue):\n" + "\n".join(summaries),
        )


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
    """Page 10: Background Processing Queue supporting Real Uploaded/Imported Multi-Cam Video Takes & Demo."""

    def __init__(self, ctrl: StudioController) -> None:
        super().__init__()
        self.ctrl = ctrl
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(10)
        root.addLayout(_header("Processing Queue", "Run non-blocking offline 3D reconstruction on recorded phone takes, imported videos, or demo data"))

        launch_box = QGroupBox("Launch Offline Reconstruction Job")
        l_row = QHBoxLayout(launch_box)
        self.combo_source = QComboBox()
        self.combo_mode = QComboBox(); self.combo_mode.addItems(["markerless_ai", "hybrid", "physical_marker"])
        self.combo_profile = QComboBox(); self.combo_profile.addItems(["standard", "cinema", "fast_preview"])
        btn_ref = QPushButton("↻")
        btn_ref.setFixedWidth(34)
        btn_ref.clicked.connect(self._refresh_sources)
        btn_run = QPushButton("Start Reconstruction Job")
        btn_run.setObjectName("PrimaryButton")
        btn_run.clicked.connect(self._queue_job)
        btn_cancel = QPushButton("Cancel Selected")
        btn_cancel.setObjectName("DangerButton")
        btn_cancel.clicked.connect(self._cancel_selected)

        l_row.addWidget(QLabel("Source:"))
        l_row.addWidget(self.combo_source, 2)
        l_row.addWidget(btn_ref)
        l_row.addWidget(QLabel("Mode:"))
        l_row.addWidget(self.combo_mode)
        l_row.addWidget(QLabel("Profile:"))
        l_row.addWidget(self.combo_profile)
        l_row.addWidget(btn_run)
        l_row.addWidget(btn_cancel)
        root.addWidget(launch_box)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Job ID", "Title", "Status", "Progress", "Message"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        root.addWidget(self.table, 1)

        ctrl.jobs_changed.connect(self.refresh)
        ctrl.project_changed.connect(self._refresh_sources)
        self._refresh_sources()
        self.refresh()

    def _refresh_sources(self) -> None:
        self.combo_source.clear()
        for take_id, vids in self.ctrl.list_recorded_take_folders():
            self.combo_source.addItem(f"Recorded Take: {take_id} ({len(vids)} videos)", ("recorded", take_id, vids))
        if self.ctrl.imported_videos:
            self.combo_source.addItem(
                f"Imported Videos ({len(self.ctrl.imported_videos)} cams)",
                ("imported", "imported_take", dict(self.ctrl.imported_videos)),
            )
        self.combo_source.addItem("Synthetic 4-Cam Demo Walk", ("demo", "synthetic_demo", {}))

    def _queue_job(self) -> None:
        mode = self.combo_mode.currentText()
        prof = self.combo_profile.currentText()
        src = self.combo_source.currentData() or ("demo", "synthetic_demo", {})
        kind, take_name, vids = src

        if kind in ("recorded", "imported"):
            def real_task(prog, cancel):
                doc = self.ctrl.reconstruct_from_video_files(
                    video_paths_by_cam=vids,
                    take_name=take_name,
                    capture_mode=mode,
                    profile_name=prof,
                    progress_cb=prog,
                    cancel_event=cancel,
                )
                return doc.take_id

            self.ctrl.job_queue.submit(f"Reconstruct '{take_name}' ({prof})", "reconstruct", real_task)
        else:
            def demo_task(prog, cancel):
                prog(0.2, f"Initializing {prof} ({mode}) demo pipeline...")
                if cancel.is_set():
                    return None
                doc = self.ctrl.generate_synthetic_demo_take(f"queued_{prof}_{mode}")
                prog(1.0, f"Completed {doc.num_frames} frames")
                return doc.take_id

            self.ctrl.job_queue.submit(f"Demo Reconstruct ({prof} / {mode})", "reconstruct", demo_task)

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
    """Page 11: Interactive Motion Editor with 3D Character/Skeleton Viewport, Timeline, Presets, and 1-Click Blender Publish."""

    def __init__(self, ctrl: StudioController) -> None:
        super().__init__()
        self.ctrl = ctrl
        self._frame = 0
        self._playing = False
        self._playback_timer = QTimer(self)
        self._playback_timer.timeout.connect(self._advance_frame)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(8)
        root.addLayout(
            _header(
                "Motion Editor",
                "Space: Play/Pause · ←/→: Step Frame · Switch 3D Character / Skeleton view, clean up motion, and publish to Blender",
            )
        )

        splitter = QSplitter(Qt.Orientation.Horizontal)

        center_w = QWidget()
        center_l = QVBoxLayout(center_w)
        center_l.setContentsMargins(0, 0, 0, 0)
        self.viewport = Viewport3DContainer()
        center_l.addWidget(self.viewport, 3)

        pb_row = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setObjectName("PrimaryButton")
        self.btn_play.clicked.connect(self._toggle_play)
        btn_prev = QPushButton("◀ Frame")
        btn_prev.clicked.connect(lambda: self.set_frame(self._frame - 1))
        btn_next = QPushButton("Frame ▶")
        btn_next.clicked.connect(lambda: self.set_frame(self._frame + 1))
        self.chk_loop = QCheckBox("Loop"); self.chk_loop.setChecked(True)
        self.combo_speed = QComboBox(); self.combo_speed.addItems(["0.25x", "0.5x", "1.0x", "1.5x", "2.0x"]); self.combo_speed.setCurrentText("1.0x")
        self.lbl_frame = QLabel("Frame 0 / 0")
        btn_blender = QPushButton("Send to Blender →")
        btn_blender.setObjectName("PrimaryButton")
        btn_blender.clicked.connect(self._publish_blender)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.set_frame)
        for w in (self.btn_play, btn_prev, btn_next, self.chk_loop, self.combo_speed, self.lbl_frame):
            pb_row.addWidget(w)
        pb_row.addStretch()
        pb_row.addWidget(btn_blender)
        center_l.addLayout(pb_row)
        center_l.addWidget(self.slider)

        self.curve_widget = CurveInspectorWidget()
        center_l.addWidget(self.curve_widget, 1)
        splitter.addWidget(center_w)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_w = QWidget()
        right_l = QVBoxLayout(right_w)
        right_l.setContentsMargins(6, 0, 0, 0)

        preset_box = QGroupBox("Quick Cleanup Presets")
        pr_row = QHBoxLayout(preset_box)
        for lbl, strength, flt, fl in [
            ("Light", 0.35, "one_euro", False),
            ("Standard", 0.65, "butterworth", True),
            ("Cinema", 0.85, "butterworth", True),
        ]:
            bp = QPushButton(lbl)
            bp.clicked.connect(
                lambda checked=False, s=strength, f=flt, lock=fl: self._apply_preset(s, f, lock)
            )
            pr_row.addWidget(bp)
        right_l.addWidget(preset_box)

        insp_box = QGroupBox("Joint & Comparison")
        insp_f = QFormLayout(insp_box)
        self.combo_bone = QComboBox(); self.combo_bone.addItems(BONE_NAMES)
        self.combo_bone.currentIndexChanged.connect(self._update_curve_and_viewport)
        self.chk_use_corrected = QCheckBox("Show Corrected Motion"); self.chk_use_corrected.setChecked(True)
        self.chk_use_corrected.toggled.connect(self._update_curve_and_viewport)
        insp_f.addRow("Inspect Bone:", self.combo_bone)
        insp_f.addRow("", self.chk_use_corrected)
        right_l.addWidget(insp_box)

        clean_box = QGroupBox("Fine-Tune Non-Destructive Cleanup")
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
        btn_undo = QPushButton("↶ Undo"); btn_undo.clicked.connect(self.ctrl.undo_motion_edit)
        btn_redo = QPushButton("↷ Redo"); btn_redo.clicked.connect(self.ctrl.redo_motion_edit)
        ur_row.addWidget(btn_undo); ur_row.addWidget(btn_redo)
        cf.addRow(btn_apply_clean)
        cf.addRow(ur_row)
        right_l.addWidget(clean_box)
        right_l.addStretch()

        right_scroll.setWidget(right_w)
        splitter.addWidget(right_scroll)
        splitter.setSizes([830, 320])
        root.addWidget(splitter, 1)

        ctrl.motion_changed.connect(self.refresh)
        ctrl.calibration_changed.connect(self.refresh)
        self.refresh()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space:
            self._toggle_play()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Left:
            self.set_frame(self._frame - 1)
            event.accept()
            return
        if event.key() == Qt.Key.Key_Right:
            self.set_frame(self._frame + 1)
            event.accept()
            return
        super().keyPressEvent(event)

    def _apply_preset(self, strength: float, flt: str, foot_lock: bool) -> None:
        self.spin_strength.setValue(strength)
        self.combo_filter.setCurrentText(flt)
        self.chk_footlock.setChecked(foot_lock)
        self._apply_cleanup()

    def _publish_blender(self) -> None:
        try:
            self.ctrl.publish_active_motion_to_blender()
            QMessageBox.information(
                self,
                "Ready in Blender",
                f"Active take published to http://127.0.0.1:{self.ctrl.settings.network.server_port}.\n"
                "Open Blender → N-panel → Nuranova Mocap → Check Studio Connection → Pull Take.",
            )
        except Exception as exc:
            QMessageBox.information(self, "Publish to Blender", str(exc))

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
            self.btn_play.setText("❚❚ Pause")
        else:
            self._playback_timer.stop()
            self.btn_play.setText("▶ Play")

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
