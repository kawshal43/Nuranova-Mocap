"""Pages 4-7: Camera Manager, Camera Configuration, Calibration Wizard, and Calibration Profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from desktop.calibration.calibrator import (
    calibrate_camera_intrinsics,
    initialize_extrinsics_from_board_frames,
    run_board_bundle_adjustment,
)
from desktop.calibration.charuco_board import (
    ARUCO_DICTIONARIES,
    ChArUcoBoardConfig,
    detect_charuco_corners,
    generate_printable_charuco_pdf,
    render_charuco_png,
)
from desktop.ui.app_controller import StudioController
from desktop.ui.pages.pages_workspace import _header
from engine.math_core.camera_model import CalibratedCamera, MultiCameraCalibration
from shared_protocol.protocol import CameraRole, RecordingConfig


class CameraManagerPage(QWidget):
    """Page 4: Dynamic 2-to-12 Camera Discovery, Pairing, Health, and Sync Monitor."""

    def __init__(self, ctrl: StudioController) -> None:
        super().__init__()
        self.ctrl = ctrl
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(10)
        root.addLayout(_header("Camera Manager", "Dynamic 2–12 body cameras + dedicated face/hand cameras with PIN pairing and health monitoring"))

        top = QHBoxLayout()
        self.btn_server = QPushButton("Start Local Server")
        self.btn_server.setObjectName("PrimaryButton")
        self.btn_server.clicked.connect(self._toggle_server)
        self.lbl_pin = QLabel(f"Pairing PIN: {ctrl.camera_manager.get_active_pin()}")
        self.lbl_pin.setStyleSheet("font-weight: 700; font-size: 14px; padding: 0 10px;")
        btn_new_pin = QPushButton("New PIN")
        btn_new_pin.clicked.connect(self._regen_pin)
        btn_add_sim = QPushButton("+ Add Simulated Test Camera")
        btn_add_sim.clicked.connect(self._add_simulated_cam)
        btn_remove = QPushButton("Remove Selected")
        btn_remove.setObjectName("DangerButton")
        btn_remove.clicked.connect(self._remove_selected)
        for w in (self.btn_server, self.lbl_pin, btn_new_pin, btn_add_sim, btn_remove):
            top.addWidget(w)
        top.addStretch()
        root.addLayout(top)

        self.table = QTableWidget(0, 11)
        self.table.setHorizontalHeaderLabels([
            "UID", "Name", "Group", "Role", "Enabled", "State", "IP", "Res / FPS", "Sync Residual", "Batt / Temp", "Calib"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        root.addWidget(self.table, 1)

        self.lbl_readiness = QLabel("")
        root.addWidget(self.lbl_readiness)

        ctrl.cameras_changed.connect(self.refresh)
        ctrl.server_state_changed.connect(self._on_server_changed)
        self.refresh()

    def _toggle_server(self) -> None:
        if self.ctrl.server_running:
            self.ctrl.stop_server()
        else:
            self.ctrl.start_server()

    def _on_server_changed(self, running: bool, msg: str) -> None:
        self.btn_server.setText("Stop Local Server" if running else "Start Local Server")

    def _regen_pin(self) -> None:
        pin = self.ctrl.camera_manager.generate_pairing_pin()
        self.lbl_pin.setText(f"Pairing PIN: {pin}")

    def _add_simulated_cam(self) -> None:
        idx = len(self.ctrl.camera_manager.list_cameras()) + 1
        self.ctrl.camera_manager.register_imported_or_local_camera(
            device_id=f"sim_cam_{idx:02d}",
            name=f"Simulated Cam {idx:02d}",
            width=1920,
            height=1080,
            fps=60.0,
            role=CameraRole.BODY,
            adapter_type="simulated",
        )

    def _remove_selected(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        cams = self.ctrl.camera_manager.list_cameras()
        if row < len(cams):
            self.ctrl.camera_manager.remove_camera(cams[row].device_id)

    def refresh(self) -> None:
        cams = self.ctrl.camera_manager.list_cameras()
        self.table.setRowCount(len(cams))
        for r, c in enumerate(cams):
            sync_txt = f"±{c.clock_model.residual_rmse_ms:.2f} ms" if c.clock_model else "Unsynced"
            batt_txt = f"{c.telemetry.battery_percent:.0f}% / {c.telemetry.temperature_celsius or '--'}°C" if c.telemetry else "--"
            cal_txt = "Valid" if c.calibration_valid else ("Invalid" if c.calibrated else "None")
            vals = [
                c.camera_uid,
                c.name,
                c.group,
                c.role.value,
                "Yes" if c.enabled else "Disabled",
                c.state.value,
                c.ip_address or c.adapter_type,
                f"{c.config.width}x{c.config.height} @ {c.config.target_fps:.0f}fps",
                sync_txt,
                batt_txt,
                cal_txt,
            ]
            for col, v in enumerate(vals):
                self.table.setItem(r, col, QTableWidgetItem(v))
        ok, msg = self.ctrl.camera_manager.can_record_multi_view()
        self.lbl_readiness.setText(f"Multi-View Status: {msg}")


class CameraConfigPage(QWidget):
    """Page 5: Camera Recording Settings with Dynamic Capability Validation."""

    def __init__(self, ctrl: StudioController) -> None:
        super().__init__()
        self.ctrl = ctrl
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(10)
        root.addLayout(_header("Camera Configuration", "Configure resolution, FPS, shutter speed, focus/WB lock, and verify device Camera2 support"))

        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Select Camera:"))
        self.cam_combo = QComboBox()
        self.cam_combo.currentIndexChanged.connect(self._load_selected_cam)
        sel_row.addWidget(self.cam_combo, 1)
        root.addLayout(sel_row)

        form_box = QGroupBox("Recording Parameters (Preferred Baseline: 1080p @ 60 FPS, 1/250s, Locked Focus/WB, Stab OFF)")
        form = QFormLayout(form_box)
        self.res_combo = QComboBox()
        self.res_combo.addItems(["1280x720 (720p)", "1920x1080 (1080p)", "2560x1440 (1440p)", "3840x2160 (4K)"])
        self.res_combo.setCurrentIndex(1)
        self.fps_combo = QComboBox()
        self.fps_combo.addItems(["24", "30", "50", "60", "90", "120"])
        self.fps_combo.setCurrentText("60")
        self.bitrate_spin = QSpinBox()
        self.bitrate_spin.setRange(2, 120)
        self.bitrate_spin.setValue(20)
        self.bitrate_spin.setSuffix(" Mbps")
        self.shutter_spin = QSpinBox()
        self.shutter_spin.setRange(30, 2000)
        self.shutter_spin.setValue(250)
        self.shutter_spin.setPrefix("1/")
        self.shutter_spin.setSuffix(" s")
        self.chk_manual_exp = QCheckBox("Manual Exposure")
        self.chk_manual_exp.setChecked(True)
        self.chk_lock_focus = QCheckBox("Lock Focus")
        self.chk_lock_focus.setChecked(True)
        self.chk_lock_wb = QCheckBox("Lock White Balance")
        self.chk_lock_wb.setChecked(True)
        self.chk_no_stab = QCheckBox("Disable Digital Stabilization")
        self.chk_no_stab.setChecked(True)

        form.addRow("Resolution:", self.res_combo)
        form.addRow("Frame Rate (FPS):", self.fps_combo)
        form.addRow("H.264 Bitrate:", self.bitrate_spin)
        form.addRow("Shutter Speed:", self.shutter_spin)
        form.addRow("", self.chk_manual_exp)
        form.addRow("", self.chk_lock_focus)
        form.addRow("", self.chk_lock_wb)
        form.addRow("", self.chk_no_stab)
        root.addWidget(form_box)

        btn_apply = QPushButton("Validate & Apply to Camera")
        btn_apply.setObjectName("PrimaryButton")
        btn_apply.clicked.connect(self._apply_config)
        root.addWidget(btn_apply)

        self.diag_text = QTextEdit()
        self.diag_text.setReadOnly(True)
        self.diag_text.setMaximumHeight(130)
        root.addWidget(self.diag_text)
        root.addStretch()

        ctrl.cameras_changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        self.cam_combo.blockSignals(True)
        self.cam_combo.clear()
        for c in self.ctrl.camera_manager.list_cameras():
            self.cam_combo.addItem(f"{c.camera_uid} — {c.name}", c.device_id)
        self.cam_combo.blockSignals(False)
        self._load_selected_cam()

    def _load_selected_cam(self) -> None:
        did = self.cam_combo.currentData()
        if not did:
            return
        cam = self.ctrl.camera_manager.get_camera(did)
        if not cam:
            return
        self.fps_combo.setCurrentText(str(int(cam.config.target_fps)))
        self.bitrate_spin.setValue(max(2, cam.config.bitrate_bps // 1_000_000))
        self.chk_lock_focus.setChecked(cam.config.lock_focus)
        self.chk_lock_wb.setChecked(cam.config.lock_white_balance)
        self.chk_no_stab.setChecked(cam.config.disable_stabilization)

    def _apply_config(self) -> None:
        did = self.cam_combo.currentData()
        if not did:
            QMessageBox.information(self, "No Camera", "Register or connect a camera first in Camera Manager.")
            return
        res_map = [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)]
        w, h = res_map[self.res_combo.currentIndex()]
        cfg = RecordingConfig(
            width=w,
            height=h,
            target_fps=float(self.fps_combo.currentText()),
            bitrate_bps=self.bitrate_spin.value() * 1_000_000,
            manual_exposure=self.chk_manual_exp.isChecked(),
            shutter_speed_ns=int(1e9 / max(self.shutter_spin.value(), 1)),
            lock_focus=self.chk_lock_focus.isChecked(),
            lock_white_balance=self.chk_lock_wb.isChecked(),
            disable_stabilization=self.chk_no_stab.isChecked(),
        )
        ok, issues, alt = self.ctrl.camera_manager.update_camera_config(did, cfg)
        lines = [f"Validation Result: {'ACCEPTED' if ok else 'UNSUPPORTED ON THIS DEVICE'}"] + issues
        if alt is not None:
            lines.append(f"Suggested Supported Alternative: {alt.width}x{alt.height} @ {alt.target_fps:.0f} FPS")
        self.diag_text.setPlainText("\n".join(lines))


class CalibrationWizardPage(QWidget):
    """Page 6: Guided 13-Step ChArUco Multi-Camera Calibration Wizard."""

    def __init__(self, ctrl: StudioController) -> None:
        super().__init__()
        self.ctrl = ctrl
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(10)
        root.addLayout(_header("Calibration Wizard", "Printable ChArUco board generation, physical dimension verification, intrinsic/extrinsic bundle adjustment, and Z-Up world alignment"))

        board_box = QGroupBox("Step 1 — Printable ChArUco Board & Measured Physical Scale")
        bf = QFormLayout(board_box)
        self.spin_sx = QSpinBox(); self.spin_sx.setRange(4, 20); self.spin_sx.setValue(7)
        self.spin_sy = QSpinBox(); self.spin_sy.setRange(4, 20); self.spin_sy.setValue(5)
        self.spin_sq = QDoubleSpinBox(); self.spin_sq.setRange(10.0, 200.0); self.spin_sq.setValue(35.0); self.spin_sq.setSuffix(" mm")
        self.spin_mk = QDoubleSpinBox(); self.spin_mk.setRange(5.0, 180.0); self.spin_mk.setValue(26.0); self.spin_mk.setSuffix(" mm")
        self.spin_measured = QDoubleSpinBox(); self.spin_measured.setRange(10.0, 200.0); self.spin_measured.setValue(35.0); self.spin_measured.setSuffix(" mm (measured with ruler!)")
        self.combo_dict = QComboBox(); self.combo_dict.addItems(list(ARUCO_DICTIONARIES.keys())); self.combo_dict.setCurrentText("DICT_5X5_250")
        self.combo_paper = QComboBox(); self.combo_paper.addItems(["A4", "A3"])
        bf.addRow("Squares X / Y:", self._pair(self.spin_sx, self.spin_sy))
        bf.addRow("Nominal Square / Marker:", self._pair(self.spin_sq, self.spin_mk))
        bf.addRow("Actual Measured Square Size:", self.spin_measured)
        bf.addRow("Dictionary / Paper:", self._pair(self.combo_dict, self.combo_paper))

        btn_row = QHBoxLayout()
        btn_pdf = QPushButton("Generate Printable PDF Board")
        btn_pdf.setObjectName("PrimaryButton")
        btn_pdf.clicked.connect(self._gen_pdf)
        btn_png = QPushButton("Export High-DPI PNG")
        btn_png.clicked.connect(self._gen_png)
        btn_calib_folder = QPushButton("Run Multi-Camera Calibration from Folder...")
        btn_calib_folder.clicked.connect(self._run_folder_calibration)
        for b in (btn_pdf, btn_png, btn_calib_folder):
            btn_row.addWidget(b)
        bf.addRow("", btn_row)
        root.addWidget(board_box)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlainText(
            "Guided 13-Step Calibration Checklist:\n"
            "1. Connect 2–12 cameras -> 2. Configure locked focus/WB/no stabilization -> 3. Validate clock sync ->\n"
            "4. Print ChArUco PDF at 100% scale & enter ruler-measured square size above -> 5. Record multi-angle board sweep ->\n"
            "6. Intrinsic solve -> 7. Extrinsic pose graph -> 8. Metric board Bundle Adjustment ->\n"
            "9–10. Floor plane & Z-Up origin -> 11. Known-distance wand validation -> 12. Actor T/A-pose -> 13. Save profile."
        )
        root.addWidget(self.log, 1)

    @staticmethod
    def _pair(a: QWidget, b: QWidget) -> QWidget:
        w = QWidget()
        l = QHBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        l.addWidget(a)
        l.addWidget(b)
        return w

    def _board_cfg(self) -> ChArUcoBoardConfig:
        return ChArUcoBoardConfig(
            squares_x=self.spin_sx.value(),
            squares_y=self.spin_sy.value(),
            square_length_mm=self.spin_sq.value(),
            marker_length_mm=self.spin_mk.value(),
            dictionary_name=self.combo_dict.currentText(),
            measured_square_length_mm=self.spin_measured.value(),
        )

    def _gen_pdf(self) -> None:
        out, _ = QFileDialog.getSaveFileName(self, "Save Printable ChArUco PDF", "Nuranova_ChArUco_Board.pdf", "PDF (*.pdf)")
        if not out:
            return
        try:
            p = generate_printable_charuco_pdf(self._board_cfg(), out, paper_size=self.combo_paper.currentText())
            self.log.append(f"[OK] Saved physical-scale ChArUco PDF: {p}")
        except Exception as exc:
            QMessageBox.warning(self, "Board Error", str(exc))

    def _gen_png(self) -> None:
        out, _ = QFileDialog.getSaveFileName(self, "Save ChArUco PNG", "Nuranova_ChArUco_Board.png", "PNG (*.png)")
        if not out:
            return
        p = render_charuco_png(self._board_cfg(), out)
        self.log.append(f"[OK] Exported ChArUco PNG: {p}")

    def _run_folder_calibration(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Calibration Images Folder (subfolders cam_01, cam_02, ...)")
        if not folder:
            return
        cfg = self._board_cfg()
        cam_dirs = sorted([d for d in Path(folder).iterdir() if d.is_dir()])
        if len(cam_dirs) < 2:
            QMessageBox.warning(self, "Calibration Folder", "Expected >= 2 camera subfolders (e.g. cam_01/, cam_02/) containing synchronized ChArUco frames.")
            return
        try:
            intr_map = {}
            per_cam_dets: Dict[str, list] = {}
            for cd in cam_dirs:
                imgs = sorted(list(cd.glob("*.png")) + list(cd.glob("*.jpg")))
                dets = []
                img_size = (1920, 1080)
                for im_p in imgs:
                    im = cv2.imread(str(im_p))
                    if im is None:
                        continue
                    img_size = (im.shape[1], im.shape[0])
                    dets.append(detect_charuco_corners(im, cfg))
                per_cam_dets[cd.name] = dets
                intr_map[cd.name] = calibrate_camera_intrinsics(cd.name, dets, cfg, img_size)
                self.log.append(f"Intrinsic {cd.name}: RMSE = {intr_map[cd.name].reprojection_error_px:.3f} px")

            n_frames = min(len(v) for v in per_cam_dets.values())
            synced = [{uid: per_cam_dets[uid][f] for uid in per_cam_dets} for f in range(n_frames)]
            extr_init, board_poses = initialize_extrinsics_from_board_frames(intr_map, synced, cfg)
            extr_opt, _, rep = run_board_bundle_adjustment(intr_map, extr_init, synced, board_poses, cfg)
            calib = MultiCameraCalibration(
                profile_name=f"Calib_{Path(folder).name}",
                cameras={u: CalibratedCamera(intr_map[u], extr_opt[u]) for u in intr_map},
                global_reprojection_rmse_px=rep.global_reprojection_rmse_px,
            )
            self.ctrl.active_calibration = calib
            if self.ctrl.project_manager.active_project:
                out_p = self.ctrl.project_manager.active_project.subdir("calibration") / f"{calib.profile_name}.json"
                calib.save(out_p)
                self.log.append(f"[SAVED] Calibration profile saved to {out_p}")
            self.ctrl.calibration_changed.emit()
            self.log.append(f"[DONE] Global Bundle Adjustment RMSE: {rep.global_reprojection_rmse_px:.3f} px across {rep.num_cameras} cameras.")
        except Exception as exc:
            QMessageBox.warning(self, "Calibration Failed", str(exc))
            self.log.append(f"[ERROR] {exc}")


class CalibrationProfilesPage(QWidget):
    """Page 7: Saved Calibration Profiles Browser & Validity Inspector."""

    def __init__(self, ctrl: StudioController) -> None:
        super().__init__()
        self.ctrl = ctrl
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(10)
        root.addLayout(_header("Calibration Profiles", "Inspect, load, save, and verify reusable multi-camera calibration profiles"))

        bar = QHBoxLayout()
        btn_load = QPushButton("Load Calibration JSON...")
        btn_load.setObjectName("PrimaryButton")
        btn_load.clicked.connect(self._load_file)
        btn_save = QPushButton("Save Active Calibration...")
        btn_save.clicked.connect(self._save_file)
        bar.addWidget(btn_load)
        bar.addWidget(btn_save)
        bar.addStretch()
        root.addLayout(bar)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Camera UID", "Resolution", "fx / fy", "Intrinsic RMSE (px)", "Extrinsic RMSE (px)", "Camera Center (m)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        root.addWidget(self.table, 1)

        self.lbl_summary = QLabel("No active calibration loaded.")
        root.addWidget(self.lbl_summary)

        ctrl.calibration_changed.connect(self.refresh)
        self.refresh()

    def _load_file(self) -> None:
        fp, _ = QFileDialog.getOpenFileName(self, "Load Calibration Profile", "", "Calibration JSON (*.json)")
        if fp:
            self.ctrl.active_calibration = MultiCameraCalibration.load(fp)
            self.ctrl.calibration_changed.emit()

    def _save_file(self) -> None:
        if not self.ctrl.active_calibration:
            return
        fp, _ = QFileDialog.getSaveFileName(self, "Save Calibration Profile", "calibration_profile.json", "Calibration JSON (*.json)")
        if fp:
            self.ctrl.active_calibration.save(fp)

    def refresh(self) -> None:
        cal = self.ctrl.active_calibration
        if cal is None:
            self.table.setRowCount(0)
            self.lbl_summary.setText("No calibration profile loaded.")
            return
        self.table.setRowCount(len(cal.cameras))
        for r, (uid, cam) in enumerate(cal.cameras.items()):
            cpos = cam.extrinsics.camera_center_world
            vals = [
                uid,
                f"{cam.intrinsics.image_width}x{cam.intrinsics.image_height}",
                f"{cam.intrinsics.fx:.1f} / {cam.intrinsics.fy:.1f}",
                f"{cam.intrinsics.reprojection_error_px:.3f}",
                f"{cam.extrinsics.reprojection_error_px:.3f}",
                f"({cpos[0]:.2f}, {cpos[1]:.2f}, {cpos[2]:.2f})",
            ]
            for col, v in enumerate(vals):
                self.table.setItem(r, col, QTableWidgetItem(v))
        wand = f" | Known-Distance Err: {cal.known_distance_error_mm:.2f} mm" if cal.known_distance_error_mm is not None else ""
        self.lbl_summary.setText(f"Profile: {cal.profile_name} | Global RMSE: {cal.global_reprojection_rmse_px:.3f} px | Up Axis: {cal.world_up_axis}{wand}")
