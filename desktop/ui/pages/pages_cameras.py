"""Pages 4-7: All-in-One Cameras & QR Pairing Hub, Camera Configuration, Guided Calibration Wizard, and Calibration Profiles."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from desktop.calibration.calibrator import (
    align_calibration_to_world_z_up,
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
    """Page 4: Unified Phone Pairing (QR Code + LAN IP + Firewall Helper) + Live Cameras & Inline Quick Config."""

    def __init__(self, ctrl: StudioController) -> None:
        super().__init__()
        self.ctrl = ctrl
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(10)
        root.addLayout(
            _header(
                "Cameras & Phone Pairing",
                "Scan the QR code from Nuranova Mocap Camera on Android (same Wi-Fi/LAN) or pair with the 6-digit PIN",
            )
        )

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ---------------- LEFT PANEL: Instant Phone Pairing Card ----------------
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(330)
        left_scroll.setMaximumWidth(400)
        left_container = QWidget()
        left_l = QVBoxLayout(left_container)
        left_l.setContentsMargins(0, 0, 6, 0)
        left_l.setSpacing(10)

        pair_box = QGroupBox("Connect Phone (Instant QR / PIN)")
        pl = QVBoxLayout(pair_box)
        pl.setSpacing(8)

        srv_row = QHBoxLayout()
        self.lbl_srv_badge = QLabel("SERVER OFFLINE")
        self.lbl_srv_badge.setObjectName("BadgeWarn")
        self.btn_server = QPushButton("Start Server")
        self.btn_server.setObjectName("PrimaryButton")
        self.btn_server.clicked.connect(self._toggle_server)
        srv_row.addWidget(self.lbl_srv_badge)
        srv_row.addStretch()
        srv_row.addWidget(self.btn_server)
        pl.addLayout(srv_row)

        # Real LAN Adapter Selector
        ip_row = QHBoxLayout()
        self.combo_lan = QComboBox()
        self.combo_lan.currentIndexChanged.connect(self._on_lan_changed)
        btn_ref_ip = QPushButton("↻")
        btn_ref_ip.setFixedWidth(34)
        btn_ref_ip.setToolTip("Refresh network adapters")
        btn_ref_ip.clicked.connect(self._populate_lan_adapters)
        ip_row.addWidget(self.combo_lan, 1)
        ip_row.addWidget(btn_ref_ip)
        pl.addWidget(QLabel("PC Network Address (same Wi-Fi/LAN as phone):"))
        pl.addLayout(ip_row)

        # QR Code Frame
        qr_frame = QFrame()
        qr_frame.setObjectName("QrCard")
        qr_l = QVBoxLayout(qr_frame)
        qr_l.setContentsMargins(10, 10, 10, 10)
        self.lbl_qr = QLabel("QR Code")
        self.lbl_qr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_qr.setMinimumSize(196, 196)
        qr_l.addWidget(self.lbl_qr)
        pl.addWidget(qr_frame, 0, Qt.AlignmentFlag.AlignHCenter)

        self.lbl_endpoint = QLabel("")
        self.lbl_endpoint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_endpoint.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_endpoint.setStyleSheet("font-weight: 700; font-size: 13px;")
        pl.addWidget(self.lbl_endpoint)

        # Giant PIN Display
        self.lbl_pin = QLabel(ctrl.camera_manager.get_active_pin())
        self.lbl_pin.setObjectName("PinDisplay")
        self.lbl_pin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_pin.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        pl.addWidget(self.lbl_pin)

        pin_btns = QHBoxLayout()
        btn_new_pin = QPushButton("New PIN")
        btn_new_pin.clicked.connect(self._regen_pin)
        btn_copy_url = QPushButton("Copy Test URL")
        btn_copy_url.clicked.connect(self._copy_health_url)
        pin_btns.addWidget(btn_new_pin)
        pin_btns.addWidget(btn_copy_url)
        pl.addLayout(pin_btns)
        left_l.addWidget(pair_box)

        # Firewall & Discovery Health Box
        fw_box = QGroupBox("Connection Troubleshooter")
        fw_l = QVBoxLayout(fw_box)
        fw_l.setSpacing(6)
        self.lbl_discovery = QLabel("mDNS: -- | UDP Beacon (:8766): --")
        self.lbl_discovery.setWordWrap(True)
        self.lbl_fw = QLabel("Firewall: Checking...")
        self.lbl_fw.setWordWrap(True)
        fw_btns = QHBoxLayout()
        btn_fw_check = QPushButton("Check Firewall")
        btn_fw_check.clicked.connect(self._run_firewall_check)
        btn_fw_fix = QPushButton("Fix Windows Firewall (Admin)")
        btn_fw_fix.setObjectName("PrimaryButton")
        btn_fw_fix.clicked.connect(self._fix_firewall)
        fw_btns.addWidget(btn_fw_check)
        fw_btns.addWidget(btn_fw_fix)
        fw_l.addWidget(self.lbl_discovery)
        fw_l.addWidget(self.lbl_fw)
        fw_l.addLayout(fw_btns)
        left_l.addWidget(fw_box)
        left_l.addStretch()

        left_scroll.setWidget(left_container)
        splitter.addWidget(left_scroll)

        # ---------------- RIGHT PANEL: Paired/Connected Cameras + Inline Quick Config ----------------
        right_w = QWidget()
        right_l = QVBoxLayout(right_w)
        right_l.setContentsMargins(6, 0, 0, 0)
        right_l.setSpacing(10)

        top_bar = QHBoxLayout()
        self.lbl_readiness = QLabel("")
        self.lbl_readiness.setObjectName("BadgeWarn")
        btn_add_sim = QPushButton("+ Add Simulated Cam")
        btn_add_sim.clicked.connect(self._add_simulated_cam)
        btn_remove = QPushButton("Forget Selected")
        btn_remove.setObjectName("DangerButton")
        btn_remove.clicked.connect(self._remove_selected)
        top_bar.addWidget(self.lbl_readiness)
        top_bar.addStretch()
        top_bar.addWidget(btn_add_sim)
        top_bar.addWidget(btn_remove)
        right_l.addLayout(top_bar)

        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(
            ["UID", "Name", "Role", "Status", "IP / Source", "Mode", "Sync", "Batt / Temp", "Enabled"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.itemSelectionChanged.connect(self._on_camera_selected)
        right_l.addWidget(self.table, 2)

        # Inline Selected Camera Preview + Quick Config Panel
        quick_box = QGroupBox("Selected Camera — Live Preview & Quick Settings")
        ql = QHBoxLayout(quick_box)
        self.lbl_sel_preview = QLabel("Select a connected camera\nto view live preview")
        self.lbl_sel_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_sel_preview.setFixedSize(240, 145)
        self.lbl_sel_preview.setStyleSheet("border: 1px solid #2D3D50; border-radius: 6px; background: #0B1016;")
        ql.addWidget(self.lbl_sel_preview)

        qform = QFormLayout()
        self.q_res = QComboBox()
        self.q_res.addItems(["1280x720 (720p)", "1920x1080 (1080p)", "2560x1440 (1440p)", "3840x2160 (4K)"])
        self.q_res.setCurrentIndex(1)
        self.q_fps = QComboBox()
        self.q_fps.addItems(["24", "30", "60", "120"])
        self.q_fps.setCurrentText("30")
        self.q_enabled = QCheckBox("Enabled for Capture")
        self.q_enabled.setChecked(True)
        btn_q_apply = QPushButton("Apply to Selected Camera")
        btn_q_apply.setObjectName("PrimaryButton")
        btn_q_apply.clicked.connect(self._apply_quick_config)
        self.lbl_q_diag = QLabel("")
        self.lbl_q_diag.setWordWrap(True)
        self.lbl_q_diag.setObjectName("Subtitle")
        qform.addRow("Resolution:", self.q_res)
        qform.addRow("Target FPS:", self.q_fps)
        qform.addRow("", self.q_enabled)
        qform.addRow(btn_q_apply)
        qform.addRow(self.lbl_q_diag)
        ql.addLayout(qform, 1)
        right_l.addWidget(quick_box)

        splitter.addWidget(right_w)
        splitter.setSizes([370, 760])
        root.addWidget(splitter, 1)

        self._preview_timer = QTimer(self)
        self._preview_timer.timeout.connect(self._refresh_selected_preview)
        self._preview_timer.start(300)

        ctrl.cameras_changed.connect(self.refresh)
        ctrl.server_state_changed.connect(self._on_server_changed)

        self._populate_lan_adapters()
        self.refresh()
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            QTimer.singleShot(500, self._run_firewall_check)

    def _populate_lan_adapters(self) -> None:
        self.combo_lan.blockSignals(True)
        self.combo_lan.clear()
        adapters = self.ctrl.get_available_lan_ips()
        active_ip = self.ctrl.get_active_lan_ip()
        selected_idx = 0
        if adapters:
            for idx, (label, ip) in enumerate(adapters):
                self.combo_lan.addItem(f"{ip}  ({label})", ip)
                if ip == active_ip:
                    selected_idx = idx
        else:
            self.combo_lan.addItem("127.0.0.1 (No LAN detected)", "127.0.0.1")
        self.combo_lan.setCurrentIndex(selected_idx)
        self.combo_lan.blockSignals(False)
        self._refresh_qr_and_endpoint()

    def _on_lan_changed(self) -> None:
        ip = self.combo_lan.currentData()
        if ip and ip != "127.0.0.1":
            self.ctrl.set_preferred_lan_ip(str(ip))
        self._refresh_qr_and_endpoint()

    def _refresh_qr_and_endpoint(self) -> None:
        lan_ip = self.ctrl.get_active_lan_ip()
        port = self.ctrl.settings.network.server_port
        pin = self.ctrl.camera_manager.get_active_pin()
        self.lbl_pin.setText(pin)
        self.lbl_endpoint.setText(f"IP: {lan_ip}   ·   Port: {port}")
        pm = self.ctrl.generate_pairing_qr_pixmap(size_px=192)
        if pm is not None:
            self.lbl_qr.setPixmap(pm)
        else:
            self.lbl_qr.setText(f"Manual Pair:\n{lan_ip}:{port}\nPIN: {pin}")

        mdns_s = "Active" if self.ctrl.network_server.mdns_active else (self.ctrl.network_server.mdns_error or "Off")
        udp_s = "Active (:8766)" if self.ctrl.network_server.udp_beacon_active else "Off"
        self.lbl_discovery.setText(f"mDNS: {mdns_s}  |  UDP Discovery: {udp_s}")

    def _copy_health_url(self) -> None:
        url = f"http://{self.ctrl.get_active_lan_ip()}:{self.ctrl.settings.network.server_port}/api/v1/health"
        QGuiApplication.clipboard().setText(url)
        self.ctrl.status_message.emit(f"Copied health check URL to clipboard: {url}")

    def _run_firewall_check(self) -> None:
        res = self.ctrl.check_firewall_status()
        self.lbl_fw.setText(res.get("summary", ""))
        self.lbl_fw.setObjectName("BadgeOk" if res.get("ok") else "BadgeWarn")
        self.lbl_fw.style().unpolish(self.lbl_fw)
        self.lbl_fw.style().polish(self.lbl_fw)

    def _fix_firewall(self) -> None:
        ok, msg = self.ctrl.fix_windows_firewall()
        self._run_firewall_check()
        if ok:
            QMessageBox.information(self, "Windows Firewall Updated", msg)
        else:
            QMessageBox.warning(self, "Windows Firewall", msg)

    def _toggle_server(self) -> None:
        if self.ctrl.server_running:
            self.ctrl.stop_server()
        else:
            self.ctrl.start_server()
        self.refresh()

    def _on_server_changed(self, running: bool, msg: str) -> None:
        self.btn_server.setText("Stop Server" if running else "Start Server")
        self.lbl_srv_badge.setText("SERVER ONLINE" if running else "SERVER OFFLINE")
        self.lbl_srv_badge.setObjectName("BadgeOk" if running else "BadgeErr")
        self.lbl_srv_badge.style().unpolish(self.lbl_srv_badge)
        self.lbl_srv_badge.style().polish(self.lbl_srv_badge)
        self._refresh_qr_and_endpoint()

    def _regen_pin(self) -> None:
        self.ctrl.camera_manager.generate_pairing_pin()
        self._refresh_qr_and_endpoint()

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
        cams = self.ctrl.camera_manager.list_cameras()
        if 0 <= row < len(cams):
            self.ctrl.camera_manager.remove_camera(cams[row].device_id)

    def _selected_camera(self):
        row = self.table.currentRow()
        cams = self.ctrl.camera_manager.list_cameras()
        if 0 <= row < len(cams):
            return cams[row]
        return None

    def _on_camera_selected(self) -> None:
        cam = self._selected_camera()
        if cam is None:
            return
        res_idx = {720: 0, 1080: 1, 1440: 2, 2160: 3}.get(cam.config.height, 1)
        self.q_res.setCurrentIndex(res_idx)
        fps_str = str(int(round(cam.config.target_fps)))
        if self.q_fps.findText(fps_str) >= 0:
            self.q_fps.setCurrentText(fps_str)
        self.q_enabled.setChecked(cam.enabled)
        self.lbl_q_diag.setText(
            f"{cam.capabilities.manufacturer} {cam.capabilities.model} ({cam.capabilities.hardware_level}) | "
            f"TS: {cam.capabilities.timestamp_source.value}"
        )
        self._refresh_selected_preview()

    def _refresh_selected_preview(self) -> None:
        cam = self._selected_camera()
        if cam is None or not cam.latest_preview_jpeg:
            return
        img = QImage.fromData(cam.latest_preview_jpeg)
        if not img.isNull():
            self.lbl_sel_preview.setPixmap(
                QPixmap.fromImage(img).scaled(self.lbl_sel_preview.size(), Qt.AspectRatioMode.KeepAspectRatio)
            )

    def _apply_quick_config(self) -> None:
        cam = self._selected_camera()
        if cam is None:
            QMessageBox.information(self, "No Camera Selected", "Select a camera row in the table first.")
            return
        res_map = [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)]
        w, h = res_map[self.q_res.currentIndex()]
        self.ctrl.camera_manager.set_camera_enabled(cam.device_id, self.q_enabled.isChecked())
        cfg = cam.config.model_copy(update={"width": w, "height": h, "target_fps": float(self.q_fps.currentText())})
        ok, issues, alt = self.ctrl.camera_manager.update_camera_config(cam.device_id, cfg)
        if ok:
            self.lbl_q_diag.setText("✓ Applied successfully" + (f" ({'; '.join(issues)})" if issues else ""))
        else:
            alt_s = f" Suggested: {alt.width}x{alt.height}@{alt.target_fps:.0f}fps" if alt else ""
            self.lbl_q_diag.setText(f"✗ Unsupported: {'; '.join(issues)}{alt_s}")

    def refresh(self) -> None:
        self._on_server_changed(self.ctrl.server_running, "")
        cams = self.ctrl.camera_manager.list_cameras()
        prev_row = self.table.currentRow()
        self.table.setRowCount(len(cams))
        for r, c in enumerate(cams):
            sync_txt = f"±{c.clock_model.residual_rmse_ms:.1f} ms" if c.clock_model else "--"
            batt_txt = (
                f"{c.telemetry.battery_percent:.0f}% · {c.telemetry.temperature_celsius or '--'}°C"
                if c.telemetry
                else "--"
            )
            state_badge = "● ONLINE" if c.state.value in ("connected", "recording", "transferring") else "○ OFFLINE"
            vals = [
                c.camera_uid,
                c.name,
                c.role.value,
                f"{state_badge} ({c.state.value})",
                c.ip_address or c.adapter_type,
                f"{c.config.height}p @ {c.config.target_fps:.0f}fps",
                sync_txt,
                batt_txt,
                "Yes" if c.enabled else "Off",
            ]
            for col, v in enumerate(vals):
                self.table.setItem(r, col, QTableWidgetItem(v))

        if 0 <= prev_row < len(cams):
            self.table.selectRow(prev_row)
        elif cams:
            self.table.selectRow(0)

        ok_mv, msg_mv = self.ctrl.camera_manager.can_record_multi_view()
        ok_any, msg_any = self.ctrl.camera_manager.can_record_any()
        self.lbl_readiness.setText(msg_mv if ok_mv else msg_any)
        self.lbl_readiness.setObjectName("BadgeOk" if ok_mv else ("BadgeWarn" if ok_any else "BadgeErr"))
        self.lbl_readiness.style().unpolish(self.lbl_readiness)
        self.lbl_readiness.style().polish(self.lbl_readiness)


class CameraConfigPage(QWidget):
    """Page 5: Full Advanced Camera Recording Parameters with Capability Validation."""

    def __init__(self, ctrl: StudioController) -> None:
        super().__init__()
        self.ctrl = ctrl
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(10)
        root.addLayout(
            _header(
                "Camera Configuration",
                "Fine-tune resolution, FPS, H.264 bitrate, shutter speed, focus/WB lock, and inspect Camera2 capabilities",
            )
        )

        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Select Camera:"))
        self.cam_combo = QComboBox()
        self.cam_combo.currentIndexChanged.connect(self._load_selected_cam)
        sel_row.addWidget(self.cam_combo, 1)
        root.addLayout(sel_row)

        form_box = QGroupBox("Recording Parameters (Preferred: 1080p @ 30/60 FPS, 1/250s, Locked Focus/WB, Stab OFF)")
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
            QMessageBox.information(self, "No Camera", "Pair or register a camera first in Cameras & Pairing.")
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
    """Page 6: Interactive Step-by-Step ChArUco Calibration Wizard + Floor Z-Up Alignment."""

    def __init__(self, ctrl: StudioController) -> None:
        super().__init__()
        self.ctrl = ctrl
        self._last_board_poses = []
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(10)
        root.addLayout(
            _header(
                "Calibration Wizard",
                "Step-by-step multi-camera calibration: Print Board → Solve Bundle Adjustment → Align Floor (Z-Up)",
            )
        )

        self.tabs = QTabWidget()

        # Tab 1: Board & Physical Ruler Measurement
        t1 = QWidget()
        t1_l = QVBoxLayout(t1)
        bf = QFormLayout()
        self.spin_sx = QSpinBox(); self.spin_sx.setRange(4, 20); self.spin_sx.setValue(7)
        self.spin_sy = QSpinBox(); self.spin_sy.setRange(4, 20); self.spin_sy.setValue(5)
        self.spin_sq = QDoubleSpinBox(); self.spin_sq.setRange(10.0, 200.0); self.spin_sq.setValue(35.0); self.spin_sq.setSuffix(" mm")
        self.spin_mk = QDoubleSpinBox(); self.spin_mk.setRange(5.0, 180.0); self.spin_mk.setValue(26.0); self.spin_mk.setSuffix(" mm")
        self.spin_measured = QDoubleSpinBox(); self.spin_measured.setRange(10.0, 200.0); self.spin_measured.setValue(35.0); self.spin_measured.setSuffix(" mm (measure printed square!)")
        self.combo_dict = QComboBox(); self.combo_dict.addItems(list(ARUCO_DICTIONARIES.keys())); self.combo_dict.setCurrentText("DICT_5X5_250")
        self.combo_paper = QComboBox(); self.combo_paper.addItems(["A4", "A3"])
        bf.addRow("Board Grid (X / Y):", self._pair(self.spin_sx, self.spin_sy))
        bf.addRow("Nominal Square / Marker:", self._pair(self.spin_sq, self.spin_mk))
        bf.addRow("Ruler-Measured Square:", self.spin_measured)
        bf.addRow("Dictionary / Paper:", self._pair(self.combo_dict, self.combo_paper))
        t1_l.addLayout(bf)

        b1_row = QHBoxLayout()
        btn_pdf = QPushButton("1. Generate Printable PDF Board")
        btn_pdf.setObjectName("PrimaryButton")
        btn_pdf.clicked.connect(self._gen_pdf)
        btn_png = QPushButton("Export High-DPI PNG")
        btn_png.clicked.connect(self._gen_png)
        btn_next1 = QPushButton("Next: Solve Calibration →")
        btn_next1.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        for b in (btn_pdf, btn_png, btn_next1):
            b1_row.addWidget(b)
        b1_row.addStretch()
        t1_l.addLayout(b1_row)
        t1_l.addStretch()
        self.tabs.addTab(t1, "Step 1 · Print & Measure Board")

        # Tab 2: Run Calibration + Align Floor Z-Up
        t2 = QWidget()
        t2_l = QVBoxLayout(t2)
        t2_info = QLabel(
            "Place synchronized calibration frames inside camera subfolders (e.g. cam_01/, cam_02/, ...), "
            "or load an existing profile. Then optionally align the floor plane to Blender Z-Up."
        )
        t2_info.setWordWrap(True)
        t2_l.addWidget(t2_info)

        b2_row = QHBoxLayout()
        btn_calib_folder = QPushButton("Run Multi-Camera Calibration from Folder...")
        btn_calib_folder.setObjectName("PrimaryButton")
        btn_calib_folder.clicked.connect(self._run_folder_calibration)
        btn_align_floor = QPushButton("Align Floor to Blender Z-Up (from Board Pose)")
        btn_align_floor.clicked.connect(self._align_floor_z_up)
        btn_load_prof = QPushButton("Load Existing Profile JSON...")
        btn_load_prof.clicked.connect(self._load_existing)
        for b in (btn_calib_folder, btn_align_floor, btn_load_prof):
            b2_row.addWidget(b)
        b2_row.addStretch()
        t2_l.addLayout(b2_row)

        self.lbl_quality = QLabel("Calibration Quality: Not solved yet")
        self.lbl_quality.setObjectName("BadgeWarn")
        t2_l.addWidget(self.lbl_quality)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlainText(
            "Calibration Checklist:\n"
            "• Print ChArUco PDF at 100% Actual Size (disable Fit-to-Page) and mount on flat rigid foamboard.\n"
            "• Measure the 100 mm bar and enter the actual square size in Step 1.\n"
            "• Sweep board visibly across overlapping pairs of cameras (target RMSE <= 1.0 px)."
        )
        t2_l.addWidget(self.log, 1)
        self.tabs.addTab(t2, "Step 2 · Solve & Floor Z-Up")

        root.addWidget(self.tabs, 1)

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
            QMessageBox.information(self, "Board Generated", f"Saved printable ChArUco PDF:\n{p}")
        except Exception as exc:
            QMessageBox.warning(self, "Board Error", str(exc))

    def _gen_png(self) -> None:
        out, _ = QFileDialog.getSaveFileName(self, "Save ChArUco PNG", "Nuranova_ChArUco_Board.png", "PNG (*.png)")
        if not out:
            return
        p = render_charuco_png(self._board_cfg(), out)
        self.log.append(f"[OK] Exported ChArUco PNG: {p}")

    def _load_existing(self) -> None:
        fp, _ = QFileDialog.getOpenFileName(self, "Load Calibration Profile", "", "Calibration JSON (*.json)")
        if not fp:
            return
        calib = MultiCameraCalibration.load(fp)
        self.ctrl.active_calibration = calib
        self.ctrl.calibration_changed.emit()
        self._update_quality_badge(calib.global_reprojection_rmse_px, len(calib.cameras))
        self.log.append(f"[LOADED] {calib.profile_name} ({len(calib.cameras)} cams, RMSE {calib.global_reprojection_rmse_px:.3f} px)")

    def _update_quality_badge(self, rmse: float, n_cams: int) -> None:
        if rmse <= 1.0:
            self.lbl_quality.setText(f"✓ PASS — {n_cams} Cameras · Global RMSE: {rmse:.3f} px (target <= 1.0 px)")
            self.lbl_quality.setObjectName("BadgeOk")
        elif rmse <= 2.0:
            self.lbl_quality.setText(f"! ACCEPTABLE — {n_cams} Cameras · Global RMSE: {rmse:.3f} px")
            self.lbl_quality.setObjectName("BadgeWarn")
        else:
            self.lbl_quality.setText(f"✗ HIGH ERROR — {n_cams} Cameras · Global RMSE: {rmse:.3f} px (re-capture board sweep)")
            self.lbl_quality.setObjectName("BadgeErr")
        self.lbl_quality.style().unpolish(self.lbl_quality)
        self.lbl_quality.style().polish(self.lbl_quality)

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
            extr_opt, opt_boards, rep = run_board_bundle_adjustment(intr_map, extr_init, synced, board_poses, cfg)
            self._last_board_poses = opt_boards
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
            self._update_quality_badge(rep.global_reprojection_rmse_px, rep.num_cameras)
            self.log.append(f"[DONE] Global Bundle Adjustment RMSE: {rep.global_reprojection_rmse_px:.3f} px across {rep.num_cameras} cameras.")
        except Exception as exc:
            QMessageBox.warning(self, "Calibration Failed", str(exc))
            self.log.append(f"[ERROR] {exc}")

    def _align_floor_z_up(self) -> None:
        if self.ctrl.active_calibration is None:
            QMessageBox.information(self, "No Calibration", "Run or load a multi-camera calibration first.")
            return
        if not self._last_board_poses:
            QMessageBox.information(
                self,
                "Floor Reference",
                "Run calibration from a folder where Frame 0 has the ChArUco board placed flat on the studio floor.",
            )
            return
        cfg = self._board_cfg()
        Rb, tb = self._last_board_poses[0]
        corners_w = (cfg.get_chessboard_corners_3d() @ Rb.T) + tb
        aligned = align_calibration_to_world_z_up(
            self.ctrl.active_calibration,
            floor_points_world=corners_w,
            origin_world=np.mean(corners_w, axis=0),
            forward_hint_world=Rb[:, 1],
            camera_up_hint=True,
        )
        self.ctrl.active_calibration = aligned
        if self.ctrl.project_manager.active_project:
            out_p = self.ctrl.project_manager.active_project.subdir("calibration") / f"{aligned.profile_name}_zup.json"
            aligned.save(out_p)
            self.log.append(f"[OK] Aligned floor to Blender Z-Up & saved: {out_p}")
        self.ctrl.calibration_changed.emit()


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
