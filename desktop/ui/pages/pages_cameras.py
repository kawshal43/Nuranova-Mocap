"""Pages 4-7: All-in-One Cameras & QR Pairing Hub, Camera Configuration, Guided Calibration Wizard, and Calibration Profiles."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
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
    align_calibration_from_upright_board,
    align_calibration_to_world_z_up,
    calibrate_camera_intrinsics,
    initialize_extrinsics_from_board_frames,
    run_board_bundle_adjustment,
)
from desktop.calibration.charuco_board import (
    ARUCO_DICTIONARIES,
    ChArUcoBoardConfig,
    ChArUcoDetection,
    detect_charuco_corners,
    generate_printable_charuco_pdf,
    render_charuco_png,
)
from desktop.calibration.live_coach import (
    CameraCoachFeedback,
    LiveVoiceCoach,
    analyze_camera_detection,
    detect_wrong_board_preset,
    draw_live_calibration_hud,
    evaluate_multi_cam_coaching,
    play_coach_sound,
    pose_spoken_instruction,
    render_board_thumbnail_bgr,
)
from desktop.ui.app_controller import StudioController
from desktop.ui.pages.pages_workspace import _header
from desktop.ui.widgets.calibration_guide import CALIBRATION_POSE_STEPS, CalibrationGuideWidget
from engine.math_core.camera_model import CalibratedCamera, CameraIntrinsics, MultiCameraCalibration
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
                "Cameras — USB Webcams & Android Phones",
                "Connect USB cameras plugged directly into your PC and/or pair Android phones over Wi-Fi (QR / 6-digit PIN)",
            )
        )

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ---------------- LEFT PANEL: Instant Phone Pairing Card ----------------
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setMinimumWidth(360)
        left_scroll.setMaximumWidth(430)
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
        self.combo_lan.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.combo_lan.setMinimumContentsLength(18)
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

        # ---------------- RIGHT PANEL: USB Cameras + Paired/Connected Cameras + Inline Quick Config ----------------
        right_w = QWidget()
        right_l = QVBoxLayout(right_w)
        right_l.setContentsMargins(6, 0, 0, 0)
        right_l.setSpacing(10)

        # USB Connected Cameras Card (Webcams / USB Capture Cards — works standalone or mixed with Android phones!)
        usb_box = QGroupBox("USB Connected Cameras (PC Webcams / Capture Cards)")
        ul = QHBoxLayout(usb_box)
        ul.setSpacing(8)
        self._discovered_usb_cams: List = []
        self.combo_usb = QComboBox()
        self.combo_usb.setMinimumWidth(220)
        self.combo_usb.addItem("Click 'Scan USB' to detect PC cameras...", None)
        btn_scan_usb = QPushButton("🔍 Scan USB")
        btn_scan_usb.setToolTip(
            "Detect USB webcams, laptop cameras, and USB capture cards connected to this PC"
        )
        btn_scan_usb.clicked.connect(self._scan_usb_cameras)
        self.btn_connect_usb = QPushButton("+ Connect USB Cam")
        self.btn_connect_usb.setObjectName("PrimaryButton")
        self.btn_connect_usb.clicked.connect(self._connect_selected_usb)
        btn_connect_all_usb = QPushButton("Connect All USB")
        btn_connect_all_usb.clicked.connect(self._connect_all_usb)
        ul.addWidget(self.combo_usb, 1)
        ul.addWidget(btn_scan_usb)
        ul.addWidget(self.btn_connect_usb)
        ul.addWidget(btn_connect_all_usb)
        right_l.addWidget(usb_box)

        top_bar = QHBoxLayout()
        self.lbl_readiness = QLabel("")
        self.lbl_readiness.setObjectName("BadgeWarn")
        btn_add_sim = QPushButton("+ Add Simulated Cam")
        btn_add_sim.clicked.connect(self._add_simulated_cam)
        btn_remove = QPushButton("Disconnect / Forget")
        btn_remove.setObjectName("DangerButton")
        btn_remove.clicked.connect(self._remove_selected)
        top_bar.addWidget(self.lbl_readiness)
        top_bar.addStretch()
        top_bar.addWidget(btn_add_sim)
        top_bar.addWidget(btn_remove)
        right_l.addLayout(top_bar)

        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(
            [
                "UID",
                "Name",
                "Role",
                "Status",
                "IP / Source",
                "Mode",
                "Sync",
                "Batt / Temp",
                "Enabled",
            ]
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
        self.lbl_sel_preview.setStyleSheet(
            "border: 1px solid #2D3D50; border-radius: 6px; background: #0B1016;"
        )
        ql.addWidget(self.lbl_sel_preview)

        qform = QFormLayout()
        self.q_res = QComboBox()
        self.q_res.addItems(
            ["1280x720 (720p)", "1920x1080 (1080p)", "2560x1440 (1440p)", "3840x2160 (4K)"]
        )
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
        ctrl.telemetry_updated.connect(lambda _did: self._refresh_telemetry_cells())
        ctrl.server_state_changed.connect(self._on_server_changed)

        self._populate_lan_adapters()
        self.refresh()
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            QTimer.singleShot(350, self._scan_usb_cameras)
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

        mdns_s = (
            "Active"
            if self.ctrl.network_server.mdns_active
            else (self.ctrl.network_server.mdns_error or "Off")
        )
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
            ok = self.ctrl.start_server()
            if not ok and self.ctrl.server_error:
                QMessageBox.warning(self, "Server Could Not Start", self.ctrl.server_error)
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

    def _scan_usb_cameras(self) -> None:
        self._discovered_usb_cams = self.ctrl.scan_usb_cameras()
        self.combo_usb.blockSignals(True)
        self.combo_usb.clear()
        if not self._discovered_usb_cams:
            self.combo_usb.addItem(
                "No USB cameras detected — plug in webcam & click Scan USB", None
            )
        else:
            for c in self._discovered_usb_cams:
                tag = " [CONNECTED]" if c.already_connected else ""
                label = f"USB #{c.index} — {c.name} ({c.width}×{c.height} @ {c.fps:.0f}fps){tag}"
                self.combo_usb.addItem(label, c.index)
        self.combo_usb.blockSignals(False)

    def _connect_selected_usb(self) -> None:
        idx = self.combo_usb.currentData()
        if idx is None:
            self._scan_usb_cameras()
            idx = self.combo_usb.currentData()
        if idx is None:
            QMessageBox.information(
                self,
                "No USB Camera Found",
                "No USB camera was detected.\nPlug in a USB webcam or capture card, make sure no other app (Zoom/Teams/OBS) is locking it, and click 'Scan USB'.",
            )
            return
        meta = next((c for c in self._discovered_usb_cams if c.index == idx), None)
        try:
            self.ctrl.connect_usb_camera(
                index=int(idx),
                name=meta.name if meta else f"USB Camera #{idx}",
                width=meta.width if meta else 1280,
                height=meta.height if meta else 720,
                fps=meta.fps if meta else 30.0,
                supported_modes=meta.supported_modes if meta else None,
            )
            self._scan_usb_cameras()
            self.refresh()
        except Exception as exc:
            QMessageBox.warning(self, "USB Camera Error", str(exc))

    def _connect_all_usb(self) -> None:
        connected = self.ctrl.connect_all_usb_cameras()
        self._scan_usb_cameras()
        self.refresh()
        if not connected:
            if any(c.already_connected for c in self._discovered_usb_cams):
                self.ctrl.status_message.emit("All detected USB cameras are already connected.")
            else:
                QMessageBox.information(
                    self,
                    "No New USB Cameras",
                    "No new USB cameras detected. Plug in a USB webcam and try 'Scan USB'.",
                )

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
            self.ctrl.disconnect_or_remove_camera(cams[row].device_id)
            self._scan_usb_cameras()

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
                QPixmap.fromImage(img).scaled(
                    self.lbl_sel_preview.size(), Qt.AspectRatioMode.KeepAspectRatio
                )
            )

    def _apply_quick_config(self) -> None:
        cam = self._selected_camera()
        if cam is None:
            QMessageBox.information(
                self, "No Camera Selected", "Select a camera row in the table first."
            )
            return
        res_map = [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)]
        w, h = res_map[self.q_res.currentIndex()]
        self.ctrl.camera_manager.set_camera_enabled(cam.device_id, self.q_enabled.isChecked())
        cfg = cam.config.model_copy(
            update={"width": w, "height": h, "target_fps": float(self.q_fps.currentText())}
        )
        ok, issues, alt = self.ctrl.camera_manager.update_camera_config(cam.device_id, cfg)
        if ok:
            self.lbl_q_diag.setText(
                "✓ Applied successfully" + (f" ({'; '.join(issues)})" if issues else "")
            )
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
                (
                    "USB / AC"
                    if c.adapter_type == "usb_camera"
                    else f"{c.telemetry.battery_percent:.0f}% · {c.telemetry.temperature_celsius or '--'}°C"
                )
                if c.telemetry
                else ("USB / AC" if c.adapter_type == "usb_camera" else "--")
            )
            state_badge = (
                "● ONLINE"
                if c.state.value in ("connected", "recording", "transferring")
                else "○ OFFLINE"
            )
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
        self.lbl_readiness.setObjectName(
            "BadgeOk" if ok_mv else ("BadgeWarn" if ok_any else "BadgeErr")
        )
        self.lbl_readiness.style().unpolish(self.lbl_readiness)
        self.lbl_readiness.style().polish(self.lbl_readiness)

    def _refresh_telemetry_cells(self) -> None:
        """Update sync & battery columns in-place on high-frequency telemetry heartbeats (zero flicker)."""
        cams = self.ctrl.camera_manager.list_cameras()
        if len(cams) != self.table.rowCount():
            self.refresh()
            return
        for r, c in enumerate(cams):
            sync_txt = f"±{c.clock_model.residual_rmse_ms:.1f} ms" if c.clock_model else "--"
            batt_txt = (
                (
                    "USB / AC"
                    if c.adapter_type == "usb_camera"
                    else f"{c.telemetry.battery_percent:.0f}% · {c.telemetry.temperature_celsius or '--'}°C"
                )
                if c.telemetry
                else ("USB / AC" if c.adapter_type == "usb_camera" else "--")
            )
            state_badge = (
                "● ONLINE"
                if c.state.value in ("connected", "recording", "transferring")
                else "○ OFFLINE"
            )
            for col, txt in ((3, f"{state_badge} ({c.state.value})"), (6, sync_txt), (7, batt_txt)):
                item = self.table.item(r, col)
                if item is not None:
                    item.setText(txt)


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

        form_box = QGroupBox(
            "Recording Parameters (Preferred: 1080p @ 30/60 FPS, 1/250s, Locked Focus/WB, Stab OFF)"
        )
        form = QFormLayout(form_box)
        self.res_combo = QComboBox()
        self.res_combo.addItems(
            ["1280x720 (720p)", "1920x1080 (1080p)", "2560x1440 (1440p)", "3840x2160 (4K)"]
        )
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
        self.shutter_spin.setValue(100)
        self.shutter_spin.setPrefix("1/")
        self.shutter_spin.setSuffix(" s")
        self.chk_manual_exp = QCheckBox("Manual Exposure (leave OFF indoors for Auto Anti-Flicker)")
        self.chk_manual_exp.setChecked(False)
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

        btn_apply = QPushButton("Validate && Apply to Camera")
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
            QMessageBox.information(
                self, "No Camera", "Pair or register a camera first in Cameras & Pairing."
            )
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
        lines = [
            f"Validation Result: {'ACCEPTED' if ok else 'UNSUPPORTED ON THIS DEVICE'}"
        ] + issues
        if alt is not None:
            lines.append(
                f"Suggested Supported Alternative: {alt.width}x{alt.height} @ {alt.target_fps:.0f} FPS"
            )
        self.diag_text.setPlainText("\n".join(lines))


_IGNORED_FOLDER_NAMES = {"__pycache__", "build", "dist", "node_modules", "venv", ".venv"}


def find_camera_image_subfolders(root_folder: Path) -> List[Path]:
    """Return sorted non-hidden subfolders that contain at least one PNG/JPG calibration image."""
    valid: List[Path] = []
    if not root_folder.exists() or not root_folder.is_dir():
        return valid
    for d in sorted(root_folder.iterdir()):
        if not d.is_dir():
            continue
        if d.name.startswith(".") or d.name.lower() in _IGNORED_FOLDER_NAMES:
            continue
        has_img = any(
            p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg") for p in d.iterdir()
        )
        if has_img:
            valid.append(d)
    return valid


def _scale_intrinsics_to_resolution(
    intr: CameraIntrinsics,
    target_w: int,
    target_h: int,
) -> CameraIntrinsics:
    """Scale pinhole camera intrinsics K linearly if target recording aspect ratio matches preview."""
    if (
        target_w <= 0
        or target_h <= 0
        or (intr.image_width == target_w and intr.image_height == target_h)
    ):
        return intr
    ar_src = intr.image_width / max(float(intr.image_height), 1.0)
    ar_dst = target_w / max(float(target_h), 1.0)
    if abs(ar_src - ar_dst) / max(ar_dst, 1e-6) > 0.06:
        # Different aspect ratios (e.g. 4:3 preview vs 16:9 recording) — keep solved preview resolution intact
        return intr
    sx = float(target_w) / float(intr.image_width)
    sy = float(target_h) / float(intr.image_height)
    K_scaled = intr.camera_matrix.copy()
    K_scaled[0, 0] *= sx
    K_scaled[0, 2] *= sx
    K_scaled[1, 1] *= sy
    K_scaled[1, 2] *= sy
    return CameraIntrinsics(
        camera_uid=intr.camera_uid,
        image_width=int(target_w),
        image_height=int(target_h),
        camera_matrix=K_scaled,
        dist_coeffs=intr.dist_coeffs.copy(),
        reprojection_error_px=float(intr.reprojection_error_px * 0.5 * (sx + sy)),
    )


class CalibrationWizardPage(QWidget):
    """Page 6: Visual Room/Camera/Board Placement Guide + Printable ChArUco + Live Guided Capture & Floor Z-Up."""

    def __init__(self, ctrl: StudioController) -> None:
        super().__init__()
        self.ctrl = ctrl
        self._last_board_poses = []
        # Captured live multi-camera snapshots: list of (pose_index, {uid: (det, (w, h), bgr_bytes)})
        self._captured_frames: List[
            Tuple[int, Dict[str, Tuple[ChArUcoDetection, Tuple[int, int], bytes]]]
        ] = []
        self._completed_pose_indices: Set[int] = set()
        self._countdown_left: int = 0
        self._live_cam_img_labels: Dict[str, QLabel] = {}
        self._live_cam_status_labels: Dict[str, QLabel] = {}
        self._tile_device_ids: List[str] = []
        self._live_pose_pills: List[QPushButton] = []

        # Live hands-free coaching & auto-capture state
        self.voice_coach = LiveVoiceCoach()
        self._auto_capture_enabled: bool = False
        self._user_paused_auto_capture: bool = False
        self._steady_ticks: int = 0
        self._steady_ticks_target: int = (
            2  # Fast ~0.35s lock-on: auto-captures & saves as soon as both cameras lock!
        )
        self._prev_board_centers: Dict[str, Tuple[float, float]] = {}
        self._cooldown_until: float = 0.0
        self._flash_until: float = 0.0
        self._flash_text: str = ""
        self._last_banner_notice: str = ""
        self._mismatch_tick: int = 0
        self._suggested_alt_board: str = ""
        self._capture_session_dir: Optional[Path] = None
        self._solve_in_progress: bool = False

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(10)
        root.addLayout(
            _header(
                "Calibration Wizard",
                "Visual Camera & Board Placement Guide → Print/Measure ChArUco → Live Hands-Free Coached Calibration & Floor Z-Up",
            )
        )

        self.tabs = QTabWidget()

        # ==================== TAB 1: Visual Camera & Board Placement Guide ====================
        t0 = QWidget()
        t0_l = QVBoxLayout(t0)
        t0_l.setContentsMargins(10, 10, 10, 10)
        t0_l.setSpacing(10)

        self.guide = CalibrationGuideWidget()
        self.guide.pose_changed.connect(self._on_guide_pose_changed)
        t0_l.addWidget(self.guide, 1)

        rules_card = QFrame()
        rules_card.setObjectName("Card")
        rl = QHBoxLayout(rules_card)
        rl.setContentsMargins(12, 8, 12, 8)
        rules_lbl = QLabel(
            "Checklist:  ① Mount phones in LANDSCAPE on sturdy tripods (1.5–1.8 m high)   ·   "
            "② Aim slightly down (~15°) so floor + head fit in frame   ·   "
            "③ Disable Stabilization & Lock Focus   ·   "
            "④ NEVER move tripods after calibrating"
        )
        rules_lbl.setWordWrap(True)
        rules_lbl.setObjectName("Subtitle")
        rl.addWidget(rules_lbl)
        t0_l.addWidget(rules_card)

        g_btns = QHBoxLayout()
        btn_jump_live = QPushButton("▶  Start Live Coached Calibration →")
        btn_jump_live.setObjectName("PrimaryButton")
        btn_jump_live.clicked.connect(self._jump_to_live_guided_calibration)
        btn_to_print = QPushButton("Print / Measure ChArUco Board (Step 2) →")
        btn_to_print.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        g_btns.addWidget(btn_jump_live)
        g_btns.addWidget(btn_to_print)
        g_btns.addStretch()
        t0_l.addLayout(g_btns)
        self.tabs.addTab(t0, "Step 1 · Camera && Board Setup Guide")

        # ==================== TAB 2: Print & Measure ChArUco Board ====================
        t1 = QWidget()
        t1_l = QVBoxLayout(t1)
        t1_l.setContentsMargins(12, 12, 12, 12)
        t1_l.setSpacing(10)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Quick Board Preset:"))
        btn_a4_preset = QPushButton("A4 Standard (7×5 · 35 mm squares)")
        btn_a4_preset.clicked.connect(self._apply_a4_preset)
        btn_a3_preset = QPushButton("A3 Studio (9×6 · 40 mm squares)")
        btn_a3_preset.clicked.connect(self._apply_a3_preset)
        btn_open_printables = QPushButton("Open Ready-Made Printables Folder")
        btn_open_printables.clicked.connect(self._open_printables_folder)
        for b in (btn_a4_preset, btn_a3_preset, btn_open_printables):
            preset_row.addWidget(b)
        preset_row.addStretch()
        t1_l.addLayout(preset_row)

        bf = QFormLayout()
        self.spin_sx = QSpinBox()
        self.spin_sx.setRange(4, 20)
        self.spin_sx.setValue(7)
        self.spin_sy = QSpinBox()
        self.spin_sy.setRange(4, 20)
        self.spin_sy.setValue(5)
        self.spin_sq = QDoubleSpinBox()
        self.spin_sq.setRange(10.0, 200.0)
        self.spin_sq.setValue(35.0)
        self.spin_sq.setSuffix(" mm")
        self.spin_mk = QDoubleSpinBox()
        self.spin_mk.setRange(5.0, 180.0)
        self.spin_mk.setValue(26.0)
        self.spin_mk.setSuffix(" mm")
        self.spin_measured = QDoubleSpinBox()
        self.spin_measured.setRange(10.0, 200.0)
        self.spin_measured.setValue(35.0)
        self.spin_measured.setSuffix(" mm (measure printed square!)")
        self.combo_dict = QComboBox()
        self.combo_dict.addItems(list(ARUCO_DICTIONARIES.keys()))
        self.combo_dict.setCurrentText("DICT_5X5_250")
        self.combo_paper = QComboBox()
        self.combo_paper.addItems(["A4", "A3"])
        bf.addRow("Board Grid (X / Y):", self._pair(self.spin_sx, self.spin_sy))
        bf.addRow("Nominal Square / Marker:", self._pair(self.spin_sq, self.spin_mk))
        bf.addRow("Ruler-Measured Square:", self.spin_measured)
        bf.addRow("Dictionary / Paper:", self._pair(self.combo_dict, self.combo_paper))
        t1_l.addLayout(bf)

        for w_chg in (self.spin_sx, self.spin_sy, self.spin_sq, self.spin_mk, self.spin_measured):
            w_chg.valueChanged.connect(self._refresh_active_board_card)
        self.combo_dict.currentIndexChanged.connect(self._refresh_active_board_card)
        self.combo_paper.currentIndexChanged.connect(self._refresh_active_board_card)

        print_tip = QLabel(
            "Printing Tips:\n"
            "• Print at 100% Actual Size (turn OFF Fit-to-Page / Shrink-to-Fit) on matte paper.\n"
            "• Glue/tape flat onto rigid foamboard or thick flat cardboard (never hold loose curling paper!).\n"
            "• Verify the 100.0 mm bar at the bottom with a physical ruler and enter your exact printed square mm above.\n"
            "• If you printed the ready-made A3 PDF (Nuranova_ChArUco_A3_9x6_40mm.pdf), click 'A3 Studio (9×6 · 40 mm)' above!"
        )
        print_tip.setObjectName("Subtitle")
        print_tip.setWordWrap(True)
        t1_l.addWidget(print_tip)

        b1_row = QHBoxLayout()
        btn_pdf = QPushButton("Generate Printable PDF Board")
        btn_pdf.setObjectName("PrimaryButton")
        btn_pdf.clicked.connect(self._gen_pdf)
        btn_png = QPushButton("Export High-DPI PNG")
        btn_png.clicked.connect(self._gen_png)
        btn_next1 = QPushButton("Next: Live Coached Calibration →")
        btn_next1.clicked.connect(self._jump_to_live_guided_calibration)
        for b in (btn_pdf, btn_png, btn_next1):
            b1_row.addWidget(b)
        b1_row.addStretch()
        t1_l.addLayout(b1_row)
        t1_l.addStretch()
        self.tabs.addTab(t1, "Step 2 · Print && Measure Board")

        # ==================== TAB 3: Live Coached Capture & Solve ====================
        t2 = QWidget()
        t2_l = QVBoxLayout(t2)
        t2_l.setContentsMargins(10, 8, 10, 8)
        t2_l.setSpacing(8)

        # Top Command Row: Left = SHOW THIS BOARD Card | Right = GIANT LIVE COACH & 6-Pose Stepper
        cmd_row = QHBoxLayout()
        cmd_row.setSpacing(10)

        # --- Active Board Preview Card ("which board to show to camera") ---
        board_card = QFrame()
        board_card.setObjectName("Card")
        board_card.setMinimumWidth(295)
        board_card.setMaximumWidth(355)
        bc_l = QHBoxLayout(board_card)
        bc_l.setContentsMargins(10, 8, 10, 8)
        bc_l.setSpacing(10)

        self.lbl_board_thumb = QLabel("Board")
        self.lbl_board_thumb.setFixedSize(108, 76)
        self.lbl_board_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_board_thumb.setStyleSheet(
            "background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 6px;"
        )
        bc_l.addWidget(self.lbl_board_thumb)

        bc_info = QVBoxLayout()
        bc_info.setSpacing(3)
        lbl_show_hdr = QLabel("SHOW THIS BOARD")
        lbl_show_hdr.setObjectName("MetricTitle")
        self.lbl_board_spec = QLabel("A4 ChArUco (7×5 · 35mm)")
        self.lbl_board_spec.setStyleSheet("font-size: 12px; font-weight: 700;")
        self.lbl_board_spec.setWordWrap(True)

        sw_row = QHBoxLayout()
        sw_row.setSpacing(4)
        btn_sw_a4 = QPushButton("A4 (7×5)")
        btn_sw_a4.setToolTip("Switch to A4 7×5 (35 mm squares)")
        btn_sw_a4.clicked.connect(self._apply_a4_preset)
        btn_sw_a3 = QPushButton("A3 (9×6)")
        btn_sw_a3.setToolTip("Switch to A3 9×6 (40 mm squares)")
        btn_sw_a3.clicked.connect(self._apply_a3_preset)
        sw_row.addWidget(btn_sw_a4)
        sw_row.addWidget(btn_sw_a3)

        self.btn_board_mismatch = QPushButton("")
        self.btn_board_mismatch.setObjectName("PrimaryButton")
        self.btn_board_mismatch.setVisible(False)
        self.btn_board_mismatch.clicked.connect(self._accept_suggested_board)

        bc_info.addWidget(lbl_show_hdr)
        bc_info.addWidget(self.lbl_board_spec)
        bc_info.addLayout(sw_row)
        bc_info.addWidget(self.btn_board_mismatch)
        bc_l.addLayout(bc_info, 1)
        cmd_row.addWidget(board_card)

        # --- GIANT Room-Readable Live Coaching Card + 6-Pose Progress Stepper ---
        coach_card = QFrame()
        coach_card.setObjectName("Card")
        cc_l = QVBoxLayout(coach_card)
        cc_l.setContentsMargins(12, 8, 12, 8)
        cc_l.setSpacing(5)

        # 6-pose checklist pills
        step_row = QHBoxLayout()
        step_row.setSpacing(4)
        self._live_pose_pills.clear()
        for step in CALIBRATION_POSE_STEPS:
            pill = QPushButton(step.short_label)
            pill.setCheckable(True)
            pill.clicked.connect(lambda checked=False, i=step.index: self.guide.set_pose_index(i))
            step_row.addWidget(pill)
            self._live_pose_pills.append(pill)
        step_row.addStretch()
        self.lbl_captured_count = QLabel("Captured: 0 / 6")
        self.lbl_captured_count.setObjectName("BadgeWarn")
        step_row.addWidget(self.lbl_captured_count)
        cc_l.addLayout(step_row)

        # Giant headline instruction (COME CLOSER / STEP BACK / MOVE LEFT / POSE 1 CAPTURED -> NEXT POSE!)
        self.lbl_coach_headline = QLabel(
            "Click '▶ Start Guided Auto-Capture' and hold your ChArUco board in view"
        )
        self.lbl_coach_headline.setStyleSheet("font-size: 16px; font-weight: 800;")
        self.lbl_coach_headline.setWordWrap(True)
        cc_l.addWidget(self.lbl_coach_headline)

        self.lbl_live_pose_title = QLabel("")
        self.lbl_live_pose_title.setStyleSheet("font-size: 12px; font-weight: 700;")
        self.lbl_live_pose_desc = QLabel("")
        self.lbl_live_pose_desc.setObjectName("Subtitle")
        self.lbl_live_pose_desc.setWordWrap(True)
        cc_l.addWidget(self.lbl_live_pose_title)
        cc_l.addWidget(self.lbl_live_pose_desc)

        # Lock-on steadiness progress bar
        self.hold_progress_bar = QProgressBar()
        self.hold_progress_bar.setRange(0, 100)
        self.hold_progress_bar.setValue(0)
        self.hold_progress_bar.setMaximumHeight(18)
        self.hold_progress_bar.setTextVisible(True)
        self.hold_progress_bar.setFormat("Waiting for both cameras to lock...")
        cc_l.addWidget(self.hold_progress_bar)

        cmd_row.addWidget(coach_card, 1)
        t2_l.addLayout(cmd_row)

        # Action Toolbar: Hands-Free Auto-Capture + Manual Capture + Voice Coach + Solve + Folder
        cap_row = QHBoxLayout()
        cap_row.setSpacing(6)

        self.btn_auto_coach = QPushButton("▶  Start Guided Auto-Capture")
        self.btn_auto_coach.setObjectName("PrimaryButton")
        self.btn_auto_coach.clicked.connect(self._toggle_auto_capture)

        self.btn_cap_now = QPushButton("📸 Capture Now")
        self.btn_cap_now.clicked.connect(lambda: self._capture_live_pose(auto_triggered=False))

        self.btn_cap_timer = QPushButton("⏱ 3s Timer")
        self.btn_cap_timer.clicked.connect(self._start_countdown_capture)

        self.chk_voice_coach = QCheckBox("🔊 Live Voice Coach")
        self.chk_voice_coach.setChecked(True)
        self.chk_voice_coach.setToolTip(
            "Continuously speaks distance, direction, and next-pose instructions aloud until each pose is captured"
        )

        btn_test_voice = QPushButton("Test Voice")
        btn_test_voice.setToolTip("Test speaker volume before walking away from the PC")
        btn_test_voice.clicked.connect(self._test_voice_coach)

        self.btn_solve_live = QPushButton("✓ Solve Calibration (+ Floor Z-Up)")
        self.btn_solve_live.setObjectName("PrimaryButton")
        self.btn_solve_live.setEnabled(False)
        self.btn_solve_live.clicked.connect(self._solve_live_calibration)

        btn_clear_cap = QPushButton("Reset")
        btn_clear_cap.clicked.connect(self._clear_captured_poses)

        btn_view_diagram = QPushButton("Room Map")
        btn_view_diagram.clicked.connect(lambda: self.tabs.setCurrentIndex(0))

        btn_debug_snap = QPushButton("Save Debug Frames")
        btn_debug_snap.setToolTip(
            "Save current live camera JPEGs and detection HUD for troubleshooting"
        )
        btn_debug_snap.clicked.connect(self._save_debug_frames)

        btn_calib_folder = QPushButton("From Folder...")
        btn_calib_folder.clicked.connect(self._run_folder_calibration)

        btn_load_prof = QPushButton("Load JSON...")
        btn_load_prof.clicked.connect(self._load_existing)

        for b in (
            self.btn_auto_coach,
            self.btn_cap_now,
            self.btn_cap_timer,
            self.chk_voice_coach,
            btn_test_voice,
            self.btn_solve_live,
            btn_clear_cap,
            btn_view_diagram,
            btn_debug_snap,
            btn_calib_folder,
            btn_load_prof,
        ):
            cap_row.addWidget(b)
        cap_row.addStretch()
        t2_l.addLayout(cap_row)

        # Live Camera Grid (with real-time ChArUco HUD, target brackets & directional coaching!) + Log
        live_split = QSplitter(Qt.Orientation.Horizontal)

        cam_box = QGroupBox(
            "Live Camera Calibration View — Place ChArUco Board Inside Dashed Guide Brackets"
        )
        self.live_cam_grid = QGridLayout(cam_box)
        self.live_cam_grid.setSpacing(8)
        live_split.addWidget(cam_box)

        right_panel = QWidget()
        rp_l = QVBoxLayout(right_panel)
        rp_l.setContentsMargins(4, 0, 0, 0)
        rp_l.setSpacing(6)
        self.lbl_quality = QLabel("Calibration Quality: Not solved yet")
        self.lbl_quality.setObjectName("BadgeWarn")
        rp_l.addWidget(self.lbl_quality)

        btn_align_floor = QPushButton("Re-Align Floor Z-Up")
        btn_align_floor.clicked.connect(self._align_floor_z_up)
        rp_l.addWidget(btn_align_floor)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlainText(
            "Live Coached Calibration:\n"
            "1. Confirm the 'SHOW THIS BOARD' card above matches your printed ChArUco board (A4 7×5 or A3 9×6).\n"
            "2. Click '▶ Start Guided Auto-Capture' — walk into view and follow the live instructions (COME CLOSER / STEP BACK / MOVE LEFT / RIGHT / HOLD STILL).\n"
            "3. Hold steady ~1s per pose: it beeps, flashes green, notifies you to move to the next pose (1 → 6), and auto-solves Calibration + Floor Z-Up!"
        )
        rp_l.addWidget(self.log, 1)
        live_split.addWidget(right_panel)
        live_split.setSizes([760, 320])
        t2_l.addWidget(live_split, 1)

        self.tabs.addTab(t2, "Step 3 · Live Guided Calibration")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        root.addWidget(self.tabs, 1)

        # Timers for live ChArUco corner detection coaching & 3s solo countdown
        self._live_timer = QTimer(self)
        self._live_timer.timeout.connect(self._refresh_live_charuco_feeds)
        self._live_timer.start(180)

        self._countdown_timer = QTimer(self)
        self._countdown_timer.timeout.connect(self._on_countdown_tick)

        ctrl.cameras_changed.connect(self._refresh_camera_tiles)
        self._refresh_active_board_card()
        self._on_guide_pose_changed(0)
        self._refresh_camera_tiles()

    @staticmethod
    def _pair(a: QWidget, b: QWidget) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(a)
        row.addWidget(b)
        return w

    def _on_tab_changed(self, tab_index: int) -> None:
        if tab_index == 2:
            self._maybe_auto_arm_capture()

    def _maybe_auto_arm_capture(self) -> None:
        """Auto-arm hands-free capture once >= 2 cameras are streaming on Step 3 (unless user paused it)."""
        if self._auto_capture_enabled or self._user_paused_auto_capture or self._solve_in_progress:
            return
        if len(self._completed_pose_indices) >= len(CALIBRATION_POSE_STEPS):
            return
        streaming = [c for c in self.ctrl.camera_manager.list_cameras() if c.latest_preview_jpeg]
        if len(streaming) >= 2:
            self._set_auto_capture(True, user_triggered=False)

    def _jump_to_live_guided_calibration(self) -> None:
        self._user_paused_auto_capture = False
        self.tabs.setCurrentIndex(2)
        self._maybe_auto_arm_capture()

    def _ensure_live_capture_session_dir(self) -> Path:
        """Return persistent directory for auto-saving captured live calibration frames (works with or without a project)."""
        if self._capture_session_dir is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            if self.ctrl.project_manager.active_project is not None:
                base = (
                    self.ctrl.project_manager.active_project.subdir("calibration") / "live_captures"
                )
            else:
                base = self.ctrl._last_calib_path.parent / "calibrations" / "live_captures"
            self._capture_session_dir = base / f"session_{ts}"
        self._capture_session_dir.mkdir(parents=True, exist_ok=True)
        return self._capture_session_dir

    def _refresh_active_board_card(self) -> None:
        cfg = self._board_cfg()
        try:
            thumb_bgr = render_board_thumbnail_bgr(cfg, 104, 72)
            rgb = cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888).copy()
            self.lbl_board_thumb.setPixmap(QPixmap.fromImage(qimg))
        except Exception:
            self.lbl_board_thumb.setText(f"{cfg.squares_x}×{cfg.squares_y}")

        if cfg.squares_x == 9 and cfg.squares_y == 6:
            preset_lbl = "A3 Studio ChArUco"
        elif cfg.squares_x == 7 and cfg.squares_y == 5:
            preset_lbl = "A4 Standard ChArUco"
        else:
            preset_lbl = f"Custom {self.combo_paper.currentText()} ChArUco"
        self.lbl_board_spec.setText(
            f"{preset_lbl}\n{cfg.squares_x}×{cfg.squares_y} grid · {cfg.measured_square_length_mm or cfg.square_length_mm:.1f} mm sq · {cfg.dictionary_name}"
        )
        self.btn_board_mismatch.setVisible(False)
        self._suggested_alt_board = ""

    def _accept_suggested_board(self) -> None:
        if self._suggested_alt_board == "A3":
            self._apply_a3_preset()
        elif self._suggested_alt_board == "A4":
            self._apply_a4_preset()
        self.btn_board_mismatch.setVisible(False)
        self._suggested_alt_board = ""

    def _apply_a4_preset(self) -> None:
        self.spin_sx.setValue(7)
        self.spin_sy.setValue(5)
        self.spin_sq.setValue(35.0)
        self.spin_mk.setValue(26.0)
        self.spin_measured.setValue(35.0)
        self.combo_dict.setCurrentText("DICT_5X5_250")
        self.combo_paper.setCurrentText("A4")
        self._refresh_active_board_card()
        self.log.append(
            "[BOARD] Active board set to A4 Standard ChArUco (7×5, 35 mm squares, 26 mm markers)."
        )

    def _apply_a3_preset(self) -> None:
        self.spin_sx.setValue(9)
        self.spin_sy.setValue(6)
        self.spin_sq.setValue(40.0)
        self.spin_mk.setValue(30.0)
        self.spin_measured.setValue(40.0)
        self.combo_dict.setCurrentText("DICT_5X5_250")
        self.combo_paper.setCurrentText("A3")
        self._refresh_active_board_card()
        self.log.append(
            "[BOARD] Active board set to A3 Studio ChArUco (9×6, 40 mm squares, 30 mm markers)."
        )

    def _open_printables_folder(self) -> None:
        pdir = Path(__file__).resolve().parents[3] / "documentation" / "printable"
        if not pdir.exists():
            QMessageBox.information(self, "Printables Folder", f"Folder not found: {pdir}")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(pdir))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(pdir)])
        except Exception as exc:
            QMessageBox.warning(self, "Open Folder", str(exc))

    def _refresh_pose_stepper_pills(self) -> None:
        cur = self.guide.canvas.pose_index
        for step, pill in zip(CALIBRATION_POSE_STEPS, self._live_pose_pills):
            done = step.index in self._completed_pose_indices
            pill.setText(f"✓ {step.short_label}" if done else step.short_label)
            pill.setChecked(step.index == cur)
            if step.index == cur:
                pill.setObjectName("PrimaryButton")
            elif done:
                pill.setObjectName("BadgeOk")
            else:
                pill.setObjectName("")
            pill.style().unpolish(pill)
            pill.style().polish(pill)

    def _on_guide_pose_changed(self, idx: int) -> None:
        step = CALIBRATION_POSE_STEPS[max(0, min(len(CALIBRATION_POSE_STEPS) - 1, idx))]
        self.lbl_live_pose_title.setText(step.title)
        self.lbl_live_pose_desc.setText(f"Target position: {step.where_to_place}")
        self._steady_ticks = 0
        self.hold_progress_bar.setValue(0)
        self._refresh_pose_stepper_pills()

    def _toggle_auto_capture(self) -> None:
        self._set_auto_capture(not self._auto_capture_enabled, user_triggered=True)

    def _active_board_short_name(self) -> str:
        cfg = self._board_cfg()
        if cfg.squares_x == 9 and cfg.squares_y == 6:
            return "A3"
        if cfg.squares_x == 7 and cfg.squares_y == 5:
            return "A4"
        return "ChArUco"

    def _test_voice_coach(self) -> None:
        bname = self._active_board_short_name()
        play_coach_sound("tick")
        self.voice_coach.announce(
            f"Voice coach is ready! Hold up your {bname} board in front of both cameras and I will talk you through every pose.",
            priority=True,
        )
        self.log.append("[VOICE] Tested Live Voice Coach speaker output.")

    def _save_debug_frames(self) -> None:
        cams = [c for c in self.ctrl.camera_manager.list_cameras() if c.latest_preview_jpeg]
        if not cams:
            QMessageBox.information(
                self, "No Live Frames", "No cameras are streaming preview frames right now."
            )
            return
        out_dir = Path.cwd() / "calibration_debug"
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = self._board_cfg()
        ts = time.strftime("%Y%m%d_%H%M%S")
        saved: List[str] = []
        for cam in cams:
            jpeg = cam.latest_preview_jpeg
            if not jpeg:
                continue
            raw_p = out_dir / f"{ts}_{cam.camera_uid}_raw.jpg"
            raw_p.write_bytes(jpeg)
            arr = np.frombuffer(jpeg, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is not None:
                det = detect_charuco_corners(bgr, cfg)
                h, w = bgr.shape[:2]
                saved.append(
                    f"{cam.camera_uid}: {w}x{h}, {det.num_corners} corners, {len(det.marker_ids)} markers"
                )
        self.log.append(
            f"[DEBUG] Saved {len(saved)} raw frame(s) to {out_dir} ({'; '.join(saved)})"
        )

    def shutdown_voice(self) -> None:
        self.voice_coach.stop()

    def _set_auto_capture(self, enabled: bool, user_triggered: bool = True) -> None:
        self._auto_capture_enabled = bool(enabled)
        if user_triggered:
            self._user_paused_auto_capture = not self._auto_capture_enabled
        elif self._auto_capture_enabled:
            self._user_paused_auto_capture = False
        self._steady_ticks = 0
        self.hold_progress_bar.setValue(0)
        self.voice_coach.reset_pacing()
        if self._auto_capture_enabled:
            self.btn_auto_coach.setText("⏸  Pause Guided Auto-Capture")
            self.btn_auto_coach.setObjectName("DangerButton")
            pose_idx = self.guide.canvas.pose_index
            step = CALIBRATION_POSE_STEPS[pose_idx]
            self.lbl_coach_headline.setText(
                f"LIVE COACH ACTIVE — Show board for {step.short_label}"
            )
            if self.chk_voice_coach.isChecked():
                play_coach_sound("tick")
                self.voice_coach.announce(
                    f"Guided calibration started. {pose_spoken_instruction(pose_idx, self._active_board_short_name())}",
                    priority=True,
                )
            src_lbl = "started" if user_triggered else "auto-armed (2+ cameras ready)"
            self.log.append(
                f"[COACH] Hands-free voice-guided calibration {src_lbl} at {step.short_label}."
            )
        else:
            self.btn_auto_coach.setText("▶  Start Guided Auto-Capture")
            self.btn_auto_coach.setObjectName("PrimaryButton")
            if user_triggered and self.chk_voice_coach.isChecked():
                self.voice_coach.announce("Guided calibration paused.", priority=True)
            if user_triggered:
                self.log.append("[COACH] Hands-free auto-capture paused.")
        self.btn_auto_coach.style().unpolish(self.btn_auto_coach)
        self.btn_auto_coach.style().polish(self.btn_auto_coach)

    def _refresh_camera_tiles(self) -> None:
        cams = self.ctrl.camera_manager.list_cameras()
        online_count = len(self.ctrl.camera_manager.get_active_recording_cameras())
        self.guide.set_num_cameras(max(2, online_count or len(cams)))

        # Never destroy & recreate live video widgets unless the set of cameras actually changed!
        new_ids = [c.device_id for c in cams]
        if new_ids == self._tile_device_ids and (cams or self.live_cam_grid.count() > 0):
            return
        self._tile_device_ids = new_ids

        while self.live_cam_grid.count():
            item = self.live_cam_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._live_cam_img_labels.clear()
        self._live_cam_status_labels.clear()

        if not cams:
            empty = QLabel(
                "No phones connected yet.\n\n"
                "• Connect 2 or more phones in '2. Cameras & Pair'\n"
                "• Or click 'From Folder...' to calibrate from saved images."
            )
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setObjectName("Subtitle")
            self.live_cam_grid.addWidget(empty, 0, 0)
            return

        cols = 2 if len(cams) <= 4 else 3
        for idx, cam in enumerate(cams):
            tile = QFrame()
            tile.setObjectName("Card")
            tl = QVBoxLayout(tile)
            tl.setContentsMargins(6, 6, 6, 6)
            tl.setSpacing(4)

            img_lbl = QLabel(f"{cam.camera_uid} · Waiting for stream...")
            img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_lbl.setMinimumSize(220, 150)
            img_lbl.setStyleSheet("background: #080C12; border-radius: 6px; color: #8EA0B5;")
            tl.addWidget(img_lbl, 1)

            st_lbl = QLabel(f"{cam.camera_uid} ({cam.name}) · Waiting for board...")
            st_lbl.setObjectName("BadgeWarn")
            tl.addWidget(st_lbl)

            self.live_cam_grid.addWidget(tile, idx // cols, idx % cols)
            self._live_cam_img_labels[cam.device_id] = img_lbl
            self._live_cam_status_labels[cam.device_id] = st_lbl

    def _refresh_live_charuco_feeds(self) -> None:
        if not self.isVisible() or self.tabs.currentIndex() != 2:
            return
        cfg = self._board_cfg()
        pose_idx = self.guide.canvas.pose_index
        now = time.monotonic()
        in_cooldown = now < self._cooldown_until
        flash_txt = self._flash_text if now < self._flash_until else ""
        hold_prog = (
            float(self._steady_ticks) / float(max(self._steady_ticks_target, 1))
            if self._auto_capture_enabled
            else 0.0
        )

        feedbacks: List[CameraCoachFeedback] = []
        live_snap: Dict[str, Tuple[ChArUcoDetection, Tuple[int, int], bytes]] = {}
        self._mismatch_tick += 1
        check_mismatch_now = self._mismatch_tick % 3 == 0
        alt_detected: str = ""

        for did, img_lbl in list(self._live_cam_img_labels.items()):
            cam = self.ctrl.camera_manager.get_camera(did)
            st_lbl = self._live_cam_status_labels.get(did)
            if cam is None or not cam.latest_preview_jpeg:
                continue
            jpeg_bytes = bytes(cam.latest_preview_jpeg)
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            h_img, w_img = bgr.shape[:2]
            det = detect_charuco_corners(bgr, cfg)
            live_snap[cam.camera_uid] = (det, (w_img, h_img), jpeg_bytes)
            prev_c = self._prev_board_centers.get(cam.camera_uid)
            fb = analyze_camera_detection(
                cam.camera_uid,
                det,
                w_img,
                h_img,
                pose_index=pose_idx,
                prev_center_norm=prev_c,
            )
            if fb.board_center_norm is not None:
                self._prev_board_centers[cam.camera_uid] = fb.board_center_norm
            else:
                self._prev_board_centers.pop(cam.camera_uid, None)
            feedbacks.append(fb)

            # Check if the user accidentally printed A3 while A4 is selected (or vice versa!)
            if check_mismatch_now and not alt_detected and det.num_corners < 6:
                alt = detect_wrong_board_preset(bgr, cfg, det.num_corners)
                if alt:
                    alt_detected = alt

            vis = draw_live_calibration_hud(
                bgr,
                det,
                fb,
                pose_index=pose_idx,
                hold_progress=hold_prog,
                flash_text=flash_txt,
            )
            ok_enc, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok_enc:
                qimg = QImage.fromData(buf.tobytes())
                if not qimg.isNull():
                    img_lbl.setPixmap(
                        QPixmap.fromImage(qimg).scaled(
                            img_lbl.size(), Qt.AspectRatioMode.KeepAspectRatio
                        )
                    )
            if st_lbl is not None:
                st_lbl.setText(
                    f"{'✓ ' if fb.status == 'ready' else ''}{cam.camera_uid} · {fb.short_hint}"
                )
                st_lbl.setObjectName("BadgeOk" if fb.status == "ready" else "BadgeWarn")
                st_lbl.style().unpolish(st_lbl)
                st_lbl.style().polish(st_lbl)

        if alt_detected:
            # If no poses are captured yet, auto-switch seamlessly to the detected A3/A4 board!
            if len(self._captured_frames) == 0:
                if alt_detected == "A3":
                    self._apply_a3_preset()
                elif alt_detected == "A4":
                    self._apply_a4_preset()
                self.ctrl.status_message.emit(
                    f"Auto-detected your printed {alt_detected} ChArUco board!"
                )
                if self._auto_capture_enabled and self.chk_voice_coach.isChecked():
                    self.voice_coach.announce(
                        f"Detected your {alt_detected} board! Switched automatically.",
                        priority=True,
                    )
            else:
                self._suggested_alt_board = alt_detected
                lbl_alt = "A3 (9×6 · 40mm)" if alt_detected == "A3" else "A4 (7×5 · 35mm)"
                self.btn_board_mismatch.setText(f"💡 Detected {lbl_alt}! Switch →")
                self.btn_board_mismatch.setVisible(True)

        if not feedbacks:
            return

        # Automatically arm hands-free auto-capture as soon as 2+ live cameras are streaming on Step 3
        if (
            len(feedbacks) >= 2
            and not self._auto_capture_enabled
            and not self._user_paused_auto_capture
            and not self._solve_in_progress
            and not in_cooldown
            and len(self._completed_pose_indices) < len(CALIBRATION_POSE_STEPS)
        ):
            self._set_auto_capture(True, user_triggered=False)

        decision = evaluate_multi_cam_coaching(feedbacks, pose_index=pose_idx)

        if self._last_banner_notice.startswith("✓ CALIBRATION SOLVED & SAVED"):
            self.lbl_coach_headline.setText(self._last_banner_notice)
            self.hold_progress_bar.setValue(100)
            return

        if in_cooldown:
            # Keep post-capture celebration & next-pose instruction clearly visible while user repositions!
            rem = max(0.0, self._cooldown_until - now)
            self.lbl_coach_headline.setText(
                f"{self._last_banner_notice}  (next pose arms in {rem:.1f}s)"
            )
            self.lbl_coach_headline.setStyleSheet(
                "font-size: 16px; font-weight: 800; color: #16A34A;"
            )
            self.hold_progress_bar.setValue(0)
            return

        # Live directional coaching banner update (never say "CAPTURING POSE..." when user explicitly paused!)
        if decision.all_ready_to_capture and not self._auto_capture_enabled:
            self.lbl_coach_headline.setText(
                "✓ BOTH CAMERAS LOCKED — Click 'Capture Now' or 'Start Guided Auto-Capture'"
            )
            self.lbl_coach_headline.setStyleSheet(
                "font-size: 16px; font-weight: 800; color: #16A34A;"
            )
        else:
            self.lbl_coach_headline.setText(decision.headline)
            if decision.severity == "ok":
                self.lbl_coach_headline.setStyleSheet(
                    "font-size: 16px; font-weight: 800; color: #16A34A;"
                )
            elif decision.severity == "guide":
                self.lbl_coach_headline.setStyleSheet(
                    "font-size: 16px; font-weight: 800; color: #D97706;"
                )
            else:
                self.lbl_coach_headline.setStyleSheet("font-size: 16px; font-weight: 800;")
        self.lbl_live_pose_desc.setText(
            f"{CALIBRATION_POSE_STEPS[pose_idx].where_to_place}  ·  {decision.detail}"
        )

        # Hands-free continuous voice coaching + auto-capture lock-on counter
        if self._auto_capture_enabled:
            if self.chk_voice_coach.isChecked() and decision.voice_prompt:
                # Keep talking continuously every ~2.6s (or faster on status change) until user nails the pose!
                self.voice_coach.coach(
                    decision.voice_key,
                    decision.voice_prompt,
                    now=now,
                    repeat_interval_s=2.6,
                    change_interval_s=0.95,
                )

            if decision.all_ready_to_capture:
                self._steady_ticks += 1
                pct = int(
                    min(100, round(100.0 * self._steady_ticks / max(self._steady_ticks_target, 1)))
                )
                self.hold_progress_bar.setValue(pct)
                self.hold_progress_bar.setFormat(
                    f"BOTH CAMERAS LOCKED — Auto-Capturing & Saving ({pct}%)..."
                )
                if (
                    self._steady_ticks < self._steady_ticks_target
                    and self.chk_voice_coach.isChecked()
                ):
                    play_coach_sound("lock_tick", step_index=self._steady_ticks)
                if self._steady_ticks >= self._steady_ticks_target:
                    self._steady_ticks = 0
                    self.hold_progress_bar.setValue(100)
                    self.hold_progress_bar.setFormat("✓ AUTO-CAPTURED & SAVED!")
                    self._capture_live_pose(auto_triggered=True, preverified_snap=live_snap)
            else:
                # Don't drop lock-on progress on a single brief 'moving' flicker while both cameras still see the board!
                only_jitter = decision.visible_count >= 2 and all(
                    f.status in ("ready", "moving") for f in feedbacks if f.status != "missing"
                )
                if not only_jitter:
                    self._steady_ticks = max(0, self._steady_ticks - 1)
                pct = int(round(100.0 * self._steady_ticks / max(self._steady_ticks_target, 1)))
                self.hold_progress_bar.setValue(pct)
                self.hold_progress_bar.setFormat(
                    "Hold steady — locking..." if pct > 0 else "Waiting for both cameras to lock..."
                )
        else:
            self.hold_progress_bar.setValue(100 if decision.all_ready_to_capture else 0)
            self.hold_progress_bar.setFormat(
                "BOTH LOCKED (Paused — click Start Guided Auto-Capture or Capture Now)"
                if decision.all_ready_to_capture
                else "Auto-Capture paused"
            )

    def _start_countdown_capture(self) -> None:
        if self._countdown_timer.isActive():
            return
        self._countdown_left = 3
        self.btn_cap_timer.setText("⏱ 3...")
        if self.chk_voice_coach.isChecked():
            play_coach_sound("tick")
        self._countdown_timer.start(1000)

    def _on_countdown_tick(self) -> None:
        self._countdown_left -= 1
        if self._countdown_left > 0:
            self.btn_cap_timer.setText(f"⏱ {self._countdown_left}...")
            if self.chk_voice_coach.isChecked():
                play_coach_sound("tick")
        else:
            self._countdown_timer.stop()
            self.btn_cap_timer.setText("⏱ 3s Timer")
            self._capture_live_pose(auto_triggered=False)

    def _capture_live_pose(
        self,
        auto_triggered: bool = False,
        preverified_snap: Optional[
            Dict[str, Tuple[ChArUcoDetection, Tuple[int, int], bytes]]
        ] = None,
    ) -> None:
        cfg = self._board_cfg()
        pose_idx = self.guide.canvas.pose_index
        snap: Dict[str, Tuple[ChArUcoDetection, Tuple[int, int], bytes]] = {}
        summary_parts: List[str] = []
        valid_cams = 0

        # Use the already-locked sharp snapshot from live detection when available (eliminates re-detect drop race!)
        if preverified_snap:
            valid_pre = {
                uid: val for uid, val in preverified_snap.items() if val[0].num_corners >= 6
            }
            if len(valid_pre) >= 2:
                snap = dict(valid_pre)
                for uid in sorted(snap.keys()):
                    det, _sz, _b = snap[uid]
                    summary_parts.append(f"{uid}:{det.num_corners}c")
                    valid_cams += 1

        if valid_cams < 2:
            cams = [c for c in self.ctrl.camera_manager.list_cameras() if c.latest_preview_jpeg]
            if len(cams) < 2:
                if not auto_triggered:
                    QMessageBox.information(
                        self,
                        "Need >= 2 Streaming Cameras",
                        f"Currently {len(cams)} phone(s) are streaming live preview frames.\n"
                        "Open Nuranova Mocap Camera on at least 2 connected phones so both streams appear here, "
                        "or use 'From Folder...'.",
                    )
                return

            snap = {}
            summary_parts = []
            valid_cams = 0
            for cam in cams:
                jpeg = cam.latest_preview_jpeg
                if not jpeg:
                    continue
                arr = np.frombuffer(jpeg, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                h_img, w_img = bgr.shape[:2]
                det = detect_charuco_corners(bgr, cfg)
                snap[cam.camera_uid] = (det, (w_img, h_img), bytes(jpeg))
                summary_parts.append(f"{cam.camera_uid}:{det.num_corners}c")
                if det.num_corners >= 6:
                    valid_cams += 1

        if valid_cams < 2:
            self.log.append(
                f"[RETRY] Only {valid_cams} camera(s) saw >= 6 corners ({', '.join(summary_parts)}). "
                "Check 'SHOW THIS BOARD' preset (A4 7×5 vs A3 9×6) and angle board toward all cameras!"
            )
            return

        self._captured_frames.append((pose_idx, snap))
        self._completed_pose_indices.add(pose_idx)
        n_cap = len(self._captured_frames)
        step = CALIBRATION_POSE_STEPS[pose_idx]

        # Immediately persist every captured multi-camera pose JPEG pair to disk (even without an open project!)
        saved_dir_str = ""
        try:
            sess_dir = self._ensure_live_capture_session_dir()
            saved_files: Dict[str, str] = {}
            for uid, (_det, _sz, jpeg_b) in snap.items():
                cam_dir = sess_dir / uid
                cam_dir.mkdir(parents=True, exist_ok=True)
                img_path = cam_dir / f"pose_{pose_idx + 1:02d}_shot_{n_cap:02d}.jpg"
                img_path.write_bytes(jpeg_b)
                saved_files[uid] = str(img_path)
            manifest_path = sess_dir / "captures_manifest.json"
            manifest = {
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "board": {
                    "squares_x": cfg.squares_x,
                    "squares_y": cfg.squares_y,
                    "square_length_mm": cfg.square_length_mm,
                    "marker_length_mm": cfg.marker_length_mm,
                    "measured_square_length_mm": cfg.measured_square_length_mm,
                    "dictionary": cfg.dictionary_name,
                },
                "completed_pose_indices": sorted(self._completed_pose_indices),
                "total_shots": n_cap,
                "latest_shot": {
                    "pose_index": pose_idx,
                    "pose_label": step.short_label,
                    "summary": summary_parts,
                    "files": saved_files,
                },
            }
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            saved_dir_str = f" → Saved to {sess_dir}"
        except Exception as exc:
            self.log.append(f"[WARN] Could not persist snapshot JPEGs: {exc}")

        self.log.append(
            f"[✓ CAPTURED & SAVED #{n_cap}] {step.short_label} ({', '.join(summary_parts)}){saved_dir_str}"
        )
        self._update_captured_badge()

        # Pick next uncaptured pose (or wrap around)
        next_pose_idx = next(
            (
                i
                for i in range(len(CALIBRATION_POSE_STEPS))
                if i not in self._completed_pose_indices
            ),
            None,
        )

        now = time.monotonic()
        self._flash_until = now + 1.15
        self._flash_text = f"SAVED: {step.short_label.upper()}!"

        if next_pose_idx is None:
            # All 6 guided poses completed! Celebrate + notify + ALWAYS auto-solve & auto-save!
            self._last_banner_notice = (
                "🎉 ALL 6 POSES CAPTURED & SAVED! Auto-Solving Calibration + Floor Z-Up..."
            )
            self.lbl_coach_headline.setText(self._last_banner_notice)
            self.lbl_coach_headline.setStyleSheet(
                "font-size: 16px; font-weight: 800; color: #16A34A;"
            )
            self._cooldown_until = now + 1.5
            if self.chk_voice_coach.isChecked():
                play_coach_sound("complete")
                self.voice_coach.announce(
                    "Awesome! All 6 poses captured and saved! Solving multi-camera calibration now, please wait.",
                    priority=True,
                )
            self.ctrl.status_message.emit(
                "All 6 calibration poses captured & saved! Auto-solving calibration..."
            )
            if self._auto_capture_enabled:
                self._auto_capture_enabled = False
                self.btn_auto_coach.setText("▶  Start Guided Auto-Capture")
                self.btn_auto_coach.setObjectName("PrimaryButton")
                self.btn_auto_coach.style().unpolish(self.btn_auto_coach)
                self.btn_auto_coach.style().polish(self.btn_auto_coach)
            if not self._solve_in_progress:
                QTimer.singleShot(300, self._solve_live_calibration)
        else:
            nxt_step = CALIBRATION_POSE_STEPS[next_pose_idx]
            self._last_banner_notice = f"✓ {step.short_label} SAVED!  →  GO TO {nxt_step.title}"
            self._cooldown_until = now + 3.0
            self.guide.set_pose_index(next_pose_idx)
            self.lbl_coach_headline.setText(self._last_banner_notice)
            self.lbl_coach_headline.setStyleSheet(
                "font-size: 16px; font-weight: 800; color: #16A34A;"
            )
            self.ctrl.status_message.emit(self._last_banner_notice)
            if self.chk_voice_coach.isChecked():
                play_coach_sound("capture")
                self.voice_coach.announce(
                    f"Great! Pose {pose_idx + 1} captured and saved! Now move to {pose_spoken_instruction(next_pose_idx, self._active_board_short_name())}",
                    priority=True,
                )

        self._refresh_pose_stepper_pills()

    def _clear_captured_poses(self) -> None:
        self._captured_frames.clear()
        self._completed_pose_indices.clear()
        self._last_banner_notice = ""
        self._capture_session_dir = None
        self._user_paused_auto_capture = False
        self._solve_in_progress = False
        self._steady_ticks = 0
        self._cooldown_until = 0.0
        self._flash_until = 0.0
        self.hold_progress_bar.setValue(0)
        self._update_captured_badge()
        self.guide.set_pose_index(0)
        self._refresh_pose_stepper_pills()
        self.lbl_coach_headline.setText("Reset complete — Ready for Pose 1 (Center Upright)")
        self.log.append(
            "[RESET] Cleared captured poses. Ready to start at Pose 1 (Center Upright)."
        )
        self._maybe_auto_arm_capture()

    def _update_captured_badge(self) -> None:
        n_unique = len(self._completed_pose_indices)
        n_cap = len(self._captured_frames)
        self.lbl_captured_count.setText(f"Captured: {n_unique}/6 ({n_cap} shots)")
        self.lbl_captured_count.setObjectName(
            "BadgeOk" if n_unique >= 6 else ("BadgeWarn" if n_cap >= 3 else "BadgeErr")
        )
        self.lbl_captured_count.style().unpolish(self.lbl_captured_count)
        self.lbl_captured_count.style().polish(self.lbl_captured_count)
        self.btn_solve_live.setEnabled(n_cap >= 3)

    def _solve_live_calibration(self) -> None:
        if self._solve_in_progress:
            return
        if len(self._captured_frames) < 3:
            QMessageBox.information(
                self,
                "Need More Poses",
                "Capture at least 3 multi-camera poses (6+ varied poses recommended) before solving calibration.",
            )
            return

        self._solve_in_progress = True
        cfg = self._board_cfg()
        try:
            # Only include cameras that actually saw >= 6 corners in >= 3 poses (never let an idle 3rd webcam fail calibration!)
            cam_valid_counts: Dict[str, int] = {}
            for _, snap in self._captured_frames:
                for uid, (det, _sz, _b) in snap.items():
                    if det.num_corners >= 6:
                        cam_valid_counts[uid] = cam_valid_counts.get(uid, 0) + 1
            all_uids = sorted(uid for uid, cnt in cam_valid_counts.items() if cnt >= 3)
            if len(all_uids) < 2:
                raise ValueError(
                    f"Need at least 2 cameras with >= 3 valid board captures each (current valid views: {cam_valid_counts})."
                )
            per_cam_dets: Dict[str, List[ChArUcoDetection]] = {u: [] for u in all_uids}
            per_cam_size: Dict[str, Tuple[int, int]] = {}
            synced: List[Dict[str, ChArUcoDetection]] = []

            # Always save synchronized snapshot JPEGs into session folder (active project OR global calibrations/)
            save_dir = self._ensure_live_capture_session_dir()

            for f_idx, (_pose_idx, snap) in enumerate(self._captured_frames):
                frame_map: Dict[str, ChArUcoDetection] = {}
                for uid in all_uids:
                    if uid in snap:
                        det, img_sz, jpeg_b = snap[uid]
                        per_cam_dets[uid].append(det)
                        per_cam_size[uid] = img_sz
                        frame_map[uid] = det
                        cd = save_dir / uid
                        cd.mkdir(parents=True, exist_ok=True)
                        (cd / f"pose_{f_idx:03d}.jpg").write_bytes(jpeg_b)
                    else:
                        empty_det = ChArUcoDetection(
                            corner_ids=np.zeros((0,), dtype=np.int32),
                            corners_2d=np.zeros((0, 2), dtype=np.float64),
                            marker_ids=np.zeros((0,), dtype=np.int32),
                            marker_corners=[],
                        )
                        per_cam_dets[uid].append(empty_det)
                        frame_map[uid] = empty_det
                synced.append(frame_map)

            intr_map: Dict[str, CameraIntrinsics] = {}
            for uid in all_uids:
                intr_map[uid] = calibrate_camera_intrinsics(
                    uid, per_cam_dets[uid], cfg, per_cam_size[uid]
                )
                self.log.append(
                    f"Intrinsic {uid}: RMSE = {intr_map[uid].reprojection_error_px:.3f} px (preview res)"
                )

            extr_init, board_poses = initialize_extrinsics_from_board_frames(intr_map, synced, cfg)
            extr_opt, opt_boards, rep = run_board_bundle_adjustment(
                intr_map, extr_init, synced, board_poses, cfg
            )
            self._last_board_poses = opt_boards

            # Scale intrinsics cleanly from preview JPEG size to each camera's configured recording resolution
            cam_by_uid = {c.camera_uid: c for c in self.ctrl.camera_manager.list_cameras()}
            scaled_cams: Dict[str, CalibratedCamera] = {}
            for uid in all_uids:
                cam_obj = cam_by_uid.get(uid)
                intr_final = intr_map[uid]
                if cam_obj is not None:
                    intr_final = _scale_intrinsics_to_resolution(
                        intr_final, cam_obj.config.width, cam_obj.config.height
                    )
                scaled_cams[uid] = CalibratedCamera(intr_final, extr_opt[uid])

            calib = MultiCameraCalibration(
                profile_name=f"LiveCalib_{len(all_uids)}cams",
                cameras=scaled_cams,
                global_reprojection_rmse_px=rep.global_reprojection_rmse_px,
                notes="Solved from live guided multi-camera ChArUco snapshots.",
            )

            # Auto-align world to Blender Z-Up using Pose 1 (Center Upright) — or first valid board pose!
            upright_idx = next(
                (
                    i
                    for i, (p_idx, _) in enumerate(self._captured_frames)
                    if p_idx == 0 and i < len(opt_boards) and opt_boards[i] is not None
                ),
                next((i for i, bp in enumerate(opt_boards) if bp is not None), None),
            )
            if upright_idx is not None and opt_boards[upright_idx] is not None:
                Rb, tb = opt_boards[upright_idx]
                calib = align_calibration_from_upright_board(
                    calib,
                    R_wb=Rb,
                    t_wb=tb,
                    board_config=cfg,
                    board_center_height_m=1.15,
                )
                self.log.append(
                    "[OK] Auto-aligned Studio Vertical Z-Up from Pose 1 (Center Upright)!"
                )

            self.ctrl.set_active_calibration(calib)
            # Always also archive a timestamped calibration JSON so nothing is ever overwritten or lost!
            ts = time.strftime("%Y%m%d_%H%M%S")
            archive_dir = self.ctrl._last_calib_path.parent / "calibrations"
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive_path = archive_dir / f"{calib.profile_name}_{ts}.json"
            calib.save(archive_path)

            # Switch automatically from 1-Cam AI mode to Multi-Camera 3D mode when 2+ cameras are calibrated
            if len(calib.cameras) >= 2 and self.ctrl.settings.general.single_camera_mode:
                self.ctrl.set_single_camera_mode(False)
                self.log.append(
                    f"[MODE] Activated Multi-Camera 3D Mode ({len(calib.cameras)} calibrated cameras)."
                )

            if self.ctrl.project_manager.active_project:
                out_p = (
                    self.ctrl.project_manager.active_project.subdir("calibration")
                    / f"{calib.profile_name}.json"
                )
                self.log.append(
                    f"[SAVED] Calibration saved to {out_p}, {archive_path}, and global startup profile."
                )
            else:
                self.log.append(
                    f"[SAVED] Calibration saved to {archive_path} & {self.ctrl._last_calib_path} (auto-loads on startup)."
                )
            self._update_quality_badge(rep.global_reprojection_rmse_px, rep.num_cameras)
            self._last_banner_notice = f"✓ CALIBRATION SOLVED & SAVED! (RMSE {rep.global_reprojection_rmse_px:.2f} px · {rep.num_cameras} cams)"
            self.lbl_coach_headline.setText(self._last_banner_notice)
            self.lbl_coach_headline.setStyleSheet(
                "font-size: 16px; font-weight: 800; color: #16A34A;"
            )
            self.ctrl.status_message.emit(self._last_banner_notice)
            self.log.append(
                f"[DONE] Multi-Camera Calibration complete! Global RMSE: {rep.global_reprojection_rmse_px:.3f} px "
                f"across {rep.num_cameras} cameras ({len(self._captured_frames)} poses)."
            )
            if self.chk_voice_coach.isChecked():
                qual = (
                    "Excellent accuracy"
                    if rep.global_reprojection_rmse_px <= 1.0
                    else (
                        "Acceptable accuracy"
                        if rep.global_reprojection_rmse_px <= 2.0
                        else "High error, consider re-calibrating"
                    )
                )
                self.voice_coach.announce(
                    f"Calibration complete and saved! {qual}, error {rep.global_reprojection_rmse_px:.1f} pixels. You are ready for motion capture!",
                    priority=True,
                )
        except Exception as exc:
            if self.chk_voice_coach.isChecked():
                self.voice_coach.announce(
                    "Calibration could not solve. Please check the monitor for details.",
                    priority=True,
                )
            QMessageBox.warning(self, "Live Calibration Error", str(exc))
            self.log.append(f"[ERROR] {exc}")
        finally:
            self._solve_in_progress = False

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
        out, _ = QFileDialog.getSaveFileName(
            self, "Save Printable ChArUco PDF", "Nuranova_ChArUco_Board.pdf", "PDF (*.pdf)"
        )
        if not out:
            return
        try:
            p = generate_printable_charuco_pdf(
                self._board_cfg(), out, paper_size=self.combo_paper.currentText()
            )
            self.log.append(f"[OK] Saved physical-scale ChArUco PDF: {p}")
            QMessageBox.information(self, "Board Generated", f"Saved printable ChArUco PDF:\n{p}")
        except Exception as exc:
            QMessageBox.warning(self, "Board Error", str(exc))

    def _gen_png(self) -> None:
        out, _ = QFileDialog.getSaveFileName(
            self, "Save ChArUco PNG", "Nuranova_ChArUco_Board.png", "PNG (*.png)"
        )
        if not out:
            return
        p = render_charuco_png(self._board_cfg(), out)
        self.log.append(f"[OK] Exported ChArUco PNG: {p}")

    def _load_existing(self) -> None:
        fp, _ = QFileDialog.getOpenFileName(
            self, "Load Calibration Profile", "", "Calibration JSON (*.json)"
        )
        if not fp:
            return
        calib = MultiCameraCalibration.load(fp)
        self.ctrl.set_active_calibration(calib)
        self._update_quality_badge(calib.global_reprojection_rmse_px, len(calib.cameras))
        self.log.append(
            f"[LOADED] {calib.profile_name} ({len(calib.cameras)} cams, RMSE {calib.global_reprojection_rmse_px:.3f} px)"
        )

    def _update_quality_badge(self, rmse: float, n_cams: int) -> None:
        if rmse <= 1.0:
            self.lbl_quality.setText(
                f"✓ PASS — {n_cams} Cameras · Global RMSE: {rmse:.3f} px (target <= 1.0 px)"
            )
            self.lbl_quality.setObjectName("BadgeOk")
        elif rmse <= 2.0:
            self.lbl_quality.setText(
                f"! ACCEPTABLE — {n_cams} Cameras · Global RMSE: {rmse:.3f} px"
            )
            self.lbl_quality.setObjectName("BadgeWarn")
        else:
            self.lbl_quality.setText(
                f"✗ HIGH ERROR — {n_cams} Cameras · Global RMSE: {rmse:.3f} px (re-capture board sweep)"
            )
            self.lbl_quality.setObjectName("BadgeErr")
        self.lbl_quality.style().unpolish(self.lbl_quality)
        self.lbl_quality.style().polish(self.lbl_quality)

    def _run_folder_calibration(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select Calibration Images Folder (subfolders cam_01, cam_02, ...)"
        )
        if not folder:
            return
        cfg = self._board_cfg()
        cam_dirs = find_camera_image_subfolders(Path(folder))
        if len(cam_dirs) < 2:
            QMessageBox.warning(
                self,
                "Calibration Folder",
                "Expected >= 2 camera subfolders containing PNG/JPG ChArUco images (e.g. cam_01/, cam_02/).\n\n"
                "Tip: With phones connected online, you can calibrate directly using '📸 Capture Pose Now' without creating folders!",
            )
            return
        try:
            intr_map = {}
            per_cam_dets: Dict[str, list] = {}
            for cd in cam_dirs:
                imgs = sorted(
                    list(cd.glob("*.png")) + list(cd.glob("*.jpg")) + list(cd.glob("*.jpeg"))
                )
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
                self.log.append(
                    f"Intrinsic {cd.name}: RMSE = {intr_map[cd.name].reprojection_error_px:.3f} px"
                )

            n_frames = min(len(v) for v in per_cam_dets.values())
            synced = [{uid: per_cam_dets[uid][f] for uid in per_cam_dets} for f in range(n_frames)]
            extr_init, board_poses = initialize_extrinsics_from_board_frames(intr_map, synced, cfg)
            extr_opt, opt_boards, rep = run_board_bundle_adjustment(
                intr_map, extr_init, synced, board_poses, cfg
            )
            self._last_board_poses = opt_boards
            calib = MultiCameraCalibration(
                profile_name=f"Calib_{Path(folder).name}",
                cameras={u: CalibratedCamera(intr_map[u], extr_opt[u]) for u in intr_map},
                global_reprojection_rmse_px=rep.global_reprojection_rmse_px,
            )
            self.ctrl.active_calibration = calib
            if self.ctrl.project_manager.active_project:
                out_p = (
                    self.ctrl.project_manager.active_project.subdir("calibration")
                    / f"{calib.profile_name}.json"
                )
                calib.save(out_p)
                self.log.append(f"[SAVED] Calibration profile saved to {out_p}")
            self.ctrl.calibration_changed.emit()
            self._update_quality_badge(rep.global_reprojection_rmse_px, rep.num_cameras)
            self.log.append(
                f"[DONE] Global Bundle Adjustment RMSE: {rep.global_reprojection_rmse_px:.3f} px across {rep.num_cameras} cameras."
            )
        except Exception as exc:
            QMessageBox.warning(self, "Calibration Failed", str(exc))
            self.log.append(f"[ERROR] {exc}")

    def _align_floor_z_up(self) -> None:
        if self.ctrl.active_calibration is None:
            QMessageBox.information(
                self, "No Calibration", "Run or load a multi-camera calibration first."
            )
            return
        if not self._last_board_poses or self._last_board_poses[0] is None:
            QMessageBox.information(
                self,
                "Z-Up Reference",
                "Run calibration first (Pose 1 upright or Folder Frame 0 flat on floor).",
            )
            return
        cfg = self._board_cfg()
        Rb, tb = self._last_board_poses[0]
        if self._captured_frames:
            aligned = align_calibration_from_upright_board(
                self.ctrl.active_calibration,
                R_wb=Rb,
                t_wb=tb,
                board_config=cfg,
                board_center_height_m=1.15,
            )
        else:
            corners_w = (cfg.get_chessboard_corners_3d() @ Rb.T) + tb
            aligned = align_calibration_to_world_z_up(
                self.ctrl.active_calibration,
                floor_points_world=corners_w,
                origin_world=np.mean(corners_w, axis=0),
                forward_hint_world=Rb[:, 1],
                camera_up_hint=True,
            )
        self.ctrl.set_active_calibration(aligned)
        self.log.append("[OK] Aligned calibration to Blender Z-Up & saved.")


class CalibrationProfilesPage(QWidget):
    """Page 7: Saved Calibration Profiles Browser & Validity Inspector."""

    def __init__(self, ctrl: StudioController) -> None:
        super().__init__()
        self.ctrl = ctrl
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(10)
        root.addLayout(
            _header(
                "Calibration Profiles",
                "Inspect, load, save, and verify reusable multi-camera calibration profiles",
            )
        )

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
        self.table.setHorizontalHeaderLabels(
            [
                "Camera UID",
                "Resolution",
                "fx / fy",
                "Intrinsic RMSE (px)",
                "Extrinsic RMSE (px)",
                "Camera Center (m)",
            ]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        root.addWidget(self.table, 1)

        self.lbl_summary = QLabel("No active calibration loaded.")
        root.addWidget(self.lbl_summary)

        ctrl.calibration_changed.connect(self.refresh)
        self.refresh()

    def _load_file(self) -> None:
        fp, _ = QFileDialog.getOpenFileName(
            self, "Load Calibration Profile", "", "Calibration JSON (*.json)"
        )
        if fp:
            self.ctrl.active_calibration = MultiCameraCalibration.load(fp)
            self.ctrl.calibration_changed.emit()

    def _save_file(self) -> None:
        if not self.ctrl.active_calibration:
            return
        fp, _ = QFileDialog.getSaveFileName(
            self,
            "Save Calibration Profile",
            "calibration_profile.json",
            "Calibration JSON (*.json)",
        )
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
        wand = (
            f" | Known-Distance Err: {cal.known_distance_error_mm:.2f} mm"
            if cal.known_distance_error_mm is not None
            else ""
        )
        self.lbl_summary.setText(
            f"Profile: {cal.profile_name} | Global RMSE: {cal.global_reprojection_rmse_px:.3f} px | Up Axis: {cal.world_up_axis}{wand}"
        )
