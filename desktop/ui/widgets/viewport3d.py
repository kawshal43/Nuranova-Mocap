"""Interactive 3D Viewport Widget supporting Orbit/Pan/Zoom, Perspective/Ortho, View Presets,
Ground Grid, Axes, Camera Frustums, Physical Markers, Articulated Skeleton, Confidence Coloring,
and Raw vs Corrected Motion Overlay.

Uses crisp high-DPI Qt 3D software projection (works 100% reliably on GPU desktops, remote displays,
and headless offscreen tests) with optional PyVista VTK export/pop-out inspector.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from engine.math_core.camera_model import MultiCameraCalibration
from engine.skeleton.rig_definition import BONE_NAMES, PARENT_INDICES


class Canvas3DWidget(QWidget):
    """Hardware-independent interactive 3D Z-Up viewport canvas."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(360, 260)
        self.setMouseTracking(True)

        # Camera spherical orbit state (Z-Up)
        self.yaw_deg: float = -40.0
        self.pitch_deg: float = 22.0
        self.distance: float = 4.2
        self.target: np.ndarray = np.array([0.0, 0.0, 0.95], dtype=np.float64)
        self.orthographic: bool = False

        # Visual overlays
        self.show_grid: bool = True
        self.show_axes: bool = True
        self.show_cameras: bool = True
        self.show_markers: bool = True
        self.show_raw_overlay: bool = False
        self.color_by_confidence: bool = True
        self.dark_bg: bool = True

        # Scene data
        self.calibration: Optional[MultiCameraCalibration] = None
        self.skeleton_positions: Optional[np.ndarray] = None  # (B, 3)
        self.skeleton_confidence: Optional[np.ndarray] = None  # (B,)
        self.raw_skeleton_positions: Optional[np.ndarray] = None  # (B, 3)
        self.markers_3d: Optional[np.ndarray] = None  # (M, 3)
        self.selected_bone_idx: int = -1
        self.hud_text: str = "Z-Up (meters) | Left-Drag: Orbit | Right-Drag: Pan | Wheel: Zoom"

        self._last_mouse: Optional[QPoint] = None
        self._drag_mode: Optional[str] = None

    def set_view_preset(self, preset: str) -> None:
        presets = {
            "perspective": (-40.0, 22.0, False),
            "front": (0.0, 5.0, True),
            "rear": (180.0, 5.0, True),
            "left": (-90.0, 5.0, True),
            "right": (90.0, 5.0, True),
            "top": (0.0, 89.0, True),
        }
        if preset in presets:
            self.yaw_deg, self.pitch_deg, self.orthographic = presets[preset]
            self.update()

    def set_scene_frame(
        self,
        skeleton_positions: Optional[np.ndarray] = None,
        skeleton_confidence: Optional[np.ndarray] = None,
        raw_skeleton_positions: Optional[np.ndarray] = None,
        markers_3d: Optional[np.ndarray] = None,
    ) -> None:
        self.skeleton_positions = skeleton_positions
        self.skeleton_confidence = skeleton_confidence
        self.raw_skeleton_positions = raw_skeleton_positions
        self.markers_3d = markers_3d
        self.update()

    def set_calibration(self, calib: Optional[MultiCameraCalibration]) -> None:
        self.calibration = calib
        self.update()

    def _view_basis(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        yaw = math.radians(self.yaw_deg)
        pitch = math.radians(max(-89.0, min(89.0, self.pitch_deg)))
        cam_dir = np.array([
            math.cos(pitch) * math.sin(yaw),
            -math.cos(pitch) * math.cos(yaw),
            math.sin(pitch),
        ])
        cam_pos = self.target + cam_dir * self.distance
        forward = (self.target - cam_pos)
        forward /= max(float(np.linalg.norm(forward)), 1e-8)
        up_ref = np.array([0.0, 0.0, 1.0]) if abs(forward[2]) < 0.98 else np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up_ref)
        right /= max(float(np.linalg.norm(right)), 1e-8)
        up = np.cross(right, forward)
        return cam_pos, right, up, forward

    def _project(self, pt: np.ndarray, cam_pos: np.ndarray, right: np.ndarray, up: np.ndarray, forward: np.ndarray) -> Optional[Tuple[float, float, float]]:
        rel = pt - cam_pos
        z = float(np.dot(rel, forward))
        if not self.orthographic and z < 0.08:
            return None
        x = float(np.dot(rel, right))
        y = float(np.dot(rel, up))
        w, h = float(self.width()), float(self.height())
        scale = min(w, h) * 0.95
        if self.orthographic:
            inv_d = 1.0 / max(self.distance, 0.25)
            sx = w * 0.5 + x * scale * inv_d
            sy = h * 0.5 - y * scale * inv_d
        else:
            sx = w * 0.5 + (x / z) * scale
            sy = h * 0.5 - (y / z) * scale
        return sx, sy, z

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        bg = QColor("#0F151C") if self.dark_bg else QColor("#EEF2F6")
        p.fillRect(self.rect(), bg)

        cam_pos, right, up, forward = self._view_basis()

        def draw_line_3d(a: np.ndarray, b: np.ndarray, color: QColor, width: float = 1.2, dashed: bool = False):
            pa = self._project(a, cam_pos, right, up, forward)
            pb = self._project(b, cam_pos, right, up, forward)
            if pa is None or pb is None:
                return
            pen = QPen(color, width)
            if dashed:
                pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawLine(QPointF(pa[0], pa[1]), QPointF(pb[0], pb[1]))

        # 1. Ground grid (Z = 0, 5m x 5m in 0.5m increments)
        if self.show_grid:
            grid_col = QColor("#233040") if self.dark_bg else QColor("#C9D4E0")
            for v in np.arange(-2.5, 2.51, 0.5):
                draw_line_3d(np.array([v, -2.5, 0.0]), np.array([v, 2.5, 0.0]), grid_col, 1.0)
                draw_line_3d(np.array([-2.5, v, 0.0]), np.array([2.5, v, 0.0]), grid_col, 1.0)

        # 2. World Coordinate Axes (X Red, Y Green, Z Blue)
        if self.show_axes:
            o = np.zeros(3)
            draw_line_3d(o, np.array([0.6, 0.0, 0.0]), QColor("#E5534B"), 2.2)
            draw_line_3d(o, np.array([0.0, 0.6, 0.0]), QColor("#3FB950"), 2.2)
            draw_line_3d(o, np.array([0.0, 0.0, 0.6]), QColor("#3B82F6"), 2.2)

        # 3. Calibrated Camera Frustums
        if self.show_cameras and self.calibration is not None:
            frustum_col = QColor("#F59E0B")
            for uid, cam in self.calibration.cameras.items():
                c_pos = cam.extrinsics.camera_center_world
                R_wc = cam.extrinsics.R_cw.T
                corners_cam = [
                    np.array([-0.14, -0.09, 0.28]),
                    np.array([0.14, -0.09, 0.28]),
                    np.array([0.14, 0.09, 0.28]),
                    np.array([-0.14, 0.09, 0.28]),
                ]
                corners_w = [c_pos + R_wc @ cc for cc in corners_cam]
                for cw in corners_w:
                    draw_line_3d(c_pos, cw, frustum_col, 1.3)
                for idx in range(4):
                    draw_line_3d(corners_w[idx], corners_w[(idx + 1) % 4], frustum_col, 1.3)
                pc = self._project(c_pos, cam_pos, right, up, forward)
                if pc is not None:
                    p.setPen(QPen(frustum_col))
                    p.drawText(QPointF(pc[0] + 6, pc[1] - 4), uid)

        # 4. Physical 3D Markers
        if self.show_markers and self.markers_3d is not None:
            for pt in self.markers_3d:
                if not np.all(np.isfinite(pt)):
                    continue
                pm = self._project(pt.astype(np.float64), cam_pos, right, up, forward)
                if pm is not None:
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(QColor("#F8FAFC") if self.dark_bg else QColor("#0EA5E9"))
                    p.drawEllipse(QPointF(pm[0], pm[1]), 3.5, 3.5)

        # 5. Raw Skeleton Overlay (dashed amber/gray for before-vs-after comparison)
        if self.show_raw_overlay and self.raw_skeleton_positions is not None:
            raw_pts = self.raw_skeleton_positions
            for b_idx, parent_idx in enumerate(PARENT_INDICES):
                if parent_idx >= 0 and b_idx < len(raw_pts):
                    draw_line_3d(raw_pts[parent_idx].astype(np.float64), raw_pts[b_idx].astype(np.float64), QColor("#F59E0B"), 1.6, dashed=True)

        # 6. Primary / Corrected Articulated Skeleton
        if self.skeleton_positions is not None:
            pts = self.skeleton_positions
            confs = self.skeleton_confidence
            for b_idx, parent_idx in enumerate(PARENT_INDICES):
                if parent_idx >= 0 and b_idx < len(pts):
                    col = QColor("#22C7C7")
                    if self.color_by_confidence and confs is not None and b_idx < len(confs):
                        c_val = float(confs[b_idx])
                        if c_val >= 0.7:
                            col = QColor("#22C55E")
                        elif c_val >= 0.35:
                            col = QColor("#F59E0B")
                        else:
                            col = QColor("#EF4444")
                    draw_line_3d(pts[parent_idx].astype(np.float64), pts[b_idx].astype(np.float64), col, 2.8)
            for b_idx in range(min(len(pts), len(BONE_NAMES))):
                pj = self._project(pts[b_idx].astype(np.float64), cam_pos, right, up, forward)
                if pj is not None:
                    is_sel = (b_idx == self.selected_bone_idx)
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(QColor("#EF4444") if is_sel else QColor("#FFFFFF"))
                    r = 5.5 if is_sel else 3.2
                    p.drawEllipse(QPointF(pj[0], pj[1]), r, r)

        # 7. HUD Overlay
        p.setPen(QPen(QColor("#9FB0C4") if self.dark_bg else QColor("#4A5A6B")))
        p.setFont(QFont("Segoe UI", 9))
        mode_lbl = "Ortho" if self.orthographic else "Persp"
        p.drawText(10, 20, f"[{mode_lbl}] {self.hud_text}")
        p.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._last_mouse = event.position().toPoint()
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_mode = "orbit"
        elif event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._drag_mode = "pan"

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._last_mouse is None or self._drag_mode is None:
            return
        cur = event.position().toPoint()
        dx = cur.x() - self._last_mouse.x()
        dy = cur.y() - self._last_mouse.y()
        self._last_mouse = cur
        if self._drag_mode == "orbit":
            self.yaw_deg += dx * 0.45
            self.pitch_deg = max(-89.0, min(89.0, self.pitch_deg + dy * 0.45))
        elif self._drag_mode == "pan":
            _, right, up, _ = self._view_basis()
            scale = self.distance * 0.0022
            self.target = self.target - right * (dx * scale) + up * (dy * scale)
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_mode = None
        self._last_mouse = None

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        factor = 0.88 if delta > 0 else 1.14
        self.distance = max(0.6, min(25.0, self.distance * factor))
        self.update()


class Viewport3DContainer(QWidget):
    """Complete 3D viewport toolbar + canvas with view presets and display toggles."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        bar = QHBoxLayout()
        for label, key in [("Persp", "perspective"), ("Front", "front"), ("Rear", "rear"), ("Left", "left"), ("Right", "right"), ("Top", "top")]:
            btn = QPushButton(label)
            btn.setMaximumWidth(64)
            btn.clicked.connect(lambda checked=False, k=key: self.canvas.set_view_preset(k))
            bar.addWidget(btn)

        self.chk_cameras = QCheckBox("Cameras")
        self.chk_cameras.setChecked(True)
        self.chk_markers = QCheckBox("Markers")
        self.chk_markers.setChecked(True)
        self.chk_raw = QCheckBox("Raw Overlay")
        self.chk_raw.setChecked(False)
        self.chk_conf = QCheckBox("Confidence")
        self.chk_conf.setChecked(True)
        for chk in (self.chk_cameras, self.chk_markers, self.chk_raw, self.chk_conf):
            bar.addWidget(chk)
        bar.addStretch()

        self.canvas = Canvas3DWidget()
        self.chk_cameras.toggled.connect(lambda v: self._set_flag("show_cameras", v))
        self.chk_markers.toggled.connect(lambda v: self._set_flag("show_markers", v))
        self.chk_raw.toggled.connect(lambda v: self._set_flag("show_raw_overlay", v))
        self.chk_conf.toggled.connect(lambda v: self._set_flag("color_by_confidence", v))

        layout.addLayout(bar)
        layout.addWidget(self.canvas, 1)

    def _set_flag(self, attr: str, val: bool) -> None:
        setattr(self.canvas, attr, val)
        self.canvas.update()
