"""Interactive 3D Viewport Widget supporting Stylized 3D Character Mannequin, Articulated Skeleton,
Orbit/Pan/Zoom, View Presets, Ground Grid + Shadow, Camera Frustums, Markers, Confidence Coloring,
Raw vs Corrected Overlay, and Live Mocap HUD.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QRadialGradient,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from desktop.ui.widgets.character_models import (
    BUILTIN_CHARACTER_PRESETS,
    RiggedCharacterMesh,
    build_full_55_skin_matrices,
    discover_user_character_files,
    get_character_mesh,
    load_custom_character_file,
    render_character_mesh_rgba,
    skin_character_vertices,
)
from engine.math_core.camera_model import MultiCameraCalibration
from engine.skeleton.rig_definition import (
    BONE_INDEX,
    BONE_NAMES,
    DEFAULT_REST_OFFSETS_M,
    PARENT_INDICES,
    compute_forward_kinematics_frame,
)
from engine.math_core.transforms import quat_identity


def _default_standing_pose() -> Tuple[np.ndarray, np.ndarray]:
    """Compute a clean default neutral/A-pose standing character facing the studio camera when no take is active."""
    from scipy.spatial.transform import Rotation

    lq = np.tile(quat_identity(), (len(BONE_NAMES), 1))
    # Face -Y toward the default front-3/4 studio camera, with natural relaxed A-pose arms down at sides
    lq[BONE_INDEX["hips"]] = Rotation.from_euler("z", 180.0, degrees=True).as_quat()
    lq[BONE_INDEX["left_upper_arm"]] = Rotation.from_euler("y", -52.0, degrees=True).as_quat()
    lq[BONE_INDEX["right_upper_arm"]] = Rotation.from_euler("y", 52.0, degrees=True).as_quat()
    lq[BONE_INDEX["left_forearm"]] = Rotation.from_euler("z", -14.0, degrees=True).as_quat()
    lq[BONE_INDEX["right_forearm"]] = Rotation.from_euler("z", 14.0, degrees=True).as_quat()
    pos, mats = compute_forward_kinematics_frame(DEFAULT_REST_OFFSETS_M[0], lq, DEFAULT_REST_OFFSETS_M)
    return pos.astype(np.float32), mats.astype(np.float32)


def _default_standing_positions() -> np.ndarray:
    return _default_standing_pose()[0]


class Canvas3DWidget(QWidget):
    """Hardware-independent interactive 3D Z-Up studio viewport with Character Mannequin & Skeleton rendering."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(380, 280)
        self.setMouseTracking(True)

        # Camera spherical orbit state (Z-Up)
        self.yaw_deg: float = -32.0
        self.pitch_deg: float = 18.0
        self.distance: float = 3.6
        self.target: np.ndarray = np.array([0.0, 0.0, 0.92], dtype=np.float64)
        self.orthographic: bool = False

        # Display mode, 3D character preset & visual overlays
        self.display_mode: str = "character"  # "character", "skeleton", "both"
        self.character_preset: str = "studio_female"  # "studio_female", "studio_male", "cyber_mannequin", "classic_capsule", or "file:<path>"
        self._custom_character_mesh: Optional[RiggedCharacterMesh] = None
        self.show_grid: bool = True
        self.show_axes: bool = True
        self.show_cameras: bool = True
        self.show_markers: bool = True
        self.show_raw_overlay: bool = False
        self.color_by_confidence: bool = True
        self.dark_bg: bool = True

        # Live Mocap status HUD overlay
        self.live_badge_text: str = ""
        self.live_badge_kind: str = "idle"  # "idle", "live", "recording", "demo"

        # Scene data (starts with default neutral standing character so viewport is never blank)
        self.calibration: Optional[MultiCameraCalibration] = None
        _def_pos, _def_mats = _default_standing_pose()
        self.skeleton_positions: Optional[np.ndarray] = _def_pos
        self.skeleton_global_matrices: Optional[np.ndarray] = _def_mats
        self.left_hand_quats: Optional[np.ndarray] = None
        self.right_hand_quats: Optional[np.ndarray] = None
        self.skeleton_confidence: Optional[np.ndarray] = None
        self.raw_skeleton_positions: Optional[np.ndarray] = None
        self.markers_3d: Optional[np.ndarray] = None
        self.selected_bone_idx: int = -1
        self.hud_text: str = "Z-Up (m) · Left-Drag: Orbit · Right-Drag: Pan · Wheel: Zoom"

        self._last_mouse: Optional[QPoint] = None
        self._drag_mode: Optional[str] = None

    def set_display_mode(self, mode: str) -> None:
        if mode in ("character", "skeleton", "both"):
            self.display_mode = mode
            self.update()

    def set_character_preset(self, preset_id: str) -> None:
        self.character_preset = preset_id
        if not preset_id.startswith("loaded:"):
            self._custom_character_mesh = None
        self.update()

    def set_custom_character_mesh(self, mesh: RiggedCharacterMesh) -> None:
        self._custom_character_mesh = mesh
        self.character_preset = f"loaded:{mesh.preset_id}"
        self.update()

    def set_live_status(self, text: str = "", kind: str = "idle") -> None:
        self.live_badge_text = text
        self.live_badge_kind = kind
        self.update()

    def set_view_preset(self, preset: str) -> None:
        presets = {
            "perspective": (-32.0, 18.0, False),
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
        global_matrices: Optional[np.ndarray] = None,
        left_hand_quats: Optional[np.ndarray] = None,
        right_hand_quats: Optional[np.ndarray] = None,
    ) -> None:
        if skeleton_positions is not None:
            self.skeleton_positions = skeleton_positions
            self.skeleton_global_matrices = global_matrices
        else:
            self.skeleton_positions, self.skeleton_global_matrices = _default_standing_pose()
        self.left_hand_quats = left_hand_quats
        self.right_hand_quats = right_hand_quats
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
        cam_dir = np.array(
            [
                math.cos(pitch) * math.sin(yaw),
                -math.cos(pitch) * math.cos(yaw),
                math.sin(pitch),
            ]
        )
        cam_pos = self.target + cam_dir * self.distance
        forward = self.target - cam_pos
        forward /= max(float(np.linalg.norm(forward)), 1e-8)
        up_ref = np.array([0.0, 0.0, 1.0]) if abs(forward[2]) < 0.98 else np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up_ref)
        right /= max(float(np.linalg.norm(right)), 1e-8)
        up = np.cross(right, forward)
        return cam_pos, right, up, forward

    def _project(
        self,
        pt: np.ndarray,
        cam_pos: np.ndarray,
        right: np.ndarray,
        up: np.ndarray,
        forward: np.ndarray,
    ) -> Optional[Tuple[float, float, float]]:
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
            return sx, sy, max(z, 0.5)
        sx = w * 0.5 + (x / z) * scale
        sy = h * 0.5 - (y / z) * scale
        return sx, sy, z

    def _radius_px(self, radius_m: float, depth_z: float) -> float:
        scale = min(float(self.width()), float(self.height())) * 0.95
        if self.orthographic:
            return max(2.0, radius_m * scale / max(self.distance, 0.25))
        return max(2.0, (radius_m / max(depth_z, 0.2)) * scale)

    def _draw_tapered_capsule(
        self,
        p: QPainter,
        pa: Tuple[float, float, float],
        pb: Tuple[float, float, float],
        r_a_m: float,
        r_b_m: float,
        fill_col: QColor,
        rim_col: QColor,
    ) -> None:
        ra = self._radius_px(r_a_m, pa[2])
        rb = self._radius_px(r_b_m, pb[2])
        dx = pb[0] - pa[0]
        dy = pb[1] - pa[1]
        length = math.hypot(dx, dy)
        if length < 1e-3:
            p.setPen(QPen(rim_col, 1.3))
            p.setBrush(fill_col)
            p.drawEllipse(QPointF(pa[0], pa[1]), ra, ra)
            return

        nx, ny = -dy / length, dx / length
        poly = QPolygonF(
            [
                QPointF(pa[0] + nx * ra, pa[1] + ny * ra),
                QPointF(pb[0] + nx * rb, pb[1] + ny * rb),
                QPointF(pb[0] - nx * rb, pb[1] - ny * rb),
                QPointF(pa[0] - nx * ra, pa[1] - ny * ra),
            ]
        )
        grad = QLinearGradient(pa[0] + nx * ra, pa[1] + ny * ra, pa[0] - nx * ra, pa[1] - ny * ra)
        grad.setColorAt(0.0, fill_col.lighter(132))
        grad.setColorAt(0.55, fill_col)
        grad.setColorAt(1.0, fill_col.darker(155))

        p.setPen(QPen(rim_col, 1.2))
        p.setBrush(QBrush(grad))
        p.drawPolygon(poly)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(pa[0], pa[1]), ra * 0.96, ra * 0.96)
        p.drawEllipse(QPointF(pb[0], pb[1]), rb * 0.96, rb * 0.96)

    def _draw_character_mannequin(
        self,
        p: QPainter,
        pts: np.ndarray,
        cam_pos: np.ndarray,
        right: np.ndarray,
        up: np.ndarray,
        forward: np.ndarray,
        translucent: bool = False,
    ) -> None:
        """Render a depth-sorted stylized 3D articulated humanoid studio character."""
        if len(pts) < 23:
            return
        alpha = 145 if translucent else 240
        suit_body = QColor(42, 168, 178, alpha) if self.dark_bg else QColor(24, 132, 142, alpha)
        suit_torso = QColor(56, 136, 205, alpha) if self.dark_bg else QColor(36, 108, 172, alpha)
        suit_limb = QColor(44, 152, 172, alpha) if self.dark_bg else QColor(30, 124, 144, alpha)
        suit_head = QColor(210, 224, 238, alpha) if self.dark_bg else QColor(74, 96, 118, alpha)
        rim_col = QColor(140, 232, 236, 190 if not translucent else 110)

        def proj_idx(idx: int) -> Optional[Tuple[float, float, float]]:
            return self._project(pts[idx].astype(np.float64), cam_pos, right, up, forward)

        # Ground contact shadow ellipse directly under hips on Z=0
        hip_ground = np.array([float(pts[0, 0]), float(pts[0, 1]), 0.002], dtype=np.float64)
        pg = self._project(hip_ground, cam_pos, right, up, forward)
        if pg is not None:
            rx = self._radius_px(0.32, pg[2])
            ry = max(4.0, rx * max(0.22, abs(math.sin(math.radians(self.pitch_deg)))))
            shad_grad = QRadialGradient(QPointF(pg[0], pg[1]), rx)
            shad_grad.setColorAt(0.0, QColor(0, 0, 0, 115 if self.dark_bg else 65))
            shad_grad.setColorAt(1.0, QColor(0, 0, 0, 0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(shad_grad))
            p.drawEllipse(QPointF(pg[0], pg[1]), rx, ry)

        # Foot ground-contact glow rings when ankles/toes are near floor
        for foot_i in (BONE_INDEX["left_foot"], BONE_INDEX["right_foot"]):
            if pts[foot_i, 2] <= 0.14:
                fg = np.array([float(pts[foot_i, 0]), float(pts[foot_i, 1]), 0.004], dtype=np.float64)
                pf = self._project(fg, cam_pos, right, up, forward)
                if pf is not None:
                    rf = self._radius_px(0.11, pf[2])
                    p.setPen(QPen(QColor(34, 197, 94, 170), 1.6))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawEllipse(QPointF(pf[0], pf[1]), rf, max(3.0, rf * 0.45))

        # Collect depth-sorted drawables: (mean_depth_z, draw_callable)
        drawables: List[Tuple[float, object]] = []

        # Sculpted Pelvis & Torso Quads
        def add_quad(indices: List[int], color: QColor) -> None:
            projs = [proj_idx(i) for i in indices]
            if any(pr is None for pr in projs):
                return
            valid_p = [pr for pr in projs if pr is not None]
            z_mean = float(np.mean([pr[2] for pr in valid_p]))

            def _draw() -> None:
                poly = QPolygonF([QPointF(pr[0], pr[1]) for pr in valid_p])
                xs = [pr[0] for pr in valid_p]
                ys = [pr[1] for pr in valid_p]
                grad = QLinearGradient(min(xs), min(ys), max(xs), max(ys))
                grad.setColorAt(0.0, color.lighter(128))
                grad.setColorAt(0.6, color)
                grad.setColorAt(1.0, color.darker(150))
                p.setPen(QPen(rim_col, 1.3))
                p.setBrush(QBrush(grad))
                p.drawPolygon(poly)

            drawables.append((z_mean, _draw))

        # Pelvis block & Chest/Torso armor
        add_quad(
            [BONE_INDEX["left_upper_leg"], BONE_INDEX["right_upper_leg"], BONE_INDEX["spine1"], BONE_INDEX["spine"]],
            suit_body,
        )
        add_quad(
            [
                BONE_INDEX["left_upper_arm"],
                BONE_INDEX["right_upper_arm"],
                BONE_INDEX["right_upper_leg"],
                BONE_INDEX["left_upper_leg"],
            ],
            suit_torso,
        )

        # Articulated tapered limb capsules: (bone_a, bone_b, r_a_m, r_b_m, color)
        capsule_specs = [
            (BONE_INDEX["hips"], BONE_INDEX["chest"], 0.115, 0.135, suit_torso),
            (BONE_INDEX["chest"], BONE_INDEX["neck"], 0.085, 0.055, suit_body),
            (BONE_INDEX["left_clavicle"], BONE_INDEX["left_upper_arm"], 0.055, 0.062, suit_torso),
            (BONE_INDEX["right_clavicle"], BONE_INDEX["right_upper_arm"], 0.055, 0.062, suit_torso),
            (BONE_INDEX["left_upper_arm"], BONE_INDEX["left_forearm"], 0.058, 0.046, suit_limb),
            (BONE_INDEX["left_forearm"], BONE_INDEX["left_hand"], 0.045, 0.034, suit_limb),
            (BONE_INDEX["left_hand"], BONE_INDEX["left_hand_end"], 0.038, 0.028, suit_head),
            (BONE_INDEX["right_upper_arm"], BONE_INDEX["right_forearm"], 0.058, 0.046, suit_limb),
            (BONE_INDEX["right_forearm"], BONE_INDEX["right_hand"], 0.045, 0.034, suit_limb),
            (BONE_INDEX["right_hand"], BONE_INDEX["right_hand_end"], 0.038, 0.028, suit_head),
            (BONE_INDEX["left_upper_leg"], BONE_INDEX["left_lower_leg"], 0.082, 0.058, suit_body),
            (BONE_INDEX["left_lower_leg"], BONE_INDEX["left_foot"], 0.056, 0.040, suit_limb),
            (BONE_INDEX["left_foot"], BONE_INDEX["left_toe"], 0.044, 0.032, suit_head),
            (BONE_INDEX["right_upper_leg"], BONE_INDEX["right_lower_leg"], 0.082, 0.058, suit_body),
            (BONE_INDEX["right_lower_leg"], BONE_INDEX["right_foot"], 0.056, 0.040, suit_limb),
            (BONE_INDEX["right_foot"], BONE_INDEX["right_toe"], 0.044, 0.032, suit_head),
        ]
        for ba, bb, ra, rb, col in capsule_specs:
            pa = proj_idx(ba)
            pb = proj_idx(bb)
            if pa is None or pb is None:
                continue
            z_mean = 0.5 * (pa[2] + pb[2])
            drawables.append(
                (z_mean, lambda _pa=pa, _pb=pb, _ra=ra, _rb=rb, _c=col: self._draw_tapered_capsule(p, _pa, _pb, _ra, _rb, _c, rim_col))
            )

        # Sculpted 3D Head Sphere + Directional Visor
        head_center = 0.45 * pts[BONE_INDEX["head"]].astype(np.float64) + 0.55 * pts[BONE_INDEX["head_end"]].astype(np.float64)
        ph = self._project(head_center, cam_pos, right, up, forward)
        if ph is not None:
            # Estimate forward direction from shoulder line x spine up
            sh_vec = pts[BONE_INDEX["right_upper_arm"]].astype(np.float64) - pts[BONE_INDEX["left_upper_arm"]].astype(np.float64)
            sp_up = pts[BONE_INDEX["head_end"]].astype(np.float64) - pts[BONE_INDEX["chest"]].astype(np.float64)
            fwd_est = np.cross(sp_up, sh_vec)
            fn = float(np.linalg.norm(fwd_est))
            fwd_est = (fwd_est / fn) if fn > 1e-5 else np.array([0.0, 1.0, 0.0])
            visor_pt = self._project(head_center + fwd_est * 0.095, cam_pos, right, up, forward)

            def _draw_head(_ph=ph, _vp=visor_pt) -> None:
                rad = self._radius_px(0.105, _ph[2])
                grad = QRadialGradient(QPointF(_ph[0] - rad * 0.3, _ph[1] - rad * 0.35), rad * 1.35)
                grad.setColorAt(0.0, suit_head.lighter(135))
                grad.setColorAt(0.65, suit_head)
                grad.setColorAt(1.0, suit_head.darker(160))
                p.setPen(QPen(rim_col, 1.4))
                p.setBrush(QBrush(grad))
                p.drawEllipse(QPointF(_ph[0], _ph[1]), rad * 0.92, rad * 1.05)
                if _vp is not None and _vp[2] < _ph[2] + 0.04:
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(QColor(45, 212, 191, 235))
                    p.drawEllipse(QPointF(_vp[0], _vp[1]), rad * 0.36, rad * 0.16)

            drawables.append((ph[2] - 0.02, _draw_head))

        # Depth sort farthest-first (descending Z) and paint
        drawables.sort(key=lambda item: item[0], reverse=True)
        for _, fn in drawables:
            fn()  # type: ignore[misc]

    def _draw_sculpted_character_mesh(
        self,
        p: QPainter,
        mesh: RiggedCharacterMesh,
        pts: np.ndarray,
        cam_pos: np.ndarray,
        right: np.ndarray,
        up: np.ndarray,
        forward: np.ndarray,
        translucent: bool = False,
    ) -> None:
        """Render a 3-point studio-lit rigged 3D character mesh (built-in or custom .GLB/.OBJ) + floor shadow."""
        if len(pts) < 23:
            return
        # Soft ground contact shadow under hips on Z = 0
        hip_ground = np.array([float(pts[0, 0]), float(pts[0, 1]), 0.002], dtype=np.float64)
        pg = self._project(hip_ground, cam_pos, right, up, forward)
        if pg is not None:
            rx = self._radius_px(0.34, pg[2])
            ry = max(4.0, rx * max(0.22, abs(math.sin(math.radians(self.pitch_deg)))))
            shad_grad = QRadialGradient(QPointF(pg[0], pg[1]), rx)
            shad_grad.setColorAt(0.0, QColor(0, 0, 0, 130 if self.dark_bg else 75))
            shad_grad.setColorAt(1.0, QColor(0, 0, 0, 0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(shad_grad))
            p.drawEllipse(QPointF(pg[0], pg[1]), rx, ry)

        # Foot ground-contact glow rings when ankles/toes are near floor
        for foot_i in (BONE_INDEX["left_foot"], BONE_INDEX["right_foot"]):
            if pts[foot_i, 2] <= 0.14:
                fg = np.array([float(pts[foot_i, 0]), float(pts[foot_i, 1]), 0.004], dtype=np.float64)
                pf = self._project(fg, cam_pos, right, up, forward)
                if pf is not None:
                    rf = self._radius_px(0.11, pf[2])
                    p.setPen(QPen(QColor(34, 197, 94, 170), 1.6))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawEllipse(QPointF(pf[0], pf[1]), rf, max(3.0, rf * 0.45))

        M_skin = build_full_55_skin_matrices(
            mesh=mesh,
            skeleton_positions=pts,
            global_matrices=self.skeleton_global_matrices,
            left_hand_quats_xyzw=self.left_hand_quats,
            right_hand_quats_xyzw=self.right_hand_quats,
        )
        posed_v = skin_character_vertices(mesh, M_skin)
        rgba = render_character_mesh_rgba(
            mesh=mesh,
            posed_vertices=posed_v,
            width=self.width(),
            height=self.height(),
            cam_pos=cam_pos,
            right=right,
            up=up,
            forward=forward,
            distance=self.distance,
            orthographic=self.orthographic,
            alpha=0.62 if translucent else 1.0,
        )
        if rgba is not None:
            qimg = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format.Format_RGBA8888)
            p.drawImage(0, 0, qimg)

    def _active_character_mesh(self) -> Optional[RiggedCharacterMesh]:
        if self._custom_character_mesh is not None and self.character_preset.startswith("loaded:"):
            return self._custom_character_mesh
        return get_character_mesh(self.character_preset)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Rich studio radial backdrop gradient
        w, h = self.width(), self.height()
        bg_grad = QRadialGradient(QPointF(w * 0.5, h * 0.45), max(w, h) * 0.72)
        if self.dark_bg:
            bg_grad.setColorAt(0.0, QColor("#15202E"))
            bg_grad.setColorAt(1.0, QColor("#090E14"))
        else:
            bg_grad.setColorAt(0.0, QColor("#F5F8FC"))
            bg_grad.setColorAt(1.0, QColor("#DCE4EE"))
        p.fillRect(self.rect(), QBrush(bg_grad))

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
            grid_col = QColor("#1F2D3D") if self.dark_bg else QColor("#C4D0DF")
            major_col = QColor("#2B3D52") if self.dark_bg else QColor("#A9B9CC")
            for v in np.arange(-2.5, 2.51, 0.5):
                col = major_col if abs(v - round(v)) < 1e-3 else grid_col
                draw_line_3d(np.array([v, -2.5, 0.0]), np.array([v, 2.5, 0.0]), col, 1.0)
                draw_line_3d(np.array([-2.5, v, 0.0]), np.array([2.5, v, 0.0]), col, 1.0)

        # 2. World Coordinate Axes (X Red, Y Green, Z Blue)
        if self.show_axes:
            o = np.zeros(3)
            draw_line_3d(o, np.array([0.55, 0.0, 0.0]), QColor("#EF4444"), 2.0)
            draw_line_3d(o, np.array([0.0, 0.55, 0.0]), QColor("#22C55E"), 2.0)
            draw_line_3d(o, np.array([0.0, 0.0, 0.55]), QColor("#3B82F6"), 2.0)

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
                    draw_line_3d(c_pos, cw, frustum_col, 1.2)
                for idx in range(4):
                    draw_line_3d(corners_w[idx], corners_w[(idx + 1) % 4], frustum_col, 1.2)
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

        # 5. Raw Skeleton Overlay (dashed amber for before-vs-after comparison)
        if self.show_raw_overlay and self.raw_skeleton_positions is not None:
            raw_pts = self.raw_skeleton_positions
            for b_idx, parent_idx in enumerate(PARENT_INDICES):
                if parent_idx >= 0 and b_idx < len(raw_pts):
                    draw_line_3d(
                        raw_pts[parent_idx].astype(np.float64),
                        raw_pts[b_idx].astype(np.float64),
                        QColor("#F59E0B"),
                        1.6,
                        dashed=True,
                    )

        # 6. 3D Character (Sculpted Humanoid / Custom .GLB/.OBJ / Classic Capsule)
        if self.skeleton_positions is not None and self.display_mode in ("character", "both"):
            char_mesh = self._active_character_mesh()
            if char_mesh is not None:
                self._draw_sculpted_character_mesh(
                    p,
                    char_mesh,
                    self.skeleton_positions,
                    cam_pos,
                    right,
                    up,
                    forward,
                    translucent=(self.display_mode == "both"),
                )
            else:
                self._draw_character_mannequin(
                    p,
                    self.skeleton_positions,
                    cam_pos,
                    right,
                    up,
                    forward,
                    translucent=(self.display_mode == "both"),
                )

        # 7. Articulated Skeleton Overlay
        if self.skeleton_positions is not None and self.display_mode in ("skeleton", "both"):
            pts = self.skeleton_positions
            confs = self.skeleton_confidence
            for b_idx, parent_idx in enumerate(PARENT_INDICES):
                if parent_idx >= 0 and b_idx < len(pts):
                    col = QColor("#2DD4BF")
                    if self.color_by_confidence and confs is not None and b_idx < len(confs):
                        c_val = float(confs[b_idx])
                        if c_val >= 0.7:
                            col = QColor("#22C55E")
                        elif c_val >= 0.35:
                            col = QColor("#F59E0B")
                        else:
                            col = QColor("#EF4444")
                    draw_line_3d(pts[parent_idx].astype(np.float64), pts[b_idx].astype(np.float64), col, 3.0)
            for b_idx in range(min(len(pts), len(BONE_NAMES))):
                pj = self._project(pts[b_idx].astype(np.float64), cam_pos, right, up, forward)
                if pj is not None:
                    is_sel = b_idx == self.selected_bone_idx
                    p.setPen(QPen(QColor("#0F172A"), 1.0))
                    p.setBrush(QColor("#EF4444") if is_sel else QColor("#FFFFFF"))
                    r = 5.8 if is_sel else 3.6
                    p.drawEllipse(QPointF(pj[0], pj[1]), r, r)

        # 8. Top-Left HUD Overlay
        p.setPen(QPen(QColor("#94A6B8") if self.dark_bg else QColor("#475569")))
        p.setFont(QFont("Segoe UI", 9))
        mode_lbl = "Ortho" if self.orthographic else "Persp"
        p.drawText(12, 22, f"[{mode_lbl} · {self.display_mode.title()}]  {self.hud_text}")

        # 9. Top-Right Live Mocap Status Pill
        if self.live_badge_text:
            badge_colors = {
                "recording": (QColor(211, 47, 47, 230), QColor("#FFFFFF")),
                "live": (QColor(13, 148, 136, 225), QColor("#FFFFFF")),
                "demo": (QColor(180, 83, 9, 220), QColor("#FFFBEB")),
            }
            bg_c, fg_c = badge_colors.get(self.live_badge_kind, (QColor(30, 41, 59, 210), QColor("#E2E8F0")))
            font = QFont("Segoe UI", 9, QFont.Weight.Bold)
            p.setFont(font)
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(self.live_badge_text)
            pill_w = tw + 22
            pill_h = 26
            x0 = w - pill_w - 12
            y0 = 10
            path = QPainterPath()
            path.addRoundedRect(x0, y0, pill_w, pill_h, 13, 13)
            p.setPen(Qt.PenStyle.NoPen)
            p.fillPath(path, bg_c)
            p.setPen(QPen(fg_c))
            p.drawText(int(x0 + 11), int(y0 + 17), self.live_badge_text)

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
    """Complete 3D viewport toolbar + canvas with Character/Skeleton mode selector and view presets."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Row 1: Display Mode + 3D Character Selector + Custom .GLB/.OBJ Loader (fits compact split panes!)
        bar1 = QHBoxLayout()
        bar1.setSpacing(5)
        bar1.addWidget(QLabel("Preview:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItem("3D Character", "character")
        self.combo_mode.addItem("Skeleton", "skeleton")
        self.combo_mode.addItem("Character + Skeleton", "both")
        self.combo_mode.currentIndexChanged.connect(
            lambda: self.canvas.set_display_mode(str(self.combo_mode.currentData()))
        )
        bar1.addWidget(self.combo_mode)

        bar1.addWidget(QLabel("Character:"))
        self.combo_character = QComboBox()
        self.combo_character.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo_character.setMinimumContentsLength(14)
        self.combo_character.setMaximumWidth(215)
        self._populate_character_combo()
        self.combo_character.currentIndexChanged.connect(self._on_character_changed)
        bar1.addWidget(self.combo_character, 1)

        self.btn_load_char = QPushButton("+ Load .GLB/.OBJ")
        self.btn_load_char.setToolTip(
            "Load any rigged .GLB (Blender / Mixamo / ReadyPlayerMe / VRM) or .OBJ character into the 3D Viewport"
        )
        self.btn_load_char.clicked.connect(self._load_custom_character)
        bar1.addWidget(self.btn_load_char)
        bar1.addStretch()

        # Row 2: Camera View Presets & Visual Overlays
        bar2 = QHBoxLayout()
        bar2.setSpacing(4)
        for label, key in [("Persp", "perspective"), ("Front", "front"), ("Left", "left"), ("Right", "right"), ("Top", "top")]:
            btn = QPushButton(label)
            btn.setMinimumWidth(56)
            btn.setMaximumWidth(64)
            btn.setStyleSheet("padding: 4px;")
            btn.clicked.connect(lambda checked=False, k=key: self.canvas.set_view_preset(k))
            bar2.addWidget(btn)

        self.chk_cameras = QCheckBox("Cams")
        self.chk_cameras.setChecked(True)
        self.chk_markers = QCheckBox("Markers")
        self.chk_markers.setChecked(True)
        self.chk_raw = QCheckBox("Raw")
        self.chk_raw.setChecked(False)
        self.chk_conf = QCheckBox("Conf")
        self.chk_conf.setChecked(True)
        for chk in (self.chk_cameras, self.chk_markers, self.chk_raw, self.chk_conf):
            bar2.addWidget(chk)
        bar2.addStretch()

        self.canvas = Canvas3DWidget()
        # Sync initial character selection cleanly with canvas
        if self.combo_character.count() > 0:
            self.canvas.set_character_preset(str(self.combo_character.currentData()))
        self.chk_cameras.toggled.connect(lambda v: self._set_flag("show_cameras", v))
        self.chk_markers.toggled.connect(lambda v: self._set_flag("show_markers", v))
        self.chk_raw.toggled.connect(lambda v: self._set_flag("show_raw_overlay", v))
        self.chk_conf.toggled.connect(lambda v: self._set_flag("color_by_confidence", v))

        layout.addLayout(bar1)
        layout.addLayout(bar2)
        layout.addWidget(self.canvas, 1)

    def _populate_character_combo(self, select_data: Optional[str] = None) -> None:
        self.combo_character.blockSignals(True)
        self.combo_character.clear()
        for pid, label in BUILTIN_CHARACTER_PRESETS:
            self.combo_character.addItem(label, pid)
        for fp in discover_user_character_files():
            self.combo_character.addItem(f"Custom: {fp.stem} ({fp.suffix.lower()})", f"file:{fp}")
        if select_data is not None:
            idx = self.combo_character.findData(select_data)
            if idx >= 0:
                self.combo_character.setCurrentIndex(idx)
        self.combo_character.blockSignals(False)

    def _on_character_changed(self) -> None:
        data = str(self.combo_character.currentData() or "studio_female")
        try:
            self.canvas.set_character_preset(data)
        except Exception as exc:
            QMessageBox.warning(self, "Character Load Error", f"Could not load character:\n{exc}")
            self.combo_character.setCurrentIndex(0)
            self.canvas.set_character_preset("studio_female")

    def _load_custom_character(self) -> None:
        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Select Custom 3D Character (.GLB or .OBJ)",
            "",
            "3D Character Models (*.glb *.obj);;glTF Binary (*.glb);;Wavefront OBJ (*.obj)",
        )
        if not fp:
            return
        try:
            mesh = load_custom_character_file(fp, copy_to_library=True)
            saved_key = f"file:{mesh.source_path}" if mesh.source_path else f"loaded:{mesh.preset_id}"
            self._populate_character_combo(select_data=saved_key)
            self.canvas.set_custom_character_mesh(mesh)
        except Exception as exc:
            QMessageBox.warning(self, "Custom Character Import Error", f"Failed to load '{Path(fp).name}':\n{exc}")

    def _set_flag(self, attr: str, val: bool) -> None:
        setattr(self.canvas, attr, val)
        self.canvas.update()
