"""Interactive Visual Studio Calibration Guide Widget.

Draws a clear Top-Down Studio Floor Map + Side Elevation View showing:
- Exact recommended smartphone tripod placement (distance, spacing angle, height, downward tilt, FOV overlap cone)
  dynamically tailored for 2, 3, 4, 6, or 8 cameras.
- Step-by-step ChArUco board placement poses (Floor Origin, Center Upright, Left/Right Tilted, Low/High Sweep)
  showing exactly where to place or hold the printed board in both top-down and side views.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class CalibrationPoseStep:
    index: int
    short_label: str
    title: str
    where_to_place: str
    why_needed: str
    # Top-down position in studio meters (x: right, y: forward; capture zone is -1.2..+1.2)
    board_xy_m: Tuple[float, float]
    board_yaw_deg: float  # Orientation in top-down view
    board_flat_on_floor: bool
    # Side-view elevation (height_m above floor, pitch_deg: 0=vertical facing cam, 90=flat on floor, +tilt up, -tilt down)
    board_height_m: float
    board_pitch_deg: float


CALIBRATION_POSE_STEPS: List[CalibrationPoseStep] = [
    CalibrationPoseStep(
        index=0,
        short_label="1. Center Upright",
        title="Pose 1 / 6 — Center Upright (Sets Vertical Z-Up)",
        where_to_place=(
            "Stand in the center and hold the ChArUco board STRAIGHT UPRIGHT at chest height (~1.15 m), "
            "bottom edge level, facing between ALL cameras."
        ),
        why_needed="Detects cleanly across all cameras with zero floor glare AND automatically aligns vertical Z-Up for Blender!",
        board_xy_m=(0.0, 0.0),
        board_yaw_deg=0.0,
        board_flat_on_floor=False,
        board_height_m=1.15,
        board_pitch_deg=0.0,
    ),
    CalibrationPoseStep(
        index=1,
        short_label="2. Tilted Angle",
        title="Pose 2 / 6 — Center Tilted (~25° Forward)",
        where_to_place=(
            "Stay in the center at chest height and tilt the top of the board ~25° forward/backward "
            "while keeping it clearly visible to both cameras."
        ),
        why_needed="Provides 3D perspective angle variation to lock exact camera focal length & lens distortion.",
        board_xy_m=(0.0, 0.05),
        board_yaw_deg=12.0,
        board_flat_on_floor=False,
        board_height_m=1.20,
        board_pitch_deg=-25.0,
    ),
    CalibrationPoseStep(
        index=2,
        short_label="3. Left Tilt",
        title="Pose 3 / 6 — Left Side & Tilted (~25°)",
        where_to_place=(
            "Move to the LEFT edge of the capture zone (~0.8 m left of center) at chest height, "
            "and tilt the board ~25° inward so all cameras still see the full grid."
        ),
        why_needed="Calibrates lens distortion across the left frame edge and improves depth triangulation.",
        board_xy_m=(-0.82, 0.15),
        board_yaw_deg=28.0,
        board_flat_on_floor=False,
        board_height_m=1.15,
        board_pitch_deg=-15.0,
    ),
    CalibrationPoseStep(
        index=3,
        short_label="4. Right Tilt",
        title="Pose 4 / 6 — Right Side & Tilted (~25°)",
        where_to_place=(
            "Move to the RIGHT edge of the capture zone (~0.8 m right of center) at chest height, "
            "and tilt the board ~25° inward toward the cameras."
        ),
        why_needed="Balances horizontal focal length and right-edge lens distortion across cameras.",
        board_xy_m=(0.82, 0.15),
        board_yaw_deg=-28.0,
        board_flat_on_floor=False,
        board_height_m=1.15,
        board_pitch_deg=15.0,
    ),
    CalibrationPoseStep(
        index=4,
        short_label="5. Low Angle",
        title="Pose 5 / 6 — Low Angle (Near Knees, Tilted Up)",
        where_to_place=(
            "Hold the board low near knee height (~0.45 m) slightly toward the front of the capture zone, "
            "tilted ~30° UPWARD toward the camera lenses."
        ),
        why_needed="Locks lower-body/foot tracking accuracy near the bottom of each phone's frame.",
        board_xy_m=(0.0, -0.45),
        board_yaw_deg=8.0,
        board_flat_on_floor=False,
        board_height_m=0.45,
        board_pitch_deg=30.0,
    ),
    CalibrationPoseStep(
        index=5,
        short_label="6. High Angle",
        title="Pose 6 / 6 — High Angle (Overhead, Tilted Down)",
        where_to_place=(
            "Hold the board high near head/overhead height (~1.85 m) slightly back in the capture zone, "
            "tilted ~25° DOWNWARD toward the cameras."
        ),
        why_needed="Locks head/arm reach accuracy and completes 3D bundle adjustment across the full volume.",
        board_xy_m=(0.0, 0.45),
        board_yaw_deg=-8.0,
        board_flat_on_floor=False,
        board_height_m=1.85,
        board_pitch_deg=-25.0,
    ),
]


def _camera_layout_for_count(num_cameras: int) -> List[Tuple[str, float, float, float]]:
    """Return list of (label, x_m, y_m, look_angle_deg) in studio top-down coordinates.
    Convention: +X right, +Y forward (away from front cameras); front cameras sit at negative Y looking toward (0,0).
    """
    r = 2.65
    if num_cameras <= 2:
        # 2 cameras at Front-Left and Front-Right (~80 deg separation -> ideal stereo baseline!)
        angles_deg = [-130.0, -50.0]
    elif num_cameras == 3:
        # 3 cameras forming a front-biased 120° wrap around the actor
        angles_deg = [-140.0, -40.0, 90.0]
    elif num_cameras == 4:
        # 4 corners around the capture volume (90° apart)
        angles_deg = [-135.0, -45.0, 45.0, 135.0]
    elif num_cameras == 6:
        angles_deg = [-150.0, -90.0, -30.0, 30.0, 90.0, 150.0]
    else:
        step = 360.0 / max(num_cameras, 2)
        angles_deg = [-135.0 + i * step for i in range(num_cameras)]

    cams: List[Tuple[str, float, float, float]] = []
    for i, ang in enumerate(angles_deg):
        rad = math.radians(ang)
        x = r * math.cos(rad)
        y = r * math.sin(rad)
        look_deg = math.degrees(math.atan2(-y, -x))
        cams.append((f"CAM {i + 1}", x, y, look_deg))
    return cams


def camera_placement_summary(num_cameras: int) -> str:
    """Return concise human-readable camera placement instructions for the selected count."""
    if num_cameras <= 2:
        return (
            "2 Phones: Place CAM 1 & CAM 2 at Front-Left and Front-Right (~75°–90° apart), "
            "2.5–3.2 m from center, on tripods 1.5–1.7 m high, tilted ~15° downward so head-to-feet + floor are fully visible."
        )
    if num_cameras == 3:
        return (
            "3 Phones: Place CAM 1 & CAM 2 at Front-Left / Front-Right and CAM 3 behind/side (~120° triangle), "
            "2.5–3.2 m from center, 1.5–1.8 m high, all aimed at chest height in the center."
        )
    if num_cameras == 4:
        return (
            "4 Phones (Recommended Studio): Place 1 phone at each of the 4 corners around a 2.4 × 2.4 m capture mat, "
            "2.5–3.5 m from center, 1.6–1.8 m high, tilted ~15°–20° downward."
        )
    return (
        f"{num_cameras} Phones (360° Ring): Space phones evenly in a ring 2.5–3.5 m from center at 1.5–1.9 m tripod height, "
        "ensuring every neighboring camera pair sees the ChArUco board together."
    )


class CalibrationGuideCanvas(QWidget):
    """Custom QPainter studio diagram showing Top-Down Room/Camera/Board map + Side Elevation view."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(290)
        self.num_cameras: int = 2
        self.pose_index: int = 0
        self.dark_bg: bool = True

    def set_num_cameras(self, n: int) -> None:
        self.num_cameras = max(2, min(8, int(n)))
        self.update()

    def set_pose_index(self, idx: int) -> None:
        self.pose_index = max(0, min(len(CALIBRATION_POSE_STEPS) - 1, int(idx)))
        self.update()

    def set_dark_bg(self, dark: bool) -> None:
        self.dark_bg = bool(dark)
        self.update()

    def _draw_mini_charuco(self, p: QPainter, rect: QRectF, cols: int = 4, rows: int = 3) -> None:
        """Draw a crisp tiny ChArUco checkerboard icon inside rect."""
        p.save()
        p.setPen(QPen(QColor("#F59E0B"), 2.0))
        p.setBrush(QBrush(QColor("#FFFFFF")))
        p.drawRoundedRect(rect, 2, 2)
        cw = rect.width() / cols
        ch = rect.height() / rows
        p.setPen(Qt.PenStyle.NoPen)
        for r in range(rows):
            for c in range(cols):
                if (r + c) % 2 == 0:
                    p.fillRect(
                        QRectF(rect.left() + c * cw, rect.top() + r * ch, cw, ch),
                        QColor("#0F172A"),
                    )
                else:
                    # Tiny ArUco inner dot inside white squares
                    m = min(cw, ch) * 0.25
                    p.fillRect(
                        QRectF(rect.left() + c * cw + m, rect.top() + r * ch + m, cw - 2 * m, ch - 2 * m),
                        QColor("#0F766E"),
                    )
        p.restore()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()

        # Palette
        bg = QColor("#0B111A") if self.dark_bg else QColor("#F8FAFC")
        panel_bg = QColor("#101926") if self.dark_bg else QColor("#FFFFFF")
        border = QColor("#223244") if self.dark_bg else QColor("#CBD5E1")
        grid_c = QColor("#1A2838") if self.dark_bg else QColor("#E2E8F0")
        text_c = QColor("#E2E8F0") if self.dark_bg else QColor("#0F172A")
        sub_c = QColor("#8DA2B8") if self.dark_bg else QColor("#475569")
        accent = QColor("#14B8A6") if self.dark_bg else QColor("#0F766E")
        amber = QColor("#F59E0B") if self.dark_bg else QColor("#D97706")

        p.fillRect(0, 0, w, h, bg)

        pad = 10
        split_x = int(w * 0.60)
        left_rect = QRectF(pad, pad, max(180, split_x - pad - 6), h - 2 * pad)
        right_rect = QRectF(split_x + 6, pad, max(160, w - split_x - pad - 6), h - 2 * pad)

        # Panel frames
        for rct in (left_rect, right_rect):
            path = QPainterPath()
            path.addRoundedRect(rct, 10, 10)
            p.fillPath(path, panel_bg)
            p.setPen(QPen(border, 1.2))
            p.drawPath(path)

        pose = CALIBRATION_POSE_STEPS[self.pose_index]
        self._paint_top_down(p, left_rect, pose, grid_c, text_c, sub_c, accent, amber)
        self._paint_side_view(p, right_rect, pose, grid_c, text_c, sub_c, accent, amber)
        p.end()

    def _paint_top_down(
        self,
        p: QPainter,
        rect: QRectF,
        pose: CalibrationPoseStep,
        grid_c: QColor,
        text_c: QColor,
        sub_c: QColor,
        accent: QColor,
        amber: QColor,
    ) -> None:
        p.save()
        p.setClipRect(rect.adjusted(1, 1, -1, -1))

        # Title
        p.setPen(QPen(text_c))
        p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        p.drawText(int(rect.left() + 12), int(rect.top() + 20), f"TOP-DOWN ROOM MAP ({self.num_cameras} Cameras)")
        p.setPen(QPen(sub_c))
        p.setFont(QFont("Segoe UI", 8))
        p.drawText(int(rect.left() + 12), int(rect.top() + 34), "Grid = 1.0 m · Keep tripods 2.5–3.2 m from center")

        cx = rect.center().x()
        cy = rect.center().y() + 14
        usable = min(rect.width() - 40, rect.height() - 62)
        ppm = max(18.0, usable / 6.8)  # pixels per meter (~6.8m span fits cameras at r=2.65m comfortably)

        def to_pt(xm: float, ym: float) -> QPointF:
            # +X right, +Y up on screen
            return QPointF(cx + xm * ppm, cy - ym * ppm)

        # 1m room grid
        p.setPen(QPen(grid_c, 1.0))
        for m in range(-3, 4):
            p.drawLine(to_pt(m, -3.2), to_pt(m, 3.2))
            p.drawLine(to_pt(-3.2, m), to_pt(3.2, m))

        # Distance ring (r = 2.65m recommended tripod circle)
        ring_pen = QPen(sub_c, 1.0, Qt.PenStyle.DotLine)
        p.setPen(ring_pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        r_px = 2.65 * ppm
        p.drawEllipse(QPointF(cx, cy), r_px, r_px)

        # Capture Zone 2.4m x 2.4m
        cz_tl = to_pt(-1.2, 1.2)
        cz_br = to_pt(1.2, -1.2)
        cz_rect = QRectF(cz_tl, cz_br)
        p.fillRect(cz_rect, QColor(accent.red(), accent.green(), accent.blue(), 28))
        p.setPen(QPen(accent, 1.6, Qt.PenStyle.DashLine))
        p.drawRect(cz_rect)
        p.setPen(QPen(accent))
        p.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        p.drawText(int(cz_rect.left() + 5), int(cz_rect.top() + 13), "Capture Zone (2.4 × 2.4 m)")

        # Origin crosshair +Y Forward arrow
        orig = to_pt(0.0, 0.0)
        p.setPen(QPen(QColor("#22C55E"), 1.5))
        p.drawLine(orig, to_pt(0.0, 0.55))
        p.setPen(QPen(QColor("#EF4444"), 1.5))
        p.drawLine(orig, to_pt(0.55, 0.0))
        p.setPen(QPen(sub_c))
        p.setFont(QFont("Segoe UI", 7))
        p.drawText(int(orig.x() + 5), int(orig.y() - 0.55 * ppm + 4), "+Y")

        # Cameras + FOV cones
        cams = _camera_layout_for_count(self.num_cameras)
        for label, xm, ym, look_deg in cams:
            cpt = to_pt(xm, ym)
            # FOV cone (~56 deg horizontal FOV)
            cone_len = 3.55 * ppm
            half_fov = 26.0
            a1 = math.radians(look_deg - half_fov)
            a2 = math.radians(look_deg + half_fov)
            p1 = QPointF(cpt.x() + cone_len * math.cos(a1), cpt.y() - cone_len * math.sin(a1))
            p2 = QPointF(cpt.x() + cone_len * math.cos(a2), cpt.y() - cone_len * math.sin(a2))
            poly = QPolygonF([cpt, p1, p2])
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(accent.red(), accent.green(), accent.blue(), 26)))
            p.drawPolygon(poly)
            p.setPen(QPen(QColor(accent.red(), accent.green(), accent.blue(), 85), 1.0))
            p.drawLine(cpt, p1)
            p.drawLine(cpt, p2)

            # Camera body icon
            p.setBrush(QBrush(accent))
            p.setPen(QPen(QColor("#FFFFFF") if self.dark_bg else QColor("#0F172A"), 1.5))
            p.drawEllipse(cpt, 8.5, 8.5)
            p.setPen(QPen(text_c))
            p.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            lx = cpt.x() + (11 if xm >= 0 else -44)
            ly = cpt.y() + (16 if ym < 0 else -10)
            p.drawText(int(lx), int(ly), label)

        # Target ChArUco Board Placement Marker
        bx, by = pose.board_xy_m
        bpt = to_pt(bx, by)
        p.setPen(QPen(QColor(amber.red(), amber.green(), amber.blue(), 110), 2.0))
        p.setBrush(QBrush(QColor(amber.red(), amber.green(), amber.blue(), 38)))
        p.drawEllipse(bpt, 24, 24)

        p.save()
        p.translate(bpt)
        p.rotate(-pose.board_yaw_deg)
        if pose.board_flat_on_floor:
            self._draw_mini_charuco(p, QRectF(-16, -11, 32, 22), cols=4, rows=3)
        else:
            # Thin rotated bar representing upright board seen from top + subtle face normal tick
            p.setPen(QPen(amber, 3.0))
            p.drawLine(QPointF(-18, 0), QPointF(18, 0))
            p.setPen(QPen(QColor("#38BDF8"), 2.0))
            p.drawLine(QPointF(0, 0), QPointF(0, 12))  # Facing cameras (-Y on screen is +Y up; +Y down is toward front cams)
        p.restore()

        p.setPen(QPen(amber))
        p.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        p.drawText(int(bpt.x() - 48), int(bpt.y() - 28), f"BOARD: {pose.short_label}")

        # Bottom Legend
        p.setPen(QPen(sub_c))
        p.setFont(QFont("Segoe UI", 7.5))
        p.drawText(int(rect.left() + 12), int(rect.bottom() - 8), "● Teal = Phone Tripods   ■ Amber = Where to place ChArUco Board")
        p.restore()

    def _paint_side_view(
        self,
        p: QPainter,
        rect: QRectF,
        pose: CalibrationPoseStep,
        grid_c: QColor,
        text_c: QColor,
        sub_c: QColor,
        accent: QColor,
        amber: QColor,
    ) -> None:
        p.save()
        p.setClipRect(rect.adjusted(1, 1, -1, -1))

        p.setPen(QPen(text_c))
        p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        p.drawText(int(rect.left() + 12), int(rect.top() + 20), "SIDE VIEW (Height & Tilt)")
        p.setPen(QPen(sub_c))
        p.setFont(QFont("Segoe UI", 8))
        p.drawText(int(rect.left() + 12), int(rect.top() + 34), "Tripod 1.5–1.8 m high · Tilt ~15° down")

        floor_y = rect.bottom() - 32
        usable_h = max(90.0, floor_y - (rect.top() + 52))
        ppm = min(usable_h / 2.35, (rect.width() - 36) / 4.2)

        cam_x = rect.left() + 34
        center_x = rect.left() + rect.width() * 0.66

        # Height reference lines (1m, 2m)
        p.setPen(QPen(grid_c, 1.0, Qt.PenStyle.DotLine))
        for hm in (1.0, 2.0):
            yy = floor_y - hm * ppm
            p.drawLine(QPointF(rect.left() + 10, yy), QPointF(rect.right() - 10, yy))
            p.setPen(QPen(sub_c))
            p.setFont(QFont("Segoe UI", 7))
            p.drawText(int(rect.right() - 28), int(yy - 2), f"{hm:.0f}m")
            p.setPen(QPen(grid_c, 1.0, Qt.PenStyle.DotLine))

        # Studio Floor line (Z = 0)
        p.setPen(QPen(text_c, 1.8))
        p.drawLine(QPointF(rect.left() + 10, floor_y), QPointF(rect.right() - 10, floor_y))
        p.setPen(QPen(sub_c))
        p.setFont(QFont("Segoe UI", 7.5))
        p.drawText(int(rect.left() + 12), int(floor_y + 14), "Floor Z = 0.0 m")

        # Tripod + Phone at 1.65 m
        cam_y = floor_y - 1.65 * ppm
        p.setPen(QPen(sub_c, 2.0))
        p.drawLine(QPointF(cam_x, cam_y + 6), QPointF(cam_x - 10, floor_y))
        p.drawLine(QPointF(cam_x, cam_y + 6), QPointF(cam_x + 10, floor_y))
        p.drawLine(QPointF(cam_x, cam_y + 6), QPointF(cam_x, floor_y))

        # Camera FOV cone tilted downward to cover floor-to-overhead at center
        fov_top = QPointF(rect.right() - 12, floor_y - 2.20 * ppm)
        fov_bot = QPointF(rect.right() - 12, floor_y + 2)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(accent.red(), accent.green(), accent.blue(), 26)))
        p.drawPolygon(QPolygonF([QPointF(cam_x, cam_y), fov_top, fov_bot]))
        p.setPen(QPen(QColor(accent.red(), accent.green(), accent.blue(), 90), 1.0, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(cam_x, cam_y), fov_top)
        p.drawLine(QPointF(cam_x, cam_y), fov_bot)

        p.setBrush(QBrush(accent))
        p.setPen(QPen(QColor("#FFFFFF") if self.dark_bg else QColor("#0F172A"), 1.4))
        p.drawRoundedRect(QRectF(cam_x - 7, cam_y - 5, 14, 10), 2, 2)
        p.setPen(QPen(text_c))
        p.setFont(QFont("Segoe UI", 7.5, QFont.Weight.Bold))
        p.drawText(int(cam_x - 16), int(cam_y - 9), "Phone 1.6m")

        # Subtle standing person scale silhouette (1.75 m tall)
        person_c = QColor(sub_c.red(), sub_c.green(), sub_c.blue(), 85)
        p.setPen(QPen(person_c, 2.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        head_y = floor_y - 1.62 * ppm
        p.drawEllipse(QPointF(center_x + 18, head_y), 7, 7)
        p.drawLine(QPointF(center_x + 18, head_y + 7), QPointF(center_x + 18, floor_y - 0.85 * ppm))
        p.drawLine(QPointF(center_x + 18, floor_y - 0.85 * ppm), QPointF(center_x + 10, floor_y))
        p.drawLine(QPointF(center_x + 18, floor_y - 0.85 * ppm), QPointF(center_x + 26, floor_y))

        # Target ChArUco Board at height & pitch
        bx = center_x - 8 + pose.board_xy_m[1] * 18.0
        by = floor_y - pose.board_height_m * ppm
        p.save()
        p.translate(QPointF(bx, by))
        p.rotate(pose.board_pitch_deg)
        if pose.board_flat_on_floor:
            p.setPen(QPen(amber, 4.0))
            p.drawLine(QPointF(0, -18), QPointF(0, 18))
        else:
            self._draw_mini_charuco(p, QRectF(-6, -16, 12, 32), cols=2, rows=4)
        p.restore()

        p.setPen(QPen(amber))
        p.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        lbl_h = "Flat on Floor" if pose.board_flat_on_floor else f"Height ~{pose.board_height_m:.1f} m"
        p.drawText(int(max(rect.left() + 12, bx - 44)), int(max(rect.top() + 50, by - 22)), lbl_h)

        # Distance callout arrow between camera and center
        arr_y = floor_y + 12
        p.setPen(QPen(accent, 1.2))
        p.drawLine(QPointF(cam_x, arr_y), QPointF(center_x, arr_y))
        p.setFont(QFont("Segoe UI", 7.5, QFont.Weight.Bold))
        p.drawText(int((cam_x + center_x) * 0.5 - 24), int(arr_y + 12), "2.5 – 3.2 m")
        p.restore()


class CalibrationGuideWidget(QWidget):
    """Complete Interactive Room, Camera & Board Placement Guide Card."""

    pose_changed = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # Top control bar: Camera count + Pose step pills
        top_bar = QHBoxLayout()
        top_bar.setSpacing(6)
        top_bar.addWidget(QLabel("Room Layout:"))
        self.combo_cams = QComboBox()
        for n in (2, 3, 4, 6, 8):
            self.combo_cams.addItem(f"{n} Cameras", n)
        self.combo_cams.currentIndexChanged.connect(self._on_cam_count_changed)
        top_bar.addWidget(self.combo_cams)

        top_bar.addSpacing(8)
        top_bar.addWidget(QLabel("Board Pose:"))
        self._pose_buttons: List[QPushButton] = []
        for step in CALIBRATION_POSE_STEPS:
            btn = QPushButton(step.short_label)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked=False, idx=step.index: self.set_pose_index(idx))
            top_bar.addWidget(btn)
            self._pose_buttons.append(btn)
        top_bar.addStretch()
        root.addLayout(top_bar)

        # Visual Canvas
        self.canvas = CalibrationGuideCanvas()
        root.addWidget(self.canvas, 1)

        # Clear instruction callout box below diagram
        info_card = QFrame()
        info_card.setObjectName("Card")
        ic = QVBoxLayout(info_card)
        ic.setContentsMargins(14, 10, 14, 10)
        ic.setSpacing(5)

        hdr = QHBoxLayout()
        self.lbl_pose_title = QLabel("")
        self.lbl_pose_title.setStyleSheet("font-size: 13px; font-weight: 700;")
        btn_prev = QPushButton("◀ Prev Pose")
        btn_prev.clicked.connect(lambda: self.set_pose_index((self.canvas.pose_index - 1) % len(CALIBRATION_POSE_STEPS)))
        btn_next = QPushButton("Next Pose ▶")
        btn_next.setObjectName("PrimaryButton")
        btn_next.clicked.connect(lambda: self.set_pose_index((self.canvas.pose_index + 1) % len(CALIBRATION_POSE_STEPS)))
        hdr.addWidget(self.lbl_pose_title, 1)
        hdr.addWidget(btn_prev)
        hdr.addWidget(btn_next)
        ic.addLayout(hdr)

        self.lbl_board_where = QLabel("")
        self.lbl_board_where.setWordWrap(True)
        self.lbl_cam_where = QLabel("")
        self.lbl_cam_where.setObjectName("Subtitle")
        self.lbl_cam_where.setWordWrap(True)
        ic.addWidget(self.lbl_board_where)
        ic.addWidget(self.lbl_cam_where)
        root.addWidget(info_card)

        self.set_num_cameras(2)
        self.set_pose_index(0)

    def set_dark_theme(self, dark: bool) -> None:
        self.canvas.set_dark_bg(dark)

    def set_num_cameras(self, n: int) -> None:
        # Pick closest supported layout preset (2, 3, 4, 6, 8)
        supported = [2, 3, 4, 6, 8]
        chosen = min(supported, key=lambda x: abs(x - max(2, n)))
        idx = self.combo_cams.findData(chosen)
        if idx >= 0:
            self.combo_cams.blockSignals(True)
            self.combo_cams.setCurrentIndex(idx)
            self.combo_cams.blockSignals(False)
        self.canvas.set_num_cameras(chosen)
        self.lbl_cam_where.setText("📷 Camera Placement: " + camera_placement_summary(chosen))

    def _on_cam_count_changed(self) -> None:
        n = int(self.combo_cams.currentData() or 2)
        self.canvas.set_num_cameras(n)
        self.lbl_cam_where.setText("📷 Camera Placement: " + camera_placement_summary(n))

    def set_pose_index(self, idx: int) -> None:
        idx = max(0, min(len(CALIBRATION_POSE_STEPS) - 1, int(idx)))
        self.canvas.set_pose_index(idx)
        step = CALIBRATION_POSE_STEPS[idx]
        for i, b in enumerate(self._pose_buttons):
            b.setChecked(i == idx)
            b.setObjectName("PrimaryButton" if i == idx else "")
            b.style().unpolish(b)
            b.style().polish(b)
        self.lbl_pose_title.setText(step.title)
        self.lbl_board_where.setText(f"🎯 Board Placement: {step.where_to_place}  ({step.why_needed})")
        self.pose_changed.emit(idx)
