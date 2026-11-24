"""Animation Curve Inspector widget for Motion Editor (Raw vs Corrected multi-channel curves + playhead)."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget


class CurveInspectorWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.raw_curves: Optional[np.ndarray] = None  # (T, C)
        self.corrected_curves: Optional[np.ndarray] = None  # (T, C)
        self.current_frame: int = 0
        self.title: str = "Animation Curve Inspector (Select a Joint)"
        self.dark_bg: bool = True

    def set_data(
        self,
        title: str,
        raw_curves: Optional[np.ndarray],
        corrected_curves: Optional[np.ndarray] = None,
        current_frame: int = 0,
    ) -> None:
        self.title = title
        self.raw_curves = raw_curves
        self.corrected_curves = corrected_curves
        self.current_frame = current_frame
        self.update()

    def set_frame(self, frame: int) -> None:
        self.current_frame = frame
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        bg = QColor("#121922") if self.dark_bg else QColor("#FFFFFF")
        p.fillRect(self.rect(), bg)

        w, h = self.width(), self.height()
        pad_l, pad_r, pad_t, pad_b = 12, 12, 24, 14
        plot_w = max(10, w - pad_l - pad_r)
        plot_h = max(10, h - pad_t - pad_b)

        p.setPen(QPen(QColor("#8EA0B5") if self.dark_bg else QColor("#4A5A6B")))
        p.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        p.drawText(pad_l, 16, self.title)

        primary = self.corrected_curves if self.corrected_curves is not None else self.raw_curves
        if primary is None or primary.shape[0] < 2:
            p.drawText(pad_l, h // 2, "Load a motion take to inspect X/Y/Z or quaternion animation curves.")
            p.end()
            return

        T = primary.shape[0]
        all_vals = [primary]
        if self.raw_curves is not None and self.raw_curves.shape == primary.shape:
            all_vals.append(self.raw_curves)
        stack = np.concatenate(all_vals, axis=0)
        vmin = float(np.nanmin(stack))
        vmax = float(np.nanmax(stack))
        if abs(vmax - vmin) < 1e-5:
            vmin -= 0.5
            vmax += 0.5

        colors = [QColor("#EF4444"), QColor("#22C55E"), QColor("#3B82F6"), QColor("#A855F7")]

        def map_pt(t_idx: int, val: float) -> QPointF:
            x = pad_l + (t_idx / max(T - 1, 1)) * plot_w
            y = pad_t + (1.0 - (val - vmin) / (vmax - vmin)) * plot_h
            return QPointF(x, y)

        # Draw dashed raw curves when comparing against corrected
        if self.raw_curves is not None and self.corrected_curves is not None:
            for c in range(min(4, self.raw_curves.shape[1])):
                col = QColor(colors[c])
                col.setAlpha(110)
                pen = QPen(col, 1.1)
                pen.setStyle(Qt.PenStyle.DashLine)
                p.setPen(pen)
                for t in range(T - 1):
                    p.drawLine(map_pt(t, float(self.raw_curves[t, c])), map_pt(t + 1, float(self.raw_curves[t + 1, c])))

        # Draw solid primary curves
        for c in range(min(4, primary.shape[1])):
            p.setPen(QPen(colors[c], 1.8))
            for t in range(T - 1):
                p.drawLine(map_pt(t, float(primary[t, c])), map_pt(t + 1, float(primary[t + 1, c])))

        # Playhead cursor
        px = pad_l + (min(max(self.current_frame, 0), T - 1) / max(T - 1, 1)) * plot_w
        p.setPen(QPen(QColor("#F59E0B"), 1.6))
        p.drawLine(QPointF(px, pad_t), QPointF(px, pad_t + plot_h))
        p.end()
