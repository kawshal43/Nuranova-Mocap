"""Small native vector graphics and bounded animations for the studio chrome."""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

from PySide6.QtCore import QEvent, Property, QEasingCurve, QPointF, QPropertyAnimation, QRectF, Qt
from PySide6.QtGui import (
    QColor, QFont, QFontDatabase, QIcon, QLinearGradient, QPainter, QPen, QPixmap, QPolygonF,
)
from PySide6.QtWidgets import (
    QApplication, QGraphicsOpacityEffect, QLabel, QPushButton, QStackedWidget, QWidget,
)


def system_reduces_motion() -> bool:
    """Respect Windows' 'Animation effects' accessibility preference."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        enabled = ctypes.c_int(1)
        if ctypes.windll.user32.SystemParametersInfoW(0x1042, 0, ctypes.byref(enabled), 0):
            return not bool(enabled.value)
    except (AttributeError, OSError):
        pass
    return False


def configure_studio_font(app):
    # Qt's offscreen Windows platform can have an empty system font database.
    if sys.platform == "win32" and "Segoe UI" not in QFontDatabase.families():
        folder = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        for name in ("segoeui.ttf", "seguisb.ttf", "segoeuib.ttf"):
            if (folder / name).exists():
                QFontDatabase.addApplicationFont(str(folder / name))
    app.setFont(QFont("Segoe UI", 10))


def paint_brand(painter: QPainter, rect: QRectF) -> None:
    painter.save()
    painter.translate(rect.topLeft())
    painter.scale(rect.width() / 48, rect.height() / 48)
    gradient = QLinearGradient(0, 0, 48, 48)
    gradient.setColorAt(0, QColor("#5EEAD4"))
    gradient.setColorAt(1, QColor("#38BDF8"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#113B43"))
    painter.drawRoundedRect(QRectF(0, 0, 48, 48), 13, 13)
    painter.setPen(QPen(gradient, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    points = [QPointF(13, 34), QPointF(13, 14), QPointF(35, 34), QPointF(35, 14)]
    for a, b in zip(points, points[1:]):
        painter.drawLine(a, b)
    painter.setPen(QPen(QColor("#0E2730"), 1.8))
    painter.setBrush(gradient)
    for point in points:
        painter.drawEllipse(point, 3.7, 3.7)
    painter.restore()


def studio_icon() -> QIcon:
    pixmap = QPixmap(96, 96)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    paint_brand(painter, QRectF(0, 0, 96, 96))
    painter.end()
    return QIcon(pixmap)


class BrandMark(QWidget):
    def __init__(self, parent=None, size: int = 42):
        super().__init__(parent)
        self.setFixedSize(size, size)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        paint_brand(painter, QRectF(self.rect()))


class MotionArtwork(QWidget):
    """A quiet vector motion study; its optional sweep never represents progress."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sweep = 0.0
        self.setMinimumSize(150, 150)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def get_sweep(self):
        return self._sweep

    def set_sweep(self, value):
        self._sweep = float(value)
        self.update()

    sweep = Property(float, get_sweep, set_sweep)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        side = min(self.width(), self.height())
        p.translate((self.width() - side) / 2, (self.height() - side) / 2)
        p.scale(side / 240, side / 240)
        p.setPen(QPen(QColor(94, 234, 212, 22), 1))
        for x in range(20, 241, 25):
            p.drawLine(x, 15, x, 225)
            p.drawLine(15, x, 225, x)
        for radius in (72, 98):
            p.drawEllipse(QPointF(120, 120), radius, radius)
        p.setPen(QPen(QColor("#2DD4BF"), 1.5))
        p.drawArc(QRectF(22, 22, 196, 196), int(self._sweep * 16), 62 * 16)
        # Two restrained trails communicate a motion capture workspace.
        points = [(122, 48), (121, 78), (92, 89), (70, 120), (52, 115),
                  (146, 88), (164, 110), (191, 92), (119, 124), (97, 159),
                  (89, 202), (139, 162), (163, 195)]
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
                 (1, 8), (8, 9), (9, 10), (8, 11), (11, 12)]
        for offset, alpha in ((-10, 24), (-5, 48), (0, 225)):
            p.setPen(QPen(QColor(94, 234, 212, alpha), 2.6,
                          Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            for a, b in edges:
                p.drawLine(QPointF(points[a][0] + offset, points[a][1]),
                           QPointF(points[b][0] + offset, points[b][1]))
        p.setBrush(QColor("#0C2530"))
        p.setPen(QPen(QColor("#67E8F9"), 2))
        for x, y in points:
            p.drawEllipse(QPointF(x, y), 3.5, 3.5)
        p.drawEllipse(QPointF(122, 47), 10, 10)
        angle = math.radians(self._sweep)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#A7F3D0"))
        p.drawEllipse(QPointF(120 + 98 * math.cos(angle), 120 - 98 * math.sin(angle)), 3, 3)


class NavButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self._accent = 0.0
        self.motion_enabled = True
        self._animation = QPropertyAnimation(self, b"accent", self)
        self._animation.setDuration(150)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def get_accent(self):
        return self._accent

    def set_accent(self, value):
        self._accent = float(value)
        self.update()

    accent = Property(float, get_accent, set_accent)

    def set_active(self, active: bool):
        self.setProperty("active", "true" if active else "false")
        self.style().unpolish(self)
        self.style().polish(self)
        self._animation.stop()
        target = 1.0 if active else 0.0
        if self.motion_enabled and self.isVisible():
            self._animation.setStartValue(self._accent)
            self._animation.setEndValue(target)
            self._animation.start()
        else:
            self.set_accent(target)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._accent > .01:
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setOpacity(self._accent)
            p.setPen(QPen(QColor("#5EEAD4"), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            half = max(2, (self.height() / 2 - 10) * self._accent)
            p.drawLine(QPointF(5, self.height() / 2 - half), QPointF(5, self.height() / 2 + half))


class AnimatedStack(QStackedWidget):
    """Fade a snapshot, never install an effect on a live video or 3D widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.motion_enabled = True
        self._cover = None
        self._animation = None

    def finish_transition(self):
        if self._animation is not None:
            self._animation.stop()
            self._animation.deleteLater()
            self._animation = None
        if self._cover is not None:
            QApplication.instance().removeEventFilter(self)
            self._cover.hide()
            self._cover.deleteLater()
            self._cover = None

    def setCurrentIndex(self, index):
        self.finish_transition()
        old = self.currentWidget()
        animate = (self.motion_enabled and self.isVisible() and old is not None
                   and index != self.currentIndex() and 0 <= index < self.count())
        snapshot = old.grab() if animate else None
        super().setCurrentIndex(index)
        if snapshot is None:
            return
        self._cover = QLabel(self)
        QApplication.instance().installEventFilter(self)
        self._cover.setPixmap(snapshot)
        self._cover.setGeometry(self.rect())
        self._cover.show()
        self._cover.raise_()
        effect = QGraphicsOpacityEffect(self._cover)
        self._cover.setGraphicsEffect(effect)
        self._animation = QPropertyAnimation(effect, b"opacity", self)
        self._animation.setDuration(160)
        self._animation.setStartValue(1.0)
        self._animation.setEndValue(0.0)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._animation.finished.connect(self.finish_transition)
        self._animation.start()

    def eventFilter(self, receiver, event):
        # Swallow input during the brief visual overlap without disabling/recoloring live views.
        if self._cover is not None and event.type() in (
            QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease,
            QEvent.Type.MouseButtonDblClick, QEvent.Type.Wheel,
            QEvent.Type.KeyPress, QEvent.Type.KeyRelease, QEvent.Type.Shortcut,
            QEvent.Type.ShortcutOverride, QEvent.Type.TouchBegin,
        ) and isinstance(receiver, QWidget) and (receiver is self or self.isAncestorOf(receiver)):
            return True
        return False

    def resizeEvent(self, event):
        self.finish_transition()
        super().resizeEvent(event)


class WorkspaceIcon(QLabel):
    """Consistent vector outlines at every display scale."""

    def __init__(self, kind, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.setObjectName("IconBadge")
        self.setFixedSize(40, 40)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor("#14B8A6"), 1.7, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.translate(8, 8)
        if self.kind == "pair":
            p.drawRoundedRect(QRectF(3, 2, 13, 20), 2, 2)
            p.drawLine(7, 5, 12, 5)
            p.drawPoint(9, 19)
            p.drawLine(19, 7, 22, 7)
            p.drawLine(19, 12, 22, 12)
        elif self.kind == "capture":
            p.drawRoundedRect(QRectF(1, 5, 15, 14), 3, 3)
            p.drawPolygon(QPolygonF([QPointF(16, 9), QPointF(23, 5),
                                    QPointF(23, 19), QPointF(16, 15)]))
        elif self.kind == "calibrate":
            p.drawRoundedRect(QRectF(3, 3, 18, 18), 2, 2)
            p.drawLine(12, 0, 12, 7)
            p.drawLine(12, 17, 12, 24)
            p.drawLine(0, 12, 7, 12)
            p.drawLine(17, 12, 24, 12)
            p.drawEllipse(QPointF(12, 12), 3, 3)
        elif self.kind == "editor":
            for y, x in ((5, 8), (12, 16), (19, 10)):
                p.drawLine(2, y, x - 3, y)
                p.drawLine(x + 3, y, 22, y)
                p.drawEllipse(QPointF(x, y), 2.5, 2.5)
        elif self.kind == "blender":
            p.drawPolygon(QPolygonF([QPointF(12, 1), QPointF(22, 7), QPointF(22, 18),
                                    QPointF(12, 24), QPointF(2, 18), QPointF(2, 7)]))
            p.drawLine(2, 7, 12, 13)
            p.drawLine(22, 7, 12, 13)
            p.drawLine(12, 13, 12, 24)
        else:
            p.drawPolygon(QPolygonF([QPointF(5, 4), QPointF(20, 12), QPointF(5, 20)]))
