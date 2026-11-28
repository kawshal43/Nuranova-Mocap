"""Video/annotation and hand viewers with persistent, independent navigation."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap, QTransform
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QLabel,
    QWidget,
)

HAND_EDGES = [(0, 1), (0, 5), (0, 9), (0, 13), (0, 17)] + [
    (i, i + 1) for start in (1, 5, 9, 13, 17) for i in range(start, start + 3)
]


def navigation_key(view, event):
    key = event.key()
    if key in (Qt.Key.Key_F, Qt.Key.Key_0):
        view.fit_view()
    elif key == Qt.Key.Key_1:
        view.set_zoom_percent(100)
    elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
        view.zoom_by(1.25)
    elif key in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
        view.zoom_by(1 / 1.25)
    elif key == Qt.Key.Key_Space:
        view.user_command.emit("play")
    elif key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
        many = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        view.user_command.emit(
            ("previous" if key == Qt.Key.Key_Left else "next") + ("10" if many else "")
        )
    elif key == Qt.Key.Key_E and getattr(view,"supports_actual_size",False):
        view.user_command.emit("edit")
    elif key in (Qt.Key.Key_Home, Qt.Key.Key_End):
        view.user_command.emit("first" if key == Qt.Key.Key_Home else "last")
    else:
        return False
    event.accept()
    return True


class EditablePoint(QGraphicsEllipseItem):
    def __init__(self, key, xy, name, source, callback):
        super().__init__(-4, -4, 8, 8)
        self.key, self.callback = key, callback
        self.original = QPointF(*xy)
        self.setPos(self.original)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setAcceptHoverEvents(True)
        self.setZValue(5)
        self.label = QGraphicsSimpleTextItem(self)
        self.label.setBrush(QBrush(QColor("#ffffff")))
        self.label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.label.setPos(9, -18)
        self.label_background = QGraphicsRectItem(self.label)
        self.label_background.setBrush(QBrush(QColor(12, 20, 34, 235)))
        self.label_background.setPen(QPen(Qt.PenStyle.NoPen))
        self.label_background.setZValue(-1)
        self.label.setVisible(False)
        self.update_point(xy, name, source, False, False)

    def update_point(self, xy, name, source, editing, labels):
        self.setPos(QPointF(*xy))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, editing and source != "fused")
        self.setBrush(
            QBrush(
                QColor(
                    "#38bdf8"
                    if source == "manual"
                    else "#a78bfa"
                    if source == "fused"
                    else "#34d399"
                )
            )
        )
        self.setPen(QPen(QColor("#ffffff"), 1))
        self.setToolTip(name + "\n" + source)
        self.label.setText(name)
        self.label_background.setRect(self.label.boundingRect().adjusted(-3, -2, 3, 2))
        self.always_label = labels
        self.label.setVisible(labels or self.isSelected())

    def hoverEnterEvent(self, event):
        self.label.setVisible(True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.label.setVisible(self.always_label or self.isSelected())
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        self.original = self.pos()
        self.callback(self.key, None)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if (
            self.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            and (self.pos() - self.original).manhattanLength() > 0.01
        ):
            self.callback(self.key, self.pos())


class PointOverlay(QGraphicsView):
    selected = Signal(str)
    edited = Signal(str, object)
    roi_edited = Signal(object)
    navigation_changed = Signal()
    user_command = Signal(str)
    supports_actual_size = True

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setMinimumSize(160, 120)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.native_to_scene = QTransform()
        self._native_scale = 1.0
        self.edit_key = None
        self.editing = False
        self.show_overlays = True
        self.show_labels = False
        self._has_frame = False
        self._fit_mode = True
        self._pending_center = None
        self._pan_start = None
        self._dragging_point = False
        self._points, self._edges = {}, {}
        self._point_data = {}
        self._pixmap = QGraphicsPixmapItem()
        self._pixmap.setZValue(-1)
        self.scene().addItem(self._pixmap)
        self.select_roi = False
        self.roi_start = self.roi_item = None
        self.setBackgroundBrush(QBrush(QColor("#111827")))
        self.message = QLabel("Waiting for frame", self.viewport())
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message.setWordWrap(True)
        self.message.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.message.setStyleSheet("background: #111827; color: #cbd5e1; padding: 12px;")
        self.setToolTip(
            "Wheel: zoom · Drag: pan · Middle/Alt-drag: pan in edit mode · F: fit · 1: actual size · Double-click: full screen · E: edit points"
        )

    def frame(self, image, native_size, points, edges=(), transform=None):
        if self._dragging_point:
            return
        previous_zoom = self.zoom_percent()
        old_rect = self.sceneRect()
        old_center = self.mapToScene(self.viewport().rect().center())
        if old_rect.width() and old_rect.height():
            normalized_center = QPointF(
                old_center.x() / old_rect.width(), old_center.y() / old_rect.height()
            )
        else:
            normalized_center = QPointF(0.5, 0.5)
        rgb = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimage = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self._pixmap.setPixmap(QPixmap.fromImage(qimage))
        self._pixmap.setVisible(True)
        if transform is None:
            nw, nh = native_size
            self._pixmap.setTransform(QTransform.fromScale(nw / w, nh / h))
            self.native_to_scene = QTransform()
        else:
            nw, nh = w, h
            self._pixmap.setTransform(QTransform())
            self.native_to_scene = QTransform(
                float(transform[0, 0]),
                float(transform[1, 0]),
                float(transform[0, 1]),
                float(transform[1, 1]),
                float(transform[0, 2]),
                float(transform[1, 2]),
            )
        self._native_scale = max(
            1e-9, float(np.hypot(self.native_to_scene.m11(), self.native_to_scene.m12()))
        )
        if not self._fit_mode:
            scale = previous_zoom / 100 / self._native_scale
            self.setTransform(QTransform.fromScale(scale, scale))
        self._has_frame = True
        self.message.hide()
        self.scene().setSceneRect(0, 0, nw, nh)
        self._point_data = points
        valid = {
            key
            for key, point in points.items()
            if point.get("valid", True) and np.isfinite(point["xy"]).all()
        }
        for key in self._points.keys() - valid:
            self.scene().removeItem(self._points.pop(key))
        for key in valid:
            point = points[key]
            xy = self.native_to_scene.map(QPointF(*point["xy"]))
            if key not in self._points:
                self._points[key] = EditablePoint(
                    key,
                    (xy.x(), xy.y()),
                    point["name"],
                    point.get("source", "detected"),
                    self._point_event,
                )
                self.scene().addItem(self._points[key])
            mark = self._points[key]
            if not self._dragging_point:
                mark.update_point(
                    (xy.x(), xy.y()),
                    point["name"],
                    point.get("source", "detected"),
                    self.editing,
                    self.show_labels,
                )
            mark.setVisible(self.show_overlays)
        current_edges = {tuple(edge) for edge in edges if all(key in valid for key in edge)}
        for edge in self._edges.keys() - current_edges:
            self.scene().removeItem(self._edges.pop(edge))
        for edge in current_edges:
            if edge not in self._edges:
                line = QGraphicsLineItem()
                pen = QPen(QColor("#22c55e"), 1.5)
                pen.setCosmetic(True)
                line.setPen(pen)
                self.scene().addItem(line)
                self._edges[edge] = line
            a, b = [self._points[key].pos() for key in edge]
            self._edges[edge].setLine(a.x(), a.y(), b.x(), b.y())
            self._edges[edge].setVisible(self.show_overlays)
        if self._fit_mode:
            self._fit(emit=False)
        elif self._pending_center is not None:
            self._restore_center()
        elif old_rect.size() != self.sceneRect().size():
            self.centerOn(normalized_center.x() * nw, normalized_center.y() * nh)

    def placeholder(self, text):
        self._has_frame = False
        self._pixmap.setVisible(False)
        for item in (*self._points.values(), *self._edges.values()):
            item.setVisible(False)
        self.message.setText(text)
        self.message.setGeometry(self.viewport().rect())
        self.message.show()

    def set_editing(self, enabled):
        self.editing = bool(enabled)
        for key, mark in self._points.items():
            mark.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
                self.editing and self._point_data.get(key, {}).get("source") != "fused",
            )
        self.viewport().setCursor(
            Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.OpenHandCursor
        )

    def set_overlays(self, visible, labels=False):
        self.show_overlays, self.show_labels = bool(visible), bool(labels)
        for mark in self._points.values():
            mark.setVisible(self._has_frame and visible)
            mark.always_label = labels
            mark.label.setVisible(labels or mark.isSelected())
        for line in self._edges.values():
            line.setVisible(self._has_frame and visible)

    def _point_event(self, key, xy):
        if xy is None:
            self.scene().clearSelection()
            if key in self._points:
                self._points[key].setSelected(True)
            for item in self._points.values():
                item.label.setVisible(self.show_labels or item.isSelected())
            self.selected.emit(key)
        else:
            inverse, okay = self.native_to_scene.inverted()
            if okay:
                native = inverse.map(xy)
                self.edited.emit(key, [native.x(), native.y()])

    def zoom_percent(self):
        return self.transform().m11() * self._native_scale * 100

    def _fit(self, *, emit):
        before = self.zoom_percent()
        if not self.sceneRect().isEmpty():
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        if emit or abs(self.zoom_percent() - before) > 0.01:
            self.navigation_changed.emit()

    def fit_view(self):
        self._fit_mode = True
        self._pending_center = None
        self._fit(emit=True)

    def set_zoom_percent(self, value, anchor=None):
        if not np.isfinite(value):
            return
        position = anchor or self.viewport().rect().center()
        before = self.mapToScene(position)
        self._fit_mode = False
        scale = float(np.clip(value / 100, 0.001, 64)) / self._native_scale
        self.setTransform(QTransform.fromScale(scale, scale))
        after = self.mapToScene(position)
        center = self.mapToScene(self.viewport().rect().center()) + before - after
        self.centerOn(center)
        self.navigation_changed.emit()

    def zoom_by(self, factor, anchor=None):
        self.set_zoom_percent(self.zoom_percent() * factor, anchor)

    def navigation_state(self):
        rect = self.sceneRect()
        center = self.mapToScene(self.viewport().rect().center())
        return {
            "fit": self._fit_mode,
            "zoom": self.zoom_percent(),
            "center": [center.x() / rect.width(), center.y() / rect.height()]
            if rect.width() and rect.height()
            else [0.5, 0.5],
        }

    def restore_navigation(self, state):
        self._fit_mode = bool(state.get("fit", True))
        zoom = state.get("zoom", 100.0)
        if isinstance(zoom, (int, float)) and np.isfinite(zoom):
            scale = float(np.clip(zoom / 100, 0.001, 64)) / self._native_scale
            self.setTransform(QTransform.fromScale(scale, scale))
        center = state.get("center", [0.5, 0.5])
        if isinstance(center, list) and len(center) == 2 and np.isfinite(center).all():
            self._pending_center = np.clip(center, 0.0, 1.0).tolist()
        if self._has_frame:
            if self._fit_mode:
                self._fit(emit=False)
            else:
                self._restore_center()
        self.navigation_changed.emit()

    def _restore_center(self):
        if self._pending_center is not None and not self.sceneRect().isEmpty():
            self.centerOn(
                self._pending_center[0] * self.sceneRect().width(),
                self._pending_center[1] * self.sceneRect().height(),
            )
            self._pending_center = None

    def mousePressEvent(self, event):
        self.setFocus()
        left = event.button() == Qt.MouseButton.LeftButton
        if self.select_roi and self._has_frame and left:
            self.roi_start = self.mapToScene(event.position().toPoint())
            self.roi_item = QGraphicsRectItem(QRectF(self.roi_start, self.roi_start))
            self.roi_item.setPen(QPen(QColor("#38bdf8"), 3))
            self.scene().addItem(self.roi_item)
            event.accept()
            return
        pan = event.button() == Qt.MouseButton.MiddleButton or (
            left and (not self.editing or event.modifiers() & Qt.KeyboardModifier.AltModifier)
        )
        if pan and self._has_frame:
            item = self._point_at(event.position().toPoint())
            if item is not None and left:
                self._point_event(item.key, None)
            self._pan_start = event.position()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if (
            self._has_frame
            and self.edit_key
            and left
            and self._point_at(event.position().toPoint()) is None
        ):
            self._point_event(self.edit_key, self.mapToScene(event.position().toPoint()))
            return
        self._dragging_point = left and self._point_at(event.position().toPoint()) is not None
        super().mousePressEvent(event)

    def _point_at(self, position):
        item = self.itemAt(position)
        while item is not None and not isinstance(item, EditablePoint):
            item = item.parentItem()
        return item

    def mouseMoveEvent(self, event):
        if self._pan_start is not None:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self._fit_mode = False
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - round(delta.x())
            )
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - round(delta.y()))
            self.navigation_changed.emit()
        elif self.select_roi and self.roi_item is not None:
            self.roi_item.setRect(
                QRectF(self.roi_start, self.mapToScene(event.position().toPoint())).normalized()
            )
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._pan_start is not None:
            self._pan_start = None
            self.viewport().setCursor(
                Qt.CursorShape.CrossCursor if self.editing else Qt.CursorShape.OpenHandCursor
            )
        elif self.select_roi and self.roi_item is not None:
            rectangle = self.roi_item.rect()
            self.scene().removeItem(self.roi_item)
            self.roi_item, self.select_roi = None, False
            inverse, okay = self.native_to_scene.inverted()
            if okay:
                a, b = inverse.map(rectangle.topLeft()), inverse.map(rectangle.bottomRight())
                self.roi_edited.emit([a.x(), a.y(), b.x() - a.x(), b.y() - a.y()])
        else:
            super().mouseReleaseEvent(event)
        self._dragging_point = False

    def resizeEvent(self, event):
        center = self.mapToScene(self.viewport().rect().center())
        super().resizeEvent(event)
        self.message.setGeometry(self.viewport().rect())
        if self._fit_mode:
            self._fit(emit=False)
        else:
            self.centerOn(center)

    def wheelEvent(self, event):
        if event.angleDelta().y():
            self.zoom_by(1.2 ** (event.angleDelta().y() / 120), event.position().toPoint())
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and self.select_roi:
            if self.roi_item is not None:
                self.scene().removeItem(self.roi_item)
            self.roi_item, self.select_roi = None, False
            event.accept()
        elif not navigation_key(self, event):
            super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and not self.editing:
            self.user_command.emit("fullscreen")
        else:
            super().mouseDoubleClickEvent(event)


class HandPoseView(QWidget):
    navigation_changed = Signal()
    user_command = Signal(str)
    supports_actual_size = False

    def __init__(self):
        super().__init__()
        self.setMinimumSize(160, 150)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.points = self.reference = None
        self.center = True
        self.caption = "3D hand available after solving"
        self.zoom, self.yaw, self.pitch = 1.0, 0.35, -0.25
        self.pan = np.zeros(2)
        self._drag = None
        self.setToolTip(
            "Drag: orbit · Middle/right drag: pan · Wheel: zoom · F: fit · Double-click: full screen"
        )

    def set_hand(self, points, caption, center=True):
        self.points = np.asarray(points, float) if points is not None else None
        self.caption, self.center = caption, center
        if self.reference is None and self.points is not None and np.isfinite(self.points).all():
            self.reference = self.points[0].copy()
        self.update()

    def _changed(self):
        self.update()
        self.navigation_changed.emit()

    def zoom_percent(self):
        return self.zoom * 100

    def set_zoom_percent(self, value):
        if np.isfinite(value):
            self.zoom = float(np.clip(value / 100, 0.001, 64))
            self._changed()

    def zoom_by(self, factor):
        self.set_zoom_percent(self.zoom_percent() * factor)

    def fit_view(self):
        self.zoom, self.pan = 1.0, np.zeros(2)
        if not self.center and self.points is not None:
            self.reference = self.points[0].copy()
        self._changed()

    def reset_view(self):
        self.yaw, self.pitch = 0.35, -0.25
        self.fit_view()

    def navigation_state(self):
        return {
            "zoom": self.zoom_percent(),
            "yaw": self.yaw,
            "pitch": self.pitch,
            "pan": self.pan.tolist(),
        }

    def restore_navigation(self, state):
        values = [
            state.get("zoom", 100),
            state.get("yaw", 0.35),
            state.get("pitch", -0.25),
            *state.get("pan", [0.0, 0.0]),
        ]
        if len(values) == 5 and np.isfinite(values).all():
            self.zoom = float(np.clip(values[0] / 100, 0.001, 64))
            self.yaw, self.pitch = (
                float(values[1]),
                float(np.clip(values[2], -np.pi / 2, np.pi / 2)),
            )
            self.pan = np.clip(values[3:], -10.0, 10.0)
        self._changed()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0c1422"))
        painter.setPen(QColor("#cbd5e1"))
        if self.points is None or not np.isfinite(self.points).all():
            painter.drawText(
                self.rect().adjusted(10, 10, -10, -10),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                self.caption,
            )
            return
        painter.drawText(
            QRectF(6, 4, self.width() - 12, 40), Qt.TextFlag.TextWordWrap, self.caption
        )
        pts = self.points - (self.points[0] if self.center else self.reference)
        cy, sy, cp, sp = np.cos(self.yaw), np.sin(self.yaw), np.cos(self.pitch), np.sin(self.pitch)
        rotation = np.array([[cy, 0, sy], [sp * sy, cp, -sp * cy], [-cp * sy, sp, cp * cy]])
        coordinates = (pts @ rotation.T)[:, :2] * [1, -1]
        if self.center:
            coordinates -= (coordinates.min(axis=0) + coordinates.max(axis=0)) / 2
        scale = min((self.width() - 24) / 0.24, (self.height() - 52) / 0.24) * self.zoom
        xy = (
            coordinates * scale
            + [self.width() / 2, (self.height() + 36) / 2]
            + self.pan * [self.width(), self.height()]
        )
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#34d399"), 2))
        for a, b in HAND_EDGES:
            painter.drawLine(QPointF(*xy[a]), QPointF(*xy[b]))
        painter.setBrush(QColor("#e2e8f0"))
        for point in xy:
            painter.drawEllipse(QPointF(*point), 3, 3)

    def mousePressEvent(self, event):
        self.setFocus()
        self._drag = event.position()
        self._pan_mode = event.button() in (
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.RightButton,
        ) or bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
        event.accept()

    def mouseMoveEvent(self, event):
        if self._drag is not None:
            delta = event.position() - self._drag
            self._drag = event.position()
            if self._pan_mode:
                self.pan += [delta.x() / max(1, self.width()), delta.y() / max(1, self.height())]
            else:
                self.yaw += delta.x() * 0.01
                self.pitch = float(np.clip(self.pitch + delta.y() * 0.01, -np.pi / 2, np.pi / 2))
            self._changed()

    def mouseReleaseEvent(self, event):
        self._drag = None

    def wheelEvent(self, event):
        if event.angleDelta().y():
            self.zoom_by(1.2 ** (event.angleDelta().y() / 120))
        event.accept()

    def keyPressEvent(self, event):
        if not navigation_key(self, event):
            super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.user_command.emit("fullscreen")
