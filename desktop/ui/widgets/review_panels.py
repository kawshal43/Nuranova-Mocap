"""Reusable floating preview panels and a resizable review grid."""

from __future__ import annotations

from PySide6.QtCore import QByteArray, QEvent, QSize, Qt, Signal
from PySide6.QtGui import QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class ElidedLabel(QLabel):
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(self.palette().windowText().color())
        painter.drawText(
            self.contentsRect(),
            Qt.AlignmentFlag.AlignVCenter,
            self.fontMetrics().elidedText(
                self.text(), Qt.TextElideMode.ElideRight, self.contentsRect().width()
            ),
        )


class _PreviewWindow(QDialog):
    def __init__(self, panel):
        super().__init__(panel, Qt.WindowType.Window)
        self.panel = panel
        self.closing = False
        self.setWindowTitle(panel.window_title + " — Nuranova")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setSizeGripEnabled(True)
        self.setMinimumSize(360, 280)
        self.resize(1000, 720)
        self.box = QVBoxLayout(self)
        self.box.setContentsMargins(5, 5, 5, 5)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            self.panel.activate()

    def reject(self):
        if self.isFullScreen():
            self.panel.toggle_fullscreen()
        else:
            self.close()

    def closeEvent(self, event):
        if not self.closing:
            self.panel.return_to_grid(from_window=True)
        event.accept()


class PreviewPanel(QWidget):
    """Move the same live view between its grid slot and a modeless window."""

    activated = Signal()
    presentation_changed = Signal()
    floating_changed = Signal(bool)
    command = Signal(str)

    def __init__(self, content, title, *, navigation=True, parent=None):
        super().__init__(parent)
        self.content, self.window_title = content, title
        self.floating_window = None
        self._geometry = ""
        self._normal_geometry = ""
        self._fullscreen_from_grid = False
        self.transport_factory = None
        self.transport = None
        self.setMinimumWidth(180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.box = QVBoxLayout(self)
        self.box.setContentsMargins(0, 0, 0, 0)
        self.surface = QFrame(self)
        self.surface.setObjectName("ReviewPreviewSurface")
        self.surface.setProperty("previewSelected", False)
        self.surface.setStyleSheet("""
            QFrame#ReviewPreviewSurface { border: 1px solid palette(mid); border-radius: 6px; }
            QFrame#ReviewPreviewSurface[previewSelected="true"] { border: 2px solid #0d9488; }
            QFrame#ReviewPreviewHeader { border: 0; border-radius: 4px; }
            QFrame#ReviewPreviewHeader[previewSelected="true"] { background: #0f766e; color: white; }
            QFrame#ReviewPreviewHeader[previewSelected="true"] QLabel { color: white; }
            QLabel#ReviewSelectionBadge { background: #0f766e; color: white; border-radius: 5px; padding: 1px 4px; }
        """)
        body = QVBoxLayout(self.surface)
        body.setContentsMargins(4, 4, 4, 4)
        body.setSpacing(3)
        self.header = QFrame()
        self.header.setObjectName("ReviewPreviewHeader")
        row = QHBoxLayout(self.header)
        row.setContentsMargins(4, 2, 2, 2)
        row.setSpacing(3)
        self.title = ElidedLabel(title)
        self.title.setToolTip(title)
        self.title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.title.setMinimumWidth(45)
        self.badge = QLabel("Selected")
        self.badge.setObjectName("ReviewSelectionBadge")
        self.badge.hide()
        row.addWidget(self.title, 1)
        row.addWidget(self.badge)
        self.pop_button = self._icon_button(
            QStyle.StandardPixmap.SP_TitleBarNormalButton,
            "Pop out into a separate window (Ctrl+Enter)",
            self.toggle_floating,
        )
        self.full_button = self._icon_button(
            QStyle.StandardPixmap.SP_TitleBarMaxButton,
            "Full screen (F11). Escape returns to your previous view.",
            self.toggle_fullscreen,
        )
        self.menu_button = QToolButton()
        self.menu_button.setText("⋯")
        self.menu_button.setToolTip("View options")
        self.menu_button.setAccessibleName("View options for " + title)
        self.menu_button.setAutoRaise(True)
        self.menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.menu = QMenu(self.menu_button)
        self.float_action = self.menu.addAction("Pop out", self.toggle_floating)
        self.menu.addAction("Full screen / return", self.toggle_fullscreen)
        self.menu.addSeparator()
        if navigation:
            self.menu.addAction("Fit view", self.fit_view)
            if hasattr(content, "reset_view"):
                self.menu.addAction("Reset orientation", content.reset_view)
        self.menu.addAction("Save view image…", self.save_snapshot)
        self.menu.addAction("Controls…", self.show_controls)
        self.menu_button.setMenu(self.menu)
        for button in (self.pop_button, self.full_button, self.menu_button):
            row.addWidget(button)
        body.addWidget(self.header)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.status.setMaximumHeight(40)
        self.status.hide()
        body.addWidget(self.status)
        body.addWidget(content, 1)
        self.zoom_value = None
        if navigation:
            nav = QHBoxLayout()
            nav.setContentsMargins(2, 0, 2, 1)
            nav.setSpacing(2)
            minus = QToolButton()
            minus.setText("−")
            minus.setToolTip("Zoom out (−)")
            minus.clicked.connect(lambda: content.zoom_by(1 / 1.25))
            plus = QToolButton()
            plus.setText("+")
            plus.setToolTip("Zoom in (+)")
            plus.clicked.connect(lambda: content.zoom_by(1.25))
            self.zoom_value = QDoubleSpinBox()
            self.zoom_value.setRange(0.1, 6400)
            self.zoom_value.setDecimals(1)
            self.zoom_value.setKeyboardTracking(False)
            self.zoom_value.setSuffix("%")
            self.zoom_value.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
            self.zoom_value.setFixedWidth(73)
            self.zoom_value.setToolTip(
                "Zoom percentage. Enter a value; each view keeps its own zoom."
            )
            self.zoom_value.setAccessibleName("Zoom for " + title)
            self.zoom_value.valueChanged.connect(content.set_zoom_percent)
            fit = QToolButton()
            fit.setText("Fit")
            fit.setToolTip("Fit the whole view (F or 0)")
            fit.clicked.connect(self.fit_view)
            for widget in (minus, self.zoom_value, plus, fit):
                nav.addWidget(widget)
            if getattr(content, "supports_actual_size", False):
                actual = QToolButton()
                actual.setText("1:1")
                actual.setToolTip("One source image pixel per screen pixel (1)")
                actual.clicked.connect(lambda: content.set_zoom_percent(100))
                nav.addWidget(actual)
            nav.addStretch()
            body.addLayout(nav)
            content.navigation_changed.connect(self._navigation_changed)
            content.user_command.connect(self._view_command)
        self.box.addWidget(self.surface)
        self.placeholder = QWidget(self)
        placeholder_box = QVBoxLayout(self.placeholder)
        placeholder_box.addStretch()
        note = QLabel(title + "\nOpen in a separate window")
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        note.setWordWrap(True)
        placeholder_box.addWidget(note)
        for text, callback in (
            ("Show window", self.show_floating),
            ("Return to grid", self.return_to_grid),
        ):
            button = QPushButton(text)
            button.clicked.connect(callback)
            placeholder_box.addWidget(button)
        placeholder_box.addStretch()
        self.placeholder.hide()
        self.setAccessibleName(title)
        for child in [
            self.surface,
            self.header,
            self.title,
            content,
            *content.findChildren(QWidget),
        ]:
            child.installEventFilter(self)
        for sequence, callback in (
            ("F11", self.toggle_fullscreen),
            ("Ctrl+Return", self.toggle_floating),
            ("Ctrl+Enter", self.toggle_floating),
        ):
            shortcut = QShortcut(QKeySequence(sequence), self.surface)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(callback)

    def _icon_button(self, icon, tooltip, callback):
        button = QToolButton()
        button.setIcon(self.style().standardIcon(icon))
        button.setIconSize(QSize(15, 15))
        button.setAutoRaise(True)
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip.split(" (")[0] + " — " + self.window_title)
        button.clicked.connect(callback)
        return button

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Type.MouseButtonPress, QEvent.Type.FocusIn, QEvent.Type.Wheel):
            self.activate()
            if event.type() == QEvent.Type.MouseButtonPress and watched in (self.surface, self.header, self.title):
                self.content.setFocus(Qt.FocusReason.MouseFocusReason)
        return super().eventFilter(watched, event)

    def activate(self):
        self.activated.emit()

    def set_selected(self, selected):
        selected = bool(selected)
        if self.surface.property("previewSelected") == selected:
            return
        for widget in (self.surface, self.header):
            widget.setProperty("previewSelected", selected)
            widget.style().unpolish(widget)
            widget.style().polish(widget)
            widget.update()
        self.badge.setVisible(selected)
        self.header.setStyleSheet(
            "QFrame#ReviewPreviewHeader { background-color: #0f766e; border: 0; border-radius: 4px; } QLabel { color: white; background: transparent; }"
            if selected
            else "QFrame#ReviewPreviewHeader { background: transparent; border: 0; } QLabel { background: transparent; }"
        )
        self.setAccessibleDescription("Selected preview" if selected else "Preview")

    def set_status(self, text):
        self.status.setText(text)
        self.status.setToolTip(text)
        self.status.setVisible(bool(text))

    def fit_view(self):
        if hasattr(self.content, "fit_view"):
            self.content.fit_view()

    def _navigation_changed(self):
        if self.zoom_value is not None:
            self.zoom_value.blockSignals(True)
            self.zoom_value.setValue(self.content.zoom_percent())
            self.zoom_value.blockSignals(False)
        self.presentation_changed.emit()

    def _view_command(self, command):
        if command == "fullscreen":
            self.toggle_fullscreen()
        elif command == "popout":
            self.toggle_floating()
        else:
            self.command.emit(command)

    def toggle_floating(self):
        if self.floating_window is None:
            self.pop_out()
        else:
            self.return_to_grid()

    def pop_out(self):
        if self.floating_window is not None:
            self.show_floating()
            return
        self.activate()
        window = _PreviewWindow(self)
        self.floating_window = window
        self.box.removeWidget(self.surface)
        self.box.addWidget(self.placeholder)
        self.placeholder.show()
        window.box.addWidget(self.surface)
        if self.transport_factory:
            self.transport = self.transport_factory(window)
            window.box.addWidget(self.transport)
        if self._geometry:
            window.restoreGeometry(QByteArray.fromBase64(self._geometry.encode("ascii")))
        self.pop_button.setToolTip("Return this view to the grid (Ctrl+Enter)")
        self.float_action.setText("Return to grid")
        window.show()
        self.surface.show()
        self.content.setFocus()
        self.floating_changed.emit(True)
        self.presentation_changed.emit()

    def show_floating(self):
        if self.floating_window is not None:
            self.floating_window.show()
            self.floating_window.raise_()
            self.floating_window.activateWindow()

    def return_to_grid(self, checked=False, *, from_window=False):
        window = self.floating_window
        if window is None:
            return
        self._geometry = (
            self._normal_geometry
            if window.isFullScreen()
            else bytes(window.saveGeometry().toBase64()).decode("ascii")
        )
        self.floating_window = self.transport = None
        self._fullscreen_from_grid = False
        window.box.removeWidget(self.surface)
        self.box.removeWidget(self.placeholder)
        self.placeholder.hide()
        self.box.addWidget(self.surface)
        self.surface.show()
        self.pop_button.setToolTip("Pop out into a separate window (Ctrl+Enter)")
        self.float_action.setText("Pop out")
        if not from_window:
            window.closing = True
            window.close()
        self.floating_changed.emit(False)
        self.presentation_changed.emit()

    def toggle_fullscreen(self):
        window = self.floating_window
        if window is not None and window.isFullScreen():
            if self._fullscreen_from_grid:
                self.return_to_grid()
            else:
                window.showNormal()
            return
        from_grid = window is None
        if from_grid:
            self.pop_out()
        window = self.floating_window
        self._fullscreen_from_grid = from_grid
        self._normal_geometry = bytes(window.saveGeometry().toBase64()).decode("ascii")
        window.showFullScreen()
        self.content.setFocus()

    def save_snapshot(self):
        path, _ = QFileDialog.getSaveFileName(
            self.window(), "Save this view", "review.png", "PNG image (*.png)"
        )
        if path and not self.content.grab().save(path, "PNG"):
            QMessageBox.warning(
                self.window(), "Image could not be saved", "Choose a writable PNG file path."
            )

    def show_controls(self):
        QMessageBox.information(
            self.window(),
            "Preview controls",
            self.content.toolTip()
            + "\n\nSpace: play/pause · Left/Right: source frame · Shift+Left/Right: ten frames"
            "\nF / 0: fit · + / −: zoom · 1: actual size for video"
            "\nF11: full screen · Ctrl+Enter: pop out / return · Escape: leave full screen / return"
            "\n\nView settings do not change tracking, calibration or exported motion.",
        )

    def presentation_state(self):
        geometry = self._geometry
        if self.floating_window:
            geometry = (
                self._normal_geometry
                if self.floating_window.isFullScreen()
                else bytes(self.floating_window.saveGeometry().toBase64()).decode("ascii")
            )
        return {
            "floating": self.floating_window is not None,
            "geometry": geometry,
            "navigation": self.content.navigation_state()
            if hasattr(self.content, "navigation_state")
            else {},
        }

    def restore_presentation(self, state, *, floating=False):
        self._geometry = str(state.get("geometry", ""))[:20000]
        if hasattr(self.content, "restore_navigation"):
            self.content.restore_navigation(state.get("navigation", {}))
        if floating and state.get("floating"):
            self.pop_out()


class ReviewGrid(QWidget):
    """User-resizable rows and columns; hidden tiles keep their live view state."""

    layout_changed = Signal()

    def __init__(self):
        super().__init__()
        self.box = QVBoxLayout(self)
        self.box.setContentsMargins(0, 0, 0, 0)
        self.parking = QWidget(self)
        self.parking.hide()
        self.tiles = []
        self.rows = []
        self.splitter = None
        self.columns = 0
        self.available_width = 900
        self.tile_height = 260
        self._signature = None
        self._layouts = {}

    def set_tiles(self, tiles, columns=0, width=None):
        if width is not None:
            self.available_width = width
        minimum = max((tile.minimumSizeHint().width() for tile in tiles), default=300)
        count = min(
            len(tiles), columns or max(1, min(3, self.available_width // max(300, minimum)))
        )
        signature = (tuple(tile.objectName() for tile in tiles), count)
        if signature == self._signature:
            return
        if self._signature:
            self._layouts[str(self._signature)] = self.split_sizes()
        for tile in self.tiles:
            tile.hide()
            tile.setParent(self.parking)
        if self.splitter:
            self.box.removeWidget(self.splitter)
            self.splitter.hide()
            self.splitter.deleteLater()
        self.tiles, self.rows, self.columns = list(tiles), [], count
        self._signature = signature
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(7)
        self.splitter.splitterMoved.connect(lambda: self.layout_changed.emit())
        for start in range(0, len(tiles), max(1, count)):
            row = QSplitter(Qt.Orientation.Horizontal)
            row.setChildrenCollapsible(False)
            row.setHandleWidth(7)
            row.splitterMoved.connect(lambda: self.layout_changed.emit())
            for tile in tiles[start : start + count]:
                row.addWidget(tile)
                tile.setMinimumHeight(190)
                tile.show()
            self.splitter.addWidget(row)
            self.rows.append(row)
        self.box.addWidget(self.splitter)
        self.setMinimumHeight(len(self.rows) * self.tile_height + max(0, len(self.rows) - 1) * 7)
        self.setMinimumWidth(count * max(180, minimum) + max(0, count - 1) * 7)
        self.restore_sizes(self._layouts.get(str(signature), {}))

    def split_sizes(self):
        return {
            "vertical": self.splitter.sizes() if self.splitter else [],
            "rows": [row.sizes() for row in self.rows],
        }

    def restore_sizes(self, state):
        vertical = state.get("vertical", [])
        if self.splitter and len(vertical) == len(self.rows):
            self.splitter.setSizes([max(1, min(10000, int(n))) for n in vertical])
        for row, sizes in zip(self.rows, state.get("rows", [])):
            if len(sizes) == row.count():
                row.setSizes([max(1, min(10000, int(n))) for n in sizes])
