"""Paint first, load libraries in a worker, then build widgets on the GUI thread."""

from __future__ import annotations

import logging
from pathlib import Path
import time
import traceback

from PySide6.QtCore import (
    QObject, QEasingCurve, QPropertyAnimation, QThread, QTimer, Qt, Signal, Slot,
)
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QProgressBar, QPushButton,
    QVBoxLayout, QWidget,
)

from desktop.ui.widgets.studio_chrome import (
    BrandMark, MotionArtwork, studio_icon, system_reduces_motion,
)


STARTUP_TASKS = 27  # preferences, engine, interface, hardware, services, 21 pages, ready


class StartupCancelled(Exception):
    pass


class StartupWorker(QThread):
    progress = Signal(int, str, str)
    motion_policy = Signal(bool)

    def __init__(self, settings_path: Path | None = None, parent=None):
        super().__init__(parent)
        self.settings_path = settings_path
        self.payload = None
        self.error = ""
        self.cancelled = False

    def checkpoint(self):
        if self.isInterruptionRequested():
            raise StartupCancelled()

    def load_runtime(self):
        self.progress.emit(0, "Opening your studio", "Reading preferences")
        from desktop.settings.app_settings import AppSettings
        settings = AppSettings.load(self.settings_path)
        self.motion_policy.emit(not (
            settings.general.reduce_interface_motion or system_reduces_motion()
        ))
        self.checkpoint()
        self.progress.emit(1, "Loading motion engine", "Preparing tracking and reconstruction libraries")
        from desktop.ui.app_controller import StudioController
        self.checkpoint()
        self.progress.emit(2, "Loading studio interface", "Preparing capture, review and export workspaces")
        from desktop.ui.main_window import NuranovaMainWindow
        self.checkpoint()
        self.progress.emit(3, "Checking your hardware", "Checking graphics acceleration and CPU fallback")
        from engine.ai_tracking.runtime import detect_hardware
        hardware = detect_hardware(prefer_gpu=settings.processing.compute_backend != "cpu")
        self.checkpoint()
        self.progress.emit(4, "Starting studio services", f"{hardware.selected_device.upper()} compute ready")
        return settings, hardware, StudioController, NuranovaMainWindow

    def run(self):
        try:
            self.payload = self.load_runtime()
        except StartupCancelled:
            self.cancelled = True
        except Exception:
            self.error = traceback.format_exc()


class StartupScreen(QWidget):
    cancel_requested = Signal()
    retry_requested = Signal()

    def __init__(self):
        super().__init__(None, Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle("Starting Nuranova Mocap Studio")
        self.setWindowIcon(studio_icon())
        self.setObjectName("StartupScreen")
        self.resize(700, 460)
        self.setMinimumWidth(540)
        self._allow_close = False
        self._started = time.monotonic()
        self.motion_enabled = False  # Wait for preferences before starting decorative motion.
        self.setStyleSheet("""
            QWidget#StartupScreen { background: #0B1420; color: #EAF2FA; }
            QLabel { background: transparent; color: #EAF2FA; border: none; font-family: 'Segoe UI'; }
            QLabel#StartupEyebrow { color: #5EEAD4; font-size: 10px; font-weight: 700; letter-spacing: 2px; }
            QLabel#StartupTitle { font-size: 28px; font-weight: 700; }
            QLabel#StartupSub { color: #A5B7CD; font-size: 12px; }
            QLabel#StartupStage { font-size: 15px; font-weight: 600; }
            QLabel#StartupDetail { color: #91A7BF; font-size: 12px; }
            QLabel#StartupMeta { color: #70879F; font-size: 11px; }
            QProgressBar { background: #1C2D40; border: none; border-radius: 3px; min-height: 6px; max-height: 6px; }
            QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #14B8A6,stop:1 #67E8F9); border-radius: 3px; }
            QPushButton { background: #142436; color: #B9CCDF; border: 1px solid #2A3E54; border-radius: 6px; padding: 5px 12px; font-size: 11px; }
            QPushButton:hover { background: #21384B; color: #FFFFFF; border-color: #5EEAD4; }
            QPushButton:focus { border-color: #5EEAD4; }
            QPushButton#StartupRetry { background: #0F766E; color: white; border-color: #14B8A6; }
            QPlainTextEdit { background: #08111C; color: #B9CCDF; border: 1px solid #2A3E54; border-radius: 6px; font-size: 11px; }
        """)
        root = QVBoxLayout(self)
        root.setContentsMargins(36, 26, 36, 22)
        root.setSpacing(0)
        top = QHBoxLayout()
        top.addWidget(BrandMark(size=38))
        wordmark = QLabel("N U R A N O V A")
        wordmark.setStyleSheet("font-size: 12px; font-weight: 700; padding-left: 10px;")
        top.addWidget(wordmark)
        top.addStretch()
        self.btn_close = QPushButton("Close")
        self.btn_close.setAccessibleName("Cancel app startup")
        self.btn_close.clicked.connect(self.cancel_requested)
        top.addWidget(self.btn_close)
        root.addLayout(top)
        root.addSpacing(23)
        hero = QHBoxLayout()
        copy = QVBoxLayout()
        eyebrow = QLabel("MOTION. CAPTURED.")
        eyebrow.setObjectName("StartupEyebrow")
        title = QLabel("Your movement.\nA new dimension.")
        title.setObjectName("StartupTitle")
        sub = QLabel("Nuranova Mocap Studio\nCapture · refine · animate")
        sub.setObjectName("StartupSub")
        copy.addStretch()
        copy.addWidget(eyebrow)
        copy.addSpacing(10)
        copy.addWidget(title)
        copy.addSpacing(10)
        copy.addWidget(sub)
        copy.addStretch()
        hero.addLayout(copy, 1)
        self.artwork = MotionArtwork()
        self.artwork.setFixedSize(220, 220)
        hero.addWidget(self.artwork)
        root.addLayout(hero)
        root.addSpacing(16)
        stage_row = QHBoxLayout()
        self.stage = QLabel("Opening your studio")
        self.stage.setObjectName("StartupStage")
        self.counter = QLabel(f"0 / {STARTUP_TASKS}")
        self.counter.setObjectName("StartupMeta")
        stage_row.addWidget(self.stage, 1)
        stage_row.addWidget(self.counter)
        root.addLayout(stage_row)
        root.addSpacing(8)
        self.progress = QProgressBar()
        self.progress.setRange(0, STARTUP_TASKS)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        root.addWidget(self.progress)
        root.addSpacing(10)
        self.detail = QLabel("Preparing startup")
        self.detail.setObjectName("StartupDetail")
        self.detail.setWordWrap(True)
        root.addWidget(self.detail)
        root.addSpacing(16)
        footer = QHBoxLayout()
        self.elapsed = QLabel("Starting locally")
        self.elapsed.setObjectName("StartupMeta")
        footer.addWidget(self.elapsed, 1)
        version = QLabel("STUDIO 1.0")
        version.setObjectName("StartupMeta")
        footer.addWidget(version)
        root.addLayout(footer)
        self.error_tools = QFrame()
        error_row = QHBoxLayout(self.error_tools)
        error_row.setContentsMargins(0, 12, 0, 0)
        self.btn_details = QPushButton("Show details")
        self.btn_details.clicked.connect(self._toggle_details)
        self.btn_retry = QPushButton("Retry startup")
        self.btn_retry.setObjectName("StartupRetry")
        self.btn_retry.clicked.connect(self.retry_requested)
        error_row.addWidget(self.btn_details)
        error_row.addStretch()
        error_row.addWidget(self.btn_retry)
        root.addWidget(self.error_tools)
        self.error_tools.hide()
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(140)
        self.details.hide()
        root.addWidget(self.details)
        self._clock = QTimer(self)
        self._clock.setInterval(250)
        self._clock.timeout.connect(self._update_elapsed)
        self._orbit = QPropertyAnimation(self.artwork, b"sweep", self)
        self._orbit.setDuration(7500)
        self._orbit.setStartValue(0.)
        self._orbit.setEndValue(360.)
        self._orbit.setLoopCount(-1)
        self._fade = QPropertyAnimation(self, b"windowOpacity", self)
        self._fade.setDuration(140)
        self._fade.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._fade.finished.connect(self.dismiss)

    def paintEvent(self, event):
        p = QPainter(self)
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0, QColor("#0B1420"))
        gradient.setColorAt(.7, QColor("#102332"))
        gradient.setColorAt(1, QColor("#0C3035"))
        p.fillRect(self.rect(), gradient)
        p.setPen(QPen(QColor("#2B4458"), 1))
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))

    def showEvent(self, event):
        self._clock.start()
        if self.motion_enabled:
            self._orbit.start()
        super().showEvent(event)

    def hideEvent(self, event):
        self._clock.stop()
        self._orbit.stop()
        super().hideEvent(event)

    def _update_elapsed(self):
        self.elapsed.setText(f"Starting locally · {time.monotonic() - self._started:.0f}s elapsed")

    def set_motion_enabled(self, enabled):
        self.motion_enabled = enabled
        if not enabled:
            self._orbit.stop()
        elif self.isVisible():
            self._orbit.start()

    def set_progress(self, completed, title, detail):
        self.progress.setValue(completed)
        self.counter.setText(f"{completed} / {self.progress.maximum()}")
        self.stage.setText(title)
        self.detail.setText(detail)
        self.progress.setAccessibleName(f"Startup: {title}")
        self.progress.setAccessibleDescription(detail)

    def show_error(self, message, log_path):
        self._orbit.stop()
        self._clock.stop()
        self.stage.setText("The studio couldn't start")
        self.detail.setText(f"Retry startup, or open details to inspect the error.\nLog: {log_path}")
        self.details.setPlainText(message)
        self.error_tools.show()
        self.btn_close.setEnabled(True)
        self.adjustSize()

    def reset(self):
        self.error_tools.hide()
        self.details.hide()
        self.btn_details.setText("Show details")
        self._started = time.monotonic()
        self.resize(700, 460)
        self._clock.start()
        self.btn_close.setEnabled(True)
        if self.motion_enabled:
            self._orbit.start()

    def _toggle_details(self):
        self.details.setVisible(self.details.isHidden())
        self.btn_details.setText("Hide details" if self.details.isVisible() else "Show details")
        self.adjustSize()

    def finish(self):
        self.btn_close.setEnabled(False)
        self._orbit.stop()
        self._clock.stop()
        if self.motion_enabled and self.isVisible():
            self._fade.setStartValue(1.)
            self._fade.setEndValue(0.)
            self._fade.start()
        else:
            self.dismiss()

    def dismiss(self):
        self._allow_close = True
        self.close()

    def closeEvent(self, event):
        if self._allow_close:
            event.accept()
        else:
            event.ignore()
            self.cancel_requested.emit()


class StartupCoordinator(QObject):
    ready = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    progress_changed = Signal(int, str, str)

    def __init__(self, app, settings_path=None, *, worker_factory=StartupWorker, exit_on_cancel=True):
        super().__init__(app)
        self.app = app
        self.settings_path = Path(settings_path) if settings_path is not None else None
        self.worker_factory = worker_factory
        self.exit_on_cancel = exit_on_cancel
        self.screen = StartupScreen()
        self.screen.cancel_requested.connect(self.request_cancel)
        self.screen.retry_requested.connect(self.start)
        self.worker = None
        self.window = None
        self.controller = None
        self._cancel_requested = False
        self._active = False
        self._old_quit = app.quitOnLastWindowClosed()
        self._page_timer = QTimer(self)
        self._page_timer.setInterval(0)
        self._page_timer.timeout.connect(self._build_page)
        self.history = []
        self.started = 0.
        self.report = {}
        self.app.aboutToQuit.connect(self.wait_for_worker)

    def start(self):
        if self._active or (self.worker is not None and self.worker.isRunning()):
            return
        self._active = True
        self._cancel_requested = False
        self.started = time.monotonic()
        self.history = []
        self.app.setQuitOnLastWindowClosed(False)
        self.screen.reset()
        self.screen.show()
        available = self.screen.screen().availableGeometry()
        self.screen.move(available.center() - self.screen.rect().center())
        self._progress(0, "Opening your studio", "Preparing startup")
        if self.worker is not None:
            self.worker.deleteLater()
        self.worker = self.worker_factory(self.settings_path, self)
        self.worker.progress.connect(self._progress)
        self.worker.motion_policy.connect(self.screen.set_motion_enabled)
        self.worker.finished.connect(self._worker_finished)
        # Return to Qt so the window can paint before imports and hardware checks begin.
        QTimer.singleShot(0, self._start_worker)

    def _start_worker(self):
        if self._active and not self._cancel_requested:
            self.worker.start()

    @Slot(int, str, str)
    def _progress(self, completed, title, detail):
        if self._cancel_requested:
            return
        self.history.append({"completed": completed, "stage": title, "detail": detail,
                             "elapsed_s": round(time.monotonic() - self.started, 3)})
        self.screen.set_progress(completed, title, detail)
        self.progress_changed.emit(completed, title, detail)

    @Slot()
    def _worker_finished(self):
        worker = self.worker
        if self._cancel_requested or worker.cancelled:
            self._finish_cancel()
            return
        if worker.error:
            self._fail(worker.error)
            return
        try:
            settings, hardware, controller_class, window_class = worker.payload
            self.screen.set_motion_enabled(not (
                settings.general.reduce_interface_motion or system_reduces_motion()
            ))
            self.controller = controller_class(self.settings_path, settings=settings, hardware=hardware)
            self.window = window_class(self.controller, deferred_setup=True)
            self.screen.progress.setMaximum(6 + self.window.page_count)
            self._progress(5, "Preparing your workspaces", f"0 / {self.window.page_count} workspaces ready")
            self._page_timer.start()
        except Exception:
            self._fail(traceback.format_exc())

    def _build_page(self):
        if self._cancel_requested:
            self._finish_cancel()
            return
        try:
            name = self.window.build_next_page()
            if name is not None:
                count = len(self.window.pages)
                self._progress(5 + count, "Preparing your workspaces",
                               f"{name} · {count} / {self.window.page_count} ready")
                return
            self._page_timer.stop()
            self.window.finish_setup()
            self._progress(6 + self.window.page_count, "Your studio is ready", "Opening your motion workspace")
            self.report = {"passed": True, "elapsed_s": round(time.monotonic() - self.started, 3),
                           "stages": self.history, "page_count": len(self.window.pages),
                           "completed": self.screen.progress.value(),
                           "total": self.screen.progress.maximum()}
            self.window.show()
            self.app.setQuitOnLastWindowClosed(self._old_quit)
            self._active = False
            self.screen.finish()
            self.ready.emit(self.window)
        except Exception:
            self._fail(traceback.format_exc())

    def _dispose_partial(self):
        self._page_timer.stop()
        if self.window is not None:
            self.window.close()
            self.window.deleteLater()
            self.window = None
            self.controller = None
        elif self.controller is not None:
            self.controller.shutdown()
            self.controller = None

    def _fail(self, message):
        self._dispose_partial()
        self._active = False
        log_path = (self.settings_path.parent if self.settings_path is not None
                    else Path.home() / ".nuranova_mocap") / "startup.log"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(message, encoding="utf-8")
        except OSError:
            logging.getLogger(__name__).exception("Could not write startup error log")
        self.report = {"passed": False, "stages": self.history, "error": message}
        self.screen.show_error(message, log_path)
        self.failed.emit(message)

    @Slot()
    def request_cancel(self):
        self._cancel_requested = True
        self._page_timer.stop()
        self.screen.stage.setText("Closing safely")
        self.screen.detail.setText("Finishing the current startup task…")
        self.screen.btn_close.setEnabled(False)
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
        else:
            self._finish_cancel()

    def _finish_cancel(self):
        self._dispose_partial()
        self._active = False
        self.screen.dismiss()
        self.app.setQuitOnLastWindowClosed(self._old_quit)
        self.cancelled.emit()
        if self.exit_on_cancel:
            self.app.quit()

    @Slot()
    def wait_for_worker(self):
        """Never destroy a running QThread, including during OS/application shutdown."""
        self._page_timer.stop()
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait()
