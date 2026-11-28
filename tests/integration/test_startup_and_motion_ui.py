"""Startup responsiveness, recovery and safe navigation through the actual Qt event loop."""

import threading
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QThread, QTimer, Qt
from PySide6.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QWidget

from desktop.settings.app_settings import AppSettings
from desktop.ui.startup import StartupCoordinator, StartupWorker
from desktop.ui.widgets.studio_chrome import AnimatedStack, configure_studio_font


def _fake_runtime(qapp, *, fail_page=False):
    records = SimpleNamespace(controllers=[], windows=[], widget_threads=[])

    class Controller:
        def __init__(self, path, **kwargs):
            assert QThread.currentThread() == qapp.thread()
            self.settings = kwargs["settings"]
            self.stopped = False
            records.controllers.append(self)

        def shutdown(self):
            self.stopped = True

    class Window(QMainWindow):
        page_count = 3

        def __init__(self, controller, *, deferred_setup):
            assert deferred_setup
            super().__init__()
            self.ctrl = controller
            self.pages = {}
            self.finished_setup = False
            records.windows.append(self)

        def build_next_page(self):
            records.widget_threads.append(QThread.currentThread())
            if fail_page and len(records.windows) == 1 and len(self.pages) == 1:
                raise RuntimeError("Test workspace construction failure")
            if len(self.pages) == self.page_count:
                return None
            name = f"Workspace {len(self.pages) + 1}"
            self.pages[name] = QWidget(self)
            return name

        def finish_setup(self):
            self.finished_setup = True

        def closeEvent(self, event):
            self.ctrl.shutdown()
            super().closeEvent(event)

    settings = AppSettings()
    settings.general.reduce_interface_motion = True
    return (settings, object(), Controller, Window), records


def _worker_factory(payload, *, gate=None, fail_first=False):
    attempts = []

    class Worker(StartupWorker):
        def load_runtime(self):
            attempts.append(self)
            self.progress.emit(1, "Loading engine", "A real test task is running")
            if fail_first and len(attempts) == 1:
                raise RuntimeError("Test import failure")
            if gate is not None:
                gate.wait(8)
            self.checkpoint()
            self.progress.emit(4, "Starting services", "Test task completed")
            return payload

    return Worker


def _cleanup(startup):
    startup.wait_for_worker()
    if startup.window is not None:
        startup.window.close()
    startup.screen.dismiss()
    startup.app.setQuitOnLastWindowClosed(startup._old_quit)


def test_loading_screen_stays_responsive_and_progress_waits_for_real_work(qtbot, qapp, tmp_path):
    configure_studio_font(qapp)
    payload, records = _fake_runtime(qapp)
    gate = threading.Event()
    startup = StartupCoordinator(qapp, tmp_path / "settings.json",
        worker_factory=_worker_factory(payload, gate=gate), exit_on_cancel=False)
    heartbeats = []
    heartbeat = QTimer()
    heartbeat.setInterval(5)
    heartbeat.timeout.connect(lambda: heartbeats.append(1))
    try:
        heartbeat.start()
        startup.start()
        qtbot.waitUntil(lambda: startup.screen.progress.value() == 1)
        qtbot.waitUntil(lambda: len(heartbeats) >= 8)
        assert startup.screen.isVisible() and startup.window is None
        assert startup.screen.progress.value() == 1
        assert not startup.screen.grab().isNull()
        gate.set()
        qtbot.waitUntil(lambda: startup.report.get("passed", False))
        assert startup.window.finished_setup and startup.window.isVisible()
        assert startup.report["completed"] == startup.report["total"] == 9
        values = [item["completed"] for item in startup.history]
        assert values == sorted(values) and values[-4:] == [6, 7, 8, 9]
        assert all(thread == qapp.thread() for thread in records.widget_threads)
    finally:
        heartbeat.stop()
        gate.set()
        _cleanup(startup)


@pytest.mark.parametrize("partial_window", [False, True])
def test_startup_failure_is_reviewable_and_retry_cleans_partial_services(
    qtbot, qapp, tmp_path, partial_window,
):
    payload, records = _fake_runtime(qapp, fail_page=partial_window)
    startup = StartupCoordinator(qapp, tmp_path / "settings.json",
        worker_factory=_worker_factory(payload, fail_first=not partial_window), exit_on_cancel=False)
    try:
        startup.start()
        qtbot.waitUntil(lambda: bool(startup.report) and not startup.report["passed"])
        assert startup.screen.isVisible() and startup.screen.btn_retry.isVisible()
        assert "Test " in (tmp_path / "startup.log").read_text()
        assert startup.controller is None and startup.window is None
        if partial_window:
            assert records.controllers[0].stopped
        startup.screen.btn_retry.click()
        qtbot.waitUntil(lambda: startup.report.get("passed", False))
        assert len(startup.window.pages) == 3 and startup.window.finished_setup
        assert not records.controllers[-1].stopped
    finally:
        _cleanup(startup)


def test_cancel_during_import_keeps_worker_alive_until_safe_exit(qtbot, qapp, tmp_path):
    payload, records = _fake_runtime(qapp)
    gate = threading.Event()
    startup = StartupCoordinator(qapp, tmp_path / "settings.json",
        worker_factory=_worker_factory(payload, gate=gate), exit_on_cancel=False)
    try:
        startup.start()
        qtbot.waitUntil(lambda: startup.screen.progress.value() == 1)
        startup.request_cancel()
        assert startup.worker.isRunning() and startup.screen.isVisible()
        assert not startup.screen.btn_close.isEnabled()
        with qtbot.waitSignal(startup.cancelled, timeout=3000):
            gate.set()
        assert not startup.worker.isRunning() and not startup.screen.isVisible()
        assert not records.controllers and not records.windows
    finally:
        gate.set()
        _cleanup(startup)


def test_cancel_before_worker_starts_does_not_start_an_orphan_thread(qtbot, qapp, tmp_path):
    payload, records = _fake_runtime(qapp)
    startup = StartupCoordinator(qapp, tmp_path / "settings.json",
        worker_factory=_worker_factory(payload), exit_on_cancel=False)
    try:
        startup.start()
        startup.request_cancel()
        qapp.processEvents()
        assert not startup.worker.isRunning()
        assert not records.controllers and not records.windows
    finally:
        _cleanup(startup)


def test_page_fades_block_edits_handle_fast_navigation_and_leave_no_effects(qtbot):
    stack = AnimatedStack()
    qtbot.addWidget(stack)
    clicked = []
    pages = []
    for index in range(3):
        page = QWidget()
        button = QPushButton(f"Edit {index}")
        button.clicked.connect(lambda checked=False, i=index: clicked.append(i))
        QVBoxLayout(page).addWidget(button)
        stack.addWidget(page)
        pages.append(button)
    stack.resize(400, 250)
    stack.show()
    stack.setCurrentIndex(1)
    assert stack._cover is not None and stack.isEnabled()
    qtbot.mouseClick(pages[1], Qt.MouseButton.LeftButton)
    assert not clicked
    stack.setCurrentIndex(2)
    stack.setCurrentIndex(1)
    qtbot.waitUntil(lambda: stack._cover is None)
    qtbot.mouseClick(pages[1], Qt.MouseButton.LeftButton)
    assert clicked == [1]
    assert all(stack.widget(i).graphicsEffect() is None for i in range(3))
    stack.setCurrentIndex(2)
    stack.resize(420, 260)
    assert stack._cover is None
    stack.motion_enabled = False
    stack.setCurrentIndex(0)
    assert stack._cover is None and stack.isEnabled()
    qtbot.mouseClick(pages[0], Qt.MouseButton.LeftButton)
    assert clicked == [1, 0]


def _cpu_hardware():
    from engine.ai_tracking.runtime import HardwareDiagnostics
    return HardwareDiagnostics("test", 4, False, "None", "N/A", 0, 0, "test",
                               ["CPUExecutionProvider"], False, "cpu", ["CPUExecutionProvider"])


def test_real_startup_builds_all_workspaces_and_settings_reduce_motion(
    qtbot, qapp, tmp_path, monkeypatch,
):
    import engine.ai_tracking.runtime as runtime
    from desktop.ui.main_window import NuranovaMainWindow
    from desktop.ui.app_controller import StudioController

    monkeypatch.setattr(runtime, "detect_hardware", lambda **kwargs: _cpu_hardware())
    settings_path = tmp_path / "settings.json"
    settings = AppSettings()
    settings.network.auto_start_server = False
    settings.save(settings_path)
    startup = StartupCoordinator(qapp, settings_path, exit_on_cancel=False)
    try:
        startup.start()
        qtbot.waitUntil(lambda: startup.report.get("passed", False), timeout=15000)
        window = startup.window
        assert isinstance(window, NuranovaMainWindow)
        assert len(window.pages) == startup.report["page_count"] == 21
        assert startup.report["completed"] == startup.report["total"] == 27
        assert window.ctrl.thread() == qapp.thread()
        window._motion_enabled = True
        window._toggle_advanced_drawer()
        window._toggle_advanced_drawer()
        qtbot.waitUntil(lambda: window.nav_list.isHidden())
        assert window.nav_list.height() == 0
        window._toggle_advanced_drawer()
        expanded_height = int(window._drawer_animation.endValue())
        settings_page = window.pages["Settings"]
        settings_page.chk_reduce_motion.setChecked(True)
        settings_page._save()
        assert AppSettings.load(settings_path).general.reduce_interface_motion
        assert not window.stack.motion_enabled
        assert window.nav_list.height() == expanded_height
        window._toggle_advanced_drawer()
        assert window.nav_list.isHidden()
        window.navigate_to("Multi-Video Cinema Lab")
        assert window.stack._cover is None
        assert window.lbl_workspace.text() == "Video motion capture"
        for theme in ("light", "dark"):
            window.apply_theme(theme)
            assert window.pages["Multi-Video Cinema Lab"].review is not None
    finally:
        _cleanup(startup)

    # The same injected hardware/settings also support closing a partially constructed window.
    controller = StudioController(settings_path, settings=settings, hardware=_cpu_hardware())
    window = NuranovaMainWindow(controller, deferred_setup=True)
    qtbot.addWidget(window)
    assert not window.pages
    window.build_next_page()
    with pytest.raises(RuntimeError, match="all workspaces"):
        window.finish_setup()
    window.close()
