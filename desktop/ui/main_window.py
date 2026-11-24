"""Main Desktop Application Window hosting all 20 Nuranova Mocap Studio pages, sidebar navigation,
Light/Dark theme switching, keyboard shortcuts, and live status bar.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from desktop.ui.app_controller import StudioController
from desktop.ui.pages.pages_assets_capture import (
    ActorManagerPage,
    FacialCapturePage,
    HandCapturePage,
    MarkerEditorPage,
)
from desktop.ui.pages.pages_cameras import (
    CalibrationProfilesPage,
    CalibrationWizardPage,
    CameraConfigPage,
    CameraManagerPage,
)
from desktop.ui.pages.pages_capture_edit import (
    CaptureStudioPage,
    MotionEditorPage,
    ProcessingQueuePage,
    RecordingLibraryPage,
)
from desktop.ui.pages.pages_export_system import (
    AboutPage,
    AnimationExportsPage,
    CharacterRetargetingPage,
    DiagnosticsPage,
    SettingsPage,
)
from desktop.ui.pages.pages_workspace import DashboardPage, ProjectsPage, WelcomePage
from desktop.ui.themes.theme import get_stylesheet


class NuranovaMainWindow(QMainWindow):
    """Primary desktop window for Nuranova Mocap Studio v1.0.0."""

    def __init__(self, controller: StudioController | None = None) -> None:
        super().__init__()
        self.ctrl = controller or StudioController()
        self.setWindowTitle("Nuranova Mocap Studio v1.0.0 — Nuranova Solutions")
        self.resize(1380, 860)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left navigation sidebar
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(226)
        side_l = QVBoxLayout(sidebar)
        side_l.setContentsMargins(8, 12, 8, 10)
        side_l.setSpacing(6)

        brand = QLabel("NURANOVA\nMOCAP STUDIO")
        brand.setStyleSheet("font-size: 15px; font-weight: 800; letter-spacing: 1px; padding: 4px 10px;")
        side_l.addWidget(brand)

        self.nav_list = QListWidget()
        self.nav_list.setObjectName("NavList")
        side_l.addWidget(self.nav_list, 1)

        self.btn_theme = QPushButton("Switch Dark / Light Theme")
        self.btn_theme.clicked.connect(self._toggle_theme)
        side_l.addWidget(self.btn_theme)
        main_layout.addWidget(sidebar)

        # Main page stack (all 20 required pages!)
        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack, 1)

        self.pages: Dict[str, QWidget] = {}
        page_specs: List[Tuple[str, QWidget]] = [
            ("Welcome", WelcomePage(self.ctrl, self.navigate_to)),
            ("Dashboard", DashboardPage(self.ctrl, self.navigate_to)),
            ("Projects", ProjectsPage(self.ctrl)),
            ("Camera Manager", CameraManagerPage(self.ctrl)),
            ("Camera Configuration", CameraConfigPage(self.ctrl)),
            ("Calibration Wizard", CalibrationWizardPage(self.ctrl)),
            ("Calibration Profiles", CalibrationProfilesPage(self.ctrl)),
            ("Capture Studio", CaptureStudioPage(self.ctrl, self.navigate_to)),
            ("Recording Library", RecordingLibraryPage(self.ctrl, self.navigate_to)),
            ("Processing Queue", ProcessingQueuePage(self.ctrl)),
            ("Motion Editor", MotionEditorPage(self.ctrl)),
            ("Marker Editor", MarkerEditorPage(self.ctrl)),
            ("Actor Manager", ActorManagerPage(self.ctrl)),
            ("Facial Capture", FacialCapturePage(self.ctrl)),
            ("Hand Capture", HandCapturePage(self.ctrl)),
            ("Character Retargeting", CharacterRetargetingPage(self.ctrl)),
            ("Animation Exports", AnimationExportsPage(self.ctrl)),
            ("Diagnostics", DiagnosticsPage(self.ctrl)),
            ("Settings", SettingsPage(self.ctrl, self.apply_theme)),
            ("About", AboutPage(self.ctrl)),
        ]

        for name, widget in page_specs:
            self.pages[name] = widget
            self.stack.addWidget(widget)
            self.nav_list.addItem(name)

        self.nav_list.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav_list.setCurrentRow(0)

        # Status bar
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.sb_compute = QLabel(f"Compute: {self.ctrl.hardware.selected_device.upper()}")
        self.sb_cams = QLabel("Cameras: 0")
        sb.addPermanentWidget(self.sb_cams)
        sb.addPermanentWidget(self.sb_compute)

        self.ctrl.status_message.connect(lambda m: sb.showMessage(m, 6000))
        self.ctrl.cameras_changed.connect(self._update_status_bar)

        self._setup_shortcuts()
        self.apply_theme(self.ctrl.settings.general.theme)
        self._update_status_bar()

    def navigate_to(self, page_name: str) -> None:
        names = list(self.pages.keys())
        if page_name in names:
            idx = names.index(page_name)
            self.nav_list.setCurrentRow(idx)

    def apply_theme(self, theme_name: str) -> None:
        self.ctrl.settings.general.theme = "dark" if theme_name == "dark" else "light"
        self.setStyleSheet(get_stylesheet(self.ctrl.settings.general.theme))
        dark = (self.ctrl.settings.general.theme == "dark")
        # Sync custom painter widgets
        for pname in ("Capture Studio", "Motion Editor"):
            p = self.pages[pname]
            if hasattr(p, "viewport"):
                p.viewport.canvas.dark_bg = dark
                p.viewport.canvas.update()
            if hasattr(p, "curve_widget"):
                p.curve_widget.dark_bg = dark
                p.curve_widget.update()

    def _toggle_theme(self) -> None:
        nxt = "dark" if self.ctrl.settings.general.theme == "light" else "light"
        self.apply_theme(nxt)
        self.ctrl.settings.save()

    def _update_status_bar(self) -> None:
        n_act = len(self.ctrl.camera_manager.get_active_body_cameras())
        n_tot = len(self.ctrl.camera_manager.list_cameras())
        self.sb_cams.setText(f"Cameras: {n_act}/{n_tot} active")

    def _setup_shortcuts(self) -> None:
        act_undo = QAction(self)
        act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        act_undo.triggered.connect(self.ctrl.undo_motion_edit)
        self.addAction(act_undo)

        act_redo = QAction(self)
        act_redo.setShortcut(QKeySequence.StandardKey.Redo)
        act_redo.triggered.connect(self.ctrl.redo_motion_edit)
        self.addAction(act_redo)

        act_save = QAction(self)
        act_save.setShortcut(QKeySequence.StandardKey.Save)
        act_save.triggered.connect(self.ctrl.project_manager.save_project)
        self.addAction(act_save)

    def closeEvent(self, event) -> None:
        self.ctrl.shutdown()
        super().closeEvent(event)
