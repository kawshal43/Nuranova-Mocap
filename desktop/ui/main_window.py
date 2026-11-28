"""Main Desktop Application Window with Guided 6-Step Studio Workflow Bar, Clean Grouped Sidebar,
Collapsible All-Tools Drawer (all 20 pages preserved), Auto-Start Server, Light/Dark Themes, and Shortcuts.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Tuple

from PySide6.QtCore import Property, QEasingCurve, QPropertyAnimation, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
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
from desktop.ui.widgets.studio_chrome import (
    AnimatedStack, BrandMark, NavButton, studio_icon, system_reduces_motion,
)
from multi_video_studio.ui_page import MultiVideoCinemaLabPage


PRIMARY_NAV_STEPS: List[Tuple[str, str]] = [
    ("1. Home && Project", "Welcome"),
    ("2. Cameras && Pair", "Camera Manager"),
    ("3. Calibrate", "Calibration Wizard"),
    ("4. Capture Studio", "Capture Studio"),
    ("5. Motion Editor", "Motion Editor"),
    ("6. Blender && Export", "Character Retargeting"),
]


class NuranovaMainWindow(QMainWindow):
    """Primary desktop window for Nuranova Mocap Studio v1.0.0."""

    def __init__(
        self, controller: StudioController | None = None, *, deferred_setup: bool = False,
    ) -> None:
        super().__init__()
        self.ctrl = controller or StudioController()
        self._initialized = False
        self._closing = False
        self._tools_expanded = False
        self._motion_enabled = not (
            self.ctrl.settings.general.reduce_interface_motion or system_reduces_motion()
        )
        self.setWindowTitle("Nuranova Mocap Studio v1.0.0 — Nuranova Solutions")
        self.setWindowIcon(studio_icon())
        self.resize(1400, 880)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ---------------- Left Sidebar ----------------
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(236)
        side_l = QVBoxLayout(sidebar)
        side_l.setContentsMargins(14, 22, 14, 14)
        side_l.setSpacing(6)

        brand_row = QHBoxLayout()
        brand_row.setSpacing(11)
        brand_row.addWidget(BrandMark())
        brand_text = QVBoxLayout()
        brand_text.setSpacing(0)
        brand = QLabel("NURANOVA")
        brand.setObjectName("BrandName")
        edition = QLabel("M O C A P   S T U D I O")
        edition.setObjectName("BrandEdition")
        brand_text.addWidget(brand)
        brand_text.addWidget(edition)
        brand_row.addLayout(brand_text, 1)
        side_l.addLayout(brand_row)
        side_l.addSpacing(18)

        lbl_wf = QLabel("STUDIO WORKFLOW")
        lbl_wf.setStyleSheet(
            "font-size: 10px; font-weight: 700; letter-spacing: 0.8px; color: #7D91A8; padding: 4px 8px 2px 8px;"
        )
        side_l.addWidget(lbl_wf)

        self._primary_buttons: Dict[str, QPushButton] = {}
        for label, target_page in PRIMARY_NAV_STEPS:
            btn = NavButton(label)
            btn.motion_enabled = self._motion_enabled
            btn.setObjectName("NavPrimary")
            btn.clicked.connect(lambda checked=False, p=target_page: self.navigate_to(p))
            side_l.addWidget(btn)
            self._primary_buttons[target_page] = btn

        # Dedicated Separate Module: Multi-Video Cinema Lab
        lbl_mv = QLabel("OFFLINE CINEMA SUITE")
        lbl_mv.setStyleSheet(
            "font-size: 10px; font-weight: 700; letter-spacing: 0.8px; color: #14B8A6; padding: 8px 8px 2px 8px;"
        )
        side_l.addWidget(lbl_mv)
        btn_mv_lab = NavButton("Import && Solve Videos")
        btn_mv_lab.motion_enabled = self._motion_enabled
        btn_mv_lab.setObjectName("NavPrimary")
        btn_mv_lab.clicked.connect(lambda checked=False: self.navigate_to("Multi-Video Cinema Lab"))
        side_l.addWidget(btn_mv_lab)
        self._primary_buttons["Multi-Video Cinema Lab"] = btn_mv_lab

        # Collapsible Advanced / All 21 Pages Drawer
        self.btn_more_tools = QPushButton("All tools  +")
        self.btn_more_tools.setObjectName("SectionToggle")
        self.btn_more_tools.clicked.connect(self._toggle_advanced_drawer)
        side_l.addWidget(self.btn_more_tools)

        self.nav_list = QListWidget()
        self.nav_list.setObjectName("NavList")
        self.nav_list.setVisible(False)  # Collapsed by default so beginners see only 6 clean steps!
        self.nav_list.setFixedHeight(0)
        side_l.addWidget(self.nav_list)
        self._drawer_animation = QPropertyAnimation(self, b"drawerHeight", self)
        self._drawer_animation.setDuration(180)
        self._drawer_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._drawer_animation.finished.connect(self._finish_drawer)

        self._sidebar_spacer = QWidget()
        self._sidebar_spacer.setObjectName("SidebarSpacer")
        side_l.addWidget(self._sidebar_spacer, 1)

        # Quick utility shortcuts at bottom of sidebar
        util_row = QHBoxLayout()
        util_row.setSpacing(6)
        btn_settings = QPushButton("Settings")
        btn_settings.clicked.connect(lambda: self.navigate_to("Settings"))
        self.btn_theme = QPushButton("Light / Dark")
        self.btn_theme.clicked.connect(self._toggle_theme)
        util_row.addWidget(btn_settings)
        util_row.addWidget(self.btn_theme)
        side_l.addLayout(util_row)
        main_layout.addWidget(sidebar)

        # ---------------- Right Workspace (Top Guided Bar + Page Stack) ----------------
        right_col = QWidget()
        right_l = QVBoxLayout(right_col)
        right_l.setContentsMargins(0, 0, 0, 0)
        right_l.setSpacing(0)

        # Top Guided Workflow Stepper Bar
        wf_bar = QFrame()
        wf_bar.setObjectName("WorkflowBar")
        wf_l = QHBoxLayout(wf_bar)
        wf_l.setContentsMargins(14, 8, 14, 8)
        wf_l.setSpacing(6)

        context = QVBoxLayout()
        context.setSpacing(1)
        self.lbl_workspace = QLabel("Studio home")
        self.lbl_workspace.setObjectName("WorkspaceTitle")
        self.lbl_project = QLabel("Ready for your next take")
        self.lbl_project.setObjectName("WorkspaceContext")
        context.addWidget(self.lbl_workspace)
        context.addWidget(self.lbl_project)
        wf_l.addLayout(context, 1)

        self._step_pills: Dict[str, QPushButton] = {}
        for label, target_page in PRIMARY_NAV_STEPS:
            pill = QPushButton(label, self)
            pill.setObjectName("StepPill")
            pill.clicked.connect(lambda checked=False, p=target_page: self.navigate_to(p))
            pill.hide()  # Workflow navigation lives in the sidebar.
            self._step_pills[target_page] = pill

        self.btn_video_only = QPushButton("Import videos")
        self.btn_video_only.setObjectName("StepPill")
        self.btn_video_only.setToolTip(
            "Open Dedicated Multi-Video Cinema Lab (.ZIP / Folder / In-App Multi-Cam -> CUDA RTMW-x + 30 Fingers + 6-DoF Head + Zero-Slide)"
        )
        self.btn_video_only.clicked.connect(lambda: self.navigate_to("Multi-Video Cinema Lab"))
        wf_l.addWidget(self.btn_video_only)

        self.btn_cam_mode = QPushButton("Multi-camera")
        self.btn_cam_mode.setObjectName("StepPill")
        self.btn_cam_mode.setToolTip(
            "Toggle between 1-Cam AI Mode (no calibration needed) and Multi-Cam Studio Mode (2–12 phones)"
        )
        self.btn_cam_mode.clicked.connect(
            lambda: self.ctrl.set_single_camera_mode(
                not self.ctrl.settings.general.single_camera_mode
            )
        )
        wf_l.addWidget(self.btn_cam_mode)

        self.btn_server_pill = QPushButton("Server: Offline")
        self.btn_server_pill.setObjectName("StepPill")
        self.btn_server_pill.setToolTip("Click to open Phone Pairing & QR Code")
        self.btn_server_pill.clicked.connect(lambda: self.navigate_to("Camera Manager"))
        wf_l.addWidget(self.btn_server_pill)

        self.btn_next_step = QPushButton("Next Step →")
        self.btn_next_step.setObjectName("PrimaryButton")
        self.btn_next_step.clicked.connect(self._go_next_step)
        wf_l.addWidget(self.btn_next_step)

        right_l.addWidget(wf_bar)

        # Main page stack (all 20 pages preserved!)
        self.stack = AnimatedStack()
        self.stack.motion_enabled = self._motion_enabled
        right_l.addWidget(self.stack, 1)
        main_layout.addWidget(right_col, 1)

        self.pages: Dict[str, QWidget] = {}
        self._page_factories: List[Tuple[str, Callable[[], QWidget]]] = [
            ("Welcome", lambda: WelcomePage(self.ctrl, self.navigate_to)),
            ("Dashboard", lambda: DashboardPage(self.ctrl, self.navigate_to)),
            ("Projects", lambda: ProjectsPage(self.ctrl)),
            ("Camera Manager", lambda: CameraManagerPage(self.ctrl)),
            ("Camera Configuration", lambda: CameraConfigPage(self.ctrl)),
            ("Calibration Wizard", lambda: CalibrationWizardPage(self.ctrl)),
            ("Calibration Profiles", lambda: CalibrationProfilesPage(self.ctrl)),
            ("Capture Studio", lambda: CaptureStudioPage(self.ctrl, self.navigate_to)),
            ("Multi-Video Cinema Lab", lambda: MultiVideoCinemaLabPage(self.ctrl, self.navigate_to)),
            ("Recording Library", lambda: RecordingLibraryPage(self.ctrl, self.navigate_to)),
            ("Processing Queue", lambda: ProcessingQueuePage(self.ctrl)),
            ("Motion Editor", lambda: MotionEditorPage(self.ctrl)),
            ("Marker Editor", lambda: MarkerEditorPage(self.ctrl)),
            ("Actor Manager", lambda: ActorManagerPage(self.ctrl)),
            ("Facial Capture", lambda: FacialCapturePage(self.ctrl)),
            ("Hand Capture", lambda: HandCapturePage(self.ctrl)),
            ("Character Retargeting", lambda: CharacterRetargetingPage(self.ctrl)),
            ("Animation Exports", lambda: AnimationExportsPage(self.ctrl)),
            ("Diagnostics", lambda: DiagnosticsPage(self.ctrl)),
            ("Settings", lambda: SettingsPage(self.ctrl, self.apply_theme)),
            ("About", lambda: AboutPage(self.ctrl)),
        ]

        self.page_scrolls: Dict[str, QScrollArea] = {}
        # Status bar (initialize before selecting row 0 so _refresh_workflow_bar sees sb_cams)
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.sb_compute = QLabel(f"Compute: {self.ctrl.hardware.selected_device.upper()}")
        self.sb_cams = QLabel("Cameras: 0")
        sb.addPermanentWidget(self.sb_cams)
        sb.addPermanentWidget(self.sb_compute)

        self.ctrl.status_message.connect(lambda m: sb.showMessage(m, 6000))
        self.setStyleSheet(get_stylesheet(self.ctrl.settings.general.theme))
        if not deferred_setup:
            while self.build_next_page() is not None:
                pass
            self.finish_setup()

    @property
    def page_count(self) -> int:
        return len(self._page_factories)

    def build_next_page(self) -> str | None:
        """Construct one real page per event-loop turn while startup stays visible."""
        if len(self.pages) == self.page_count:
            return None
        name, factory = self._page_factories[len(self.pages)]
        widget = factory()
        self.pages[name] = widget
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setMinimumSize(0, 0)
        scroll.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        scroll.setWidget(widget)
        self.page_scrolls[name] = scroll
        self.stack.addWidget(scroll)
        self.nav_list.addItem(name)
        return name

    def finish_setup(self) -> None:
        if self._initialized:
            return
        if len(self.pages) != self.page_count:
            raise RuntimeError("The studio cannot open until all workspaces are ready.")
        self.nav_list.currentRowChanged.connect(self._on_page_index_changed)
        self.nav_list.setCurrentRow(0)
        self.ctrl.cameras_changed.connect(self._refresh_workflow_bar)
        self.ctrl.project_changed.connect(self._refresh_workflow_bar)
        self.ctrl.calibration_changed.connect(self._refresh_workflow_bar)
        self.ctrl.motion_changed.connect(self._refresh_workflow_bar)
        self.ctrl.server_state_changed.connect(lambda ok, msg: self._refresh_workflow_bar())
        self.ctrl.recording_state_changed.connect(lambda rec, msg: self._refresh_workflow_bar())
        self.pages["Multi-Video Cinema Lab"].workflow_changed.connect(self._refresh_workflow_bar)

        self._setup_shortcuts()
        self.apply_theme(self.ctrl.settings.general.theme)
        self._initialized = True
        self._refresh_workflow_bar()

        # Auto-start local LAN server for instant phone discovery & QR pairing (skipped in offscreen tests)
        if (
            self.ctrl.settings.network.auto_start_server
            and os.environ.get("QT_QPA_PLATFORM") != "offscreen"
        ):
            QTimer.singleShot(250, self._auto_start_server)

    def _auto_start_server(self) -> None:
        if not self._closing and not self.ctrl.server_running:
            self.ctrl.start_server()
            self._refresh_workflow_bar()

    def _toggle_advanced_drawer(self) -> None:
        self._tools_expanded = not self._tools_expanded
        self._drawer_animation.stop()
        self.btn_more_tools.setText("All tools  −" if self._tools_expanded else "All tools  +")
        self.btn_more_tools.setAccessibleName("Collapse all tools" if self._tools_expanded else "Expand all tools")
        available = self._sidebar_spacer.height() + self.nav_list.height() - 6
        target = min(300, max(80, available)) if self._tools_expanded else 0
        if self._tools_expanded:
            self.nav_list.show()
        if self._motion_enabled and self.isVisible():
            self._drawer_animation.setStartValue(self.nav_list.height())
            self._drawer_animation.setEndValue(target)
            self._drawer_animation.start()
        else:
            self.set_drawer_height(target)
            self._finish_drawer()

    def get_drawer_height(self):
        return self.nav_list.height()

    def set_drawer_height(self, value):
        self.nav_list.setFixedHeight(int(value))

    drawerHeight = Property(int, get_drawer_height, set_drawer_height)

    def _finish_drawer(self):
        if not self._tools_expanded:
            self.nav_list.hide()

    def _on_page_index_changed(self, idx: int) -> None:
        if idx < 0:
            return
        self.stack.setCurrentIndex(idx)
        names = list(self.pages.keys())
        current_name = names[idx] if idx < len(names) else ""
        self.lbl_workspace.setText("Studio home" if current_name == "Welcome" else (
            "Video motion capture" if current_name == "Multi-Video Cinema Lab" else current_name
        ))
        # Map secondary pages to their primary workflow bucket for clear visual highlight
        bucket_map = {
            "Welcome": "Welcome",
            "Dashboard": "Welcome",
            "Projects": "Welcome",
            "Camera Manager": "Camera Manager",
            "Camera Configuration": "Camera Manager",
            "Calibration Wizard": "Calibration Wizard",
            "Calibration Profiles": "Calibration Wizard",
            "Capture Studio": "Capture Studio",
            "Recording Library": "Capture Studio",
            "Processing Queue": "Capture Studio",
            "Motion Editor": "Motion Editor",
            "Character Retargeting": "Character Retargeting",
            "Animation Exports": "Character Retargeting",
            "Multi-Video Cinema Lab": "Multi-Video Cinema Lab",
        }
        active_bucket = bucket_map.get(current_name, "")
        for key, btn in self._primary_buttons.items():
            btn.set_active(key == active_bucket)
        self._refresh_workflow_bar()

    def navigate_to(self, page_name: str) -> None:
        names = list(self.pages.keys())
        if page_name in names:
            idx = names.index(page_name)
            self.nav_list.setCurrentRow(idx)

    def _go_next_step(self) -> None:
        if self.stack.currentIndex() == list(self.pages).index("Multi-Video Cinema Lab"):
            page = self.pages["Multi-Video Cinema Lab"]
            if page._package is None:
                page._import_files()
            elif page._last_result is not None:
                page.workflow_tabs.setCurrentIndex(1)
            else:
                page._start_cinema_solve()
            return
        single_cam = self.ctrl.settings.general.single_camera_mode
        if self.ctrl.project_manager.active_project is None:
            self.navigate_to("Welcome")
        elif len(self.ctrl.camera_manager.get_active_recording_cameras()) == 0:
            self.navigate_to("Camera Manager")
        elif (not single_cam) and self.ctrl.active_calibration is None:
            self.navigate_to("Calibration Wizard")
        elif self.ctrl.active_motion is None:
            self.navigate_to("Capture Studio")
        else:
            self.navigate_to("Character Retargeting")

    def _refresh_workflow_bar(self) -> None:
        project = self.ctrl.project_manager.active_project
        self.lbl_project.setText(project.name if project is not None else "Ready for your next take")
        single_cam = self.ctrl.settings.general.single_camera_mode
        has_proj = self.ctrl.project_manager.active_project is not None
        active_cams = len(self.ctrl.camera_manager.get_active_recording_cameras())
        body_cams = len(self.ctrl.camera_manager.get_active_body_cameras())
        total_cams = len(self.ctrl.camera_manager.list_cameras())
        has_calib = self.ctrl.active_calibration is not None
        calib_satisfied = has_calib or single_cam
        cams_ready = (active_cams >= 1) if single_cam else (body_cams >= 2)
        has_motion = self.ctrl.active_motion is not None

        self.sb_cams.setText(
            f"Cameras: {active_cams}/{total_cams} online · {'1-Cam AI' if single_cam else 'Multi-Cam'}"
        )

        # Update top 1-Cam / Multi-Cam toggle pill
        self.btn_cam_mode.setText("Single-camera AI" if single_cam else "Multi-camera")
        self.btn_cam_mode.setProperty("stepState", "done" if single_cam else "todo")
        self.btn_cam_mode.style().unpolish(self.btn_cam_mode)
        self.btn_cam_mode.style().polish(self.btn_cam_mode)

        # Update sidebar Step 3 label to show it's optional in 1-Cam Mode
        if "Calibration Wizard" in self._primary_buttons:
            self._primary_buttons["Calibration Wizard"].setText(
                "3. Calibrate (Optional)" if single_cam else "3. Calibrate"
            )

        names = list(self.pages.keys())
        cur_idx = self.stack.currentIndex()
        cur_page = names[cur_idx] if 0 <= cur_idx < len(names) else ""
        offline = cur_page == "Multi-Video Cinema Lab"
        self.btn_video_only.setVisible(not offline)
        self.btn_cam_mode.setVisible(not offline)
        self.btn_server_pill.setVisible(not offline)

        states = {
            "Welcome": (
                "done"
                if has_proj
                else ("active" if cur_page in ("Welcome", "Projects", "Dashboard") else "todo")
            ),
            "Camera Manager": (
                "done"
                if cams_ready
                else (
                    "active"
                    if cur_page in ("Camera Manager", "Camera Configuration") or active_cams >= 1
                    else "todo"
                )
            ),
            "Calibration Wizard": (
                "done"
                if calib_satisfied
                else (
                    "active"
                    if cur_page in ("Calibration Wizard", "Calibration Profiles")
                    else "todo"
                )
            ),
            "Capture Studio": (
                "recording"
                if self.ctrl.is_recording
                else (
                    "done" if has_motion else ("active" if cur_page == "Capture Studio" else "todo")
                )
            ),
            "Motion Editor": (
                "active"
                if cur_page == "Motion Editor"
                else (
                    "done"
                    if (has_motion and self.ctrl.active_motion.corrected_motion is not None)
                    else "todo"
                )
            ),
            "Character Retargeting": (
                "active" if cur_page in ("Character Retargeting", "Animation Exports") else "todo"
            ),
        }

        labels = {
            "Welcome": f"{'✓ ' if has_proj else ''}1. Project",
            "Camera Manager": f"{'✓ ' if cams_ready else ''}2. Cameras ({active_cams})",
            "Calibration Wizard": (
                "✓ 3. Skip Calib" if single_cam else f"{'✓ ' if has_calib else ''}3. Calibrate"
            ),
            "Capture Studio": "● REC" if self.ctrl.is_recording else "4. Capture",
            "Motion Editor": "5. Edit",
            "Character Retargeting": "6. Blender",
        }

        for page_key, pill in self._step_pills.items():
            pill.setText(labels[page_key])
            pill.setProperty("stepState", states[page_key])
            pill.style().unpolish(pill)
            pill.style().polish(pill)

        if self.ctrl.server_running:
            lan_ip = self.ctrl.get_active_lan_ip()
            port = self.ctrl.settings.network.server_port
            pin = self.ctrl.camera_manager.get_active_pin()
            self.btn_server_pill.setText("Phones: server online")
            self.btn_server_pill.setToolTip(f"Pair phones at {lan_ip}:{port} · PIN {pin}")
            self.btn_server_pill.setProperty("stepState", "done")
        else:
            self.btn_server_pill.setText("Pair phones")
            self.btn_server_pill.setProperty("stepState", "todo")
        self.btn_server_pill.style().unpolish(self.btn_server_pill)
        self.btn_server_pill.style().polish(self.btn_server_pill)

        # Contextual Next Step CTA label
        if not has_proj:
            self.btn_next_step.setText("Next: Create Project →")
        elif active_cams == 0:
            self.btn_next_step.setText("Next: Pair Phone (QR) →")
        elif not calib_satisfied:
            self.btn_next_step.setText("Next: Calibrate →")
        elif not has_motion:
            self.btn_next_step.setText("Next: Capture Take →")
        else:
            self.btn_next_step.setText("Next: Send to Blender →")
        if offline:
            page = self.pages["Multi-Video Cinema Lab"]
            self.btn_next_step.setText(
                "Review motion"
                if page._last_result
                else ("Solve motion" if page._package else "Add videos")
            )

    def apply_theme(self, theme_name: str) -> None:
        self.ctrl.settings.general.theme = "dark" if theme_name == "dark" else "light"
        self.setStyleSheet(get_stylesheet(self.ctrl.settings.general.theme))
        dark = self.ctrl.settings.general.theme == "dark"
        self.btn_theme.setText("Light theme" if dark else "Dark theme")
        self._motion_enabled = not (
            self.ctrl.settings.general.reduce_interface_motion or system_reduces_motion()
        )
        self.stack.motion_enabled = self._motion_enabled
        self.stack.finish_transition()
        if not self._motion_enabled:
            target_height = self._drawer_animation.endValue()
            self._drawer_animation.stop()
            self.set_drawer_height(int(target_height or self.nav_list.height()) if self._tools_expanded else 0)
            self._finish_drawer()
        for btn in self._primary_buttons.values():
            btn.motion_enabled = self._motion_enabled
            btn.set_active(btn.property("active") == "true")
        for pname in ("Capture Studio", "Motion Editor", "Calibration Wizard"):
            p = self.pages.get(pname)
            if hasattr(p, "viewport"):
                p.viewport.canvas.dark_bg = dark
                p.viewport.canvas.update()
            if hasattr(p, "curve_widget"):
                p.curve_widget.dark_bg = dark
                p.curve_widget.update()
            if hasattr(p, "guide"):
                p.guide.set_dark_theme(dark)

    def _toggle_theme(self) -> None:
        nxt = "dark" if self.ctrl.settings.general.theme == "light" else "light"
        self.apply_theme(nxt)
        self.ctrl.settings.save(self.ctrl._settings_path)

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
        act_save.triggered.connect(self.ctrl.save_project)
        self.addAction(act_save)

    def closeEvent(self, event) -> None:
        if self._closing:
            super().closeEvent(event)
            return
        self._closing = True
        self.stack.finish_transition()
        self._drawer_animation.stop()
        video_page = self.pages.get("Multi-Video Cinema Lab")
        if video_page is not None:
            video_page.shutdown()
        for page in self.pages.values():
            if page.__class__.__name__ == "AnimationExportsPage":
                page.shutdown()
        cal_pg = self.pages.get("Calibration Wizard")
        if cal_pg is not None and hasattr(cal_pg, "shutdown_voice"):
            cal_pg.shutdown_voice()
        self.ctrl.shutdown()
        super().closeEvent(event)
