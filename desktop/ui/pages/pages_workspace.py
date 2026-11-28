"""Pages 1-3: Modern Card-Type Studio Home, Dashboard, and Portable Projects."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from desktop.ui.app_controller import StudioController
from desktop.ui.widgets.studio_chrome import MotionArtwork, WorkspaceIcon


def _header(title: str, subtitle: str) -> QVBoxLayout:
    box = QVBoxLayout()
    box.setSpacing(2)
    lbl = QLabel(title)
    lbl.setObjectName("PageTitle")
    lbl.setWordWrap(True)
    sub = QLabel(subtitle)
    sub.setObjectName("Subtitle")
    sub.setWordWrap(True)
    box.addWidget(lbl)
    box.addWidget(sub)
    return box


class WelcomePage(QWidget):
    """Page 1: Modern Card-Type Studio Home Hub with Live Readiness KPIs, Action Cards & Project Cards."""

    def __init__(self, ctrl: StudioController, navigate: Callable[[str], None]) -> None:
        super().__init__()
        self.ctrl = ctrl
        self.navigate = navigate

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        root = QVBoxLayout(content)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(18)

        # ---------------- 1. Hero Studio Banner Card ----------------
        banner = QFrame()
        banner.setObjectName("BannerCard")
        bl = QHBoxLayout(banner)
        bl.setContentsMargins(22, 18, 22, 18)
        bl.setSpacing(16)

        b_left = QVBoxLayout()
        b_left.setSpacing(6)
        eyebrow = QLabel("YOUR MOTION WORKSPACE")
        eyebrow.setObjectName("HeroEyebrow")
        b_left.addWidget(eyebrow)
        b_title = QLabel("Turn movement\ninto animation.")
        b_title.setObjectName("HeroTitle")
        b_title.setWordWrap(True)
        b_sub = QLabel(
            "Bring your cameras together. Refine every detail.\nCreate animation ready for Blender."
        )
        b_sub.setObjectName("Subtitle")
        b_sub.setWordWrap(True)
        self.lbl_hero_pill = QLabel("Local studio")
        self.lbl_hero_pill.setObjectName("BadgeWarn")
        self.lbl_hero_pill.setSizePolicy(
            self.lbl_hero_pill.sizePolicy().horizontalPolicy(),
            self.lbl_hero_pill.sizePolicy().verticalPolicy(),
        )
        pill_row = QHBoxLayout()
        pill_row.addWidget(self.lbl_hero_pill)
        pill_row.addStretch()
        b_left.addWidget(b_title)
        b_left.addWidget(b_sub)
        b_left.addLayout(pill_row)
        bl.addLayout(b_left, 1)

        artwork = MotionArtwork()
        artwork.setFixedSize(170, 170)
        bl.addWidget(artwork)

        b_right = QVBoxLayout()
        b_right.setSpacing(8)
        btn_hero_mocap = QPushButton("Import recorded videos")
        btn_hero_mocap.setObjectName("PrimaryButton")
        btn_hero_mocap.setMinimumHeight(40)
        btn_hero_mocap.clicked.connect(lambda: self.navigate("Multi-Video Cinema Lab"))

        btn_hero_new = QPushButton("+  New Mocap Project")
        btn_hero_new.setMinimumHeight(34)
        btn_hero_new.clicked.connect(self._new_project)

        self.btn_hero_single_cam = QPushButton("Use single-camera AI")
        self.btn_hero_single_cam.setMinimumHeight(34)
        self.btn_hero_single_cam.clicked.connect(
            lambda: self.ctrl.set_single_camera_mode(
                not self.ctrl.settings.general.single_camera_mode
            )
        )
        b_right.addWidget(btn_hero_mocap)
        b_right.addWidget(btn_hero_new)
        b_right.addWidget(self.btn_hero_single_cam)
        bl.addLayout(b_right)
        root.addWidget(banner)

        # ---------------- 2. Live KPI Metric Cards Strip ----------------
        kpi_row = QHBoxLayout()
        kpi_row.setSpacing(12)
        self._kpi_labels: Dict[str, Tuple[QLabel, QLabel]] = {}
        for key, title in [
            ("project", "ACTIVE PROJECT"),
            ("cameras", "PHONES ONLINE"),
            ("calib", "CALIBRATION"),
            ("compute", "AI COMPUTE ENGINE"),
        ]:
            card = QFrame()
            card.setObjectName("MetricCard")
            cl = QVBoxLayout(card)
            cl.setContentsMargins(14, 12, 14, 12)
            cl.setSpacing(3)
            t = QLabel(title)
            t.setObjectName("MetricTitle")
            v = QLabel("--")
            v.setObjectName("MetricValue")
            s = QLabel("")
            s.setObjectName("Subtitle")
            s.setWordWrap(True)
            cl.addWidget(t)
            cl.addWidget(v)
            cl.addWidget(s)
            kpi_row.addWidget(card, 1)
            self._kpi_labels[key] = (v, s)
        root.addLayout(kpi_row)

        # ---------------- 3. Studio Workflow Action Cards Grid (3x2) ----------------
        lbl_sec1 = QLabel("STUDIO WORKSPACES")
        lbl_sec1.setObjectName("MetricTitle")
        root.addWidget(lbl_sec1)

        cards_grid = QGridLayout()
        cards_grid.setSpacing(14)
        self._card_badges: Dict[str, QLabel] = {}

        card_specs = [
            (
                "pair",
                "Pair Android Phones",
                "Connect 1 to 12 smartphones over Wi-Fi using instant QR code scan or auto-discovery.",
                "Open QR Pairing Hub →",
                True,
                self._go_pair,
            ),
            (
                "capture",
                "Live Capture & 3D Preview",
                "Launch live mocap with real-time 3D Character & Skeleton preview and synchronized phone recording.",
                "Open Capture Studio →",
                True,
                self._start_live_mocap,
            ),
            (
                "calibrate",
                "Multi-Cam Calibration",
                "Print exact-mm ChArUco boards, calibrate camera lenses, and align the floor to Blender Z-Up.",
                "Calibration Wizard →",
                False,
                lambda: self.navigate("Calibration Wizard"),
            ),
            (
                "editor",
                "Motion Editor & Cleanup",
                "Inspect 3D character & animation curves, apply Butterworth / One Euro filters and 2-bone IK foot-lock.",
                "Open Motion Editor →",
                False,
                lambda: self.navigate("Motion Editor"),
            ),
            (
                "blender",
                "Blender Bridge & Export",
                "Stream takes live to Blender 4.5 / 5.x for Rigify & custom character retargeting, or export FBX/BVH.",
                "Blender && Exports →",
                False,
                lambda: self.navigate("Character Retargeting"),
            ),
            (
                "demo",
                "Instant 3D Demo Take",
                "Generate a 4-camera synthetic walk cycle immediately to test the 3D character viewport & Blender bridge.",
                "Load 3D Demo Walk →",
                False,
                self._load_demo,
            ),
        ]

        for idx, (key, title, desc, btn_txt, is_primary, cb) in enumerate(card_specs):
            card = QFrame()
            card.setObjectName("ActionCard")
            card.setMinimumHeight(168)
            cl = QVBoxLayout(card)
            cl.setContentsMargins(16, 15, 16, 15)
            cl.setSpacing(10)

            top_r = QHBoxLayout()
            ic = WorkspaceIcon(key)
            badge = QLabel("Ready")
            badge.setObjectName("BadgeWarn")
            top_r.addWidget(ic)
            top_r.addStretch()
            top_r.addWidget(badge, 0, Qt.AlignmentFlag.AlignVCenter)
            cl.addLayout(top_r)

            t_lbl = QLabel(title)
            t_lbl.setStyleSheet("font-size: 15px; font-weight: 700;")
            d_lbl = QLabel(desc)
            d_lbl.setObjectName("Subtitle")
            d_lbl.setWordWrap(True)
            cl.addWidget(t_lbl)
            cl.addWidget(d_lbl, 1)

            btn = QPushButton(btn_txt)
            if is_primary:
                btn.setObjectName("PrimaryButton")
            btn.clicked.connect(cb)
            cl.addWidget(btn)

            self._card_badges[key] = badge
            cards_grid.addWidget(card, idx // 3, idx % 3)

        root.addLayout(cards_grid)

        # ---------------- 4. Recent & Sample Projects Card Strip ----------------
        proj_hdr = QHBoxLayout()
        lbl_sec2 = QLabel("PROJECTS & SAMPLES")
        lbl_sec2.setObjectName("MetricTitle")
        btn_open_folder = QPushButton("Open Project Folder...")
        btn_open_folder.clicked.connect(self._open_project)
        proj_hdr.addWidget(lbl_sec2)
        proj_hdr.addStretch()
        proj_hdr.addWidget(btn_open_folder)
        root.addLayout(proj_hdr)

        self.projects_grid = QGridLayout()
        self.projects_grid.setSpacing(12)
        root.addLayout(self.projects_grid)
        root.addStretch()

        scroll.setWidget(content)
        outer.addWidget(scroll)

        ctrl.project_changed.connect(self.refresh)
        ctrl.cameras_changed.connect(self.refresh)
        ctrl.calibration_changed.connect(self.refresh)
        ctrl.motion_changed.connect(self.refresh)
        ctrl.server_state_changed.connect(lambda ok, msg: self.refresh())
        self.refresh()

    def _set_badge(self, key: str, text: str, style_name: str) -> None:
        lbl = self._card_badges.get(key)
        if lbl is None:
            return
        lbl.setText(text)
        lbl.setObjectName(style_name)
        lbl.style().unpolish(lbl)
        lbl.style().polish(lbl)

    def refresh(self) -> None:
        lan_ip = self.ctrl.get_active_lan_ip()
        port = self.ctrl.settings.network.server_port
        pin = self.ctrl.camera_manager.get_active_pin()

        if self.ctrl.server_running:
            self.lbl_hero_pill.setText(f"● Server Online · {lan_ip}:{port} · PIN {pin}")
            self.lbl_hero_pill.setObjectName("BadgeOk")
        else:
            self.lbl_hero_pill.setText("Local studio · phones offline")
            self.lbl_hero_pill.setObjectName("BadgeWarn")
        self.lbl_hero_pill.style().unpolish(self.lbl_hero_pill)
        self.lbl_hero_pill.style().polish(self.lbl_hero_pill)

        # Update KPI cards
        proj = self.ctrl.project_manager.active_project
        self._kpi_labels["project"][0].setText(proj.name if proj else "None Open")
        self._kpi_labels["project"][1].setText(
            str(proj.root_dir.name) if proj else "Create or open below"
        )

        n_online = len(self.ctrl.camera_manager.get_active_recording_cameras())
        n_total = len(self.ctrl.camera_manager.list_cameras())
        self._kpi_labels["cameras"][0].setText(f"{n_online} Online")
        self._kpi_labels["cameras"][1].setText(f"{n_total} paired · PIN {pin}")

        single_cam = self.ctrl.settings.general.single_camera_mode
        self.btn_hero_single_cam.setText(
            "Single-camera AI · enabled"
            if single_cam
            else "Use single-camera AI"
        )

        cal = self.ctrl.active_calibration
        if single_cam:
            self._kpi_labels["calib"][0].setText("Skipped (1-Cam)")
            self._kpi_labels["calib"][1].setText("3D AI + Root Walk active")
        elif cal:
            self._kpi_labels["calib"][0].setText(f"{cal.global_reprojection_rmse_px:.2f} px RMSE")
            self._kpi_labels["calib"][1].setText(f"{len(cal.cameras)} calibrated cameras")
        else:
            self._kpi_labels["calib"][0].setText("Not Calibrated")
            self._kpi_labels["calib"][1].setText("Or enable 1-Cam Mode above")

        hw = self.ctrl.hardware
        self._kpi_labels["compute"][0].setText(hw.selected_device.upper())
        self._kpi_labels["compute"][1].setText(
            hw.gpu_name if hw.gpu_detected else f"{hw.cpu_count} CPU threads"
        )

        # Update Action Card status chips
        self._set_badge(
            "pair",
            f"● {n_online} Online" if n_online > 0 else f"PIN {pin}",
            "BadgeOk" if n_online > 0 else "BadgeWarn",
        )
        self._set_badge(
            "capture",
            "● 1-Cam Ready"
            if (single_cam and n_online >= 1)
            else ("● Live Ready" if n_online > 0 else "3D Preview Ready"),
            "BadgeOk" if n_online > 0 else "BadgeWarn",
        )
        self._set_badge(
            "calibrate",
            "✓ Skipped (1-Cam)"
            if single_cam
            else (f"✓ {len(cal.cameras)} Cams" if cal else "Multi-Cam Only"),
            "BadgeOk" if (cal or single_cam) else "BadgeWarn",
        )
        doc = self.ctrl.active_motion
        self._set_badge(
            "editor",
            f"{doc.num_frames} Frames" if doc else "No Take Loaded",
            "BadgeOk" if doc else "BadgeWarn",
        )
        bl = self.ctrl.settings.blender.blender_executable
        self._set_badge(
            "blender", "Blender Ready" if bl else "Configure Path", "BadgeOk" if bl else "BadgeWarn"
        )
        self._set_badge("demo", "Offline Sandbox", "BadgeOk")

        # Rebuild Project Cards grid
        while self.projects_grid.count():
            item = self.projects_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        cards_data = []
        seen_paths = set()
        for item in self.ctrl.project_manager.list_recent_projects()[:5]:
            p_str = str(item["path"])
            if p_str not in seen_paths:
                seen_paths.add(p_str)
                cards_data.append((item["name"], p_str, "Recent Project"))

        sample_dir = (
            Path(__file__).resolve().parents[3] / "sample_projects" / "Nuranova_Sample_Walk"
        )
        if (
            sample_dir.exists()
            and (sample_dir / "project.nmocap").exists()
            and str(sample_dir) not in seen_paths
        ):
            cards_data.append(("Nuranova Sample Walk", str(sample_dir), "Built-in Sample"))

        if not cards_data:
            empty_card = QFrame()
            empty_card.setObjectName("ProjectCard")
            el = QHBoxLayout(empty_card)
            el.setContentsMargins(16, 14, 16, 14)
            el.addWidget(
                QLabel(
                    "No projects yet. Click '+ New Mocap Project' above or open an existing folder."
                )
            )
            self.projects_grid.addWidget(empty_card, 0, 0, 1, 3)
        else:
            for idx, (pname, ppath, tag) in enumerate(cards_data[:6]):
                pc = QFrame()
                pc.setObjectName("ProjectCard")
                pl = QVBoxLayout(pc)
                pl.setContentsMargins(14, 12, 14, 12)
                pl.setSpacing(6)
                row_top = QHBoxLayout()
                t = QLabel(pname)
                t.setStyleSheet("font-size: 14px; font-weight: 700;")
                tg = QLabel(tag)
                tg.setObjectName("BadgeOk" if tag == "Built-in Sample" else "BadgeWarn")
                row_top.addWidget(t)
                row_top.addStretch()
                row_top.addWidget(tg)
                p_lbl = QLabel(ppath)
                p_lbl.setObjectName("Subtitle")
                p_lbl.setWordWrap(True)
                btn_op = QPushButton("Open Project →")
                btn_op.clicked.connect(lambda checked=False, p=ppath: self._open_path(p))
                pl.addLayout(row_top)
                pl.addWidget(p_lbl, 1)
                pl.addWidget(btn_op)
                self.projects_grid.addWidget(pc, idx // 3, idx % 3)

    def _new_project(self) -> None:
        name, ok = QInputDialog.getText(
            self, "New Project", "Project Name:", text="Studio_Scene_01"
        )
        if not ok or not name.strip():
            return
        parent = Path(self.ctrl.settings.general.default_project_dir)
        parent.mkdir(parents=True, exist_ok=True)
        try:
            self.ctrl.create_project(parent, name.strip())
            self.navigate("Camera Manager")
        except Exception as exc:
            QMessageBox.warning(self, "Create Project Failed", str(exc))

    def _open_project(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Open Nuranova Project Directory", self.ctrl.settings.general.default_project_dir
        )
        if folder:
            self._open_path(folder)

    def _open_path(self, path: str) -> None:
        try:
            self.ctrl.open_project(path)
            self.navigate("Capture Studio")
        except Exception as exc:
            QMessageBox.warning(self, "Open Project Failed", str(exc))

    def _load_demo(self) -> None:
        self.ctrl.generate_synthetic_demo_take("synthetic_demo_walk")
        self.navigate("Motion Editor")

    def _go_pair(self) -> None:
        if not self.ctrl.server_running:
            self.ctrl.start_server()
        self.navigate("Camera Manager")

    def _start_live_mocap(self) -> None:
        if not self.ctrl.server_running:
            self.ctrl.start_server()
        self.navigate("Capture Studio")
        # Start live 3D Character/Skeleton preview right away (uses live phones when streaming, else live demo character!)
        self.ctrl.live_preview.start(simulate_when_no_frames=True)


class DashboardPage(QWidget):
    """Page 2: Studio Dashboard."""

    def __init__(self, ctrl: StudioController, navigate: Callable[[str], None]) -> None:
        super().__init__()
        self.ctrl = ctrl
        self.navigate = navigate
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(12)
        root.addLayout(
            _header(
                "Studio Dashboard",
                "Live overview of active project, cameras, calibration, and motion takes",
            )
        )

        stats_row = QHBoxLayout()
        self.lbl_proj = QLabel("No Active Project")
        self.lbl_cams = QLabel("0 Connected")
        self.lbl_calib = QLabel("Not Calibrated")
        self.lbl_takes = QLabel("0 Takes")
        for title, lbl in [
            ("Active Project", self.lbl_proj),
            ("Cameras (2–12)", self.lbl_cams),
            ("Multi-Cam Calibration", self.lbl_calib),
            ("Recorded Takes", self.lbl_takes),
        ]:
            box = QGroupBox(title)
            v = QVBoxLayout(box)
            lbl.setWordWrap(True)
            v.addWidget(lbl)
            stats_row.addWidget(box)
        root.addLayout(stats_row)

        nav_group = QGroupBox("Production Pipeline Shortcuts")
        nav_row = QHBoxLayout(nav_group)
        for label, page in [
            ("1. Pair Phones", "Camera Manager"),
            ("2. Calibrate Volume", "Calibration Wizard"),
            ("3. Capture Studio", "Capture Studio"),
            ("4. Motion Editor", "Motion Editor"),
            ("5. Blender Retarget", "Character Retargeting"),
        ]:
            b = QPushButton(label)
            b.clicked.connect(lambda checked=False, p=page: self.navigate(p))
            nav_row.addWidget(b)
        root.addWidget(nav_group)

        self.activity_list = QListWidget()
        act_box = QGroupBox("Active Project Takes & Artifacts")
        QVBoxLayout(act_box).addWidget(self.activity_list)
        root.addWidget(act_box, 1)

        ctrl.project_changed.connect(self.refresh)
        ctrl.cameras_changed.connect(self.refresh)
        ctrl.calibration_changed.connect(self.refresh)
        ctrl.motion_changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        proj = self.ctrl.project_manager.active_project
        self.lbl_proj.setText(
            f"{proj.name}\n{proj.root_dir}" if proj else "None open (create or open in Projects)"
        )
        active_c = len(self.ctrl.camera_manager.get_active_recording_cameras())
        total_c = len(self.ctrl.camera_manager.list_cameras())
        self.lbl_cams.setText(f"{active_c} online / {total_c} paired")
        cal = self.ctrl.active_calibration
        if cal:
            dist_txt = (
                f" | Wand err: {cal.known_distance_error_mm:.1f} mm"
                if cal.known_distance_error_mm is not None
                else ""
            )
            self.lbl_calib.setText(
                f"{cal.profile_name}\n{len(cal.cameras)} cams | RMSE: {cal.global_reprojection_rmse_px:.2f} px{dist_txt}"
            )
        else:
            self.lbl_calib.setText("Uncalibrated — open Calibration Wizard")

        takes = self.ctrl.project_manager.list_takes()
        self.lbl_takes.setText(f"{len(takes)} indexed takes")
        self.activity_list.clear()
        for t in takes:
            self.activity_list.addItem(
                f"[{t['capture_mode']}] {t['name']} — {t['num_frames']} frames @ {t['fps']:.0f} FPS ({t['status']})"
            )


class ProjectsPage(QWidget):
    """Page 3: Full Project Lifecycle Management."""

    def __init__(self, ctrl: StudioController) -> None:
        super().__init__()
        self.ctrl = ctrl
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(12)
        root.addLayout(
            _header(
                "Projects",
                "Create, open, duplicate, rename, backup, export, import, and delete portable .nmocap projects",
            )
        )

        create_box = QGroupBox("Create New Portable Project")
        c_row = QHBoxLayout(create_box)
        self.edit_name = QLineEdit("Scene_01_Mocap")
        self.edit_dir = QLineEdit(ctrl.settings.general.default_project_dir)
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse_dir)
        btn_create = QPushButton("Create Project")
        btn_create.setObjectName("PrimaryButton")
        btn_create.clicked.connect(self._create)
        c_row.addWidget(QLabel("Name:"))
        c_row.addWidget(self.edit_name, 1)
        c_row.addWidget(QLabel("Folder:"))
        c_row.addWidget(self.edit_dir, 2)
        c_row.addWidget(btn_browse)
        c_row.addWidget(btn_create)
        root.addWidget(create_box)

        ops_box = QGroupBox("Active Project Operations")
        ops_row = QHBoxLayout(ops_box)
        for label, fn, style in [
            ("Open...", self._open, ""),
            ("Save Manifest", self._save, ""),
            ("Rename", self._rename, ""),
            ("Duplicate", self._duplicate, ""),
            ("Backup / Export ZIP", self._export_zip, "PrimaryButton"),
            ("Import / Restore ZIP", self._import_zip, ""),
            ("Delete Project...", self._delete, "DangerButton"),
        ]:
            b = QPushButton(label)
            if style:
                b.setObjectName(style)
            b.clicked.connect(fn)
            ops_row.addWidget(b)
        root.addWidget(ops_box)

        self.info_label = QLabel("No project currently open.")
        self.info_label.setWordWrap(True)
        root.addWidget(self.info_label)

        self.recent_list = QListWidget()
        rec_box = QGroupBox("Recent Projects (Double-click to open)")
        QVBoxLayout(rec_box).addWidget(self.recent_list)
        self.recent_list.itemDoubleClicked.connect(
            lambda it: self._open_path(it.text().split(" — ", 1)[-1])
        )
        root.addWidget(rec_box, 1)

        ctrl.project_changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        p = self.ctrl.project_manager.active_project
        self.info_label.setText(
            f"Active Project: {p.name} | ID: {p.project_id} | Path: {p.root_dir} | Last Saved: {p.modified_at}"
            if p
            else "No active project."
        )
        self.recent_list.clear()
        for it in self.ctrl.project_manager.list_recent_projects():
            self.recent_list.addItem(f"{it['name']} — {it['path']}")

    def _browse_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Parent Folder", self.edit_dir.text())
        if d:
            self.edit_dir.setText(d)

    def _create(self) -> None:
        try:
            self.ctrl.create_project(self.edit_dir.text(), self.edit_name.text())
        except Exception as exc:
            QMessageBox.warning(self, "Create Failed", str(exc))

    def _open(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Open Project Folder", self.edit_dir.text())
        if d:
            self._open_path(d)

    def _open_path(self, path: str) -> None:
        try:
            self.ctrl.open_project(path)
        except Exception as exc:
            QMessageBox.warning(self, "Open Failed", str(exc))

    def _save(self) -> None:
        self.ctrl.project_manager.save_project()
        self.ctrl.project_changed.emit()
        self.ctrl.status_message.emit("Project saved.")

    def _rename(self) -> None:
        if not self.ctrl.project_manager.active_project:
            return
        new_name, ok = QInputDialog.getText(
            self, "Rename Project", "New name:", text=self.ctrl.project_manager.active_project.name
        )
        if ok and new_name.strip():
            self.ctrl.project_manager.rename_project(new_name.strip())
            self.ctrl.project_changed.emit()

    def _duplicate(self) -> None:
        if not self.ctrl.project_manager.active_project:
            return
        if not self._project_is_idle():
            return
        new_name, ok = QInputDialog.getText(
            self,
            "Duplicate Project",
            "Copy name:",
            text=self.ctrl.project_manager.active_project.name + "_Copy",
        )
        if ok and new_name.strip():
            try:
                duplicate = self.ctrl.project_manager.duplicate_project(
                    self.ctrl.project_manager.active_project.root_dir.parent, new_name.strip()
                )
                self.ctrl.open_project(duplicate.root_dir)
            except Exception as error:
                QMessageBox.warning(self, "Project copy failed", str(error))

    def _project_is_idle(self):
        try:
            self.ctrl._require_idle_project()
            return True
        except RuntimeError as error:
            QMessageBox.warning(self, "Project is busy", str(error))
            return False

    def _export_zip(self) -> None:
        if not self.ctrl.project_manager.active_project:
            QMessageBox.information(self, "No Project", "Open or create a project first.")
            return
        if not self._project_is_idle():
            return
        out, _ = QFileDialog.getSaveFileName(
            self, "Export Portable Project Archive", "", "Nuranova Project ZIP (*.nmocap.zip)"
        )
        if out:
            try:
                zp = self.ctrl.project_manager.backup_or_export_project_zip(out)
                QMessageBox.information(self, "Exported", f"Portable project archive saved to:\n{zp}")
            except Exception as error:
                QMessageBox.warning(self, "Project archive failed", str(error))

    def _import_zip(self) -> None:
        if not self._project_is_idle():
            return
        zp, _ = QFileDialog.getOpenFileName(
            self, "Import Project ZIP", "", "Project Archive (*.zip)"
        )
        if not zp:
            return
        dst = QFileDialog.getExistingDirectory(
            self, "Select Destination Directory", self.edit_dir.text()
        )
        if dst:
            try:
                proj = self.ctrl.project_manager.import_or_restore_project_zip(
                    zp, Path(dst) / Path(zp).stem
                )
                self.ctrl.open_project(proj.root_dir)
            except Exception as error:
                QMessageBox.warning(self, "Project restore failed", str(error))

    def _delete(self) -> None:
        proj = self.ctrl.project_manager.active_project
        if not proj:
            return
        if not self._project_is_idle():
            return
        reply = QMessageBox.question(
            self,
            "Confirm Project Deletion",
            f"Permanently delete project '{proj.name}' at:\n{proj.root_dir}?\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.ctrl.project_manager.delete_project(proj.root_dir, confirmed=True)
                self.ctrl._reset_project_state()
                self.ctrl.project_changed.emit()
            except Exception as error:
                QMessageBox.warning(self, "Project deletion failed", str(error))
