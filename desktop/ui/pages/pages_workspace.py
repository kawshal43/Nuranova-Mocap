"""Pages 1-3: Welcome & Onboarding, Dashboard, and Projects."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from desktop.ui.app_controller import StudioController


def _header(title: str, subtitle: str) -> QVBoxLayout:
    box = QVBoxLayout()
    box.setSpacing(2)
    lbl = QLabel(title)
    lbl.setObjectName("PageTitle")
    sub = QLabel(subtitle)
    sub.setObjectName("Subtitle")
    box.addWidget(lbl)
    box.addWidget(sub)
    return box


class WelcomePage(QWidget):
    """Page 1: Welcome and onboarding."""

    def __init__(self, ctrl: StudioController, navigate: Callable[[str], None]) -> None:
        super().__init__()
        self.ctrl = ctrl
        self.navigate = navigate
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(14)
        root.addLayout(_header("Welcome to Nuranova Mocap Studio", "Version 1.0.0 — Multi-Camera Optical, Markerless AI & Hybrid Cinema Motion Capture"))

        # Quick start actions
        qa_group = QGroupBox("Quick Start")
        qa_row = QHBoxLayout(qa_group)
        btn_new = QPushButton("New Project")
        btn_new.setObjectName("PrimaryButton")
        btn_new.clicked.connect(self._new_project)
        btn_open = QPushButton("Open Existing Project")
        btn_open.clicked.connect(self._open_project)
        btn_demo = QPushButton("Load Synthetic 4-Cam Demo Take")
        btn_demo.clicked.connect(self._load_demo)
        btn_srv = QPushButton("Start Camera Network Server")
        btn_srv.clicked.connect(self._start_server)
        for b in (btn_new, btn_open, btn_demo, btn_srv):
            qa_row.addWidget(b)
        root.addWidget(qa_group)

        # Onboarding checklist & hardware status
        split = QHBoxLayout()
        hw_group = QGroupBox("System Readiness Check")
        hw_form = QFormLayout(hw_group)
        hw = ctrl.hardware
        hw_form.addRow("Compute Backend:", QLabel(f"{hw.selected_device.upper()} ({hw.gpu_name})"))
        hw_form.addRow("GPU VRAM:", QLabel(f"{hw.vram_free_mb} MiB free / {hw.vram_total_mb} MiB total"))
        hw_form.addRow("Blender Detected:", QLabel(ctrl.settings.blender.blender_executable or "Not found (configure in Settings)"))
        hw_form.addRow("Local Processing:", QLabel("100% Local LAN — No Cloud Uploads"))
        split.addWidget(hw_group, 1)

        rec_group = QGroupBox("Recent Projects")
        rec_lay = QVBoxLayout(rec_group)
        self.recent_list = QListWidget()
        self.recent_list.itemDoubleClicked.connect(self._open_recent_item)
        rec_lay.addWidget(self.recent_list)
        split.addWidget(rec_group, 1)
        root.addLayout(split, 1)

        ctrl.project_changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        self.recent_list.clear()
        for item in self.ctrl.project_manager.list_recent_projects():
            self.recent_list.addItem(f"{item['name']} — {item['path']}")

    def _new_project(self) -> None:
        name, ok = QInputDialog.getText(self, "New Project", "Project Name:", text="Animated_Short_01")
        if not ok or not name.strip():
            return
        parent = Path(self.ctrl.settings.general.default_project_dir)
        parent.mkdir(parents=True, exist_ok=True)
        try:
            self.ctrl.create_project(parent, name.strip())
            self.navigate("Dashboard")
        except Exception as exc:
            QMessageBox.warning(self, "Create Project Failed", str(exc))

    def _open_project(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open Nuranova Project Directory", self.ctrl.settings.general.default_project_dir)
        if not folder:
            return
        try:
            self.ctrl.open_project(folder)
            self.navigate("Dashboard")
        except Exception as exc:
            QMessageBox.warning(self, "Open Project Failed", str(exc))

    def _open_recent_item(self, item) -> None:
        path = item.text().split(" — ", 1)[-1]
        try:
            self.ctrl.open_project(path)
            self.navigate("Dashboard")
        except Exception as exc:
            QMessageBox.warning(self, "Open Failed", str(exc))

    def _load_demo(self) -> None:
        self.ctrl.generate_synthetic_demo_take("synthetic_demo_walk")
        self.navigate("Motion Editor")

    def _start_server(self) -> None:
        self.ctrl.start_server()
        self.navigate("Camera Manager")


class DashboardPage(QWidget):
    """Page 2: Studio Dashboard."""

    def __init__(self, ctrl: StudioController, navigate: Callable[[str], None]) -> None:
        super().__init__()
        self.ctrl = ctrl
        self.navigate = navigate
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(12)
        root.addLayout(_header("Studio Dashboard", "Live overview of active project, cameras, calibration, and motion takes"))

        stats_row = QHBoxLayout()
        self.lbl_proj = QLabel("No Active Project")
        self.lbl_cams = QLabel("0 Connected")
        self.lbl_calib = QLabel("Not Calibrated")
        self.lbl_takes = QLabel("0 Takes")
        for title, lbl in [
            ("Active Project", self.lbl_proj),
            ("Body Cameras (2–12)", self.lbl_cams),
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
            ("1. Connect Cameras", "Camera Manager"),
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
        self.lbl_proj.setText(f"{proj.name}\n{proj.root_dir}" if proj else "None open (create or open in Projects)")
        active_c = len(self.ctrl.camera_manager.get_active_body_cameras())
        total_c = len(self.ctrl.camera_manager.list_cameras())
        self.lbl_cams.setText(f"{active_c} active / {total_c} registered")
        cal = self.ctrl.active_calibration
        if cal:
            dist_txt = f" | Wand err: {cal.known_distance_error_mm:.1f} mm" if cal.known_distance_error_mm is not None else ""
            self.lbl_calib.setText(f"{cal.profile_name}\n{len(cal.cameras)} cams | RMSE: {cal.global_reprojection_rmse_px:.2f} px{dist_txt}")
        else:
            self.lbl_calib.setText("Uncalibrated — open Calibration Wizard")

        takes = self.ctrl.project_manager.list_takes()
        self.lbl_takes.setText(f"{len(takes)} indexed takes")
        self.activity_list.clear()
        for t in takes:
            self.activity_list.addItem(f"[{t['capture_mode']}] {t['name']} — {t['num_frames']} frames @ {t['fps']:.0f} FPS ({t['status']})")


class ProjectsPage(QWidget):
    """Page 3: Full Project Lifecycle Management."""

    def __init__(self, ctrl: StudioController) -> None:
        super().__init__()
        self.ctrl = ctrl
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(12)
        root.addLayout(_header("Projects", "Create, open, duplicate, rename, backup, export, import, and delete portable .nmocap projects"))

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
        self.recent_list.itemDoubleClicked.connect(lambda it: self._open_path(it.text().split(" — ", 1)[-1]))
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
        new_name, ok = QInputDialog.getText(self, "Rename Project", "New name:", text=self.ctrl.project_manager.active_project.name)
        if ok and new_name.strip():
            self.ctrl.project_manager.rename_project(new_name.strip())
            self.ctrl.project_changed.emit()

    def _duplicate(self) -> None:
        if not self.ctrl.project_manager.active_project:
            return
        new_name, ok = QInputDialog.getText(self, "Duplicate Project", "Copy name:", text=self.ctrl.project_manager.active_project.name + "_Copy")
        if ok and new_name.strip():
            self.ctrl.project_manager.duplicate_project(self.ctrl.project_manager.active_project.root_dir.parent, new_name.strip())
            self.ctrl.project_changed.emit()

    def _export_zip(self) -> None:
        if not self.ctrl.project_manager.active_project:
            QMessageBox.information(self, "No Project", "Open or create a project first.")
            return
        out, _ = QFileDialog.getSaveFileName(self, "Export Portable Project Archive", "", "Nuranova Project ZIP (*.nmocap.zip)")
        if out:
            zp = self.ctrl.project_manager.backup_or_export_project_zip(out)
            QMessageBox.information(self, "Exported", f"Portable project archive saved to:\n{zp}")

    def _import_zip(self) -> None:
        zp, _ = QFileDialog.getOpenFileName(self, "Import Project ZIP", "", "Project Archive (*.zip)")
        if not zp:
            return
        dst = QFileDialog.getExistingDirectory(self, "Select Destination Directory", self.edit_dir.text())
        if dst:
            proj = self.ctrl.project_manager.import_or_restore_project_zip(zp, Path(dst) / Path(zp).stem)
            self.ctrl.open_project(proj.root_dir)

    def _delete(self) -> None:
        proj = self.ctrl.project_manager.active_project
        if not proj:
            return
        reply = QMessageBox.question(
            self,
            "Confirm Project Deletion",
            f"Permanently delete project '{proj.name}' at:\n{proj.root_dir}?\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.ctrl.project_manager.delete_project(proj.root_dir, confirmed=True)
            self.ctrl.project_changed.emit()
