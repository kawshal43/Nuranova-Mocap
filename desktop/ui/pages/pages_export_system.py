"""Pages 16-20: Character Retargeting, Animation Exports, Diagnostics, Settings, and About."""

from __future__ import annotations

import json
import threading
import tempfile
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Signal

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from desktop.ui.app_controller import StudioController
from desktop.ui.pages.pages_workspace import _header
from engine.ai_tracking.runtime import detect_hardware
from engine.data_format.motion_io import export_blender_json, save_motion_file
from engine.export.bvh_exporter import export_bvh


class CharacterRetargetingPage(QWidget):
    """Page 16: Blender Rigify / Custom Character Mapping & Bridge Publishing."""

    def __init__(self, ctrl: StudioController) -> None:
        super().__init__()
        self.ctrl = ctrl
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(10)
        root.addLayout(
            _header(
                "Character Retargeting",
                "Edit reusable Blender Rigify / Custom Rig bone mappings and publish active take to Nuranova Mocap Bridge",
            )
        )

        bar = QHBoxLayout()
        btn_rigify = QPushButton("Load Default Rigify Preset")
        btn_rigify.clicked.connect(self._load_rigify_preset)
        btn_save = QPushButton("Save Mapping Profile JSON...")
        btn_save.clicked.connect(self._save_profile)
        btn_pub = QPushButton("Publish Active Take to Blender Bridge")
        btn_pub.setObjectName("PrimaryButton")
        btn_pub.clicked.connect(self._publish_bridge)
        for b in (btn_rigify, btn_save, btn_pub):
            bar.addWidget(b)
        bar.addStretch()
        root.addLayout(bar)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(
            ["Nuranova Source Bone", "Target Character Bone (Editable)"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        root.addWidget(self.table, 1)
        self._load_rigify_preset()

    def _load_rigify_preset(self) -> None:
        preset = {
            "hips": "DEF-spine",
            "spine": "DEF-spine.001",
            "spine1": "DEF-spine.002",
            "chest": "DEF-spine.003",
            "neck": "DEF-spine.004",
            "head": "DEF-spine.006",
            "left_clavicle": "DEF-shoulder.L",
            "left_upper_arm": "DEF-upper_arm.L",
            "left_forearm": "DEF-forearm.L",
            "left_hand": "DEF-hand.L",
            "right_clavicle": "DEF-shoulder.R",
            "right_upper_arm": "DEF-upper_arm.R",
            "right_forearm": "DEF-forearm.R",
            "right_hand": "DEF-hand.R",
            "left_upper_leg": "DEF-thigh.L",
            "left_lower_leg": "DEF-shin.L",
            "left_foot": "DEF-foot.L",
            "left_toe": "DEF-toe.L",
            "right_upper_leg": "DEF-thigh.R",
            "right_lower_leg": "DEF-shin.R",
            "right_foot": "DEF-foot.R",
            "right_toe": "DEF-toe.R",
        }
        self.table.setRowCount(len(preset))
        for r, (src, tgt) in enumerate(preset.items()):
            self.table.setItem(r, 0, QTableWidgetItem(src))
            self.table.setItem(r, 1, QTableWidgetItem(tgt))

    def _save_profile(self) -> None:
        fp, _ = QFileDialog.getSaveFileName(
            self, "Save Character Mapping Profile", "character_mapping.json", "JSON (*.json)"
        )
        if not fp:
            return
        bmap = {}
        for r in range(self.table.rowCount()):
            s = self.table.item(r, 0).text().strip()
            t = self.table.item(r, 1).text().strip()
            if s and t:
                bmap[s] = t
        Path(fp).write_text(
            json.dumps({"schema": "nuranova-rig-map-v1", "bone_map": bmap}, indent=2),
            encoding="utf-8",
        )
        self.ctrl.status_message.emit(
            f"Saved character mapping profile ({len(bmap)} bones) to {fp}"
        )

    def _publish_bridge(self) -> None:
        doc = self.ctrl.active_motion
        if doc is None:
            QMessageBox.information(self, "No Active Take", "Load or generate a motion take first.")
            return
        if not self.ctrl.server_running:
            self.ctrl.start_server()
        out_dir = (
            self.ctrl.project_manager.active_project.subdir("exports")
            if self.ctrl.project_manager.active_project
            else Path(self.ctrl.settings.storage.export_dir)
        )
        out_path = export_blender_json(
            doc, out_dir / f"{doc.take_id}_bridge.nmocap.json", use_corrected=True
        )
        self.ctrl.network_server.register_bridge_take(doc.take_id, out_path)
        QMessageBox.information(
            self,
            "Published to Blender Bridge",
            f"Take '{doc.take_id}' is now live on http://127.0.0.1:{self.ctrl.settings.network.server_port}\n"
            "Open Blender -> View3D Sidebar -> Nuranova Mocap -> Check Studio Connection -> Pull Take.",
        )


class AnimationExportsPage(QWidget):
    """Export motion archives and verified FBX/GLB/glTF character animation."""

    _export_done = Signal(object)
    _export_error = Signal(str)

    def __init__(self, ctrl: StudioController) -> None:
        super().__init__()
        self.ctrl = ctrl
        self._export_cancel = threading.Event()
        self._export_thread = None
        self._export_done.connect(self._on_export_done)
        self._export_error.connect(self._on_export_error)
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(10)
        root.addLayout(
            _header(
                "Animation Exports",
                "Export editable motion to HDF5, Blender JSON and BVH, or character animation to FBX, GLB and glTF",
            )
        )

        opt_box = QGroupBox("Export Options")
        of = QFormLayout(opt_box)
        self.chk_corrected = QCheckBox("Use Corrected Motion Track (recommended)")
        self.chk_corrected.setChecked(True)
        self.chk_bvh_cm = QCheckBox("Scale BVH to Centimeters (standard Maya/Blender BVH)")
        self.chk_bvh_cm.setChecked(True)
        of.addRow("", self.chk_corrected)
        of.addRow("", self.chk_bvh_cm)
        self.export_variant = QComboBox()
        self.export_variant.addItem("Motion only (skeleton + animation)", "motion")
        self.export_variant.addItem("Character + animation", "character")
        self.export_variant.addItem("Static character pose", "static")
        of.addRow("3D content", self.export_variant)
        self.character_file = QLineEdit()
        self.character_file.setPlaceholderText("Choose the actual rigged character: .blend / .fbx / .glb / .gltf")
        browse_character = QPushButton("Browse…")
        browse_character.clicked.connect(self._choose_export_character)
        character_row = QHBoxLayout()
        character_row.addWidget(self.character_file, 1)
        character_row.addWidget(browse_character)
        of.addRow("Character file", character_row)
        self.target_armature = QLineEdit()
        self.target_armature.setPlaceholderText("Required when the file contains several armatures")
        of.addRow("Target armature name", self.target_armature)
        self.keep_scene_action = QCheckBox("Use this .blend's existing animation and Blender contact corrections")
        of.addRow("", self.keep_scene_action)
        self.range_start, self.range_end = QSpinBox(), QSpinBox()
        self.range_start.setRange(1, 1000000)
        self.range_end.setRange(0, 1000000)
        self.range_end.setSpecialValueText("Last frame")
        range_row = QHBoxLayout()
        range_row.addWidget(self.range_start)
        range_row.addWidget(self.range_end)
        of.addRow("Frame range", range_row)
        self.in_place = QCheckBox("In-place root motion")
        self.in_place.setToolTip("Applies to generated take exports. Saved scenes retain their existing root motion.")
        self.keep_scene_action.toggled.connect(lambda checked: self.in_place.setEnabled(not checked))
        self.embed_textures = QCheckBox("Include textures")
        self.embed_textures.setChecked(True)
        of.addRow("", self.in_place)
        of.addRow("", self.embed_textures)
        note = QLabel("Generated exports use the take's FPS. Saved .blend actions use scene FPS and range; set an end frame to trim. Poses are verified after reimport. Character exports require your chosen rigged asset.")
        note.setWordWrap(True)
        of.addRow(note)
        root.addWidget(opt_box)

        btn_row = QGridLayout()
        for index, (label, fn, style) in enumerate([
            ("Export .nmocap.h5 (Full Archive)", self._exp_h5, ""),
            ("Export Blender .nmocap.json", self._exp_json, "PrimaryButton"),
            ("Export .bvh Skeleton", self._exp_bvh, "PrimaryButton"),
            ("Export .fbx via Installed Blender", self._exp_fbx, ""),
            ("Export animated .glb", lambda: self._exp_3d(".glb"), "PrimaryButton"),
            ("Export .gltf + assets", lambda: self._exp_3d(".gltf"), ""),
        ]):
            b = QPushButton(label)
            if style:
                b.setObjectName(style)
            b.clicked.connect(fn)
            btn_row.addWidget(b, index // 3, index % 3)
        root.addLayout(btn_row)
        self.cancel_export = QPushButton("Cancel export")
        self.cancel_export.setEnabled(False)
        self.cancel_export.clicked.connect(self._export_cancel.set)
        root.addWidget(self.cancel_export)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        root.addWidget(self.log, 1)

    def _require_doc(self):
        if self.ctrl.active_motion is None:
            QMessageBox.information(
                self, "No Active Motion", "Open or generate a motion take first."
            )
            return None
        return self.ctrl.active_motion

    def _exp_h5(self) -> None:
        doc = self._require_doc()
        if not doc:
            return
        fp, _ = QFileDialog.getSaveFileName(
            self, "Save HDF5 Motion", f"{doc.take_id}.nmocap.h5", "Nuranova HDF5 (*.nmocap.h5)"
        )
        if fp:
            p = save_motion_file(doc, fp)
            self.log.append(f"[OK] Saved versioned HDF5 motion + JSON sidecar: {p}")

    def _exp_json(self) -> None:
        doc = self._require_doc()
        if not doc:
            return
        fp, _ = QFileDialog.getSaveFileName(
            self, "Save Blender JSON", f"{doc.take_id}.nmocap.json", "Nuranova JSON (*.nmocap.json)"
        )
        if fp:
            p = export_blender_json(doc, fp, use_corrected=self.chk_corrected.isChecked())
            self.log.append(f"[OK] Exported Blender Bridge JSON: {p}")

    def _exp_bvh(self) -> None:
        doc = self._require_doc()
        if not doc:
            return
        fp, _ = QFileDialog.getSaveFileName(self, "Export BVH", f"{doc.take_id}.bvh", "BVH (*.bvh)")
        if fp:
            p = export_bvh(
                doc,
                fp,
                use_corrected=self.chk_corrected.isChecked(),
                scale_to_cm=self.chk_bvh_cm.isChecked(),
            )
            self.log.append(f"[OK] Exported standard BVH: {p}")

    def _exp_fbx(self) -> None:
        self._exp_3d(".fbx")

    def _choose_export_character(self):
        path, _ = QFileDialog.getOpenFileName(self, "Rigged character", "", "Rigged assets (*.blend *.fbx *.glb *.gltf)")
        if path:
            self.character_file.setText(path)

    def _exp_3d(self, suffix) -> None:
        scene_mode = self.keep_scene_action.isChecked()
        doc = self.ctrl.active_motion if scene_mode else self._require_doc()
        if doc is None and not scene_mode:
            return
        blender = self.ctrl.settings.blender.blender_executable
        if not blender or not Path(blender).exists():
            QMessageBox.warning(
                self,
                "Blender Not Configured",
                "Set Blender executable path in Settings -> Blender.",
            )
            return
        if self._export_thread is not None and self._export_thread.is_alive():
            self.log.append("An export is already running.")
            return
        variant = self.export_variant.currentData()
        if (variant != "motion" or scene_mode) and not Path(self.character_file.text()).is_file():
            self.log.append("Choose the actual rigged character file first.")
            return
        name = Path(self.character_file.text()).stem if scene_mode else doc.take_id
        fp, _ = QFileDialog.getSaveFileName(self, "Export animated 3D asset", f"{name}{suffix}", f"3D asset (*{suffix})")
        if not fp:
            return
        temporary = tempfile.TemporaryDirectory(prefix="nuranova_motion_export_")
        settings = {"output": fp, "variant": variant,
                    "character_file": self.character_file.text(), "target_armature": self.target_armature.text(),
                    "frame_start": self.range_start.value(), "embed_textures": self.embed_textures.isChecked()}
        if scene_mode:
            if Path(self.character_file.text()).suffix.lower() != ".blend":
                temporary.cleanup()
                self.log.append("Keeping a corrected action requires a saved .blend file.")
                return
            settings["scene_file"] = self.character_file.text()
            settings["frame_end"] = self.range_end.value() or None
        else:
            motion_json = Path(temporary.name) / "motion.json"
            export_blender_json(doc, motion_json, use_corrected=self.chk_corrected.isChecked())
            settings.update(motion_json=str(motion_json),frame_end=self.range_end.value() or doc.num_frames,
                            fps=doc.fps,in_place=self.in_place.isChecked())
        self._export_cancel.clear()
        self.cancel_export.setEnabled(True)
        self.log.append("Export started. Blender will bake the chosen scope and verify the imported file.")

        def work():
            from engine.export.blender_job import run_blender_export
            try:
                self._export_done.emit(run_blender_export(blender, settings, self._export_cancel))
            except Exception as exc:
                self._export_error.emit(str(exc))
            finally:
                temporary.cleanup()
        self._export_thread = threading.Thread(target=work, name="nuranova-animation-export", daemon=True)
        self._export_thread.start()

    def _on_export_done(self, report):
        self.cancel_export.setEnabled(False)
        self.log.append(f"[VERIFIED] {report['file']} · {report['samples']} checked poses · max joint difference {report['max_joint_error_m'] * 1000:.2f} mm")

    def _on_export_error(self, message):
        self.cancel_export.setEnabled(False)
        self.log.append("[EXPORT FAILED] " + message)

    def shutdown(self):
        self._export_cancel.set()
        if self._export_thread is not None:
            self._export_thread.join(timeout=2)


class DiagnosticsPage(QWidget):
    """Page 18: Hardware, CUDA/CPU ORT, Model Registry, and System Diagnostics."""

    def __init__(self, ctrl: StudioController) -> None:
        super().__init__()
        self.ctrl = ctrl
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(10)
        root.addLayout(
            _header(
                "Diagnostics",
                "Inspect GPU/CUDA acceleration health, ONNX Runtime providers, pretrained model registry, and system environment",
            )
        )

        bar = QHBoxLayout()
        btn_probe = QPushButton("Re-Run Hardware & CUDA Probe")
        btn_probe.setObjectName("PrimaryButton")
        btn_probe.clicked.connect(self.refresh)
        btn_dl = QPushButton("Download Selected Model (Verify SHA-256)")
        btn_dl.clicked.connect(self._download_selected_model)
        bar.addWidget(btn_probe)
        bar.addWidget(btn_dl)
        bar.addStretch()
        root.addLayout(bar)

        self.hw_text = QTextEdit()
        self.hw_text.setReadOnly(True)
        self.hw_text.setMaximumHeight(175)
        root.addWidget(self.hw_text)

        self.model_table = QTableWidget(0, 5)
        self.model_table.setHorizontalHeaderLabels(
            ["Model ID", "Family", "Task", "Cached Locally", "License Advisory"]
        )
        self.model_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.model_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        root.addWidget(self.model_table, 1)
        self.refresh()

    def refresh(self) -> None:
        hw = detect_hardware(prefer_gpu=(self.ctrl.settings.processing.compute_backend != "cpu"))
        self.ctrl.hardware = hw
        lines = [
            f"OS / CPU: {hw.os_info} ({hw.cpu_count} logical CPUs)",
            f"NVIDIA GPU: {hw.gpu_name} | Driver: {hw.driver_version} | VRAM: {hw.vram_free_mb} MiB free / {hw.vram_total_mb} MiB total",
            f"ONNX Runtime: v{hw.ort_version} | Available EPs: {', '.join(hw.ort_available_providers)}",
            f"Selected Compute Device: {hw.selected_device.upper()} ({', '.join(hw.selected_providers)})",
            f"Blender Executable: {self.ctrl.settings.blender.blender_executable or 'Not detected'}",
        ]
        for w in hw.warnings:
            lines.append(f"[WARNING] {w}")
        self.hw_text.setPlainText("\n".join(lines))

        models = self.ctrl.model_registry.list_models()
        self.model_table.setRowCount(len(models))
        for r, m in enumerate(models):
            cached = "YES" if self.ctrl.model_registry.is_cached(m.id) else "Not Downloaded"
            for c, v in enumerate([m.id, m.family, m.task, cached, m.license]):
                self.model_table.setItem(r, c, QTableWidgetItem(v))

    def _download_selected_model(self) -> None:
        row = self.model_table.currentRow()
        models = self.ctrl.model_registry.list_models()
        if not (0 <= row < len(models)):
            return
        m = models[row]
        try:
            dest = self.ctrl.model_registry.ensure_downloaded(m.id)
            QMessageBox.information(
                self, "Model Verified", f"Downloaded and SHA-256 verified:\n{dest}"
            )
            self.refresh()
        except Exception as exc:
            QMessageBox.warning(self, "Download Failed", str(exc))


class SettingsPage(QWidget):
    """Page 19: Comprehensive Persisted Application Settings."""

    def __init__(self, ctrl: StudioController, on_theme_changed: Callable[[str], None]) -> None:
        super().__init__()
        self.ctrl = ctrl
        self.on_theme_changed = on_theme_changed
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(10)
        root.addLayout(
            _header(
                "Settings",
                "Configure General, Network, Camera, Processing, Animation, Storage, Blender, and Privacy preferences",
            )
        )

        tabs = QTabWidget()
        s = ctrl.settings

        # Tab 1: General & Storage
        t_gen = QWidget()
        fg = QFormLayout(t_gen)
        self.combo_theme = QComboBox()
        self.combo_theme.addItems(["light", "dark"])
        self.combo_theme.setCurrentText(s.general.theme)
        self.chk_reduce_motion = QCheckBox("Reduce interface animations")
        self.chk_reduce_motion.setChecked(s.general.reduce_interface_motion)
        self.chk_reduce_motion.setToolTip("Windows' Animation effects preference is also respected.")
        self.edit_proj_dir = QLineEdit(s.general.default_project_dir)
        self.chk_autosave = QCheckBox("Enable Autosave")
        self.chk_autosave.setChecked(s.general.autosave_enabled)
        self.edit_export_dir = QLineEdit(s.storage.export_dir)
        fg.addRow("UI Theme:", self.combo_theme)
        fg.addRow("Accessibility:", self.chk_reduce_motion)
        fg.addRow("Default Project Directory:", self.edit_proj_dir)
        fg.addRow("Default Export Directory:", self.edit_export_dir)
        fg.addRow("", self.chk_autosave)
        tabs.addTab(t_gen, "General & Storage")

        # Tab 2: Network & Privacy
        t_net = QWidget()
        fn = QFormLayout(t_net)
        self.edit_bind = QLineEdit(s.network.bind_ip)
        self.spin_port = QSpinBox()
        self.spin_port.setRange(1024, 65535)
        self.spin_port.setValue(s.network.server_port)
        self.chk_mdns = QCheckBox("Advertise mDNS / Zeroconf on Local Network")
        self.chk_mdns.setChecked(s.network.enable_mdns_discovery)
        self.chk_local_only = QCheckBox("Enforce Local-Network-Only Operation (No Cloud)")
        self.chk_local_only.setChecked(s.privacy.local_network_only)
        self.chk_sha_req = QCheckBox("Require SHA-256 Verification Before Allowing Phone Cleanup")
        self.chk_sha_req.setChecked(s.privacy.require_checksum_before_phone_cleanup)
        fn.addRow("Server Bind IP:", self.edit_bind)
        fn.addRow("Server Port:", self.spin_port)
        fn.addRow("", self.chk_mdns)
        fn.addRow("", self.chk_local_only)
        fn.addRow("", self.chk_sha_req)
        tabs.addTab(t_net, "Network & Privacy")

        # Tab 3: Processing & Animation
        t_proc = QWidget()
        fp = QFormLayout(t_proc)
        self.combo_compute = QComboBox()
        self.combo_compute.addItems(["auto", "cuda", "cpu"])
        self.combo_compute.setCurrentText(s.processing.compute_backend)
        self.combo_ai = QComboBox()
        self.combo_ai.addItems(["rtmlib_wholebody", "rtmlib_halpe26", "mediapipe_pose"])
        self.combo_ai.setCurrentText(s.processing.ai_backend)
        self.combo_prof = QComboBox()
        self.combo_prof.addItems(["cinema", "standard", "fast_preview"])
        self.combo_prof.setCurrentText(s.processing.default_profile)
        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(1, 16)
        self.spin_workers.setValue(s.processing.worker_count)
        fp.addRow("Compute Backend:", self.combo_compute)
        fp.addRow("AI Pose Backend:", self.combo_ai)
        fp.addRow("Default Quality Profile:", self.combo_prof)
        fp.addRow("Worker Count:", self.spin_workers)
        tabs.addTab(t_proc, "Processing & AI")

        # Tab 4: Blender
        t_bl = QWidget()
        fb = QFormLayout(t_bl)
        self.edit_blender = QLineEdit(s.blender.blender_executable)
        btn_b = QPushButton("Browse Blender.exe...")
        btn_b.clicked.connect(self._browse_blender)
        row_b = QHBoxLayout()
        row_b.addWidget(self.edit_blender, 1)
        row_b.addWidget(btn_b)
        fb.addRow("Blender Executable:", row_b)
        tabs.addTab(t_bl, "Blender")

        root.addWidget(tabs, 1)

        btn_save = QPushButton("Save & Apply Settings")
        btn_save.setObjectName("PrimaryButton")
        btn_save.clicked.connect(self._save)
        root.addWidget(btn_save)

    def _browse_blender(self) -> None:
        fp, _ = QFileDialog.getOpenFileName(
            self, "Select blender.exe", "C:/Program Files/Blender Foundation", "Executable (*.exe)"
        )
        if fp:
            self.edit_blender.setText(fp)

    def _save(self) -> None:
        s = self.ctrl.settings
        s.general.theme = self.combo_theme.currentText()  # type: ignore[assignment]
        s.general.reduce_interface_motion = self.chk_reduce_motion.isChecked()
        s.general.default_project_dir = self.edit_proj_dir.text()
        s.general.autosave_enabled = self.chk_autosave.isChecked()
        s.storage.export_dir = self.edit_export_dir.text()
        s.network.bind_ip = self.edit_bind.text()
        s.network.server_port = self.spin_port.value()
        s.network.enable_mdns_discovery = self.chk_mdns.isChecked()
        s.privacy.local_network_only = self.chk_local_only.isChecked()
        s.privacy.require_checksum_before_phone_cleanup = self.chk_sha_req.isChecked()
        s.processing.compute_backend = self.combo_compute.currentText()  # type: ignore[assignment]
        s.processing.ai_backend = self.combo_ai.currentText()  # type: ignore[assignment]
        s.processing.default_profile = self.combo_prof.currentText()  # type: ignore[assignment]
        s.processing.worker_count = self.spin_workers.value()
        s.blender.blender_executable = self.edit_blender.text()
        s.save(self.ctrl._settings_path)
        self.on_theme_changed(s.general.theme)
        self.ctrl.status_message.emit("Settings saved and applied.")


class AboutPage(QWidget):
    """Page 20: About Nuranova Mocap Studio & Third-Party License Register."""

    def __init__(self, ctrl: StudioController) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(10)
        root.addLayout(
            _header(
                "About Nuranova Mocap Studio", "Version 1.0.0 — Developed by Nuranova Solutions"
            )
        )

        info = QLabel(
            "<b>Nuranova Mocap Studio v1.0.0</b><br>"
            "Developer: <b>Nuranova Solutions</b><br>"
            "Development Team: <b>Kawshal, Shanal, and Shehan</b><br>"
            "Website: <a href='https://nuranova.lk/'>https://nuranova.lk/</a><br>"
            "Target Ecosystem: Windows 10/11 x64 Studio · Android Camera2 App · Blender 4.5 LTS / 5.0+ Bridge"
        )
        info.setOpenExternalLinks(True)
        root.addWidget(info)

        lic_box = QGroupBox("Third-Party License Register & Credits")
        lic_txt = QTextEdit()
        lic_txt.setReadOnly(True)
        lic_path = Path(__file__).resolve().parents[3] / "LICENSE_REGISTER.md"
        if lic_path.exists():
            lic_txt.setPlainText(lic_path.read_text(encoding="utf-8"))
        else:
            lic_txt.setPlainText(
                "NumPy (BSD-3), SciPy (BSD-3), OpenCV (Apache-2.0), PySide6 (LGPL-3.0), FastAPI (MIT), RTMLib (Apache-2.0), MediaPipe (Apache-2.0), ONNX Runtime (MIT), HDF5/h5py (BSD-3)."
            )
        QVBoxLayout(lic_box).addWidget(lic_txt)
        root.addWidget(lic_box, 1)
