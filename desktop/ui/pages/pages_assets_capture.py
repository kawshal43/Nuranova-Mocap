"""Pages 12-15: Marker Editor, Actor Manager, Facial Capture, and Hand Capture."""

from __future__ import annotations


import numpy as np
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from desktop.ui.app_controller import StudioController
from desktop.ui.pages.pages_workspace import _header
from engine.face.face_tracker import MEDIAPIPE_52_BLENDSHAPES, FacialCaptureEngine
from engine.hands.hand_tracker import HandCaptureEngine
from engine.marker_tracking.marker_kit import (
    MARKER_GROUPS,
    MarkerDefinition,
    MarkerKitConfig,
    generate_printable_marker_sheet_pdf,
)
from engine.skeleton.rig_definition import BONE_NAMES, ActorProfile
from engine.video.video_io import iter_video_frames


class MarkerEditorPage(QWidget):
    """Page 12: Wearable Physical Marker Kit Editor & Printable AprilTag/ArUco Sheet Generator."""

    def __init__(self, ctrl: StudioController) -> None:
        super().__init__()
        self.ctrl = ctrl
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(10)
        root.addLayout(
            _header(
                "Marker Editor",
                "Configure 20–100+ visible-light markers, asymmetric rigid clusters, and generate valid printable AprilTag/ArUco templates",
            )
        )

        bar = QHBoxLayout()
        btn_add = QPushButton("+ Add Custom Marker")
        btn_add.clicked.connect(self._add_marker)
        btn_pdf = QPushButton("Export Printable AprilTag 36h11 Sheet (PDF)...")
        btn_pdf.setObjectName("PrimaryButton")
        btn_pdf.clicked.connect(self._export_pdf)
        btn_save = QPushButton("Save Kit JSON...")
        btn_save.clicked.connect(self._save_kit)
        btn_load = QPushButton("Load Kit JSON...")
        btn_load.clicked.connect(self._load_kit)
        for b in (btn_add, btn_pdf, btn_save, btn_load):
            bar.addWidget(b)
        bar.addStretch()
        root.addLayout(bar)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            [
                "Marker ID",
                "Name",
                "Body Group",
                "Attached Bone",
                "Type",
                "Tag ID",
                "Local Offset (m)",
            ]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        root.addWidget(self.table, 1)

        self.lbl_clusters = QLabel("")
        root.addWidget(self.lbl_clusters)
        self.refresh()

    def refresh(self) -> None:
        kit = self.ctrl.active_marker_kit
        self.table.setRowCount(len(kit.markers))
        for r, m in enumerate(kit.markers):
            vals = [
                m.marker_id,
                m.name,
                m.group,
                m.attached_bone,
                m.marker_type,
                str(m.tag_id) if m.tag_id is not None else "--",
                f"({m.local_offset_m[0]:.3f}, {m.local_offset_m[1]:.3f}, {m.local_offset_m[2]:.3f})",
            ]
            for c, v in enumerate(vals):
                self.table.setItem(r, c, QTableWidgetItem(v))
        self.lbl_clusters.setText(
            f"Active Kit: {kit.kit_name} | Total Markers: {len(kit.markers)} | Rigid 4-Point Clusters: {len(kit.clusters)}"
        )

    def _add_marker(self) -> None:
        kit = self.ctrl.active_marker_kit
        idx = len(kit.markers) + 1
        kit.markers.append(
            MarkerDefinition(
                marker_id=f"custom_m_{idx:02d}",
                name=f"Custom Marker {idx}",
                group=MARKER_GROUPS[idx % len(MARKER_GROUPS)],
                attached_bone="chest",
                local_offset_m=[0.0, 0.08, 0.0],
                marker_type="sphere_white",
            )
        )
        self.refresh()

    def _export_pdf(self) -> None:
        out, _ = QFileDialog.getSaveFileName(
            self,
            "Save Printable Marker Sheet PDF",
            "Nuranova_Wearable_AprilTags.pdf",
            "PDF (*.pdf)",
        )
        if not out:
            return
        kit = self.ctrl.active_marker_kit
        tag_markers = [m for m in kit.markers if m.tag_id is not None]
        tag_ids = [int(m.tag_id) for m in tag_markers] or list(range(12))
        labels = (
            [m.name for m in tag_markers] if tag_markers else [f"Cluster Tag #{i}" for i in tag_ids]
        )
        p = generate_printable_marker_sheet_pdf(
            tag_ids, out, dictionary_name="DICT_APRILTAG_36h11", tag_size_mm=40.0, labels=labels
        )
        QMessageBox.information(
            self, "Printable Marker Sheet Saved", f"Generated valid AprilTag 36h11 PDF:\n{p}"
        )

    def _save_kit(self) -> None:
        fp, _ = QFileDialog.getSaveFileName(
            self, "Save Marker Kit", "nuranova_marker_kit.json", "JSON (*.json)"
        )
        if fp:
            self.ctrl.active_marker_kit.save(fp)

    def _load_kit(self) -> None:
        fp, _ = QFileDialog.getOpenFileName(self, "Load Marker Kit", "", "JSON (*.json)")
        if fp:
            self.ctrl.active_marker_kit = MarkerKitConfig.load(fp)
            self.refresh()


class ActorManagerPage(QWidget):
    """Page 13: Actor Proportions & Skeletal Calibration Manager."""

    def __init__(self, ctrl: StudioController) -> None:
        super().__init__()
        self.ctrl = ctrl
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(10)
        root.addLayout(
            _header(
                "Actor Manager",
                "Manage actor skeletal proportions, T-pose/A-pose calibration, and saved actor profiles",
            )
        )

        form_box = QGroupBox("Active Actor Profile")
        f = QFormLayout(form_box)
        self.edit_name = QLineEdit(ctrl.active_actor.name)
        self.spin_height = QDoubleSpinBox()
        self.spin_height.setRange(0.9, 2.3)
        self.spin_height.setSingleStep(0.01)
        self.spin_height.setValue(ctrl.active_actor.height_m)
        self.spin_height.setSuffix(" m")
        self.combo_pose = QComboBox()
        self.combo_pose.addItems(["T-pose", "A-pose", "Neutral"])
        btn_scale = QPushButton("Apply Height Scale & Update Rig")
        btn_scale.setObjectName("PrimaryButton")
        btn_scale.clicked.connect(self._apply_actor)
        btn_save = QPushButton("Save Actor Profile JSON...")
        btn_save.clicked.connect(self._save_actor)
        btn_load = QPushButton("Load Actor Profile JSON...")
        btn_load.clicked.connect(self._load_actor)
        row = QHBoxLayout()
        row.addWidget(btn_scale)
        row.addWidget(btn_save)
        row.addWidget(btn_load)
        f.addRow("Actor Name:", self.edit_name)
        f.addRow("Actor Height:", self.spin_height)
        f.addRow("Calibration Pose:", self.combo_pose)
        f.addRow("", row)
        root.addWidget(form_box)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(
            ["Bone Name", "Rest Offset [X, Y, Z] (m)", "Bone Length (cm)"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        root.addWidget(self.table, 1)
        self.refresh()

    def refresh(self) -> None:
        a = self.ctrl.active_actor
        self.table.setRowCount(len(BONE_NAMES))
        for r, name in enumerate(BONE_NAMES):
            off = a.rest_offsets_m[r]
            length_cm = float(np.linalg.norm(off)) * 100.0
            self.table.setItem(r, 0, QTableWidgetItem(name))
            self.table.setItem(
                r, 1, QTableWidgetItem(f"[{off[0]:.3f}, {off[1]:.3f}, {off[2]:.3f}]")
            )
            self.table.setItem(r, 2, QTableWidgetItem(f"{length_cm:.1f} cm"))

    def _apply_actor(self) -> None:
        a = self.ctrl.active_actor
        ratio = self.spin_height.value() / max(a.height_m, 0.5)
        a.name = self.edit_name.text().strip() or a.name
        a.height_m = self.spin_height.value()
        a.calibration_pose_type = self.combo_pose.currentText()
        a.rest_offsets_m = a.rest_offsets_m * ratio
        self.refresh()
        self.ctrl.status_message.emit(f"Updated actor '{a.name}' ({a.height_m:.2f} m).")

    def _save_actor(self) -> None:
        fp, _ = QFileDialog.getSaveFileName(
            self, "Save Actor Profile", "actor_profile.json", "JSON (*.json)"
        )
        if fp:
            self.ctrl.active_actor.save(fp)

    def _load_actor(self) -> None:
        fp, _ = QFileDialog.getOpenFileName(self, "Load Actor Profile", "", "JSON (*.json)")
        if fp:
            self.ctrl.active_actor = ActorProfile.load(fp)
            self.edit_name.setText(self.ctrl.active_actor.name)
            self.spin_height.setValue(self.ctrl.active_actor.height_m)
            self.refresh()


class FacialCapturePage(QWidget):
    """Page 14: Dedicated Close-Up Facial Capture (MediaPipe 52 Blendshapes + Neutral Calibration)."""

    def __init__(self, ctrl: StudioController) -> None:
        super().__init__()
        self.ctrl = ctrl
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(10)
        root.addLayout(
            _header(
                "Facial Capture",
                "MediaPipe Face Landmarker (52 blendshape coefficients, head pose, eye gaze, neutral calibration)",
            )
        )

        bar = QHBoxLayout()
        btn_vid = QPushButton("Process Close-Up Face Video...")
        btn_vid.setObjectName("PrimaryButton")
        btn_vid.clicked.connect(self._process_face_video)
        self.spin_max_frames = QSpinBox()
        self.spin_max_frames.setRange(5, 5000)
        self.spin_max_frames.setValue(120)
        bar.addWidget(btn_vid)
        bar.addWidget(QLabel("Max Frames:"))
        bar.addWidget(self.spin_max_frames)
        bar.addStretch()
        root.addLayout(bar)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(
            ["Blendshape Channel (52)", "Mean Score", "Peak Score"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        root.addWidget(self.table, 1)

        self.lbl_info = QLabel(
            "Note: MediaPipe 52 blendshape names align with ARKit naming; tune character gains inside Blender Bridge."
        )
        root.addWidget(self.lbl_info)

    def _process_face_video(self) -> None:
        fp, _ = QFileDialog.getOpenFileName(
            self, "Select Close-Up Face Video", "", "Video (*.mp4 *.mov *.avi)"
        )
        if not fp:
            return
        max_f = self.spin_max_frames.value()
        eng = FacialCaptureEngine()
        results = []
        try:
            for _, _, frame in iter_video_frames(fp, start_frame=0, end_frame=max_f):
                results.append(eng.infer_frame(frame))
            if len(results) >= 5:
                eng.calibrate_neutral(results[:5])
            track = eng.build_track(results)
        finally:
            eng.close()

        if self.ctrl.active_motion is not None:
            self.ctrl.active_motion.face = track
            self.ctrl.motion_changed.emit()

        self.table.setRowCount(len(MEDIAPIPE_52_BLENDSHAPES))
        for i, name in enumerate(MEDIAPIPE_52_BLENDSHAPES):
            col = track.blendshapes[:, i] if track.blendshapes.shape[0] > 0 else np.zeros(1)
            self.table.setItem(i, 0, QTableWidgetItem(name))
            self.table.setItem(i, 1, QTableWidgetItem(f"{float(np.mean(col)):.3f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{float(np.max(col)):.3f}"))
        self.lbl_info.setText(
            f"Processed {len(results)} frames using {eng.backend_label}. Attached to active motion take."
        )


class HandCapturePage(QWidget):
    """Page 15: Dedicated Close-Up Hand & Finger Capture (21 Landmarks -> 16 Finger Bones)."""

    def __init__(self, ctrl: StudioController) -> None:
        super().__init__()
        self.ctrl = ctrl
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(10)
        root.addLayout(
            _header(
                "Hand & Finger Capture",
                "Dedicated close-up hand camera tracking (MediaPipe 21 landmarks / RTMW hands -> 16 finger rotations)",
            )
        )

        bar = QHBoxLayout()
        btn_vid = QPushButton("Process Close-Up Hand Video...")
        btn_vid.setObjectName("PrimaryButton")
        btn_vid.clicked.connect(self._process_hand_video)
        self.spin_max_frames = QSpinBox()
        self.spin_max_frames.setRange(5, 5000)
        self.spin_max_frames.setValue(120)
        bar.addWidget(btn_vid)
        bar.addWidget(QLabel("Max Frames:"))
        bar.addWidget(self.spin_max_frames)
        bar.addStretch()
        root.addLayout(bar)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlainText(
            "Hand Capture Advisory:\n"
            "Distant full-body camera views rarely provide enough pixel resolution for accurate finger articulation.\n"
            "Assign a dedicated close-up smartphone (role: hand_dual / hand_left / hand_right) or import a dedicated hand video here."
        )
        root.addWidget(self.log, 1)

    def _process_hand_video(self) -> None:
        fp, _ = QFileDialog.getOpenFileName(
            self, "Select Close-Up Hand Video", "", "Video (*.mp4 *.mov *.avi)"
        )
        if not fp:
            return
        max_f = self.spin_max_frames.value()
        eng = HandCaptureEngine()
        frames = []
        try:
            for _, _, bgr in iter_video_frames(fp, start_frame=0, end_frame=max_f):
                frames.append(eng.infer_frame(bgr))
            track = eng.build_track(frames)
        finally:
            eng.close()

        l_det = sum(1 for f in frames if "left" in f)
        r_det = sum(1 for f in frames if "right" in f)
        if self.ctrl.active_motion is not None:
            self.ctrl.active_motion.hands = track
            self.ctrl.motion_changed.emit()
        self.log.append(
            f"[OK] Processed {len(frames)} frames: Left hand detected in {l_det} frames, Right hand detected in {r_det} frames. Attached to active motion."
        )
