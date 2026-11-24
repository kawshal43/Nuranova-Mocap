"""Central Studio Application Controller wiring all services, server thread, undo/redo, and signals."""

from __future__ import annotations

import copy
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np
import uvicorn
from PySide6.QtCore import QObject, Signal
from scipy.spatial.transform import Rotation

from desktop.camera_manager.manager import CameraManager
from desktop.network.server import StudioNetworkServer
from desktop.project_manager.manager import MocapProject, ProjectManager
from desktop.settings.app_settings import AppSettings
from desktop.workers.job_queue import ProcessingQueueManager
from engine.ai_tracking.runtime import HardwareDiagnostics, detect_hardware
from engine.data_format.motion_io import (
    MotionCaptureDocument,
    SkeletalMotionTrack,
    export_blender_json,
    load_motion_file,
    save_motion_file,
)
from engine.marker_tracking.marker_kit import MarkerKitConfig, create_default_cinema_marker_kit
from engine.math_core.camera_model import (
    CalibratedCamera,
    CameraExtrinsics,
    CameraIntrinsics,
    MultiCameraCalibration,
)
from engine.math_core.transforms import quat_identity
from engine.motion_cleanup.cleanup import MotionCleanupConfig, clean_skeletal_motion
from engine.optimization.hybrid_engine import HybridCinemaReconstructor
from engine.skeleton.rig_definition import (
    BONE_INDEX,
    BONE_NAMES,
    ActorProfile,
    compute_forward_kinematics_sequence,
)
from model_registry.registry import ModelRegistry


class StudioController(QObject):
    """Single source of truth for Nuranova Mocap Studio state and signals."""

    project_changed = Signal()
    cameras_changed = Signal()
    calibration_changed = Signal()
    motion_changed = Signal()
    jobs_changed = Signal()
    server_state_changed = Signal(bool, str)
    status_message = Signal(str)

    def __init__(self, settings_path: Optional[Path] = None) -> None:
        super().__init__()
        self.settings = AppSettings.load(settings_path)
        self.project_manager = ProjectManager()
        self.camera_manager = CameraManager()
        self.job_queue = ProcessingQueueManager(max_workers=max(1, min(4, self.settings.processing.worker_count)))
        self.model_registry = ModelRegistry()
        self.hardware: HardwareDiagnostics = detect_hardware(
            prefer_gpu=(self.settings.processing.compute_backend != "cpu")
        )

        self.active_calibration: Optional[MultiCameraCalibration] = None
        self.active_actor: ActorProfile = ActorProfile(actor_id="default_actor", name="Default Adult Actor (1.72m)")
        self.active_marker_kit: MarkerKitConfig = create_default_cinema_marker_kit()
        self.active_motion: Optional[MotionCaptureDocument] = None
        self._undo_stack: List[Optional[SkeletalMotionTrack]] = []
        self._redo_stack: List[Optional[SkeletalMotionTrack]] = []

        default_storage = Path(self.settings.storage.cache_dir) / "incoming"
        self.network_server = StudioNetworkServer(
            self.camera_manager,
            storage_root=default_storage,
            port=self.settings.network.server_port,
        )
        self._uvicorn_server: Optional[uvicorn.Server] = None
        self._server_thread: Optional[threading.Thread] = None
        self.server_running: bool = False

        self.camera_manager.add_listener(lambda ev, did: self.cameras_changed.emit())
        self.job_queue.add_listener(lambda job: self.jobs_changed.emit())

    # ---------------- Server Control ----------------
    def start_server(self) -> bool:
        if self.server_running:
            return True
        port = self.settings.network.server_port
        host = self.settings.network.bind_ip
        self.network_server.port = port
        if self.settings.network.enable_mdns_discovery:
            self.network_server.start_mdns_advertisement()

        config = uvicorn.Config(
            self.network_server.app,
            host=host,
            port=port,
            log_level="warning",
        )
        self._uvicorn_server = uvicorn.Server(config)
        self._server_thread = threading.Thread(
            target=self._uvicorn_server.run, name="nuranova-api-server", daemon=True
        )
        self._server_thread.start()
        self.server_running = True
        msg = f"Local Studio Server running on {host}:{port} (PIN: {self.camera_manager.get_active_pin()})"
        self.server_state_changed.emit(True, msg)
        self.status_message.emit(msg)
        return True

    def stop_server(self) -> None:
        if not self.server_running:
            return
        self.network_server.stop_mdns_advertisement()
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
        self.server_running = False
        self.server_state_changed.emit(False, "Local Studio Server stopped")
        self.status_message.emit("Local Studio Server stopped.")

    # ---------------- Project Helpers ----------------
    def create_project(self, parent_dir: str | Path, name: str) -> MocapProject:
        proj = self.project_manager.create_project(parent_dir, name)
        self._bind_project_paths(proj)
        self.project_changed.emit()
        self.status_message.emit(f"Created project '{proj.name}' at {proj.root_dir}")
        return proj

    def open_project(self, path: str | Path) -> MocapProject:
        proj = self.project_manager.open_project(path)
        self._bind_project_paths(proj)
        # Load most recent calibration if present
        cals = sorted(proj.subdir("calibration").glob("*.json"))
        if cals:
            try:
                self.active_calibration = MultiCameraCalibration.load(cals[-1])
                self.calibration_changed.emit()
            except Exception:
                pass
        # Load most recent take if present
        takes = self.project_manager.list_takes()
        if takes and takes[0].get("motion_h5_path") and Path(takes[0]["motion_h5_path"]).exists():
            try:
                self.set_active_motion(load_motion_file(takes[0]["motion_h5_path"]))
            except Exception:
                pass
        self.project_changed.emit()
        self.status_message.emit(f"Opened project '{proj.name}'")
        return proj

    def _bind_project_paths(self, proj: MocapProject) -> None:
        self.network_server.set_storage_root(proj.root_dir)
        self.network_server.bridge_project_name = proj.name
        self.network_server.bridge_project_path = str(proj.root_dir)
        for take in self.project_manager.list_takes():
            bj = take.get("blender_json_path")
            if bj and Path(bj).exists():
                self.network_server.register_bridge_take(take["take_id"], Path(bj))

    # ---------------- Motion & Undo/Redo ----------------
    def set_active_motion(self, doc: MotionCaptureDocument) -> None:
        self.active_motion = doc
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.motion_changed.emit()

    def apply_motion_cleanup(self, cfg: MotionCleanupConfig, frame_range: Optional[tuple[int, int]] = None) -> None:
        if self.active_motion is None:
            return
        self._undo_stack.append(copy.deepcopy(self.active_motion.corrected_motion))
        self._redo_stack.clear()
        cleaned = clean_skeletal_motion(
            self.active_motion.raw_motion,
            actor=self.active_actor,
            fps=self.active_motion.fps,
            config=cfg,
            frame_range=frame_range,
        )
        self.active_motion.corrected_motion = cleaned
        self.motion_changed.emit()
        self.status_message.emit(f"Applied non-destructive cleanup ({cfg.filter_type}, strength={cfg.strength:.2f}).")

    def undo_motion_edit(self) -> bool:
        if self.active_motion is None or not self._undo_stack:
            return False
        self._redo_stack.append(copy.deepcopy(self.active_motion.corrected_motion))
        self.active_motion.corrected_motion = self._undo_stack.pop()
        self.motion_changed.emit()
        self.status_message.emit("Undid motion cleanup edit.")
        return True

    def redo_motion_edit(self) -> bool:
        if self.active_motion is None or not self._redo_stack:
            return False
        self._undo_stack.append(copy.deepcopy(self.active_motion.corrected_motion))
        self.active_motion.corrected_motion = self._redo_stack.pop()
        self.motion_changed.emit()
        self.status_message.emit("Redid motion cleanup edit.")
        return True

    def generate_synthetic_demo_take(self, take_name: str = "synthetic_demo_walk") -> MotionCaptureDocument:
        """Generate a clearly labeled SYNTHETIC demo capture take & 4-camera rig for offline UI inspection & Blender testing."""
        # 4 synthetic ring cameras
        target = np.array([0.0, 0.0, 1.0])
        up = np.array([0.0, 0.0, 1.0])
        K = np.array([[1400.0, 0.0, 960.0], [0.0, 1400.0, 540.0], [0.0, 0.0, 1.0]])
        dist = np.array([-0.1, 0.03, 0.0, 0.0, 0.0])
        cams = {}
        for i in range(4):
            ang = 2.0 * np.pi * i / 4.0 + 0.3
            pos = np.array([3.0 * np.cos(ang), 3.0 * np.sin(ang), 1.85])
            zc = (target - pos)
            zc /= np.linalg.norm(zc)
            xc = np.cross(zc, up)
            xc /= np.linalg.norm(xc)
            yc = np.cross(zc, xc)
            R_cw = np.vstack([xc, yc, zc])
            t_cw = -R_cw @ pos
            uid = f"cam_{i+1:02d}"
            cams[uid] = CalibratedCamera(
                CameraIntrinsics(uid, 1920, 1080, K.copy(), dist.copy(), reprojection_error_px=0.28),
                CameraExtrinsics(uid, R_cw, t_cw, reprojection_error_px=0.31),
            )
        calib = MultiCameraCalibration(
            profile_name="Synthetic 4-Camera Demo Rig (NOT Physical Hardware)",
            cameras=cams,
            global_reprojection_rmse_px=0.31,
            known_distance_expected_m=1.0,
            known_distance_measured_m=1.0004,
            known_distance_error_mm=0.4,
            notes="SYNTHETIC_GROUND_TRUTH demonstration calibration.",
        )
        self.active_calibration = calib
        self.calibration_changed.emit()

        T = 90
        fps = 60.0
        root_t = np.zeros((T, 3))
        local_q = np.tile(quat_identity(), (T, len(BONE_NAMES), 1))
        for t in range(T):
            ph = 2.0 * np.pi * (t / 45.0)
            root_t[t] = [0.0, -0.5 + 1.0 * (t / T), 0.96 + 0.018 * np.sin(2 * ph)]
            local_q[t, BONE_INDEX["chest"]] = Rotation.from_euler("z", 10.0 * np.sin(ph), degrees=True).as_quat()
            local_q[t, BONE_INDEX["left_upper_arm"]] = Rotation.from_euler("yz", [58.0, 22.0 * np.sin(ph)], degrees=True).as_quat()
            local_q[t, BONE_INDEX["left_forearm"]] = Rotation.from_euler("z", -38.0 - 14.0 * np.sin(ph), degrees=True).as_quat()
            local_q[t, BONE_INDEX["right_upper_arm"]] = Rotation.from_euler("yz", [-58.0, -22.0 * np.sin(ph)], degrees=True).as_quat()
            local_q[t, BONE_INDEX["right_forearm"]] = Rotation.from_euler("z", 38.0 + 14.0 * np.sin(ph), degrees=True).as_quat()
            local_q[t, BONE_INDEX["left_upper_leg"]] = Rotation.from_euler("x", 20.0 * np.sin(ph), degrees=True).as_quat()
            local_q[t, BONE_INDEX["left_lower_leg"]] = Rotation.from_euler("x", -max(5.0, 42.0 * np.sin(ph)), degrees=True).as_quat()
            local_q[t, BONE_INDEX["right_upper_leg"]] = Rotation.from_euler("x", -20.0 * np.sin(ph), degrees=True).as_quat()
            local_q[t, BONE_INDEX["right_lower_leg"]] = Rotation.from_euler("x", -max(5.0, -42.0 * np.sin(ph)), degrees=True).as_quat()

        fk_pos, _ = compute_forward_kinematics_sequence(root_t, local_q, self.active_actor.rest_offsets_m)
        body26 = np.zeros((T, 26, 3))
        m = {
            19: "hips", 18: "neck", 17: "head_end",
            5: "left_upper_arm", 7: "left_forearm", 9: "left_hand",
            6: "right_upper_arm", 8: "right_forearm", 10: "right_hand",
            11: "left_upper_leg", 13: "left_lower_leg", 15: "left_foot", 20: "left_toe", 22: "left_toe", 24: "left_foot",
            12: "right_upper_leg", 14: "right_lower_leg", 16: "right_foot", 21: "right_toe", 23: "right_toe", 25: "right_foot",
        }
        for k, b in m.items():
            body26[:, k] = fk_pos[:, BONE_INDEX[b]]
        hp = fk_pos[:, BONE_INDEX["head"]]
        body26[:, 0] = hp + [0.0, 0.09, 0.02]
        body26[:, 3] = hp + [-0.08, 0.0, 0.02]
        body26[:, 4] = hp + [0.08, 0.0, 0.02]

        rng = np.random.default_rng(2026)
        ai_obs = {}
        for uid, cam in cams.items():
            arr = np.zeros((T, 26, 3), dtype=np.float32)
            for t in range(T):
                arr[t, :, :2] = cam.project_world_points(body26[t], True) + rng.normal(0.0, 1.2, size=(26, 2))
                arr[t, :, 2] = 0.92
            ai_obs[uid] = arr

        recon = HybridCinemaReconstructor(calibration=calib, actor=self.active_actor, marker_kit=self.active_marker_kit)
        proj_id = self.project_manager.active_project.project_id if self.project_manager.active_project else "demo_proj"
        doc, _ = recon.reconstruct_from_2d_observations(
            project_id=proj_id,
            session_id="synthetic_demo",
            take_id=take_name,
            fps=fps,
            frame_timestamps_ns=np.arange(T, dtype=np.int64) * int(1e9 / fps),
            ai_obs_2d_per_cam=ai_obs,
            capture_mode="markerless_ai",
            profile_name="standard",
        )

        if self.project_manager.active_project is not None:
            proj = self.project_manager.active_project
            calib.save(proj.subdir("calibration") / "synthetic_demo_calib.json")
            h5_p = save_motion_file(doc, proj.subdir("corrected_motion") / f"{take_name}.nmocap.h5")
            bj_p = export_blender_json(doc, proj.subdir("exports") / f"{take_name}_blender.json", use_corrected=True)
            self.project_manager.register_take(
                take_id=take_name,
                session_id="synthetic_demo",
                name=f"{take_name} [SYNTHETIC DEMO]",
                capture_mode="markerless_ai",
                fps=fps,
                num_frames=T,
                motion_h5_path=h5_p,
                blender_json_path=bj_p,
            )
            self.network_server.register_bridge_take(take_name, bj_p)
            self.project_changed.emit()

        self.set_active_motion(doc)
        self.status_message.emit(f"Generated SYNTHETIC demo take '{take_name}' ({T} frames @ 60 FPS).")
        return doc

    def shutdown(self) -> None:
        self.stop_server()
        self.job_queue.shutdown()
        self.project_manager.close_project()
        self.settings.save()
