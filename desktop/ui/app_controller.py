"""Central Studio Application Controller wiring all services, server thread, QR pairing, recording, undo/redo, and signals."""

from __future__ import annotations

import copy
import socket
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import numpy as np
import uvicorn
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage, QPixmap
from scipy.spatial.transform import Rotation

from desktop.camera_manager.manager import CameraManager
from desktop.network.server import (
    StudioNetworkServer,
    check_windows_firewall,
    enumerate_lan_ipv4_addresses,
    request_elevated_firewall_fix,
)
from desktop.project_manager.manager import MocapProject, ProjectManager
from desktop.settings.app_settings import AppSettings
from desktop.workers.job_queue import ProcessingQueueManager
from desktop.workers.live_preview import LiveMocapPreviewWorker
from engine.ai_tracking.pose_tracker import AIPoseTracker2D
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
from engine.optimization.hybrid_engine import PROCESSING_PROFILES, HybridCinemaReconstructor
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
    recording_state_changed = Signal(bool, str)
    status_message = Signal(str)

    def __init__(self, settings_path: Optional[Path] = None) -> None:
        super().__init__()
        self._settings_path = settings_path
        self.settings = AppSettings.load(settings_path)
        base_state_dir = settings_path.parent if settings_path is not None else AppSettings.default_path().parent

        self.project_manager = ProjectManager(
            recent_registry_path=(base_state_dir / "recent_projects.json") if settings_path is not None else None
        )
        self.camera_manager = CameraManager(persistence_path=base_state_dir / "paired_cameras.json")
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

        self.imported_videos: Dict[str, Path] = {}
        self.is_recording: bool = False
        self.active_session_id: Optional[str] = None
        self.active_take_id: Optional[str] = None
        self.last_firewall_check: Optional[Dict[str, Any]] = None

        default_storage = Path(self.settings.storage.cache_dir) / "incoming"
        self.network_server = StudioNetworkServer(
            self.camera_manager,
            storage_root=default_storage,
            port=self.settings.network.server_port,
        )
        if self.settings.network.preferred_lan_ip:
            self.network_server.preferred_lan_ip = self.settings.network.preferred_lan_ip

        self._uvicorn_server: Optional[uvicorn.Server] = None
        self._server_thread: Optional[threading.Thread] = None
        self.server_running: bool = False
        self.server_error: Optional[str] = None

        self.camera_manager.add_listener(lambda ev, did: self.cameras_changed.emit())
        self.job_queue.add_listener(lambda job: self.jobs_changed.emit())
        self.network_server.add_transfer_listener(self._on_transfer_completed)

        self.live_overlay_jpegs: Dict[str, bytes] = {}
        self.live_preview = LiveMocapPreviewWorker(
            camera_manager=self.camera_manager,
            get_calibration=lambda: self.active_calibration,
            get_actor=lambda: self.active_actor,
        )
        self.live_preview.overlay_updated.connect(self._on_live_overlay)

    def _on_live_overlay(self, device_id: str, jpeg_bytes: bytes) -> None:
        self.live_overlay_jpegs[device_id] = jpeg_bytes

    # ---------------- Network, LAN IP & QR Pairing ----------------
    def get_available_lan_ips(self) -> List[Tuple[str, str]]:
        return enumerate_lan_ipv4_addresses()

    def get_active_lan_ip(self) -> str:
        return self.network_server.get_active_lan_ip()

    def set_preferred_lan_ip(self, ip: str) -> None:
        cleaned = ip.strip()
        self.settings.network.preferred_lan_ip = cleaned
        self.network_server.preferred_lan_ip = cleaned or None
        self.settings.save(self._settings_path)
        if self.server_running and self.settings.network.enable_mdns_discovery:
            self.network_server.start_mdns_advertisement(bind_ip=cleaned or None)
        lan = self.get_active_lan_ip()
        self.server_state_changed.emit(
            self.server_running,
            f"Studio LAN endpoint: http://{lan}:{self.settings.network.server_port}",
        )

    def get_pairing_uri(self) -> str:
        lan_ip = self.get_active_lan_ip()
        port = self.settings.network.server_port
        pin = self.camera_manager.get_active_pin()
        params = urlencode(
            {
                "host": lan_ip,
                "port": str(port),
                "pin": pin,
                "name": self.network_server.studio_name,
            }
        )
        return f"nuranova://pair?{params}"

    def generate_pairing_qr_pixmap(self, size_px: int = 210) -> Optional[QPixmap]:
        """Generate a crisp scannable QR code QPixmap for 1-tap Android phone pairing."""
        try:
            import qrcode

            qr = qrcode.QRCode(
                version=None,
                error_correction=qrcode.constants.ERROR_CORRECT_M,
                box_size=8,
                border=3,
            )
            qr.add_data(self.get_pairing_uri())
            qr.make(fit=True)
            pil_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
            pil_img = pil_img.resize((size_px, size_px))
            raw = pil_img.tobytes("raw", "RGB")
            qimg = QImage(raw, size_px, size_px, size_px * 3, QImage.Format.Format_RGB888).copy()
            return QPixmap.fromImage(qimg)
        except Exception:
            return None

    def check_firewall_status(self) -> Dict[str, Any]:
        self.last_firewall_check = check_windows_firewall(
            port=self.settings.network.server_port,
            udp_port=self.settings.network.udp_discovery_port,
        )
        return self.last_firewall_check

    def fix_windows_firewall(self) -> Tuple[bool, str]:
        ok, msg = request_elevated_firewall_fix(
            port=self.settings.network.server_port,
            udp_port=self.settings.network.udp_discovery_port,
        )
        self.check_firewall_status()
        self.status_message.emit(msg)
        return ok, msg

    # ---------------- Server Control ----------------
    def start_server(self) -> bool:
        if self.server_running:
            return True
        port = self.settings.network.server_port
        host = self.settings.network.bind_ip or "0.0.0.0"

        # Pre-check port availability before launching uvicorn thread so errors are never silent
        probe_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe_sock.bind((host, port))
        except OSError as exc:
            self.server_error = f"Port {port} is unavailable ({exc}). Close any other Studio instance or change port in Settings."
            self.server_running = False
            self.server_state_changed.emit(False, self.server_error)
            self.status_message.emit(self.server_error)
            return False
        finally:
            try:
                probe_sock.close()
            except Exception:
                pass

        self.server_error = None
        self.network_server.port = port
        if self.settings.network.preferred_lan_ip:
            self.network_server.preferred_lan_ip = self.settings.network.preferred_lan_ip

        config = uvicorn.Config(
            self.network_server.app,
            host=host,
            port=port,
            log_level="warning",
        )
        self._uvicorn_server = uvicorn.Server(config)

        def _run_uvicorn() -> None:
            try:
                assert self._uvicorn_server is not None
                self._uvicorn_server.run()
            except Exception as exc:
                self.server_error = str(exc)
                self.server_running = False

        self._server_thread = threading.Thread(
            target=_run_uvicorn, name="nuranova-api-server", daemon=True
        )
        self._server_thread.start()

        # Wait briefly until uvicorn confirms startup (or fails)
        deadline = time.monotonic() + 2.5
        while time.monotonic() < deadline:
            if self._uvicorn_server.started:
                break
            if not self._server_thread.is_alive():
                break
            time.sleep(0.05)

        if not self._uvicorn_server.started:
            self.server_running = False
            err = self.server_error or f"Studio server failed to bind {host}:{port}."
            self.server_state_changed.emit(False, err)
            self.status_message.emit(err)
            return False

        if self.settings.network.enable_mdns_discovery:
            self.network_server.start_mdns_advertisement(bind_ip=self.settings.network.preferred_lan_ip or None)
        if self.settings.network.enable_udp_beacon:
            self.network_server.start_udp_discovery(udp_port=self.settings.network.udp_discovery_port)

        self.server_running = True
        lan_ip = self.get_active_lan_ip()
        msg = f"Studio Server Online at http://{lan_ip}:{port} (PIN: {self.camera_manager.get_active_pin()})"
        self.server_state_changed.emit(True, msg)
        self.status_message.emit(msg)
        return True

    def stop_server(self) -> None:
        if not self.server_running:
            return
        self.network_server.stop_mdns_advertisement()
        self.network_server.stop_udp_discovery()
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
        if self._server_thread is not None and self._server_thread.is_alive():
            self._server_thread.join(timeout=2.0)
        self.server_running = False
        self.server_state_changed.emit(False, "Local Studio Server stopped")
        self.status_message.emit("Local Studio Server stopped.")

    # ---------------- Live Synchronized Recording ----------------
    def start_synchronized_recording(
        self,
        take_name: Optional[str] = None,
        allow_single_camera: bool = True,
    ) -> Dict[str, Any]:
        if not self.server_running:
            if not self.start_server():
                raise RuntimeError("Studio Server could not be started.")
        ok, msg = (
            self.camera_manager.can_record_any()
            if allow_single_camera
            else self.camera_manager.can_record_multi_view()
        )
        if not ok:
            raise RuntimeError(msg)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"sess_{stamp}"
        take_id = take_name or f"take_{stamp}"
        res = self.network_server.schedule_record_from_thread(
            session_id=session_id,
            take_id=take_id,
            lead_time_s=1.2,
            allow_single_camera=allow_single_camera,
        )
        self.is_recording = True
        self.active_session_id = session_id
        self.active_take_id = take_id
        n_disp = len(res.get("dispatched_devices", []))
        info = f"Recording '{take_id}' on {n_disp} connected phone(s)."
        if res.get("errors"):
            info += f" ({len(res['errors'])} camera warnings)"
        self.recording_state_changed.emit(True, info)
        self.status_message.emit(info)
        return res

    def stop_synchronized_recording(self) -> List[str]:
        session_id = self.active_session_id or "sess_manual"
        take_id = self.active_take_id or "take_manual"
        stopped = self.network_server.stop_record_from_thread(session_id=session_id, take_id=take_id)
        self.is_recording = False
        info = f"Stopped '{take_id}' on {len(stopped)} phone(s); awaiting SHA-256 verified transfers."
        self.recording_state_changed.emit(False, info)
        self.status_message.emit(info)
        return stopped

    def _on_transfer_completed(self, slot) -> None:
        if slot.verified and slot.file_kind == "video":
            self.status_message.emit(f"Verified video upload: {slot.dest_path.name} (take '{slot.take_id}')")
            self.project_changed.emit()

    def list_recorded_take_folders(self) -> List[Tuple[str, Dict[str, Path]]]:
        """Find multi-camera recorded take folders containing MP4 files ready for reconstruction."""
        roots: List[Path] = []
        if self.project_manager.active_project is not None:
            roots.append(self.project_manager.active_project.subdir("recordings"))
        roots.append(self.network_server.storage_root / "recordings")

        found: Dict[str, Dict[str, Path]] = {}
        for r in roots:
            if not r.exists():
                continue
            for take_dir in sorted(r.iterdir()):
                if not take_dir.is_dir():
                    continue
                vids: Dict[str, Path] = {}
                for mp4 in sorted(take_dir.glob("*.mp4")):
                    # Filename format from StudioNetworkServer: cam_01_take_xxx.mp4
                    parts = mp4.stem.split("_")
                    cam_uid = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 and parts[0] == "cam" else mp4.stem
                    vids[cam_uid] = mp4
                if vids:
                    found[take_dir.name] = vids
        return list(found.items())

    # ---------------- Real Multi-Camera Offline Reconstruction ----------------
    def reconstruct_from_video_files(
        self,
        video_paths_by_cam: Dict[str, Path],
        take_name: str = "reconstructed_take",
        capture_mode: str = "markerless_ai",
        profile_name: str = "standard",
        progress_cb: Optional[Callable[[float, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> MotionCaptureDocument:
        """Run real 2D AI tracking -> multi-view triangulation -> skeletal IK -> cleanup on MP4 files."""
        if self.active_calibration is None:
            raise RuntimeError("No active multi-camera calibration loaded. Calibrate or load a profile first.")
        calib_uids = list(self.active_calibration.cameras.keys())
        matched_vids: Dict[str, Path] = {}
        for idx, (uid, vpath) in enumerate(video_paths_by_cam.items()):
            target_uid = uid if uid in self.active_calibration.cameras else (calib_uids[idx] if idx < len(calib_uids) else uid)
            if target_uid in self.active_calibration.cameras and Path(vpath).exists():
                matched_vids[target_uid] = Path(vpath)
        if len(matched_vids) < 2:
            raise RuntimeError(
                f"Need at least 2 calibrated camera videos for 3D reconstruction (matched {len(matched_vids)})."
            )

        prof = PROCESSING_PROFILES.get(profile_name, PROCESSING_PROFILES["standard"])
        tracker = AIPoseTracker2D(
            backend=self.settings.processing.ai_backend,
            mode=prof.ai_model_mode,
            prefer_gpu=(self.settings.processing.compute_backend != "cpu"),
        )
        ai_obs: Dict[str, np.ndarray] = {}
        n_cams = len(matched_vids)
        try:
            for c_idx, (uid, vpath) in enumerate(matched_vids.items()):
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("Cancelled by user.")
                if progress_cb:
                    progress_cb(0.05 + 0.65 * (c_idx / n_cams), f"Running 2D AI tracking on {uid} ({vpath.name})...")
                arr, _, _ = tracker.process_video(
                    vpath,
                    step=prof.frame_stride,
                    cancel_check=(lambda: cancel_event.is_set()) if cancel_event else None,
                )
                ai_obs[uid] = arr
        finally:
            tracker.close()

        min_frames = min(arr.shape[0] for arr in ai_obs.values())
        if min_frames < 2:
            raise RuntimeError("Tracked fewer than 2 synchronized frames across cameras.")
        trimmed = {uid: arr[:min_frames] for uid, arr in ai_obs.items()}
        fps = float(self.settings.animation.output_fps / max(prof.frame_stride, 1))
        ts_ns = np.arange(min_frames, dtype=np.int64) * int(round(1e9 / max(fps, 1.0)))

        if progress_cb:
            progress_cb(0.75, f"Triangulating & solving 25-bone skeleton ({min_frames} frames)...")
        recon = HybridCinemaReconstructor(
            calibration=self.active_calibration,
            actor=self.active_actor,
            marker_kit=self.active_marker_kit,
        )
        proj_id = self.project_manager.active_project.project_id if self.project_manager.active_project else "local_proj"
        doc, _ = recon.reconstruct_from_2d_observations(
            project_id=proj_id,
            session_id="offline_recon",
            take_id=take_name,
            fps=fps,
            frame_timestamps_ns=ts_ns,
            ai_obs_2d_per_cam=trimmed,
            capture_mode=capture_mode,
            profile_name=profile_name,
        )

        if self.project_manager.active_project is not None:
            proj = self.project_manager.active_project
            h5_p = save_motion_file(doc, proj.subdir("corrected_motion") / f"{take_name}.nmocap.h5")
            bj_p = export_blender_json(doc, proj.subdir("exports") / f"{take_name}_blender.json", use_corrected=True)
            self.project_manager.register_take(
                take_id=take_name,
                session_id="offline_recon",
                name=take_name,
                capture_mode=capture_mode,
                fps=fps,
                num_frames=min_frames,
                motion_h5_path=h5_p,
                blender_json_path=bj_p,
            )
            self.network_server.register_bridge_take(take_name, bj_p)
            self.project_changed.emit()

        self.set_active_motion(doc)
        if progress_cb:
            progress_cb(1.0, f"Completed {min_frames} frames ({profile_name})")
        return doc

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
        cals = sorted(proj.subdir("calibration").glob("*.json"))
        if cals:
            try:
                self.active_calibration = MultiCameraCalibration.load(cals[-1])
                self.calibration_changed.emit()
            except Exception:
                pass
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

    def publish_active_motion_to_blender(self) -> Path:
        doc = self.active_motion
        if doc is None:
            raise RuntimeError("No active motion take loaded.")
        if not self.server_running:
            self.start_server()
        out_dir = (
            self.project_manager.active_project.subdir("exports")
            if self.project_manager.active_project
            else Path(self.settings.storage.export_dir)
        )
        out_path = export_blender_json(doc, out_dir / f"{doc.take_id}_bridge.nmocap.json", use_corrected=True)
        self.network_server.register_bridge_take(doc.take_id, out_path)
        self.status_message.emit(f"Published '{doc.take_id}' to Blender Bridge on port {self.settings.network.server_port}.")
        return out_path

    def generate_synthetic_demo_take(self, take_name: str = "synthetic_demo_walk") -> MotionCaptureDocument:
        """Generate a clearly labeled SYNTHETIC demo capture take & 4-camera rig for offline UI inspection & Blender testing."""
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
        self.live_preview.stop()
        self.stop_server()
        self.job_queue.shutdown()
        self.project_manager.close_project()
        self.settings.save(self._settings_path)
