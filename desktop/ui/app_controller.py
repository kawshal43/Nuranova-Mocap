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

import cv2
import numpy as np
import uvicorn
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage, QPixmap
from scipy.spatial.transform import Rotation

from desktop.calibration.calibrator import (
    align_calibration_from_upright_board,
    calibrate_camera_intrinsics,
    initialize_extrinsics_from_board_frames,
    run_board_bundle_adjustment,
)
from desktop.calibration.charuco_board import (
    ChArUcoBoardConfig,
    ChArUcoDetection,
    detect_charuco_corners,
)
from desktop.camera_manager.manager import CameraManager
from desktop.camera_manager.usb_cameras import DiscoveredUSBCamera, USBCameraManager
from desktop.network.server import (
    StudioNetworkServer,
    check_windows_firewall,
    enumerate_lan_ipv4_addresses,
    request_elevated_firewall_fix,
)
from desktop.project_manager.manager import MocapProject, ProjectManager
from desktop.settings.app_settings import AppSettings
from desktop.synchronization.sync_engine import (
    estimate_audio_sync_offset_s,
    extract_audio_pcm_from_video,
    synchronize_multiview_video_observations,
)
from desktop.workers.job_queue import ProcessingQueueManager
from desktop.workers.live_preview import (
    LiveMocapPreviewWorker,
    SingleCamRootTracker,
    _mediapipe_world33_to_body26_zup,
)
from engine.ai_tracking.pose_tracker import AIPoseTracker2D, _mediapipe33_to_body26
from engine.ai_tracking.runtime import HardwareDiagnostics, detect_hardware
from engine.data_format.motion_io import (
    HandMotionTrack,
    MotionCaptureDocument,
    SkeletalMotionTrack,
    export_blender_json,
    load_motion_file,
    save_motion_file,
)
from engine.hands.hand_tracker import HandCaptureEngine, triangulate_hands_from_wholebody
from engine.marker_tracking.marker_kit import MarkerKitConfig, create_default_cinema_marker_kit
from engine.math_core.camera_model import (
    CalibratedCamera,
    CameraExtrinsics,
    CameraIntrinsics,
    MultiCameraCalibration,
)
from engine.math_core.transforms import quat_identity
from engine.motion_cleanup.cleanup import (
    MotionCleanupConfig,
    clean_skeletal_motion,
    despike_and_fill_3d_points,
)
from engine.optimization.hybrid_engine import PROCESSING_PROFILES, HybridCinemaReconstructor
from engine.skeleton.rig_definition import (
    BONE_INDEX,
    BONE_NAMES,
    PARENT_INDICES,
    ActorProfile,
    calibrate_actor_from_body26_frames,
    compute_forward_kinematics_sequence,
)
from engine.skeleton.solver import ArticulatedSkeletonSolver, SolverConfig
from model_registry.registry import ModelRegistry


class StudioController(QObject):
    """Single source of truth for Nuranova Mocap Studio state and signals."""

    project_changed = Signal()
    cameras_changed = Signal()
    telemetry_updated = Signal(
        str
    )  # device_id — high-frequency updates (never rebuild video tiles!)
    calibration_changed = Signal()
    motion_changed = Signal()
    jobs_changed = Signal()
    server_state_changed = Signal(bool, str)
    recording_state_changed = Signal(bool, str)
    take_upload_updated = Signal(str, int, int, str)  # take_id, received_n, expected_n, summary
    status_message = Signal(str)

    def __init__(
        self, settings_path: Optional[Path] = None, *,
        settings: Optional[AppSettings] = None,
        hardware: Optional[HardwareDiagnostics] = None,
    ) -> None:
        super().__init__()
        self._settings_path = settings_path
        self.settings = settings if settings is not None else AppSettings.load(settings_path)
        base_state_dir = (
            settings_path.parent if settings_path is not None else AppSettings.default_path().parent
        )

        self.project_manager = ProjectManager(
            recent_registry_path=(base_state_dir / "recent_projects.json")
            if settings_path is not None
            else None
        )
        self.camera_manager = CameraManager(persistence_path=base_state_dir / "paired_cameras.json")
        self.usb_cameras = USBCameraManager(self.camera_manager)
        self.job_queue = ProcessingQueueManager(
            max_workers=max(1, min(4, self.settings.processing.worker_count))
        )
        self.model_registry = ModelRegistry()
        self.hardware: HardwareDiagnostics = hardware if hardware is not None else detect_hardware(
            prefer_gpu=(self.settings.processing.compute_backend != "cpu")
        )

        self._base_state_dir = base_state_dir
        self._last_calib_path = base_state_dir / "last_calibration.json"
        self.active_calibration: Optional[MultiCameraCalibration] = None
        if self._last_calib_path.exists():
            try:
                self.active_calibration = MultiCameraCalibration.load(self._last_calib_path)
            except Exception:
                self.active_calibration = None
        self.active_actor: ActorProfile = ActorProfile(
            actor_id="default_actor", name="Default Adult Actor (1.72m)"
        )
        self.active_marker_kit: MarkerKitConfig = create_default_cinema_marker_kit()
        self.active_motion: Optional[MotionCaptureDocument] = None
        self._undo_stack: List[Optional[SkeletalMotionTrack]] = []
        self._redo_stack: List[Optional[SkeletalMotionTrack]] = []

        self.imported_videos: Dict[str, Path] = {}
        self.is_recording: bool = False
        self.offline_processing = False
        self.active_session_id: Optional[str] = None
        self.active_take_id: Optional[str] = None
        self._pending_take: Optional[Dict[str, Any]] = None
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

        self.camera_manager.add_listener(self._on_camera_manager_event)
        self.job_queue.add_listener(lambda job: self.jobs_changed.emit())
        self.network_server.add_transfer_listener(self._on_transfer_completed)

        self.live_overlay_jpegs: Dict[str, bytes] = {}
        self.live_preview = LiveMocapPreviewWorker(
            camera_manager=self.camera_manager,
            get_calibration=lambda: (
                None if self.settings.general.single_camera_mode else self.active_calibration
            ),
            get_actor=lambda: self.active_actor,
            get_single_cam_root_motion=lambda: self.settings.general.single_cam_root_motion,
            get_enable_hands=lambda: self.settings.processing.enable_hand_capture,
        )
        self.live_preview.overlay_updated.connect(self._on_live_overlay)
        self.live_preview.live_frame_updated.connect(self.network_server.update_live_pose)

    def set_hand_capture_enabled(self, enabled: bool) -> None:
        self.settings.processing.enable_hand_capture = bool(enabled)
        self.settings.save(self._settings_path)
        self.status_message.emit(
            "Finger capture (30 hand bones) enabled for live preview, Blender & offline takes"
            if enabled
            else "Finger capture paused (body-only mode)"
        )

    def set_single_camera_mode(self, enabled: bool) -> None:
        """Toggle 1-Camera AI Mode (skips multi-camera calibration) and persist setting."""
        self.settings.general.single_camera_mode = bool(enabled)
        self.settings.save(self._settings_path)
        self.live_preview.reset_root_origin()
        self.cameras_changed.emit()
        self.calibration_changed.emit()
        self.status_message.emit(
            "Single-Cam AI Mode ON (1 phone · no calibration needed!)"
            if enabled
            else "Multi-Cam Studio Mode ON (2–12 calibrated phones)"
        )

    def set_single_cam_root_motion(self, enabled: bool) -> None:
        """Toggle monocular 3D root walking & jump estimation for Single-Cam live preview & reconstruction."""
        self.settings.general.single_cam_root_motion = bool(enabled)
        self.settings.save(self._settings_path)
        self.live_preview.reset_root_origin()
        self.status_message.emit(
            "1-Cam 3D Root Walking enabled (left/right, forward/back & jumps)"
            if enabled
            else "1-Cam Root pinned in-place at origin"
        )

    def _on_camera_manager_event(self, event: str, device_id: str) -> None:
        if event == "config_changed":
            cam = self.camera_manager.get_camera(device_id)
            if cam is not None and cam.adapter_type == "usb_camera":
                self.usb_cameras.reconfigure_camera(
                    device_id, cam.config.width, cam.config.height, cam.config.target_fps
                )
        elif event == "removed":
            self.usb_cameras.disconnect_camera(device_id)

        if event in ("telemetry", "clock_synced"):
            self.telemetry_updated.emit(device_id)
        else:
            self.cameras_changed.emit()

    # ---------------- USB / Local Connected Cameras ----------------
    def scan_usb_cameras(self) -> List[DiscoveredUSBCamera]:
        return self.usb_cameras.scan_cameras()

    def connect_usb_camera(
        self,
        index: int,
        name: Optional[str] = None,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        supported_modes: Optional[List[Any]] = None,
    ):
        rec = self.usb_cameras.connect_camera(
            index=index,
            name=name,
            width=width,
            height=height,
            fps=fps,
            supported_modes=supported_modes,
        )
        self.status_message.emit(
            f"Connected USB camera '{rec.name}' as {rec.camera_uid} ({rec.config.width}x{rec.config.height} @ {rec.config.target_fps:.0f}fps)."
        )
        return rec

    def connect_all_usb_cameras(self) -> List[Any]:
        found = self.scan_usb_cameras()
        connected = []
        for cam in found:
            if cam.already_connected:
                continue
            try:
                rec = self.connect_usb_camera(
                    index=cam.index,
                    name=cam.name,
                    width=cam.width,
                    height=cam.height,
                    fps=cam.fps,
                    supported_modes=cam.supported_modes,
                )
                connected.append(rec)
            except Exception:
                continue
        return connected

    def disconnect_or_remove_camera(self, device_id: str) -> bool:
        self.usb_cameras.disconnect_camera(device_id)
        return self.camera_manager.remove_camera(device_id)

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
            self.network_server.start_mdns_advertisement(
                bind_ip=self.settings.network.preferred_lan_ip or None
            )
        if self.settings.network.enable_udp_beacon:
            self.network_server.start_udp_discovery(
                udp_port=self.settings.network.udp_discovery_port
            )

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

    # ---------------- Live Synchronized Recording (Phones + USB Cameras) ----------------
    def start_synchronized_recording(
        self,
        take_name: Optional[str] = None,
        allow_single_camera: bool = True,
        purpose: str = "mocap",  # "mocap" or "calibration"
        auto_process: bool = True,
        board_preset: str = "A3",
    ) -> Dict[str, Any]:
        ok, msg = (
            self.camera_manager.can_record_any()
            if allow_single_camera
            else self.camera_manager.can_record_multi_view()
        )
        if not ok:
            raise RuntimeError(msg)

        active_cams = self.camera_manager.get_active_recording_cameras()
        has_network_phones = any(c.adapter_type == "android_wifi" for c in active_cams)
        has_usb_cams = any(c.adapter_type == "usb_camera" for c in active_cams)

        if has_network_phones and not self.server_running:
            if not self.start_server():
                raise RuntimeError("Studio Server could not be started for paired Wi-Fi phones.")

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"sess_{stamp}"
        prefix = "calib" if purpose == "calibration" else "take"
        take_id = take_name or f"{prefix}_{stamp}"
        lead_s = 1.2 if has_network_phones else 0.1
        start_at_ns = time.monotonic_ns() + int(lead_s * 1e9)

        dispatched: List[str] = []
        errors: Dict[str, str] = {}
        res: Dict[str, Any] = {
            "session_id": session_id,
            "take_id": take_id,
            "purpose": purpose,
            "dispatched_devices": dispatched,
            "errors": errors,
        }

        if has_network_phones and self.server_running:
            net_res = self.network_server.schedule_record_from_thread(
                session_id=session_id,
                take_id=take_id,
                lead_time_s=lead_s,
                allow_single_camera=allow_single_camera,
            )
            dispatched.extend(net_res.get("dispatched_devices", []))
            errors.update(net_res.get("errors", {}))

        if has_usb_cams:
            rec_root = (
                self.project_manager.active_project.subdir("recordings")
                if self.project_manager.active_project is not None
                else (self.network_server.storage_root / "recordings")
            )
            usb_started = self.usb_cameras.start_recording_all(
                recordings_root=rec_root,
                take_id=take_id,
                start_at_monotonic_ns=start_at_ns,
            )
            dispatched.extend(usb_started)

        expected_uids = set()
        for did in dispatched:
            cam_rec = self.camera_manager.get_camera(did)
            if cam_rec is not None:
                expected_uids.add(cam_rec.camera_uid)

        self.is_recording = True
        self.active_session_id = session_id
        self.active_take_id = take_id
        self._pending_take = {
            "take_id": take_id,
            "purpose": purpose,
            "expected_uids": expected_uids,
            "completed_uids": set(),
            "auto_process": bool(auto_process),
            "board_preset": board_preset,
            "triggered": False,
        }
        n_disp = len(dispatched)
        kind_lbl = "Calibration Board Video" if purpose == "calibration" else "Mocap Video"
        info = f"Recording {kind_lbl} '{take_id}' on {n_disp} connected camera(s)."
        if errors:
            info += f" ({len(errors)} camera warnings)"
        self.recording_state_changed.emit(True, info)
        self.take_upload_updated.emit(take_id, 0, len(expected_uids), info)
        self.status_message.emit(info)
        return res

    def stop_synchronized_recording(self) -> List[str]:
        session_id = self.active_session_id or "sess_manual"
        take_id = self.active_take_id or "take_manual"
        stopped: List[str] = []
        if self.server_running:
            stopped.extend(
                self.network_server.stop_record_from_thread(session_id=session_id, take_id=take_id)
            )
        saved_usb = self.usb_cameras.stop_recording_all()
        stopped.extend(saved_usb.keys())
        self.is_recording = False

        if self._pending_take and self._pending_take["take_id"] == take_id:
            self._pending_take["completed_uids"].update(saved_usb.keys())
            done_n = len(self._pending_take["completed_uids"])
            exp_n = max(len(self._pending_take["expected_uids"]), done_n)
            if exp_n > 0 and done_n >= exp_n:
                msg = f"All {done_n}/{exp_n} video(s) saved for '{take_id}'!"
                self.take_upload_updated.emit(take_id, done_n, exp_n, msg)
                self.project_changed.emit()
                self._maybe_auto_process_pending_take()
            else:
                msg = f"Stopped '{take_id}': {done_n}/{exp_n} video(s) saved — receiving uploads from phones..."
                self.take_upload_updated.emit(take_id, done_n, exp_n, msg)
                if saved_usb:
                    self.project_changed.emit()
            info = msg
        elif saved_usb:
            self.project_changed.emit()
            info = f"Stopped '{take_id}': saved {len(saved_usb)} USB recording(s) (ready in Processing Queue)."
        else:
            info = f"Stopped '{take_id}' on {len(stopped)} camera(s); awaiting SHA-256 verified transfers."

        self.recording_state_changed.emit(False, info)
        self.status_message.emit(info)
        return stopped

    def _on_transfer_completed(self, slot) -> None:
        if not (slot.verified and slot.file_kind == "video"):
            return
        self.status_message.emit(
            f"Verified video upload: {slot.dest_path.name} (take '{slot.take_id}')"
        )
        self.project_changed.emit()

        pending = self._pending_take
        if pending is not None and slot.take_id == pending["take_id"]:
            cam_rec = self.camera_manager.get_camera(slot.device_id)
            if cam_rec is not None:
                pending["completed_uids"].add(cam_rec.camera_uid)
            else:
                pending["completed_uids"].add(slot.dest_path.stem)
            done_n = len(pending["completed_uids"])
            exp_n = max(len(pending["expected_uids"]), done_n)
            if done_n >= exp_n:
                msg = f"✓ All {done_n}/{exp_n} video(s) verified for '{slot.take_id}'!"
                self.take_upload_updated.emit(slot.take_id, done_n, exp_n, msg)
                self._maybe_auto_process_pending_take()
            else:
                msg = f"Uploaded {done_n}/{exp_n} video(s) for '{slot.take_id}' — waiting for remaining phone(s)..."
                self.take_upload_updated.emit(slot.take_id, done_n, exp_n, msg)

    def _maybe_auto_process_pending_take(self) -> None:
        pending = self._pending_take
        if not pending or pending.get("triggered") or not pending.get("auto_process"):
            return
        vids = self.get_videos_for_take(pending["take_id"])
        if not vids:
            return
        pending["triggered"] = True
        self.queue_recorded_take_processing(
            take_id=pending["take_id"],
            purpose=pending.get("purpose", "mocap"),
            board_preset=pending.get("board_preset", "A3"),
            profile_name="cinema",
        )

    def queue_recorded_take_processing(
        self,
        take_id: str,
        purpose: str = "mocap",
        board_preset: str = "A3",
        profile_name: str = "cinema",
    ) -> Optional[str]:
        """Submit a background Video Calibration or Cinema Offline Reconstruction job for a recorded take."""
        vids = self.get_videos_for_take(take_id)
        if not vids:
            return None

        if purpose == "calibration":
            cfg = (
                ChArUcoBoardConfig(
                    squares_x=7,
                    squares_y=5,
                    square_length_mm=35.0,
                    marker_length_mm=26.0,
                    dictionary_name="DICT_5X5_250",
                )
                if board_preset == "A4"
                else ChArUcoBoardConfig(
                    squares_x=9,
                    squares_y=6,
                    square_length_mm=40.0,
                    marker_length_mm=30.0,
                    dictionary_name="DICT_5X5_250",
                )
            )

            def cal_task(prog, cancel):
                cal = self.calibrate_from_video_files(
                    vids, board_cfg=cfg, progress_cb=prog, cancel_event=cancel
                )
                return f"{len(cal.cameras)} cams ({cal.global_reprojection_rmse_px:.2f}px)"

            job = self.job_queue.submit(
                f"Auto-Calibrate '{take_id}' ({len(vids)} cams)", "calibrate", cal_task
            )
            self.take_upload_updated.emit(
                take_id,
                len(vids),
                len(vids),
                f"✓ {len(vids)} video(s) saved — Auto-Calibrating multi-camera rig in background...",
            )
            return job.job_id

        def recon_task(prog, cancel):
            doc = self.reconstruct_from_video_files(
                video_paths_by_cam=vids,
                take_name=take_id,
                capture_mode="markerless_ai",
                profile_name=profile_name,
                auto_sync_videos=True,
                progress_cb=prog,
                cancel_event=cancel,
            )
            return doc.take_id

        job = self.job_queue.submit(
            f"Cinema Reconstruct '{take_id}' ({len(vids)} cams)", "reconstruct", recon_task
        )
        self.take_upload_updated.emit(
            take_id,
            len(vids),
            len(vids),
            f"✓ {len(vids)} video(s) saved — Cinema 3D + 30-Finger Reconstruction running!",
        )
        return job.job_id

    def get_videos_for_take(self, take_id: str) -> Dict[str, Path]:
        """Return all recorded MP4 files (merging Android uploads + PC USB recordings) for `take_id`."""
        for tid, vids in self.list_recorded_take_folders():
            if tid == take_id:
                return vids
        return {}

    def list_recorded_take_folders(self) -> List[Tuple[str, Dict[str, Path]]]:
        """Find multi-camera recorded take folders (merging project + incoming roots, newest first)."""
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
                    parts = mp4.stem.split("_")
                    cam_uid = (
                        f"{parts[0]}_{parts[1]}"
                        if len(parts) >= 2 and parts[0] == "cam"
                        else mp4.stem
                    )
                    vids[cam_uid] = mp4
                if vids:
                    found.setdefault(take_dir.name, {}).update(vids)
        return sorted(found.items(), key=lambda kv: kv[0], reverse=True)

    def set_active_calibration(self, calib: MultiCameraCalibration) -> None:
        """Set active multi-camera calibration and persist globally so it survives restarts even without a project."""
        self.active_calibration = calib
        try:
            self._last_calib_path.parent.mkdir(parents=True, exist_ok=True)
            calib.save(self._last_calib_path)
        except Exception:
            pass
        if self.project_manager.active_project is not None:
            try:
                out_p = (
                    self.project_manager.active_project.subdir("calibration")
                    / f"{calib.profile_name}.json"
                )
                calib.save(out_p)
            except Exception:
                pass
        self.calibration_changed.emit()

    @staticmethod
    def _probe_readable_video(vpath: Path) -> Tuple[bool, int, int, float, int]:
        """Verify an MP4 file is complete and readable via OpenCV before queuing reconstruction."""
        if not vpath.exists() or vpath.stat().st_size < 4096:
            return False, 0, 0, 0.0, 0
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            cap.release()
            return False, 0, 0, 0.0, 0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None or w <= 0 or h <= 0:
            return False, 0, 0, 0.0, 0
        if fc <= 0:
            fc = 1
        return True, w, h, (fps if 5.0 <= fps <= 240.0 else 30.0), fc

    def calibrate_from_video_files(
        self,
        video_paths_by_cam: Dict[str, Path],
        board_cfg: Optional[ChArUcoBoardConfig] = None,
        sample_count: int = 36,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> MultiCameraCalibration:
        """100% Video-Only Multi-Camera Calibration from 2+ recorded ChArUco board MP4 videos.

        Auto-aligns videos via audio clap (if present), samples synchronized board frames, solves
        per-camera intrinsics, multi-view extrinsics, bundle adjustment, and upright Z-Up world frame.
        """
        if len(video_paths_by_cam) < 2:
            raise ValueError(
                "Video-based multi-camera calibration requires at least 2 camera videos."
            )

        cfg = board_cfg or ChArUcoBoardConfig(
            squares_x=9,
            squares_y=6,
            square_length_mm=40.0,
            marker_length_mm=30.0,
            dictionary_name="DICT_5X5_250",
        )
        cam_uids = list(video_paths_by_cam.keys())
        meta: Dict[str, Tuple[Path, int, int, float, int]] = {}
        for uid, vp in video_paths_by_cam.items():
            ok, w, h, fps, fc = self._probe_readable_video(Path(vp))
            if not ok or fc < 5:
                raise RuntimeError(f"Calibration video unreadable or too short: {vp}")
            meta[uid] = (Path(vp), w, h, fps, fc)

        # Optional audio clap sync between calibration videos so moving board frames match in time!
        offsets_s: Dict[str, float] = {cam_uids[0]: 0.0}
        ref_audio = extract_audio_pcm_from_video(meta[cam_uids[0]][0])
        if ref_audio is not None:
            for uid in cam_uids[1:]:
                tgt_audio = extract_audio_pcm_from_video(meta[uid][0])
                if tgt_audio is not None:
                    a_off, a_snr = estimate_audio_sync_offset_s(
                        ref_audio, tgt_audio, sample_rate_hz=16000, max_lag_s=4.0
                    )
                    offsets_s[uid] = -a_off if (a_snr >= 6.0 and abs(a_off) <= 4.0) else 0.0
                else:
                    offsets_s[uid] = 0.0
        else:
            for uid in cam_uids[1:]:
                offsets_s[uid] = 0.0

        # Compute overlapping time window
        durations = {uid: meta[uid][4] / max(meta[uid][3], 1.0) for uid in cam_uids}
        t_start = max(offsets_s[uid] for uid in cam_uids)
        t_end = min(offsets_s[uid] + durations[uid] for uid in cam_uids)
        if t_end <= t_start + 0.5:
            offsets_s = {uid: 0.0 for uid in cam_uids}
            t_start = 0.0
            t_end = min(durations.values())

        sample_times = np.linspace(
            t_start + 0.2, max(t_start + 0.3, t_end - 0.2), max(12, int(sample_count))
        )
        detections_per_cam: Dict[str, List[ChArUcoDetection]] = {uid: [] for uid in cam_uids}
        synced_frames: List[Dict[str, ChArUcoDetection]] = [{} for _ in range(len(sample_times))]

        caps = {uid: cv2.VideoCapture(str(meta[uid][0])) for uid in cam_uids}
        try:
            for s_idx, t_glob in enumerate(sample_times):
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("Cancelled by user.")
                if progress_cb and s_idx % 3 == 0:
                    progress_cb(
                        0.05 + 0.55 * (s_idx / len(sample_times)),
                        f"Scanning ChArUco board (sample {s_idx + 1}/{len(sample_times)})...",
                    )
                for uid in cam_uids:
                    cap = caps[uid]
                    fps = meta[uid][3]
                    local_t = max(0.0, float(t_glob - offsets_s[uid]))
                    f_no = int(min(meta[uid][4] - 1, max(0, round(local_t * fps))))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f_no)
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        continue
                    det = detect_charuco_corners(frame, cfg)
                    if det.num_corners >= 6:
                        detections_per_cam[uid].append(det)
                        synced_frames[s_idx][uid] = det
        finally:
            for cap in caps.values():
                cap.release()

        # Filter to synchronized multi-view board frames seen by >= 2 cameras
        shared_frames = [f for f in synced_frames if len(f) >= 2]
        if len(shared_frames) < 4:
            counts = ", ".join(f"{u}: {len(detections_per_cam[u])} views" for u in cam_uids)
            raise RuntimeError(
                f"Only {len(shared_frames)} shared ChArUco frames detected across cameras ({counts}). "
                "Make sure the printed board matches your A3/A4 preset and is clearly visible to all cameras at the same time."
            )

        if progress_cb:
            progress_cb(
                0.68,
                f"Solving camera intrinsics & extrinsics from {len(shared_frames)} shared board views...",
            )

        intrinsics_map: Dict[str, CameraIntrinsics] = {}
        for uid in cam_uids:
            w, h = meta[uid][1], meta[uid][2]
            dets = detections_per_cam[uid]
            if len(dets) < 3:
                raise RuntimeError(
                    f"Camera {uid} only saw the board in {len(dets)} frames (need >= 3)."
                )
            intrinsics_map[uid] = calibrate_camera_intrinsics(uid, dets, cfg, (w, h))

        extr_init, board_poses = initialize_extrinsics_from_board_frames(
            intrinsics_map, shared_frames, cfg
        )
        if progress_cb:
            progress_cb(0.84, "Running multi-camera bundle adjustment & Z-Up world alignment...")
        extr_opt, opt_boards, ba_rep = run_board_bundle_adjustment(
            intrinsics_map, extr_init, shared_frames, board_poses, cfg
        )

        calib_ba = MultiCameraCalibration(
            profile_name=f"VideoCalib_{len(cam_uids)}cams",
            cameras={uid: CalibratedCamera(intrinsics_map[uid], extr_opt[uid]) for uid in cam_uids},
            global_reprojection_rmse_px=ba_rep.global_reprojection_rmse_px,
            notes=f"Solved from {len(shared_frames)} synchronized ChArUco video frames.",
        )

        # Anchor Blender Z-Up world frame using the clearest shared upright board pose
        upright_idx = next((i for i, bp in enumerate(opt_boards) if bp is not None), None)
        if upright_idx is not None and opt_boards[upright_idx] is not None:
            Rb, tb = opt_boards[upright_idx]
            calib_final = align_calibration_from_upright_board(
                calib_ba, R_wb=Rb, t_wb=tb, board_config=cfg, board_center_height_m=1.15
            )
        else:
            calib_final = calib_ba

        self.set_active_calibration(calib_final)
        if len(calib_final.cameras) >= 2 and self.settings.general.single_camera_mode:
            self.set_single_camera_mode(False)

        if progress_cb:
            progress_cb(
                1.0,
                f"Video calibration solved & saved ({len(cam_uids)} cams, {len(shared_frames)} shared views, RMS {calib_final.global_reprojection_rmse_px:.2f} px)!",
            )
        return calib_final

    def _reconstruct_single_view_ai_3d(
        self,
        cam_uid: str,
        vpath: Path,
        video_fps: float,
        total_frames: int,
        take_name: str,
        capture_mode: str,
        profile_name: str,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> MotionCaptureDocument:
        """High-accuracy offline Single-Video AI 3D reconstruction (Heavy pose + full-res wrist-cropped 30 fingers + foot lock)."""
        import mediapipe as mp

        prof = PROCESSING_PROFILES.get(profile_name, PROCESSING_PROFILES["cinema"])
        step = max(1, int(prof.frame_stride))
        complexity = {"lightweight": 0, "balanced": 1, "performance": 2}.get(prof.ai_model_mode, 2)
        pose_max_w = 1280 if prof.name == "cinema" else (960 if prof.name == "standard" else 640)

        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {vpath}")

        pts_list: List[np.ndarray] = []
        conf_list: List[np.ndarray] = []
        hand_results: List[Dict[str, Any]] = []
        capture_hands = bool(self.settings.processing.enable_hand_capture)
        hand_engine: Optional[HandCaptureEngine] = (
            HandCaptureEngine(max_num_hands=2, model_complexity=0 if complexity == 0 else 1)
            if capture_hands
            else None
        )
        root_tracker = SingleCamRootTracker(enabled=self.settings.general.single_cam_root_motion)
        pose_model = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        frame_idx = 0
        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("Cancelled by user.")
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    break
                if frame_idx % step == 0:
                    h, w = bgr.shape[:2]
                    if w > pose_max_w:
                        scale = float(pose_max_w) / float(w)
                        bgr_in = cv2.resize(
                            bgr,
                            (pose_max_w, max(1, int(round(h * scale)))),
                            interpolation=cv2.INTER_AREA,
                        )
                    else:
                        bgr_in = bgr
                    h_in, w_in = bgr_in.shape[:2]
                    rgb = cv2.cvtColor(bgr_in, cv2.COLOR_BGR2RGB)
                    res = pose_model.process(rgb)
                    b26_full: Optional[np.ndarray] = None
                    if res.pose_world_landmarks:
                        p3, c3 = _mediapipe_world33_to_body26_zup(res.pose_world_landmarks.landmark)
                        if res.pose_landmarks:
                            b26_in = _mediapipe33_to_body26(res.pose_landmarks.landmark, w_in, h_in)
                            b26_full = _mediapipe33_to_body26(res.pose_landmarks.landmark, w, h)
                            p3 = root_tracker.apply(p3, c3, b26_in, w_in, h_in)
                        pts_list.append(p3.astype(np.float32))
                        conf_list.append(c3.astype(np.float32))
                    else:
                        pts_list.append(np.full((26, 3), np.nan, dtype=np.float32))
                        conf_list.append(np.zeros((26,), dtype=np.float32))

                    # High-resolution wrist-cropped hand tracking directly on the native 1080p/4K frame!
                    if hand_engine is not None:
                        try:
                            if b26_full is not None:
                                hand_results.append(
                                    hand_engine.infer_hands_from_body26(bgr, b26_full)
                                )
                            else:
                                hand_results.append(
                                    hand_engine.infer_frame(bgr_in, assume_rear_camera=True)
                                )
                        except Exception:
                            hand_results.append({})

                    if progress_cb and (len(pts_list) % 15 == 0):
                        frac = min(0.75, 0.05 + 0.70 * (frame_idx / max(float(total_frames), 1.0)))
                        progress_cb(
                            frac,
                            f"Tracking Studio AI pose + wrist-cropped fingers on {cam_uid} (frame {frame_idx}/{total_frames})...",
                        )
                frame_idx += 1
        finally:
            pose_model.close()
            if hand_engine is not None:
                hand_engine.close()
            cap.release()

        if len(pts_list) < 2:
            raise RuntimeError("Video contained fewer than 2 readable frames.")

        pts_arr = np.stack(pts_list, axis=0)
        conf_arr = np.stack(conf_list, axis=0)
        valid_frames = int(np.sum(np.any(conf_arr > 0.25, axis=1)))
        if valid_frames == 0:
            raise RuntimeError(
                f"No person detected in {vpath.name}. Make sure the actor's body is visible in the recording."
            )

        pts_filled, interp_mask = despike_and_fill_3d_points(pts_arr, conf_arr)
        conf_filled = np.where(interp_mask, np.maximum(conf_arr, 0.35), conf_arr).astype(np.float32)
        eff_fps = max(5.0, float(video_fps) / float(step))
        ts_ns = np.arange(len(pts_list), dtype=np.int64) * int(round(1e9 / eff_fps))

        # Auto-calibrate actor bone lengths directly from the actor in this video!
        try:
            actor = calibrate_actor_from_body26_frames(
                actor_id=f"actor_{take_name}",
                name=f"Auto-Calibrated Actor ({take_name})",
                body26_3d_frames=pts_filled,
                body26_conf=conf_filled,
            )
            self.active_actor = actor
        except Exception:
            actor = self.active_actor

        if progress_cb:
            progress_cb(
                0.82,
                f"Solving 25-bone skeleton, zero-phase Butterworth & foot-locking ({len(pts_list)} frames)...",
            )

        solver = ArticulatedSkeletonSolver(
            actor=actor,
            config=SolverConfig(
                enforce_anatomical_limits=True,
                enable_nonlinear_refinement=bool(prof.enable_solver_nonlinear_refinement),
            ),
        )
        raw_track = solver.solve_sequence(
            pts_filled.astype(np.float32), conf_filled.astype(np.float32)
        )
        corrected_track = clean_skeletal_motion(
            raw_track,
            actor=actor,
            fps=eff_fps,
            config=MotionCleanupConfig(
                strength=0.78 if prof.name == "cinema" else 0.65,
                filter_type="butterworth",
                butterworth_cutoff_hz=7.0,
                enable_foot_locking=True,
                enable_ground_clamp=True,
            ),
        )

        hand_track: Optional[HandMotionTrack] = None
        if hand_results and any(bool(r) for r in hand_results):
            hc_builder = HandCaptureEngine()
            hand_track = hc_builder.build_track(hand_results, fps=eff_fps, smooth=True)

        proj_id = (
            self.project_manager.active_project.project_id
            if self.project_manager.active_project
            else "local_proj"
        )
        doc = MotionCaptureDocument(
            project_id=proj_id,
            session_id="offline_recon_single_ai",
            take_id=take_name,
            actor_id=actor.actor_id,
            fps=eff_fps,
            frame_timestamps_ns=ts_ns,
            bone_names=list(BONE_NAMES),
            parent_indices=list(PARENT_INDICES),
            rest_offsets=actor.rest_offsets_m.astype(np.float32).copy(),
            raw_motion=raw_track,
            corrected_motion=corrected_track,
            capture_mode=f"{capture_mode}_single_cam_3d",
            processing_profile=profile_name,
            cameras=[{"camera_uid": cam_uid, "source": vpath.name}],
            processing_config={
                "is_approximate": True,
                "fallback": "single_camera_mediapipe_heavy_3d",
                "hands_captured": hand_track is not None,
                "actor_height_m": round(float(actor.height_m), 3),
                "note": (
                    f"Reconstructed with Single-Video AI 3D (complexity={complexity}, wrist-cropped fingers, Butterworth + foot-lock) from {cam_uid}. "
                    "For true metric 3D depth across 360° turns, calibrate 2–4 camera videos!"
                ),
            },
            model_versions={"pose_3d": f"mediapipe_pose_world:complexity_{complexity}"},
            joints_3d_raw=pts_filled.astype(np.float32),
            joints_conf_raw=conf_filled.astype(np.float32),
            hands=hand_track,
        )
        return doc

    # ---------------- Real Multi-Camera (and Automatic Single-View Fallback) Offline Reconstruction ----------------
    def reconstruct_from_video_files(
        self,
        video_paths_by_cam: Dict[str, Path],
        take_name: str = "reconstructed_take",
        capture_mode: str = "markerless_ai",
        profile_name: str = "cinema",
        auto_sync_videos: bool = True,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> MotionCaptureDocument:
        """Run high-accuracy offline 3D reconstruction on MP4 files.

        - Calibrated 2+ Videos: RTMW-133 WholeBody -> Audio/Motion Auto-Sync -> Nonlinear Triangulation ->
          Triangulated + Wrist-Cropped 30 Fingers -> Auto-Calibrated Actor IK -> Zero-Phase Butterworth + Foot Lock!
        - Single Video / Uncalibrated: MediaPipe Heavy 3D + 3D Root Walk + Wrist-Cropped 30 Fingers + Foot Lock!
        """
        valid_vids: Dict[str, Tuple[Path, int, int, float, int]] = {}
        skipped: List[str] = []
        for uid, vp in video_paths_by_cam.items():
            p = Path(vp)
            ok, w, h, fps, fc = self._probe_readable_video(p)
            if ok and fc >= 2:
                valid_vids[uid] = (p, w, h, fps, fc)
            else:
                skipped.append(f"{uid} ({p.name})")

        if not valid_vids:
            raise RuntimeError(
                "No complete playable MP4 videos found in this take "
                + (f"(incomplete/unreadable: {', '.join(skipped)})." if skipped else ".")
            )

        prof = PROCESSING_PROFILES.get(profile_name, PROCESSING_PROFILES["cinema"])
        if capture_mode == "markerless_ai":
            # Use the same timing, camera estimation, and export path as the video workspace.
            # Import lazily because the workspace also imports StudioController.
            from multi_video_studio.package_ingest import MultiVideoPackage, probe_camera_clip
            from multi_video_studio.pipeline import CinemaPipelineConfig, MultiVideoCinemaPipeline

            clips = {uid: probe_camera_clip(info[0], uid) for uid, info in valid_vids.items()}
            package = MultiVideoPackage(
                package_name=take_name,
                source_path=next(iter(valid_vids.values()))[0].parent,
                motion_clips={uid: clip for uid, clip in clips.items() if clip is not None},
            )
            project = self.project_manager.active_project
            output_dir = project.subdir("exports") if project else self._base_state_dir / "exports"
            result = MultiVideoCinemaPipeline(
                CinemaPipelineConfig(
                    ai_backend=self.settings.processing.ai_backend,
                    ai_mode=prof.ai_model_mode,
                    frame_stride=prof.frame_stride,
                    prefer_gpu=self.settings.processing.compute_backend != "cpu",
                    enable_fingers=self.settings.processing.enable_hand_capture,
                    enable_auto_sync=auto_sync_videos,
                    actor_height_m=self.active_actor.height_m,
                )
            ).run_from_source(
                package,
                output_dir,
                existing_calibration=self.active_calibration,
                project_id=project.project_id if project else "offline_video",
                take_name=take_name,
                progress_cb=progress_cb,
                cancel_event=cancel_event,
            )
            doc = result.document
            if result.calibration is not None:
                self.set_active_calibration(result.calibration)
            if project:
                self.project_manager.register_take(
                    doc.take_id,
                    doc.session_id,
                    doc.take_id,
                    capture_mode=doc.capture_mode,
                    fps=doc.fps,
                    num_frames=doc.num_frames,
                    motion_h5_path=result.h5_path,
                    blender_json_path=result.blender_json_path,
                )
            if result.blender_json_path is not None:
                self.network_server.register_bridge_take(doc.take_id, result.blender_json_path)
            self.set_active_motion(doc)
            self.project_changed.emit()
            return doc
        use_multiview = (
            (not self.settings.general.single_camera_mode)
            and self.active_calibration is not None
            and len(valid_vids) >= 2
            and len(self.active_calibration.cameras) >= 2
        )
        doc: Optional[MotionCaptureDocument] = None
        mode_note = "Multi-View Calibrated 3D"

        if use_multiview and self.active_calibration is not None:
            calib_uids = list(self.active_calibration.cameras.keys())
            matched_vids: Dict[str, Tuple[Path, int, int, float, int]] = {}
            used_calib: set[str] = set()
            for uid, info in valid_vids.items():
                if uid in self.active_calibration.cameras:
                    matched_vids[uid] = info
                    used_calib.add(uid)
            rem_calib = [u for u in calib_uids if u not in used_calib]
            for uid, info in valid_vids.items():
                if uid in matched_vids:
                    continue
                if rem_calib:
                    target_uid = rem_calib.pop(0)
                    matched_vids[target_uid] = info
                    used_calib.add(target_uid)

            if len(matched_vids) >= 2:
                tracker = AIPoseTracker2D(
                    backend=self.settings.processing.ai_backend,
                    mode=prof.ai_model_mode,
                    prefer_gpu=(self.settings.processing.compute_backend != "cpu"),
                )
                ai_obs: Dict[str, np.ndarray] = {}
                lh_obs: Dict[str, np.ndarray] = {}
                rh_obs: Dict[str, np.ndarray] = {}
                pts_s_map: Dict[str, List[float]] = {}
                raw_body26_unscaled: Dict[str, np.ndarray] = {}
                n_cams = len(matched_vids)
                try:
                    for c_idx, (uid, (vpath, vw, vh, _vfps, _vfc)) in enumerate(
                        matched_vids.items()
                    ):
                        if cancel_event is not None and cancel_event.is_set():
                            raise RuntimeError("Cancelled by user.")
                        if progress_cb:
                            progress_cb(
                                0.05 + 0.55 * (c_idx / n_cams),
                                f"Running {tracker.backend} ({prof.ai_model_mode}) on {uid} ({vpath.name})...",
                            )
                        body_arr, lh_arr, rh_arr, pts_s, _ = tracker.process_video_wholebody(
                            vpath,
                            step=prof.frame_stride,
                            cancel_check=(lambda: cancel_event.is_set()) if cancel_event else None,
                        )
                        raw_body26_unscaled[uid] = body_arr.copy()
                        # Scale 2D keypoints cleanly if video resolution differs from calibrated intrinsics
                        ci = self.active_calibration.cameras[uid].intrinsics
                        sx = (
                            float(ci.image_width) / float(vw)
                            if (vw > 0 and vw != ci.image_width)
                            else 1.0
                        )
                        sy = (
                            float(ci.image_height) / float(vh)
                            if (vh > 0 and vh != ci.image_height)
                            else 1.0
                        )
                        if sx != 1.0 or sy != 1.0:
                            body_arr = body_arr.copy()
                            body_arr[:, :, 0] *= sx
                            body_arr[:, :, 1] *= sy
                            if lh_arr is not None:
                                lh_arr = lh_arr.copy()
                                lh_arr[:, :, 0] *= sx
                                lh_arr[:, :, 1] *= sy
                            if rh_arr is not None:
                                rh_arr = rh_arr.copy()
                                rh_arr[:, :, 0] *= sx
                                rh_arr[:, :, 1] *= sy
                        ai_obs[uid] = body_arr
                        if lh_arr is not None:
                            lh_obs[uid] = lh_arr
                        if rh_arr is not None:
                            rh_obs[uid] = rh_arr
                        pts_s_map[uid] = pts_s
                finally:
                    tracker.close()

                min_frames = min(arr.shape[0] for arr in ai_obs.values())
                if min_frames >= 2:
                    base_fps = float(np.median([info[3] for info in matched_vids.values()]))
                    fps = max(5.0, base_fps / max(prof.frame_stride, 1))

                    if progress_cb:
                        progress_cb(
                            0.64,
                            "Auto-synchronizing multi-camera timelines (Audio Clap / Vertical Motion)...",
                        )

                    extra_tracks: Dict[str, Dict[str, np.ndarray]] = {}
                    if len(lh_obs) >= 2:
                        extra_tracks["left_hand21"] = lh_obs
                    if len(rh_obs) >= 2:
                        extra_tracks["right_hand21"] = rh_obs

                    vpaths_only = {uid: info[0] for uid, info in matched_vids.items()}
                    ts_ns, aligned_body26, aligned_extra, sync_report = (
                        synchronize_multiview_video_observations(
                            video_paths_by_cam=vpaths_only,
                            pts_s_by_cam=pts_s_map,
                            body26_by_cam=ai_obs,
                            target_fps=fps,
                            extra_obs_by_cam=extra_tracks,
                            enable_auto_sync=auto_sync_videos,
                        )
                    )
                    n_synced = len(ts_ns)

                    if progress_cb:
                        progress_cb(
                            0.72,
                            f"Solving 3D fingers & triangulating calibrated skeleton ({n_synced} frames)...",
                        )

                    mv_hand_track: Optional[HandMotionTrack] = None
                    if self.settings.processing.enable_hand_capture:
                        try:
                            # Fallback wrist-cropped high-res MediaPipe hands on primary video (full native resolution!)
                            first_uid = next(iter(matched_vids.keys()))
                            first_vpath = matched_vids[first_uid][0]
                            first_b26 = raw_body26_unscaled.get(first_uid)
                            he = HandCaptureEngine(max_num_hands=2, model_complexity=1)
                            cap_h = cv2.VideoCapture(str(first_vpath))
                            h_res_list: List[Dict[str, Any]] = []
                            f_i = 0
                            try:
                                while len(h_res_list) < n_synced:
                                    ok_h, bgr_h = cap_h.read()
                                    if not ok_h or bgr_h is None:
                                        break
                                    if f_i % max(prof.frame_stride, 1) == 0:
                                        idx_k = len(h_res_list)
                                        if first_b26 is not None and idx_k < first_b26.shape[0]:
                                            h_res_list.append(
                                                he.infer_hands_from_body26(bgr_h, first_b26[idx_k])
                                            )
                                        else:
                                            h_res_list.append(
                                                he.infer_frame(bgr_h, assume_rear_camera=True)
                                            )
                                    f_i += 1
                            finally:
                                cap_h.release()
                                he.close()

                            # Fuse true multi-view 3D triangulated RTMW-133 hands with wrist-cropped fallback!
                            mv_hand_track = triangulate_hands_from_wholebody(
                                cameras=self.active_calibration.cameras,
                                left_hand21_per_cam=aligned_extra.get("left_hand21"),
                                right_hand21_per_cam=aligned_extra.get("right_hand21"),
                                fps=fps,
                                fallback_obs_per_frame=h_res_list,
                            )
                        except Exception:
                            mv_hand_track = None

                    # Pass actor=None so HybridCinemaReconstructor auto-calibrates the actor's real limb lengths!
                    recon = HybridCinemaReconstructor(
                        calibration=self.active_calibration,
                        actor=None,
                        marker_kit=self.active_marker_kit,
                    )
                    proj_id = (
                        self.project_manager.active_project.project_id
                        if self.project_manager.active_project
                        else "local_proj"
                    )
                    doc, summary = recon.reconstruct_from_2d_observations(
                        project_id=proj_id,
                        session_id="offline_recon",
                        take_id=take_name,
                        fps=fps,
                        frame_timestamps_ns=ts_ns,
                        ai_obs_2d_per_cam=aligned_body26,
                        capture_mode=capture_mode,
                        profile_name=profile_name,
                        hand_track=mv_hand_track,
                        model_versions={"pose_2d": tracker.model_version_tag},
                    )
                    doc.processing_config["sync_report"] = sync_report
                    doc.processing_config["mean_reprojection_error_px"] = round(
                        float(summary.mean_reprojection_error_px), 2
                    )
                    mode_note = (
                        f"Multi-View Calibrated 3D ({len(matched_vids)} cams, "
                        f"reproj {summary.mean_reprojection_error_px:.1f}px)"
                    )

        # Fallback (or explicit Single-Cam Mode): Single-Camera AI 3D World Pose + 3D Root Walk!
        if doc is None:
            best_uid, (best_path, _bw, _bh, best_fps, best_fc) = max(
                valid_vids.items(), key=lambda kv: kv[1][4]
            )
            if self.settings.general.single_camera_mode:
                reason = "Single-Cam AI Mode"
            elif self.active_calibration is None:
                reason = "no calibration loaded"
            else:
                reason = f"only 1 complete video ({best_uid})"
            if progress_cb:
                progress_cb(0.04, f"Using Single-Cam 3D AI + Root Walk ({reason}) on {best_uid}...")
            mode_note = f"Single-Cam AI 3D + Root Walk ({reason})"
            doc = self._reconstruct_single_view_ai_3d(
                cam_uid=best_uid,
                vpath=best_path,
                video_fps=best_fps,
                total_frames=best_fc,
                take_name=take_name,
                capture_mode=capture_mode,
                profile_name=profile_name,
                progress_cb=progress_cb,
                cancel_event=cancel_event,
            )

        # Save outputs (inside active project if open, else inside ~/.nuranova_mocap/exports)
        if self.project_manager.active_project is not None:
            proj = self.project_manager.active_project
            h5_p = save_motion_file(doc, proj.subdir("corrected_motion") / f"{take_name}.nmocap.h5")
            bj_p = export_blender_json(
                doc, proj.subdir("exports") / f"{take_name}_blender.json", use_corrected=True
            )
            self.project_manager.register_take(
                take_id=take_name,
                session_id=doc.session_id,
                name=take_name,
                capture_mode=doc.capture_mode,
                fps=doc.fps,
                num_frames=doc.num_frames,
                motion_h5_path=h5_p,
                blender_json_path=bj_p,
            )
            self.network_server.register_bridge_take(take_name, bj_p)
            self.project_changed.emit()
        else:
            exp_dir = self._base_state_dir / "exports"
            exp_dir.mkdir(parents=True, exist_ok=True)
            save_motion_file(doc, exp_dir / f"{take_name}.nmocap.h5")
            bj_p = export_blender_json(
                doc, exp_dir / f"{take_name}_blender.json", use_corrected=True
            )
            self.network_server.register_bridge_take(take_name, bj_p)

        self.set_active_motion(doc)
        skip_note = f" · skipped incomplete: {', '.join(skipped)}" if skipped else ""
        if progress_cb:
            progress_cb(1.0, f"Completed {doc.num_frames} frames · {mode_note}{skip_note}")
        return doc

    # ---------------- Project Helpers ----------------
    def create_project(self, parent_dir: str | Path, name: str) -> MocapProject:
        self._require_idle_project()
        proj = self.project_manager.create_project(parent_dir, name)
        self._reset_project_state()
        self._bind_project_paths(proj)
        self.project_changed.emit()
        self.status_message.emit(f"Created project '{proj.name}' at {proj.root_dir}")
        return proj

    def open_project(self, path: str | Path) -> MocapProject:
        self._require_idle_project()
        proj = self.project_manager.open_project(path)
        self._reset_project_state()
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

    def _require_idle_project(self) -> None:
        if self.offline_processing:
            raise RuntimeError("Finish or cancel the video solve before switching projects.")
        if self.is_recording:
            raise RuntimeError("Stop recording before switching projects.")
        if any(j.status.value in ("queued", "running") for j in self.job_queue.list_jobs()):
            raise RuntimeError("Finish or cancel processing before switching projects.")

    def _reset_project_state(self) -> None:
        self.live_preview.stop()
        self.active_motion = None
        self.active_calibration = None
        self.active_actor = ActorProfile(actor_id="default_actor", name="Default Actor (1.72m)")
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.imported_videos.clear()
        self.active_session_id = None
        self.active_take_id = None
        self._pending_take = None
        self.network_server.bridge_takes.clear()
        self.network_server.clear_live_pose()
        self.motion_changed.emit()
        self.calibration_changed.emit()

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
        self.active_actor = ActorProfile(
            actor_id=doc.actor_id,
            name=f"Actor for {doc.take_id}",
            rest_offsets_m=np.asarray(doc.rest_offsets, dtype=np.float64).copy(),
        )
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.motion_changed.emit()

    def save_project(self) -> None:
        project = self.project_manager.active_project
        if project is None:
            self.status_message.emit("Create or open a project to save the motion.")
            return
        doc = self.active_motion
        if doc is not None:
            doc.project_id = project.project_id
            h5_path = save_motion_file(
                doc, project.subdir("corrected_motion") / f"{doc.take_id}.nmocap.h5"
            )
            blender_path = export_blender_json(
                doc, project.subdir("exports") / f"{doc.take_id}_blender.json"
            )
            self.project_manager.register_take(
                doc.take_id,
                doc.session_id,
                doc.take_id,
                capture_mode=doc.capture_mode,
                fps=doc.fps,
                num_frames=doc.num_frames,
                motion_h5_path=h5_path,
                blender_json_path=blender_path,
            )
            self.network_server.register_bridge_take(doc.take_id, blender_path)
        self.project_manager.save_project()
        self.status_message.emit("Project and motion edits saved.")

    def apply_motion_cleanup(
        self, cfg: MotionCleanupConfig, frame_range: Optional[tuple[int, int]] = None
    ) -> None:
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
        self.status_message.emit(
            f"Applied non-destructive cleanup ({cfg.filter_type}, strength={cfg.strength:.2f})."
        )

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
        out_path = export_blender_json(
            doc, out_dir / f"{doc.take_id}_bridge.nmocap.json", use_corrected=True
        )
        self.network_server.register_bridge_take(doc.take_id, out_path)
        self.status_message.emit(
            f"Published '{doc.take_id}' to Blender Bridge on port {self.settings.network.server_port}."
        )
        return out_path

    def generate_synthetic_demo_take(
        self, take_name: str = "synthetic_demo_walk"
    ) -> MotionCaptureDocument:
        """Generate a clearly labeled SYNTHETIC demo capture take & 4-camera rig for offline UI inspection & Blender testing."""
        target = np.array([0.0, 0.0, 1.0])
        up = np.array([0.0, 0.0, 1.0])
        K = np.array([[1400.0, 0.0, 960.0], [0.0, 1400.0, 540.0], [0.0, 0.0, 1.0]])
        dist = np.array([-0.1, 0.03, 0.0, 0.0, 0.0])
        cams = {}
        for i in range(4):
            ang = 2.0 * np.pi * i / 4.0 + 0.3
            pos = np.array([3.0 * np.cos(ang), 3.0 * np.sin(ang), 1.85])
            zc = target - pos
            zc /= np.linalg.norm(zc)
            xc = np.cross(zc, up)
            xc /= np.linalg.norm(xc)
            yc = np.cross(zc, xc)
            R_cw = np.vstack([xc, yc, zc])
            t_cw = -R_cw @ pos
            uid = f"cam_{i+1:02d}"
            cams[uid] = CalibratedCamera(
                CameraIntrinsics(
                    uid, 1920, 1080, K.copy(), dist.copy(), reprojection_error_px=0.28
                ),
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
            local_q[t, BONE_INDEX["chest"]] = Rotation.from_euler(
                "z", 10.0 * np.sin(ph), degrees=True
            ).as_quat()
            local_q[t, BONE_INDEX["left_upper_arm"]] = Rotation.from_euler(
                "yz", [58.0, 22.0 * np.sin(ph)], degrees=True
            ).as_quat()
            local_q[t, BONE_INDEX["left_forearm"]] = Rotation.from_euler(
                "z", -38.0 - 14.0 * np.sin(ph), degrees=True
            ).as_quat()
            local_q[t, BONE_INDEX["right_upper_arm"]] = Rotation.from_euler(
                "yz", [-58.0, -22.0 * np.sin(ph)], degrees=True
            ).as_quat()
            local_q[t, BONE_INDEX["right_forearm"]] = Rotation.from_euler(
                "z", 38.0 + 14.0 * np.sin(ph), degrees=True
            ).as_quat()
            local_q[t, BONE_INDEX["left_upper_leg"]] = Rotation.from_euler(
                "x", 20.0 * np.sin(ph), degrees=True
            ).as_quat()
            local_q[t, BONE_INDEX["left_lower_leg"]] = Rotation.from_euler(
                "x", -max(5.0, 42.0 * np.sin(ph)), degrees=True
            ).as_quat()
            local_q[t, BONE_INDEX["right_upper_leg"]] = Rotation.from_euler(
                "x", -20.0 * np.sin(ph), degrees=True
            ).as_quat()
            local_q[t, BONE_INDEX["right_lower_leg"]] = Rotation.from_euler(
                "x", -max(5.0, -42.0 * np.sin(ph)), degrees=True
            ).as_quat()

        fk_pos, _ = compute_forward_kinematics_sequence(
            root_t, local_q, self.active_actor.rest_offsets_m
        )
        body26 = np.zeros((T, 26, 3))
        m = {
            19: "hips",
            18: "neck",
            17: "head_end",
            5: "left_upper_arm",
            7: "left_forearm",
            9: "left_hand",
            6: "right_upper_arm",
            8: "right_forearm",
            10: "right_hand",
            11: "left_upper_leg",
            13: "left_lower_leg",
            15: "left_foot",
            20: "left_toe",
            22: "left_toe",
            24: "left_foot",
            12: "right_upper_leg",
            14: "right_lower_leg",
            16: "right_foot",
            21: "right_toe",
            23: "right_toe",
            25: "right_foot",
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
                arr[t, :, :2] = cam.project_world_points(body26[t], True) + rng.normal(
                    0.0, 1.2, size=(26, 2)
                )
                arr[t, :, 2] = 0.92
            ai_obs[uid] = arr

        recon = HybridCinemaReconstructor(
            calibration=calib, actor=self.active_actor, marker_kit=self.active_marker_kit
        )
        proj_id = (
            self.project_manager.active_project.project_id
            if self.project_manager.active_project
            else "demo_proj"
        )
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
            bj_p = export_blender_json(
                doc, proj.subdir("exports") / f"{take_name}_blender.json", use_corrected=True
            )
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
        self.status_message.emit(
            f"Generated SYNTHETIC demo take '{take_name}' ({T} frames @ 60 FPS)."
        )
        return doc

    def shutdown(self) -> None:
        self.live_preview.stop()
        self.network_server.clear_live_pose()
        self.usb_cameras.shutdown()
        self.stop_server()
        self.job_queue.shutdown()
        self.project_manager.close_project()
        self.settings.save(self._settings_path)
