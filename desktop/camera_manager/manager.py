"""Dynamic 2-to-12 Camera Manager, Pairing Auth, Token Persistence, Capability Validation, and Adapters."""

from __future__ import annotations

import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Tuple

from desktop.synchronization.sync_engine import ClockModel, ClockSyncSample, fit_clock_sync_model
from shared_protocol.protocol import (
    MAX_BODY_CAMERAS,
    MIN_BODY_CAMERAS,
    CameraCapabilities,
    CameraConnectionState,
    CameraRole,
    CameraSummary,
    DeviceTelemetry,
    PairRequest,
    PairResponse,
    RecordingConfig,
    SupportedMode,
)


class CameraAdapter(Protocol):
    """Replaceable camera adapter interface enabling Android Wi-Fi, USB, GigE, IR, and Simulated cameras."""

    adapter_type: str

    def get_capabilities(self) -> CameraCapabilities: ...
    def is_connected(self) -> bool: ...


@dataclass
class ManagedCameraRecord:
    camera_uid: str
    device_id: str
    name: str
    auth_token: str
    group: str = "Main Volume"
    role: CameraRole = CameraRole.BODY
    enabled: bool = True
    state: CameraConnectionState = CameraConnectionState.CONNECTED
    ip_address: str = ""
    adapter_type: str = "android_wifi"
    capabilities: CameraCapabilities = field(default_factory=CameraCapabilities)
    config: RecordingConfig = field(default_factory=RecordingConfig)
    telemetry: Optional[DeviceTelemetry] = None
    calibrated: bool = False
    calibration_valid: bool = False
    calibration_invalidation_reason: Optional[str] = None
    sync_samples: List[ClockSyncSample] = field(default_factory=list)
    clock_model: Optional[ClockModel] = None
    last_heartbeat_s: float = field(default_factory=time.monotonic)
    latest_preview_jpeg: Optional[bytes] = None

    def to_summary(self) -> CameraSummary:
        return CameraSummary(
            camera_uid=self.camera_uid,
            device_id=self.device_id,
            name=self.name,
            group=self.group,
            role=self.role,
            enabled=self.enabled,
            state=self.state,
            ip_address=self.ip_address,
            calibrated=self.calibrated,
            calibration_valid=self.calibration_valid,
            calibration_invalidation_reason=self.calibration_invalidation_reason,
            sync_offset_ns=self.clock_model.intercept_ns if self.clock_model else None,
            sync_drift_ppm=self.clock_model.drift_ppm if self.clock_model else None,
            sync_residual_ms=self.clock_model.residual_rmse_ms if self.clock_model else None,
            rtt_ms=self.clock_model.median_rtt_ms if self.clock_model else None,
            capabilities=self.capabilities,
            config=self.config,
            telemetry=self.telemetry,
        )

    def to_persisted_dict(self) -> Dict:
        return {
            "camera_uid": self.camera_uid,
            "device_id": self.device_id,
            "name": self.name,
            "auth_token": self.auth_token,
            "group": self.group,
            "role": self.role.value,
            "enabled": self.enabled,
            "ip_address": self.ip_address,
            "adapter_type": self.adapter_type,
            "capabilities": self.capabilities.model_dump(),
            "config": self.config.model_dump(),
        }

    @classmethod
    def from_persisted_dict(cls, data: Dict) -> "ManagedCameraRecord":
        role_raw = str(data.get("role", CameraRole.BODY.value))
        try:
            role = CameraRole(role_raw)
        except ValueError:
            role = CameraRole.BODY
        return cls(
            camera_uid=str(data["camera_uid"]),
            device_id=str(data["device_id"]),
            name=str(data.get("name", data["camera_uid"])),
            auth_token=str(data["auth_token"]),
            group=str(data.get("group", "Main Volume")),
            role=role,
            enabled=bool(data.get("enabled", True)),
            state=CameraConnectionState.DISCONNECTED,
            ip_address=str(data.get("ip_address", "")),
            adapter_type=str(data.get("adapter_type", "android_wifi")),
            capabilities=CameraCapabilities.model_validate(data.get("capabilities", {})),
            config=RecordingConfig.model_validate(data.get("config", {})),
        )


def validate_recording_config_against_capabilities(
    config: RecordingConfig,
    caps: CameraCapabilities,
) -> Tuple[bool, List[str], Optional[RecordingConfig]]:
    """Validate requested recording configuration against detected hardware capabilities.

    Never performs silent changes; returns clear diagnostic errors and an explicit alternative config.
    """
    issues: List[str] = []
    if not caps.supported_modes:
        return True, ["Camera reported no explicit mode list; hardware will verify at session start."], None

    exact = [
        m
        for m in caps.supported_modes
        if m.width == config.width and m.height == config.height and m.max_fps >= config.target_fps - 0.5
    ]
    if exact:
        if config.manual_exposure and not caps.supports_manual_exposure:
            issues.append("Manual exposure not supported by this camera's Camera2 hardware level; AE lock will be requested.")
        if config.lock_focus and not caps.supports_manual_focus:
            issues.append("Manual focus distance not exposed; AF lock fallback required.")
        return True, issues, None

    issues.append(
        f"Requested {config.width}x{config.height} @ {config.target_fps:.0f} FPS is NOT exposed by {caps.manufacturer} {caps.model} Camera2 API."
    )
    same_res = [m for m in caps.supported_modes if m.width == config.width and m.height == config.height]
    candidates = same_res if same_res else caps.supported_modes
    best_mode: SupportedMode = min(
        candidates,
        key=lambda m: (abs(m.width * m.height - config.width * config.height) / 1e6, abs(m.max_fps - config.target_fps)),
    )
    alt = config.model_copy(
        update={
            "width": best_mode.width,
            "height": best_mode.height,
            "target_fps": min(config.target_fps, best_mode.max_fps),
        }
    )
    return False, issues, alt


class CameraManager:
    """Thread-safe manager for 2 to 12 body cameras + dedicated face/hand cameras with optional token persistence."""

    def __init__(
        self,
        max_body_cameras: int = MAX_BODY_CAMERAS,
        persistence_path: Optional[Path] = None,
    ) -> None:
        self.max_body_cameras = max_body_cameras
        self.persistence_path = Path(persistence_path) if persistence_path is not None else None
        self._lock = threading.RLock()
        self._cameras: Dict[str, ManagedCameraRecord] = {}  # keyed by device_id
        self._tokens: Dict[str, str] = {}  # token -> device_id
        self._active_pin: Optional[str] = None
        self._pin_expires_at: float = 0.0
        self._camera_counter = 0
        self._listeners: List[Callable[[str, str], None]] = []
        self._load_persisted_devices()

    def _load_persisted_devices(self) -> None:
        if self.persistence_path is None or not self.persistence_path.exists():
            return
        try:
            raw = json.loads(self.persistence_path.read_text(encoding="utf-8"))
            items = raw.get("cameras", []) if isinstance(raw, dict) else []
            with self._lock:
                for item in items:
                    try:
                        rec = ManagedCameraRecord.from_persisted_dict(item)
                    except Exception:
                        continue
                    self._cameras[rec.device_id] = rec
                    self._tokens[rec.auth_token] = rec.device_id
                    if rec.camera_uid.startswith("cam_"):
                        try:
                            num = int(rec.camera_uid.split("_", 1)[1])
                            self._camera_counter = max(self._camera_counter, num)
                        except ValueError:
                            pass
        except Exception:
            pass

    def _save_persisted_devices(self) -> None:
        if self.persistence_path is None:
            return
        try:
            with self._lock:
                persisted = [
                    rec.to_persisted_dict()
                    for rec in self._cameras.values()
                    if rec.adapter_type == "android_wifi"
                ]
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            self.persistence_path.write_text(
                json.dumps({"schema": "nuranova-paired-cameras-v1", "cameras": persisted}, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def add_listener(self, cb: Callable[[str, str], None]) -> None:
        with self._lock:
            self._listeners.append(cb)

    def _notify(self, event: str, device_id: str) -> None:
        with self._lock:
            callbacks = list(self._listeners)
        for cb in callbacks:
            try:
                cb(event, device_id)
            except Exception:
                pass

    def generate_pairing_pin(self, ttl_seconds: int = 600) -> str:
        with self._lock:
            pin = f"{secrets.randbelow(900000) + 100000}"
            self._active_pin = pin
            self._pin_expires_at = time.monotonic() + ttl_seconds
            return pin

    def get_active_pin(self) -> str:
        with self._lock:
            if self._active_pin is None or time.monotonic() > self._pin_expires_at:
                return self.generate_pairing_pin()
            return self._active_pin

    def get_pin_remaining_seconds(self) -> int:
        with self._lock:
            if self._active_pin is None:
                return 0
            return max(0, int(round(self._pin_expires_at - time.monotonic())))

    def pair_device(self, req: PairRequest, ip_address: str = "") -> PairResponse:
        with self._lock:
            if self._active_pin is None or time.monotonic() > self._pin_expires_at:
                raise PermissionError("No active pairing PIN. Generate a new pairing PIN in Mocap Studio.")
            if not secrets.compare_digest(req.pin.strip(), self._active_pin):
                raise PermissionError("Invalid pairing PIN. Check the 6-digit PIN shown in Mocap Studio.")

            if req.device_id not in self._cameras:
                body_count = sum(1 for c in self._cameras.values() if c.role == CameraRole.BODY)
                if req.role == CameraRole.BODY and body_count >= self.max_body_cameras:
                    raise ValueError(f"Maximum body camera limit ({self.max_body_cameras}) reached.")
                self._camera_counter += 1
                cam_uid = f"cam_{self._camera_counter:02d}"
                token = secrets.token_urlsafe(32)
                # Negotiate initial safe config based on capabilities if 1080p@60 isn't exposed
                default_cfg = RecordingConfig()
                ok_cfg, _, alt_cfg = validate_recording_config_against_capabilities(default_cfg, req.capabilities)
                initial_cfg = default_cfg if ok_cfg else (alt_cfg or default_cfg)
                rec = ManagedCameraRecord(
                    camera_uid=cam_uid,
                    device_id=req.device_id,
                    name=req.device_name,
                    auth_token=token,
                    role=req.role,
                    ip_address=ip_address,
                    capabilities=req.capabilities,
                    config=initial_cfg,
                    state=CameraConnectionState.CONNECTED,
                )
                self._cameras[req.device_id] = rec
                self._tokens[token] = req.device_id
            else:
                rec = self._cameras[req.device_id]
                rec.name = req.device_name or rec.name
                rec.role = req.role
                rec.capabilities = req.capabilities
                rec.ip_address = ip_address or rec.ip_address
                rec.state = CameraConnectionState.CONNECTED
                rec.last_heartbeat_s = time.monotonic()

            resp = PairResponse(
                paired=True,
                device_id=req.device_id,
                assigned_camera_uid=rec.camera_uid,
                auth_token=rec.auth_token,
                studio_time_ns=time.monotonic_ns(),
            )
        self._save_persisted_devices()
        self._notify("paired", req.device_id)
        return resp

    def authenticate_token(self, token: str, device_id: Optional[str] = None) -> Optional[ManagedCameraRecord]:
        if not token:
            return None
        with self._lock:
            matched_did = self._tokens.get(token)
            if matched_did is None:
                return None
            if device_id is not None and matched_did != device_id:
                return None
            return self._cameras.get(matched_did)

    def mark_connected(self, device_id: str, ip_address: str = "") -> None:
        with self._lock:
            rec = self._cameras.get(device_id)
            if rec is None:
                return
            rec.state = CameraConnectionState.CONNECTED
            if ip_address:
                rec.ip_address = ip_address
            rec.last_heartbeat_s = time.monotonic()
        self._save_persisted_devices()
        self._notify("connected", device_id)

    def register_imported_or_local_camera(
        self,
        device_id: str,
        name: str,
        width: int = 1920,
        height: int = 1080,
        fps: float = 60.0,
        role: CameraRole = CameraRole.BODY,
        adapter_type: str = "imported_video",
    ) -> ManagedCameraRecord:
        with self._lock:
            if device_id in self._cameras:
                return self._cameras[device_id]
            self._camera_counter += 1
            uid = f"cam_{self._camera_counter:02d}"
            token = secrets.token_urlsafe(32)
            caps = CameraCapabilities(
                manufacturer="Offline/Imported" if adapter_type == "imported_video" else "Simulated",
                model=name,
                supported_modes=[SupportedMode(width=width, height=height, max_fps=fps, fps_ranges=[[int(fps), int(fps)]])],
            )
            rec = ManagedCameraRecord(
                camera_uid=uid,
                device_id=device_id,
                name=name,
                auth_token=token,
                role=role,
                adapter_type=adapter_type,
                capabilities=caps,
                config=RecordingConfig(width=width, height=height, target_fps=fps),
                state=CameraConnectionState.CONNECTED,
            )
            self._cameras[device_id] = rec
            self._tokens[token] = device_id
        self._notify("added", device_id)
        return rec

    def remove_camera(self, device_id: str) -> bool:
        with self._lock:
            rec = self._cameras.pop(device_id, None)
            if rec is None:
                return False
            self._tokens.pop(rec.auth_token, None)
        self._save_persisted_devices()
        self._notify("removed", device_id)
        return True

    def set_camera_enabled(self, device_id: str, enabled: bool) -> None:
        with self._lock:
            if device_id not in self._cameras:
                return
            self._cameras[device_id].enabled = enabled
        self._save_persisted_devices()
        self._notify("updated", device_id)

    def rename_camera(self, device_id: str, name: str, group: Optional[str] = None) -> None:
        with self._lock:
            if device_id not in self._cameras:
                return
            self._cameras[device_id].name = name
            if group is not None:
                self._cameras[device_id].group = group
        self._save_persisted_devices()
        self._notify("updated", device_id)

    def update_camera_config(self, device_id: str, config: RecordingConfig) -> Tuple[bool, List[str], Optional[RecordingConfig]]:
        with self._lock:
            rec = self._cameras[device_id]
            ok, issues, alt = validate_recording_config_against_capabilities(config, rec.capabilities)
            if ok:
                if rec.calibrated and (
                    config.width != rec.config.width
                    or config.height != rec.config.height
                    or config.disable_stabilization != rec.config.disable_stabilization
                ):
                    rec.calibration_valid = False
                    rec.calibration_invalidation_reason = "Recording resolution or stabilization setting changed after calibration."
                rec.config = config
        if ok:
            self._save_persisted_devices()
            self._notify("config_changed", device_id)
        return ok, issues, alt

    def update_telemetry(self, telemetry: DeviceTelemetry) -> None:
        with self._lock:
            rec = self._cameras.get(telemetry.device_id)
            if rec is None:
                return
            rec.telemetry = telemetry
            rec.last_heartbeat_s = time.monotonic()
            if telemetry.recording_state == "recording":
                rec.state = CameraConnectionState.RECORDING
            elif rec.state == CameraConnectionState.DISCONNECTED:
                rec.state = CameraConnectionState.CONNECTED
        self._notify("telemetry", telemetry.device_id)

    def update_preview_frame(self, device_id: str, jpeg_bytes: bytes) -> None:
        with self._lock:
            rec = self._cameras.get(device_id)
            if rec is not None:
                rec.latest_preview_jpeg = jpeg_bytes
                rec.last_heartbeat_s = time.monotonic()

    def record_clock_sample(self, device_id: str, sample: ClockSyncSample) -> Optional[ClockModel]:
        with self._lock:
            rec = self._cameras.get(device_id)
            if rec is None:
                return None
            rec.sync_samples.append(sample)
            if len(rec.sync_samples) > 40:
                rec.sync_samples = rec.sync_samples[-40:]
            rec.clock_model = fit_clock_sync_model(rec.camera_uid, rec.sync_samples)
            model = rec.clock_model
        self._notify("clock_sync", device_id)
        return model

    def mark_disconnected(self, device_id: str) -> None:
        with self._lock:
            rec = self._cameras.get(device_id)
            if rec is None:
                return
            rec.state = CameraConnectionState.DISCONNECTED
        self._notify("disconnected", device_id)

    def get_camera(self, device_id: str) -> Optional[ManagedCameraRecord]:
        with self._lock:
            return self._cameras.get(device_id)

    def list_cameras(self) -> List[ManagedCameraRecord]:
        with self._lock:
            return list(self._cameras.values())

    def list_summaries(self) -> List[CameraSummary]:
        with self._lock:
            return [c.to_summary() for c in self._cameras.values()]

    def get_active_body_cameras(self) -> List[ManagedCameraRecord]:
        """Return currently enabled and connected body cameras (allows recording with 4 of 12 registered!)."""
        with self._lock:
            return [
                c
                for c in self._cameras.values()
                if c.role == CameraRole.BODY
                and c.enabled
                and c.state in (CameraConnectionState.CONNECTED, CameraConnectionState.RECORDING)
            ]

    def get_active_recording_cameras(self) -> List[ManagedCameraRecord]:
        """Return all enabled and connected cameras across body, face, and hand roles."""
        with self._lock:
            return [
                c
                for c in self._cameras.values()
                if c.enabled
                and c.state in (CameraConnectionState.CONNECTED, CameraConnectionState.RECORDING)
            ]

    def can_record_multi_view(self) -> Tuple[bool, str]:
        active = self.get_active_body_cameras()
        if len(active) < MIN_BODY_CAMERAS:
            return (
                False,
                f"Insufficient active body cameras ({len(active)} connected/enabled; minimum {MIN_BODY_CAMERAS} required for 3D reconstruction).",
            )
        return True, f"Ready to record with {len(active)} active body cameras."

    def can_record_any(self) -> Tuple[bool, str]:
        active = self.get_active_recording_cameras()
        if not active:
            return False, "No cameras connected. Pair an Android phone first."
        body_n = len(self.get_active_body_cameras())
        if body_n < MIN_BODY_CAMERAS:
            return True, f"Single-camera test mode ({len(active)} connected; connect {MIN_BODY_CAMERAS}+ body cameras for 3D mocap)."
        return True, f"Ready to record with {len(active)} active cameras ({body_n} body)."
