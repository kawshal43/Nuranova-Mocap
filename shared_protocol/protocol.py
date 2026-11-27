"""Shared versioned network protocol models for Nuranova Mocap Ecosystem.

Protocol version: nuranova-cam-proto/1.0
Used identically across Desktop Studio (FastAPI/WebSocket), Android Camera App (Kotlin mirror),
and Blender Bridge HTTP API.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator

PROTOCOL_VERSION = "1.0.0"
MDNS_SERVICE_TYPE = "_nuranova-mocap._tcp.local."
DEFAULT_SERVER_PORT = 8765
MIN_BODY_CAMERAS = 2
MAX_BODY_CAMERAS = 12


class CameraRole(str, Enum):
    BODY = "body"
    FACE = "face"
    HAND_LEFT = "hand_left"
    HAND_RIGHT = "hand_right"
    HAND_DUAL = "hand_dual"
    REFERENCE = "reference"


class CameraConnectionState(str, Enum):
    DISCOVERED = "discovered"
    PAIRING = "pairing"
    CONNECTED = "connected"
    RECORDING = "recording"
    TRANSFERRING = "transferring"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class TimestampSource(str, Enum):
    UNKNOWN = "UNKNOWN"
    REALTIME = "REALTIME"  # CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME
    MONOTONIC = "MONOTONIC"  # SENSOR_INFO_TIMESTAMP_SOURCE_UNKNOWN (CLOCK_MONOTONIC)


class SupportedMode(BaseModel):
    width: int = Field(..., ge=320, le=7680)
    height: int = Field(..., ge=240, le=4320)
    fps_ranges: List[List[int]] = Field(default_factory=list)
    max_fps: float = Field(..., gt=0)
    supports_high_speed: bool = False


class CameraCapabilities(BaseModel):
    manufacturer: str = "Unknown"
    model: str = "Unknown"
    android_api_level: int = 26
    camera_id: str = "0"
    lens_facing: Literal["back", "front", "external"] = "back"
    hardware_level: str = "LIMITED"
    timestamp_source: TimestampSource = TimestampSource.UNKNOWN
    supported_modes: List[SupportedMode] = Field(default_factory=list)
    supports_manual_exposure: bool = False
    supports_manual_focus: bool = False
    supports_awb_lock: bool = False
    supports_video_stabilization_off: bool = True
    exposure_time_range_ns: Optional[List[int]] = None
    iso_range: Optional[List[int]] = None
    focal_lengths_mm: List[float] = Field(default_factory=list)
    sensor_physical_size_mm: Optional[List[float]] = None


class RecordingConfig(BaseModel):
    width: int = Field(1920, ge=640, le=4096)
    height: int = Field(1080, ge=480, le=2160)
    target_fps: float = Field(60.0, ge=15.0, le=240.0)
    bitrate_bps: int = Field(20_000_000, ge=1_000_000, le=150_000_000)
    codec: Literal["H264"] = "H264"
    camera_lens: Literal["rear_main", "front", "rear_wide"] = "rear_main"
    lock_focus: bool = True
    focus_distance_diopters: Optional[float] = None
    manual_exposure: bool = False  # Auto-exposure + anti-banding default prevents indoor 50Hz/60Hz LED flicker
    shutter_speed_ns: int = 10_000_000  # 1/100s flicker-free indoor default when manual exposure is enabled
    iso: Optional[int] = None
    lock_white_balance: bool = True
    disable_stabilization: bool = True
    disable_hdr: bool = True
    record_audio: bool = True
    preview_width: int = 960
    preview_height: int = 540
    preview_fps: float = 8.0
    preview_jpeg_quality: int = 82


class DeviceTelemetry(BaseModel):
    device_id: str
    battery_percent: float = Field(..., ge=0.0, le=100.0)
    is_charging: bool = False
    temperature_celsius: Optional[float] = None
    available_storage_mb: float = Field(..., ge=0.0)
    actual_width: int = 1920
    actual_height: int = 1080
    measured_fps: float = 0.0
    dropped_frames: int = 0
    shutter_speed_ns: Optional[int] = None
    iso: Optional[int] = None
    focus_locked: bool = False
    wb_locked: bool = False
    orientation_degrees: int = 0
    recording_state: Literal["idle", "armed", "recording", "finalizing", "error"] = "idle"
    active_take_id: Optional[str] = None
    recording_duration_s: float = 0.0
    monotonic_time_ns: int = 0


class PairRequest(BaseModel):
    protocol_version: str = PROTOCOL_VERSION
    device_id: str = Field(..., min_length=4, max_length=128)
    device_name: str = Field(..., min_length=1, max_length=64)
    pin: str = Field(..., min_length=6, max_length=12)
    capabilities: CameraCapabilities
    role: CameraRole = CameraRole.BODY


class PairResponse(BaseModel):
    protocol_version: str = PROTOCOL_VERSION
    paired: bool
    device_id: str
    assigned_camera_uid: str
    auth_token: str
    studio_name: str = "Nuranova Mocap Studio"
    studio_time_ns: int


class ClockSyncRequest(BaseModel):
    device_id: str
    seq: int
    client_t1_ns: int


class ClockSyncResponse(BaseModel):
    device_id: str
    seq: int
    client_t1_ns: int
    server_t2_ns: int
    server_t3_ns: int


class ScheduleRecordCommand(BaseModel):
    command: Literal["SCHEDULE_RECORD"] = "SCHEDULE_RECORD"
    session_id: str
    take_id: str
    scheduled_start_studio_ns: int
    scheduled_start_device_ns: int
    config: RecordingConfig
    flash_sync_pulse: bool = False
    audio_clap_cue: bool = True


class StopRecordCommand(BaseModel):
    command: Literal["STOP_RECORD"] = "STOP_RECORD"
    session_id: str
    take_id: str


class FrameTimestampRecord(BaseModel):
    frame_index: int = Field(..., ge=0)
    sensor_timestamp_ns: int
    pts_us: int
    monotonic_receive_ns: int
    exposure_ns: Optional[int] = None
    rolling_shutter_skew_ns: Optional[int] = None
    frame_duration_ns: Optional[int] = None


class TransferInitRequest(BaseModel):
    device_id: str
    session_id: str
    take_id: str
    filename: str
    file_kind: Literal["video", "timestamps", "audio", "metadata"]
    total_size_bytes: int = Field(..., gt=0)
    sha256_hex: str = Field(..., min_length=64, max_length=64)

    @field_validator("filename")
    @classmethod
    def validate_safe_filename(cls, v: str) -> str:
        if "/" in v or "\\" in v or ".." in v or not v.strip():
            raise ValueError("Unsafe filename rejected")
        return v


class TransferInitResponse(BaseModel):
    transfer_id: str
    received_bytes: int
    chunk_size_bytes: int = 1024 * 1024
    already_complete: bool = False


class TransferStatusResponse(BaseModel):
    transfer_id: str
    received_bytes: int
    total_size_bytes: int
    verified: bool
    computed_sha256: Optional[str] = None
    expected_sha256: str
    can_delete_local_copy: bool = False


class CameraSummary(BaseModel):
    camera_uid: str
    device_id: str
    name: str
    group: str = "Main Volume"
    role: CameraRole = CameraRole.BODY
    enabled: bool = True
    state: CameraConnectionState = CameraConnectionState.DISCONNECTED
    ip_address: str = ""
    calibrated: bool = False
    calibration_valid: bool = False
    calibration_invalidation_reason: Optional[str] = None
    sync_offset_ns: Optional[float] = None
    sync_drift_ppm: Optional[float] = None
    sync_residual_ms: Optional[float] = None
    rtt_ms: Optional[float] = None
    capabilities: Optional[CameraCapabilities] = None
    config: RecordingConfig = Field(default_factory=RecordingConfig)
    telemetry: Optional[DeviceTelemetry] = None


class BridgeStatusPayload(BaseModel):
    studio_version: str = "1.0.0"
    protocol_version: str = PROTOCOL_VERSION
    active_project_name: Optional[str] = None
    active_project_path: Optional[str] = None
    available_takes: List[Dict[str, Any]] = Field(default_factory=list)
    latest_motion_file: Optional[str] = None
