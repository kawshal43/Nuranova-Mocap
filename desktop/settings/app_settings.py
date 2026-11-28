"""Comprehensive persisted application settings for Nuranova Mocap Studio."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field


class GeneralSettings(BaseModel):
    language: str = "en-US"
    theme: Literal["light", "dark"] = "dark"
    reduce_interface_motion: bool = False
    single_camera_mode: bool = False  # True = 1-phone AI 3D workflow (skips multi-cam calibration)
    single_cam_root_motion: bool = True  # Estimate 3D left/right/forward/back walking & jumps from 1 phone
    open_last_project_on_startup: bool = True
    default_project_dir: str = str(Path.home() / "Documents" / "NuranovaMocapProjects")
    autosave_enabled: bool = True
    autosave_interval_minutes: int = 5
    backup_retention_count: int = 5


class NetworkSettings(BaseModel):
    bind_ip: str = "0.0.0.0"
    preferred_lan_ip: str = ""
    server_port: int = 8765
    udp_discovery_port: int = 8766
    auto_start_server: bool = True
    enable_mdns_discovery: bool = True
    enable_udp_beacon: bool = True
    connection_timeout_s: int = 15
    transfer_concurrency: int = 3
    preview_max_fps: float = 8.0
    preview_jpeg_quality: int = 55


class CameraDefaultSettings(BaseModel):
    default_width: int = 1920
    default_height: int = 1080
    default_fps: float = 60.0
    default_bitrate_mbps: int = 20
    default_shutter_denom: int = 250  # 1/250 s
    lock_focus: bool = True
    lock_white_balance: bool = True
    disable_stabilization: bool = True
    sync_probe_count: int = 16


class ProcessingSettings(BaseModel):
    compute_backend: Literal["auto", "cuda", "cpu"] = "auto"
    ai_backend: Literal["rtmlib_wholebody", "rtmlib_halpe26", "mediapipe_pose"] = "rtmlib_wholebody"
    default_profile: Literal["fast_preview", "standard", "cinema"] = "cinema"
    model_storage_dir: str = str(Path("model_registry/cache").resolve())
    batch_size: int = 4
    worker_count: int = 4
    max_memory_mb: int = 8192
    default_filter: Literal["butterworth", "one_euro", "none"] = "butterworth"
    default_filter_cutoff_hz: float = 7.5
    enable_hand_capture: bool = True  # Capture 30 articulated finger bones (left + right hands)


class AnimationSettings(BaseModel):
    skeleton_convention: str = "nuranova_humanoid_25_zup"
    output_fps: float = 60.0
    extract_root_motion: bool = True
    enable_foot_locking: bool = True
    retarget_scale_compensation: bool = True


class StorageSettings(BaseModel):
    cache_dir: str = str(Path.home() / ".nuranova_mocap" / "cache")
    export_dir: str = str(Path.home() / "Documents" / "NuranovaMocapExports")
    min_free_disk_gb: float = 10.0


class BlenderSettings(BaseModel):
    blender_executable: str = ""
    bridge_poll_interval_s: float = 1.0
    default_export_format: Literal["nmocap_json", "bvh", "fbx"] = "nmocap_json"


class PrivacySecuritySettings(BaseModel):
    local_network_only: bool = True
    require_recording_consent_indicator: bool = True
    keep_diagnostic_logs_days: int = 30
    require_checksum_before_phone_cleanup: bool = True


class AppSettings(BaseModel):
    general: GeneralSettings = Field(default_factory=GeneralSettings)
    network: NetworkSettings = Field(default_factory=NetworkSettings)
    camera: CameraDefaultSettings = Field(default_factory=CameraDefaultSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    animation: AnimationSettings = Field(default_factory=AnimationSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    blender: BlenderSettings = Field(default_factory=BlenderSettings)
    privacy: PrivacySecuritySettings = Field(default_factory=PrivacySecuritySettings)

    @classmethod
    def default_path(cls) -> Path:
        p = Path.home() / ".nuranova_mocap" / "settings.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AppSettings":
        p = path or cls.default_path()
        if p.exists():
            try:
                settings = cls.model_validate_json(p.read_text(encoding="utf-8"))
                if not settings.blender.blender_executable:
                    settings.blender.blender_executable = detect_blender_executable()
                return settings
            except Exception:
                pass
        s = cls()
        s.blender.blender_executable = detect_blender_executable()
        return s

    def save(self, path: Optional[Path] = None) -> Path:
        p = path or self.default_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return p


def detect_blender_executable() -> str:
    """Auto-detect installed Blender 4.5 LTS / 5.x executable on Windows."""
    on_path = shutil.which("blender")
    if on_path:
        return on_path
    base = Path("C:/Program Files/Blender Foundation")
    if base.exists():
        candidates = sorted(base.glob("Blender */blender.exe"), reverse=True)
        if candidates:
            return str(candidates[0])
    return ""
