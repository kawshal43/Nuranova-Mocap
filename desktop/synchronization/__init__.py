"""Multi-camera synchronization package."""
from .sync_engine import (
    ClockSyncSample,
    ClockModel,
    CameraTimelineReport,
    fit_clock_sync_model,
    map_frame_timestamps_to_session_timeline,
    estimate_audio_sync_offset_s,
    detect_flash_sync_frame,
    align_observations_to_common_timeline,
)

__all__ = [
    "ClockSyncSample",
    "ClockModel",
    "CameraTimelineReport",
    "fit_clock_sync_model",
    "map_frame_timestamps_to_session_timeline",
    "estimate_audio_sync_offset_s",
    "detect_flash_sync_frame",
    "align_observations_to_common_timeline",
]
