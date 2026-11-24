"""Camera manager package."""
from .manager import (
    CameraAdapter,
    ManagedCameraRecord,
    validate_recording_config_against_capabilities,
    CameraManager,
)

__all__ = [
    "CameraAdapter",
    "ManagedCameraRecord",
    "validate_recording_config_against_capabilities",
    "CameraManager",
]
