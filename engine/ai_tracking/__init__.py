"""AI tracking package."""
from .runtime import HardwareDiagnostics, detect_hardware
from .pose_tracker import (
    CANONICAL_BODY26_NAMES,
    FramePose2D,
    PoseBenchmarkStats,
    AIPoseTracker2D,
)

__all__ = [
    "HardwareDiagnostics",
    "detect_hardware",
    "CANONICAL_BODY26_NAMES",
    "FramePose2D",
    "PoseBenchmarkStats",
    "AIPoseTracker2D",
]
