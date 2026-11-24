"""Multi-camera calibration package."""
from .charuco_board import (
    ARUCO_DICTIONARIES,
    ChArUcoBoardConfig,
    ChArUcoDetection,
    render_charuco_png,
    generate_printable_charuco_pdf,
    detect_charuco_corners,
)
from .calibrator import (
    CalibrationReport,
    calibrate_camera_intrinsics,
    estimate_board_pose_pnp,
    initialize_extrinsics_from_board_frames,
    run_bundle_adjustment,
    run_board_bundle_adjustment,
    align_calibration_to_world_z_up,
    validate_known_distance,
)

__all__ = [
    "ARUCO_DICTIONARIES",
    "ChArUcoBoardConfig",
    "ChArUcoDetection",
    "render_charuco_png",
    "generate_printable_charuco_pdf",
    "detect_charuco_corners",
    "CalibrationReport",
    "calibrate_camera_intrinsics",
    "estimate_board_pose_pnp",
    "initialize_extrinsics_from_board_frames",
    "run_bundle_adjustment",
    "run_board_bundle_adjustment",
    "align_calibration_to_world_z_up",
    "validate_known_distance",
]
