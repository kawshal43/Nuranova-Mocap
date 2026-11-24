"""Physical marker tracking package."""
from .marker_kit import (
    MARKER_GROUPS,
    MarkerDefinition,
    RigidClusterDefinition,
    MarkerKitConfig,
    create_default_cinema_marker_kit,
    generate_valid_marker_image,
    generate_printable_marker_sheet_pdf,
)
from .detector import (
    DetectedTag2D,
    DetectedBlob2D,
    RigidClusterPose3D,
    FiducialMarkerDetector,
    VisibleBlobMarkerDetector,
    TemporalBlobTracker2D,
    match_and_triangulate_unlabeled_blobs,
    solve_rigid_cluster_pose,
)

__all__ = [
    "MARKER_GROUPS",
    "MarkerDefinition",
    "RigidClusterDefinition",
    "MarkerKitConfig",
    "create_default_cinema_marker_kit",
    "generate_valid_marker_image",
    "generate_printable_marker_sheet_pdf",
    "DetectedTag2D",
    "DetectedBlob2D",
    "RigidClusterPose3D",
    "FiducialMarkerDetector",
    "VisibleBlobMarkerDetector",
    "TemporalBlobTracker2D",
    "match_and_triangulate_unlabeled_blobs",
    "solve_rigid_cluster_pose",
]
