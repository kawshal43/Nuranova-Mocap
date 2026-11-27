"""Nuranova Multi-Video Cinema Mocap Suite — Dedicated Offline Multi-Camera Video Analysis Module.

Supports importing synchronized multi-camera videos from:
- Portable `.zip` archives (with optional `calib/` + `motion/` subfolders or flat camera videos)
- Folders of videos recorded on phones, USB cameras, GoPros, or DSLRs
- In-app synchronized recordings (Android Wi-Fi phones + PC USB webcams)

Outputs studio-grade, glitch-free, zero-foot-sliding mocap with:
- Full 25-bone body skeleton
- All 30 articulated fingers (triangulated 3D + full-resolution wrist crops)
- 6-DoF Head Orientation + 52 ARKit Facial Expressions
- Direct Blender Bridge export (`.nmocap.json`, `.nmocap.h5`, `.bvh`)
"""

from multi_video_studio.package_ingest import (
    CameraVideoClip,
    MultiVideoPackage,
    export_take_to_zip,
    ingest_multi_video_source,
)
from multi_video_studio.pipeline import (
    CinemaPipelineConfig,
    CinemaPipelineResult,
    MultiVideoCinemaPipeline,
)
from multi_video_studio.zero_slide_physics import QualityScorecard

__all__ = [
    "CameraVideoClip",
    "CinemaPipelineConfig",
    "CinemaPipelineResult",
    "MultiVideoCinemaPipeline",
    "MultiVideoPackage",
    "QualityScorecard",
    "export_take_to_zip",
    "ingest_multi_video_source",
]
