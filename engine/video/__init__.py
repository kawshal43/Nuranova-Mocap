"""Streaming video I/O package."""
from .video_io import VideoMetadata, probe_video_metadata, iter_video_frames

__all__ = ["VideoMetadata", "probe_video_metadata", "iter_video_frames"]
