"""Bounded streaming video decoding, frame PTS extraction, VFR and drop-frame analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

try:
    import av
except ImportError:  # pragma: no cover
    av = None


@dataclass
class VideoMetadata:
    path: str
    width: int
    height: int
    nominal_fps: float
    measured_fps: float
    frame_count: int
    duration_s: float
    codec: str
    is_variable_frame_rate: bool
    dropped_frame_indices: List[int]
    duplicate_frame_indices: List[int]
    pts_seconds: List[float]


def probe_video_metadata(video_path: str | Path, scan_pts: bool = True) -> VideoMetadata:
    """Inspect video resolution, nominal & measured FPS, VFR, and missing/duplicate frames."""
    p = Path(video_path)
    if not p.exists():
        raise FileNotFoundError(f"Video file not found: {p}")

    if av is not None:
        with av.open(str(p)) as container:
            vstream = container.streams.video[0]
            width = int(vstream.width)
            height = int(vstream.height)
            nominal_fps = float(vstream.average_rate or vstream.guessed_rate or 30.0)
            codec = str(vstream.codec_context.name)
            pts_list: List[float] = []
            if scan_pts:
                time_base = float(vstream.time_base)
                for packet in container.demux(vstream):
                    if packet.pts is not None:
                        pts_list.append(float(packet.pts) * time_base)
                pts_list.sort()
            frame_count = len(pts_list) if pts_list else int(vstream.frames or 0)
            duration_s = (pts_list[-1] - pts_list[0]) if len(pts_list) >= 2 else (frame_count / max(nominal_fps, 1e-6))
    else:
        cap = cv2.VideoCapture(str(p))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        nominal_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pts_list = [i / max(nominal_fps, 1e-6) for i in range(frame_count)]
        duration_s = frame_count / max(nominal_fps, 1e-6)
        codec = "unknown"
        cap.release()

    dropped: List[int] = []
    duplicates: List[int] = []
    is_vfr = False
    measured_fps = nominal_fps

    if len(pts_list) >= 3:
        dts = np.diff(pts_list)
        positive_dts = dts[dts > 1e-6]
        if len(positive_dts) > 0:
            med_dt = float(np.median(positive_dts))
            measured_fps = 1.0 / med_dt
            rel_jitter = float(np.std(positive_dts) / med_dt)
            is_vfr = rel_jitter > 0.05
            for idx, dt in enumerate(dts):
                if dt > med_dt * 1.65:
                    dropped.append(idx + 1)
                elif dt < med_dt * 0.25:
                    duplicates.append(idx + 1)

    return VideoMetadata(
        path=str(p),
        width=width,
        height=height,
        nominal_fps=nominal_fps,
        measured_fps=measured_fps,
        frame_count=frame_count,
        duration_s=duration_s,
        codec=codec,
        is_variable_frame_rate=is_vfr,
        dropped_frame_indices=dropped,
        duplicate_frame_indices=duplicates,
        pts_seconds=pts_list,
    )


def iter_video_frames(
    video_path: str | Path,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    step: int = 1,
    resize_wh: Optional[Tuple[int, int]] = None,
) -> Iterator[Tuple[int, float, np.ndarray]]:
    """Bounded streaming video frame iterator yielding (frame_index, pts_s, bgr_frame).

    Never buffers an entire video into RAM.
    """
    p = Path(video_path)
    if not p.exists():
        raise FileNotFoundError(f"Video file not found: {p}")

    if av is not None:
        with av.open(str(p)) as container:
            vstream = container.streams.video[0]
            vstream.thread_type = "AUTO"
            time_base = float(vstream.time_base)
            fps = float(vstream.average_rate or 30.0)
            for idx, frame in enumerate(container.decode(vstream)):
                if idx < start_frame:
                    continue
                if end_frame is not None and idx >= end_frame:
                    break
                if (idx - start_frame) % max(step, 1) != 0:
                    continue
                pts_s = float(frame.pts) * time_base if frame.pts is not None else idx / fps
                bgr = frame.to_ndarray(format="bgr24")
                if resize_wh is not None:
                    bgr = cv2.resize(bgr, resize_wh, interpolation=cv2.INTER_AREA)
                yield idx, pts_s, bgr
    else:
        cap = cv2.VideoCapture(str(p))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx >= start_frame and (end_frame is None or idx < end_frame):
                    if (idx - start_frame) % max(step, 1) == 0:
                        if resize_wh is not None:
                            frame = cv2.resize(frame, resize_wh, interpolation=cv2.INTER_AREA)
                        yield idx, idx / fps, frame
                idx += 1
                if end_frame is not None and idx >= end_frame:
                    break
        finally:
            cap.release()
