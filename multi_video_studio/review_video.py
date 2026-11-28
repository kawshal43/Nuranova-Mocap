"""Bounded, timestamp-based video access for camera review, independent of Qt."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np


@dataclass
class ReviewVideoFrame:
    pts_s: float
    image: np.ndarray
    native_size: tuple
    frame_index: int | None

    def __iter__(self):
        return iter((self.pts_s, self.image, self.native_size))

    def __getitem__(self, index):
        return (self.pts_s, self.image, self.native_size)[index]


class ReviewVideoReader:
    def __init__(self, path: Path, cache_frames=10):
        self.container = av.open(str(path))
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        self.stream.codec_context.thread_count = 2
        self.cache_frames = cache_frames
        self.cache = OrderedDict()
        self.frames = iter(())
        self.previous = self.following = None
        self.last_request = None
        # Container presentation timestamps can be indexed without decoding 4K images.
        with av.open(str(path)) as index_container:
            index_stream = index_container.streams.video[0]
            self.presentation_times = np.unique([float(packet.pts * index_stream.time_base)
                for packet in index_container.demux(index_stream) if packet.pts is not None and packet.size > 0])
        intervals = np.diff(self.presentation_times)
        self.sampling_tolerance = min(.25, max(.005, float(np.median(intervals)) * .65)) if len(intervals) else .05

    def close(self):
        self.cache.clear()
        self.container.close()

    def at(self, pts_s: float, max_width=960):
        if not np.isfinite(pts_s) or pts_s < 0:
            return None
        key = (round(pts_s, 5), max_width)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        if self.last_request is None or pts_s < self.last_request - .001 or pts_s > self.last_request + 1.2:
            self.container.seek(int(max(0, pts_s - .35) * av.time_base), backward=True)
            self.frames = self.container.decode(self.stream)
            self.previous = self.following = None
        self.last_request = pts_s
        while self.following is None or self.following.time < pts_s:
            self.previous = self.following
            self.following = next(self.frames, None)
            if self.following is None:
                break
        candidates = [f for f in (self.previous, self.following) if f is not None and f.time is not None]
        if not candidates:
            return None
        frame = min(candidates, key=lambda f: abs(f.time - pts_s))
        # Never present an end-of-video frame for a much later requested instant.
        if abs(frame.time - pts_s) > self.sampling_tolerance:
            return None
        width, height = self.stream.width, self.stream.height
        if max_width and width > max_width:
            frame_small = frame.reformat(width=max_width, height=max(1, round(height * max_width / width)), format="bgr24")
            image = frame_small.to_ndarray()
        else:
            image = frame.to_ndarray(format="bgr24")
        frame_index = int(np.argmin(abs(self.presentation_times - frame.time))) if len(self.presentation_times) else None
        if frame_index is not None and abs(self.presentation_times[frame_index] - frame.time) > .001:
            frame_index = None
        result = ReviewVideoFrame(float(frame.time), image, (width, height), frame_index)
        self.cache[key] = result
        while len(self.cache) > self.cache_frames:
            self.cache.popitem(last=False)
        return result


def hand_crop(image, points, native_size, center=True, side=384, roi=None, size_override=None):
    """Display-only crop transform. Returns image and native-pixel-to-crop affine map."""
    height, width = image.shape[:2]
    native_w, native_h = native_size
    valid = np.isfinite(points[:, :2]).all(axis=1) & (points[:, 2] >= .2)
    if np.count_nonzero(valid) < 3 and roi is None:
        return None
    xy = points[valid, :2]
    if roi is None:
        low, high = xy.min(axis=0), xy.max(axis=0)
        size = max(48., float(np.max(high - low)) * 1.4)
        midpoint = (low + high) / 2
        if size_override is not None:
            size = max(size, float(size_override))
    else:
        x, y, width_roi, height_roi = roi
        size = max(width_roi, height_roi)
        midpoint = np.array([x + width_roi / 2, y + height_roi / 2])
    if center and valid[0] and roi is None:
        anchors = [i for i in (0, 5, 9, 13, 17) if valid[i]]
        palm = np.mean(points[anchors, :2], axis=0)
        midpoint = .7 * midpoint + .3 * palm
    scale = side / size
    native_to_crop = np.array([[scale, 0, side / 2 - midpoint[0] * scale],
                               [0, scale, side / 2 - midpoint[1] * scale]], float)
    preview_to_crop = native_to_crop.copy()
    preview_to_crop[:, 0] *= native_w / width
    preview_to_crop[:, 1] *= native_h / height
    crop = cv2.warpAffine(image, preview_to_crop, (side, side), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(16, 22, 30))
    return crop, native_to_crop
