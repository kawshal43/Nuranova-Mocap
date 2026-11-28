"""Bounded hand ROI overrides, attached to native source time and wrist motion."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from engine.hands.hand_tracker import HandCaptureEngine
from engine.video.video_io import iter_video_frames


def roi_for_frame(review, uid, side, pts, body):
    candidates = [roi for roi in review.hand_rois if roi["camera_uid"] == uid and roi["side"] == side
                  and roi["start_s"] <= pts <= roi["end_s"]]
    if not candidates:
        return None
    roi = candidates[-1]
    x, y, width, height = roi["rect"]
    wrist = body[9 if side == "left" else 10]
    if roi.get("wrist_offset") is not None and wrist[2] >= .2:
        center = wrist[:2] + roi["wrist_offset"]
        x, y = center - [width / 2, height / 2]
    return np.array([x, y, width, height], float)


def crop_override(image, rect):
    h, w = image.shape[:2]
    x, y, width, height = np.asarray(rect, float)
    size = max(width, height)
    x, y = x + (width - size) / 2, y + (height - size) / 2
    width = height = size
    x0, y0 = max(0, round(x)), max(0, round(y))
    x1, y1 = min(w, round(x + width)), min(h, round(y + height))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    box = image[y0:y1, x0:x1]
    left, top = max(0, round(-x)), max(0, round(-y))
    right, bottom = max(0, round(x + size - w)), max(0, round(y + size - h))
    box = cv2.copyMakeBorder(box, top, bottom, left, right, cv2.BORDER_CONSTANT)
    crop = cv2.resize(box, (224, 224), interpolation=cv2.INTER_CUBIC)
    return crop, .8, (x0 - left, y0 - top, box.shape[1], box.shape[0])


def refine_tracks_with_rois(package, review, tracks, cache_dir=None, cancel=None, progress=None):
    """Reuse body inference and cache only the ROI-dependent hand pass."""
    from multi_video_studio.tracking_cache import tracking_cache_key, load_tracking_cache, save_tracking_cache

    output = dict(tracks)
    for uid, original in tracks.items():
        rois = [roi for roi in review.hand_rois if roi["camera_uid"] == uid]
        if not review.camera_policy.get(uid, {}).get("hands", True):
            continue
        if not rois:
            continue
        path = None
        if cache_dir:
            path = Path(cache_dir) / (tracking_cache_key(original.video_path, {
                "kind": "hand_roi", "camera_uid": uid, "rois": rois,
                "body_pins": [p for p in review.points if p["camera_uid"] == uid and p["key"].startswith("body:")],
                "pts": [original.pts_s[0], original.pts_s[-1], len(original.pts_s)],
            }) + ".npz")
            if path.exists():
                output[uid] = load_tracking_cache(path)
                continue
        n = len(original.pts_s)
        hands = [dict(frame) for frame in original.wrist_crop_hands] if original.wrist_crop_hands else [{} for _ in range(n)]
        left = original.left_hand21_uvc.copy() if original.left_hand21_uvc is not None else np.zeros((n, 21, 3), np.float32)
        right = original.right_hand21_uvc.copy() if original.right_hand21_uvc is not None else np.zeros((n, 21, 3), np.float32)
        body, _ = review.apply_points(uid, original.pts_s, original.body26_uvc)
        times = np.asarray(original.pts_s)
        start, end = max(times[0], min(r["start_s"] for r in rois)), min(times[-1], max(r["end_s"] for r in rois))
        engine = HandCaptureEngine()
        try:
            for _, pts, image in iter_video_frames(original.video_path, start_pts_s=max(0., start), end_pts_s=end):
                if cancel is not None and cancel.is_set():
                    raise RuntimeError("Hand crop retracking cancelled.")
                t = int(np.argmin(abs(times - pts)))
                if abs(times[t] - pts) > .001:
                    continue
                crops = {}
                for side in ("left", "right"):
                    roi = roi_for_frame(review, uid, side, pts, body[t])
                    if roi is not None:
                        data = crop_override(image, roi)
                        if data is not None:
                            crops[side] = data
                detected = engine.infer_hands_from_crops(crops)
                for side in crops:
                    values = left if side == "left" else right
                    if side in detected:
                        observation = detected[side]
                        observation.source = "manual_roi_model"
                        hands[t][side] = observation
                        if observation.landmarks_2d is not None:
                            values[t] = np.c_[observation.landmarks_2d, np.full(21, observation.confidence)]
                    else:
                        values[t, :, 2] = 0
                        hands[t].pop(side, None)
                if progress and t % 30 == 0:
                    progress(float((pts - start) / max(end - start, .01)), f"Retracking hand crop: {uid} {pts:.3f}s")
        finally:
            engine.close()
        track = replace(original, left_hand21_uvc=left, right_hand21_uvc=right, wrist_crop_hands=hands)
        output[uid] = track
        if path:
            save_tracking_cache(path, track)
    return output
