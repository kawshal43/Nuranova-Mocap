"""Versioned, pickle-free observation cache for repeatable reconstruction experiments."""

from __future__ import annotations

import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from shared_protocol.atomic_io import atomic_path

from engine.face.face_tracker import FaceFrameResult
from engine.hands.hand_tracker import HandObservation
from multi_video_studio.wholebody_cuda import CameraTrackingResult


@lru_cache(maxsize=1)
def _bundle_digest(executable):
    # Frozen modules live in PYZ, not in loose .py files. The executable embeds
    # that archive, so a changed build invalidates its cached detector results.
    with Path(executable).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").digest()


def tracking_cache_key(video: Path, settings: dict) -> str:
    stat = video.stat()
    code = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    names = [
        "multi_video_studio/wholebody_cuda.py",
        "engine/ai_tracking/pose_tracker.py",
        "engine/hands/hand_tracker.py",
        "engine/face/face_tracker.py",
        "engine/video/video_io.py",
    ]
    if getattr(sys, "frozen", False):
        code.update(_bundle_digest(sys.executable))
    else:
        if str(settings.get("kind", "")).startswith("monocular"):
            names.append("multi_video_studio/monocular_tracker.py")
        if settings.get("kind") == "hand_roi":
            names.append("multi_video_studio/roi_tracking.py")
        for name in names:
            code.update((root/name).read_bytes())
    payload = {
        "version": 1,
        "path": str(video.resolve()),
        "size": stat.st_size,
        "mtime": stat.st_mtime_ns,
        "code": code.hexdigest(),
        "settings": settings,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def save_tracking_cache(path: Path, track: CameraTrackingResult) -> None:
    arrays = {"body26_uvc": track.body26_uvc, "pts_s": np.asarray(track.pts_s)}
    for name in ("left_hand21_uvc", "right_hand21_uvc", "face68_uvc", "relative_body26"):
        value = getattr(track, name)
        if value is not None:
            arrays[name] = value
    for side in ("left", "right"):
        frames = track.wrist_crop_hands
        arrays[side + "_source"] = np.asarray([f[side].source if side in f else "missing" for f in frames])
        arrays[side + "_uv"] = np.asarray([f[side].landmarks_2d if side in f and f[side].landmarks_2d is not None
                                           else np.full((21, 2), np.nan) for f in frames])
        arrays[side + "_point_conf"] = np.asarray([f[side].landmark_confidence if side in f and f[side].landmark_confidence is not None
                                                   else np.full(21, f[side].confidence if side in f else 0) for f in frames])
        arrays[side + "_confidence"] = np.array(
            [f[side].confidence if side in f else 0 for f in frames]
        )
        arrays[side + "_landmarks"] = np.array(
            [f[side].landmarks_3d if side in f else np.zeros((21, 3)) for f in frames]
        )
        arrays[side + "_rotations"] = np.array(
            [
                f[side].local_rotations_xyzw if side in f else np.tile([0, 0, 0, 1], (16, 1))
                for f in frames
            ]
        )
    faces = track.head_crop_faces
    arrays["face_detected"] = np.array([bool(f is not None and f.detected) for f in faces])
    arrays["face_blendshapes"] = np.array([f.blendshapes if f else np.zeros(52) for f in faces])
    arrays["face_rotation"] = np.array([f.head_rotation_xyzw if f else [0, 0, 0, 1] for f in faces])
    arrays["face_gaze"] = np.array([f.gaze_left_right if f else np.zeros(4) for f in faces])
    arrays["face_confidence"] = np.array([f.confidence if f else 0 for f in faces])
    metadata = {
        k: getattr(track, k)
        for k in (
            "camera_uid",
            "width",
            "height",
            "fps",
            "backend_tag",
            "device",
            "detection_rate",
            "measured_fps",
        )
    }
    metadata["video_path"] = str(track.video_path)
    metadata["runtime_metadata"] = track.runtime_metadata
    arrays["metadata"] = np.array(json.dumps(metadata))
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as temporary:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)


def load_tracking_cache(path: Path) -> CameraTrackingResult:
    with np.load(path, allow_pickle=False) as archive:
        # NpzFile decompresses a whole member on every __getitem__. Materialize
        # once: indexing it inside the per-frame hand loop made full takes O(T²).
        data = {name: archive[name] for name in archive.files}
        metadata = json.loads(str(data["metadata"]))
        metadata["video_path"] = Path(metadata["video_path"])
        kwargs = {
            k: data[k].copy()
            for k in ("body26_uvc", "left_hand21_uvc", "right_hand21_uvc", "face68_uvc", "relative_body26")
            if k in data
        }
        n_hands = len(data["left_confidence"])
        hands = [{} for _ in range(n_hands)]
        for side in ("left", "right"):
            for i, confidence in enumerate(data[side + "_confidence"]):
                if confidence > 0:
                    hands[i][side] = HandObservation(
                        side,
                        data[side + "_landmarks"][i].copy(),
                        float(confidence),
                        data[side + "_rotations"][i].copy(),
                        landmark_confidence=data[side + "_point_conf"][i].copy() if side + "_point_conf" in data else None,
                        landmarks_2d=data[side + "_uv"][i].copy() if side + "_uv" in data and np.isfinite(data[side + "_uv"][i]).all() else None,
                        source=str(data[side + "_source"][i]) if side + "_source" in data else "single_view",
                    )
        faces = [
            FaceFrameResult(
                True,
                data["face_blendshapes"][i].copy(),
                data["face_rotation"][i].copy(),
                data["face_gaze"][i].copy(),
                confidence=float(data["face_confidence"][i]),
                source_backend="cached",
            )
            if detected
            else None
            for i, detected in enumerate(data["face_detected"])
        ]
        return CameraTrackingResult(
            **metadata,
            **kwargs,
            pts_s=data["pts_s"].tolist(),
            wrist_crop_hands=hands,
            head_crop_faces=faces,
        )
