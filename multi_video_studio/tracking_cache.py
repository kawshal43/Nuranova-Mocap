"""Versioned, pickle-free observation cache for repeatable reconstruction experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from engine.face.face_tracker import FaceFrameResult
from engine.hands.hand_tracker import HandObservation
from multi_video_studio.wholebody_cuda import CameraTrackingResult


def tracking_cache_key(video: Path, settings: dict) -> str:
    stat = video.stat()
    code = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for name in (
        "multi_video_studio/wholebody_cuda.py",
        "engine/ai_tracking/pose_tracker.py",
        "engine/hands/hand_tracker.py",
        "engine/face/face_tracker.py",
        "engine/video/video_io.py",
    ):
        code.update((root / name).read_bytes())
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
    for name in ("left_hand21_uvc", "right_hand21_uvc", "face68_uvc"):
        value = getattr(track, name)
        if value is not None:
            arrays[name] = value
    for side in ("left", "right"):
        frames = track.wrist_crop_hands
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
    arrays["metadata"] = np.array(json.dumps(metadata))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def load_tracking_cache(path: Path) -> CameraTrackingResult:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        metadata["video_path"] = Path(metadata["video_path"])
        kwargs = {
            k: data[k].copy()
            for k in ("body26_uvc", "left_hand21_uvc", "right_hand21_uvc", "face68_uvc")
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
