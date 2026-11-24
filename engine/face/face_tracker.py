"""Dedicated Facial Motion Capture module using MediaPipe Face Landmarker (52 blendshapes + head pose + gaze).

Note on ARKit naming: MediaPipe's 52 blendshape names follow the standard ARKit naming schema,
but numerical response curves differ slightly from Apple TrueDepth sensors; users can apply
per-character gain/offset mapping profiles inside Nuranova Mocap Studio and Blender Bridge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from engine.data_format.motion_io import FaceMotionTrack
from engine.math_core.transforms import enforce_quaternion_continuity, matrix_to_quat, quat_identity

MEDIAPIPE_52_BLENDSHAPES: List[str] = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft",
    "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft",
    "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft",
    "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
]


@dataclass
class FaceFrameResult:
    detected: bool
    blendshapes: np.ndarray  # (52,) float32 [0..1]
    head_rotation_xyzw: np.ndarray  # (4,) float32
    gaze_left_right: np.ndarray  # (4,) float32 [l_yaw, l_pitch, r_yaw, r_pitch] rad
    landmarks_478: Optional[np.ndarray] = None  # (478, 3) normalized
    confidence: float = 0.0
    source_backend: str = "mediapipe_face_landmarker"


@dataclass
class FacialCalibrationProfile:
    neutral_baseline: np.ndarray = field(default_factory=lambda: np.zeros(52, dtype=np.float32))
    gains: np.ndarray = field(default_factory=lambda: np.ones(52, dtype=np.float32))
    shape_key_map: Dict[str, str] = field(default_factory=lambda: {name: name for name in MEDIAPIPE_52_BLENDSHAPES})

    def apply(self, raw_blendshapes: np.ndarray) -> np.ndarray:
        """Apply neutral calibration baseline subtraction and per-channel gain."""
        arr = np.asarray(raw_blendshapes, dtype=np.float32)
        calibrated = (arr - self.neutral_baseline) * self.gains
        return np.clip(calibrated, 0.0, 1.0)


class FacialCaptureEngine:
    """Tracks facial landmarks, 52 blendshape coefficients, head pose, and eye gaze."""

    def __init__(self, task_model_path: Optional[str | Path] = None) -> None:
        self.task_model_path = Path(task_model_path) if task_model_path else None
        self._landmarker = None
        self._face_mesh = None
        self.backend_label = "uninitialized"
        self.calibration = FacialCalibrationProfile()

    def _ensure_loaded(self) -> None:
        if self._landmarker is not None or self._face_mesh is not None:
            return
        import mediapipe as mp

        if self.task_model_path and self.task_model_path.exists():
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            options = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(self.task_model_path)),
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1,
            )
            self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            self.backend_label = "mediapipe_tasks_face_landmarker_52bs"
        else:
            # Built-in MediaPipe FaceMesh fallback (468+10 iris landmarks, geometric proxy blendshapes)
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.backend_label = "mediapipe_facemesh_geometric_fallback"

    def infer_frame(self, frame_bgr: np.ndarray) -> FaceFrameResult:
        self._ensure_loaded()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self._landmarker is not None:
            import mediapipe as mp

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = self._landmarker.detect(mp_img)
            if not res.face_landmarks:
                return FaceFrameResult(
                    detected=False,
                    blendshapes=np.zeros(52, dtype=np.float32),
                    head_rotation_xyzw=quat_identity().astype(np.float32),
                    gaze_left_right=np.zeros(4, dtype=np.float32),
                    confidence=0.0,
                    source_backend=self.backend_label,
                )
            lms = np.array([[p.x, p.y, p.z] for p in res.face_landmarks[0]], dtype=np.float32)
            bs = np.zeros(52, dtype=np.float32)
            if res.face_blendshapes:
                name_to_idx = {n: i for i, n in enumerate(MEDIAPIPE_52_BLENDSHAPES)}
                for cat in res.face_blendshapes[0]:
                    if cat.category_name in name_to_idx:
                        bs[name_to_idx[cat.category_name]] = float(cat.score)
            head_q = quat_identity()
            if res.facial_transformation_matrixes:
                M = np.asarray(res.facial_transformation_matrixes[0], dtype=np.float64)
                head_q = matrix_to_quat(M[:3, :3])
            gaze = self._estimate_gaze_from_blendshapes(bs)
            return FaceFrameResult(
                detected=True,
                blendshapes=self.calibration.apply(bs),
                head_rotation_xyzw=head_q.astype(np.float32),
                gaze_left_right=gaze,
                landmarks_478=lms,
                confidence=0.9,
                source_backend=self.backend_label,
            )

        # Fallback: MediaPipe FaceMesh with refined iris landmarks
        res = self._face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return FaceFrameResult(
                detected=False,
                blendshapes=np.zeros(52, dtype=np.float32),
                head_rotation_xyzw=quat_identity().astype(np.float32),
                gaze_left_right=np.zeros(4, dtype=np.float32),
                confidence=0.0,
                source_backend=self.backend_label,
            )
        lms = np.array([[p.x, p.y, p.z] for p in res.multi_face_landmarks[0].landmark], dtype=np.float32)
        bs, head_q, gaze = self._derive_geometric_face_features(lms)
        return FaceFrameResult(
            detected=True,
            blendshapes=self.calibration.apply(bs),
            head_rotation_xyzw=head_q.astype(np.float32),
            gaze_left_right=gaze,
            landmarks_478=lms,
            confidence=0.75,
            source_backend=self.backend_label,
        )

    @staticmethod
    def _estimate_gaze_from_blendshapes(bs: np.ndarray) -> np.ndarray:
        idx = {n: i for i, n in enumerate(MEDIAPIPE_52_BLENDSHAPES)}
        l_yaw = (bs[idx["eyeLookOutLeft"]] - bs[idx["eyeLookInLeft"]]) * 0.5
        l_pitch = (bs[idx["eyeLookUpLeft"]] - bs[idx["eyeLookDownLeft"]]) * 0.4
        r_yaw = (bs[idx["eyeLookInRight"]] - bs[idx["eyeLookOutRight"]]) * 0.5
        r_pitch = (bs[idx["eyeLookUpRight"]] - bs[idx["eyeLookDownRight"]]) * 0.4
        return np.array([l_yaw, l_pitch, r_yaw, r_pitch], dtype=np.float32)

    @staticmethod
    def _derive_geometric_face_features(lms: np.ndarray):
        """Derive jaw, blink, smile, brow, and head rotation from 468/478 FaceMesh landmarks."""
        bs = np.zeros(52, dtype=np.float32)
        idx = {n: i for i, n in enumerate(MEDIAPIPE_52_BLENDSHAPES)}
        face_scale = max(float(np.linalg.norm(lms[10] - lms[152])), 1e-4)  # forehead to chin
        # Jaw open (upper inner lip 13 vs lower inner lip 14)
        jaw_dist = float(np.linalg.norm(lms[13, :2] - lms[14, :2])) / face_scale
        bs[idx["jawOpen"]] = float(np.clip((jaw_dist - 0.015) / 0.14, 0.0, 1.0))
        # Eye blinks (upper/lower eyelid distances: L 159-145, R 386-374)
        l_eye = float(np.linalg.norm(lms[159, :2] - lms[145, :2])) / face_scale
        r_eye = float(np.linalg.norm(lms[386, :2] - lms[374, :2])) / face_scale
        bs[idx["eyeBlinkLeft"]] = float(np.clip(1.0 - l_eye / 0.045, 0.0, 1.0))
        bs[idx["eyeBlinkRight"]] = float(np.clip(1.0 - r_eye / 0.045, 0.0, 1.0))
        # Smile width (mouth corners 61-291)
        mouth_w = float(np.linalg.norm(lms[61, :2] - lms[291, :2])) / face_scale
        smile = float(np.clip((mouth_w - 0.28) / 0.15, 0.0, 1.0))
        bs[idx["mouthSmileLeft"]] = smile
        bs[idx["mouthSmileRight"]] = smile
        # Brow lift (inner brow 107/336 to eye)
        brow_l = float(np.linalg.norm(lms[107, :2] - lms[159, :2])) / face_scale
        bs[idx["browInnerUp"]] = float(np.clip((brow_l - 0.09) / 0.08, 0.0, 1.0))

        # Approximate head rotation frame from forehead(10), chin(152), left cheek(234), right cheek(454)
        x_ax = lms[454] - lms[234]
        x_ax = x_ax / max(float(np.linalg.norm(x_ax)), 1e-6)
        z_ax = lms[10] - lms[152]
        z_ax = z_ax / max(float(np.linalg.norm(z_ax)), 1e-6)
        y_ax = np.cross(z_ax, x_ax)
        y_ax = y_ax / max(float(np.linalg.norm(y_ax)), 1e-6)
        z_ax = np.cross(x_ax, y_ax)
        R = np.column_stack([x_ax, y_ax, z_ax])
        head_q = matrix_to_quat(R)
        return bs, head_q, np.zeros(4, dtype=np.float32)

    def calibrate_neutral(self, neutral_frames: List[FaceFrameResult]) -> FacialCalibrationProfile:
        valid = [f.blendshapes for f in neutral_frames if f.detected]
        if valid:
            self.calibration.neutral_baseline = np.median(np.vstack(valid), axis=0).astype(np.float32)
        return self.calibration

    def build_track(self, frames: List[FaceFrameResult], smooth_alpha: float = 0.35) -> FaceMotionTrack:
        T = len(frames)
        bs = np.zeros((T, 52), dtype=np.float32)
        head_q = np.tile(quat_identity().astype(np.float32), (T, 1))
        gaze = np.zeros((T, 4), dtype=np.float32)
        conf = np.zeros((T,), dtype=np.float32)

        for t, f in enumerate(frames):
            bs[t] = f.blendshapes
            head_q[t] = f.head_rotation_xyzw
            gaze[t] = f.gaze_left_right
            conf[t] = f.confidence

        if T >= 2 and 0.0 < smooth_alpha < 1.0:
            for t in range(1, T):
                bs[t] = smooth_alpha * bs[t] + (1.0 - smooth_alpha) * bs[t - 1]
                gaze[t] = smooth_alpha * gaze[t] + (1.0 - smooth_alpha) * gaze[t - 1]

        head_q = enforce_quaternion_continuity(head_q).astype(np.float32)
        return FaceMotionTrack(
            blendshapes=bs,
            blendshape_names=list(MEDIAPIPE_52_BLENDSHAPES),
            head_rotation_xyzw=head_q,
            gaze_left_right=gaze,
            confidence=conf,
        )

    def close(self) -> None:
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
