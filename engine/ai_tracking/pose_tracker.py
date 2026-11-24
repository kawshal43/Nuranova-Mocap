"""Markerless 2D AI pose estimation adapters (RTMLib RTMW-133, HALPE-26, MediaPipe Pose-33).

Maps all backends into Nuranova's canonical 26-keypoint body+foot observation convention
plus optional full 133-keypoint wholebody outputs (hands + face).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from engine.ai_tracking.runtime import HardwareDiagnostics, detect_hardware
from engine.video.video_io import iter_video_frames

# Canonical 26 body+foot keypoint order (HALPE-26 compatible)
CANONICAL_BODY26_NAMES: List[str] = [
    "nose",            # 0
    "left_eye",        # 1
    "right_eye",       # 2
    "left_ear",        # 3
    "right_ear",       # 4
    "left_shoulder",   # 5
    "right_shoulder",  # 6
    "left_elbow",      # 7
    "right_elbow",     # 8
    "left_wrist",      # 9
    "right_wrist",     # 10
    "left_hip",        # 11
    "right_hip",       # 12
    "left_knee",       # 13
    "right_knee",      # 14
    "left_ankle",      # 15
    "right_ankle",     # 16
    "head_top",        # 17
    "neck",            # 18
    "hips",            # 19
    "left_big_toe",    # 20
    "right_big_toe",   # 21
    "left_small_toe",  # 22
    "right_small_toe", # 23
    "left_heel",       # 24
    "right_heel",      # 25
]


@dataclass
class FramePose2D:
    frame_index: int
    pts_s: float
    body26_uvc: np.ndarray  # (26, 3) float32 [u_px, v_px, confidence]
    wholebody133_uvc: Optional[np.ndarray] = None  # (133, 3) float32 when RTMW active
    left_hand21_uvc: Optional[np.ndarray] = None  # (21, 3) float32
    right_hand21_uvc: Optional[np.ndarray] = None  # (21, 3) float32
    backend_name: str = ""
    person_detected: bool = False


@dataclass
class PoseBenchmarkStats:
    backend: str
    device: str
    frames_processed: int
    total_seconds: float
    measured_fps: float
    mean_confidence: float
    detection_rate: float


def _coco133_to_body26(kpts: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Convert COCO-WholeBody 133 keypoints to canonical Body26 layout."""
    out = np.zeros((26, 3), dtype=np.float32)
    # 0..16 match COCO-17 directly
    out[:17, :2] = kpts[:17, :2]
    out[:17, 2] = scores[:17]
    # Derived synthetic anatomical midpoints (neck, hips, head_top) with min parent confidence
    out[18, :2] = 0.5 * (kpts[5, :2] + kpts[6, :2])
    out[18, 2] = min(float(scores[5]), float(scores[6]))
    out[19, :2] = 0.5 * (kpts[11, :2] + kpts[12, :2])
    out[19, 2] = min(float(scores[11]), float(scores[12]))
    ear_mid = 0.5 * (kpts[3, :2] + kpts[4, :2])
    out[17, :2] = ear_mid + 0.35 * (ear_mid - out[18, :2])
    out[17, 2] = min(float(scores[3]), float(scores[4]))
    # Foot keypoints in COCO-WholeBody: 17=L_big_toe, 18=L_small_toe, 19=L_heel, 20=R_big_toe, 21=R_small_toe, 22=R_heel
    if kpts.shape[0] >= 23:
        mapping = {20: 17, 22: 18, 24: 19, 21: 20, 23: 21, 25: 22}
        for dst, src in mapping.items():
            out[dst, :2] = kpts[src, :2]
            out[dst, 2] = scores[src]
    return out


def _mediapipe33_to_body26(landmarks, width: int, height: int) -> np.ndarray:
    """Map MediaPipe Pose 33 landmarks to canonical Body26 pixels."""
    out = np.zeros((26, 3), dtype=np.float32)
    mp_map = {
        0: 0,   # nose
        1: 2,   # left_eye
        2: 5,   # right_eye
        3: 7,   # left_ear
        4: 8,   # right_ear
        5: 11,  # left_shoulder
        6: 12,  # right_shoulder
        7: 13,  # left_elbow
        8: 14,  # right_elbow
        9: 15,  # left_wrist
        10: 16, # right_wrist
        11: 23, # left_hip
        12: 24, # right_hip
        13: 25, # left_knee
        14: 26, # right_knee
        15: 27, # left_ankle
        16: 28, # right_ankle
        20: 31, # left_foot_index -> left_big_toe
        21: 32, # right_foot_index -> right_big_toe
        24: 29, # left_heel
        25: 30, # right_heel
    }
    for dst, src in mp_map.items():
        lm = landmarks[src]
        out[dst, 0] = float(lm.x * width)
        out[dst, 1] = float(lm.y * height)
        out[dst, 2] = float(np.clip(getattr(lm, "visibility", 0.5), 0.0, 1.0))

    # Synthesize neck, hips, head_top, small toes
    out[18, :2] = 0.5 * (out[5, :2] + out[6, :2])
    out[18, 2] = min(out[5, 2], out[6, 2])
    out[19, :2] = 0.5 * (out[11, :2] + out[12, :2])
    out[19, 2] = min(out[11, 2], out[12, 2])
    ear_mid = 0.5 * (out[3, :2] + out[4, :2])
    out[17, :2] = ear_mid + 0.35 * (ear_mid - out[18, :2])
    out[17, 2] = min(out[3, 2], out[4, 2])
    out[22] = out[20]
    out[23] = out[21]
    return out


class AIPoseTracker2D:
    """Replaceable 2D AI pose estimation engine with safe CUDA/CPU selection."""

    def __init__(
        self,
        backend: str = "rtmlib_wholebody",  # "rtmlib_wholebody", "rtmlib_halpe26", "mediapipe_pose"
        mode: str = "balanced",  # "lightweight", "balanced", "performance"
        prefer_gpu: bool = True,
    ) -> None:
        self.backend = backend
        self.mode = mode
        self.hw: HardwareDiagnostics = detect_hardware(prefer_gpu=prefer_gpu)
        self.device = self.hw.selected_device
        self._model = None
        self.model_version_tag = f"{backend}:{mode}:{self.device}"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if self.backend == "rtmlib_wholebody":
            from rtmlib import Wholebody

            self._model = Wholebody(
                to_openpose=False,
                mode=self.mode,
                backend="onnxruntime",
                device=self.device,
            )
        elif self.backend == "rtmlib_halpe26":
            from rtmlib import BodyWithFeet

            self._model = BodyWithFeet(
                to_openpose=False,
                mode=self.mode,
                backend="onnxruntime",
                device=self.device,
            )
        elif self.backend == "mediapipe_pose":
            import mediapipe as mp

            complexity = {"lightweight": 0, "balanced": 1, "performance": 2}.get(self.mode, 1)
            self._model = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=complexity,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4,
            )
        else:
            raise ValueError(f"Unsupported AI pose backend: {self.backend}")

    def infer_frame(self, frame_bgr: np.ndarray, frame_index: int = 0, pts_s: float = 0.0) -> FramePose2D:
        """Run actual AI model inference on a single BGR video frame."""
        self._ensure_loaded()
        h, w = frame_bgr.shape[:2]

        if self.backend == "mediapipe_pose":
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = self._model.process(rgb)
            if not res.pose_landmarks:
                return FramePose2D(
                    frame_index=frame_index,
                    pts_s=pts_s,
                    body26_uvc=np.zeros((26, 3), dtype=np.float32),
                    backend_name=self.model_version_tag,
                    person_detected=False,
                )
            body26 = _mediapipe33_to_body26(res.pose_landmarks.landmark, w, h)
            return FramePose2D(
                frame_index=frame_index,
                pts_s=pts_s,
                body26_uvc=body26,
                backend_name=self.model_version_tag,
                person_detected=True,
            )

        # RTMLib backends (Wholebody 133 or BodyWithFeet 26)
        keypoints, scores = self._model(frame_bgr)
        if keypoints is None or len(keypoints) == 0:
            return FramePose2D(
                frame_index=frame_index,
                pts_s=pts_s,
                body26_uvc=np.zeros((26, 3), dtype=np.float32),
                backend_name=self.model_version_tag,
                person_detected=False,
            )

        # Select primary actor (highest mean keypoint score)
        best_idx = int(np.argmax(np.mean(scores, axis=1)))
        kpts = np.asarray(keypoints[best_idx], dtype=np.float32)
        sc = np.asarray(scores[best_idx], dtype=np.float32)
        # Normalize scores to [0, 1] safely (RTMLib SimCC scores can occasionally exceed 1.0 slightly)
        sc_norm = np.clip(sc, 0.0, 1.0)

        if self.backend == "rtmlib_wholebody" and kpts.shape[0] >= 133:
            body26 = _coco133_to_body26(kpts, sc_norm)
            wb = np.concatenate([kpts[:, :2], sc_norm[:, None]], axis=1).astype(np.float32)
            # COCO-WholeBody hand slices: left hand = 91..111 (21 pts), right hand = 112..132 (21 pts)
            l_hand = wb[91:112].copy()
            r_hand = wb[112:133].copy()
            return FramePose2D(
                frame_index=frame_index,
                pts_s=pts_s,
                body26_uvc=body26,
                wholebody133_uvc=wb,
                left_hand21_uvc=l_hand,
                right_hand21_uvc=r_hand,
                backend_name=self.model_version_tag,
                person_detected=True,
            )

        # HALPE-26 directly matches canonical 26 order
        body26 = np.zeros((26, 3), dtype=np.float32)
        n_copy = min(26, kpts.shape[0])
        body26[:n_copy, :2] = kpts[:n_copy, :2]
        body26[:n_copy, 2] = sc_norm[:n_copy]
        return FramePose2D(
            frame_index=frame_index,
            pts_s=pts_s,
            body26_uvc=body26,
            backend_name=self.model_version_tag,
            person_detected=True,
        )

    def process_video(
        self,
        video_path: str | Path,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        step: int = 1,
        resize_wh: Optional[Tuple[int, int]] = None,
        progress_cb: Optional[Callable[[int], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Tuple[np.ndarray, List[float], PoseBenchmarkStats]:
        """Stream-process a video file into (T, 26, 3) observations without loading video into RAM."""
        t0 = time.perf_counter()
        results: List[np.ndarray] = []
        pts_list: List[float] = []
        detected_count = 0
        confs: List[float] = []

        for idx, pts_s, bgr in iter_video_frames(
            video_path, start_frame=start_frame, end_frame=end_frame, step=step, resize_wh=resize_wh
        ):
            if cancel_check is not None and cancel_check():
                break
            fp = self.infer_frame(bgr, frame_index=idx, pts_s=pts_s)
            results.append(fp.body26_uvc)
            pts_list.append(pts_s)
            if fp.person_detected:
                detected_count += 1
                confs.append(float(np.mean(fp.body26_uvc[:, 2])))
            if progress_cb is not None:
                progress_cb(len(results))

        elapsed = max(time.perf_counter() - t0, 1e-6)
        n = len(results)
        arr = np.stack(results, axis=0) if results else np.zeros((0, 26, 3), dtype=np.float32)
        stats = PoseBenchmarkStats(
            backend=self.model_version_tag,
            device=self.device,
            frames_processed=n,
            total_seconds=elapsed,
            measured_fps=n / elapsed,
            mean_confidence=float(np.mean(confs)) if confs else 0.0,
            detection_rate=detected_count / max(n, 1),
        )
        return arr, pts_list, stats

    def close(self) -> None:
        if self.backend == "mediapipe_pose" and self._model is not None:
            self._model.close()
            self._model = None
