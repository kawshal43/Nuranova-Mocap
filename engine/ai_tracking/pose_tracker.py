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


def calibrate_rtm_scores(scores: np.ndarray) -> np.ndarray:
    """Map raw RTMLib SimCC peak scores cleanly into calibrated [0, 1] visibility probabilities.

    RTMW-x SimCC outputs unnormalized peak logits in ~[1.5 .. 8.0] (occluded guesses ~2.2-3.2,
    clean visible joints ~4.5-7.2). Naive `np.clip(scores, 0, 1)` saturated 100% of occluded
    joints to `1.000`, corrupting calibration and triangulation!
    """
    sc = np.asarray(scores, dtype=np.float32)
    if sc.size == 0:
        return sc
    # Distinguish unnormalized SimCC logit scale (max > 1.5) from standard [0, 1] probability models
    if float(np.max(sc)) > 1.5:
        # Linear-clamped calibration grounded on empirical RTMW-x percentiles:
        # <=2.3 -> 0.05 (occluded), 3.3 -> 0.24, 4.9 (p50) -> 0.62, >=6.5 -> 1.00
        return np.clip((sc - 2.3) / 4.2, 0.05, 1.0).astype(np.float32)
    return np.clip(sc, 0.0, 1.0).astype(np.float32)


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
        self.fallback_reason: Optional[str] = None
        self.model_version_tag = f"{backend}:{mode}:{self.device}"

    def _tune_cuda_sessions(self) -> None:
        """Re-configure RTMLib ONNX Runtime CUDA sessions for max RTX Tensor Core throughput."""
        if self.device != "cuda" or self._model is None:
            return
        try:
            import onnxruntime as ort
        except Exception:
            return
        for attr, algo in (("pose_model", "HEURISTIC"), ("det_model", "HEURISTIC")):
            sub = getattr(self._model, attr, None)
            onnx_path = getattr(sub, "onnx_model", None)
            if sub is None or not onnx_path or not Path(str(onnx_path)).exists():
                continue
            try:
                so = ort.SessionOptions()
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                cuda_opts = {
                    "device_id": 0,
                    "cudnn_conv_algo_search": algo,
                    "cudnn_conv_use_max_workspace": "1",
                    "do_copy_in_default_stream": "1",
                }
                new_sess = ort.InferenceSession(
                    str(onnx_path),
                    sess_options=so,
                    providers=[("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"],
                )
                if "CUDAExecutionProvider" in new_sess.get_providers():
                    sub.session = new_sess
            except Exception:
                pass

    def active_providers(self) -> List[str]:
        """Return actual ONNX Runtime execution providers active on the loaded model."""
        if self._model is None:
            return []
        provs: List[str] = []
        for attr in ("det_model", "pose_model"):
            sub = getattr(self._model, attr, None)
            sess = getattr(sub, "session", None)
            if sess is not None and hasattr(sess, "get_providers"):
                for p in sess.get_providers():
                    if p not in provs:
                        provs.append(str(p))
        return provs

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if self.backend == "rtmlib_wholebody":
            try:
                from rtmlib import Wholebody

                self._model = Wholebody(
                    to_openpose=False,
                    mode=self.mode,
                    backend="onnxruntime",
                    device=self.device,
                )
                self._tune_cuda_sessions()
            except Exception as exc:
                # Graceful offline fallback to MediaPipe Pose if ONNX weights cannot be downloaded
                self.fallback_reason = f"RTMW-x load failed ({exc}); fell back to CPU MediaPipe Pose."
                self.backend = "mediapipe_pose"
                self.model_version_tag = f"mediapipe_pose:{self.mode}:fallback"
        elif self.backend == "rtmlib_halpe26":
            try:
                from rtmlib import BodyWithFeet

                self._model = BodyWithFeet(
                    to_openpose=False,
                    mode=self.mode,
                    backend="onnxruntime",
                    device=self.device,
                )
                self._tune_cuda_sessions()
            except Exception as exc:
                self.fallback_reason = f"HALPE-26 load failed ({exc}); fell back to CPU MediaPipe Pose."
                self.backend = "mediapipe_pose"
                self.model_version_tag = f"mediapipe_pose:{self.mode}:fallback"

        if self.backend == "mediapipe_pose":
            import mediapipe as mp

            self.device = "cpu"
            self.model_version_tag = f"mediapipe_pose:{self.mode}:cpu" + (":fallback" if self.fallback_reason else "")
            complexity = {"lightweight": 0, "balanced": 1, "performance": 2}.get(self.mode, 1)
            self._model = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=complexity,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4,
            )
        elif self._model is None:
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
        sc_norm = calibrate_rtm_scores(sc)

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

    def process_video_wholebody(
        self,
        video_path: str | Path,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        step: int = 1,
        resize_wh: Optional[Tuple[int, int]] = None,
        progress_cb: Optional[Callable[[int], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], List[float], PoseBenchmarkStats]:
        """Single-pass offline video tracking returning Body26 PLUS optional Left/Right 21-keypoint hands (RTMW-133)."""
        t0 = time.perf_counter()
        body_list: List[np.ndarray] = []
        lh_list: List[np.ndarray] = []
        rh_list: List[np.ndarray] = []
        any_hands = False
        pts_list: List[float] = []
        detected_count = 0
        confs: List[float] = []

        for idx, pts_s, bgr in iter_video_frames(
            video_path, start_frame=start_frame, end_frame=end_frame, step=step, resize_wh=resize_wh
        ):
            if cancel_check is not None and cancel_check():
                break
            fp = self.infer_frame(bgr, frame_index=idx, pts_s=pts_s)
            body_list.append(fp.body26_uvc)
            if fp.left_hand21_uvc is not None and fp.right_hand21_uvc is not None:
                any_hands = True
                lh_list.append(fp.left_hand21_uvc)
                rh_list.append(fp.right_hand21_uvc)
            else:
                lh_list.append(np.zeros((21, 3), dtype=np.float32))
                rh_list.append(np.zeros((21, 3), dtype=np.float32))
            pts_list.append(pts_s)
            if fp.person_detected:
                detected_count += 1
                confs.append(float(np.mean(fp.body26_uvc[:, 2])))
            if progress_cb is not None:
                progress_cb(len(body_list))

        elapsed = max(time.perf_counter() - t0, 1e-6)
        n = len(body_list)
        body_arr = np.stack(body_list, axis=0) if body_list else np.zeros((0, 26, 3), dtype=np.float32)
        lh_arr = np.stack(lh_list, axis=0) if (any_hands and lh_list) else None
        rh_arr = np.stack(rh_list, axis=0) if (any_hands and rh_list) else None
        stats = PoseBenchmarkStats(
            backend=self.model_version_tag,
            device=self.device,
            frames_processed=n,
            total_seconds=elapsed,
            measured_fps=n / elapsed,
            mean_confidence=float(np.mean(confs)) if confs else 0.0,
            detection_rate=detected_count / max(n, 1),
        )
        return body_arr, lh_arr, rh_arr, pts_list, stats

    def close(self) -> None:
        if self.backend == "mediapipe_pose" and self._model is not None:
            self._model.close()
            self._model = None
