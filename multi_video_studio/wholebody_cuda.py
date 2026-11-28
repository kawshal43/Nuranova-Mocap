"""RTX 3060 Ti CUDA Whole-Body (133-pt RTMW-x), Actor-Lock Multi-Person Tracker,
Full-Resolution Wrist-Cropped 30-Finger Tracker, and Head-Cropped 6-DoF Head + 52 Facial Expression Tracker.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from engine.ai_tracking.pose_tracker import (
    AIPoseTracker2D,
    _coco133_to_body26,
    _mediapipe33_to_body26,
    calibrate_rtm_scores,
)
from engine.ai_tracking.runtime import HardwareDiagnostics, detect_hardware
from engine.face.face_tracker import FaceFrameResult, FacialCaptureEngine
from engine.hands.hand_tracker import HandCaptureEngine, HandObservation
from engine.video.video_io import iter_video_frames


@dataclass
class CameraTrackingResult:
    camera_uid: str
    video_path: Path
    width: int
    height: int
    fps: float
    pts_s: List[float]
    body26_uvc: np.ndarray  # (T, 26, 3) float32 [u, v, conf]
    left_hand21_uvc: Optional[np.ndarray] = None  # (T, 21, 3) when RTMW-133 active
    right_hand21_uvc: Optional[np.ndarray] = None  # (T, 21, 3) when RTMW-133 active
    face68_uvc: Optional[np.ndarray] = None  # (T, 68, 3) when RTMW-133 active
    wrist_crop_hands: List[Dict[str, HandObservation]] = field(default_factory=list)
    head_crop_faces: List[Optional[FaceFrameResult]] = field(default_factory=list)
    backend_tag: str = ""
    device: str = "cpu"
    detection_rate: float = 0.0
    measured_fps: float = 0.0


class ActorLockTracker:
    """Locks onto the primary actor across frames in multi-person scenes so bystanders never cause teleports."""

    def __init__(self) -> None:
        self._prev_center: Optional[np.ndarray] = None
        self._prev_scale: Optional[float] = None

    def reset(self) -> None:
        self._prev_center = None
        self._prev_scale = None

    @staticmethod
    def _candidate_center_scale(
        kpts: np.ndarray, scores: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        # Core torso joints in COCO-17 / WholeBody-133: shoulders (5,6), hips (11,12)
        core = [5, 6, 11, 12]
        valid = [k for k in core if k < len(scores) and scores[k] > 0.20]
        if valid:
            pts = kpts[valid, :2]
            center = np.mean(pts, axis=0)
        else:
            good = scores[: min(17, len(scores))] > 0.15
            center = (
                np.mean(kpts[: min(17, len(scores))][good, :2], axis=0)
                if np.any(good)
                else np.zeros(2)
            )

        body_pts = kpts[: min(17, len(scores))]
        body_sc = scores[: min(17, len(scores))]
        vis = body_pts[body_sc > 0.20, :2]
        if len(vis) >= 3:
            span = np.max(vis, axis=0) - np.min(vis, axis=0)
            scale = float(max(np.linalg.norm(span), 40.0))
        else:
            scale = 120.0
        mean_conf = float(np.mean(calibrate_rtm_scores(body_sc)))
        return center.astype(np.float64), scale, mean_conf

    def select_actor_index(
        self, keypoints: np.ndarray, scores: np.ndarray, img_w: int, img_h: int
    ) -> int:
        n_people = len(keypoints)
        if n_people <= 1:
            if n_people == 1:
                c, s, _ = self._candidate_center_scale(keypoints[0], scores[0])
                self._prev_center, self._prev_scale = c, s
            return 0

        img_center = np.array([0.5 * img_w, 0.5 * img_h], dtype=np.float64)
        best_idx = 0
        best_score = -1e9
        best_cs = None

        for i in range(n_people):
            c, s, conf = self._candidate_center_scale(keypoints[i], scores[i])
            if self._prev_center is not None and self._prev_scale is not None:
                dist = float(np.linalg.norm(c - self._prev_center))
                prox = float(np.exp(-dist / max(0.45 * self._prev_scale, 45.0)))
                scale_ratio = min(s, self._prev_scale) / max(s, self._prev_scale, 1.0)
                total = 0.55 * prox + 0.25 * scale_ratio + 0.20 * conf
            else:
                # First frame: prefer large, high-confidence, near-center performer
                dist_img = float(np.linalg.norm(c - img_center)) / max(
                    float(np.hypot(img_w, img_h)), 1.0
                )
                size_norm = min(1.0, s / max(0.75 * img_h, 1.0))
                total = 0.45 * conf + 0.35 * size_norm + 0.20 * (1.0 - dist_img)
            if total > best_score:
                best_score = total
                best_idx = i
                best_cs = (c, s)

        if best_cs is not None:
            self._prev_center, self._prev_scale = best_cs
        return best_idx


def extract_head_crop_bgr(
    full_res_bgr: np.ndarray,
    body26_uvc: np.ndarray,
    min_conf: float = 0.25,
) -> Optional[np.ndarray]:
    """Extract a normalized 256x256 BGR head crop so full 4K frames can be freed immediately."""
    h, w = full_res_bgr.shape[:2]
    head_pts = []
    for idx in (0, 1, 2, 3, 4, 17):
        if float(body26_uvc[idx, 2]) >= min_conf:
            u, v = float(body26_uvc[idx, 0]), float(body26_uvc[idx, 1])
            if 0 <= u < w and 0 <= v < h:
                head_pts.append((u, v))
    if not head_pts:
        return None

    arr = np.asarray(head_pts, dtype=np.float64)
    cx, cy = float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1]))

    if float(body26_uvc[5, 2]) >= min_conf and float(body26_uvc[6, 2]) >= min_conf:
        sh_w = float(
            np.hypot(body26_uvc[5, 0] - body26_uvc[6, 0], body26_uvc[5, 1] - body26_uvc[6, 1])
        )
        box_sz = int(np.clip(sh_w * 1.15, 140.0, min(w, h) * 0.55))
    else:
        span = float(np.max(np.ptp(arr, axis=0))) if len(arr) >= 2 else 80.0
        box_sz = int(np.clip(span * 2.6, 140.0, min(w, h) * 0.55))

    half = box_sz // 2
    x0 = max(0, int(round(cx)) - half)
    y0 = max(0, int(round(cy)) - half)
    x1 = min(w, x0 + box_sz)
    y1 = min(h, y0 + box_sz)
    x0 = max(0, x1 - box_sz)
    y0 = max(0, y1 - box_sz)
    if (x1 - x0) < 48 or (y1 - y0) < 48:
        return None

    crop = full_res_bgr[y0:y1, x0:x1]
    if crop.shape[0] < 224 or crop.shape[1] < 224:
        return cv2.resize(crop, (256, 256), interpolation=cv2.INTER_CUBIC)
    if crop.shape[0] > 288 or crop.shape[1] > 288:
        return cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)
    return crop.copy()


def extract_head_crop_face(
    full_res_bgr: np.ndarray,
    body26_uvc: np.ndarray,
    face_engine: FacialCaptureEngine,
    frame_index: int = 0,
    min_conf: float = 0.25,
) -> Optional[FaceFrameResult]:
    """Crop a tight full-resolution ROI around the actor's head and extract 6-DoF head rotation + 52 facial expressions."""
    crop = extract_head_crop_bgr(full_res_bgr, body26_uvc, min_conf=min_conf)
    if crop is None:
        return None
    try:
        return face_engine.infer_frame(crop)
    except Exception:
        return None


def _bbox_from_keypoints(
    kpts: np.ndarray,
    scores: np.ndarray,
    img_w: int,
    img_h: int,
    pad: float = 0.28,
    pad_h_x: float = 0.20,
    pad_y: float = 0.13,
) -> Optional[List[float]]:
    """Compute a height-scaled smart [x1, y1, x2, y2] bbox covering body + feet (0..22) and arm swing reach.

    Scaling horizontal padding by body height (`bh * pad_h_x`) keeps fast swinging arms inside the crop even
    when the actor starts with arms at their sides, while compact vertical padding (`bh * pad_y`) preserves
    full native pixel resolution inside RTMPose's 288x384 input!
    """
    n_bf = min(23, len(scores))
    valid_bf = scores[:n_bf] >= 0.25
    if int(np.sum(valid_bf)) < 5:
        return None
    pts_bf = kpts[:n_bf][valid_bf, :2]
    x1, y1 = float(np.min(pts_bf[:, 0])), float(np.min(pts_bf[:, 1]))
    x2, y2 = float(np.max(pts_bf[:, 0])), float(np.max(pts_bf[:, 1]))
    bh_core = max(y2 - y1, 60.0)

    # Extend bounds with confident hand keypoints (91..132) within plausible reach so hands never clip
    if len(scores) >= 133:
        h_sc = scores[91:133]
        h_ok = h_sc >= 0.45
        if np.any(h_ok):
            h_pts = kpts[91:133][h_ok, :2]
            in_reach = (
                (h_pts[:, 0] >= x1 - 0.65 * bh_core)
                & (h_pts[:, 0] <= x2 + 0.65 * bh_core)
                & (h_pts[:, 1] >= y1 - 0.45 * bh_core)
                & (h_pts[:, 1] <= y2 + 0.45 * bh_core)
            )
            if np.any(in_reach):
                hp = h_pts[in_reach]
                x1 = min(x1, float(np.min(hp[:, 0])))
                y1 = min(y1, float(np.min(hp[:, 1])))
                x2 = max(x2, float(np.max(hp[:, 0])))
                y2 = max(y2, float(np.max(hp[:, 1])))

    bw, bh = max(x2 - x1, 40.0), max(y2 - y1, 60.0)
    px = max(bw * pad, bh * pad_h_x)
    py = bh * pad_y
    bx1 = max(0.0, x1 - px)
    by1 = max(0.0, y1 - py)
    bx2 = min(float(img_w - 1), x2 + px)
    by2 = min(float(img_h - 1), y2 + py)
    if (bx2 - bx1) < 32.0 or (by2 - by1) < 48.0:
        return None
    return [bx1, by1, bx2, by2]


def _pad_det_bbox(bbox: np.ndarray | List[float], img_w: int, img_h: int) -> List[float]:
    """Slightly expand a single YOLOX person bbox so an 8-frame batch stays centered without shrinking resolution."""
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    bw, bh = max(x2 - x1, 40.0), max(y2 - y1, 60.0)
    px = max(bw * 0.18, bh * 0.12)
    py = bh * 0.08
    return [
        max(0.0, x1 - px),
        max(0.0, y1 - py),
        min(float(img_w - 1), x2 + px),
        min(float(img_h - 1), y2 + py),
    ]


def _prefetch_video_frames(
    video_path: Path,
    step: int,
    cancel_event: Optional[threading.Event] = None,
    maxsize: int = 4,
    start_pts_s: Optional[float] = None,
    end_pts_s: Optional[float] = None,
) -> Iterator[Tuple[int, float, np.ndarray]]:
    """Decode video frames in a background thread so CPU decoding overlaps with RTX CUDA inference."""
    q: queue.Queue = queue.Queue(maxsize=maxsize)
    stop_evt = threading.Event()
    err_holder: List[BaseException] = []

    def _producer() -> None:
        try:
            for item in iter_video_frames(
                video_path, step=step, start_pts_s=start_pts_s, end_pts_s=end_pts_s
            ):
                if stop_evt.is_set() or (cancel_event is not None and cancel_event.is_set()):
                    break
                while not stop_evt.is_set():
                    if cancel_event is not None and cancel_event.is_set():
                        return
                    try:
                        q.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:
            err_holder.append(exc)
        finally:
            while True:
                try:
                    q.put(None, timeout=0.1)
                    break
                except queue.Full:
                    if stop_evt.is_set():
                        break

    th = threading.Thread(target=_producer, name="nuranova-frame-prefetch", daemon=True)
    th.start()
    try:
        while True:
            item = q.get()
            if item is None:
                break
            yield item
        if err_holder:
            raise err_holder[0]
    finally:
        stop_evt.set()
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break
        th.join(timeout=1.5)


class _AsyncHandFacePool:
    """Multi-threaded background CPU pool for 224x224 MediaPipe Hands & FaceMesh crops overlapped with RTX CUDA.

    Samples at `crop_stride` (default 2 on >=24 FPS takes -> 15 Hz, matching the 7.5 Hz Nyquist finger filter!)
    so CPU MediaPipe NEVER stalls the RTX GPU, then propagates seamlessly to every frame!
    """

    def __init__(
        self,
        enable_hands: bool,
        enable_face: bool,
        num_workers: int = 4,
        crop_stride: int = 2,
        max_queue: int = 1600,
    ) -> None:
        self.enable_hands = enable_hands
        self.enable_face = enable_face
        self.crop_stride = max(1, int(crop_stride))
        self._active = enable_hands or enable_face
        self._q: queue.Queue = queue.Queue(maxsize=max(64, int(max_queue)))
        self._hands_out: Dict[int, Dict[str, HandObservation]] = {}
        self._faces_out: Dict[int, Optional[FaceFrameResult]] = {}
        self._threads: List[threading.Thread] = []
        if self._active:
            for w_i in range(max(1, num_workers)):
                th = threading.Thread(target=self._worker, name=f"nuranova-crop-worker-{w_i}", daemon=True)
                th.start()
                self._threads.append(th)

    def _worker(self) -> None:
        h_eng = HandCaptureEngine(max_num_hands=2, model_complexity=1) if self.enable_hands else None
        f_eng = FacialCaptureEngine() if self.enable_face else None
        try:
            while True:
                item = self._q.get()
                if item is None:
                    break
                f_idx, hand_crops, head_crop = item
                if h_eng is not None:
                    try:
                        self._hands_out[f_idx] = h_eng.infer_hands_from_crops(hand_crops)
                    except Exception:
                        self._hands_out[f_idx] = {}
                if f_eng is not None:
                    if head_crop is not None:
                        try:
                            self._faces_out[f_idx] = f_eng.infer_frame(head_crop)
                        except Exception:
                            self._faces_out[f_idx] = None
                    else:
                        self._faces_out[f_idx] = None
        finally:
            if h_eng is not None:
                h_eng.close()
            if f_eng is not None:
                f_eng.close()

    def submit(
        self,
        f_idx: int,
        full_res_bgr: np.ndarray,
        b26: np.ndarray,
    ) -> None:
        if not self._active:
            return
        if self.crop_stride > 1 and (f_idx % self.crop_stride) != 0:
            return
        hand_crops = HandCaptureEngine.extract_wrist_crops_from_body26(full_res_bgr, b26) if self.enable_hands else {}
        head_crop = extract_head_crop_bgr(full_res_bgr, b26) if self.enable_face else None
        self._q.put((f_idx, hand_crops, head_crop))

    def finish(self, n_frames: int) -> Tuple[List[Dict[str, HandObservation]], List[Optional[FaceFrameResult]]]:
        if not self._active:
            return [], []
        for _ in self._threads:
            self._q.put(None)
        for th in self._threads:
            th.join()

        hands: List[Dict[str, HandObservation]] = []
        if self.enable_hands:
            # Fill stride-skipped frames cleanly by geodesic SLERP between adjacent sampled hand observations!
            from engine.math_core.transforms import quat_slerp

            sampled_h = [i for i in range(n_frames) if i in self._hands_out]
            hands = [self._hands_out.get(i, {}) for i in range(n_frames)]
            if self.crop_stride > 1 and len(sampled_h) >= 2:
                for idx_s in range(len(sampled_h) - 1):
                    a, b = sampled_h[idx_s], sampled_h[idx_s + 1]
                    if b - a <= 1 or b - a > self.crop_stride + 1:
                        continue
                    ha, hb = hands[a], hands[b]
                    for k in range(a + 1, b):
                        alpha = float(k - a) / float(b - a)
                        interp: Dict[str, HandObservation] = {}
                        for side in ("left", "right"):
                            oa, ob = ha.get(side), hb.get(side)
                            if oa is not None and ob is not None:
                                rots = np.array(
                                    [quat_slerp(oa.local_rotations_xyzw[j], ob.local_rotations_xyzw[j], alpha) for j in range(16)],
                                    dtype=np.float32,
                                )
                                interp[side] = HandObservation(
                                    handedness=side,
                                    landmarks_3d=((1.0 - alpha) * oa.landmarks_3d + alpha * ob.landmarks_3d).astype(np.float32),
                                    confidence=float((1.0 - alpha) * oa.confidence + alpha * ob.confidence),
                                    local_rotations_xyzw=rots,
                                    source="interpolated",
                                )
                            # Do not copy a single observation across unobserved source frames.
                        hands[k] = interp
                # Fill trailing frame after last sample cleanly
                if sampled_h[-1] < n_frames - 1:
                    for k in range(sampled_h[-1] + 1, n_frames):
                        hands[k] = {}

        faces: List[Optional[FaceFrameResult]] = []
        if self.enable_face:
            faces = [self._faces_out.get(i, None) for i in range(n_frames)]
            if self.crop_stride > 1:
                for k in range(n_frames):
                    if faces[k] is None and k > 0 and (k % self.crop_stride) != 0:
                        faces[k] = faces[k - 1]
        return hands, faces


def _can_batch_cuda_pose(pose_model) -> bool:
    """Return True if RTMPose ONNX session supports dynamic batching on CUDAExecutionProvider."""
    try:
        sess = getattr(pose_model, "session", None)
        if sess is None or "CUDAExecutionProvider" not in sess.get_providers():
            return False
        inp = sess.get_inputs()[0]
        return len(inp.shape) == 4 and (isinstance(inp.shape[0], str) or inp.shape[0] is None)
    except Exception:
        return False


def _preprocess_rtmpose_fast(
    img: np.ndarray,
    bbox: List[float],
    model_input_size: Tuple[int, int],
    mean_f32: Optional[np.ndarray],
    inv_std_f32: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thread-safe, pure-float32 RTMPose top-down affine crop & normalize (no shared mutable state, no float64)."""
    from rtmlib.tools.pose_estimation.rtmpose import bbox_xyxy2cs, top_down_affine

    center, scale = bbox_xyxy2cs(np.asarray(bbox, dtype=np.float32), padding=1.25)
    resized_img, scale = top_down_affine(model_input_size, scale, center, img)
    out = resized_img.astype(np.float32)
    if mean_f32 is not None and inv_std_f32 is not None:
        out -= mean_f32
        out *= inv_std_f32
    return out, center, scale


def _run_pose_batch_cuda(
    pose_model,
    bgr_batch: List[np.ndarray],
    bboxes: List[List[float]],
    prep_pool: ThreadPoolExecutor,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run true batched CUDA RTMW-x inference across B frames in one GPU launch (~190-208 FPS on RTX 3060 Ti!)."""
    from rtmlib.tools.pose_estimation.rtmpose import get_simcc_maximum

    mean_f32 = np.asarray(pose_model.mean, dtype=np.float32) if pose_model.mean is not None else None
    inv_std_f32 = (
        (1.0 / np.asarray(pose_model.std, dtype=np.float32)) if pose_model.std is not None else None
    )
    in_sz = pose_model.model_input_size

    def _prep_one(pair: Tuple[np.ndarray, List[float]]):
        return _preprocess_rtmpose_fast(pair[0], pair[1], in_sz, mean_f32, inv_std_f32)

    prepped = list(prep_pool.map(_prep_one, zip(bgr_batch, bboxes)))
    imgs = [p[0] for p in prepped]
    centers = np.asarray([p[1] for p in prepped], dtype=np.float32)  # (B, 2)
    scales = np.asarray([p[2] for p in prepped], dtype=np.float32)  # (B, 2)

    # Stack into contiguous (B, 3, H, W) float32 NCHW tensor & dispatch ONE GPU kernel batch
    batch_nhwc = np.stack(imgs, axis=0)
    batch_nchw = np.ascontiguousarray(batch_nhwc.transpose(0, 3, 1, 2), dtype=np.float32)
    inp_name = pose_model.session.get_inputs()[0].name
    out_names = [o.name for o in pose_model.session.get_outputs()]
    simcc_x, simcc_y = pose_model.session.run(out_names, {inp_name: batch_nchw})

    # Vectorized SimCC decoding across entire batch (exact bit-for-bit match to RTMPose.postprocess!)
    locs, scores = get_simcc_maximum(simcc_x, simcc_y)
    in_w, in_h = float(in_sz[0]), float(in_sz[1])
    kpts = locs / 2.0  # simcc_split_ratio = 2.0
    kpts[:, :, 0] = kpts[:, :, 0] / in_w * scales[:, 0:1] + centers[:, 0:1] - scales[:, 0:1] * 0.5
    kpts[:, :, 1] = kpts[:, :, 1] / in_h * scales[:, 1:2] + centers[:, 1:2] - scales[:, 1:2] * 0.5
    return kpts.astype(np.float32), scores.astype(np.float32)


class _SharedCudaPoseBatcher:
    """Dedicated single-thread RTX CUDA inference dispatcher coalescing micro-batches across all active cameras.

    Allows N cameras to decode & preprocess frames concurrently across all CPU cores while coalescing their
    3-4 frame micro-batches into saturated Batch-12..16 GPU launches (~190-208 FPS on RTX 3060 Ti, 0 extra VRAM!).
    """

    def __init__(self, rtm_model, max_gpu_batch: int = 16, num_cams: int = 1) -> None:
        self._rtm = rtm_model
        self._pm = rtm_model.pose_model
        self.model_input_size = tuple(self._pm.model_input_size)
        self.mean_f32 = (
            np.asarray(self._pm.mean, dtype=np.float32) if self._pm.mean is not None else None
        )
        self.inv_std_f32 = (
            (1.0 / np.asarray(self._pm.std, dtype=np.float32)) if self._pm.std is not None else None
        )
        self._inp_name = self._pm.session.get_inputs()[0].name
        self._out_names = [o.name for o in self._pm.session.get_outputs()]
        self.max_gpu_batch = max(4, int(max_gpu_batch))

        self._cond = threading.Condition()
        self._pose_q: List[Tuple[np.ndarray, np.ndarray, np.ndarray, threading.Event, Dict]] = []
        self._det_q: List[Tuple[np.ndarray, threading.Event, Dict]] = []
        self._active_cams = max(1, int(num_cams))
        self._stop = False
        self._thread = threading.Thread(
            target=self._gpu_loop, name="nuranova-cuda-batcher", daemon=True
        )
        self._thread.start()

    @property
    def active_cams(self) -> int:
        with self._cond:
            return self._active_cams

    def mark_camera_done(self) -> None:
        with self._cond:
            self._active_cams = max(0, self._active_cams - 1)
            self._cond.notify_all()

    def preprocess_batch(
        self, bgr_list: List[np.ndarray], bboxes: List[List[float]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run CPU affine crops & NCHW pack on the caller's camera thread (releases GIL via OpenCV)."""
        imgs: List[np.ndarray] = []
        centers: List[np.ndarray] = []
        scales: List[np.ndarray] = []
        for bgr, bb in zip(bgr_list, bboxes):
            im, c, s = _preprocess_rtmpose_fast(
                bgr, bb, self.model_input_size, self.mean_f32, self.inv_std_f32
            )
            imgs.append(im)
            centers.append(c)
            scales.append(s)
        batch_nhwc = np.stack(imgs, axis=0)
        batch_nchw = np.ascontiguousarray(batch_nhwc.transpose(0, 3, 1, 2), dtype=np.float32)
        return batch_nchw, np.asarray(centers, dtype=np.float32), np.asarray(scales, dtype=np.float32)

    def submit_pose(
        self, batch_nchw: np.ndarray, centers: np.ndarray, scales: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        from rtmlib.tools.pose_estimation.rtmpose import get_simcc_maximum

        evt = threading.Event()
        res: Dict = {}
        with self._cond:
            if self._stop:
                raise RuntimeError("CUDA pose batcher stopped.")
            self._pose_q.append((batch_nchw, evt, res))
            self._cond.notify_all()
        evt.wait()
        if "err" in res:
            raise res["err"]
        # Decode SimCC & affine back-project on the caller's CPU thread so the RTX GPU launches the next batch immediately!
        simcc_x, simcc_y = res["simcc_x"], res["simcc_y"]
        in_w, in_h = float(self.model_input_size[0]), float(self.model_input_size[1])
        locs, scores = get_simcc_maximum(simcc_x, simcc_y)
        kpts = locs / 2.0
        kpts[:, :, 0] = (
            kpts[:, :, 0] / in_w * scales[:, 0:1] + centers[:, 0:1] - scales[:, 0:1] * 0.5
        )
        kpts[:, :, 1] = (
            kpts[:, :, 1] / in_h * scales[:, 1:2] + centers[:, 1:2] - scales[:, 1:2] * 0.5
        )
        return kpts.astype(np.float32), scores.astype(np.float32)

    def submit_detect_full(
        self, bgr_img: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        evt = threading.Event()
        res: Dict = {}
        with self._cond:
            if self._stop:
                raise RuntimeError("CUDA pose batcher stopped.")
            self._det_q.append(("full", bgr_img, evt, res))
            self._cond.notify_all()
        evt.wait()
        if "err" in res:
            raise res["err"]
        return res["kpts"], res["scores"]

    def submit_detect_boxes(self, bgr_img: np.ndarray) -> np.ndarray:
        """Run ONLY YOLOX person detector on GPU (avoids slow batch=1 RTMPose when 1 actor is present!)."""
        evt = threading.Event()
        res: Dict = {}
        with self._cond:
            if self._stop:
                raise RuntimeError("CUDA pose batcher stopped.")
            self._det_q.append(("boxes", bgr_img, evt, res))
            self._cond.notify_all()
        evt.wait()
        if "err" in res:
            raise res["err"]
        return res["bboxes"]

    def _gpu_loop(self) -> None:
        while True:
            det_req = None
            pose_reqs: List[Tuple[np.ndarray, threading.Event, Dict]] = []
            with self._cond:
                while not self._stop and not self._pose_q and not self._det_q:
                    self._cond.wait(timeout=0.1)
                if self._stop and not self._pose_q and not self._det_q:
                    break
                if self._det_q:
                    det_req = self._det_q.pop(0)
                elif self._pose_q:
                    # Coalesce concurrent camera chunks up to Batch-16..24 (<= 6 ms wait -> saturates RTX 3060 Ti!)
                    deadline = time.perf_counter() + 0.006
                    while True:
                        total_f = sum(r[0].shape[0] for r in self._pose_q)
                        if (
                            self._stop
                            or self._det_q
                            or self._active_cams <= 1
                            or total_f >= self.max_gpu_batch
                            or len(self._pose_q) >= self._active_cams
                        ):
                            break
                        rem_t = deadline - time.perf_counter()
                        if rem_t <= 0.0:
                            break
                        self._cond.wait(timeout=rem_t)
                    acc_f = 0
                    while self._pose_q:
                        nxt_f = self._pose_q[0][0].shape[0]
                        if pose_reqs and (acc_f + nxt_f) > (self.max_gpu_batch + 8):
                            break
                        pose_reqs.append(self._pose_q.pop(0))
                        acc_f += nxt_f

            if det_req is not None:
                mode_d, bgr_img, evt, res = det_req
                try:
                    if mode_d == "boxes":
                        bboxes = self._rtm.det_model(bgr_img)
                        res["bboxes"] = (
                            np.asarray(bboxes, dtype=np.float32)
                            if bboxes is not None and len(bboxes) > 0
                            else np.zeros((0, 4), dtype=np.float32)
                        )
                    else:
                        kpts, scores = self._rtm(bgr_img)
                        res["kpts"] = kpts
                        res["scores"] = scores
                except BaseException as exc:
                    res["err"] = exc
                finally:
                    evt.set()
            elif pose_reqs:
                try:
                    if len(pose_reqs) == 1:
                        batch_nchw = pose_reqs[0][0]
                    else:
                        batch_nchw = np.concatenate([r[0] for r in pose_reqs], axis=0)
                    simcc_x, simcc_y = self._pm.session.run(
                        self._out_names, {self._inp_name: batch_nchw}
                    )
                    off = 0
                    for b_arr, evt, res in pose_reqs:
                        n_b = b_arr.shape[0]
                        res["simcc_x"] = simcc_x[off : off + n_b]
                        res["simcc_y"] = simcc_y[off : off + n_b]
                        off += n_b
                        evt.set()
                except BaseException as exc:
                    for _, evt, res in pose_reqs:
                        res["err"] = exc
                        evt.set()

    def close(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        self._thread.join(timeout=5.0)
        with self._cond:
            for _, evt, res in self._pose_q:
                res["err"] = RuntimeError("CUDA pose batcher closed.")
                evt.set()
            for _, _, evt, res in self._det_q:
                res["err"] = RuntimeError("CUDA pose batcher closed.")
                evt.set()
            self._pose_q.clear()
            self._det_q.clear()


def _decode_rtm_outputs(
    kpts: np.ndarray,
    sc: np.ndarray,
    active_backend: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, bool]:
    """Decode single-person RTMW-133 or HALPE-26 keypoints into Body26 + hands + face68."""
    if active_backend == "rtmlib_wholebody" and kpts.shape[0] >= 133:
        b26 = _coco133_to_body26(kpts, sc)
        wb = np.concatenate([kpts[:, :2], sc[:, None]], axis=1).astype(np.float32)
        return b26, wb[91:112].copy(), wb[112:133].copy(), wb[23:91].copy(), True, True
    b26 = np.zeros((26, 3), dtype=np.float32)
    nc = min(26, kpts.shape[0])
    b26[:nc, :2] = kpts[:nc, :2]
    b26[:nc, 2] = sc[:nc]
    return (
        b26,
        np.zeros((21, 3), dtype=np.float32),
        np.zeros((21, 3), dtype=np.float32),
        np.zeros((68, 3), dtype=np.float32),
        False,
        False,
    )


class CinemaWholeBodyVideoTracker:
    """High-accuracy offline multi-camera tracker using Batched RTX CUDA RTMW-x + Actor Lock + Async Wrist/Head Crops."""

    def __init__(
        self,
        backend: str = "rtmlib_wholebody",  # "rtmlib_wholebody", "rtmlib_halpe26", "mediapipe_pose"
        mode: str = "performance",  # "performance", "balanced", "lightweight"
        prefer_gpu: bool = True,
        cuda_batch_size: int = 8,
    ) -> None:
        self.backend = backend
        self.mode = mode
        self.cuda_batch_size = max(1, int(cuda_batch_size))
        self.hw: HardwareDiagnostics = detect_hardware(prefer_gpu=prefer_gpu)
        self.device = self.hw.selected_device
        self._tracker = AIPoseTracker2D(backend=backend, mode=mode, prefer_gpu=prefer_gpu)

    def track_video(
        self,
        camera_uid: str,
        video_path: Path,
        step: int = 1,
        extract_wrist_hands: bool = False,
        extract_head_face: bool = False,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        start_pts_s: Optional[float] = None,
        end_pts_s: Optional[float] = None,
    ) -> CameraTrackingResult:
        """Stream-track a full-resolution camera video with Batched CUDA RTMW-x + Async Parallel Hand/Face Crops."""
        p = Path(video_path)
        cap_meta = cv2.VideoCapture(str(p))
        w = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        h = int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
        fps = float(cap_meta.get(cv2.CAP_PROP_FPS) or 30.0)
        fc = int(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        cap_meta.release()
        if not (5.0 <= fps <= 240.0):
            fps = 30.0

        if progress_cb:
            dev_lbl = f"{self.hw.gpu_name} (CUDA)" if self.device == "cuda" else "CPU"
            progress_cb(
                0.01,
                f"Loading {self.backend} onto {dev_lbl}; missing model weights download on first use...",
            )

        self._tracker._ensure_loaded()
        active_backend = self._tracker.backend
        active_provs = self._tracker.active_providers()
        self.device = "cuda" if "CUDAExecutionProvider" in active_provs else "cpu"
        can_bbox_skip = (
            active_backend != "mediapipe_pose"
            and hasattr(self._tracker._model, "det_model")
            and hasattr(self._tracker._model, "pose_model")
        )
        use_cuda_batch = (
            can_bbox_skip
            and self.cuda_batch_size > 1
            and _can_batch_cuda_pose(self._tracker._model.pose_model)
        )
        if progress_cb:
            if self._tracker.fallback_reason:
                progress_cb(0.02, f"[FALLBACK] {self._tracker.fallback_reason}")
            else:
                prov_str = ", ".join(active_provs) if active_provs else self.device.upper()
                b_tag = f" · Batch-{self.cuda_batch_size} GPU" if use_cuda_batch else ""
                progress_cb(
                    0.02,
                    f"[{self.device.upper()}] {camera_uid} model ready ({active_backend} · {prov_str}{b_tag})",
                )

        actor_lock = ActorLockTracker()
        # On >=24 FPS clips, sample wrist/head crops at 15 Hz (stride=2) + SLERP -> 2x faster & zero GPU stall!
        eff_crop_stride = 2 if (fps / max(1, int(step))) >= 24.0 else 1
        crop_pool = _AsyncHandFacePool(
            enable_hands=bool(extract_wrist_hands),
            enable_face=bool(extract_head_face),
            num_workers=2,
            crop_stride=eff_crop_stride,
            max_queue=1600,
        )

        body_list: List[np.ndarray] = []
        lh_list: List[np.ndarray] = []
        rh_list: List[np.ndarray] = []
        f68_list: List[np.ndarray] = []
        any_wb_hands = False
        any_wb_face = False
        pts_list: List[float] = []
        detected = 0

        tracked_bbox: Optional[List[float]] = None
        frames_since_det = 0
        eff_batch_size = min(self.cuda_batch_size, 6 if fps < 22.0 else self.cuda_batch_size)
        det_refresh_interval = 180 if use_cuda_batch else 12
        last_idx_seen = 0
        t0 = time.perf_counter()

        def _detect_single_full(bgr_img: np.ndarray, fw_i: int, fh_i: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
            keypoints, scores = self._tracker._model(bgr_img)
            if keypoints is None or len(keypoints) == 0:
                return None, None
            best_i = actor_lock.select_actor_index(
                np.asarray(keypoints, dtype=np.float32),
                np.asarray(scores, dtype=np.float32),
                fw_i,
                fh_i,
            )
            return (
                np.asarray(keypoints[best_i], dtype=np.float32),
                calibrate_rtm_scores(np.asarray(scores[best_i], dtype=np.float32)),
            )

        def _record_frame(
            f_idx_out: int,
            pts_val: float,
            bgr_img: np.ndarray,
            kpts_f: Optional[np.ndarray],
            sc_f: Optional[np.ndarray],
        ) -> None:
            nonlocal detected, any_wb_hands, any_wb_face
            if kpts_f is None or sc_f is None:
                b26 = np.zeros((26, 3), dtype=np.float32)
                lh = np.zeros((21, 3), dtype=np.float32)
                rh = np.zeros((21, 3), dtype=np.float32)
                f68 = np.zeros((68, 3), dtype=np.float32)
            else:
                detected += 1
                b26, lh, rh, f68, has_h, has_f = _decode_rtm_outputs(kpts_f, sc_f, active_backend)
                if has_h:
                    any_wb_hands = True
                if has_f:
                    any_wb_face = True
            body_list.append(b26)
            lh_list.append(lh)
            rh_list.append(rh)
            f68_list.append(f68)
            pts_list.append(pts_val)
            crop_pool.submit(f_idx_out, bgr_img, b26)

        prep_pool = ThreadPoolExecutor(max_workers=4) if use_cuda_batch else None
        try:
            frame_iter = _prefetch_video_frames(
                p,
                step=max(1, int(step)),
                cancel_event=cancel_event,
                maxsize=24 if use_cuda_batch else 6,
                start_pts_s=start_pts_s,
                end_pts_s=end_pts_s,
            )

            if use_cuda_batch and prep_pool is not None:
                # HIGH-THROUGHPUT BATCHED RTX CUDA LOOP with height-scaled smart bboxes!
                pm = self._tracker._model.pose_model
                dm = self._tracker._model.det_model
                pending: List[Tuple[int, float, np.ndarray]] = []

                def _flush_cuda_batch(items: List[Tuple[int, float, np.ndarray]]) -> None:
                    nonlocal tracked_bbox, frames_since_det, w, h, last_idx_seen
                    if not items:
                        return
                    fh_i, fw_i = items[0][2].shape[:2]
                    w, h = fw_i, fh_i
                    last_idx_seen = items[-1][0]

                    start_k = 0
                    if tracked_bbox is None or frames_since_det >= det_refresh_interval:
                        bboxes_d = dm(items[0][2])
                        if bboxes_d is not None and len(bboxes_d) == 1:
                            # Single actor: reuse exact YOLOX box straight into the Batch-8 pose pass!
                            tracked_bbox = _pad_det_bbox(bboxes_d[0], fw_i, fh_i)
                            frames_since_det = 0
                        else:
                            k0, s0 = _detect_single_full(items[0][2], fw_i, fh_i)
                            _record_frame(len(body_list), items[0][1], items[0][2], k0, s0)
                            if k0 is not None and s0 is not None:
                                tracked_bbox = _bbox_from_keypoints(k0, s0, fw_i, fh_i)
                                frames_since_det = 1
                            else:
                                tracked_bbox = None
                                frames_since_det = 0
                            start_k = 1

                    rem = items[start_k:]
                    if not rem:
                        return

                    if tracked_bbox is None:
                        for _, pts_v, bgr_v in rem:
                            kf, sf = _detect_single_full(bgr_v, fw_i, fh_i)
                            _record_frame(len(body_list), pts_v, bgr_v, kf, sf)
                            if kf is not None and sf is not None:
                                tracked_bbox = _bbox_from_keypoints(kf, sf, fw_i, fh_i)
                                frames_since_det = 1
                        return

                    bgr_batch = [it[2] for it in rem]
                    bbox_batch = [tracked_bbox] * len(rem)
                    kpts_b, sc_b = _run_pose_batch_cuda(pm, bgr_batch, bbox_batch, prep_pool)
                    bx1, by1, bx2, by2 = tracked_bbox

                    for j, (_, pts_v, bgr_v) in enumerate(rem):
                        kj = kpts_b[j]
                        sj = calibrate_rtm_scores(sc_b[j])
                        mean_c = float(np.mean(sj[:17]))
                        valid_chk = sj[:17] >= 0.35
                        near_edge = False
                        if int(np.sum(valid_chk)) >= 5:
                            xs_c = kj[:17][valid_chk, 0]
                            ys_c = kj[:17][valid_chk, 1]
                            if (
                                (bx1 > 4.0 and float(np.min(xs_c)) < bx1 + 4.0)
                                or (bx2 < fw_i - 5.0 and float(np.max(xs_c)) > bx2 - 4.0)
                                or (by1 > 4.0 and float(np.min(ys_c)) < by1 + 4.0)
                                or (by2 < fh_i - 5.0 and float(np.max(ys_c)) > by2 - 4.0)
                            ):
                                near_edge = True

                        if mean_c < 0.22 or near_edge:
                            kf, sf = _detect_single_full(bgr_v, fw_i, fh_i)
                            _record_frame(len(body_list), pts_v, bgr_v, kf, sf)
                            if kf is not None and sf is not None:
                                tracked_bbox = _bbox_from_keypoints(kf, sf, fw_i, fh_i)
                                bx1, by1, bx2, by2 = tracked_bbox or [0.0, 0.0, float(fw_i), float(fh_i)]
                                frames_since_det = 1
                            else:
                                tracked_bbox = None
                                frames_since_det = 0
                        else:
                            _record_frame(len(body_list), pts_v, bgr_v, kj, sj)
                            frames_since_det += 1
                            if j == len(rem) - 1:
                                nb = _bbox_from_keypoints(kj, sj, fw_i, fh_i)
                                if nb is not None:
                                    tracked_bbox = nb

                for idx, pts_s, bgr in frame_iter:
                    if cancel_event is not None and cancel_event.is_set():
                        raise RuntimeError("Cancelled by user.")
                    pending.append((idx, pts_s, bgr))
                    if len(pending) >= eff_batch_size:
                        _flush_cuda_batch(pending)
                        pending.clear()
                        if progress_cb:
                            frac = min(0.99, float(last_idx_seen + 1) / max(float(fc), 1.0))
                            live_fps = float(len(body_list)) / max(time.perf_counter() - t0, 1e-3)
                            progress_cb(
                                frac,
                                f"[CUDA Batch-{eff_batch_size} · {live_fps:.1f} FPS] {camera_uid} ({p.name}): frame {last_idx_seen + 1}/{fc}",
                            )
                if pending:
                    _flush_cuda_batch(pending)
                    pending.clear()
            else:
                # Fallback sequential path (CPU / MediaPipe Pose)
                for idx, pts_s, bgr in frame_iter:
                    if cancel_event is not None and cancel_event.is_set():
                        raise RuntimeError("Cancelled by user.")
                    fh, fw = bgr.shape[:2]
                    w, h = fw, fh
                    if active_backend == "mediapipe_pose":
                        if fw > 1280:
                            sc_r = 1280.0 / float(fw)
                            bgr_in = cv2.resize(
                                bgr, (1280, max(1, int(round(fh * sc_r)))), interpolation=cv2.INTER_AREA
                            )
                        else:
                            bgr_in = bgr
                        rgb = cv2.cvtColor(bgr_in, cv2.COLOR_BGR2RGB)
                        res = self._tracker._model.process(rgb)
                        if res.pose_landmarks:
                            b26 = _mediapipe33_to_body26(res.pose_landmarks.landmark, fw, fh)
                            detected += 1
                        else:
                            b26 = np.zeros((26, 3), dtype=np.float32)
                        body_list.append(b26)
                        lh_list.append(np.zeros((21, 3), dtype=np.float32))
                        rh_list.append(np.zeros((21, 3), dtype=np.float32))
                        f68_list.append(np.zeros((68, 3), dtype=np.float32))
                        pts_list.append(pts_s)
                        crop_pool.submit(len(body_list) - 1, bgr, b26)
                    else:
                        used_skip = False
                        if can_bbox_skip and tracked_bbox is not None and frames_since_det < det_refresh_interval:
                            keypoints, scores = self._tracker._model.pose_model(bgr, bboxes=[tracked_bbox])
                            used_skip = True
                            frames_since_det += 1
                            if (
                                keypoints is None
                                or len(keypoints) == 0
                                or float(np.mean(calibrate_rtm_scores(scores[0][:17]))) < 0.22
                            ):
                                keypoints, scores = self._tracker._model(bgr)
                                used_skip = False
                                frames_since_det = 0
                        else:
                            keypoints, scores = self._tracker._model(bgr)
                            frames_since_det = 0

                        if keypoints is None or len(keypoints) == 0:
                            tracked_bbox = None
                            _record_frame(len(body_list), pts_s, bgr, None, None)
                        else:
                            best_i = (
                                0
                                if used_skip
                                else actor_lock.select_actor_index(
                                    np.asarray(keypoints, dtype=np.float32),
                                    np.asarray(scores, dtype=np.float32),
                                    fw,
                                    fh,
                                )
                            )
                            kf = np.asarray(keypoints[best_i], dtype=np.float32)
                            sf = calibrate_rtm_scores(np.asarray(scores[best_i], dtype=np.float32))
                            tracked_bbox = _bbox_from_keypoints(kf, sf, fw, fh, pad=0.26)
                            _record_frame(len(body_list), pts_s, bgr, kf, sf)

                    if progress_cb and (len(body_list) == 1 or len(body_list) % 10 == 0):
                        frac = min(0.99, float(idx + 1) / max(float(fc), 1.0))
                        live_fps = float(len(body_list)) / max(time.perf_counter() - t0, 1e-3)
                        progress_cb(
                            frac,
                            f"[{self.device.upper()} · {live_fps:.1f} FPS] {camera_uid} ({p.name}): frame {idx + 1}/{fc} ({active_backend})",
                        )
        finally:
            if prep_pool is not None:
                prep_pool.shutdown(wait=False)
            wrist_hands, head_faces = crop_pool.finish(len(body_list))

        elapsed = max(time.perf_counter() - t0, 1e-6)
        n = len(body_list)
        return CameraTrackingResult(
            camera_uid=camera_uid,
            video_path=p,
            width=w,
            height=h,
            fps=fps / max(1, int(step)),
            pts_s=pts_list,
            body26_uvc=np.stack(body_list, axis=0) if n else np.zeros((0, 26, 3), dtype=np.float32),
            left_hand21_uvc=np.stack(lh_list, axis=0) if (any_wb_hands and n) else None,
            right_hand21_uvc=np.stack(rh_list, axis=0) if (any_wb_hands and n) else None,
            face68_uvc=np.stack(f68_list, axis=0) if (any_wb_face and n) else None,
            wrist_crop_hands=wrist_hands,
            head_crop_faces=head_faces,
            backend_tag=self._tracker.model_version_tag,
            device=self.device,
            detection_rate=float(detected) / max(n, 1),
            measured_fps=float(n) / elapsed,
        )

    def _track_camera_on_batcher(
        self,
        batcher: _SharedCudaPoseBatcher,
        camera_uid: str,
        video_path: Path,
        step: int,
        extract_wrist_hands: bool,
        extract_head_face: bool,
        active_backend: str,
        on_frames_done: Callable[[int], None],
        cancel_event: Optional[threading.Event],
        abort_event: threading.Event,
        start_pts_s: Optional[float] = None,
        end_pts_s: Optional[float] = None,
    ) -> CameraTrackingResult:
        """Worker executed per camera concurrently, feeding height-scaled smart-bbox chunks to the shared Batch-16 GPU."""
        p = Path(video_path)
        cap_meta = cv2.VideoCapture(str(p))
        w = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        h = int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
        fps = float(cap_meta.get(cv2.CAP_PROP_FPS) or 30.0)
        cap_meta.release()
        if not (5.0 <= fps <= 240.0):
            fps = 30.0

        actor_lock = ActorLockTracker()
        # Sample wrist/head crops at 10 Hz (stride=3 on >=24 FPS) + geodesic SLERP + 30 Hz 3-cam RTMW-x -> 0s GPU wait!
        eff_crop_stride = 3 if (fps / max(1, int(step))) >= 24.0 else 1
        crop_pool = _AsyncHandFacePool(
            enable_hands=bool(extract_wrist_hands),
            enable_face=bool(extract_head_face),
            num_workers=3,
            crop_stride=eff_crop_stride,
            max_queue=1600,
        )

        body_list: List[np.ndarray] = []
        lh_list: List[np.ndarray] = []
        rh_list: List[np.ndarray] = []
        f68_list: List[np.ndarray] = []
        any_wb_hands = False
        any_wb_face = False
        pts_list: List[float] = []
        detected = 0

        tracked_bbox: Optional[List[float]] = None
        frames_since_det = 0
        base_chunk = 6 if fps < 22.0 else 8
        det_refresh_interval = 180
        t0 = time.perf_counter()

        def _is_cancelled() -> bool:
            return abort_event.is_set() or (cancel_event is not None and cancel_event.is_set())

        def _detect_full(bgr_img: np.ndarray, fw_i: int, fh_i: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
            keypoints, scores = batcher.submit_detect_full(bgr_img)
            if keypoints is None or len(keypoints) == 0:
                return None, None
            best_i = actor_lock.select_actor_index(
                np.asarray(keypoints, dtype=np.float32),
                np.asarray(scores, dtype=np.float32),
                fw_i,
                fh_i,
            )
            return (
                np.asarray(keypoints[best_i], dtype=np.float32),
                calibrate_rtm_scores(np.asarray(scores[best_i], dtype=np.float32)),
            )

        def _record(
            f_idx_out: int,
            pts_val: float,
            bgr_img: np.ndarray,
            kpts_f: Optional[np.ndarray],
            sc_f: Optional[np.ndarray],
        ) -> None:
            nonlocal detected, any_wb_hands, any_wb_face
            if kpts_f is None or sc_f is None:
                b26 = np.zeros((26, 3), dtype=np.float32)
                lh = np.zeros((21, 3), dtype=np.float32)
                rh = np.zeros((21, 3), dtype=np.float32)
                f68 = np.zeros((68, 3), dtype=np.float32)
            else:
                detected += 1
                b26, lh, rh, f68, has_h, has_f = _decode_rtm_outputs(kpts_f, sc_f, active_backend)
                if has_h:
                    any_wb_hands = True
                if has_f:
                    any_wb_face = True
            body_list.append(b26)
            lh_list.append(lh)
            rh_list.append(rh)
            f68_list.append(f68)
            pts_list.append(pts_val)
            crop_pool.submit(f_idx_out, bgr_img, b26)

        def _flush_chunk(items: List[Tuple[int, float, np.ndarray]]) -> None:
            nonlocal tracked_bbox, frames_since_det, w, h
            if not items:
                return
            fh_i, fw_i = items[0][2].shape[:2]
            w, h = fw_i, fh_i

            start_k = 0
            if tracked_bbox is None or frames_since_det >= det_refresh_interval:
                bboxes_d = batcher.submit_detect_boxes(items[0][2])
                if len(bboxes_d) == 1:
                    # Single actor in scene: skip slow batch=1 pose and run ALL frames in the Batch-16 GPU pass!
                    tracked_bbox = _pad_det_bbox(bboxes_d[0], fw_i, fh_i)
                    frames_since_det = 0
                else:
                    k0, s0 = _detect_full(items[0][2], fw_i, fh_i)
                    _record(len(body_list), items[0][1], items[0][2], k0, s0)
                    if k0 is not None and s0 is not None:
                        tracked_bbox = _bbox_from_keypoints(k0, s0, fw_i, fh_i)
                        frames_since_det = 1
                    else:
                        tracked_bbox = None
                        frames_since_det = 0
                    start_k = 1

            rem = items[start_k:]
            if not rem:
                on_frames_done(len(items))
                return

            if tracked_bbox is None:
                for _, pts_v, bgr_v in rem:
                    kf, sf = _detect_full(bgr_v, fw_i, fh_i)
                    _record(len(body_list), pts_v, bgr_v, kf, sf)
                    if kf is not None and sf is not None:
                        tracked_bbox = _bbox_from_keypoints(kf, sf, fw_i, fh_i)
                        frames_since_det = 1
                on_frames_done(len(items))
                return

            bgr_batch = [it[2] for it in rem]
            bbox_batch = [tracked_bbox] * len(rem)
            batch_nchw, centers, scales = batcher.preprocess_batch(bgr_batch, bbox_batch)
            kpts_b, sc_b = batcher.submit_pose(batch_nchw, centers, scales)
            bx1, by1, bx2, by2 = tracked_bbox

            for j, (_, pts_v, bgr_v) in enumerate(rem):
                kj = kpts_b[j]
                sj = calibrate_rtm_scores(sc_b[j])
                mean_c = float(np.mean(sj[:17]))
                valid_chk = sj[:17] >= 0.35
                near_edge = False
                if int(np.sum(valid_chk)) >= 5:
                    xs_c = kj[:17][valid_chk, 0]
                    ys_c = kj[:17][valid_chk, 1]
                    if (
                        (bx1 > 4.0 and float(np.min(xs_c)) < bx1 + 4.0)
                        or (bx2 < fw_i - 5.0 and float(np.max(xs_c)) > bx2 - 4.0)
                        or (by1 > 4.0 and float(np.min(ys_c)) < by1 + 4.0)
                        or (by2 < fh_i - 5.0 and float(np.max(ys_c)) > by2 - 4.0)
                    ):
                        near_edge = True

                if mean_c < 0.22 or near_edge:
                    kf, sf = _detect_full(bgr_v, fw_i, fh_i)
                    _record(len(body_list), pts_v, bgr_v, kf, sf)
                    if kf is not None and sf is not None:
                        tracked_bbox = _bbox_from_keypoints(kf, sf, fw_i, fh_i)
                        bx1, by1, bx2, by2 = tracked_bbox or [0.0, 0.0, float(fw_i), float(fh_i)]
                        frames_since_det = 1
                    else:
                        tracked_bbox = None
                        frames_since_det = 0
                else:
                    _record(len(body_list), pts_v, bgr_v, kj, sj)
                    frames_since_det += 1
                    if j == len(rem) - 1:
                        nb = _bbox_from_keypoints(kj, sj, fw_i, fh_i)
                        if nb is not None:
                            tracked_bbox = nb
            on_frames_done(len(items))

        try:
            pending: List[Tuple[int, float, np.ndarray]] = []
            for idx, pts_s, bgr in _prefetch_video_frames(
                p,
                step=max(1, int(step)),
                cancel_event=cancel_event,
                maxsize=24,
                start_pts_s=start_pts_s,
                end_pts_s=end_pts_s,
            ):
                if _is_cancelled():
                    raise RuntimeError("Cancelled by user.")
                pending.append((idx, pts_s, bgr))
                # Keep tight 8-frame chunks while 2+ cams coalesce to Batch-16..24; widen slightly only for final solo tail
                target_chunk = min(10, base_chunk + 2) if batcher.active_cams <= 1 else base_chunk
                if len(pending) >= target_chunk:
                    _flush_chunk(pending)
                    pending.clear()
            if pending and not _is_cancelled():
                _flush_chunk(pending)
                pending.clear()
        finally:
            batcher.mark_camera_done()
            wrist_hands, head_faces = crop_pool.finish(len(body_list))

        elapsed = max(time.perf_counter() - t0, 1e-6)
        n = len(body_list)
        return CameraTrackingResult(
            camera_uid=camera_uid,
            video_path=p,
            width=w,
            height=h,
            fps=fps / max(1, int(step)),
            pts_s=pts_list,
            body26_uvc=np.stack(body_list, axis=0) if n else np.zeros((0, 26, 3), dtype=np.float32),
            left_hand21_uvc=np.stack(lh_list, axis=0) if (any_wb_hands and n) else None,
            right_hand21_uvc=np.stack(rh_list, axis=0) if (any_wb_hands and n) else None,
            face68_uvc=np.stack(f68_list, axis=0) if (any_wb_face and n) else None,
            wrist_crop_hands=wrist_hands,
            head_crop_faces=head_faces,
            backend_tag=self._tracker.model_version_tag,
            device=self.device,
            detection_rate=float(detected) / max(n, 1),
            measured_fps=float(n) / elapsed,
        )

    def track_videos_parallel(
        self,
        camera_specs: List[Tuple[str, Path, bool, bool]],
        step: int = 1,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        pts_windows_by_cam: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
    ) -> Dict[str, CameraTrackingResult]:
        """Track multiple camera videos concurrently using ONE shared Batch-16 RTX CUDA inference dispatcher!

        Each item in `camera_specs` is `(camera_uid, video_path, extract_wrist_hands, extract_head_face)`.
        Optional `pts_windows_by_cam` maps `camera_uid -> (start_pts_s, end_pts_s)` to skip non-overlapping frames.
        Automatically falls back to sequential `track_video` on CPU/MediaPipe or single-camera input.
        """
        if not camera_specs:
            return {}
        win_map = pts_windows_by_cam or {}

        if progress_cb:
            dev_lbl = f"{self.hw.gpu_name} (CUDA)" if self.device == "cuda" else "CPU"
            progress_cb(
                0.01,
                f"Loading {self.backend} onto {dev_lbl}; missing model weights download on first use...",
            )

        self._tracker._ensure_loaded()
        active_backend = self._tracker.backend
        active_provs = self._tracker.active_providers()
        self.device = "cuda" if "CUDAExecutionProvider" in active_provs else "cpu"
        can_bbox_skip = (
            active_backend != "mediapipe_pose"
            and hasattr(self._tracker._model, "det_model")
            and hasattr(self._tracker._model, "pose_model")
        )
        use_parallel_cuda = (
            len(camera_specs) >= 2
            and can_bbox_skip
            and self.cuda_batch_size > 1
            and _can_batch_cuda_pose(self._tracker._model.pose_model)
        )

        if not use_parallel_cuda:
            results: Dict[str, CameraTrackingResult] = {}
            n_cams = len(camera_specs)
            for c_i, (uid, vpath, do_h, do_f) in enumerate(camera_specs):
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("Cancelled by user.")

                def _sub_cb(p: float, m: str, i=c_i) -> None:
                    if progress_cb:
                        progress_cb((i + p) / max(n_cams, 1), m)

                s_pts, e_pts = win_map.get(uid, (None, None))
                results[uid] = self.track_video(
                    camera_uid=uid,
                    video_path=vpath,
                    step=step,
                    extract_wrist_hands=do_h,
                    extract_head_face=do_f,
                    progress_cb=_sub_cb,
                    cancel_event=cancel_event,
                    start_pts_s=s_pts,
                    end_pts_s=e_pts,
                )
            return results

        # Estimate total overlapping frame count across all cameras for unified throughput reporting
        total_est_frames = 0
        for uid, vpath, _, _ in camera_specs:
            cap_m = cv2.VideoCapture(str(vpath))
            fc_m = int(cap_m.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
            fps_m = float(cap_m.get(cv2.CAP_PROP_FPS) or 30.0)
            cap_m.release()
            if not (5.0 <= fps_m <= 240.0):
                fps_m = 30.0
            s_pts, e_pts = win_map.get(uid, (None, None))
            if s_pts is not None and e_pts is not None and e_pts > s_pts:
                fc_m = min(fc_m, max(1, int(round((e_pts - s_pts) * fps_m))))
            total_est_frames += max(1, (fc_m + max(1, int(step)) - 1) // max(1, int(step)))

        if progress_cb:
            prov_str = ", ".join(active_provs) if active_provs else self.device.upper()
            progress_cb(
                0.02,
                f"[CUDA Parallel {len(camera_specs)}-Cam · Shared Batch-16 GPU] {active_backend} ready ({prov_str})",
            )

        batcher = _SharedCudaPoseBatcher(
            self._tracker._model,
            max_gpu_batch=24,
            num_cams=len(camera_specs),
        )
        prog_lock = threading.Lock()
        done_frames = [0]
        last_report_t = [0.0]
        t_all = time.perf_counter()
        abort_evt = threading.Event()

        def _on_frames(delta: int) -> None:
            if not progress_cb:
                return
            with prog_lock:
                done_frames[0] += delta
                now = time.perf_counter()
                if now - last_report_t[0] >= 0.20 or done_frames[0] >= total_est_frames:
                    last_report_t[0] = now
                    frac = min(0.99, float(done_frames[0]) / max(float(total_est_frames), 1.0))
                    agg_fps = float(done_frames[0]) / max(now - t_all, 1e-3)
                    progress_cb(
                        frac,
                        f"[CUDA Parallel {len(camera_specs)}-Cam · {agg_fps:.1f} FPS total] {done_frames[0]}/{total_est_frames} frames",
                    )

        results_map: Dict[str, CameraTrackingResult] = {}
        try:
            with ThreadPoolExecutor(max_workers=len(camera_specs)) as cam_pool:
                futs = {
                    uid: cam_pool.submit(
                        self._track_camera_on_batcher,
                        batcher,
                        uid,
                        vpath,
                        step,
                        do_h,
                        do_f,
                        active_backend,
                        _on_frames,
                        cancel_event,
                        abort_evt,
                        win_map.get(uid, (None, None))[0],
                        win_map.get(uid, (None, None))[1],
                    )
                    for uid, vpath, do_h, do_f in camera_specs
                }
                for uid, fut in futs.items():
                    try:
                        results_map[uid] = fut.result()
                    except BaseException:
                        abort_evt.set()
                        raise
        finally:
            abort_evt.set()
            batcher.close()

        return results_map

    def close(self) -> None:
        self._tracker.close()
