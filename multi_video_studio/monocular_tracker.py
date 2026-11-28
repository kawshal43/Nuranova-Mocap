"""Single-view adapters. Native observations are immutable, corrections are a later stage."""
from __future__ import annotations

import time
import os
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from engine.ai_tracking.monocular_geometry import unproject_native, map_coco_xyz_to_body26
from engine.ai_tracking.pose_tracker import _mediapipe33_to_body26, _coco133_to_body26, calibrate_rtm_scores
from engine.hands.hand_tracker import HandCaptureEngine, HandObservation, solve_hand_rotations_from_landmarks21
from engine.face.face_tracker import FacialCaptureEngine
from engine.video.video_io import iter_video_frames
from multi_video_studio.wholebody_cuda import CameraTrackingResult, extract_head_crop_bgr, ActorLockTracker


def resolved_backend(cfg):
    if cfg.single_camera_backend != "auto":
        return cfg.single_camera_backend
    from model_registry.registry import ModelRegistry
    from engine.ai_tracking.runtime import detect_hardware
    return ("rtmw3d" if cfg.prefer_gpu and ModelRegistry().is_cached("rtmw3d_x")
            and detect_hardware().cuda_usable else "mediapipe")


class MonocularTracker:
    def __init__(self, cfg, progress=None, cancel=None):
        self.cfg, self.progress = cfg, progress
        self.model = None
        self.backend = cfg.single_camera_backend
        self.providers = ["CPUExecutionProvider"]
        self.fallback_reason = None
        self.box = None
        self.actor_lock = ActorLockTracker()
        self.n = 0
        if self.backend == "auto":
            # Auto uses available local weights; never hides a large startup download.
            self.backend = resolved_backend(cfg)
        if self.backend == "rtmw3d":
            try:
                from engine.ai_tracking.runtime import detect_hardware
                from model_registry.registry import ModelRegistry
                from rtmlib import Wholebody3d
                device = "cuda" if cfg.prefer_gpu and detect_hardware().cuda_usable else "cpu"
                if progress:
                    progress(.02, "Loading 3D body tracker; the first download is approximately 369 MB…")
                path = ModelRegistry().ensure_downloaded("rtmw3d_x", timeout_s=30, cancel=cancel)
                self.model = Wholebody3d(pose=str(path), device="cpu", backend="onnxruntime")
                # RTMLib's default CUDA exhaustive cuDNN search stalls the first frame.
                # Explicit sessions keep startup bounded and avoid excessive CPU thread pools.
                import onnxruntime as ort
                for component in (self.model.det_model, self.model.pose_model):
                    options = ort.SessionOptions()
                    options.intra_op_num_threads = min(4, os.cpu_count() or 1)
                    options.log_severity_level = 3
                    providers = ([('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'HEURISTIC',
                        'cudnn_conv_use_max_workspace': '1', 'device_id': 0}), 'CPUExecutionProvider']
                        if device == 'cuda' else ['CPUExecutionProvider'])
                    component.session = ort.InferenceSession(str(component.onnx_model),
                        sess_options=options, providers=providers)
                self.providers = self.model.pose_model.session.get_providers()
            except Exception as exc:
                if cancel is not None and cancel.is_set():
                    raise RuntimeError("Cancelled by user.") from exc
                # Explicit selection must fail visibly; auto may fall back with evidence.
                if cfg.single_camera_backend != "auto":
                    raise RuntimeError(f"Cannot start RTMW3D: {exc}") from exc
                self.fallback_reason = str(exc)
                self.backend = "mediapipe"
                self.model = None
        if self.backend == "mediapipe":
            import mediapipe as mp
            complexity = {"lightweight": 0, "balanced": 1, "performance": 2}.get(cfg.ai_mode, 2)
            self.model = mp.solutions.pose.Pose(model_complexity=complexity, smooth_landmarks=True,
                min_detection_confidence=.45, min_tracking_confidence=.45)
            self.providers = ["MediaPipe CPU / XNNPACK"]

    def infer(self, image):
        h, w = image.shape[:2]
        if self.backend == "mediapipe":
            from desktop.workers.live_preview import _mediapipe_world33_to_body26_zup
            small = image if w <= 1280 else cv2.resize(image, (1280, round(h * 1280 / w)))
            result = self.model.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            if not result.pose_world_landmarks or not result.pose_landmarks:
                return None
            p, _ = _mediapipe_world33_to_body26_zup(result.pose_world_landmarks.landmark, ground_feet=False)
            uv = _mediapipe33_to_body26(result.pose_landmarks.landmark, w, h)
            return p.astype(np.float32), uv, None
        if self.box is None or self.n % 8 == 0:
            boxes = self.model.det_model(image)
            if len(boxes) == 0:
                self.box = None
                return None
            # Score all detected people, retaining anatomical actor identity across detections.
            raw, scores, _, native = self.model.pose_model(image, bboxes=boxes)
            scores = calibrate_rtm_scores(scores)
            selected = self.actor_lock.select_actor_index(native, scores, w, h)
        else:
            raw, scores, _, native = self.model.pose_model(image, bboxes=[self.box])
            scores = calibrate_rtm_scores(scores)
            selected = 0
        self.n += 1
        xyz, uv, scores = raw[selected], native[selected], scores[selected]
        good = (scores[:23] > .25) & np.isfinite(uv[:23]).all(axis=-1)
        if good.sum() < 5 or (scores[[5, 6, 11, 12]] > .3).sum() < 3:
            self.box = None
            return None
        visible = uv[:23][good]
        lo, hi = visible.min(axis=0), visible.max(axis=0)
        center, size = (lo + hi) / 2, np.maximum(hi - lo, [60, 100]) * 1.2
        self.box = np.r_[np.maximum(0, center - size / 2), np.minimum([w, h], center + size / 2)]
        focal = self.cfg.single_focal_length_px or .84 * max(w, h)
        span = max(float(np.ptp(visible[:, 1])), h * .10)
        root_depth = focal * self.cfg.actor_height_m / span
        depth = xyz[:, 2] - np.mean(xyz[[11, 12], 2])
        points = unproject_native(uv, depth, root_depth, w, h, focal)
        return map_coco_xyz_to_body26(points), _coco133_to_body26(uv, scores), (points, uv, scores)

    def close(self):
        if self.backend == "mediapipe" and self.model is not None:
            self.model.close()
        self.model = None


def wholebody_hands(prediction, body):
    if prediction is None:
        return {}
    points, native, scores = prediction
    output = {}
    for side, start, wrist in (("left", 91, 9), ("right", 112, 10)):
        p, uv, confidence = points[start:start+21], native[start:start+21], scores[start:start+21].copy()
        palm_px = float(np.linalg.norm(uv[5] - uv[17]))
        # A small or contradictory hand remains uncertain regardless of upsampling.
        confidence *= np.clip((palm_px - 5.) / 20., 0, 1)
        if confidence.mean() < .2 or np.linalg.norm(uv[0] - body[wrist, :2]) > max(15, 2.5 * palm_px):
            continue
        local = p - p[0]
        span = np.linalg.norm(local[9])
        if span < .015 or span > .30:
            continue
        output[side] = HandObservation(side, local.astype(np.float32), float(confidence.mean()),
            solve_hand_rotations_from_landmarks21(local, handedness=side),
            landmark_confidence=confidence, landmarks_2d=uv.copy(), source="rtmw3d_estimate")
    return output


def gate_native_hands(estimates, body):
    """Reject wrong-wrist crops and report tiny palms with reduced confidence."""
    output = {}
    for side, obs in estimates.items():
        uv = obs.landmarks_2d
        if uv is None or not np.isfinite(uv).all():
            continue
        own, other = (9, 10) if side == "left" else (10, 9)
        palm_px = float(np.linalg.norm(uv[5] - uv[17]))
        own_distance = np.linalg.norm(uv[0] - body[own, :2])
        other_distance = np.linalg.norm(uv[0] - body[other, :2])
        if palm_px < 5 or own_distance > max(35, palm_px * 2.5):
            continue
        if body[other, 2] > .4 and other_distance + 15 < own_distance:
            continue
        weight = min(1., palm_px / 25.)
        if body[other, 2] > .4 and np.linalg.norm(body[own, :2] - body[other, :2]) < palm_px:
            weight *= .5
        obs.confidence *= weight
        if obs.landmark_confidence is not None:
            obs.landmark_confidence = obs.landmark_confidence * weight
        output[side] = obs
    return output


class BackgroundMotionMonitor:
    """Actor-masked background motion is a diagnostic, never measured actor travel."""
    def __init__(self):
        self.previous = None
        self.previous_mask = None
        self.measurements = []

    def sample(self, image, body):
        h, w = image.shape[:2]
        scale = 640 / w
        gray = cv2.cvtColor(cv2.resize(image, (640, round(h*scale))), cv2.COLOR_BGR2GRAY)
        mask = np.full(gray.shape, 255, np.uint8)
        visible = body[(body[:, 2] > .2) & np.isfinite(body).all(axis=1), :2] * scale
        if len(visible):
            low, high = visible.min(axis=0)-35, visible.max(axis=0)+35
            cv2.rectangle(mask, tuple(np.maximum(0, low).astype(int)), tuple(high.astype(int)), 0, -1)
        if self.previous is not None:
            points = cv2.goodFeaturesToTrack(self.previous, 180, .02, 10, mask=self.previous_mask)
            if points is not None and len(points) >= 20:
                moved, ok, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, points, None)
                if moved is not None and ok.sum() >= 15:
                    affine, inliers = cv2.estimateAffinePartial2D(points[ok.ravel() > 0], moved[ok.ravel() > 0],
                        method=cv2.RANSAC, ransacReprojThreshold=2)
                    if affine is not None and float(inliers.mean()) >= .6:
                        self.measurements.append({"translation_px_at_640": float(np.linalg.norm(affine[:, 2])),
                            "scale": float(np.linalg.norm(affine[:, 0])), "inliers": float(inliers.mean())})
        self.previous, self.previous_mask = gray, mask

    def report(self):
        moving = any(m["translation_px_at_640"] > 8 or abs(m["scale"]-1) > .015 for m in self.measurements)
        return {"detected": moving, "valid_pairs": len(self.measurements),
                "status": "background motion detected" if moving else "stationary camera assumed",
                "measurements": self.measurements}


def track_single_camera(clip, cfg, cancel=None, progress=None):
    tracker = MonocularTracker(cfg, progress, cancel)
    hands_engine = HandCaptureEngine() if cfg.enable_fingers else None
    face_engine = FacialCaptureEngine() if cfg.enable_head_face else None
    relative, bodies, hands, faces, times = [], [], [], [], []
    left, right = [], []
    start = time.perf_counter()
    motion_monitor = BackgroundMotionMonitor()
    # Only tiny native-resolution crops cross this bounded worker queue. Body
    # inference/decode overlap CPU hand/face work without retaining full videos.
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="monocular-detail")
    pending = deque()

    def refine(crops, head):
        if cancel is not None and cancel.is_set():
            return {}, None
        detail = hands_engine.infer_hands_from_crops(crops) if crops else {}
        face = face_engine.infer_frame(head) if head is not None else None
        return detail, face

    def collect():
        position, future = pending.popleft()
        detail, faces[position] = future.result()
        hands[position].update(detail)
        hands[position] = gate_native_hands(hands[position], bodies[position])

    try:
        for index, pts, image in iter_video_frames(clip.path, step=cfg.frame_stride,
                start_pts_s=cfg.single_start_s, end_pts_s=cfg.single_end_s):
            if cancel is not None and cancel.is_set():
                raise RuntimeError("Cancelled by user.")
            result = tracker.infer(image)
            body = result[1] if result else np.zeros((26, 3), np.float32)
            relative.append(result[0] if result else np.full((26, 3), np.nan, np.float32))
            bodies.append(body)
            times.append(pts)
            estimates = wholebody_hands(result[2], body) if result and cfg.enable_fingers else {}
            crops = {}
            if hands_engine and result and len(times) % cfg.single_hand_interval == 1 % cfg.single_hand_interval:
                crops = hands_engine.extract_wrist_crops_from_body26(image, body)
            hands.append(estimates)
            faces.append(None)
            head = extract_head_crop_bgr(image, body) if face_engine and result else None
            pending.append((len(times)-1, pool.submit(refine, crops, head)))
            if len(pending) >= 2:
                collect()
            if len(times) % 15 == 1:
                motion_monitor.sample(image, body)
            if progress and len(times) % 20 == 0:
                progress(.04 + .62 * index / max(clip.frame_count, 1),
                    f"Tracking source frame {index} · {tracker.backend} · {tracker.providers[0]}")
        while pending:
            collect()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        tracker.close()
        if hands_engine:
            hands_engine.close()
        if face_engine:
            face_engine.close()
    if len(times) < 2:
        raise RuntimeError("Selected video range contains fewer than two frames.")
    for frame in hands:
        for side, target in (("left", left), ("right", right)):
            obs = frame.get(side)
            uvc = np.zeros((21, 3), np.float32)
            if obs is not None and obs.landmarks_2d is not None:
                uvc[:, :2] = obs.landmarks_2d
                uvc[:, 2] = obs.landmark_confidence if obs.landmark_confidence is not None else obs.confidence
            target.append(uvc)
    runtime = {"backend": tracker.backend, "providers": tracker.providers,
               "requested_backend": cfg.single_camera_backend, "prefer_gpu": cfg.prefer_gpu,
               "fallback_reason": tracker.fallback_reason, "inference_seconds": time.perf_counter()-start,
               "native_timestamps": True, "hand_interval": cfg.single_hand_interval}
    runtime["detail_queue_capacity"] = 2
    runtime["camera_motion"] = motion_monitor.report()
    from importlib.metadata import version
    runtime["model_version"] = ("rtmw3d-b0a0eab7_20240626" if tracker.backend == "rtmw3d"
                                else "mediapipe-" + version("mediapipe"))
    if tracker.backend == "rtmw3d":
        runtime["checkpoint_sha256"] = "4a289c0e99d47eb595e99679d9d4a2d1def1b4241f9adcbafba44b9ff585ebcd"
        runtime["rtmlib_version"] = version("rtmlib")
    return CameraTrackingResult(clip.camera_uid, Path(clip.path), clip.width, clip.height, clip.fps,
        times, np.asarray(bodies), np.asarray(left), np.asarray(right), wrist_crop_hands=hands,
        head_crop_faces=faces, backend_tag=tracker.backend,
        device="cuda" if "CUDAExecutionProvider" in tracker.providers else "cpu",
        detection_rate=float(np.mean(np.asarray(bodies)[:, 19, 2] > .3)),
        relative_body26=np.asarray(relative), runtime_metadata=runtime)
