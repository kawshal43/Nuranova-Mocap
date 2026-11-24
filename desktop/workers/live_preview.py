"""Real-Time Non-Blocking Live Mocap Preview Worker.

Runs lightweight 2D/3D pose estimation on incoming smartphone JPEG preview streams
(supporting both single-phone 3D world pose preview and multi-camera triangulated preview),
draws crisp 2D skeleton overlays on camera feeds, and emits live 25-bone articulated 3D
character/skeleton poses for the Capture Studio 3D viewport!
"""

from __future__ import annotations

import math
import threading
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal
from scipy.spatial.transform import Rotation

from desktop.camera_manager.manager import CameraManager
from engine.ai_tracking.pose_tracker import _mediapipe33_to_body26
from engine.math_core.camera_model import MultiCameraCalibration
from engine.math_core.transforms import quat_identity
from engine.skeleton.rig_definition import (
    BONE_INDEX,
    BONE_NAMES,
    ActorProfile,
    compute_forward_kinematics_frame,
)
from engine.skeleton.solver import ArticulatedSkeletonSolver, SolverConfig
from engine.triangulation.triangulator import Observation2D, triangulate_point_robust

# 2D skeleton bone pairs in canonical Body26 indices for camera preview overlay
_OVERLAY_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (15, 20), (15, 24),
    (12, 14), (14, 16), (16, 21), (16, 25),
    (18, 0), (18, 19),
]


def draw_body26_overlay_bgr(frame_bgr: np.ndarray, body26_uvc: np.ndarray) -> bytes:
    """Draw a glowing studio skeleton overlay onto a BGR camera frame and return JPEG bytes."""
    img = frame_bgr.copy()
    for a, b in _OVERLAY_EDGES:
        if body26_uvc[a, 2] >= 0.35 and body26_uvc[b, 2] >= 0.35:
            pa = (int(round(body26_uvc[a, 0])), int(round(body26_uvc[a, 1])))
            pb = (int(round(body26_uvc[b, 0])), int(round(body26_uvc[b, 1])))
            cv2.line(img, pa, pb, (205, 212, 45), 2, cv2.LINE_AA)
    for k in range(26):
        c = float(body26_uvc[k, 2])
        if c >= 0.35:
            pt = (int(round(body26_uvc[k, 0])), int(round(body26_uvc[k, 1])))
            col = (94, 197, 34) if c >= 0.7 else (11, 158, 245)
            cv2.circle(img, pt, 3, col, -1, cv2.LINE_AA)
            cv2.circle(img, pt, 4, (255, 255, 255), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes() if ok else b""


def _mediapipe_world33_to_body26_zup(world_landmarks) -> Tuple[np.ndarray, np.ndarray]:
    """Convert MediaPipe 3D pose_world_landmarks (meters, Y-down) to Nuranova Z-Up Body26."""
    pts = np.full((26, 3), np.nan, dtype=np.float64)
    conf = np.zeros(26, dtype=np.float64)
    mp_map = {
        0: 0, 1: 2, 2: 5, 3: 7, 4: 8,
        5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16,
        11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28,
        20: 31, 21: 32, 24: 29, 25: 30,
    }
    for dst, src in mp_map.items():
        lm = world_landmarks[src]
        # MediaPipe world: +X right (actor's left), +Y down, +Z toward camera
        # Nuranova Z-up facing +Y: X = -lm.x, Y = -lm.z, Z = -lm.y
        pts[dst] = [-float(lm.x), -float(lm.z), -float(lm.y)]
        conf[dst] = float(np.clip(getattr(lm, "visibility", 0.75), 0.0, 1.0))

    pts[18] = 0.5 * (pts[5] + pts[6])
    conf[18] = min(conf[5], conf[6])
    pts[19] = 0.5 * (pts[11] + pts[12])
    conf[19] = min(conf[11], conf[12])
    ear_mid = 0.5 * (pts[3] + pts[4])
    pts[17] = ear_mid + 0.35 * (ear_mid - pts[18])
    conf[17] = min(conf[3], conf[4])
    pts[22], conf[22] = pts[20].copy(), conf[20]
    pts[23], conf[23] = pts[21].copy(), conf[21]

    # Ground feet cleanly on Z = 0.04m so mannequin stands naturally on the studio floor
    foot_zs = [pts[i, 2] for i in (15, 16, 20, 21, 24, 25) if conf[i] > 0.25 and np.isfinite(pts[i, 2])]
    min_z = min(foot_zs) if foot_zs else -0.92
    pts[:, 2] -= (min_z - 0.04)
    return pts, conf


def generate_live_procedural_pose(phase_s: float, actor: ActorProfile) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a smooth live animated character pose when previewing without a live video stream."""
    ph = phase_s * 2.4
    lq = np.tile(quat_identity(), (len(BONE_NAMES), 1))
    root_t = np.array([0.0, 0.0, 0.95 + 0.022 * math.sin(2.0 * ph)], dtype=np.float64)
    lq[BONE_INDEX["hips"]] = Rotation.from_euler("z", 8.0 * math.sin(ph * 0.5), degrees=True).as_quat()
    lq[BONE_INDEX["chest"]] = Rotation.from_euler("zx", [12.0 * math.sin(ph), 5.0 * math.cos(ph)], degrees=True).as_quat()
    lq[BONE_INDEX["head"]] = Rotation.from_euler("z", -10.0 * math.sin(ph * 0.5), degrees=True).as_quat()
    lq[BONE_INDEX["left_upper_arm"]] = Rotation.from_euler("yz", [54.0, 26.0 * math.sin(ph)], degrees=True).as_quat()
    lq[BONE_INDEX["left_forearm"]] = Rotation.from_euler("z", -35.0 - 16.0 * math.sin(ph), degrees=True).as_quat()
    lq[BONE_INDEX["right_upper_arm"]] = Rotation.from_euler("yz", [-54.0, -26.0 * math.sin(ph)], degrees=True).as_quat()
    lq[BONE_INDEX["right_forearm"]] = Rotation.from_euler("z", 35.0 + 16.0 * math.sin(ph), degrees=True).as_quat()
    lq[BONE_INDEX["left_upper_leg"]] = Rotation.from_euler("x", 22.0 * math.sin(ph), degrees=True).as_quat()
    lq[BONE_INDEX["left_lower_leg"]] = Rotation.from_euler("x", -max(4.0, 42.0 * math.sin(ph)), degrees=True).as_quat()
    lq[BONE_INDEX["right_upper_leg"]] = Rotation.from_euler("x", -22.0 * math.sin(ph), degrees=True).as_quat()
    lq[BONE_INDEX["right_lower_leg"]] = Rotation.from_euler("x", -max(4.0, -42.0 * math.sin(ph)), degrees=True).as_quat()
    pos, _ = compute_forward_kinematics_frame(root_t, lq, actor.rest_offsets_m)
    conf = np.full(len(BONE_NAMES), 0.95, dtype=np.float32)
    return pos.astype(np.float32), conf


class LiveMocapPreviewWorker(QObject):
    """Background live pose estimator feeding real-time 3D character poses & 2D overlays to Capture Studio."""

    pose_updated = Signal(object, object, str, float)  # (positions (B,3), conf (B,), source_label, fps)
    overlay_updated = Signal(str, bytes)  # (device_id, jpeg_bytes_with_skeleton)

    def __init__(
        self,
        camera_manager: CameraManager,
        get_calibration,
        get_actor,
    ) -> None:
        super().__init__()
        self.camera_manager = camera_manager
        self._get_calibration = get_calibration
        self._get_actor = get_actor

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.running: bool = False
        self.simulate_when_no_frames: bool = False
        self._last_smoothed_pos: Optional[np.ndarray] = None
        self._last_jpeg_hash: Dict[str, int] = {}

    def start(self, simulate_when_no_frames: bool = False) -> None:
        self.simulate_when_no_frames = simulate_when_no_frames
        if self.running and self._thread is not None and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, name="nuranova-live-mocap", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        self.running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None
        self._last_smoothed_pos = None

    def _run_loop(self) -> None:
        mp_pose = None
        mp_init_attempted = False

        t_start = time.perf_counter()
        last_emit_t = t_start
        ema_fps = 0.0

        try:
            while not self._stop_evt.is_set():
                loop_t0 = time.perf_counter()
                actor: ActorProfile = self._get_actor()
                calib: Optional[MultiCameraCalibration] = self._get_calibration()
                cams = self.camera_manager.list_cameras()
                streaming_cams = [c for c in cams if c.latest_preview_jpeg]

                if streaming_cams and not mp_init_attempted:
                    mp_init_attempted = True
                    try:
                        import mediapipe as mp

                        mp_pose = mp.solutions.pose.Pose(
                            static_image_mode=False,
                            model_complexity=0,  # Fastest live preview (~20-30 FPS on CPU)
                            smooth_landmarks=True,
                            enable_segmentation=False,
                            min_detection_confidence=0.45,
                            min_tracking_confidence=0.45,
                        )
                    except Exception:
                        mp_pose = None

                if streaming_cams and mp_pose is not None:
                    obs_2d_by_uid: Dict[str, np.ndarray] = {}
                    obs_size_by_uid: Dict[str, Tuple[int, int]] = {}
                    single_world_pts: Optional[np.ndarray] = None
                    single_world_conf: Optional[np.ndarray] = None

                    for cam in streaming_cams[:4]:  # Cap live preview tracking at 4 views for snappy FPS
                        jpeg = cam.latest_preview_jpeg
                        if not jpeg:
                            continue
                        h_val = hash(jpeg)
                        arr = np.frombuffer(jpeg, dtype=np.uint8)
                        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if bgr is None:
                            continue
                        h_img, w_img = bgr.shape[:2]
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        res = mp_pose.process(rgb)
                        if res.pose_landmarks:
                            body26_2d = _mediapipe33_to_body26(res.pose_landmarks.landmark, w_img, h_img)
                            obs_2d_by_uid[cam.camera_uid] = body26_2d
                            obs_size_by_uid[cam.camera_uid] = (w_img, h_img)
                            if self._last_jpeg_hash.get(cam.device_id) != h_val:
                                self._last_jpeg_hash[cam.device_id] = h_val
                                ov = draw_body26_overlay_bgr(bgr, body26_2d)
                                if ov:
                                    self.overlay_updated.emit(cam.device_id, ov)
                            if single_world_pts is None and res.pose_world_landmarks:
                                single_world_pts, single_world_conf = _mediapipe_world33_to_body26_zup(
                                    res.pose_world_landmarks.landmark
                                )

                    body26_3d: Optional[np.ndarray] = None
                    body26_conf: Optional[np.ndarray] = None
                    src_label = ""

                    # Path A: Multi-camera calibrated live triangulation
                    if calib is not None and len(obs_2d_by_uid) >= 2:
                        matched_uids = [u for u in obs_2d_by_uid if u in calib.cameras]
                        if len(matched_uids) >= 2:
                            pts3 = np.full((26, 3), np.nan, dtype=np.float64)
                            confs3 = np.zeros(26, dtype=np.float64)
                            for k in range(26):
                                obs_list = []
                                for uid in matched_uids:
                                    u, v, c = obs_2d_by_uid[uid][k]
                                    if c >= 0.35:
                                        ci = calib.cameras[uid].intrinsics
                                        pw, ph = obs_size_by_uid[uid]
                                        sx = ci.image_width / max(float(pw), 1.0)
                                        sy = ci.image_height / max(float(ph), 1.0)
                                        obs_list.append(Observation2D(uid, float(u) * sx, float(v) * sy, float(c)))
                                if len(obs_list) >= 2:
                                    tr = triangulate_point_robust(
                                        calib.cameras,
                                        obs_list,
                                        max_reproj_error_px=25.0,
                                        refine_nonlinear=False,
                                    )
                                    if tr.valid:
                                        pts3[k] = tr.point_3d
                                        confs3[k] = tr.confidence
                            if int(np.sum(confs3 > 0.25)) >= 8:
                                body26_3d, body26_conf = pts3, confs3
                                src_label = f"Multi-Cam 3D ({len(matched_uids)} cams)"

                    # Path B: Single-phone (or uncalibrated) live 3D MediaPipe World Pose -> Articulated Skeleton
                    if body26_3d is None and single_world_pts is not None and single_world_conf is not None:
                        body26_3d, body26_conf = single_world_pts, single_world_conf
                        src_label = f"Live Phone AI ({streaming_cams[0].camera_uid})"

                    if body26_3d is not None and body26_conf is not None:
                        solver = ArticulatedSkeletonSolver(
                            actor=actor,
                            config=SolverConfig(enforce_anatomical_limits=True, enable_nonlinear_refinement=False),
                        )
                        trk = solver.solve_sequence(
                            body26_3d[None, :, :].astype(np.float32),
                            body26_conf[None, :].astype(np.float32),
                        )
                        pos = trk.joint_positions[0].astype(np.float32)
                        conf = trk.joint_confidence[0].astype(np.float32)
                        if self._last_smoothed_pos is not None and self._last_smoothed_pos.shape == pos.shape:
                            pos = (0.55 * pos + 0.45 * self._last_smoothed_pos).astype(np.float32)
                        self._last_smoothed_pos = pos.copy()

                        now = time.perf_counter()
                        dt = max(now - last_emit_t, 1e-3)
                        ema_fps = 0.25 * (1.0 / dt) + 0.75 * ema_fps if ema_fps > 0 else (1.0 / dt)
                        last_emit_t = now
                        self.pose_updated.emit(pos, conf, src_label, float(ema_fps))
                elif self.simulate_when_no_frames:
                    # Live simulated character preview when starting mocap with simulated/no streams
                    now = time.perf_counter()
                    pos, conf = generate_live_procedural_pose(now - t_start, actor)
                    dt = max(now - last_emit_t, 1e-3)
                    ema_fps = 0.25 * (1.0 / dt) + 0.75 * ema_fps if ema_fps > 0 else (1.0 / dt)
                    last_emit_t = now
                    self.pose_updated.emit(pos, conf, "Live Demo Character", float(ema_fps))

                elapsed = time.perf_counter() - loop_t0
                sleep_s = max(0.03, 0.065 - elapsed)  # ~15 FPS smooth UI updates without hogging CPU
                self._stop_evt.wait(sleep_s)
        finally:
            if mp_pose is not None:
                try:
                    mp_pose.close()
                except Exception:
                    pass
            self.running = False
