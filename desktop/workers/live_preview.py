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
from engine.hands.hand_tracker import HAND_BONE_NAMES, HandCaptureEngine
from engine.math_core.camera_model import MultiCameraCalibration
from engine.math_core.transforms import quat_identity
from engine.skeleton.rig_definition import (
    BONE_INDEX,
    BONE_NAMES,
    PARENT_INDICES,
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


def _mediapipe_world33_to_body26_zup(world_landmarks, *, ground_feet=True) -> Tuple[np.ndarray, np.ndarray]:
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
    if ground_feet:
        pts[:, 2] -= (min_z - 0.04)
    return pts, conf


class SingleCamRootTracker:
    """Monocular 3D root translation estimator for Single-Camera Mode (left/right, forward/back & jump Z).

    Combines 2D normalized image landmarks with hip-centered 3D metric world landmarks using a
    typical smartphone pinhole FOV (~62 deg horizontal) to estimate smooth metric (X, Y, Z)
    studio-floor movement without needing multi-camera calibration.
    """

    def __init__(self, enabled: bool = True, max_radius_m: float = 2.8) -> None:
        self.enabled: bool = bool(enabled)
        self.max_radius_m: float = float(max_radius_m)
        self._ref_depth_m: Optional[float] = None
        self._ref_foot_y_norm: Optional[float] = None
        self._ema_depth_m: Optional[float] = None
        self._ema_xy: Optional[np.ndarray] = None
        self._ema_jump_z: float = 0.0

    def reset(self) -> None:
        self._ref_depth_m = None
        self._ref_foot_y_norm = None
        self._ema_depth_m = None
        self._ema_xy = None
        self._ema_jump_z = 0.0

    def apply(
        self,
        body26_zup: np.ndarray,
        conf: np.ndarray,
        body26_2d_uvc: np.ndarray,
        img_w: int,
        img_h: int,
    ) -> np.ndarray:
        """Apply estimated 3D studio-floor (X, Y, jump-Z) translation onto hip-centered Z-Up Body26 points."""
        pts = body26_zup.copy()
        if not self.enabled or img_w <= 0 or img_h <= 0:
            return pts

        # Typical rear smartphone wide lens (~62 deg horizontal FOV -> f_px ~= 0.84 * max(w, h))
        f_px = 0.84 * float(max(img_w, img_h))

        # 1. Estimate metric distance from camera using visible 3D vs 2D anatomical segment lengths
        segment_pairs = [
            (5, 11),   # left shoulder -> left hip
            (6, 12),   # right shoulder -> right hip
            (18, 19),  # neck -> pelvis center
            (11, 13),  # left thigh
            (12, 14),  # right thigh
            (13, 15),  # left shin
            (14, 16),  # right shin
        ]
        depth_samples = []
        for a, b in segment_pairs:
            if conf[a] >= 0.45 and conf[b] >= 0.45 and body26_2d_uvc[a, 2] >= 0.45 and body26_2d_uvc[b, 2] >= 0.45:
                if np.all(np.isfinite(pts[a])) and np.all(np.isfinite(pts[b])):
                    l3d = float(np.linalg.norm(pts[a] - pts[b]))
                    l2d = float(np.linalg.norm(body26_2d_uvc[a, :2] - body26_2d_uvc[b, :2]))
                    if l3d >= 0.18 and l2d >= 12.0:
                        d_est = f_px * (l3d / l2d)
                        if 0.9 <= d_est <= 6.5:
                            depth_samples.append(d_est)

        if depth_samples:
            raw_depth = float(np.clip(np.median(depth_samples), 1.1, 5.5))
        else:
            raw_depth = self._ema_depth_m if self._ema_depth_m is not None else 2.6

        if self._ema_depth_m is None:
            self._ema_depth_m = raw_depth
        else:
            # Gentle EMA on depth prevents monocular forward/back jitter while tracking real walks cleanly
            self._ema_depth_m = 0.18 * raw_depth + 0.82 * self._ema_depth_m

        if self._ref_depth_m is None:
            self._ref_depth_m = self._ema_depth_m

        depth_m = self._ema_depth_m

        # 2. Back-project 2D hip center (Body26 index 19) to studio X (left/right) and Y (forward/back)
        if body26_2d_uvc[19, 2] >= 0.30:
            hx_px = float(body26_2d_uvc[19, 0])
        elif body26_2d_uvc[18, 2] >= 0.30:
            hx_px = float(body26_2d_uvc[18, 0])
        else:
            hx_px = 0.5 * float(img_w)

        # Sign matches _mediapipe_world33_to_body26_zup: X = -lm_world.x, Y = -lm_world.z
        raw_x = -((hx_px - 0.5 * float(img_w)) / f_px) * depth_m
        raw_y = float(depth_m - self._ref_depth_m)
        raw_xy = np.clip(np.array([raw_x, raw_y], dtype=np.float64), -self.max_radius_m, self.max_radius_m)

        if self._ema_xy is None:
            self._ema_xy = raw_xy
        else:
            # Responsive X (side-to-side), smooth Y (depth)
            self._ema_xy[0] = 0.38 * raw_xy[0] + 0.62 * self._ema_xy[0]
            self._ema_xy[1] = 0.22 * raw_xy[1] + 0.78 * self._ema_xy[1]

        # 3. Jump-aware vertical lift (keeps feet grounded during walking, lifts smoothly when both feet rise!)
        foot_ys = [
            float(body26_2d_uvc[i, 1]) / float(img_h)
            for i in (15, 16, 20, 21, 24, 25)
            if body26_2d_uvc[i, 2] >= 0.45
        ]
        if foot_ys:
            lowest_foot_y_norm = max(foot_ys)  # larger y = closer to bottom of image (floor)
            if self._ref_foot_y_norm is None:
                self._ref_foot_y_norm = lowest_foot_y_norm
            elif lowest_foot_y_norm > self._ref_foot_y_norm:
                # Slowly adapt floor baseline downward if feet step closer to bottom
                self._ref_foot_y_norm = 0.25 * lowest_foot_y_norm + 0.75 * self._ref_foot_y_norm
            else:
                # Gently drift baseline back toward standing foot level so camera tilt doesn't stick
                self._ref_foot_y_norm = 0.02 * lowest_foot_y_norm + 0.98 * self._ref_foot_y_norm

            lift_norm = max(0.0, (self._ref_foot_y_norm - lowest_foot_y_norm) - 0.035)
            raw_jump_z = float(np.clip((lift_norm * float(img_h) / f_px) * depth_m, 0.0, 0.85))
            self._ema_jump_z = 0.35 * raw_jump_z + 0.65 * self._ema_jump_z
        else:
            self._ema_jump_z *= 0.8

        pts[:, 0] += self._ema_xy[0]
        pts[:, 1] += self._ema_xy[1]
        pts[:, 2] += self._ema_jump_z
        return pts


def generate_live_procedural_frame(
    phase_s: float,
    actor: ActorProfile,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a smooth live animated body + 30-finger pose for Desktop & Blender Live Preview."""
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

    # Subtle natural procedural finger curl wave (16 bones per hand: wrist + 5x3 phalanges)
    l_hand = np.tile(quat_identity().astype(np.float32), (16, 1))
    r_hand = np.tile(quat_identity().astype(np.float32), (16, 1))
    for f_i in range(5):
        curl_l = 14.0 + 18.0 * (0.5 + 0.5 * math.sin(ph + f_i * 0.4))
        curl_r = 14.0 + 18.0 * (0.5 + 0.5 * math.cos(ph + f_i * 0.4))
        for seg in range(3):
            b_idx = 1 + f_i * 3 + seg
            l_hand[b_idx] = Rotation.from_euler("x", curl_l * (0.65 if seg == 0 else 1.0), degrees=True).as_quat()
            r_hand[b_idx] = Rotation.from_euler("x", curl_r * (0.65 if seg == 0 else 1.0), degrees=True).as_quat()

    return pos.astype(np.float32), conf, root_t.astype(np.float32), lq.astype(np.float32), l_hand, r_hand


def generate_live_procedural_pose(phase_s: float, actor: ActorProfile) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a smooth live animated character pose when previewing without a live video stream."""
    pos, conf, _, _, _, _ = generate_live_procedural_frame(phase_s, actor)
    return pos, conf


class LiveMocapPreviewWorker(QObject):
    """Background live body + finger pose estimator feeding real-time 3D poses to Capture Studio & Blender."""

    pose_updated = Signal(object, object, str, float)  # (positions (B,3), conf (B,), source_label, fps)
    live_frame_updated = Signal(dict)  # Full JSON-serializable live body + 30-finger frame for Blender Bridge!
    overlay_updated = Signal(str, bytes)  # (device_id, jpeg_bytes_with_skeleton)

    def __init__(
        self,
        camera_manager: CameraManager,
        get_calibration,
        get_actor,
        get_single_cam_root_motion=None,
        get_enable_hands=None,
    ) -> None:
        super().__init__()
        self.camera_manager = camera_manager
        self._get_calibration = get_calibration
        self._get_actor = get_actor
        self._get_single_cam_root_motion = get_single_cam_root_motion or (lambda: True)
        self._get_enable_hands = get_enable_hands or (lambda: True)

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.running: bool = False
        self.simulate_when_no_frames: bool = False
        self._last_smoothed_pos: Optional[np.ndarray] = None
        self._last_jpeg_hash: Dict[str, int] = {}
        self._root_tracker = SingleCamRootTracker(enabled=True)
        self._last_left_hand: np.ndarray = np.tile(quat_identity().astype(np.float32), (16, 1))
        self._last_right_hand: np.ndarray = np.tile(quat_identity().astype(np.float32), (16, 1))

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
        self._root_tracker.reset()

    def reset_root_origin(self) -> None:
        """Re-center the Single-Cam 3D walking origin to where the actor is standing right now."""
        self._root_tracker.reset()

    def _build_live_blender_dict(
        self,
        actor: ActorProfile,
        root_t: np.ndarray,
        local_q: np.ndarray,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
        left_detected: bool,
        right_detected: bool,
        src_label: str,
        fps: float,
    ) -> dict:
        return {
            "source": src_label,
            "fps": round(float(fps), 1),
            "skeleton": {
                "bone_names": list(BONE_NAMES),
                "parent_indices": list(PARENT_INDICES),
                "rest_offsets": np.asarray(actor.rest_offsets_m, dtype=np.float64).tolist(),
                "hand_bone_names": list(HAND_BONE_NAMES),
            },
            "root_translation": np.asarray(root_t, dtype=np.float64).reshape(3).tolist(),
            "local_rotations_xyzw": np.asarray(local_q, dtype=np.float64).reshape(-1, 4).tolist(),
            "hands": {
                "left_local_rotations_xyzw": np.asarray(left_hand, dtype=np.float64).reshape(16, 4).tolist(),
                "right_local_rotations_xyzw": np.asarray(right_hand, dtype=np.float64).reshape(16, 4).tolist(),
                "left_detected": bool(left_detected),
                "right_detected": bool(right_detected),
            },
        }

    def _run_loop(self) -> None:
        mp_pose = None
        mp_init_attempted = False
        hand_engine: Optional[HandCaptureEngine] = None
        hand_init_attempted = False

        t_start = time.perf_counter()
        last_emit_t = t_start
        ema_fps = 0.0

        try:
            while not self._stop_evt.is_set():
                loop_t0 = time.perf_counter()
                actor: ActorProfile = self._get_actor()
                calib: Optional[MultiCameraCalibration] = self._get_calibration()
                enable_hands = bool(self._get_enable_hands())
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

                if streaming_cams and enable_hands and not hand_init_attempted:
                    hand_init_attempted = True
                    try:
                        hand_engine = HandCaptureEngine(max_num_hands=2, model_complexity=0)
                    except Exception:
                        hand_engine = None

                if streaming_cams and mp_pose is not None:
                    obs_2d_by_uid: Dict[str, np.ndarray] = {}
                    obs_size_by_uid: Dict[str, Tuple[int, int]] = {}
                    single_world_pts: Optional[np.ndarray] = None
                    single_world_conf: Optional[np.ndarray] = None
                    left_det = False
                    right_det = False
                    primary_bgr: Optional[np.ndarray] = None

                    for cam in streaming_cams[:4]:  # Cap live preview tracking at 4 views for snappy FPS
                        jpeg = cam.latest_preview_jpeg
                        if not jpeg:
                            continue
                        h_val = hash(jpeg)
                        arr = np.frombuffer(jpeg, dtype=np.uint8)
                        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if bgr is None:
                            continue
                        if primary_bgr is None:
                            primary_bgr = bgr
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
                                w_pts, w_conf = _mediapipe_world33_to_body26_zup(
                                    res.pose_world_landmarks.landmark
                                )
                                self._root_tracker.enabled = bool(self._get_single_cam_root_motion())
                                single_world_pts = self._root_tracker.apply(w_pts, w_conf, body26_2d, w_img, h_img)
                                single_world_conf = w_conf

                    # Real-time 30-finger capture (left + right hands) from primary live camera!
                    if enable_hands and hand_engine is not None and primary_bgr is not None:
                        try:
                            h_res = hand_engine.infer_frame(primary_bgr)
                            if "left" in h_res:
                                self._last_left_hand = h_res["left"].local_rotations_xyzw.astype(np.float32)
                                left_det = True
                            if "right" in h_res:
                                self._last_right_hand = h_res["right"].local_rotations_xyzw.astype(np.float32)
                                right_det = True
                        except Exception:
                            pass

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
                        if left_det or right_det:
                            src_label += " + Fingers"
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
                        self.live_frame_updated.emit(
                            self._build_live_blender_dict(
                                actor=actor,
                                root_t=trk.root_translation[0],
                                local_q=trk.local_rotations_xyzw[0],
                                left_hand=self._last_left_hand,
                                right_hand=self._last_right_hand,
                                left_detected=left_det,
                                right_detected=right_det,
                                src_label=src_label,
                                fps=ema_fps,
                            )
                        )
                elif self.simulate_when_no_frames:
                    # Live simulated body + fingers preview when testing without a live video stream
                    now = time.perf_counter()
                    pos, conf, root_t, lq, lh, rh = generate_live_procedural_frame(now - t_start, actor)
                    dt = max(now - last_emit_t, 1e-3)
                    ema_fps = 0.25 * (1.0 / dt) + 0.75 * ema_fps if ema_fps > 0 else (1.0 / dt)
                    last_emit_t = now
                    self.pose_updated.emit(pos, conf, "Live Demo Character + Fingers", float(ema_fps))
                    self.live_frame_updated.emit(
                        self._build_live_blender_dict(
                            actor=actor,
                            root_t=root_t,
                            local_q=lq,
                            left_hand=lh,
                            right_hand=rh,
                            left_detected=True,
                            right_detected=True,
                            src_label="Live Demo Character + Fingers",
                            fps=ema_fps,
                        )
                    )

                elapsed = time.perf_counter() - loop_t0
                sleep_s = max(0.03, 0.065 - elapsed)  # ~15 FPS smooth UI updates without hogging CPU
                self._stop_evt.wait(sleep_s)
        finally:
            if mp_pose is not None:
                try:
                    mp_pose.close()
                except Exception:
                    pass
            if hand_engine is not None:
                try:
                    hand_engine.close()
                except Exception:
                    pass
            self.running = False
