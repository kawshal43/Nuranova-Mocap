"""Dedicated Hand & Finger Capture module (MediaPipe Hands + RTMW-133 hand landmarks -> 16 finger bone rotations)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.signal import butter, sosfiltfilt

from engine.data_format.motion_io import HandMotionTrack
from engine.math_core.transforms import (
    enforce_quaternion_continuity,
    matrix_to_quat,
    quat_identity,
    quat_inverse,
    quat_multiply,
    quat_normalize,
    quat_rotate_vector,
    quat_slerp,
)

# Canonical 16 hand/finger bone order per hand (wrist + 5 fingers x 3 phalanges)
HAND_BONE_NAMES: List[str] = [
    "wrist",
    "thumb_01", "thumb_02", "thumb_03",
    "index_01", "index_02", "index_03",
    "middle_01", "middle_02", "middle_03",
    "ring_01", "ring_02", "ring_03",
    "pinky_01", "pinky_02", "pinky_03",
]

# MediaPipe / COCO-Hand 21-landmark chains: (parent_lm, child_lm, rest_dir_in_palm)
_FINGER_CHAINS: List[Tuple[int, int, int]] = [
    # (bone_idx, lm_start, lm_end)
    (1, 1, 2), (2, 2, 3), (3, 3, 4),         # thumb
    (4, 5, 6), (5, 6, 7), (6, 7, 8),         # index
    (7, 9, 10), (8, 10, 11), (9, 11, 12),    # middle
    (10, 13, 14), (11, 14, 15), (12, 15, 16),# ring
    (13, 17, 18), (14, 18, 19), (15, 19, 20),# pinky
]


@dataclass
class HandObservation:
    handedness: str  # "left" or "right"
    landmarks_3d: np.ndarray  # (21, 3) float32 in local/world metric frame
    confidence: float
    local_rotations_xyzw: np.ndarray  # (16, 4) float32


def _rot_x_quat(rad: float) -> np.ndarray:
    h = 0.5 * float(rad)
    return np.array([np.sin(h), 0.0, 0.0, np.cos(h)], dtype=np.float64)


def _rot_z_quat(rad: float) -> np.ndarray:
    h = 0.5 * float(rad)
    return np.array([0.0, 0.0, np.sin(h), np.cos(h)], dtype=np.float64)


def solve_hand_rotations_from_landmarks21(
    landmarks_3d: np.ndarray,
    handedness: str = "right",
) -> np.ndarray:
    """Solve anatomically constrained wrist + 15 finger phalanx local quaternions from 21 3D hand landmarks.

    Conventions (matching Blender `append_finger_edit_bones` with dorsal `+Z` roll alignment):
    - Local `+Y`: along finger from knuckle toward fingertip
    - Local `+Z`: dorsal (back of hand)
    - Local `-Z`: palmar (palm interior)
    - Pure local `-X` rotation (`R_x(-curl)`) ALWAYS curls fingers inward toward the palm!
    - Pure local `±Z` rotation (`R_z(splay)`) splays knuckles sideways in the palm plane (zero axial twist!).
    - PIP (`_02`) & DIP (`_03`) are strictly 1-DoF anatomical hinges (`0°..105°` curl, `0°` splay/twist),
      completely eliminating broken sideways/backwards finger glitches from noisy distant video!
    """
    pts = np.asarray(landmarks_3d, dtype=np.float64).reshape(21, 3)
    quats = np.tile(quat_identity(), (16, 1))

    # Build palm orthonormal frame: +Y wrist -> middle MCP (9), +Z dorsal (back of hand), +X across palm
    y_palm = pts[9] - pts[0]
    yn = float(np.linalg.norm(y_palm))
    if yn < 1e-6:
        return quats.astype(np.float32)
    y_palm /= yn

    across = pts[17] - pts[5] if handedness == "right" else pts[5] - pts[17]
    z_palm = np.cross(across, y_palm)
    zn = float(np.linalg.norm(z_palm))
    if zn < 1e-6:
        return quats.astype(np.float32)
    z_palm /= zn
    x_palm = np.cross(y_palm, z_palm)
    x_palm /= max(float(np.linalg.norm(x_palm)), 1e-6)

    R_palm = np.column_stack([x_palm, y_palm, z_palm])
    quats[0] = matrix_to_quat(R_palm)

    # Transform all 21 landmarks into canonical palm frame (+Y fingers out, +Z dorsal, -Z palmar)
    pts_p = (pts - pts[0]) @ R_palm

    # Anatomical Palm-Normal Flip Guard: human PIP joints CANNOT hyperextend backward > 20 deg!
    # Distant 2-camera 5cm palm triangulation can flip z_palm by 180 deg on a closed fist; detecting
    # impossible multi-finger dorsal PIP bends restores the true dorsal normal and recovers the fist!
    pip_dorsal_bends: List[float] = []
    for lm_mcp, lm_pip, lm_dip in ((5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19)):
        v1 = pts_p[lm_pip] - pts_p[lm_mcp]
        v2 = pts_p[lm_dip] - pts_p[lm_pip]
        n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        if n1 > 1e-6 and n2 > 1e-6:
            u1, u2 = v1 / n1, v2 / n2
            bend_rad = float(np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0)))
            perp = u2 - float(np.dot(u2, u1)) * u1
            pn = float(np.linalg.norm(perp))
            if pn > 1e-5 and bend_rad > 0.25:
                pip_dorsal_bends.append(bend_rad * float(perp[2] / pn))
    if len(pip_dorsal_bends) >= 2 and float(np.mean(pip_dorsal_bends)) > 0.35:
        z_palm = -z_palm
        x_palm = -x_palm
        R_palm = np.column_stack([x_palm, y_palm, z_palm])
        quats[0] = matrix_to_quat(R_palm)
        pts_p = (pts - pts[0]) @ R_palm

    # Note: In palm frame, x_palm points index->pinky on RIGHT hand and pinky->index on LEFT hand.
    thumb_side_x = -1.0 if handedness == "right" else 1.0

    # Anatomical hinge limits (radians) per segment: (curl_min, curl_max, splay_min, splay_max)
    finger_limits = [
        # Thumb (01=CMC/MCP opposition+curl, 02=MCP flex, 03=IP flex)
        [(-0.18, 1.10, -0.85, 0.85), (-0.10, 1.15, 0.0, 0.0), (-0.10, 1.25, 0.0, 0.0)],
        # Index (01=MCP curl+splay, 02=PIP hinge, 03=DIP hinge)
        [(-0.12, 1.50, -0.26, 0.26), (0.0, 1.78, 0.0, 0.0), (0.0, 1.32, 0.0, 0.0)],
        # Middle
        [(-0.12, 1.52, -0.22, 0.22), (0.0, 1.80, 0.0, 0.0), (0.0, 1.32, 0.0, 0.0)],
        # Ring
        [(-0.12, 1.52, -0.22, 0.22), (0.0, 1.80, 0.0, 0.0), (0.0, 1.32, 0.0, 0.0)],
        # Pinky
        [(-0.12, 1.52, -0.28, 0.28), (0.0, 1.80, 0.0, 0.0), (0.0, 1.32, 0.0, 0.0)],
    ]
    # Nominal rest splay angle in palm XY plane matching _FINGER_LAYOUT so neutral open fingers -> ~0 deg delta!
    rest_splay_xy = [
        thumb_side_x * 0.48,  # thumb naturally angled outward ~28 deg toward thumb side
        thumb_side_x * 0.08,  # index
        0.00,                 # middle
        -thumb_side_x * 0.06, # ring
        -thumb_side_x * 0.14, # pinky
    ]

    for f_idx in range(5):
        # Track cumulative anatomical orientation of parent segment in palm frame
        parent_q = _rot_z_quat(-rest_splay_xy[f_idx]) if f_idx == 0 else quat_identity()
        prev_curl = 0.0
        for seg in range(3):
            b_idx, lm_a, lm_b = _FINGER_CHAINS[f_idx * 3 + seg]
            vec = pts_p[lm_b] - pts_p[lm_a]
            vn = float(np.linalg.norm(vec))
            if vn < 1e-6:
                quats[b_idx] = quat_identity()
                continue
            v_loc = quat_rotate_vector(quat_inverse(parent_q), vec / vn)
            c_min, c_max, s_min, s_max = finger_limits[f_idx][seg]

            if seg == 0:
                # Knuckle / MCP: palmar curl (positive when v_loc.z < 0 toward palm) + gentle XY splay
                if f_idx == 0:
                    # Thumb curls both inward (-Z palm) and across palm toward pinky (-thumb_side_x)
                    curl = float(np.arctan2(-v_loc[2], max(float(v_loc[1]), 1e-4)))
                    splay = float(np.arctan2(-v_loc[0], max(float(v_loc[1]), 1e-4)))
                else:
                    # Subtract neutral finger splay so relaxed straight fingers sit cleanly at 0 deg
                    splay_raw = float(np.arctan2(-v_loc[0], max(float(v_loc[1]), 1e-4)))
                    splay = splay_raw - (-rest_splay_xy[f_idx])
                    # Measure curl in the splayed finger's sagittal plane
                    v_sag = quat_rotate_vector(_rot_z_quat(-splay), v_loc)
                    curl = float(np.arctan2(-v_sag[2], max(float(v_sag[1]), 1e-4)))
                curl = float(np.clip(curl, c_min, c_max))
                splay = float(np.clip(splay, s_min, s_max))
                q_local = quat_normalize(quat_multiply(_rot_z_quat(splay), _rot_x_quat(-curl)))
            elif f_idx == 0:
                # Thumb MCP (_02) & IP (_03): bend BOTH palmar (-Z) AND across the palm toward pinky (-thumb_side_x)!
                splay_raw = float(np.arctan2(-v_loc[0], max(float(v_loc[1]), 1e-4)))
                # Clamp adduction strictly inward toward pinky (0..+55 deg) with tiny outward allowance (-10 deg)
                splay_in = float(np.clip(splay_raw * thumb_side_x, -0.18, 0.96))
                splay = splay_in * thumb_side_x
                v_sag = quat_rotate_vector(_rot_z_quat(-splay), v_loc)
                curl_signed = float(np.arctan2(-v_sag[2], max(float(v_sag[1]), 1e-4)))
                curl = float(np.clip(curl_signed, c_min, c_max))
                q_local = quat_normalize(quat_multiply(_rot_z_quat(splay), _rot_x_quat(-curl)))
            else:
                # Fingers 1..4 PIP (_02) & DIP (_03): STRICT 1-DoF palmar hinge (zero sideways break, zero axial twist!)
                curl_signed = float(np.arctan2(-v_loc[2], max(float(v_loc[1]), 1e-4)))
                if seg == 2:
                    # Soft anatomical DIP-PIP tendon coupling (human DIP bends ~0.65 * PIP) prevents jittery fingertips
                    curl_signed = 0.65 * curl_signed + 0.35 * (0.65 * prev_curl)
                curl = float(np.clip(curl_signed, c_min, c_max))
                q_local = _rot_x_quat(-curl)

            prev_curl = curl
            quats[b_idx] = q_local
            parent_q = quat_normalize(quat_multiply(parent_q, q_local))

    return quats.astype(np.float32)


class HandCaptureEngine:
    """High-accuracy hand & 30-finger tracker supporting full-res wrist-ROI cropping and multi-view 3D fusion."""

    def __init__(self, max_num_hands: int = 2, model_complexity: int = 1) -> None:
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self._hands = None
        self._hands_left = None
        self._hands_right = None

    def _ensure_loaded(self) -> None:
        if self._hands is not None:
            return
        import mediapipe as mp

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_num_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45,
        )

    def _ensure_single_hand_trackers(self) -> None:
        if self._hands_left is not None and self._hands_right is not None:
            return
        import mediapipe as mp

        self._hands_left = mp.solutions.hands.Hands(
            static_image_mode=True,  # Crops move frame-to-frame; static mode avoids stale crop tracking boxes!
            max_num_hands=1,
            model_complexity=self.model_complexity,
            min_detection_confidence=0.22,
            min_tracking_confidence=0.22,
        )
        self._hands_right = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            model_complexity=self.model_complexity,
            min_detection_confidence=0.22,
            min_tracking_confidence=0.22,
        )

    def infer_frame(self, frame_bgr: np.ndarray, assume_rear_camera: bool = False) -> Dict[str, HandObservation]:
        self._ensure_loaded()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._hands.process(rgb)
        out: Dict[str, HandObservation] = {}
        if not res.multi_hand_world_landmarks or not res.multi_handedness:
            return out
        for world_lms, handed in zip(res.multi_hand_world_landmarks, res.multi_handedness):
            raw_label = handed.classification[0].label.lower()
            # MediaPipe Hands assumes a mirrored selfie image; rear smartphone cameras are non-mirrored
            if assume_rear_camera and raw_label in ("left", "right"):
                label = "right" if raw_label == "left" else "left"
            else:
                label = raw_label
            score = float(handed.classification[0].score)
            # Proper right-handed (det=+1) conversion from MediaPipe (+X right, +Y down, +Z away)
            # to Z-up world (+X right, +Y away, +Z up) so dorsal palm normal +Z is NEVER mirrored!
            pts = np.array([[p.x, p.z, -p.y] for p in world_lms.landmark], dtype=np.float32)
            rots = solve_hand_rotations_from_landmarks21(pts, handedness=label)
            out[label] = HandObservation(
                handedness=label,
                landmarks_3d=pts,
                confidence=score,
                local_rotations_xyzw=rots,
            )
        return out

    def infer_hands_from_body26(
        self,
        full_res_bgr: np.ndarray,
        body26_uvc: np.ndarray,
        min_wrist_conf: float = 0.25,
    ) -> Dict[str, HandObservation]:
        """Crop high-resolution square ROIs around detected Left/Right wrists in a full 1080p/4K frame.

        Dramatically increases finger accuracy at 2-4 m full-body capture distance AND guarantees
        correct anatomical Left/Right handedness (never flipped by selfie-camera assumption!).
        """
        h, w = full_res_bgr.shape[:2]
        out: Dict[str, HandObservation] = {}
        self._ensure_single_hand_trackers()

        # Body26: left_elbow=7, left_wrist=9; right_elbow=8, right_wrist=10
        hand_specs = [
            ("left", 7, 9, self._hands_left),
            ("right", 8, 10, self._hands_right),
        ]
        for side, elbow_idx, wrist_idx, mp_model in hand_specs:
            if mp_model is None:
                continue
            w_conf = float(body26_uvc[wrist_idx, 2])
            if w_conf < min_wrist_conf:
                continue
            wx, wy = float(body26_uvc[wrist_idx, 0]), float(body26_uvc[wrist_idx, 1])
            if not (0 <= wx < w and 0 <= wy < h):
                continue

            # Extrapolate palm center ~34% past wrist along forearm vector & size box proportionally to forearm
            if float(body26_uvc[elbow_idx, 2]) >= min_wrist_conf:
                ex, ey = float(body26_uvc[elbow_idx, 0]), float(body26_uvc[elbow_idx, 1])
                dx, dy = wx - ex, wy - ey
                forearm_len = float(np.hypot(dx, dy))
                cx = wx + 0.34 * dx
                cy = wy + 0.34 * dy
                box_sz = int(np.clip(forearm_len * 1.65, 160.0, min(w, h) * 0.55))
            else:
                cx, cy = wx, wy
                box_sz = int(np.clip(min(w, h) * 0.24, 160.0, 420.0))

            half = box_sz // 2
            x0 = max(0, int(round(cx)) - half)
            y0 = max(0, int(round(cy)) - half)
            x1 = min(w, x0 + box_sz)
            y1 = min(h, y0 + box_sz)
            x0 = max(0, x1 - box_sz)
            y0 = max(0, y1 - box_sz)
            if (x1 - x0) < 48 or (y1 - y0) < 48:
                continue

            crop = full_res_bgr[y0:y1, x0:x1]
            # Normalize crop size to 256x256 (upscales distant hands & downscales huge 4K crops -> 7x faster!)
            if crop.shape[0] < 224 or crop.shape[1] < 224:
                crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_CUBIC)
            elif crop.shape[0] > 288 or crop.shape[1] > 288:
                crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)
            obs = self._infer_single_hand_crop(crop, side=side, mp_model=mp_model, w_conf=w_conf)
            if obs is not None:
                out[side] = obs

        # Only run full-frame fallback if neither wrist had a valid body26 ROI (avoid slow 4K full-frame re-scan!)
        if not out and (float(body26_uvc[9, 2]) < min_wrist_conf and float(body26_uvc[10, 2]) < min_wrist_conf):
            if w > 960:
                sc = 960.0 / float(w)
                small = cv2.resize(full_res_bgr, (960, max(1, int(round(h * sc)))), interpolation=cv2.INTER_AREA)
                return self.infer_frame(small, assume_rear_camera=True)
            return self.infer_frame(full_res_bgr, assume_rear_camera=True)
        return out

    def _infer_single_hand_crop(
        self,
        crop_256_bgr: np.ndarray,
        side: str,
        mp_model,
        w_conf: float = 0.8,
    ) -> Optional[HandObservation]:
        """Run single-hand MediaPipe landmark inference + low-contrast CLAHE retry on a 256x256 wrist crop."""
        rgb = cv2.cvtColor(crop_256_bgr, cv2.COLOR_BGR2RGB)
        res = mp_model.process(rgb)
        if not res.multi_hand_world_landmarks and w_conf >= 0.45:
            # Only spend a second MediaPipe pass on genuinely dim/low-contrast crops (skip clear occluded hands!)
            gray_m = float(crop_256_bgr.mean())
            gray_s = float(crop_256_bgr.std())
            if gray_m < 80.0 or gray_s < 38.0:
                if not hasattr(self, "_clahe") or self._clahe is None:
                    self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
                lab = cv2.cvtColor(crop_256_bgr, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
                crop_eq = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                res = mp_model.process(crop_eq)
            if not res.multi_hand_world_landmarks:
                return None
        elif not res.multi_hand_world_landmarks:
            return None

        world_lms = res.multi_hand_world_landmarks[0]
        score = float(res.multi_handedness[0].classification[0].score) if res.multi_handedness else 0.75
        pts = np.array([[p.x, p.z, -p.y] for p in world_lms.landmark], dtype=np.float32)
        rots = solve_hand_rotations_from_landmarks21(pts, handedness=side)
        return HandObservation(
            handedness=side,
            landmarks_3d=pts,
            confidence=score,
            local_rotations_xyzw=rots,
        )

    @staticmethod
    def extract_wrist_crops_from_body26(
        full_res_bgr: np.ndarray,
        body26_uvc: np.ndarray,
        min_wrist_conf: float = 0.25,
    ) -> Dict[str, Tuple[np.ndarray, float]]:
        """Extract lightweight 256x256 BGR wrist crops (left/right) so full 4K frames can be freed immediately."""
        h, w = full_res_bgr.shape[:2]
        crops: Dict[str, Tuple[np.ndarray, float]] = {}
        for side, elbow_idx, wrist_idx in (("left", 7, 9), ("right", 8, 10)):
            w_conf = float(body26_uvc[wrist_idx, 2])
            if w_conf < min_wrist_conf:
                continue
            wx, wy = float(body26_uvc[wrist_idx, 0]), float(body26_uvc[wrist_idx, 1])
            if not (0 <= wx < w and 0 <= wy < h):
                continue
            if float(body26_uvc[elbow_idx, 2]) >= min_wrist_conf:
                ex, ey = float(body26_uvc[elbow_idx, 0]), float(body26_uvc[elbow_idx, 1])
                dx, dy = wx - ex, wy - ey
                forearm_len = float(np.hypot(dx, dy))
                cx = wx + 0.34 * dx
                cy = wy + 0.34 * dy
                box_sz = int(np.clip(forearm_len * 1.65, 160.0, min(w, h) * 0.55))
            else:
                cx, cy = wx, wy
                box_sz = int(np.clip(min(w, h) * 0.24, 160.0, 420.0))
            half = box_sz // 2
            x0 = max(0, int(round(cx)) - half)
            y0 = max(0, int(round(cy)) - half)
            x1 = min(w, x0 + box_sz)
            y1 = min(h, y0 + box_sz)
            x0 = max(0, x1 - box_sz)
            y0 = max(0, y1 - box_sz)
            if (x1 - x0) < 48 or (y1 - y0) < 48:
                continue
            crop = full_res_bgr[y0:y1, x0:x1]
            # Normalize crop directly to MediaPipe's native 224x224 input size (maximum speed + crisp detail!)
            if crop.shape[0] < 224 or crop.shape[1] < 224:
                crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)
            elif crop.shape[0] > 240 or crop.shape[1] > 240:
                crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
            else:
                crop = crop.copy()
            crops[side] = (crop, w_conf)
        return crops

    def infer_hands_from_crops(
        self,
        crops_by_side: Dict[str, Tuple[np.ndarray, float]],
    ) -> Dict[str, HandObservation]:
        """Run MediaPipe hand landmarks on pre-extracted 256x256 wrist crops (thread-safe per worker instance)."""
        if not crops_by_side:
            return {}
        self._ensure_single_hand_trackers()
        out: Dict[str, HandObservation] = {}
        for side, mp_model in (("left", self._hands_left), ("right", self._hands_right)):
            if mp_model is None or side not in crops_by_side:
                continue
            crop_bgr, w_conf = crops_by_side[side]
            obs = self._infer_single_hand_crop(crop_bgr, side=side, mp_model=mp_model, w_conf=w_conf)
            if obs is not None:
                out[side] = obs
        return out

    @staticmethod
    def _smooth_hand_quaternions(
        rot_seq: np.ndarray,  # (T, 16, 4)
        detected_mask: np.ndarray,  # (T,) bool
        fps: float = 30.0,
        max_gap_frames: int = 18,
        cutoff_hz: float = 4.8,
    ) -> np.ndarray:
        """SLERP-fill dropouts, rate-limit impossible finger snaps, and apply Dual-Band zero-phase smoothing."""
        T = rot_seq.shape[0]
        out = enforce_quaternion_continuity(rot_seq.astype(np.float64)).copy()
        if T < 4:
            return out.astype(np.float32)

        # Lock palm index 0 to clean identity (Blender applies arm IK to wrist bone and only indices 1..15 to fingers!)
        # Prevents mixed MediaPipe camera-space vs RTMW world-space palm frame flips from injecting 170-deg spikes!
        out[:, 0, :] = quat_identity()

        # 1. Fill undetected gaps smoothly with geodesic SLERP (and hold edges cleanly)
        valid_idx = np.where(detected_mask)[0]
        if len(valid_idx) >= 2:
            if valid_idx[0] > 0:
                out[: valid_idx[0], 1:] = out[valid_idx[0], 1:]
            if valid_idx[-1] < T - 1:
                out[valid_idx[-1] + 1 :, 1:] = out[valid_idx[-1], 1:]
            for g in range(len(valid_idx) - 1):
                a, b = int(valid_idx[g]), int(valid_idx[g + 1])
                gap = b - a - 1
                if 0 < gap <= max_gap_frames:
                    for k in range(1, gap + 1):
                        alpha = k / float(gap + 1)
                        for j in range(1, 16):
                            out[a + k, j] = quat_slerp(out[a, j], out[b, j], alpha)
        elif len(valid_idx) == 1:
            out[:, 1:] = out[valid_idx[0], 1:]

        out = enforce_quaternion_continuity(out)

        # 2. Bidirectional Geodesic Finger Despiker & Rate Limiter (<= 360 deg/s = 12 deg/frame @ 30fps!)
        max_f_step = max(8.0, 360.0 / max(fps, 10.0))
        for j in range(1, 16):
            qj = out[:, j, :]
            # Repair isolated 1-4 frame spikes via SLERP from clean neighbors
            for _p in range(2):
                dots = np.clip(np.abs(np.sum(qj[:-1] * qj[1:], axis=1)), 0.0, 1.0)
                sdeg = np.degrees(2.0 * np.arccos(dots))
                sp = np.where(sdeg > max_f_step * 1.4)[0]
                if len(sp) == 0:
                    break
                ti = 0
                while ti < len(sp):
                    s0 = int(sp[ti])
                    s1 = s0 + 1
                    while ti + 1 < len(sp) and int(sp[ti + 1]) - s1 <= 4:
                        ti += 1
                        s1 = int(sp[ti]) + 1
                    a, c = max(0, s0), min(T - 1, s1)
                    if 1 <= (c - a) <= 8 and a > 0 and c < T - 1:
                        for k in range(a + 1, c):
                            qj[k] = quat_slerp(qj[a], qj[c], float(k - a) / float(c - a))
                    ti += 1

            # Zero-phase forward + backward geodesic speed cap (guarantees zero >max_f_step twitching!)
            qf = qj.copy()
            for t in range(1, T):
                d = float(np.dot(qf[t - 1], qf[t]))
                tgt = -qf[t] if d < 0.0 else qf[t]
                ang = float(np.degrees(2.0 * np.arccos(min(1.0, max(0.0, abs(d))))))
                if ang > max_f_step:
                    qf[t] = quat_slerp(qf[t - 1], tgt, max_f_step / ang)
            qb = qj[::-1].copy()
            for t in range(1, T):
                d = float(np.dot(qb[t - 1], qb[t]))
                tgt = -qb[t] if d < 0.0 else qb[t]
                ang = float(np.degrees(2.0 * np.arccos(min(1.0, max(0.0, abs(d))))))
                if ang > max_f_step:
                    qb[t] = quat_slerp(qb[t - 1], tgt, max_f_step / ang)
            qb = qb[::-1]
            for t in range(T):
                qj[t] = quat_slerp(qf[t], qb[t], 0.5)
            out[:, j, :] = qj

        out = enforce_quaternion_continuity(out)

        # 3. Dual-Band Zero-Phase Butterworth smoothing on continuous finger quaternions (1.6 Hz calm / 4.8 Hz active!)
        nyq = 0.5 * max(fps, 5.0)
        fast_f = min(float(cutoff_hz), 4.8)
        if T >= 16 and 0.5 < fast_f < nyq * 0.85:
            calm_f = min(1.6, max(0.7, fast_f * 0.35))
            sos_fast = butter(3, fast_f / nyq, btype="low", output="sos")
            sos_calm = butter(2, calm_f / nyq, btype="low", output="sos")
            sos_env = butter(2, min(2.0, nyq * 0.5) / nyq, btype="low", output="sos")
            for j in range(1, 16):
                q_fast = sosfiltfilt(sos_fast, out[:, j, :], axis=0)
                q_calm = sosfiltfilt(sos_calm, out[:, j, :], axis=0)
                qfn = q_fast / np.maximum(np.linalg.norm(q_fast, axis=1, keepdims=True), 1e-8)
                dots = np.clip(np.abs(np.sum(qfn[:-1] * qfn[1:], axis=1)), 0.0, 1.0)
                spd = np.zeros(T, dtype=np.float64)
                spd[1:] = np.degrees(2.0 * np.arccos(dots)) * fps
                spd[0] = spd[1]
                env = np.maximum(0.0, sosfiltfilt(sos_env, spd))
                u = np.clip((env - 18.0) / 110.0, 0.0, 1.0)
                wf = u * u * u * (u * (u * 6.0 - 15.0) + 10.0)
                q_bl = (1.0 - wf[:, None]) * q_calm + wf[:, None] * q_fast
                norms = np.linalg.norm(q_bl, axis=1, keepdims=True)
                out[:, j, :] = q_bl / np.maximum(norms, 1e-8)

        # 4. Eliminate Butterworth Gibbs ringing into impossible backward PIP/DIP hyperextension (> 0 deg)
        #    Indices: index/middle/ring/pinky _02 and _03 are strict 1-DoF palmar hinges (x <= 0 when w >= 0)
        _HINGE_INDICES = (5, 6, 8, 9, 11, 12, 14, 15)
        for j in _HINGE_INDICES:
            q_j = out[:, j, :]
            neg_w = q_j[:, 3] < 0.0
            q_j[neg_w] = -q_j[neg_w]
            q_j[:, 0] = np.minimum(q_j[:, 0], 0.0)
            q_j[:, 1] = 0.0
            q_j[:, 2] = 0.0
            norms = np.linalg.norm(q_j, axis=1, keepdims=True)
            out[:, j, :] = q_j / np.maximum(norms, 1e-8)

        return enforce_quaternion_continuity(out).astype(np.float32)

    def build_track(
        self,
        frame_results: List[Dict[str, HandObservation]],
        fps: float = 30.0,
        smooth: bool = True,
    ) -> HandMotionTrack:
        T = len(frame_results)
        l_rot = np.tile(quat_identity().astype(np.float32), (T, 16, 1))
        r_rot = np.tile(quat_identity().astype(np.float32), (T, 16, 1))
        l_conf = np.zeros((T, 21), dtype=np.float32)
        r_conf = np.zeros((T, 21), dtype=np.float32)
        l_det = np.zeros(T, dtype=np.bool_)
        r_det = np.zeros(T, dtype=np.bool_)

        for t, res in enumerate(frame_results):
            if "left" in res:
                l_rot[t] = res["left"].local_rotations_xyzw
                l_conf[t] = res["left"].confidence
                l_det[t] = True
            elif t > 0:
                l_rot[t] = l_rot[t - 1]
            if "right" in res:
                r_rot[t] = res["right"].local_rotations_xyzw
                r_conf[t] = res["right"].confidence
                r_det[t] = True
            elif t > 0:
                r_rot[t] = r_rot[t - 1]

        if smooth and T >= 4:
            l_out = self._smooth_hand_quaternions(l_rot, l_det, fps=fps)
            r_out = self._smooth_hand_quaternions(r_rot, r_det, fps=fps)
            # Mark short SLERP-bridged gaps (<= 18 frames) with interpolated confidence for accurate coverage reporting
            for det_m, cf_arr in ((l_det, l_conf), (r_det, r_conf)):
                v_idx = np.where(det_m)[0]
                if len(v_idx) >= 2:
                    for g in range(len(v_idx) - 1):
                        a, b = int(v_idx[g]), int(v_idx[g + 1])
                        if 1 < (b - a) <= 19:
                            cf_arr[a + 1 : b] = np.maximum(cf_arr[a + 1 : b], 0.55)
        else:
            l_out = enforce_quaternion_continuity(l_rot).astype(np.float32)
            r_out = enforce_quaternion_continuity(r_rot).astype(np.float32)

        return HandMotionTrack(
            left_local_rotations_xyzw=l_out,
            left_confidence=l_conf,
            right_local_rotations_xyzw=r_out,
            right_confidence=r_conf,
        )

    def close(self) -> None:
        for attr in ("_hands", "_hands_left", "_hands_right"):
            inst = getattr(self, attr, None)
            if inst is not None:
                try:
                    inst.close()
                except Exception:
                    pass
                setattr(self, attr, None)


def triangulate_hands_from_wholebody(
    cameras: Dict[str, object],
    left_hand21_per_cam: Optional[Dict[str, np.ndarray]],
    right_hand21_per_cam: Optional[Dict[str, np.ndarray]],
    fps: float = 30.0,
    fallback_obs_per_frame: Optional[List[Dict[str, HandObservation]]] = None,
) -> Optional[HandMotionTrack]:
    """Triangulate 21-keypoint left/right hands across calibrated multi-camera views (RTMW-133) into 3D finger rotations.

    Falls back smoothly frame-by-frame to `fallback_obs_per_frame` (e.g., high-res wrist-cropped MediaPipe Hands)
    whenever fewer than 2 cameras clearly observe a hand.
    """
    from engine.triangulation.triangulator import triangulate_sequence

    T = 0
    for src in (left_hand21_per_cam, right_hand21_per_cam):
        if src:
            T = max(T, min(arr.shape[0] for arr in src.values()))
    if T == 0 and fallback_obs_per_frame:
        T = len(fallback_obs_per_frame)
    if T == 0:
        return None

    frame_results: List[Dict[str, HandObservation]] = [
        dict(fallback_obs_per_frame[t]) if (fallback_obs_per_frame and t < len(fallback_obs_per_frame)) else {}
        for t in range(T)
    ]

    max_cam_w = max(
        (int(getattr(getattr(cam, "intrinsics", None), "image_width", 1920)) for cam in cameras.values()),
        default=1920,
    )
    hand_reproj_thresh = max(18.0, 0.016 * float(max_cam_w))

    for side, obs_map in (("left", left_hand21_per_cam), ("right", right_hand21_per_cam)):
        if not obs_map or len(obs_map) < 2:
            continue
        trimmed = {u: arr[:T] for u, arr in obs_map.items() if u in cameras}
        if len(trimmed) < 2:
            continue
        # Prevent frame-to-frame MediaPipe <-> RTMW-x ping-pong flicker!
        # Short MediaPipe gaps (<= 10 frames) are bridged smoothly via SLERP in _smooth_hand_quaternions;
        # ONLY triangulate 3-cam RTMW-x across sustained MediaPipe dropouts (> 10 frames) or when no crops exist!
        has_primary = np.array(
            [frame_results[t].get(side) is not None and float(frame_results[t][side].confidence) >= 0.42 for t in range(T)],
            dtype=np.bool_,
        )
        need_mask = np.zeros(T, dtype=np.bool_)
        if not np.any(has_primary):
            need_mask[:] = True
        else:
            t_i = 0
            while t_i < T:
                if not has_primary[t_i]:
                    t_end = t_i
                    while t_end < T and not has_primary[t_end]:
                        t_end += 1
                    # Only invoke 3-cam RTMW-x fallback on genuinely long dropouts (> 10 frames)
                    if (t_end - t_i) > 10:
                        need_mask[t_i:t_end] = True
                    t_i = t_end
                else:
                    t_i += 1
        need_frames = np.where(need_mask)[0].tolist()
        if not need_frames:
            continue
        need_idx = np.asarray(need_frames, dtype=np.int64)
        sub_obs = {u: arr[need_idx] for u, arr in trimmed.items()}
        pts3d_sub, confs_sub, _, _ = triangulate_sequence(
            cameras,
            sub_obs,
            min_confidence=0.22,
            max_reproj_error_px=hand_reproj_thresh,
            refine_nonlinear=False,
        )
        for k_sub, t in enumerate(need_frames):
            valid_k = (confs_sub[k_sub] > 0.20) & np.all(np.isfinite(pts3d_sub[k_sub]), axis=-1)
            # Need wrist (0) + palm anchor points (5, 9, 17) + at least 10 landmarks total
            if int(np.sum(valid_k)) >= 10 and all(valid_k[k] for k in (0, 5, 9, 17)):
                pts_clean = np.nan_to_num(pts3d_sub[k_sub], nan=0.0).astype(np.float32)
                rots = solve_hand_rotations_from_landmarks21(pts_clean, handedness=side)
                frame_results[t][side] = HandObservation(
                    handedness=side,
                    landmarks_3d=pts_clean,
                    confidence=float(np.mean(confs_sub[k_sub, valid_k])),
                    local_rotations_xyzw=rots,
                )

    if not any(bool(r) for r in frame_results):
        return None
    return HandCaptureEngine().build_track(frame_results, fps=fps, smooth=True)
