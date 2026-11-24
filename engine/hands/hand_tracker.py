"""Dedicated Hand & Finger Capture module (MediaPipe Hands + RTMW-133 hand landmarks -> 16 finger bone rotations)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

from engine.data_format.motion_io import HandMotionTrack
from engine.math_core.transforms import (
    enforce_quaternion_continuity,
    matrix_to_quat,
    quat_from_two_vectors,
    quat_identity,
    quat_inverse,
    quat_multiply,
    quat_rotate_vector,
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


def solve_hand_rotations_from_landmarks21(
    landmarks_3d: np.ndarray,
    handedness: str = "right",
) -> np.ndarray:
    """Solve wrist orientation and 15 finger phalanx local quaternions from 21 3D hand landmarks.

    Landmarks follow MediaPipe/COCO-Hand convention:
    0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky.
    """
    pts = np.asarray(landmarks_3d, dtype=np.float64).reshape(21, 3)
    quats = np.tile(quat_identity(), (16, 1))

    # Build palm orthonormal frame: +Y along wrist -> middle MCP (9), +X across index(5)->pinky(17)
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

    # Transform landmarks into palm frame (+Y finger extension rest axis)
    pts_palm = (pts - pts[0]) @ R_palm

    # Solve hierarchical local rotations along each 3-bone finger chain
    rest_dir = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    for finger_idx in range(5):
        parent_q = quat_identity()
        for seg in range(3):
            b_idx, lm_a, lm_b = _FINGER_CHAINS[finger_idx * 3 + seg]
            vec = pts_palm[lm_b] - pts_palm[lm_a]
            vn = float(np.linalg.norm(vec))
            if vn < 1e-6:
                quats[b_idx] = quat_identity()
                continue
            vec_local = quat_rotate_vector(quat_inverse(parent_q), vec / vn)
            q_local = quat_from_two_vectors(rest_dir, vec_local)
            quats[b_idx] = q_local
            parent_q = quat_multiply(parent_q, q_local)

    return quats.astype(np.float32)


class HandCaptureEngine:
    """Dedicated close-up camera hand tracker using MediaPipe Hands (21 landmarks/hand)."""

    def __init__(self, max_num_hands: int = 2, model_complexity: int = 1) -> None:
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self._hands = None

    def _ensure_loaded(self) -> None:
        if self._hands is not None:
            return
        import mediapipe as mp

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_num_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def infer_frame(self, frame_bgr: np.ndarray) -> Dict[str, HandObservation]:
        self._ensure_loaded()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._hands.process(rgb)
        out: Dict[str, HandObservation] = {}
        if not res.multi_hand_world_landmarks or not res.multi_handedness:
            return out
        for world_lms, handed in zip(res.multi_hand_world_landmarks, res.multi_handedness):
            label = handed.classification[0].label.lower()
            score = float(handed.classification[0].score)
            # Convert MediaPipe hand coords (+X right, +Y down, +Z away) to Z-up right-handed
            pts = np.array([[p.x, -p.z, -p.y] for p in world_lms.landmark], dtype=np.float32)
            rots = solve_hand_rotations_from_landmarks21(pts, handedness=label)
            out[label] = HandObservation(
                handedness=label,
                landmarks_3d=pts,
                confidence=score,
                local_rotations_xyzw=rots,
            )
        return out

    def build_track(self, frame_results: List[Dict[str, HandObservation]]) -> HandMotionTrack:
        T = len(frame_results)
        l_rot = np.tile(quat_identity().astype(np.float32), (T, 16, 1))
        r_rot = np.tile(quat_identity().astype(np.float32), (T, 16, 1))
        l_conf = np.zeros((T, 21), dtype=np.float32)
        r_conf = np.zeros((T, 21), dtype=np.float32)

        for t, res in enumerate(frame_results):
            if "left" in res:
                l_rot[t] = res["left"].local_rotations_xyzw
                l_conf[t] = res["left"].confidence
            elif t > 0:
                l_rot[t] = l_rot[t - 1]
            if "right" in res:
                r_rot[t] = res["right"].local_rotations_xyzw
                r_conf[t] = res["right"].confidence
            elif t > 0:
                r_rot[t] = r_rot[t - 1]

        return HandMotionTrack(
            left_local_rotations_xyzw=enforce_quaternion_continuity(l_rot).astype(np.float32),
            left_confidence=l_conf,
            right_local_rotations_xyzw=enforce_quaternion_continuity(r_rot).astype(np.float32),
            right_confidence=r_conf,
        )

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()
            self._hands = None
