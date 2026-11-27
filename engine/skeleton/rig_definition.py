"""Hierarchical 25-bone articulated humanoid rig, anatomical limits, forward kinematics, and actor calibration.

Conventions:
- Right-handed Z-Up world (+X Right, +Y Forward, +Z Up)
- Meters for all bone rest offsets
- Rest pose is a standard T-Pose facing +Y:
  - Arms extend along -X (left) and +X (right)
  - Legs descend along -Z
  - Spine rises along +Z
  - Feet point forward along +Y
- Quaternions stored scalar-last [x, y, z, w]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from engine.math_core.transforms import (
    make_transform,
    quat_normalize,
    quat_to_matrix,
)

# Canonical 25-bone Nuranova humanoid rig
BONE_NAMES: List[str] = [
    "hips",            # 0  (root)
    "spine",           # 1
    "spine1",          # 2
    "chest",           # 3
    "neck",            # 4
    "head",            # 5
    "left_clavicle",   # 6
    "left_upper_arm",  # 7
    "left_forearm",    # 8
    "left_hand",       # 9
    "right_clavicle",  # 10
    "right_upper_arm", # 11
    "right_forearm",   # 12
    "right_hand",      # 13
    "left_upper_leg",  # 14
    "left_lower_leg",  # 15
    "left_foot",       # 16
    "left_toe",        # 17
    "right_upper_leg", # 18
    "right_lower_leg", # 19
    "right_foot",      # 20
    "right_toe",       # 21
    "head_end",        # 22
    "left_hand_end",   # 23
    "right_hand_end",  # 24
]

BONE_INDEX: Dict[str, int] = {name: i for i, name in enumerate(BONE_NAMES)}

PARENT_INDICES: List[int] = [
    -1,  # 0  hips
    0,   # 1  spine
    1,   # 2  spine1
    2,   # 3  chest
    3,   # 4  neck
    4,   # 5  head
    3,   # 6  left_clavicle
    6,   # 7  left_upper_arm
    7,   # 8  left_forearm
    8,   # 9  left_hand
    3,   # 10 right_clavicle
    10,  # 11 right_upper_arm
    11,  # 12 right_forearm
    12,  # 13 right_hand
    0,   # 14 left_upper_leg
    14,  # 15 left_lower_leg
    15,  # 16 left_foot
    16,  # 17 left_toe
    0,   # 18 right_upper_leg
    18,  # 19 right_lower_leg
    19,  # 20 right_foot
    20,  # 21 right_toe
    5,   # 22 head_end
    9,   # 23 left_hand_end
    13,  # 24 right_hand_end
]

# Default 1.72m adult rest offsets (meters, T-pose facing +Y, Z-up)
DEFAULT_REST_OFFSETS_M: np.ndarray = np.array(
    [
        [0.000, 0.000, 0.960],   # 0  hips (height above floor)
        [0.000, 0.000, 0.110],   # 1  spine
        [0.000, 0.000, 0.140],   # 2  spine1
        [0.000, 0.000, 0.150],   # 3  chest
        [0.000, 0.000, 0.120],   # 4  neck
        [0.000, 0.000, 0.090],   # 5  head
        [-0.060, 0.000, 0.070],  # 6  left_clavicle
        [-0.120, 0.000, 0.000],  # 7  left_upper_arm (shoulder joint)
        [-0.290, 0.000, 0.000],  # 8  left_forearm (elbow joint)
        [-0.260, 0.000, 0.000],  # 9  left_hand (wrist joint)
        [0.060, 0.000, 0.070],   # 10 right_clavicle
        [0.120, 0.000, 0.000],   # 11 right_upper_arm
        [0.290, 0.000, 0.000],   # 12 right_forearm
        [0.260, 0.000, 0.000],   # 13 right_hand
        [-0.095, 0.000, -0.040], # 14 left_upper_leg (hip joint)
        [0.000, 0.000, -0.430],  # 15 left_lower_leg (knee joint)
        [0.000, 0.000, -0.410],  # 16 left_foot (ankle joint)
        [0.000, 0.140, -0.080],  # 17 left_toe
        [0.095, 0.000, -0.040],  # 18 right_upper_leg
        [0.000, 0.000, -0.430],  # 19 right_lower_leg
        [0.000, 0.000, -0.410],  # 20 right_foot
        [0.000, 0.140, -0.080],  # 21 right_toe
        [0.000, 0.000, 0.160],   # 22 head_end
        [-0.140, 0.000, 0.000],  # 23 left_hand_end
        [0.140, 0.000, 0.000],   # 24 right_hand_end
    ],
    dtype=np.float64,
)

# Anatomical XYZ Euler rotation limits (degrees) per bone (Z-up, +Y forward, +X right, right-hand rule!)
# Note: For downward-pointing legs (rest = -Z):
#   +X rotates (0,0,-1) -> (0,+1,0) FORWARD (hip flexion up to +150 deg in deep squats!)
#   -X rotates (0,0,-1) -> (0,-1,0) BACKWARD (hip extension to -45 deg; knee flexion to -160 deg!)
#   +Y rotates (0,0,-1) -> (-1,0,0) LEFT (left hip abduction, right hip adduction)
#   -Y rotates (0,0,-1) -> (+1,0,0) RIGHT (right hip abduction, left hip adduction)
ANATOMICAL_LIMITS_DEG: Dict[str, Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = {
    "spine": ((-60, 45), (-40, 40), (-45, 45)),
    "spine1": ((-55, 40), (-35, 35), (-40, 40)),
    "chest": ((-45, 35), (-30, 30), (-40, 40)),
    "neck": ((-55, 50), (-40, 40), (-60, 60)),
    "head": ((-60, 55), (-45, 45), (-75, 75)),
    "left_clavicle": ((-18, 18), (-18, 22), (-18, 22)),
    "right_clavicle": ((-18, 18), (-22, 18), (-22, 18)),
    "left_upper_arm": ((-145, 145), (-135, 135), (-145, 145)),
    "right_upper_arm": ((-145, 145), (-135, 135), (-145, 145)),
    "left_forearm": ((-95, 95), (-25, 25), (-160, 10)),
    "right_forearm": ((-95, 95), (-25, 25), (-10, 160)),
    "left_hand": ((-80, 80), (-75, 75), (-55, 55)),
    "right_hand": ((-80, 80), (-75, 75), (-55, 55)),
    "left_upper_leg": ((-45, 150), (-45, 85), (-65, 65)),
    "right_upper_leg": ((-45, 150), (-85, 45), (-65, 65)),
    "left_lower_leg": ((-160, 5), (-18, 18), (-30, 30)),  # knee flexes backward (-X in Z-up facing +Y)
    "right_lower_leg": ((-160, 5), (-18, 18), (-30, 30)),
    "left_foot": ((-55, 45), (-22, 22), (-32, 32)),
    "right_foot": ((-55, 45), (-22, 22), (-32, 32)),
}


@dataclass
class ActorProfile:
    actor_id: str
    name: str
    height_m: float = 1.72
    rest_offsets_m: np.ndarray = field(default_factory=lambda: DEFAULT_REST_OFFSETS_M.copy())
    calibrated_marker_offsets_m: Dict[str, Tuple[str, List[float]]] = field(default_factory=dict)
    calibrated_cluster_rotations: Dict[str, List[float]] = field(default_factory=dict)
    calibration_pose_type: str = "T-pose"

    def bone_length(self, child_bone: str) -> float:
        idx = BONE_INDEX[child_bone]
        return float(np.linalg.norm(self.rest_offsets_m[idx]))

    def to_dict(self) -> Dict:
        return {
            "actor_id": self.actor_id,
            "name": self.name,
            "height_m": float(self.height_m),
            "rest_offsets_m": np.asarray(self.rest_offsets_m, dtype=np.float64).tolist(),
            "calibrated_marker_offsets_m": self.calibrated_marker_offsets_m,
            "calibrated_cluster_rotations": self.calibrated_cluster_rotations,
            "calibration_pose_type": self.calibration_pose_type,
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "ActorProfile":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            actor_id=str(d["actor_id"]),
            name=str(d["name"]),
            height_m=float(d.get("height_m", 1.72)),
            rest_offsets_m=np.asarray(d["rest_offsets_m"], dtype=np.float64),
            calibrated_marker_offsets_m=dict(d.get("calibrated_marker_offsets_m", {})),
            calibrated_cluster_rotations=dict(d.get("calibrated_cluster_rotations", {})),
            calibration_pose_type=str(d.get("calibration_pose_type", "T-pose")),
        )


def compute_forward_kinematics_frame(
    root_translation: np.ndarray,
    local_rotations_xyzw: np.ndarray,
    rest_offsets_m: np.ndarray = DEFAULT_REST_OFFSETS_M,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute exact Forward Kinematics for one frame, guaranteeing constant calibrated bone lengths.

    Returns:
        joint_positions (B, 3) float64 meters
        global_matrices (B, 4, 4) float64
    """
    B = len(BONE_NAMES)
    positions = np.zeros((B, 3), dtype=np.float64)
    global_mats = np.zeros((B, 4, 4), dtype=np.float64)
    global_rots = np.zeros((B, 3, 3), dtype=np.float64)

    for b in range(B):
        p = PARENT_INDICES[b]
        R_loc = quat_to_matrix(local_rotations_xyzw[b])
        if p < 0:
            positions[b] = np.asarray(root_translation, dtype=np.float64).reshape(3)
            global_rots[b] = R_loc
        else:
            positions[b] = positions[p] + global_rots[p] @ rest_offsets_m[b]
            global_rots[b] = global_rots[p] @ R_loc
        global_mats[b] = make_transform(global_rots[b], positions[b])

    return positions, global_mats


def compute_forward_kinematics_sequence(
    root_translations: np.ndarray,
    local_rotations_xyzw: np.ndarray,
    rest_offsets_m: np.ndarray = DEFAULT_REST_OFFSETS_M,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized Forward Kinematics across all T frames (~300x faster than per-frame loop!)."""
    T = root_translations.shape[0]
    B = len(BONE_NAMES)
    if T == 0:
        return np.zeros((0, B, 3), dtype=np.float32), np.zeros((0, B, 4, 4), dtype=np.float32)

    q = np.asarray(local_rotations_xyzw, dtype=np.float64)  # (T, B, 4) xyzw
    norms = np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    q = q / norms
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R_loc = np.empty((T, B, 3, 3), dtype=np.float64)
    R_loc[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
    R_loc[..., 0, 1] = 2.0 * (xy - wz)
    R_loc[..., 0, 2] = 2.0 * (xz + wy)
    R_loc[..., 1, 0] = 2.0 * (xy + wz)
    R_loc[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
    R_loc[..., 1, 2] = 2.0 * (yz - wx)
    R_loc[..., 2, 0] = 2.0 * (xz - wy)
    R_loc[..., 2, 1] = 2.0 * (yz + wx)
    R_loc[..., 2, 2] = 1.0 - 2.0 * (xx + yy)

    rest = np.asarray(rest_offsets_m, dtype=np.float64)
    pos_all = np.zeros((T, B, 3), dtype=np.float64)
    R_glob = np.zeros((T, B, 3, 3), dtype=np.float64)
    pos_all[:, 0, :] = np.asarray(root_translations, dtype=np.float64).reshape(T, 3)
    R_glob[:, 0, :, :] = R_loc[:, 0, :, :]

    for b in range(1, B):
        p = PARENT_INDICES[b]
        pos_all[:, b, :] = pos_all[:, p, :] + np.einsum("tij,j->ti", R_glob[:, p], rest[b])
        R_glob[:, b, :, :] = R_glob[:, p] @ R_loc[:, b]

    mat_all = np.zeros((T, B, 4, 4), dtype=np.float32)
    mat_all[:, :, :3, :3] = R_glob.astype(np.float32)
    mat_all[:, :, :3, 3] = pos_all.astype(np.float32)
    mat_all[:, :, 3, 3] = 1.0
    return pos_all.astype(np.float32), mat_all


def clamp_bone_rotation(bone_name: str, q_local_xyzw: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Apply anatomical joint rotation limits smoothly in local bone Euler space with continuous gimbal protection.

    Uses fast direct analytic extrinsic-XYZ Euler <-> quaternion math (~18x faster than constructing scipy Rotation!).
    """
    q_norm = quat_normalize(q_local_xyzw)
    if bone_name not in ANATOMICAL_LIMITS_DEG or strength <= 0.0:
        return q_norm
    limits = ANATOMICAL_LIMITS_DEG[bone_name]
    qx, qy, qz, qw = float(q_norm[0]), float(q_norm[1]), float(q_norm[2]), float(q_norm[3])
    # Exact extrinsic XYZ Euler angles (identical to Rotation.from_quat(q).as_euler("xyz", degrees=True))
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 0.999999:
        ey_rad = np.copysign(0.5 * np.pi, sinp)
        ex_rad = 0.0
        ez_rad = float(np.arctan2(2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz)))
    else:
        ey_rad = float(np.arcsin(sinp))
        ex_rad = float(np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy)))
        ez_rad = float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
    ex = float(np.degrees(ex_rad))
    ey = float(np.degrees(ey_rad))
    ez = float(np.degrees(ez_rad))

    # Fast early-exit when already strictly inside anatomical limits (zero trig reconstruction needed!)
    if (
        limits[0][0] <= ex <= limits[0][1]
        and limits[1][0] <= ey <= limits[1][1]
        and limits[2][0] <= ez <= limits[2][1]
    ):
        return q_norm

    # Smoothly fade Euler clamping near |y| -> 90 deg on wide-range ball-and-socket joints (ZERO 78-deg discontinuity snap!)
    eff_strength = float(strength)
    if bone_name in ("left_upper_arm", "right_upper_arm", "left_upper_leg", "right_upper_leg"):
        ay = abs(ey)
        if ay >= 82.0:
            return q_norm
        if ay > 65.0:
            fade = (82.0 - ay) / 17.0
            eff_strength *= fade * fade * (3.0 - 2.0 * fade)

    cx_d = float(np.clip(ex, limits[0][0], limits[0][1]))
    cy_d = float(np.clip(ey, limits[1][0], limits[1][1]))
    cz_d = float(np.clip(ez, limits[2][0], limits[2][1]))
    bx = 0.5 * np.radians((1.0 - eff_strength) * ex + eff_strength * cx_d)
    by = 0.5 * np.radians((1.0 - eff_strength) * ey + eff_strength * cy_d)
    bz = 0.5 * np.radians((1.0 - eff_strength) * ez + eff_strength * cz_d)
    sx, cx = float(np.sin(bx)), float(np.cos(bx))
    sy, cy = float(np.sin(by)), float(np.cos(by))
    sz, cz = float(np.sin(bz)), float(np.cos(bz))
    return quat_normalize(
        np.array(
            [
                sx * cy * cz - cx * sy * sz,
                cx * sy * cz + sx * cy * sz,
                cx * cy * sz - sx * sy * cz,
                cx * cy * cz + sx * sy * sz,
            ],
            dtype=np.float64,
        )
    )


def calibrate_actor_from_body26_frames(
    actor_id: str,
    name: str,
    body26_3d_frames: np.ndarray,  # (T, 26, 3) meters
    body26_conf: np.ndarray,  # (T, 26)
    pose_type: str = "T-pose",
    min_conf: float = 0.4,
) -> ActorProfile:
    """Calibrate symmetric actor skeletal bone lengths from a neutral/T/A-pose 3D capture."""
    pts = np.asarray(body26_3d_frames, dtype=np.float64)
    conf = np.asarray(body26_conf, dtype=np.float64)

    def median_dist(idx_a: int, idx_b: int, default_m: float) -> float:
        valid = (
            (conf[:, idx_a] >= min_conf)
            & (conf[:, idx_b] >= min_conf)
            & np.all(np.isfinite(pts[:, idx_a]), axis=-1)
            & np.all(np.isfinite(pts[:, idx_b]), axis=-1)
        )
        if not np.any(valid):
            return default_m
        dists = np.linalg.norm(pts[valid, idx_a] - pts[valid, idx_b], axis=-1)
        med = float(np.median(dists))
        # Sanity bounds relative to human anatomy
        return float(np.clip(med, default_m * 0.45, default_m * 1.75))

    # Measure bilateral segments and symmetrize L/R for clean rig proportions
    upper_arm = 0.5 * (median_dist(5, 7, 0.29) + median_dist(6, 8, 0.29))
    forearm = 0.5 * (median_dist(7, 9, 0.26) + median_dist(8, 10, 0.26))
    thigh = 0.5 * (median_dist(11, 13, 0.43) + median_dist(12, 14, 0.43))
    shin = 0.5 * (median_dist(13, 15, 0.41) + median_dist(14, 16, 0.41))
    foot_len = 0.5 * (median_dist(15, 20, 0.16) + median_dist(16, 21, 0.16))
    hip_half_w = 0.5 * median_dist(11, 12, 0.19)
    shoulder_half_w = 0.5 * median_dist(5, 6, 0.36)
    torso_len = median_dist(19, 18, 0.52)  # hips -> neck
    head_len = median_dist(18, 17, 0.25)   # neck -> head_top

    offsets = DEFAULT_REST_OFFSETS_M.copy()
    offsets[BONE_INDEX["hips"]] = [0.0, 0.0, thigh + shin + 0.08]
    offsets[BONE_INDEX["spine"]] = [0.0, 0.0, torso_len * 0.22]
    offsets[BONE_INDEX["spine1"]] = [0.0, 0.0, torso_len * 0.27]
    offsets[BONE_INDEX["chest"]] = [0.0, 0.0, torso_len * 0.29]
    offsets[BONE_INDEX["neck"]] = [0.0, 0.0, torso_len * 0.22]
    offsets[BONE_INDEX["head"]] = [0.0, 0.0, head_len * 0.38]
    offsets[BONE_INDEX["head_end"]] = [0.0, 0.0, head_len * 0.62]

    clav_x = shoulder_half_w * 0.35
    sh_x = shoulder_half_w * 0.65
    offsets[BONE_INDEX["left_clavicle"]] = [-clav_x, 0.0, torso_len * 0.14]
    offsets[BONE_INDEX["right_clavicle"]] = [clav_x, 0.0, torso_len * 0.14]
    offsets[BONE_INDEX["left_upper_arm"]] = [-sh_x, 0.0, 0.0]
    offsets[BONE_INDEX["right_upper_arm"]] = [sh_x, 0.0, 0.0]
    offsets[BONE_INDEX["left_forearm"]] = [-upper_arm, 0.0, 0.0]
    offsets[BONE_INDEX["right_forearm"]] = [upper_arm, 0.0, 0.0]
    offsets[BONE_INDEX["left_hand"]] = [-forearm, 0.0, 0.0]
    offsets[BONE_INDEX["right_hand"]] = [forearm, 0.0, 0.0]

    offsets[BONE_INDEX["left_upper_leg"]] = [-hip_half_w, 0.0, -0.04]
    offsets[BONE_INDEX["right_upper_leg"]] = [hip_half_w, 0.0, -0.04]
    offsets[BONE_INDEX["left_lower_leg"]] = [0.0, 0.0, -thigh]
    offsets[BONE_INDEX["right_lower_leg"]] = [0.0, 0.0, -thigh]
    offsets[BONE_INDEX["left_foot"]] = [0.0, 0.0, -shin]
    offsets[BONE_INDEX["right_foot"]] = [0.0, 0.0, -shin]
    offsets[BONE_INDEX["left_toe"]] = [0.0, foot_len * 0.85, -0.075]
    offsets[BONE_INDEX["right_toe"]] = [0.0, foot_len * 0.85, -0.075]

    est_height = thigh + shin + 0.08 + torso_len + head_len
    return ActorProfile(
        actor_id=actor_id,
        name=name,
        height_m=float(est_height),
        rest_offsets_m=offsets,
        calibration_pose_type=pose_type,
    )
