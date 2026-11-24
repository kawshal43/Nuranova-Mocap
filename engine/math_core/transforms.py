"""Core 3D linear algebra, quaternion, SE(3), and Kabsch rigid-body math.

Canonical conventions:
- Right-handed Z-Up world coordinate system (+X Right, +Y Forward, +Z Up)
- Quaternions stored scalar-last [x, y, z, w]
- Rotation matrices are 3x3 active right-handed column-vector matrices
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation


def quat_identity() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def quat_normalize(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    safe_norm = np.where(norm > eps, norm, 1.0)
    out = q / safe_norm
    if q.ndim == 1:
        if norm.item() <= eps:
            return quat_identity()
        return out
    bad = (norm[..., 0] <= eps)
    if np.any(bad):
        out[bad] = quat_identity()
    return out


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    out = q.copy()
    out[..., :3] *= -1.0
    return out


def quat_inverse(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    return quat_conjugate(q)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 * q2 with scalar-last [x, y, z, w]."""
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return quat_normalize(np.stack([x, y, z, w], axis=-1))


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    return Rotation.from_quat(q).as_matrix()


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    # Orthonormalize via SVD to guard against floating-point drift
    if R.ndim == 2:
        U, _, Vt = np.linalg.svd(R)
        R_clean = U @ Vt
        if np.linalg.det(R_clean) < 0:
            U[:, -1] *= -1.0
            R_clean = U @ Vt
        return quat_normalize(Rotation.from_matrix(R_clean).as_quat())
    return quat_normalize(Rotation.from_matrix(R).as_quat())


def quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    R = quat_to_matrix(q)
    v = np.asarray(v, dtype=np.float64)
    if R.ndim == 2 and v.ndim == 1:
        return R @ v
    return np.einsum("...ij,...j->...i", R, v)


def quat_from_two_vectors(v_from: np.ndarray, v_to: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Shortest-arc unit quaternion rotating v_from onto v_to."""
    a = np.asarray(v_from, dtype=np.float64)
    b = np.asarray(v_to, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < eps or nb < eps:
        return quat_identity()
    a = a / na
    b = b / nb
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 1.0 - 1e-10:
        return quat_identity()
    if dot < -1.0 + 1e-10:
        # 180-degree rotation around any orthogonal axis
        ortho = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, ortho)
        axis /= np.linalg.norm(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0], dtype=np.float64)
    cross = np.cross(a, b)
    w = 1.0 + dot
    return quat_normalize(np.array([cross[0], cross[1], cross[2], w], dtype=np.float64))


def quat_angle_between(q1: np.ndarray, q2: np.ndarray) -> float:
    """Geodesic angular difference in radians between two unit quaternions."""
    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)
    dot = abs(float(np.clip(np.dot(q1, q2), -1.0, 1.0)))
    return float(2.0 * np.arccos(dot))


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between q0 and q1 for t in [0, 1]."""
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return quat_normalize(q0 + t * (q1 - q0))
    theta0 = np.arccos(dot)
    sin_t0 = np.sin(theta0)
    s0 = np.sin((1.0 - t) * theta0) / sin_t0
    s1 = np.sin(t * theta0) / sin_t0
    return quat_normalize(s0 * q0 + s1 * q1)


def enforce_quaternion_continuity(quats: np.ndarray) -> np.ndarray:
    """Resolve antipodal sign flips (q == -q) across a time sequence (T, ..., 4)."""
    out = quat_normalize(np.asarray(quats, dtype=np.float64).copy())
    if out.shape[0] < 2:
        return out
    for t in range(1, out.shape[0]):
        dots = np.sum(out[t] * out[t - 1], axis=-1, keepdims=True)
        out[t] = np.where(dots < 0.0, -out[t], out[t])
    return out


def swing_twist_decomposition(q: np.ndarray, twist_axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose unit quaternion q = q_swing * q_twist where q_twist rotates about twist_axis."""
    q = quat_normalize(q)
    axis = np.asarray(twist_axis, dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    v = q[:3]
    proj = np.dot(v, axis) * axis
    q_twist = np.array([proj[0], proj[1], proj[2], q[3]], dtype=np.float64)
    if np.linalg.norm(q_twist) < 1e-10:
        q_twist = quat_identity()
    else:
        q_twist = quat_normalize(q_twist)
    q_swing = quat_multiply(q, quat_conjugate(q_twist))
    return q_swing, q_twist


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert canonical [x, y, z, w] to Blender [w, x, y, z]."""
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """Convert Blender [w, x, y, z] to canonical [x, y, z, w]."""
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return M


def invert_transform(M: np.ndarray) -> np.ndarray:
    R = M[:3, :3]
    t = M[:3, 3]
    Minv = np.eye(4, dtype=np.float64)
    Minv[:3, :3] = R.T
    Minv[:3, 3] = -R.T @ t
    return Minv


def transform_points(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    return (pts @ M[:3, :3].T) + M[:3, 3]


def solve_rigid_transform_kabsch(
    local_points: np.ndarray,
    world_points: np.ndarray,
    weights: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Weighted Kabsch / Procrustes SVD solver for rigid body 6-DoF pose.

    Finds R (3x3, det=+1) and t (3,) minimizing sum_i w_i || R @ local_i + t - world_i ||^2.
    Requires at least 3 non-collinear points.

    Returns:
        (R, t, rms_error_m)
    """
    P = np.asarray(local_points, dtype=np.float64)
    Q = np.asarray(world_points, dtype=np.float64)
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3 or P.shape[0] < 3:
        raise ValueError("Kabsch solver requires matching (N, 3) point arrays with N >= 3.")

    n = P.shape[0]
    if weights is None:
        w = np.ones(n, dtype=np.float64) / n
    else:
        w = np.asarray(weights, dtype=np.float64)
        w_sum = np.sum(w)
        if w_sum <= 1e-12:
            raise ValueError("Kabsch weights must sum to a positive value.")
        w = w / w_sum

    p_mean = np.sum(P * w[:, None], axis=0)
    q_mean = np.sum(Q * w[:, None], axis=0)
    Pc = P - p_mean
    Qc = Q - q_mean

    H = (Pc * w[:, None]).T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1.0, 1.0, 1.0 if d >= 0 else -1.0])
    R = Vt.T @ S @ U.T
    t = q_mean - R @ p_mean

    pred = (P @ R.T) + t
    residuals = np.linalg.norm(pred - Q, axis=1)
    rms = float(np.sqrt(np.sum(w * residuals**2)))
    return R, t, rms
