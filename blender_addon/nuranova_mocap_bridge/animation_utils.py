"""Quaternion interpolation guards, also usable without Blender for validation."""
import numpy as np


def continuous_quaternions(sequence):
    """Normalize and select a consistent hemisphere before writing linear keys."""
    q = np.asarray(sequence, dtype=float).copy()
    if q.ndim != 3 or q.shape[-1] != 4:
        raise ValueError("Quaternion animation must have shape (frames, bones, 4).")
    length = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.isfinite(q).all() or np.any(length < 1e-8):
        raise ValueError("Quaternion animation contains a non-finite or zero rotation.")
    q /= length
    for frame in range(1, len(q)):
        q[frame] *= np.where(np.sum(q[frame-1] * q[frame], axis=-1, keepdims=True) < 0, -1, 1)
    return q
