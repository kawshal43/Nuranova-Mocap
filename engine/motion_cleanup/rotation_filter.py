"""Symmetric robust smoothing on SO(3), independent of Euler branches and quaternion signs."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from engine.math_core.transforms import enforce_quaternion_continuity
from engine.skeleton.rig_definition import BONE_NAMES, compute_forward_kinematics_sequence


def observed_fast_motion(quaternions, confidence, observable, fps, base_step, airborne=None):
    """Protect coherent multi-frame movement, rather than accepting every fast step."""
    from scipy.ndimage import gaussian_filter1d
    rotations = Rotation.from_quat(quaternions)
    vectors = (rotations[:-1].inv()*rotations[1:]).as_rotvec()
    angles = np.degrees(np.linalg.norm(vectors, axis=1))
    ceiling = max(base_step, 1200/max(fps, 10.))
    valid = (angles > base_step) & (angles <= ceiling)
    valid &= (np.minimum(confidence[:-1], confidence[1:]) >= .6) & observable[:-1] & observable[1:]
    if airborne is not None:
        valid &= airborne[:-1] & airborne[1:]
    directions = vectors/np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
    same_direction = np.sum(directions[:-1]*directions[1:], axis=1) >= .85
    protected = np.zeros(len(angles), bool)
    for start in np.flatnonzero(valid[:-2] & valid[1:-1] & valid[2:] & same_direction[:-1] & same_direction[1:]):
        protected[start:start+3] = True
    extra_limits = np.where(protected, np.minimum(angles*1.08, ceiling), 0.)
    frames = np.r_[protected, False] | np.r_[False, protected]
    weight = gaussian_filter1d(frames.astype(float), .6)
    return extra_limits, weight


def smooth_rotations(quaternions, fps, cutoff_hz=4.0, confidence=None):
    """Robust local rotation means with symmetric Gaussian time weights and no phase delay.

    Unlike filtering Euler angles, this preserves turns through +/-180 degrees. Raw
    observations remain separate; low confidence reduces a sample's influence.
    """
    raw = np.asarray(quaternions, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[-1] != 4:
        raise ValueError("Rotations must have shape (frames, bones, 4).")
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(cutoff_hz) or cutoff_hz <= 0:
        raise ValueError("Frame rate and rotation cutoff must be positive and finite.")
    norms = np.linalg.norm(raw, axis=-1, keepdims=True)
    if not np.all(np.isfinite(raw)) or np.any(norms < 1e-8):
        raise ValueError("Rotations must be finite nonzero quaternions.")
    raw = raw / norms
    frames, bones, _ = raw.shape
    if frames < 3:
        return raw.astype(np.float32)
    conf = np.ones((frames, bones)) if confidence is None else np.asarray(confidence)
    if conf.shape != (frames, bones):
        raise ValueError("Rotation confidence must match the frame and bone counts.")
    conf = np.clip(np.nan_to_num(conf, nan=0.0), 0.05, 1.0)
    # Gaussian -3 dB frequency: sigma_t = sqrt(log(2)) / (2*pi*cutoff).
    sigma = max(0.45, np.sqrt(np.log(2)) * fps / (2 * np.pi * cutoff_hz))
    radius = max(1, int(np.ceil(3 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    temporal = np.exp(-0.5 * (offsets / sigma) ** 2)
    indices = np.pad(np.arange(frames), (radius, radius), mode="reflect")
    # A hemisphere-aligned chordal mean initializes the intrinsic robust mean.
    accum = np.zeros_like(raw)
    total = np.zeros((frames, bones))
    for delta, weight in zip(offsets, temporal):
        idx = indices[radius + delta : radius + delta + frames]
        neighbor = raw[idx]
        signs = np.where(np.sum(neighbor * raw, axis=-1, keepdims=True) < 0, -1, 1)
        weights = weight * conf[idx]
        accum += neighbor * signs * weights[..., None]
        total += weights
    current = accum / np.maximum(np.linalg.norm(accum, axis=-1, keepdims=True), 1e-12)
    for _ in range(3):
        center = Rotation.from_quat(current.reshape(-1, 4))
        tangent = np.zeros((frames, bones, 3))
        total.fill(0)
        for delta, weight in zip(offsets, temporal):
            idx = indices[radius + delta : radius + delta + frames]
            relative = (
                (center.inv() * Rotation.from_quat(raw[idx].reshape(-1, 4)))
                .as_rotvec()
                .reshape(frames, bones, 3)
            )
            angle = np.linalg.norm(relative, axis=-1)
            robust = np.minimum(1, np.radians(25) / np.maximum(angle, 1e-9))
            weights = weight * conf[idx] * robust
            tangent += relative * weights[..., None]
            total += weights
        update = tangent / np.maximum(total[..., None], 1e-12)
        current = (
            (center * Rotation.from_rotvec(update.reshape(-1, 3)))
            .as_quat()
            .reshape(frames, bones, 4)
        )
    return enforce_quaternion_continuity(current).astype(np.float32)


def stabilize_upper_body(track, actor, fps, cutoff_hz=4.5):
    """Remove residual IK branch flicker without changing the planted leg chains or root."""
    indices = [
        i
        for i, name in enumerate(BONE_NAMES)
        if any(
            part in name
            for part in ("spine", "neck", "head", "clavicle", "upper_arm", "forearm", "hand")
        )
    ]
    rotations = track.local_rotations_xyzw.copy()
    rotations[:, indices] = smooth_rotations(
        rotations[:, indices], fps, cutoff_hz, track.joint_confidence[:, indices]
    )
    if track.rotation_observable is not None and track.num_frames >= 7:
        for index in indices:
            if not any(part in BONE_NAMES[index] for part in ('upper_arm', 'forearm', 'hand')):
                continue
            original = track.local_rotations_xyzw[:, index]
            _, weight = observed_fast_motion(original, track.joint_confidence[:, index],
                track.rotation_observable[:, index], fps, 540/max(fps, 10.))
            filtered = rotations[:, index]
            original = original*np.where(np.sum(original*filtered, axis=1, keepdims=True) < 0, -1, 1)
            blended = (1-.8*weight[:, None])*filtered+.8*weight[:, None]*original
            rotations[:, index] = blended/np.maximum(np.linalg.norm(blended, axis=1, keepdims=True), 1e-9)
    positions, matrices = compute_forward_kinematics_sequence(
        track.root_translation, rotations, actor.rest_offsets_m
    )
    return replace(
        track,
        local_rotations_xyzw=rotations,
        joint_positions=positions.astype(np.float32),
        global_matrices=matrices.astype(np.float32),
        cleanup_config={
            **track.cleanup_config,
            "rotation_filter": "robust_so3",
            "rotation_cutoff_hz": cutoff_hz,
        },
    )
