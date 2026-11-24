"""Non-destructive motion cleanup, adaptive filtering, gap recovery, and clean-room 2-bone IK foot-locking.

LICENSING NOTE: This module is an original Nuranova Solutions clean-room implementation using standard
SciPy signal processing (BSD-3-Clause) and analytical 2-bone leg IK. No code or weights from the
non-commercial UnderPressure research repository are used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, sosfiltfilt

from engine.data_format.motion_io import SkeletalMotionTrack
from engine.math_core.transforms import (
    enforce_quaternion_continuity,
    quat_from_two_vectors,
    quat_normalize,
    quat_slerp,
    quat_to_matrix,
)
from engine.skeleton.rig_definition import (
    BONE_INDEX,
    BONE_NAMES,
    ActorProfile,
    clamp_bone_rotation,
    compute_forward_kinematics_sequence,
)


@dataclass
class MotionCleanupConfig:
    strength: float = 0.75  # Master strength [0.0 .. 1.0]
    enable_outlier_removal: bool = True
    outlier_mad_sigma: float = 4.5
    enable_gap_interpolation: bool = True
    max_gap_frames: int = 12
    filter_type: str = "butterworth"  # "butterworth", "one_euro", "none"
    butterworth_cutoff_hz: float = 7.5
    butterworth_order: int = 4
    one_euro_min_cutoff: float = 1.5
    one_euro_beta: float = 0.08
    enable_anatomical_limits: bool = True
    enable_foot_locking: bool = True
    floor_z_m: float = 0.0
    foot_contact_height_m: float = 0.115
    foot_contact_vel_mps: float = 0.25
    foot_release_vel_mps: float = 0.38
    foot_min_hold_frames: int = 3
    foot_blend_frames: int = 4
    enable_ground_clamp: bool = True
    enable_root_stabilization: bool = True


def _one_euro_filter_1d(signal: np.ndarray, fps: float, min_cutoff: float = 1.5, beta: float = 0.08, d_cutoff: float = 1.0) -> np.ndarray:
    """Casiez et al. (2012) One Euro adaptive low-pass filter."""
    T = len(signal)
    if T < 3:
        return signal.copy()
    dt = 1.0 / max(fps, 1.0)

    def alpha(cutoff: float) -> float:
        tau = 1.0 / (2.0 * np.pi * max(cutoff, 1e-4))
        return 1.0 / (1.0 + tau / dt)

    out = np.empty_like(signal, dtype=np.float64)
    out[0] = signal[0]
    dx_hat = 0.0
    a_d = alpha(d_cutoff)
    for t in range(1, T):
        dx = (signal[t] - out[t - 1]) / dt
        dx_hat = a_d * dx + (1.0 - a_d) * dx_hat
        cutoff = min_cutoff + beta * abs(dx_hat)
        a = alpha(cutoff)
        out[t] = a * signal[t] + (1.0 - a) * out[t - 1]
    return out


def _butterworth_zero_phase(arr: np.ndarray, fps: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth filter along axis 0 without introducing phase lag."""
    T = arr.shape[0]
    nyq = 0.5 * fps
    if T < max(15, 3 * order + 3) or cutoff_hz >= nyq * 0.92 or cutoff_hz <= 0.1:
        return arr.copy()
    sos = butter(order, cutoff_hz / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, arr, axis=0)


def despike_and_fill_3d_points(
    points_3d: np.ndarray,  # (T, K, 3)
    confidences: np.ndarray,  # (T, K)
    max_gap_frames: int = 12,
    mad_sigma: float = 4.5,
    min_conf: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove velocity outliers and interpolate short missing gaps via shape-preserving PCHIP splines."""
    T, K, _ = points_3d.shape
    out = np.asarray(points_3d, dtype=np.float64).copy()
    interpolated_mask = np.zeros((T, K), dtype=np.bool_)

    for k in range(K):
        valid = (confidences[:, k] >= min_conf) & np.all(np.isfinite(out[:, k]), axis=-1)
        # Velocity-based MAD despiking
        if np.sum(valid) >= 6:
            idx_v = np.where(valid)[0]
            diffs = np.linalg.norm(np.diff(out[idx_v, k], axis=0), axis=-1)
            dt = np.maximum(np.diff(idx_v), 1)
            speeds = diffs / dt
            med = float(np.median(speeds))
            mad = float(np.median(np.abs(speeds - med))) + 1e-6
            spike_pos = np.where(speeds > med + mad_sigma * 1.4826 * mad)[0]
            for sp in spike_pos:
                # Mark the transient spike frame invalid if it jumps and returns
                cand = idx_v[sp + 1]
                valid[cand] = False

        idx_valid = np.where(valid)[0]
        if len(idx_valid) < 2:
            continue

        # Identify contiguous gaps <= max_gap_frames
        for g_i in range(len(idx_valid) - 1):
            a, b = int(idx_valid[g_i]), int(idx_valid[g_i + 1])
            gap_len = b - a - 1
            if 0 < gap_len <= max_gap_frames:
                # Local PCHIP spline over surrounding valid anchors
                lo = max(0, g_i - 2)
                hi = min(len(idx_valid), g_i + 4)
                anch_t = idx_valid[lo:hi]
                if len(anch_t) >= 2:
                    t_fill = np.arange(a + 1, b)
                    for axis in range(3):
                        interp = PchipInterpolator(anch_t, out[anch_t, k, axis])
                        out[t_fill, k, axis] = interp(t_fill)
                    interpolated_mask[t_fill, k] = True

    return out.astype(np.float32), interpolated_mask


def detect_foot_contacts(
    joint_positions: np.ndarray,  # (T, B, 3)
    fps: float,
    floor_z_m: float = 0.0,
    height_thresh_m: float = 0.095,
    vel_enter_mps: float = 0.22,
    vel_exit_mps: float = 0.35,
    min_hold_frames: int = 3,
) -> np.ndarray:
    """Detect binary foot contact states [L_heel, L_toe, R_heel, R_toe] using height + dual-threshold velocity hysteresis."""
    T = joint_positions.shape[0]
    contacts = np.zeros((T, 4), dtype=np.uint8)
    if T < 2:
        return contacts

    bone_ids = [
        BONE_INDEX["left_foot"],
        BONE_INDEX["left_toe"],
        BONE_INDEX["right_foot"],
        BONE_INDEX["right_toe"],
    ]

    for c_idx, b_idx in enumerate(bone_ids):
        pos = np.asarray(joint_positions[:, b_idx], dtype=np.float64)
        vel_xy = np.zeros(T, dtype=np.float64)
        vel_xy[1:] = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=-1) * fps
        vel_xy[0] = vel_xy[1]
        heights = pos[:, 2] - floor_z_m

        state = False
        raw = np.zeros(T, dtype=np.uint8)
        for t in range(T):
            if not state:
                if heights[t] <= height_thresh_m and vel_xy[t] <= vel_enter_mps:
                    state = True
            else:
                if heights[t] > height_thresh_m * 1.35 or vel_xy[t] > vel_exit_mps:
                    state = False
            raw[t] = 1 if state else 0

        # Remove ultra-short spurious 1-2 frame flicker contacts
        t = 0
        while t < T:
            if raw[t] == 1:
                end = t
                while end < T and raw[end] == 1:
                    end += 1
                if (end - t) < min_hold_frames:
                    raw[t:end] = 0
                t = end
            else:
                t += 1
        contacts[:, c_idx] = raw

    return contacts


def _solve_2bone_leg_ik_frame(
    hip_pos: np.ndarray,
    knee_pos_orig: np.ndarray,
    ankle_target: np.ndarray,
    thigh_len: float,
    shin_len: float,
    parent_R: np.ndarray,
    thigh_rest: np.ndarray,
    shin_rest: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact analytical 2-bone leg IK preserving constant thigh & shin lengths to reach ankle_target."""
    hip_to_target = ankle_target - hip_pos
    dist = float(np.linalg.norm(hip_to_target))
    max_reach = max(thigh_len + shin_len - 1e-4, 1e-4)
    min_reach = abs(thigh_len - shin_len) + 1e-4
    d_clamped = float(np.clip(dist, min_reach, max_reach))

    dir_ht = hip_to_target / max(dist, 1e-8)

    # Pole plane determined by original knee offset from hip-target line
    knee_vec = knee_pos_orig - hip_pos
    pole = knee_vec - np.dot(knee_vec, dir_ht) * dir_ht
    pn = float(np.linalg.norm(pole))
    if pn < 1e-5:
        pole = parent_R[:, 1] - np.dot(parent_R[:, 1], dir_ht) * dir_ht
        pn = float(np.linalg.norm(pole))
    pole = pole / max(pn, 1e-8)

    # Law of cosines for hip-to-knee angle
    cos_a = float(np.clip((thigh_len**2 + d_clamped**2 - shin_len**2) / (2.0 * thigh_len * d_clamped), -1.0, 1.0))
    sin_a = float(np.sqrt(max(0.0, 1.0 - cos_a**2)))
    new_knee = hip_pos + thigh_len * (cos_a * dir_ht + sin_a * pole)
    new_ankle = hip_pos + d_clamped * dir_ht

    # Recover exact local rotations from new hip -> new_knee -> new_ankle
    u_dir = (new_knee - hip_pos) / max(thigh_len, 1e-6)
    l_dir = (new_ankle - new_knee) / max(shin_len, 1e-6)

    u_rest = thigh_rest / max(float(np.linalg.norm(thigh_rest)), 1e-6)
    l_rest = shin_rest / max(float(np.linalg.norm(shin_rest)), 1e-6)

    # Build consistent upper-leg frame using pole as +Y forward reference
    u_sec = pole - np.dot(pole, u_dir) * u_dir
    u_sec /= max(float(np.linalg.norm(u_sec)), 1e-8)
    u_third = np.cross(u_dir, u_sec)
    R_world_u = np.column_stack([u_dir, u_sec, u_third])

    r_sec = np.array([0.0, 1.0, 0.0]) - np.dot([0.0, 1.0, 0.0], u_rest) * u_rest
    r_sec /= max(float(np.linalg.norm(r_sec)), 1e-8)
    R_rest_u = np.column_stack([u_rest, r_sec, np.cross(u_rest, r_sec)])

    R_upper_w = R_world_u @ R_rest_u.T
    q_bend = quat_from_two_vectors(R_upper_w @ l_rest, l_dir)
    R_lower_w = quat_to_matrix(q_bend) @ R_upper_w

    from engine.math_core.transforms import matrix_to_quat

    q_ul_local = matrix_to_quat(parent_R.T @ R_upper_w)
    q_ll_local = matrix_to_quat(R_upper_w.T @ R_lower_w)
    return q_ul_local, q_ll_local


def clean_skeletal_motion(
    raw_track: SkeletalMotionTrack,
    actor: ActorProfile,
    fps: float,
    config: Optional[MotionCleanupConfig] = None,
    frame_range: Optional[Tuple[int, int]] = None,
) -> SkeletalMotionTrack:
    """Run configurable non-destructive motion cleanup returning a new corrected SkeletalMotionTrack."""
    cfg = config or MotionCleanupConfig()
    strength = float(np.clip(cfg.strength, 0.0, 1.0))

    root_t = np.asarray(raw_track.root_translation, dtype=np.float64).copy()
    local_q = enforce_quaternion_continuity(np.asarray(raw_track.local_rotations_xyzw, dtype=np.float64).copy())
    T, B, _ = local_q.shape
    rest = actor.rest_offsets_m

    f0 = 0 if frame_range is None else max(0, int(frame_range[0]))
    f1 = T if frame_range is None else min(T, int(frame_range[1]))

    if strength > 0.0 and (f1 - f0) >= 3:
        sub_root = root_t[f0:f1].copy()
        sub_q = local_q[f0:f1].copy()

        # 1. Temporal smoothing on root & continuous unit quaternions
        if cfg.filter_type == "butterworth":
            cutoff = max(1.5, cfg.butterworth_cutoff_hz * (1.5 - 0.8 * strength))
            if cfg.enable_root_stabilization:
                filt_root = _butterworth_zero_phase(sub_root, fps, cutoff_hz=cutoff, order=cfg.butterworth_order)
                sub_root = (1.0 - strength) * sub_root + strength * filt_root
            for b in range(B):
                filt_q = _butterworth_zero_phase(sub_q[:, b, :], fps, cutoff_hz=cutoff, order=cfg.butterworth_order)
                filt_q = quat_normalize(filt_q)
                for ti in range(len(sub_q)):
                    sub_q[ti, b] = quat_slerp(sub_q[ti, b], filt_q[ti], strength)
        elif cfg.filter_type == "one_euro":
            mc = max(0.3, cfg.one_euro_min_cutoff * (1.4 - 0.7 * strength))
            if cfg.enable_root_stabilization:
                for ax in range(3):
                    sub_root[:, ax] = _one_euro_filter_1d(sub_root[:, ax], fps, min_cutoff=mc, beta=cfg.one_euro_beta)
            for b in range(B):
                for ax in range(4):
                    sub_q[:, b, ax] = _one_euro_filter_1d(sub_q[:, b, ax], fps, min_cutoff=mc, beta=cfg.one_euro_beta)
                sub_q[:, b] = quat_normalize(sub_q[:, b])

        # 2. Anatomical joint limits
        if cfg.enable_anatomical_limits:
            for ti in range(len(sub_q)):
                for b_idx, b_name in enumerate(BONE_NAMES):
                    if b_idx > 0:
                        sub_q[ti, b_idx] = clamp_bone_rotation(b_name, sub_q[ti, b_idx], strength=strength)

        root_t[f0:f1] = sub_root
        local_q[f0:f1] = enforce_quaternion_continuity(sub_q)

    # Compute FK before foot-locking
    pos_fk, mats_fk = compute_forward_kinematics_sequence(root_t, local_q, rest)
    foot_contacts = detect_foot_contacts(
        pos_fk,
        fps=fps,
        floor_z_m=cfg.floor_z_m,
        height_thresh_m=cfg.foot_contact_height_m,
        vel_enter_mps=cfg.foot_contact_vel_mps,
        vel_exit_mps=cfg.foot_release_vel_mps,
        min_hold_frames=cfg.foot_min_hold_frames,
    )

    # 3. Ground penetration clamping (raise root smoothly if lowest foot/toe penetrates floor_z)
    if cfg.enable_ground_clamp and strength > 0.0:
        foot_indices = [
            BONE_INDEX["left_foot"],
            BONE_INDEX["left_toe"],
            BONE_INDEX["right_foot"],
            BONE_INDEX["right_toe"],
        ]
        min_z = np.min(pos_fk[:, foot_indices, 2], axis=1)
        penetration = np.maximum(0.0, cfg.floor_z_m - min_z)
        root_t[f0:f1, 2] += penetration[f0:f1] * strength
        pos_fk, mats_fk = compute_forward_kinematics_sequence(root_t, local_q, rest)

    # 4. Clean-room 2-Bone IK Foot-Locking (eliminates footskate while preserving constant leg bone lengths)
    if cfg.enable_foot_locking and strength > 0.0:
        for c_col, ul_b, ll_b, ft_b in [
            (0, BONE_INDEX["left_upper_leg"], BONE_INDEX["left_lower_leg"], BONE_INDEX["left_foot"]),
            (2, BONE_INDEX["right_upper_leg"], BONE_INDEX["right_lower_leg"], BONE_INDEX["right_foot"]),
        ]:
            thigh_len = float(np.linalg.norm(rest[ll_b]))
            shin_len = float(np.linalg.norm(rest[ft_b]))
            contact_col = foot_contacts[:, c_col]

            # Find contiguous contact intervals and compute anchor XY + floor-clamped Z
            t = f0
            while t < f1:
                if contact_col[t] == 1:
                    end = t
                    while end < f1 and contact_col[end] == 1:
                        end += 1
                    anchor_xy = np.median(pos_fk[t:end, ft_b, :2], axis=0)
                    anchor_z = max(cfg.floor_z_m + 0.075, float(np.median(pos_fk[t:end, ft_b, 2])))
                    anchor_pos = np.array([anchor_xy[0], anchor_xy[1], anchor_z], dtype=np.float64)

                    # Apply IK over contact span with smooth ramp-in / ramp-out blending
                    blend_n = max(1, min(cfg.foot_blend_frames, (end - t) // 2))
                    for ti in range(t, end):
                        w_in = min(1.0, (ti - t + 1) / blend_n)
                        w_out = min(1.0, (end - ti) / blend_n)
                        w = strength * min(w_in, w_out)
                        orig_ankle = pos_fk[ti, ft_b].astype(np.float64)
                        target_ankle = (1.0 - w) * orig_ankle + w * anchor_pos
                        hip_pos = pos_fk[ti, ul_b].astype(np.float64)
                        knee_pos = pos_fk[ti, ll_b].astype(np.float64)
                        parent_R = mats_fk[ti, 0, :3, :3].astype(np.float64)

                        q_ul, q_ll = _solve_2bone_leg_ik_frame(
                            hip_pos, knee_pos, target_ankle, thigh_len, shin_len, parent_R, rest[ll_b], rest[ft_b]
                        )
                        local_q[ti, ul_b] = quat_slerp(local_q[ti, ul_b], q_ul, w)
                        local_q[ti, ll_b] = quat_slerp(local_q[ti, ll_b], q_ll, w)
                    t = end
                else:
                    t += 1

        local_q = enforce_quaternion_continuity(local_q)
        pos_fk, mats_fk = compute_forward_kinematics_sequence(root_t, local_q, rest)

    return SkeletalMotionTrack(
        root_translation=root_t.astype(np.float32),
        root_rotation_xyzw=local_q[:, 0, :].astype(np.float32),
        local_rotations_xyzw=local_q.astype(np.float32),
        global_matrices=mats_fk.astype(np.float32),
        joint_positions=pos_fk.astype(np.float32),
        joint_confidence=raw_track.joint_confidence.copy(),
        rotation_observable=raw_track.rotation_observable.copy() if raw_track.rotation_observable is not None else None,
        foot_contacts=foot_contacts,
        cleanup_config=asdict(cfg),
    )
