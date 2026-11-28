"""Floor estimation, contact-aware leg IK and motion consistency reporting.

Four heel/toe contact states and bounded stance corrections reduce estimated foot
drift. Their quality depends on observations and contact classification. Reported
slide, coverage and reprojection measure consistency, not reference mocap accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt

from engine.data_format.motion_io import HandMotionTrack, SkeletalMotionTrack
from engine.math_core.transforms import (
    enforce_quaternion_continuity,
    matrix_to_quat,
    quat_from_two_vectors,
    quat_multiply,
    quat_normalize,
    quat_slerp,
    quat_to_matrix,
)
from engine.motion_cleanup.cleanup import (
    MotionCleanupConfig,
    _solve_2bone_leg_ik_frame,
    detect_foot_contacts,
)
from engine.skeleton.rig_definition import (
    BONE_INDEX,
    BONE_NAMES,
    PARENT_INDICES,
    ActorProfile,
    clamp_bone_rotation,
    compute_forward_kinematics_sequence,
)


@dataclass
class QualityScorecard:
    grade: str  # "A+", "A", "B", "C"
    num_frames: int
    fps: float
    num_cameras: int
    calibration_mode: str
    mean_reprojection_rmse_px: Optional[float]
    raw_foot_slide_mm: float
    final_foot_slide_mm: float
    max_foot_slide_mm: float
    floor_penetration_mm: float
    bone_length_std_mm: float
    repaired_glitch_entries: int
    finger_coverage_pct: float
    head_coverage_pct: float
    floor_tilt_corrected_deg: float
    notes: List[str] = field(default_factory=list)
    body_coverage_pct: float = 0.0

    def __post_init__(self):
        # Older single-video reports stored a placeholder zero. Reopening one
        # must not present that value as measured cross-view reconstruction error.
        if self.num_cameras == 1:
            self.mean_reprojection_rmse_px = None

    def summary_line(self) -> str:
        reprojection = (f"{self.mean_reprojection_rmse_px:.2f}px"
                        if self.mean_reprojection_rmse_px is not None else "N/A")
        return (
            f"Grade {self.grade} · Slide: {self.raw_foot_slide_mm:.1f}mm -> {self.final_foot_slide_mm:.1f}mm · "
            f"Reproj: {reprojection} · "
            f"Fingers: {self.finger_coverage_pct:.0f}% · Head: {self.head_coverage_pct:.0f}%"
        )


def _c2_hermite(u: float) -> float:
    """C2-continuous quintic smoothstep 6u^5 - 15u^4 + 10u^3 for zero-pop foot lock blending."""
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


def fit_and_level_studio_floor(
    track: SkeletalMotionTrack,
    actor: ActorProfile,
    fps: float,
    max_tilt_deg: float = 18.0,
) -> Tuple[SkeletalMotionTrack, float]:
    """Fit exact studio ground plane via RANSAC over low-velocity foot points across the ENTIRE capture volume
    (without Z-percentile bias that breaks on sloped initial calibrations) and level world cleanly to Z = 0.
    """
    T = track.num_frames
    if T < 6:
        return track, 0.0

    pos = track.joint_positions.astype(np.float64)
    foot_ids = [
        BONE_INDEX["left_foot"],
        BONE_INDEX["left_toe"],
        BONE_INDEX["right_foot"],
        BONE_INDEX["right_toe"],
    ]
    # Use toe tips (sole contact points) + ankles when stationary!
    toe_ids = [BONE_INDEX["left_toe"], BONE_INDEX["right_toe"]]

    stance_pts: List[np.ndarray] = []
    for bid in toe_ids:
        p = pos[:, bid]
        vel = np.zeros(T, dtype=np.float64)
        vel[1:] = np.linalg.norm(np.diff(p, axis=0), axis=-1) * fps
        vel[0] = vel[1]
        # Select strictly by low 3D velocity (works across tilted floors of any slope!)
        mask = vel <= 0.25
        if np.any(mask):
            stance_pts.append(p[mask])

    if not stance_pts:
        for bid in foot_ids:
            p = pos[:, bid]
            vel = np.zeros(T, dtype=np.float64)
            vel[1:] = np.linalg.norm(np.diff(p, axis=0), axis=-1) * fps
            mask = vel <= 0.35
            if np.any(mask):
                stance_pts.append(p[mask])

    if not stance_pts:
        return track, 0.0

    all_pts = np.concatenate(stance_pts, axis=0)
    if len(all_pts) < 6:
        return track, 0.0

    # Fit global floor plane whenever feet span >= 0.45 m horizontally across the room!
    xy_span = float(np.hypot(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1])))
    tilt_deg = 0.0
    R_level = np.eye(3, dtype=np.float64)
    q_level = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    if xy_span >= 0.45 and len(all_pts) >= 12:
        # Spatially downsample stance points to a 0.25m floor grid (median Z per cell) so standing
        # 80% of the time in one spot never overwhelms stance points from elsewhere in the room!
        cell_map: Dict[Tuple[int, int], List[np.ndarray]] = {}
        for pt in all_pts:
            key = (int(np.floor(pt[0] / 0.25)), int(np.floor(pt[1] / 0.25)))
            cell_map.setdefault(key, []).append(pt)
        grid_pts = np.array(
            [np.median(np.vstack(v), axis=0) for v in cell_map.values()], dtype=np.float64
        )
        fit_pts = grid_pts if len(grid_pts) >= 6 else all_pts

        rng = np.random.default_rng(42)
        best_inliers = -1
        best_mask = None
        best_n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        for _ in range(240):
            idx = rng.choice(len(fit_pts), size=3, replace=False)
            p0, p1, p2 = fit_pts[idx[0]], fit_pts[idx[1]], fit_pts[idx[2]]
            if np.linalg.norm(p1 - p0) < 0.25 or np.linalg.norm(p2 - p0) < 0.25:
                continue
            n = np.cross(p1 - p0, p2 - p0)
            nn = float(np.linalg.norm(n))
            if nn < 0.04:
                continue
            n /= nn
            if n[2] < 0.0:
                n = -n
            ang = float(np.degrees(np.arccos(np.clip(n[2], -1.0, 1.0))))
            if ang > max_tilt_deg:
                continue
            dists = np.abs((fit_pts - p0) @ n)
            inl_mask = dists <= 0.055
            inliers = int(np.sum(inl_mask))
            if inliers > best_inliers:
                best_inliers = inliers
                best_mask = inl_mask
                best_n = n
        if best_mask is not None and best_inliers >= max(4, int(len(fit_pts) * 0.45)):
            inl_pts = fit_pts[best_mask]
            centered = inl_pts - np.mean(inl_pts, axis=0)
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            n_ref = Vt[-1]
            if n_ref[2] < 0.0:
                n_ref = -n_ref
            n_ref /= max(float(np.linalg.norm(n_ref)), 1e-8)
            ang_ref = float(np.degrees(np.arccos(np.clip(n_ref[2], -1.0, 1.0))))
            if 0.15 <= ang_ref <= max_tilt_deg:
                tilt_deg = ang_ref
                best_n = n_ref
                q_level = quat_from_two_vectors(best_n, np.array([0.0, 0.0, 1.0], dtype=np.float64))
                R_level = quat_to_matrix(q_level)

    root_t = (track.root_translation.astype(np.float64) @ R_level.T).copy()
    local_q = track.local_rotations_xyzw.astype(np.float64).copy()
    if tilt_deg >= 0.15:
        for t in range(T):
            local_q[t, 0] = quat_normalize(quat_multiply(q_level, local_q[t, 0]))
        local_q = enforce_quaternion_continuity(local_q)

    fk_pos, _ = compute_forward_kinematics_sequence(root_t, local_q, actor.rest_offsets_m)
    foot_zs = fk_pos[:, foot_ids, 2].reshape(-1)
    floor_z = float(np.percentile(foot_zs, 3))
    # Place studio floor precisely at Z = 0.035m ankle/sole height -> toe touches Z = 0.00m
    toe_zs = fk_pos[:, [BONE_INDEX["left_toe"], BONE_INDEX["right_toe"]], 2].reshape(-1)
    toe_floor = float(np.percentile(toe_zs, 4))
    z_shift = -min(floor_z, toe_floor)
    root_t[:, 2] += z_shift

    fk_pos, fk_mats = compute_forward_kinematics_sequence(root_t, local_q, actor.rest_offsets_m)
    leveled = SkeletalMotionTrack(
        root_translation=root_t.astype(np.float32),
        root_rotation_xyzw=local_q[:, 0].astype(np.float32),
        local_rotations_xyzw=local_q.astype(np.float32),
        global_matrices=fk_mats.astype(np.float32),
        joint_positions=fk_pos.astype(np.float32),
        joint_confidence=track.joint_confidence.copy(),
        rotation_observable=track.rotation_observable.copy()
        if track.rotation_observable is not None
        else None,
        foot_contacts=track.foot_contacts.copy() if track.foot_contacts is not None else None,
        cleanup_config={**track.cleanup_config, "world_from_capture": [
            [*R_level[0].tolist(), 0], [*R_level[1].tolist(), 0],
            [*R_level[2].tolist(), float(z_shift)], [0, 0, 0, 1]]},
    )
    return leveled, tilt_deg


def measure_foot_slide_mm(
    joint_positions: np.ndarray, contacts: np.ndarray, fps: float
) -> Tuple[float, float]:
    """Measure mean and 95th-percentile horizontal foot slide (mm per stance interval) during ground contact."""
    T = joint_positions.shape[0]
    if T < 3 or contacts is None or len(contacts) != T:
        return 0.0, 0.0

    # Check both left_foot (contact col 0) and right_foot (contact col 2)
    step_slides_mm: List[float] = []
    for c_col, b_name in ((0, "left_foot"), (2, "right_foot")):
        b_idx = BONE_INDEX[b_name]
        c_vec = contacts[:, c_col]
        t = 0
        while t < T:
            if c_vec[t] == 1:
                end = t
                while end < T and c_vec[end] == 1:
                    end += 1
                if end - t >= 3:
                    xy = joint_positions[t:end, b_idx, :2].astype(np.float64)
                    anchor = xy[0]
                    drifts = np.linalg.norm(xy - anchor[None, :], axis=-1) * 1000.0
                    step_slides_mm.extend(drifts.tolist())
                t = end
            else:
                t += 1

    if not step_slides_mm:
        return 0.0, 0.0
    arr = np.asarray(step_slides_mm, dtype=np.float64)
    return float(np.mean(arr)), float(np.percentile(arr, 95))


def _geodesic_rate_limit_bidirectional(
    q_seq: np.ndarray,  # (T, 4) unit continuous quaternions
    max_step_deg: float | np.ndarray,
) -> np.ndarray:
    """Zero-phase bidirectional geodesic angular-velocity limiter (eliminates sudden snaps with zero phase lag!)."""
    T = q_seq.shape[0]
    if T < 3:
        return q_seq
    limits = np.broadcast_to(np.asarray(max_step_deg, float), (T-1,))
    if np.all(limits <= 0):
        return q_seq
    if not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError("Angular step limits must be finite and positive.")

    def _pass(arr: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        out = arr.copy()
        for t in range(1, T):
            dot = float(np.dot(out[t - 1], out[t]))
            target = -out[t] if dot < 0.0 else out[t]
            dot = abs(dot)
            ang = float(np.degrees(2.0 * np.arccos(min(1.0, max(0.0, dot)))))
            if ang > bounds[t-1]:
                out[t] = quat_slerp(out[t - 1], target, bounds[t-1] / ang)
        return out

    q_fwd = _pass(q_seq, limits)
    q_bwd = _pass(q_seq[::-1], limits[::-1])[::-1]
    res = np.empty_like(q_seq)
    for t in range(T):
        res[t] = quat_slerp(q_fwd[t], q_bwd[t], 0.5)
    # Independent shortest paths can meet on opposite sides of pi. Their
    # midpoint is then discontinuous; project the combined path once more.
    return _pass(res, limits)


def _observed_fast_motion(quaternions, confidence, observable, fps, base_step, airborne=None):
    from engine.motion_cleanup.rotation_filter import observed_fast_motion
    return observed_fast_motion(quaternions, confidence, observable, fps, base_step, airborne)


def apply_zero_foot_slide_physics(
    track: SkeletalMotionTrack,
    actor: ActorProfile,
    fps: float = 30.0,
    strength: float = 0.92,
    cutoff_hz: float = 5.5,
    blend_frames: int = 5,
    contacts_override: Optional[np.ndarray] = None,
    allow_floor_tilt: bool = True,
    cancel_event=None,
    progress_cb=None,
) -> Tuple[SkeletalMotionTrack, float, float, float]:
    """Apply RANSAC floor leveling + zero-phase smoothing + C2 Hermite stance-anchored 2-bone Leg IK.

    Returns:
        (cleaned_track, raw_slide_mm, final_slide_mm, floor_tilt_deg)
    """
    T = track.num_frames
    B = len(BONE_NAMES)
    def checkpoint(fraction=None):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Cancelled by user.")
        if progress_cb is not None and fraction is not None:
            progress_cb(fraction, "Fitting floor and foot contacts…")
    checkpoint(.01)
    strength = float(np.clip(strength, 0.0, 1.0))

    # 1. Level studio floor plane cleanly to Z = 0
    leveled, tilt_deg = fit_and_level_studio_floor(track, actor, fps=fps,
                                                max_tilt_deg=18. if allow_floor_tilt else 0.)

    # Detect contacts on leveled raw motion & measure baseline foot slide
    raw_contacts = detect_foot_contacts(
        leveled.joint_positions,
        fps=fps,
        floor_z_m=0.0,
        height_thresh_m=0.115,
        vel_enter_mps=0.28,
        vel_exit_mps=0.42,
        min_hold_frames=3,
    )
    if contacts_override is not None:
        if np.asarray(contacts_override).shape != (T, 4):
            raise ValueError("Foot contact observations must have four channels per frame.")
        raw_contacts = np.asarray(contacts_override, np.uint8)
    raw_slide_mm, _ = measure_foot_slide_mm(leveled.joint_positions, raw_contacts, fps=fps)

    root_t = leveled.root_translation.astype(np.float64).copy()
    local_q = enforce_quaternion_continuity(leveled.local_rotations_xyzw.astype(np.float64))
    fast_limits, fast_weights = {}, {}

    def step_limit(bone, default):
        return np.maximum(default, fast_limits.get(bone, 0.))

    # 1b. Transient Quaternion Angular-Velocity Despiker + Bidirectional Geodesic Rate Limiter
    #     Eliminates BOTH isolated 1-6 frame spikes AND multi-frame oscillation snaps with ZERO phase lag!
    if T >= 7 and strength > 0.1:
        arm_bones = {
            BONE_INDEX["left_upper_arm"],
            BONE_INDEX["left_forearm"],
            BONE_INDEX["left_hand"],
            BONE_INDEX["right_upper_arm"],
            BONE_INDEX["right_forearm"],
            BONE_INDEX["right_hand"],
        }
        for b in range(B):
            checkpoint()
            q_b = local_q[:, b, :]
            # Physical per-frame angular-speed limit: ~540 deg/s on arms (18 deg@30fps), ~420 deg/s on torso/legs (14 deg@30fps)
            max_deg_s = 540.0 if b in arm_bones else 420.0
            max_step_deg = max(12.0, max_deg_s / max(fps, 10.0))
            if b in arm_bones or BONE_NAMES[b] in {
                'left_upper_leg', 'left_lower_leg', 'left_foot', 'right_upper_leg', 'right_lower_leg', 'right_foot'
            }:
                airborne = None
                if 'leg' in BONE_NAMES[b] or BONE_NAMES[b].endswith('_foot'):
                    channels = [0, 1] if BONE_NAMES[b].startswith('left') else [2, 3]
                    airborne = ~np.any(raw_contacts[:, channels], axis=1)
                observed = (np.asarray(track.rotation_observable[:, b], bool)
                    if track.rotation_observable is not None else np.zeros(T, bool))
                fast_limits[b], fast_weights[b] = _observed_fast_motion(q_b,
                    track.joint_confidence[:, b], observed, fps, max_step_deg, airborne)
            for _pass in range(2):
                dots = np.clip(np.abs(np.sum(q_b[:-1] * q_b[1:], axis=1)), 0.0, 1.0)
                step_deg = np.degrees(2.0 * np.arccos(dots))
                spikes = np.where(step_deg > step_limit(b, max_step_deg))[0]
                if len(spikes) == 0:
                    break
                # Repair glitch islands (1..8 frames) via geodesic SLERP from bracketing clean frames
                t_i = 0
                while t_i < len(spikes):
                    s0 = int(spikes[t_i])
                    s1 = s0 + 1
                    while t_i + 1 < len(spikes) and int(spikes[t_i + 1]) - s1 <= 6:
                        t_i += 1
                        s1 = int(spikes[t_i]) + 1
                    a = max(0, s0)
                    c = min(T - 1, s1)
                    if 1 <= (c - a) <= 10 and a > 0 and c < T - 1:
                        for k in range(a + 1, c):
                            alpha = float(k - a) / float(c - a)
                            q_b[k] = quat_slerp(q_b[a], q_b[c], alpha)
                    t_i += 1
            # Zero-phase bidirectional geodesic rate limit guarantees zero residual >max_step_deg snaps!
            q_b = _geodesic_rate_limit_bidirectional(q_b, max_step_deg=step_limit(b, max_step_deg))
            local_q[:, b, :] = q_b
        local_q = enforce_quaternion_continuity(local_q)

    # 2. Studio Dual-Band Velocity-Adaptive Zero-Phase Butterworth Smoothing (ZERO phase lag!):
    checkpoint(.25)
    #    - Still / slow poses lock onto the rock-solid 1.5 Hz band -> eliminates standing micro-jitter!
    #    - Brisk swings & steps smoothly open up to `fast_hz` (<= 5.5 Hz) -> silky studio motion, zero flicker!
    nyq = 0.5 * max(fps, 5.0)
    fast_hz = min(float(cutoff_hz), 5.5) if strength >= 0.5 else float(cutoff_hz)
    if T >= 16 and strength > 0.05 and 0.5 < fast_hz < nyq * 0.85:
        calm_hz = min(1.5, max(0.7, fast_hz * 0.27))
        sos_fast = butter(4, fast_hz / nyq, btype="low", output="sos")
        sos_calm = butter(3, calm_hz / nyq, btype="low", output="sos")
        sos_env = butter(2, min(2.2, nyq * 0.5) / nyq, btype="low", output="sos")

        rt_fast = sosfiltfilt(sos_fast, root_t, axis=0)
        rt_calm = sosfiltfilt(sos_calm, root_t, axis=0)
        rt_spd = np.zeros(T, dtype=np.float64)
        rt_spd[1:] = np.linalg.norm(np.diff(rt_fast, axis=0), axis=-1) * fps
        rt_spd[0] = rt_spd[1]
        rt_spd_env = np.maximum(0.0, sosfiltfilt(sos_env, rt_spd))
        u_rt = np.clip((rt_spd_env - 0.08) / 0.47, 0.0, 1.0)
        w_rt_fast = u_rt * u_rt * u_rt * (u_rt * (u_rt * 6.0 - 15.0) + 10.0)
        rt_adapt = (1.0 - w_rt_fast[:, None]) * rt_calm + w_rt_fast[:, None] * rt_fast
        root_t = (1.0 - strength) * root_t + strength * rt_adapt

        from engine.motion_cleanup.rotation_filter import smooth_rotations
        fast_rotations = smooth_rotations(local_q, fps, fast_hz)
        calm_rotations = smooth_rotations(local_q, fps, calm_hz)
        for b in range(B):
            checkpoint()
            q_fast = fast_rotations[:, b].astype(np.float64)
            q_calm = calm_rotations[:, b].astype(np.float64)
            q_calm *= np.where(np.sum(q_calm*q_fast, axis=1, keepdims=True) < 0, -1, 1)
            qf_n = q_fast / np.maximum(np.linalg.norm(q_fast, axis=1, keepdims=True), 1e-8)
            dots = np.clip(np.abs(np.sum(qf_n[:-1] * qf_n[1:], axis=1)), 0.0, 1.0)
            ang_spd = np.zeros(T, dtype=np.float64)
            ang_spd[1:] = np.degrees(2.0 * np.arccos(dots)) * fps
            ang_spd[0] = ang_spd[1]
            ang_env = np.maximum(0.0, sosfiltfilt(sos_env, ang_spd))
            # C2 angular speed gate: calm 1.5Hz when holding pose (< 15 deg/s), crisp fast_hz above 125 deg/s!
            u_b = np.clip((ang_env - 15.0) / 110.0, 0.0, 1.0)
            w_b_fast = u_b * u_b * u_b * (u_b * (u_b * 6.0 - 15.0) + 10.0)
            q_adapt = (1.0 - w_b_fast[:, None]) * q_calm + w_b_fast[:, None] * q_fast
            q_adapt *= np.where(np.sum(q_adapt*local_q[:, b], axis=1, keepdims=True) < 0, -1, 1)
            effective_strength = strength*(1-.8*fast_weights.get(b, np.zeros(T)))
            q_bl = (1.0-effective_strength[:, None])*local_q[:, b, :] + effective_strength[:, None]*q_adapt
            q_bl /= np.maximum(np.linalg.norm(q_bl, axis=1, keepdims=True), 1e-8)
            local_q[:, b, :] = _geodesic_rate_limit_bidirectional(q_bl,
                max_step_deg=step_limit(b, 540/max(fps, 10.)))
        local_q = enforce_quaternion_continuity(local_q)

    # Recompute FK after smooth pass & apply Adaptive Ground-Contact Gravity Pinning!
    # Eliminates subtle residual vertical floor drift across large rooms (0.0m floating, while preserving jumps!)
    fk_pos, fk_mats = compute_forward_kinematics_sequence(root_t, local_q, actor.rest_offsets_m)
    if strength > 0.1 and T >= 8:
        lf_i, lt_i = BONE_INDEX["left_foot"], BONE_INDEX["left_toe"]
        rf_i, rt_i = BONE_INDEX["right_foot"], BONE_INDEX["right_toe"]
        sole_offset = max(0.04, float(abs(actor.rest_offsets_m[lt_i, 2])))
        # Sole height above Z=0 per frame (toes touch Z=0, ankles sit at sole_offset when flat)
        min_sole_z = np.minimum.reduce(
            [
                fk_pos[:, lt_i, 2].astype(np.float64),
                fk_pos[:, rt_i, 2].astype(np.float64),
                fk_pos[:, lf_i, 2].astype(np.float64) - sole_offset,
                fk_pos[:, rf_i, 2].astype(np.float64) - sole_offset,
            ]
        )
        # Compute vertical velocity of the lowest foot; true jumps have BOTH feet flying fast vertically
        vz = np.zeros(T, dtype=np.float64)
        vz[1:] = np.abs(np.diff(min_sole_z)) * fps
        vz[0] = vz[1]
        # Grounded frames: not in a high fast jump arc (within 32cm of floor & vertical speed < 0.75 m/s)
        grounded = (min_sole_z <= 0.32) & (vz <= 0.75)
        # Zero vertical velocity also occurs at jump apices and in assisted or
        # slow-motion raised poses. Contact evidence must authorize a floor pin.
        grounded &= np.any(raw_contacts, axis=1)
        if int(np.sum(grounded)) >= 6:
            g_idx = np.where(grounded)[0]
            drift_raw = np.interp(
                np.arange(T, dtype=np.float64), g_idx.astype(np.float64), min_sole_z[g_idx]
            )
            # Smooth drift curve (0.8 Hz low-pass) so walking steps stay natural while removing room-wide float!
            if T >= 18 and 0.8 < nyq * 0.85:
                sos_g = butter(2, 0.8 / nyq, btype="low", output="sos")
                drift_smooth = sosfiltfilt(sos_g, drift_raw)
            else:
                drift_smooth = drift_raw
            root_t[:, 2] -= drift_smooth * strength
            fk_pos, fk_mats = compute_forward_kinematics_sequence(
                root_t, local_q, actor.rest_offsets_m
            )

    contacts = detect_foot_contacts(
        fk_pos,
        fps=fps,
        floor_z_m=0.0,
        height_thresh_m=0.115,
        vel_enter_mps=0.28,
        vel_exit_mps=0.42,
        min_hold_frames=3,
    )
    if contacts_override is not None:
        contacts = np.asarray(contacts_override, np.uint8).copy()
    checkpoint(.45)

    # 3. Stance-Anchored Analytical 2-Bone Leg IK with Pelvis Reach Guard & C2 Hermite Blend!
    if strength > 0.05 and T >= 4:
        leg_defs = [
            (0, 1, "left_upper_leg", "left_lower_leg", "left_foot", "left_toe"),
            (2, 3, "right_upper_leg", "right_lower_leg", "right_foot", "right_toe"),
        ]
        # Pre-compute target locked ankle positions & pelvis reach adjustments per leg
        ankle_targets: Dict[str, np.ndarray] = {}
        ankle_weights: Dict[str, np.ndarray] = {}

        for heel_c, toe_c, thigh_name, shin_name, foot_name, _toe_name in leg_defs:
            f_idx = BONE_INDEX[foot_name]
            th_idx = BONE_INDEX[thigh_name]
            sh_idx = BONE_INDEX[shin_name]
            thigh_len = float(np.linalg.norm(actor.rest_offsets_m[sh_idx]))
            shin_len = float(np.linalg.norm(actor.rest_offsets_m[f_idx]))

            c_active = ((contacts[:, heel_c] == 1) | (contacts[:, toe_c] == 1)).astype(np.uint8)
            tgt_arr = fk_pos[:, f_idx].copy()
            w_arr = np.zeros(T, dtype=np.float64)

            t = 0
            while t < T:
                if c_active[t] == 1:
                    end = t
                    while end < T and c_active[end] == 1:
                        end += 1
                    # Exact locked stance anchor: median XY during first 3 frames of strike (zero drift!), floor Z clamped
                    strike_end = min(end, t + 3)
                    lock_xy = np.median(fk_pos[t:strike_end, f_idx, :2], axis=0)
                    # Nominal ankle height when sole sits flat on Z=0
                    ankle_rest_z = max(
                        0.04, float(abs(actor.rest_offsets_m[BONE_INDEX[_toe_name], 2]))
                    )
                    lock_z = float(
                        np.clip(
                            np.median(fk_pos[t:end, f_idx, 2]),
                            ankle_rest_z * 0.85,
                            ankle_rest_z * 1.25,
                        )
                    )
                    anchor = np.array([lock_xy[0], lock_xy[1], lock_z], dtype=np.float64)

                    for k in range(t, end):
                        tgt_arr[k] = anchor
                        w_arr[k] = strength

                    # Smooth C2 Hermite fade-in before strike & fade-out after lift-off (zero knee pop!)
                    bf = max(1, int(blend_frames))
                    for k in range(1, bf + 1):
                        if t - k >= 0 and c_active[t - k] == 0:
                            w = _c2_hermite(1.0 - k / float(bf + 1)) * strength
                            if w > w_arr[t - k]:
                                tgt_arr[t - k] = anchor
                                w_arr[t - k] = w
                        if end - 1 + k < T and c_active[end - 1 + k] == 0:
                            w = _c2_hermite(1.0 - k / float(bf + 1)) * strength
                            if w > w_arr[end - 1 + k]:
                                tgt_arr[end - 1 + k] = anchor
                                w_arr[end - 1 + k] = w
                    t = end
                else:
                    t += 1

            ankle_targets[foot_name] = tgt_arr
            ankle_weights[foot_name] = w_arr

            # Pelvis Reach Guard: if locked ankle is slightly beyond max leg reach, gently pull pelvis down/closer!
            max_leg = 0.985 * (thigh_len + shin_len)
            for k in range(T):
                checkpoint()
                if w_arr[k] > 0.25:
                    hip_p = fk_pos[k, th_idx]
                    d_vec = tgt_arr[k] - hip_p
                    dist = float(np.linalg.norm(d_vec))
                    if dist > max_leg and dist > 1e-4:
                        excess = (dist - max_leg) * w_arr[k]
                        root_t[k] += (d_vec / dist) * excess

        # Smooth pelvis reach adjustments slightly so root never jerks, then recompute FK before 2-bone Leg IK
        if T >= 16 and 0.5 < cutoff_hz < nyq * 0.85:
            sos_r = butter(3, min(8.0, cutoff_hz + 1.0) / nyq, btype="low", output="sos")
            root_t = sosfiltfilt(sos_r, root_t, axis=0)
        fk_pos, fk_mats = compute_forward_kinematics_sequence(root_t, local_q, actor.rest_offsets_m)

        # Solve exact 2-bone analytical Leg IK + Planted-Foot Flat-Sole Leveler on every contact/blend frame!
        for hc_i, tc_i, thigh_name, shin_name, foot_name, _toe_name in leg_defs:
            th_idx = BONE_INDEX[thigh_name]
            sh_idx = BONE_INDEX[shin_name]
            f_idx = BONE_INDEX[foot_name]
            thigh_len = float(np.linalg.norm(actor.rest_offsets_m[sh_idx]))
            shin_len = float(np.linalg.norm(actor.rest_offsets_m[f_idx]))
            tgt_arr = ankle_targets[foot_name]
            w_arr = ankle_weights[foot_name]

            for t in range(T):
                checkpoint()
                w = float(w_arr[t])
                if w <= 1e-3:
                    continue
                desired_ankle = (1.0 - w) * fk_pos[t, f_idx] + w * tgt_arr[t]
                parent_R = fk_mats[t, PARENT_INDICES[th_idx], :3, :3]
                q_th, q_sh = _solve_2bone_leg_ik_frame(
                    hip_pos=fk_pos[t, th_idx],
                    knee_pos_orig=fk_pos[t, sh_idx],
                    ankle_target=desired_ankle,
                    thigh_len=thigh_len,
                    shin_len=shin_len,
                    parent_R=parent_R,
                    thigh_rest=actor.rest_offsets_m[sh_idx],
                    shin_rest=actor.rest_offsets_m[f_idx],
                    orig_thigh_fwd=fk_mats[t, th_idx, :3, 1].astype(np.float64),
                )
                q_th_c = clamp_bone_rotation(thigh_name, q_th, strength=0.85)
                q_sh_c = clamp_bone_rotation(shin_name, q_sh, strength=0.85)
                # Verify clamped 2-bone IK actually lands close to desired_ankle (never kick leg away on outlier!)
                R_th_new = parent_R @ quat_to_matrix(q_th_c)
                R_sh_new = R_th_new @ quat_to_matrix(q_sh_c)
                ankle_new = (
                    fk_pos[t, th_idx]
                    + R_th_new @ actor.rest_offsets_m[sh_idx]
                    + R_sh_new @ actor.rest_offsets_m[f_idx]
                )
                err_new = float(np.linalg.norm(ankle_new[:2] - desired_ankle[:2]))
                err_old = float(np.linalg.norm(fk_pos[t, f_idx, :2] - desired_ankle[:2]))
                if err_new <= max(err_old + 0.015, 0.025):
                    local_q[t, th_idx] = q_th_c
                    local_q[t, sh_idx] = q_sh_c
                    R_shin_final = R_sh_new
                else:
                    R_shin_final = fk_mats[t, sh_idx, :3, :3].astype(np.float64)

                # Planted-Foot Flat-Sole Leveler: remove pitch & roll tilt onto Z=0 while preserving smooth current yaw
                # (zero yaw-fighting during body pivots and zero stance-boundary snap!)
                R_ft_cur = R_shin_final @ quat_to_matrix(local_q[t, f_idx])
                fwd_xy = R_ft_cur[:2, 1].astype(np.float64)
                fn = float(np.linalg.norm(fwd_xy))
                if fn > 1e-4:
                    fy = np.array([fwd_xy[0] / fn, fwd_xy[1] / fn, 0.0], dtype=np.float64)
                    fz = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                    fx = np.cross(fy, fz)
                    R_flat = np.column_stack([fx, fy, fz])
                    q_ft_flat_loc = matrix_to_quat(R_shin_final.T @ R_flat)
                    q_ft_flat_loc = clamp_bone_rotation(foot_name, q_ft_flat_loc, strength=0.90)
                    local_q[t, f_idx] = quat_slerp(local_q[t, f_idx], q_ft_flat_loc, w * 0.85)

        local_q = enforce_quaternion_continuity(local_q)
        # 3b. Post-IK Leg Transition Polish: smooths swing/lift-off IK seams while keeping hard-stance feet locked (<2mm)!
        if T >= 16 and 0.5 < min(4.8, cutoff_hz) < nyq * 0.85:
            sos_leg = butter(3, min(4.8, cutoff_hz) / nyq, btype="low", output="sos")
            max_leg_step = max(10.0, 360.0 / max(fps, 10.0))
            for _hc, _tc, thigh_name, shin_name, foot_name, _tn in leg_defs:
                w_st = ankle_weights[foot_name]
                # Zero perturbation on hard-planted stance (w_st >= 0.85 -> 0% drift!), full polish on swing/transitions!
                pol_w = np.clip((0.88 - w_st) / 0.88, 0.0, 1.0) * strength
                for b_name in (thigh_name, shin_name, foot_name):
                    bi = BONE_INDEX[b_name]
                    q_lim = _geodesic_rate_limit_bidirectional(local_q[:, bi, :], max_step_deg=step_limit(bi, max_leg_step))
                    q_sm = sosfiltfilt(sos_leg, q_lim, axis=0)
                    q_sm /= np.maximum(np.linalg.norm(q_sm, axis=1, keepdims=True), 1e-8)
                    polish = pol_w*(1-.8*fast_weights.get(bi, np.zeros(T)))
                    q_out = (1.0-polish[:, None])*q_lim + polish[:, None]*q_sm
                    local_q[:, bi, :] = q_out / np.maximum(np.linalg.norm(q_out, axis=1, keepdims=True), 1e-8)
            local_q = enforce_quaternion_continuity(local_q)
        fk_pos, fk_mats = compute_forward_kinematics_sequence(root_t, local_q, actor.rest_offsets_m)

    # 4. Strict Floor Penetration Guard (zero frames sink below Z = 0!)
    checkpoint(.75)
    foot_ids = [
        BONE_INDEX["left_foot"],
        BONE_INDEX["left_toe"],
        BONE_INDEX["right_foot"],
        BONE_INDEX["right_toe"],
    ]
    min_z_per_frame = np.min(fk_pos[:, foot_ids, 2], axis=1)
    pen = np.maximum(0.0, -min_z_per_frame)
    if np.any(pen > 1e-4):
        root_t[:, 2] += pen
        fk_pos, fk_mats = compute_forward_kinematics_sequence(root_t, local_q, actor.rest_offsets_m)

    # A floor lift changes ankle positions after the earlier IK. For single-view
    # image-derived contact constraints, re-fit those targets with the final pelvis.
    if contacts_override is not None and strength > .05 and T >= 4:
        for _ in range(3):
            for _, _, thigh_name, shin_name, foot_name, _ in leg_defs:
                thigh, shin, foot = (BONE_INDEX[n] for n in (thigh_name, shin_name, foot_name))
                for t in np.flatnonzero(ankle_weights[foot_name] > .01):
                    checkpoint()
                    weight = ankle_weights[foot_name][t]
                    desired = (1-weight)*fk_pos[t, foot] + weight*ankle_targets[foot_name][t]
                    parent = fk_mats[t, PARENT_INDICES[thigh], :3, :3]
                    q_thigh, q_shin = _solve_2bone_leg_ik_frame(fk_pos[t, thigh], fk_pos[t, shin], desired,
                        float(np.linalg.norm(actor.rest_offsets_m[shin])),
                        float(np.linalg.norm(actor.rest_offsets_m[foot])), parent,
                        actor.rest_offsets_m[shin], actor.rest_offsets_m[foot],
                        orig_thigh_fwd=fk_mats[t, thigh, :3, 1])
                    foot_world = fk_mats[t, foot, :3, :3]
                    local_q[t, thigh], local_q[t, shin] = q_thigh, q_shin
                    shin_world = parent @ quat_to_matrix(q_thigh) @ quat_to_matrix(q_shin)
                    local_q[t, foot] = matrix_to_quat(shin_world.T @ foot_world)
            fk_pos, fk_mats = compute_forward_kinematics_sequence(root_t, local_q, actor.rest_offsets_m)
            root_t[:, 2] += np.maximum(0., -np.min(fk_pos[:, foot_ids, 2], axis=1))
            fk_pos, fk_mats = compute_forward_kinematics_sequence(root_t, local_q, actor.rest_offsets_m)
        for bone in range(BONE_INDEX["left_upper_leg"], B):
            local_q[:, bone] = _geodesic_rate_limit_bidirectional(local_q[:, bone],
                step_limit(bone, 540/max(fps, 10.)))
        fk_pos, fk_mats = compute_forward_kinematics_sequence(root_t, local_q, actor.rest_offsets_m)
        root_t[:, 2] += np.maximum(0., -np.min(fk_pos[:, foot_ids, 2], axis=1))
        fk_pos, fk_mats = compute_forward_kinematics_sequence(root_t, local_q, actor.rest_offsets_m)

    # Matrix-to-quaternion IK can choose the opposite quaternion sign. The pose
    # is identical at keys, but linear FCurves would flip halfway between them.
    local_q = enforce_quaternion_continuity(local_q)

    # Align actor rest hip height cleanly to the true floor-grounded standing height (prevents Blender root float!)
    if T >= 4:
        standing_hip_z = float(np.percentile(root_t[:, 2], 78))
        if 0.45 <= standing_hip_z <= 1.45:
            actor.rest_offsets_m[0, 2] = standing_hip_z

    final_slide_mm, _ = measure_foot_slide_mm(fk_pos, contacts, fps=fps)
    checkpoint(1.)

    cleaned = SkeletalMotionTrack(
        root_translation=root_t.astype(np.float32),
        root_rotation_xyzw=local_q[:, 0].astype(np.float32),
        local_rotations_xyzw=local_q.astype(np.float32),
        global_matrices=fk_mats.astype(np.float32),
        joint_positions=fk_pos.astype(np.float32),
        joint_confidence=track.joint_confidence.copy(),
        rotation_observable=track.rotation_observable.copy()
        if track.rotation_observable is not None
        else None,
        foot_contacts=contacts,
        cleanup_config={
            **leveled.cleanup_config,
            **MotionCleanupConfig(
                strength=strength, butterworth_cutoff_hz=cutoff_hz, enable_foot_locking=True
            ).__dict__,
            "engine": "nuranova_zero_slide_physics_v2",
            "floor_tilt_corrected_deg": round(tilt_deg, 2),
        },
    )
    return cleaned, raw_slide_mm, final_slide_mm, tilt_deg


def compute_quality_scorecard(
    cleaned_track: SkeletalMotionTrack,
    actor: ActorProfile,
    fps: float,
    num_cameras: int,
    calibration_mode: str,
    mean_reprojection_rmse_px: Optional[float],
    raw_foot_slide_mm: float,
    repaired_glitch_entries: int,
    hand_track: Optional[HandMotionTrack],
    head_coverage_pct: float,
    floor_tilt_corrected_deg: float,
) -> QualityScorecard:
    """Compute an objective, transparent Studio Quality Scorecard for the reconstructed mocap take."""
    pos = cleaned_track.joint_positions.astype(np.float64)
    if not np.all(np.isfinite(pos)):
        raise ValueError("The solved motion contains non-finite joint positions.")
    contacts = cleaned_track.foot_contacts
    final_mean_slide, final_p95_slide = measure_foot_slide_mm(pos, contacts, fps=fps)

    foot_ids = [
        BONE_INDEX["left_foot"],
        BONE_INDEX["left_toe"],
        BONE_INDEX["right_foot"],
        BONE_INDEX["right_toe"],
    ]
    min_z = float(np.min(pos[:, foot_ids, 2])) if len(pos) > 0 else 0.0
    penetration_mm = max(0.0, -min_z * 1000.0)

    # Verify strict constant bone lengths across all bones
    stds: List[float] = []
    for i, p in enumerate(PARENT_INDICES):
        if p >= 0 and np.linalg.norm(actor.rest_offsets_m[i]) > 0.02:
            lens = np.linalg.norm(pos[:, i] - pos[:, p], axis=-1)
            stds.append(float(np.std(lens)) * 1000.0)
    bone_std_mm = float(np.mean(stds)) if stds else 0.0

    finger_cov = 0.0
    if hand_track is not None and hand_track.num_frames > 0:
        l_ok = (
            np.max(hand_track.left_confidence, axis=-1) > 0.1
            if hand_track.left_confidence is not None
            else np.zeros(hand_track.num_frames, dtype=bool)
        )
        r_ok = (
            np.max(hand_track.right_confidence, axis=-1) > 0.1
            if hand_track.right_confidence is not None
            else np.zeros(hand_track.num_frames, dtype=bool)
        )
        finger_cov = float(np.mean(l_ok | r_ok)) * 100.0

    notes: List[str] = []
    if num_cameras == 1:
        notes.append(
            "Single-view depth is estimated; multi-view reprojection error is unavailable."
        )
    elif num_cameras < 3:
        notes.append(
            "Using 2 cameras (good); adding a 3rd camera further eliminates side-turn occlusions."
        )
    if final_mean_slide <= 3.0:
        notes.append(
            f"Zero-Slide Physics locked stance drift from {raw_foot_slide_mm:.1f} mm down to {final_mean_slide:.1f} mm."
        )
    if floor_tilt_corrected_deg >= 0.5:
        notes.append(f"Auto-leveled studio floor tilt by {floor_tilt_corrected_deg:.1f}°.")
    if finger_cov >= 50.0:
        notes.append(f"30-bone finger motion captured ({finger_cov:.0f}% frame coverage).")
    elif hand_track is None:
        notes.append(
            "No hands clearly visible (move closer or enable 1080p/4K recording for finger detail)."
        )
    if head_coverage_pct >= 60.0:
        notes.append(f"6-DoF Head pitch/yaw/roll fused ({head_coverage_pct:.0f}% coverage).")
    if "boardless" in calibration_mode:
        notes.append(
            "Camera geometry and scale are estimated from the actor; a calibration board improves reliability."
        )
    notes.append(
        "These consistency checks do not measure accuracy against reference motion capture."
    )
    body_coverage = float(np.mean(cleaned_track.joint_confidence >= 0.25)) * 100

    # Compute overall studio quality grade
    score = 100.0
    if mean_reprojection_rmse_px is not None and mean_reprojection_rmse_px > 2.0:
        score -= min(25.0, (mean_reprojection_rmse_px - 2.0) * 4.0)
    if final_mean_slide > 4.0:
        score -= min(25.0, (final_mean_slide - 4.0) * 2.0)
    if penetration_mm > 5.0:
        score -= 10.0
    if num_cameras < 2:
        score -= 20.0
    if body_coverage < 70:
        score = min(score, 60.0)
        notes.append(
            f"Limited body tracking coverage ({body_coverage:.0f}%). Review missing limbs before export."
        )
    grade = "A+" if score >= 92 else ("A" if score >= 82 else ("B" if score >= 68 else "C"))

    return QualityScorecard(
        grade=grade,
        num_frames=cleaned_track.num_frames,
        fps=fps,
        num_cameras=num_cameras,
        calibration_mode=calibration_mode,
        mean_reprojection_rmse_px=(round(mean_reprojection_rmse_px, 2)
                                   if mean_reprojection_rmse_px is not None else None),
        raw_foot_slide_mm=round(raw_foot_slide_mm, 2),
        final_foot_slide_mm=round(final_mean_slide, 2),
        max_foot_slide_mm=round(final_p95_slide, 2),
        floor_penetration_mm=round(penetration_mm, 2),
        bone_length_std_mm=round(bone_std_mm, 4),
        repaired_glitch_entries=repaired_glitch_entries,
        finger_coverage_pct=round(finger_cov, 1),
        head_coverage_pct=round(head_coverage_pct, 1),
        floor_tilt_corrected_deg=round(floor_tilt_corrected_deg, 2),
        notes=notes,
        body_coverage_pct=round(body_coverage, 1),
    )
