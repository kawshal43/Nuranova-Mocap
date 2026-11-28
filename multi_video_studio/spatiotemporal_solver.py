"""Spatio-Temporal Multi-View 3D Trajectory Optimizer + 6-DoF Head/Neck Fusion & Full-Body Kinematic Solver.

Eliminates single-frame occlusion glitches, limb stretching, and stiff/jittery head turns by:
1. MAD despiking + PCHIP gap recovery on triangulated 3D landmarks.
2. Coupled Iterative Bone-Length & Zero-Phase Temporal Projection (enforces rigid anatomical segment
   lengths across the entire take BEFORE skeletal IK runs!).
3. Auto-calibrating actor bone proportions from high-confidence 3D frames.
4. Fusing 3D triangulated head landmarks (eyes/ears/nose) + high-res head-cropped PnP head orientation
   naturally across Neck (35%) + Head (65%) with anatomical limits and quaternion continuity.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import butter, savgol_filter, sosfiltfilt
from scipy.spatial.transform import Rotation

from engine.data_format.motion_io import FaceMotionTrack, SkeletalMotionTrack
from engine.math_core.transforms import (
    enforce_quaternion_continuity,
    matrix_to_quat,
    quat_identity,
    quat_inverse,
    quat_multiply,
    quat_normalize,
    quat_slerp,
)
from engine.motion_cleanup.cleanup import despike_and_fill_3d_points
from engine.skeleton.rig_definition import (
    BONE_INDEX,
    ActorProfile,
    calibrate_actor_from_body26_frames,
    clamp_bone_rotation,
    compute_forward_kinematics_sequence,
)
from engine.skeleton.solver import ArticulatedSkeletonSolver, SolverConfig

# Canonical anatomical rigid bone segments in Body26 indices
_RIGID_SEGMENTS_BODY26: List[Tuple[int, int]] = [
    (19, 18),  # pelvis -> neck (torso)
    (18, 5),  # neck -> L shoulder
    (18, 6),  # neck -> R shoulder
    (5, 7),  # L upper arm
    (7, 9),  # L forearm
    (6, 8),  # R upper arm
    (8, 10),  # R forearm
    (19, 11),  # pelvis -> L hip
    (19, 12),  # pelvis -> R hip
    (11, 13),  # L thigh
    (13, 15),  # L shin
    (12, 14),  # R thigh
    (14, 16),  # R shin
    (15, 20),  # L ankle -> L big toe
    (16, 21),  # R ankle -> R big toe
    (15, 24),  # L ankle -> L heel
    (16, 25),  # R ankle -> R heel
    (18, 17),  # neck -> head_top
]


def optimize_3d_trajectories_spatiotemporal(
    joints_3d: np.ndarray,  # (T, 26, 3)
    joints_conf: np.ndarray,  # (T, 26)
    fps: float = 30.0,
    iterations: int = 8,
    cutoff_hz: float = 8.0,
    pin_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Despike, gap-fill, and jointly enforce constant anatomical bone lengths + temporal smoothness.

    Returns:
        (optimized_3d (T, 26, 3), interp_mask (T, 26), num_repaired_entries)
    """
    T, K, _ = joints_3d.shape
    pinned = np.zeros((T, K), bool) if pin_mask is None else np.asarray(pin_mask, bool)
    if pinned.shape != (T, K):
        raise ValueError("Manual pin mask does not match the trajectory.")
    pinned = pinned & np.isfinite(joints_3d).all(axis=-1) & (joints_conf > 0)
    if T < 3:
        return joints_3d.astype(np.float32), np.zeros((T, K), dtype=np.bool_), 0

    # 0. Bone-Length Outlier & Max-Reach Gate: reject 2-camera occlusion depth explosions BEFORE PCHIP gap filling!
    #    - Short limb stretches (<= 20 frames) are NaN'd for clean PCHIP gap recovery.
    #    - Sustained narrow-baseline leg over-reach (> 20 frames, e.g. 1.24m on a 0.79m leg) is radially clamped
    #      back onto the hip-centered anatomical reach sphere so limbs NEVER explode in depth!
    pts_gated = np.asarray(joints_3d, dtype=np.float64).copy()
    conf_gated = np.asarray(joints_conf, dtype=np.float64).copy()
    gated_outliers = 0
    _LIMB_GATE_CHAINS = [
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),  # thighs & shins
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),  # upper arms & forearms
    ]
    med_seg_len: Dict[Tuple[int, int], float] = {}
    for a, b in _LIMB_GATE_CHAINS:
        ok_med = (
            (conf_gated[:, a] >= 0.35)
            & (conf_gated[:, b] >= 0.35)
            & np.all(np.isfinite(pts_gated[:, a]), axis=-1)
            & np.all(np.isfinite(pts_gated[:, b]), axis=-1)
        )
        ok_ab = (
            (conf_gated[:, a] >= 0.18)
            & (conf_gated[:, b] >= 0.18)
            & np.all(np.isfinite(pts_gated[:, a]), axis=-1)
            & np.all(np.isfinite(pts_gated[:, b]), axis=-1)
        )
        ref_mask = ok_med if int(np.sum(ok_med)) >= 8 else ok_ab
        if int(np.sum(ref_mask)) >= 8:
            d_all = np.linalg.norm(pts_gated[:, b] - pts_gated[:, a], axis=-1)
            med_l = float(np.median(d_all[ref_mask]))
            if 0.12 <= med_l <= 0.75:
                med_seg_len[(a, b)] = med_l
                bad = ok_ab & ((d_all > med_l * 1.32) | (d_all < med_l * 0.62))
                if np.any(bad):
                    # Split contiguous bad runs: NaN short runs (<= 20 frames) for PCHIP, clamp long runs radially
                    bad_idx = np.where(bad)[0]
                    splits = np.where(np.diff(bad_idx) > 1)[0] + 1
                    for run in np.split(bad_idx, splits):
                        if len(run) <= 20:
                            pts_gated[run, b] = np.nan
                            conf_gated[run, b] = 0.0
                        else:
                            # Radially project long stretched segment back to anatomical length along measured ray
                            v_ab = pts_gated[run, b] - pts_gated[run, a]
                            vn = np.linalg.norm(v_ab, axis=-1, keepdims=True)
                            scale_l = np.clip(vn, med_l * 0.85, med_l * 1.05)
                            pts_gated[run, b] = pts_gated[run, a] + v_ab * (
                                scale_l / np.maximum(vn, 1e-6)
                            )
                            conf_gated[run, b] = np.minimum(conf_gated[run, b], 0.25)
                    gated_outliers += int(np.sum(bad))

    # 0b. Two-segment Hip->Ankle / Shoulder->Wrist Max-Reach Clamping (prevents depth-elongated phantom lunges!)
    for hip_i, knee_i, ank_i in ((11, 13, 15), (12, 14, 16), (5, 7, 9), (6, 8, 10)):
        l1 = med_seg_len.get((hip_i, knee_i))
        l2 = med_seg_len.get((knee_i, ank_i))
        if l1 is None or l2 is None:
            continue
        max_reach = 0.99 * (l1 + l2)
        fin_hka = (
            np.all(np.isfinite(pts_gated[:, hip_i]), axis=-1)
            & np.all(np.isfinite(pts_gated[:, ank_i]), axis=-1)
            & (conf_gated[:, hip_i] >= 0.18)
            & (conf_gated[:, ank_i] >= 0.18)
        )
        if not np.any(fin_hka):
            continue
        vec_ha = pts_gated[:, ank_i] - pts_gated[:, hip_i]
        dist_ha = np.linalg.norm(vec_ha, axis=-1)
        over = fin_hka & (dist_ha > max_reach * 1.04)
        if np.any(over):
            ratio = (max_reach / np.maximum(dist_ha[over], 1e-6))[:, None]
            pts_gated[over, ank_i] = pts_gated[over, hip_i] + vec_ha[over] * ratio
            # Pull knee proportionally when knee is also finite and over-extended
            ok_k = over & np.all(np.isfinite(pts_gated[:, knee_i]), axis=-1)
            if np.any(ok_k):
                vec_hk = pts_gated[ok_k, knee_i] - pts_gated[ok_k, hip_i]
                dk = np.linalg.norm(vec_hk, axis=-1, keepdims=True)
                pts_gated[ok_k, knee_i] = pts_gated[ok_k, hip_i] + vec_hk * np.minimum(
                    1.0, (l1 * 1.02) / np.maximum(dk, 1e-6)
                )
            # Pull foot/toe landmarks on the same leg by the same ankle correction delta
            if ank_i in (15, 16):
                foot_children = (20, 22, 24) if ank_i == 15 else (21, 23, 25)
                delta_a = pts_gated[over, ank_i] - (pts_gated[over, hip_i] + vec_ha[over])
                for fc_i in foot_children:
                    fin_fc = np.all(np.isfinite(pts_gated[over, fc_i]), axis=-1)
                    if np.any(fin_fc):
                        idx_over = np.where(over)[0][fin_fc]
                        pts_gated[idx_over, fc_i] += delta_a[fin_fc]
            gated_outliers += int(np.sum(over))

    # 1. Remove hard velocity spikes & fill gaps (up to 28 frames) with shape-preserving PCHIP splines
    pts_clean, interp_mask = despike_and_fill_3d_points(
        pts_gated.astype(np.float32),
        conf_gated.astype(np.float32),
        max_gap_frames=28,
        mad_sigma=3.8,
        min_conf=0.18,
    )
    repaired = int(np.sum(interp_mask)) + gated_outliers

    pts = pts_clean.astype(np.float64).copy()
    pts[pinned] = joints_3d[pinned]
    conf = conf_gated.copy()
    conf = np.where(interp_mask, np.maximum(conf, 0.32), conf)

    # Fill any remaining edge NaNs by holding nearest finite value so filters never propagate NaN
    for k in range(K):
        fin = np.all(np.isfinite(pts[:, k]), axis=-1)
        if np.any(fin) and not np.all(fin):
            idx = np.where(fin)[0]
            for ax in range(3):
                pts[:, k, ax] = np.interp(np.arange(T), idx, pts[idx, k, ax])
        elif not np.any(fin):
            pts[:, k] = 0.0
            conf[:, k] = 0.0

    # 2. Measure canonical rigid segment lengths across high-confidence frames
    target_lengths: Dict[Tuple[int, int], float] = {}
    for a, b in _RIGID_SEGMENTS_BODY26:
        good = (conf[:, a] >= 0.45) & (conf[:, b] >= 0.45)
        if int(np.sum(good)) >= 3:
            dists = np.linalg.norm(pts[good, b] - pts[good, a], axis=-1)
            med_len = float(np.median(dists))
            if 0.03 <= med_len <= 1.20:
                target_lengths[(a, b)] = med_len

    # 3. Alternating Temporal Smoothing + Rigid Bone-Length Projection (fast & rock-solid!)
    pts_meas = pts.copy()
    nyq = 0.5 * max(fps, 5.0)
    use_butter = T >= 18 and 0.8 < cutoff_hz < nyq * 0.85
    sos = butter(3, cutoff_hz / nyq, btype="low", output="sos") if use_butter else None
    sg_win = min(T if T % 2 == 1 else T - 1, 9)

    for _ in range(max(1, iterations)):
        # A. Confidence-aware temporal smoothing (high-conf measured frames stay sharp; low-conf frames smooth strongly)
        if sos is not None:
            smoothed = sosfiltfilt(sos, pts, axis=0)
        elif sg_win >= 5:
            smoothed = savgol_filter(pts, window_length=sg_win, polyorder=2, axis=0)
        else:
            smoothed = pts.copy()

        # Blend toward smoothed trajectory inversely proportional to measurement confidence
        alpha = np.clip(0.85 - 0.55 * conf, 0.25, 0.88)[:, :, None]
        pts = (1.0 - alpha) * pts_meas + alpha * smoothed

        # B. Exact bone-length projection along every rigid segment weighted by inverse joint confidence
        #    Give proximal torso/hip anchors 3.5x higher stiffness so distal limb noise never drags the pelvis!
        _TORSO_ANCHORS = {19, 18, 11, 12, 5, 6}
        for (a, b), tgt_len in target_lengths.items():
            vec = pts[:, b] - pts[:, a]
            cur_len = np.linalg.norm(vec, axis=-1)
            ok = cur_len > 1e-5
            if not np.any(ok):
                continue
            err = (cur_len[ok] - tgt_len) / cur_len[ok]
            # Move lower-confidence / distal endpoint more than higher-confidence proximal anchor
            wa = (0.28 if a in _TORSO_ANCHORS else 1.0) / np.maximum(conf[ok, a], 0.15)
            wb = (0.28 if b in _TORSO_ANCHORS else 1.0) / np.maximum(conf[ok, b], 0.15)
            w_sum = wa + wb
            corr = vec[ok] * (err[:, None] * 0.55)
            pts[ok, a] += corr * (wa / w_sum)[:, None]
            pts[ok, b] -= corr * (wb / w_sum)[:, None]

        # User corrections remain targets; temporal filtering must not erase them.
        pts[pinned] = joints_3d[pinned]

    # Final rigid bone-length projection pass (2x) so output trajectories strictly honor anatomical segment lengths
    _TORSO_ANCHORS = {19, 18, 11, 12, 5, 6}
    for _ in range(2):
        for (a, b), tgt_len in target_lengths.items():
            vec = pts[:, b] - pts[:, a]
            cur_len = np.linalg.norm(vec, axis=-1)
            ok = cur_len > 1e-5
            if not np.any(ok):
                continue
            err = (cur_len[ok] - tgt_len) / cur_len[ok]
            wa = (0.28 if a in _TORSO_ANCHORS else 1.0) / np.maximum(conf[ok, a], 0.15)
            wb = (0.28 if b in _TORSO_ANCHORS else 1.0) / np.maximum(conf[ok, b], 0.15)
            w_sum = wa + wb
            corr = vec[ok] * (err[:, None] * 0.65)
            pts[ok, a] += corr * (wa / w_sum)[:, None]
            pts[ok, b] -= corr * (wb / w_sum)[:, None]

    pts[pinned] = joints_3d[pinned]
    interp_mask[pinned] = False
    return pts.astype(np.float32), interp_mask, repaired


def _solve_3d_head_world_frame(body26_t: np.ndarray, conf_t: np.ndarray) -> Optional[np.ndarray]:
    """Compute a clean right-handed world rotation matrix [X=Right, Y=Forward, Z=Up] from 3D head landmarks.

    Uses the long ~25 cm `neck -> head_top` vector as the primary Z-up axis (immune to 180-deg nose-pitch flips!)
    and `left_ear -> right_ear` (~16 cm) as the lateral X-right axis, with `nose - ear_mid` only guarding L/R ear swaps.
    """
    # Body26: 0=nose, 1=L_eye, 2=R_eye, 3=L_ear, 4=R_ear, 17=head_top, 18=neck
    has_ears = conf_t[3] >= 0.30 and conf_t[4] >= 0.30
    has_eyes = conf_t[1] >= 0.30 and conf_t[2] >= 0.30
    has_nose = conf_t[0] >= 0.30
    has_up = conf_t[17] >= 0.25 and conf_t[18] >= 0.25

    if has_ears:
        x_raw = body26_t[4] - body26_t[3]  # Left ear -> Right ear (+X right)
        ear_mid = 0.5 * (body26_t[3] + body26_t[4])
    elif has_eyes:
        x_raw = body26_t[2] - body26_t[1]
        ear_mid = 0.5 * (body26_t[1] + body26_t[2])
    else:
        return None

    nx = float(np.linalg.norm(x_raw))
    if nx < 0.03:
        return None
    x_cand = x_raw / nx

    if has_up:
        z_raw = body26_t[17] - body26_t[18]  # Neck -> Head Top (+Z up, ~25cm baseline!)
        nz = float(np.linalg.norm(z_raw))
        if nz < 0.08:
            return None
        z_axis = z_raw / nz
        # Orthogonalize lateral X against Z-up so pitch & yaw never flip Z upside down!
        x_orth = x_cand - float(np.dot(x_cand, z_axis)) * z_axis
        nxo = float(np.linalg.norm(x_orth))
        if nxo < 0.15:
            return None
        x_axis = x_orth / nxo
        y_axis = np.cross(z_axis, x_axis)
        ny = float(np.linalg.norm(y_axis))
        if ny < 1e-5:
            return None
        y_axis /= ny
        # Guard against true 180-deg L/R ear swap ONLY when nose clearly points behind (-Y)
        if has_nose:
            nf = body26_t[0] - ear_mid
            nf_horiz = nf - float(np.dot(nf, z_axis)) * z_axis
            nf_n = float(np.linalg.norm(nf_horiz))
            if nf_n >= 0.03 and float(np.dot(y_axis, nf_horiz / nf_n)) < -0.55:
                x_axis = -x_axis
                y_axis = -y_axis
        return np.column_stack([x_axis, y_axis, z_axis])

    # Fallback when head_top/neck unavailable: build from ears + nose
    if not has_nose:
        return None
    x_axis = x_cand
    y_raw = body26_t[0] - ear_mid
    y_orth = y_raw - float(np.dot(y_raw, x_axis)) * x_axis
    ny = float(np.linalg.norm(y_orth))
    if ny < 0.02:
        return None
    y_axis = y_orth / ny
    z_axis = np.cross(x_axis, y_axis)
    nz = float(np.linalg.norm(z_axis))
    if nz < 1e-5:
        return None
    z_axis /= nz
    return np.column_stack([x_axis, y_axis, z_axis])


def fuse_6dof_head_orientation(
    track: SkeletalMotionTrack,
    actor: ActorProfile,
    body26_3d: np.ndarray,
    body26_conf: np.ndarray,
    face_track: Optional[FaceMotionTrack] = None,
    fps: float = 30.0,
) -> Tuple[SkeletalMotionTrack, float]:
    """Fuse 3D triangulated head landmarks + optional high-res face PnP head pose cleanly into Neck (35%) + Head (65%).

    Returns:
        (updated_track, head_tracking_coverage_pct)
    """
    T = track.num_frames
    local_q = track.local_rotations_xyzw.astype(np.float64).copy()
    root_t = track.root_translation.astype(np.float64).copy()

    chest_idx = BONE_INDEX["chest"]
    neck_idx = BONE_INDEX["neck"]
    head_idx = BONE_INDEX["head"]

    # Recompute FK global matrices to get exact world chest orientation per frame
    _, global_mats = compute_forward_kinematics_sequence(root_t, local_q, actor.rest_offsets_m)

    head_rel_quats = np.tile(quat_identity(), (T, 1))
    valid_head = np.zeros(T, dtype=np.bool_)

    # Optional calibrated baseline from FaceMotionTrack head_rotation_xyzw (delta relative to neutral median)
    face_delta_quats: Optional[np.ndarray] = None
    if face_track is not None and len(face_track.head_rotation_xyzw) >= T:
        f_conf = face_track.confidence[:T]
        f_q = enforce_quaternion_continuity(face_track.head_rotation_xyzw[:T].astype(np.float64))
        ok_f = f_conf >= 0.40
        if int(np.sum(ok_f)) >= 3:
            q0 = quat_normalize(np.mean(f_q[ok_f], axis=0))
            q0_inv = quat_inverse(q0)
            face_delta_quats = np.array(
                [quat_multiply(q0_inv, f_q[t]) for t in range(T)], dtype=np.float64
            )

    for t in range(T):
        R_chest = global_mats[t, chest_idx, :3, :3]
        R_head_3d = _solve_3d_head_world_frame(
            body26_3d[t].astype(np.float64), body26_conf[t].astype(np.float64)
        )
        if R_head_3d is not None:
            R_rel = R_chest.T @ R_head_3d
            q_rel = matrix_to_quat(R_rel)
            # Blend subtle high-precision pitch/yaw/roll from head-cropped face tracker when confident
            if (
                face_delta_quats is not None
                and face_track is not None
                and float(face_track.confidence[t]) >= 0.55
            ):
                q_rel = quat_slerp(q_rel, face_delta_quats[t], 0.30)
            head_rel_quats[t] = q_rel
            valid_head[t] = True
        elif (
            face_delta_quats is not None
            and face_track is not None
            and float(face_track.confidence[t]) >= 0.45
        ):
            head_rel_quats[t] = face_delta_quats[t]
            valid_head[t] = True

    coverage = float(np.mean(valid_head)) * 100.0 if T > 0 else 0.0
    if int(np.sum(valid_head)) < 2:
        return track, coverage

    # Fill missing head frames via SLERP + smooth with zero-phase Butterworth
    v_idx = np.where(valid_head)[0]
    if v_idx[0] > 0:
        head_rel_quats[: v_idx[0]] = head_rel_quats[v_idx[0]]
    if v_idx[-1] < T - 1:
        head_rel_quats[v_idx[-1] + 1 :] = head_rel_quats[v_idx[-1]]
    for g in range(len(v_idx) - 1):
        a, b = int(v_idx[g]), int(v_idx[g + 1])
        if b - a > 1:
            for k in range(1, b - a):
                head_rel_quats[a + k] = quat_slerp(
                    head_rel_quats[a], head_rel_quats[b], k / float(b - a)
                )

    head_rel_quats = enforce_quaternion_continuity(head_rel_quats)
    nyq = 0.5 * max(fps, 5.0)
    if T >= 16 and 6.5 < nyq * 0.85:
        sos = butter(3, 6.5 / nyq, btype="low", output="sos")
        filt = sosfiltfilt(sos, head_rel_quats, axis=0)
        head_rel_quats = np.array([quat_normalize(filt[t]) for t in range(T)], dtype=np.float64)

    # Split total chest->head rotation anatomically: 35% Neck + 65% Head (clamped smoothly to human range!)
    id_q = quat_identity()
    prev_clamped = id_q.copy()
    for t in range(T):
        rot = Rotation.from_quat(head_rel_quats[t])
        rv = rot.as_rotvec()
        ang = float(np.linalg.norm(rv))
        # Guard against +/-180 deg yaw wrap (e.g. actor walking off-camera) which otherwise snaps +50 <-> -50 deg!
        if ang > np.radians(115.0):
            q_clamped = quat_slerp(prev_clamped, id_q, 0.12)
        elif ang > np.radians(78.0):
            q_clamped = Rotation.from_rotvec(rv * (np.radians(78.0) / ang)).as_quat()
        else:
            q_clamped = head_rel_quats[t]
        # Rate-limit sudden head snaps to <= 22 deg/frame (~660 deg/s real head turn limit)
        if t > 0:
            q_step = quat_multiply(quat_inverse(prev_clamped), q_clamped)
            step_ang = float(2.0 * np.arccos(np.clip(abs(q_step[3]), 0.0, 1.0)))
            max_head_step = float(np.radians(22.0))
            if step_ang > max_head_step:
                q_clamped = quat_slerp(prev_clamped, q_clamped, max_head_step / step_ang)
        prev_clamped = q_clamped
        q_neck = clamp_bone_rotation("neck", quat_slerp(id_q, q_clamped, 0.35), strength=0.90)
        q_head = clamp_bone_rotation("head", quat_slerp(id_q, q_clamped, 0.65), strength=0.90)
        local_q[t, neck_idx] = q_neck
        local_q[t, head_idx] = q_head

    local_q = enforce_quaternion_continuity(local_q)
    fk_pos, fk_mats = compute_forward_kinematics_sequence(root_t, local_q, actor.rest_offsets_m)
    updated = SkeletalMotionTrack(
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
        cleanup_config=dict(track.cleanup_config),
    )
    return updated, coverage


def solve_cinema_skeleton_from_3d(
    joints_3d: np.ndarray,
    joints_conf: np.ndarray,
    fps: float = 30.0,
    face_track: Optional[FaceMotionTrack] = None,
    enable_nonlinear_refinement: bool = False,
    progress_cb=None,
    cancel_event=None,
    pin_mask: Optional[np.ndarray] = None,
) -> Tuple[SkeletalMotionTrack, ActorProfile, np.ndarray, int, float]:
    """Full Spatio-Temporal 3D Optimization + Auto Actor Calibration + 25-Bone IK + 6-DoF Head Fusion."""
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Cancelled by user.")
    if progress_cb:
        progress_cb(
            0.10, "Running Spatio-Temporal bone-length despike & trajectory optimization..."
        )
    opt_3d, interp_mask, repaired_count = optimize_3d_trajectories_spatiotemporal(
        joints_3d, joints_conf, fps=fps, iterations=8, cutoff_hz=8.0, pin_mask=pin_mask
    )
    conf_clean = np.where(interp_mask, np.maximum(joints_conf, 0.35), joints_conf).astype(
        np.float32
    )

    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Cancelled by user.")
    if progress_cb:
        progress_cb(0.35, "Auto-calibrating actor bone lengths & solving 25-bone articulated IK...")
    actor = calibrate_actor_from_body26_frames(
        actor_id="cinema_auto_actor",
        name="Auto-Calibrated Cinema Actor",
        body26_3d_frames=opt_3d,
        body26_conf=conf_clean,
    )

    solver = ArticulatedSkeletonSolver(
        actor=actor,
        config=SolverConfig(
            enforce_anatomical_limits=True,
            limit_strength=0.90,
            enable_nonlinear_refinement=enable_nonlinear_refinement,
        ),
    )
    raw_track = solver.solve_sequence(opt_3d, conf_clean)
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Cancelled by user.")
    if progress_cb:
        progress_cb(0.80, "Fusing 6-DoF Head & Neck orientation...")
    fused_track, head_cov = fuse_6dof_head_orientation(
        raw_track, actor, opt_3d, conf_clean, face_track=face_track, fps=fps
    )
    if progress_cb:
        progress_cb(1.0, "Skeleton & 6-DoF Head solve complete.")
    return fused_track, actor, opt_3d, repaired_count, head_cov
