"""Articulated Skeletal Motion Solver with constant bone lengths, constrained IK, rigid cluster fusion, and observability flags.

This solver NEVER simply points bones at raw landmarks. It computes a full articulated hierarchy:
1. Solves Pelvis 6-DoF from hip geometry + optional pelvis rigid marker cluster.
2. Solves Spine / Spine1 / Chest / Neck / Head orthonormal frames + optional chest/head rigid clusters.
3. Solves Clavicle elevation/protraction.
4. Solves 2-bone analytical + pole-vector Limb IK for Left/Right Arms & Legs (upper + lower segments)
   with explicit twist observability tracking and anatomical joint rotation limits.
5. Solves Foot/Ankle and Wrist/Hand orientation from heel/toe/wrist/cluster frames.
6. Optionally runs Cinema-grade per-frame constrained Levenberg-Marquardt refinement incorporating
   both AI joint targets and calibrated physical marker surface offsets (without naive averaging!).
7. Computes final joint positions strictly via Forward Kinematics to guarantee invariant bone lengths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from engine.data_format.motion_io import SkeletalMotionTrack
from engine.marker_tracking.detector import RigidClusterPose3D
from engine.math_core.transforms import (
    enforce_quaternion_continuity,
    matrix_to_quat,
    quat_from_two_vectors,
    quat_identity,
    quat_inverse,
    quat_multiply,
    quat_slerp,
    quat_to_matrix,
)
from engine.skeleton.rig_definition import (
    BONE_INDEX,
    BONE_NAMES,
    PARENT_INDICES,
    ActorProfile,
    clamp_bone_rotation,
    compute_forward_kinematics_frame,
)


def _safe_unit(v: np.ndarray, fallback: np.ndarray) -> Tuple[np.ndarray, bool]:
    n = float(np.linalg.norm(v))
    if n < 1e-6 or not np.all(np.isfinite(v)):
        return fallback.copy(), False
    return v / n, True


def _build_orthonormal_frame(primary_axis: np.ndarray, secondary_hint: np.ndarray, axis_order: str = "zx") -> Tuple[np.ndarray, bool]:
    """Build a right-handed 3x3 rotation matrix from two non-collinear direction hints."""
    if axis_order == "zx":
        z, ok1 = _safe_unit(primary_axis, np.array([0.0, 0.0, 1.0]))
        x_raw = secondary_hint - np.dot(secondary_hint, z) * z
        x, ok2 = _safe_unit(x_raw, np.array([1.0, 0.0, 0.0]))
        y = np.cross(z, x)
        y, ok3 = _safe_unit(y, np.array([0.0, 1.0, 0.0]))
        x = np.cross(y, z)
        return np.column_stack([x, y, z]), (ok1 and ok2 and ok3)
    if axis_order == "xz":
        x, ok1 = _safe_unit(primary_axis, np.array([1.0, 0.0, 0.0]))
        z_raw = secondary_hint - np.dot(secondary_hint, x) * x
        z, ok2 = _safe_unit(z_raw, np.array([0.0, 0.0, 1.0]))
        y = np.cross(z, x)
        y, ok3 = _safe_unit(y, np.array([0.0, 1.0, 0.0]))
        z = np.cross(x, y)
        return np.column_stack([x, y, z]), (ok1 and ok2 and ok3)
    if axis_order == "yz":
        y, ok1 = _safe_unit(primary_axis, np.array([0.0, 1.0, 0.0]))
        z_raw = secondary_hint - np.dot(secondary_hint, y) * y
        z, ok2 = _safe_unit(z_raw, np.array([0.0, 0.0, 1.0]))
        x = np.cross(y, z)
        x, ok3 = _safe_unit(x, np.array([1.0, 0.0, 0.0]))
        z = np.cross(x, y)
        return np.column_stack([x, y, z]), (ok1 and ok2 and ok3)
    raise ValueError(f"Unsupported axis_order: {axis_order}")


def solve_two_bone_chain_rotations(
    root_pos: np.ndarray,
    mid_target: np.ndarray,
    end_target: np.ndarray,
    parent_world_R: np.ndarray,
    upper_rest_vec: np.ndarray,
    lower_rest_vec: np.ndarray,
    bend_hint_world: np.ndarray,
    prev_upper_R: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Solve 3D global orientations for a 2-bone articulated limb (upper + lower) with pole-plane twist.

    Returns:
        (R_upper_world, R_lower_world, twist_observable)
    """
    u_dir_world, ok_u = _safe_unit(mid_target - root_pos, parent_world_R @ (upper_rest_vec / max(np.linalg.norm(upper_rest_vec), 1e-6)))
    l_dir_world, ok_l = _safe_unit(end_target - mid_target, parent_world_R @ (lower_rest_vec / max(np.linalg.norm(lower_rest_vec), 1e-6)))

    u_rest = upper_rest_vec / max(float(np.linalg.norm(upper_rest_vec)), 1e-6)
    l_rest = lower_rest_vec / max(float(np.linalg.norm(lower_rest_vec)), 1e-6)

    # Check bend angle to know whether the limb plane uniquely observes axial twist
    bend_cos = float(np.clip(np.dot(u_dir_world, l_dir_world), -1.0, 1.0))
    bend_angle_deg = float(np.degrees(np.arccos(bend_cos)))
    twist_observable = ok_u and ok_l and (bend_angle_deg >= 10.0)

    if twist_observable:
        # Pole vector lies in the limb bending plane
        plane_normal = np.cross(u_dir_world, l_dir_world)
        pn = float(np.linalg.norm(plane_normal))
        plane_normal = plane_normal / max(pn, 1e-6)
        pole_dir = np.cross(plane_normal, u_dir_world)
        # Ensure pole_dir points consistently with anatomical bend_hint_world
        if np.dot(pole_dir, bend_hint_world) < 0:
            pole_dir = -pole_dir
    else:
        # Straight limb: axial twist is ambiguous from 3 points alone -> stabilize using hint/previous frame
        if prev_upper_R is not None:
            pole_dir = prev_upper_R[:, 1]
        else:
            pole_dir = bend_hint_world

    # Map rest frame (u_rest aligned to u_dir_world, rest secondary axis aligned to pole_dir)
    # Determine rest primary & secondary axes cleanly
    rest_sec = np.array([0.0, 1.0, 0.0]) if abs(u_rest[1]) < 0.8 else np.array([0.0, 0.0, 1.0])
    rest_sec = rest_sec - np.dot(rest_sec, u_rest) * u_rest
    rest_sec /= max(float(np.linalg.norm(rest_sec)), 1e-6)
    rest_third = np.cross(u_rest, rest_sec)
    R_rest_basis = np.column_stack([u_rest, rest_sec, rest_third])

    p_orth = pole_dir - np.dot(pole_dir, u_dir_world) * u_dir_world
    p_orth, _ = _safe_unit(p_orth, np.cross(u_dir_world, np.array([0.0, 0.0, 1.0]) if abs(u_dir_world[2]) < 0.9 else np.array([1.0, 0.0, 0.0])))
    w_third = np.cross(u_dir_world, p_orth)
    R_world_basis = np.column_stack([u_dir_world, p_orth, w_third])

    R_upper = R_world_basis @ R_rest_basis.T

    # Lower bone: shortest-arc bend from upper bone's aligned lower rest axis onto actual l_dir_world
    l_rest_in_upper = R_upper @ l_rest
    q_bend = quat_from_two_vectors(l_rest_in_upper, l_dir_world)
    R_lower = quat_to_matrix(q_bend) @ R_upper

    return R_upper, R_lower, twist_observable


@dataclass
class SolverConfig:
    enforce_anatomical_limits: bool = True
    limit_strength: float = 0.85
    enable_nonlinear_refinement: bool = False  # Enabled automatically in Cinema / Hybrid mode
    cluster_orientation_weight: float = 0.65
    temporal_smooth_weight: float = 0.15


class ArticulatedSkeletonSolver:
    """Replaceable articulated full-body skeletal solver."""

    def __init__(self, actor: ActorProfile, config: Optional[SolverConfig] = None) -> None:
        self.actor = actor
        self.config = config or SolverConfig()

    def solve_sequence(
        self,
        body26_3d: np.ndarray,  # (T, 26, 3) meters
        body26_conf: np.ndarray,  # (T, 26) [0..1]
        rigid_clusters_per_frame: Optional[List[Dict[str, RigidClusterPose3D]]] = None,
        surface_markers_per_frame: Optional[List[Dict[str, Tuple[np.ndarray, float]]]] = None,
    ) -> SkeletalMotionTrack:
        """Solve full hierarchical skeletal animation over T frames."""
        T = body26_3d.shape[0]
        B = len(BONE_NAMES)
        rest = self.actor.rest_offsets_m

        root_trans = np.zeros((T, 3), dtype=np.float64)
        root_rot = np.tile(quat_identity(), (T, 1))
        local_quats = np.tile(quat_identity(), (T, B, 1))
        global_mats = np.zeros((T, B, 4, 4), dtype=np.float64)
        joint_pos = np.zeros((T, B, 3), dtype=np.float64)
        joint_conf = np.zeros((T, B), dtype=np.float64)
        rot_obs = np.zeros((T, B), dtype=np.bool_)

        prev_global_R: Optional[np.ndarray] = None

        for t in range(T):
            pts = np.asarray(body26_3d[t], dtype=np.float64)
            conf = np.asarray(body26_conf[t], dtype=np.float64)
            clusters = rigid_clusters_per_frame[t] if (rigid_clusters_per_frame and t < len(rigid_clusters_per_frame)) else {}
            markers = surface_markers_per_frame[t] if (surface_markers_per_frame and t < len(surface_markers_per_frame)) else {}

            r_t, l_q, g_R, j_c, r_o = self._solve_single_frame(pts, conf, clusters, markers, prev_global_R)

            if self.config.enforce_anatomical_limits:
                for b_idx, b_name in enumerate(BONE_NAMES):
                    if b_idx > 0:
                        l_q[b_idx] = clamp_bone_rotation(b_name, l_q[b_idx], strength=self.config.limit_strength)

            # Recompute exact FK so bone lengths are strictly constant
            fk_pos, fk_mats = compute_forward_kinematics_frame(r_t, l_q, rest)
            prev_global_R = fk_mats[:, :3, :3].copy()

            root_trans[t] = r_t
            root_rot[t] = l_q[0]
            local_quats[t] = l_q
            global_mats[t] = fk_mats
            joint_pos[t] = fk_pos
            joint_conf[t] = j_c
            rot_obs[t] = r_o

        # Enforce quaternion hemisphere continuity over time
        local_quats = enforce_quaternion_continuity(local_quats)
        root_rot = local_quats[:, 0, :].copy()

        return SkeletalMotionTrack(
            root_translation=root_trans.astype(np.float32),
            root_rotation_xyzw=root_rot.astype(np.float32),
            local_rotations_xyzw=local_quats.astype(np.float32),
            global_matrices=global_mats.astype(np.float32),
            joint_positions=joint_pos.astype(np.float32),
            joint_confidence=joint_conf.astype(np.float32),
            rotation_observable=rot_obs,
        )

    def _valid_pt(self, pts: np.ndarray, conf: np.ndarray, idx: int, min_c: float = 0.15) -> Optional[np.ndarray]:
        if conf[idx] >= min_c and np.all(np.isfinite(pts[idx])):
            return pts[idx]
        return None

    def _solve_single_frame(
        self,
        pts: np.ndarray,
        conf: np.ndarray,
        clusters: Dict[str, RigidClusterPose3D],
        surface_markers: Dict[str, Tuple[np.ndarray, float]],
        prev_global_R: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        B = len(BONE_NAMES)
        rest = self.actor.rest_offsets_m
        global_R = np.tile(np.eye(3, dtype=np.float64), (B, 1, 1))
        local_q = np.tile(quat_identity(), (B, 1))
        j_conf = np.zeros(B, dtype=np.float64)
        rot_obs = np.zeros(B, dtype=np.bool_)

        # 1. Pelvis / Hips root translation & orientation
        l_hip = self._valid_pt(pts, conf, 11)
        r_hip = self._valid_pt(pts, conf, 12)
        hips_pt = self._valid_pt(pts, conf, 19)
        neck_pt = self._valid_pt(pts, conf, 18)
        l_sh = self._valid_pt(pts, conf, 5)
        r_sh = self._valid_pt(pts, conf, 6)

        if hips_pt is None and l_hip is not None and r_hip is not None:
            hips_pt = 0.5 * (l_hip + r_hip)
        if hips_pt is None:
            hips_pt = rest[0].copy()

        root_t = hips_pt.copy()

        # Pelvis orientation: +X from left_hip -> right_hip, +Z from hips -> neck
        if l_hip is not None and r_hip is not None:
            pelvis_x = r_hip - l_hip
            pelvis_z = (neck_pt - hips_pt) if neck_pt is not None else np.array([0.0, 0.0, 1.0])
            R_pelvis, ok_p = _build_orthonormal_frame(pelvis_z, pelvis_x, axis_order="zx")
            j_conf[0] = float(0.5 * (conf[11] + conf[12]))
            rot_obs[0] = ok_p
        else:
            R_pelvis = prev_global_R[0].copy() if prev_global_R is not None else np.eye(3)
            j_conf[0] = 0.1

        # Fuse rigid pelvis cluster orientation if visible
        if "cl_pelvis" in clusters and clusters["cl_pelvis"].valid:
            cp = clusters["cl_pelvis"]
            w_c = float(np.clip(self.config.cluster_orientation_weight * cp.confidence, 0.0, 0.9))
            q_fused = quat_slerp(matrix_to_quat(R_pelvis), cp.rotation_quat_xyzw, w_c)
            R_pelvis = quat_to_matrix(q_fused)
            rot_obs[0] = True
            j_conf[0] = max(j_conf[0], cp.confidence)

        global_R[0] = R_pelvis
        local_q[0] = matrix_to_quat(R_pelvis)

        # 2. Spine -> Spine1 -> Chest orientation (distribute twist evenly across 3 spine segments!)
        if l_sh is not None and r_sh is not None and neck_pt is not None:
            chest_x = r_sh - l_sh
            chest_z = neck_pt - hips_pt
            R_chest_target, ok_ch = _build_orthonormal_frame(chest_z, chest_x, axis_order="zx")
            chest_c = float(min(conf[5], conf[6], conf[18]))
        else:
            R_chest_target = R_pelvis.copy()
            ok_ch = False
            chest_c = 0.2

        if "cl_chest" in clusters and clusters["cl_chest"].valid:
            cc = clusters["cl_chest"]
            w_c = float(np.clip(self.config.cluster_orientation_weight * cc.confidence, 0.0, 0.9))
            R_chest_target = quat_to_matrix(quat_slerp(matrix_to_quat(R_chest_target), cc.rotation_quat_xyzw, w_c))
            ok_ch = True
            chest_c = max(chest_c, cc.confidence)

        q_pelvis_to_chest = matrix_to_quat(R_pelvis.T @ R_chest_target)
        # Distribute spine rotation smoothly: 30% spine, 35% spine1, 35% chest
        q_s0 = quat_slerp(quat_identity(), q_pelvis_to_chest, 0.30)
        q_s1 = quat_slerp(quat_identity(), q_pelvis_to_chest, 0.35)
        q_s2 = quat_multiply(quat_inverse(quat_multiply(q_s0, q_s1)), q_pelvis_to_chest)

        for idx_s, q_s in [(1, q_s0), (2, q_s1), (3, q_s2)]:
            local_q[idx_s] = q_s
            global_R[idx_s] = global_R[PARENT_INDICES[idx_s]] @ quat_to_matrix(q_s)
            j_conf[idx_s] = chest_c
            rot_obs[idx_s] = ok_ch

        # 3. Neck & Head orientation
        nose = self._valid_pt(pts, conf, 0)
        l_ear = self._valid_pt(pts, conf, 3)
        r_ear = self._valid_pt(pts, conf, 4)
        head_top = self._valid_pt(pts, conf, 17)

        if l_ear is not None and r_ear is not None and neck_pt is not None:
            head_x = r_ear - l_ear
            head_z = (head_top - neck_pt) if head_top is not None else global_R[3][:, 2]
            R_head_target, ok_hd = _build_orthonormal_frame(head_z, head_x, axis_order="zx")
            if nose is not None:
                # Verify +Y faces toward nose
                ear_mid = 0.5 * (l_ear + r_ear)
                if np.dot(R_head_target[:, 1], nose - ear_mid) < 0:
                    R_head_target[:, 0] *= -1
                    R_head_target[:, 1] *= -1
            head_c = float(min(conf[3], conf[4]))
        else:
            R_head_target = global_R[3].copy()
            ok_hd = False
            head_c = 0.2

        if "cl_head" in clusters and clusters["cl_head"].valid:
            ch = clusters["cl_head"]
            w_c = float(np.clip(self.config.cluster_orientation_weight * ch.confidence, 0.0, 0.9))
            R_head_target = quat_to_matrix(quat_slerp(matrix_to_quat(R_head_target), ch.rotation_quat_xyzw, w_c))
            ok_hd = True
            head_c = max(head_c, ch.confidence)

        q_chest_to_head = matrix_to_quat(global_R[3].T @ R_head_target)
        q_neck = quat_slerp(quat_identity(), q_chest_to_head, 0.45)
        q_head = quat_multiply(quat_inverse(q_neck), q_chest_to_head)
        local_q[4], local_q[5] = q_neck, q_head
        global_R[4] = global_R[3] @ quat_to_matrix(q_neck)
        global_R[5] = global_R[4] @ quat_to_matrix(q_head)
        j_conf[4] = j_conf[5] = j_conf[22] = head_c
        rot_obs[4] = rot_obs[5] = ok_hd

        # 4. Left & Right Arms (Clavicle -> Upper Arm -> Forearm -> Hand)
        chest_pos_est = hips_pt + global_R[0] @ rest[1] + global_R[1] @ rest[2] + global_R[2] @ rest[3]
        for side, sh_k, el_k, wr_k, cl_b, ua_b, fa_b, ha_b, cl_id in [
            ("left", 5, 7, 9, 6, 7, 8, 9, "cl_l_forearm"),
            ("right", 6, 8, 10, 10, 11, 12, 13, "cl_r_forearm"),
        ]:
            sh_p = self._valid_pt(pts, conf, sh_k)
            el_p = self._valid_pt(pts, conf, el_k)
            wr_p = self._valid_pt(pts, conf, wr_k)

            # Clavicle modest elevation/protraction toward observed shoulder
            if sh_p is not None:
                clav_vec_local = global_R[3].T @ (sh_p - chest_pos_est)
                rest_sh_total = rest[cl_b] + rest[ua_b]
                q_clav = quat_slerp(quat_identity(), quat_from_two_vectors(rest_sh_total, clav_vec_local), 0.35)
                local_q[cl_b] = q_clav
                global_R[cl_b] = global_R[3] @ quat_to_matrix(q_clav)
                j_conf[cl_b] = float(conf[sh_k])
                rot_obs[cl_b] = True
            else:
                global_R[cl_b] = global_R[3].copy()

            if sh_p is not None and el_p is not None and wr_p is not None:
                # Elbows flex forward (+Y in chest frame), keeping pole_dir aligned with rest +Y
                bend_hint = global_R[3][:, 1]
                prev_u = prev_global_R[ua_b] if prev_global_R is not None else None
                R_ua, R_fa, obs_arm = solve_two_bone_chain_rotations(
                    sh_p, el_p, wr_p, global_R[cl_b], rest[fa_b], rest[ha_b], bend_hint, prev_upper_R=prev_u
                )
                if cl_id in clusters and clusters[cl_id].valid:
                    cf = clusters[cl_id]
                    R_fa = quat_to_matrix(quat_slerp(matrix_to_quat(R_fa), cf.rotation_quat_xyzw, self.config.cluster_orientation_weight * cf.confidence))
                    obs_arm = True
                global_R[ua_b], global_R[fa_b] = R_ua, R_fa
                local_q[ua_b] = matrix_to_quat(global_R[cl_b].T @ R_ua)
                local_q[fa_b] = matrix_to_quat(R_ua.T @ R_fa)
                global_R[ha_b] = R_fa.copy()
                local_q[ha_b] = quat_identity()
                j_conf[ua_b] = float(min(conf[sh_k], conf[el_k]))
                j_conf[fa_b] = float(min(conf[el_k], conf[wr_k]))
                j_conf[ha_b] = float(conf[wr_k])
                rot_obs[ua_b] = obs_arm
                rot_obs[fa_b] = obs_arm
            else:
                global_R[ua_b] = global_R[cl_b].copy()
                global_R[fa_b] = global_R[ua_b].copy()
                global_R[ha_b] = global_R[fa_b].copy()

        # 5. Left & Right Legs (Upper Leg -> Lower Leg -> Foot -> Toe)
        for side, hip_k, kn_k, an_k, toe_k, heel_k, ul_b, ll_b, ft_b, to_b, cl_id in [
            ("left", 11, 13, 15, 20, 24, 14, 15, 16, 17, "cl_l_foot"),
            ("right", 12, 14, 16, 21, 25, 18, 19, 20, 21, "cl_r_foot"),
        ]:
            hp = self._valid_pt(pts, conf, hip_k)
            kp = self._valid_pt(pts, conf, kn_k)
            ap = self._valid_pt(pts, conf, an_k)
            tp = self._valid_pt(pts, conf, toe_k)
            hl = self._valid_pt(pts, conf, heel_k)

            if hp is not None and kp is not None and ap is not None:
                # Knees point forward (+Y of pelvis)
                knee_hint = global_R[0][:, 1]
                prev_u = prev_global_R[ul_b] if prev_global_R is not None else None
                R_ul, R_ll, obs_leg = solve_two_bone_chain_rotations(
                    hp, kp, ap, global_R[0], rest[ll_b], rest[ft_b], knee_hint, prev_upper_R=prev_u
                )
                global_R[ul_b], global_R[ll_b] = R_ul, R_ll
                local_q[ul_b] = matrix_to_quat(global_R[0].T @ R_ul)
                local_q[ll_b] = matrix_to_quat(R_ul.T @ R_ll)
                j_conf[ul_b] = float(min(conf[hip_k], conf[kn_k]))
                j_conf[ll_b] = float(min(conf[kn_k], conf[an_k]))
                rot_obs[ul_b] = obs_leg
                rot_obs[ll_b] = obs_leg

                # Foot orientation from heel/ankle -> toe
                if tp is not None:
                    foot_fwd = tp - (hl if hl is not None else ap)
                    R_ft, ok_ft = _build_orthonormal_frame(foot_fwd, R_ll[:, 2], axis_order="yz")
                    if cl_id in clusters and clusters[cl_id].valid:
                        cf = clusters[cl_id]
                        R_ft = quat_to_matrix(quat_slerp(matrix_to_quat(R_ft), cf.rotation_quat_xyzw, self.config.cluster_orientation_weight * cf.confidence))
                        ok_ft = True
                    global_R[ft_b] = R_ft
                    local_q[ft_b] = matrix_to_quat(R_ll.T @ R_ft)
                    j_conf[ft_b] = float(min(conf[an_k], conf[toe_k]))
                    j_conf[to_b] = float(conf[toe_k])
                    rot_obs[ft_b] = ok_ft
                else:
                    global_R[ft_b] = R_ll.copy()
                    local_q[ft_b] = quat_identity()
                    j_conf[ft_b] = float(conf[an_k])
                global_R[to_b] = global_R[ft_b].copy()
            else:
                global_R[ul_b] = global_R[0].copy()
                global_R[ll_b] = global_R[ul_b].copy()
                global_R[ft_b] = global_R[ll_b].copy()
                global_R[to_b] = global_R[ft_b].copy()

        # 6. Cinema-mode nonlinear refinement incorporating calibrated surface markers + AI joints
        if self.config.enable_nonlinear_refinement and (surface_markers or np.sum(conf >= 0.4) >= 10):
            root_t, local_q = self._refine_frame_nonlinear(
                root_t, local_q, pts, conf, surface_markers
            )

        return root_t, local_q, global_R, j_conf, rot_obs

    def _refine_frame_nonlinear(
        self,
        root_t_init: np.ndarray,
        local_q_init: np.ndarray,
        pts26: np.ndarray,
        conf26: np.ndarray,
        surface_markers: Dict[str, Tuple[np.ndarray, float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Constrained LM refinement jointly fitting root + key articulated joints to AI landmarks and calibrated surface marker offsets."""
        # Map key Body26 joints -> rig bone indices for residual computation
        joint_pairs = [
            (19, 0), (5, 7), (7, 8), (9, 9), (6, 11), (8, 12), (10, 13),
            (11, 14), (13, 15), (15, 16), (20, 17), (12, 18), (14, 19), (16, 20), (21, 21), (18, 4),
        ]
        active_joints = [
            (k26, b_idx, pts26[k26], float(conf26[k26]))
            for k26, b_idx in joint_pairs
            if conf26[k26] >= 0.35 and np.all(np.isfinite(pts26[k26]))
        ]
        # Calibrated physical surface markers mapped via attached bone transform (NEVER averaged with AI joints!)
        active_markers = []
        for mid, (m_world, m_conf) in surface_markers.items():
            if mid in self.actor.calibrated_marker_offsets_m and m_conf >= 0.25 and np.all(np.isfinite(m_world)):
                bone_name, local_off = self.actor.calibrated_marker_offsets_m[mid]
                if bone_name in BONE_INDEX:
                    active_markers.append(
                        (BONE_INDEX[bone_name], np.asarray(local_off, dtype=np.float64), np.asarray(m_world, dtype=np.float64), float(m_conf))
                    )

        if not active_joints and not active_markers:
            return root_t_init, local_q_init

        # Optimize root translation (3) + incremental axis-angle deltas for primary movable joints (15 joints * 3)
        opt_bones = [0, 1, 3, 4, 7, 8, 11, 12, 14, 15, 16, 18, 19, 20]
        x0 = np.zeros(3 + 3 * len(opt_bones), dtype=np.float64)
        rest = self.actor.rest_offsets_m

        def apply_delta(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            rt = root_t_init + x[:3]
            lq = local_q_init.copy()
            for idx_i, b_idx in enumerate(opt_bones):
                rotvec = x[3 + 3 * idx_i : 6 + 3 * idx_i]
                if np.linalg.norm(rotvec) > 1e-8:
                    dq = Rotation.from_rotvec(rotvec).as_quat()
                    lq[b_idx] = quat_multiply(lq[b_idx], dq)
            return rt, lq

        def residuals(x: np.ndarray) -> np.ndarray:
            rt, lq = apply_delta(x)
            fk_pos, fk_mats = compute_forward_kinematics_frame(rt, lq, rest)
            res: List[np.ndarray] = []
            for _, b_idx, target_p, w in active_joints:
                res.append((fk_pos[b_idx] - target_p) * np.sqrt(w))
            for b_idx, local_off, target_m, w in active_markers:
                pred_m = fk_mats[b_idx, :3, :3] @ local_off + fk_mats[b_idx, :3, 3]
                # Higher spatial trust on physical markers (2.0x weight)
                res.append((pred_m - target_m) * (2.0 * np.sqrt(w)))
            # Mild regularization toward analytical solution to prevent gauge drift on unobserved DoFs
            res.append(x * 0.08)
            return np.concatenate(res)

        sol = least_squares(residuals, x0, method="trf", bounds=(-0.45, 0.45), max_nfev=25)
        return apply_delta(sol.x)
