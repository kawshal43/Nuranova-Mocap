"""Hybrid Cinema Capture & multi-profile reconstruction pipeline (Fast Preview / Standard / Cinema)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from engine.data_format.motion_io import (
    FaceMotionTrack,
    HandMotionTrack,
    MotionCaptureDocument,
    ProvenanceCode,
    SkeletalMotionTrack,
)
from engine.marker_tracking.detector import RigidClusterPose3D, solve_rigid_cluster_pose
from engine.marker_tracking.marker_kit import MarkerKitConfig
from engine.math_core.camera_model import MultiCameraCalibration
from engine.motion_cleanup.cleanup import (
    MotionCleanupConfig,
    clean_skeletal_motion,
    despike_and_fill_3d_points,
)
from engine.skeleton.rig_definition import BONE_NAMES, PARENT_INDICES, ActorProfile, calibrate_actor_from_body26_frames
from engine.skeleton.solver import ArticulatedSkeletonSolver, SolverConfig
from engine.triangulation.triangulator import triangulate_sequence


@dataclass
class ProcessingProfileSpec:
    name: str  # "fast_preview", "standard", "cinema"
    label: str
    is_approximate: bool
    frame_stride: int
    refine_triangulation_nonlinear: bool
    enable_solver_nonlinear_refinement: bool
    enable_motion_cleanup: bool
    enable_foot_locking: bool
    ai_model_mode: str  # "lightweight", "balanced", "performance"


PROCESSING_PROFILES: Dict[str, ProcessingProfileSpec] = {
    "fast_preview": ProcessingProfileSpec(
        name="fast_preview",
        label="Fast Preview (Approximate)",
        is_approximate=True,
        frame_stride=2,
        refine_triangulation_nonlinear=False,
        enable_solver_nonlinear_refinement=False,
        enable_motion_cleanup=False,
        enable_foot_locking=False,
        ai_model_mode="lightweight",
    ),
    "standard": ProcessingProfileSpec(
        name="standard",
        label="Standard",
        is_approximate=False,
        frame_stride=1,
        refine_triangulation_nonlinear=True,
        enable_solver_nonlinear_refinement=False,
        enable_motion_cleanup=True,
        enable_foot_locking=False,
        ai_model_mode="balanced",
    ),
    "cinema": ProcessingProfileSpec(
        name="cinema",
        label="Cinema (Production Quality)",
        is_approximate=False,
        frame_stride=1,
        refine_triangulation_nonlinear=True,
        enable_solver_nonlinear_refinement=True,
        enable_motion_cleanup=True,
        enable_foot_locking=True,
        ai_model_mode="performance",
    ),
}


@dataclass
class ReconstructionSummary:
    profile: str
    capture_mode: str
    num_frames: int
    mean_reprojection_error_px: float
    mean_joint_confidence: float
    interpolated_fraction: float
    low_confidence_joints: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class HybridCinemaReconstructor:
    """Orchestrates multi-camera triangulation, calibrated physical marker fusion, skeletal IK, and cleanup."""

    def __init__(
        self,
        calibration: MultiCameraCalibration,
        actor: Optional[ActorProfile] = None,
        marker_kit: Optional[MarkerKitConfig] = None,
    ) -> None:
        self.calibration = calibration
        self.actor = actor
        self.marker_kit = marker_kit

    def reconstruct_from_2d_observations(
        self,
        project_id: str,
        session_id: str,
        take_id: str,
        fps: float,
        frame_timestamps_ns: np.ndarray,
        ai_obs_2d_per_cam: Optional[Dict[str, np.ndarray]] = None,  # cam_uid -> (T, 26, 3)
        marker_obs_2d_per_cam: Optional[Dict[str, np.ndarray]] = None,  # cam_uid -> (T, M, 3)
        marker_names: Optional[List[str]] = None,
        capture_mode: str = "hybrid",  # "physical_marker", "markerless_ai", "hybrid"
        profile_name: str = "cinema",
        cleanup_config: Optional[MotionCleanupConfig] = None,
        face_track: Optional[FaceMotionTrack] = None,
        hand_track: Optional[HandMotionTrack] = None,
        model_versions: Optional[Dict[str, str]] = None,
    ) -> Tuple[MotionCaptureDocument, ReconstructionSummary]:
        profile = PROCESSING_PROFILES.get(profile_name, PROCESSING_PROFILES["cinema"])
        warnings: List[str] = []
        if profile.is_approximate:
            warnings.append("Fast Preview profile produces approximate motion for quick inspection only.")

        cams = self.calibration.cameras
        if len(cams) < 2:
            raise ValueError("Multi-view reconstruction requires at least 2 calibrated cameras.")

        T = len(frame_timestamps_ns)
        joints_3d = np.full((T, 26, 3), np.nan, dtype=np.float32)
        joints_conf = np.zeros((T, 26), dtype=np.float32)
        joints_reproj = np.full((T, 26), np.nan, dtype=np.float32)
        joints_prov = np.full((T, 26), int(ProvenanceCode.MISSING), dtype=np.uint8)

        # 1. Triangulate AI anatomical landmarks when available
        if ai_obs_2d_per_cam and capture_mode in ("markerless_ai", "hybrid"):
            joints_3d, joints_conf, joints_reproj, joints_prov = triangulate_sequence(
                cams,
                ai_obs_2d_per_cam,
                refine_nonlinear=profile.refine_triangulation_nonlinear,
                provenance_if_valid=ProvenanceCode.MEASURED_AI,
            )

        # 2. Triangulate physical markers & solve rigid marker clusters when available
        markers_3d = None
        markers_conf = None
        markers_reproj = None
        markers_prov = None
        rigid_clusters_per_frame: List[Dict[str, RigidClusterPose3D]] = [{} for _ in range(T)]
        surface_markers_per_frame: List[Dict[str, Tuple[np.ndarray, float]]] = [{} for _ in range(T)]

        if marker_obs_2d_per_cam and marker_names and capture_mode in ("physical_marker", "hybrid"):
            markers_3d, markers_conf, markers_reproj, markers_prov = triangulate_sequence(
                cams,
                marker_obs_2d_per_cam,
                max_reproj_error_px=8.0,
                refine_nonlinear=profile.refine_triangulation_nonlinear,
                provenance_if_valid=ProvenanceCode.MEASURED_MARKER,
            )
            for t in range(T):
                frame_m: Dict[str, Tuple[np.ndarray, float]] = {}
                for m_idx, m_name in enumerate(marker_names):
                    if markers_conf[t, m_idx] > 0.1 and np.all(np.isfinite(markers_3d[t, m_idx])):
                        frame_m[m_name] = (markers_3d[t, m_idx].astype(np.float64), float(markers_conf[t, m_idx]))
                surface_markers_per_frame[t] = frame_m
                if self.marker_kit is not None:
                    for cl in self.marker_kit.clusters:
                        pose = solve_rigid_cluster_pose(cl, frame_m)
                        if pose.valid:
                            rigid_clusters_per_frame[t][cl.cluster_id] = pose
                            if pose.possible_identity_swap:
                                warnings.append(f"Frame {t}: possible marker swap in cluster '{cl.cluster_id}'.")

            if capture_mode == "hybrid":
                # Mark joints supported by hybrid constraints without averaging surface markers onto joints
                valid_j = joints_prov == int(ProvenanceCode.MEASURED_AI)
                joints_prov[valid_j] = int(ProvenanceCode.HYBRID_FUSED)

        # 3. Despike & fill short gaps on 3D observations (tracking interpolated provenance accurately!)
        joints_3d_clean, interp_mask = despike_and_fill_3d_points(joints_3d, joints_conf)
        joints_prov[interp_mask] = int(ProvenanceCode.INTERPOLATED)
        interp_frac = float(np.mean(interp_mask))

        # 4. Calibrate actor automatically if not supplied
        actor = self.actor
        if actor is None:
            actor = calibrate_actor_from_body26_frames(
                actor_id="actor_auto",
                name="Auto-Calibrated Actor",
                body26_3d_frames=joints_3d_clean,
                body26_conf=joints_conf,
            )

        # 5. Articulated skeletal solve
        solver = ArticulatedSkeletonSolver(
            actor=actor,
            config=SolverConfig(
                enforce_anatomical_limits=True,
                enable_nonlinear_refinement=profile.enable_solver_nonlinear_refinement,
            ),
        )
        raw_track = solver.solve_sequence(
            body26_3d=joints_3d_clean,
            body26_conf=joints_conf,
            rigid_clusters_per_frame=rigid_clusters_per_frame,
            surface_markers_per_frame=surface_markers_per_frame,
        )

        # 6. Non-destructive motion cleanup (raw_motion always preserved!)
        corrected_track: Optional[SkeletalMotionTrack] = None
        if profile.enable_motion_cleanup:
            cfg = cleanup_config or MotionCleanupConfig(
                strength=0.75 if profile.name == "cinema" else 0.5,
                enable_foot_locking=profile.enable_foot_locking,
            )
            corrected_track = clean_skeletal_motion(raw_track, actor=actor, fps=fps, config=cfg)

        # Collect per-camera retained 2D observations for inspection/reprocessing
        obs_2d_store: Dict[str, Dict[str, np.ndarray]] = {}
        if ai_obs_2d_per_cam:
            for uid, arr in ai_obs_2d_per_cam.items():
                obs_2d_store.setdefault(uid, {})["keypoints_2d"] = arr.astype(np.float32)
        if marker_obs_2d_per_cam:
            for uid, arr in marker_obs_2d_per_cam.items():
                obs_2d_store.setdefault(uid, {})["markers_2d"] = arr.astype(np.float32)

        doc = MotionCaptureDocument(
            project_id=project_id,
            session_id=session_id,
            take_id=take_id,
            actor_id=actor.actor_id,
            fps=fps,
            frame_timestamps_ns=np.asarray(frame_timestamps_ns, dtype=np.int64),
            bone_names=list(BONE_NAMES),
            parent_indices=list(PARENT_INDICES),
            rest_offsets=actor.rest_offsets_m.astype(np.float32),
            raw_motion=raw_track,
            corrected_motion=corrected_track,
            capture_mode=capture_mode,
            processing_profile=profile.name,
            cameras=[{"camera_uid": u, "width": c.intrinsics.image_width, "height": c.intrinsics.image_height} for u, c in cams.items()],
            calibration=self.calibration.to_dict(),
            processing_config={"profile": profile.name, "capture_mode": capture_mode},
            model_versions=model_versions or {},
            marker_names=list(marker_names or []),
            markers_3d=markers_3d,
            markers_conf=markers_conf,
            markers_reproj_err_px=markers_reproj,
            markers_provenance=markers_prov,
            joints_3d_raw=joints_3d,
            joints_conf_raw=joints_conf,
            joints_reproj_err_px=joints_reproj,
            joints_provenance=joints_prov,
            observations_2d=obs_2d_store,
            face=face_track,
            hands=hand_track,
        )

        valid_reproj = joints_reproj[np.isfinite(joints_reproj)]
        mean_reproj = float(np.mean(valid_reproj)) if valid_reproj.size > 0 else 0.0
        mean_conf = float(np.mean(raw_track.joint_confidence))
        low_conf = [
            BONE_NAMES[b]
            for b in range(len(BONE_NAMES))
            if float(np.mean(raw_track.joint_confidence[:, b])) < 0.35
        ]
        summary = ReconstructionSummary(
            profile=profile.name,
            capture_mode=capture_mode,
            num_frames=T,
            mean_reprojection_error_px=mean_reproj,
            mean_joint_confidence=mean_conf,
            interpolated_fraction=interp_frac,
            low_confidence_joints=low_conf,
            warnings=warnings,
        )
        return doc, summary
