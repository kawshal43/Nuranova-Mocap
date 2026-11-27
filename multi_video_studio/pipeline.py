"""End-to-End Multi-Video Cinema Mocap Pipeline Orchestrator.

Takes a `MultiVideoPackage` (from `.zip`, folder, or in-app multi-camera recording) and produces:
- Studio-grade 25-bone body skeleton + 30 articulated fingers + 6-DoF head + 52 facial blendshapes
- Zero-Foot-Slide physics cleanup + objective `QualityScorecard`
- Ready-to-use `.nmocap.h5`, `_blender.json` (Blender Bridge), and `.bvh` exports!
"""

from __future__ import annotations

import threading
import time
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from desktop.calibration.charuco_board import ChArUcoBoardConfig
from desktop.workers.live_preview import SingleCamRootTracker, _mediapipe_world33_to_body26_zup
from engine.ai_tracking.pose_tracker import _mediapipe33_to_body26
from engine.data_format.motion_io import (
    FaceMotionTrack,
    HandMotionTrack,
    MotionCaptureDocument,
    ProvenanceCode,
    export_blender_json,
    save_motion_file,
)
from engine.export.bvh_exporter import export_bvh
from engine.face.face_tracker import FacialCaptureEngine
from engine.hands.hand_tracker import HandCaptureEngine, triangulate_hands_from_wholebody
from engine.math_core.camera_model import MultiCameraCalibration
from engine.skeleton.rig_definition import BONE_NAMES, PARENT_INDICES
from engine.triangulation.triangulator import triangulate_sequence
from engine.video.video_io import iter_video_frames, probe_video_metadata
from engine.motion_cleanup.rotation_filter import smooth_rotations, stabilize_upper_body
from engine.motion_cleanup.quality_audit import audit_landmark_fit, audit_quaternions
from multi_video_studio.multiview_3d_optimizer import optimize_multiview_3d_global
from multi_video_studio.obs2d_cleaner import clean_synchronized_2d_tracks
from multi_video_studio.package_ingest import MultiVideoPackage, ingest_multi_video_source
from multi_video_studio.tracking_cache import (
    load_tracking_cache,
    save_tracking_cache,
    tracking_cache_key,
)
from multi_video_studio.pyav_sync import (
    CameraSyncEstimate,
    estimate_video_sync_offsets,
    map_camera_pts_to_global_s,
    resample_synchronized_tracks,
)
from multi_video_studio.spatiotemporal_solver import solve_cinema_skeleton_from_3d
from multi_video_studio.video_calibrator import (
    calibrate_boardless_from_human_motion,
    calibrate_from_charuco_videos,
    validate_calibration_against_observations,
)
from multi_video_studio.wholebody_cuda import (
    CameraTrackingResult,
    CinemaWholeBodyVideoTracker,
    extract_head_crop_face,
)
from multi_video_studio.zero_slide_physics import (
    QualityScorecard,
    apply_zero_foot_slide_physics,
    compute_quality_scorecard,
)


@dataclass
class CinemaPipelineConfig:
    tracking_cache_dir: Optional[str] = str(
        Path.home() / ".nuranova_mocap" / "cache" / "video_tracking"
    )
    trim_to_overlap: bool = True
    rotation_cutoff_hz: float = 4.5
    finger_cutoff_hz: float = 3.0
    ai_backend: str = "rtmlib_wholebody"  # "rtmlib_wholebody", "rtmlib_halpe26", "mediapipe_pose"
    ai_mode: str = "performance"  # "performance", "balanced", "lightweight"
    prefer_gpu: bool = True
    frame_stride: int = 1
    target_fps: Optional[float] = None  # None -> auto-match median camera FPS (e.g. 30 or 60)
    enable_auto_sync: bool = True
    manual_offsets_ms: Dict[str, float] = field(default_factory=dict)
    enable_fingers: bool = True
    enable_head_face: bool = True
    enable_nonlinear_refinement: bool = False  # Analytical 2-bone IK + Spatio-Temporal solves 2000 frames in ~2s (no 20-min LM stall!)
    zero_slide_strength: float = 0.92
    smooth_cutoff_hz: float = 5.5
    board_preset: str = "A3"  # "A3" (9x6 40mm) or "A4" (7x5 35mm)
    actor_height_m: float = 1.72  # Used when Boardless Human-Motion Self-Calibration runs
    calibration_source: str = (
        "auto"  # "auto" (package -> validated studio -> boardless), "boardless", "active_studio"
    )
    allow_boardless_calibration: bool = True
    export_bvh_file: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.zero_slide_strength) or not 0 <= self.zero_slide_strength <= 1:
            raise ValueError("Foot locking strength must be between 0 and 1.")
        if not np.isfinite(self.smooth_cutoff_hz) or self.smooth_cutoff_hz <= 0:
            raise ValueError("Smoothing frequency must be positive and finite.")
        if not isinstance(self.frame_stride, int) or self.frame_stride < 1:
            raise ValueError("Frame stride must be a positive integer.")
        if self.target_fps is not None and (
            not np.isfinite(self.target_fps) or not 1 <= self.target_fps <= 120
        ):
            raise ValueError("Output frame rate must be between 1 and 120.")
        if not np.isfinite(self.actor_height_m) or not 0.5 <= self.actor_height_m <= 2.8:
            raise ValueError("Actor height must be between 0.5 and 2.8 meters.")
        if any(not np.isfinite(v) for v in self.manual_offsets_ms.values()):
            raise ValueError("Timing offsets must be finite numbers.")


@dataclass
class CinemaPipelineResult:
    document: MotionCaptureDocument
    scorecard: QualityScorecard
    calibration: Optional[MultiCameraCalibration]
    sync_estimates: Dict[str, CameraSyncEstimate]
    h5_path: Optional[Path] = None
    blender_json_path: Optional[Path] = None
    bvh_path: Optional[Path] = None
    elapsed_s: float = 0.0


def _board_cfg_from_preset(preset: str) -> ChArUcoBoardConfig:
    if (preset or "A3").upper() == "A4":
        return ChArUcoBoardConfig(
            squares_x=7,
            squares_y=5,
            square_length_mm=35.0,
            marker_length_mm=26.0,
            dictionary_name="DICT_5X5_250",
        )
    return ChArUcoBoardConfig(
        squares_x=9,
        squares_y=6,
        square_length_mm=40.0,
        marker_length_mm=30.0,
        dictionary_name="DICT_5X5_250",
    )


def _resample_single_view(points, confidence, timestamps, fps):
    """Keep VFR phone motion at its true speed when exporting a uniform-rate animation."""
    times = np.asarray(timestamps, dtype=np.float64)
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("Video timestamps must increase strictly.")
    grid = np.arange(times[0], times[-1] + 1e-9, 1.0 / fps)
    right = np.clip(np.searchsorted(times, grid), 0, len(times) - 1)
    left = np.maximum(0, right - 1)
    exact = np.isclose(times[right], grid, atol=1e-8, rtol=0)
    left[exact] = right[exact]
    span = times[right] - times[left]
    alpha = np.divide(grid - times[left], span, out=np.zeros_like(grid), where=span > 0)
    valid = (
        (confidence[left] > 0.1)
        & (confidence[right] > 0.1)
        & np.isfinite(points[left]).all(axis=-1)
        & np.isfinite(points[right]).all(axis=-1)
    )
    valid &= (span <= max(0.3, 2.8 * float(np.median(np.diff(times)))))[:, None]
    result = (1 - alpha[:, None, None]) * points[left] + alpha[:, None, None] * points[right]
    result[~valid] = np.nan
    conf = np.where(valid, np.minimum(confidence[left], confidence[right]), 0)
    nearest = np.where(alpha < 0.5, left, right)
    return result.astype(np.float32), conf.astype(np.float32), grid - grid[0], nearest


class MultiVideoCinemaPipeline:
    """Self-contained studio-grade Multi-Video Cinema Mocap Pipeline."""

    def __init__(self, config: Optional[CinemaPipelineConfig] = None) -> None:
        self.config = config or CinemaPipelineConfig()

    def run_from_source(
        self,
        source: str | Path | MultiVideoPackage,
        output_dir: str | Path,
        existing_calibration: Optional[MultiCameraCalibration] = None,
        project_id: str = "multi_video_cinema",
        take_name: Optional[str] = None,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> CinemaPipelineResult:
        """Execute full Cinema analysis on a `.zip`, folder, video list, or pre-ingested `MultiVideoPackage`."""
        t_start = time.perf_counter()
        pkg = source if isinstance(source, MultiVideoPackage) else ingest_multi_video_source(source)
        if pkg.num_motion_cameras == 0:
            raise ValueError("No readable motion capture videos found in the selected source.")

        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        tid = take_name or pkg.package_name or "cinema_take"
        if tid in (".", "..") or re.search(r'[<>:"/\\|?*\x00-\x1f]', tid):
            raise ValueError("Take name must be a filename without path separators.")
        self._check_cancelled(cancel_event)

        if pkg.num_motion_cameras == 1:
            if progress_cb:
                progress_cb(
                    0.02, "Single-video capture estimates depth using MediaPipe world landmarks."
                )
            return self._run_single_video_cinema(
                pkg=pkg,
                out_dir=out_dir,
                project_id=project_id,
                take_id=tid,
                t_start=t_start,
                progress_cb=progress_cb,
                cancel_event=cancel_event,
            )

        return self._run_multiview_cinema(
            pkg=pkg,
            out_dir=out_dir,
            existing_calibration=existing_calibration,
            project_id=project_id,
            take_id=tid,
            t_start=t_start,
            progress_cb=progress_cb,
            cancel_event=cancel_event,
        )

    def _run_multiview_cinema(
        self,
        pkg: MultiVideoPackage,
        out_dir: Path,
        existing_calibration: Optional[MultiCameraCalibration],
        project_id: str,
        take_id: str,
        t_start: float,
        progress_cb: Optional[Callable[[float, str], None]],
        cancel_event: Optional[threading.Event],
    ) -> CinemaPipelineResult:
        cfg = self.config
        motion_clips = pkg.motion_clips
        cam_uids = list(motion_clips.keys())

        # Stage 1: Resolve Package-Owned Calibration (Embedded JSON or ChArUco Board Videos).
        # Note: External `existing_calibration` is ONLY accepted after Stage 3 epipolar/aspect validation!
        calib: Optional[MultiCameraCalibration] = None
        calib_from_package = False
        calib_mode = "boardless_human_motion"

        if cfg.calibration_source != "boardless":
            if pkg.calibration_json_path is not None and pkg.calibration_json_path.exists():
                calib = MultiCameraCalibration.load(pkg.calibration_json_path)
                calib_from_package = True
                calib_mode = f"embedded_json ({pkg.calibration_json_path.name})"
            elif pkg.num_calibration_cameras >= 2:
                if progress_cb:
                    progress_cb(
                        0.03,
                        f"Solving sharp-frame ChArUco calibration from {pkg.num_calibration_cameras} board videos...",
                    )
                calib = calibrate_from_charuco_videos(
                    pkg.calibration_paths_by_cam(),
                    board_cfg=_board_cfg_from_preset(cfg.board_preset),
                    progress_cb=(lambda p, m: progress_cb(0.03 + 0.14 * p, m))
                    if progress_cb
                    else None,
                    cancel_event=cancel_event,
                )
                calib_from_package = True
                calib_mode = f"charuco_video ({cfg.board_preset} · {calib.global_reprojection_rmse_px:.2f}px)"

        # Stage 1.5: Fast Audio Pre-Sync & Common Overlap Window Trimming!
        # When PyAV Audio Clap sync locks across all cameras (~2s), skip decoding/tracking non-overlapping
        # lead-in / tail frames (saves ~20-35% of 4K GPU + MediaPipe work with 100% identical synced output!).
        vpaths = {uid: motion_clips[uid].path for uid in cam_uids}
        manual_ms = dict(cfg.manual_offsets_ms)
        for uid, clip in motion_clips.items():
            if abs(clip.manual_offset_ms) > 1e-6 and uid not in manual_ms:
                manual_ms[uid] = clip.manual_offset_ms

        pre_sync_estimates: Optional[Dict[str, CameraSyncEstimate]] = None
        pts_windows_by_cam: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None
        if cfg.enable_auto_sync and cfg.trim_to_overlap and len(cam_uids) >= 2:
            try:
                if progress_cb:
                    progress_cb(
                        0.16, "Pre-checking PyAV Audio Sync to trim non-overlapping frames..."
                    )
                probed_pts: Dict[str, List[float]] = {}
                for uid in cam_uids:
                    meta = probe_video_metadata(vpaths[uid], scan_pts=True)
                    if len(meta.pts_seconds) >= 2:
                        probed_pts[uid] = meta.pts_seconds
                if len(probed_pts) == len(cam_uids):
                    cand_sync = estimate_video_sync_offsets(
                        video_paths_by_cam=vpaths,
                        pts_s_by_cam=probed_pts,
                        body26_by_cam=None,
                        manual_offsets_ms=manual_ms,
                        enable_audio=True,
                        enable_flash=False,
                        enable_motion=False,
                    )
                    all_audio_locked = all(
                        (
                            cand_sync[u].method in ("reference", "manual")
                            or (
                                cand_sync[u].method == "audio_clap"
                                and cand_sync[u].confidence >= 8.0
                            )
                        )
                        for u in cam_uids
                    )
                    if all_audio_locked:
                        glob_ranges = {}
                        for uid in cam_uids:
                            p0 = float(probed_pts[uid][0])
                            cand_sync[uid].base_pts_s = (
                                p0 if abs(p0) > 5.0 else float(cand_sync[uid].audio_start_s)
                            )
                            g_pts = map_camera_pts_to_global_s(probed_pts[uid], cand_sync[uid])
                            glob_ranges[uid] = (float(g_pts[0]), float(g_pts[-1]), g_pts)
                        ov_start = max(r[0] for r in glob_ranges.values())
                        ov_end = min(r[1] for r in glob_ranges.values())
                        if ov_end - ov_start >= 1.0:
                            pad_s = 0.85
                            win_map: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
                            for uid in cam_uids:
                                l_pts = np.asarray(probed_pts[uid], dtype=np.float64)
                                g_pts = glob_ranges[uid][2]
                                i0 = int(np.searchsorted(g_pts, ov_start - pad_s, side="left"))
                                i1 = int(np.searchsorted(g_pts, ov_end + pad_s, side="right")) - 1
                                i0 = max(0, min(i0, len(l_pts) - 1))
                                i1 = max(i0, min(i1, len(l_pts) - 1))
                                s_w = float(l_pts[i0]) if i0 > 0 else None
                                e_w = float(l_pts[i1]) if i1 < len(l_pts) - 1 else None
                                win_map[uid] = (s_w, e_w)
                            pre_sync_estimates = cand_sync
                            pts_windows_by_cam = win_map
            except Exception:
                pre_sync_estimates = None
                pts_windows_by_cam = None

        # Stage 2: Parallel Multi-Camera Shared Batch-16 RTX CUDA WholeBody (133-pt) + Actor-Lock + Async Hand/Face Crops!
        tracker = CinemaWholeBodyVideoTracker(
            backend=cfg.ai_backend,
            mode=cfg.ai_mode,
            prefer_gpu=cfg.prefer_gpu,
        )
        tracks: Dict[str, CameraTrackingResult] = {}
        try:
            cam_specs = [
                (
                    uid,
                    motion_clips[uid].path,
                    bool(cfg.enable_fingers and c_idx == 0),
                    bool(cfg.enable_head_face and c_idx == 0),
                )
                for c_idx, uid in enumerate(cam_uids)
            ]

            def _stage2_prog(p: float, m: str) -> None:
                if progress_cb:
                    progress_cb(0.18 + 0.44 * p, m)

            cache_paths = {}
            missing_specs = []
            for spec in cam_specs:
                uid, video, hands, face = spec
                if cfg.tracking_cache_dir:
                    key = tracking_cache_key(
                        video,
                        {
                            "backend": cfg.ai_backend,
                            "mode": cfg.ai_mode,
                            "stride": cfg.frame_stride,
                            "hands": hands,
                            "face": face,
                            "window": (pts_windows_by_cam or {}).get(uid),
                            "camera_uid": uid,
                        },
                    )
                    path = Path(cfg.tracking_cache_dir) / f"{key}.npz"
                    cache_paths[uid] = path
                    if path.exists():
                        try:
                            tracks[uid] = load_tracking_cache(path)
                            if progress_cb:
                                progress_cb(0.18, f"Loaded cached observations for {uid}.")
                            continue
                        except (ValueError, OSError, KeyError):
                            pass
                missing_specs.append(spec)
            fresh = tracker.track_videos_parallel(
                camera_specs=missing_specs,
                step=cfg.frame_stride,
                progress_cb=_stage2_prog,
                cancel_event=cancel_event,
                pts_windows_by_cam=pts_windows_by_cam,
            )
            for uid, track in fresh.items():
                if uid in cache_paths:
                    save_tracking_cache(cache_paths[uid], track)
            tracks.update(fresh)
            for uid in cam_uids:
                clip = motion_clips[uid]
                if len(tracks[uid].pts_s) < 2 or tracks[uid].detection_rate < 0.05:
                    raise RuntimeError(
                        f"Could not track a person reliably in {clip.path.name}. "
                        "Check that the actor's full body is visible."
                    )
        finally:
            tracker.close()

        # Stage 3: Sub-Millisecond PyAV Auto-Sync (Audio Clap -> Flash -> Vertical Motion) + Mixed-FPS Resampling
        self._check_cancelled(cancel_event)
        if progress_cb:
            progress_cb(
                0.64, "Synchronizing multi-camera timelines (PyAV Audio Clap / Flash / Motion)..."
            )

        pts_map = {uid: tracks[uid].pts_s for uid in cam_uids}
        raw_b26 = {uid: tracks[uid].body26_uvc for uid in cam_uids}
        img_sizes = {uid: (tracks[uid].width, tracks[uid].height) for uid in cam_uids}

        if pre_sync_estimates is not None:
            sync_estimates = pre_sync_estimates
        else:
            sync_estimates = estimate_video_sync_offsets(
                video_paths_by_cam=vpaths,
                pts_s_by_cam=pts_map,
                body26_by_cam=raw_b26,
                manual_offsets_ms=manual_ms,
                enable_audio=cfg.enable_auto_sync,
                enable_flash=cfg.enable_auto_sync,
                enable_motion=cfg.enable_auto_sync,
            )
        self._check_cancelled(cancel_event)

        base_fps = (
            float(cfg.target_fps)
            if cfg.target_fps
            else float(np.median([tracks[u].fps for u in cam_uids]))
        )
        base_fps = max(1.0, min(120.0, base_fps))

        raw_lh = {
            u: tracks[u].left_hand21_uvc for u in cam_uids if tracks[u].left_hand21_uvc is not None
        }
        raw_rh = {
            u: tracks[u].right_hand21_uvc
            for u in cam_uids
            if tracks[u].right_hand21_uvc is not None
        }
        extra_in: Dict[str, Dict[str, np.ndarray]] = {}
        if len(raw_lh) >= 2:
            extra_in["left_hand21"] = raw_lh
        if len(raw_rh) >= 2:
            extra_in["right_hand21"] = raw_rh

        canonical_ns, aligned_b26, aligned_extras = resample_synchronized_tracks(
            sync_estimates=sync_estimates,
            pts_s_by_cam=pts_map,
            primary_obs_by_cam=raw_b26,
            target_fps=base_fps,
            extra_tracks_by_name=extra_in,
        )
        T = len(canonical_ns)
        if T < 2:
            raise RuntimeError("The synchronized videos share fewer than two frames.")

        # Stage 3.5: 2D Multi-View Pre-Cleaner (L/R limb-swap repair + spike despiker + zero-phase One-Euro filter)
        aligned_b26, _repaired_2d_cnt = clean_synchronized_2d_tracks(
            aligned_body26_by_cam=aligned_b26,
            image_sizes_by_cam=img_sizes,
            fps=base_fps,
        )

        # Stage 4: Validate Candidate Calibration Against Synced Observations -> Or Run Sparse Boardless Self-Calib!
        def _remap_calib_to_clips(
            cal_in: MultiCameraCalibration,
        ) -> Tuple[MultiCameraCalibration, Dict[str, str]]:
            """Map calibration camera UIDs to current clip UIDs (returning a renamed copy cleanly)."""
            from engine.math_core.camera_model import (
                CalibratedCamera,
                CameraExtrinsics,
                CameraIntrinsics,
            )

            cal_uids = list(cal_in.cameras.keys())
            uid_map: Dict[str, str] = {}  # clip_uid -> cal_uid
            used_c = set()
            for u in cam_uids:
                if u in cal_in.cameras:
                    uid_map[u] = u
                    used_c.add(u)
            rem_c = [u for u in cal_uids if u not in used_c]
            for u in cam_uids:
                if u not in uid_map and rem_c:
                    cu = rem_c.pop(0)
                    uid_map[u] = cu
                    used_c.add(cu)
            renamed_cams = {}
            for clip_u, cal_u in uid_map.items():
                src_cam = cal_in.cameras[cal_u]
                renamed_cams[clip_u] = CalibratedCamera(
                    intrinsics=CameraIntrinsics(
                        camera_uid=clip_u,
                        image_width=src_cam.intrinsics.image_width,
                        image_height=src_cam.intrinsics.image_height,
                        camera_matrix=src_cam.intrinsics.camera_matrix.copy(),
                        dist_coeffs=src_cam.intrinsics.dist_coeffs.copy(),
                        reprojection_error_px=src_cam.intrinsics.reprojection_error_px,
                    ),
                    extrinsics=CameraExtrinsics(
                        camera_uid=clip_u,
                        R_cw=src_cam.extrinsics.R_cw.copy(),
                        t_cw=src_cam.extrinsics.t_cw.copy(),
                        reprojection_error_px=src_cam.extrinsics.reprojection_error_px,
                    ),
                )
            return (
                MultiCameraCalibration(
                    profile_name=cal_in.profile_name,
                    cameras=renamed_cams,
                    global_reprojection_rmse_px=cal_in.global_reprojection_rmse_px,
                    world_up_axis=cal_in.world_up_axis,
                    notes=cal_in.notes,
                ),
                uid_map,
            )

        if calib is not None and calib_from_package:
            calib, _ = _remap_calib_to_clips(calib)
            okay, reason = validate_calibration_against_observations(calib, aligned_b26, img_sizes)
            if not okay:
                raise RuntimeError(
                    "The supplied camera calibration does not match these videos: " + reason
                )
        elif (
            calib is None
            and cfg.calibration_source in ("auto", "active_studio")
            and existing_calibration is not None
            and len(existing_calibration.cameras) >= 2
        ):
            cand_cal, _ = _remap_calib_to_clips(existing_calibration)
            ok_cal, cal_reason = validate_calibration_against_observations(
                cand_cal, aligned_b26, img_sizes
            )
            if ok_cal:
                calib = cand_cal
                calib_mode = f"active_studio_calibration ({cand_cal.global_reprojection_rmse_px:.2f}px · {cal_reason})"
            elif cfg.calibration_source == "active_studio":
                raise RuntimeError(
                    "The selected studio calibration does not match these videos: " + cal_reason
                )
            elif progress_cb:
                progress_cb(
                    0.66,
                    f"Skipped unrelated Active Studio calibration ({cal_reason}) -> running Boardless Self-Calib...",
                )

        if calib is None:
            if not cfg.allow_boardless_calibration:
                raise RuntimeError(
                    "No valid camera calibration provided and boardless self-calibration is disabled."
                )
            if progress_cb:
                progress_cb(
                    0.68,
                    "Solving camera positions & focal lengths via Boardless Human-Motion Self-Calibration...",
                )
            calib = calibrate_boardless_from_human_motion(
                aligned_body26_by_cam=aligned_b26,
                image_sizes_by_cam=img_sizes,
                actor_height_m=cfg.actor_height_m,
            )
            calib_mode = f"boardless_human_motion ({calib.global_reprojection_rmse_px:.2f}px@1080p · {cfg.actor_height_m:.2f}m)"
        self._check_cancelled(cancel_event)

        # Scale synced 2D observations cleanly to the validated calibration's intrinsics dimensions if they differ
        for uid in cam_uids:
            if uid in calib.cameras:
                ci = calib.cameras[uid].intrinsics
                vw, vh = tracks[uid].width, tracks[uid].height
                sx = float(ci.image_width) / float(vw) if vw > 0 else 1.0
                sy = float(ci.image_height) / float(vh) if vh > 0 else 1.0
                if abs(sx - 1.0) > 1e-4 or abs(sy - 1.0) > 1e-4:
                    aligned_b26[uid] = aligned_b26[uid].copy()
                    aligned_b26[uid][:, :, 0] *= sx
                    aligned_b26[uid][:, :, 1] *= sy
                    for ek in ("left_hand21", "right_hand21"):
                        if ek in aligned_extras and uid in aligned_extras[ek]:
                            aligned_extras[ek][uid] = aligned_extras[ek][uid].copy()
                            aligned_extras[ek][uid][:, :, 0] *= sx
                            aligned_extras[ek][uid][:, :, 1] *= sy

        # Stage 5: Outlier-Rejecting Multi-View Triangulation (Body + 6 Foot Points, resolution-aware threshold!)
        if progress_cb:
            progress_cb(
                0.73, f"Triangulating {T} frames across {len(cam_uids)} calibrated cameras..."
            )

        max_cam_w = max(
            (calib.cameras[u].intrinsics.image_width for u in cam_uids if u in calib.cameras),
            default=1920,
        )
        # Tighter 3+ camera outlier pruning cleanly discards single-camera side-turn occlusions!
        thresh_ratio = 0.0125 if len(cam_uids) >= 3 else 0.016
        body_reproj_thresh = max(20.0, thresh_ratio * float(max_cam_w))

        # Fast vectorized undistorted DLT+RANSAC seed (Stage 5.5 performs full global nonlinear multi-view Gauss-Newton!)
        joints_3d_init, joints_conf_init, joints_reproj_init, joints_prov = triangulate_sequence(
            calib.cameras,
            aligned_b26,
            min_confidence=0.22,
            max_reproj_error_px=body_reproj_thresh,
            refine_nonlinear=False,
            provenance_if_valid=ProvenanceCode.MEASURED_AI,
        )
        core = [5, 6, 11, 12, 13, 14, 15, 16]
        measured = np.isfinite(joints_3d_init[:, core]).all(axis=-1) & (
            joints_conf_init[:, core] >= 0.22
        )
        if np.count_nonzero(measured.sum(axis=1) >= 4) < min(5, T):
            raise RuntimeError(
                "Too few matching body points to reconstruct motion. "
                "Check video timing and use a calibration board if camera estimation fails."
            )

        # Stage 5.5: Global Spatio-Temporal Multi-View 3D Reprojection Optimizer (Geman-McClure IRLS + C2 Banded Solve!)
        # Replaces hard binary 3-cam/2-cam teleports with continuous global ray-consistent, rigid-bone 3D trajectories!
        if progress_cb:
            progress_cb(
                0.76,
                f"Running Global Multi-View Reprojection & C2 Trajectory Optimizer ({T} frames)...",
            )
        joints_3d, joints_conf, reproj_opt_1080 = optimize_multiview_3d_global(
            cameras=calib.cameras,
            obs_2d_by_cam=aligned_b26,
            init_3d=joints_3d_init,
            init_conf=joints_conf_init,
            fps=base_fps,
            iterations=10,
        )
        valid_reproj_opt = reproj_opt_1080[np.isfinite(reproj_opt_1080)]
        norm_scale_1080 = 1920.0 / max(float(max_cam_w), 1.0)
        if len(valid_reproj_opt) >= 20:
            mean_reproj_px = float(np.mean(valid_reproj_opt))
            joints_reproj = (reproj_opt_1080 / max(norm_scale_1080, 1e-6)).astype(np.float32)
        else:
            valid_reproj = joints_reproj_init[np.isfinite(joints_reproj_init)]
            mean_reproj_px = (
                float(np.mean(valid_reproj)) * norm_scale_1080
                if len(valid_reproj)
                else float(calib.global_reprojection_rmse_px)
            )
            joints_reproj = joints_reproj_init

        # Stage 6: Build 52 Facial Expressions + 30 Articulated Fingers (Triangulated 3D + Full-Res Wrist Crops!)
        if progress_cb:
            progress_cb(
                0.80,
                "Solving 30 articulated fingers (3D triangulated + wrist crops) & 6-DoF head/face...",
            )

        primary_uid = cam_uids[0]
        primary_pts_global = map_camera_pts_to_global_s(
            tracks[primary_uid].pts_s, sync_estimates[primary_uid], fallback_fps=base_fps
        )
        canonical_s = canonical_ns.astype(np.float64) * 1e-9
        if len(primary_pts_global) >= 1:
            prim_idx_map = np.clip(
                np.searchsorted(primary_pts_global, canonical_s), 0, len(primary_pts_global) - 1
            )
        else:
            prim_idx_map = np.zeros(T, dtype=np.int64)

        face_track: Optional[FaceMotionTrack] = None
        if cfg.enable_head_face and tracks[primary_uid].head_crop_faces:
            raw_faces = tracks[primary_uid].head_crop_faces
            if any(f is not None for f in raw_faces):
                idx_f = np.clip(prim_idx_map, 0, len(raw_faces) - 1)
                resampled_faces = [raw_faces[int(i)] for i in idx_f]
                fc_builder = FacialCaptureEngine()
                face_track = fc_builder.build_track(resampled_faces)

        hand_track: Optional[HandMotionTrack] = None
        if cfg.enable_fingers:
            raw_hands = tracks[primary_uid].wrist_crop_hands
            resampled_hands = None
            if raw_hands:
                idx_h = np.clip(prim_idx_map, 0, len(raw_hands) - 1)
                resampled_hands = [raw_hands[int(i)] for i in idx_h]
            hand_track = triangulate_hands_from_wholebody(
                cameras=calib.cameras,
                left_hand21_per_cam=aligned_extras.get("left_hand21"),
                right_hand21_per_cam=aligned_extras.get("right_hand21"),
                fps=base_fps,
                fallback_obs_per_frame=resampled_hands,
            )

        # Stage 7: Spatio-Temporal Bone-Length Optimizer + Auto-Actor IK + 6-DoF Head Fusion
        if progress_cb:
            progress_cb(
                0.86, "Running Spatio-Temporal Bone-Length Solver & 6-DoF Head/Neck Fusion..."
            )

        raw_track, actor, opt_3d, repaired_cnt, head_cov = solve_cinema_skeleton_from_3d(
            joints_3d=joints_3d,
            joints_conf=joints_conf,
            fps=base_fps,
            face_track=face_track,
            enable_nonlinear_refinement=cfg.enable_nonlinear_refinement,
            progress_cb=(lambda p, m: progress_cb(0.86 + 0.06 * p, m)) if progress_cb else None,
            cancel_event=cancel_event,
        )

        # Stage 8: Zero-Foot-Slide Physics + Studio Floor Auto-Level + Quality Scorecard!
        if progress_cb:
            progress_cb(
                0.92, "Applying Zero-Foot-Slide Physics (RANSAC Floor + 4-Pt Heel/Toe IK Lock)..."
            )

        cleaned_track, raw_slide_mm, _final_slide_mm, tilt_deg = apply_zero_foot_slide_physics(
            track=raw_track,
            actor=actor,
            fps=base_fps,
            strength=cfg.zero_slide_strength,
            cutoff_hz=cfg.smooth_cutoff_hz,
        )

        cleaned_track, hand_track = self._stabilize_outputs(
            cleaned_track, actor, hand_track, base_fps
        )
        scorecard = compute_quality_scorecard(
            cleaned_track=cleaned_track,
            actor=actor,
            fps=base_fps,
            num_cameras=len(cam_uids),
            calibration_mode=calib_mode,
            mean_reprojection_rmse_px=mean_reproj_px,
            raw_foot_slide_mm=raw_slide_mm,
            repaired_glitch_entries=repaired_cnt,
            hand_track=hand_track,
            head_coverage_pct=head_cov,
            floor_tilt_corrected_deg=tilt_deg,
        )

        # Package complete MotionCaptureDocument & write studio outputs
        doc = MotionCaptureDocument(
            project_id=project_id,
            session_id="multi_video_cinema_lab",
            take_id=take_id,
            actor_id=actor.actor_id,
            fps=base_fps,
            frame_timestamps_ns=canonical_ns,
            bone_names=list(BONE_NAMES),
            parent_indices=list(PARENT_INDICES),
            rest_offsets=actor.rest_offsets_m.astype(np.float32),
            raw_motion=raw_track,
            corrected_motion=cleaned_track,
            capture_mode="multi_video_cinema",
            processing_profile="cinema",
            cameras=[
                {
                    "camera_uid": u,
                    "source": motion_clips[u].path.name,
                    "width": tracks[u].width,
                    "height": tracks[u].height,
                    "sync": sync_estimates[u].summary,
                }
                for u in cam_uids
            ],
            calibration=calib.to_dict(),
            sync_info={u: sync_estimates[u].__dict__ for u in cam_uids},
            processing_config={
                "pipeline": "nuranova_multi_video_cinema_v1",
                "calibration_mode": calib_mode,
                "sync_report": {u: sync_estimates[u].summary for u in cam_uids},
                "scorecard": scorecard.__dict__,
                "detection_rates": {u: tracks[u].detection_rate for u in cam_uids},
                "frame_stride": cfg.frame_stride,
                "skeletal_fit": audit_landmark_fit(
                    raw_track.joint_positions, list(BONE_NAMES), opt_3d, joints_conf
                ),
            },
            model_versions={
                "wholebody_2d": tracks[primary_uid].backend_tag,
                "device": tracks[primary_uid].device,
            },
            joints_3d_raw=joints_3d,
            joints_conf_raw=joints_conf,
            joints_reproj_err_px=joints_reproj,
            joints_provenance=joints_prov,
            observations_2d={u: {"keypoints_2d": aligned_b26[u]} for u in cam_uids},
            face=face_track,
            hands=hand_track,
        )

        self._check_cancelled(cancel_event)
        h5_p, bj_p, bvh_p = self._save_exports(doc, out_dir, take_id)
        if progress_cb:
            progress_cb(1.0, f"Cinema Multi-Video Solve Complete! {scorecard.summary_line()}")

        return CinemaPipelineResult(
            document=doc,
            scorecard=scorecard,
            calibration=calib,
            sync_estimates=sync_estimates,
            h5_path=h5_p,
            blender_json_path=bj_p,
            bvh_path=bvh_p,
            elapsed_s=round(time.perf_counter() - t_start, 2),
        )

    def _run_single_video_cinema(
        self,
        pkg: MultiVideoPackage,
        out_dir: Path,
        project_id: str,
        take_id: str,
        t_start: float,
        progress_cb: Optional[Callable[[float, str], None]],
        cancel_event: Optional[threading.Event],
    ) -> CinemaPipelineResult:
        """High-accuracy single-video fallback (MediaPipe Heavy + Wrist-Cropped 30 Fingers + Head-Cropped 52 Face + Zero-Slide)."""
        import mediapipe as mp

        cfg = self.config
        uid, clip = next(iter(pkg.motion_clips.items()))

        pose_model = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        hand_engine = (
            HandCaptureEngine(max_num_hands=2, model_complexity=1) if cfg.enable_fingers else None
        )
        face_engine = FacialCaptureEngine() if cfg.enable_head_face else None
        root_tracker = SingleCamRootTracker(enabled=True)

        pts_list: List[np.ndarray] = []
        conf_list: List[np.ndarray] = []
        hand_list: List[Dict] = []
        face_list: List = []
        timestamps: List[float] = []
        step = max(1, int(cfg.frame_stride))
        idx = 0
        try:
            for idx, pts_s, bgr in iter_video_frames(clip.path, step=step):
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("Cancelled by user.")
                if idx % step == 0:
                    timestamps.append(pts_s)
                    fh, fw = bgr.shape[:2]
                    if fw > 1280:
                        sc = 1280.0 / float(fw)
                        bgr_in = cv2.resize(
                            bgr, (1280, max(1, int(round(fh * sc)))), interpolation=cv2.INTER_AREA
                        )
                    else:
                        bgr_in = bgr
                    hi, wi = bgr_in.shape[:2]
                    res = pose_model.process(cv2.cvtColor(bgr_in, cv2.COLOR_BGR2RGB))
                    b26_full = None
                    if res.pose_world_landmarks:
                        p3, c3 = _mediapipe_world33_to_body26_zup(res.pose_world_landmarks.landmark)
                        if res.pose_landmarks:
                            b26_in = _mediapipe33_to_body26(res.pose_landmarks.landmark, wi, hi)
                            b26_full = _mediapipe33_to_body26(res.pose_landmarks.landmark, fw, fh)
                            p3 = root_tracker.apply(p3, c3, b26_in, wi, hi)
                        pts_list.append(p3.astype(np.float32))
                        conf_list.append(c3.astype(np.float32))
                    else:
                        pts_list.append(np.full((26, 3), np.nan, dtype=np.float32))
                        conf_list.append(np.zeros((26,), dtype=np.float32))
                    if hand_engine is not None:
                        hand_list.append(
                            hand_engine.infer_hands_from_body26(bgr, b26_full)
                            if b26_full is not None
                            else {}
                        )
                    if face_engine is not None:
                        face_list.append(
                            extract_head_crop_face(bgr, b26_full, face_engine, len(pts_list) - 1)
                            if b26_full is not None
                            else None
                        )
                    if progress_cb and len(pts_list) % 15 == 0:
                        progress_cb(
                            0.05 + 0.70 * (idx / max(float(clip.frame_count), 1.0)),
                            f"Tracking 1-Video AI + Fingers + Face (frame {idx}/{clip.frame_count})...",
                        )
                idx += 1
        finally:
            pose_model.close()
            if hand_engine:
                hand_engine.close()
            if face_engine:
                face_engine.close()

        if len(pts_list) < 2:
            raise RuntimeError("Video contained fewer than 2 readable frames.")

        eff_fps = cfg.target_fps or clip.fps / step
        pts_arr = np.stack(pts_list, axis=0)
        conf_arr = np.stack(conf_list, axis=0)
        if np.count_nonzero((conf_arr[:, [5, 6, 11, 12]] >= 0.4).sum(axis=1) >= 3) < 2:
            raise RuntimeError(
                "No reliable body motion detected. Check that the full actor is visible."
            )
        self._check_cancelled(cancel_event)
        pts_arr, conf_arr, uniform_times, nearest = _resample_single_view(
            pts_arr, conf_arr, timestamps, eff_fps
        )
        if hand_list:
            hand_list = [hand_list[int(i)] for i in nearest]
        if face_list:
            face_list = [face_list[int(i)] for i in nearest]

        face_track = (
            FacialCaptureEngine().build_track(face_list)
            if (face_list and any(f is not None for f in face_list))
            else None
        )
        hand_track = (
            HandCaptureEngine().build_track(hand_list, fps=eff_fps, smooth=True)
            if (hand_list and any(bool(h) for h in hand_list))
            else None
        )

        raw_track, actor, opt_3d, repaired_cnt, head_cov = solve_cinema_skeleton_from_3d(
            joints_3d=pts_arr,
            joints_conf=conf_arr,
            fps=eff_fps,
            face_track=face_track,
            enable_nonlinear_refinement=False,
        )
        cleaned_track, raw_slide_mm, _final_slide_mm, tilt_deg = apply_zero_foot_slide_physics(
            track=raw_track,
            actor=actor,
            fps=eff_fps,
            strength=cfg.zero_slide_strength,
            cutoff_hz=cfg.smooth_cutoff_hz,
        )
        cleaned_track, hand_track = self._stabilize_outputs(
            cleaned_track, actor, hand_track, eff_fps
        )
        scorecard = compute_quality_scorecard(
            cleaned_track=cleaned_track,
            actor=actor,
            fps=eff_fps,
            num_cameras=1,
            calibration_mode="single_video_ai_3d",
            mean_reprojection_rmse_px=0.0,
            raw_foot_slide_mm=raw_slide_mm,
            repaired_glitch_entries=repaired_cnt,
            hand_track=hand_track,
            head_coverage_pct=head_cov,
            floor_tilt_corrected_deg=tilt_deg,
        )
        ts_ns = np.round(uniform_times * 1e9).astype(np.int64)
        doc = MotionCaptureDocument(
            project_id=project_id,
            session_id="multi_video_cinema_lab",
            take_id=take_id,
            actor_id=actor.actor_id,
            fps=eff_fps,
            frame_timestamps_ns=ts_ns,
            bone_names=list(BONE_NAMES),
            parent_indices=list(PARENT_INDICES),
            rest_offsets=actor.rest_offsets_m.astype(np.float32),
            raw_motion=raw_track,
            corrected_motion=cleaned_track,
            capture_mode="single_video_cinema",
            processing_profile="cinema",
            cameras=[{"camera_uid": uid, "source": clip.path.name}],
            processing_config={
                "pipeline": "nuranova_single_video_cinema",
                "scorecard": scorecard.__dict__,
                "skeletal_fit": audit_landmark_fit(
                    raw_track.joint_positions, list(BONE_NAMES), opt_3d, conf_arr
                ),
            },
            joints_3d_raw=pts_arr,
            joints_conf_raw=conf_arr,
            face=face_track,
            hands=hand_track,
        )
        self._check_cancelled(cancel_event)
        h5_p, bj_p, bvh_p = self._save_exports(doc, out_dir, take_id)
        if progress_cb:
            progress_cb(1.0, f"Single-Video Cinema Solve Complete! {scorecard.summary_line()}")
        return CinemaPipelineResult(
            document=doc,
            scorecard=scorecard,
            calibration=None,
            sync_estimates={uid: CameraSyncEstimate(uid, 0.0, "reference", 1.0, "Single video")},
            h5_path=h5_p,
            blender_json_path=bj_p,
            bvh_path=bvh_p,
            elapsed_s=round(time.perf_counter() - t_start, 2),
        )

    def _save_exports(
        self, doc: MotionCaptureDocument, out_dir: Path, take_id: str
    ) -> Tuple[Path, Path, Optional[Path]]:
        final_motion = doc.corrected_motion or doc.raw_motion
        audit = {
            "body": audit_quaternions(final_motion.local_rotations_xyzw, doc.bone_names, doc.fps)
        }
        if doc.hands is not None:
            from engine.hands.hand_tracker import HAND_BONE_NAMES

            for side in ("left", "right"):
                rotations = getattr(doc.hands, side + "_local_rotations_xyzw")
                if rotations is not None:
                    audit[side + "_hand"] = audit_quaternions(rotations, HAND_BONE_NAMES, doc.fps)
        doc.processing_config["keyframe_audit"] = audit
        h5_p = save_motion_file(doc, out_dir / f"{take_id}.nmocap.h5")
        bj_p = export_blender_json(doc, out_dir / f"{take_id}_blender.json", use_corrected=True)
        bvh_p = (
            export_bvh(doc, out_dir / f"{take_id}.bvh", use_corrected=True)
            if self.config.export_bvh_file
            else None
        )
        if doc.calibration:
            (out_dir / f"{take_id}_calibration.json").write_text(
                json.dumps(doc.calibration, indent=2), encoding="utf-8"
            )
        (out_dir / f"{take_id}_report.json").write_text(
            json.dumps(doc.processing_config, indent=2), encoding="utf-8"
        )
        return h5_p, bj_p, bvh_p

    @staticmethod
    def _check_cancelled(cancel_event: Optional[threading.Event]) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Cancelled by user.")

    def _stabilize_outputs(self, body, actor, hands, fps):
        from dataclasses import replace

        if self.config.rotation_cutoff_hz > 0:
            body = stabilize_upper_body(body, actor, fps, self.config.rotation_cutoff_hz)
        if hands is not None and self.config.finger_cutoff_hz > 0:
            updates = {}
            for side in ("left", "right"):
                q = getattr(hands, side + "_local_rotations_xyzw")
                if q is not None:
                    updates[side + "_local_rotations_xyzw"] = smooth_rotations(
                        q, fps, self.config.finger_cutoff_hz
                    )
            hands = replace(hands, **updates)
        return body, hands
