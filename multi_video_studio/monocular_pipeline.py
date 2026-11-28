"""Reviewable monocular reconstruction, sharing the studio solver and export path."""
from __future__ import annotations

import time
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np

from engine.ai_tracking.monocular_geometry import fit_monocular_sequence, align_camera_observations
from engine.data_format.motion_io import MotionCaptureDocument, ProvenanceCode
from engine.face.face_tracker import FacialCaptureEngine
from engine.hands.hand_tracker import HandCaptureEngine, HandObservation, solve_hand_rotations_from_landmarks21
from engine.hands.hand_fusion import apply_observed_wrist_orientation
from engine.hands.kinematics import calibrate_hand_template
from engine.math_core.transforms import matrix_to_quat, quat_to_matrix
from engine.motion_cleanup.quality_audit import audit_landmark_fit
from engine.skeleton.rig_definition import BONE_NAMES, PARENT_INDICES
from engine.skeleton.rig_definition import compute_forward_kinematics_sequence
from multi_video_studio.monocular_tracker import track_single_camera, resolved_backend
from multi_video_studio.review_observations import ReviewObservations
from multi_video_studio.review_state import ReviewState
from multi_video_studio.roi_tracking import refine_tracks_with_rois
from multi_video_studio.tracking_cache import tracking_cache_key, save_tracking_cache, load_tracking_cache
from multi_video_studio.camera_stabilization import ALGORITHM, estimate_camera_alignment, alignment_for_times
from multi_video_studio.pyav_sync import CameraSyncEstimate
from multi_video_studio.spatiotemporal_solver import solve_cinema_skeleton_from_3d
from multi_video_studio.zero_slide_physics import apply_zero_foot_slide_physics, compute_quality_scorecard


def correction_frame(observations, review, uid, pts):
    """Saved raw previews never override newly tracked data or a changed source."""
    if observations is None or uid not in observations.cameras:
        return None
    metadata = observations.cameras[uid]["metadata"]
    if metadata.get("asset_id") != review.sources.get(uid, {}).get("asset_id"):
        return None
    requested = (any(abs(float(t)-pts) < .001 for t in metadata.get("correction_pts", [])) or
                 any(a <= pts <= b for a, b in metadata.get("correction_ranges", [])))
    return observations.frame(uid, pts) if requested else None


def apply_hand_review(track, review, observations=None):
    """Re-fit manual image-space fingers in their estimated palm-depth frame."""
    output = [{} for _ in track.pts_s]
    uv_by_side = {}
    for side in ("left", "right"):
        raw = getattr(track, side + "_hand21_uvc")
        uv = raw.copy() if raw is not None else np.zeros((len(track.pts_s), 21, 3), np.float32)
        detections = {}
        if observations is not None:
            for i, pts in enumerate(track.pts_s):
                frame = correction_frame(observations, review, track.camera_uid, pts)
                has_roi = any(r["camera_uid"] == track.camera_uid and r["side"] == side
                              and r["start_s"] <= pts <= r["end_s"] for r in review.hand_rois)
                if frame is not None and not has_roi:
                    uv[i] = frame["hand:" + side]
                    detections[i] = frame
        uv, pins = review.apply_points(track.camera_uid, track.pts_s, uv, "hand:" + side)
        uv = review.apply_exclusions(track.camera_uid, track.pts_s, uv, "hands")
        uv_by_side[side] = uv
        for i, frame in enumerate(track.wrist_crop_hands):
            original = frame.get(side)
            if i in detections:
                local_points = detections[i]["local:" + side]
                if np.isfinite(local_points).all() and np.max(uv[i, :, 2]) > .1:
                    original = HandObservation(side, local_points, float(uv[i, :, 2].mean()),
                        solve_hand_rotations_from_landmarks21(local_points, handedness=side),
                        landmark_confidence=uv[i, :, 2], landmarks_2d=uv[i, :, :2], source="review_detection")
            if original is None or np.max(uv[i, :, 2]) <= .1:
                continue
            p = original.landmarks_3d.copy()
            if pins[i].any() and original.landmarks_2d is not None:
                # Keep estimated local depth; convert edited pixel displacement using palm scale.
                pixel_span = np.linalg.norm(original.landmarks_2d[5] - original.landmarks_2d[17])
                metric_span = np.linalg.norm(p[5] - p[17])
                ratio = metric_span / max(pixel_span, 5.)
                delta = uv[i, :, :2] - original.landmarks_2d
                image_x_sign = -1 if original.source.startswith("rtmw3d") else 1
                p[pins[i], 0] += image_x_sign * delta[pins[i], 0] * ratio
                p[pins[i], 2] -= delta[pins[i], 1] * ratio
            output[i][side] = HandObservation(side, p, float(np.mean(uv[i, :, 2])),
                solve_hand_rotations_from_landmarks21(p, handedness=side),
                landmark_confidence=uv[i, :, 2], landmarks_2d=uv[i, :, :2],
                source=(original.source + "_manual") if pins[i].any() else original.source)
    return replace(track, wrist_crop_hands=output, left_hand21_uvc=uv_by_side["left"],
                   right_hand21_uvc=uv_by_side["right"])


def uncertainty_intervals(times, confidence, threshold=.3):
    """Compress missing/weak source observations into actionable review ranges."""
    weak = (confidence[:, [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]] < threshold).any(axis=1)
    edges = np.diff(np.r_[False, weak, False].astype(int))
    return [{"start_s": float(times[a]), "end_s": float(times[b-1]),
             "reason": "One or more limbs has weak image evidence"}
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def image_foot_contacts(body, fps, times=None):
    """Stationary-camera stance evidence, independent of model depth jitter."""
    from scipy.ndimage import gaussian_filter1d, binary_closing, binary_opening
    uv = np.asarray(body, float)
    feet = uv[:, [24, 20, 25, 21], :2]
    smooth = gaussian_filter1d(feet, max(.5, fps*.05), axis=0)
    velocity = np.gradient(smooth, np.asarray(times), axis=0) if times is not None else np.gradient(smooth, axis=0)*fps
    speed = np.linalg.norm(velocity, axis=-1)
    height = np.percentile(np.max(uv[..., 1], axis=1)-np.min(uv[..., 1], axis=1), 75)
    floor_y = np.max(smooth[..., 1], axis=1)
    near_floor = (floor_y[:, None]-smooth[..., 1]) < max(12., height*.055)
    observed = uv[:, [24, 20, 25, 21], 2] >= .35
    contacts = (speed < max(10., height*.16)) & near_floor & observed
    for channel in range(4):
        contacts[:, channel] = binary_opening(binary_closing(contacts[:, channel], iterations=1), iterations=1)
    return contacts.astype(np.uint8)


def run_single_video(pipeline, pkg, out_dir, project_id, take_id, t_start, progress, cancel):
    from multi_video_studio.pipeline import CinemaPipelineResult, _resample_single_view
    cfg = pipeline.config
    uid, clip = next(iter(pkg.motion_clips.items()))
    review = ReviewState.from_dict(cfg.review_state)
    # Content identity is checked even if a replacement reuses the same filename.
    review.bind_sources(pkg.motion_clips)
    key_options = {"kind": "monocular_v3", "backend": resolved_backend(cfg),
        "resolved_backend": resolved_backend(cfg),
        "mode": cfg.ai_mode, "gpu": cfg.prefer_gpu, "stride": cfg.frame_stride,
        "height": cfg.actor_height_m, "fingers": cfg.enable_fingers, "face": cfg.enable_head_face,
        "hand_interval": cfg.single_hand_interval, "focal": cfg.single_focal_length_px,
        "range": [cfg.single_start_s, cfg.single_end_s],
        "model": "rtmw3d-b0a0eab7_20240626/mediapipe-0.10"}
    cache_path = (Path(cfg.tracking_cache_dir) / (tracking_cache_key(clip.path, key_options) + ".npz")
                  if cfg.tracking_cache_dir else None)
    full_cache = None
    if cache_path and (cfg.single_start_s is not None or cfg.single_end_s is not None):
        full_options = {**key_options, "range": [None, None]}
        full_cache = Path(cfg.tracking_cache_dir) / (tracking_cache_key(clip.path, full_options) + ".npz")
        if full_cache.exists():
            cache_path = full_cache
    cache_hit = False
    if cache_path and cache_path.exists():
        try:
            track = load_tracking_cache(cache_path)
            if track.relative_body26 is None:
                raise ValueError("Cache lacks native 3D estimates")
            if track.backend_tag != key_options["resolved_backend"]:
                raise ValueError("Cache uses a different tracking model")
            cache_hit = True
            if progress:
                progress(.65, "Reusing saved native tracking; applying review corrections…")
        except (OSError, ValueError, KeyError):
            track = track_single_camera(clip, cfg, cancel, progress)
    else:
        track = track_single_camera(clip, cfg, cancel, progress)
    pipeline._check_cancelled(cancel)
    if cache_path and not cache_hit and track.backend_tag == key_options["resolved_backend"]:
        save_tracking_cache(cache_path, track)
    # Camera slot names may change when importing the same content alone.
    track = replace(track, camera_uid=uid, video_path=clip.path)
    camera_moving = track.runtime_metadata.get("camera_motion", {}).get("detected", False)
    profile = track.runtime_metadata.get("camera_stabilization", {})
    if camera_moving and profile.get("algorithm") != ALGORITHM:
        try:
            profile = estimate_camera_alignment(clip.path, track.pts_s, track.body26_uvc, cancel,
                (lambda f, message: progress(.65+.015*f, message)) if progress else None)
        except (OSError, ValueError) as error:
            profile = {"algorithm":ALGORITHM, "accepted":False, "status":str(error)}
        track.runtime_metadata = {**track.runtime_metadata, "camera_stabilization":profile}
        if cache_path and track.backend_tag == key_options["resolved_backend"]:
            save_tracking_cache(cache_path, track)
    if full_cache is not None and cache_hit and cache_path == full_cache:
        # Retain half a second of context so interval boundaries can be smoothed.
        times = np.asarray(track.pts_s)
        low = 0. if cfg.single_start_s is None else max(0., cfg.single_start_s-.5)
        high = float('inf') if cfg.single_end_s is None else cfg.single_end_s+.5
        indices = np.flatnonzero((times >= low) & (times <= high))
        updates = {name: getattr(track, name)[indices] for name in
            ("body26_uvc", "left_hand21_uvc", "right_hand21_uvc", "face68_uvc", "relative_body26")
            if getattr(track, name) is not None}
        track = replace(track, **updates, pts_s=times[indices].tolist(),
            wrist_crop_hands=[track.wrist_crop_hands[i] for i in indices],
            head_crop_faces=[track.head_crop_faces[i] for i in indices])
    if len(track.pts_s) < 2:
        raise RuntimeError("Selected range has fewer than two source observations.")
    if cfg.enable_fingers and review.hand_rois:
        track = refine_tracks_with_rois(pkg, review, {uid: track}, cfg.tracking_cache_dir,
            cancel, (lambda f, message: progress(.66+.01*f, message)) if progress else None)[uid]
    native = ReviewObservations()
    local = {}
    for side in ("left", "right"):
        local[side] = np.asarray([f[side].landmarks_3d if side in f else np.full((21, 3), np.nan)
                                 for f in track.wrist_crop_hands])
    native.set_track(uid, track.pts_s, track.body26_uvc, track.left_hand21_uvc, track.right_hand21_uvc,
        local["left"], local["right"], source="native single-camera observations",
        asset_id=review.sources[uid]["asset_id"],
        backend=track.backend_tag, providers=track.runtime_metadata.get("providers", []))
    reviewed = None
    if cfg.review_observations_path and Path(cfg.review_observations_path).exists():
        reviewed = ReviewObservations.load(Path(cfg.review_observations_path))
    body = track.body26_uvc.copy()
    if reviewed:
        for i, pts in enumerate(track.pts_s):
            frame = correction_frame(reviewed, review, uid, pts)
            if frame is not None:
                body[i] = frame["body"]
    changed = np.linalg.norm(body[..., :2] - track.body26_uvc[..., :2], axis=-1) > .1
    body, pins = review.apply_points(uid, track.pts_s, body)
    manual_pin_count = int(pins.sum())
    pins |= changed
    for hint in cfg.single_depth_hints:
        hint_frames = ((np.asarray(track.pts_s) >= hint["start_s"]) &
                       (np.asarray(track.pts_s) <= hint["end_s"]))
        pins[hint_frames, hint["joint"]] = True
    body = review.apply_exclusions(uid, track.pts_s, body)
    if np.count_nonzero((body[:, [5, 6, 11, 12], 2] >= .4).sum(axis=1) >= 3) < 2:
        raise RuntimeError("No reliable body motion detected. Check the full performer is visible and enabled.")
    camera_transforms = alignment_for_times(profile, track.pts_s)
    contact_body = body
    camera_basis = None
    camera_supported = None
    if camera_transforms is not None:
        profile_times = np.asarray(profile["native_pts_s"])
        source_indices = np.clip(np.searchsorted(profile_times, track.pts_s), 0, len(profile_times)-1)
        camera_supported = np.asarray(profile["supported_mask"], bool)[source_indices]
        # Camera support governs global travel, not visible relative limb poses.
        _, contact_body, camera_basis = align_camera_observations(track.relative_body26, body, camera_transforms)
    points, confidence, geometry = fit_monocular_sequence(track.relative_body26, body, track.pts_s,
        clip.width, clip.height, cfg.actor_height_m, cfg.single_root_mode, cfg.single_focal_length_px,
        pins, cfg.single_depth_hints, camera_moving, camera_transforms, camera_supported)
    # Animation cadence is independent of variable phone frame arrival times.
    fps = cfg.target_fps or 30.
    points, confidence, uniform, nearest = _resample_single_view(points, confidence, track.pts_s, fps)
    source_grid = uniform + track.pts_s[0]
    keep = np.ones(len(source_grid), bool)
    if cfg.single_start_s is not None:
        keep &= source_grid >= cfg.single_start_s-1e-8
    if cfg.single_end_s is not None:
        keep &= source_grid <= cfg.single_end_s+1e-8
    points, confidence, source_grid, nearest = points[keep], confidence[keep], source_grid[keep], nearest[keep]
    if len(source_grid) < 2:
        raise RuntimeError("Selected range is too short to solve.")
    global_grid = np.asarray([review.global_time(uid, t) for t in source_grid])
    # Clock drift changes animation sample spacing as well as its start time.
    fps = 1 / float(np.median(np.diff(global_grid)))
    if progress:
        progress(.67, "Fitting source timing, body and hand corrections…")
    hand_track = None
    corrected_track = apply_hand_review(track, review, reviewed)
    if cfg.enable_fingers and any(corrected_track.wrist_crop_hands):
        builder = HandCaptureEngine()
        try:
            hand_track = builder.build_track([corrected_track.wrist_crop_hands[int(i)] for i in nearest],
                                             fps=fps, smooth=True)
        finally:
            builder.close()
        for side in ("left", "right"):
            palms = np.full((len(nearest), 4), np.nan, np.float32)
            landmarks = np.full((len(nearest), 21, 3), np.nan, np.float32)
            provenance = np.zeros((len(nearest), 21), np.uint8)
            for t, source_index in enumerate(nearest):
                obs = corrected_track.wrist_crop_hands[int(source_index)].get(side)
                if obs is None:
                    continue
                # MP hand world uses (x,z,-y); body uses (-x,-z,-y).
                rotation = np.eye(3) if obs.source.startswith("rtmw3d") else np.diag([-1., -1., 1.])
                if camera_basis is not None:
                    rotation = camera_basis[int(source_index)]@rotation
                palms[t] = matrix_to_quat(rotation @ quat_to_matrix(obs.local_rotations_xyzw[0]))
                wrist = points[t, 9 if side == "left" else 10]
                landmarks[t] = (obs.landmarks_3d - obs.landmarks_3d[0]) @ rotation.T + wrist
                provenance[t] = np.where(getattr(hand_track, side + "_confidence")[t] > .1, 2, 0)
            setattr(hand_track, side + "_palm_world_xyzw", palms)
            setattr(hand_track, side + "_landmarks_world", landmarks)
            setattr(hand_track, side + "_provenance", provenance)
            setattr(hand_track, side + "_template", calibrate_hand_template(landmarks, getattr(hand_track, side + "_confidence"), side))
    face_track = None
    if cfg.enable_head_face and any(f is not None for f in track.head_crop_faces):
        builder = FacialCaptureEngine()
        try:
            face_track = builder.build_track([track.head_crop_faces[int(i)] for i in nearest])
        finally:
            builder.close()
    raw, actor, optimized, repaired, head_cov = solve_cinema_skeleton_from_3d(points, confidence,
        # Face PnP is native-camera based; use aligned body head geometry instead
        # of mixing its camera roll back into the compensated skeleton.
        fps=fps, face_track=face_track if camera_basis is None else None,
        enable_nonlinear_refinement=cfg.enable_nonlinear_refinement,
        progress_cb=(lambda f, message: progress(.68+.12*f, message)) if progress else None,
        cancel_event=cancel, pin_mask=pins[nearest])
    native_fps = 1/float(np.median(np.diff(track.pts_s)))
    contacts = image_foot_contacts(contact_body, native_fps, track.pts_s)[nearest]
    if geometry["moving_camera_fallback"]:
        contacts = None
    cleaned, raw_slide, _, tilt = apply_zero_foot_slide_physics(raw, actor, fps,
        strength=cfg.zero_slide_strength, cutoff_hz=cfg.smooth_cutoff_hz,
        contacts_override=contacts, allow_floor_tilt=False, cancel_event=cancel,
        progress_cb=(lambda f, message: progress(.81+.13*f, message)) if progress else None)
    cleaned, hand_track = pipeline._stabilize_outputs(cleaned, actor, hand_track, fps)
    if geometry["root_mode"] == "in_place":
        root = cleaned.root_translation.copy()
        root[:, :2] = 0
        positions, matrices = compute_forward_kinematics_sequence(root, cleaned.local_rotations_xyzw, actor.rest_offsets_m)
        cleaned = replace(cleaned, root_translation=root, joint_positions=positions, global_matrices=matrices)
    cleaned = apply_observed_wrist_orientation(cleaned, hand_track, actor, fps)
    score = compute_quality_scorecard(cleaned, actor, fps, 1, "single_view_estimated", None,
                                     raw_slide, repaired, hand_track, head_cov, tilt)
    score.notes.append("Trajectory, hidden depth and fingers are estimates. Review weak evidence before export.")
    observation_path = out_dir / f"{take_id}_review_observations_{uuid.uuid4().hex[:12]}.npz"
    pipeline._check_cancelled(cancel)
    native.save(observation_path)
    sync = CameraSyncEstimate(uid, 0., "reference", 1., "Single video; original presentation timestamps")
    sync_data = review.clocks.get(uid, sync.__dict__)
    intervals = uncertainty_intervals(np.asarray(track.pts_s), body[..., 2])
    if camera_transforms is not None:
        intervals.extend(profile.get("uncertain_intervals", []))
    configuration = {"pipeline": "nuranova_single_video_v2", "scorecard": score.__dict__,
        "review_state": review.to_dict(), "review_observations_path": str(observation_path),
        "single_camera": {**track.runtime_metadata, **geometry, "cache_hit": cache_hit,
            "requested_backend": cfg.single_camera_backend, "prefer_gpu": cfg.prefer_gpu,
            "uncertain_intervals": intervals, "source_start_s": track.pts_s[0],
            "source_end_s": track.pts_s[-1], "native_frames": len(track.pts_s),
            "manual_body_points": manual_pin_count, "review_detection_points": int(changed.sum()),
            "depth_hints": cfg.single_depth_hints,
            "head_orientation_source":"aligned body landmarks" if camera_basis is not None else "body and native face estimates",
            "camera_motion_status": ("Background-aligned movement is estimated; camera translation/parallax is unresolved"
                if camera_transforms is not None else "Background motion detected; camera-derived movement disabled"
                if geometry["moving_camera_fallback"] else "Stationary camera assumed; world travel is not measured")},
        "skeletal_fit": audit_landmark_fit(raw.joint_positions, list(BONE_NAMES), optimized, confidence)}
    doc = MotionCaptureDocument(project_id, "single_video_studio", take_id, actor.actor_id, fps,
        np.round(global_grid * 1e9).astype(np.int64), list(BONE_NAMES), list(PARENT_INDICES),
        actor.rest_offsets_m.astype(np.float32), raw, corrected_motion=cleaned,
        capture_mode="single_video_cinema", processing_profile="cinema",
        cameras=[{"camera_uid": uid, "source": clip.path.name, **review.sources[uid]}],
        sync_info={uid: sync_data}, processing_config=configuration,
        model_versions={"body_3d": track.backend_tag, "device": track.device},
        joints_3d_raw=points, joints_conf_raw=confidence,
        joints_provenance=np.where(confidence > .1, int(ProvenanceCode.IK_ESTIMATED),
                                   int(ProvenanceCode.MISSING)).astype(np.uint8),
        observations_2d={uid: {"keypoints_2d": body[nearest],
            "left_hand21": corrected_track.left_hand21_uvc[nearest],
            "right_hand21": corrected_track.right_hand21_uvc[nearest],
            "native_pts_s": np.asarray(track.pts_s), "native_keypoints_2d": track.body26_uvc,
            "native_relative_body26": track.relative_body26,
            "native_left_hand21": track.left_hand21_uvc, "native_right_hand21": track.right_hand21_uvc,
            "native_left_local": local["left"], "native_right_local": local["right"]}},
        face=face_track, hands=hand_track)
    pipeline._check_cancelled(cancel)
    try:
        if progress:
            progress(.96, "Saving motion, native review points and exports…")
        h5, blender, bvh = pipeline._save_exports(doc, out_dir, take_id, cancel_event=cancel)
    except BaseException:
        observation_path.unlink(missing_ok=True)
        raise
    if progress:
        progress(1., score.summary_line())
    return CinemaPipelineResult(doc, score, None, {uid: sync}, h5, blender, bvh,
                                round(time.perf_counter()-t_start, 2))
