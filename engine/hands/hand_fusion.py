"""Visibility-aware multi-view palm orientation and constrained finger fitting."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from engine.data_format.motion_io import HandMotionTrack
from engine.hands.kinematics import calibrate_hand_template, hand_forward_kinematics, palm_frame
from engine.math_core.transforms import enforce_quaternion_continuity, quat_slerp


def detailed_hand_uv(track, side):
    """Use observed native crop landmarks, retaining whole-body points between crop samples."""
    original = getattr(track, side + "_hand21_uvc")
    result = original.copy() if original is not None else np.zeros((len(track.pts_s), 21, 3), np.float32)
    wrist_index = 9 if side == "left" else 10
    for t, hands in enumerate(track.wrist_crop_hands):
        hand = hands.get(side)
        if hand is None or hand.source == "interpolated" or hand.landmarks_2d is None:
            continue
        uv = np.asarray(hand.landmarks_2d)
        wrist = track.body26_uvc[t, wrist_index]
        # Reject another hand entering the ROI or normalized UV from an old cache.
        width = np.linalg.norm(uv[17] - uv[5])
        if width < 5 or not np.isfinite(uv).all() or (hand.source != "manual_roi_model" and np.linalg.norm(uv[0] - wrist[:2]) > max(60, width * 2.5)):
            continue
        confidence = min(hand.confidence, float(np.clip(width / 30., .15, 1.)))
        result[t, :, :2], result[t, :, 2] = uv, confidence
    return result


def select_crop_views(observations_by_camera, frame_count, visibility=None):
    """Select articulation priors with temporal hysteresis; camera-local palms are never world poses."""
    result = [{} for _ in range(frame_count)]
    previous_camera = {"left": None, "right": None}
    previous_q = {}
    for t in range(frame_count):
        for side in ("left", "right"):
            candidates = []
            for uid, frames in observations_by_camera.items():
                obs = frames[t].get(side) if t < len(frames) else None
                if obs is None or obs.confidence < .2:
                    continue
                weight = visibility[uid][t, 9 if side == "left" else 10] if visibility and uid in visibility else 1.
                score = obs.confidence * weight * (.55 if obs.source == "interpolated" else 1.)
                if previous_camera[side] == uid:
                    score += .12
                if side in previous_q:
                    dots = np.clip(abs(np.sum(previous_q[side][1:] * obs.local_rotations_xyzw[1:], axis=1)), 0, 1)
                    score -= min(.2, float(np.median(np.arccos(dots))))
                weighted = replace(obs, confidence=obs.confidence * weight,
                    landmark_confidence=obs.landmark_confidence * weight if obs.landmark_confidence is not None else None)
                candidates.append((score, uid, weighted))
            if candidates:
                _, uid, obs = max(candidates, key=lambda item: item[0])
                result[t][side] = obs
                previous_camera[side], previous_q[side] = uid, obs.local_rotations_xyzw
    return result


def _parameters_to_quaternions(parameters):
    curls = np.asarray(parameters[:15]).reshape(5, 3)
    splays = np.asarray(parameters[15:20])
    quats = np.zeros((16, 4), float)
    quats[0, 3] = 1
    for finger in range(5):
        for segment in range(3):
            curl, splay = curls[finger, segment], splays[finger] if segment == 0 else 0.
            sx, cx = np.sin(-curl / 2), np.cos(curl / 2)
            sz, cz = np.sin(splay / 2), np.cos(splay / 2)
            quats[1 + 3 * finger + segment] = [cz * sx, sz * sx, sz * cx, cz * cx]
    return quats


def fit_hand_frame(points_local, confidence, side, template, initial=None, previous=None):
    """Fit fixed segment lengths with bounded MCP splay and PIP/DIP hinges.

    Missing points contribute no residual. The prior supplies pose, not observation confidence.
    """
    confidence = np.asarray(confidence, float).copy()
    valid = (confidence >= .2) & np.isfinite(points_local).all(axis=1)
    if initial is None:
        initial = np.tile([0., 0., 0., 1.], (16, 1))
    angles = Rotation.from_quat(initial[1:]).as_euler("xyz")
    parameters = np.r_[-angles[:, 0], angles[::3, 2]]
    lower = np.r_[np.tile([-.12, 0., 0.], 5), [-.85, -.28, -.24, -.24, -.30]]
    upper = np.r_[np.tile([1.52, 1.80, 1.35], 5), [.85, .28, .24, .24, .30]]
    parameters = np.clip(parameters, lower + 1e-7, upper - 1e-7)
    if valid.sum() < 4:
        return np.asarray(previous if previous is not None else initial).copy()
    target = np.asarray(points_local)[valid]
    weights = np.sqrt(confidence[valid])[:, None]
    prior = parameters.copy()
    if previous is not None:
        angles_prev = Rotation.from_quat(previous[1:]).as_euler("xyz")
        prior = np.r_[-angles_prev[:, 0], angles_prev[::3, 2]]

    def residual(value):
        predicted = hand_forward_kinematics(_parameters_to_quaternions(value), side, template)
        # 1 cm image-derived position residual; soft prior resists underconstrained motion.
        position = ((predicted[valid] - target) * weights / .01).ravel()
        tendon = (value[:15].reshape(5, 3)[1:, 2] - .65 * value[:15].reshape(5, 3)[1:, 1]) * .12
        return np.r_[position, (value - prior) * .18, tendon]

    solution = least_squares(residual, parameters, bounds=(lower, upper), loss="soft_l1",
                             diff_step=.002, max_nfev=7, ftol=.02, xtol=.005)
    return _parameters_to_quaternions(solution.x).astype(np.float32)


def fuse_hands(cameras, left_hand21_per_cam, right_hand21_per_cam, fps=30., fallback_obs_per_frame=None):
    from engine.hands.hand_tracker import HandCaptureEngine, HandObservation, solve_hand_rotations_from_landmarks21
    from engine.triangulation.triangulator import triangulate_sequence

    frame_count = max([len(a) for source in (left_hand21_per_cam, right_hand21_per_cam) if source
                       for a in source.values()] + [len(fallback_obs_per_frame or [])])
    if frame_count == 0:
        return None
    frames = [dict(fallback_obs_per_frame[t]) if fallback_obs_per_frame and t < len(fallback_obs_per_frame) else {}
              for t in range(frame_count)]
    extras = {}
    for side, source in (("left", left_hand21_per_cam), ("right", right_hand21_per_cam)):
        world = np.full((frame_count, 21, 3), np.nan, np.float32)
        measured_conf = np.zeros((frame_count, 21), np.float32)
        if source and len(source) >= 2:
            usable = {u: a[:frame_count] for u, a in source.items() if u in cameras}
            if len(usable) >= 2:
                world, measured_conf, errors, _ = triangulate_sequence(
                    cameras, usable, min_confidence=.25,
                    max_reproj_error_px=max(10., min(c.intrinsics.image_width for c in cameras.values()) * .006),
                    refine_nonlinear=False)
                measured_conf[~np.isfinite(world).all(axis=-1)] = 0
                # Implausible finger segments invalidate their endpoints, never become zero positions.
                for start in (1, 5, 9, 13, 17):
                    for k in range(start, start + 3):
                        lengths = np.linalg.norm(world[:, k + 1] - world[:, k], axis=1)
                        bad = (lengths < .007) | (lengths > .095)
                        measured_conf[bad, k + 1] = 0
        template = calibrate_hand_template(world, measured_conf, side)
        palms = np.full((frame_count, 4), np.nan, np.float32)
        provenance = np.zeros((frame_count, 21), np.uint8)
        previous = None
        for t in range(frame_count):
            palm = palm_frame(world[t], side, measured_conf[t])
            fallback = frames[t].get(side)
            if fallback is not None:
                provenance[t] = 4 if fallback.source == "interpolated" else 2
            if palm is None:
                continue
            palms[t] = Rotation.from_matrix(palm).as_quat()
            local = (world[t] - world[t, 0]) @ palm
            initial = fallback.local_rotations_xyzw if fallback is not None else solve_hand_rotations_from_landmarks21(world[t], side)
            fitted = fit_hand_frame(local, measured_conf[t], side, template, initial, previous)
            # Use reliable multi-view evidence; detailed image crops remain an articulation prior.
            if fallback is not None:
                coverage = float(np.mean(measured_conf[t] >= .25))
                weight = float(np.clip((coverage - .35) / .5, 0, 1))
                for j in range(1, 16):
                    fitted[j] = quat_slerp(initial[j], fitted[j], weight)
            provenance[t, measured_conf[t] >= .25] = 3
            frames[t][side] = HandObservation(side, world[t], float(measured_conf[t].mean()), fitted,
                                             measured_conf[t].copy(), source="multiview")
            previous = fitted
        extras[side + "_palm_world_xyzw"] = palms
        extras[side + "_landmarks_world"] = world
        extras[side + "_provenance"] = provenance
        extras[side + "_template"] = template
    if not any(frames):
        return None
    track = HandCaptureEngine().build_track(frames, fps, smooth=True)
    return replace(track, **extras)


def apply_observed_wrist_orientation(body, hands: HandMotionTrack | None, actor, fps):
    """Fit palm swing and continuous forearm roll with visibility and rate constraints."""
    if hands is None:
        return body
    from engine.skeleton.rig_definition import BONE_INDEX, compute_forward_kinematics_sequence
    from engine.motion_cleanup.rotation_filter import smooth_rotations
    from scipy.signal import butter, sosfiltfilt

    local = body.local_rotations_xyzw.copy()
    observable = body.rotation_observable.copy() if body.rotation_observable is not None else np.zeros(local.shape[:2], bool)
    capture_to_world = np.asarray(body.cleanup_config.get('world_from_capture', np.eye(4)))[:3, :3]
    count = len(local)
    for side, sign in (('left', -1.), ('right', 1.)):
        palms = getattr(hands, side + '_palm_world_xyzw')
        if palms is None:
            continue
        wrist, forearm = BONE_INDEX[side + '_hand'], BONE_INDEX[side + '_forearm']
        rest = np.column_stack([[0, -sign, 0], [sign, 0, 0], [0, 0, 1]])
        confidence = getattr(hands, side + '_confidence')
        observed = np.isfinite(palms).all(axis=1)
        if confidence is not None:
            observed &= np.min(confidence[:, [0, 5, 9, 17]], axis=1) >= .35
        target = np.tile([0., 0., 0., 1.], (count, 1))
        for t in np.flatnonzero(observed):
            desired = capture_to_world @ Rotation.from_quat(palms[t]).as_matrix() @ rest.T
            parent = body.global_matrices[t, forearm, :3, :3]
            target[t] = Rotation.from_matrix(parent.T @ desired).as_quat()
        valid = np.flatnonzero(observed)
        if not len(valid):
            continue
        # Missing poses carry a decaying prior and remain explicitly unobserved.
        last = np.array([0., 0., 0., 1.])
        missing = 0
        for t in range(count):
            if observed[t]:
                last, missing = target[t].copy(), 0
            else:
                missing += 1
                alpha = 1 - np.exp(-max(0., missing / fps - .25) / .7)
                target[t] = quat_slerp(last, np.array([0., 0., 0., 1.]), alpha)
        for a, b in zip(valid[:-1], valid[1:]):
            if b - a <= max(2, round(fps * .25)):
                for t in range(a + 1, b):
                    target[t] = quat_slerp(target[a], target[b], (t - a) / (b - a))
        target = smooth_rotations(enforce_quaternion_continuity(target)[:, None], fps, 2.8)[:, 0]
        # Swing/twist decomposition avoids the discontinuity of clipping a rotation vector at pi.
        twist = target.copy()
        twist[:, 1:3] = 0
        norms = np.linalg.norm(twist, axis=1)
        bad = norms < 1e-5
        twist[bad] = [0, 0, 0, 1]
        twist /= np.maximum(np.linalg.norm(twist, axis=1, keepdims=True), 1e-8)
        angle = np.unwrap(2 * np.arctan2(twist[:, 0], twist[:, 3]))
        angle -= 2 * np.pi * np.round(np.median(angle[observed]) / (2 * np.pi))
        if count >= 18 and fps > 8:
            angle = sosfiltfilt(butter(2, min(2.8, fps * .35), fs=fps, output='sos'), angle)
        angle = np.clip(angle, -np.radians(160), np.radians(160))
        swing = (Rotation.from_quat(target) * Rotation.from_quat(twist).inv()).as_rotvec()
        length = np.linalg.norm(swing, axis=1)
        swing *= np.minimum(1., np.radians(80) / np.maximum(length, 1e-8))[:, None]
        desired_local = Rotation.from_rotvec(swing) * Rotation.from_rotvec(np.c_[angle, np.zeros((count, 2))])
        roll = np.clip(angle * .7, -np.radians(110), np.radians(110))
        # Limit temporal changes of the correction, preserving measured arm endpoints.
        for indices in (range(1, count), range(count - 2, -1, -1)):
            previous_delta = -1 if indices.start == 1 else 1
            for t in indices:
                roll[t] = np.clip(roll[t], roll[t + previous_delta] - np.radians(120) / fps,
                                  roll[t + previous_delta] + np.radians(120) / fps)
        parent_delta = Rotation.from_rotvec(np.c_[roll, np.zeros((count, 2))])
        wrist_quaternions = (parent_delta.inv() * desired_local).as_quat()
        wrist_quaternions = enforce_quaternion_continuity(wrist_quaternions)
        for indices in (range(1, count), range(count - 2, -1, -1)):
            previous_delta = -1 if indices.start == 1 else 1
            for t in indices:
                previous = wrist_quaternions[t + previous_delta]
                dot = np.clip(abs(np.dot(previous, wrist_quaternions[t])), 0, 1)
                angle_step = 2 * np.arccos(dot)
                if angle_step > np.radians(360) / fps:
                    wrist_quaternions[t] = quat_slerp(previous, wrist_quaternions[t], (np.radians(360) / fps) / angle_step)
        local[:, forearm] = (Rotation.from_quat(local[:, forearm]) * parent_delta).as_quat()
        local[:, wrist] = wrist_quaternions
        observable[:, wrist] = observed
    local = enforce_quaternion_continuity(local)
    positions, matrices = compute_forward_kinematics_sequence(body.root_translation, local, actor.rest_offsets_m)
    return replace(body, local_rotations_xyzw=local, joint_positions=positions, global_matrices=matrices,
                   rotation_observable=observable)
