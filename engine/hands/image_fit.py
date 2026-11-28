"""Fit the existing anatomical hand to native image observations.

This is a reconstruction pass, not another detector. It never changes native
pixels or increases their confidence. Weak perspective is local to one hand;
the body's calibrated/world trajectory is not part of this fit.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from engine.hands.hand_fusion import _parameters_to_quaternions
from engine.hands.kinematics import calibrate_hand_template, default_hand_template


def _parameters(rotations):
    angles = Rotation.from_quat(np.asarray(rotations)[1:]).as_euler("xyz")
    return np.r_[-angles[:, 0], angles[::3, 2]]


def _points(parameters, template):
    """Batch the five fingers, retaining fixed segment lengths."""
    matrices = Rotation.from_quat(_parameters_to_quaternions(parameters)[1:]).as_matrix().reshape(5, 3, 3, 3)
    chains = np.asarray(template)[1:].reshape(5, 4, 3)
    output = chains.copy()
    basis = np.tile(np.eye(3), (5, 1, 1))
    for segment in range(3):
        basis = basis @ matrices[:, segment]
        output[:, segment + 1] = output[:, segment] + np.einsum(
            "fij,fj->fi", basis, chains[:, segment + 1] - chains[:, segment])
    return np.vstack([np.zeros(3), output.reshape(20, 3)])


def fit_hand_image(observation, template=None, previous=None):
    """Return a bounded refinement and its diagnostic, or the unchanged prior.

    The template is constant across the take. Palm scale/offset are nuisance
    parameters fitted only to the wrist/knuckles, never to the finger tips.
    """
    report = {"accepted": False, "reason": "insufficient image evidence"}
    uv = observation.landmarks_2d
    if uv is None or observation.source == "interpolated":
        return observation, report
    uv = np.asarray(uv, float)
    confidence = np.asarray(observation.landmark_confidence if observation.landmark_confidence is not None
                            else np.full(21, observation.confidence), float)
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(confidence) & (confidence >= .25)
    anchors = np.array([0, 5, 9, 13, 17])
    palm_valid = anchors[valid[anchors]]
    if len(palm_valid) < 4 or valid.sum() < 12 or not valid[[5, 17]].all():
        return observation, report
    span = float(np.linalg.norm(uv[17] - uv[5]))
    if span < 12 or not np.isfinite(observation.local_rotations_xyzw).all():
        return observation, report
    if template is None:
        template = default_hand_template(observation.handedness)
    template = np.asarray(template, float)
    palm = Rotation.from_quat(observation.local_rotations_xyzw[0]).as_matrix()
    projection = np.array([[-1. if observation.source.startswith("rtmw3d") else 1., 0., 0.], [0., 0., -1.]])
    initial = _parameters(observation.local_rotations_xyzw)
    lower = np.r_[np.tile([-.12, 0., 0.], 5), [-.85, -.28, -.24, -.24, -.30]]
    upper = np.r_[np.tile([1.52, 1.80, 1.35], 5), [.85, .28, .24, .24, .30]]
    initial = np.clip(initial, lower + 1e-7, upper - 1e-7)
    previous_parameters = initial if previous is None else np.clip(_parameters(previous.local_rotations_xyzw), lower, upper)
    weights = np.sqrt(np.clip(confidence[valid], 0, 1))[:, None]
    palm_weights = np.clip(confidence[palm_valid], 0, 1)
    observed_palm = uv[palm_valid]
    target_center = np.average(observed_palm, axis=0, weights=palm_weights)

    def predict(parameters):
        rotation = Rotation.from_rotvec(parameters[20:]).as_matrix() @ palm
        projected = _points(parameters[:20], template) @ rotation.T @ projection.T
        base = projected[palm_valid]
        center = np.average(base, axis=0, weights=palm_weights)
        centered = base - center
        denominator = float(np.sum(centered * centered * palm_weights[:, None]))
        scale = float(np.sum(centered * (observed_palm - target_center) * palm_weights[:, None]) / max(denominator, 1e-10))
        return (projected - center) * scale + target_center, rotation, scale

    seed = np.r_[initial, np.zeros(3)]
    before, _, scale = predict(seed)
    if scale <= 0 or not np.isfinite(before).all():
        report["reason"] = "inconsistent palm projection"
        return observation, report
    before_error = float(np.sqrt(np.mean((before[valid] - uv[valid]) ** 2)))

    def residual(parameters):
        predicted, _, _ = predict(parameters)
        curls = parameters[:15].reshape(5, 3)
        return np.r_[((predicted[valid] - uv[valid]) * weights / span).ravel(),
                     (parameters[:20] - initial) * .035,
                     (parameters[:20] - previous_parameters) * .018,
                     (curls[1:, 2] - .65 * curls[1:, 1]) * .025,
                     parameters[20:] * .15]

    result = least_squares(residual, seed, bounds=(np.r_[lower, [-.3] * 3], np.r_[upper, [.3] * 3]),
                           loss="soft_l1", f_scale=.08, diff_step=.002, max_nfev=12, ftol=.005, xtol=.002)
    after, rotation, scale = predict(result.x)
    error = float(np.sqrt(np.mean((after[valid] - uv[valid]) ** 2)))
    palm_error = float(np.sqrt(np.mean((after[palm_valid] - uv[palm_valid]) ** 2)))
    report.update(before_px=before_error, after_px=error, palm_px=palm_error, span_px=span)
    if (scale <= 0 or not np.isfinite(error) or error >= before_error * .98 or error > span * .20
            or palm_error > span * .15 or np.linalg.norm(result.x[20:]) > .35):
        report["reason"] = "fit did not improve supported image agreement"
        return observation, report
    q = _parameters_to_quaternions(result.x[:20]).astype(np.float32)
    q[0] = Rotation.from_matrix(rotation).as_quat()
    # Keep the original hemisphere so a valid refinement cannot insert a sign flip.
    q[np.sum(q * observation.local_rotations_xyzw, axis=1) < 0] *= -1
    points = _points(result.x[:20], template) @ rotation.T + observation.landmarks_3d[0]
    report.update(accepted=True, reason="bounded fit to native image points")
    return replace(observation, local_rotations_xyzw=q, landmarks_3d=points.astype(np.float32)), report


def refine_hand_observations(frames, times, cancel=None):
    """Refine native samples once, resetting temporal priors across long gaps."""
    if len(frames) != len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Hand observations require matching, ordered native timestamps.")
    output = [dict(frame) for frame in frames]
    reports = {side: [] for side in ("left", "right")}
    for side in reports:
        observations = [frame.get(side) for frame in frames]
        available = [obs for obs in observations if obs is not None]
        if not available:
            continue
        template = calibrate_hand_template(np.asarray([o.landmarks_3d for o in available]),
            np.asarray([o.landmark_confidence if o.landmark_confidence is not None
                        else np.full(21, o.confidence) for o in available]), side)
        previous, previous_time = None, -np.inf
        for i, (stamp, obs) in enumerate(zip(times, observations)):
            if cancel is not None and cancel.is_set():
                raise RuntimeError("Cancelled by user.")
            if obs is None:
                continue
            prior = previous if stamp - previous_time <= .20 else None
            fitted, report = fit_hand_image(obs, template, prior)
            output[i][side] = fitted
            reports[side].append({"pts_s": float(stamp), **report})
            if obs.source != "interpolated":
                previous, previous_time = fitted, stamp
    return output, reports


def summarize_hand_fits(reports):
    """Report paired image errors only on the samples actually accepted."""
    summary = {}
    for side, rows in reports.items():
        accepted = [row for row in rows if row["accepted"]]
        summary[side] = {"samples":len(rows), "accepted":len(accepted),
            "evaluated":sum("before_px" in row for row in rows),
            "accepted_native_pts_s":[row["pts_s"] for row in accepted],
            "before_median_px":float(np.median([row["before_px"] for row in accepted])) if accepted else None,
            "after_median_px":float(np.median([row["after_px"] for row in accepted])) if accepted else None}
    return summary
