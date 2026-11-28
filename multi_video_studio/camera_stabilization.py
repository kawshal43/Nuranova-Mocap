"""Reference-background alignment for estimated monocular movement.

Similarities compensate image translation, zoom and roll. They do not recover a
metric camera trajectory or resolve parallax, perspective changes or hidden limbs.
"""
from __future__ import annotations

import cv2
import numpy as np

from engine.video.video_io import iter_video_frames

ALGORITHM = "masked_reference_similarity_v2"


class BackgroundReference:
    """Match each image directly to one reference, avoiding cumulative flow drift."""
    def __init__(self):
        self.detector = cv2.SIFT_create(nfeatures=1400, contrastThreshold=.025)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.reference = None

    def estimate(self, image, body):
        height, width = image.shape[:2]
        ratio = min(1., 640 / max(height, width))
        size = round(width*ratio), round(height*ratio)
        gray = cv2.cvtColor(cv2.resize(image, size), cv2.COLOR_BGR2GRAY)
        mask = np.full(gray.shape, 255, np.uint8)
        body = np.asarray(body)
        visible = body[(body[:, 2] > .2) & np.isfinite(body).all(axis=1), :2]*ratio
        if len(visible) < 4:
            return None, {"reason":"Performer mask unavailable"}
        low, high = visible.min(axis=0)-24, visible.max(axis=0)+24
        cv2.rectangle(mask, tuple(np.maximum(0, low).astype(int)), tuple(high.astype(int)), 0, -1)
        keys, descriptors = self.detector.detectAndCompute(gray, mask)
        if descriptors is None or len(keys) < 10:
            return None, {"reason":"Insufficient background texture"}
        if self.reference is None:
            self.reference = keys, descriptors, ratio, size
        ref_keys, ref_descriptors, ref_ratio, ref_size = self.reference
        if size != ref_size:
            return None, {"reason":"Video resolution changed"}
        matches = self.matcher.knnMatch(descriptors, ref_descriptors, k=2)
        pairs = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < .7*pair[1].distance]
        # A repeated texture must not count the same reference feature many times.
        pairs = sorted(pairs, key=lambda pair: pair.distance)
        unique = {}
        for pair in pairs:
            unique.setdefault(pair.trainIdx, pair)
        pairs = list(unique.values())
        if len(pairs) < 10:
            return None, {"reason":"Insufficient distinct background matches"}
        current = np.float32([keys[p.queryIdx].pt for p in pairs])
        reference = np.float32([ref_keys[p.trainIdx].pt for p in pairs])
        affine, inliers = cv2.estimateAffinePartial2D(current, reference, method=cv2.RANSAC,
            ransacReprojThreshold=1.8, maxIters=3000, confidence=.995)
        if affine is None or inliers is None:
            return None, {"reason":"Background transform could not be estimated"}
        good = inliers.ravel().astype(bool)
        error = np.linalg.norm(current@affine[:, :2].T+affine[:, 2]-reference, axis=-1)
        span = np.ptp(reference[good], axis=0) if good.any() else np.zeros(2)
        scale = float(np.linalg.norm(affine[:, 0]))
        report = {"matches":len(pairs), "inliers":int(good.sum()), "inlier_ratio":float(good.mean()),
                  "median_error_preview_px":float(np.median(error[good])) if good.any() else None}
        if (good.sum() < 10 or good.mean() < .6 or np.median(error[good]) > 1.
                or span[0] < ref_size[0]*.2 or span[1] < ref_size[1]*.1 or not .5 <= scale <= 2.):
            return None, {**report, "reason":"Background support is too weak or localized"}
        matrix = np.eye(3)
        matrix[:2] = affine
        matrix = np.diag([1/ref_ratio, 1/ref_ratio, 1.])@matrix@np.diag([ratio, ratio, 1.])
        return matrix, report


def interpolate_alignment(sample_times, matrices, times, max_gap_s=.65):
    """Interpolate similarity parameters, never shear matrices or bridge long gaps."""
    times = np.asarray(times, float)
    stamps = np.asarray(sample_times, float)
    output = np.tile(np.eye(3), (len(times), 1, 1))
    supported = np.zeros(len(times), bool)
    if len(stamps) < 2:
        return output, supported
    matrices = np.asarray(matrices, float)
    scale = np.linalg.norm(matrices[:, :2, 0], axis=1)
    angle = np.unwrap(np.arctan2(matrices[:, 1, 0], matrices[:, 0, 0]))
    scales = np.exp(np.interp(times, stamps, np.log(scale)))
    angles = np.interp(times, stamps, angle)
    output[:, 0, 0] = output[:, 1, 1] = scales*np.cos(angles)
    output[:, 1, 0] = scales*np.sin(angles)
    output[:, 0, 1] = -output[:, 1, 0]
    output[:, :2, 2] = np.stack([np.interp(times, stamps, matrices[:, axis, 2]) for axis in (0, 1)], axis=1)
    right = np.clip(np.searchsorted(stamps, times), 0, len(stamps)-1)
    left = np.maximum(0, right-1)
    supported = (times >= stamps[0]-1e-6) & (times <= stamps[-1]+1e-6)
    supported &= ((stamps[right]-stamps[left] <= max_gap_s)
                  | np.isclose(times, stamps[right], atol=1e-6, rtol=0))
    return output, supported


def estimate_camera_alignment(path, times, body, cancel=None, progress=None):
    """Track native source times. A failed camera model never silently enables travel."""
    times = np.asarray(times, float)
    reference = BackgroundReference()
    sample_times, matrices, diagnostics = [], [], []
    last_sample = -np.inf
    for _, stamp, image in iter_video_frames(path, start_pts_s=float(times[0]), end_pts_s=float(times[-1])+.001):
        if cancel is not None and cancel.is_set():
            raise RuntimeError("Cancelled by user.")
        if stamp-last_sample < .09 and stamp < times[-1]-.001:
            continue
        last_sample = stamp
        index = int(np.argmin(abs(times-stamp)))
        if abs(times[index]-stamp) > .001:
            continue
        matrix, report = reference.estimate(image, body[index])
        diagnostics.append({"pts_s":float(stamp), "supported":matrix is not None, **report})
        if matrix is not None:
            sample_times.append(stamp)
            matrices.append(matrix)
        if progress and len(diagnostics) % 40 == 0:
            progress(float(index/max(1, len(times)-1)), "Matching stationary background to retain performer movement…")
    transforms, supported = interpolate_alignment(sample_times, matrices, times)
    observed = (np.asarray(body)[:, [5, 6, 11, 12], 2] > .3).sum(axis=1) >= 3
    accepted, uncertain = assess_alignment_support(times, supported)
    accepted = accepted and len(sample_times) >= 6 and observed.sum() >= 6
    return {"algorithm":ALGORITHM, "accepted":bool(accepted), "native_pts_s":times.tolist(),
            "native_to_reference":transforms.tolist() if accepted else [],
            "supported_frames":int(supported.sum()), "total_frames":len(times),
            "supported_mask":supported.tolist(), "uncertain_intervals":uncertain,
            "body_observed_frames":int(observed.sum()), "camera_supported_body_frames":int((observed & supported).sum()),
            "samples":diagnostics, "reference_pts_s":sample_times[0] if sample_times else None,
            "status":"Background-aligned movement is estimated" if accepted else "Insufficient background support; movement fallback retained",
            "metric_camera_trajectory_measured":False}


def assess_alignment_support(times, supported):
    """Keep useful footage around a short fade without claiming observed motion there."""
    times, supported = np.asarray(times, float), np.asarray(supported, bool)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.
    edges = np.diff(np.r_[False, ~supported, False].astype(int))
    uncertain, bounded = [], True
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        duration = float(times[b-1]-times[a]+dt)
        bounded &= duration <= (2. if a == 0 or b == len(times) else .35)
        uncertain.append({"start_s":float(times[a]), "end_s":float(times[b-1]),
                          "reason":"Background alignment unavailable; movement is held or interpolated"})
    return bool(len(times) >= 2 and supported.mean() >= .9 and bounded), uncertain


def alignment_for_times(profile, times):
    if not profile or profile.get("algorithm") != ALGORITHM or not profile.get("accepted"):
        return None
    native = np.asarray(profile.get("native_pts_s", []), float)
    transforms = np.asarray(profile.get("native_to_reference", []), float)
    requested = np.asarray(times, float)
    if len(native) < 2 or transforms.shape != (len(native), 3, 3) or not np.isfinite(transforms).all():
        return None
    index = np.clip(np.searchsorted(native, requested), 0, len(native)-1)
    if np.any(abs(native[index]-requested) > 1e-6):
        return None
    return transforms[index]
