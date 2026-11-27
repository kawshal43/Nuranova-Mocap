"""Consensus audio alignment: fit clock drift from many independently matched windows."""

from __future__ import annotations

import numpy as np

from desktop.synchronization.sync_engine import estimate_audio_sync_offset_s


def fit_clock_consensus(times, offsets, strengths, initial_offset, tolerance_s=0.008):
    """RANSAC rejects echoes/unrelated transients before a weighted affine clock fit."""
    x = np.asarray(times, dtype=float)
    y = np.asarray(offsets, dtype=float)
    weights = np.clip(np.asarray(strengths, dtype=float), 1, 40)
    good = np.isfinite(x) & np.isfinite(y) & (weights >= 10) & (np.abs(y - initial_offset) < 0.2)
    x, y, weights = x[good], y[good], weights[good]
    if len(x) < 3:
        return initial_offset, 1.0, 0.0, {"inliers": 0, "residual_ms": None}
    pivot = float(np.median(x))
    xc = x - pivot
    models = [(float(np.median(y)), 0.0)]
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if abs(x[j] - x[i]) >= 10:
                slope = (y[j] - y[i]) / (x[j] - x[i])
                if abs(slope) <= 0.002:
                    models.append((float(y[i] - slope * xc[i]), float(slope)))
    best_mask = None
    best_score = -1.0
    for intercept, slope in models:
        mask = np.abs(y - intercept - slope * xc) <= tolerance_s
        # Prefer evidence distributed through the recording over a cluster near one clap.
        span = float(np.ptp(x[mask])) if np.any(mask) else 0
        score = float(np.sum(weights[mask])) * (0.6 + 0.4 * span / max(float(np.ptp(x)), 1))
        if mask.sum() >= 3 and score > best_score:
            best_score, best_mask = score, mask
    if best_mask is None:
        return initial_offset, 1.0, pivot, {"inliers": 0, "residual_ms": None}
    mask = best_mask
    span = float(np.ptp(x[mask]))
    use_drift = mask.sum() >= 4 and span >= max(20, float(np.ptp(x)) * 0.5)
    if use_drift:
        design = np.column_stack([np.ones(mask.sum()), xc[mask]])
        coeff = np.linalg.lstsq(
            design * np.sqrt(weights[mask, None]), y[mask] * np.sqrt(weights[mask]), rcond=None
        )[0]
        offset, drift = float(coeff[0]), float(coeff[1])
        if abs(drift) > 0.002:
            drift = 0.0
    else:
        offset, drift = float(np.median(y[mask])), 0.0
    constant_offset = float(np.average(y[mask], weights=weights[mask]))
    constant_error = float(np.average((y[mask] - constant_offset) ** 2, weights=weights[mask]))
    drift_error = float(
        np.average((y[mask] - offset - drift * xc[mask]) ** 2, weights=weights[mask])
    )
    if (
        mask.sum() < max(4, int(np.ceil(len(x) * 0.6)))
        or abs(drift) * span < 0.012
        or drift_error > constant_error * 0.75
    ):
        offset, drift, use_drift = constant_offset, 0.0, False
    residual = y[mask] - offset - drift * xc[mask]
    return (
        offset,
        1 + drift,
        pivot,
        {
            "inliers": int(mask.sum()),
            "windows": int(len(x)),
            "residual_ms": float(np.sqrt(np.average(residual**2, weights=weights[mask])) * 1000),
            "span_s": span,
            "drift_verified": bool(use_drift),
        },
    )


def estimate_audio_clock(ref, target, sample_rate, max_lag_s, diagnostics=None):
    lag, snr = estimate_audio_sync_offset_s(
        ref, target, sample_rate_hz=sample_rate, max_lag_s=max_lag_s
    )
    initial = -lag
    ref_duration, target_duration = len(ref) / sample_rate, len(target) / sample_rate
    start, end = max(0.0, -initial), min(target_duration, ref_duration - initial)
    overlap = end - start
    details = {"inliers": 0, "windows": 0, "residual_ms": None, "drift_verified": False}
    offset, slope, pivot = initial, 1.0, max(0, (start + end) / 2)
    if snr >= 10 and overlap >= 20:
        half = min(6.0, overlap / 5)
        centers = np.linspace(start + half + 0.4, end - half - 0.4, 15)
        times, offsets, strengths = [], [], []
        matches = []
        for center in centers:
            t0, t1 = center - half, center + half
            r0, r1 = max(0, t0 + initial - 0.2), min(ref_duration, t1 + initial + 0.2)
            ir0, ir1 = int(round(r0 * sample_rate)), int(round(r1 * sample_rate))
            it0, it1 = int(round(t0 * sample_rate)), int(round(t1 * sample_rate))
            if min(ir1 - ir0, it1 - it0) < 3 * sample_rate:
                continue
            local_lag, strength = estimate_audio_sync_offset_s(
                ref[ir0:ir1], target[it0:it1], sample_rate_hz=sample_rate, max_lag_s=0.45
            )
            local_offset = (ir0 - it0) / sample_rate - local_lag
            times.append(center)
            offsets.append(local_offset)
            strengths.append(strength)
            matches.append(
                {"time_s": float(center), "offset_s": float(local_offset), "snr": float(strength)}
            )
        offset, slope, pivot, details = fit_clock_consensus(times, offsets, strengths, initial)
        details["matches"] = matches
    if diagnostics is not None:
        diagnostics.update(details)
    return float(offset), float(snr), float(slope), float(pivot)
