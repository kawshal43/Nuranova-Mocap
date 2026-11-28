"""Consensus audio alignment: fit clock drift from many independently matched windows."""

from __future__ import annotations

import numpy as np

from desktop.synchronization.sync_engine import estimate_audio_sync_offset_s


def estimate_onset_alignment(ref, target, sample_rate, max_lag_s):
    """Independent spectral-envelope check when microphone waveforms differ.

    A repeated beat is explicitly ambiguous. Accepted offsets need agreement
    through several sections of the shared recording, not just one loud event.
    Returned lag has the waveform estimator's sign convention.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import correlate, find_peaks, stft

    report = {"usable":False, "ambiguous":False, "score":0., "witnesses":0}
    a, b = np.asarray(ref, float).reshape(-1), np.asarray(target, float).reshape(-1)
    if sample_rate < 1000 or min(len(a), len(b)) < 4*sample_rate or not np.isfinite(a).all() or not np.isfinite(b).all():
        return report
    window, hop = max(64, round(sample_rate*.016)), max(1, round(sample_rate*.004))

    def envelope(samples):
        frequencies, _, spectrum = stft(samples, fs=sample_rate, nperseg=window,
            noverlap=window-hop, boundary=None, padded=False)
        bands = []
        for low, high in ((100, 800), (800, 2400), (2400, sample_rate*.49)):
            selected = (frequencies >= low) & (frequencies < high)
            if not selected.any():
                continue
            energy = np.sqrt(np.mean(abs(spectrum[selected])**2, axis=0))
            floor = max(float(np.percentile(energy, 25)), float(np.max(energy))*1e-4, 1e-12)
            compressed = np.log1p(energy/floor)
            changes = np.maximum(np.diff(compressed, prepend=compressed[0]), 0)
            changes = gaussian_filter1d(changes, 1.)
            scale = max(float(np.percentile(changes, 95)), 1e-6)
            bands.append(np.minimum(changes/scale, 6.))
        if not bands:
            return np.zeros(1), 0
        combined = np.mean(bands, axis=0)
        loudness = np.sqrt(np.sum(abs(spectrum)**2, axis=0))
        peaks, _ = find_peaks(loudness, prominence=max(float(loudness.max())*.15, 1e-12),
                              distance=max(1, round(.12*sample_rate/hop)))
        return combined - combined.mean(), len(peaks)

    (ea, events_a), (eb, events_b) = envelope(a), envelope(b)
    report["salient_events"] = [events_a, events_b]
    if min(np.linalg.norm(ea), np.linalg.norm(eb)) < .01:
        return report
    correlation = correlate(eb, ea, mode="full", method="fft")
    lags = np.arange(-len(ea)+1, len(eb))
    start, end = np.maximum(0, -lags), np.minimum(len(ea), len(eb)-lags)
    end = np.maximum(start, end)
    pa, pb = np.r_[0, np.cumsum(ea*ea)], np.r_[0, np.cumsum(eb*eb)]
    energy = (pa[end]-pa[start])*(pb[end+lags]-pb[start+lags])
    scores = correlation / np.sqrt(np.maximum(energy, 1e-12))
    usable = (abs(lags)*hop/sample_rate <= max_lag_s) & (end-start >= min(8*sample_rate/hop, .6*min(len(ea), len(eb))))
    scores = np.where(usable, scores, -1.)
    best = int(np.argmax(scores))
    score, lag = float(scores[best]), int(lags[best])
    alternatives = usable & (abs(lags-lag)*hop/sample_rate > .15)
    runner = max(0., float(np.max(scores[alternatives]))) if alternatives.any() else 0.
    ratio = score/max(runner, .001)
    report.update(score=score, peak_ratio=ratio, lag_s=lag*hop/sample_rate)
    if score < .45:
        return report
    if ratio < 1.12:
        # A lone clap cannot establish periodic ambiguity. Sparse/silent
        # overlap can give several similar envelope peaks; retain the precise
        # waveform result in that case.
        report["ambiguous"] = min(events_a, events_b) >= 4
        return report
    boundaries = np.linspace(start[best], end[best], 5).astype(int)
    witnesses = 0
    for low, high in zip(boundaries[:-1], boundaries[1:]):
        x, y = ea[low:high].copy(), eb[low+lag:high+lag].copy()
        x -= x.mean()
        y -= y.mean()
        if min(float(x.std()), float(y.std())) < .03*min(float(ea.std()),float(eb.std())):
            continue
        norm = float(np.linalg.norm(x)*np.linalg.norm(y))
        if norm > 1e-8 and float(x@y)/norm >= .4:
            witnesses += 1
    report.update(witnesses=witnesses, usable=witnesses >= 3)
    return report


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
    onset = estimate_onset_alignment(ref, target, sample_rate, max_lag_s)
    use_onsets = onset["usable"] and (snr < 10 or abs(lag-onset["lag_s"]) > .035)
    if use_onsets:
        lag, snr = onset["lag_s"], 40.*onset["score"]
    elif onset["ambiguous"]:
        snr = 0.
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
            if use_onsets:
                local = estimate_onset_alignment(ref[ir0:ir1], target[it0:it1], sample_rate, .45)
                if not local["usable"]:
                    continue
                local_lag, strength = local["lag_s"], 40.*local["score"]
            else:
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
        diagnostics.update(feature_method="spectral_onsets" if use_onsets else "waveform",
                           onset_check=onset)
    return float(offset), float(snr), float(slope), float(pivot)
