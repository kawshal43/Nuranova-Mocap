"""Multi-camera synchronization engine: NTP-style clock drift regression, Camera2 sensor timeline mapping,
audio cross-correlation sync, flash-event sync, VFR/missing/duplicate detection, and mixed-FPS temporal alignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import correlate

from shared_protocol.protocol import FrameTimestampRecord, TimestampSource


@dataclass
class ClockSyncSample:
    seq: int
    t1_client_ns: int
    t2_server_ns: int
    t3_server_ns: int
    t4_client_ns: int

    @property
    def rtt_ns(self) -> int:
        return (self.t4_client_ns - self.t1_client_ns) - (self.t3_server_ns - self.t2_server_ns)

    @property
    def midpoint_client_ns(self) -> float:
        return 0.5 * (self.t1_client_ns + self.t4_client_ns)

    @property
    def midpoint_server_ns(self) -> float:
        return 0.5 * (self.t2_server_ns + self.t3_server_ns)

    @property
    def offset_ns(self) -> float:
        """Estimated offset such that t_server ~= t_client + offset_ns."""
        return self.midpoint_server_ns - self.midpoint_client_ns


@dataclass
class ClockModel:
    camera_uid: str
    slope: float  # ~1.0 (1 + drift_ppm * 1e-6)
    intercept_ns: float
    t0_client_ns: float
    drift_ppm: float
    residual_rmse_ms: float
    min_rtt_ms: float
    median_rtt_ms: float
    sample_count: int
    meets_5ms_target: bool
    warnings: List[str] = field(default_factory=list)

    def device_to_studio_ns(self, device_ns: float | np.ndarray) -> float | np.ndarray:
        return self.intercept_ns + self.slope * (device_ns - self.t0_client_ns)

    def studio_to_device_ns(self, studio_ns: float | np.ndarray) -> float | np.ndarray:
        return self.t0_client_ns + (studio_ns - self.intercept_ns) / max(self.slope, 1e-9)


def fit_clock_sync_model(
    camera_uid: str,
    samples: Sequence[ClockSyncSample],
    low_rtt_keep_fraction: float = 0.6,
    target_residual_ms: float = 5.0,
) -> ClockModel:
    """Fit drift-compensated linear clock model t_studio = intercept + slope * (t_device - t0).

    Filters high-RTT network jitter outliers before weighted least-squares regression.
    """
    if not samples:
        raise ValueError(f"No clock synchronization samples provided for camera '{camera_uid}'.")

    rtts_ns = np.array([max(s.rtt_ns, 0) for s in samples], dtype=np.float64)
    min_rtt_ms = float(np.min(rtts_ns)) * 1e-6
    med_rtt_ms = float(np.median(rtts_ns)) * 1e-6

    # Keep lowest-RTT quantile to minimize asymmetric Wi-Fi queueing delay bias
    keep_n = max(1, min(len(samples), int(np.ceil(len(samples) * low_rtt_keep_fraction))))
    order = np.argsort(rtts_ns)[:keep_n]
    kept = [samples[i] for i in order]

    t_c = np.array([s.midpoint_client_ns for s in kept], dtype=np.float64)
    t_s = np.array([s.midpoint_server_ns for s in kept], dtype=np.float64)
    t0 = float(np.mean(t_c))
    dt_c = t_c - t0

    warnings: List[str] = []
    span_s = float(np.max(t_c) - np.min(t_c)) * 1e-9

    if len(kept) >= 4 and span_s >= 0.5:
        # Linear regression t_s = slope * (t_c - t0) + intercept
        A = np.column_stack([dt_c, np.ones_like(dt_c)])
        sol, _, _, _ = np.linalg.lstsq(A, t_s, rcond=None)
        slope = float(sol[0])
        intercept = float(sol[1])
        # Guard against unphysical clock drift (> 5000 ppm) caused by short noisy burst
        if abs(slope - 1.0) > 5e-3:
            warnings.append(
                f"Clamped extreme estimated clock drift ({(slope - 1.0)*1e6:.1f} ppm) due to noisy RTT span."
            )
            slope = 1.0
            intercept = float(np.median(t_s - dt_c))
    else:
        slope = 1.0
        intercept = float(np.median(t_s - dt_c))

    preds = intercept + slope * dt_c
    residuals_ms = (t_s - preds) * 1e-6
    # Total timing uncertainty combines regression residual and half-RTT asymmetry bound
    residual_rmse_ms = float(np.sqrt(np.mean(residuals_ms**2) + (0.25 * min_rtt_ms) ** 2))
    drift_ppm = (slope - 1.0) * 1e6
    meets_target = residual_rmse_ms <= target_residual_ms

    if not meets_target:
        warnings.append(
            f"Camera '{camera_uid}' sync residual ({residual_rmse_ms:.2f} ms) exceeds {target_residual_ms:.1f} ms target; "
            "fast motions may exhibit multi-view triangulation lag. Consider audio clap or flash sync refinement."
        )

    return ClockModel(
        camera_uid=camera_uid,
        slope=slope,
        intercept_ns=intercept,
        t0_client_ns=t0,
        drift_ppm=drift_ppm,
        residual_rmse_ms=residual_rmse_ms,
        min_rtt_ms=min_rtt_ms,
        median_rtt_ms=med_rtt_ms,
        sample_count=len(samples),
        meets_5ms_target=meets_target,
        warnings=warnings,
    )


@dataclass
class CameraTimelineReport:
    camera_uid: str
    frame_times_studio_ns: np.ndarray  # (N,) int64 on common studio timeline
    nominal_fps: float
    measured_fps: float
    is_vfr: bool
    missing_frame_indices: List[int]
    duplicate_frame_indices: List[int]
    timestamp_source: TimestampSource
    warnings: List[str] = field(default_factory=list)


def map_frame_timestamps_to_session_timeline(
    camera_uid: str,
    records: Sequence[FrameTimestampRecord],
    clock_model: ClockModel,
    timestamp_source: TimestampSource = TimestampSource.REALTIME,
    additional_offset_ns: int = 0,
    compensate_exposure_midpoint: bool = True,
    compensate_rolling_shutter_center: bool = True,
) -> CameraTimelineReport:
    """Convert raw Android Camera2 sensor/monotonic timestamps to the canonical studio session timeline."""
    if not records:
        return CameraTimelineReport(
            camera_uid=camera_uid,
            frame_times_studio_ns=np.zeros((0,), dtype=np.int64),
            nominal_fps=0.0,
            measured_fps=0.0,
            is_vfr=False,
            missing_frame_indices=[],
            duplicate_frame_indices=[],
            timestamp_source=timestamp_source,
            warnings=["No frame timestamp records found."],
        )

    warnings: List[str] = []
    # Align sensor_timestamp_ns to the device monotonic clock domain used by ClockModel
    sensor_ns = np.array([r.sensor_timestamp_ns for r in records], dtype=np.float64)
    recv_ns = np.array([r.monotonic_receive_ns for r in records], dtype=np.float64)

    # Estimate constant offset between Camera2 sensor clock domain and SystemClock.elapsedRealtimeNanos/uptimeNanos
    if timestamp_source == TimestampSource.UNKNOWN:
        warnings.append(
            f"Camera '{camera_uid}' reports SENSOR_INFO_TIMESTAMP_SOURCE_UNKNOWN; "
            "estimating sensor-to-monotonic domain offset from frame delivery timestamps."
        )
    sensor_to_monotonic_offset = float(np.median(recv_ns - sensor_ns))
    # Only apply domain shift if clocks clearly differ by > 50 ms (different epoch bases)
    if abs(sensor_to_monotonic_offset) > 50_000_000.0:
        device_aligned_ns = sensor_ns + sensor_to_monotonic_offset
    else:
        device_aligned_ns = sensor_ns.copy()

    # Compensate mid-exposure and rolling-shutter mid-frame timing
    for i, r in enumerate(records):
        if compensate_exposure_midpoint and r.exposure_ns:
            device_aligned_ns[i] += 0.5 * float(r.exposure_ns)
        if compensate_rolling_shutter_center and r.rolling_shutter_skew_ns:
            device_aligned_ns[i] += 0.5 * float(r.rolling_shutter_skew_ns)

    studio_ns = np.asarray(
        clock_model.device_to_studio_ns(device_aligned_ns), dtype=np.float64
    ) + float(additional_offset_ns)
    studio_ns_int = np.round(studio_ns).astype(np.int64)

    missing: List[int] = []
    duplicates: List[int] = []
    measured_fps = 0.0
    is_vfr = False

    if len(studio_ns_int) >= 3:
        dts_s = np.diff(studio_ns_int) * 1e-9
        pos = dts_s[dts_s > 1e-5]
        if len(pos) > 0:
            med_dt = float(np.median(pos))
            measured_fps = 1.0 / med_dt
            rel_std = float(np.std(pos) / med_dt)
            is_vfr = rel_std > 0.05
            for idx, dt in enumerate(dts_s):
                if dt > med_dt * 1.65:
                    missing.append(idx + 1)
                elif dt < med_dt * 0.25:
                    duplicates.append(idx + 1)

    if is_vfr:
        warnings.append(
            f"Variable frame rate detected on '{camera_uid}'; offline timestamp resampling enabled."
        )
    if missing:
        warnings.append(f"Detected {len(missing)} dropped frame gaps on '{camera_uid}'.")

    return CameraTimelineReport(
        camera_uid=camera_uid,
        frame_times_studio_ns=studio_ns_int,
        nominal_fps=measured_fps,
        measured_fps=measured_fps,
        is_vfr=is_vfr,
        missing_frame_indices=missing,
        duplicate_frame_indices=duplicates,
        timestamp_source=timestamp_source,
        warnings=warnings,
    )


def estimate_audio_sync_offset_s(
    ref_audio: np.ndarray,
    target_audio: np.ndarray,
    sample_rate_hz: int,
    max_lag_s: float = 2.0,
) -> Tuple[float, float]:
    """Estimate sub-millisecond time offset (seconds) between two audio tracks via FFT cross-correlation.

    Returns:
        (offset_s, peak_correlation_ratio) where positive offset_s means `target_audio` starts later than `ref_audio`.
    """
    a = np.asarray(ref_audio, dtype=np.float64).reshape(-1)
    b = np.asarray(target_audio, dtype=np.float64).reshape(-1)
    a = a - np.mean(a)
    b = b - np.mean(b)
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm < 1e-12:
        return 0.0, 0.0

    # Envelope / energy cross-correlation works cleanly for sharp slate claps
    corr = correlate(b, a, mode="full", method="fft") / norm
    lags = np.arange(-(len(a) - 1), len(b))
    max_lag_samples = int(round(max_lag_s * sample_rate_hz))
    mask = np.abs(lags) <= max_lag_samples
    masked_corr = np.where(mask, np.abs(corr), -1.0)
    best_idx = int(np.argmax(masked_corr))
    best_lag = float(lags[best_idx])

    # Parabolic sub-sample peak interpolation for sub-millisecond refinement
    if 0 < best_idx < len(corr) - 1:
        y0, y1, y2 = abs(corr[best_idx - 1]), abs(corr[best_idx]), abs(corr[best_idx + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            best_lag += 0.5 * (y0 - y2) / denom

    peak_val = float(abs(corr[best_idx]))
    med_val = float(np.median(np.abs(corr[mask]))) + 1e-9
    snr = peak_val / med_val
    return float(best_lag / sample_rate_hz), float(snr)


def detect_flash_sync_frame(luma_means: np.ndarray, z_threshold: float = 5.0) -> Optional[int]:
    """Detect visible flash/LED synchronization pulse frame index from per-frame mean brightness."""
    arr = np.asarray(luma_means, dtype=np.float64)
    if len(arr) < 5:
        return None
    diff = np.diff(arr, prepend=arr[0])
    med = float(np.median(diff))
    mad = float(np.median(np.abs(diff - med))) * 1.4826 + 1e-6
    z = (diff - med) / mad
    best_idx = int(np.argmax(z))
    if z[best_idx] >= z_threshold:
        return best_idx
    return None


def align_observations_to_common_timeline(
    camera_timelines_ns: Dict[str, np.ndarray],
    camera_observations: Dict[str, np.ndarray],  # cam_uid -> (N_cam, K, 3) [u, v, conf]
    target_fps: float = 60.0,
    max_interp_gap_s: float = 0.06,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Temporally resample mixed-FPS / VFR multi-camera 2D observations onto a canonical uniform session timeline.

    Returns:
        (canonical_timestamps_ns (T,), aligned_obs_per_cam {cam_uid: (T, K, 3)})
    """
    valid_cams = [
        u
        for u in camera_timelines_ns
        if u in camera_observations and len(camera_timelines_ns[u]) >= 2
    ]
    if len(valid_cams) < 2:
        raise ValueError("Temporal alignment requires at least 2 cameras with valid timestamps.")
    if not np.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("Target frame rate must be a positive finite number.")
    for uid in valid_cams:
        ts = np.asarray(camera_timelines_ns[uid])
        obs = np.asarray(camera_observations[uid])
        if len(ts) != len(obs) or obs.ndim != 3 or obs.shape[-1] != 3:
            raise ValueError(f"Invalid observation/timestamp shape for {uid}.")
        if not np.all(np.isfinite(ts)) or np.any(np.diff(ts) <= 0):
            raise ValueError(f"Timestamps must increase strictly for {uid}.")

    start_ns = max(int(camera_timelines_ns[u][0]) for u in valid_cams)
    end_ns = min(int(camera_timelines_ns[u][-1]) for u in valid_cams)
    if end_ns <= start_ns:
        raise ValueError("Camera recording timelines do not overlap.")

    dt_ns = int(round(1e9 / max(target_fps, 1.0)))
    canonical_ns = np.arange(start_ns, end_ns + 1, dt_ns, dtype=np.int64)
    T = len(canonical_ns)
    aligned: Dict[str, np.ndarray] = {}

    for uid in valid_cams:
        cam_ts = np.asarray(camera_timelines_ns[uid], dtype=np.float64)
        obs = np.asarray(camera_observations[uid], dtype=np.float64)
        K = obs.shape[1]
        out = np.zeros((T, K, 3), dtype=np.float32)

        idx_right = np.searchsorted(cam_ts, canonical_ns.astype(np.float64), side="left")
        for t in range(T):
            r = int(idx_right[t])
            if r < len(cam_ts) and cam_ts[r] == canonical_ns[t]:
                out[t] = obs[r]
            elif r == 0:
                if abs(cam_ts[0] - canonical_ns[t]) * 1e-9 <= max_interp_gap_s:
                    out[t] = obs[0]
            elif r >= len(cam_ts):
                if abs(cam_ts[-1] - canonical_ns[t]) * 1e-9 <= max_interp_gap_s:
                    out[t] = obs[-1]
            else:
                left_idx = r - 1
                gap_s = (cam_ts[r] - cam_ts[left_idx]) * 1e-9
                if gap_s <= max_interp_gap_s and gap_s > 1e-9:
                    alpha = (canonical_ns[t] - cam_ts[left_idx]) / (cam_ts[r] - cam_ts[left_idx])
                    # Interpolate u,v only when both bracketing observations are confident
                    both_ok = (obs[left_idx, :, 2] > 0.1) & (obs[r, :, 2] > 0.1)
                    out[t, both_ok, :2] = (1.0 - alpha) * obs[left_idx, both_ok, :2] + alpha * obs[
                        r, both_ok, :2
                    ]
                    out[t, both_ok, 2] = np.minimum(obs[left_idx, both_ok, 2], obs[r, both_ok, 2])
        aligned[uid] = out

    return canonical_ns, aligned


def extract_audio_pcm_from_video(
    video_path: str | Path, sample_rate_hz: int = 16000
) -> Optional[np.ndarray]:
    """Extract mono float32 PCM audio from a video file (via ffmpeg if installed) for clap/waveform sync."""
    import shutil
    import subprocess
    from pathlib import Path

    vp = Path(video_path)
    if not vp.exists():
        return None

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            proc = subprocess.run(
                [
                    ffmpeg,
                    "-v",
                    "error",
                    "-i",
                    str(vp),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    str(int(sample_rate_hz)),
                    "-f",
                    "f32le",
                    "-",
                ],
                capture_output=True,
                timeout=20,
            )
            if proc.returncode == 0 and len(proc.stdout) >= sample_rate_hz * 2:
                pcm = np.frombuffer(proc.stdout, dtype=np.float32)
                if np.std(pcm) > 1e-5:
                    return pcm
        except Exception:
            pass
    return None


def _build_vertical_motion_curve(
    obs_2d: np.ndarray,
    pts_s: Sequence[float],
    grid_rate_hz: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a view-invariant vertical velocity curve from 2D body keypoints (wrists, shoulders, hips, knees, ankles)."""
    n = min(len(pts_s), obs_2d.shape[0])
    if n < 6:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    t_arr = np.asarray(pts_s[:n], dtype=np.float64)
    if t_arr[-1] <= t_arr[0] + 0.1:
        t_arr = np.arange(n, dtype=np.float64) / 30.0

    # Keypoints whose vertical velocity is consistent across all camera angles around a room
    k_ids = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    y_vals = np.zeros(n, dtype=np.float64)
    valid = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        c = obs_2d[i, k_ids, 2].astype(np.float64)
        mask = c > 0.25
        if np.any(mask):
            w = c[mask]
            y_vals[i] = float(np.sum(obs_2d[i, k_ids, 1][mask] * w) / np.sum(w))
            valid[i] = True

    if int(np.sum(valid)) < 6:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    # Interpolate onto uniform 60 Hz grid and differentiate for view-invariant vertical velocity
    t_valid = t_arr[valid]
    y_valid = y_vals[valid]
    t_grid = np.arange(t_valid[0], t_valid[-1], 1.0 / grid_rate_hz, dtype=np.float64)
    if len(t_grid) < 10:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    y_grid = np.interp(t_grid, t_valid, y_valid)
    vel = np.gradient(y_grid)
    # Normalize by scale (std) so cameras at different distances/resolutions match cleanly
    std = float(np.std(vel))
    if std < 1e-6:
        return t_grid, np.zeros_like(vel)
    return t_grid, (vel - float(np.mean(vel))) / std


def estimate_motion_sync_offset_s(
    ref_obs_2d: np.ndarray,
    ref_pts_s: Sequence[float],
    tgt_obs_2d: np.ndarray,
    tgt_pts_s: Sequence[float],
    max_lag_s: float = 4.0,
) -> Tuple[float, float]:
    """Estimate time offset (seconds) between two camera views using view-invariant vertical body motion.

    Works even on completely silent MP4 videos recorded from different angles!
    Returns (offset_s, normalized_corr_peak) where positive offset_s means `tgt` started AFTER `ref`.
    """
    grid_hz = 60.0
    t_ref, v_ref = _build_vertical_motion_curve(ref_obs_2d, ref_pts_s, grid_rate_hz=grid_hz)
    t_tgt, v_tgt = _build_vertical_motion_curve(tgt_obs_2d, tgt_pts_s, grid_rate_hz=grid_hz)
    if len(v_ref) < 12 or len(v_tgt) < 12:
        return 0.0, 0.0
    n_ref = float(np.linalg.norm(v_ref))
    n_tgt = float(np.linalg.norm(v_tgt))
    if n_ref < 1e-6 or n_tgt < 1e-6:
        return 0.0, 0.0

    # Cross-correlate v_ref(t) with v_tgt(t - lag): lag > 0 means event happens EARLIER in tgt -> tgt started LATER
    corr = correlate(v_ref, v_tgt, mode="full", method="fft") / (n_ref * n_tgt)
    lags = np.arange(-(len(v_tgt) - 1), len(v_ref))
    max_samples = int(round(max_lag_s * grid_hz))
    mask = np.abs(lags) <= max_samples
    if not np.any(mask):
        return 0.0, 0.0
    masked = np.where(mask, corr, -1e9)
    best_idx = int(np.argmax(masked))
    best_lag = float(lags[best_idx])
    peak_corr = float(corr[best_idx])

    if 0 < best_idx < len(corr) - 1:
        y0, y1, y2 = float(corr[best_idx - 1]), float(corr[best_idx]), float(corr[best_idx + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            best_lag += 0.5 * (y0 - y2) / denom

    # Account for grid start difference (t_ref[0] - t_tgt[0])
    offset_s = (best_lag / grid_hz) + float(t_ref[0] - t_tgt[0])
    return float(offset_s), float(peak_corr)


def synchronize_multiview_video_observations(
    video_paths_by_cam: Dict[str, "Path"],
    pts_s_by_cam: Dict[str, Sequence[float]],
    body26_by_cam: Dict[str, np.ndarray],
    target_fps: float = 30.0,
    extra_obs_by_cam: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    enable_auto_sync: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]], Dict[str, str]]:
    """Automatically synchronize 2+ multi-camera video timelines (Audio Clap -> Vertical Motion -> Native PTS)
    and resample all 2D body + hand observations onto one uniform overlapping session timeline.

    Returns:
        (canonical_ts_ns, aligned_body26, aligned_extra_tracks, sync_report_per_cam)
    """
    cam_uids = [u for u in body26_by_cam if body26_by_cam[u].shape[0] >= 2]
    if len(cam_uids) < 2:
        raise ValueError("Multi-video synchronization requires at least 2 cameras.")

    ref_uid = cam_uids[0]
    ref_audio = (
        extract_audio_pcm_from_video(video_paths_by_cam[ref_uid])
        if (enable_auto_sync and ref_uid in video_paths_by_cam)
        else None
    )

    timelines_ns: Dict[str, np.ndarray] = {}
    sync_report: Dict[str, str] = {ref_uid: "reference (0.0 ms)"}

    def _pts_to_ns(uid: str, offset_s: float) -> np.ndarray:
        n = body26_by_cam[uid].shape[0]
        pts = np.asarray(pts_s_by_cam.get(uid, []), dtype=np.float64)
        if len(pts) != n or (n >= 2 and pts[-1] <= pts[0] + 1e-3):
            pts = np.arange(n, dtype=np.float64) / max(target_fps, 1.0)
        pts = pts - pts[0] + offset_s
        return np.round(pts * 1e9).astype(np.int64)

    timelines_ns[ref_uid] = _pts_to_ns(ref_uid, 0.0)

    for uid in cam_uids[1:]:
        offset_s = 0.0
        method = "native_pts (0.0 ms)"
        if enable_auto_sync:
            # Tier 1: Audio Clap / Waveform cross-correlation (< 0.5 ms precision!)
            audio_done = False
            if ref_audio is not None and uid in video_paths_by_cam:
                tgt_audio = extract_audio_pcm_from_video(video_paths_by_cam[uid])
                if tgt_audio is not None:
                    a_off, a_snr = estimate_audio_sync_offset_s(
                        ref_audio, tgt_audio, sample_rate_hz=16000, max_lag_s=4.0
                    )
                    # estimate_audio_sync_offset_s returns positive when target starts EARLIER (shifted right in target)
                    if a_snr >= 6.0 and abs(a_off) <= 4.0:
                        offset_s = -a_off
                        method = f"audio_sync ({offset_s * 1000.0:+.1f} ms, SNR={a_snr:.1f})"
                        audio_done = True

            # Tier 2: View-invariant Vertical Motion cross-correlation (works even on silent MP4s!)
            if not audio_done:
                m_off, m_corr = estimate_motion_sync_offset_s(
                    body26_by_cam[ref_uid],
                    pts_s_by_cam.get(ref_uid, []),
                    body26_by_cam[uid],
                    pts_s_by_cam.get(uid, []),
                    max_lag_s=4.0,
                )
                if m_corr >= 0.42 and abs(m_off) <= 4.0:
                    offset_s = m_off
                    method = f"motion_sync ({offset_s * 1000.0:+.1f} ms, r={m_corr:.2f})"
                else:
                    method = f"native_pts (motion r={m_corr:.2f})"

        timelines_ns[uid] = _pts_to_ns(uid, offset_s)
        sync_report[uid] = method

    # Guard against non-overlapping timelines if an extreme offset occurred
    try:
        canonical_ns, aligned_body26 = align_observations_to_common_timeline(
            timelines_ns,
            body26_by_cam,
            target_fps=target_fps,
            max_interp_gap_s=max(0.08, 2.5 / max(target_fps, 1.0)),
        )
    except ValueError:
        # Fallback cleanly to zero-offset native timelines so reconstruction never crashes
        for uid in cam_uids:
            timelines_ns[uid] = _pts_to_ns(uid, 0.0)
            sync_report[uid] = "native_pts_fallback (0.0 ms)"
        canonical_ns, aligned_body26 = align_observations_to_common_timeline(
            timelines_ns,
            body26_by_cam,
            target_fps=target_fps,
            max_interp_gap_s=max(0.08, 2.5 / max(target_fps, 1.0)),
        )

    aligned_extra: Dict[str, Dict[str, np.ndarray]] = {}
    if extra_obs_by_cam:
        for track_name, per_cam_arr in extra_obs_by_cam.items():
            valid_sub = {
                u: per_cam_arr[u]
                for u in cam_uids
                if u in per_cam_arr and per_cam_arr[u].shape[0] >= 2
            }
            if len(valid_sub) >= 2:
                _, aligned_sub = align_observations_to_common_timeline(
                    {u: timelines_ns[u] for u in valid_sub},
                    valid_sub,
                    target_fps=target_fps,
                    max_interp_gap_s=max(0.08, 2.5 / max(target_fps, 1.0)),
                )
                aligned_extra[track_name] = aligned_sub

    return canonical_ns, aligned_body26, aligned_extra, sync_report
