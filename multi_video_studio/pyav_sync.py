"""PyAV multi-video synchronization using original presentation timestamps.

Match waveform windows by FFT correlation and robustly estimate clock offset/drift.
Report matching evidence and residuals; recording conditions determine accuracy.
Flash and body motion provide fallbacks, with optional manual offsets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from desktop.synchronization.sync_engine import (
    align_observations_to_common_timeline,
    detect_flash_sync_frame,
    estimate_motion_sync_offset_s,
)


@dataclass
class CameraSyncEstimate:
    camera_uid: str
    offset_s: float  # Positive => this camera started AFTER the reference camera
    method: (
        str  # "reference", "audio_clap", "flash_pulse", "vertical_motion", "manual", "native_pts"
    )
    confidence: float  # SNR for audio, corr peak [0..1] for motion
    summary: str
    drift_slope: float = 1.0  # Linear clock-rate ratio vs reference (1.0 = zero drift)
    drift_pivot_s: float = 0.0  # Local audio time pivot (s) where offset_s is exact
    audio_start_s: float = 0.0  # Container audio stream start_time (s)
    base_pts_s: Optional[float] = (
        None  # Explicit video timeline origin (preserved when seeking overlap windows)
    )
    alignment_residual_ms: Optional[float] = None
    matched_audio_windows: int = 0
    alignment_verified: bool = False
    pairwise_closure_ms: Optional[float] = None
    rejected_audio_bridges: int = 0


def _probe_audio_start_s(video_path: str | Path) -> float:
    """Return container audio stream start time (seconds), or 0.0 if absent/unknown."""
    p = Path(video_path)
    if not p.exists():
        return 0.0
    try:
        import av

        with av.open(str(p)) as container:
            if not container.streams.audio:
                return 0.0
            a_stream = container.streams.audio[0]
            if a_stream.start_time is not None and a_stream.time_base is not None:
                val = float(a_stream.start_time * a_stream.time_base)
                return val if -5.0 <= val <= 5.0 else 0.0
    except Exception:
        pass
    return 0.0


def extract_mono_audio_pyav(
    video_path: str | Path,
    sample_rate_hz: int = 16000,
    max_duration_s: float = 120.0,
) -> Optional[np.ndarray]:
    """Extract normalized mono float32 PCM audio directly using PyAV (`av`) — zero external FFmpeg needed."""
    p = Path(video_path)
    if not p.exists():
        return None
    try:
        import av

        with av.open(str(p)) as container:
            if not container.streams.audio:
                return None
            audio_stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="flt", layout="mono", rate=int(sample_rate_hz))
            chunks: List[np.ndarray] = []
            max_samples = int(max_duration_s * sample_rate_hz)
            total_samples = 0
            for frame in container.decode(audio_stream):
                resampled = resampler.resample(frame)
                frames_out = resampled if isinstance(resampled, list) else [resampled]
                for rf in frames_out:
                    if rf is None:
                        continue
                    arr = rf.to_ndarray().reshape(-1).astype(np.float32)
                    chunks.append(arr)
                    total_samples += len(arr)
                if total_samples >= max_samples:
                    break
            # Flush resampler
            try:
                flushed = resampler.resample(None)
                if flushed:
                    frames_out = flushed if isinstance(flushed, list) else [flushed]
                    for rf in frames_out:
                        if rf is not None:
                            chunks.append(rf.to_ndarray().reshape(-1).astype(np.float32))
            except Exception:
                pass
            if not chunks:
                return None
            pcm = np.concatenate(chunks, axis=0)[:max_samples]
            if len(pcm) < int(0.25 * sample_rate_hz) or float(np.std(pcm)) < 1e-5:
                return None
            return pcm
    except Exception:
        return None


def _compute_luma_series(video_path: Path, max_frames: int = 450) -> Tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.zeros(0, dtype=np.float64), 30.0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if not (5.0 <= fps <= 240.0):
        fps = 30.0
    lumas: List[float] = []
    try:
        for _ in range(max_frames):
            ok, bgr = cap.read()
            if not ok or bgr is None:
                break
            small = cv2.resize(bgr, (96, 54), interpolation=cv2.INTER_AREA)
            lumas.append(float(np.mean(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))))
    finally:
        cap.release()
    return np.asarray(lumas, dtype=np.float64), fps


def _estimate_audio_pair_with_drift(ref_hp, tgt_hp, sr, max_lag_s, diagnostics=None):
    from multi_video_studio.audio_alignment import estimate_audio_clock

    return estimate_audio_clock(ref_hp, tgt_hp, sr, max_lag_s, diagnostics)


def estimate_video_sync_offsets(
    video_paths_by_cam: Dict[str, Path],
    pts_s_by_cam: Optional[Dict[str, Sequence[float]]] = None,
    body26_by_cam: Optional[Dict[str, np.ndarray]] = None,
    manual_offsets_ms: Optional[Dict[str, float]] = None,
    max_lag_s: Optional[float] = None,
    enable_audio: bool = True,
    enable_flash: bool = True,
    enable_motion: bool = True,
) -> Dict[str, CameraSyncEstimate]:
    """Compute per-camera start offsets (seconds) relative to the first camera using PyAV Audio -> Flash -> Motion.

    Supports wide start-time offsets (up to 60s), automatic phone crystal clock-drift compensation,
    exact container audio/video start-PTS preservation, and 3+ camera pairwise triangle-closure verification!
    """
    cam_uids = list(video_paths_by_cam.keys())
    if not cam_uids:
        return {}

    ref_uid = cam_uids[0]
    manual = manual_offsets_ms or {}
    sr = 16000

    audio_starts: Dict[str, float] = {
        u: _probe_audio_start_s(video_paths_by_cam[u]) for u in cam_uids
    }
    results: Dict[str, CameraSyncEstimate] = {
        ref_uid: CameraSyncEstimate(
            camera_uid=ref_uid,
            offset_s=float(manual.get(ref_uid, 0.0)) * 1e-3,
            method="reference",
            confidence=1.0,
            summary="Reference camera (0.0 ms)",
            drift_slope=1.0,
            drift_pivot_s=0.0,
            audio_start_s=audio_starts[ref_uid],
            alignment_verified=True,
        )
    }

    audio_hp_by_cam: Dict[str, np.ndarray] = {}
    if enable_audio:
        for u in cam_uids:
            pcm = extract_mono_audio_pyav(video_paths_by_cam[u], sample_rate_hz=sr)
            if pcm is not None:
                audio_hp_by_cam[u] = np.diff(pcm, prepend=pcm[0])

    ref_hp = audio_hp_by_cam.get(ref_uid)
    ref_flash_t: Optional[float] = None

    for uid in cam_uids[1:]:
        nudge_s = float(manual.get(uid, 0.0)) * 1e-3
        # Dynamically allow wide start-time lags (up to 60s / 80% of clip duration) when user walks between phones!
        if max_lag_s is not None:
            lag_limit = float(max_lag_s)
        elif ref_hp is not None and uid in audio_hp_by_cam:
            max_dur = max(len(ref_hp), len(audio_hp_by_cam[uid])) / float(sr)
            lag_limit = float(np.clip(0.85 * max_dur, 6.0, 60.0))
        else:
            lag_limit = 30.0

        # Tier 1: Sub-millisecond PyAV Audio Clap / Waveform Cross-Correlation + Linear Drift Compensation
        if enable_audio and ref_hp is not None and uid in audio_hp_by_cam:
            tgt_hp = audio_hp_by_cam[uid]
            fit_details = {}
            off_est, a_snr, slope_est, pivot_est = _estimate_audio_pair_with_drift(
                ref_hp, tgt_hp, sr, lag_limit, fit_details
            )

            # For 3+ cameras, if direct SNR is modest (< 28), check triangle bridge via any already-synced high-SNR cam!
            if a_snr < 28.0:
                for bridge_uid, b_est in list(results.items()):
                    if (
                        bridge_uid == ref_uid
                        or b_est.method != "audio_clap"
                        or b_est.confidence < 25.0
                    ):
                        continue
                    b_hp = audio_hp_by_cam.get(bridge_uid)
                    if b_hp is None:
                        continue
                    b_off, b_snr, b_slope, b_piv = _estimate_audio_pair_with_drift(
                        b_hp, tgt_hp, sr, lag_limit
                    )
                    if b_snr > a_snr * 1.25 and b_snr >= 12.0:
                        bridge_intercept = (
                            b_est.offset_s - (b_est.drift_slope - 1) * b_est.drift_pivot_s
                        )
                        target_intercept = b_off - (b_slope - 1) * b_piv
                        combined_intercept = bridge_intercept + b_est.drift_slope * target_intercept
                        a_snr = b_snr
                        slope_est = b_est.drift_slope * b_slope
                        pivot_est = b_piv
                        off_est = combined_intercept + (slope_est - 1) * pivot_est

            if a_snr >= 5.5 and abs(off_est) <= lag_limit + 5.0:
                off_s = off_est + nudge_s
                drift_ppm = (slope_est - 1.0) * 1e6
                drift_tag = f" · drift {drift_ppm:+.0f}ppm" if abs(drift_ppm) >= 80.0 else ""
                results[uid] = CameraSyncEstimate(
                    camera_uid=uid,
                    offset_s=off_s,
                    method="audio_clap",
                    confidence=float(a_snr),
                    summary=f"Audio waveform ({off_s * 1000.0:+.1f} ms · {fit_details.get('inliers', 0)} matching windows{drift_tag})",
                    drift_slope=slope_est,
                    drift_pivot_s=pivot_est,
                    audio_start_s=audio_starts[uid],
                    alignment_residual_ms=fit_details.get("residual_ms"),
                    matched_audio_windows=fit_details.get("inliers", 0),
                    alignment_verified=fit_details.get("inliers", 0) >= 3,
                )
                continue

        # Tier 2: Visible Flash / Brightness Pulse Sync
        if enable_flash:
            if ref_flash_t is None:
                l_ref, fps_ref = _compute_luma_series(video_paths_by_cam[ref_uid])
                f_ref = detect_flash_sync_frame(l_ref, z_threshold=5.5)
                ref_flash_t = (float(f_ref) / fps_ref) if f_ref is not None else -1.0
            if ref_flash_t is not None and ref_flash_t >= 0.0:
                l_tgt, fps_tgt = _compute_luma_series(video_paths_by_cam[uid])
                f_tgt = detect_flash_sync_frame(l_tgt, z_threshold=5.5)
                if f_tgt is not None:
                    t_tgt = float(f_tgt) / fps_tgt
                    raw_off = ref_flash_t - t_tgt
                    if abs(raw_off) <= lag_limit:
                        off_s = raw_off + nudge_s
                        results[uid] = CameraSyncEstimate(
                            camera_uid=uid,
                            offset_s=off_s,
                            method="flash_pulse",
                            confidence=0.95,
                            summary=f"Flash Sync ({off_s * 1000.0:+.1f} ms)",
                            audio_start_s=audio_starts[uid],
                        )
                        continue

        # Tier 3: View-Invariant Vertical Body Motion Cross-Correlation (Silent Video Fallback!)
        if enable_motion and body26_by_cam and ref_uid in body26_by_cam and uid in body26_by_cam:
            pts_ref = (pts_s_by_cam or {}).get(ref_uid, [])
            pts_tgt = (pts_s_by_cam or {}).get(uid, [])
            m_off, m_corr = estimate_motion_sync_offset_s(
                body26_by_cam[ref_uid], pts_ref, body26_by_cam[uid], pts_tgt, max_lag_s=lag_limit
            )
            if m_corr >= 0.40 and abs(m_off) <= lag_limit:
                off_s = m_off + nudge_s
                results[uid] = CameraSyncEstimate(
                    camera_uid=uid,
                    offset_s=off_s,
                    method="vertical_motion",
                    confidence=float(m_corr),
                    summary=f"Motion Sync ({off_s * 1000.0:+.1f} ms · r={m_corr:.2f})",
                    audio_start_s=audio_starts[uid],
                )
                continue

        # Fallback: Native PTS (+ any user manual nudge)
        results[uid] = CameraSyncEstimate(
            camera_uid=uid,
            offset_s=nudge_s,
            method="manual" if abs(nudge_s) > 1e-6 else "native_pts",
            confidence=0.5,
            summary=f"Native PTS ({nudge_s * 1000.0:+.1f} ms)",
            audio_start_s=audio_starts[uid],
        )

    # Verify independent audio routes. A weak echo match must not override stronger direct evidence.
    for i, a_uid in enumerate(cam_uids[1:], start=1):
        for b_uid in cam_uids[i + 1 :]:
            if a_uid not in audio_hp_by_cam or b_uid not in audio_hp_by_cam:
                continue
            ea, eb = results[a_uid], results[b_uid]
            if ea.method != "audio_clap" or eb.method != "audio_clap":
                continue
            detail = {}
            off, snr, slope, pivot = _estimate_audio_pair_with_drift(
                audio_hp_by_cam[a_uid], audio_hp_by_cam[b_uid], sr, float(max_lag_s or 60), detail
            )
            mid = 0.5 * len(audio_hp_by_cam[b_uid]) / sr
            direct = mid + eb.offset_s + (eb.drift_slope - 1) * (mid - eb.drift_pivot_s)
            via_a = mid + off + (slope - 1) * (mid - pivot)
            via_ref = via_a + ea.offset_s + (ea.drift_slope - 1) * (via_a - ea.drift_pivot_s)
            closure = abs(direct - via_ref) * 1000
            eb.pairwise_closure_ms = float(closure)
            if snr >= 10 and closure > 20:
                if detail.get("inliers", 0) < min(
                    ea.matched_audio_windows, eb.matched_audio_windows
                ):
                    eb.rejected_audio_bridges += 1
                else:
                    eb.alignment_verified = False
                    eb.summary += f" · verify timing ({closure:.0f} ms pair disagreement)"
    return results


def map_camera_pts_to_global_s(
    pts_s: Sequence[float], est: CameraSyncEstimate, fallback_fps: float = 30.0
) -> np.ndarray:
    """Convert raw video frame PTS (seconds) into exact audio-aligned global timeline seconds.

    Preserves container `v_start - a_start` offset (avoiding 100-150ms multi-frame shifts!) and applies
    linear crystal clock-drift compensation when present.
    """
    n = len(pts_s)
    pts = np.asarray(pts_s, dtype=np.float64)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n >= 2 and pts[-1] <= pts[0] + 1e-3:
        pts = np.arange(n, dtype=np.float64) / max(fallback_fps, 1.0)
    # Preserve explicit origin when Stage 1.5 trimmed leading non-overlapping frames; otherwise strip only epoch-scale (>1h) PTS
    if est.base_pts_s is not None:
        base = float(est.base_pts_s)
    else:
        base = float(pts[0]) if abs(float(pts[0])) > 3600.0 else float(est.audio_start_s)
    local_t = pts - base
    slope = float(est.drift_slope) if est.drift_slope else 1.0
    pivot = float(est.drift_pivot_s) if est.drift_pivot_s else 0.0
    return local_t + float(est.offset_s) + (slope - 1.0) * (local_t - pivot)


def resample_synchronized_tracks(
    sync_estimates: Dict[str, CameraSyncEstimate],
    pts_s_by_cam: Dict[str, Sequence[float]],
    primary_obs_by_cam: Dict[str, np.ndarray],
    target_fps: float = 60.0,
    extra_tracks_by_name: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
    """Resample multi-camera primary & extra 2D tracks onto one canonical synchronized timeline."""
    cam_uids = [
        u for u in primary_obs_by_cam if u in sync_estimates and primary_obs_by_cam[u].shape[0] >= 2
    ]
    if len(cam_uids) < 2:
        raise ValueError("Synchronized resampling requires at least 2 cameras.")

    def _build_timeline_ns(uid: str, est: CameraSyncEstimate) -> np.ndarray:
        n = primary_obs_by_cam[uid].shape[0]
        raw_pts = pts_s_by_cam.get(uid, [])
        if len(raw_pts) != n:
            raw_pts = (np.arange(n, dtype=np.float64) / max(target_fps, 1.0)).tolist()
        t_glob = map_camera_pts_to_global_s(raw_pts, est, fallback_fps=target_fps)
        return np.round(t_glob * 1e9).astype(np.int64)

    timelines_ns = {u: _build_timeline_ns(u, sync_estimates[u]) for u in cam_uids}
    # Support low-FPS / VFR mobile cameras (e.g. 15fps -> 67ms dt) cleanly without dropping observations!
    max_cam_dt = 0.035
    for u in cam_uids:
        dts = np.diff(timelines_ns[u].astype(np.float64) * 1e-9)
        pos = dts[dts > 1e-4]
        if len(pos):
            max_cam_dt = max(max_cam_dt, float(np.percentile(pos, 95)))
    max_gap = max(0.16, 2.8 * max_cam_dt, 2.5 / max(target_fps, 1.0))

    try:
        canonical_ns, aligned_primary = align_observations_to_common_timeline(
            timelines_ns, primary_obs_by_cam, target_fps=target_fps, max_interp_gap_s=max_gap
        )
    except ValueError as exc:
        raise ValueError(
            "The synchronized videos have no usable shared timeline. Check the start offsets "
            "and select videos of the same performance."
        ) from exc

    aligned_extras: Dict[str, Dict[str, np.ndarray]] = {}
    if extra_tracks_by_name:
        for tname, per_cam in extra_tracks_by_name.items():
            sub = {u: per_cam[u] for u in cam_uids if u in per_cam and per_cam[u].shape[0] >= 2}
            if len(sub) >= 2:
                # Include all body timelines so hand tracks use exactly the same frame range.
                extra_obs = {
                    u: sub.get(
                        u,
                        np.zeros(
                            (len(timelines_ns[u]), *next(iter(sub.values())).shape[1:]),
                            dtype=np.float32,
                        ),
                    )
                    for u in cam_uids
                }
                extra_ns, aligned_sub = align_observations_to_common_timeline(
                    timelines_ns,
                    extra_obs,
                    target_fps=target_fps,
                    max_interp_gap_s=max_gap,
                )
                if not np.array_equal(extra_ns, canonical_ns):
                    raise ValueError("Additional tracks do not match the body timeline.")
                aligned_extras[tname] = {u: aligned_sub[u] for u in sub}

    return canonical_ns, aligned_primary, aligned_extras
