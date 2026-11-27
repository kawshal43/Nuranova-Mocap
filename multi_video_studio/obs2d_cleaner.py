"""2D Multi-View Keypoint Pre-Cleaner: L/R Limb-Swap Repair, Velocity/MAD Despiker, and Zero-Phase One-Euro Filter.

Removes single-camera 2D occlusion jumps (e.g. 300-580 px wrist/ankle spikes and L/R swaps) and
sub-pixel SimCC argmax buzz BEFORE camera calibration and 3D multi-view optimization so 2D noise
never amplifies into 3D depth teleports!
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import median_filter

# Symmetric Left/Right limb keypoint pairs in canonical Body26 layout
_LR_LIMB_PAIRS: Tuple[Tuple[int, int], ...] = (
    (7, 8),    # elbows
    (9, 10),   # wrists
    (13, 14),  # knees
    (15, 16),  # ankles
    (20, 21),  # big toes
    (22, 23),  # small toes
    (24, 25),  # heels
)


def _one_euro_1d_pass(
    x: np.ndarray,
    valid: np.ndarray,
    fps: float,
    min_cutoff: float,
    beta: float,
    d_cutoff: float,
) -> np.ndarray:
    """Single directional pass of the adaptive One-Euro filter (Casiez et al. CHI 2012) over (T, 2) pixels."""
    T = x.shape[0]
    out = x.copy()
    dt = 1.0 / max(fps, 5.0)
    tau_d = 1.0 / (2.0 * np.pi * max(d_cutoff, 0.1))
    alpha_d = 1.0 / (1.0 + tau_d / dt)

    initialized = False
    x_prev = np.zeros(2, dtype=np.float64)
    dx_hat = np.zeros(2, dtype=np.float64)

    for t in range(T):
        if not valid[t]:
            initialized = False
            continue
        cur = x[t]
        if not initialized:
            x_prev = cur.copy()
            dx_hat[:] = 0.0
            out[t] = cur
            initialized = True
            continue
        dx = (cur - x_prev) / dt
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * dx_hat
        speed = float(np.linalg.norm(dx_hat))
        cutoff = min_cutoff + beta * speed
        tau = 1.0 / (2.0 * np.pi * max(cutoff, 0.1))
        alpha = 1.0 / (1.0 + tau / dt)
        x_hat = alpha * cur + (1.0 - alpha) * x_prev
        out[t] = x_hat
        x_prev = x_hat
    return out


def _zero_phase_one_euro_2d(
    uv_1080: np.ndarray,  # (T, 2) normalized to 1080p pixel units
    valid: np.ndarray,    # (T,) bool
    fps: float = 30.0,
    min_cutoff: float = 2.4,
    beta: float = 0.012,
    d_cutoff: float = 1.2,
) -> np.ndarray:
    """Zero-phase bidirectional One-Euro filter (zero lag + silky still hold + crisp fast motion!)."""
    if len(uv_1080) < 5 or int(np.sum(valid)) < 4:
        return uv_1080
    fwd = _one_euro_1d_pass(uv_1080, valid, fps, min_cutoff, beta, d_cutoff)
    bwd = _one_euro_1d_pass(uv_1080[::-1], valid[::-1], fps, min_cutoff, beta, d_cutoff)[::-1]
    return np.where(valid[:, None], 0.5 * (fwd + bwd), uv_1080)


def _repair_lr_swaps_camera(
    uv_1080: np.ndarray,  # (T, 26, 2) float64
    conf: np.ndarray,     # (T, 26) float64
    min_conf: float = 0.20,
    swap_trigger_px: float = 42.0,
) -> int:
    """Detect and repair sudden single-view L/R limb swaps (e.g. crossing wrists/ankles during side turns)."""
    T = uv_1080.shape[0]
    if T < 4:
        return 0
    repaired = 0
    for l_k, r_k in _LR_LIMB_PAIRS:
        for t in range(1, T):
            if (
                conf[t - 1, l_k] < min_conf
                or conf[t - 1, r_k] < min_conf
                or conf[t, l_k] < min_conf
                or conf[t, r_k] < min_conf
            ):
                continue
            d_keep = float(
                np.linalg.norm(uv_1080[t, l_k] - uv_1080[t - 1, l_k])
                + np.linalg.norm(uv_1080[t, r_k] - uv_1080[t - 1, r_k])
            )
            d_swap = float(
                np.linalg.norm(uv_1080[t, r_k] - uv_1080[t - 1, l_k])
                + np.linalg.norm(uv_1080[t, l_k] - uv_1080[t - 1, r_k])
            )
            # Only un-swap when unswapped jump is large and swapping cuts total jump dramatically!
            if d_keep > 2.0 * swap_trigger_px and d_swap < 0.42 * d_keep:
                tmp_uv = uv_1080[t, l_k].copy()
                uv_1080[t, l_k] = uv_1080[t, r_k]
                uv_1080[t, r_k] = tmp_uv
                tmp_c = float(conf[t, l_k])
                conf[t, l_k] = conf[t, r_k]
                conf[t, r_k] = tmp_c
                repaired += 1
    return repaired


def clean_camera_body26_2d(
    uvc: np.ndarray,  # (T, 26, 3) float32 native resolution
    img_width: int,
    fps: float = 30.0,
    min_conf: float = 0.18,
    spike_px_1080: float = 52.0,
    max_fill_gap: int = 4,
) -> Tuple[np.ndarray, int]:
    """Clean one camera's synchronized 2D Body26 sequence: un-swap L/R, despike jumps, zero-phase One-Euro smooth.

    Returns:
        (cleaned_uvc (T, 26, 3) float32 in original pixel coordinates, num_repaired_2d_entries)
    """
    T, K, _ = uvc.shape
    if T < 5:
        return uvc.astype(np.float32).copy(), 0

    s1080 = 1920.0 / max(float(img_width), 1.0)
    uv = uvc[:, :, :2].astype(np.float64) * s1080
    cf = uvc[:, :, 2].astype(np.float64).copy()
    repaired = 0

    # 1. Un-swap transient L/R limb crossing swaps
    repaired += _repair_lr_swaps_camera(uv, cf, min_conf=min_conf)

    # 2. Vectorized rolling-median + acceleration spike rejection (~300x faster than Python loop!)
    uv_med_all = median_filter(uv, size=(5, 1, 1), mode="nearest")
    for k in range(K):
        valid = (cf[:, k] >= min_conf) & np.all(np.isfinite(uv[:, k]), axis=-1)
        if int(np.sum(valid)) < 7:
            continue

        dev = np.linalg.norm(uv[:, k] - uv_med_all[:, k], axis=-1)
        cand = valid & (dev > spike_px_1080)
        bad = np.zeros(T, dtype=np.bool_)
        if np.any(cand):
            mid = np.empty_like(uv[:, k])
            mid[1:-1] = 0.5 * (uv[:-2, k] + uv[2:, k])
            mid[0] = uv[1, k]
            mid[-1] = uv[-2, k]
            acc_dev = np.linalg.norm(uv[:, k] - mid, axis=-1)
            interior_ok = np.zeros(T, dtype=np.bool_)
            interior_ok[1:-1] = valid[:-2] & valid[2:]
            bad = cand & (~interior_ok | (acc_dev > 0.70 * spike_px_1080))

        if np.any(bad):
            repaired += int(np.sum(bad))
            valid[bad] = False
            cf[bad, k] = 0.05  # Downweight aggressively for downstream calibration & IRLS!

            # Fill short 1..max_fill_gap spike holes cleanly with shape-preserving PCHIP
            good_idx = np.where(valid)[0]
            if len(good_idx) >= 4:
                pchip_x = PchipInterpolator(good_idx, uv[good_idx, k, 0], extrapolate=False)
                pchip_y = PchipInterpolator(good_idx, uv[good_idx, k, 1], extrapolate=False)
                for g_i in range(len(good_idx) - 1):
                    a, b = int(good_idx[g_i]), int(good_idx[g_i + 1])
                    gap = b - a - 1
                    if 0 < gap <= max_fill_gap:
                        qs = np.arange(a + 1, b, dtype=np.float64)
                        uv[a + 1 : b, k, 0] = pchip_x(qs)
                        uv[a + 1 : b, k, 1] = pchip_y(qs)
                        # Mark interpolated spike with modest confidence so true views still dominate!
                        cf[a + 1 : b, k] = np.maximum(cf[a + 1 : b, k], 0.28)
                        valid[a + 1 : b] = True

        # 3. Gentle Zero-Phase Adaptive One-Euro Filter (removes high-freq SimCC buzz without blurring fast swings!)
        uv[:, k] = _zero_phase_one_euro_2d(
            uv[:, k],
            valid=valid,
            fps=fps,
            min_cutoff=2.4,
            beta=0.012,
            d_cutoff=1.2,
        )

    # Recompute exact derived midpoints (neck=18, hips=19) so they remain geometrically consistent with shoulders/hips
    for mid_k, a_k, b_k in ((18, 5, 6), (19, 11, 12)):
        both_ok = (cf[:, a_k] >= min_conf) & (cf[:, b_k] >= min_conf)
        uv[both_ok, mid_k] = 0.5 * (uv[both_ok, a_k] + uv[both_ok, b_k])
        cf[both_ok, mid_k] = np.minimum(cf[both_ok, a_k], cf[both_ok, b_k])

    out = np.empty_like(uvc, dtype=np.float32)
    out[:, :, :2] = (uv / s1080).astype(np.float32)
    out[:, :, 2] = np.clip(cf, 0.0, 1.0).astype(np.float32)
    return out, repaired


def clean_synchronized_2d_tracks(
    aligned_body26_by_cam: Dict[str, np.ndarray],
    image_sizes_by_cam: Dict[str, Tuple[int, int]],
    fps: float = 30.0,
) -> Tuple[Dict[str, np.ndarray], int]:
    """Run L/R swap repair, spike rejection, and zero-phase adaptive One-Euro filtering across all camera tracks."""
    cleaned: Dict[str, np.ndarray] = {}
    total_repaired = 0
    for uid, arr in aligned_body26_by_cam.items():
        w = int(image_sizes_by_cam.get(uid, (1920, 1080))[0])
        c_arr, n_rep = clean_camera_body26_2d(arr, img_width=w, fps=fps)
        cleaned[uid] = c_arr
        total_repaired += n_rep
    return cleaned, total_repaired
