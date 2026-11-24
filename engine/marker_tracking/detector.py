"""Physical marker detection (ArUco/AprilTag + subpixel visible blobs), temporal tracking, epipolar matching, and rigid cluster 6-DoF solving."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from desktop.calibration.charuco_board import ARUCO_DICTIONARIES
from engine.data_format.motion_io import ProvenanceCode
from engine.marker_tracking.marker_kit import RigidClusterDefinition
from engine.math_core.camera_model import CalibratedCamera, epipolar_line_distance_px
from engine.math_core.transforms import matrix_to_quat, solve_rigid_transform_kabsch
from engine.triangulation.triangulator import Observation2D, TriangulationResult, triangulate_point_robust


@dataclass
class DetectedTag2D:
    tag_id: int
    dictionary_name: str
    center_uv: np.ndarray  # (2,) float64
    corners_uv: np.ndarray  # (4, 2) float64
    confidence: float = 1.0


@dataclass
class DetectedBlob2D:
    track_id: int
    center_uv: np.ndarray  # (2,) float64 subpixel centroid
    radius_px: float
    circularity: float
    contrast_score: float
    confidence: float


@dataclass
class RigidClusterPose3D:
    cluster_id: str
    valid: bool
    position_world: np.ndarray  # (3,) meters
    rotation_matrix: np.ndarray  # (3, 3)
    rotation_quat_xyzw: np.ndarray  # (4,)
    rms_fit_error_m: float
    visible_marker_count: int
    confidence: float
    possible_identity_swap: bool = False
    warnings: List[str] = field(default_factory=list)


class FiducialMarkerDetector:
    """Detects valid ArUco and AprilTag wearable fiducials with subpixel corner refinement."""

    def __init__(self, dictionary_name: str = "DICT_APRILTAG_36h11") -> None:
        if dictionary_name not in ARUCO_DICTIONARIES:
            raise ValueError(f"Unsupported marker dictionary: {dictionary_name}")
        self.dictionary_name = dictionary_name
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[dictionary_name])
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 30
        self._detector = cv2.aruco.ArucoDetector(self._aruco_dict, params)

    def detect(self, image: np.ndarray) -> List[DetectedTag2D]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        corners, ids, _ = self._detector.detectMarkers(gray)
        results: List[DetectedTag2D] = []
        if ids is None or len(ids) == 0:
            return results
        for corner_arr, tid in zip(corners, ids.reshape(-1)):
            pts = corner_arr.reshape(4, 2).astype(np.float64)
            center = np.mean(pts, axis=0)
            # Confidence scales gently with tag pixel perimeter (tiny distant tags -> lower conf)
            perim = float(cv2.arcLength(pts.astype(np.float32), True))
            conf = float(np.clip(perim / 80.0, 0.35, 1.0))
            results.append(
                DetectedTag2D(
                    tag_id=int(tid),
                    dictionary_name=self.dictionary_name,
                    center_uv=center,
                    corners_uv=pts,
                    confidence=conf,
                )
            )
        return results


class VisibleBlobMarkerDetector:
    """Detects high-contrast visible white circular/spherical markers on matte black clothing.

    Uses adaptive luminance thresholding, contour circularity + area filtering, and
    intensity-weighted subpixel centroid localization.
    """

    def __init__(
        self,
        brightness_threshold: int = 200,
        min_area_px: float = 10.0,
        max_area_px: float = 3500.0,
        min_circularity: float = 0.62,
        min_contrast: float = 45.0,
    ) -> None:
        self.brightness_threshold = int(brightness_threshold)
        self.min_area_px = float(min_area_px)
        self.max_area_px = float(max_area_px)
        self.min_circularity = float(min_circularity)
        self.min_contrast = float(min_contrast)

    def detect(self, image: np.ndarray) -> List[DetectedBlob2D]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        blurred = cv2.GaussianBlur(gray, (3, 3), 0.8)
        _, binary = cv2.threshold(blurred, self.brightness_threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        h, w = gray.shape[:2]
        detections: List[DetectedBlob2D] = []
        for idx, cnt in enumerate(contours):
            area = float(cv2.contourArea(cnt))
            if area < self.min_area_px or area > self.max_area_px:
                continue
            perim = float(cv2.arcLength(cnt, True))
            if perim <= 1e-6:
                continue
            circularity = float(4.0 * np.pi * area / (perim * perim))
            if circularity < self.min_circularity:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            # Local ring contrast check (marker bright interior vs surrounding matte black cloth)
            pad = max(3, int(max(bw, bh) * 0.5))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
            roi = gray[y0:y1, x0:x1].astype(np.float64)
            mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            shifted = cnt - np.array([[[x0, y0]]])
            cv2.drawContours(mask, [shifted], -1, 255, thickness=-1)
            fg_vals = roi[mask > 0]
            bg_vals = roi[mask == 0]
            if fg_vals.size == 0 or bg_vals.size == 0:
                continue
            contrast = float(np.mean(fg_vals) - np.median(bg_vals))
            if contrast < self.min_contrast:
                continue

            # Intensity-weighted subpixel centroid above local background
            bg_level = float(np.median(bg_vals))
            weights = np.clip(roi - bg_level, 0.0, None) * (mask > 0)
            w_sum = float(np.sum(weights))
            if w_sum <= 1e-6:
                continue
            ys, xs = np.mgrid[y0:y1, x0:x1]
            cx = float(np.sum(xs * weights) / w_sum)
            cy = float(np.sum(ys * weights) / w_sum)
            radius = float(np.sqrt(area / np.pi))
            conf = float(np.clip(0.5 * circularity + 0.5 * min(contrast / 120.0, 1.0), 0.1, 1.0))

            detections.append(
                DetectedBlob2D(
                    track_id=idx,
                    center_uv=np.array([cx, cy], dtype=np.float64),
                    radius_px=radius,
                    circularity=circularity,
                    contrast_score=contrast,
                    confidence=conf,
                )
            )
        return detections


class TemporalBlobTracker2D:
    """Maintains persistent 2D track IDs across frames using constant-velocity prediction + Hungarian matching."""

    def __init__(self, max_jump_px: float = 45.0, max_missed_frames: int = 8) -> None:
        self.max_jump_px = float(max_jump_px)
        self.max_missed_frames = int(max_missed_frames)
        self._next_id = 1
        self._tracks: Dict[int, Dict] = {}

    def update(self, detections: List[DetectedBlob2D]) -> Tuple[List[DetectedBlob2D], bool]:
        """Assign persistent track IDs and report whether an ambiguous close-crossing swap risk occurred."""
        possible_swap = False
        if not self._tracks:
            for d in detections:
                d.track_id = self._next_id
                self._tracks[self._next_id] = {
                    "pos": d.center_uv.copy(),
                    "vel": np.zeros(2, dtype=np.float64),
                    "missed": 0,
                }
                self._next_id += 1
            return detections, False

        track_ids = list(self._tracks.keys())
        predicted = np.array([self._tracks[tid]["pos"] + self._tracks[tid]["vel"] for tid in track_ids])

        if detections:
            det_pts = np.array([d.center_uv for d in detections])
            cost = np.linalg.norm(predicted[:, None, :] - det_pts[None, :, :], axis=2)
            # Flag close crossing ambiguity if two detections are very close to the same predicted track
            for row in cost:
                if len(row) >= 2:
                    s = np.sort(row)
                    if s[0] < self.max_jump_px and s[1] < s[0] * 1.35 + 4.0:
                        possible_swap = True

            row_ind, col_ind = linear_sum_assignment(cost)
            matched_tracks = set()
            matched_dets = set()
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] <= self.max_jump_px:
                    tid = track_ids[r]
                    d = detections[c]
                    d.track_id = tid
                    prev_pos = self._tracks[tid]["pos"]
                    self._tracks[tid]["vel"] = 0.6 * (d.center_uv - prev_pos) + 0.4 * self._tracks[tid]["vel"]
                    self._tracks[tid]["pos"] = d.center_uv.copy()
                    self._tracks[tid]["missed"] = 0
                    matched_tracks.add(tid)
                    matched_dets.add(c)

            for c, d in enumerate(detections):
                if c not in matched_dets:
                    d.track_id = self._next_id
                    self._tracks[self._next_id] = {
                        "pos": d.center_uv.copy(),
                        "vel": np.zeros(2, dtype=np.float64),
                        "missed": 0,
                    }
                    self._next_id += 1

            for tid in track_ids:
                if tid not in matched_tracks:
                    self._tracks[tid]["missed"] += 1
        else:
            for tid in track_ids:
                self._tracks[tid]["missed"] += 1

        # Prune stale tracks
        stale = [tid for tid, st in self._tracks.items() if st["missed"] > self.max_missed_frames]
        for tid in stale:
            del self._tracks[tid]

        return detections, possible_swap


def match_and_triangulate_unlabeled_blobs(
    cameras_map: Dict[str, CalibratedCamera],
    blobs_per_camera: Dict[str, List[DetectedBlob2D]],
    max_epipolar_px: float = 6.0,
    max_reproj_px: float = 6.0,
) -> List[TriangulationResult]:
    """Cross-camera epipolar association + multi-view triangulation for unlabeled spherical markers."""
    cam_uids = [u for u in blobs_per_camera if u in cameras_map and blobs_per_camera[u]]
    if len(cam_uids) < 2:
        return []

    used: Dict[str, set[int]] = {u: set() for u in cam_uids}
    triangulated: List[TriangulationResult] = []

    for i in range(len(cam_uids)):
        ca = cam_uids[i]
        for bi_idx, ba in enumerate(blobs_per_camera[ca]):
            if bi_idx in used[ca]:
                continue
            obs = [Observation2D(ca, float(ba.center_uv[0]), float(ba.center_uv[1]), ba.confidence)]
            matched_indices: List[Tuple[str, int]] = [(ca, bi_idx)]

            for j in range(i + 1, len(cam_uids)):
                cb = cam_uids[j]
                best_b_idx = None
                best_dist = max_epipolar_px
                for bj_idx, bb in enumerate(blobs_per_camera[cb]):
                    if bj_idx in used[cb]:
                        continue
                    d_epi = epipolar_line_distance_px(cameras_map[ca], ba.center_uv, cameras_map[cb], bb.center_uv)
                    if d_epi < best_dist:
                        best_dist = d_epi
                        best_b_idx = bj_idx
                if best_b_idx is not None:
                    bb = blobs_per_camera[cb][best_b_idx]
                    obs.append(Observation2D(cb, float(bb.center_uv[0]), float(bb.center_uv[1]), bb.confidence))
                    matched_indices.append((cb, best_b_idx))

            if len(obs) >= 2:
                res = triangulate_point_robust(
                    cameras_map,
                    obs,
                    max_reproj_error_px=max_reproj_px,
                    refine_nonlinear=True,
                    provenance_if_valid=ProvenanceCode.MEASURED_MARKER,
                )
                if res.valid:
                    triangulated.append(res)
                    for u_cam, b_idx in matched_indices:
                        if u_cam in res.inlier_camera_uids:
                            used[u_cam].add(b_idx)

    return triangulated


def solve_rigid_cluster_pose(
    cluster_def: RigidClusterDefinition,
    observed_markers_3d: Dict[str, Tuple[np.ndarray, float]],
    max_rms_m: float = 0.015,
) -> RigidClusterPose3D:
    """Estimate 6-DoF rigid body pose of an asymmetric marker cluster via weighted Kabsch SVD.

    Args:
        cluster_def: Rigid cluster definition (>= 3 non-collinear local points).
        observed_markers_3d: Mapping marker_id -> (world_pos_3d_m, confidence).
    """
    local_pts = []
    world_pts = []
    weights = []
    for mid, lpt in zip(cluster_def.marker_ids, cluster_def.local_points_m):
        if mid in observed_markers_3d:
            wpt, conf = observed_markers_3d[mid]
            if np.all(np.isfinite(wpt)) and conf > 0.05:
                local_pts.append(lpt)
                world_pts.append(wpt)
                weights.append(conf)

    if len(local_pts) < 3:
        return RigidClusterPose3D(
            cluster_id=cluster_def.cluster_id,
            valid=False,
            position_world=np.full(3, np.nan),
            rotation_matrix=np.eye(3),
            rotation_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            rms_fit_error_m=float("inf"),
            visible_marker_count=len(local_pts),
            confidence=0.0,
            warnings=["Fewer than 3 cluster markers visible; 6-DoF orientation unobservable."],
        )

    P = np.asarray(local_pts, dtype=np.float64)
    Q = np.asarray(world_pts, dtype=np.float64)
    W = np.asarray(weights, dtype=np.float64)

    # Pairwise rigidity check to catch swapped marker identities before Kabsch fit
    swap_flag = False
    warnings: List[str] = []
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            d_local = float(np.linalg.norm(P[i] - P[j]))
            d_world = float(np.linalg.norm(Q[i] - Q[j]))
            if abs(d_local - d_world) > max_rms_m * 2.5:
                swap_flag = True
                warnings.append(
                    f"Pairwise rigid distance mismatch ({abs(d_local - d_world)*1000:.1f} mm) — possible marker swap."
                )

    R, t, rms = solve_rigid_transform_kabsch(P, Q, W)
    if rms > max_rms_m:
        swap_flag = True
        warnings.append(f"Rigid cluster RMS residual ({rms*1000:.1f} mm) exceeds {max_rms_m*1000:.1f} mm threshold.")

    conf = float(np.clip(np.mean(W) * np.exp(-rms / max(max_rms_m, 1e-4)), 0.0, 1.0))
    return RigidClusterPose3D(
        cluster_id=cluster_def.cluster_id,
        valid=rms <= max_rms_m * 2.0,
        position_world=t,
        rotation_matrix=R,
        rotation_quat_xyzw=matrix_to_quat(R),
        rms_fit_error_m=rms,
        visible_marker_count=len(P),
        confidence=conf,
        possible_identity_swap=swap_flag,
        warnings=warnings,
    )
