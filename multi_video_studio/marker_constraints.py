"""Wearable tag observations and calibrated tag-to-joint constraints.

Four corners establish tag orientation. A center annotation alone cannot establish it.
"""

from __future__ import annotations

import cv2
import numpy as np

from engine.marker_tracking.detector import FiducialMarkerDetector
from engine.video.video_io import iter_video_frames
from multi_video_studio.review_video import ReviewVideoReader


def detect_take_tags(package, review, observations, cancel=None, progress=None):
    detector = FiducialMarkerDetector()
    cameras = [uid for uid in package.motion_clips if uid in observations.cameras]
    for ci, uid in enumerate(cameras):
        camera = observations.cameras[uid]
        target_times = camera["pts"]
        if not len(target_times):
            continue
        for index, pts, image in iter_video_frames(package.motion_clips[uid].path,
                start_pts_s=float(target_times[0]), end_pts_s=float(target_times[-1])):
            if cancel is not None and cancel.is_set():
                raise RuntimeError("Marker detection cancelled.")
            key = f"{pts:.6f}"
            if key not in camera["tags"]:
                camera["tags"][key] = [{"id": tag.tag_id, "corners": tag.corners_uv.tolist(),
                    "confidence": tag.confidence, "dictionary": tag.dictionary_name} for tag in detector.detect(image)]
            if progress and index % 60 == 0:
                progress((ci + (pts - target_times[0]) / max(target_times[-1] - target_times[0], .01)) / len(cameras),
                         f"Detecting wearable tags: {uid} {pts:.2f}s")


def native_tag_tracks(observations, review):
    extras = {}
    for tag_id, assignment in review.marker_assignments.items():
        per_camera = {}
        for uid, camera in observations.cameras.items():
            array = np.zeros((len(camera["pts"]), 4, 3), np.float32)
            for i, pts in enumerate(camera["pts"]):
                tags = camera["tags"].get(f"{pts:.6f}", [])
                matches = [v for v in tags if str(v["id"]) == str(tag_id)]
                if len(matches) == 1:
                    array[i, :, :2] = matches[0]["corners"]
                    array[i, :, 2] = matches[0].get("confidence", .8)
                for edit in review.points:
                    if edit["camera_uid"] != uid or abs(edit["pts_s"] - pts) > 1e-5:
                        continue
                    prefix = f"tag:{tag_id}:corner"
                    if edit["key"].startswith(prefix):
                        k = int(edit["key"][len(prefix):])
                        if 0 <= k < 4:
                            if edit["valid"] and edit["xy"] is not None:
                                array[i, k] = [*edit["xy"], 1.]
                            else:
                                array[i, k, 2] = 0
            per_camera[uid] = array
        extras["tag:" + str(tag_id)] = per_camera
    return extras


def tag_pose_from_corners(camera, corners_native, native_size, size_m):
    """For neutral-pose offset calibration only; return tag-to-world matrix and error."""
    corners = np.asarray(corners_native, float).reshape(4, 2).copy()
    corners *= [camera.intrinsics.image_width / native_size[0], camera.intrinsics.image_height / native_size[1]]
    half = float(size_m) / 2
    square = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], float)
    okay, rv, tv = cv2.solvePnP(square, corners, camera.intrinsics.camera_matrix,
                               camera.intrinsics.dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not okay or tv[2, 0] <= 0:
        raise ValueError("Tag pose could not be recovered from its four corners.")
    projected = cv2.projectPoints(square, rv, tv, camera.intrinsics.camera_matrix, camera.intrinsics.dist_coeffs)[0].reshape(4, 2)
    error = float(np.sqrt(np.mean(np.sum((projected - corners) ** 2, axis=1))))
    if error > 3:
        raise ValueError("Tag corners are inconsistent with a square; review the detection.")
    rotation = camera.extrinsics.R_cw.T @ cv2.Rodrigues(rv)[0]
    center = camera.extrinsics.R_cw.T @ (tv.ravel() - camera.extrinsics.t_cw)
    return rotation, center, error


def apply_tag_constraints(calibration, aligned_tags, review, body_observations, canonical_s):
    from engine.triangulation.triangulator import triangulate_sequence

    report = {"assigned": len(review.marker_assignments), "calibrated": 0, "constrained_samples": 0,
              "warnings": [], "tags": {}}
    constrained_joints = np.zeros(next(iter(body_observations.values())).shape[:2], bool)
    for tag_id, assignment in review.marker_assignments.items():
        source = aligned_tags.get("tag:" + str(tag_id))
        if not source or "joint_offset_tag_m" not in assignment or not assignment.get("offset_calibrated"):
            report["warnings"].append(f"Tag {tag_id}: calibrate the marker-to-joint offset before using it in the solve.")
            continue
        report["calibrated"] += 1
        points, confidence, errors, _ = triangulate_sequence(calibration.cameras, source,
            min_confidence=.3, max_reproj_error_px=8., refine_nonlinear=False)
        offset = np.asarray(assignment["joint_offset_tag_m"], float)
        size = float(assignment["size_m"])
        half = size / 2
        square = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], float)
        k = int(assignment["body_index"])
        valid_count = 0
        for t in range(len(points)):
            if np.min(confidence[t]) < .3 or not np.isfinite(points[t]).all():
                continue
            center = points[t].mean(axis=0)
            u, _, vt = np.linalg.svd((square - square.mean(axis=0)).T @ (points[t] - center))
            rotation = vt.T @ u.T
            if np.linalg.det(rotation) < 0:
                vt[-1] *= -1
                rotation = vt.T @ u.T
            fit = float(np.sqrt(np.mean(np.sum((square @ rotation.T + center - points[t]) ** 2, axis=1))))
            if fit > max(.004, size * .12):
                continue
            joint = rotation @ offset + center
            for uid, camera in calibration.cameras.items():
                if uid not in body_observations or uid not in source or np.min(source[uid][t, :, 2]) < .3:
                    continue
                if review.excluded(uid, review.local_time(uid, canonical_s[t]), "body"):
                    continue
                uv = camera.project_world_points(joint[None], True)[0]
                body_observations[uid][t, k] = [*uv, .98]
            constrained_joints[t, k] = True
            valid_count += 1
        report["tags"][str(tag_id)] = {"samples": valid_count, "body_index": k,
                                       "offset_method": assignment.get("offset_method", "manual")}
        report["constrained_samples"] += valid_count
    return constrained_joints, report


def calibrate_review_marker_offsets(package, review, observations, calibration, document, global_s):
    if document is None or document.joints_3d_raw is None:
        raise ValueError("Solve the markerless body first, or enter measured marker offsets manually.")
    index = int(np.argmin(abs(document.frame_timestamps_ns * 1e-9 - global_s)))
    count = 0
    for tag_id, assignment in review.marker_assignments.items():
        candidates = []
        for uid, camera in observations.cameras.items():
            if uid not in calibration.cameras:
                continue
            target = review.local_time(uid, global_s)
            # Use the displayed source frame rather than nominal FPS frame numbers.
            reader = ReviewVideoReader(package.motion_clips[uid].path, cache_frames=1)
            try:
                frame = reader.at(target)
            finally:
                reader.close()
            if not frame:
                continue
            for tag in camera["tags"].get(f"{frame[0]:.6f}", []):
                if str(tag["id"]) == str(tag_id):
                    source = review.sources[uid]
                    rotation, center, error = tag_pose_from_corners(calibration.cameras[uid], tag["corners"],
                        (source["width"], source["height"]), assignment["size_m"])
                    joint = document.joints_3d_raw[index, assignment["body_index"]]
                    if np.isfinite(joint).all():
                        candidates.append((error, rotation.T @ (joint - center)))
        if candidates:
            _, offset = min(candidates, key=lambda v: v[0])
            if np.linalg.norm(offset) <= .25:
                assignment.update(joint_offset_tag_m=offset.tolist(), offset_calibrated=True,
                    offset_method="body-assisted neutral pose", calibrated_global_s=float(global_s))
                count += 1
    if not count:
        raise ValueError("No assigned tags with reliable four-corner detections are visible at this instant.")
    return count
