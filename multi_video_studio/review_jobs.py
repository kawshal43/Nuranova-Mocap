"""Background jobs for source-frame detection and anchored camera repair."""

from __future__ import annotations

import numpy as np

from engine.marker_tracking.detector import FiducialMarkerDetector
from engine.video.video_io import iter_video_frames
from multi_video_studio.review_observations import ReviewObservations
from multi_video_studio.review_video import ReviewVideoReader


def detect_review_points(package, review, camera_uids, start_s, end_s, current,
                         backend="rtmlib_wholebody", prefer_gpu=True,
                         progress=None, cancel=None):
    """Return new native observations; never mutate annotations or Qt objects."""
    if not camera_uids or end_s < start_s:
        raise ValueError("Select cameras and a valid time range.")
    output = ReviewObservations()
    detector = FiducialMarkerDetector() if review.capture_mode != "markerless" else None
    pose = None
    tracker = None
    try:
        if review.capture_mode != "markers":
            if current:
                from engine.ai_tracking.pose_tracker import AIPoseTracker2D
                pose = AIPoseTracker2D(backend, "performance", prefer_gpu)
            else:
                from multi_video_studio.wholebody_cuda import CinemaWholeBodyVideoTracker
                tracker = CinemaWholeBodyVideoTracker(backend, "performance", prefer_gpu)
                tracks = tracker.track_videos_parallel(
                    [(u, package.motion_clips[u].path, True, False) for u in camera_uids],
                    pts_windows_by_cam={u: (review.local_time(u, start_s), review.local_time(u, end_s))
                                        for u in camera_uids},
                    progress_cb=progress, cancel_event=cancel,
                )
                if review.hand_rois:
                    from multi_video_studio.roi_tracking import refine_tracks_with_rois
                    tracks = refine_tracks_with_rois(package, review, tracks, cancel=cancel, progress=progress)
                for uid, track in tracks.items():
                    output.set_track(uid, track.pts_s, track.body26_uvc,
                                     track.left_hand21_uvc, track.right_hand21_uvc,
                                     **{side + "_local": np.array([frame[side].landmarks_3d
                                         if side in frame and frame[side].source != "interpolated" else np.full((21, 3), np.nan)
                                         for frame in track.wrist_crop_hands]) if track.wrist_crop_hands else None
                                        for side in ("left", "right")},
                                     source="raw detection", asset_id=review.sources[uid]["asset_id"])
        for ci, uid in enumerate(camera_uids):
            if cancel is not None and cancel.is_set():
                raise RuntimeError("Review detection cancelled.")
            clip = package.motion_clips[uid]
            lo, hi = review.local_time(uid, start_s), review.local_time(uid, end_s)
            if current:
                reader = ReviewVideoReader(clip.path, cache_frames=1)
                try:
                    frame = reader.at(lo, max_width=None)
                finally:
                    reader.close()
                frames = [(0, frame[0], frame[1])] if frame else []
            elif detector is not None:
                frames = iter_video_frames(clip.path, start_pts_s=max(0., lo), end_pts_s=hi)
            else:
                frames = []
            for index, pts, image in frames:
                if cancel is not None and cancel.is_set():
                    raise RuntimeError("Review detection cancelled.")
                body, left, right = np.zeros((26, 3), np.float32), None, None
                if pose is not None:
                    prediction = pose.infer_frame(image, index, pts)
                    body, left, right = prediction.body26_uvc, prediction.left_hand21_uvc, prediction.right_hand21_uvc
                elif uid in output.cameras:
                    previous = output.frame(uid, pts)
                    if previous:
                        body, left, right = previous["body"], previous["hand:left"], previous["hand:right"]
                tags = [{"id": tag.tag_id, "corners": tag.corners_uv.tolist(), "confidence": tag.confidence,
                         "dictionary": tag.dictionary_name} for tag in detector.detect(image)] if detector else None
                hand_local = {}
                if pose is not None:
                    from engine.hands.hand_tracker import HandCaptureEngine
                    engine = HandCaptureEngine()
                    try:
                        from multi_video_studio.roi_tracking import crop_override, roi_for_frame
                        edited_body, _ = review.apply_points(uid, [pts], body[None])
                        crops = engine.extract_wrist_crops_from_body26(image, edited_body[0])
                        for side in ("left", "right"):
                            roi = roi_for_frame(review, uid, side, pts, edited_body[0])
                            if roi is not None:
                                override = crop_override(image, roi)
                                if override is not None:
                                    crops[side] = override
                        hands = engine.infer_hands_from_crops(crops)
                        for side, hand in hands.items():
                            hand_local[side + "_local"] = hand.landmarks_3d
                            if hand.landmarks_2d is not None:
                                values = np.c_[hand.landmarks_2d, np.full(21, hand.confidence)]
                                if side == "left":
                                    left = values
                                else:
                                    right = values
                    finally:
                        engine.close()
                output.put_frame(uid, pts, body, left, right, tags, **hand_local)
                if progress and index % 30 == 0:
                    frac = (ci + np.clip((pts - lo) / max(hi - lo, .01), 0, 1)) / len(camera_uids)
                    progress(float(frac), f"{uid}: detected source frame {pts:.3f}s")
        return output
    finally:
        if pose is not None:
            pose.close()
        if tracker is not None:
            tracker.close()


def refine_review_camera(calibration, uid, observations, review):
    from multi_video_studio.pyav_sync import CameraSyncEstimate, resample_synchronized_tracks
    from multi_video_studio.camera_refinement import refine_camera_from_views

    if not review.alignment_accepted:
        raise ValueError("Align and review the videos before refining camera geometry.")
    if uid == review.reference_camera:
        raise ValueError("The reference camera anchors this take. Select another camera to refine.")
    pts, tracks, clocks = {}, {}, {}
    for camera_uid, camera in observations.cameras.items():
        if camera_uid not in calibration.cameras or camera_uid not in review.clocks or len(camera["pts"]) < 2:
            continue
        pts[camera_uid] = camera["pts"]
        tracks[camera_uid], _ = review.apply_points(camera_uid, camera["pts"], camera["body"])
        if camera_uid == uid:
            # A globally excluded camera can still be repaired. Explicit bad
            # observation ranges remain excluded from the calibration fit.
            for exclusion in review.exclusions:
                if exclusion["camera_uid"]==uid and exclusion["channel"] in {"body","all"}:
                    mask=(camera["pts"]>=exclusion["start_s"]) & (camera["pts"]<=exclusion["end_s"])
                    tracks[camera_uid][mask,:,2]=0
        else:
            tracks[camera_uid] = review.apply_exclusions(camera_uid, camera["pts"], tracks[camera_uid])
        intrinsic = calibration.cameras[camera_uid].intrinsics
        source = review.sources[camera_uid]
        tracks[camera_uid][..., 0] *= intrinsic.image_width / source["width"]
        tracks[camera_uid][..., 1] *= intrinsic.image_height / source["height"]
        clocks[camera_uid] = CameraSyncEstimate(**review.clocks[camera_uid])
    if uid not in tracks:
        raise ValueError("Track this camera's selected range before recalibrating it.")
    _, aligned, _ = resample_synchronized_tracks(clocks, pts, tracks, target_fps=15)
    return refine_camera_from_views(calibration, uid, aligned)
