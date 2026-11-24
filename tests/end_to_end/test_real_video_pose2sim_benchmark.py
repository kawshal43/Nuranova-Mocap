"""Real-video multi-camera integration test using cited Pose2Sim BSD-3-Clause 4-camera demo footage.

NOTE: This validates actual streaming video decoding and pretrained AI inference on real recorded
multi-camera video (Pose2Sim Demo_SinglePerson, BSD-3-Clause). It is NOT a substitute for physical
smartphone validation with Galaxy S20 FE / S21 FE / Redmi Note 12 / Redmi Note 15.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from engine.ai_tracking.pose_tracker import AIPoseTracker2D
from engine.video.video_io import probe_video_metadata


def test_pose2sim_4cam_real_video_inference_pipeline():
    vid_dir = Path("e:/Backup/KAWSHAL/Nuranova-Mocap/references/pose2sim-main/Pose2Sim/Demo_SinglePerson/videos")
    vids = sorted(vid_dir.glob("cam*.mp4"))
    if len(vids) < 2:
        pytest.skip("Pose2Sim BSD-3-Clause demo videos not found under references/.")

    tracker = AIPoseTracker2D(backend="mediapipe_pose", mode="lightweight", prefer_gpu=False)
    try:
        per_cam_stats = {}
        for v in vids[:4]:
            meta = probe_video_metadata(v, scan_pts=True)
            assert meta.width > 0 and meta.height > 0 and meta.frame_count >= 12
            arr, pts, stats = tracker.process_video(v, start_frame=0, end_frame=12)
            assert arr.shape == (12, 26, 3)
            assert stats.frames_processed == 12
            assert stats.detection_rate >= 0.5
            per_cam_stats[v.stem] = stats
        assert len(per_cam_stats) == 4
    finally:
        tracker.close()
