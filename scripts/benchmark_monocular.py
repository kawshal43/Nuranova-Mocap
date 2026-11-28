"""Reproduce native-video single-camera solves and source/round-trip checks.

Example: python scripts/benchmark_monocular.py video.mp4 --output results --start 20 --end 28
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["auto", "mediapipe", "rtmw3d"], default="auto")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--mode", choices=["performance", "balanced", "lightweight"], default="performance")
    parser.add_argument("--start", type=float)
    parser.add_argument("--end", type=float)
    parser.add_argument("--hands-every", type=int, default=2)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    from engine.data_format.motion_io import load_motion_file
    from multi_video_studio.pipeline import CinemaPipelineConfig, MultiVideoCinemaPipeline
    from shared_protocol.atomic_io import atomic_write_text

    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for video in args.videos:
        before = file_hash(video)
        cfg = CinemaPipelineConfig(single_camera_backend=args.backend, prefer_gpu=not args.cpu,
            ai_mode=args.mode, single_start_s=args.start, single_end_s=args.end,
            single_hand_interval=args.hands_every,
            tracking_cache_dir=None if args.no_cache else str(args.output/"tracking_cache"))
        result = MultiVideoCinemaPipeline(cfg).run_from_source(video, args.output/video.stem,
            take_name=video.stem,
            progress_cb=lambda fraction, message: print(json.dumps({"progress":round(fraction, 3),
                                                                    "stage":message}), flush=True))
        restored = load_motion_file(result.h5_path)
        native = next(iter(restored.observations_2d.values()))
        difference = float(np.max(abs(restored.corrected_motion.joint_positions-
                                      result.document.corrected_motion.joint_positions)))
        assert before == file_hash(video), "Original video changed"
        assert difference == 0, "H5 round trip changed poses"
        assert np.all(np.diff(native["native_pts_s"]) > 0), "Native timestamps are not ordered"
        reports.append({"source":video.name, "sha256":before, "source_unchanged":True,
            "elapsed_s":result.elapsed_s, "native_frames":len(native["native_pts_s"]),
            "output_frames":restored.num_frames, "fps":restored.fps,
            "h5_roundtrip_max_difference_m":difference, "scorecard":result.scorecard.__dict__,
            "keyframe_audit":restored.processing_config["keyframe_audit"],
            "single_camera":restored.processing_config["single_camera"],
            "reference_3d_accuracy_measured":False})
        atomic_write_text(args.output/"benchmark.json", json.dumps(reports, indent=2, allow_nan=False))
    print(json.dumps({"passed":True, "videos":len(reports), "report":str(args.output/"benchmark.json")}))


if __name__ == "__main__":
    main()
