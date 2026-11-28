"""Standalone CLI for Nuranova Multi-Video Cinema Mocap Suite.

Usage:
    python -m multi_video_studio.cli process path/to/take.zip --out ./exports
    python -m multi_video_studio.cli process path/to/videos_folder --board A3 --height 1.74 --out ./exports
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from multi_video_studio.pipeline import CinemaPipelineConfig, MultiVideoCinemaPipeline
from engine.math_core.camera_model import MultiCameraCalibration


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(
        prog="nuranova-multi-video",
        description="Nuranova Multi-Video Cinema Mocap — Analyze synchronized multi-camera .zip or folder into zero-slide Blender mocap.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser(
        "process", help="Analyze a .zip archive or folder of synchronized multi-camera videos."
    )
    p_run.add_argument(
        "source", help="Path to a .zip archive or folder containing multi-camera MP4/MOV videos."
    )
    p_run.add_argument(
        "--out",
        default="./cinema_exports",
        help="Output directory for .nmocap.h5, _blender.json, and .bvh.",
    )
    p_run.add_argument(
        "--backend",
        default="rtmlib_wholebody",
        choices=["rtmlib_wholebody", "rtmlib_halpe26", "mediapipe_pose"],
    )
    p_run.add_argument(
        "--mode", default="performance", choices=["performance", "balanced", "lightweight"]
    )
    p_run.add_argument(
        "--board",
        default="A3",
        choices=["A3", "A4"],
        help="ChArUco board preset if calib/ videos are present.",
    )
    p_run.add_argument(
        "--height",
        type=float,
        default=1.72,
        help="Actor height in meters (for boardless self-calibration).",
    )
    p_run.add_argument("--no-fingers", action="store_true", help="Disable 30-finger capture.")
    p_run.add_argument(
        "--no-face", action="store_true", help="Disable 6-DoF head & 52 facial expression capture."
    )
    p_run.add_argument(
        "--cpu", action="store_true", help="Force CPU execution instead of CUDA GPU."
    )
    p_run.add_argument(
        "--stride", type=int, default=1, help="Track every Nth frame (1 keeps all frames)."
    )
    p_run.add_argument(
        "--fps", type=float, help="Output frame rate; defaults to the tracked median camera rate."
    )
    p_run.add_argument(
        "--calibration",
        type=Path,
        help="Existing camera calibration JSON, checked against the videos.",
    )

    args = parser.parse_args(argv)
    if args.cmd == "process":
        try:
            cfg = CinemaPipelineConfig(
                ai_backend=args.backend,
                ai_mode=args.mode,
                prefer_gpu=not args.cpu,
                board_preset=args.board,
                actor_height_m=float(args.height),
                enable_fingers=not args.no_fingers,
                enable_head_face=not args.no_face,
                frame_stride=args.stride,
                target_fps=args.fps,
                calibration_source="active_studio" if args.calibration else "auto",
            )
            calibration = MultiCameraCalibration.load(args.calibration) if args.calibration else None
            pipe = MultiVideoCinemaPipeline(cfg)
        except (ValueError, RuntimeError, OSError, KeyError, TypeError, AttributeError) as exc:
            print(f"Processing failed: {exc}", file=sys.stderr)
            return 1

        last_update = 0.0

        def _prog(frac: float, msg: str) -> None:
            nonlocal last_update
            now = time.monotonic()
            if now - last_update >= 5 or frac >= 0.64:
                print(f"[{int(frac * 100):3d}%] {msg}", flush=True)
                last_update = now

        try:
            res = pipe.run_from_source(
                Path(args.source),
                Path(args.out),
                existing_calibration=calibration,
                progress_cb=_prog,
            )
        except (ValueError, RuntimeError, OSError) as exc:
            print(f"Processing failed: {exc}", file=sys.stderr)
            return 1
        print("\n=== NURANOVA MULTI-VIDEO CINEMA QUALITY SCORECARD ===")
        print(res.scorecard.summary_line())
        print(f"Elapsed: {res.elapsed_s:.1f}s")
        print(f"Blender JSON: {res.blender_json_path}")
        print(f"Motion H5:    {res.h5_path}")
        print(f"Export BVH:   {res.bvh_path}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
