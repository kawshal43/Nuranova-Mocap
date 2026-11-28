"""Run a real multi-video take and retain reproducible quality evidence outside source clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.math_core.camera_model import MultiCameraCalibration
from multi_video_studio.pipeline import CinemaPipelineConfig, MultiVideoCinemaPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    config = CinemaPipelineConfig(calibration_source="active_studio" if args.calibration else "auto")
    calibration = MultiCameraCalibration.load(args.calibration) if args.calibration else None
    result = MultiVideoCinemaPipeline(config).run_from_source(args.source, args.output,
        existing_calibration=calibration, take_name=args.source.name,
        progress_cb=lambda fraction, message: print(f"{fraction:.3f} {message}", flush=True),
        cancel_event=threading.Event())
    report = {"scorecard": result.scorecard.__dict__, "processing": result.document.processing_config,
              "elapsed_s": result.elapsed_s, "h5_path": str(result.h5_path)}
    destination = args.output / "validation.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"validation": str(destination), "elapsed_s": result.elapsed_s}), flush=True)


if __name__ == "__main__":
    main()
