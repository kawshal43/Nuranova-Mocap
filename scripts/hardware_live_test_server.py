"""Live hardware validation server for real connected Android smartphones.

Runs StudioNetworkServer on 0.0.0.0:8765 with a known test PIN and periodically writes
real measured device telemetry, Camera2 capabilities, preview frame counts, NTP clock sync metrics,
and verified SHA-256 uploaded files to a JSON report.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import uvicorn

from desktop.camera_manager.manager import CameraManager
from desktop.network.server import StudioNetworkServer


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("e:/Backup/KAWSHAL/Nuranova-Mocap/benchmark_datasets/live_hw_test")
    pin = sys.argv[2] if len(sys.argv) > 2 else "654321"
    duration_s = int(sys.argv[3]) if len(sys.argv) > 3 else 45

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "live_hw_report.json"

    cm = CameraManager(max_body_cameras=12)
    cm._active_pin = pin
    cm._pin_expires_at = time.monotonic() + 3600.0

    server = StudioNetworkServer(cm, storage_root=out_dir / "uploads", port=8765)
    preview_counts: dict[str, int] = {}
    orig_upd = cm.update_preview_frame

    def tracked_preview(did: str, data: bytes) -> None:
        preview_counts[did] = preview_counts.get(did, 0) + 1
        orig_upd(did, data)

    cm.update_preview_frame = tracked_preview  # type: ignore[method-assign]

    config = uvicorn.Config(server.app, host="0.0.0.0", port=8765, log_level="warning")
    uv_srv = uvicorn.Server(config)
    t = threading.Thread(target=uv_srv.run, daemon=True)
    t.start()

    start = time.monotonic()
    while time.monotonic() - start < duration_s:
        summaries = [c.model_dump() for c in cm.list_summaries()]
        transfers = [
            {
                "transfer_id": sl.transfer_id,
                "take_id": sl.take_id,
                "file_kind": sl.file_kind,
                "path": str(sl.dest_path),
                "received_bytes": sl.received_bytes,
                "total_size_bytes": sl.total_size_bytes,
                "verified": sl.verified,
                "sha256": sl.computed_sha256,
            }
            for sl in server._transfers.values()
        ]
        report = {
            "elapsed_s": round(time.monotonic() - start, 2),
            "pin": pin,
            "preview_frame_counts": preview_counts,
            "cameras": summaries,
            "transfers": transfers,
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        time.sleep(1.0)

    uv_srv.should_exit = True
    t.join(timeout=3)


if __name__ == "__main__":
    main()
