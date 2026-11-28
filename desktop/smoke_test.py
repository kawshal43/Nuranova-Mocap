"""Isolated CLI check for the actual standalone application and its Blender worker."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PySide6.QtCore import QTimer


def schedule_smoke_test(app, output, window, workdir, startup_report):
    from desktop.settings.app_settings import detect_blender_executable
    from engine.data_format.motion_io import MotionCaptureDocument, SkeletalMotionTrack, export_blender_json
    from engine.export.blender_job import run_blender_export
    from engine.skeleton.rig_definition import BONE_NAMES, PARENT_INDICES, DEFAULT_REST_OFFSETS_M, compute_forward_kinematics_sequence

    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"passed":False, "frozen":bool(getattr(sys,"frozen",False))}
    controller = window.ctrl
    folder = workdir
    result["startup"] = startup_report

    def verify():
        try:
            assert startup_report["passed"] and startup_report["completed"] == startup_report["total"]
            assert startup_report["page_count"] == 21
            result["pages"] = list(window.pages)
            assert "Multi-Video Cinema Lab" in window.pages
            assert "Animation Exports" in window.pages
            bundle = Path(__file__).resolve().parents[1]
            assert (bundle/"scripts/blender_export_job.py").is_file()
            assert (bundle/"blender_addon/nuranova_mocap_bridge/character_export.py").is_file()
            window.navigate_to("Multi-Video Cinema Lab")
            window.stack.finish_transition()
            app.processEvents()
            window.grab().save(str(output.with_suffix(".png")))
            video = os.environ.get("NURANOVA_SMOKE_VIDEO")
            if video:
                from engine.ai_tracking.pose_tracker import AIPoseTracker2D
                from engine.video.video_io import iter_video_frames
                tracker = AIPoseTracker2D("rtmlib_wholebody","performance",True)
                try:
                    target_time = float(os.environ.get("NURANOVA_SMOKE_TIME","0"))
                    index, pts, image = next(iter_video_frames(Path(video),start_pts_s=target_time,end_pts_s=target_time+.5))
                    detected = tracker.infer_frame(image,index,pts)
                    values = detected.body26_uvc
                    providers = tracker.active_providers()
                    result["video_inference"] = {"body_points_over_02":int((values[:,2]>.2).sum()),
                        "providers":providers,"backend":tracker.backend,"pts_s":float(pts),"shape":list(values.shape)}
                    assert values.shape == (26,3) and (values[:,2]>.2).sum()>=10
                    if controller.hardware.cuda_usable:
                        assert "CUDAExecutionProvider" in providers, "CUDA inference unexpectedly fell back to CPU."
                finally:
                    tracker.close()
                page = window.pages["Multi-Video Cinema Lab"]
                page._load_source([Path(video)])
                page.workflow_tabs.setCurrentIndex(1)
                page.review.time_slider.setValue(round(float(pts)*1000))
                deadline = time.monotonic()+20
                while time.monotonic()<deadline:
                    app.processEvents()
                    if page.review.last_frames and all(page.review.last_frames.values()):
                        break
                    time.sleep(.02)
                assert page.review.last_frames and all(page.review.last_frames.values())
                uid = next(iter(page.review.package.motion_clips))
                tile = page.review.tiles[("cameras",uid,None)]
                view = tile.overlay
                tile.video_panel.pop_out()
                app.processEvents()
                assert tile.video_panel.floating_window is not None
                view.set_zoom_percent(175)
                assert abs(view.zoom_percent()-175)<.01
                assert tile.video_panel.surface.property("previewSelected")
                tile.video_panel.floating_window.grab().save(str(output.with_name(output.stem+"_preview.png")))
                tile.video_panel.return_to_grid()
                assert tile.video_panel.floating_window is None and tile.overlay is view
                result["review_ui"] = {"source_frame_loaded":True,"popout_return_same_view":True,
                    "selection_visible":True,"native_zoom_percent":view.zoom_percent()}
            count = 8
            root = np.tile([0.,0.,.96],(count,1))
            rotations = np.tile([0.,0.,0.,1.],(count,len(BONE_NAMES),1))
            positions, matrices = compute_forward_kinematics_sequence(root,rotations,DEFAULT_REST_OFFSETS_M)
            track = SkeletalMotionTrack(root,rotations[:,0],rotations,matrices,positions,np.ones((count,len(BONE_NAMES))))
            document = MotionCaptureDocument("smoke","smoke","smoke","smoke",30.,
                (np.arange(count)/30*1e9).astype(np.int64),BONE_NAMES,PARENT_INDICES,DEFAULT_REST_OFFSETS_M,track)
            payload = export_blender_json(document,Path(folder)/"motion.json")
            blender = detect_blender_executable()
            if not blender:
                raise RuntimeError("Blender is required for the standalone export smoke check.")
            result["blender_export"] = run_blender_export(blender,{"motion_json":str(payload),
                "variant":"motion","output":str(Path(folder)/"smoke.glb"),"frame_start":1,"frame_end":count},timeout_s=180)
            assert result["blender_export"]["passed"]
            result["passed"] = True
        except Exception:
            result["error"] = traceback.format_exc()
        finally:
            window.close()
            output.write_text(json.dumps(result,indent=2),encoding="utf-8")
            app.exit(0 if result["passed"] else 1)

    QTimer.singleShot(500, verify)
