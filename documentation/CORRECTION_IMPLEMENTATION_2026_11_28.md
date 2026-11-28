# Correction implementation and validation — 28 November 2026

The review workflow, marker constraints, hand fitting, character contact tools and verified 3D exports are implemented. This is a working correction pass, not a claim that every requirement in the [42-item correction sheet](FINAL_CORRECTION_SHEET.md) has passed its complete acceptance criteria. The [tracker](FINAL_CORRECTION_TRACKER.csv) separates implemented features from remaining validation.

## Implemented workflow

1. Import the camera videos, check waveform timing, inspect shared playback and accept or adjust timing. A camera can be made the time reference. Presentation timestamps, drift and native frame indices remain attached to the source.
2. Open Cameras & calibration, Markers & body, or Hands & fingers. Choose one, all or selected preview cameras independently of solve inclusion. Exclude body/hand observations by camera or time range; refine one camera while retaining trusted geometry.
3. Select/drag anatomical points, mark missing observations, restore detections, undo/redo and save portable review packages. Manual corrections survive re-solving. Draw a hand crop for a bounded interval; it follows the wrist and retracks hands while reusing body inference.
4. Use physical markers enables AprilTag-assisted solving. Detect or place tag centers/corners, assign anatomical names and measured square sizes, and calibrate marker-to-joint offsets from a neutral body estimate or enter measured offsets. Marker-only reconstruction requires supplied geometry and sufficient calibrated, visible limb tags; it cannot silently fall back to AI.
5. Inspect native hand crops, per-camera model estimates, interpolated previews and fused hand articulation. Small palms, missing data and overlapping hands in a view are identified. Centering changes the preview only. The camera subset selector uses a popup, point/crop tools expand when needed, and import controls stay on the video tab to leave usable preview space at 1366 x 768.
6. In Blender, Analyze evaluates the character's deformed meshes. Fix range creates a separate action/rig/mesh copy; Compare switches between copies. Contact intent, visual baking of control rigs and editable BLEND copies are available. Surface crossings and capped-region containment estimates have separate labels. Added contact corrections are temporally regularized before final validation; control rigs, Euler-keyed animation and active NLA layers require a visual bake first.
7. FBX, GLB and glTF export a scoped armature and bound meshes. Choose motion only, character animation or a baked static mesh. To retain Blender contact corrections, use the saved scene's existing action. Desktop export runs in a cancellable background process and publishes a new file only after validation. Blender add-on exports perform the same checks synchronously.

## Engineering choices

Keep PySide6, PyAV, OpenCV, ONNX Runtime CUDA, NumPy/SciPy and Blender. This avoids a framework rewrite and addresses demonstrated timing, visibility, geometry and rig problems. Detailed wrist crops now run in every useful camera. Palm orientation is separate from camera-local hand estimates; continuous swing/twist fitting distributes forearm roll without moving reliable elbow/wrist endpoints. Fingers use calibrated fixed lengths and bounded joints. Missing fingertips remain missing; interpolation no longer counts as measured coverage.

No new AI model is declared more accurate without labeled comparisons. RTMPose Hand5 versus MediaPipe benchmarking, controlled marker recordings and gesture/contact reference captures remain required. The supplied recordings contain views with very small tracked palms and projected hand overlap; those images do not establish reliable fingertip depth by themselves.

Export sampling and deformation checks follow the [Blender export API](https://docs.blender.org/api/5.1/bpy.ops.export_scene.html), with behavior checked in installed Blender 5.2.3. File validation checks every integer/halfway pose, hierarchy, duration, supported shape-key motion and sampled deformed surfaces. This checks preservation of the chosen animation, not capture accuracy against ground truth. Static auxiliary armatures before the main skin are baked into temporary basis/morph geometry, with the original evaluated stack retained as reference. Moving auxiliary rigs require a fuller bake and produce an explicit error. The original character and morph actions remain intact.

## Reproducible evidence

The three supplied videos remain unchanged. Local artifacts are under `D:\2026\nuranova-mocap\Mcapn_results\correction_sheet_v1`.

- `validation.json`: the full 2,309-frame, 30 FPS take; waveform matches across all three cameras; camera reprojection 6.82 px normalized to 1920-pixel width; foot consistency 21.68 → 4.55 mm, P95 17.74 mm. These are consistency metrics, not reference motion accuracy.
- `tests_release_final.xml`: 105 collected tests: 104 passed, one optional Pose2Sim-data benchmark skipped because its external footage is unavailable. Coverage includes Qt editing/portable packages, VFR decoding, point validity, anchored camera refinement, wrist continuity, real Blender control-rig baking, non-uniform scale, FBX/GLB/glTF round trips, static export and cancellation.
- `review_hands_768.png`, `review_cameras_768.png` and `review_exports_768.png`: actual desktop previews using the supplied footage at 1366 x 768.
- `Mcapn_Review.blend`: complete body/finger retarget on a copy of the supplied character, prepared through Blender MCP.
- `character_review/contacts_before.json`: indexed original mesh failures over every frame. The batch worker writes `character_contact_report.json` and a separate `Mcapn_Character_Reviewed.blend` after correction and intermediate-frame validation. An incomplete worker or a report containing residual contacts is not a clean validation result.
- `Mcapn_Tracked.glb.validation.json` and `Mcapn_Tracked.fbx.validation.json`: full-take character exports, each checked at all 4,617 integer/halfway poses. Maximum sampled skin differences are 0.0221 mm (GLB) and 0.0254 mm (FBX); all 55 bones and 60 morph channels are retained. Supplemental `Mcapn_Tracked.glb.orientation.json` checks the full hierarchy and world deformation matrices. These files contain the stable retarget before the experimental surface-correction copy, whose residual contacts need separate review.
- `addon_manifest.json`: source, installed add-on and ZIP payload hash checks for Blender bridge 1.3.0.

Run the relevant checks with:

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe scripts/validate_cinema_take.py <video-folder> <output-folder> --calibration <camera-profile.json>
```

Character correction is bounded and reports infeasible contacts. It does not guarantee collision-free motion on every rig, infer intent perfectly, repair arbitrary skin weights automatically, or validate detail hidden in the recordings. Additional wardrobe/accessory rigs, twist-bone adapters, labeled model comparisons and physical marker captures remain acceptance work. Android builds and phone hardware were not rebuilt or newly validated in this pass.

Rendered synthetic AprilTag videos reconstruct 12 tagged limb joints through the real detector and solver without creating an AI tracker. Physical marker recordings remain untested. A replaced source cannot inherit the old camera slot's annotations/calibration; changing a source after review blocks solving.

The additional cross-stage review-solve test passes: a globally excluded camera with deliberately corrupted tracks cannot veto trusted calibration, and a native wrist pin reaches the actual solver without modifying raw observations. Quoted/Unicode glTF dependency relocation, motion-only GLB and persistent static morph weights are also covered by the Blender integration fixture.

The earlier repair remains available under `quality_repair/final`: its stance-slide mean/P95 was 2.89/9.55 mm, compared with 4.55/17.74 mm in this pass. This pass improves review and hand constraints; it is not a demonstrated improvement on every motion metric. Finger coverage is now 74.1% because missing/interpolated estimates no longer count as measured observations.
