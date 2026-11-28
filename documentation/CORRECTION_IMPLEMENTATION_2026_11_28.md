# Correction implementation and validation — 28 November 2026

The review workflow, marker constraints, hand fitting, character contact tools and verified 3D exports are implemented. This is a working correction pass, not a claim that every requirement in the [42-item correction sheet](FINAL_CORRECTION_SHEET.md) has passed its complete acceptance criteria. The [tracker](FINAL_CORRECTION_TRACKER.csv) separates implemented features from remaining validation.

## Review UI/UX update

Camera/body videos, left/right hand videos, their 3D hand viewers and the main character preview now open independently in resizable windows. They share the review clock and use the same live widgets when returned to the grid. A teal border, selected header and badge identify the active view; selecting a camera view updates the camera selector without changing solve inclusion. Floating previews stay live when another tab or camera subset is being inspected.

| Action | Control |
| --- | --- |
| Separate window / return | Header pop-out button, Ctrl+Enter, window close, or Return to grid |
| Fullscreen / previous arrangement | F11; Escape exits fullscreen and then returns a floating view |
| Video zoom and pan | Wheel, editable native-pixel percentage, Fit, 1:1; drag to pan |
| Annotate deliberately | Edit points checkbox / E; middle or Alt-drag still pans |
| Inspect a 3D hand | Drag to orbit; middle/right-drag to pan; wheel/percentage to zoom; menu resets orientation |
| Playback and frame inspection | Space; Left/Right source-native frames; Shift for ten; Home/End source edges in video/hand views |
| Arrange and compare | Auto or 1–3 columns; row/column and video/model dividers; Fit all, Return all and Reset views |
| Inspect image detail | Overlay/name toggles, high-contrast point labels, per-hand full frame/crop and camera-local/fused choices |
| Capture a review image | View menu → Save view image (PNG) |

Video zoom no longer resets on each incoming frame or window resize. Detailed zoom requests native-resolution decoding. Annotation coordinates remain in the source image frame; in-flight decoding cannot retarget a point drag to a different source PTS. Navigation is the default mode. Floating video windows expose their editing mode and a shared transport. Outside-recording ranges show their source time bounds, with source-edge navigation available; loading and unavailable frames have different states.

UI preferences live in `take_review/preview_layout.json` below the active settings directory, with a bounded history of 12 source sets. Per-view zoom/pan/orbit, floating geometry and splitter layout are separate from timing, annotations and calibration. Replacing a capture closes its old windows and resets the decoder's source epoch. Qt's [window geometry](https://doc.qt.io/qt-6/restoring-geometry.html) and [splitter](https://doc.qt.io/qt-6/qsplitter.html) facilities underpin the window/layout behavior.

Validation artifacts are under `D:\2026\nuranova-mocap\Mcapn_results\ui_ux_review`: 115 collected tests, 114 passed and one optional external-footage benchmark skipped; actual three-camera project checks in light/dark themes at 1366 × 768; and a rebuilt executable with matching bytecode for nine modules. Its real-footage smoke check loads a video, pops out the same selected viewer, zooms to 175% and returns it, then completes a bundled Blender export/reimport. The earlier motion/contact metrics below retain their separate acceptance status.

## Implemented workflow

1. Import the camera videos, check waveform timing, inspect shared playback and accept or adjust timing. A camera can be made the time reference. Presentation timestamps, drift and native frame indices remain attached to the source.
2. Open Cameras & calibration, Markers & body, or Hands & fingers. Choose one, all or selected preview cameras independently of solve inclusion. Exclude body/hand observations by camera or time range; refine one camera while retaining trusted geometry.
3. Select/drag anatomical points, mark missing observations, restore detections, undo/redo and save portable review packages. Manual corrections survive re-solving. Draw a hand crop for a bounded interval; it follows the wrist and retracks hands while reusing body inference.
4. Use physical markers enables AprilTag-assisted solving. Detect or place tag centers/corners, assign anatomical names and measured square sizes, and calibrate marker-to-joint offsets from a neutral body estimate or enter measured offsets. Marker-only reconstruction requires supplied geometry and sufficient calibrated, visible limb tags; it cannot silently fall back to AI.
5. Inspect native hand crops, per-camera model estimates, interpolated previews and fused hand articulation. Small palms, missing data and overlapping hands in a view are identified. Centering changes the preview only. Camera and character playback share one timestamp; solved previews outside the solved interval are labeled missing. Reopening the same source set restores its saved result. The camera subset selector uses a popup, point/crop tools expand when needed, and import controls stay on the video tab to leave usable preview space at 1366 x 768.
6. In Blender, Analyze evaluates the character's deformed meshes. Fix range creates a separate action/rig/mesh copy; Compare switches between copies. Contact intent, visual baking of control rigs and editable BLEND copies are available. Surface crossings and capped-region containment estimates have separate labels. Added contact corrections are temporally regularized before final validation, including the final frame. Local fixes retain unrelated artist curves. Control rigs, Euler-keyed or curved quaternion animation, and active NLA layers require a visual bake first.
7. FBX, GLB and glTF export a scoped armature and bound meshes. Choose motion only, character animation or a baked static mesh. To retain Blender contact corrections, use the saved scene's existing action; it uses that scene's FPS and range and does not require an active capture take. Desktop export runs in a cancellable background process and publishes a new file only after validation. Blender add-on exports perform the same checks synchronously.

## Engineering choices

Keep PySide6, PyAV, OpenCV, ONNX Runtime CUDA, NumPy/SciPy and Blender. This avoids a framework rewrite and addresses demonstrated timing, visibility, geometry and rig problems. Detailed wrist crops now run in every useful camera. Palm orientation is separate from camera-local hand estimates; continuous swing/twist fitting distributes forearm roll without moving reliable elbow/wrist endpoints. Fingers use calibrated fixed lengths and bounded joints. Missing fingertips remain missing; interpolation no longer counts as measured coverage.

No new AI model is declared more accurate without labeled comparisons. RTMPose Hand5 versus MediaPipe benchmarking, controlled marker recordings and gesture/contact reference captures remain required. The supplied recordings contain views with very small tracked palms and projected hand overlap; those images do not establish reliable fingertip depth by themselves.

`camera_local_hand_coverage.json` records finite camera-local hand-model availability over the native source frames: cam_01 left/right 7.25/8.94%, cam_02 6.31/7.52%, and cam_03 52.64/58.56%. These model estimates and the separate 74.1% fused finger-observation coverage use different criteria. Review tiles label unavailable model estimates explicitly. Controlled close hand views are still needed for gesture-level accuracy validation.

Export sampling and deformation checks follow the [Blender export API](https://docs.blender.org/api/5.1/bpy.ops.export_scene.html), with behavior checked in installed Blender 5.2.3. File validation checks every integer/halfway pose, hierarchy, duration, supported shape-key motion and sampled deformed surfaces. This checks preservation of the chosen animation, not capture accuracy against ground truth. Static auxiliary armatures before the main skin are baked into temporary basis/morph geometry, with the original evaluated stack retained as reference. Moving auxiliary rigs require a fuller bake and produce an explicit error. The original character and morph actions remain intact.

## Reproducible evidence

The three supplied videos remain unchanged. Local artifacts are under `D:\2026\nuranova-mocap\Mcapn_results\correction_sheet_v1`.

- `validation.json`: the full 2,309-frame, 30 FPS take; waveform matches across all three cameras; camera reprojection 6.82 px normalized to 1920-pixel width; foot consistency 21.68 → 4.55 mm, P95 17.74 mm. These are consistency metrics, not reference motion accuracy.
- `tests_release_final.xml`: 107 collected tests: 106 passed, one optional Pose2Sim-data benchmark skipped because its external footage is unavailable. Coverage includes Qt editing/portable packages, saved-result reopening and shared playback, VFR decoding, point validity, anchored camera refinement, wrist continuity, real Blender control-rig baking, non-uniform scale, FBX/GLB/glTF round trips, static export and cancellation.
- `review_hands_768.png`, `review_cameras_768.png` and `review_exports_768.png`: actual desktop previews using the supplied footage at 1366 x 768.
- `Mcapn_Review.blend`: complete body/finger retarget on a copy of the supplied character, prepared through Blender MCP.
- `character_review/contacts_before.json`: indexed original mesh failures over every frame. The first experimental correction copy developed runaway bone scale and was rejected as `REJECTED_UnstableScale_DoNotUse.blend`; it is excluded from the prepared project and exports. Scale/location rollback and numerical extent checks now reject this failure. Later bounded passes retain separate actions and validate intermediate poses; residual contacts remain explicit review failures.
- `character_review_local/Mcapn_Character_Review_Ready.blend`: the completed contact-correction comparison, opened and checked through live Blender MCP with bridge 1.3.0. Original, raw correction and temporally regularized actions remain available. The full mesh scan checks 7,442 integer/intermediate poses; 5,586 samples still report contacts, so the status is **review required**. The fresh keyframe audit checks all 2,309 integer frames and 2,308 halfway poses, with finite values, maximum step 20.921 degrees and zero steps above 30 degrees. Maximum local scale difference from the original is 0.00000728.
- `character_delivery_checks.json`: proof that the exported checkpoint and final comparison have identical sampled bone positions/quaternions and unchanged face actions. `Mcapn_Character_Review.fbx` and `.glb` each pass 4,617 pose checks, retain the main body mesh, 55 bones and 60 morph channels, and verify hierarchy/deformation matrices. Maximum sampled surface differences are 0.0291 mm for FBX and 0.0219 mm for GLB.
- `Mcapn_Tracked.glb.validation.json` and `Mcapn_Tracked.fbx.validation.json`: full-take character exports, each checked at all 4,617 integer/halfway poses. Maximum sampled skin differences are 0.0221 mm (GLB) and 0.0254 mm (FBX); all 55 bones and 60 morph channels are retained. Supplemental `Mcapn_Tracked.glb.orientation.json` checks the full hierarchy and world deformation matrices. These files contain the stable retarget before the experimental surface-correction copy, whose residual contacts need separate review.
- `addon_manifest.json`: source, installed add-on and ZIP payload hash checks for Blender bridge 1.3.0.
- `frozen_smoke.json` and `windows_bundle_manifest.json`: the built Windows executable loads all 21 pages, detects 26 body points in a supplied-video sample with CUDA, and completes a real Blender export/reimport using its bundled worker. Run `dist/NuranovaMocapStudio/NuranovaMocapStudio.exe` with its complete folder. Inno Setup is unavailable; no new setup installer was compiled.
- `Mcapn_Correction_Review/project.nmocap`: prepared desktop project with the stable native take, source observations, review settings and verified 3D exports. The original video folder is referenced separately.
- `START_HERE.txt`, `handoff_manifest.json` and the project's `OPEN_THIS_PROJECT.txt` identify the Windows app, comparison scene and export workflow. Before/after contact CSVs map findings to Blender frames, global time and each camera's native frame/PTS. `priority_contact_intervals.csv` groups estimates of at least 20 mm for review; the full index retains smaller contacts. `prepared_project_ui_check.json` verifies actual project reopening, all three source previews and camera-local/fused hand views at 1366 x 768.

## Full-take character comparison

These figures compare the same 2,309 integer poses on the supplied character. Contact depths include capped-region containment estimates; contact labels still need intent review. The adaptive scan above also checks between those poses.

| Check | Stable retarget before contacts | Regularized correction |
| --- | ---: | ---: |
| Integer poses with reported contacts | 2,293 | 1,396 |
| Mean maximum estimated contact depth | 20.45 mm | 6.23 mm |
| P95 maximum estimated contact depth | 71.07 mm | 41.83 mm |
| Largest integer-pose estimated contact depth | 99.51 mm | 89.41 mm |
| Target-character flat-stance drift mean / P95 | 25.98 / 64.31 mm | 8.96 / 31.58 mm |
| Target-character heel-contact drift mean / P95, including pivots | 38.18 / 108.06 mm | 28.42 / 93.43 mm |

At frame 1777, the tested hand/thigh depth falls from 62.49 to 5.17 mm. The larger remaining case is the hand-to-head gesture around frame 1451 (global 54.122 seconds). All three source views show a forehead touch with mostly hidden finger detail. The supplied character has a larger head relative to the arms; the current bounded limb correction leaves palm/finger containment there. Character-specific contact goals and palm orientation still need work. The largest adaptive estimate is 92.26 mm at frame 1451.25. This build does not pass collision-free acceptance.

The priority index contains 95 interval groups with estimates of at least 20 mm, including remaining hand/torso and hand/thigh cases. That threshold prioritizes review; it does not declare smaller contacts acceptable. Both original and corrected actions remain available for checking intent before further pose changes.

The stance figures here measure the target character's foot joints using the source contact masks. They are separate from the native skeleton's 4.55 mm pipeline slide figure. Neither metric is ground-truth accuracy. Workbench before/after images at frames 1451 and 1777 are saved as `character_final_before_*.png` and `character_final_after_*.png`; the forehead issue remains visible in those images.

Run the relevant checks with:

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe scripts/validate_cinema_take.py <video-folder> <output-folder> --calibration <camera-profile.json>
```

Character correction is bounded and reports infeasible contacts. It does not guarantee collision-free motion on every rig, infer intent perfectly, repair arbitrary skin weights automatically, or validate detail hidden in the recordings. Additional wardrobe/accessory rigs, twist-bone adapters, labeled model comparisons and physical marker captures remain acceptance work. Android builds and phone hardware were not rebuilt or newly validated in this pass.

The supplied asset's verified FBX/GLB exports contain the primary rig and its one bound body mesh. Separate eye, hair and wardrobe rigs are outside this verified export scope.

Rendered synthetic AprilTag videos reconstruct 12 tagged limb joints through the real detector and solver without creating an AI tracker. Physical marker recordings remain untested. A replaced source cannot inherit the old camera slot's annotations/calibration; changing a source after review blocks solving.

The additional cross-stage review-solve test passes: a globally excluded camera with deliberately corrupted tracks cannot veto trusted calibration, and a native wrist pin reaches the actual solver without modifying raw observations. Quoted/Unicode glTF dependency relocation, motion-only GLB and persistent static morph weights are also covered by the Blender integration fixture.

The earlier repair remains available under `quality_repair/final`: its stance-slide mean/P95 was 2.89/9.55 mm, compared with 4.55/17.74 mm in this pass. This pass improves review and hand constraints; it is not a demonstrated improvement on every motion metric. Finger coverage is now 74.1% because missing/interpolated estimates no longer count as measured observations.
