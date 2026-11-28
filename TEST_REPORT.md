# Automated Test Report — Nuranova Mocap Studio v1.0.0

## Correction workflow and character exports — 2026-11-28

The correction pass has **104 passing tests and one optional external-footage
benchmark skipped** (105 collected). Ruff passes. Tests cover native-PTS point
editing, portable review state, replaced-source protection, camera refinement,
synthetic video marker-only reconstruction without an AI tracker, wrist/finger
constraints, Blender contact copies and continuity, control-rig baking, and
scoped FBX/GLB/glTF exports with cancellation.

The supplied character's complete 2,309-frame take passes FBX and GLB round trips
at 4,617 integer/halfway poses, retaining 55 bones and 60 morph channels. Maximum
sampled skin differences are 0.0254 mm and 0.0221 mm respectively. These checks
verify export preservation; remaining mesh contacts and capture accuracy have
separate acceptance requirements. Detailed evidence and current limitations are
in the [implementation report](documentation/CORRECTION_IMPLEMENTATION_2026_11_28.md)
and [42-item tracker](documentation/FINAL_CORRECTION_TRACKER.csv).

## Motion Quality Repair — 2026-11-27

The supplied videos in `D:\2026\nuranova-mocap\Mcapn` were processed repeatedly at
full frame coverage on the RTX 3060 Ti. The open Blender 5.2.3 scene was inspected
and updated through its **actual MCP server**, including the installed bridge.
The original videos and `D:\2026\animation\Untitled.blend` were preserved.

### Measured results

| Check | Original / before repair | Repaired |
|---|---:|---:|
| Largest body rotation between Blender frames | 53.32 degrees | 21.11 degrees |
| Largest finger rotation between Blender frames | 60.26 degrees | 13.19 degrees |
| Bone/frame jumps above 30 degrees | 1,187 | 0 |
| Worst wrist fitting error in the intermediate IK solve | 48.7 cm | 9.4 cm |
| Right / left wrist P95 fitting error | 10.8 / 13.3 cm | 5.9 / 5.3 cm |
| Estimated stance drift within the repaired pipeline | 34.13 mm before contact cleanup | 2.89 mm; P95 9.55 mm |

The Blender comparison evaluates the user's original `Mcapn_Final_Cinema` action
and the repaired character action. They contain 2,310 and 2,309 frames respectively.
Wrist fitting compares the previous and revised IK solver against the same optimized
model landmarks, before floor leveling. These targets are **not ground truth**.

- The export contains **2,309 frames at 30 fps**, 55 body/finger bones, and mapped
  facial channels on a copy of the supplied character.
- Both source and character rigs passed evaluation at **every integer frame and
  all 2,308 halfway frames**. Positions, scales, and quaternions are finite; the
  largest halfway interpolation discrepancy is below 0.00005 degrees.
- Original-PTS waveform alignment: cam_02 **+5.7722 s**, cam_03 **-7.2606 s**.
  Independent window consensus retained 14 and 9 matches, with residuals of about
  4.0 and 3.0 ms. Drift was not supported by sufficient evidence and was not applied.
  A weak five-window bridge disagreed by about 43 ms and was rejected.
- Camera geometry was checked for all three views. Three focal-length starting
  candidates were compared by cross-view prediction. A separately validated camera
  solution from the same footage performed better and was retained for this export.
- Mean model-landmark reprojection error: **6.64 px**, normalized to 1920 image width.
  Reported body confidence coverage is 90.8%; coverage is not pose accuracy.
- Bridge **1.2.2** was installed, reloaded in the live Blender session, and packaged
  in `installers/NuranovaMocapBridge.zip`. The previous installation is backed up.

### Fixes and verification

Final full Python suite: **89 passed, 1 skipped**, in 80.52 seconds, with one
upstream Starlette deprecation warning. Ruff: **all checks passed**; Git whitespace
validation: clean.

Multi-window audio consensus replaces the fragile two-window drift estimate.
Camera validation now rejects missing or mismatched third views. Safe observation
caches allow reconstruction experiments without rerunning inference. The limb
solver preserves measured segment directions instead of displacing endpoints to
rate-limit twist, and rotation smoothing operates on SO(3). Blender imports use
linear dense curves, continuous quaternion signs, and muted NLA archive tracks.

The test suite covers calibration, synchronization, network/project handling,
tracking, reconstruction, cleanup, desktop navigation, and real headless Blender
import, retargeting, fractional-frame playback, FBX and BVH export. The optional
Pose2Sim footage benchmark is skipped because its reference videos are absent.
Ruff and Git whitespace validation pass. Android rebuilding, physical phone
capture, and clean-machine installer deployment were not repeated; the Android
toolchain is unavailable on this machine.

### Review artifacts

All artifacts are under `../Mcapn_results/quality_repair`:

- `Mcapn_Repaired.blend`: the saved and currently open repaired character scene.
- `Mcapn_aligned_review.mp4` and `.jpg`: all three timestamp-aligned sources beside
  the reconstructed pose; complete 76.93-second review at 10 preview fps.
- `motion_quality_comparison.png` / `.json`: original/repaired all-frame comparison.
- `blender_repaired_audit.json` and `blender_repaired_keyframes.npz`: full 55-bone
  source/target audit with subframe evaluation.
- `final/`: HDF5, Blender JSON, BVH, validated camera calibration, and processing
  report with per-bone continuity and skeletal fitting measurements.
- `Mcapn_Reviewed_Project/project.nmocap`: desktop project containing the repaired
  motion and calibration. Reopening it and scrubbing nine positions through the
  real 2,309-frame take passed using the Windows Qt backend; a screenshot is saved
  as `desktop_repaired_take.png`.
- `scene_original_copy.blend` and `installed_addon_backup_1_2_1/`: preserved backups.

These changes substantially improve this take, but cannot certify perfect motion
capture from blurred, occluded footage. The third phone records roughly 15.8 fps,
which limits fast motion reconstruction. Boardless absolute scale uses an assumed
1.72 m height; accurate physical scale and reliable lens geometry require measured
calibration. Hidden fingers and face details remain estimates. Grade B is a
consistency score, not proof of reference capture accuracy.

## Earlier Pipeline Validation — 2026-11-27

Environment: Windows x64, Python 3.11, PySide6 6.7.3, ONNX Runtime 1.19.2,
NVIDIA RTX 3060 Ti, Blender 5.2.3 LTS.

- Full Python suite: **61 passed, 1 skipped**, one upstream Starlette deprecation warning.
- The skipped Pose2Sim benchmark requires footage absent from this checkout.
- Ruff: **all checks passed**. Git whitespace validation: clean.
- New regressions cover ZIP escapes, project isolation, motion edit persistence,
  actor proportions, missing hand tracks, exact timestamp alignment, VFR timing,
  shared hand/body timelines, cancellation, upload bounds, model cache persistence,
  laptop layout, and retaining all imported camera views without a studio calibration.
- Blender integration also verifies fractional frame rates (59.94 fps).

### Supplied Three-Video Take

Source: `D:\2026\nuranova-mocap\Mcapn`. Original videos were left unchanged.

| Camera | Source | Resolution | Average FPS | Duration |
|---|---|---|---:|---:|
| cam_01 | 20261127_015632.mp4 | 3840 × 2160 | 30.01 | 100.61 s |
| cam_02 | VID_20261127_015637.mp4 | 1920 × 1080 | 30.06 | 76.98 s |
| cam_03 | VID_20261127_015801.mp4 | 1920 × 1080 | 15.83 | 96.42 s |

Audio alignment estimated cam_02 at +5.771 s and cam_03 at −7.228 s relative to
cam_01, with clock drift compensation on cam_03. All three sources were tracked
without frame skipping using RTMW WholeBody on CUDA. Camera geometry was estimated
without a board, using the default 1.72 m actor height.

- Output: **2,309 frames at 30 fps**, spanning 76.93 seconds of shared recording time.
- Runtime: 660.5 seconds, including body, hand, face, calibration, and cleanup stages.
- Mean reprojection error: **6.44 px**, normalized to a 1920-pixel image width.
- Estimated stance drift: **16.76 mm → 1.95 mm**; final P95: 3.93 mm.
- Floor penetration: 0 mm; measured body confidence coverage: 91.5%.
- Joint positions and matrices are finite; timestamps increase strictly.
- Maximum quaternion norm error: approximately 6 × 10⁻⁸.
- The real export was loaded in Blender and saved with **55 bones and 2,309 frames**.
- Sampled source frames and reconstructed poses were visually inspected.

Artifacts are in `../Mcapn_results/full`: HDF5 motion, Blender JSON, BVH,
camera calibration, processing and validation reports, a source/pose comparison
image, and `Mcapn_verified.blend`. UI screenshots are in `../Mcapn_results`.

These metrics describe consistency, not ground-truth motion accuracy. Boardless
scale depends on the entered height, and occluded limbs or fingers can remain
uncertain. Physical phone capture, Android rebuilding, and a clean-machine Windows
installer were not revalidated in this run; the Android toolchain is absent here.

## Historical Validation — 2026-11-24

**Date:** 2026-11-24  
**Environment:** Windows 10 x64 · Python 3.11.9 · PySide6 6.7.3 · Blender 5.0.1 · JDK 21.0.12 · Android SDK 36 · Inno Setup 6

---

## 1. Python Desktop, Engine, Blender & UI Suite (`pytest`)

- **Total Executed:** 18 tests
- **Passed:** **18 / 18 (100%)**
- **Failed:** 0
- **Skipped:** 0

| Test Module | Category | Tests | Result | What Was Verified |
|---|---|---:|---|---|
| `tests/unit/test_math_calibration_triangulation.py` | Unit + Synthetic | 4 | **PASS** | Quaternion/Kabsch algebra, ChArUco PDF/PNG generation & OpenCV corner detection, 4-camera outlier-rejecting triangulation, metric board Bundle Adjustment + Z-Up floor alignment + calibration invalidation check |
| `tests/unit/test_motion_format.py` | Unit | 1 | **PASS** | Versioned `.nmocap.h5` HDF5 + `.nmocap.json` sidecar + Blender JSON round-trip (raw & corrected tracks, 52 blendshapes, hands, provenance) |
| `tests/unit/test_markers_ai_face_hands.py` | Unit + Real Video | 4 | **PASS** | Valid AprilTag 36h11 generation/PDF/detection, visible subpixel spherical blobs + epipolar matching + 6-DoF rigid cluster pose, hardware CUDA/CPU probe + real Pose2Sim video inference, 16-bone hand solver + facial neutral calibration |
| `tests/unit/test_skeleton_hybrid_cleanup.py` | Unit + Synthetic | 3 | **PASS** | Actor calibration & constant bone lengths, 4-camera Hybrid Cinema reconstruction + BVH export, MAD despiking + PCHIP gap recovery + 2-bone IK foot locking (<10 mm target) |
| `tests/integration/test_network_sync_project_scalability.py` | Integration + 12-Cam Simulation | 3 | **PASS** | 12 simulated camera clients pairing, 401/403 auth rejection, partial camera availability (4 of 12), unsupported capability alternatives, resumable chunked SHA-256 upload, WebSocket preview/ping, NTP clock drift + audio clap + flash + mixed 30/60 FPS sync, portable project ZIP backup/restore & background job queue |
| `tests/integration/test_blender_bridge_real.py` | Real Blender 5.0.1 Integration | 1 | **PASS** | Actual headless `blender.exe --background` running `nuranova_mocap_bridge`: imports `.nmocap.json`, creates 25-bone Z-Up source rig, retargets onto stylized 0.72x Rigify rig with non-zero bone rolls, maps facial shape keys & hand bones, creates NLA strips, exports real `.fbx` & `.bvh` |
| `tests/end_to_end/test_real_video_pose2sim_benchmark.py` | Real Multi-Camera Video | 1 | **PASS** | Streams all 4 real Pose2Sim BSD-3-Clause multi-camera `.mp4` videos via PyAV and executes real pretrained AI pose inference |
| `tests/end_to_end/test_desktop_ui_and_pipeline.py` | Desktop UI (`pytest-qt`) | 1 | **PASS** | Launches `NuranovaMainWindow`, navigates all 20 required pages, switches Light/Dark themes, creates project, runs synthetic demo capture, scrubs Motion Editor, applies cleanup + Undo/Redo, renders all 3D viewport presets |

---

## 2. Android Unit Tests & APK Build (`Gradle 8.11.1`)

- **Command:** `.\gradlew.bat testDebugUnitTest assembleDebug --no-daemon`
- **Result:** **BUILD SUCCESSFUL**
- **Unit Tests Verified (`ClockSyncAndProtocolTest.kt`):**
  - `clockSyncRejectsHighRttOutliers`: Verified NTP lowest-RTT outlier filter (`< 2 ms` residual, `< 1 ms` offset error despite 60 ms Wi-Fi RTT spikes) — **PASS**
  - `protocolSerializationMatchesDesktopSchema`: Verified Kotlinx Serialization JSON wire format matches Python `shared_protocol/protocol.py` (`protocol_version = "1.0.0"`) — **PASS**
- **APK Artifact:** `installers/NuranovaMocapCamera.apk` (17,688,212 bytes, debug-signed test APK)

---

## 3. Static Analysis (`ruff`)

- Fixed all 53 unused imports, 1 unused variable, and 1 missing `Tuple` typing import (`F821`). Zero `F821`, `F401`, or `F841` errors remain.

---

## 4. Tests Requiring Physical Hardware (Not Yet Executed)

- Clean second-PC Windows installation/uninstallation test of `NuranovaMocapStudio_Setup.exe`
- Physical installation and Camera2/sync/optical validation on `Samsung Galaxy S20 FE`, `Samsung Galaxy S21 FE`, `Redmi Note 15`, and `Redmi Note 12` (documented in `HARDWARE_TEST_GUIDE.md`)
