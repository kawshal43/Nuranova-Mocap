# Project Plan — Nuranova Mocap Studio v1.0.0

**Developer:** Nuranova Solutions (Kawshal, Shanal, Shehan — https://nuranova.lk/)  
**Target Release:** v1.0.0  

## Motion Quality Repair — 2026-11-27

1. Preserve the open Blender scene and original videos. Connect to the user's live
   Blender MCP bridge and audit source/target rigs, actions, NLA blending, rest axes,
   scale, frame rate, and every evaluated frame. Record a numerical baseline.
2. Align audio with multiple independent windows, robustly fit clock offset/drift,
   check all camera pairs for closure, and preserve original video PTS. Export
   aligned visual evidence. Reject uncertain synchronization instead of silently
   guessing simultaneous starts.
3. Verify calibration against synchronized, confident correspondences. Compare
   held-out reprojection/cheirality and bone proportions; inspect occlusion and
   opposite-camera degeneracy. Refine geometry only when the evidence improves.
4. Cache raw model observations for repeatable experiments. Audit tracking, 3D,
   skeletal IK, cleanup, and retargeting separately. Fix discontinuities at their
   source and use rotation-aware temporal regularization without freezing motion.
5. Compare complete captures with the baseline using per-bone angular jumps,
   isolated spikes, displacement, contact drift, reprojection, and confidence.
   Unobserved joints remain identified as estimates. Check rotations through a
   full turn, squats, arm crossings, missing hands, and low-confidence intervals.
6. Import the improved take into a copy of the supplied character scene through
   Blender MCP. Audit all evaluated keyframes and NLA behavior; save source and
   retargeted actions, review images/video, and machine-readable audit results.
7. Run the full test suite and document measured improvements and residual limits.
   "Perfect" cannot be certified without reference motion capture; completion
   requires reproducible improvements, no known unaddressed deterministic failure,
   and explicit reporting of remaining capture uncertainty.

Stack to evaluate: Python 3.11/PySide6, PyAV for original timestamps/audio, SciPy
signal correlation and robust least squares, OpenCV calibration/triangulation,
RTMW via ONNX Runtime CUDA, rotation-aware smoothing, and Blender's native
quaternion actions via MCP. Keep a component when it passes measured checks;
replace a component only when an alternative improves this take's result.

Implemented and measured: robust multi-window audio clocks, all-camera calibration
validation and candidate selection, cached observations, endpoint-preserving limb
IK, SO(3) rotation smoothing, per-bone audits, and Blender bridge 1.2.2. The installed
extension was updated through the live MCP session. The repaired character has no
adjacent-frame jumps over 30 degrees across 55 bones, and all 2,309 keyframes plus
2,308 midpoints were evaluated on both source and target rigs. Detailed evidence
and remaining capture limitations are recorded in [TEST_REPORT.md](TEST_REPORT.md).

The selected stack remains Python 3.11/PySide6 + PyAV + OpenCV/SciPy + RTMW/ONNX
Runtime CUDA + native Blender quaternion actions. This machine's model inference
already works on CUDA; the measured failures were synchronization, geometry checks,
IK, and interpolation. A framework replacement would not address those causes.

Primary references for implementation decisions:
[SciPy correlation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html),
[robust least squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html),
[rotation splines](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.RotationSpline.html),
[OpenCV camera geometry](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html),
[Pose2Sim workflow](https://github.com/perfanalytics/pose2sim),
[RTMLib WholeBody implementation](https://github.com/Tau-J/rtmlib/blob/main/rtmlib/tools/solution/wholebody.py).

## Release Phase Roadmap

| Phase | Scope | Exit Criteria |
|---|---|---|
| **Phase 0** | Workspace audit, environment verification, licensing review, clean repository structure & pinned `.venv` | `REPOSITORY_AUDIT.md`, `ARCHITECTURE.md`, `LICENSE_REGISTER.md`, working Python 3.11 `.venv` |
| **Phase 1** | Mathematical foundation: camera projection/undistortion, ChArUco intrinsics/extrinsics, bundle adjustment, weighted DLT + RANSAC triangulation, quaternion/SE(3) algebra, synthetic validation | Unit & synthetic tests passing with sub-pixel reprojection and <1 mm synthetic triangulation error |
| **Phase 2** | Android Camera Application (`Nuranova Mocap Camera`): Kotlin, Jetpack Compose, MVVM, Camera2 API, MediaCodec/MediaMuxer H.264, Camera2 timestamps, NSD, pairing, resumable upload, Room/DataStore, all required UI screens | Gradle build compiles clean APK (`NuranovaMocapCamera.apk`) + JUnit unit tests passing |
| **Phase 3** | Desktop Camera Network & Synchronization: FastAPI/WebSocket server, mDNS Zeroconf advertiser, HMAC/token pairing, 2–12 camera manager, repeated NTP-style clock sync + drift regression, audio cross-correlation sync, flash/LED visual sync, resumable verified transfers | Integration tests passing for 2–12 simulated camera clients, clock offset estimation, resumable SHA-256 transfer |
| **Phase 4** | Physical Marker Tracking: valid ChArUco/ArUco/AprilTag PDF & PNG generator, subpixel blob/contour detector, ArUco/AprilTag detector, temporal Hungarian/Kalman tracker, cross-view epipolar association, asymmetric rigid marker clusters (6-DoF Kabsch pose) | Unit & synthetic multi-camera marker + cluster reconstruction tests passing (<2 mm synthetic error, <0.5° rotation error) |
| **Phase 5** | AI Tracking: ONNX Runtime CUDA/CPU safe hardware detector, RTMLib (`Wholebody` RTMW-133, `HALPE-26`, `Hand-21`), MediaPipe (`Pose`, `Face Landmarker` 52 blendshapes, `Hand Landmarker`), model registry with checksums & licenses, benchmark harness | Real AI inference validated on actual multi-camera video frames (Pose2Sim BSD-3-Clause demo dataset) with measured FPS & VRAM diagnostics |
| **Phase 6** | Full-Body Articulated Skeletal Solver: hierarchical 24+ bone humanoid rig (with clavicles, spine chain, forearm twist, fingers), actor calibration (T/A/neutral pose), constant bone lengths, constrained IK + numerical optimization, quaternion continuity, confidence/ambiguity flags | Synthetic ground-truth skeletal solve validated (<10 mm joint pos, <3° major joint angle, zero bone-length drift) |
| **Phase 7** | Hybrid Cinema Mode & Processing Profiles: calibrated marker-to-segment/joint offset model (never naive averaging), joint + rigid-cluster fusion optimization, Fast Preview / Standard / Cinema profiles, HDF5 `.nmocap.h5` versioned data format with full provenance | Hybrid mode tested against synthetic occluded/noisy AI + marker data showing measurable error reduction over AI-only |
| **Phase 8** | Motion Cleanup: raw vs corrected non-destructive storage, outlier removal, swap correction, confidence-aware gap interpolation, adaptive Butterworth & One Euro filters, kinematic foot-contact detection + 2-bone IK foot-locking, ground penetration clamping, root stabilization | Synthetic noisy/footskating tests verify planted-foot displacement <5 mm and bone length invariant |
| **Phase 9** | Blender Integration (`Nuranova Mocap Bridge`): Blender 4.5 LTS / 5.0+ extension, live HTTP polling via `bpy.app.timers` + offline import, source armature builder, Rigify & custom rig auto/manual bone mapper, rest-pose delta & scale compensation, facial shape-key mapping, finger animation, baking to editable keyframes/Actions/NLA, FBX/BVH export | Tested headlessly inside real installed **Blender 5.0.1** CLI creating armatures, retargeting to a secondary stylized rig, baking keyframes, and exporting FBX & BVH |
| **Phase 10** | Dedicated Facial & Hand Capture: MediaPipe Face Landmarker (478 landmarks + 52 blendshapes + head pose + neutral calibration + smoothing), MediaPipe/RTMW Hand Landmarker (left/right handedness, finger bend/splay angles, dedicated camera sync) | Unit & real-image inference tests passing, Blender shape-key & finger bone mapping verified |
| **Phase 11** | Complete Desktop UI (`Nuranova Mocap Studio`): PySide6 modern Light & Dark themes, all 20 required pages with real wired functionality (no dead buttons), Capture Studio 3D viewport (PyVista + Qt fallback) & live camera grid, Calibration Wizard, Marker Kit Editor, Motion Editor (timeline, curves, undo/redo, raw vs corrected), Project Manager, Processing Queue, Diagnostics, Settings | Launches cleanly offscreen & GUI; pytest-qt UI smoke tests verify all 20 pages, theme switching, project CRUD, viewport rendering, and editor actions |
| **Phase 12** | Comprehensive Testing, Benchmarks & Hardware Guides: unit, integration, synthetic ground-truth, 12-camera scalability simulation, real-video Pose2Sim integration test, performance profiling | `TEST_REPORT.md`, `ACCURACY_REPORT.md`, `HARDWARE_TEST_GUIDE.md` updated with actual measured numbers |
| **Phase 13** | Packaging & Deliverables: PyInstaller bundle + Inno Setup `NuranovaMocapStudio_Setup.exe`, Android `NuranovaMocapCamera.apk`, Blender `NuranovaMocapBridge.zip`, printable calibration boards & marker sheets, sample project, complete manuals | All three installers/packages built, verified, and documented in `BUILD_GUIDE.md` & `USER_GUIDE.md` |
