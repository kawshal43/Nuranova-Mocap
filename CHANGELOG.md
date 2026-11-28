# Changelog — Nuranova Mocap Studio

## Maintenance - 2026-11-28

- Validate two new dance recordings at every native timestamp. Add guarded
  background alignment for image translation/zoom/roll; preserve relative poses
  while flagging unsupported camera intervals.
- Prevent missing observations from resetting estimated camera distance to zero.
- Preserve coherent fast limbs through cleanup and the final body filter while
  retaining isolated-spike rejection.
- Fix inference caching in the frozen executable and invalidate development
  caches when the monocular adapter changes. Delivery checks now exercise actual
  cached CPU/CUDA solves and camera compensation.
- Restrict GLB/glTF export to the active Blender scene, preserving shared-rig
  animation and skinning in multi-scene files. Bridge version 1.3.3.
- Sample evaluated mesh coordinates in bulk for surface validation, retaining
  full pose sampling and fresh-import deformation checks.
- Record the [new video results and accuracy limits](documentation/SINGLE_VIDEO_TEST_REPORT_2026_11_28.md).
- Preserve airborne feet when contact evidence excludes a floor pin.
- Prevent Blender character corrections from taking the long path across 180 degrees;
  retain partial-range boundaries and report rejected corrections. Bridge version 1.3.2.
- Preserve native video start timestamps in initial review bounds and source-edge navigation.
- Avoid discarding a pending preview frame when layout or selection refreshes request the same image.
- Add four reproduced regression cases; repeat the full suite, supplied-video solves,
  Blender MCP pose checks and character FBX/GLB round trips. See the
  [follow-up report](documentation/FOLLOWUP_BUG_REPORT_2026_11_28.md) for remaining quality issues.

## [1.0.0] - 2026-11-24

### Fixed (Physical Hardware Validation — Samsung Galaxy S20 FE 5G `SM-G781B`)
- Updated Android `Camera2Controller` to dynamically negotiate supported `CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES` and `YUV_420_888` / `TextureView` preview buffer sizes (resolved Samsung CamX HAL `errorCode=4/5` when 60 FPS is requested on standard Camera2 sessions that cap at 30 FPS).
- Switched preview to `TextureView` with explicit supported buffer dimensions, improved dark-theme button contrast (`onPrimary = Color.White`), and fixed TopAppBar wrapping.
- Verified end-to-end on physical `SM-G781B`: pairing, 166 WebSocket previews, `4.13 ms` NTP sync residual, 1080p H.264 MP4 + 231 Camera2 `SENSOR_TIMESTAMP` JSONL recording, and byte-for-byte SHA-256 upload.

### Added
- **Phase 0:** Full audit of 8 local reference repositories, license/dependency registers, AGPL & non-commercial exclusion guardrails, and isolated Python 3.11 `.venv`.
- **Phase 1:** Core quaternion/SE(3)/Kabsch math (`engine/math_core`), ChArUco printable A4/A3 board generator & detector, intrinsic/extrinsic pose-graph calibration, metric-scale board Bundle Adjustment, Z-Up floor alignment, weighted DLT + RANSAC + LM triangulation (`engine/triangulation`), and streaming PyAV video I/O (`engine/video`).
- **Phase 2 & 3:** Android Camera application (`android/`, Kotlin + Jetpack Compose + Camera2 + MediaCodec H.264 + MediaMuxer + NSD + Room + DataStore + OkHttp) compiled to `installers/NuranovaMocapCamera.apk`; Desktop FastAPI/WebSocket/Zeroconf local server supporting 2–12 cameras, 6-digit PIN pairing, bearer auth, NTP clock drift regression, audio cross-correlation & flash sync, mixed-FPS temporal resampling, and resumable SHA-256 verified file transfers.
- **Phase 4:** Configurable wearable physical marker kit (39 markers + 7 asymmetric rigid 4-point clusters), valid OpenCV-generated AprilTag 36h11 / ArUco PDF generator, subpixel visible blob detector, Hungarian temporal tracker, epipolar matching, and Kabsch 6-DoF cluster pose solver.
- **Phase 5 & 10:** Safe ONNX Runtime CUDA/CPU hardware probe, RTMLib (`Wholebody` RTMW-133, `HALPE-26`) & MediaPipe (`Pose-33`, `Face Landmarker` 52 blendshapes + neutral calibration, `Hand Landmarker` 21 landmarks -> 16 finger bones), SHA-256 model registry, and real Pose2Sim 4-camera video inference validation.
- **Phase 6, 7 & 8:** 25-bone hierarchical articulated humanoid rig, actor calibration, constant-bone-length articulated IK solver with twist observability flags & anatomical limits, Hybrid Cinema Mode (calibrated surface-marker offset constraints without naive averaging), non-destructive motion cleanup (Butterworth, One Euro, MAD despiking, PCHIP gap fill, clean-room 2-bone IK foot locking, ground clamp), `.nmocap.h5` versioned HDF5 container, and standard `.bvh` exporter.
- **Phase 9:** Installable Blender 4.5 LTS / 5.0+ Extension (`installers/NuranovaMocapBridge.zip`) with main-thread-safe `bpy.app.timers` bridge, source armature generation, Rigify & custom rig auto/manual bone mapping with rest-pose delta & scale compensation, facial 52-blendshape shape keys, hand bones, editable baked keyframes, NLA tracks, and FBX/BVH export — verified inside real headless Blender 5.0.1.
- **Phase 11:** Complete PySide6 Windows Desktop Studio with Light & Dark themes, all 20 required pages, interactive 3D Z-Up viewport, Animation Curve Inspector, Motion Editor with Undo/Redo, portable `.nmocap` Project Manager, and background Processing Queue.
- **Phase 12 & 13:** Full automated pytest & Android JUnit suites, synthetic & real-video accuracy benchmarks, printable calibration/marker PDFs, portable sample project, PyInstaller bundle, Inno Setup Windows installer (`NuranovaMocapStudio_Setup.exe`), and complete manuals.
