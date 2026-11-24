# Automated Test Report — Nuranova Mocap Studio v1.0.0

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
