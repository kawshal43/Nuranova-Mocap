# Software Architecture — Nuranova Mocap Studio

**Version:** 1.0.0  
**Developer:** Nuranova Solutions (Kawshal, Shanal, Shehan — https://nuranova.lk/)

---

## 1. High-Level Ecosystem Architecture

Nuranova Mocap Studio is architected as three independently installable, locally networked applications sharing a strict versioned protocol (`shared_protocol/`) and motion data format (`engine/data_format/`).

```
+-------------------------------------------------------------------------+
|                 APPLICATION B: Nuranova Mocap Camera                    |
|              (2..12 Android Phones — Kotlin / Compose / Camera2)        |
|  - Local H.264 high-bitrate recording (MediaCodec + MediaMuxer)         |
|  - Camera2 SENSOR_TIMESTAMP + monotonic clock + NTP-style sync client   |
|  - Low-bandwidth MJPEG preview stream -> Desktop                        |
|  - Resumable chunked HTTP upload with SHA-256 verification              |
|  - NSD (mDNS) discovery + manual IP + HMAC-SHA256 pairing token auth    |
+------------------------------------+------------------------------------+
                                     | Local Wi-Fi / LAN (No cloud/internet)
                                     | FastAPI HTTP + WebSocket + mDNS
+------------------------------------v------------------------------------+
|                 APPLICATION A: Nuranova Mocap Studio                    |
|                (Windows 10/11 x64 — Python 3.11 / PySide6)              |
|                                                                         |
|  +-------------------------+   +-------------------------------------+  |
|  | Desktop Control Plane   |   | Processing & Reconstruction Engine  |  |
|  | - CameraManager (2..12) |   | - engine/math_core (SE3, Quat, DLT) |  |
|  | - Pairing & Auth Guard  |   | - engine/video (PyAV streaming)     |  |
|  | - ClockSyncService      |   | - engine/marker_tracking (ArUco,    |  |
|  | - Resumable Transfer    |   |   AprilTag, Blob, Rigid Clusters)   |  |
|  | - ProjectManager/SQLite |   | - engine/ai_tracking (RTMLib/RTMW,  |  |
|  | - Background Job Queue  |   |   MediaPipe, CUDA/CPU ORT runtime)  |  |
|  | - 20-Page Studio UI     |   | - engine/triangulation (Weighted    |  |
|  |   + 3D Viewport         |   |   DLT, RANSAC, Bundle Adjustment)   |  |
|  +-------------------------+   | - engine/skeleton (Articulated IK,  |  |
|                                |   Actor Calibration, Constraints)   |  |
|                                | - engine/optimization (Hybrid Mode) |  |
|                                | - engine/motion_cleanup (One Euro,  |  |
|                                |   Butterworth, Foot Lock, Gap Fill) |  |
|                                | - engine/face & engine/hands        |  |
|                                | - engine/export (HDF5, BVH, JSON)   |  |
|                                +-------------------------------------+  |
+------------------------------------+------------------------------------+
                                     | Local HTTP Bridge API / .nmocap.h5 / .nmocap.json
+------------------------------------v------------------------------------+
|                 APPLICATION C: Nuranova Mocap Bridge                    |
|              (Blender 4.5 LTS / 5.0+ Extension — Python bpy)            |
|  - Connects to Studio local bridge (main-thread safe bpy.app.timers)    |
|  - Offline import of .nmocap.json / .nmocap.h5 / .bvh                   |
|  - Generates calibrated Nuranova Source Armature (Z-up, meters, XYZW)   |
|  - Guided auto + manual bone mapping (Rigify & Custom Rigs)             |
|  - Rest-pose delta compensation + scale & root-motion handling          |
|  - Facial 52-blendshape shape-key & bone driver mapping                 |
|  - Finger animation, NLA tracks, editable baked keyframes, FBX & BVH    |
+-------------------------------------------------------------------------+
```

---

## 2. Canonical Coordinate System, Units & Conventions

Documented across all modules, schemas, HDF5 datasets, and Blender bridge:

| Property | Convention |
|---|---|
| **World Coordinate System** | Right-handed, **Z-Up** (`+X` Right, `+Y` Forward, `+Z` Up — matches Blender native world space) |
| **Camera Coordinate System** | OpenCV pinhole convention (`+X` Right, `+Y` Down, `+Z` Forward along optical axis) |
| **Spatial Units** | **Meters (`m`)** everywhere in 3D world space, calibration boards, skeleton bone lengths, and HDF5 arrays. Printed dimensions in UI accept millimeters (`mm`) and convert to meters internally. |
| **Angles** | **Radians** in engine math; degrees displayed in UI and exported in BVH per BVH spec. |
| **Quaternion Storage Order** | **Scalar-last `[x, y, z, w]`** (matches SciPy `Rotation.as_quat()` and internal arrays). Converted explicitly to Blender's `[w, x, y, z]` (`mathutils.Quaternion((w, x, y, z))`) at the Blender Bridge boundary. |
| **Rotation Matrices** | Active right-handed `3x3` column-vector convention: `p_world = R @ p_local + t`. Camera extrinsics store world-to-camera `R_cw, t_cw` (`X_cam = R_cw @ X_world + t_cw`), with camera center `C_world = -R_cw.T @ t_cw`. |
| **Timestamps** | Integer **nanoseconds (`int64`)** on monotonic/session timeline (`t_ns`), with floating-point seconds (`float64`) derived relative to `session_start_ns`. |
| **Internal Data Format Version** | `nmocap-motion-v1.0` (`.nmocap.h5` HDF5 binary arrays + `.nmocap.json` human-readable manifest/metadata; full spec in `documentation/MOTION_DATA_FORMAT.md`). |

---

## 3. Modular Layer Boundaries & Expandability

All hardware-specific and algorithm-specific subsystems use explicit Python `Protocol` / ABC adapter interfaces so future hardware (IR global-shutter cameras, OV9281/OV2311, USB/GigE cameras, ESP32 IMUs, TrueDepth/ARKit) and future solvers can plug in without modifying the UI or pipeline orchestration:

- `CameraAdapter` (`desktop/camera_manager/adapters.py`): Android Wi-Fi camera client, imported offline video camera, simulated multi-camera client (for 12-camera scalability testing), and stub interfaces for future USB/GigE/IR hardware.
- `Tracker2DAdapter` (`engine/ai_tracking/base.py`, `engine/marker_tracking/`): Physical marker tracker (ArUco/AprilTag/Blob), RTMLib (RTMW-133, HALPE-26, Hand-21), MediaPipe (Face 478+52 blendshapes, Hand 21, Pose 33).
- `TriangulatorAdapter` (`engine/triangulation/`): Weighted DLT + RANSAC outlier rejection + Levenberg-Marquardt nonlinear reprojection refinement + global bundle adjustment.
- `SkeletonSolverAdapter` (`engine/skeleton/`): Hierarchical articulated skeleton with actor bone-length calibration, analytical + constrained Levenberg-Marquardt IK, rigid marker cluster orientation fusion, anatomical joint rotation limits, and quaternion temporal continuity.
- `MotionCleanerAdapter` (`engine/motion_cleanup/`): Non-destructive chain storing raw and corrected motion separately — outlier despiking, confidence-weighted PCHIP/cubic gap interpolation, adaptive One Euro / zero-phase Butterworth filtering, bone-length enforcement, kinematic foot-contact detection + 2-bone IK foot-locking, ground penetration clamping, and root-motion stabilization.
- `ExporterAdapter` (`engine/export/`): `.nmocap.h5` + `.nmocap.json`, `.bvh`, Blender Bridge live/offline payload, and Blender-driven `.fbx`.

---

## 4. Network Protocol & Security Architecture

- **Protocol Version:** `nuranova-cam-proto/1.0` (schemas in `shared_protocol/` & `schemas/`).
- **Discovery:** mDNS / Zeroconf service `_nuranova-mocap._tcp.local.` advertised by desktop Studio; manual IP:port fallback supported.
- **Pairing & Authentication:** 6-digit pairing PIN displayed in Studio (expiring, single-use) exchanges a per-device 256-bit cryptographic bearer token (`secrets.token_urlsafe(32)`) stored in SQLite/DataStore. Every camera REST and WebSocket command requires valid `Authorization: Bearer <token>` and `X-Device-Id` validation. Unauthenticated recording commands are rejected with `401 Unauthorized`.
- **Network Binding:** Configurable bind interface (default LAN interface or explicit user-selected IP; no cloud relay, no external telemetry).
- **Recording & File Transfer:**
  - Studio sends `SCHEDULE_RECORD` with synchronized `target_start_studio_ns` converted to each phone's monotonic clock via fitted linear clock model (`t_studio = a * t_device + b`, drift-compensated).
  - Phone records locally to H.264 MP4 + sidecar JSONL frame timestamps (`frame_index`, `sensor_timestamp_ns`, `pts_us`, `exposure_ns`, `rolling_shutter_skew_ns`).
  - Phone streams low-bandwidth JPEG frames (`<= 320x180`, 5–10 FPS) over WebSocket for UI monitoring.
  - Post-recording: resumable chunked HTTP upload (`Content-Range` / chunk offset API) with SHA-256 checksum verification before the phone marks a take transferable/cleanable. Recordings are never auto-deleted before server checksum confirmation.

---

## 5. Project Storage Architecture

Each project is a self-contained, portable directory with a SQLite metadata index (`project.db`) and manifest (`project.nmocap`), never storing raw video blobs inside SQLite:

```text
<ProjectName>/
    project.nmocap          # JSON manifest (UUID, schema version, settings, paths)
    project.db              # SQLite metadata index (takes, cameras, calibrations, jobs)
    calibration/            # Intrinsic/extrinsic/world calibration profiles (.json)
    actors/                 # Actor profiles & calibrated bone lengths (.json)
    recordings/             # Original camera MP4 files + verified SHA-256 manifests
    timestamps/             # Per-camera JSONL sensor timestamps & sync reports
    tracking/               # Raw 2D marker & AI keypoint detections (.h5/.json)
    reconstructed/          # Raw 3D triangulated markers & joints (.h5)
    raw_motion/             # Unfiltered solved skeletal motion (.nmocap.h5)
    corrected_motion/       # Non-destructive cleaned motion (.nmocap.h5)
    facial_motion/          # 52 blendshape trajectories + head pose (.h5/.json)
    hand_motion/            # Left/right finger joint angles & landmarks (.h5/.json)
    character_mappings/     # Reusable Blender rig mapping profiles (.json)
    exports/                # Exported BVH, FBX, Blender bridge bundles
    backups/                # Point-in-time portable project archive snapshots (.zip)
    logs/                   # Structured JSONL session & processing logs
```

---

## 6. Architectural Decisions & Deviations from Initial Preferences

| Decision | Justification |
|---|---|
| `numpy==1.26.4` instead of NumPy 2.x | Ensures clean wheel ABI compatibility across `mediapipe==0.10.14`, `opencv-contrib-python==4.10.0.84`, `scipy==1.13.1`, and `pyvista==0.44.2` on Python 3.11 Windows. |
| Native OpenCV Contrib AprilTag + ArUco (`cv2.aruco`) | Avoids C compiler/CMake build dependency on Windows while delivering Apache-2.0 licensed, library-generated valid ArUco and AprilTag (`36h11`, `25h9`, `16h5`) detection and printable generation. |
| Clean-room kinematic + IK foot-locking instead of `UnderPressure` | `UnderPressure` carries a non-commercial research-only InterDigital evaluation license incompatible with Nuranova Solutions product distribution. |
| Clean-room desktop/bridge instead of `FreeMoCap` | `FreeMoCap` is AGPL-3.0, incompatible with commercial proprietary distribution. |
| PyVista `QtInteractor` with automatic Qt 3D software fallback | Guarantees 3D viewport works on full GPU workstations while remaining resilient on headless/CI/remote-display environments. |
