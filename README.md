# Nuranova Mocap Studio v1.0.0

**Developer:** Nuranova Solutions — Kawshal, Shanal, and Shehan ([https://nuranova.lk/](https://nuranova.lk/))  
**Platform:** Windows 10/11 x64 (Desktop) · Android 8.0+ (Camera App) · Blender 4.5 LTS / 5.0+ (Extension)

Nuranova Mocap Studio is a local-first, production-oriented multi-camera motion capture ecosystem built to transform 2 to 12 ordinary Android smartphones into a calibrated, synchronized, full-body, facial, and hand motion capture studio with editable Blender character animation output.

## Ecosystem Applications

1. **Nuranova Mocap Studio (`desktop/`, `engine/`)** — Standalone Windows Qt 6 desktop studio supporting project management, 2–12 camera discovery/pairing/sync, ChArUco multi-camera calibration, physical marker tracking (ArUco/AprilTag/Blob/Rigid Clusters), markerless AI pose estimation (RTMLib RTMW/HALPE-26 + MediaPipe), Hybrid Cinema reconstruction, articulated skeletal IK solving, non-destructive motion cleanup & foot locking, facial 52-blendshape tracking, finger tracking, and interactive 3D viewport & motion editor.
2. **Nuranova Mocap Camera (`android/`)** — Kotlin + Jetpack Compose + Camera2 Android camera application supporting dynamic capability probing, locked manual exposure/focus/WB, local H.264 recording with sensor timestamps, low-bandwidth MJPEG preview streaming, NTP-style clock drift sync, and resumable SHA-256 verified file transfer.
3. **Nuranova Mocap Bridge (`blender_addon/`)** — Installable Blender 4.5 LTS / 5.0+ extension supporting live Studio polling (`bpy.app.timers`), offline `.nmocap.json`/`.nmocap.h5`/`.bvh` import, source armature generation, Rigify and custom character rig mapping with rest-pose delta & scale compensation, facial shape-key mapping, finger animation, editable baked keyframes, Actions/NLA tracks, and FBX/BVH export.

## Quick Start (Development)

```powershell
# Activate Python 3.11 environment
.\.venv\Scripts\Activate.ps1

# Run full automated test suite
pytest

# Launch Nuranova Mocap Studio
python -m desktop.main
```

## Process Recorded Videos

Open **Import & Solve Videos** in the sidebar, choose **Add videos** or **Add folder**,
then **Check timing** and **Solve motion**. Use the **Review and export** tab to inspect
the result and **Open exports** to find the motion files. A project and connected
phones are optional for this workflow.

The recording library uses the same pipeline, including automatic camera estimation
when a calibration board is unavailable. Estimated geometry depends on the actor height;
the tracking report measures consistency rather than reference motion capture accuracy.

For a repeatable command-line run:

```powershell
.\.venv\Scripts\python.exe -m multi_video_studio.cli process "D:\path\to\videos" --out "D:\path\to\exports" --fps 30
```

Exports include HDF5 motion, Blender JSON, BVH, a camera calibration JSON, and a processing report.

For reusable camera geometry, choose **More → Load camera calibration…**, or pass
`--calibration "D:\path\to\calibration.json"` to the CLI. The supplied profile is
checked against every video camera before reconstruction. The export report includes
per-bone rotation continuity and skeletal fitting measurements; neither establishes
accuracy against reference motion capture.

The November 27 motion-quality repair, Blender MCP validation, measured results,
and review artifacts are documented in [TEST_REPORT.md](TEST_REPORT.md). The bridge
installer in `installers/NuranovaMocapBridge.zip` is version 1.2.2.

## Documentation Index

- [Architecture](ARCHITECTURE.md)
- [Project Plan](PROJECT_PLAN.md)
- [Repository Audit](REPOSITORY_AUDIT.md)
- [Development Progress](DEVELOPMENT_PROGRESS.md)
- [Dependency Register](DEPENDENCY_REGISTER.md)
- [License Register](LICENSE_REGISTER.md)
- [Motion Data Format Specification](documentation/MOTION_DATA_FORMAT.md)
- [Network Protocol Specification](documentation/NETWORK_PROTOCOL.md)
- [Test Report](TEST_REPORT.md)
- [Accuracy Report](ACCURACY_REPORT.md)
- [Build Guide](BUILD_GUIDE.md)
- [User Guide](USER_GUIDE.md)
- [Hardware Test Guide](HARDWARE_TEST_GUIDE.md)
- [Known Issues](KNOWN_ISSUES.md)
