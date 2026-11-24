# Dependency Register — Nuranova Mocap Studio

**Last Updated:** 2026-11-24  
**Python Runtime:** Python 3.11.9 x64 (`.venv`)

## 1. Desktop & Engine Python Packages (Pinned)

| Package | Version | Purpose |
|---|---|---|
| `numpy` | 1.26.4 | Vectorized linear algebra, transforms, array storage |
| `scipy` | 1.13.1 | Nonlinear least-squares BA/IK, Butterworth filtering, splines, rotations |
| `opencv-contrib-python` | 4.10.0.84 | Calibration, ChArUco/ArUco/AprilTag detection, undistortion, blob tracking |
| `PySide6` | 6.7.3 | Cross-platform Qt 6 desktop application UI |
| `pyvista` | 0.44.2 | VTK-based 3D viewport rendering |
| `pyvistaqt` | 0.11.1 | Qt 6 `QtInteractor` integration for PyVista |
| `fastapi` | 0.115.6 | Local REST & WebSocket server for Android camera network & Blender bridge |
| `uvicorn[standard]` | 0.32.1 | ASGI server for FastAPI |
| `websockets` | 13.1 | Low-latency preview, command, and clock-sync WebSocket transport |
| `httpx` | 0.28.1 | Async/sync HTTP test & client requests |
| `python-multipart` | 0.0.20 | Multipart/chunked file transfer endpoints |
| `zeroconf` | 0.136.2 | mDNS service advertisement & camera discovery |
| `SQLAlchemy` | 2.0.36 | Project & device metadata persistence over SQLite |
| `pydantic` | 2.10.4 | Strict schema & network protocol validation |
| `pydantic-settings` | 2.7.0 | Typed application settings persistence |
| `h5py` | 3.12.1 | Versioned HDF5 `.nmocap.h5` motion data read/write |
| `av` | 13.1.0 | FFmpeg-backed streaming video decoding & frame PTS/timebase inspection |
| `Pillow` | 11.0.0 | Image manipulation & thumbnail generation |
| `reportlab` | 4.2.5 | Millimeter-accurate printable A4/A3 ChArUco & marker PDF generator |
| `mediapipe` | 0.10.14 | Face Landmarker (52 blendshapes), Hand Landmarker, Pose Landmarker |
| `onnxruntime-gpu` | 1.19.2 | CUDA & CPU ONNX model execution |
| `rtmlib` | 0.0.16 | Lightweight YOLOX + RTMW / RTMPose ONNX inference (installed from `references/rtmlib-main`) |
| `tqdm` | 4.67.1 | Progress utilities (required by `rtmlib`) |
| `pytest`, `pytest-asyncio`, `pytest-qt` | 8.3.4 / 0.24.0 / 4.4.0 | Automated unit, integration, async, and Qt UI testing |
| `ruff` | 0.8.4 | Static analysis and formatting |
| `pyinstaller` | 6.11.1 | Standalone Windows executable bundling |

## 2. Android Dependencies (Gradle)

- Kotlin 2.0.21, Android Gradle Plugin 8.7.3, Compile/Target SDK 35 (or 36 matching installed SDK), Min SDK 26 (Android 8.0, covers Galaxy S20 FE / S21 FE / Redmi Note 12 / Redmi Note 15 and broad devices).
- Jetpack Compose BOM, Material3, Navigation Compose, Lifecycle ViewModel Compose, Room, DataStore Preferences, WorkManager, OkHttp 4.12.0, Kotlinx Coroutines, Kotlinx Serialization JSON.

## 3. Pretrained Model Registry

See `model_registry/models.json` for exact URLs, versions, licenses, and verified SHA-256 checksums populated upon model download/verification.
