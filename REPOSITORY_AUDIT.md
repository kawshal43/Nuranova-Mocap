# Repository Audit Report — Nuranova Mocap Studio

**Project:** Nuranova Mocap Studio v1.0.0  
**Developer:** Nuranova Solutions (Kawshal, Shanal, Shehan — https://nuranova.lk/)  
**Audit Date:** 2026-11-24  
**Workspace:** `e:\Backup\KAWSHAL\Nuranova-Mocap`

---

## 1. Environment Audit

| Component | Detected Status | Notes |
|---|---|---|
| Operating System | Windows 10/11 x64 | Primary target OS |
| Python (Preferred) | Python 3.11.9 (`py -3.11`) | Matches preferred baseline; isolated `.venv` created |
| Python (System default) | Python 3.14.3 | Not used for `.venv` due to wheel compatibility |
| GPU (Dev Machine) | NVIDIA GeForce RTX 2060 SUPER (8 GB VRAM), Driver 560.94, CUDA 12.6 driver support | Target deployment spec is RTX 3060 Ti (8 GB VRAM); same 8 GB VRAM envelope |
| Java / JDK | Oracle JDK 21.0.12 LTS (`C:\Program Files\Java\jdk-21.0.12`) | Available for Android Gradle builds |
| Android Studio / SDK | Android Studio installed; SDK at `%LOCALAPPDATA%\Android\Sdk` (`android-36`, `android-37.0`, `build-tools 36.0.0`, `platform-tools`, `adb`) | Ready to build real APKs |
| Gradle Cache | `gradle-9.1.0-bin`, `gradle-9.6.0-bin` cached locally | Android project will include pinned Gradle Wrapper |
| Blender | Blender 5.0.1 (`C:\Program Files\Blender Foundation\Blender 5.0\blender.exe`) | Extension targets Blender 4.5 LTS+ and tested directly in Blender 5.0.1 CLI |
| FFmpeg | FFmpeg 8.1 (`winget` Gyan build) | Available alongside PyAV (`av`) |
| Inno Setup | Inno Setup 6 (`%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe`) | Ready for building `NuranovaMocapStudio_Setup.exe` |
| Connected Android Devices | None currently connected over ADB (`adb devices` empty) | Physical smartphone tests documented in `HARDWARE_TEST_GUIDE.md` |
| Disk Free Space | `C:` 217.56 GB free; `E:` 249.13 GB free | Ample space for models, builds, installers |

---

## 2. Downloaded Reference Repositories Audit

All 8 repositories under `references/` were inspected without modification. Original files in `references/` are preserved untouched.

| Repository | Version / Snapshot | License | Commercial Suitability | Reusable Components / Integration Strategy |
|---|---|---|---|---|
| `apriltag-master` | 3.4.5 (C library) | **BSD 2-Clause** | Permissive (commercial-safe) | Reference tag families (`tag36h11`, `tag25h9`, `tag16h5`, `tagStandard41h12`). On Windows without CMake/MSVC build overhead, OpenCV contrib's native `cv2.aruco` AprilTag detectors (`DICT_APRILTAG_36h11`, `DICT_APRILTAG_25h9`, `DICT_APRILTAG_16h5`) provide zero-build Apache-2.0 support; optional native `apriltag` adapter hook included. |
| `freemocap-main` | 2.0.0a26 | **AGPL-3.0** | **INCOMPATIBLE** with closed-source commercial distribution | **DO NOT COPY CODE OR DEPEND ON IT.** Reviewed purely for architectural study of multi-camera workflows and Blender export UX. Completely excluded from runtime dependencies and code reuse. |
| `mediapipe-master` | dev snapshot | **Apache 2.0** | Permissive (commercial-safe) | Used via official PyPI wheel `mediapipe==0.10.14` (Tasks API & Solutions) for Face Landmarker (478 landmarks + 52 blendshapes), Hand Landmarker (21 landmarks/hand), and Pose Landmarker (33 landmarks). |
| `opencv-5.x` | 5.x source tree | **Apache 2.0** | Permissive (commercial-safe) | Used via pinned binary wheel `opencv-contrib-python==4.10.0.84` for ChArUco/ArUco/AprilTag detection, camera intrinsic/extrinsic calibration, undistortion, blob/contour subpixel marker localization, and epipolar geometry. |
| `pose2sim-main` | main snapshot | **BSD 3-Clause** | Permissive (commercial-safe) | Studied algorithms for weighted DLT triangulation, reprojection outlier pruning, camera synchronization, and Butterworth/Loess filtering. Also ships 4-camera real test footage (`Demo_SinglePerson/videos/cam01..04.mp4` + calibration data) used strictly as a cited BSD-3-Clause integration dataset (never claimed as smartphone hardware validation). Note: Pose2Sim's `opensim` dependency is heavy and biomechanical; Nuranova implements a clean animation-first articulated skeletal solver in NumPy/SciPy instead. |
| `rtmlib-main` | 0.0.16 | **Apache 2.0** | Permissive (commercial-safe) | Directly integrated as a clean runtime dependency (installed from `references/rtmlib-main` without modifying source). Provides lightweight ONNX Runtime inference for **YOLOX + RTMW (Wholebody 133 keypoints: body+feet+hands+face)**, **RTMPose HALPE-26 (body+feet)**, and **Hand-21** without heavy `mmcv`/`mmpose` build issues. Note: pretrained model weights hosted by OpenMMLab carry Apache-2.0 / dataset-specific licensing notes tracked in `LICENSE_REGISTER.md` and `model_registry/`. |
| `scipy-main` | 2.0.0.dev0 | **BSD 3-Clause** | Permissive (commercial-safe) | Used via pinned binary wheel `scipy==1.13.1` for Levenberg-Marquardt / Trust-Region Reflective nonlinear bundle adjustment (`least_squares`), Procrustes / Kabsch rigid body alignment (`Rotation`), Butterworth zero-phase filtering (`sosfiltfilt`), cubic/PCHIP gap interpolation, and sparse optimization. |
| `UnderPressure-main` | SCA 2022 release | **InterDigital Limited Software Evaluation License (Non-Commercial Research Only)** | **INCOMPATIBLE** with commercial or product redistribution | **DO NOT COPY CODE OR WEIGHTS (`pretrained.tar`).** Strictly research-only evaluation license forbidding commercial use or third-party distribution. Nuranova implements a clean-room, commercially-permissive kinematic foot-contact detector (height + linear/angular velocity + dual-threshold hysteresis + ankle/toe geometry) and analytical/optimization 2-bone leg IK foot-locking engine instead. |

---

## 3. Missing Reference Repositories & Resolution

The prompt noted that the following might or might not be present; audit confirmed they are **not** in `references/`:
- `SkellyCam`, `SkellyForge`, `SkellyBlender` (AGPL-3.0 FreeMoCap ecosystem — excluded by design due to commercial licensing)
- `Anipose`, `Aniposelib` (BSD-2-Clause; core mathematical calibration, ChArUco detection, weighted DLT triangulation, and sparse bundle adjustment are implemented cleanly using OpenCV + SciPy with full unit & synthetic verification)
- `MMPose` (replaced cleanly by `rtmlib` + `onnxruntime`, eliminating `mmcv` C++/CUDA compilation fragility on Windows)
- `AsterTrack` (not present; optical multi-camera rigid marker cluster tracking and cross-view epipolar matching implemented cleanly in `engine/marker_tracking/` and `engine/triangulation/`)

---

## 4. Compatibility Risks & Mitigations

1. **AGPL / Non-Commercial License Contamination**: Strictly excluded `freemocap-main` and `UnderPressure-main` from code reuse and runtime dependencies.
2. **CUDA / cuDNN Runtime Fragility on End-User Windows PCs**: Driver 560.94 supports CUDA 12.6, but end-user PCs may lack full CUDA Toolkit/cuDNN DLLs or have limited free VRAM. Mitigation: graceful runtime probe in `engine/ai_tracking/runtime.py` that tests CUDA EP safely in an isolated check, reports exact diagnostics, and falls back seamlessly to ONNX Runtime CPU EP without crashing.
3. **NumPy 2.x ABI Breakage**: Pinned `numpy==1.26.4` for broad compatibility across `mediapipe==0.10.14`, `opencv-contrib-python==4.10.0.84`, `scipy==1.13.1`, `pyvista==0.44.2`, and `onnxruntime-gpu==1.19.2`.
4. **Blender 4.5 LTS vs Blender 5.0 API Differences**: Blender 5.0 updated Action/Channelbag/Slotted-Action APIs and Extension Manifest (`blender_manifest.toml`). Our Blender extension uses Blender 4.2+ Extension packaging (`blender_manifest.toml`, `blender_version_min = "4.5.0"`) and handles both Blender 4.5 LTS and Blender 5.0 Action/FCurve APIs safely.
5. **PyVista / VTK Headless or Driver Issues**: Desktop 3D viewport uses PyVista `QtInteractor` when OpenGL/VTK initializes cleanly, with an automatic pure-Qt interactive 3D software-projected viewport fallback so rendering never crashes on remote desktops, CI/pytest offscreen runs, or driver-constrained systems.
