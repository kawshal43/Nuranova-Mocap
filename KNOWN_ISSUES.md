# Known Issues & Hardware Blockers — Nuranova Mocap Studio v1.0.0

**Last Updated:** 2026-11-28

## Latest motion-quality verification

The latest suite passed 173 tests with one optional external-footage test skipped.
The third supplied single-camera recording still has unreliable feet, and the
supplied character still has sampled intersections after correction. See the
[follow-up bug report](documentation/FOLLOWUP_BUG_REPORT_2026_11_28.md) for measured
limits and separate editable Blender outputs. Passing export checks does not
establish collision-free animation or reference capture accuracy.

## 1. Physical Hardware Validation Blockers (Not Physically Validated)

- **Smartphones Not Connected (`adb devices` empty):** Physical testing on the 4 target devices (`Samsung Galaxy S20 FE`, `Samsung Galaxy S21 FE`, `Redmi Note 15`, `Redmi Note 12`) could not be executed in this session.
  - **What is validated:** Real Android APK compiled and unit-tested (`installers/NuranovaMocapCamera.apk`, 17.7 MB); Desktop 2–12 camera network, PIN pairing, WebSocket control, NTP clock drift regression, audio clap sync, and resumable SHA-256 transfers verified via integration tests; CV/AI/calibration/skeletal/foot-locking validated via synthetic ground truth and Pose2Sim BSD-3-Clause real 4-camera video.
  - **What remains blocked pending physical hardware:** Real Camera2 mode availability on Samsung/Xiaomi HALs, real multi-phone Wi-Fi sync residuals, real printed ChArUco calibration on phone optics, real physical marker suit tracking, thermal throttling behavior over long shoots, and clean-PC installer testing on a second Windows machine. Follow `HARDWARE_TEST_GUIDE.md` to complete physical validation.

## 2. Environment & Runtime Notes

- **Debug-Signed Test APK:** Because no private release signing keystore was supplied (and secrets must never be hardcoded or fabricated), `installers/NuranovaMocapCamera.apk` is built as a debug-signed sideloadable test APK. Follow `BUILD_GUIDE.md` Section 5 for production release keystore signing.
- **GPU VRAM Contention / CUDA Fallback:** On the development workstation (`RTX 2060 SUPER` 8 GB VRAM), external desktop apps consumed ~7.4 GB of VRAM during automated runs, causing the hardware detector to intentionally fall back to `CPUExecutionProvider` to prevent CUDA OOM. On a clean RTX 3060 Ti (8 GB VRAM) workstation with >= 1.2 GB free VRAM and CUDA/cuDNN DLLs installed, `CUDAExecutionProvider` activates automatically.
- **Pretrained AI Weights Not Hard-Bundled in Installer:** To keep the Windows installer compact and respect OpenMMLab dataset licensing advisories (see `LICENSE_REGISTER.md`), large ONNX model weights (`RTMW-x`, `YOLOX-m`, `MediaPipe Tasks`) are downloaded on-demand via **Diagnostics -> Model Registry** with automatic SHA-256 checksum verification.
- **Excluded Reference Repositories:** `references/freemocap-main` (AGPL-3.0) and `references/UnderPressure-main` (InterDigital Non-Commercial Evaluation License) are deliberately excluded from runtime dependencies and installers.
