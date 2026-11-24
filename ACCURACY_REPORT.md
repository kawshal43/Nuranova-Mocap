# Accuracy & Performance Validation Report — Nuranova Mocap Studio v1.0.0

**Generated:** 2026-11-24T10:31:49.519506+00:00  
**Test Workstation:** Windows 10 (AMD64) | 8 logical CPUs | GPU: NVIDIA GeForce RTX 2060 SUPER (Driver 560.94, 689 MiB free / 8192 MiB total)  
**Active ONNX Runtime Device:** `cpu` (`CPUExecutionProvider`)

> **IMPORTANT SCIENTIFIC INTEGRITY NOTICE:**  
> All optical geometry, calibration, synchronization, skeletal solving, and foot-locking measurements below were obtained on **explicitly labeled synthetic mathematical ground truth** and the **cited Pose2Sim BSD-3-Clause public demo video**. **Physical multi-smartphone validation on Samsung Galaxy S20 FE, Galaxy S21 FE, Redmi Note 12, and Redmi Note 15 has NOT YET been performed** because no physical phones were connected during automated build execution (`adb devices` empty). Do not extrapolate synthetic errors as physical optical accuracy until `HARDWARE_TEST_GUIDE.md` is executed on real hardware.

---

## 1. Synthetic Mathematical Ground-Truth Benchmarks (SYNTHETIC_GROUND_TRUTH)

| Metric | Engineering Target | Measured Synthetic Result | Status |
|---|---|---|---|
| 4-Camera ChArUco Bundle Adjustment Reprojection RMSE (0.25 px 2D noise) | <= 1.0 px | **0.347 px** (solved in 0.73 s) | Meets target (synthetic) |
| 1.000 m Known-Distance Wand Reconstruction Error | <= 10.0 mm | **0.21 mm** | Meets target (synthetic) |
| NTP-Style Clock Sync Residual (40 ppm drift + 2× 45 ms RTT spikes) | <= 5.0 ms | **0.28 ms** (est drift: 40.0 ppm) | Meets target (synthetic) |
| Audio Slate Clap Cross-Correlation Offset Error (48 kHz + noise) | <= 5.0 ms | **0.000 ms** (SNR: 284583.0x) | Meets target (synthetic) |
| Articulated Full-Body Walk Major Joint Position Error (mean / p95) | <= 20–30 mm | **0.00 mm** / **0.00 mm** (169 frames/s) | Meets target (synthetic) |
| Constant Bone-Length Invariance Across Solved Sequence | 0.0 mm drift | **0.00010 mm** max drift | Invariant |
| Planted-Foot Horizontal Displacement (Before -> After 2-Bone IK Foot Lock) | <= 10.0 mm | **25.1 mm -> 0.00 mm** | Meets target (synthetic) |

---

## 2. Real Multi-Camera Video Inference Benchmark (Pose2Sim BSD-3-Clause Demo Dataset)

- **Source Dataset:** `references/pose2sim-main/Pose2Sim/Demo_SinglePerson/videos/cam01.mp4` (cited open-source dataset; NOT Nuranova smartphone footage)
- **Backend Tested:** `mediapipe_pose:lightweight:cpu` (24 streamed frames via PyAV)
- **Measured Throughput:** **14.83 FPS** (CPU)
- **Actor Detection Rate:** **100.0%**
- **Mean Landmark Confidence:** **0.971**

---

## 3. Physical Hardware Validation Results

### Validated Device #1 — Samsung Galaxy S20 FE 5G (`SM-G781B`, Android 13, API 33) — **PHYSICALLY VALIDATED (Single-Phone Pipeline)**

Executed live on real hardware connected via USB (`adb reverse tcp:8765`) to `StudioNetworkServer`:

| Hardware Check | Measured Result | Notes |
|---|---|---|
| APK Install & Cold Launch | **Success (1,289 ms cold launch, 0 crashes)** | Debug-signed `NuranovaMocapCamera.apk` installed & launched cleanly |
| Camera2 Hardware Level & Timestamps | **`LEVEL_3` · `REALTIME` timestamp source** | Exposed manual exposure (`57.5 µs – 146.7 ms`), ISO (`50–3200`), manual focus, AWB lock, stab OFF, focal `5.4 mm`, sensor `7.26 × 5.44 mm` |
| Camera2 Standard vs High-Speed Modes | **Standard 720p/1080p/4K @ 30 FPS** (`[15,15],[7,24],[24,24],[7,30],[30,30]`); **High-Speed 720p/1080p @ 120/240 FPS** | Caught real Samsung HAL behavior: standard sessions cap at 30 FPS; dynamic FPS negotiation added & verified (`30.0–30.1 fps` measured) |
| PIN Pairing & WebSocket Preview Stream | **Paired (`cam_01`), 166 JPEG preview frames received** | Live `TextureView` preview + remote low-bandwidth JPEG stream verified |
| Live NTP Clock Sync (16 probes) | **Residual `4.13 ms` (median RTT `5.45 ms`)** | Meets `<= 5.0 ms` target over ADB TCP bridge; Wi-Fi multi-phone residual pending 2+ phones |
| Local H.264 MP4 Recording + Sensor Timestamps | **`1920×1080` H.264, `224 frames`, `30.00 fps` measured, `7.48 s`, `14.57 MB` + `231` JSONL sensor records** | Recorded real per-frame `sensor_timestamp_ns`, `exposure_ns`, and **`rolling_shutter_skew_ns = 28.74 ms`** |
| Automatic Resumable SHA-256 Transfer | **Both `.mp4` and `.jsonl` verified `true` byte-for-byte** | `MP4 SHA-256: d79dacdceccce4...903b9007`, `JSONL SHA-256: d4c2c1bece27b9...41098f2b7`, zero `AndroidRuntime` errors |

### Pending Multi-Phone Physical Validation

| Hardware Test | Status | Blocker / Next Action |
|---|---|---|
| Galaxy S21 FE / Redmi Note 12 / Redmi Note 15 Camera2 probing | **Not Physically Validated** | Connect remaining 3 phones and repeat live probe |
| Real Wi-Fi 2–4 phone simultaneous clock drift & audio clap sync | **Not Physically Validated** | Requires 2+ physical phones paired simultaneously over 5 GHz Wi-Fi |
| Printed A4/A3 ChArUco calibration with real smartphone lenses | **Not Physically Validated** | Print `documentation/printable/Nuranova_ChArUco_A4_7x5_35mm.pdf` at 100% scale |
| Physical matte-black suit + AprilTag 36h11 / 19mm sphere tracking | **Not Physically Validated** | Print `Nuranova_Wearable_AprilTag_36h11_40mm.pdf` and capture volume test |

---

## 4. Known Limitations

1. **Rolling Shutter & Exposure Jitter:** Consumer smartphones use rolling-shutter CMOS sensors; extreme whip-pan or sprint speeds require high shutter speed (`<= 1/250s`), ample studio lighting, and audio/flash sync verification.
2. **Distant Full-Body Finger Accuracy:** Distant body cameras lack sufficient pixel resolution on fingers; use dedicated close-up hand cameras for cinema finger capture.
3. **VRAM Contention on 8 GB GPUs:** On workstations where editing suites (e.g. Premiere Pro) consume >6.5 GB VRAM, Nuranova safely falls back to CPUExecutionProvider rather than crashing with CUDA OOM. Close competing GPU workloads for maximum CUDA throughput on RTX 3060 Ti.
