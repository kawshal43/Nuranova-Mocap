# Physical Hardware Test Guide — Nuranova Mocap Studio v1.0.0

**Target Devices:**
1. Samsung Galaxy S20 FE
2. Samsung Galaxy S21 FE
3. Redmi Note 15
4. Redmi Note 12

**Target PC:** Intel Core i5-10400 · NVIDIA GeForce RTX 3060 Ti (8 GB VRAM) · 32 GB RAM · Windows 10/11 x64

> **Validation Status:** Physical smartphone tests have **not yet been executed** because no Android devices were attached during automated build verification. Complete the tests below on real hardware and record actual measured numbers in `ACCURACY_REPORT.md` before declaring physical optical accuracy validated.

---

## Test 1 — APK Installation & Dynamic Camera2 Capability Probing (All 4 Phones)

1. Enable **USB Debugging** on each phone or transfer `installers/NuranovaMocapCamera.apk`.
2. Install via ADB (`adb install -r installers\NuranovaMocapCamera.apk`) or Android package installer.
3. Open **Nuranova Mocap Camera -> Permissions -> Diagnostics -> Re-Probe Camera2**.
4. Record for each of the 4 phones:
   - `INFO_SUPPORTED_HARDWARE_LEVEL` (`FULL` / `LEVEL_3` / `LIMITED`)
   - `SENSOR_INFO_TIMESTAMP_SOURCE` (`REALTIME` vs `MONOTONIC`)
   - Verified third-party Camera2 exposed modes (confirm whether 1080p@60fps and 4K@30/60fps are actually exposed by Samsung/Xiaomi Camera2 HAL!)
   - Whether `CONTROL_AE_MODE_OFF` (manual 1/250s exposure) and `CONTROL_VIDEO_STABILIZATION_MODE_OFF` succeed without session errors.

## Test 2 — Multi-Phone Wi-Fi Discovery, PIN Pairing & Synchronization (2, 3, and 4 Phones)

1. Connect PC (preferably wired Ethernet) and all 4 phones to a dedicated 5 GHz Wi-Fi router (disable mobile data & battery optimization for Nuranova Mocap Camera).
2. In Studio **Camera Manager**, start Local Server and verify all 4 phones discover `_nuranova-mocap._tcp.` (or connect via manual IP).
3. Pair all 4 phones with the 6-digit PIN; verify wrong PIN is rejected (`403`).
4. Trigger **Sync** on each phone and record:
   - `minRTT (ms)`, `Sync Residual (ms)` (engineering target: `<= 5.0 ms`), and `Clock Drift (ppm)`.
5. Record a 30-second test take with a sharp visual LED flash + loud wooden clap at t=3s and t=27s.
6. Verify resumable transfer integrity: mid-upload, toggle Wi-Fi OFF for 10 seconds on 1 phone, then toggle ON and verify upload resumes from exact byte offset and passes **SHA-256 verification**.
7. Measure inter-camera audio clap offset and flash-frame difference across all 4 phones; test mixed FPS (e.g., S20 FE + S21 FE @ 60 FPS, Redmi Note 12 @ 30 FPS).

## Test 3 — Printed ChArUco Calibration & Known-Distance Wand Test

1. Print `documentation/printable/Nuranova_ChArUco_A3_9x6_40mm.pdf` (recommended for 3m×3m volume) or A4 board at **100% scale** on matte paper mounted to rigid board.
2. Measure the 100 mm scale bar and 5 consecutive squares with calipers; enter the exact measured square length in **Calibration Wizard**.
3. Position 4 tripods surrounding a 2.5m × 2.5m capture floor at ~1.6–2.0m height, angled slightly downward, with focus locked on volume center and stabilization OFF.
4. Record a 60-second slow ChArUco board sweep covering floor, mid-height, and overhead corners visible to overlapping pairs.
5. Run calibration and record:
   - Per-camera intrinsic & extrinsic reprojection RMSE (target `<= 1.0 px`)
   - Known-distance check: hold two printed AprilTag markers mounted exactly `1000.0 mm` apart on a rigid bar at 5 volume positions -> record reconstructed distance error in mm (target `<= 10.0 mm`).

## Test 4 — Physical Marker Kit & Hybrid Cinema Capture

1. Print `documentation/printable/Nuranova_Wearable_AprilTag_36h11_40mm.pdf` on matte cardstock; attach rigid 4-point clusters (Headband, Chest, Pelvis, Forearms, Shoes) and matte 19 mm white spheres onto matte-black attire.
2. Capture 6 actor animation tests (Neutral/T-pose, Idle, Walking, Turning 360°, Sitting/Standing, Arm Crossing) + 1 close-up Facial & Hand clip.
3. Process in **Cinema Mode** on RTX 3060 Ti (close competing GPU apps so >= 6 GB VRAM is free). Record:
   - GPU VRAM usage, processing speed (FPS), thermal temperature rise on phones over 10-minute session, dropped frame count, planted-foot drift in mm, and Blender retargeting quality on a stylized character rig.
