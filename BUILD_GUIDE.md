# Build & Packaging Guide — Nuranova Mocap Studio v1.0.0

**Developer:** Nuranova Solutions (Kawshal, Shanal, Shehan — https://nuranova.lk/)

---

## 1. Prerequisites (Windows 10/11 x64 Build Machine)

- **Python 3.11 (64-bit)** (isolated virtual environment `.venv`)
- **Inno Setup 6** (`ISCC.exe`) for building `NuranovaMocapStudio_Setup.exe`
- **JDK 17 or JDK 21** + **Android SDK** (`compileSdk 36`, `build-tools 36.0.0`) for building `NuranovaMocapCamera.apk`
- **Blender 4.5 LTS or 5.0+** (for automated Blender Bridge CLI verification)

---

## 2. Python Environment & Automated Tests

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\pip.exe install -e .[dev]
.\.venv\Scripts\pip.exe install --no-deps .\references\rtmlib-main

# Run all unit, integration, Blender 5.0, real-video Pose2Sim, and PySide6 UI tests
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\pytest.exe -v
```

---

## 3. Generate Deliverables (Printable Boards, Marker Sheets, Sample Project, Blender Extension)

```powershell
.\.venv\Scripts\python.exe scripts\generate_deliverables.py
```

Outputs:
- `installers/NuranovaMocapBridge.zip` (installable Blender 4.5 LTS / 5.0+ Extension)
- `documentation/printable/Nuranova_ChArUco_A4_7x5_35mm.pdf` & `Nuranova_ChArUco_A3_9x6_40mm.pdf`
- `documentation/printable/Nuranova_Wearable_AprilTag_36h11_40mm.pdf`
- `sample_projects/Nuranova_Sample_Walk/`

---

## 4. Build Windows Standalone Installer (`NuranovaMocapStudio_Setup.exe`)

```powershell
# Step 1: Bundle standalone Windows application with PyInstaller
.\.venv\Scripts\pyinstaller.exe --noconfirm --clean installers\nuranova_studio.spec

# Step 2: Compile Windows Setup installer with Inno Setup 6
& "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe" installers\NuranovaMocapStudio.iss
```

Output: `installers/output/NuranovaMocapStudio_Setup.exe`

---

## 5. Build Android Camera Application (`NuranovaMocapCamera.apk`)

### Debug Test APK (No production keystore required)

```powershell
$env:JAVA_HOME = "C:\Program Files\Java\jdk-21.0.12"
$env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk"
cd android
.\gradlew.bat testDebugUnitTest assembleDebug --no-daemon
```

Output copied to: `installers/NuranovaMocapCamera.apk` (**Debug-signed test APK** — safe for local sideloading via `adb install` or file manager).

### Production Release Signing Instructions (Do NOT commit keys to Git!)

1. Generate a private release keystore outside the repository:
   ```powershell
   keytool -genkeypair -v -keystore C:\secure\nuranova-release.jks -alias nuranova -keyalg RSA -keysize 2048 -validity 10000
   ```
2. Configure environment variables (`NURANOVA_KEYSTORE_PATH`, `NURANOVA_KEYSTORE_PASS`, `NURANOVA_KEY_ALIAS`, `NURANOVA_KEY_PASS`) and add a `signingConfigs { create("release") { ... } }` block in `android/app/build.gradle.kts`.
3. Run `.\gradlew.bat assembleRelease` to produce a production-signed APK/AAB.

---

## 6. Install Blender Extension (`NuranovaMocapBridge.zip`)

1. Open **Blender 4.5 LTS** or **Blender 5.0+**.
2. Go to **Edit -> Preferences -> Get Extensions** (click the top-right dropdown arrow) -> **Install from Disk...**.
3. Select `installers/NuranovaMocapBridge.zip`.
4. Enable **Nuranova Mocap Bridge** and open **View3D Sidebar (`N`) -> Nuranova Mocap**.
