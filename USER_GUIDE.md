# User Guide — Nuranova Mocap Studio v1.0.0

**Developer:** Nuranova Solutions — Kawshal, Shanal, and Shehan ([https://nuranova.lk/](https://nuranova.lk/))

---

## 1. Studio Setup Overview

Nuranova Mocap Studio operates **100% locally on your Wi-Fi / LAN** without cloud accounts or external uploads:
1. Install **Nuranova Mocap Studio** on your Windows 10/11 PC (`NuranovaMocapStudio_Setup.exe`).
2. Install **Nuranova Mocap Camera** (`NuranovaMocapCamera.apk`) on 2 to 12 Android smartphones (plus optional dedicated close-up face/hand phones).
3. Install **Nuranova Mocap Bridge** (`NuranovaMocapBridge.zip`) into Blender 4.5 LTS or 5.0+.

---

## 2. Step-by-Step Capture Workflow

### Step 1: Create a Portable Project
- Open **Projects** (or **Welcome -> New Project**). Each project creates a self-contained folder with `project.nmocap`, `project.db`, `calibration/`, `recordings/`, `raw_motion/`, `corrected_motion/`, and `exports/`.
- Use **Backup / Export ZIP** to move projects seamlessly to another PC.

### Step 2: Connect & Pair Smartphones (2 to 12 Cameras)
- Go to **Camera Manager** and click **Start Local Server**. Note the displayed **6-digit Pairing PIN**.
- Open **Nuranova Mocap Camera** on each smartphone, grant Camera/Audio/Notification permissions, select your discovered PC (or enter IP & Port `8765`), enter the PIN, and assign role (`body`, `face`, or `hand_dual`).
- Go to **Camera Configuration** to verify each phone's supported resolution/FPS. Preferred baseline: **1080p @ 60 FPS**, **1/250s shutter**, **Locked Focus**, **Locked White Balance**, **Digital Stabilization OFF**. Mixed resolutions/frame rates are supported automatically.

### Step 3: Multi-Camera Calibration
- Open **Calibration Wizard**.
- Print `documentation/printable/Nuranova_ChArUco_A4_7x5_35mm.pdf` (or A3) at **100% Actual Size** (disable Fit-to-Page!).
- Mount the printed board on a rigid flat foamcore/acrylic backing.
- **Measure the actual printed square length and 100 mm bar with a ruler/caliper**, and enter the exact measured value in millimeters in the Calibration Wizard.
- Capture a multi-angle sweep visible across overlapping camera pairs, run calibration, verify per-camera reprojection error (target `<= 1.0 px`) and known-distance validation error (target `<= 10 mm`), and save your calibration profile.
- **Warning:** Changing camera tripod position, zoom, resolution, or enabling stabilization invalidates calibration!

### Step 4: Wearable Physical Marker Kit (Optional / Hybrid Cinema Mode)
- Open **Marker Editor** to customize markers across 16 anatomical groups and export `Nuranova_Wearable_AprilTag_36h11_40mm.pdf`.
- Wear matte black clothing with high-contrast 19 mm white matte spheres and rigid asymmetric 4-marker clusters on Head, Chest, Pelvis, Forearms, and Feet.

### Step 5: Record & Reconstruct
- Open **Capture Studio** to inspect the 3D volume, camera previews, sync residuals, battery, temperature, and storage.
- Click **Start Synchronized Recording**. Phones record full-quality H.264 MP4 + sensor timestamps locally (never saturating Wi-Fi), then upload via resumable SHA-256 verified transfer. Local footage is retained until checksum verification succeeds.
- Queue offline processing in **Processing Queue**:
  - **Fast Preview:** Quick approximate inspection.
  - **Standard:** Full-body reconstruction with basic cleanup.
  - **Cinema (Recommended):** Highest validated quality, nonlinear triangulation & skeletal IK refinement, hybrid marker constraints, and 2-bone IK foot locking.

### Step 6: Clean Up in Motion Editor
- Open **Motion Editor** to scrub frames, inspect joint confidence colors, compare **Raw vs Corrected** skeleton overlays & curves, tune **Butterworth / One Euro** filtering, toggle **2-Bone IK Foot Locking** and **Ground Clamp**, reprocess selected frame ranges, and **Undo / Redo** non-destructively.

### Step 7: Retarget & Animate in Blender
- Click **Character Retargeting -> Publish Active Take to Blender Bridge** (or export `.nmocap.json` / `.bvh` / `.fbx` in **Animation Exports**).
- Inside Blender (`N`-panel -> **Nuranova Mocap**):
  1. Click **Check Studio Connection** -> **Pull Take** (or import offline `.nmocap.json`).
  2. Pick your **Target Character Rig** (Rigify or custom stylized character) and optional **Target Face Mesh** (with 52 shape keys).
  3. Click **Auto-Match Rig Bones** (adjust manually if needed and save a mapping profile).
  4. Click **Retarget & Bake Editable Keyframes** — generates pure editable quaternion keyframes, Actions, and NLA strips with rest-pose delta & proportion scale compensation!
  5. Export directly to **FBX** or **BVH** when finished.
