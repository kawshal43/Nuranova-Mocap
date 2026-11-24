# Third-Party License Register — Nuranova Mocap Studio

**Developer:** Nuranova Solutions (https://nuranova.lk/)  
**Last Updated:** 2026-11-24

This register audits all reference repositories, runtime libraries, pretrained models, and build tools for commercial product distribution compliance.

## 1. Excluded / Prohibited Reference Repositories (NOT Incorporated)

| Component | License | Risk / Obligation | Action Taken |
|---|---|---|---|
| `references/freemocap-main` | **AGPL-3.0** | Strong network copyleft; requires disclosing complete application source under AGPL-3.0 if incorporated or derived. | **STRICTLY EXCLUDED.** Zero code copied; not a dependency. |
| `references/UnderPressure-main` (`pretrained.tar` + source) | **InterDigital Limited Software Evaluation License** | Strictly non-commercial research evaluation only; explicitly prohibits commercial use, redistribution, or product inclusion. | **STRICTLY EXCLUDED.** Zero code or weights used. Replaced by clean-room kinematic + IK foot-locking module. |

## 2. Python Desktop & Engine Runtime Dependencies

| Package | Pinned Version | License | Commercial Obligations |
|---|---|---|---|
| `numpy` | 1.26.4 | BSD-3-Clause | Include copyright notice & license text in `licenses/` |
| `scipy` | 1.13.1 | BSD-3-Clause | Include copyright notice & license text |
| `opencv-contrib-python` | 4.10.0.84 | Apache-2.0 | Include Apache 2.0 notice; avoid patented non-free modules (`OPENCV_ENABLE_NONFREE` is OFF in standard wheels) |
| `PySide6` (Qt 6 for Python) | 6.7.3 | **LGPL-3.0** | Dynamic linking (standard PyInstaller external DLL/pyd bundling), preserve Qt LGPL notice, allow user relinking of Qt shared libraries, do not statically link Qt. |
| `pyvista` / `pyvistaqt` / `vtk` | 0.44.2 / 0.11.1 | MIT / BSD-3-Clause | Include copyright notices |
| `fastapi` / `uvicorn` / `starlette` / `pydantic` | 0.115.6 / 0.32.1 / 2.10.4 | MIT / BSD-3-Clause | Include copyright notices |
| `websockets` / `httpx` | 13.1 / 0.28.1 | BSD-3-Clause | Include copyright notices |
| `zeroconf` | 0.136.2 | **LGPL-2.1** | Dynamic import / standard Python wheel; preserve LGPL-2.1 notice |
| `SQLAlchemy` | 2.0.36 | MIT | Include copyright notice |
| `h5py` / HDF5 | 3.12.1 | BSD-3-Clause | Include copyright notice |
| `av` (PyAV) | 13.1.0 | BSD-3-Clause (bundles FFmpeg **LGPL-2.1+** wheels) | PyAV PyPI wheels bundle LGPL-compiled FFmpeg (no GPL `--enable-gpl` encoders like x264 unless separately installed). Verify LGPL FFmpeg build before commercial release. |
| `Pillow` | 11.0.0 | MIT-CMU | Include copyright notice |
| `reportlab` | 4.2.5 | BSD-3-Clause | Include copyright notice |
| `mediapipe` | 0.10.14 | Apache-2.0 | Include Apache 2.0 notice |
| `onnxruntime-gpu` / `onnxruntime` | 1.19.2 | MIT | Include MIT notice |
| `rtmlib` | 0.0.16 (from `references/rtmlib-main`) | Apache-2.0 | Include Apache 2.0 notice |
| `tqdm` | 4.67.1 | MPL-2.0 / MIT | Dual-licensed; used under MIT |

## 3. Build & Packaging Tools (Not Redistributed as Proprietary Code)

| Tool | License | Redistribution Notes |
|---|---|---|
| `PyInstaller` 6.11.1 | GPL-2.0 **with Bootloader Exception** | Bootloader exception explicitly allows bundling closed-source commercial applications. |
| `Inno Setup` 6 | Modified BSD-like (Inno Setup License) | Free for commercial installer distribution; preserve license notice. |
| `pytest`, `ruff` | MIT | Dev/test only; not shipped in installer. |

## 4. Android Application Dependencies

| Library | License | Notes |
|---|---|---|
| Kotlin Stdlib / Coroutines | Apache-2.0 | Permissive |
| AndroidX / Jetpack Compose / Room / DataStore / WorkManager / Camera2 | Apache-2.0 | Permissive |
| OkHttp 4.x | Apache-2.0 | Permissive |

## 5. Blender Extension Licensing Note

| Component | License | Notes |
|---|---|---|
| `Nuranova Mocap Bridge` (`blender_addon/`) | **GPL-3.0-or-later** | Blender's Python API (`bpy`) is GPL-3.0. Per Blender Foundation licensing guidelines, Python extensions importing `bpy` are distributed under GPL-3.0-or-later as an independent installable extension (`NuranovaMocapBridge.zip`), communicating with Nuranova Mocap Studio exclusively over documented local HTTP/JSON or open file formats (`.nmocap.h5`, `.nmocap.json`, `.bvh`), preserving clean license separation. |

## 6. Pretrained AI Model Weights Register (Review Before Commercial Sale)

| Model | Source | Code License | Weights / Training Data Advisory |
|---|---|---|---|
| **YOLOX-m / YOLOX-tiny** (via RTMLib) | OpenMMLab (`download.openmmlab.com`) | Apache-2.0 | Trained on COCO / HumanArt; verify dataset commercial terms before commercial redistribution of bundled weights. Recommended: download on-demand via Model Manager with user acknowledgement. |
| **RTMW-x / RTMW-l / RTMPose-m** (via RTMLib) | OpenMMLab MMPose | Apache-2.0 | Apache-2.0 code; trained on Cocktail14 / COCO-WholeBody / HALPE (some constituent academic datasets have non-commercial research clauses). **Action:** Do not hard-bundle weights into closed installer without legal review; provide on-demand Model Registry downloader with explicit provenance, SHA-256 verification, and license display. |
| **MediaPipe Pose / Face / Hand Landmarker** | Google AI Edge (`storage.googleapis.com/mediapipe-models/`) | Apache-2.0 | Apache-2.0 code and TFLite task models released by Google under Apache-2.0. Commercially permissive with attribution. |
