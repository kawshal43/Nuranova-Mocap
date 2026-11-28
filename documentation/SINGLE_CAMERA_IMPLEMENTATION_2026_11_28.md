# Single-camera implementation and validation

28 November 2026 · delivery branch `kawshal/bugFix`.

The single-video workspace now uses a dedicated, reviewable reconstruction
pipeline. It supports RTMW3D through ONNX Runtime CUDA, with MediaPipe as the CPU
fallback. Automatic selection uses locally installed, verified models. Selecting
RTMW3D explicitly downloads its pinned checkpoint on first use.

This is a tested upgrade, not a claim of perfect monocular reconstruction.
The third recording still has unreliable foot placement. Hidden finger depth
and absolute travel cannot be verified from these recordings alone.

## Changes delivered

- Preserve native image points, hand estimates, presentation timestamps, source
  identities and review state in the capture document. The H5 file can restore
  native review data even without its separate preview NPZ.
- Apply body and finger edits, exclusions, hand-crop overrides and bounded depth
  adjustments to the solve. Saved raw previews cannot override a newer model or
  a changed video. Explicit review detections carry separate correction metadata.
- Reuse tracking for corrected solves. Range solves can reuse the full recording's
  cache with boundary context and produce a separate range take.
- Honor model, CPU/GPU preference, performer height, lens estimate and movement
  controls. Both desktop offline entry points use this pipeline.
- Fit one sequence scale and floor, retain relative vertical movement, and use
  source-image foot-contact evidence. Export a uniform 30 FPS animation while
  keeping every original observation and timestamp.
- Use native hand crops, reject contradictory wrist assignments, reduce confidence
  for tiny palms, and apply the existing anatomical finger and forearm constraints.
- Bound the detail-worker queue to two small crop jobs. Avoid exhaustive CUDA
  convolution search at the first frame. Decompress each cache member once.
- Add one-video controls, orange uncertainty bands, next-interval navigation,
  depth adjustments and range solving. Existing independent previews and zoom remain available.
- Display unavailable cross-view reprojection as N/A; export JSON `null`.
  Legacy single-video scorecards also stop displaying their placeholder zero.
- Preserve quaternion signs through physics, wrist fitting and Blender import.
  This fixes flips between otherwise correct keyframes.
- Keep the rest pose in motion-only FBX files. Compare fractional-rate FBX
  animation using the Blender importer's actual key-time units. Continue checking
  joint positions, deformation matrices, hierarchy, duration, surfaces and morphs.

## Recorded tests

The automated suite passed **169 tests**, with one optional Pose2Sim footage test
skipped because its external reference videos are absent. The new tests cover
camera units/axes, height scaling, jump preservation, in-place motion, manual
pins, native-time persistence, cache/range reuse, source identity, wrong-wrist
rejection, moving-background fallback and real Blender exports with opposite
quaternion signs and a fractional frame rate.

Each supplied recording was processed independently. These figures are internal
consistency and model-confidence statistics, **not measured 3D accuracy**:

| Source | Native frames | Output frames at 30 FPS | Grade | Estimated stance drift |
| --- | ---: | ---: | --- | ---: |
| `20261127_015632.mp4` | 3,019 | 3,018 | B | 3.86 mm |
| `VID_20261127_015637.mp4` | 2,314 | 2,309 | B | 1.58 mm |
| `VID_20261127_015801.mp4` | 1,526 | 2,891 | C | 17.50 mm |

All three source-file hashes remained unchanged. H5 reload reproduced corrected
positions exactly. Output quaternion tracks have no antipodal sign changes.
Additional output frames on the last video are interpolated estimates; they add
no new observations.

The earlier 20–28 second CPU baseline took 44.24 seconds. The final fresh GPU run
took **27.29 seconds in Detailed mode** and **22.12 seconds in Faster mode**, about
1.62× and 2.0× faster. Both processed 240 native frames and produced 239 output
poses from the same source interval. Reference accuracy and the no-regression
quality gate remain unmeasured. A full 3,019-frame tracking
cache now loads in about 0.105 seconds after removing repeated NPZ decompression;
the subsequent motion solve still takes additional time.

Blender MCP checked all **8,218 integer poses and 8,215 halfway poses** on 55-bone
source rigs. Maximum halfway interpolation error was below 0.00005 degrees.
Both motion-only and character FBX/GLB files passed fresh-import checks. The
corrected character exports also passed deformed-surface and facial-channel checks.
The character repair copy was inspected at 477 integer/halfway samples; residual
contacts remain and are recorded separately from export integrity.

The actual Windows bundle passed 21-page startup, native preview selection/zoom,
CUDA tracking, single-camera persistence and Blender export checks. Forced CPU
mode selected MediaPipe/XNNPACK and saved native observations correctly. Code
for 31 packaged modules/entry points and all bundled Blender sources was compared
against the delivered source. The warm packaged startup check took 3.484 seconds.

Detailed evidence is under
`D:/2026/nuranova-mocap/Mcapn_results/single_camera_implementation_2026_11_28/`.
The compact [validation record](SINGLE_CAMERA_VALIDATION_2026_11_28.json) and
[task tracker](SINGLE_CAMERA_OPTIMIZATION_TASKS_2026_11_28.csv) distinguish completed
engineering from remaining quality gates.
The [bug sheet](SINGLE_CAMERA_BUG_REPORT_2026_11_28.csv) records fixes and open
quality limitations. Reproduce a solve with:

```powershell
.venv/Scripts/python.exe scripts/benchmark_monocular.py VIDEO.mp4 --output RESULTS --backend rtmw3d --start 20 --end 28 --no-cache
```

## Remaining gates

- The low-frame-rate recording remains Grade C. Review its feet and weak-evidence
  intervals. A small average drift in the other takes does not prove correct
  physical placement or absence of character intersections.
- Character contact correction operates on the chosen rig/mesh and keeps an
  editable copy. Residual sampled contacts must be inspected; no guarantee of
  collision-free motion on every character is made.
- Background-motion detection can disable unreliable estimated travel. Full
  moving-camera/world-motion separation is not implemented.
- SAM 3D Body is not installed. Checkpoint access, its separate terms, an isolated
  worker and measured memory/accuracy remain prerequisites. No custom model was
  trained on the three test views.
- Labeled 3D reference data and curated fast/crossing/occlusion gestures are still
  needed to establish accuracy and compare backends beyond confidence statistics.
- Manual finger adjustments need an available hand estimate. Re-detect a clearer
  hand crop or supply another view when its local 3D estimate is missing.

## Windows and Blender delivery

Run `dist/NuranovaMocapStudio/NuranovaMocapStudio.exe` with its whole adjacent
folder. The Blender bridge is version **1.3.1**, packaged separately as
`installers/NuranovaMocapBridge.zip`. Model weights remain outside the Windows
bundle and retain their checksum/provenance in the registry.
