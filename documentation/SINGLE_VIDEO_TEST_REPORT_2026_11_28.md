# Tests of the two new single-camera videos

28 November 2026. Sources: `D:\2026\nuranova-mocap\single\1.mp4` and `2.mp4`.
Delivery branch: `kawshal/bugFix`.

Both videos were decoded and solved at every native presentation timestamp.
The fixes below improve movement preservation and delivery reliability. These
tests do **not** establish perfect movement prediction or reference 3D accuracy.
Camera motion, occlusion, loose clothing and small hands still limit the result.

## Corrections

| ID | Reproduced problem | Delivered change |
| --- | --- | --- |
| SV-01 | A moving-camera fallback made the entire take stationary, also removing the performer's travel. | Match masked background features directly to a reference image, estimate translation/zoom/roll, and retain supported movement. Insufficient support retains the fallback. Short fades receive uncertainty intervals rather than disabling the whole take. |
| SV-02 | A long detection gap could make estimated camera distance become zero. | Hold/interpolate missing roots from supported estimates. The 40-frame synthetic gap changed from a minimum distance of 0 m to 4.023 m, with a 4 m reference. |
| SV-03 | Cleanup erased a coherent fast gesture as a short glitch. Repeated smoothing also weakened recovered movement. | Preserve coherent, confident and observable fast limbs through physics and the final body filter. A known 100-degree synthetic gesture changed from 0 degrees after cleanup to 97.89 degrees after physics and 95.15 degrees after both filters. Isolated rotation spikes remain rejected. |
| SV-04 | Standalone cached solves required loose Python source files absent from the executable bundle. | Fingerprint the embedded application through its executable. The delivery smoke test now solves twice with caching enabled and asserts a cache hit on the second solve. |
| SV-05 | Changes to the monocular detector adapter did not invalidate its native cache. | Include that adapter in development cache fingerprints. New executable builds invalidate frozen inference caches. |
| SV-06 | Exporting a shared character from a secondary Blender scene could omit its animation or use another scene's skin. | Restrict animated and static GLB/glTF export to the active scene. Bridge 1.3.3 includes a real multi-scene animation/deformation regression. |

The camera correction uses existing OpenCV SIFT and robust feature matching.
It compensates a two-dimensional similarity transform, including roll in body
and palm coordinates. It does not estimate a metric 3D camera path, translation
parallax or arbitrary perspective changes. Face blendshape capture remains
available; compensated head orientation uses aligned body landmarks so native
face-camera roll is not mixed back into the skeleton. The method follows
[OpenCV's feature-matching guidance](https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html);
[camera calibration](https://docs.opencv.org/4.x/d4/d94/tutorial_camera_calibration.html)
remains necessary for measured camera geometry.

## Verification

- **186 automated tests passed; 1 optional test skipped.** The skipped Pose2Sim
  test needs external reference videos absent from this checkout. Ruff passed.
- Fresh RTMW3D/CUDA inference processed both complete recordings. Repeated solves
  reused the verified native inference caches while exercising the new geometry,
  camera correction, cleanup, serialization and export paths. Cache provenance is
  recorded separately; detector results were not silently replaced or relabeled.
- Original source SHA-256 hashes stayed unchanged. H5 reload reproduced corrected
  joint positions exactly. Native timestamps are ordered, and no source frames
  were omitted from the solve. Header counts include three extra non-decoded
  entries in each file; the table lists actual decoded frames.
- Blender MCP checked **2,380 integer poses and 2,378 halfway poses** on the
  55-bone source rigs. Position/scale values were finite and interpolation stayed
  on the short quaternion path. Confident fast limbs may exceed the old
  30-degree-per-frame diagnostic threshold; observed maxima were 38.78 and 37.21
  degrees. Body and hand exports have zero antipodal quaternion steps.
- The rebuilt Windows executable passed CUDA and explicit CPU checks using the
  new videos. Each checked all 21 workspaces, preview selection/pop-out/zoom,
  an actual monocular solve, a cached second solve, camera compensation and
  bundled Blender GLB export. CPU used MediaPipe/XNNPACK for 3D; GPU used RTMW3D.
  **34 packaged code components** were compared with current source.
- Both complete retargeted characters passed fresh-import FBX and GLB checks:
  **2,669 pose samples per format for video 1** and **2,089 per format for video 2**,
  with 55 bones and 60 shape channels. Maximum sampled export surface difference
  was below 0.04 mm. This establishes export fidelity, not pose accuracy.
- The earlier three recordings also completed regression solves, with unchanged
  source hashes and exact H5 reloads. Estimated stance drift was 3.84, 1.65 and
  17.39 mm; grades remained B, B and C.
- Bulk mesh-coordinate sampling made the surface-tree construction step about
  **3.3 times faster** on the 16,710-vertex test mesh. The measured nearest-surface
  distances were identical for the sampled queries. This is a component benchmark,
  not a claim that complete export is 3.3 times faster.

## Per-video measurements

These measure model confidence and consistency, not motion capture accuracy.

| Source | Native/output frames | Grade | Estimated stance drift | Body confidence coverage | Finger confidence coverage |
| --- | ---: | --- | ---: | ---: | ---: |
| `1.mp4` | 1,335 / 1,335 | B | 5.65 mm | 91.9% | 87.3% |
| `2.mp4` | 1,045 / 1,045 | B | 1.93 mm | 89.4% | 87.4% |

Restored travel is an estimate in the reference camera's coordinates. It must
not be treated as measured world-space travel. Actor height used the configured
default of 1.72 m; the recordings provide no measured height or calibrated lens.

The source-versus-solve videos project the resulting skeleton through an
estimated camera. Their diagnostics show improvement in the median but also
remaining errors and tradeoffs:

| Diagnostic | Video 1 before → after | Video 2 before → after |
| --- | ---: | ---: |
| Median source-point projection difference | 83.62 → 59.55 px | 79.57 → 64.63 px |
| 95th-percentile source-point projection difference | 148.10 → 140.34 px | 142.67 → 146.97 px |
| Median root-relative model-target difference | 70.42 → 82.95 mm | 62.16 → 67.66 mm |
| 95th-percentile root-relative model-target difference | 280.15 → 330.39 mm | 213.17 → 232.67 mm |

The last two rows compare with model estimates, not physical reference data.
They became worse even while estimated travel and median image agreement improved.
No universal accuracy improvement or accuracy non-regression gate is claimed.
Projection differences also depend on estimated camera/floor registration.

## Moments requiring review

- Video 1: **44.367–44.500 s**, camera/background support is lost during the fade.
- Video 2: **33.567–34.833 s**, the fade loses camera/background support and later
  body evidence. Unsupported global motion is held/interpolated; visible relative
  limb poses remain separate from camera confidence.
- Turned-away hands, hidden fingers, knees beneath clothing and inferred depth
  remain uncertain. The comparison videos include visible pose mismatches;
  passing interpolation or export checks does not make those predictions exact.
- Character intersections are checked separately from motion and export integrity.
  Collision-free animation on every character has not been established.

The character audit checked all 4,758 integer/halfway poses at 20 samples per
region. These are retargeted review copies, not collision-cleaned final animation:

| Character take | Poses with any reported contact | Poses with sampled depth above 10 mm | Maximum sampled depth |
| --- | ---: | ---: | ---: |
| Video 1 | 2,540 / 2,669 | 1,012 / 2,669 | 99.76 mm |
| Video 2 | 2,080 / 2,089 | 1,239 / 2,089 | 100.45 mm |

The first column includes finger triangle crossings with zero or tiny sampled
depth, but the deeper arm/head/torso/leg intersections are material remaining
problems. Passing export checks must not be described as fixing these contacts.
The original character input was not overwritten; the live MCP test scenes were
removed after their audit while preserving the user's active scene.

## Evidence and review files

Workstation output directory:

`D:\2026\nuranova-mocap\Mcapn_results\single_new_videos_2026_11_28`

- `source_manifest.json`, `native_clock_audit.json`: source hashes and decoded timing.
- `baseline/benchmark.json`, `corrected/benchmark.json`: complete before/after solves.
- `native_cache_provenance.json`: verified reuse of unchanged native inference.
- `missing_root_regression.json`, `fast_motion_regression.json`: reproduced failures and fixes.
- `motion_fidelity_review.json`, `*_frame_audit.npz`: per-frame diagnostic arrays.
- `1_before_after_review.mp4`, `2_before_after_review.mp4`: silent, full-length
  source/skeleton comparisons, red before and green after. These use estimated
  projection and are review aids rather than accuracy certificates.
- `blender_mcp_audit.json`: all source key poses and halfway poses.
- `character_validation.json`, `character_*_geometry.json`: full character export
  results and per-pose contact measurements.
- `SingleVideo_1_Character_Review.blend`, `SingleVideo_2_Character_Review.blend`:
  editable retargeted review copies; the corresponding `.glb` and `.fbx` files are
  verified exports of those same poses, including their remaining intersections.
- `surface_benchmark.json`: surface-check component timings and equivalence probe.
- `regressions.xml`: the full test suite.
- `windows_bundle_manifest.json`, `frozen_smoke.json`, `frozen_cpu_delivery.json`:
  checks of the actual rebuilt executable and enabled inference cache.

Run `scripts/benchmark_monocular.py` with `--backend rtmw3d` to reproduce a solve.
The Windows app is `dist/NuranovaMocapStudio/NuranovaMocapStudio.exe`; retain its
whole directory, including `_internal`.
Install `installers/NuranovaMocapBridge.zip` for the interactive Blender bridge
1.3.3 scene-export fix as well.

One repeated test run stopped correctly at the C: free-space guard. The final
suite ran on D: and passed. This session's old test folders 218–220 were preserved
on D: rather than deleted. C: still has only about **0.50 GiB free**; normal work
there can hit the application's disk-space guard. No unrelated files were removed.
