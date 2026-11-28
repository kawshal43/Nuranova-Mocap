# Follow-up bug fixes and verification

28 November 2026. Branch: `kawshal/bugFix`. Blender Bridge: **1.3.2**.

Four regressions were reproduced before applying their fixes. Each now passes
against the production implementation in
[`test_followup_reliability.py`](../tests/integration/test_followup_reliability.py).

| ID | Reproduced behavior | Correction and regression check |
| --- | --- | --- |
| FOLLOW-01 | Floor cleanup reduced a 25 cm airborne pose to about 8 mm even with explicit no-contact evidence. | Require foot-contact evidence before applying a gravity floor pin. The raised feet retain more than 20 cm clearance. |
| FOLLOW-02 | A smooth 170–190 degree wrist correction took the long rotation path, with about 175 degrees of error. | Smooth and limit corrections directly with quaternion interpolation. The crossing stays within 3 degrees of the intended rotation; correction steps stay within 6.001 degrees at 30 FPS. |
| FOLLOW-03 | Initial review bounds and first/last-frame fallback assumed the recording started at timestamp zero. | Probe the native stream origin and duration. A real encoded MP4 starting at 2 seconds has the correct timeline and first/last-frame navigation before a frame index or observations are loaded. |
| FOLLOW-04 | Repeated identical review refreshes invalidated an image still being decoded, so the completed image was discarded. | Reuse an identical pending decode. A deterministic delayed-decoder test now accepts the frame despite repeated layout/selection refreshes; a changed timestamp or resolution still replaces the request. |

Partial Blender corrections retain their boundary rotations and preserve curves
outside the reviewed range. If a boundary cannot be preserved safely, the bridge
restores that bone's original rotations and lists the rejected correction in its
report. Raw correction actions remain available for further editing.

An early standalone CPU check timed out waiting for a preview. Subsequent CPU
checks passed, and tracing identified the separate reproducible refresh race
above. The smoke report now records decoder liveness, requested cameras,
generations and source times on failure. The timeout is retained in
`preview_delivery_incident.json`; its exact cause was not conclusively established.

## Verification

- Full Python suite: **173 passed, 1 skipped**, no failures or errors. The skipped
  test requires optional Pose2Sim reference videos absent from this checkout.
- Ruff and `git diff --check` passed.
- All three supplied recordings were solved again, reusing their existing native
  detector caches. Reconstruction, physics, serialization and motion exports ran
  again. Each source SHA-256 stayed unchanged; reloaded H5 positions matched the
  in-memory result exactly. Body and hand tracks contain zero antipodal steps.
- Blender MCP evaluated **8,218 integer poses and 8,215 halfway poses** across the
  three 55-bone source rigs. Positions and scales stayed finite; maximum halfway
  interpolation error was below 0.00005 degrees. The original open scene was
  restored without saving it.
- The supplied character was regularized again from its saved raw corrections
  and inspected at **477 integer/halfway poses**. Both FBX and GLB passed fresh
  import checks with 55 bones and 60 shape channels. Maximum sampled export
  surface error was below 0.03 mm. This checks export fidelity, not capture accuracy.
- The Windows bundle was rebuilt. Its 32 checked code components match current
  source, and packaged Blender code matches the bridge source. All 12 bridge ZIP
  entries were compared with their source files.
- The final executable passed isolated CUDA and fresh-state CPU runs: all 21
  workspaces, real startup progress, source preview, visible selection, pop-out
  return, 175% video zoom and bundled Blender GLB export. CUDA ran RTMLib 2D and
  RTMW3D; explicit CPU mode ran RTMLib CPU and MediaPipe/XNNPACK 3D. Both retained
  native source observations in the solved take. Recorded startup was 10.578 s
  for the CUDA run and 2.094 s for CPU; these are individual workstation runs.

The repeated single-video consistency measurements are:

| Recording | Native observations | Output poses at 30 FPS | Grade | Estimated stance drift |
| --- | ---: | ---: | --- | ---: |
| `20261127_015632.mp4` | 3,019 | 3,018 | B | 3.86 mm |
| `VID_20261127_015637.mp4` | 2,314 | 2,309 | B | 1.58 mm |
| `VID_20261127_015801.mp4` | 1,526 | 2,891 | C | 17.43 mm |

These are internal consistency statistics. The extra output poses in the last
recording are interpolated estimates. No reference 3D accuracy is claimed.

## Remaining quality issues

The third recording still has unreliable foot placement; its maximum measured
stance drift remains 86.18 mm. Occluded finger depth and absolute performer travel
also remain estimates.

The character geometry audit still reports contacts at **310 of 477 sampled
poses**, with maximum sampled penetration about **70.78 mm**. The previous result
was 311 poses and 70.82 mm; this is not a meaningful contact-quality improvement.
The rotation fix removes the reproduced wrap bug, but does not solve every
character intersection. Export validation and interpolation checks must not be
interpreted as a collision-free or bug-free claim.

## Workstation evidence and outputs

Evidence is stored outside Git at:

`D:\2026\nuranova-mocap\Mcapn_results\followup_bugfix_2026_11_28`

- `regressions.xml`: full test results.
- `full_clips/report.json`: all three solves, source hashes and consistency checks.
- `blender_mcp_release_audit.json`: all source poses and midpoint checks.
- `character_review.json`: character contacts, correction continuity and exports.
- `SingleCamera_Character_ContactReview.blend`: separate editable review copy.
- `single_camera_contact_review.fbx` and `.glb`: verified character exports.
- `windows_bundle_manifest.json`: packaged code verification and CUDA smoke test.
- `frozen_smoke.json` and `cpu_final_fresh/frozen_cpu_delivery.json`: final standalone checks.

The Windows app is `dist/NuranovaMocapStudio/NuranovaMocapStudio.exe`; retain its
whole directory, including `_internal`. The installable Blender extension is
`installers/NuranovaMocapBridge.zip`.
