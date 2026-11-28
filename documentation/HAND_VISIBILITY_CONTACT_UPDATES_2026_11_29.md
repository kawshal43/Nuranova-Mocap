# Hand, visibility, audio and character refinements

29 November 2026. Delivery branch: `kawshal/bugFix`.

The BlendCap review identified useful approaches that fit Nuranova's existing
solvers. This change implements those approaches with the existing NumPy/SciPy
dependencies. No BlendCap source, model weights, detector, installer or runtime
dependency is incorporated. SAM 3D Body remains an unbenchmarked candidate and is
not added to the app by this change.

## Delivered behavior

- **Image-based hands:** fit the existing fixed-length anatomical hand to native
  video points, with bounded finger angles, palm correction, tendon coupling and
  a weak temporal prior. Small, missing or inconsistent hands keep their prior.
  Accept only fits that improve supported image agreement; never increase the
  observation confidence. The single-camera and multi-camera paths both use this
  pass. Manual image edits are used by the fit, and the inference cache stays raw.
- **Camera visibility:** use reliable estimated body segments to downweight
  obstructed sightlines before calibrated reconstruction and hand-view selection.
  Keep nonzero weights because the body itself is estimated. Preserve manual
  point decisions and bypass this weighting for marker-only capture. Native
  pixels and calibrated camera geometry are retained.
- **Audio alignment:** add spectral-onset matching for differing microphone
  responses, require agreement across multiple sections, and reject repeated-beat
  ambiguity. Keep precise waveform alignment when supported, together with the
  existing native timestamps and clock-drift fit. A single clap is not classified
  as repeated music merely because its surrounding silence is ambiguous.
- **Blender bridge 1.3.4:** record stationary foot positions before correction;
  refine remaining arm/hand intersections by moving the penetrating surface with
  small joint rotations. Validate proposals against the evaluated mesh and
  endpoint displacement bounds. Apply this after the established IK correction,
  regularize the added correction, and inspect integer/intermediate poses.
  An incomplete bone map now produces the intended review error. Preserve the
  keyed quaternion hemisphere when decomposing poses for partial-range cleanup,
  preventing artificial intermediate-frame twists at 180-degree crossings.

Hands run automatically when finger capture is enabled. Visibility weighting
runs on calibrated multi-camera AI/hybrid solves. Audio refinement is part of
automatic synchronization. In Blender, install the updated bridge and use
**Character contacts & correction > Fix range (copy)**. The character operation
creates a separate action/mesh copy and retains unresolved contact reports.

## Verification

**205 tests passed; 1 optional external-footage test skipped. Ruff passed.**
The skipped Pose2Sim test needs reference videos absent from this checkout.

Nineteen additional tests cover known image poses for both hands and coordinate
conventions; constant bone lengths; tiny/missing/interpolated evidence;
calibrated hidden-view reconstruction; manual/marker precedence; microphone
frequency differences; repeated beats; silence/unrelated events; a 7.2-second clap
offset; and real skinned-mesh correction with a fixed wrist.
The partial-range quaternion test also checks stored signs, halfway poses and
unchanged interpolation outside the corrected range across a 180-degree turn.

The first complete regression run caught a new spectral ambiguity check rejecting
the existing 7.2-second clap case. The check was corrected, the specific regression
passed again, and the full suite then passed.

Full-character GLB validation subsequently exposed a latent correction-boundary
bug: decomposing pose matrices discarded the stored quaternion hemisphere. Six
negative quaternion steps caused a maximum joint mismatch of 843 mm and a sampled
surface mismatch of 406 mm on the failed review copy. The new regression reproduced
the problem before the fix and passed after preserving the keyed hemisphere.
Character review copies were regenerated with the corrected code and rechecked.
The regenerated first character has zero negative quaternion steps across the
complete clip, compared with six on the failed copy.

Both supplied videos were re-solved in full from the verified unchanged native
inference caches. Every decoded source timestamp is represented, source hashes
are unchanged, and H5 reload reproduces the corrected body positions exactly.
Cached inference reuse is recorded; the hand fitting and reconstruction still run.

| Source | Native/output frames | Accepted hand samples | Estimated stance drift |
| --- | ---: | ---: | ---: |
| `single/1.mp4` | 1,335 / 1,335 | 562 | 5.65 mm |
| `single/2.mp4` | 1,045 / 1,045 | 345 | 1.93 mm |

A sample is one hand at one native time. There were 3,289 available hand samples,
2,081 eligible image fits and 907 accepted refinements. Others retain their prior.
The paired image-error medians below concern accepted fits before final temporal
filtering; they are not physical 3D accuracy measurements.

| Source/hand | Before | After |
| --- | ---: | ---: |
| 1 / left | 3.66 px | 1.45 px |
| 1 / right | 3.78 px | 1.51 px |
| 2 / left | 5.07 px | 1.60 px |
| 2 / right | 4.98 px | 1.72 px |

The final output preserves the previous root trajectories exactly. Finger tracks
retain zero antipodal quaternion steps; maximum finger steps remain below 13.6
degrees per frame. Some step percentiles increase slightly with recovered motion;
this is not a claim that every smoothness metric improves.

An isolated Blender MCP session checked all **2,380 integer poses and 2,378 halfway
poses** on the 55-bone source rigs. Positions/scales are finite and quaternion
interpolation follows the short path. The user's live MCP endpoint was unavailable;
the installed add-on's background CLI supplied the isolated session. Test scenes
were removed afterward and only that owned helper process was stopped.

The rebuilt Windows application passed explicit CUDA and CPU checks, each with
37 native video samples, actual hand fitting, an enabled inference cache round
trip, the 21 workspaces, preview selection/pop-out/zoom and Blender GLB export.
The packaged spectral-onset example aligned within 3 ms. Thirty-eight packaged
code components and Blender resources were compared with source. The CUDA sample
accepted 27 hand fits; the CPU/MediaPipe sample accepted four.

On 14 fixed character poses, the added surface polish reduced aggregate sampled
squared penetration from 0.021166 to 0.014419 square meters (31.9%) relative to the
previous correction, with no maximum-depth regression above 0.1 mm. This is a
limited pose probe, not a full-animation improvement percentage.

The delivered correction path was also exercised on these explicit character
ranges: video 1 frames **1–30, 690–735, 980–1020**; video 2 frames **280–320,
520–560, 700–740**. Other frames retain the retargeted motion. Final checks include
integer/halfway poses and additional quarter poses where contacts are found.

| Character | Contact validation poses | Before/after summed squared depth | Poses still reporting contacts | Maximum remaining sampled depth |
| --- | ---: | ---: | ---: | ---: |
| Video 1 | 428 | 1.9497 / 0.6403 m² | 395 | 68.28 mm |
| Video 2 | 479 | 3.4746 / 1.9524 m² | 473 | 99.67 mm |

Contact counts include triangle crossings with tiny or zero sampled depth.
The deeper remaining intersections still require review. These totals cover the
selected ranges, not every pose of the entire character animation.

Fresh-import export checks cover each complete character recording, including
55 bones and 60 shape channels:

| Character / format | Pose samples | Maximum joint difference | Maximum sampled surface difference |
| --- | ---: | ---: | ---: |
| Video 1 / GLB | 2,669 | 0.031 mm | 0.032 mm |
| Video 2 / FBX | 2,089 | 0.026 mm | 0.024 mm |

These differences establish export fidelity, not movement accuracy. Original
character inputs were not overwritten. The review files select the corrected
character and retain a separate comparison base.

## Remaining limits

Single-view depth, hidden fingers, loose clothing, camera parallax and ambiguous
contacts remain uncertain. Image fitting is local weak perspective; an improved
projection can still have incorrect depth. Neural confidence is not measured
accuracy. The earlier camera/fade uncertainty intervals still apply.
The visibility improvement is checked against calibrated synthetic data. The two
new recordings are separate takes, so this update does not establish accuracy on
a new real synchronized multi-camera recording.

Character intersections remain material: even the improved 14-pose probe retains
an 86 mm sampled maximum in one difficult pose. Angular bounds, fixed feet,
intentional contacts and temporal continuity can prevent full clearance. Final
intermediate-pose reports retain those contacts instead of labelling the animation
collision-free. No every-moment accuracy or bug-free guarantee is established.

## Local evidence

Output directory:
`D:\2026\nuranova-mocap\Mcapn_results\blendcap_methods_2026_11_28`

- `benchmark.json`, `1/`, `2/`: full source-preserving solves and H5/Blender/BVH files.
- `regressions_final_boundary.xml`: complete test suite, including the boundary regression.
- `boundary_failure_before_fix.json`, `character_quaternion_inspection.json`:
  failed export measurements and the six reproduced quaternion sign changes.
- `blender_mcp_audit.json`, `blender_mcp_reply_isolated.json`, `mcp_cleanup_reply.json`:
  source key poses, halfway poses and isolated MCP lifecycle.
- `character_baseline.json`, `character_polished.json`: the paired 14-pose probe.
- `character_range_summary.json`, `character_*_range_results.json`: final contact
  checks for the selected correction intervals.
- `Methods_1_Character_Review.blend`, `Methods_2_Character_Review.blend`: complete
  retargeted review copies with the selected ranges corrected. The corresponding
  `Methods_1_Character.glb` and `Methods_2_Character.fbx` passed full-length fresh-import checks.
- `frozen_smoke.json`, `frozen_cpu_delivery.json`, `windows_bundle_manifest.json`:
  tests of the actual executable, including hands and spectral audio.

Keep the full `dist/NuranovaMocapStudio` folder when using its executable. Install
`installers/NuranovaMocapBridge.zip` for the Blender correction changes.

Design references: [BlendCap hand fitting](https://github.com/Arcomade/BlendCap/blob/76d6d9dad375c48d35bd5efa004cadf90f1dcdb4/pipeline/hand_solver.py),
[visibility-aware fusion](https://github.com/Arcomade/BlendCap/blob/76d6d9dad375c48d35bd5efa004cadf90f1dcdb4/pipeline/two_camera_merge.py),
and [audio synchronization](https://github.com/Arcomade/BlendCap/blob/76d6d9dad375c48d35bd5efa004cadf90f1dcdb4/blendcap/multicam_sound.py).
