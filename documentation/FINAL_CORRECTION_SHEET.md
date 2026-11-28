# Nuranova Mocap Final Correction Sheet

**Date:** 28 November 2026

**Implementation branch:** `kawshal/bugFix`

**Baseline commit:** `db1a70e`

**Status:** approved requirements; implementation and acceptance evidence are tracked below.

The next release must improve hand, finger and foot motion, make each camera's
tracking reviewable and editable, support physical markers on the performer,
correct intersections on the chosen Blender character, and export the corrected
character animation to FBX and GLB.

Completion requires the stated acceptance checks and recorded evidence. See the
[implementation report](CORRECTION_IMPLEMENTATION_2026_11_28.md) and
[live CSV tracker](FINAL_CORRECTION_TRACKER.csv) for delivered features and remaining
acceptance work. The original baseline below records the starting point.

The [CSV tracker](FINAL_CORRECTION_TRACKER.csv) contains the same requirement IDs
for filtering and progress tracking. Detailed algorithms, source references and
module changes remain in [PROJECT_PLAN.md](../PROJECT_PLAN.md).

## Existing verified baseline

The November 27 repair added robust waveform alignment, original video timestamp
handling, camera validation, observation caching, improved limb fitting and
rotation smoothing. The Blender bridge has continuous quaternion signs, linear
dense animation curves and muted NLA archive tracks. Basic FBX/BVH and native
motion exports existed at that baseline; the new implementation adds GLB and the
unified character export described in the implementation report.

The recorded baseline validation is **89 tests passed and one optional footage
benchmark skipped**. Both source and target rigs were evaluated over 2,309 frames
and 2,308 halfway frames. This establishes the earlier checks, not completion of
the new hand/contact/mesh requirements. See [TEST_REPORT.md](../TEST_REPORT.md).

Physical-marker detection and rigid-cluster code also existed. Their integration
into multi-video review is part of the new implementation.

## Final workflow

1. Import or record the take and choose its capture mode.
2. Align the videos and review timing on one shared timeline.
3. Validate camera calibration and repair any suspect camera.
4. Detect markers/body points, assign names and correct observations.
5. Review each camera's hands and fingers with centered crop previews.
6. Solve body, wrist, finger and foot motion using reliable observations and constraints.
7. Retarget to the selected character; scan and correct contacts/intersections in Blender.
8. Bake, export, reopen and verify the corrected result.

The interface uses five linked tabs: **Videos & Sync**, **Cameras & Calibration**,
**Markers & Body**, **Hands & Fingers**, and **Character & Export**. Camera
selection, the global timeline and correction history stay consistent across tabs.

## Capture modes and synchronization

| ID | Required correction | Complete when |
|---|---|---|
| MODE01 | Add a **Use physical markers** checkbox to the shared take/capture settings. Retain AI-only capture; checked behavior enables marker-assisted capture. Provide marker-only operation when enough observations are available. | The selected mode persists, controls the actual pipeline and reports insufficient marker evidence rather than claiming successful marker capture. |
| MODE02 | Associate the performer with a marker kit, unique IDs, anatomical names and calibrated marker-to-body offsets. | Known IDs map consistently across cameras; unknown IDs can be assigned manually; changing the kit invalidates affected results. |
| SYNC01 | Align videos before multi-camera solving. Provide automatic waveform matching, evidence/status, manual timing adjustment and an alignment preview. | Users can check and correct shared movement timing before reconstruction, with an explicit reference camera and accepted timing revision. |
| SYNC02 | Preserve original presentation timestamps and handle mixed frame rates, variable frame rates, gaps and supported clock drift. | All previews show the same global instant with each source's actual frame/time; missing frames are labeled and resampling preserves motion duration. |

Physical markers belong on the person, not only on the Blender model. Forearm,
hand and shoe markers can improve segment orientation. Accurate fingers require
additional visible finger measurements, sufficiently detailed hand views, or a
calibrated glove. The software must not imply that a wrist tag measures every
finger joint or adds observations to older recordings without physical markers.

## Camera review and repair

| ID | Required correction | Complete when |
|---|---|---|
| CAM01 | Provide **One camera**, **All cameras**, and **Selected cameras** preview modes, with solo/enlarge controls and shared playback. | Any chosen subset can be reviewed together without losing synchronization or changing camera identity when tiles are reordered. |
| CAM02 | Separate cameras selected for preview from cameras included in solving. Allow solving exclusions by camera, time range and body/hand channel. | Viewing one camera leaves other solve inputs unchanged; an unusable hand view can be excluded while retaining that camera's useful body observations. |
| CAM03 | Show each camera's video, tracked/edited point overlay and the fused motion projected into that view. Clearly label any single-camera 3D model estimate. | Users can compare the image with the corresponding motion at the same instant and identify which camera contributes a bad observation. |
| CAM04 | Provide separate **Fix timing**, **Edit points / Re-track range**, **Recalibrate this camera**, and **Exclude range** actions. | Timing, observation and geometry problems use distinct repair paths; selected-camera calibration can be compared/restored while trusted cameras and the world reference remain anchored. |

Recalibration needs suitable known geometry or trustworthy shared references.
Camera movement during a take needs a calibration interval or range exclusion.
Blurred, corrupted or absent image detail needs better observations; lens/pose
calibration alone cannot restore it. Re-solving may change the fused motion while
the other cameras' raw observations remain intact.

## Marker and body point editing

| ID | Required correction | Complete when |
|---|---|---|
| MARK01 | Add **Detect points** for the selected cameras and range, using physical markers or anatomical point detection according to capture mode. | Detection displays the correct source/type and exposes unassigned or ambiguous IDs instead of silently assigning them to body parts. |
| MARK02 | Allow users to select, add, drag, relabel, delete or mark points missing/occluded; support bounded retracking and restoring the raw detection. | Edits use native source coordinates, affect the intended camera/range and remain protected from automatic cleaning unless explicitly replaced. |
| MARK03 | Show each point's anatomical name on hover. Clicking pins details such as camera, marker ID, source time and provenance. | Names are correct and readable; hover/selection works through zoom, letterboxing, hand crops and optional display mirroring. |
| MARK04 | Match physical/anatomical identities across views and validate tag corners when orientation is required. | Corresponding views share the correct identity while retaining their own pixel coordinates; a manually selected center alone is not treated as complete marker orientation. |

## Hands and fingers review

| ID | Required correction | Complete when |
|---|---|---|
| HAND01 | Add a dedicated **Hands & Fingers** tab with left/right/both selection and a hand-only 3D viewer containing wrist, palm and fingers, without arm bones. | Each selected camera's hand can be inspected independently or compared in the camera grid. |
| HAND02 | Show native-resolution video hand crops, point overlays and corresponding hand motion. Retain per-camera observations alongside the fused result. | The user can compare the same hand/time across cameras; crops retain fingertips and clearly distinguish camera-local estimates from fused 3D motion. |
| HAND03 | Add **Center hand**, stable zoom and reset-to-full-frame preview controls. | Centering keeps the visible hand in view and changes no source coordinates, world motion, calibration or exported animation. |
| HAND04 | Allow manual hand/finger point correction, ROI adjustment, retry detection and camera/hand/range exclusion. | A correction to one camera's hand is saved, applied to re-solving and leaves other cameras' raw detections unchanged. |
| HAND05 | Show missing, occluded, low-confidence and stale hand data explicitly. | The viewer does not swap anatomical sides, snap to the image origin or display a held old pose as a fresh measurement. |

## Motion quality and solving

| ID | Required correction | Complete when |
|---|---|---|
| SOLVE01 | Preserve per-landmark validity, uncertainty and observed/interpolated/manual/estimated provenance. Remove zero-filled missing fingertips and artificial confidence promotion. | Missing points never become geometric measurements; the quality report separates observations from interpolation and estimates. |
| SOLVE02 | Run detailed hand observations in every useful camera and choose/fuse evidence by quality. Benchmark dedicated RTMPose Hand5 and MediaPipe against the same labeled clips. | Selection is based on measured visible-hand error and runtime, with no fixed first-camera preference or unsupported claim that a new model is better. |
| SOLVE03 | Recover palm orientation and jointly solve forearm roll and wrist movement in calibrated camera/world/rig rest frames. | Controlled palms-up/down, wrist-bend and near-straight-elbow tests avoid false flips and preserve reliable limb endpoints. |
| SOLVE04 | Calibrate hand proportions and fit fixed-length fingers, joint limits, thumb opposition and temporal behavior. | Open hand, fist, pinch, splay and thumb gestures remain plausible, retain intended movement and avoid noisy independent finger rotations. |
| SOLVE05 | Couple pelvis and both legs with heel/sole/toe contact states. Correct foot sliding, ground penetration and unwanted foot rotation. | Flat stance, heel lift, toe pivot, steps and jumps pass separate checks; genuine motion is not suppressed to improve a slide metric. |
| SOLVE06 | Fit the actor with anatomical, temporal, contact and collision constraints while preserving reliable image evidence and intended self-contact. | Annotated source-pose failures improve without stretching limbs, freezing movement or destroying claps, crossed limbs and hands resting on the body. |

Retain the working Python/PySide6, PyAV, OpenCV, NumPy/SciPy, ONNX Runtime CUDA
and Blender stack. Use pretrained models as observation providers and geometry
for fitting/contact correction. Training a new network is an optional later step
after a remaining detection failure and suitable labeled data justify it.

## Blender character correction

| ID | Required correction | Complete when |
|---|---|---|
| BLEND01 | Build a separate rig/geometry/contact profile for each character, including units, rest frames, controls, body shape, clothing and hand/foot anchors. | Supported Nuranova, Mixamo, Rigify, Unreal, VRM and custom mappings work across different proportions; unusual rigs receive concrete setup requirements. |
| BLEND02 | Detect unwanted 3D penetration involving hands, fingers, arms, legs, feet, torso, head, floor and configured props using suitable proxies and evaluated meshes. | Scanning catches relevant intersections and containment while excluding adjacent anatomy, intended garment layers and harmless overlap in the video image. |
| BLEND03 | Correct target-character intersections using coupled joint/IK adjustments and contact goals after retargeting. | The target preserves reachable motion intent, limb lengths, intended contacts and stance anchors; infeasible intervals are identified rather than hidden. |
| BLEND04 | Provide **Analyze**, **Fix range**, **Compare**, contact pins, local re-solving and **Bake corrected action** inside Blender. | Users can review and finish uncertain ranges reversibly; original actions, rigs and artist controls remain recoverable. |
| BLEND05 | Diagnose skin weights, bone rolls, finger mapping and twist distribution separately from pose errors; test optional twist bones and Preserve Volume per character. | Skinning changes improve the actual deformed character and are checked in the selected export path rather than treated as universal fixes. |
| BLEND06 | Validate the final deformed mesh after all hand/face animation, filtering, constraints and baking. | Integer frames, halfway frames and adaptive checks near fast contacts detect remaining problems; later operations do not silently overwrite corrections. |

Every character needs its own profile; a generic skeleton does not describe every
body volume. The goal is correction on mapped, properly bound characters with a
clear setup/manual finish for unusual anatomy, broken rigs or unreachable poses.
Ordinary Blender Collision physics settings alone are not this animation solver.

## Animated 3D exports

| ID | Required correction | Complete when |
|---|---|---|
| EXPORT01 | Make **FBX** and **GLB** primary animated-character exports; provide glTF with complete dependencies. | The chosen character mesh, rig and corrected body/finger motion are exported; supported facial channels and materials are verified per format. |
| EXPORT02 | Offer **Motion only**, **Character + animation**, and **Static pose**, with clip/range, FPS, root motion/in-place, units/axes and texture options. | Options control the actual output, selected clips are explicit and no source mannequin is substituted for another selected character. |
| EXPORT03 | Retain BVH skeleton motion, editable BLEND copies and native motion/JSON archives with clear content labels. | Users understand which files contain meshes, motion, textures or full capture metadata; optional finger channels match the actual exported skeleton. |
| EXPORT04 | Use shared background export jobs with explicit object/action scope, cancellation, structured paths and completed-file verification. | Exports handle quoted/Unicode paths and exclude unrelated rigs, cameras, helper markers and archived noisy actions; failed jobs cannot report success using an old file. |
| EXPORT05 | Export and reimport into a fresh scene, then verify the exact delivered variant. | Animation, frame rate/duration, hierarchy, scale, fingers, supported face/material channels and character contacts meet the declared format checks. |

Alembic geometry caches, USD/USDZ and static OBJ are optional later outputs.
Additional formats require their own tested capabilities. They must not be
advertised as equivalent ways to preserve every rig, material and animation feature.

## Project data and responsiveness

| ID | Required correction | Complete when |
|---|---|---|
| DATA01 | Keep original observations immutable and store edits against stable camera/source IDs, native frame index and original PTS. | Reordering cameras or retiming a take does not attach a correction to another image or camera. |
| DATA02 | Persist marker mode/mapping, camera inclusion, calibration revisions, hand review, contact pins and edit history in project/take packages. | Save/reopen and portable-package tests reproduce the corrected state; older projects remain readable. |
| DATA03 | Track result dependencies and re-solve affected ranges after timing, calibration, annotation, model or character-profile changes. | Stale results are visible; raw inference is reused where valid, and manual edits survive derived-data regeneration. |
| DATA04 | Keep decoding, detection, solving and export responsive with bounded caches, progress and cancellation. | Multi-camera preview remains usable; cancellation preserves the previous complete result and does not leave partial actions or project state active. |

Undo/Redo applies to manual corrections and local re-solving. Folder/ZIP imports
and in-app recordings use the same review pipeline. Standalone take review must
also retain edits through an explicit save/package path when no project is open.

## Testing and release criteria

| ID | Required correction | Complete when |
|---|---|---|
| QA01 | Maintain an indexed failure set using the supplied three videos plus controlled gesture and marker-assisted captures. | Cases identify camera/time, visibility, intended contact and the stage where the error starts; successful intervals remain in the regression set. |
| QA02 | Extend real Blender/MCP tests across character proportions, control rigs, clothing, transforms and export round trips. | Supplied and additional rig/mesh fixtures pass the defined checks; incomplete mappings and impossible contacts produce explicit reports. |
| QA03 | Measure pose fidelity, contact/penetration, finger behavior and temporal quality with valid references. | Reports distinguish accuracy from confidence/coverage and do not declare success solely from low angular jumps or smooth motion. |
| QA04 | Validate dependencies, new model assets, performance, project safety and release artifacts. | Model revisions/hashes/terms are recorded; relevant regression, UI, hardware and packaging checks have evidence or are explicitly outside the tested release scope. |

Use these initial acceptance targets, then calibrate them against the labeled
cases and character dimensions:

- Synthetic observable endpoints within **1 mm**, known wrist/finger angles within
  **1 degree**, and fixed bone lengths.
- On verified flat stances, mean anchor drift below **5 mm** and P95 below **10 mm**;
  score pivots and swing separately.
- Body collision checks initially use the detailed plan's **5 mm** proxy tolerance
  and **10 mm** persistent-error review threshold for adult-sized characters;
  scale to the character and use smaller, separately defined finger tolerances.
- On controlled still holds, reduce high-frequency finger noise by at least
  **40%**, retaining at least **90%** of known gesture range.
- Source-actor corrections preserve reliable reprojection within the agreed
  tolerance. Target characters are evaluated using mapped motion intent and
  contact fidelity, not source-camera reprojection of a different body shape.
- Export checks run on the delivered format, including optional compression,
  using the final corrected motion and actual target mesh.

These are implementation targets, not newly measured results. Release requires
resolution of the reproducible blocking failures in the agreed test set, a usable
manual correction path, and visible uncertainty where observations are absent.

## Delivery sequence

| Stage | Delivery | Requirements |
|---|---|---|
| 1 | Shared source identity, review data model, accepted synchronization and camera preview | MODE01, SYNC01–02, CAM01–03, DATA01 |
| 2 | Marker/body editing, hover labels, per-camera repair and persistent history | MODE02, CAM04, MARK01–04, DATA02–03 |
| 3 | All-camera hand observations, hand-only review and quality/provenance corrections | HAND01–05, SOLVE01–02 |
| 4 | Calibrated wrist/finger fitting and coupled actor/foot contact solving | SOLVE03–06 |
| 5 | Character profiling, Blender collision correction, review controls and baking | BLEND01–06 |
| 6 | Unified FBX/GLB/glTF and existing-format export verification | EXPORT01–05 |
| 7 | Full regression, real capture/Blender validation, performance and release packaging | DATA04, QA01–04 |

Build the failure set and read-only character scanner early; run relevant tests
at each stage. These stages state dependencies, not permission checkpoints.

## Optional later work

- Benchmark HaMeR or other hand-mesh priors where the initial hand detectors still
  fail, subject to model/dependency terms and measured GPU/runtime limits.
- Fine-tune a suitable pretrained detector only after collecting corrected labels
  and holding out actors/sessions. Do not train on the current reconstructed
  fingers as if they were ground truth.
- Add calibrated glove/finger-marker device adapters for recordings that need
  measurements beyond what the cameras can observe.
- Add tested Alembic/USD/static-model exports and a dynamics engine only where a
  concrete delivery or motion case needs them.

For this hardware, retain the existing RTX 3060 Ti **8 GB** deployment budget.
Better lighting, measured camera calibration, useful viewing angles and sufficient
hand detail remain part of capture quality. Arbitrary hidden or blurred finger
poses need more observations or manual correction to establish their actual motion.

## Supporting records

- [Detailed project plan and research references](../PROJECT_PLAN.md)
- [Existing validation and limitations](../TEST_REPORT.md)
- [Motion format specification](MOTION_DATA_FORMAT.md)
- [Correction tracker](FINAL_CORRECTION_TRACKER.csv)
