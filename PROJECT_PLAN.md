# Project Plan — Nuranova Mocap Studio v1.0.0

**Developer:** Nuranova Solutions (Kawshal, Shanal, Shehan — https://nuranova.lk/)  
**Target Release:** v1.0.0  

The consolidated implementation requirements are in the
[Final Correction Sheet](documentation/FINAL_CORRECTION_SHEET.md), with an
[editable CSV tracker](documentation/FINAL_CORRECTION_TRACKER.csv). The sections
below retain the research, engineering details and validation plans behind them.

## Hand, Contact, and Character Quality Plan — 2026-11-28

**Status:** researched implementation plan; the changes below are not yet implemented.
**Baseline:** `db1a70e` on `kawshal/bugFix`, 89 tests passed and one optional
footage benchmark skipped. The repaired take is smoother, but those tests and the
previous angular-jump counts do not establish correct wrists, fingers, contacts,
or mesh deformation.

### Decision

Build a geometry-constrained hand/body solver and character-aware retargeting on
top of the existing stack. Improve observations from every camera, recover wrist
orientation, fit a calibrated articulated hand over time, and enforce contact and
collision constraints on both the source actor and the target character. Evaluate
pretrained hand models as interchangeable observation providers. Training a new
network is a later option, contingent on a measured remaining detection failure
and a suitable annotated dataset.

Default engineering stack: Python/PySide6, PyAV timestamps, OpenCV camera geometry,
ONNX Runtime CUDA/RTMLib observations, NumPy/SciPy constrained fitting, and Blender
actions plus geometry audits through MCP. Start with analytic capsule distances
and bounded sparse optimization. A dynamics simulator is optional; adding one
does not itself infer which contacts or hand poses actually occurred.

### Evidence from this app and capture

These are observations from the code, cached inference results, and a read-only
inspection of the user's live Blender scene, rather than claims about which new
model performs best. Phase 0 must establish which mechanism causes each visible
failure; structural findings alone do not prove the cause at a particular frame.

| Finding | Evidence | Consequence for the implementation |
|---|---|---|
| Detailed hand crops use only the first camera | `multi_video_studio/pipeline.py` enables wrist crops only when `c_idx == 0`; later reads fallback hands from `primary_uid` | Process native-resolution hand crops in every usable view and choose evidence by quality, rather than camera order. |
| Fine hand detail is small in the recordings | Median estimated MCP5-to-MCP17 palm span: cam_01 left/right 31.8/30.4 px at 4K; cam_02 19.7/29.3 px and cam_03 17.4/21.3 px at 1080p | Measure hand visibility and image detail. Enlarging a crop cannot recover finger detail missing from the source. |
| Primary-camera hand detections are intermittent | Cached detailed crop observations exist on 26.7% of left-hand frames and 32.4% of right-hand frames for cam_01; detailed crops were not run for cam_02/03 | Separate direct observations from triangulation, interpolation, and pose-prior output in the report. These percentages are not final hand coverage. |
| Wrist orientation is deliberately discarded | `HandCaptureEngine._smooth_hand_quaternions` sets palm channel 0 to identity; body solver also sets hand local rotation to identity; Blender finger import skips the wrist by default | Recover a world-space palm frame, solve forearm roll and wrist articulation together, and convert to the actual rig's local rest frame. Simply enabling the old wrist channel is unsafe. |
| Missing fingers can become false geometric measurements | `triangulate_hands_from_wholebody` permits a partially valid hand and replaces missing 3D coordinates with zero before fitting; a scalar hand confidence is then broadcast to all landmarks | Preserve per-landmark validity and uncertainty; exclude missing points from fitting. Never use the world origin as a missing fingertip. |
| Confidence and source changes can hide uncertainty | Short hand gaps are promoted to at least 0.55 confidence; detailed-crop priority and the ten-frame fallback rule govern backend selection | Preserve measured confidence, record provenance and missing duration, and blend compatible evidence in a common coordinate frame. |
| The visible character can deform badly even with reasonable joints | Live MCP inspection: both 55-bone character rigs have no twist bones; their 16,710-vertex bound meshes have Preserve Volume disabled | Independently test skin weights, bone rolls, twist distribution, and skinning. This is a possible contributor, not a confirmed explanation for each reported twist. |
| Target retargeting copies rotations and adjusts root height | `blender_addon/.../retargeting.py` has no character-volume or self-contact solve | Add a target contact/collision pass; a clear source skeleton can still intersect a differently proportioned character. |
| Foot cleanup still relies on contact heuristics and sequential fixes | `zero_slide_physics.py` detects contact, adjusts root/legs, and flattens feet; it also uses Euler clamping in that path | Fit root, both legs, contact state, and ground constraints together; preserve toe pivots and airborne motion. |

Palm-span measurements use raw RTMW landmarks whose two MCP scores exceed 0.4;
they are estimated projected distances, not manually measured hand dimensions or
a universal minimum model input size. Crop percentages describe availability in
the complete cached source tracks, not the shared output timeline. Cached crop
results may include held/interpolated samples; the old cache does not retain enough
provenance to call all of them independent detector observations.
Machine-readable evidence is saved locally
at `../Mcapn_results/quality_repair/research_evidence_2026_11_28.json`; the live rig
inspection is `research_rig_structure.json` in the same directory.

### Model and library choices

| Candidate | Use in this plan | Decision and practical limits |
|---|---|---|
| Existing RTMW whole-body ONNX | Body initialization and inexpensive hand-region proposals | Retain the working CUDA path; benchmark improvements with the same cached body observations. |
| Dedicated RTMPose-m Hand5, 256 x 256 | Native-resolution hand-crop 2D landmarks, then calibrated multi-view fitting | First new model to benchmark. Official MMPose/RTMLib releases provide this model, including an ONNX download. It fits the present runtime; actual accuracy and throughput on this footage remain unmeasured. [Hand demo](https://github.com/open-mmlab/mmpose/blob/main/demo/docs/en/2d_hand_demo.md), [RTMLib model table](https://github.com/Tau-J/rtmlib/blob/main/README.md). |
| MediaPipe Hand Landmarker Tasks | Independent lightweight hand baseline and continuity aid | Benchmark VIDEO mode per camera/track. Its hand-centered world landmarks must be aligned to the calibrated capture world, not treated as absolute world measurements. [Official guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker?tab=Python). |
| HaMeR | Optional hand-mesh/pose prior for difficult visible hands | Benchmark in an isolated worker only after the geometry baseline. Repository code is MIT, but it requires a separate MANO model; model assets, dependencies, and redistribution rights need separate review. Do not assume the code license licenses the entire model bundle. [Repository](https://github.com/geopavlakos/hamer), [code license](https://github.com/geopavlakos/hamer/blob/main/LICENSE.md), [MANO terms](https://mano.is.tue.mpg.de/license.html). |
| WiLoR / WiLoR-mini | Optional research comparison of hand localization/reconstruction | The official WiLoR license is CC BY-NC-ND 4.0 and MANO has separate terms. A convenient wrapper does not remove upstream restrictions. Do not make it a bundled default without suitable rights. [Official implementation](https://github.com/rolpotamias/WiLoR), [license](https://github.com/rolpotamias/WiLoR/blob/main/license.txt). |
| HaWoR | Research reference for temporal hand recovery | Its egocentric setting differs from these fixed external cameras; its model license is also restrictive. Not the first implementation choice. [Official implementation](https://github.com/ThunderVVV/HaWoR). |
| SciPy + analytic geometry | Articulated hand/body fitting, robust residuals, sparse windows, joint bounds | Selected implementation foundation. `least_squares` supports bounds, robust loss and Jacobian sparsity; collision penalties remain soft unless an explicit feasibility step is added. [API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html). |
| Anipose/aniposelib | Independent calibration/triangulation comparison | Useful validation reference for temporal/spatial reconstruction; it does not supply target-character collision handling or finger retargeting. Avoid replacing the complete desktop pipeline. [Documentation](https://anipose.readthedocs.io/en/latest/). |
| UnderPressure | Reference for contact estimation and IK cleanup | Do not import its implementation into the shipping app by default: its evaluation license restricts use and modification. Implement the needed contact model independently and assess learned contact detection only if heuristics remain inadequate. [Project](https://github.com/InterDigitalInc/UnderPressure), [license](https://github.com/InterDigitalInc/UnderPressure/blob/main/LICENCE.txt). |
| MuJoCo | Optional later validation of more complex dynamics/contact | Apache-2.0 engine; not required for the initial kinematic solution. Adopt only if measured cases need dynamics and justify extra integration. [Official repository](https://github.com/google-deepmind/mujoco). |

All new models require pinned revisions, hashes, explicit model/data terms, and a
Windows/RTX 3060 Ti benchmark. New candidate models have not been benchmarked on this
take during this research. Do not promise that a research demonstration will fit
8 GB VRAM, export to ONNX unchanged, or outperform current inference.

### Ordered implementation and completion gates

| Phase | Concrete implementation | Required evidence before proceeding |
|---|---|---|
| **0. Reproduce and label** | Freeze the current exports and model/cache revisions. Locate high-twist, close-hand, crossed-arm, crossed-leg, foot-pivot and fast-finger ranges automatically; visually label the actual failures. Compare raw detections, reconstructed joints, source rig, and final skinned character. | An indexed failure list with camera timestamps, Blender frames, visibility, intended contacts, and whether the error starts in detection, solving, retargeting, or skinning. Include successful intervals to prevent regressions. |
| **1. Better observations and honest uncertainty** | Run hand crops from all useful cameras. Preserve crop-to-image transforms and original PTS. Benchmark Hand5 and MediaPipe on identical crops. Add per-landmark validity, uncertainty, handedness/identity, backend, camera frame, and provenance. Fuse observations using reprojection and visibility; remove zero-filled missing points and confidence promotion. | Visible-joint annotation error and hand-identity swaps improve on held-out intervals. Missing landmarks never become measured points. Old captures remain readable. Low-detail/occluded intervals remain visibly marked as estimates. |
| **2. Calibrated hand and wrist articulation** | Calibrate palm width, finger lengths, rest splay, thumb axes and left/right rig frames from neutral/open/fist/pinch examples. Fit a fixed-length articulated hand directly to valid multi-view 2D data. Use MCP curl/splay, PIP/DIP flexion and a suitable thumb opposition model with soft tendon coupling. Jointly solve palm orientation, forearm axial roll, and wrist bend/deviation over time. | Synthetic mirrored-hand and full forearm-turn tests pass without flips, endpoint displacement, or finger-length drift. Verify independent wrist flexion versus forearm pronation and real fist/open/pinch gestures. |
| **3. Contact and collision fitting** | Add actor/character segment capsules and torso/palm/foot proxies with editable dimensions. Couple root, limbs and contacts over temporal windows. Penalize actual 3D penetration; preserve intentional hand-to-body, hand-to-hand and crossed-limb contacts. Use an explicit contact state model for heel strike, flat stance, toe pivot and swing. | Fewer annotated penetrations and incorrect contact transitions, without pushing apart legitimate contacts or treating 2D image overlap as a collision. Contact accuracy is assessed on independently labeled intervals. |
| **4. Target-character correction** | Export a rest/geometry/contact profile from Blender. Convert through source/target rest frames and solve target hand/foot anchors using target proportions. Audit bone rolls, bind weights, thumb layout, wrist axes and finger mapping. Test Preserve Volume and optional distributed twist bones on a copy of the character. | Correct source poses remain correct on the supplied character and a second rig with different proportions/rolls. Check evaluated mesh deformation and export playback, not just quaternion channels. |
| **5. Temporal refinement and review controls** | Use one coordinated temporal fitting stage; tune SO(3) and joint-angle regularization by observation uncertainty. Apply constraints after smoothing or jointly during fitting. Add timeline issue markers, palm/contact pins, short gesture key poses, and reversible local re-solving. | Suppress noise on annotated still holds while retaining real gesture range and timing. No silent long-gap fabrication; low-evidence ranges are reviewable and corrections survive save/reopen/export. |
| **6. Benchmark and release** | Compare baseline, each new stage, and full result on the complete three-video take plus separate gesture/occlusion captures. Run desktop/engine tests and Blender MCP audits for all frames and subframes; package the tested bridge/model manifests. | Publish per-case measurements and review videos, reject regressions or infeasible cases, and retain the old pipeline as a selectable rollback until the new path passes. |

Phase 0 and Phase 1 are the first coding milestone. Do not spend time training or
installing the heaviest model before the observation contract and evaluation set
can distinguish a better model from smoother hallucinated motion.

### Solver contract

Fit fixed actor proportions and a continuous pose sequence against all valid
views. A practical objective combines uncertainty-weighted robust reprojection,
3D target agreement where well conditioned, anatomical joint bounds, SO(3) angular
velocity/acceleration regularization, contact-anchor error, ground clearance and
nonadjacent-segment penetration. Determine contact and hand-identity hypotheses
from several frames rather than switching from a single confidence threshold.

Use warm-started sparse windows, initially 0.5–1.0 seconds with overlap. Measure
window size, convergence, runtime and cancellation responsiveness on this machine.
Reconcile overlap before a final constraint validation. Bounds belong to the
articulated variables; do not clamp arbitrary Euler decompositions after IK.
Retain genuine fast turns and finger motion instead of making a fixed angular
speed cap the quality criterion.

For palm pose, align a calibrated palm template to reliable MCP/wrist evidence,
resolve normal/handedness ambiguities over time, and express the result relative
to the forearm and rig rest transform. Separate camera-centered monocular priors
from calibrated world evidence. Add an explicit world-palm orientation field if
needed; do not change the interpretation of the old hand channel 0 silently.
Legacy takes keep their existing interpretation and remain importable.

Collision proxies are an approximation. Use soft penalties for initialization,
then audit feasibility and locally repair or flag remaining violations. A bounded
least-squares solve alone does not guarantee nonpenetration. Adjacent segments
and intentional surface contacts need explicit handling. Avoid a global repulsion
rule that destroys a clap, crossed arms, clasped hands, a squat, or a toe pivot.
Contact-aware retargeting research supports treating geometry and self-contact
as part of retargeting, rather than assuming a skeleton rotation transfer suffices.
[Primary paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Villegas_Contact-Aware_Retargeting_of_Skinned_Motion_ICCV_2021_paper.pdf).

Keep optimization in the desktop worker. Blender's embedded Python does not need
SciPy or an AI framework: export a compact rig/contact profile, return native baked
actions, and use Blender/MCP for evaluated-pose/mesh validation. MCP is a testing
interface, not a mandatory production dependency. Any final filter or retarget
operation must be followed by contact/collision checks because it can invalidate
previously solved constraints.

### Expected code changes

| Area | Planned work |
|---|---|
| `multi_video_studio/pipeline.py`, `wholebody_cuda.py`, `tracking_cache.py` | All-view hand crops; backend adapters; cache keys include the observation schema, model hash, crop transform and settings; retain original PTS. |
| `engine/hands/hand_tracker.py` and new hand observation/fitting modules | Per-landmark validity/provenance, multi-view hand identity, calibrated hand template and constrained temporal fitting. |
| `engine/skeleton/solver.py` and a shared twist/contact module | Separate swing, axial roll and wrist articulation; couple body and hand evidence without moving reliable endpoints to satisfy a pole filter. |
| New `engine/geometry` collision utilities | Segment distances, capsules, exclusion/contact pairs and calibrated target proxies, with signed-distance and degenerate-segment tests. |
| `multi_video_studio/zero_slide_physics.py`, cleanup and rotation filtering | Contact states and coupled root/leg constraints; no post-IK Euler clamp/filter that silently invalidates anchors. |
| `engine/data_format/motion_io.py` | Additive uncertainty, provenance, palm orientation and contact metadata with explicit compatibility tests. |
| Blender `retargeting.py`, `facial_hands.py`, `auto_rig.py` | Rig/contact profile exchange, rest-frame mapping, wrist/finger/twist handling and final character correction. |
| `multi_video_studio/ui_page.py`, Motion Editor | Show observed/interpolated/estimated intervals, reason for uncertainty and reversible pose/contact pins. Avoid a single coverage percentage as the accuracy result. |
| Quality audits and integration tests | Measured detection error, twist accuracy where observable, finger constraints, collision/contact metrics, target mesh checks and real export round trips. |

### Acceptance criteria

The following are **provisional engineering gates**, to be calibrated against the
Phase 0 labels and character geometry. They are not claimed current results or
guaranteed accuracy for invisible joints.

- Preserve prior synchronization, file safety, project handling and Blender import
  regressions. Validate old and new motion files and both original/custom rigs.
- On noiseless synthetic fixtures, fit observable limb endpoints within 1 mm,
  preserve bone lengths, and recover known wrist/finger angles within 1 degree.
  Include mirrored hands, near-straight elbows, palm rotations through 180 degrees,
  independent wrist flexion, missing fingertips and conflicting views.
- On visible manually annotated hand frames, improve palm-scale-normalized joint
  error/PCK relative to the baseline; account for labeling uncertainty. Report
  results separately for small, blurred, occluded and clear hands. Do not use a
  model's own confidence or 3D predictions as ground truth.
- Reduce high-frequency finger motion by at least 40% on verified still holds,
  while preserving at least 90% of the known gesture range on controlled tests.
  Inspect pinch, fist closure, splay and thumb opposition, not just open hands.
- On clearly labeled flat stances, target mean anchor drift below 5 mm and P95
  below 10 mm. Score toe pivots and swing separately. Do not improve the number by
  declaring difficult stance frames airborne or planting a genuinely lifted foot.
- Use a provisional 5 mm penetration tolerance for coarse body/foot proxies;
  flag persistent errors above 10 mm for review. Establish separate smaller
  finger tolerances from the actual character dimensions. Validate intended
  surface contacts rather than requiring every pair of body parts to separate.
- Source-actor contact/collision improvement must not increase reliable body
  reprojection P95 by more than 5% without an explained outlier correction and
  visual confirmation. Different target-character proportions cannot be judged
  against source-camera reprojection directly; evaluate their contact, motion
  intent and mapped-pose fidelity separately.
- Audit every keyframe, halfway frames, and adaptive subframes near fast contact
  changes. Use swept or adaptively refined proxy checks to detect crossings between
  samples; finite quaternions and low frame-step angles are insufficient.
- Record peak VRAM, runtime, dropped source observations and cancellation latency.
  Use the existing 8 GB GPU as the deployment budget; heavier models run in a
  separate process and are accepted only after measured improvement.
- Full annotated failures must resolve or remain explicitly marked for correction.
  A release cannot be declared physically correct merely because all unit tests pass.

### Capture, non-AI options, and training

The supplied videos remain the regression set. For future captures, make measured
lens/studio calibration, fixed camera positions, useful side/front coverage and
adequate exposure the normal high-quality workflow. Add a short palm-open,
palms-up/down, fist and pinch calibration sequence. Keep hands large and sharp in
at least two complementary views; try closer hand cameras or a separate hand pass.
Select exposure/fps based on blur and available light; resampling a 15.8 fps
recording does not create the missing temporal observations.

Provide three modes rather than one quality promise:

| Mode | What it offers | What new input it needs |
|---|---|---|
| Existing-video repair | Multi-view observations, articulated fitting, contact-aware correction and uncertain-range review | Existing videos; cannot recover uniquely unobserved finger poses. |
| Calibrated marker-assisted capture | Direct wrist/foot orientation from rigid marker patterns, fused with body/hand evidence | Visible calibrated patterns or noncollinear rigid clusters, with enough image detail and views. |
| Sensor/marker hand capture | Hand articulation from a calibrated glove or sufficiently visible finger markers, optionally corrected against cameras | Additional recorded measurements and device/marker calibration. Wrist tags alone do not measure every finger joint. |

The marker path can use existing OpenCV/AprilTag infrastructure. Marker pose
estimation uses identified corner geometry and camera calibration; it is a viable
non-neural source of orientation when the pattern is visible. It still has
occlusion and resolution limits. [OpenCV marker guide](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html),
[AprilTag implementation](https://github.com/AprilRobotics/apriltag).

Do not train from the three recordings as if their current reconstructed fingers
were correct labels. First benchmark pretrained hand models and the constrained
solver. If a repeatable detection gap remains, collect diverse, sharp hand images
with corrected labels, camera/actor/session-separated evaluation, and appropriate
dataset/model rights. Fine-tune the smallest suitable pretrained detector on the
measured failure cases; reserve training from scratch for evidence that this is
necessary. Promotion requires improvement on held-out real captures and controlled
gestures without losing runtime, anatomical correctness, or contact quality.

The deliverable is improved, inspectable motion with clear uncertainty and a
short correction path. Arbitrary blurred or hidden fingers cannot be guaranteed
perfect from ordinary video; better measurements are the route to higher fidelity
when geometry and pretrained models have exhausted the available evidence.

## Marker and Multi-Camera Review Workflow — 2026-11-28

**Status:** product and implementation requirements from the user's proposed
workflow. These controls and review stages are planned, not implemented by this
documentation update. Integrate them into Import & Solve Videos and the shared
capture/review pipeline rather than creating a separate disconnected application.

### Workflow and tabs

Use one shared take, camera list, global timeline and correction history across:

**Videos & Sync → Cameras & Calibration → Markers & Body → Hands & Fingers →
Character & Export**.

Import and time alignment come before cross-camera marker/body/hand solving. The
user can inspect a single source video at any time, but multi-view reconstruction
requires an accepted timing solution and valid camera geometry. Existing validated
timing/calibration may be reused with their validation status visible.

| Tab | Main controls | What the user can review or correct |
|---|---|---|
| **Videos & Sync** | Add videos; reference camera; **Align videos**; waveform preview; manual offset/drift adjustment; common playback/scrubbing | Same physical instant across cameras; unmatched frames, uncertain audio matches and VFR sampling differences. |
| **Cameras & Calibration** | Camera picker; per-camera status; **Recalibrate this camera**; compare/restore calibration; camera inclusion for solving | Lens/pose errors, calibration residuals, calibration video/known-point correspondences and affected intervals. |
| **Markers & Body** | **Use physical markers** checkbox; **Detect points**; assign anatomical names; add/drag/relabel/delete a point; missing/occluded flag; Undo/Redo | Each camera's video, detected points, manually corrected points, and projected fused skeleton. Hover reveals the point name; selection pins its details. |
| **Hands & Fingers** | Left/right/both; camera picker; video/overlay/3D comparison; **Center hand**; optional stable zoom; point edits; retry detection/exclude hand range | Native-resolution hand crops and hand-only motion for each selected camera, alongside the fused result. No arm bones in the 3D hand viewer. |
| **Character & Export** | Choose character/profile; retarget; collision/contact scan; correction and contact pins; bake/export | The final skinned character, unresolved issues and source-versus-corrected comparisons from the Blender plan below. |

Keep the common camera/timeline controls visible across tabs. Show a persistent
take summary for timing, calibration, marker mapping and solve revision, so the
user can tell when an edit has made a result stale. Detailed model/solver options
belong in Advanced rather than the main review path.

### Physical marker selection and point editing

- Add **Use physical markers** to take/capture settings and persist it with the
  project. Normal checked behavior combines reliable marker observations with the
  existing body/hand tracking. Expose a marker-only option separately for users
  with enough measurements. Unchecked behavior retains the markerless workflow.
- With markers enabled, **Detect points** finds physical tags/blobs in the selected
  cameras and frame range, displays IDs/corners/centers, and maps known IDs through
  the selected marker kit. Unknown IDs remain explicitly unassigned until the user
  chooses their anatomical name. With markers disabled, the action detects the
  configured anatomical body points. Keep the active detection type clear.
- Allow manual selection/placement of a point on the person and assignment of a
  name such as **Left wrist**, **Right heel**, or **Left index fingertip**. Drag to
  correct its location; mark it occluded/missing, relabel a mistaken identity,
  restore the raw detection, or propagate/retrack a bounded interval. A manual
  point is recorded as a manual annotation, rather than a newly observed tag.
- Hover shows the anatomical name and camera. Selected details can additionally
  show tag/point ID, native image coordinates, source time, confidence/provenance
  and reprojection error. Selected and edited points need distinguishable styling
  beyond color alone; labels should not obscure the entire image.
- Marker matching across cameras means matching physical/anatomical identity at
  the same global time. Corresponding points normally have different pixel
  coordinates in different views. Do not copy one camera's pixel position to all
  other views or force their 2D overlays into the same image position.
- A square tag's corners and known dimensions carry pose information. Provide
  corner correction/validation when orientation is needed. A single manually
  placed center point alone cannot establish complete hand/forearm orientation.
  [OpenCV marker geometry](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html).
- Manual corrections are protected inputs to re-solving; automatic cleaning must
  not silently erase them. Cross-view consistency checks can identify a conflicting
  manual edit for review instead of assuming that every manual point is correct.
- If no physical markers are visible, report that state and offer the relevant
  markerless/manual path. Adding the checkbox does not introduce physical marker
  measurements into older videos that never recorded them.

### Camera selection and synchronized preview

Provide **One camera**, **All cameras**, and **Selected cameras** preview modes,
with a checkbox list, Select all/Clear selection, and solo/enlarge actions. Use a
grid for comparisons and a full-size tile for detailed point edits. Each tile
shows its stable camera name, source frame/time, availability and tracking status.

Keep **Preview cameras** separate from **Use for solving**. Looking at camera 2
alone must not silently remove cameras 1 and 3 from reconstruction. Likewise,
changing tile order must not change the reference camera, anatomical labels or
calibration mapping. Solving exclusions may be per camera, interval and channel:
a camera can remain useful for body tracking while its blurred hand observations
are excluded.

One global scrub/play control drives all selected tiles using each video's real
PTS and the accepted clock mapping. Display the nearest available source frame
and its time difference; do not imply that 15.8 fps footage contains every 30 fps
output instant. Missing/out-of-range frames show an explicit placeholder rather
than an unrelated stale image. Retiming preserves the original file timestamps.
Audio waveform correlation remains the automatic starting point, with evidence
and manual refinement available. [SciPy correlation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html).

Per-camera review should show **Video**, **Video + detected/edited points**, and
**Fused pose projected into this camera**. Camera-local 3D model estimates may be
shown as an additional clearly labeled view. A monocular estimate is not an
independent measured 3D reconstruction or ground truth; multi-view consistency is
judged against the combined solution and the actual images.

Batch actions such as detection, re-tracking or exclusion apply only to explicitly
selected cameras/ranges and report their scope. Shared marker-name assignments
can be reused by physical ID, but camera transforms and pixel edits remain
camera-specific. Preserve annotations when preview selection changes.

### Repairing one camera

The camera review must distinguish four actions:

| Problem seen by the user | Correct action |
|---|---|
| The movement occurs at a different instant | **Fix timing**: review waveform/event alignment and update that camera's clock mapping. |
| A wrist, fingertip or marker is detected in the wrong place | **Edit points / Re-track range**: correct observations and re-solve the affected interval. |
| Many otherwise good points disagree systematically with the 3D projection | **Recalibrate camera**: refine the selected camera's lens/pose from suitable calibration evidence. |
| The recording is blurred, corrupted, missing or fully occluded | **Exclude camera/range/channel** or use a better recording; calibration cannot recover absent image detail. |

For recalibration, keep trusted cameras and the world reference anchored while
solving and validating the selected camera. Prefer an existing/new calibration
board recording with known geometry and a consistent world reference. When two
or more other trusted views are available, their well-conditioned shared 3D
correspondences can also support refining the suspect view. One remaining ordinary
camera alone does not establish arbitrary unknown 3D geometry.

Show old/new reprojection results on fitting and held-out correspondences before
applying the revision, and retain a restore option. Changing one camera's
calibration can legitimately change the fused motion across all views; recompute
the dependent result without rewriting the other cameras' raw observations or
silently moving their calibration frames. If the camera moved mid-take, use
calibration intervals or exclude the affected interval; a single fixed pose is
not valid for the entire recording in that case.

### Dedicated Hands & Fingers tab

Every selected camera has independent left/right hand tracks. The tab should
support a single enlarged hand view or a multi-camera comparison grid. Each camera
tile offers these synchronized panes:

1. **Video hand crop:** crop around the observed wrist/palm and fingers from the
   native-resolution image, with enough margin to retain fingertips. Base the crop
   on the hand landmarks; avoid showing the whole arm as the hand preview.
2. **Tracked points:** the same crop with named 21-point hand landmarks, anatomical
   side, confidence/provenance and manual corrections. Preserve the source-image
   crop transform so edits refer to the correct original pixels.
3. **Hand-only motion:** wrist/palm/finger skeleton or mesh with no arm chain,
   viewed from the corresponding camera orientation. Distinguish the camera-local
   model estimate from the fused multi-view 3D hand, and allow a linked comparison.

**Center hand** is a display stabilization option. When reliable wrist/palm
evidence exists, keep that hand centered in the video crop and hand-only viewport.
Offer a stable zoom option and a reset-to-full-frame view; show where the crop
came from in a small source overview. Preserve visible finger motion, wrist
rotation and anatomical identity. Centering must not alter source observations,
world translation, calibration or exported motion.

When confidence is lost or the hand leaves the frame, show **Hand not visible /
Tracking uncertain**. A held preview must be explicitly labeled with its original
timestamp and stale status. Do not snap the crop to the image origin, swap hands,
or present a repeated old pose as a fresh observation. Allow the user to choose
or correct the hand ROI and rerun detection on that interval.

Point names include wrist and each thumb/finger joint/tip. The user can inspect,
correct, disable or retry one camera's hand data without changing the other
cameras' detections. Apply corrections before calibrated hand fitting, then
revalidate the combined result and the final character contacts.

### Data, performance and implementation

Store immutable source observations plus a versioned edit history. Each annotation
contains a stable source-asset/camera ID, native frame index and raw PTS, anatomical
point or marker/corner ID, native image coordinates, validity, provenance and the
edited interval. Link derived results to timing, calibration, marker-kit, hand
model and annotation revisions. Camera import order is not an identity contract.

Changing synchronization remaps existing annotations from their original PTS;
it does not move them to a different source image. Calibration edits invalidate
derived 3D results, not raw inference. Point edits invalidate the affected fit
range. Persist camera inclusion, marker mode/mapping, manual edits and calibration
revisions through save/reopen and portable project export. Older projects keep
their existing behavior with defaults for the new optional fields.

Use cached/proxy images for the comparison grid and on-demand full-resolution
frames for precise edits and hand crops. Keep proxy/native/crop/view transforms
explicit and test zoom, letterboxing, rotation and optional display mirroring.
Decode and detect asynchronously with cancellation and bounded caches. Run hand
inference in each useful view, as required by the hand-quality plan, and preserve
camera-specific raw hand landmarks/priors for review instead of only saving the
final fused rotations.

Implementation areas are `multi_video_studio/ui_page.py` with extracted shared
camera-grid, timeline, overlay editor and hand-review widgets; `package_ingest.py`
and project state for stable assets/settings; `pyav_sync.py` for timing revisions;
marker detectors/kit mapping and the pipeline for manual/automatic observations;
camera calibration refinement; and versioned observation/edit data in motion I/O.
Qt's graphics-item hover/tooltips and coordinate mapping support the overlay
interaction. [Qt graphics-item API](https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QGraphicsItem.html).

### Ordered milestones and tests

| Order | Deliverable | Acceptance check |
|---|---|---|
| **U0** | Shared source identities, timeline and camera preview grid | One/all/selected views stay synchronized using real PTS; preview selection never changes solve inclusion or camera calibration identity. |
| **U1** | Use-markers setting, automatic detection, named overlay editing and history | Hover names and point selection work; native coordinates remain correct through zoom/crop/mirroring; manual edits survive save/reopen and cannot be silently overwritten by detection. |
| **U2** | Timing/point/calibration repair actions per camera | A synthetic bad camera can be corrected with trusted views anchored; a point edit is not treated as a lens change. Revision tracking refreshes dependent 3D outputs and permits rollback. |
| **U3** | All-camera hand crops and dedicated hand-only comparison | Left/right identity and camera labels remain correct; centering/zoom leave motion data unchanged; missing hands show uncertainty instead of fabricated fresh motion. |
| **U4** | Incremental re-solving, final contact/collision validation and project round trip | Camera/range/channel exclusions are honored; insufficient remaining evidence is reported; edited results persist through desktop review and Blender export. |

Test markerless, hybrid and marker-only takes, absent tags, unknown/mistaken IDs,
VFR and dropped frames, manual offsets, moved cameras, partial hand occlusion,
edge-of-frame crops, repeated import/reordering and a missing source file. Include
synthetic ground-truth point/camera fixtures and the existing three-video take for
actual playback and crop review. UI tests should exercise hover, drag, Undo/Redo,
camera selection, centering and save/reopen, not just the existence of buttons.
No new result should be labeled reviewed merely because automatic detection ran.

## Blender Character Collision and Contact Plan — 2026-11-28

**Status:** implementation plan, extending Phase 4 above. No new collision solver
has been implemented by this planning update.

### Required behavior

Every imported target character receives its own geometry and rig profile. After
retargeting, the system must detect and attempt to correct unwanted 3D penetration
of hands, fingers, arms, legs, feet, torso and head, including floor and configured
prop collisions. It must preserve intended surface contact, pose intent, planted
feet and temporal continuity. The user must be able to inspect and correct a
remaining problem directly in Blender, then bake a verified result.

The generic target is any mappable, correctly bound humanoid, including stylized
proportions, large hands, short arms, broad torsos and clothing/armor. Automatic
mapping is supplemented by editable semantic roles and collision surfaces.
Nonhumanoid anatomy and unusual control rigs require an appropriate adapter or
manual profile. A missing hand chain or an unreachable gesture is an explicit
capability/feasibility issue, rather than an automatic success. This is how the
design supports new characters without hard-coding the supplied model's shape.

Physical markers improve the actor observations. They do not establish clearance
on a target character with different proportions, so this finishing pass remains
necessary with AI, marker, and hybrid capture alike.

### Blender workflow

The proposed panel flow is **Set Up Character → Scan Motion → Fix Range → Compare
and Pin Contacts → Bake Corrected Action → Validate Export**. Reports show the
body-part pair, frame range, penetration/clearance, applied correction and any
unresolved constraint. Useful controls are character-relative clearance,
correction strength, protected contacts, locked channels, frame range and an
editable floor/prop selection. Keep technical solver settings in an advanced panel.

Work on a duplicate action and, when necessary, an explicit working/export rig
copy. Preserve the original rig, skin weights, artist controls and source action.
The corrected result is editable, supports Undo/Redo, and saves with its profile
and issue report. A completed operation reports the actual result: clear within
the stated checks, corrected with remaining issues, or failed/infeasible.

### Character profile and evaluated geometry

| Profile component | What must be stored or checked |
|---|---|
| Rig identity and mapping | Stable asset/profile ID, semantic bone chains, parent structure, rest matrices, local axes, control/deform bone distinction, IK/FK switches, constraints, drivers and active NLA behavior. Start with Nuranova, Mixamo, Rigify, Unreal and VRM adapters; allow explicit custom mappings. |
| Coordinate contract | Object/armature transforms, scene units, world-to-rig conversion and left/right orientation. Perform distances in consistent world meters. Test mirrored and nonuniform object scale; preserve artist transforms instead of silently applying them. |
| Body geometry | Fitted torso/head/palm volumes and limb/finger capsules using mesh shape and skin-weight regions. Give each character editable dimensions and adjacency exclusions. Bone positions alone do not describe a broad torso or thick forearm. |
| Contact anchors | Palm surface/normal, thumb/finger pads, heel, ball, toe and sole offsets; semantic body regions such as hip, chest and face; configured prop handles. Store local/surface-relative anchors so contacts follow the moving body or prop. |
| Collision layers | Selected body skin and outer clothing/armor envelopes, floor and optional props. Exclude nested eyes, teeth and intended garment/body layering from ordinary body collision tests. Hair or cloth simulation needs its own configured treatment. |
| Validity and revision | Bone/rest-pose, mesh, binding and profile fingerprints; modifier/shape-key dependencies; unresolved mappings and proxy confidence. Rebuild or revalidate when the character changes; do not reuse another character's radii or contact offsets. |

Blender extracts the **evaluated deformed mesh**, including the relevant armature,
constraints, drivers, modifiers and shape keys. Transform participating geometry
into a common coordinate system before queries. Update geometry at the tested
frame; a rest-mesh BVH cannot certify a deformed animation. Keep topology/region
mapping stable where possible and mark topology-changing cases for an appropriate
proxy or evaluation path.

Use fast analytic proxies for interactive scanning, then mesh/surface checks near
contacts and in final validation. Blender's BVH supports nearest-point, ray and
overlap queries. An overlap list alone does not prove that one closed volume is
not entirely contained in another. Pair surface tests with containment/clearance
checks on suitable closed colliders. Open clothing meshes need explicit closed
collision proxies or a conservative surface policy; do not infer a reliable
inside/outside sign from an arbitrary nearest face normal.
[Blender BVH API](https://docs.blender.org/api/current/mathutils.bvhtree.html?highlight=ray).

### Correction algorithm

1. **Prepare and scan.** Resolve mapping and control ownership, verify evaluated
   playback, and create the target profile. Broad-phase bounds and semantic pair
   exclusions identify candidate collisions; capsule and mesh checks measure
   clearance. Distinguish 3D intersections from harmless overlap in a camera view.
   Treat adjacent skin regions at an elbow or finger knuckle as connected anatomy.
2. **Establish contact intent.** Use source contact metadata and temporal evidence,
   with Blender contact pins for uncertain intervals. Preserve claps, clasped hands,
   hands on hips/head, crossed limbs and object grips. A contact goal belongs to a
   surface region and orientation, not merely the nearest point anywhere on the
   whole character. Track that region smoothly as the body moves.
3. **Solve a local temporal window.** Adjust arm/shoulder chains, wrist and finger
   articulation, both legs and pelvis as needed. Minimize deviation from the
   retargeted motion while satisfying valid joint bounds, nonpenetration, contact
   anchors and floor clearance. Preserve hand orientation and elbow/knee bend
   continuity. Avoid a wrist-only push that stretches an arm or swaps elbow poles.
4. **Respect the rig.** Drive control bones and IK targets through a known adapter.
   Do not write deform-bone rotations while active controls override them. For rigs
   without a usable control adapter, provide a documented bake/export duplicate
   with evaluated transforms and verified deformation. Native Blender IK can make
   an arm or leg chain follow a corrected goal; it is part of the correction,
   rather than a complete whole-character collision solver.
   [Blender IK documentation](https://docs.blender.org/manual/en/5.0/animation/constraints/tracking/ik_solver.html).
5. **Preserve the feet and motion.** Keep heel/sole/toe anchors appropriate to flat
   stance, heel lift, toe pivot or swing. Couple both legs with reachable pelvis
   adjustments, and retain real steps and jumps. Transferring root height alone
   cannot ensure target-foot contact when limb proportions differ.
6. **Re-evaluate actual skin.** Apply the candidate pose and inspect evaluated
   geometry. Refine proxy errors with actual contact-surface feedback. Test
   skinning, bone rolls, weights and optional twist distribution separately from
   animation pose errors. Preserve Volume is a per-character experiment; it is
   not a universal correction for bad weights or wrong motion.
7. **Validate after smoothing and baking.** Blend temporal-window boundaries, apply
   the final action, and repeat contact/mesh checks. If constraints are mutually
   infeasible, retain a flagged interval with its reason and corrective controls.
   Do not hide the failure by excessive stretching, teleporting, freezing the hand
   or deleting the intended contact.

The preferred implementation shares NumPy/SciPy geometry and pose fitting with
the desktop engine. The Blender add-on handles profiles, actual rig evaluation,
native controls/IK, review, mesh feedback and baking. Numerical fitting runs in a
local desktop worker or configured standalone worker process, so the embedded
Blender Python does not require the application's SciPy/AI installation. Blender
API access stays on its main thread and is chunked to preserve progress updates
and cancellation. MCP exercises the real Blender workflow in tests; normal use
goes through the add-on and local bridge/worker.

This is deliberate animation correction. Blender's ordinary Collision physics
settings participate in particle/cloth/soft-body simulations; switching them on
does not implement armature pose correction. Shrinkwrap can help construct a
surface goal, but its constraint operates on an owner origin rather than the
whole hand's volume. Neither is sufficient as the complete solution.
[Collision documentation](https://docs.blender.org/manual/en/5.0/physics/collision.html),
[Shrinkwrap documentation](https://docs.blender.org/manual/id/5.0/animation/constraints/relationship/shrinkwrap.html).

### Bake, persistence, and export

The operation order is body/finger/face retargeting → evaluated geometry → contact
and collision correction → final bake → final geometry audit → export and
reimport validation. Later finger application, shape-key evaluation, filtering or
active constraints must not overwrite the corrected pose without revalidation.

Bake evaluated motion into a new dense action with continuous quaternion signs.
Avoid double evaluation of the active action and its NLA archive. Use visual
keying where rig constraints or drivers contribute to the result; clear or disable
helpers only on the appropriate working/export copy after verifying the bake.
Persist contact pins, profile revision, input action identity, tested frame range,
clearance policy and unresolved issues alongside the result.
[Blender Bake Action documentation](https://docs.blender.org/manual/en/dev/editors/nla/editing/strip.html).

Check both the saved `.blend` and a real exported/reimported asset. Export behavior
for skinning, constraints, corrective shape keys and target formats can differ.
Do not assume Blender's Preserve Volume appearance will be reproduced by every
FBX consumer. A skeleton-only BVH round trip can verify motion, but cannot certify
the appearance or collisions of a skinned mesh.

### Work packages and release gates

| Order | Deliverable | Gate |
|---|---|---|
| **B0** | Character profiler and read-only collision scanner, extending the existing MCP audit | Detect known hand-inside-torso, finger, crossed-leg, foot/floor and prop fixtures; distinguish intentional contact and containment from surface overlap. No pose edits in this milestone. |
| **B1** | Saved contact/proxy profile with control-rig adapters | Correct world units, transformed rigs, left/right mapping, outer collision layers and per-character anchors. Invalid binding or mapping produces a concrete setup report. |
| **B2** | Coupled hand/arm, body and foot correction in a local worker | Reduce annotated penetration while preserving contact intent, endpoint reach, bone length, motion timing and stance anchors. Detect infeasible targets. |
| **B3** | Blender review panel, contact pins, local re-solve and new-action bake | Reversible edits survive save/reopen; controls and evaluated skin agree with the corrected result; cancellation does not leave a partial replacement action active. |
| **B4** | Full mesh/subframe audit and export round trips | All annotated failures resolve or remain explicitly reported. Saved and exported results agree within the declared tolerances. Performance and geometry limits are documented. |

Suggested implementation modules: Blender `character_profile.py`, `collision_scan.py`,
`collision_fix.py` and rig adapters; shared `engine/geometry` collision primitives
and constrained pose fitting; `retargeting.py` integration after all pose channels;
and mesh/contact extensions to `scripts/blender_motion_audit.py` and real Blender
integration tests. Expose an **Analyze** operation before enabling automated fixing.

The validation set must include the supplied character, procedural rigs with short
arms/broad torsos/large hands, different scales and rest poses, a control rig with
active constraints, clothing/armor with nested geometry, dynamic shape keys and
an explicit missing-finger or unreachable-contact case. Include successful claps,
hands on hips, clasped fingers, crossed arms/legs, crouches, kicks and toe pivots.

Use profile-scaled tolerances, initially the main plan's 5 mm body-proxy tolerance
and 10 mm persistent-error review threshold for adult-sized characters; define
smaller finger tolerances from geometry. Report maximum/P95 penetration, affected
body pairs and duration, stance drift, lost intended contacts, pose deviation,
joint-limit violations and runtime. Score target motion intent and contact
fidelity separately from source-actor reprojection error.

Evaluate all integer frames and halfway frames, then use swept/adaptive subframe
checks around fast movements to catch between-frame crossings. A clean capsule
scan alone is not a mesh-level certificate. Store exactly which meshes, collision
pairs, intervals and tolerances were checked, and label incomplete checks. The
goal is reliable correction across profiled characters with a clear manual finish
for ambiguous or impossible cases, rather than an unsupported guarantee for every
possible rig, mesh and gesture.

## 3D Asset and Animation Export Plan — 2026-11-28

**Status:** requirements and implementation plan. Existing paths export native
motion archives, Blender JSON, BVH and FBX. GLB/glTF and the unified character
export flow below are not implemented by this documentation update.

### Formats and export scope

Make **FBX** and **GLB** the primary animated-character choices. Distinguish three
outputs in the interface: **Motion only**, **Character + animation**, and
**Current pose / static model**. A skeleton-only file must not be presented as an
export of the selected skinned character. Keep the native project/motion archive
available for complete editing history and capture metadata.

| Format | Planned role | Content and limitations |
|---|---|---|
| **FBX `.fbx`** | Primary skeletal animation interchange | Selected character mesh(es), armature and baked body/hand motion, or a motion-only armature when explicitly chosen. Verify materials and facial/shape-key behavior against the installed exporter and destination; do not assume every Blender feature transfers. |
| **GLB `.glb`** | Primary portable animated-character package | Mesh, skin, supported materials/textures and selected skeletal/morph animation in a binary glTF asset. Make asset dependencies portable and validate them in a fresh import. |
| **glTF `.gltf`** | Alternate glTF delivery | The same supported asset types with the required `.bin`/texture dependencies packaged together; show that this is a multi-file export. |
| **BVH `.bvh`** | Motion-only skeleton interchange | Hierarchy and motion channels; no character mesh/materials. Include finger motion only where the exported skeleton/mapping actually contains it and the destination supports it. |
| **Blender `.blend`** | Editable character/motion handoff | Selected character, corrected actions and appropriate rig/profile data in a saved copy. Preserve the original project separately. |
| **Alembic `.abc`** | Later cinematic geometry-cache option | Baked evaluated mesh motion for cases that need final deformation rather than an editable skeletal rig. Treat cache size and material delivery separately. |
| **USD `.usd/.usdc/.usdz`** | Later pipeline-interchange option | Enable tested capabilities of the installed Blender build; validate skeletal/geometry animation, textures and dependencies for the chosen destination. |
| **OBJ `.obj`** | Optional static geometry handoff | A chosen pose/model snapshot. It is not a portable skeletal mocap animation file. Keep other static formats out of the main motion-export path until needed. |
| **Native `.nmocap.h5` / Blender JSON** | Existing editable capture archive/bridge data | Retain observations, timing/calibration references, provenance and supported motion channels. These serve the capture workflow rather than replacing standard character assets. |

Blender documents glTF animation for object/bone transforms and shape-key values;
arbitrary driver/physics/material behavior needs appropriate baking or a different
delivery path. Action/NLA selection also determines which clips are exported.
[glTF export documentation](https://docs.blender.org/manual/en/latest/addons/scene_gltf2.html).
Blender exposes additional interchange formats, but support must be implemented
and tested per format instead of promising that any extension preserves every
asset feature. [Blender interchange formats](https://docs.blender.org/manual/en/latest/files/import_export/index.html).

### User controls and shared export job

Use a format dropdown and one **Export** action in Character & Export and the
Blender panel. Provide target character/rig selection, motion-only versus character
scope, corrected/raw action selection, clip/frame range, output frame rate,
root-motion versus in-place mode, hands/fingers, supported face channels, texture
packaging and destination/unit-axis presets. Hide unsupported options or explain
their effect for the selected format. Selected-clip export is the default; exporting
an action library is an explicit option.

Both the desktop and add-on should submit the same validated export-job contract.
Use a structured job/config file and argument list for background Blender tasks;
do not construct executable Python by interpolating arbitrary output filenames.
Support paths with spaces, quotes and Unicode. Run long exports asynchronously
with progress, cancellation and useful error reports. Check available Blender
operators/build capabilities and destination directories before starting.

Without a target asset, motion-only export remains available. Character export
requires the actual selected character and mapping, loaded through the live bridge
or an explicit asset/project reference in a background job. Do not silently export
the source mannequin when a user requested another character. Normal operation
does not depend on MCP; use MCP for real Blender validation.

### Existing gaps to correct

- `AnimationExportsPage._exp_fbx` currently creates a source armature in an empty
  Blender scene. Label and preserve that as the motion-only path; add the actual
  target asset, mapping and corrected action for character exports.
- The add-on FBX operator currently uses `use_selection=False`. Build an explicit
  export collection/selection containing the chosen rig, bound meshes and required
  dependencies. Exclude old source rigs, other characters, calibration cameras,
  debug markers and collision helpers unless explicitly requested.
- There is no GLB/glTF export operator or desktop export choice in the current
  implementation. Add it through the shared exporter rather than maintaining an
  unrelated second workflow.
- Archived/stashed actions may be considered by exporters even when they are not
  part of the desired result. Explicitly control action/NLA scope so old takes,
  original noisy actions and temporary correction actions do not appear as
  unexpected clips or double motion.

### Correctness and validation

Export the final body, fingers and supported face channels after retargeting,
contact/collision correction and baking. Recheck the exact exported variant,
including any optional compression, rather than validating one asset and then
shipping a differently processed one. Preserve real duration and sample times
at rates such as 29.97/59.94 fps; verify units, axis conversion, root motion and
in-place behavior. Avoid format-specific extra bones, duplicated armatures or
unwanted scaling.

Material conversion and deformation are explicit compatibility concerns. Validate
textures and alpha/PBR material appearance for GLB; report procedural materials
that need baking. Check target-specific shape-key and skinning behavior for FBX
and GLB. Blender constraints, drivers and Preserve Volume cannot be assumed to
remain live or visually identical in every downstream application. Use an evaluated
geometry cache when final surface deformation is the selected deliverable.

Each primary-format test must export and reimport into a fresh scene, then check:

- Correct rig/mesh scope, clip names, duration, sample times, materials and asset
  dependencies; no unrelated scene objects or archived takes.
- Body/root and finger animation at representative poses, plus all-frame numerical
  checks appropriate to the format. Test non-default fps and transformed rigs.
- Supported face/morph animation, joint hierarchy, skin weights and rest/bind pose.
  Report unsupported features rather than silently claiming complete export.
- Character clearance, intended contacts and planted feet after the round trip,
  using the same tolerances as the target-character correction plan.
- Missing Blender, unavailable export operators, cancellation, output failures,
  Unicode/quoted paths, stale existing files and incomplete multi-file packaging.

Write through a temporary destination and publish the completed asset only after
success/validation. An old file already present at the requested path is not
evidence that a new export succeeded. Restore the user's Blender selection,
active action, frame and helper visibility after the export job.

### Delivery order

1. **E0 — Shared scope and existing-format repair:** unify export settings and fix
   FBX selection/action scope, background-job handling and motion/character labels.
2. **E1 — Animated GLB/glTF:** add both desktop and Blender entry points, portable
   textures and animation selection, with real mesh/skin/animation round trips.
3. **E2 — Final-character validation:** connect exported assets to collision/contact
   audits; verify fingers, supported face channels and destination presets.
4. **E3 — Optional interchange:** add tested Alembic/USD/static-model choices based
   on their actual use case and build capabilities. Keep FBX and GLB complete first.

Implementation touches `desktop/ui/pages/pages_export_system.py`, the shared
export-job/worker layer, `blender_addon/nuranova_mocap_bridge` export operators,
target/action selection, and `tests/integration/test_blender_bridge_real.py`.
Do not advertise a new format as complete until the corresponding animated
character round-trip test passes.

## Motion Quality Repair — 2026-11-27

1. Preserve the open Blender scene and original videos. Connect to the user's live
   Blender MCP bridge and audit source/target rigs, actions, NLA blending, rest axes,
   scale, frame rate, and every evaluated frame. Record a numerical baseline.
2. Align audio with multiple independent windows, robustly fit clock offset/drift,
   check all camera pairs for closure, and preserve original video PTS. Export
   aligned visual evidence. Reject uncertain synchronization instead of silently
   guessing simultaneous starts.
3. Verify calibration against synchronized, confident correspondences. Compare
   held-out reprojection/cheirality and bone proportions; inspect occlusion and
   opposite-camera degeneracy. Refine geometry only when the evidence improves.
4. Cache raw model observations for repeatable experiments. Audit tracking, 3D,
   skeletal IK, cleanup, and retargeting separately. Fix discontinuities at their
   source and use rotation-aware temporal regularization without freezing motion.
5. Compare complete captures with the baseline using per-bone angular jumps,
   isolated spikes, displacement, contact drift, reprojection, and confidence.
   Unobserved joints remain identified as estimates. Check rotations through a
   full turn, squats, arm crossings, missing hands, and low-confidence intervals.
6. Import the improved take into a copy of the supplied character scene through
   Blender MCP. Audit all evaluated keyframes and NLA behavior; save source and
   retargeted actions, review images/video, and machine-readable audit results.
7. Run the full test suite and document measured improvements and residual limits.
   "Perfect" cannot be certified without reference motion capture; completion
   requires reproducible improvements, no known unaddressed deterministic failure,
   and explicit reporting of remaining capture uncertainty.

Stack to evaluate: Python 3.11/PySide6, PyAV for original timestamps/audio, SciPy
signal correlation and robust least squares, OpenCV calibration/triangulation,
RTMW via ONNX Runtime CUDA, rotation-aware smoothing, and Blender's native
quaternion actions via MCP. Keep a component when it passes measured checks;
replace a component only when an alternative improves this take's result.

Implemented and measured: robust multi-window audio clocks, all-camera calibration
validation and candidate selection, cached observations, endpoint-preserving limb
IK, SO(3) rotation smoothing, per-bone audits, and Blender bridge 1.2.2. The installed
extension was updated through the live MCP session. The repaired character has no
adjacent-frame jumps over 30 degrees across 55 bones, and all 2,309 keyframes plus
2,308 midpoints were evaluated on both source and target rigs. Detailed evidence
and remaining capture limitations are recorded in [TEST_REPORT.md](TEST_REPORT.md).

The selected stack remains Python 3.11/PySide6 + PyAV + OpenCV/SciPy + RTMW/ONNX
Runtime CUDA + native Blender quaternion actions. This machine's model inference
already works on CUDA; the measured failures were synchronization, geometry checks,
IK, and interpolation. A framework replacement would not address those causes.

Primary references for implementation decisions:
[SciPy correlation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html),
[robust least squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html),
[rotation splines](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.RotationSpline.html),
[OpenCV camera geometry](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html),
[Pose2Sim workflow](https://github.com/perfanalytics/pose2sim),
[RTMLib WholeBody implementation](https://github.com/Tau-J/rtmlib/blob/main/rtmlib/tools/solution/wholebody.py).

## Release Phase Roadmap

| Phase | Scope | Exit Criteria |
|---|---|---|
| **Phase 0** | Workspace audit, environment verification, licensing review, clean repository structure & pinned `.venv` | `REPOSITORY_AUDIT.md`, `ARCHITECTURE.md`, `LICENSE_REGISTER.md`, working Python 3.11 `.venv` |
| **Phase 1** | Mathematical foundation: camera projection/undistortion, ChArUco intrinsics/extrinsics, bundle adjustment, weighted DLT + RANSAC triangulation, quaternion/SE(3) algebra, synthetic validation | Unit & synthetic tests passing with sub-pixel reprojection and <1 mm synthetic triangulation error |
| **Phase 2** | Android Camera Application (`Nuranova Mocap Camera`): Kotlin, Jetpack Compose, MVVM, Camera2 API, MediaCodec/MediaMuxer H.264, Camera2 timestamps, NSD, pairing, resumable upload, Room/DataStore, all required UI screens | Gradle build compiles clean APK (`NuranovaMocapCamera.apk`) + JUnit unit tests passing |
| **Phase 3** | Desktop Camera Network & Synchronization: FastAPI/WebSocket server, mDNS Zeroconf advertiser, HMAC/token pairing, 2–12 camera manager, repeated NTP-style clock sync + drift regression, audio cross-correlation sync, flash/LED visual sync, resumable verified transfers | Integration tests passing for 2–12 simulated camera clients, clock offset estimation, resumable SHA-256 transfer |
| **Phase 4** | Physical Marker Tracking: valid ChArUco/ArUco/AprilTag PDF & PNG generator, subpixel blob/contour detector, ArUco/AprilTag detector, temporal Hungarian/Kalman tracker, cross-view epipolar association, asymmetric rigid marker clusters (6-DoF Kabsch pose) | Unit & synthetic multi-camera marker + cluster reconstruction tests passing (<2 mm synthetic error, <0.5° rotation error) |
| **Phase 5** | AI Tracking: ONNX Runtime CUDA/CPU safe hardware detector, RTMLib (`Wholebody` RTMW-133, `HALPE-26`, `Hand-21`), MediaPipe (`Pose`, `Face Landmarker` 52 blendshapes, `Hand Landmarker`), model registry with checksums & licenses, benchmark harness | Real AI inference validated on actual multi-camera video frames (Pose2Sim BSD-3-Clause demo dataset) with measured FPS & VRAM diagnostics |
| **Phase 6** | Full-Body Articulated Skeletal Solver: hierarchical 24+ bone humanoid rig (with clavicles, spine chain, forearm twist, fingers), actor calibration (T/A/neutral pose), constant bone lengths, constrained IK + numerical optimization, quaternion continuity, confidence/ambiguity flags | Synthetic ground-truth skeletal solve validated (<10 mm joint pos, <3° major joint angle, zero bone-length drift) |
| **Phase 7** | Hybrid Cinema Mode & Processing Profiles: calibrated marker-to-segment/joint offset model (never naive averaging), joint + rigid-cluster fusion optimization, Fast Preview / Standard / Cinema profiles, HDF5 `.nmocap.h5` versioned data format with full provenance | Hybrid mode tested against synthetic occluded/noisy AI + marker data showing measurable error reduction over AI-only |
| **Phase 8** | Motion Cleanup: raw vs corrected non-destructive storage, outlier removal, swap correction, confidence-aware gap interpolation, adaptive Butterworth & One Euro filters, kinematic foot-contact detection + 2-bone IK foot-locking, ground penetration clamping, root stabilization | Synthetic noisy/footskating tests verify planted-foot displacement <5 mm and bone length invariant |
| **Phase 9** | Blender Integration (`Nuranova Mocap Bridge`): Blender 4.5 LTS / 5.0+ extension, live HTTP polling via `bpy.app.timers` + offline import, source armature builder, Rigify & custom rig auto/manual bone mapper, rest-pose delta & scale compensation, facial shape-key mapping, finger animation, baking to editable keyframes/Actions/NLA, FBX/BVH export | Tested headlessly inside real installed **Blender 5.0.1** CLI creating armatures, retargeting to a secondary stylized rig, baking keyframes, and exporting FBX & BVH |
| **Phase 10** | Dedicated Facial & Hand Capture: MediaPipe Face Landmarker (478 landmarks + 52 blendshapes + head pose + neutral calibration + smoothing), MediaPipe/RTMW Hand Landmarker (left/right handedness, finger bend/splay angles, dedicated camera sync) | Unit & real-image inference tests passing, Blender shape-key & finger bone mapping verified |
| **Phase 11** | Complete Desktop UI (`Nuranova Mocap Studio`): PySide6 modern Light & Dark themes, all 20 required pages with real wired functionality (no dead buttons), Capture Studio 3D viewport (PyVista + Qt fallback) & live camera grid, Calibration Wizard, Marker Kit Editor, Motion Editor (timeline, curves, undo/redo, raw vs corrected), Project Manager, Processing Queue, Diagnostics, Settings | Launches cleanly offscreen & GUI; pytest-qt UI smoke tests verify all 20 pages, theme switching, project CRUD, viewport rendering, and editor actions |
| **Phase 12** | Comprehensive Testing, Benchmarks & Hardware Guides: unit, integration, synthetic ground-truth, 12-camera scalability simulation, real-video Pose2Sim integration test, performance profiling | `TEST_REPORT.md`, `ACCURACY_REPORT.md`, `HARDWARE_TEST_GUIDE.md` updated with actual measured numbers |
| **Phase 13** | Packaging & Deliverables: PyInstaller bundle + Inno Setup `NuranovaMocapStudio_Setup.exe`, Android `NuranovaMocapCamera.apk`, Blender `NuranovaMocapBridge.zip`, printable calibration boards & marker sheets, sample project, complete manuals | All three installers/packages built, verified, and documented in `BUILD_GUIDE.md` & `USER_GUIDE.md` |
