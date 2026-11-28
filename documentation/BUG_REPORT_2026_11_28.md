# Bug audit and correction report — 28 November 2026

Baseline: `d85db02`, branch `kawshal/bugFix`. Environment: Windows, Python 3.11,
PySide6 6.7.3, ONNX Runtime CUDA, RTX 3060 Ti and Blender 5.2.3 LTS.

This audit found and fixed **15 defect groups** in storage, project operations,
review, calibration, diagnostics and the command line. Three rounds of focused
reproductions recorded **23 failing cases before their fixes**. Additional tests
cover rollback, cancellation, background downloads and project context cleanup.
An existing cross-stage solver test also checks that reviewed geometry overrides
an older packaged calibration.

P1 means possible data loss, wrong processing inputs, or blocked startup. P2
means an unresponsive workflow, conflicting jobs, or an avoidable command error.
“Fixed” below means the stated reproducer passes; it does not establish that
every possible input, camera, character or hardware configuration is defect-free.

## Defect index

| ID | Priority | Confirmed defect | Status |
| --- | --- | --- | --- |
| NMC-001 | P1 | Interrupted settings/project saves truncate the previous file | Fixed |
| NMC-002 | P1 | Failed motion serialization damages the previous HDF5 take | Fixed |
| NMC-003 | P1 | Failed or cancelled export can replace part of a completed delivery | Fixed |
| NMC-004 | P1 | Failed project copy closes the source; descendant copying is not rejected | Fixed |
| NMC-005 | P1 | Project restore overwrites nonempty folders or leaves invalid imports behind | Fixed |
| NMC-006 | P1 | Project ZIP includes its own output; a failed ZIP destroys the old archive | Fixed |
| NMC-007 | P1 | Invalid model downloads are published; bad caches cannot recover | Fixed |
| NMC-008 | P1 | Failed review import changes the source before validating its saved data | Fixed |
| NMC-009 | P2 | Diagnostics probes/downloads block the GUI, including a repeated startup probe | Fixed |
| NMC-010 | P1 | Project buttons bypass the processing lock and retain old project context | Fixed |
| NMC-011 | P1 | Calibration import/undo and the solver disagree about current geometry | Fixed |
| NMC-012 | P2 | Camera refinement can start while alignment is running | Fixed |
| NMC-013 | P2 | Invalid CLI configuration/calibration escapes as an uncaught exception | Fixed |
| NMC-014 | P1 | Explicit CLI calibration can silently fall back to estimated geometry | Fixed |
| NMC-015 | P1 | A malformed checksum cache prevents model registry/app startup | Fixed |

## Reproduction, effect and correction

<a id="nmc-001"></a>

### NMC-001 — interrupted metadata saves

Save settings or a project, change a value, then interrupt the next text write
after ten characters. The old writer opened the destination in write mode first;
the completed JSON was replaced by a short, unreadable fragment. The expected
behavior is to retain the previous complete file when the replacement fails.

Settings, project manifests and recent-project history now use unique staged
files in the destination directory, flushed before atomic replacement. The
fault-injection tests verify the previous bytes and valid JSON remain available.

<a id="nmc-002"></a>

### NMC-002 — native motion-file corruption

Save a valid take, then save to the same path with a processing metadata value
that cannot be serialized. Previously the HDF5 destination was truncated before
metadata serialization failed. Its old sidecar could remain beside a damaged
take.

The complete HDF5 and its manifest are now written in a staging directory before
publication. Serialization failures preserve both previous files. The full
supplied take is also checked through HDF5 save/load, including body, fingers,
face and original frame timestamps.

<a id="nmc-003"></a>

### NMC-003 — partial export and cancellation

Export a take, run it again under the same name, then fail the Blender JSON
writer after HDF5 completes. Previously the output directory contained a mixture
of old and new results. A cancellation during final serialization had a similar
problem: the UI could report cancellation after new files had replaced the old
delivery.

Exports are now staged as a set. Publication keeps recovery copies and restores
completed replacements if a later replacement fails or cancellation arrives.
Cancellation is checked before and between replacements. A result that has
already committed is reported as completed even if a cancel click arrives late.
Native review observations use a unique run filename so a failed re-solve cannot
overwrite the previous result's raw observations. Consumers read that path from
`processing_config.review_observations_path`.

Each rename is atomic; a multi-file batch is not a filesystem transaction against
power loss. If rollback itself fails, the error identifies retained recovery
copies rather than deleting them.

<a id="nmc-004"></a>

### NMC-004 — project copying

Duplicate a project into a destination that already exists. Previously copying
failed after the active project had already been closed. A destination inside
the source also lacked a guard against copying a directory into itself.

Destinations and names are checked before mutation. Copying and manifest changes
finish in a temporary directory, with a coherent SQLite backup. The source stays
active if preparation fails; a completed copy receives its new identity before
it becomes active.

<a id="nmc-005"></a>

### NMC-005 — project restoring

Restore a valid archive into a nonempty folder, or restore an archive without
`project.nmocap` into an empty folder. Previously extraction could overwrite
existing files or leave extracted content behind before validation failed.

Nonempty destinations are rejected. Extraction, path checks, manifest validation
and database integrity checks happen in a separate staging directory. The
destination and active project stay unchanged when validation fails.

<a id="nmc-006"></a>

### NMC-006 — unsafe backup output

Save a portable ZIP inside the project root. The old directory walk could include
the growing ZIP itself. Separately, interrupt ZIP writing: the previous archive
had already been truncated.

The archive destination and staging files are excluded from the input set. ZIP
publication happens only after writing completes. SQLite's backup API captures
the database coherently, without packaging its live journal/WAL files. An
injected ZIP-write failure retains the previous archive byte for byte.

<a id="nmc-007"></a>

### NMC-007 — failed model verification

Download bytes that do not match a registered SHA-256, or start with a corrupt
cached model larger than the size threshold. Previously verification happened
after publication; the bad file then counted as cached and subsequent attempts
failed without downloading a replacement.

Downloads now use unique staged files, verify before publication, and retry
invalid cached files. Concurrent download calls on one registry are serialized.
Checksum history is saved atomically. Models without a registered checksum still
use the existing first-download checksum policy; this is not independent proof
of model provenance.

<a id="nmc-008"></a>

### NMC-008 — failed review import

Open a take with manual edits, then import another take whose review JSON is
incomplete. Previously the old windows were removed and the incoming source/save
path assigned before loading failed. Old observations could then be associated
with the wrong source or saved to the wrong review path.

Incoming review state, source identity, geometry and observations are prepared
before changing views or the active source. A failed import retains the same
old views and edits. Review JSON and observation arrays are also saved as a
completed pair. An unsuccessful save prevents source replacement; reloading the
same source preserves pending manual edits.

<a id="nmc-009"></a>

### NMC-009 — blocking diagnostics

Construct the Diagnostics page after startup has already checked hardware, or
start a slow hardware probe/model download. Previously these operations ran on
the GUI thread; a second hardware probe also ran while startup built that page.

The page initially renders the existing startup result. Probes and downloads run
in workers, with queued GUI updates, explicit busy states and close guards.
Hardware changes update the main compute indicator. Tests hold each worker open
while GUI timer events continue and verify the work runs outside the GUI thread.

<a id="nmc-010"></a>

### NMC-010 — project-operation guards and context

While a solve is active, invoke Duplicate, Import ZIP or Delete from Projects.
Those buttons called the project manager directly, bypassing the controller's
idle check. They could switch or remove the project underneath processing.

These operations, and portable backup, now check the controller's processing and
recording guard and display an error instead of mutating the project. Operation
failures are handled in the UI. Successful switching/deletion clears the old
motion, calibration, Blender bridge identity and incoming-upload destination.

<a id="nmc-011"></a>

### NMC-011 — calibration dependencies and undo

Import a replacement profile after a solved review. Previously the result could
remain marked current and the saved review could retain the old geometry. Undo
also changed stored calibration without updating the live review geometry.
Finally, an older calibration inside a package could override reviewed geometry
when solving.

Calibration import now creates an undoable review revision and marks the solve
stale. Reviewed geometry is stored in the portable review, synchronized after
undo, and takes precedence during solving. It is still validated against the
selected observations. A cross-stage test passes competing packaged/reviewed
profiles through the actual reconstruction pipeline and checks the chosen one.

<a id="nmc-012"></a>

### NMC-012 — overlapping alignment and refinement

Start alignment, then request refinement from the camera review controls. The
refinement handler previously checked solve/review busy state but missed the
alignment flag. Both jobs could operate on different clock snapshots.

Refinement and calibration import now refuse to start while alignment or another
conflicting job is active. The reproducer verifies no refinement worker starts.

<a id="nmc-013"></a>

### NMC-013 — command-line error handling

Run `process` with `--height 0.2`, `--fps 0`, or a missing calibration file.
Configuration creation and calibration loading previously happened outside the
CLI's error handler and raised an uncaught exception.

Configuration/file validation now returns status 1 with a `Processing failed:`
message. The three input cases are tested through the CLI entry function.

<a id="nmc-014"></a>

### NMC-014 — explicit CLI calibration

Use `--calibration profile.json`. The old CLI still selected automatic geometry
mode, which could reject that profile and silently estimate another solution.

An explicit profile now selects validated studio-calibration mode. A mismatch
fails instead of silently substituting boardless geometry. A CLI regression test
checks the configuration passed to the pipeline.

<a id="nmc-015"></a>

### NMC-015 — malformed checksum metadata

Replace the checksum cache's JSON object with `[]`, then start the model registry.
Previously startup failed when the code called `.get` on the list.

The optional cache is now checked for the required object type and ignored when
malformed. The registry still loads its configured models and can rebuild the
checksum history on successful downloads.

## Validation evidence

Local artifacts are in
`D:\2026\nuranova-mocap\Mcapn_results\bug_audit_2026_11_28`.

- `reproduced_failures.xml`: 13 original failing cases.
- `second_pass_reproductions.xml`: five further failures and two passing recovery checks.
- `cli_cache_reproductions.xml`: five CLI/cache failures before correction.
- `full_suite_1.xml`: 144 passed, one optional benchmark skipped.
- `full_suite_2.xml`: 149 passed, one optional benchmark skipped.
- `full_suite_final.xml`: **151 passed**, one optional benchmark skipped,
  152 collected in 97.19 seconds. Ruff and Git whitespace checks also pass.
- `repeated_1.xml` through `repeated_5.xml`: 215 additional case executions,
  all passing, with rotated/reversed test-file order.
- `real_take_audit.json`: fresh full three-camera solve, 2,309 frames at 30 FPS,
  HDF5 round-trip checks and unchanged SHA-256 values for the original recordings
  and supplied Blender file. The first verification attempt found a boolean
  subtraction error in the audit script; the corrected verifier was rerun.
  The successful solve completes in 105.08 seconds. Tested body/finger/face arrays
  and source timestamps round-trip with maximum measured difference zero.
- `blender_keyframe_audit.json`: live Blender MCP evaluation of 2,309 integer and
  2,308 halfway poses on the prepared corrected character. Values are finite;
  maximum adjacent-frame turn is 20.921 degrees and maximum local scale is
  1.0000055. The prior scene time is restored and the file is not saved.
- `frozen_smoke.json` and `windows_bundle_manifest.json`: rebuilt Windows app
  launched outside the repository, all 21 pages and 27 startup tasks ready, CUDA
  inference on supplied footage, native video load/pop-out/175% zoom/return, and
  a real bundled Blender GLB export/reimport. Twenty-one archived modules plus
  the entry script match source bytecode; bundled Blender scripts match source.
- `frozen_cpu.json`: separate CPU-configured startup in the light theme with
  reduced motion, plus Blender export. This check does not perform CPU video
  inference; the packaged video-inference check above uses CUDA.
- `visual_qa.json`: actual light/dark camera and hand review at 1366 × 768, with
  all three source cameras and unchanged capture-annotation revision.

Regression sources:
[failure safety](../tests/integration/test_bug_audit_failure_safety.py),
[workflow recovery](../tests/integration/test_bug_audit_workflow_recovery.py),
[actual solver constraints](../tests/integration/test_review_solve_constraints.py).

Reproduce the automated checks with:

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
.venv/Scripts/python.exe -m pytest
.venv/Scripts/ruff.exe check .
```

## Open acceptance limits

The fixes above close reproducible software failures. The supplied capture and
character still have previously documented contact/visibility limits. The earlier
deformed-mesh scan reported remaining contacts; this audit's continuity test does
not establish contact-free skinning or reference motion accuracy. Some recordings
hide fingers or show very small palms. Those intervals need capture/character
review, with evidence retained in the
[correction implementation report](CORRECTION_IMPLEMENTATION_2026_11_28.md).

Physical phone/marker capture, additional wardrobe/control-rig combinations and
the optional unavailable Pose2Sim reference-footage benchmark remain validation
gates. Android and a setup installer were not rebuilt in this Windows audit.
