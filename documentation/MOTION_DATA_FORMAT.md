# Nuranova Internal Motion Data Format Specification (`nmocap-motion-v1.0`)

**Format Version:** `1.0.0` (`nmocap-motion-v1.0`)  
**Primary Binary Container:** HDF5 (`.nmocap.h5`)  
**Companion Metadata Manifest:** JSON (`.nmocap.json`)

---

## 1. Conventions (Mandatory)

- **Units:** Meters (`m`) for 3D positions & bone lengths; radians (`rad`) for rotations; seconds (`s`) for derived time, integer nanoseconds (`int64`) for monotonic/session timestamps.
- **Coordinate System:** Right-handed **Z-Up** (`+X` Right, `+Y` Forward, `+Z` Up — matches Blender native space).
- **Quaternion Component Order:** Scalar-last **`[x, y, z, w]`** unit quaternions, representing **local bone rotation relative to parent rest pose** (`q_local`).
- **Matrix Convention:** Row-major `4x4` homogeneous column-vector matrices `M_world` (`p_world = M_world[:3,:3] @ p_local + M_world[:3,3]`).
- **Observation Provenance Codes (`uint8`):**
  - `0` = `MISSING` (no observation / not solved)
  - `1` = `MEASURED_MARKER` (multi-view physical marker triangulation)
  - `2` = `MEASURED_AI` (multi-view AI landmark triangulation)
  - `3` = `HYBRID_FUSED` (calibrated physical marker + AI constrained fusion)
  - `4` = `INTERPOLATED` (temporal gap fill)
  - `5` = `IK_ESTIMATED` (kinematically inferred or constrained)

---

## 2. HDF5 Group & Dataset Hierarchy

```text
/                                   # Root attrs: format_version="1.0.0", units="meters", up_axis="Z", quat_order="xyzw"
├── /metadata                       # JSON-serialized UTF-8 string attributes:
│   ├── project_id, session_id, take_id, actor_id, capture_mode, processing_profile
│   ├── cameras_json, calibration_json, sync_json, skeleton_json, processing_config_json, model_versions_json
├── /timeline
│   ├── frame_timestamps_ns         # int64 (T,) — canonical session timeline in nanoseconds
│   ├── frame_times_s               # float64 (T,) — seconds relative to frame 0
│   └── nominal_fps                 # attr float64
├── /observations_2d                # Raw retained 2D observations per camera for reprocessing
│   └── /<camera_uid>
│       ├── keypoints_2d            # float32 (T, K, 3) -> [u_px, v_px, confidence]
│       ├── markers_2d              # float32 (T, M, 3) -> [u_px, v_px, confidence]
│       └── frame_valid             # bool (T,)
├── /reconstructed_3d
│   ├── markers_pos                 # float32 (T, M, 3) meters, NaN when unobserved
│   ├── markers_conf                # float32 (T, M) [0..1]
│   ├── markers_reproj_err_px       # float32 (T, M) pixels
│   ├── markers_provenance          # uint8 (T, M)
│   ├── joints_pos_raw              # float32 (T, J, 3) meters
│   ├── joints_conf                 # float32 (T, J) [0..1]
│   ├── joints_reproj_err_px        # float32 (T, J) pixels
│   └── joints_provenance           # uint8 (T, J)
├── /raw_motion                     # Unfiltered solved skeletal motion
│   ├── root_translation            # float32 (T, 3) meters
│   ├── root_rotation_xyzw          # float32 (T, 4) unit quaternion
│   ├── local_rotations_xyzw        # float32 (T, B, 4) unit quaternion per bone
│   ├── global_matrices             # float32 (T, B, 4, 4)
│   ├── joint_positions             # float32 (T, B, 3) meters
│   ├── joint_confidence            # float32 (T, B)
│   └── rotation_observable         # bool (T, B) — flags observable vs ambiguous twist/axis
├── /corrected_motion               # Non-destructive cleaned motion (same schema as /raw_motion)
│   ├── root_translation            # float32 (T, 3)
│   ├── root_rotation_xyzw          # float32 (T, 4)
│   ├── local_rotations_xyzw        # float32 (T, B, 4)
│   ├── global_matrices             # float32 (T, B, 4, 4)
│   ├── joint_positions             # float32 (T, B, 3)
│   ├── joint_confidence            # float32 (T, B)
│   ├── foot_contacts               # uint8 (T, 4) -> [left_heel, left_toe, right_heel, right_toe]
│   └── cleanup_config_json         # attr JSON
├── /hands
│   ├── left_local_rotations_xyzw   # float32 (T, 16, 4)
│   ├── left_confidence             # float32 (T, 21)
│   ├── right_local_rotations_xyzw  # float32 (T, 16, 4)
│   └── right_confidence            # float32 (T, 21)
└── /face
    ├── blendshapes                 # float32 (T, 52) — MediaPipe 52 blendshape scores [0..1]
    ├── blendshape_names_json       # attr JSON list of 52 canonical names
    ├── head_rotation_xyzw          # float32 (T, 4)
    ├── gaze_left_right             # float32 (T, 4) -> [l_yaw, l_pitch, r_yaw, r_pitch] rad
    └── face_confidence             # float32 (T,)
```

## 3. Backwards-Compatible Schema Migration

`engine.data_format.motion_io.load_motion_file()` inspects root attribute `format_version` and runs deterministic versioned migrations (`MIGRATIONS` registry) so future v1.x/v2.x builds continue opening v1.0.0 project files cleanly.
