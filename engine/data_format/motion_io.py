"""Versioned HDF5 (.nmocap.h5) & JSON (.nmocap.json) motion data container.

Conventions:
- Units: meters (m), radians (rad), integer nanoseconds (int64)
- Coordinate system: Right-handed Z-up (+X Right, +Y Forward, +Z Up)
- Quaternions: scalar-last [x, y, z, w]
- Matrices: 4x4 row-major active column-vector homogeneous transforms
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import h5py
import numpy as np

FORMAT_VERSION = "1.0.0"


class ProvenanceCode(IntEnum):
    MISSING = 0
    MEASURED_MARKER = 1
    MEASURED_AI = 2
    HYBRID_FUSED = 3
    INTERPOLATED = 4
    IK_ESTIMATED = 5


@dataclass
class SkeletalMotionTrack:
    root_translation: np.ndarray  # (T, 3) float32
    root_rotation_xyzw: np.ndarray  # (T, 4) float32
    local_rotations_xyzw: np.ndarray  # (T, B, 4) float32
    global_matrices: np.ndarray  # (T, B, 4, 4) float32
    joint_positions: np.ndarray  # (T, B, 3) float32
    joint_confidence: np.ndarray  # (T, B) float32
    rotation_observable: Optional[np.ndarray] = None  # (T, B) bool
    foot_contacts: Optional[np.ndarray] = None  # (T, 4) uint8 [L_heel, L_toe, R_heel, R_toe]
    cleanup_config: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return int(self.root_translation.shape[0])


@dataclass
class FaceMotionTrack:
    blendshapes: np.ndarray  # (T, 52) float32
    blendshape_names: List[str]
    head_rotation_xyzw: np.ndarray  # (T, 4) float32
    gaze_left_right: Optional[np.ndarray] = None  # (T, 4) float32
    confidence: Optional[np.ndarray] = None  # (T,) float32

    @property
    def num_frames(self) -> int:
        return int(self.blendshapes.shape[0])


@dataclass
class HandMotionTrack:
    left_local_rotations_xyzw: Optional[np.ndarray] = None  # (T, 16, 4) float32
    left_confidence: Optional[np.ndarray] = None  # (T, 21) float32
    right_local_rotations_xyzw: Optional[np.ndarray] = None  # (T, 16, 4) float32
    right_confidence: Optional[np.ndarray] = None  # (T, 21) float32

    @property
    def num_frames(self) -> int:
        if self.left_local_rotations_xyzw is not None:
            return int(self.left_local_rotations_xyzw.shape[0])
        if self.right_local_rotations_xyzw is not None:
            return int(self.right_local_rotations_xyzw.shape[0])
        return 0


@dataclass
class MotionCaptureDocument:
    project_id: str
    session_id: str
    take_id: str
    actor_id: str
    fps: float
    frame_timestamps_ns: np.ndarray  # (T,) int64
    bone_names: List[str]
    parent_indices: List[int]
    rest_offsets: np.ndarray  # (B, 3) float32 meters
    raw_motion: SkeletalMotionTrack
    corrected_motion: Optional[SkeletalMotionTrack] = None
    capture_mode: str = "hybrid"
    processing_profile: str = "cinema"
    cameras: List[Dict[str, Any]] = field(default_factory=list)
    calibration: Dict[str, Any] = field(default_factory=dict)
    sync_info: Dict[str, Any] = field(default_factory=dict)
    processing_config: Dict[str, Any] = field(default_factory=dict)
    model_versions: Dict[str, str] = field(default_factory=dict)
    marker_names: List[str] = field(default_factory=list)
    markers_3d: Optional[np.ndarray] = None  # (T, M, 3)
    markers_conf: Optional[np.ndarray] = None  # (T, M)
    markers_reproj_err_px: Optional[np.ndarray] = None  # (T, M)
    markers_provenance: Optional[np.ndarray] = None  # (T, M) uint8
    joints_3d_raw: Optional[np.ndarray] = None  # (T, J, 3)
    joints_conf_raw: Optional[np.ndarray] = None  # (T, J)
    joints_reproj_err_px: Optional[np.ndarray] = None  # (T, J)
    joints_provenance: Optional[np.ndarray] = None  # (T, J) uint8
    observations_2d: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    face: Optional[FaceMotionTrack] = None
    hands: Optional[HandMotionTrack] = None
    format_version: str = FORMAT_VERSION

    @property
    def num_frames(self) -> int:
        return int(self.frame_timestamps_ns.shape[0])

    @property
    def frame_times_s(self) -> np.ndarray:
        if self.num_frames == 0:
            return np.zeros((0,), dtype=np.float64)
        t0 = self.frame_timestamps_ns[0]
        return (self.frame_timestamps_ns - t0).astype(np.float64) * 1e-9


def _write_track(group: h5py.Group, track: SkeletalMotionTrack) -> None:
    group.create_dataset("root_translation", data=np.asarray(track.root_translation, dtype=np.float32), compression="gzip")
    group.create_dataset("root_rotation_xyzw", data=np.asarray(track.root_rotation_xyzw, dtype=np.float32), compression="gzip")
    group.create_dataset("local_rotations_xyzw", data=np.asarray(track.local_rotations_xyzw, dtype=np.float32), compression="gzip")
    group.create_dataset("global_matrices", data=np.asarray(track.global_matrices, dtype=np.float32), compression="gzip")
    group.create_dataset("joint_positions", data=np.asarray(track.joint_positions, dtype=np.float32), compression="gzip")
    group.create_dataset("joint_confidence", data=np.asarray(track.joint_confidence, dtype=np.float32), compression="gzip")
    if track.rotation_observable is not None:
        group.create_dataset("rotation_observable", data=np.asarray(track.rotation_observable, dtype=np.bool_), compression="gzip")
    if track.foot_contacts is not None:
        group.create_dataset("foot_contacts", data=np.asarray(track.foot_contacts, dtype=np.uint8), compression="gzip")
    group.attrs["cleanup_config_json"] = json.dumps(track.cleanup_config)


def _read_track(group: h5py.Group) -> SkeletalMotionTrack:
    return SkeletalMotionTrack(
        root_translation=group["root_translation"][:],
        root_rotation_xyzw=group["root_rotation_xyzw"][:],
        local_rotations_xyzw=group["local_rotations_xyzw"][:],
        global_matrices=group["global_matrices"][:],
        joint_positions=group["joint_positions"][:],
        joint_confidence=group["joint_confidence"][:],
        rotation_observable=group["rotation_observable"][:] if "rotation_observable" in group else None,
        foot_contacts=group["foot_contacts"][:] if "foot_contacts" in group else None,
        cleanup_config=json.loads(group.attrs.get("cleanup_config_json", "{}")),
    )


def save_motion_file(doc: MotionCaptureDocument, h5_path: str | Path, write_json_sidecar: bool = True) -> Path:
    """Save a complete versioned `.nmocap.h5` motion file and optional `.nmocap.json` manifest."""
    path = Path(h5_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = doc.format_version
        f.attrs["units"] = "meters"
        f.attrs["angle_units"] = "radians"
        f.attrs["timestamp_units"] = "nanoseconds"
        f.attrs["up_axis"] = "Z"
        f.attrs["handedness"] = "right"
        f.attrs["quaternion_order"] = "xyzw"
        f.attrs["matrix_convention"] = "column_vector_row_major"

        meta = f.create_group("metadata")
        meta.attrs["project_id"] = doc.project_id
        meta.attrs["session_id"] = doc.session_id
        meta.attrs["take_id"] = doc.take_id
        meta.attrs["actor_id"] = doc.actor_id
        meta.attrs["capture_mode"] = doc.capture_mode
        meta.attrs["processing_profile"] = doc.processing_profile
        meta.attrs["cameras_json"] = json.dumps(doc.cameras)
        meta.attrs["calibration_json"] = json.dumps(doc.calibration)
        meta.attrs["sync_json"] = json.dumps(doc.sync_info)
        meta.attrs["processing_config_json"] = json.dumps(doc.processing_config)
        meta.attrs["model_versions_json"] = json.dumps(doc.model_versions)
        meta.attrs["skeleton_json"] = json.dumps({
            "bone_names": doc.bone_names,
            "parent_indices": doc.parent_indices,
            "rest_offsets": np.asarray(doc.rest_offsets, dtype=np.float64).tolist(),
        })
        meta.attrs["marker_names_json"] = json.dumps(doc.marker_names)

        tl = f.create_group("timeline")
        tl.attrs["nominal_fps"] = float(doc.fps)
        tl.create_dataset("frame_timestamps_ns", data=np.asarray(doc.frame_timestamps_ns, dtype=np.int64))
        tl.create_dataset("frame_times_s", data=doc.frame_times_s)

        if doc.observations_2d:
            obs_grp = f.create_group("observations_2d")
            for cam_uid, payload in doc.observations_2d.items():
                cg = obs_grp.create_group(str(cam_uid))
                for k, arr in payload.items():
                    cg.create_dataset(k, data=np.asarray(arr), compression="gzip")

        rec = f.create_group("reconstructed_3d")
        if doc.markers_3d is not None:
            rec.create_dataset("markers_pos", data=np.asarray(doc.markers_3d, dtype=np.float32), compression="gzip")
        if doc.markers_conf is not None:
            rec.create_dataset("markers_conf", data=np.asarray(doc.markers_conf, dtype=np.float32), compression="gzip")
        if doc.markers_reproj_err_px is not None:
            rec.create_dataset("markers_reproj_err_px", data=np.asarray(doc.markers_reproj_err_px, dtype=np.float32), compression="gzip")
        if doc.markers_provenance is not None:
            rec.create_dataset("markers_provenance", data=np.asarray(doc.markers_provenance, dtype=np.uint8), compression="gzip")
        if doc.joints_3d_raw is not None:
            rec.create_dataset("joints_pos_raw", data=np.asarray(doc.joints_3d_raw, dtype=np.float32), compression="gzip")
        if doc.joints_conf_raw is not None:
            rec.create_dataset("joints_conf", data=np.asarray(doc.joints_conf_raw, dtype=np.float32), compression="gzip")
        if doc.joints_reproj_err_px is not None:
            rec.create_dataset("joints_reproj_err_px", data=np.asarray(doc.joints_reproj_err_px, dtype=np.float32), compression="gzip")
        if doc.joints_provenance is not None:
            rec.create_dataset("joints_provenance", data=np.asarray(doc.joints_provenance, dtype=np.uint8), compression="gzip")

        _write_track(f.create_group("raw_motion"), doc.raw_motion)
        if doc.corrected_motion is not None:
            _write_track(f.create_group("corrected_motion"), doc.corrected_motion)

        if doc.face is not None:
            fg = f.create_group("face")
            fg.create_dataset("blendshapes", data=np.asarray(doc.face.blendshapes, dtype=np.float32), compression="gzip")
            fg.attrs["blendshape_names_json"] = json.dumps(doc.face.blendshape_names)
            fg.create_dataset("head_rotation_xyzw", data=np.asarray(doc.face.head_rotation_xyzw, dtype=np.float32), compression="gzip")
            if doc.face.gaze_left_right is not None:
                fg.create_dataset("gaze_left_right", data=np.asarray(doc.face.gaze_left_right, dtype=np.float32), compression="gzip")
            if doc.face.confidence is not None:
                fg.create_dataset("face_confidence", data=np.asarray(doc.face.confidence, dtype=np.float32), compression="gzip")

        if doc.hands is not None:
            hg = f.create_group("hands")
            if doc.hands.left_local_rotations_xyzw is not None:
                hg.create_dataset("left_local_rotations_xyzw", data=np.asarray(doc.hands.left_local_rotations_xyzw, dtype=np.float32), compression="gzip")
            if doc.hands.left_confidence is not None:
                hg.create_dataset("left_confidence", data=np.asarray(doc.hands.left_confidence, dtype=np.float32), compression="gzip")
            if doc.hands.right_local_rotations_xyzw is not None:
                hg.create_dataset("right_local_rotations_xyzw", data=np.asarray(doc.hands.right_local_rotations_xyzw, dtype=np.float32), compression="gzip")
            if doc.hands.right_confidence is not None:
                hg.create_dataset("right_confidence", data=np.asarray(doc.hands.right_confidence, dtype=np.float32), compression="gzip")

    if write_json_sidecar:
        sidecar = path.with_suffix(".json")
        if path.name.endswith(".nmocap.h5"):
            sidecar = path.with_name(path.name[:-len(".nmocap.h5")] + ".nmocap.json")
        sidecar_payload = {
            "format_version": doc.format_version,
            "conventions": {
                "units": "meters",
                "up_axis": "Z",
                "handedness": "right",
                "quaternion_order": "xyzw",
                "timestamp_units": "nanoseconds",
            },
            "h5_file": path.name,
            "project_id": doc.project_id,
            "session_id": doc.session_id,
            "take_id": doc.take_id,
            "actor_id": doc.actor_id,
            "capture_mode": doc.capture_mode,
            "processing_profile": doc.processing_profile,
            "fps": float(doc.fps),
            "num_frames": doc.num_frames,
            "skeleton": {
                "bone_names": doc.bone_names,
                "parent_indices": doc.parent_indices,
                "rest_offsets": np.asarray(doc.rest_offsets, dtype=np.float64).tolist(),
            },
            "has_corrected_motion": doc.corrected_motion is not None,
            "has_face": doc.face is not None,
            "has_hands": doc.hands is not None,
            "cameras": doc.cameras,
            "sync_info": doc.sync_info,
            "model_versions": doc.model_versions,
        }
        sidecar.write_text(json.dumps(sidecar_payload, indent=2), encoding="utf-8")

    return path


def export_blender_json(doc: MotionCaptureDocument, json_path: str | Path, use_corrected: bool = True) -> Path:
    """Export a self-contained JSON payload directly readable by Blender without requiring h5py inside Blender."""
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    track = doc.corrected_motion if (use_corrected and doc.corrected_motion is not None) else doc.raw_motion

    payload: Dict[str, Any] = {
        "format_version": doc.format_version,
        "conventions": {
            "units": "meters",
            "up_axis": "Z",
            "handedness": "right",
            "quaternion_order": "xyzw",
        },
        "project_id": doc.project_id,
        "session_id": doc.session_id,
        "take_id": doc.take_id,
        "actor_id": doc.actor_id,
        "fps": float(doc.fps),
        "num_frames": doc.num_frames,
        "capture_mode": doc.capture_mode,
        "processing_profile": doc.processing_profile,
        "source_track": "corrected" if (use_corrected and doc.corrected_motion is not None) else "raw",
        "skeleton": {
            "bone_names": doc.bone_names,
            "parent_indices": doc.parent_indices,
            "rest_offsets": np.asarray(doc.rest_offsets, dtype=np.float64).tolist(),
        },
        "frames": {
            "times_s": doc.frame_times_s.tolist(),
            "root_translation": np.asarray(track.root_translation, dtype=np.float64).tolist(),
            "root_rotation_xyzw": np.asarray(track.root_rotation_xyzw, dtype=np.float64).tolist(),
            "local_rotations_xyzw": np.asarray(track.local_rotations_xyzw, dtype=np.float64).tolist(),
            "joint_positions": np.asarray(track.joint_positions, dtype=np.float64).tolist(),
            "joint_confidence": np.asarray(track.joint_confidence, dtype=np.float64).tolist(),
            "foot_contacts": track.foot_contacts.tolist() if track.foot_contacts is not None else None,
        },
    }

    if doc.face is not None:
        payload["face"] = {
            "blendshape_names": doc.face.blendshape_names,
            "blendshapes": np.asarray(doc.face.blendshapes, dtype=np.float64).tolist(),
            "head_rotation_xyzw": np.asarray(doc.face.head_rotation_xyzw, dtype=np.float64).tolist(),
            "confidence": np.asarray(doc.face.confidence, dtype=np.float64).tolist() if doc.face.confidence is not None else None,
        }

    if doc.hands is not None:
        payload["hands"] = {
            "left_local_rotations_xyzw": (
                np.asarray(doc.hands.left_local_rotations_xyzw, dtype=np.float64).tolist()
                if doc.hands.left_local_rotations_xyzw is not None
                else None
            ),
            "right_local_rotations_xyzw": (
                np.asarray(doc.hands.right_local_rotations_xyzw, dtype=np.float64).tolist()
                if doc.hands.right_local_rotations_xyzw is not None
                else None
            ),
        }

    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# Schema migration hooks for future backwards compatibility
_MIGRATIONS: Dict[str, Callable[[MotionCaptureDocument], MotionCaptureDocument]] = {}


def load_motion_file(h5_path: str | Path) -> MotionCaptureDocument:
    """Load a `.nmocap.h5` motion file and apply any schema migrations."""
    path = Path(h5_path)
    with h5py.File(path, "r") as f:
        fmt_ver = str(f.attrs.get("format_version", "1.0.0"))
        meta = f["metadata"]
        skel = json.loads(meta.attrs["skeleton_json"])

        tl = f["timeline"]
        fps = float(tl.attrs["nominal_fps"])
        timestamps = tl["frame_timestamps_ns"][:]

        obs_2d: Dict[str, Dict[str, np.ndarray]] = {}
        if "observations_2d" in f:
            for cam_uid in f["observations_2d"]:
                cg = f["observations_2d"][cam_uid]
                obs_2d[cam_uid] = {k: cg[k][:] for k in cg}

        rec = f["reconstructed_3d"] if "reconstructed_3d" in f else None
        raw_motion = _read_track(f["raw_motion"])
        corrected_motion = _read_track(f["corrected_motion"]) if "corrected_motion" in f else None

        face = None
        if "face" in f:
            fg = f["face"]
            face = FaceMotionTrack(
                blendshapes=fg["blendshapes"][:],
                blendshape_names=json.loads(fg.attrs["blendshape_names_json"]),
                head_rotation_xyzw=fg["head_rotation_xyzw"][:],
                gaze_left_right=fg["gaze_left_right"][:] if "gaze_left_right" in fg else None,
                confidence=fg["face_confidence"][:] if "face_confidence" in fg else None,
            )

        hands = None
        if "hands" in f:
            hg = f["hands"]
            hands = HandMotionTrack(
                left_local_rotations_xyzw=hg["left_local_rotations_xyzw"][:] if "left_local_rotations_xyzw" in hg else None,
                left_confidence=hg["left_confidence"][:] if "left_confidence" in hg else None,
                right_local_rotations_xyzw=hg["right_local_rotations_xyzw"][:] if "right_local_rotations_xyzw" in hg else None,
                right_confidence=hg["right_confidence"][:] if "right_confidence" in hg else None,
            )

        doc = MotionCaptureDocument(
            project_id=str(meta.attrs["project_id"]),
            session_id=str(meta.attrs["session_id"]),
            take_id=str(meta.attrs["take_id"]),
            actor_id=str(meta.attrs["actor_id"]),
            fps=fps,
            frame_timestamps_ns=timestamps,
            bone_names=list(skel["bone_names"]),
            parent_indices=list(skel["parent_indices"]),
            rest_offsets=np.asarray(skel["rest_offsets"], dtype=np.float32),
            raw_motion=raw_motion,
            corrected_motion=corrected_motion,
            capture_mode=str(meta.attrs.get("capture_mode", "hybrid")),
            processing_profile=str(meta.attrs.get("processing_profile", "cinema")),
            cameras=json.loads(meta.attrs.get("cameras_json", "[]")),
            calibration=json.loads(meta.attrs.get("calibration_json", "{}")),
            sync_info=json.loads(meta.attrs.get("sync_json", "{}")),
            processing_config=json.loads(meta.attrs.get("processing_config_json", "{}")),
            model_versions=json.loads(meta.attrs.get("model_versions_json", "{}")),
            marker_names=json.loads(meta.attrs.get("marker_names_json", "[]")),
            markers_3d=rec["markers_pos"][:] if (rec is not None and "markers_pos" in rec) else None,
            markers_conf=rec["markers_conf"][:] if (rec is not None and "markers_conf" in rec) else None,
            markers_reproj_err_px=rec["markers_reproj_err_px"][:] if (rec is not None and "markers_reproj_err_px" in rec) else None,
            markers_provenance=rec["markers_provenance"][:] if (rec is not None and "markers_provenance" in rec) else None,
            joints_3d_raw=rec["joints_pos_raw"][:] if (rec is not None and "joints_pos_raw" in rec) else None,
            joints_conf_raw=rec["joints_conf"][:] if (rec is not None and "joints_conf" in rec) else None,
            joints_reproj_err_px=rec["joints_reproj_err_px"][:] if (rec is not None and "joints_reproj_err_px" in rec) else None,
            joints_provenance=rec["joints_provenance"][:] if (rec is not None and "joints_provenance" in rec) else None,
            observations_2d=obs_2d,
            face=face,
            hands=hands,
            format_version=fmt_ver,
        )

    while doc.format_version in _MIGRATIONS:
        doc = _MIGRATIONS[doc.format_version](doc)
    return doc
