"""Unit tests for versioned HDF5 (.nmocap.h5) and JSON motion container."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from engine.data_format.motion_io import (
    FaceMotionTrack,
    HandMotionTrack,
    MotionCaptureDocument,
    ProvenanceCode,
    SkeletalMotionTrack,
    export_blender_json,
    load_motion_file,
    save_motion_file,
)


def test_motion_hdf5_and_blender_json_roundtrip(tmp_path: Path):
    T, B, M = 30, 6, 4
    ts_ns = np.arange(T, dtype=np.int64) * int(1e9 / 60)
    bone_names = ["hips", "spine", "chest", "neck", "head", "left_upper_leg"]
    parents = [-1, 0, 1, 2, 3, 0]
    rest = np.array([[0, 0, 0.95], [0, 0, 0.15], [0, 0, 0.2], [0, 0, 0.12], [0, 0, 0.12], [-0.1, 0, -0.05]], dtype=np.float32)

    local_q = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (T, B, 1))
    mats = np.tile(np.eye(4, dtype=np.float32), (T, B, 1, 1))
    raw = SkeletalMotionTrack(
        root_translation=np.zeros((T, 3), dtype=np.float32),
        root_rotation_xyzw=np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (T, 1)),
        local_rotations_xyzw=local_q,
        global_matrices=mats,
        joint_positions=np.zeros((T, B, 3), dtype=np.float32),
        joint_confidence=np.ones((T, B), dtype=np.float32),
        rotation_observable=np.ones((T, B), dtype=np.bool_),
    )
    corr = SkeletalMotionTrack(
        root_translation=np.ones((T, 3), dtype=np.float32) * 0.1,
        root_rotation_xyzw=raw.root_rotation_xyzw.copy(),
        local_rotations_xyzw=local_q.copy(),
        global_matrices=mats.copy(),
        joint_positions=raw.joint_positions.copy(),
        joint_confidence=raw.joint_confidence.copy(),
        foot_contacts=np.ones((T, 4), dtype=np.uint8),
        cleanup_config={"filter": "butterworth", "cutoff_hz": 8.0},
    )
    face = FaceMotionTrack(
        blendshapes=np.linspace(0, 1, T * 52, dtype=np.float32).reshape(T, 52),
        blendshape_names=[f"bs_{i}" for i in range(52)],
        head_rotation_xyzw=raw.root_rotation_xyzw.copy(),
        confidence=np.ones((T,), dtype=np.float32),
    )
    hands = HandMotionTrack(
        left_local_rotations_xyzw=np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (T, 16, 1)),
        left_confidence=np.ones((T, 21), dtype=np.float32),
    )

    doc = MotionCaptureDocument(
        project_id="proj-1",
        session_id="sess-1",
        take_id="take-01",
        actor_id="actor-1",
        fps=60.0,
        frame_timestamps_ns=ts_ns,
        bone_names=bone_names,
        parent_indices=parents,
        rest_offsets=rest,
        raw_motion=raw,
        corrected_motion=corr,
        marker_names=[f"m{i}" for i in range(M)],
        markers_3d=np.zeros((T, M, 3), dtype=np.float32),
        markers_conf=np.ones((T, M), dtype=np.float32),
        markers_provenance=np.full((T, M), int(ProvenanceCode.MEASURED_MARKER), dtype=np.uint8),
        face=face,
        hands=hands,
        model_versions={"rtmw": "rtmw-x-384x288"},
    )

    h5_file = save_motion_file(doc, tmp_path / "take_01.nmocap.h5", write_json_sidecar=True)
    assert h5_file.exists()
    assert (tmp_path / "take_01.nmocap.json").exists()

    loaded = load_motion_file(h5_file)
    assert loaded.num_frames == T
    assert loaded.bone_names == bone_names
    assert loaded.corrected_motion is not None
    assert np.allclose(loaded.corrected_motion.root_translation, corr.root_translation)
    assert loaded.face is not None and loaded.face.blendshapes.shape == (T, 52)
    assert loaded.hands is not None and loaded.hands.left_local_rotations_xyzw.shape == (T, 16, 4)

    b_json = export_blender_json(loaded, tmp_path / "take_01_blender.json", use_corrected=True)
    parsed = json.loads(b_json.read_text(encoding="utf-8"))
    assert parsed["source_track"] == "corrected"
    assert len(parsed["frames"]["root_translation"]) == T
    assert len(parsed["face"]["blendshape_names"]) == 52
