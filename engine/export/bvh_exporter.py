"""Standard Biovision Hierarchy (.bvh) animation exporter."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.spatial.transform import Rotation

from engine.data_format.motion_io import MotionCaptureDocument


def export_bvh(
    doc: MotionCaptureDocument,
    output_path: str | Path,
    use_corrected: bool = True,
    scale_to_cm: bool = True,
) -> Path:
    """Export articulated skeletal animation to standard .bvh format (ZYX Euler degrees)."""
    track = doc.corrected_motion if (use_corrected and doc.corrected_motion is not None) else doc.raw_motion
    scale = 100.0 if scale_to_cm else 1.0
    names = doc.bone_names
    parents = doc.parent_indices
    offsets = np.asarray(doc.rest_offsets, dtype=np.float64) * scale

    children_map: Dict[int, List[int]] = {i: [] for i in range(len(names))}
    root_idx = 0
    for idx, p in enumerate(parents):
        if p < 0:
            root_idx = idx
        else:
            children_map[p].append(idx)

    traversal_order: List[int] = []
    lines: List[str] = ["HIERARCHY"]

    def write_joint(idx: int, depth: int) -> None:
        indent = "  " * depth
        kind = "ROOT" if parents[idx] < 0 else "JOINT"
        lines.append(f"{indent}{kind} {names[idx]}")
        lines.append(f"{indent}{{")
        off = np.zeros(3) if parents[idx] < 0 else offsets[idx]
        lines.append(f"{indent}  OFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}")
        if parents[idx] < 0:
            lines.append(f"{indent}  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation")
        else:
            lines.append(f"{indent}  CHANNELS 3 Zrotation Yrotation Xrotation")
        traversal_order.append(idx)

        kids = children_map[idx]
        if not kids:
            lines.append(f"{indent}  End Site")
            lines.append(f"{indent}  {{")
            lines.append(f"{indent}    OFFSET 0.000000 0.000000 5.000000")
            lines.append(f"{indent}  }}")
        else:
            for c_idx in kids:
                write_joint(c_idx, depth + 1)
        lines.append(f"{indent}}}")

    write_joint(root_idx, 0)

    T = doc.num_frames
    frame_time = 1.0 / max(float(doc.fps), 1.0)
    lines.append("MOTION")
    lines.append(f"Frames: {T}")
    lines.append(f"Frame Time: {frame_time:.6f}")

    root_trans = np.asarray(track.root_translation, dtype=np.float64) * scale
    local_q = np.asarray(track.local_rotations_xyzw, dtype=np.float64)

    for t in range(T):
        vals: List[str] = []
        for b_idx in traversal_order:
            if b_idx == root_idx:
                rt = root_trans[t]
                vals.extend([f"{rt[0]:.5f}", f"{rt[1]:.5f}", f"{rt[2]:.5f}"])
            q = local_q[t, b_idx]
            nrm = np.linalg.norm(q)
            q = q / nrm if nrm > 1e-8 else np.array([0.0, 0.0, 0.0, 1.0])
            zyx = Rotation.from_quat(q).as_euler("ZYX", degrees=True)
            vals.extend([f"{zyx[0]:.4f}", f"{zyx[1]:.4f}", f"{zyx[2]:.4f}"])
        lines.append(" ".join(vals))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
