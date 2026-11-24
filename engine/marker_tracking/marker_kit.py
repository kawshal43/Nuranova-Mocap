"""Configurable wearable physical marker kit, rigid marker clusters, and printable ArUco/AprilTag sheets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from desktop.calibration.charuco_board import ARUCO_DICTIONARIES

MARKER_GROUPS: List[str] = [
    "Head",
    "Neck",
    "Chest",
    "Back",
    "Pelvis",
    "Shoulders",
    "Upper Arms",
    "Elbows",
    "Forearms",
    "Wrists",
    "Hands",
    "Thighs",
    "Knees",
    "Lower Legs",
    "Ankles",
    "Feet",
]


@dataclass
class MarkerDefinition:
    marker_id: str
    name: str
    group: str
    attached_bone: str
    local_offset_m: List[float]  # [x, y, z] in attached bone local frame (meters)
    marker_type: str = "sphere_white"  # "sphere_white", "aruco", "apriltag"
    tag_id: Optional[int] = None
    tag_dictionary: str = "DICT_APRILTAG_36h11"
    diameter_mm: float = 19.0
    cluster_id: Optional[str] = None


@dataclass
class RigidClusterDefinition:
    cluster_id: str
    name: str
    attached_bone: str
    marker_ids: List[str]  # >= 3 non-collinear markers (asymmetric geometry recommended)
    local_points_m: List[List[float]]  # (K, 3) coordinates in rigid cluster local frame
    cluster_to_bone_quat_xyzw: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    cluster_to_bone_offset_m: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class MarkerKitConfig:
    kit_name: str
    description: str
    markers: List[MarkerDefinition] = field(default_factory=list)
    clusters: List[RigidClusterDefinition] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "kit_name": self.kit_name,
            "description": self.description,
            "markers": [asdict(m) for m in self.markers],
            "clusters": [asdict(c) for c in self.clusters],
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def from_dict(cls, d: Dict) -> "MarkerKitConfig":
        return cls(
            kit_name=str(d["kit_name"]),
            description=str(d.get("description", "")),
            markers=[MarkerDefinition(**m) for m in d.get("markers", [])],
            clusters=[RigidClusterDefinition(**c) for c in d.get("clusters", [])],
        )

    @classmethod
    def load(cls, path: str | Path) -> "MarkerKitConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def create_default_cinema_marker_kit() -> MarkerKitConfig:
    """Create Nuranova's default 38-marker + 7 asymmetric rigid-cluster hybrid cinema wearable kit.

    Designed for matte-black wearable garments under visible RGB smartphone illumination:
    - High-contrast 19mm white matte spheres/dots at bony landmarks
    - Asymmetric 4-marker rigid clusters on Head, Chest, Pelvis, L/R Forearms, L/R Feet
    - Unique AprilTag 36h11 IDs assigned to primary rigid plates for disambiguation
    """
    markers: List[MarkerDefinition] = []
    clusters: List[RigidClusterDefinition] = []

    # Asymmetric 4-point cluster template (meters, non-collinear & uniquely chiral)
    def add_cluster(cid: str, cname: str, bone: str, group: str, base_tag: int, scale: float = 1.0, offset=(0.0, 0.05, 0.0)):
        pts = [
            [0.000 * scale, 0.000 * scale, 0.000 * scale],
            [0.055 * scale, 0.000 * scale, 0.008 * scale],
            [0.012 * scale, 0.048 * scale, 0.000 * scale],
            [-0.025 * scale, 0.022 * scale, 0.032 * scale],
        ]
        m_ids = []
        for idx, pt in enumerate(pts):
            mid = f"{cid}_m{idx+1}"
            m_ids.append(mid)
            markers.append(
                MarkerDefinition(
                    marker_id=mid,
                    name=f"{cname} M{idx+1}",
                    group=group,
                    attached_bone=bone,
                    local_offset_m=[pt[0] + offset[0], pt[1] + offset[1], pt[2] + offset[2]],
                    marker_type="apriltag" if idx == 0 else "sphere_white",
                    tag_id=base_tag + idx if idx == 0 else None,
                    diameter_mm=28.0 if idx == 0 else 19.0,
                    cluster_id=cid,
                )
            )
        clusters.append(
            RigidClusterDefinition(
                cluster_id=cid,
                name=cname,
                attached_bone=bone,
                marker_ids=m_ids,
                local_points_m=pts,
                cluster_to_bone_offset_m=list(offset),
            )
        )

    add_cluster("cl_head", "Headband Cluster", "head", "Head", base_tag=0, scale=1.0, offset=(0.0, 0.09, 0.04))
    add_cluster("cl_chest", "Sternum Cluster", "chest", "Chest", base_tag=4, scale=1.2, offset=(0.0, 0.12, 0.0))
    add_cluster("cl_pelvis", "Pelvis Belt Cluster", "hips", "Pelvis", base_tag=8, scale=1.2, offset=(0.0, 0.11, 0.0))
    add_cluster("cl_l_forearm", "Left Wrist/Forearm Cluster", "left_forearm", "Forearms", base_tag=12, scale=0.75, offset=(0.0, 0.03, 0.12))
    add_cluster("cl_r_forearm", "Right Wrist/Forearm Cluster", "right_forearm", "Forearms", base_tag=16, scale=0.75, offset=(0.0, 0.03, 0.12))
    add_cluster("cl_l_foot", "Left Shoe Cluster", "left_foot", "Feet", base_tag=20, scale=0.8, offset=(0.0, 0.06, 0.03))
    add_cluster("cl_r_foot", "Right Shoe Cluster", "right_foot", "Feet", base_tag=24, scale=0.8, offset=(0.0, 0.06, 0.03))

    # Additional anatomical point markers (shoulders, elbows, hands, knees, ankles)
    singles = [
        ("m_c7", "C7 Neck", "Neck", "neck", [0.0, -0.07, 0.0]),
        ("m_l_shoulder", "Left Acromion", "Shoulders", "left_upper_arm", [0.0, 0.0, 0.04]),
        ("m_r_shoulder", "Right Acromion", "Shoulders", "right_upper_arm", [0.0, 0.0, 0.04]),
        ("m_l_elbow", "Left Lateral Elbow", "Elbows", "left_forearm", [-0.04, 0.0, 0.0]),
        ("m_r_elbow", "Right Lateral Elbow", "Elbows", "right_forearm", [0.04, 0.0, 0.0]),
        ("m_l_hand", "Left Hand Dorsum", "Hands", "left_hand", [0.0, 0.02, 0.05]),
        ("m_r_hand", "Right Hand Dorsum", "Hands", "right_hand", [0.0, 0.02, 0.05]),
        ("m_l_knee", "Left Lateral Knee", "Knees", "left_lower_leg", [-0.05, 0.0, 0.0]),
        ("m_r_knee", "Right Lateral Knee", "Knees", "right_lower_leg", [0.05, 0.0, 0.0]),
        ("m_l_ankle", "Left Lateral Malleolus", "Ankles", "left_foot", [-0.04, 0.0, 0.0]),
        ("m_r_ankle", "Right Lateral Malleolus", "Ankles", "right_foot", [0.04, 0.0, 0.0]),
    ]
    for idx, (mid, name, grp, bone, off) in enumerate(singles):
        markers.append(
            MarkerDefinition(
                marker_id=mid,
                name=name,
                group=grp,
                attached_bone=bone,
                local_offset_m=off,
                marker_type="sphere_white",
                tag_id=None,
                diameter_mm=19.0,
            )
        )

    return MarkerKitConfig(
        kit_name="Nuranova Hybrid Cinema Kit v1 (39 Markers / 7 Rigid Clusters)",
        description="Visible-light high-contrast markers + AprilTag 36h11 asymmetric clusters for matte black suit.",
        markers=markers,
        clusters=clusters,
    )


def generate_valid_marker_image(
    tag_id: int,
    dictionary_name: str = "DICT_APRILTAG_36h11",
    size_px: int = 400,
    border_bits: int = 1,
) -> np.ndarray:
    """Generate a genuine OpenCV-verified ArUco or AprilTag binary marker image."""
    if dictionary_name not in ARUCO_DICTIONARIES:
        raise ValueError(f"Unsupported dictionary: {dictionary_name}")
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[dictionary_name])
    img = cv2.aruco.generateImageMarker(aruco_dict, int(tag_id), int(size_px), borderBits=border_bits)
    return img


def generate_printable_marker_sheet_pdf(
    tag_ids: List[int],
    output_pdf: str | Path,
    dictionary_name: str = "DICT_APRILTAG_36h11",
    tag_size_mm: float = 40.0,
    quiet_zone_mm: float = 6.0,
    labels: Optional[List[str]] = None,
) -> Path:
    """Generate a printable A4 PDF sheet containing real, library-generated AprilTag/ArUco markers."""
    from PIL import Image

    out = Path(output_pdf)
    out.parent.mkdir(parents=True, exist_ok=True)
    page_w, page_h = A4
    c = canvas.Canvas(str(out), pagesize=A4)
    c.setTitle(f"Nuranova Wearable Marker Sheet ({dictionary_name})")

    margin_x = 15 * mm
    margin_top = 22 * mm
    cell_w = (tag_size_mm + 2 * quiet_zone_mm + 6.0) * mm
    cell_h = (tag_size_mm + 2 * quiet_zone_mm + 12.0) * mm
    cols = max(1, int((page_w - 2 * margin_x) // cell_w))
    rows = max(1, int((page_h - margin_top - 15 * mm) // cell_h))
    per_page = cols * rows

    for idx, tid in enumerate(tag_ids):
        slot = idx % per_page
        if slot == 0:
            if idx > 0:
                c.showPage()
            c.setFont("Helvetica-Bold", 11)
            c.drawString(
                margin_x,
                page_h - 13 * mm,
                f"Nuranova Mocap Studio — Printable Wearable Marker Sheet ({dictionary_name}, {tag_size_mm:.1f} mm)",
            )
            c.setFont("Helvetica", 8)
            c.drawString(
                margin_x,
                page_h - 18 * mm,
                "Print at 100% actual size on matte paper/cardstock. Preserve white quiet border around every marker.",
            )

        col = slot % cols
        row = slot // cols
        cx = margin_x + col * cell_w
        cy = page_h - margin_top - (row + 1) * cell_h

        marker_img = generate_valid_marker_image(tid, dictionary_name=dictionary_name, size_px=512, border_bits=1)
        # Add white quiet zone border so detectors always find outer black square
        pad = 64
        padded = cv2.copyMakeBorder(marker_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
        pil_img = Image.fromarray(padded)

        draw_size = (tag_size_mm + 2 * quiet_zone_mm) * mm
        c.setStrokeColorRGB(0.75, 0.75, 0.75)
        c.setLineWidth(0.4)
        c.rect(cx, cy + 8 * mm, draw_size, draw_size, stroke=1, fill=0)
        c.drawImage(ImageReader(pil_img), cx, cy + 8 * mm, width=draw_size, height=draw_size)

        lbl = labels[idx] if (labels and idx < len(labels)) else f"ID {tid}"
        c.setFillColorRGB(0.1, 0.1, 0.1)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(cx + 2 * mm, cy + 3 * mm, f"{lbl} ({dictionary_name}: #{tid})")

    c.showPage()
    c.save()
    return out
