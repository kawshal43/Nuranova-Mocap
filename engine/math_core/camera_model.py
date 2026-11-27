"""Pinhole + Brown-Conrady distortion camera models and multi-camera calibration containers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class CameraIntrinsics:
    camera_uid: str
    image_width: int
    image_height: int
    camera_matrix: np.ndarray  # (3, 3) float64
    dist_coeffs: np.ndarray  # (5,) or (8,) float64 [k1, k2, p1, p2, k3, ...]
    reprojection_error_px: float = 0.0
    focus_distance_diopters: Optional[float] = None
    stabilization_disabled: bool = True
    calibrated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.camera_matrix[1, 2])

    def undistort_points(self, pts_2d: np.ndarray) -> np.ndarray:
        """Return undistorted pixel coordinates using the original camera matrix K."""
        pts = np.asarray(pts_2d, dtype=np.float64).reshape(-1, 1, 2)
        if pts.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float64)
        undist = cv2.undistortPoints(
            pts,
            self.camera_matrix,
            self.dist_coeffs,
            P=self.camera_matrix,
        )
        return undist.reshape(-1, 2)

    def undistort_to_normalized(self, pts_2d: np.ndarray) -> np.ndarray:
        """Return undistorted normalized image coordinates [x_n, y_n] on z=1 plane."""
        pts = np.asarray(pts_2d, dtype=np.float64).reshape(-1, 1, 2)
        if pts.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float64)
        undist = cv2.undistortPoints(pts, self.camera_matrix, self.dist_coeffs)
        return undist.reshape(-1, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_uid": self.camera_uid,
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
            "camera_matrix": np.asarray(self.camera_matrix, dtype=np.float64).tolist(),
            "dist_coeffs": np.asarray(self.dist_coeffs, dtype=np.float64).reshape(-1).tolist(),
            "reprojection_error_px": float(self.reprojection_error_px),
            "focus_distance_diopters": self.focus_distance_diopters,
            "stabilization_disabled": bool(self.stabilization_disabled),
            "calibrated_at": self.calibrated_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CameraIntrinsics":
        return cls(
            camera_uid=str(d["camera_uid"]),
            image_width=int(d["image_width"]),
            image_height=int(d["image_height"]),
            camera_matrix=np.asarray(d["camera_matrix"], dtype=np.float64).reshape(3, 3),
            dist_coeffs=np.asarray(d["dist_coeffs"], dtype=np.float64).reshape(-1),
            reprojection_error_px=float(d.get("reprojection_error_px", 0.0)),
            focus_distance_diopters=d.get("focus_distance_diopters"),
            stabilization_disabled=bool(d.get("stabilization_disabled", True)),
            calibrated_at=str(d.get("calibrated_at", "")),
        )


@dataclass
class CameraExtrinsics:
    """World-to-camera rigid transform: X_cam = R_cw @ X_world + t_cw (meters)."""

    camera_uid: str
    R_cw: np.ndarray  # (3, 3)
    t_cw: np.ndarray  # (3,)
    reprojection_error_px: float = 0.0

    @property
    def rvec(self) -> np.ndarray:
        rv, _ = cv2.Rodrigues(np.asarray(self.R_cw, dtype=np.float64))
        return rv.reshape(3)

    @property
    def camera_center_world(self) -> np.ndarray:
        """Optical center position in world coordinates (meters)."""
        return -(self.R_cw.T @ self.t_cw.reshape(3))

    @property
    def world_pose_matrix(self) -> np.ndarray:
        """Camera-to-world 4x4 homogeneous transform."""
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = self.R_cw.T
        M[:3, 3] = self.camera_center_world
        return M

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_uid": self.camera_uid,
            "R_cw": np.asarray(self.R_cw, dtype=np.float64).tolist(),
            "t_cw": np.asarray(self.t_cw, dtype=np.float64).reshape(3).tolist(),
            "reprojection_error_px": float(self.reprojection_error_px),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CameraExtrinsics":
        return cls(
            camera_uid=str(d["camera_uid"]),
            R_cw=np.asarray(d["R_cw"], dtype=np.float64).reshape(3, 3),
            t_cw=np.asarray(d["t_cw"], dtype=np.float64).reshape(3),
            reprojection_error_px=float(d.get("reprojection_error_px", 0.0)),
        )


@dataclass
class CalibratedCamera:
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics

    @property
    def camera_uid(self) -> str:
        return self.intrinsics.camera_uid

    @property
    def projection_matrix(self) -> np.ndarray:
        """3x4 pinhole projection matrix P = K [R_cw | t_cw] (for undistorted points)."""
        Rt = np.hstack([self.extrinsics.R_cw, self.extrinsics.t_cw.reshape(3, 1)])
        return self.intrinsics.camera_matrix @ Rt

    def undistort_points(self, pts_2d: np.ndarray) -> np.ndarray:
        """Delegate undistortion to camera intrinsics (returns undistorted pixel coordinates)."""
        return self.intrinsics.undistort_points(pts_2d)

    def project_world_points(self, pts_world: np.ndarray, apply_distortion: bool = True) -> np.ndarray:
        """Project (N, 3) world points into (N, 2) image pixels."""
        pts = np.asarray(pts_world, dtype=np.float64).reshape(-1, 3)
        if pts.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float64)
        dist = self.intrinsics.dist_coeffs if apply_distortion else np.zeros(5, dtype=np.float64)
        rvec, _ = cv2.Rodrigues(self.extrinsics.R_cw)
        proj, _ = cv2.projectPoints(
            pts,
            rvec,
            self.extrinsics.t_cw.reshape(3, 1),
            self.intrinsics.camera_matrix,
            dist,
        )
        return proj.reshape(-1, 2)

    def is_in_front(self, pts_world: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_world, dtype=np.float64).reshape(-1, 3)
        cam_pts = (pts @ self.extrinsics.R_cw.T) + self.extrinsics.t_cw.reshape(1, 3)
        return cam_pts[:, 2] > 1e-4

    def pixel_to_world_ray(self, pixel_uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (origin_world, unit_direction_world) for a distorted image pixel [u, v]."""
        norm_xy = self.intrinsics.undistort_to_normalized(np.asarray(pixel_uv, dtype=np.float64).reshape(1, 2))[0]
        d_cam = np.array([norm_xy[0], norm_xy[1], 1.0], dtype=np.float64)
        d_cam /= np.linalg.norm(d_cam)
        d_world = self.extrinsics.R_cw.T @ d_cam
        d_world /= np.linalg.norm(d_world)
        return self.extrinsics.camera_center_world.copy(), d_world

    def check_config_validity(
        self,
        width: int,
        height: int,
        stabilization_disabled: bool = True,
        focus_distance_diopters: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Detect camera parameter changes that invalidate this calibration."""
        if width != self.intrinsics.image_width or height != self.intrinsics.image_height:
            return (
                False,
                f"Resolution mismatch: calibrated at {self.intrinsics.image_width}x{self.intrinsics.image_height}, "
                f"current is {width}x{height}",
            )
        if self.intrinsics.stabilization_disabled and not stabilization_disabled:
            return False, "Digital video stabilization was enabled after calibration (alters effective focal length/crop)"
        if (
            self.intrinsics.focus_distance_diopters is not None
            and focus_distance_diopters is not None
            and abs(self.intrinsics.focus_distance_diopters - focus_distance_diopters) > 0.25
        ):
            return False, "Focus distance changed significantly since calibration"
        return True, None


def compute_fundamental_matrix(cam_a: CalibratedCamera, cam_b: CalibratedCamera) -> np.ndarray:
    """Compute fundamental matrix F such that x_b^T F x_a = 0 for undistorted pixels."""
    R_rel = cam_b.extrinsics.R_cw @ cam_a.extrinsics.R_cw.T
    t_rel = cam_b.extrinsics.t_cw.reshape(3) - R_rel @ cam_a.extrinsics.t_cw.reshape(3)
    tx = np.array(
        [
            [0.0, -t_rel[2], t_rel[1]],
            [t_rel[2], 0.0, -t_rel[0]],
            [-t_rel[1], t_rel[0], 0.0],
        ],
        dtype=np.float64,
    )
    E = tx @ R_rel
    Ka_inv = np.linalg.inv(cam_a.intrinsics.camera_matrix)
    Kb_inv = np.linalg.inv(cam_b.intrinsics.camera_matrix)
    F = Kb_inv.T @ E @ Ka_inv
    norm = np.linalg.norm(F)
    return F / norm if norm > 1e-12 else F


def epipolar_line_distance_px(
    cam_a: CalibratedCamera,
    pt_a_distorted: np.ndarray,
    cam_b: CalibratedCamera,
    pt_b_distorted: np.ndarray,
) -> float:
    """Symmetric epipolar distance in pixels between two camera observations."""
    ua = cam_a.intrinsics.undistort_points(np.asarray(pt_a_distorted).reshape(1, 2))[0]
    ub = cam_b.intrinsics.undistort_points(np.asarray(pt_b_distorted).reshape(1, 2))[0]
    F = compute_fundamental_matrix(cam_a, cam_b)
    xa = np.array([ua[0], ua[1], 1.0])
    xb = np.array([ub[0], ub[1], 1.0])
    lb = F @ xa
    la = F.T @ xb
    num = abs(float(xb @ F @ xa))
    d_b = num / max(float(np.hypot(lb[0], lb[1])), 1e-12)
    d_a = num / max(float(np.hypot(la[0], la[1])), 1e-12)
    return 0.5 * (d_a + d_b)


@dataclass
class MultiCameraCalibration:
    profile_name: str
    cameras: Dict[str, CalibratedCamera] = field(default_factory=dict)
    world_up_axis: str = "Z"
    units: str = "meters"
    global_reprojection_rmse_px: float = 0.0
    known_distance_measured_m: Optional[float] = None
    known_distance_expected_m: Optional[float] = None
    known_distance_error_mm: Optional[float] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "world_up_axis": self.world_up_axis,
            "units": self.units,
            "global_reprojection_rmse_px": float(self.global_reprojection_rmse_px),
            "known_distance_measured_m": self.known_distance_measured_m,
            "known_distance_expected_m": self.known_distance_expected_m,
            "known_distance_error_mm": self.known_distance_error_mm,
            "created_at": self.created_at,
            "notes": self.notes,
            "cameras": {
                uid: {
                    "intrinsics": cam.intrinsics.to_dict(),
                    "extrinsics": cam.extrinsics.to_dict(),
                }
                for uid, cam in self.cameras.items()
            },
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MultiCameraCalibration":
        cams = {
            uid: CalibratedCamera(
                intrinsics=CameraIntrinsics.from_dict(c["intrinsics"]),
                extrinsics=CameraExtrinsics.from_dict(c["extrinsics"]),
            )
            for uid, c in d.get("cameras", {}).items()
        }
        return cls(
            profile_name=str(d.get("profile_name", "Unnamed")),
            cameras=cams,
            world_up_axis=str(d.get("world_up_axis", "Z")),
            units=str(d.get("units", "meters")),
            global_reprojection_rmse_px=float(d.get("global_reprojection_rmse_px", 0.0)),
            known_distance_measured_m=d.get("known_distance_measured_m"),
            known_distance_expected_m=d.get("known_distance_expected_m"),
            known_distance_error_mm=d.get("known_distance_error_mm"),
            created_at=str(d.get("created_at", "")),
            notes=str(d.get("notes", "")),
        )

    @classmethod
    def load(cls, path: str | Path) -> "MultiCameraCalibration":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
