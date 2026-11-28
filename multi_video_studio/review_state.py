"""Versioned, non-destructive camera/point review state in original video coordinates."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np


BODY_POINT_NAMES = [
    "Nose", "Left eye", "Right eye", "Left ear", "Right ear",
    "Left shoulder", "Right shoulder", "Left elbow", "Right elbow",
    "Left wrist", "Right wrist", "Left hip", "Right hip", "Left knee", "Right knee",
    "Left ankle", "Right ankle", "Head", "Neck", "Pelvis", "Left big toe",
    "Right big toe", "Left small toe", "Right small toe", "Left heel", "Right heel",
]
HAND_POINT_NAMES = ["Wrist"] + [
    f"{finger} {joint}"
    for finger, joints in [
        ("thumb", ("base", "MCP", "IP", "tip")),
        ("index", ("MCP", "PIP", "DIP", "tip")),
        ("middle", ("MCP", "PIP", "DIP", "tip")),
        ("ring", ("MCP", "PIP", "DIP", "tip")),
        ("pinky", ("MCP", "PIP", "DIP", "tip")),
    ]
    for joint in joints
]


def source_fingerprint(path: Path) -> str:
    """A portable content identity; independent of filename, directory and import order."""
    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode())
    with path.open("rb") as stream:
        for offset in sorted({0, max(0, size // 2 - 32768), max(0, size - 65536)}):
            stream.seek(offset)
            digest.update(stream.read(65536))
    return digest.hexdigest()


def point_name(key: str, assignments: dict | None = None) -> str:
    parts = key.split(":")
    try:
        if parts[0] == "body":
            return BODY_POINT_NAMES[int(parts[1])]
        if parts[0] == "hand":
            return parts[1].capitalize() + " " + HAND_POINT_NAMES[int(parts[2])].lower()
        if parts[0] == "tag":
            record = (assignments or {}).get(parts[1], {})
            return record.get("name") or f"Unassigned tag {parts[1]}"
    except (IndexError, ValueError):
        pass
    return key


@dataclass
class ReviewState:
    sources: dict = field(default_factory=dict)
    capture_mode: str = "markerless"
    clocks: dict = field(default_factory=dict)
    manual_offsets_ms: dict = field(default_factory=dict)
    alignment_accepted: bool = False
    reference_camera: str | None = None
    camera_policy: dict = field(default_factory=dict)
    points: list = field(default_factory=list)
    exclusions: list = field(default_factory=list)
    marker_assignments: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    contact_pins: list = field(default_factory=list)
    hand_rois: list = field(default_factory=list)
    processing_settings: dict = field(default_factory=dict)
    revision: int = 0
    solved_revision: int = -1
    result_path: str | None = None
    _undo: list = field(default_factory=list, repr=False)
    _redo: list = field(default_factory=list, repr=False)

    def to_dict(self, include_history=False) -> dict:
        result = {name: deepcopy(getattr(self, name)) for name in self.__dataclass_fields__
                  if not name.startswith("_")}
        result["schema_version"] = 1
        if include_history:
            result["history"] = {"undo": self._undo[-40:], "redo": self._redo[-40:]}
        return result

    @classmethod
    def from_dict(cls, value: dict) -> "ReviewState":
        if value.get("schema_version", 1) != 1:
            raise ValueError("Unsupported take review version.")
        result = cls(**{k: deepcopy(v) for k, v in value.items()
                        if k in cls.__dataclass_fields__ and not k.startswith("_")})
        if result.capture_mode not in {"markerless", "hybrid", "markers"}:
            raise ValueError("Invalid capture mode.")
        history = value.get("history", {})
        result._undo = deepcopy(history.get("undo", [])[-40:])
        result._redo = deepcopy(history.get("redo", [])[-40:])
        return result

    @classmethod
    def load(cls, path: Path) -> "ReviewState":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8-sig")))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(True), indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)

    def checkpoint(self) -> None:
        self._undo.append(self.to_dict())
        self._undo = self._undo[-40:]
        self._redo.clear()
        self.revision += 1

    def _restore(self, snapshot: dict) -> None:
        previous = self.revision
        restored = ReviewState.from_dict(snapshot)
        for name in self.__dataclass_fields__:
            if not name.startswith("_"):
                setattr(self, name, getattr(restored, name))
        # Undo is a new revision, not permission to revive a stale derived solve.
        self.revision = previous + 1
        self.solved_revision = -1

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self.to_dict())
        self._restore(self._undo.pop())
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self.to_dict())
        self._restore(self._redo.pop())
        return True

    @property
    def stale(self) -> bool:
        return self.revision != self.solved_revision

    def bind_sources(self, clips: dict) -> dict:
        """Remap a portable review by content identity, retaining camera-specific edits."""
        original_assets = sorted(v["asset_id"] for v in self.sources.values())
        initially_empty = not self.sources
        old_ids = {}
        for key, value in self.sources.items():
            old_ids.setdefault(value["asset_id"], []).append(key)
        remap, sources = {}, {}
        for uid, clip in clips.items():
            asset_id = source_fingerprint(clip.path)
            matches = old_ids.get(asset_id, [])
            # Match content before camera slots. A replacement in the same slot
            # must not inherit the preceding recording's points or calibration.
            available = [key for key in matches if key not in remap]
            old_uid = uid if uid in available else available[0] if available else uid if initially_empty else None
            if old_uid is not None:
                remap[old_uid] = uid
            sources[uid] = {"asset_id": asset_id, "filename": clip.path.name,
                            "width": clip.width, "height": clip.height}
        if remap or self.sources:
            for name in ("clocks", "camera_policy", "manual_offsets_ms"):
                original = getattr(self, name)
                setattr(self, name, {remap[k]: v for k, v in original.items() if k in remap})
            for uid, clock in self.clocks.items():
                clock["camera_uid"] = uid
            for name in ("points", "exclusions", "hand_rois"):
                setattr(self, name, [dict(v, camera_uid=remap[v["camera_uid"]])
                                     for v in getattr(self, name) if v["camera_uid"] in remap])
            self.reference_camera = remap.get(self.reference_camera, next(iter(clips), None))
            # Portable history and calibration must follow the same source identities.
            for snapshot in self._undo + self._redo:
                state = ReviewState.from_dict(snapshot)
                state.bind_sources(clips)
                snapshot.clear()
                snapshot.update(state.to_dict())
            if self.calibration.get("cameras"):
                cameras = self.calibration["cameras"]
                if isinstance(cameras, dict):
                    renamed = {}
                    for old_uid, camera in cameras.items():
                        if old_uid not in remap:
                            continue
                        uid = remap[old_uid]
                        for component in (camera, camera.get("intrinsics", {}), camera.get("extrinsics", {})):
                            if "camera_uid" in component:
                                component["camera_uid"] = uid
                        renamed[uid] = camera
                    self.calibration["cameras"] = renamed
        if not initially_empty and original_assets != sorted(v["asset_id"] for v in sources.values()):
            self.revision += 1
            self.solved_revision = -1
            self.result_path = None
            self.alignment_accepted = False
            self.contact_pins = []
        self.sources = sources
        return remap

    def set_clocks(self, values: dict, accepted=True) -> None:
        self.checkpoint()
        self.clocks = {uid: deepcopy(v.__dict__ if hasattr(v, "__dict__") else v)
                       for uid, v in values.items()}
        self.alignment_accepted = bool(accepted and self.clocks)
        self.reference_camera = next((u for u, v in self.clocks.items()
                                      if v.get("method") == "reference"), self.reference_camera)

    def global_time(self, uid: str, local_pts: float) -> float:
        clock = self.clocks.get(uid, {})
        origin = clock.get("base_pts_s")
        if origin is None:
            origin = clock.get("audio_start_s", 0)
        local = local_pts - float(origin)
        slope = float(clock.get("drift_slope", 1))
        return local + float(clock.get("offset_s", 0)) + (slope - 1) * (local - float(clock.get("drift_pivot_s", 0)))

    def local_time(self, uid: str, global_time: float) -> float:
        clock = self.clocks.get(uid, {})
        slope = float(clock.get("drift_slope", 1))
        if not np.isfinite(slope) or slope <= 0:
            raise ValueError("Camera clock slope must be positive.")
        origin = clock.get("base_pts_s")
        if origin is None:
            origin = clock.get("audio_start_s", 0)
        return (global_time - float(clock.get("offset_s", 0)) + (slope - 1) * float(clock.get("drift_pivot_s", 0))) / slope + float(origin)

    def edit_point(self, uid: str, pts_s: float, key: str, xy=None, valid=True, frame_index=None) -> None:
        if uid not in self.sources or not np.isfinite(pts_s):
            raise ValueError("Select a source camera and a valid frame first.")
        if xy is not None:
            xy = [float(v) for v in xy]
            if len(xy) != 2 or not np.all(np.isfinite(xy)):
                raise ValueError("Point coordinates must be finite image pixels.")
            source = self.sources[uid]
            if not (0 <= xy[0] < source["width"] and 0 <= xy[1] < source["height"]):
                raise ValueError("Point lies outside the source image.")
        self.checkpoint()
        self.points = [p for p in self.points if not (
            p["camera_uid"] == uid and p["key"] == key and abs(p["pts_s"] - pts_s) < 1e-5)]
        self.points.append({"camera_uid": uid, "asset_id": self.sources[uid]["asset_id"],
                            "pts_s": float(pts_s), "frame_index": frame_index,
                            "key": key, "xy": xy, "valid": bool(valid), "source": "manual"})

    def reset_point(self, uid: str, pts_s: float, key: str) -> None:
        self.checkpoint()
        self.points = [p for p in self.points if not (
            p["camera_uid"] == uid and p["key"] == key and abs(p["pts_s"] - pts_s) < 1e-5)]

    def excluded(self, uid: str, pts_s: float, channel: str) -> bool:
        policy = self.camera_policy.get(uid, {})
        if not policy.get("enabled", True) or not policy.get(channel, True):
            return True
        return any(p["camera_uid"] == uid and p["channel"] in (channel, "all")
                   and p["start_s"] <= pts_s <= p["end_s"] for p in self.exclusions)

    def apply_points(self, uid: str, pts, values: np.ndarray, channel="body") -> tuple[np.ndarray, np.ndarray]:
        """Apply annotations to copies; return a pin mask so later filters respect edits."""
        out = np.asarray(values).copy()
        pins = np.zeros(out.shape[:2], dtype=bool)
        times = np.asarray(pts, float)
        if len(times) != len(out):
            raise ValueError("Observation timestamps do not match the track.")
        if not len(times):
            return out, pins
        tolerance = max(.005, float(np.median(np.diff(times))) * .55) if len(times) > 1 else .02
        prefix = channel + ":"
        for edit in self.points:
            if edit["camera_uid"] != uid or not edit["key"].startswith(prefix):
                continue
            try:
                k = int(edit["key"][len(prefix):])
            except ValueError:
                continue
            t = int(np.argmin(abs(times - edit["pts_s"])))
            if abs(times[t] - edit["pts_s"]) > tolerance or not 0 <= k < out.shape[1]:
                continue
            if edit["valid"] and edit["xy"] is not None:
                out[t, k, :2] = edit["xy"]
                out[t, k, 2] = 1.0  # Explicit manual constraint; provenance remains separate.
            else:
                out[t, k, 2] = 0.0
            pins[t, k] = True
        return out, pins

    def apply_exclusions(self, uid: str, pts, values: np.ndarray, channel="body") -> np.ndarray:
        out = np.asarray(values).copy()
        for i, time_s in enumerate(pts):
            if self.excluded(uid, float(time_s), channel):
                out[i, :, 2] = 0
        return out
