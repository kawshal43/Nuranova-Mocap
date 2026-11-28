"""Camera-local review observations, stored separately from manual corrections."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np


class ReviewObservations:
    def __init__(self):
        self.cameras: dict[str, dict] = {}

    def set_track(self, uid, pts, body, left=None, right=None, left_local=None, right_local=None, **metadata):
        n = len(pts)
        if len(body) != n:
            raise ValueError("Camera observations do not match their source timestamps.")
        self.cameras[uid] = {
            "pts": np.asarray(pts, float), "body": np.asarray(body, np.float32),
            "hand:left": np.asarray(left, np.float32) if left is not None else np.zeros((n, 21, 3), np.float32),
            "hand:right": np.asarray(right, np.float32) if right is not None else np.zeros((n, 21, 3), np.float32),
            "local:left": np.asarray(left_local, np.float32) if left_local is not None else np.full((n, 21, 3), np.nan, np.float32),
            "local:right": np.asarray(right_local, np.float32) if right_local is not None else np.full((n, 21, 3), np.nan, np.float32),
            "metadata": metadata, "tags": {},
        }

    def frame(self, uid, pts_s):
        camera = self.cameras.get(uid)
        if not camera or not len(camera["pts"]):
            return None
        i = int(np.argmin(abs(camera["pts"] - pts_s)))
        times = camera["pts"]
        tolerance = .001
        if camera["metadata"].get("source") == "aligned solve observations" and len(times) > 1:
            tolerance = max(.001, float(np.median(np.diff(times))) * .55)
        if abs(times[i] - pts_s) > tolerance:
            return None
        result = {k: camera[k][i].copy() for k in ("body", "hand:left", "hand:right", "local:left", "local:right")}
        for side in ("left", "right"):
            key = "local:" + side
            result["local_source:" + side] = "model estimate"
            if not np.isfinite(result[key]).all():
                valid = np.flatnonzero(np.isfinite(camera[key]).all(axis=(1,2)))
                before, after = valid[valid < i], valid[valid > i]
                if len(before) and len(after):
                    a, b = before[-1], after[0]
                    if times[b] - times[a] <= .12:
                        alpha = (times[i] - times[a]) / (times[b] - times[a])
                        result[key] = camera[key][a] * (1-alpha) + camera[key][b] * alpha
                        result["local_source:" + side] = "interpolated estimate"
        return result

    def put_frame(self, uid, pts_s, body, left=None, right=None, tags=None, left_local=None, right_local=None):
        camera = self.cameras.get(uid)
        if camera is None:
            self.set_track(uid, [pts_s], np.asarray(body)[None],
                           np.asarray(left)[None] if left is not None else None,
                           np.asarray(right)[None] if right is not None else None,
                           np.asarray(left_local)[None] if left_local is not None else None,
                           np.asarray(right_local)[None] if right_local is not None else None)
            camera = self.cameras[uid]
        else:
            times = camera["pts"]
            matched = np.flatnonzero(abs(times - pts_s) < 1e-5)
            incoming = {"body": body, "hand:left": left, "hand:right": right,
                        "local:left": left_local, "local:right": right_local}
            if len(matched):
                for key, value in incoming.items():
                    if value is not None:
                        camera[key][int(matched[0])] = value
            else:
                camera["pts"] = np.append(times, pts_s)
                order = np.argsort(camera["pts"])
                camera["pts"] = camera["pts"][order]
                for key, value in incoming.items():
                    shape = (26, 3) if key == "body" else (21, 3)
                    value = (np.full(shape, np.nan) if key.startswith("local:") else np.zeros(shape)) if value is None else value
                    camera[key] = np.concatenate([camera[key], np.asarray(value)[None]], axis=0)[order]
        if tags is not None:
            camera["tags"][f"{pts_s:.6f}"] = tags

    def save(self, path: Path):
        arrays, metadata = {}, {}
        for uid, camera in self.cameras.items():
            # Array names are constructed here, never used as paths or executable code.
            metadata[uid] = {"metadata": camera["metadata"], "tags": camera["tags"]}
            for key in ("pts", "body", "hand:left", "hand:right", "local:left", "local:right"):
                arrays[uid + "/" + key] = camera[key]
        arrays["metadata"] = np.array(json.dumps(metadata))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path):
        result = cls()
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            for uid, fields in metadata.items():
                result.set_track(uid, data[uid + "/pts"], data[uid + "/body"],
                                 data[uid + "/hand:left"], data[uid + "/hand:right"],
                                 data[uid + "/local:left"] if uid + "/local:left" in data else None,
                                 data[uid + "/local:right"] if uid + "/local:right" in data else None,
                                 **fields.get("metadata", {}))
                result.cameras[uid]["tags"] = fields.get("tags", {})
        return result

    def merge(self, other):
        """Merge a completed range in one allocation; edited points live in ReviewState."""
        for uid, new in other.cameras.items():
            if uid not in self.cameras:
                self.cameras[uid] = new
                continue
            old = self.cameras[uid]
            all_times = np.concatenate([old["pts"], new["pts"]])
            _, indices = np.unique(np.round(all_times, 6)[::-1], return_index=True)
            indices = len(all_times) - 1 - indices
            indices = indices[np.argsort(all_times[indices])]
            for key in ("pts", "body", "hand:left", "hand:right", "local:left", "local:right"):
                old[key] = np.concatenate([old[key], new[key]], axis=0)[indices]
            old["tags"].update(new["tags"])
            for key in ("correction_pts", "correction_ranges"):
                if key in old["metadata"] or key in new["metadata"]:
                    new["metadata"][key] = old["metadata"].get(key, []) + new["metadata"].get(key, [])
            old["metadata"].update(new["metadata"])

    def from_document(self, doc, review):
        for uid, values in doc.observations_2d.items():
            if "native_pts_s" in values:
                self.set_track(uid, values["native_pts_s"], values["native_keypoints_2d"],
                               values.get("native_left_hand21"), values.get("native_right_hand21"),
                               values.get("native_left_local"), values.get("native_right_local"),
                               source="native single-camera observations")
                continue
            pts = [review.local_time(uid, t * 1e-9) for t in doc.frame_timestamps_ns]
            self.set_track(uid, pts, values["keypoints_2d"], values.get("left_hand21"),
                           values.get("right_hand21"), source="aligned solve observations")
