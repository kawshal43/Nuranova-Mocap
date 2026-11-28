"""Multi-Video Package Ingestion: .ZIP Archive, Folder, or In-App Take Loader + Media Probing.

Automatically discovers camera video clips, separates optional calibration board videos from
motion performance videos, detects embedded calibration JSON files, probes resolution/FPS/audio
using PyAV + OpenCV, and bundles recorded takes into portable `.zip` packages.
"""

from __future__ import annotations

import re
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2

from shared_protocol.path_safety import archive_member_path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
CALIB_KEYWORDS = ("calib", "calibration", "charuco", "checker", "board")


@dataclass
class CameraVideoClip:
    camera_uid: str
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    has_audio: bool
    role: str = "motion"  # "motion" or "calibration"
    manual_offset_ms: float = 0.0

    @property
    def summary_label(self) -> str:
        aud = "🔊 Audio" if self.has_audio else "🔇 Silent"
        return f"{self.width}×{self.height} @ {self.fps:.1f}fps · {self.duration_s:.1f}s ({self.frame_count}f) · {aud}"


@dataclass
class MultiVideoPackage:
    package_name: str
    source_path: Path
    extracted_dir: Optional[Path] = None
    motion_clips: Dict[str, CameraVideoClip] = field(default_factory=dict)
    calibration_clips: Dict[str, CameraVideoClip] = field(default_factory=dict)
    calibration_json_path: Optional[Path] = None
    review_path: Optional[Path] = None
    review_observations_path: Optional[Path] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def num_motion_cameras(self) -> int:
        return len(self.motion_clips)

    @property
    def num_calibration_cameras(self) -> int:
        return len(self.calibration_clips)

    def motion_paths_by_cam(self) -> Dict[str, Path]:
        return {uid: clip.path for uid, clip in self.motion_clips.items()}

    def calibration_paths_by_cam(self) -> Dict[str, Path]:
        return {uid: clip.path for uid, clip in self.calibration_clips.items()}


def _has_audio_stream_pyav(video_path: Path) -> bool:
    try:
        import av

        with av.open(str(video_path)) as container:
            return len(container.streams.audio) > 0
    except Exception:
        return False


def probe_camera_clip(
    video_path: Path, camera_uid: str, role: str = "motion"
) -> Optional[CameraVideoClip]:
    """Probe resolution, FPS, frame count, duration, and audio availability safely."""
    p = Path(video_path)
    if not p.exists() or not p.is_file() or p.stat().st_size < 1024:
        return None
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        cap.release()
        return None
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        if w <= 0 or h <= 0:
            h, w = frame.shape[:2]
        if not (5.0 <= fps <= 240.0):
            fps = 30.0
        if fc <= 1:
            # Count frames accurately if container didn't report frame count
            cnt = 1
            while cnt < 10000:
                ok_n, _ = cap.read()
                if not ok_n:
                    break
                cnt += 1
            fc = cnt
    finally:
        cap.release()

    duration_s = float(fc) / max(fps, 1.0)
    has_audio = _has_audio_stream_pyav(p)
    return CameraVideoClip(
        camera_uid=camera_uid,
        path=p.resolve(),
        width=w,
        height=h,
        fps=fps,
        frame_count=fc,
        duration_s=duration_s,
        has_audio=has_audio,
        role=role,
    )


def _infer_camera_uid(video_path: Path, fallback_index: int) -> str:
    """Infer a stable camera UID (`cam_01`, `cam_02`, ...) from filename or parent folder."""
    stem = video_path.stem.lower()
    # Match cam_01, cam01, cam-1, camera_2, phone_1, etc.
    m = re.search(r"(?:cam|camera|phone|view)[_-]?(\d+)", stem)
    if m:
        return f"cam_{int(m.group(1)):02d}"
    parent_low = video_path.parent.name.lower()
    m2 = re.search(r"(?:cam|camera|phone|view)[_-]?(\d+)", parent_low)
    if m2:
        return f"cam_{int(m2.group(1)):02d}"
    return f"cam_{fallback_index:02d}"


def _is_calibration_video(path: Path, root: Path) -> bool:
    rel_parts = [part.lower() for part in path.relative_to(root).parts]
    return any(any(kw in part for kw in CALIB_KEYWORDS) for part in rel_parts)


def _safe_extract_zip(zip_path: Path, dest_dir: Path) -> Path:
    """Safely extract a `.zip` archive with strict zip-slip path traversal protection."""
    dest_resolved = dest_dir.resolve()
    dest_resolved.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [(m, archive_member_path(dest_resolved, m.filename)) for m in zf.infolist()]
        for member, target in members:
            if member.is_dir():
                continue
            # Ignore macOS metadata folders
            if "__MACOSX" in member.filename or Path(member.filename).name.startswith("._"):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src, target.open("wb") as dst:
                import shutil

                shutil.copyfileobj(src, dst)
    return dest_resolved


def ingest_multi_video_source(
    source: str | Path | Sequence[str | Path],
    extract_root: Optional[Path] = None,
    package_name: Optional[str] = None,
) -> MultiVideoPackage:
    """Ingest synchronized multi-camera videos from a `.zip` file, a folder, or an explicit list of MP4 paths.

    Automatically separates:
    - Calibration board videos (`calib/` subfolder or filenames containing `calib` / `charuco` / `board`)
    - Motion performance videos (everything else)
    - Embedded calibration `.json` files (containing `"cameras"` key)
    """
    warnings: List[str] = []

    if isinstance(source, (list, tuple)):
        file_paths = [Path(p).resolve() for p in source if Path(p).is_file()]
        if not file_paths:
            raise ValueError("No readable video files provided.")
        pkg = MultiVideoPackage(
            package_name=package_name or file_paths[0].parent.name or "multi_video_take",
            source_path=file_paths[0].parent,
        )
        used_uids = set()
        for idx, vp in enumerate(sorted(file_paths), start=1):
            uid = _infer_camera_uid(vp, idx)
            while uid in used_uids:
                idx += 1
                uid = f"cam_{idx:02d}"
            used_uids.add(uid)
            clip = probe_camera_clip(vp, uid, role="motion")
            if clip is not None:
                pkg.motion_clips[uid] = clip
            else:
                warnings.append(f"Skipped unreadable video: {vp.name}")
        pkg.warnings = warnings
        return pkg

    src_path = Path(source).resolve()
    if not src_path.exists():
        raise FileNotFoundError(f"Source not found: {src_path}")

    extracted_dir: Optional[Path] = None
    scan_root = src_path
    name = package_name or src_path.stem

    if src_path.is_file():
        if src_path.suffix.lower() == ".zip":
            base_tmp = (
                Path(extract_root)
                if extract_root is not None
                else Path(tempfile.mkdtemp(prefix="nuranova_mv_zip_"))
            )
            extracted_dir = _safe_extract_zip(src_path, base_tmp / src_path.stem)
            scan_root = extracted_dir
        elif src_path.suffix.lower() in VIDEO_EXTENSIONS:
            return ingest_multi_video_source([src_path], package_name=name)
        else:
            raise ValueError(
                f"Unsupported file type '{src_path.suffix}'. Provide a .zip archive, folder, or video file."
            )

    all_videos = sorted(
        p
        for p in scan_root.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS and not p.name.startswith("._")
    )
    if not all_videos:
        raise ValueError(
            f"No video files ({', '.join(sorted(VIDEO_EXTENSIONS))}) found in '{src_path.name}'."
        )

    # Detect embedded calibration JSON profile if packaged inside the folder/zip
    calib_json: Optional[Path] = None
    for jp in sorted(scan_root.rglob("*.json")):
        if any(kw in jp.name.lower() for kw in ("calib", "camera")):
            try:
                txt = jp.read_text(encoding="utf-8", errors="ignore")
                if '"cameras"' in txt and '"camera_matrix"' in txt:
                    calib_json = jp.resolve()
                    break
            except Exception:
                continue

    calib_files: List[Path] = []
    motion_files: List[Path] = []
    for vp in all_videos:
        if _is_calibration_video(vp, scan_root):
            calib_files.append(vp)
        else:
            motion_files.append(vp)

    # If ALL videos happened to live in a folder named e.g. "calib_test", treat them as motion if none left
    if not motion_files and calib_files:
        motion_files = calib_files
        calib_files = []

    pkg = MultiVideoPackage(
        package_name=name,
        source_path=src_path,
        extracted_dir=extracted_dir,
        calibration_json_path=calib_json,
        review_path=next(iter(sorted(scan_root.rglob("review.json"))), None),
        review_observations_path=next(iter(sorted(scan_root.rglob("review_observations.npz"))), None),
    )

    def _register_group(paths: List[Path], target: Dict[str, CameraVideoClip], role: str) -> None:
        used = set()
        next_idx = 1
        for vp in paths:
            uid = _infer_camera_uid(vp, next_idx)
            while uid in used:
                next_idx += 1
                uid = f"cam_{next_idx:02d}"
            used.add(uid)
            next_idx += 1
            clip = probe_camera_clip(vp, uid, role=role)
            if clip is not None:
                target[uid] = clip
            else:
                warnings.append(f"Skipped unreadable {role} video: {vp.name}")

    _register_group(motion_files, pkg.motion_clips, role="motion")
    _register_group(calib_files, pkg.calibration_clips, role="calibration")
    pkg.warnings = warnings
    return pkg


def export_take_to_zip(
    motion_videos: Dict[str, Path],
    output_zip: str | Path,
    calibration_videos: Optional[Dict[str, Path]] = None,
    calibration_json: Optional[Path] = None,
    review_json: Optional[Path] = None,
    review_observations: Optional[Path] = None,
) -> Path:
    """Bundle multi-camera motion videos (+ optional calibration videos/JSON) into a portable `.zip` archive."""
    out = Path(output_zip).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as zf:
        for uid, vp in sorted(motion_videos.items()):
            p = Path(vp)
            if p.exists():
                zf.write(
                    p,
                    arcname=f"motion/{uid}_{p.name}"
                    if not p.name.lower().startswith(uid.lower())
                    else f"motion/{p.name}",
                )
        if calibration_videos:
            for uid, vp in sorted(calibration_videos.items()):
                p = Path(vp)
                if p.exists():
                    zf.write(
                        p,
                        arcname=f"calib/{uid}_{p.name}"
                        if not p.name.lower().startswith(uid.lower())
                        else f"calib/{p.name}",
                    )
        if calibration_json is not None and Path(calibration_json).exists():
            zf.write(calibration_json, arcname="calibration.json")
        if review_json is not None and Path(review_json).exists():
            zf.write(review_json, arcname="review.json")
        if review_observations is not None and Path(review_observations).exists():
            zf.write(review_observations, arcname="review_observations.npz")
    temporary.replace(out)
    return out
