"""Portable Project Lifecycle Management with SQLite metadata index and standard folder layout."""

from __future__ import annotations

import json
import shutil
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

PROJECT_SUBDIRECTORIES: List[str] = [
    "calibration",
    "actors",
    "recordings",
    "timestamps",
    "tracking",
    "reconstructed",
    "raw_motion",
    "corrected_motion",
    "facial_motion",
    "hand_motion",
    "character_mappings",
    "exports",
    "backups",
    "logs",
]


class Base(DeclarativeBase):
    pass


class TakeRecordORM(Base):
    __tablename__ = "takes"
    id = Column(String(64), primary_key=True)
    session_id = Column(String(64), nullable=False)
    name = Column(String(200), nullable=False)
    capture_mode = Column(String(32), default="hybrid")
    fps = Column(Float, default=60.0)
    num_frames = Column(Integer, default=0)
    status = Column(String(32), default="recorded")
    motion_h5_relpath = Column(String(500), nullable=True)
    blender_json_relpath = Column(String(500), nullable=True)
    metadata_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class CameraRecordORM(Base):
    __tablename__ = "cameras"
    camera_uid = Column(String(64), primary_key=True)
    device_id = Column(String(128), unique=True, nullable=False)
    name = Column(String(128), nullable=False)
    group_name = Column(String(128), default="Main Volume")
    role = Column(String(32), default="body")
    config_json = Column(Text, default="{}")


@dataclass
class MocapProject:
    project_id: str
    name: str
    root_dir: Path
    schema_version: str = "1.0.0"
    created_at: str = ""
    modified_at: str = ""
    preferences: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.preferences is None:
            self.preferences = {}

    @property
    def manifest_path(self) -> Path:
        return self.root_dir / "project.nmocap"

    @property
    def db_path(self) -> Path:
        return self.root_dir / "project.db"

    def subdir(self, name: str) -> Path:
        p = self.root_dir / name
        p.mkdir(parents=True, exist_ok=True)
        return p


class ProjectManager:
    """Manages portable `.nmocap` projects without storing large video blobs inside SQLite."""

    def __init__(self, recent_registry_path: Optional[Path] = None) -> None:
        self.recent_registry_path = recent_registry_path or (Path.home() / ".nuranova_mocap" / "recent_projects.json")
        self.recent_registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.active_project: Optional[MocapProject] = None
        self._engine = None
        self._SessionLocal = None

    def _init_db(self, db_path: Path) -> None:
        if self._engine is not None:
            self._engine.dispose()
        self._engine = create_engine(f"sqlite:///{db_path.as_posix()}", future=True)
        Base.metadata.create_all(self._engine)
        self._SessionLocal = sessionmaker(bind=self._engine, expire_on_commit=False)

    def close_project(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._SessionLocal = None
        self.active_project = None

    def create_project(self, parent_dir: str | Path, name: str, preferences: Optional[Dict[str, Any]] = None) -> MocapProject:
        safe = "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in name).strip()
        if not safe:
            raise ValueError("Project name cannot be empty.")
        root = Path(parent_dir) / safe
        if root.exists() and (root / "project.nmocap").exists():
            raise FileExistsError(f"Project already exists at {root}")

        root.mkdir(parents=True, exist_ok=True)
        for sub in PROJECT_SUBDIRECTORIES:
            (root / sub).mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc).isoformat()
        proj = MocapProject(
            project_id=uuid.uuid4().hex,
            name=name,
            root_dir=root,
            created_at=now,
            modified_at=now,
            preferences=preferences or {"default_fps": 60.0, "capture_mode": "hybrid", "profile": "cinema"},
        )
        self._write_manifest(proj)
        self._init_db(proj.db_path)
        self.active_project = proj
        self._record_recent(proj)
        return proj

    def open_project(self, project_dir_or_manifest: str | Path) -> MocapProject:
        p = Path(project_dir_or_manifest)
        root = p.parent if p.is_file() else p
        manifest = root / "project.nmocap"
        if not manifest.exists():
            raise FileNotFoundError(f"Missing project.nmocap manifest in {root}")
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for sub in PROJECT_SUBDIRECTORIES:
            (root / sub).mkdir(parents=True, exist_ok=True)
        proj = MocapProject(
            project_id=str(data["project_id"]),
            name=str(data["name"]),
            root_dir=root,
            schema_version=str(data.get("schema_version", "1.0.0")),
            created_at=str(data.get("created_at", "")),
            modified_at=str(data.get("modified_at", "")),
            preferences=dict(data.get("preferences", {})),
        )
        self._init_db(proj.db_path)
        self.active_project = proj
        self._record_recent(proj)
        return proj

    def save_project(self) -> None:
        if self.active_project is None:
            return
        self.active_project.modified_at = datetime.now(timezone.utc).isoformat()
        self._write_manifest(self.active_project)

    def rename_project(self, new_name: str) -> MocapProject:
        if self.active_project is None:
            raise RuntimeError("No active project.")
        self.active_project.name = new_name
        self.save_project()
        self._record_recent(self.active_project)
        return self.active_project

    def duplicate_project(self, dest_parent_dir: str | Path, new_name: str) -> MocapProject:
        if self.active_project is None:
            raise RuntimeError("No active project.")
        src = self.active_project.root_dir
        safe = "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in new_name).strip()
        dst = Path(dest_parent_dir) / safe
        self.close_project()
        shutil.copytree(src, dst)
        dup = self.open_project(dst)
        dup.project_id = uuid.uuid4().hex
        dup.name = new_name
        self.save_project()
        return dup

    def backup_or_export_project_zip(self, output_zip: Optional[str | Path] = None) -> Path:
        if self.active_project is None:
            raise RuntimeError("No active project.")
        self.save_project()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out = Path(output_zip) if output_zip else (self.active_project.subdir("backups") / f"{self.active_project.name}_{stamp}.nmocap.zip")
        out.parent.mkdir(parents=True, exist_ok=True)
        root = self.active_project.root_dir
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for item in root.rglob("*"):
                if item.is_file() and "backups" not in item.relative_to(root).parts:
                    zf.write(item, arcname=str(item.relative_to(root)))
        return out

    def import_or_restore_project_zip(self, zip_path: str | Path, dest_dir: str | Path) -> MocapProject:
        zp = Path(zip_path)
        dst = Path(dest_dir)
        dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zp, "r") as zf:
            for member in zf.namelist():
                target = (dst / member).resolve()
                if not str(target).startswith(str(dst.resolve())):
                    raise ValueError(f"Unsafe zip member blocked: {member}")
            zf.extractall(dst)
        return self.open_project(dst)

    def delete_project(self, project_root: str | Path, confirmed: bool = False) -> None:
        if not confirmed:
            raise PermissionError("Explicit confirmation is required to delete a project.")
        p = Path(project_root).resolve()
        if self.active_project and self.active_project.root_dir.resolve() == p:
            self.close_project()
        if p.exists() and (p / "project.nmocap").exists():
            shutil.rmtree(p)

    def register_take(
        self,
        take_id: str,
        session_id: str,
        name: str,
        capture_mode: str = "hybrid",
        fps: float = 60.0,
        num_frames: int = 0,
        status: str = "processed",
        motion_h5_path: Optional[Path] = None,
        blender_json_path: Optional[Path] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._SessionLocal is None or self.active_project is None:
            raise RuntimeError("No active project open.")
        root = self.active_project.root_dir
        rel_h5 = str(motion_h5_path.relative_to(root)) if (motion_h5_path and motion_h5_path.is_relative_to(root)) else (str(motion_h5_path) if motion_h5_path else None)
        rel_bj = str(blender_json_path.relative_to(root)) if (blender_json_path and blender_json_path.is_relative_to(root)) else (str(blender_json_path) if blender_json_path else None)
        with self._SessionLocal() as db:
            existing = db.get(TakeRecordORM, take_id)
            if existing is None:
                db.add(
                    TakeRecordORM(
                        id=take_id,
                        session_id=session_id,
                        name=name,
                        capture_mode=capture_mode,
                        fps=fps,
                        num_frames=num_frames,
                        status=status,
                        motion_h5_relpath=rel_h5,
                        blender_json_relpath=rel_bj,
                        metadata_json=json.dumps(metadata or {}),
                    )
                )
            else:
                existing.name = name
                existing.num_frames = num_frames
                existing.status = status
                existing.motion_h5_relpath = rel_h5
                existing.blender_json_relpath = rel_bj
                existing.metadata_json = json.dumps(metadata or {})
            db.commit()

    def list_takes(self) -> List[Dict[str, Any]]:
        if self._SessionLocal is None or self.active_project is None:
            return []
        root = self.active_project.root_dir
        with self._SessionLocal() as db:
            rows = db.query(TakeRecordORM).order_by(TakeRecordORM.created_at.desc()).all()
            return [
                {
                    "take_id": r.id,
                    "session_id": r.session_id,
                    "name": r.name,
                    "capture_mode": r.capture_mode,
                    "fps": r.fps,
                    "num_frames": r.num_frames,
                    "status": r.status,
                    "motion_h5_path": str(root / r.motion_h5_relpath) if r.motion_h5_relpath else None,
                    "blender_json_path": str(root / r.blender_json_relpath) if r.blender_json_relpath else None,
                    "metadata": json.loads(r.metadata_json or "{}"),
                }
                for r in rows
            ]

    def list_recent_projects(self) -> List[Dict[str, str]]:
        if not self.recent_registry_path.exists():
            return []
        try:
            items = json.loads(self.recent_registry_path.read_text(encoding="utf-8"))
            return [it for it in items if Path(it["path"]).exists()]
        except Exception:
            return []

    def _record_recent(self, proj: MocapProject) -> None:
        items = [it for it in self.list_recent_projects() if it.get("path") != str(proj.root_dir)]
        items.insert(0, {"project_id": proj.project_id, "name": proj.name, "path": str(proj.root_dir)})
        self.recent_registry_path.write_text(json.dumps(items[:20], indent=2), encoding="utf-8")

    @staticmethod
    def _write_manifest(proj: MocapProject) -> None:
        payload = {
            "project_id": proj.project_id,
            "name": proj.name,
            "schema_version": proj.schema_version,
            "created_at": proj.created_at,
            "modified_at": proj.modified_at,
            "preferences": proj.preferences,
        }
        proj.manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
