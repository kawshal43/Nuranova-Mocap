"""Model registry with SHA-256 checksum verification, local caching, and license tracking."""

from __future__ import annotations

import hashlib
import json
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from shared_protocol.atomic_io import atomic_path, atomic_write_text

REGISTRY_FILE = Path(__file__).parent / "models.json"
DEFAULT_CACHE_DIR = Path.home() / ".nuranova_mocap" / "models"


@dataclass
class ModelEntry:
    id: str
    family: str
    task: str
    version: str
    url: str
    license: str
    sha256: Optional[str] = None
    input_size: Optional[List[int]] = None


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelRegistry:
    def __init__(
        self, registry_path: Path = REGISTRY_FILE, cache_dir: Path = DEFAULT_CACHE_DIR
    ) -> None:
        self.registry_path = registry_path
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._download_lock = threading.RLock()
        self._checksum_path = self.cache_dir / "checksums.json"
        try:
            self._checksums = json.loads(self._checksum_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._checksums = {}
        if not isinstance(self._checksums, dict):
            self._checksums = {}
        self._data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        self.models: Dict[str, ModelEntry] = {
            m["id"]: ModelEntry(
                id=m["id"],
                family=m["family"],
                task=m["task"],
                version=m["version"],
                url=m["url"],
                license=m["license"],
                sha256=m.get("sha256") or self._checksums.get(m["id"]),
                input_size=m.get("input_size"),
            )
            for m in self._data.get("models", [])
        }

    def list_models(self) -> List[ModelEntry]:
        return list(self.models.values())

    def get_local_path(self, model_id: str) -> Path:
        entry = self.models[model_id]
        filename = Path(entry.url).name
        return self.cache_dir / f"{model_id}_{filename}"

    def is_cached(self, model_id: str) -> bool:
        p = self.get_local_path(model_id)
        return p.exists() and p.stat().st_size > 1024

    def ensure_downloaded(self, model_id: str, timeout_s: int = 120) -> Path:
        with self._download_lock:
            return self._ensure_downloaded(model_id, timeout_s)

    def _ensure_downloaded(self, model_id: str, timeout_s: int) -> Path:
        entry = self.models[model_id]
        dest = self.get_local_path(model_id)
        digest = compute_sha256(dest) if self.is_cached(model_id) else None
        if digest is None or (entry.sha256 is not None and digest != entry.sha256):
            req = urllib.request.Request(
                entry.url, headers={"User-Agent": "NuranovaMocapStudio/1.0"}
            )
            with atomic_path(dest) as staged:
                with urllib.request.urlopen(req, timeout=timeout_s) as resp, staged.open("wb") as out:
                    while True:
                        chunk = resp.read(256 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                digest = compute_sha256(staged)
                if entry.sha256 is not None and entry.sha256 != digest:
                    raise ValueError(
                        f"Checksum mismatch for model '{model_id}': expected {entry.sha256}, got {digest}"
                    )
                if staged.stat().st_size == 0:
                    raise ValueError(f"Empty model download for '{model_id}'.")
        if entry.sha256 is None:
            entry.sha256 = digest
            self._persist_checksum(model_id, digest)
        return dest

    def _persist_checksum(self, model_id: str, sha256_hex: str) -> None:
        self._checksums[model_id] = sha256_hex
        atomic_write_text(self._checksum_path, json.dumps(self._checksums, indent=2))
