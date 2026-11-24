"""Background processing queue with progress reporting, cancellation, disk checks, and checkpointing."""

from __future__ import annotations

import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProcessingJob:
    job_id: str
    title: str
    kind: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0  # 0.0 .. 1.0
    message: str = "Queued"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    measured_fps: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)


class ProcessingQueueManager:
    """Bounded background worker pool keeping the desktop UI responsive during offline reconstruction."""

    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="nuranova-worker")
        self._jobs: Dict[str, ProcessingJob] = {}
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[ProcessingJob], None]] = []

    def add_listener(self, cb: Callable[[ProcessingJob], None]) -> None:
        self._callbacks.append(cb)

    def _emit(self, job: ProcessingJob) -> None:
        for cb in list(self._callbacks):
            try:
                cb(job)
            except Exception:
                pass

    @staticmethod
    def check_free_disk_space_gb(target_path: str | Path) -> float:
        p = Path(target_path)
        while not p.exists() and p.parent != p:
            p = p.parent
        usage = shutil.disk_usage(str(p))
        return usage.free / (1024**3)

    def submit(
        self,
        title: str,
        kind: str,
        fn: Callable[[Callable[[float, str], None], threading.Event], Any],
        min_free_gb: float = 0.5,
        check_path: Optional[Path] = None,
    ) -> ProcessingJob:
        if check_path is not None:
            free_gb = self.check_free_disk_space_gb(check_path)
            if free_gb < min_free_gb:
                raise OSError(f"Insufficient disk space ({free_gb:.2f} GB free; {min_free_gb:.2f} GB required).")

        job = ProcessingJob(job_id=uuid.uuid4().hex[:12], title=title, kind=kind)
        with self._lock:
            self._jobs[job.job_id] = job
        self._emit(job)

        def runner() -> None:
            if job.cancel_event.is_set():
                job.status = JobStatus.CANCELLED
                job.message = "Cancelled before start"
                self._emit(job)
                return
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            job.message = "Starting..."
            self._emit(job)

            def progress_cb(pct: float, msg: str) -> None:
                job.progress = max(0.0, min(1.0, float(pct)))
                job.message = msg
                self._emit(job)

            try:
                res = fn(progress_cb, job.cancel_event)
                if job.cancel_event.is_set():
                    job.status = JobStatus.CANCELLED
                    job.message = "Cancelled by user"
                else:
                    job.result = res
                    job.progress = 1.0
                    job.status = JobStatus.COMPLETED
                    job.message = "Completed"
            except Exception as exc:
                job.error = str(exc)
                job.status = JobStatus.FAILED
                job.message = f"Failed: {exc}"
            finally:
                job.finished_at = time.time()
                self._emit(job)

        self._executor.submit(runner)
        return job

    def cancel_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        job.cancel_event.set()
        if job.status == JobStatus.QUEUED:
            job.status = JobStatus.CANCELLED
            job.message = "Cancelled"
            self._emit(job)
        return True

    def get_job(self, job_id: str) -> Optional[ProcessingJob]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> List[ProcessingJob]:
        with self._lock:
            return list(self._jobs.values())

    def shutdown(self) -> None:
        for j in self.list_jobs():
            j.cancel_event.set()
        self._executor.shutdown(wait=False)
