"""Publish completed files without truncating the previous successful delivery."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tempfile


@contextmanager
def atomic_path(destination):
    """Stage beside the destination so publication is one same-filesystem rename."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp",
                                    dir=destination.parent)
    os.close(handle)
    staged = Path(name)
    try:
        yield staged
        os.replace(staged, destination)
    finally:
        staged.unlink(missing_ok=True)


def atomic_write_text(destination, text, encoding="utf-8"):
    with atomic_path(destination) as staged:
        with staged.open("w", encoding=encoding) as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())


def publish_files(pairs, *, cancel=None):
    """Publish a fully staged set, rolling back completed replacements on failure.

    Each rename is atomic. This is an exception-safe batch, not a filesystem-wide
    transaction against power loss. Failed rollback retains recovery copies.
    """
    pairs = [(Path(source), Path(target)) for source, target in pairs]
    def check_cancelled():
        if cancel is not None and cancel.is_set():
            raise RuntimeError("Cancelled before completing delivery. Previous files are preserved.")

    check_cancelled()
    if not pairs:
        return
    targets = [target.resolve() for _, target in pairs]
    if len(set(targets)) != len(targets):
        raise ValueError("A delivery cannot contain duplicate destination paths.")
    for source, target in pairs:
        if not source.is_file():
            raise FileNotFoundError(source)
        target.parent.mkdir(parents=True, exist_ok=True)
    backup_parent = pairs[0][1].parent.resolve()
    recovery = Path(tempfile.mkdtemp(prefix=".nuranova-recovery-", dir=backup_parent)).resolve()
    backups, published = {}, []
    retain = False
    try:
        # Preserve every old file before modifying any destination.
        for index, (_, target) in enumerate(pairs):
            check_cancelled()
            if target.exists():
                backup = recovery / str(index)
                try:
                    os.link(target, backup)
                except OSError:
                    shutil.copy2(target, backup)
                backups[target] = backup
        try:
            for source, target in pairs:
                check_cancelled()
                os.replace(source, target)
                published.append(target)
            check_cancelled()
        except BaseException as original:
            errors = []
            for target in reversed(published):
                try:
                    if target in backups:
                        os.replace(backups[target], target)
                    else:
                        target.unlink(missing_ok=True)
                except OSError as error:
                    errors.append(str(error))
            if errors:
                retain = True
                raise OSError(f"Delivery failed; recovery copies retained at {recovery}: "
                              + "; ".join(errors)) from original
            raise
    finally:
        # Only this generated directory under the exact output parent is removable.
        if not retain and recovery.parent == backup_parent and recovery.name.startswith(".nuranova-recovery-"):
            shutil.rmtree(recovery, ignore_errors=True)
