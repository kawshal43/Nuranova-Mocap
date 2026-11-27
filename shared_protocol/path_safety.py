"""Filesystem containment checks shared by the studio's archive importers."""

from pathlib import Path, PureWindowsPath


def archive_member_path(destination: Path, member: str) -> Path:
    """Resolve an archive member without allowing absolute paths or sibling escapes."""
    # ZIPs may contain Windows paths even when inspected on another platform.
    windows_path = PureWindowsPath(member)
    if windows_path.drive or windows_path.root or "\x00" in member:
        raise ValueError(f"Unsafe path inside zip archive: {member}")
    root = destination.resolve()
    target = (root / member.replace("\\", "/")).resolve()
    if not target.is_relative_to(root) or target == root:
        raise ValueError(f"Unsafe path inside zip archive: {member}")
    return target
