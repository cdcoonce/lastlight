"""Raw camera frames stay on the capture host: the repository must hold no JPEG at all."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKIPPED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "data",
    "node_modules",
}


def _tracked_files() -> list[Path]:
    try:
        listing = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
            capture_output=True,
            check=True,
            timeout=60,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        # An sdist or a checkout without git: walk the tree, skipping what git would ignore.
        return [
            path
            for path in REPO_ROOT.rglob("*")
            if path.is_file() and not SKIPPED_DIRS & set(path.relative_to(REPO_ROOT).parts)
        ]
    return [REPO_ROOT / name for name in listing.decode().split("\0") if name]


def test_the_repository_tree_holds_no_jpeg() -> None:
    files = _tracked_files()
    assert files, "the listing must not be empty, or the check proves nothing"
    jpegs = [str(p.relative_to(REPO_ROOT)) for p in files if p.suffix.lower() in {".jpg", ".jpeg"}]
    assert jpegs == []


def test_no_tracked_file_starts_with_a_jpeg_signature() -> None:
    # A frame renamed to hide it would still start with the JPEG start-of-image marker.
    offenders = []
    for path in _tracked_files():
        if path.is_file() and path.stat().st_size >= 3:
            with path.open("rb") as handle:
                if handle.read(3) == b"\xff\xd8\xff":
                    offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == []
