"""The private frame archive on the capture host.

Layout: `<archive>/<camera id>/<YYYY-MM-DD>/<frame>.jpg`, one `<frame>.json` sidecar beside
each frame, and one `log.jsonl` per camera-evening that records every poll, including the ones
that saved nothing. `<frame>` is the frame's own Last-Modified time in UTC as `YYYYmmddTHHMMSSZ`.

Raw frames are third-party images and must never reach git (`docs/design.md` section 3), so
the collector refuses to run with an archive inside a git work tree. The archive is private
by being outside every repository; nothing here is ever published.

The date folder is the local mean solar date of the evening, derived from longitude alone (the
convention of `lastlight.geometry`), so an evening that crosses UTC midnight stays in one folder.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

ARCHIVE_ENV_VAR: Final = "LASTLIGHT_ARCHIVE"
DEFAULT_ARCHIVE_SUBPATH: Final = ".local/share/lastlight/frames"
FRAME_TIME_FORMAT: Final = "%Y%m%dT%H%M%SZ"
LOG_NAME: Final = "log.jsonl"


def resolve_archive(
    cli_value: str | None,
    *,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    """The archive root: `--archive`, else `$LASTLIGHT_ARCHIVE`, else the per-user default.

    The result is absolute with symlinks resolved, so a link that points into a work tree is
    seen for what it is. A leading `~` means `home`.
    """
    environment = os.environ if environ is None else environ
    base = Path.home() if home is None else home
    chosen = cli_value or environment.get(ARCHIVE_ENV_VAR) or str(base / DEFAULT_ARCHIVE_SUBPATH)
    if chosen == "~" or chosen.startswith("~/"):
        chosen = str(base / chosen[2:])
    return Path(chosen).resolve()


def inside_git_worktree(path: Path) -> bool:
    """True if `path`, which need not exist yet, is a git work tree or lies inside one.

    A work tree has a `.git` entry at its top: a directory for a clone, a file for a linked
    worktree or submodule. Looking for that in the path and each ancestor needs no git binary
    and no existing directory, so the check can run before anything is created.
    """
    resolved = path.resolve()
    return any(os.path.lexists(directory / ".git") for directory in (resolved, *resolved.parents))


def evening_date(when_utc: dt.datetime, lon: float) -> dt.date:
    """The local mean solar date of an instant at east longitude `lon`."""
    return (when_utc + dt.timedelta(hours=lon / 15.0)).date()


def frame_stem(last_modified: dt.datetime) -> str:
    """A frame's file name stem: its Last-Modified time in UTC, `YYYYmmddTHHMMSSZ`."""
    return last_modified.astimezone(dt.UTC).strftime(FRAME_TIME_FORMAT)


def _write_atomically(path: Path, data: bytes) -> None:
    """Write via a temporary file and a rename, so a crash never leaves a half-written file."""
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class EveningArchive:
    """One camera's folder for one evening. Directories are created on the first write."""

    def __init__(self, root: Path, camera_id: str, evening: dt.date) -> None:
        self.directory = root / camera_id / evening.isoformat()

    def has_frame(self, stem: str) -> bool:
        return (self.directory / f"{stem}.jpg").exists()

    def save_frame(self, stem: str, body: bytes, sidecar: Mapping[str, Any]) -> None:
        """Write the JPEG, then its sidecar. Raises OSError if either cannot be written."""
        self.directory.mkdir(parents=True, exist_ok=True)
        _write_atomically(self.directory / f"{stem}.jpg", body)
        text = json.dumps(sidecar, indent=2, sort_keys=True) + "\n"
        _write_atomically(self.directory / f"{stem}.json", text.encode())

    def append_log(self, entry: Mapping[str, Any]) -> None:
        """Append one poll record to the evening's `log.jsonl`. Raises OSError on failure."""
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / LOG_NAME).open("a", encoding="utf-8") as log:
            log.write(json.dumps(entry, sort_keys=True) + "\n")
