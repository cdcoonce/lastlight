"""Write the almanac: stage it, scan it with the privacy scanner, and only then put it in place.

`build_site` renders every page into a staging directory next to the output directory, re-encodes
the paintings, and runs `lastlight.privacy.scan` over the staged files. Findings leave the output
directory exactly as it was (and the staging directory removed): a site that leaks is never on disk
where a deploy step could pick it up. Only a clean build replaces the output directory, and only
if that directory is empty, absent, or an earlier build of this site.

The paintings are the PNGs beside each sealed forecast. They are not copied: each is decoded and
written again with `render.save_strip`, so the published file holds pixels and nothing else, no
matter what the original carried. The scan then checks the result for metadata (rule P004).

The site also says so when an evening has no seal (`lastlight.publish.missed`). It reports on
every evening up to `through`: the one given, or else the newest evening whose deadline has
passed at the build's clock (`now_utc`, or the real one). The build reads the clock once, for
that and for one more thing: a `through` later than that evening is refused, because the site
would call an evening that has not happened missed.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from lastlight.privacy import Finding, scan
from lastlight.publish import pages
from lastlight.publish.ledger import Evening, Ledger, read_ledger
from lastlight.publish.missed import default_through, missed_evenings
from lastlight.render.curve import save_strip

SITE_MODE = 0o755


class BuildError(Exception):
    """The site could not be built because of the directories given, not because of a leak."""


@dataclass(frozen=True)
class BuildReport:
    out_dir: Path
    pages: int
    evenings: int
    verified: int
    findings: tuple[Finding, ...]
    """Privacy findings in the generated site; any at all means nothing was written."""
    warnings: tuple[str, ...]
    missed: tuple[dt.date, ...] = ()
    """The evenings the site says have no seal, newest first."""
    through: dt.date | None = None
    """The newest evening the site reports on."""

    @property
    def ok(self) -> bool:
        return not self.findings


def utc_now() -> dt.datetime:
    """The build's clock, used only to choose the default `through` and to bound an explicit one."""
    return dt.datetime.now(dt.UTC)


def build_site(
    ledger_dir: Path,
    out_dir: Path,
    *,
    through: dt.date | None = None,
    now_utc: dt.datetime | None = None,
) -> BuildReport:
    """Build the almanac of `ledger_dir` into `out_dir`, unless the privacy scan finds a leak.

    `through` is the newest evening the site reports on; an evening between the first sealed one
    and `through` with no row in the index is shown as missed. By default it is the newest
    evening whose deadline had passed at `now_utc` (the real clock if not given), and it cannot
    be later: a `BuildError` is raised, before anything is written, for an evening that has not
    ended.
    """
    if ledger_dir.exists() and not ledger_dir.is_dir():
        raise BuildError(f"the ledger {ledger_dir} is not a directory")
    newest_closed = default_through(now_utc if now_utc is not None else utc_now())
    if through is not None and through > newest_closed:
        raise BuildError(
            f"--through {through.isoformat()} is an evening that has not ended; the newest "
            f"evening that has is {newest_closed.isoformat()}"
        )
    target = Path(os.path.abspath(out_dir))
    _check_output(target)

    ledger = read_ledger(ledger_dir)
    if through is None:
        through = newest_closed
    missed = missed_evenings((evening.date for evening in ledger.evenings), through)
    ledger = dataclasses.replace(ledger, missed=missed)
    warnings = _ledger_warnings(ledger, ledger_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", suffix=".build", dir=target.parent))
    try:
        evenings, strip_warnings = _publish_strips(ledger.evenings, staging)
        written = _write_pages(dataclasses.replace(ledger, evenings=evenings), staging)
        findings = tuple(
            dataclasses.replace(finding, path=(out_dir / finding.path).as_posix())
            for finding in scan([staging], root=staging).findings
        )
        if not findings:
            _install(staging, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)  # already renamed away after a clean build
    return BuildReport(
        out_dir=out_dir,
        pages=written,
        evenings=len(evenings),
        verified=sum(evening.verified for evening in evenings),
        findings=findings,
        warnings=(*warnings, *strip_warnings),
        missed=missed,
        through=through,
    )


def _ledger_warnings(ledger: Ledger, ledger_dir: Path) -> list[str]:
    warnings = []
    if not ledger.found:
        warnings.append(f"no ledger at {ledger_dir}; building the empty-state site")
    if ledger.skipped_index_lines:
        count = ledger.skipped_index_lines
        noun = "line" if count == 1 else "lines"
        warnings.append(f"{count} {noun} of index.jsonl could not be used and were left out")
    return warnings


# --- paintings --------------------------------------------------------------------------------


def _publish_strips(
    evenings: tuple[Evening, ...], staging: Path
) -> tuple[tuple[Evening, ...], list[str]]:
    """Re-encode each painting into the staged site.

    A date with no painting, or one that is not a readable image, still gets its page, with a
    placeholder; each such date is a warning, so a ledger layout the reader no longer matches
    shows up as a warning per date rather than as a quietly empty gallery. A date that verifies
    with a forecast naming no painting is not a warning: nothing is missing, none was sealed.
    """
    kept: list[Evening] = []
    warnings: list[str] = []
    for evening in evenings:
        iso = evening.date.isoformat()
        if evening.strip is None:
            if not evening.names_no_painting:
                warnings.append(f"there is no painting for {iso}; its page shows a placeholder")
        elif not _reencode(evening.strip, staging / pages.strip_name(evening.date)):
            warnings.append(f"the painting for {iso} is not a readable image and was left out")
            evening = dataclasses.replace(evening, strip=None)
        kept.append(evening)
    return tuple(kept), warnings


def _reencode(source: Path, destination: Path) -> bool:
    try:
        with Image.open(source) as image:
            image.load()
            flat = image.convert("RGB")
    except Exception:  # Pillow raises a different type per format and per kind of corruption
        return False
    destination.parent.mkdir(exist_ok=True)
    save_strip(flat, destination)
    return True


# --- pages ------------------------------------------------------------------------------------


def _write_pages(ledger: Ledger, staging: Path) -> int:
    documents = {
        "index.html": pages.home_page(ledger),
        "archive.html": pages.archive_page(ledger),
        "scoreboard.html": pages.scoreboard_page(),
        **{pages.page_name(e.date): pages.date_page(e) for e in ledger.evenings},
    }
    for name, text in documents.items():
        (staging / name).write_text(text, encoding="utf-8")
    (staging / "style.css").write_text(pages.stylesheet(), encoding="utf-8")
    return len(documents)


# --- the output directory ---------------------------------------------------------------------


def _check_output(target: Path) -> None:
    if target.is_symlink():
        raise BuildError(
            f"{target} is a symlink; the build replaces its output and will not follow one"
        )
    if not target.exists():
        return
    if not target.is_dir():
        raise BuildError(f"{target} exists and is not a directory")
    if any(target.iterdir()) and not _is_earlier_build(target):
        raise BuildError(
            f"{target} is not empty and is not a lastlight site (it has no index.html written by "
            "`lastlight site build`), so it will not be replaced"
        )


def _is_earlier_build(directory: Path) -> bool:
    home = directory / "index.html"
    if not home.is_file():
        return False
    return pages.GENERATOR_META in home.read_text(encoding="utf-8", errors="replace")


def _install(staging: Path, target: Path) -> None:
    staging.chmod(SITE_MODE)
    if target.exists():
        shutil.rmtree(target)
    staging.rename(target)
