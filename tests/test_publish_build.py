"""Writing the almanac: the privacy scan gates the build, strips are re-encoded, output is safe.

The scan is the repository's own (`lastlight.privacy`), run over the staged output before
anything reaches the output directory.
"""

from __future__ import annotations

import io
from pathlib import Path

import image_factory
import ledger_factory as factory
import pytest
import seal_helpers as helpers
from PIL import Image

from lastlight import privacy
from lastlight.privacy.images import image_issues
from lastlight.publish import BuildError, build_site
from lastlight.seal.ledger import VerifyResult

OLDEST, MIDDLE, NEWEST = factory.DATES
LEAK = f"checked at {helpers.clock()}"


@pytest.fixture
def site_built(ledger_dir: Path, tmp_path: Path) -> Path:
    out = tmp_path / "site"
    assert build_site(ledger_dir, out).ok
    return out


def tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def chunk_types(png: bytes) -> list[bytes]:
    return image_factory.png_chunk_types(png)


# --- the build is gated by the privacy scan -------------------------------------------------


def test_a_planted_clock_time_in_a_forecast_note_fails_the_build_and_writes_nothing(
    ledger_dir: Path, tmp_path: Path
) -> None:
    factory.overwrite_notes(ledger_dir, MIDDLE, LEAK)
    out = tmp_path / "site"
    report = build_site(ledger_dir, out)
    assert not report.ok
    assert [finding.rule for finding in report.findings] == ["P001"]
    assert report.findings[0].path.endswith("2026-10-12.html")
    assert not out.exists(), "a leaking site must never be left where it could be uploaded"


@pytest.mark.parametrize(
    ("reason", "rule"),
    [
        (f"late seal at {helpers.clock()}", "P001"),
        (f"seen near {helpers.NEAR_REFERENCE_PAIR}", "P003"),
    ],
    ids=["clock time", "coordinates"],
)
def test_a_verification_reason_the_scanner_flags_fails_the_build_instead_of_being_hidden(
    ledger_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str, rule: str
) -> None:
    monkeypatch.setattr(
        "lastlight.publish.ledger.verify", lambda date, ledger: VerifyResult(False, reason)
    )
    out = tmp_path / "site"
    report = build_site(ledger_dir, out)
    assert not report.ok
    assert {finding.rule for finding in report.findings} == {rule}
    assert {Path(finding.path).name for finding in report.findings} >= {"2026-10-13.html"}
    assert not out.exists()


def test_a_failed_build_leaves_the_previous_site_untouched(
    ledger_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "site"
    assert build_site(ledger_dir, out).ok
    before = tree(out)
    factory.overwrite_notes(ledger_dir, NEWEST, LEAK)
    assert not build_site(ledger_dir, out).ok
    assert tree(out) == before
    assert [path.name for path in tmp_path.iterdir() if path.name != "ledger"] == ["site"]


def test_a_clean_build_passes_the_repository_scanner_over_every_file(site_built: Path) -> None:
    result = privacy.scan([site_built], root=site_built)
    assert result.findings == []
    assert result.files_scanned >= 10


def test_the_report_counts_what_was_built(ledger_dir: Path, tmp_path: Path) -> None:
    report = build_site(ledger_dir, tmp_path / "site")
    assert (report.evenings, report.verified) == (3, 3)
    assert report.pages == 6  # home, archive, scoreboard and one per date
    assert report.warnings == ()


def test_the_report_counts_only_the_evenings_that_verify(
    ledger_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory.seal_date(ledger_dir, NEWEST.replace(day=14), monkeypatch, late=True)
    factory.overwrite_notes(ledger_dir, OLDEST, "edited after sealing")
    report = build_site(ledger_dir, tmp_path / "site")
    assert report.ok
    assert (report.evenings, report.verified) == (4, 2)
    assert report.pages == 7  # home, archive, scoreboard and one per date


# --- strips carry no metadata -----------------------------------------------------------------

PLANTED = {
    "tEXt": image_factory.png_with_text(kind="tEXt"),
    "zTXt": image_factory.png_with_text(kind="zTXt"),
    "iTXt": image_factory.png_with_text(kind="iTXt"),
    "tIME": image_factory.png_with_chunk(b"tIME"),
    "eXIf": image_factory.png_with_chunk(b"eXIf"),
}


@pytest.mark.parametrize("planted", sorted(PLANTED))
def test_strips_in_the_site_are_re_encoded_without_metadata(
    ledger_dir: Path, tmp_path: Path, planted: str
) -> None:
    source = PLANTED[planted]
    assert planted.encode() in chunk_types(source), "the fixture must really carry the chunk"
    factory.write_strip(ledger_dir, NEWEST, source)
    out = tmp_path / "site"
    assert build_site(ledger_dir, out).ok
    published = (out / "strips" / "2026-10-13.png").read_bytes()
    assert chunk_types(published) == [b"IHDR", b"IDAT", b"IEND"]
    assert image_issues(published, ".png") == []


def test_re_encoding_keeps_the_pixels(ledger_dir: Path, tmp_path: Path) -> None:
    painted = Image.new("RGB", (4, 2), (10, 20, 30))
    painted.putpixel((3, 1), (250, 128, 7))
    buffer = io.BytesIO()
    painted.save(buffer, format="PNG")
    factory.write_strip(ledger_dir, NEWEST, buffer.getvalue())
    out = tmp_path / "site"
    assert build_site(ledger_dir, out).ok
    published = Image.open(out / "strips" / "2026-10-13.png")
    assert published.size == (4, 2)
    assert published.convert("RGB").tobytes() == painted.tobytes()


def test_a_paletted_strip_with_transparency_is_flattened_to_plain_rgb(
    ledger_dir: Path, tmp_path: Path
) -> None:
    palette = Image.new("P", (3, 3))
    palette.putpalette([255, 0, 0, 0, 255, 0] + [0] * 762)
    buffer = io.BytesIO()
    palette.save(buffer, format="PNG", transparency=0)
    factory.write_strip(ledger_dir, NEWEST, buffer.getvalue())
    out = tmp_path / "site"
    assert build_site(ledger_dir, out).ok
    published = (out / "strips" / "2026-10-13.png").read_bytes()
    assert chunk_types(published) == [b"IHDR", b"IDAT", b"IEND"]
    assert Image.open(out / "strips" / "2026-10-13.png").mode == "RGB"


def test_an_unreadable_strip_is_a_warning_and_a_placeholder_not_a_failed_build(
    ledger_dir: Path, tmp_path: Path
) -> None:
    factory.strip_path(ledger_dir, MIDDLE).write_bytes(b"this is not a png")
    out = tmp_path / "site"
    report = build_site(ledger_dir, out)
    assert report.ok
    assert len(report.warnings) == 1 and "2026-10-12" in report.warnings[0]
    assert not (out / "strips" / "2026-10-12.png").exists()
    assert "No painting for this date" in (out / "2026-10-12.html").read_text(encoding="utf-8")


def test_a_painting_behind_a_symlinked_directory_is_not_published(
    ledger_dir: Path, tmp_path: Path
) -> None:
    month = ledger_dir / "2026" / "10"
    moved = tmp_path / "outside-month"
    month.rename(moved)
    month.symlink_to(moved, target_is_directory=True)
    out = tmp_path / "site"
    assert build_site(ledger_dir, out).ok
    assert [path for path in out.rglob("*.png")] == []
    page = (out / "2026-10-13.html").read_text(encoding="utf-8")
    assert "No painting for this date" in page and "<img" not in page


def test_a_missing_ledger_is_a_warning_and_an_empty_site(tmp_path: Path) -> None:
    report = build_site(tmp_path / "absent", tmp_path / "site")
    assert report.ok and report.evenings == 0
    assert len(report.warnings) == 1 and "no ledger" in report.warnings[0]


def test_unusable_index_lines_are_reported_as_a_warning(ledger_dir: Path, tmp_path: Path) -> None:
    with (ledger_dir / "index.jsonl").open("a") as index:
        index.write("garbage\n")
    report = build_site(ledger_dir, tmp_path / "site")
    assert report.ok and report.evenings == 3
    assert len(report.warnings) == 1 and "1 line" in report.warnings[0]


def test_a_ledger_path_that_is_a_file_is_an_error(tmp_path: Path) -> None:
    file = tmp_path / "ledger"
    file.write_text("x")
    with pytest.raises(BuildError, match="not a directory"):
        build_site(file, tmp_path / "site")


# --- the output directory ---------------------------------------------------------------------


def test_building_twice_gives_the_same_bytes(ledger_dir: Path, tmp_path: Path) -> None:
    assert build_site(ledger_dir, tmp_path / "one").ok
    assert build_site(ledger_dir, tmp_path / "two").ok
    assert tree(tmp_path / "one") == tree(tmp_path / "two")


def test_a_rebuild_replaces_the_previous_site_and_drops_pages_that_are_gone(
    ledger_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "site"
    assert build_site(ledger_dir, out).ok
    assert (out / "2026-10-11.html").exists()
    rows = (ledger_dir / "index.jsonl").read_text().splitlines(keepends=True)
    (ledger_dir / "index.jsonl").write_text("".join(rows[1:]))
    assert build_site(ledger_dir, out).ok
    assert not (out / "2026-10-11.html").exists()
    assert not (out / "strips" / "2026-10-11.png").exists()
    assert (out / "2026-10-13.html").exists()


def test_an_existing_empty_output_directory_is_fine(ledger_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "site"
    out.mkdir()
    assert build_site(ledger_dir, out).ok
    assert (out / "index.html").is_file()


def test_the_output_directory_is_created_with_its_parents(ledger_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "deep" / "er" / "site"
    assert build_site(ledger_dir, out).ok
    assert (out / "index.html").is_file()


def test_a_directory_that_is_not_a_previous_site_is_never_replaced(
    ledger_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "precious"
    out.mkdir()
    (out / "notes.txt").write_text("keep me")
    with pytest.raises(BuildError, match="not a lastlight site"):
        build_site(ledger_dir, out)
    assert tree(out) == {"notes.txt": b"keep me"}


FOREIGN_HOME = b"<!doctype html><title>someone else's web root</title>"


def test_a_directory_with_a_foreign_index_html_is_never_replaced(
    ledger_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "webroot"
    out.mkdir()
    (out / "index.html").write_bytes(FOREIGN_HOME)
    (out / "about.html").write_bytes(b"<p>about me</p>")
    before = tree(out)
    with pytest.raises(BuildError, match="not a lastlight site"):
        build_site(ledger_dir, out)
    assert tree(out) == before == {"index.html": FOREIGN_HOME, "about.html": b"<p>about me</p>"}
    assert [path.name for path in tmp_path.iterdir() if path.name != "ledger"] == ["webroot"]


def test_an_earlier_build_with_extra_files_beside_it_is_still_replaced(
    ledger_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "site"
    assert build_site(ledger_dir, out).ok
    (out / "leftover.txt").write_text("from an older version of the generator")
    assert build_site(ledger_dir, out).ok
    assert not (out / "leftover.txt").exists()
    assert (out / "index.html").is_file()


def test_an_output_path_that_is_a_file_or_a_symlink_is_an_error(
    ledger_dir: Path, tmp_path: Path
) -> None:
    file = tmp_path / "file"
    file.write_text("x")
    with pytest.raises(BuildError, match="not a directory"):
        build_site(ledger_dir, file)
    target = tmp_path / "elsewhere"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(BuildError, match="symlink"):
        build_site(ledger_dir, link)


def test_the_output_directory_is_readable_by_others(ledger_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "site"
    assert build_site(ledger_dir, out).ok
    assert out.stat().st_mode & 0o755 == 0o755
