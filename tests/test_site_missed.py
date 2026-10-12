"""The almanac says so when an evening has no seal (issue #26).

A scorer must never treat an absent seal as a forecast, and a visitor must not mistake the newest
sealed painting for tonight's. `build_site(..., through=D)` reports on every evening up to D: an
evening between the first sealed one and D that has no row in `index.jsonl` was missed, and the
site says "No forecast was sealed for D." Without `through`, D is the newest evening whose
deadline (the sun at the horizon at the reference point) has passed at the build's `now_utc`.

Every state is built from small sealed ledgers (`ledger_factory`, offline) and asserted through
`html_probe`. Dates only: nothing here spells a time of day.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import ledger_factory as factory
import pytest
from html_probe import read_page

from lastlight.cli import main
from lastlight.daily import default_evening, evening_deadline
from lastlight.privacy import contains_clock_time
from lastlight.publish import BuildReport, build_site, pages
from lastlight.publish.missed import default_through, missed_evenings
from lastlight.site import REFERENCE_POINT

OLDEST, MIDDLE, NEWEST = factory.DATES  # 2026-10-11, 2026-10-12, 2026-10-13
DAY = dt.timedelta(days=1)
SECOND = dt.timedelta(seconds=1)
LON = REFERENCE_POINT[1]


def sentence(date: dt.date) -> str:
    return f"No forecast was sealed for {date.isoformat()}."


def page_names(site: Path) -> set[str]:
    return {path.name for path in site.glob("*.html")}


def sealed_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *dates: dt.date, late: tuple[dt.date, ...] = ()
) -> Path:
    """A ledger with exactly `dates` sealed, each with its painting; `late` ones too late."""
    ledger = tmp_path / "ledger"
    for date in dates:
        factory.seal_date(ledger, date, monkeypatch, strip=factory.strip_png(), late=date in late)
    return ledger


def build(ledger: Path, tmp_path: Path, **options: object) -> tuple[BuildReport, Path]:
    out = tmp_path / "site"
    report = build_site(ledger, out, **options)  # type: ignore[arg-type]
    assert report.findings == ()
    return report, out


def missed_in_archive(out: Path) -> list[str]:
    return read_page(out, "archive.html").texts("li", "missed")


def archive_order(out: Path) -> list[str]:
    """The archive's entries, newest first: 'sealed D' or 'missed D'."""
    order = []
    for item in read_page(out, "archive.html").find("li"):
        if item.has_class("evening"):
            order.append("sealed " + item.text.split()[0])
        elif item.has_class("missed"):
            order.append("missed " + item.text.removeprefix("No forecast was sealed for ")[:-1])
    return order


# --- which evenings were missed -------------------------------------------------------------


def test_missed_evenings_are_the_dates_between_the_first_seal_and_through_with_no_seal() -> None:
    sealed = [dt.date(2026, 10, 11), dt.date(2026, 10, 13), dt.date(2026, 10, 14)]
    assert missed_evenings(sealed, dt.date(2026, 10, 17)) == (
        dt.date(2026, 10, 17),
        dt.date(2026, 10, 16),
        dt.date(2026, 10, 15),
        dt.date(2026, 10, 12),
    )


def test_through_itself_is_included_and_dates_after_it_are_not() -> None:
    sealed = [dt.date(2026, 10, 11), dt.date(2026, 10, 15)]
    assert missed_evenings(sealed, dt.date(2026, 10, 13)) == (
        dt.date(2026, 10, 13),
        dt.date(2026, 10, 12),
    )


def test_the_dates_before_the_first_seal_are_not_missed() -> None:
    assert missed_evenings([dt.date(2026, 10, 13)], dt.date(2026, 10, 14)) == (
        dt.date(2026, 10, 14),
    )


@pytest.mark.parametrize(
    ("sealed", "through"),
    [
        ([], dt.date(2026, 10, 20)),
        ([dt.date(2026, 10, 13)], dt.date(2026, 10, 12)),
        ([dt.date(2026, 10, 13)], dt.date(2026, 10, 13)),
        ([dt.date(2026, 10, 11), dt.date(2026, 10, 12)], dt.date(2026, 10, 12)),
    ],
    ids=["empty ledger", "through before the first seal", "through is sealed", "nothing missed"],
)
def test_nothing_is_missed_when_there_is_no_gap(sealed: list[dt.date], through: dt.date) -> None:
    assert missed_evenings(sealed, through) == ()


# --- the default `through`: the newest evening whose deadline has passed --------------------


def test_through_is_the_day_before_until_the_sun_reaches_the_horizon() -> None:
    evening = dt.date(2026, 10, 14)
    just_before = evening_deadline(evening) - SECOND
    assert default_evening(just_before) == evening, "the run is still inside that evening"
    assert default_through(just_before) == evening - DAY


def test_through_is_the_evening_itself_from_the_instant_the_sun_reaches_the_horizon() -> None:
    evening = dt.date(2026, 10, 14)
    deadline = evening_deadline(evening)
    assert default_through(deadline) == evening
    assert default_through(deadline + SECOND) == evening


def test_through_stays_on_the_evening_through_the_night_after_it() -> None:
    evening = dt.date(2026, 10, 14)
    # Local mean solar midnight at the reference point, an hour past it, and the small hours.
    midnight = dt.datetime(2026, 10, 15, tzinfo=dt.UTC) - dt.timedelta(hours=LON / 15.0)
    for offset in (-1, 1, 60, 180, 300):
        assert default_through(midnight + dt.timedelta(minutes=offset)) == evening


def test_through_changes_only_at_a_deadline() -> None:
    first, second = dt.date(2026, 10, 14), dt.date(2026, 10, 15)
    assert default_through(evening_deadline(second) - SECOND) == first
    assert default_through(evening_deadline(second)) == second


def test_through_agrees_with_the_instant_seal_daily_calls_an_evening_late() -> None:
    # An evening is called missed at the same instant sealing it is refused: the deadline.
    evening = dt.date(2026, 6, 21)
    deadline = evening_deadline(evening)
    for now in (deadline - SECOND, deadline, deadline + SECOND):
        closed = default_through(now) >= evening
        assert closed == (now >= deadline)


def test_a_clock_that_is_not_timezone_aware_is_refused() -> None:
    with pytest.raises(ValueError, match="timezone"):
        default_through(dt.datetime(2026, 10, 14, 12))


# --- newest evening missed ------------------------------------------------------------------


@pytest.fixture
def newest_missed(ledger_dir: Path, tmp_path: Path) -> tuple[BuildReport, Path]:
    """2026-10-11 to 2026-10-13 sealed, and the site reports through 2026-10-15."""
    return build(ledger_dir, tmp_path, through=dt.date(2026, 10, 15))


def test_the_home_page_leads_with_the_newest_missed_evening(
    newest_missed: tuple[BuildReport, Path],
) -> None:
    _report, out = newest_missed
    home = read_page(out, "index.html")
    assert home.texts("p", "missed") == [sentence(dt.date(2026, 10, 15))]
    raw = (out / "index.html").read_text(encoding="utf-8")
    assert raw.index(sentence(dt.date(2026, 10, 15))) < raw.index('class="latest"')
    assert sentence(dt.date(2026, 10, 14)) not in home.text, "only the newest is the lead"


def test_the_home_page_still_shows_the_newest_sealed_painting_under_its_own_date(
    newest_missed: tuple[BuildReport, Path],
) -> None:
    _report, out = newest_missed
    home = read_page(out, "index.html")
    assert "2026-10-13" in home.headings()
    assert "Latest sealed painting" in home.headings()
    assert home.find("img")[0].attrs["src"] == "strips/2026-10-13.png"
    assert "2026-10-15" not in home.headings() and "2026-10-14" not in home.headings()


def test_the_archive_lists_every_missed_evening_among_the_sealed_ones(
    newest_missed: tuple[BuildReport, Path],
) -> None:
    _report, out = newest_missed
    assert missed_in_archive(out) == [
        sentence(dt.date(2026, 10, 15)),
        sentence(dt.date(2026, 10, 14)),
    ]
    assert archive_order(out) == [
        "missed 2026-10-15",
        "missed 2026-10-14",
        "sealed 2026-10-13",
        "sealed 2026-10-12",
        "sealed 2026-10-11",
    ]


def test_a_missed_evening_gets_no_page_and_no_link_and_no_painting(
    newest_missed: tuple[BuildReport, Path],
) -> None:
    report, out = newest_missed
    assert page_names(out) == {
        "index.html",
        "archive.html",
        "scoreboard.html",
        "2026-10-11.html",
        "2026-10-12.html",
        "2026-10-13.html",
    }
    assert not (out / "strips" / "2026-10-14.png").exists()
    archive = read_page(out, "archive.html")
    assert "2026-10-14.html" not in archive.hrefs() and "2026-10-15.html" not in archive.hrefs()
    assert report.pages == 6


def test_the_report_counts_the_missed_evenings_and_the_sealed_ones_separately(
    newest_missed: tuple[BuildReport, Path],
) -> None:
    report, _out = newest_missed
    assert report.missed == (dt.date(2026, 10, 15), dt.date(2026, 10, 14))
    assert report.through == dt.date(2026, 10, 15)
    assert (report.evenings, report.verified) == (3, 3)
    assert report.warnings == ()


# --- a gap in the middle --------------------------------------------------------------------


def test_a_gap_in_the_middle_is_listed_in_the_archive_but_does_not_lead_the_home_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = sealed_ledger(tmp_path, monkeypatch, OLDEST, NEWEST)
    _report, out = build(ledger, tmp_path, through=NEWEST)

    assert archive_order(out) == ["sealed 2026-10-13", "missed 2026-10-12", "sealed 2026-10-11"]
    assert missed_in_archive(out) == [sentence(MIDDLE)]
    home = read_page(out, "index.html")
    assert home.find("p", "missed") == []
    assert "No forecast was sealed" not in home.text
    assert "2026-10-13" in home.headings()
    assert not (out / "2026-10-12.html").exists()


def test_several_gaps_are_each_listed_once_in_date_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = sealed_ledger(
        tmp_path, monkeypatch, OLDEST, dt.date(2026, 10, 15), dt.date(2026, 10, 17)
    )
    _report, out = build(ledger, tmp_path, through=dt.date(2026, 10, 17))
    assert [entry for entry in archive_order(out) if entry.startswith("missed")] == [
        "missed 2026-10-16",
        "missed 2026-10-14",
        "missed 2026-10-13",
        "missed 2026-10-12",
    ]


# --- nothing missed -------------------------------------------------------------------------


def test_when_nothing_is_missed_no_page_says_a_forecast_was_not_sealed(
    ledger_dir: Path, tmp_path: Path
) -> None:
    report, out = build(ledger_dir, tmp_path, through=NEWEST)
    assert report.missed == ()
    for name in page_names(out):
        probe = read_page(out, name)
        assert "No forecast was sealed" not in probe.text, name
        assert probe.find("p", "missed") == [] and probe.find("li", "missed") == [], name
    assert archive_order(out) == ["sealed 2026-10-13", "sealed 2026-10-12", "sealed 2026-10-11"]


# --- through older than the newest seal -----------------------------------------------------


def test_through_older_than_the_newest_seal_makes_no_statement_on_the_home_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 2026-10-12 is missed and is `through`, but a later evening is sealed: nothing leads.
    ledger = sealed_ledger(tmp_path, monkeypatch, OLDEST, NEWEST)
    report, out = build(ledger, tmp_path, through=MIDDLE)

    assert report.missed == (MIDDLE,)
    home = read_page(out, "index.html")
    assert home.find("p", "missed") == [] and "No forecast was sealed" not in home.text
    assert missed_in_archive(out) == [sentence(MIDDLE)], "the archive still lists the gap"


def test_a_missed_evening_after_through_is_not_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = sealed_ledger(tmp_path, monkeypatch, OLDEST, dt.date(2026, 10, 16))
    _report, out = build(ledger, tmp_path, through=MIDDLE)
    assert missed_in_archive(out) == [sentence(MIDDLE)]


def test_through_before_every_seal_reports_nothing(ledger_dir: Path, tmp_path: Path) -> None:
    report, out = build(ledger_dir, tmp_path, through=dt.date(2026, 10, 1))
    assert report.missed == ()
    assert "No forecast was sealed" not in read_page(out, "archive.html").text


# --- an empty ledger ------------------------------------------------------------------------


@pytest.mark.parametrize("how", ["absent", "empty"])
def test_an_empty_ledger_keeps_the_empty_state(tmp_path: Path, how: str) -> None:
    ledger = tmp_path / "ledger"
    if how == "empty":
        ledger.mkdir()
    report, out = build(ledger, tmp_path, through=dt.date(2026, 10, 20))
    assert report.missed == ()
    home = read_page(out, "index.html")
    assert "No sealed painting yet" in home.headings()
    assert home.find("p", "missed") == [] and "No forecast was sealed" not in home.text
    archive = read_page(out, "archive.html")
    assert "No evenings sealed yet" in archive.text
    assert archive.find("li", "missed") == []


# --- sealed, but not in time, is still sealed -----------------------------------------------


def test_an_evening_with_an_index_row_is_never_missed_even_if_it_does_not_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    late = dt.date(2026, 10, 14)
    ledger = sealed_ledger(tmp_path, monkeypatch, OLDEST, MIDDLE, NEWEST, late, late=(late,))
    report, out = build(ledger, tmp_path, through=dt.date(2026, 10, 15))

    assert report.missed == (dt.date(2026, 10, 15),)
    assert archive_order(out)[:2] == ["missed 2026-10-15", "sealed 2026-10-14"]
    home = read_page(out, "index.html")
    assert home.texts("p", "missed") == [sentence(dt.date(2026, 10, 15))]
    notice = home.find("p", "notice")[0].text
    assert "2026-10-14" in notice and "did not verify" in notice, "today's notice is kept"
    assert "2026-10-13" in home.headings(), "the newest verified painting is still featured"


# --- the default through, from an injected clock --------------------------------------------


def test_the_default_through_is_decided_by_the_injected_clock_on_either_side_of_the_deadline(
    ledger_dir: Path, tmp_path: Path
) -> None:
    evening = dt.date(2026, 10, 14)
    deadline = evening_deadline(evening)

    before_report, before = build(ledger_dir, tmp_path / "a", now_utc=deadline - SECOND)
    assert before_report.through == NEWEST
    assert before_report.missed == ()
    assert "No forecast was sealed" not in read_page(before, "index.html").text

    after_report, after = build(ledger_dir, tmp_path / "b", now_utc=deadline)
    assert after_report.through == evening
    assert after_report.missed == (evening,)
    assert read_page(after, "index.html").texts("p", "missed") == [sentence(evening)]


def test_the_default_through_counts_every_evening_that_closed_since_the_last_seal(
    ledger_dir: Path, tmp_path: Path
) -> None:
    now = evening_deadline(dt.date(2026, 10, 16)) + dt.timedelta(hours=1)
    report, out = build(ledger_dir, tmp_path, now_utc=now)
    assert report.through == dt.date(2026, 10, 16)
    assert report.missed == (dt.date(2026, 10, 16), dt.date(2026, 10, 15), dt.date(2026, 10, 14))
    assert read_page(out, "index.html").texts("p", "missed") == [sentence(dt.date(2026, 10, 16))]


def test_an_explicit_through_wins_over_the_clock(ledger_dir: Path, tmp_path: Path) -> None:
    now = evening_deadline(dt.date(2026, 10, 20))
    report, _out = build(ledger_dir, tmp_path, through=NEWEST, now_utc=now)
    assert report.through == NEWEST and report.missed == ()


def test_without_now_utc_the_build_reads_the_clock_once_through_utc_now(
    ledger_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pinned = evening_deadline(dt.date(2026, 10, 14)) + SECOND
    monkeypatch.setattr("lastlight.publish.build.utc_now", lambda: pinned)
    report, _out = build(ledger_dir, tmp_path)
    assert report.through == dt.date(2026, 10, 14)


# --- the pages stay private, valid and stable -----------------------------------------------


def test_no_page_shows_a_time_of_day_and_the_scan_passes(
    newest_missed: tuple[BuildReport, Path],
) -> None:
    _report, out = newest_missed
    for name in sorted(page_names(out)):
        text = (out / name).read_text(encoding="utf-8")
        assert not contains_clock_time(text), name
        assert "UTC" not in text, name


def test_every_statement_is_the_one_sentence_with_a_date_and_nothing_else(
    newest_missed: tuple[BuildReport, Path],
) -> None:
    _report, out = newest_missed
    statements = [
        *read_page(out, "index.html").texts("p", "missed"),
        *read_page(out, "archive.html").texts("li", "missed"),
    ]
    assert len(statements) == 3
    for text in statements:
        assert re.fullmatch(r"No forecast was sealed for \d{4}-\d{2}-\d{2}\.", text), text


def test_building_twice_with_the_same_through_gives_the_same_bytes(
    ledger_dir: Path, tmp_path: Path
) -> None:
    _r1, one = build(ledger_dir, tmp_path / "one", through=dt.date(2026, 10, 15))
    _r2, two = build(ledger_dir, tmp_path / "two", through=dt.date(2026, 10, 15))
    for name in sorted(page_names(one)):
        assert (one / name).read_bytes() == (two / name).read_bytes(), name


def test_the_archive_says_a_missing_seal_is_never_back_filled_or_scored(
    newest_missed: tuple[BuildReport, Path],
) -> None:
    _report, out = newest_missed
    text = read_page(out, "archive.html").text
    assert "never back-filled" in text and "not a forecast" in text


def test_the_stylesheet_styles_the_missed_entries_with_the_existing_tokens() -> None:
    css = pages.stylesheet()
    assert ".missed" in css
    assert len(re.findall(r":root\s*\{", css)) == 2, "no new palette: the tokens are reused"


# --- the command ----------------------------------------------------------------------------


def run(capsys: pytest.CaptureFixture[str], *argv: str | Path) -> tuple[int, str, str]:
    code = main([str(arg) for arg in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_the_site_command_takes_through_and_reports_the_missed_evenings(
    capsys: pytest.CaptureFixture[str], ledger_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "site"
    code, stdout, stderr = run(
        capsys, "site", "build", "--ledger-dir", ledger_dir, "--out", out, "--through", "2026-10-15"
    )
    assert code == 0 and stderr == ""
    assert "built 6 pages for 3 evenings (3 verified)" in stdout
    assert "2 missed" in stdout
    assert read_page(out, "index.html").texts("p", "missed") == [sentence(dt.date(2026, 10, 15))]


def test_a_site_built_through_the_newest_seal_reports_no_missed_evening(
    capsys: pytest.CaptureFixture[str], ledger_dir: Path, tmp_path: Path
) -> None:
    code, stdout, _stderr = run(
        capsys,
        "site", "build", "--ledger-dir", ledger_dir, "--out", tmp_path / "site",
        "--through", "2026-10-13",
    )  # fmt: skip
    assert code == 0 and "missed" not in stdout


@pytest.mark.parametrize("bad", ["2026-13-01", "20261015", "tomorrow", ""])
def test_a_bad_through_date_is_a_usage_error(
    capsys: pytest.CaptureFixture[str], ledger_dir: Path, tmp_path: Path, bad: str
) -> None:
    out = tmp_path / "site"
    code, stdout, stderr = run(
        capsys, "site", "build", "--ledger-dir", ledger_dir, "--out", out, "--through", bad
    )
    assert code == 2 and stdout == ""
    assert "argument --through:" in stderr, "rejected by the option's own check, not as unknown"
    assert not out.exists()


def test_help_lists_through(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["site", "build", "--help"]) == 0
    assert "--through" in capsys.readouterr().out
