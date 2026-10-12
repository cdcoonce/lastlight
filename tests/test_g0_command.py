"""`lastlight g0`: collect a year of archived inputs in resumable chunks, then write the report.

    lastlight g0 --start 2025-10-01 --end 2026-09-30 --out docs/gates/G0/ [--cache-dir DIR]
                 [--stage fetch|report|all] [--budget-seconds N]

`fetch` downloads and renders dates into the cache, `report` reads only the cache (no network at
all) and writes the report, results file and figures, and `all` does both. Exit codes: 0 done,
1 not every date has a record (a network fault, or a date not collected yet; the report is still
written and lists them), 2 usage error.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import re

import hrrr_support
from g0_support import EVENING, NOW, combined_transport, dates

from lastlight.cli import Command, discover, main
from lastlight.commands import g0
from lastlight.g0.record import RecordStore
from lastlight.privacy.scanner import scan


class NoNetwork:
    """Any request is a bug: the report stage must read only the cache."""

    def exists(self, url: str) -> bool:
        raise AssertionError(f"the report stage asked for {url}")

    def get(self, url, byte_range=None):
        raise AssertionError(f"the report stage asked for {url}")


def command(transport) -> Command:
    def run(args: argparse.Namespace) -> int:
        return g0.execute(args, transport=transport, now_utc=NOW)

    return dataclasses.replace(g0.COMMAND, run=run)


def served(days: list[dt.date]):
    files: dict[str, bytes] = {}
    for day in days:
        files.update(combined_transport(day).files)
    return hrrr_support.FakeTransport(files)


def run(capsys, tmp_path, transport, *extra, days=2):
    out = tmp_path / "out"
    end = EVENING + dt.timedelta(days=days - 1)
    argv = [
        "g0",
        "--start",
        EVENING.isoformat(),
        "--end",
        end.isoformat(),
        "--out",
        str(out),
        "--cache-dir",
        str(tmp_path / "cache"),
        *extra,
    ]
    code = main(argv, commands=[command(transport)])
    captured = capsys.readouterr()
    return code, captured.out, captured.err, out


# --- registration ------------------------------------------------------------------------------


def test_the_command_is_discovered_by_name() -> None:
    assert "g0" in {c.name for c in discover()}


def test_help_lists_the_options(capsys) -> None:
    assert main(["g0", "--help"], commands=[g0.COMMAND]) == 0
    out = capsys.readouterr().out
    for option in ("--start", "--end", "--out", "--cache-dir", "--stage", "--budget-seconds"):
        assert option in out
    assert "~/.cache/lastlight" in out


def test_the_default_cache_is_outside_the_repository() -> None:
    assert g0.DEFAULT_CACHE_DIR.parts[-2:] == (".cache", "lastlight")
    assert g0.DEFAULT_CACHE_DIR.is_absolute()


# --- the stages --------------------------------------------------------------------------------


def test_fetch_collects_the_dates_into_the_cache_and_writes_no_report(capsys, tmp_path) -> None:
    days = dates(EVENING, 2)
    code, out, err, report_dir = run(capsys, tmp_path, served(days), "--stage", "fetch")
    assert code == 0, err
    store = RecordStore(tmp_path / "cache" / "g0" / "records")
    assert store.dates() == set(days)
    assert not report_dir.exists()
    assert len(re.findall(r"^\d{4}-\d{2}-\d{2} ", out, re.MULTILINE)) == 2
    assert "2 collected" in out and "0 skipped" in out


def test_a_second_fetch_makes_no_request(capsys, tmp_path) -> None:
    run(capsys, tmp_path, served(dates(EVENING, 2)), "--stage", "fetch")
    code, out, _, _ = run(capsys, tmp_path, NoNetwork(), "--stage", "fetch")
    assert code == 0
    assert "0 collected" in out and "2 skipped" in out


def test_report_reads_only_the_cache_and_writes_the_outputs(capsys, tmp_path) -> None:
    run(capsys, tmp_path, served(dates(EVENING, 2)), "--stage", "fetch")
    code, out, err, report_dir = run(capsys, tmp_path, NoNetwork(), "--stage", "report")
    assert code == 0, err
    assert sorted(p.name for p in report_dir.iterdir()) == sorted(
        ["report.md", "results.csv", "d_distribution.png", "d_by_date.png", "profile_spread.png"]
    )
    assert "2 attempted, 2 used, 0 dropped" in out
    assert re.search(r"G0: (PASS|FAIL)", out)


def test_all_does_both_and_the_outputs_pass_the_privacy_scanner(capsys, tmp_path) -> None:
    code, _, err, report_dir = run(capsys, tmp_path, served(dates(EVENING, 2)), "--stage", "all")
    assert code == 0, err
    result = scan([report_dir], root=report_dir)
    assert result.findings == [], [str(f) for f in result.findings]
    text = (report_dir / "report.md").read_text()
    assert EVENING.isoformat() in (report_dir / "results.csv").read_text()
    assert "## Verdict" in text


def test_all_is_the_default_stage(capsys, tmp_path) -> None:
    code, _, _, report_dir = run(capsys, tmp_path, served(dates(EVENING, 2)))
    assert code == 0 and (report_dir / "report.md").is_file()


def test_a_date_missing_from_the_archive_is_a_dropped_date_and_not_a_failure(
    capsys, tmp_path
) -> None:
    days = dates(EVENING, 3)
    transport = hrrr_support.FakeTransport(
        {**served(days[:1]).files, **served(days[2:]).files}  # nothing for the middle date
    )
    code, out, _, report_dir = run(capsys, tmp_path, transport, days=3)
    assert code == 0
    assert "3 attempted, 2 used, 1 dropped" in out
    assert "hrrr_run_missing" in (report_dir / "report.md").read_text()


def test_the_report_stage_with_dates_not_yet_collected_exits_1_but_still_writes(
    capsys, tmp_path
) -> None:
    run(capsys, tmp_path, served(dates(EVENING, 2)), "--stage", "fetch")
    code, out, err, report_dir = run(capsys, tmp_path, NoNetwork(), "--stage", "report", days=4)
    assert code == 1
    assert "2 not collected" in err or "2 not collected" in out
    text = (report_dir / "report.md").read_text()
    assert text.count("not_collected") >= 2


def test_a_network_fault_exits_1_and_leaves_the_date_for_the_next_run(capsys, tmp_path) -> None:
    class Down:
        def exists(self, url):
            raise g0.FetchError("down")

        def get(self, url, byte_range=None):
            raise g0.FetchError(f"{url}: timed out")

    code, out, err, _ = run(capsys, tmp_path, Down(), "--stage", "fetch", days=1)
    assert code == 1
    assert "fault" in (out + err)
    assert RecordStore(tmp_path / "cache" / "g0" / "records").dates() == set()


def test_the_budget_stops_the_fetch_stage_politely(capsys, tmp_path) -> None:
    code, out, _, _ = run(
        capsys,
        tmp_path,
        served(dates(EVENING, 3)),
        "--stage",
        "fetch",
        "--budget-seconds",
        "0",
        days=3,
    )
    assert code == 1  # dates are left to do
    assert "stopped at the time budget" in out
    assert RecordStore(tmp_path / "cache" / "g0" / "records").dates() == set()


# --- usage -------------------------------------------------------------------------------------


def test_an_end_before_the_start_is_a_usage_error(capsys, tmp_path) -> None:
    code = main(
        ["g0", "--start", "2026-03-02", "--end", "2026-03-01", "--out", str(tmp_path)],
        commands=[command(NoNetwork())],
    )
    assert code == 2
    assert "end" in capsys.readouterr().err


def test_a_malformed_date_is_a_usage_error(capsys, tmp_path) -> None:
    code = main(
        ["g0", "--start", "20260301", "--end", "2026-03-02", "--out", str(tmp_path)],
        commands=[command(NoNetwork())],
    )
    assert code == 2


def test_a_period_longer_than_two_years_is_refused(capsys, tmp_path) -> None:
    code = main(
        ["g0", "--start", "2020-01-01", "--end", "2026-01-01", "--out", str(tmp_path)],
        commands=[command(NoNetwork())],
    )
    assert code == 2
