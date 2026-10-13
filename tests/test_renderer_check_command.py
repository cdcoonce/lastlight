"""`lastlight renderer-check`: the pre-registered camera magnitude check, end to end.

The report stage is exercised with input records written straight into the check's store and a
transport that fails the test on any request, so what these tests pin is the criterion's
definitions: the frames used, the 3-frame minimum, the cloud filter, the first 5, the bounds, the
refusals and the outputs. The fetch stage is exercised against served copies of the recorded
HRRR and GEFS files.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import io
import json
import math
import re
from pathlib import Path

import pytest
from hrrr_support import FakeTransport
from test_renderer_check_support import (
    CRITERION,
    NOW,
    SUNS,
    NoNetwork,
    expected_rendered,
    observed_for_ratio,
    rendered_inputs,
    save_inputs,
    served_run,
    write_observed,
)

from lastlight.cli import Command, discover, main
from lastlight.commands import g0, renderer_check
from lastlight.privacy.scanner import scan
from lastlight.renderer_check import guards, store

D0 = dt.date(2026, 10, 10)
AOD, OZONE = 0.05, 280.0


def day(n: int) -> dt.date:
    """The n-th evening after the first registered one (day(0) is 2026-10-10)."""
    return D0 + dt.timedelta(days=n)


def command(transport=None, now=NOW) -> Command:
    transport = NoNetwork() if transport is None else transport

    def run(args: argparse.Namespace) -> int:
        return renderer_check.execute(args, transport=transport, now_utc=now)

    return dataclasses.replace(renderer_check.COMMAND, run=run)


class Rig:
    """A temporary observed directory, cache and output directory, and a way to run the check."""

    def __init__(self, tmp_path: Path, capsys) -> None:
        self.observed = tmp_path / "observed"
        self.cache = tmp_path / "cache"
        self.out = tmp_path / "out"
        self.capsys = capsys

    def evening(
        self,
        n: int,
        *,
        ratio: float | None = 1.0,
        cloud: float | None = 0.0,
        aod: float = AOD,
        record: str | bool = True,
        triples=None,
    ) -> dt.date:
        """One evening: an observed file whose v1 r_e is `ratio` and an input record.

        `record` is True (a rendered record with `cloud`), False (no record: pending), or the
        code of a drop reason.
        """
        date = day(n)
        if triples is None:
            triples = observed_for_ratio(date, ratio, aod=aod, ozone=OZONE)
        write_observed(self.observed, date, triples)
        if record is True:
            save_inputs(self.cache, rendered_inputs(date, cloud=cloud, aod=aod, ozone=OZONE))
        elif record:
            save_inputs(self.cache, store.EveningInputs.dropped(date, record, "test"))
        return date

    def run(self, *extra, transport=None, now=NOW, out=None, default_stage="report"):
        """Run the check; the report stage unless a stage is named (None: the command's own)."""
        if default_stage and "--stage" not in extra:
            extra = (*extra, "--stage", default_stage)
        argv = [
            "renderer-check",
            "--observed-dir",
            str(self.observed),
            "--out",
            str(out or self.out),
            "--cache-dir",
            str(self.cache),
            *extra,
        ]
        code = main(argv, commands=[command(transport, now)])
        captured = self.capsys.readouterr()
        return code, captured.out, captured.err

    def text(self, name: str, out=None) -> str:
        return ((out or self.out) / name).read_text(encoding="utf-8")

    def rows(self, name: str, out=None) -> list[dict[str, str]]:
        return list(csv.DictReader(io.StringIO(self.text(name, out))))

    def evenings(self, out=None) -> dict[str, dict[str, str]]:
        return {row["date"]: row for row in self.rows("evenings.csv", out)}


@pytest.fixture
def rig(tmp_path, capsys) -> Rig:
    return Rig(tmp_path, capsys)


def verdict_line(rig: Rig) -> str:
    lines = [ln for ln in rig.text("report.md").splitlines() if ln.startswith("Verdict: ")]
    assert len(lines) == 1, lines
    return lines[0].removeprefix("Verdict: ")


# --- registration ---------------------------------------------------------------------------------


def test_the_command_is_discovered_by_name() -> None:
    assert "renderer-check" in {c.name for c in discover()}


def test_help_lists_the_options_and_there_is_no_camera_argument(capsys) -> None:
    assert main(["renderer-check", "--help"], commands=[renderer_check.COMMAND]) == 0
    out = capsys.readouterr().out
    for option in ("--observed-dir", "--out", "--cache-dir", "--stage"):
        assert option in out
    assert "--camera" not in out


def test_the_observed_dir_and_out_are_required(capsys) -> None:
    for argv in (["renderer-check", "--out", "x"], ["renderer-check", "--observed-dir", "x"]):
        assert main(argv, commands=[renderer_check.COMMAND]) == 2
        capsys.readouterr()


def test_the_default_cache_is_g0s_default() -> None:
    assert renderer_check.DEFAULT_CACHE_DIR == g0.DEFAULT_CACHE_DIR


def test_the_default_stage_is_all(capsys) -> None:
    holder: dict = {}

    def run(args: argparse.Namespace) -> int:
        holder.update(vars(args))
        return 0

    spy = dataclasses.replace(renderer_check.COMMAND, run=run)
    main(["renderer-check", "--observed-dir", "a", "--out", "b"], commands=[spy])
    assert holder["stage"] == "all"
    assert holder["cache_dir"] == renderer_check.DEFAULT_CACHE_DIR


# --- evenings considered: the 3-frame minimum and the first evening -------------------------------


def test_exactly_three_used_frames_is_considered_and_two_is_not(rig: Rig) -> None:
    three = observed_for_ratio(day(0), 1.0)[:3]
    two = observed_for_ratio(day(1), 1.0)[:2]
    rig.evening(0, triples=three)
    rig.evening(1, triples=two)
    code, out, err = rig.run()
    assert code == 0, err
    rows = rig.evenings()
    assert rows["2026-10-10"]["status"] == "counted"
    assert rows["2026-10-10"]["frames_used"] == "3"
    assert rows["2026-10-11"]["status"] == "not_considered"
    assert rows["2026-10-11"]["frames_used"] == "2"
    assert "not considered: 2 used frames" in rig.text("report.md")
    assert verdict_line(rig) == "NOT YET DECIDABLE (1 of 5 counted evenings)"


def test_a_frame_with_a_null_log_bg_is_not_used_so_four_frames_may_be_two(rig: Rig) -> None:
    triples = observed_for_ratio(day(0), 1.0)
    triples = [triples[0], triples[1], (0.0, None, 0.1), (-1.5, None, 0.1)]
    rig.evening(0, triples=triples)
    code, _, _ = rig.run()
    assert code == 0
    row = rig.evenings()["2026-10-10"]
    assert row["status"] == "not_considered" and row["frames_used"] == "2"


def test_the_light_dome_flag_does_not_remove_a_frame_from_the_count(rig: Rig, tmp_path) -> None:
    date = day(0)
    triples = observed_for_ratio(date, 1.0)[:3]
    write_observed(rig.observed, date, triples)
    path = rig.observed / f"{date.isoformat()}.json"
    document = path.read_text(encoding="utf-8").replace(
        '"light_dome_excluded": false', '"light_dome_excluded": true'
    )
    path.write_text(document, encoding="utf-8")
    save_inputs(rig.cache, rendered_inputs(date, cloud=0.0))
    rig.run()
    assert rig.evenings()["2026-10-10"]["status"] == "counted"


def test_an_evening_before_the_first_registered_one_is_not_considered(rig: Rig) -> None:
    earlier = day(-1)
    write_observed(rig.observed, earlier, observed_for_ratio(earlier, 1.0))
    save_inputs(rig.cache, rendered_inputs(earlier, cloud=0.0))
    rig.evening(0)
    code, _, _ = rig.run()
    assert code == 0
    rows = rig.evenings()
    assert rows["2026-10-09"]["status"] == "not_considered"
    assert rows["2026-10-09"]["reason"] == "before_first_evening"
    assert rows["2026-10-10"]["status"] == "counted"
    assert verdict_line(rig) == "NOT YET DECIDABLE (1 of 5 counted evenings)"


# --- the cloud filter -----------------------------------------------------------------------------


def test_exactly_ten_percent_counts_and_a_hair_over_does_not(rig: Rig) -> None:
    rig.evening(0, cloud=10.0)
    rig.evening(1, cloud=10.01)
    rig.evening(2, cloud=0.0)
    code, _, _ = rig.run()
    assert code == 0
    rows = rig.evenings()
    assert rows["2026-10-10"]["status"] == "counted"
    assert rows["2026-10-11"]["status"] == "not_counted"
    assert rows["2026-10-12"]["status"] == "counted"
    assert verdict_line(rig) == "NOT YET DECIDABLE (2 of 5 counted evenings)"


def test_a_not_counted_evening_is_reported_with_its_cloud_and_its_ratios(rig: Rig) -> None:
    rig.evening(0, cloud=42.5, ratio=3.0)
    code, _, _ = rig.run()
    assert code == 0
    row = rig.evenings()["2026-10-10"]
    assert row["status"] == "not_counted" and row["reason"] == "cloud"
    assert float(row["cloud_mean_pct"]) == pytest.approx(42.5)
    assert float(row["r_e_v1_bg"]) == pytest.approx(3.0, rel=1e-4)  # still computed, reported
    assert row["slot"] == ""
    assert "cloud 42.5" in rig.text("report.md")
    assert verdict_line(rig) == "NOT YET DECIDABLE (0 of 5 counted evenings)"


# --- the ratios -----------------------------------------------------------------------------------


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    return ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2


def observed_values(rig: Rig, date: dt.date):
    """([log_rg], [log_bg]) of the used frames as written in the observed file, in file order."""
    document = json.loads((rig.observed / f"{date.isoformat()}.json").read_text(encoding="utf-8"))
    bands = [frame["bands"][0] for frame in document["frames"]]
    return [b["log_rg"] for b in bands], [b["log_bg"] for b in bands]


def test_the_reported_ratios_are_what_the_definitions_say(rig: Rig) -> None:
    rig.evening(0, ratio=1.7)
    assert rig.run()[0] == 0
    row = rig.evenings()["2026-10-10"]
    date = day(0)

    def r_e(engine, channel, scale="nominal"):
        rendered = expected_rendered(engine, date, AOD, OZONE, SUNS, scale)[:, channel]
        observed = observed_values(rig, date)[channel]
        return _median([abs(x) for x in rendered]) / _median([abs(x) for x in observed])

    assert float(row["r_e_v1_bg"]) == pytest.approx(r_e("v1", 1), rel=1e-4)
    assert float(row["r_e_v0_bg"]) == pytest.approx(r_e("v0", 1), rel=1e-4)
    assert float(row["r_e_v1_bg_scale_low"]) == pytest.approx(r_e("v1", 1, "low"), rel=1e-4)
    assert float(row["r_e_v1_bg_scale_high"]) == pytest.approx(r_e("v1", 1, "high"), rel=1e-4)
    assert float(row["r_e_v1_rg"]) == pytest.approx(r_e("v1", 0), rel=1e-4)
    assert float(row["r_e_v0_rg"]) == pytest.approx(r_e("v0", 0), rel=1e-4)
    assert float(row["r_e_v1_bg"]) == pytest.approx(1.7, rel=1e-4)
    assert row["aod_550"] == f"{AOD:.4f}"


def test_a_zero_observed_median_makes_the_ratio_infinite_and_it_still_takes_part(rig: Rig) -> None:
    zero = [(sun, 0.0, 0.0) for sun in SUNS]
    rig.evening(0, triples=zero)
    rig.evening(1, triples=zero)
    for n in (2, 3, 4):
        rig.evening(n, ratio=1.0)
    code, _, _ = rig.run()
    assert code == 0
    rows = rig.evenings()
    assert rows["2026-10-10"]["r_e_v1_bg"] == "inf"
    assert math.isinf(float(rows["2026-10-10"]["r_e_v1_bg"]))
    assert "inf" in rig.text("report.md")
    assert verdict_line(rig) == "ADOPT"  # [inf, inf, 1, 1, 1]: the median is 1


def test_three_infinite_ratios_among_the_first_five_keep_v0(rig: Rig) -> None:
    zero = [(sun, 0.0, 0.0) for sun in SUNS]
    for n in (0, 1, 2):
        rig.evening(n, triples=zero)
    for n in (3, 4):
        rig.evening(n, ratio=1.0)
    rig.run()
    assert verdict_line(rig) == "KEEP V0"


# --- the verdict ----------------------------------------------------------------------------------


def test_five_counted_evenings_inside_the_window_adopt(rig: Rig) -> None:
    for n in range(5):
        rig.evening(n, ratio=1.0)
    code, out, _ = rig.run()
    assert code == 0
    assert verdict_line(rig) == "ADOPT"
    assert "ADOPT" in out
    slots = [rig.evenings()[day(n).isoformat()]["slot"] for n in range(5)]
    assert slots == ["1", "2", "3", "4", "5"]


def test_the_median_of_the_first_five_decides_not_the_mean(rig: Rig) -> None:
    for n, ratio in enumerate([1.0, 1.0, 1.0, 1.0, 50.0]):
        rig.evening(n, ratio=ratio)
    rig.run()
    assert verdict_line(rig) == "ADOPT"


def test_a_median_below_the_window_keeps_v0(rig: Rig) -> None:
    for n, ratio in enumerate([0.1, 0.1, 0.1, 3.0, 3.0]):
        rig.evening(n, ratio=ratio)
    rig.run()
    assert verdict_line(rig) == "KEEP V0"


@pytest.mark.parametrize(
    ("ratio", "verdict"),
    [(0.5001, "ADOPT"), (0.4999, "KEEP V0"), (1.9999, "ADOPT"), (2.0001, "KEEP V0")],
)
def test_the_window_edges_are_the_criterions(rig: Rig, ratio: float, verdict: str) -> None:
    for n in range(3):
        rig.evening(n, ratio=ratio)
    for n in (3, 4):
        rig.evening(n, ratio=1.0)
    rig.run()
    assert verdict_line(rig) == verdict


def test_only_the_five_earliest_counted_evenings_decide(rig: Rig) -> None:
    for n in range(5):
        rig.evening(n, ratio=1.0)
    for n in (5, 6):
        rig.evening(n, ratio=9.0)  # would flip a median taken over seven
    rig.run()
    assert verdict_line(rig) == "ADOPT"
    rows = rig.evenings()
    assert rows["2026-10-15"]["status"] == "counted" and rows["2026-10-15"]["slot"] == ""
    assert "after the first 5" in rig.text("report.md")


def test_later_evenings_never_change_a_keep_either(rig: Rig) -> None:
    for n in range(5):
        rig.evening(n, ratio=9.0)
    for n in (5, 6, 7):
        rig.evening(n, ratio=1.0)
    rig.run()
    assert verdict_line(rig) == "KEEP V0"


def test_a_not_counted_evening_never_takes_a_slot(rig: Rig) -> None:
    rig.evening(0, ratio=9.0, cloud=60.0)  # cloudy: reported, not counted
    rig.evening(1, ratio=9.0)
    rig.evening(2, ratio=9.0)
    for n in (3, 4, 5, 6):
        rig.evening(n, ratio=1.0)
    rig.run()
    # Counted: days 1..6 -> first five are [9, 9, 1, 1, 1], median 1.
    assert verdict_line(rig) == "ADOPT"
    assert rig.evenings()["2026-10-10"]["slot"] == ""


def test_a_dropped_evening_never_takes_a_slot(rig: Rig) -> None:
    rig.evening(0, record="aerosol_unreadable")
    rig.evening(1, ratio=9.0)
    rig.evening(2, ratio=9.0)
    for n in (3, 4, 5, 6):
        rig.evening(n, ratio=1.0)
    rig.run()
    assert verdict_line(rig) == "ADOPT"
    row = rig.evenings()["2026-10-10"]
    assert row["status"] == "dropped" and row["reason"] == "aerosol_unreadable"
    assert row["r_e_v1_bg"] == ""
    assert "dropped, aerosol_unreadable" in rig.text("report.md")


def test_an_evening_outside_the_v1_table_is_dropped_with_its_reason(rig: Rig) -> None:
    flat = [(sun, -0.1, 0.1) for sun in SUNS]
    rig.evening(0, aod=1.5, triples=flat)  # beyond the table's aod_550 of 1.0
    rig.evening(1, ratio=1.0)
    rig.run()
    row = rig.evenings()["2026-10-10"]
    assert row["status"] == "dropped" and row["reason"] == "v1_outside_table"
    assert verdict_line(rig) == "NOT YET DECIDABLE (1 of 5 counted evenings)"


def test_a_frame_outside_the_v1_table_drops_the_evening(rig: Rig) -> None:
    triples = observed_for_ratio(day(0), 1.0)
    triples.append((13.5, -0.1, 0.1))  # the table stops at a sun elevation of 12
    rig.evening(0, triples=triples)
    rig.run()
    row = rig.evenings()["2026-10-10"]
    assert row["status"] == "dropped" and row["reason"] == "v1_outside_table"


# --- pending evenings -----------------------------------------------------------------------------


def test_an_evening_without_an_input_record_is_pending_and_the_exit_code_says_so(rig: Rig) -> None:
    rig.evening(0, ratio=1.0)
    rig.evening(1, record=False)
    code, out, err = rig.run()
    assert code == 1
    assert (rig.out / "report.md").is_file()
    row = rig.evenings()["2026-10-11"]
    assert row["status"] == "pending"
    assert verdict_line(rig) == "NOT YET DECIDABLE (1 of 5 counted evenings)"
    assert "2026-10-11" in err


def test_a_pending_evening_blocks_later_evenings_from_taking_a_first_five_slot(rig: Rig) -> None:
    rig.evening(0, record=False)  # may still turn out to be counted, and then it is first
    for n in range(1, 7):
        rig.evening(n, ratio=1.0)
    code, _, _ = rig.run()
    assert code == 1
    assert verdict_line(rig) == "NOT YET DECIDABLE (0 of 5 counted evenings)"
    rows = rig.evenings()
    assert rows["2026-10-11"]["status"] == "counted" and rows["2026-10-11"]["slot"] == ""


def test_a_pending_evening_after_four_counted_leaves_four(rig: Rig) -> None:
    for n in range(4):
        rig.evening(n, ratio=1.0)
    rig.evening(4, record=False)
    rig.evening(5, ratio=1.0)
    rig.run()
    assert verdict_line(rig) == "NOT YET DECIDABLE (4 of 5 counted evenings)"


def test_the_report_stage_never_touches_the_network(rig: Rig) -> None:
    rig.evening(0, record=False)
    code, _, _ = rig.run("--stage", "report", transport=NoNetwork())
    assert code == 1


# A run missing from an archive may only be a publication lag, so the fetch stage keeps no record
# of it while the evening is less than two calendar days old: the evening stays pending, exactly
# like an analysis not published yet, and the first-5 rule holds the later evenings' slots. The
# report stage has no clock; it only ever sees final records or none.

MISSING_RUN_REASONS = ("hrrr_run_missing", "aerosol_run_missing")


def clock(evening: dt.date, days_after: int) -> dt.datetime:
    """A moment `days_after` calendar days after `evening` (the archives are long published)."""
    return dt.datetime.combine(evening + dt.timedelta(days=days_after), dt.time(12), tzinfo=dt.UTC)


def archives_without(reason: str, evening: dt.date, now: dt.datetime) -> FakeTransport:
    """The archives of `evening` as served at `now`, except the run that `reason` names."""
    served = served_run(evening, now, tcdc=5.0)
    missing = "gefs" if reason == "aerosol_run_missing" else "hrrr"
    return FakeTransport({url: body for url, body in served.files.items() if missing not in url})


def around_a_missing_run(rig: Rig) -> dt.date:
    """Days 0 to 3 counted (r_e 1, 1, 9, 9), day 4 observed with no record, day 5 counted (r_e 9).

    With day 4 out, the first five are [1, 1, 9, 9, 9] (KEEP V0); with day 4 counted at r_e of
    about 1 they are [1, 1, 9, 9, 1] (ADOPT): a verdict given too early would flip.
    """
    for n, ratio in enumerate([1.0, 1.0, 9.0, 9.0]):
        rig.evening(n, ratio=ratio)
    missing = rig.evening(4, ratio=1.0, record=False)
    rig.evening(5, ratio=9.0)
    return missing


def stored_record(rig: Rig, evening: dt.date) -> dict | None:
    path = rig.cache / "renderer-v1" / f"{evening.isoformat()}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


@pytest.mark.parametrize("reason", MISSING_RUN_REASONS)
def test_a_recent_missing_run_drop_holds_the_slots_of_later_evenings(rig: Rig, reason: str) -> None:
    missing = around_a_missing_run(rig)
    opened = clock(missing, 1)
    code, _, err = rig.run(
        transport=archives_without(reason, missing, opened), now=opened, default_stage=None
    )
    # No record: the evening is pending, as if its analysis were not published yet.
    assert code == 1 and "2026-10-14" in err
    assert stored_record(rig, missing) is None
    rows = rig.evenings()
    assert rows["2026-10-14"]["status"] == "pending" and rows["2026-10-14"]["slot"] == ""
    # The later counted evening holds no slot, so no verdict is given that a retry could flip.
    assert rows["2026-10-15"]["status"] == "counted" and rows["2026-10-15"]["slot"] == ""
    assert "counted, waiting for an earlier evening's inputs" in rig.text("report.md")
    assert verdict_line(rig) == "NOT YET DECIDABLE (4 of 5 counted evenings)"

    # The archive publishes the run. The next run records the evening, and it takes slot 5.
    code, _, err = rig.run(
        transport=served_run(missing, opened, tcdc=5.0), now=opened, default_stage=None
    )
    assert code == 0, err
    rows = rig.evenings()
    assert rows["2026-10-14"]["status"] == "counted" and rows["2026-10-14"]["slot"] == "5"
    assert rows["2026-10-15"]["slot"] == ""
    assert verdict_line(rig) == "ADOPT"


@pytest.mark.parametrize("reason", MISSING_RUN_REASONS)
def test_a_missing_run_drop_is_stored_as_final_once_the_window_has_passed(
    rig: Rig, reason: str
) -> None:
    missing = around_a_missing_run(rig)
    opened = clock(missing, 1)
    rig.run(transport=archives_without(reason, missing, opened), now=opened, default_stage=None)
    assert stored_record(rig, missing) is None  # still inside the window: nothing kept

    closed = clock(missing, 2)
    code, _, err = rig.run(
        transport=archives_without(reason, missing, closed), now=closed, default_stage=None
    )
    assert code == 0, err
    record = stored_record(rig, missing)
    assert record is not None and record["drop_reason"] == reason
    rows = rig.evenings()
    assert rows["2026-10-14"]["status"] == "dropped" and rows["2026-10-14"]["reason"] == reason
    assert rows["2026-10-15"]["status"] == "counted" and rows["2026-10-15"]["slot"] == "5"
    assert verdict_line(rig) == "KEEP V0"

    # Final: the run turning up later changes nothing and is not even asked for.
    published = served_run(missing, closed, tcdc=5.0)
    code, _, _ = rig.run(transport=published, now=closed, default_stage=None)
    assert code == 0 and published.calls == []
    assert stored_record(rig, missing) == record
    assert verdict_line(rig) == "KEEP V0"


# --- refusals -------------------------------------------------------------------------------------


def test_a_table_whose_bytes_differ_is_refused_and_nothing_is_written(
    rig: Rig, monkeypatch
) -> None:
    rig.evening(0)
    real = guards.read_table_bytes()
    monkeypatch.setattr(guards, "read_table_bytes", lambda: real[:-1] + b"\x00")
    code, out, err = rig.run()
    assert code == 3
    assert not rig.out.exists()
    assert len(err.strip().splitlines()) == 1 and "SHA-256" in err
    assert hashlib.sha256(real[:-1] + b"\x00").hexdigest() in err
    assert "verdict" not in out.lower()


def test_a_changed_v1_fingerprint_is_refused_and_nothing_is_written(rig: Rig, monkeypatch) -> None:
    rig.evening(0)
    real = guards.probe_xyz

    def drifted(engine: str):
        values = real(engine)
        return values * 1.001 if engine == "v1" else values

    monkeypatch.setattr(guards, "probe_xyz", drifted)
    code, out, err = rig.run()
    assert code == 3
    assert not rig.out.exists()
    assert len(err.strip().splitlines()) == 1 and "v1" in err
    assert "verdict" not in out.lower()


def test_a_refusal_in_the_all_stage_happens_before_any_fetch_or_cache_write(
    rig: Rig, monkeypatch
) -> None:
    write_observed(rig.observed, day(0), observed_for_ratio(day(0), 1.0))
    monkeypatch.setattr(guards, "read_table_bytes", lambda: b"another table")
    transport = NoNetwork()  # a request would raise AssertionError
    code, _, err = rig.run(transport=transport, default_stage=None)
    assert code == 3
    assert not rig.out.exists() and not rig.cache.exists()


def test_a_v0_probe_difference_is_not_a_refusal(rig: Rig, monkeypatch) -> None:
    rig.evening(0)
    real = guards.probe_xyz
    monkeypatch.setattr(
        guards, "probe_xyz", lambda engine: real(engine) * (1.5 if engine == "v0" else 1.0)
    )
    code, _, _ = rig.run()
    assert code == 0


@pytest.mark.parametrize(
    ("overrides", "needle"),
    [
        ({"camera_id": "OTHER9"}, "camera_id"),
        ({"schema_version": 2}, "schema_version"),
        ({"date": "2026-10-11"}, "date"),
    ],
)
def test_a_refused_observed_file_is_exit_2_with_one_line_and_nothing_written(
    rig: Rig, overrides, needle: str
) -> None:
    write_observed(rig.observed, day(0), observed_for_ratio(day(0), 1.0), **overrides)
    code, _, err = rig.run()
    assert code == 2
    assert not rig.out.exists()
    lines = err.strip().splitlines()
    assert len(lines) == 1 and "2026-10-10.json" in lines[0] and needle in lines[0]


def test_a_missing_observed_directory_is_exit_2(rig: Rig) -> None:
    code, _, err = rig.run()
    assert code == 2 and not rig.out.exists()


# --- the outputs ----------------------------------------------------------------------------------


def seven(rig: Rig) -> None:
    for n in range(4):
        rig.evening(n, ratio=1.0, aod=0.04 + 0.01 * n)
    rig.evening(4, ratio=3.0, cloud=55.0)
    rig.evening(5, record="hrrr_field_missing")
    rig.evening(6, record=False)


def test_the_three_outputs_are_written(rig: Rig) -> None:
    seven(rig)
    code, out, _ = rig.run()
    assert code == 1  # the last evening is pending
    assert sorted(p.name for p in rig.out.iterdir()) == ["evenings.csv", "frames.csv", "report.md"]
    assert "NOT YET DECIDABLE (4 of 5 counted evenings)" in out


def test_a_rerun_is_byte_identical(rig: Rig, tmp_path) -> None:
    seven(rig)
    rig.run()
    first = {n: (rig.out / n).read_bytes() for n in ("report.md", "evenings.csv", "frames.csv")}
    rig.run()
    again = {n: (rig.out / n).read_bytes() for n in first}
    other = tmp_path / "elsewhere"
    rig.run(out=other)
    elsewhere = {n: (other / n).read_bytes() for n in first}
    assert first == again == elsewhere


def test_the_outputs_pass_the_privacy_scanner_and_carry_nothing_private(rig: Rig, tmp_path) -> None:
    seven(rig)
    rig.run()
    result = scan([rig.out], root=rig.out)
    assert result.findings == [], [str(f) for f in result.findings]
    everything = "".join(rig.text(n) for n in ("report.md", "evenings.csv", "frames.csv"))
    assert str(tmp_path) not in everything
    assert "0123456789ab" not in everything  # the observed frames' hash prefix
    assert not re.search(r"\b\d{1,2}:\d{2}\b", everything)  # no clock time
    assert not re.search(r"\b\d{1,2}Z\b", everything)  # no UTC hour
    for hexdigits in re.findall(r"\b[0-9a-f]{40,64}\b", everything):
        assert hexdigits == guards.REGISTERED_TABLE_SHA256  # the table's, nothing else


def test_the_report_states_the_verdict_the_digest_the_first_five_and_each_evening(
    rig: Rig,
) -> None:
    seven(rig)
    rig.run()
    report = rig.text("report.md")
    assert report.splitlines()[0].startswith("# ")
    assert verdict_line(rig) == "NOT YET DECIDABLE (4 of 5 counted evenings)"
    assert guards.REGISTERED_TABLE_SHA256 in report
    for n in range(7):
        assert day(n).isoformat() in report
    assert "cloud 55.0" in report  # not counted, with its cloud
    assert "dropped, hrrr_field_missing" in report
    assert "pending" in report
    assert "4 of 5" in report


def test_the_cannot_say_paragraph_is_linked_and_not_copied(rig: Rig) -> None:
    rig.evening(0)
    rig.run()
    report = rig.text("report.md")
    assert "criterion.md#what-this-check-cannot-say" in report
    paragraph = [
        line
        for line in CRITERION.read_text(encoding="utf-8").splitlines()
        if line.startswith("The webcam's tone curve")
    ]
    assert paragraph and paragraph[0] not in report
    assert "tone curve" not in report


def test_the_evenings_file_lists_every_observed_evening_sorted_by_date(rig: Rig) -> None:
    seven(rig)
    write_observed(rig.observed, day(-3), observed_for_ratio(day(-3), 1.0)[:2])
    rig.run()
    dates = [row["date"] for row in rig.rows("evenings.csv")]
    assert dates == sorted(dates) and len(dates) == 8
    header = rig.text("evenings.csv").splitlines()[0].split(",")
    assert header[:3] == ["date", "status", "reason"]


def test_the_frames_file_has_one_row_per_used_frame_of_each_considered_evening(rig: Rig) -> None:
    rig.evening(0, ratio=1.0)
    rig.evening(1, record=False)  # considered, pending: observed values only
    short = observed_for_ratio(day(2), 1.0)[:2]
    rig.evening(2, triples=short)  # not considered: no rows
    rig.run()
    rows = rig.rows("frames.csv")
    assert {r["date"] for r in rows} == {"2026-10-10", "2026-10-11"}
    first = [r for r in rows if r["date"] == "2026-10-10"]
    assert [r["frame"] for r in first] == ["1", "2", "3", "4"]
    assert [float(r["sun_elevation_deg"]) for r in first] == list(SUNS)
    assert all(re.fullmatch(r"-?\d+\.\d\d", r["sun_elevation_deg"]) for r in rows)
    want_v1 = expected_rendered("v1", day(0), AOD, OZONE, SUNS)
    want_v0 = expected_rendered("v0", day(0), AOD, OZONE, SUNS)
    for r, v1_row, v0_row in zip(first, want_v1, want_v0, strict=True):
        assert float(r["rendered_log_bg_v1"]) == pytest.approx(v1_row[1], abs=1e-4)
        assert float(r["rendered_log_rg_v1"]) == pytest.approx(v1_row[0], abs=1e-4)
        assert float(r["rendered_log_bg_v0"]) == pytest.approx(v0_row[1], abs=1e-4)
        assert float(r["rendered_log_rg_v0"]) == pytest.approx(v0_row[0], abs=1e-4)
    pending = [r for r in rows if r["date"] == "2026-10-11"]
    assert len(pending) == 4 and all(r["rendered_log_bg_v1"] == "" for r in pending)
    assert all(r["observed_log_bg"] != "" for r in pending)


def test_a_used_frame_is_published_by_sun_elevation_never_by_name_or_hash(rig: Rig) -> None:
    rig.evening(0)
    rig.run()
    header = rig.text("frames.csv").splitlines()[0]
    assert header == (
        "date,frame,sun_elevation_deg,observed_log_bg,rendered_log_bg_v0,rendered_log_bg_v1,"
        "observed_log_rg,rendered_log_rg_v0,rendered_log_rg_v1"
    )


def test_the_exit_code_is_zero_when_every_considered_evening_has_a_record(rig: Rig) -> None:
    rig.evening(0)
    rig.evening(1, record="tcdc_missing")
    rig.evening(2, cloud=99.0)
    assert rig.run()[0] == 0


# --- the fetch stage against served archives ------------------------------------------------------


def test_the_all_stage_fetches_then_reports_and_writes_only_the_checks_own_store(
    rig: Rig, tmp_path
) -> None:
    date = day(0)
    write_observed(rig.observed, date, observed_for_ratio(date, 1.0))
    transport = served_run(date, NOW, tcdc=5.0)
    code, out, err = rig.run(transport=transport, default_stage=None)
    assert code == 0, err
    assert (rig.cache / "renderer-v1" / "2026-10-10.json").is_file()
    assert not (rig.cache / "g0").exists()  # G0's store is never written by this command
    assert sorted(p.name for p in rig.cache.iterdir()) == ["gefs", "hrrr", "renderer-v1"]
    row = rig.evenings()["2026-10-10"]
    assert row["status"] == "counted" and float(row["cloud_mean_pct"]) == pytest.approx(
        5.0, abs=1e-3
    )
    assert "2026-10-10" in out


def test_the_cloudy_recorded_crop_is_not_counted(rig: Rig) -> None:
    date = day(0)
    write_observed(rig.observed, date, observed_for_ratio(date, 1.0))
    code, _, _ = rig.run(
        transport=served_run(date, NOW), default_stage=None
    )  # recorded TCDC is cloudy
    assert code == 0
    assert rig.evenings()["2026-10-10"]["status"] == "not_counted"


def test_the_fetch_stage_alone_writes_the_cache_and_nothing_to_out(rig: Rig) -> None:
    date = day(0)
    write_observed(rig.observed, date, observed_for_ratio(date, 1.0))
    code, _, _ = rig.run("--stage", "fetch", transport=served_run(date, NOW, tcdc=5.0))
    assert code == 0
    assert (rig.cache / "renderer-v1" / "2026-10-10.json").is_file()
    assert not rig.out.exists()


def test_the_fetch_stage_leaves_a_pending_evening_unrecorded_and_exits_1(rig: Rig) -> None:
    date = day(0)
    write_observed(rig.observed, date, observed_for_ratio(date, 1.0))
    from lastlight.renderer_check import fetch

    plan = fetch.analysis_plan(date, NOW)
    cycle = dt.datetime.combine(plan.cycle_date, dt.time(plan.cycle_hour), tzinfo=dt.UTC)
    early = cycle + dt.timedelta(hours=1)
    transport = served_run(date, NOW, tcdc=5.0)
    code, _, err = rig.run("--stage", "fetch", transport=transport, now=early)
    assert code == 1
    assert not (rig.cache / "renderer-v1" / "2026-10-10.json").exists()
    assert transport.calls == []
    assert "2026-10-10" in err


def test_only_considered_evenings_are_fetched(rig: Rig) -> None:
    short, early = day(1), day(-2)
    write_observed(rig.observed, short, observed_for_ratio(short, 1.0)[:2])
    write_observed(rig.observed, early, observed_for_ratio(early, 1.0))
    transport = served_run(day(0), NOW, tcdc=5.0)
    code, _, _ = rig.run("--stage", "fetch", transport=transport)
    assert code == 0
    assert transport.calls == []
    assert not rig.cache.exists()


def test_a_second_fetch_makes_no_new_request(rig: Rig) -> None:
    date = day(0)
    write_observed(rig.observed, date, observed_for_ratio(date, 1.0))
    transport = served_run(date, NOW, tcdc=5.0)
    rig.run("--stage", "fetch", transport=transport)
    calls = len(transport.calls)
    code, _, _ = rig.run("--stage", "fetch", transport=transport)
    assert code == 0 and len(transport.calls) == calls
