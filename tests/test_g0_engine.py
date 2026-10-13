"""G0 with an engine: `rendered_contrasts(..., engine=)` and `lastlight g0 --engine v1`.

The cache holds v0-rendered contrasts, so the v1 re-run cannot just switch engines in the report
stage: each cached date is re-rendered from its `AtmosphereRecord` (aod_total and ozone_du, every
other `Atmosphere` field at its default, as G0 builds it) with v1, through the same seams, into a
sibling store (`g0-v1/records`). The v0 store, its report and its results are never touched, and
every v0 default stays byte for byte what it was.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
from g0_support import (
    LAT,
    LON,
    dropped_record,
    synthetic_record,
    whtm1_views,
)

from lastlight.cli import Command, main
from lastlight.commands import g0
from lastlight.g0 import analysis, collect, report
from lastlight.g0.contrast import day_contrasts, log_chromaticity, rendered_contrasts
from lastlight.g0.record import DROP_REASONS, RENDER_SCALES, RecordStore
from lastlight.g0.view import relative_azimuth, sun_azimuths
from lastlight.privacy.scanner import scan
from lastlight.render import Atmosphere, render_curve, sky_xyz

START = dt.date(2025, 10, 1)
NOW = dt.datetime(2026, 10, 12, tzinfo=dt.UTC)


class NoNetwork:
    """Any request is a bug: the v1 re-run reads the v0 cache and nothing else."""

    def exists(self, url: str) -> bool:
        raise AssertionError(f"the re-run asked for {url}")

    def get(self, url, byte_range=None):
        raise AssertionError(f"the re-run asked for {url}")


def command(transport=None) -> Command:
    transport = NoNetwork() if transport is None else transport

    def run(args: argparse.Namespace) -> int:
        return g0.execute(args, transport=transport, now_utc=NOW)

    return dataclasses.replace(g0.COMMAND, run=run)


def seed_v0(cache: Path) -> list[dt.date]:
    """Two rendered v0 dates, one dropped, and one date (the fourth) with no record at all."""
    store = RecordStore(cache / "g0" / "records")
    days = [START + dt.timedelta(days=i) for i in range(4)]
    store.save(synthetic_record(days[0], aod=0.08, shifts={"golden": 0.01}))
    store.save(synthetic_record(days[1], aod=0.03, shifts={"golden": 0.02}))
    store.save(dropped_record(days[2], "hrrr_run_missing", "HRRR: the file is not in the archive"))
    return days


def run(capsys, tmp_path, *extra, out="out", days=4):
    argv = [
        "g0",
        "--start",
        START.isoformat(),
        "--end",
        (START + dt.timedelta(days=days - 1)).isoformat(),
        "--out",
        str(tmp_path / out),
        "--cache-dir",
        str(tmp_path / "cache"),
        *extra,
    ]
    code = main(argv, commands=[command()])
    captured = capsys.readouterr()
    return code, captured.out, captured.err, tmp_path / out


def tree_digest(directory: Path) -> dict[str, str]:
    return {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def v1_store(tmp_path) -> RecordStore:
    return RecordStore(tmp_path / "cache" / "g0-v1" / "records")


# --- the seam: rendered_contrasts and day_contrasts take an engine --------------------------------


@pytest.fixture(scope="module")
def view():
    return whtm1_views()["nominal"]


ATM = Atmosphere(aod_550=0.05, ozone_du=280.0)


def test_the_default_engine_is_v0_and_unchanged(view) -> None:
    default, floored = rendered_contrasts(ATM, 0.0, view, 25.0)
    explicit, floored_explicit = rendered_contrasts(ATM, 0.0, view, 25.0, engine="v0")
    np.testing.assert_array_equal(default, explicit)
    assert floored == floored_explicit
    xyz = sky_xyz(ATM, 0.0, view.elevations_deg, 25.0)
    logs, _ = log_chromaticity(xyz)
    np.testing.assert_allclose(default, logs[:-1] - logs[-1], rtol=0, atol=0)


def test_engine_v1_goes_through_sky_xyz_with_that_engine(view) -> None:
    got, _ = rendered_contrasts(ATM, 0.0, view, 25.0, engine="v1")
    xyz = sky_xyz(ATM, 0.0, view.elevations_deg, 25.0, engine="v1")
    logs, _ = log_chromaticity(xyz)
    np.testing.assert_allclose(got, logs[:-1] - logs[-1], rtol=1e-12)
    v0, _ = rendered_contrasts(ATM, 0.0, view, 25.0)
    assert not np.allclose(got, v0)
    assert got[0, 1] > v0[0, 1] + 1.0  # v1's blue is far less negative than v0's at sun 0


def test_an_unknown_engine_is_an_error(view) -> None:
    with pytest.raises(ValueError, match="engine"):
        rendered_contrasts(ATM, 0.0, view, 25.0, engine="v2")


def test_day_contrasts_pass_the_engine_down_and_default_to_v0(view) -> None:
    azimuths = tuple(
        relative_azimuth(view.azimuth_deg, sun)
        for sun in sun_azimuths(dt.date(2026, 3, 1), LAT, LON)
    )
    default, floor_default = day_contrasts(ATM, view, azimuths)
    explicit, floor_explicit = day_contrasts(ATM, view, azimuths, engine="v0")
    v1, _ = day_contrasts(ATM, view, azimuths, engine="v1")
    np.testing.assert_array_equal(default, explicit)
    assert floor_default == floor_explicit
    assert v1.shape == default.shape == (29, 5, 2) and np.isfinite(v1).all()
    assert not np.allclose(v1, default)


# --- the re-render, by function -------------------------------------------------------------------


def test_a_v0_record_re_renders_from_its_atmosphere_record_only(tmp_path) -> None:
    v0 = synthetic_record(START, aod=0.08, shifts={"golden": 0.5})
    views = whtm1_views()
    v1 = collect.render_v1(v0, views)
    assert v1.ok and v1.date == START
    assert v1.atmosphere == v0.atmosphere
    atmosphere = Atmosphere(aod_550=v0.atmosphere.aod_total, ozone_du=v0.atmosphere.ozone_du)
    azimuths = tuple(
        relative_azimuth(views["nominal"].azimuth_deg, sun) for sun in sun_azimuths(START, LAT, LON)
    )
    for scale in RENDER_SCALES:
        want, floored = day_contrasts(atmosphere, views[scale], azimuths, engine="v1")
        np.testing.assert_allclose(v1.contrast[scale], want, rtol=1e-12)
        assert v1.floor_channels[scale] == floored
    rows = render_curve(atmosphere, engine="v1")
    lab = np.array([[r["L"], r["a"], r["b"]] for r in rows]).reshape(29, -1, 3)
    np.testing.assert_allclose(v1.curve_lab, lab, rtol=1e-12)
    assert not np.allclose(v1.contrast["nominal"], v0.contrast["nominal"])  # not the v0 numbers
    assert v1.inputs["engine"] == "v1" and v1.inputs["hrrr_cycle_date"] == START.isoformat()


def test_a_dropped_v0_record_stays_dropped_with_the_same_reason() -> None:
    v0 = dropped_record(START, "aerosol_unreadable", "GEFS-Aerosols: could not be read")
    v1 = collect.render_v1(v0, whtm1_views())
    assert not v1.ok
    assert (v1.drop_reason, v1.drop_detail) == (v0.drop_reason, v0.drop_detail)


def test_an_atmosphere_outside_the_table_is_dropped_with_its_own_reason() -> None:
    assert "v1_outside_table" in DROP_REASONS
    assert DROP_REASONS["v1_outside_table"].endswith(".")
    v1 = collect.render_v1(synthetic_record(START, aod=1.4), whtm1_views())
    assert not v1.ok and v1.drop_reason == "v1_outside_table"
    assert "aod_550" in v1.drop_detail


# --- the command: the fetch stage re-renders into the sibling store -------------------------------


def invoke(tmp_path: Path, *args: str) -> int:
    """Run `lastlight g0` over the four test dates against `tmp_path / "cache"`."""
    argv = [
        "g0",
        *args,
        "--start",
        START.isoformat(),
        "--end",
        (START + dt.timedelta(days=3)).isoformat(),
        "--cache-dir",
        str(tmp_path / "cache"),
    ]
    return main(argv, commands=[command()])


@pytest.fixture(scope="module")
def fetched(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("g0-v1")
    seed_v0(tmp_path / "cache")
    before = tree_digest(tmp_path / "cache")
    code = invoke(tmp_path, "--engine", "v1", "--stage", "fetch", "--out", str(tmp_path / "out"))
    return tmp_path, before, code


def test_the_fetch_stage_writes_only_the_sibling_store(fetched) -> None:
    tmp_path, before, _ = fetched
    cache = tmp_path / "cache"
    after = tree_digest(cache)
    unchanged = {k: v for k, v in after.items() if k.startswith("g0/")}
    assert unchanged == {k: v for k, v in before.items() if k.startswith("g0/")}
    new = sorted(k for k in after if not k.startswith("g0/"))
    assert new == [
        f"g0-v1/records/{(START + dt.timedelta(days=i)).isoformat()}.json" for i in range(3)
    ]
    assert not (tmp_path / "out").exists()


def test_the_fetch_stage_exit_code_says_a_date_has_no_v0_record(fetched) -> None:
    _, _, code = fetched
    assert code == 1  # the fourth date was never collected, so there is nothing to re-render


def test_the_v1_store_holds_v1_numbers_a_kept_drop_and_no_phantom(fetched) -> None:
    tmp_path, _, _ = fetched
    store = v1_store(tmp_path)
    first, second, third, fourth = (START + dt.timedelta(days=i) for i in range(4))
    for day in (first, second):
        record = store.load(day)
        assert record is not None and record.ok
        assert record.inputs["engine"] == "v1"
    dropped = store.load(third)
    assert dropped is not None and dropped.drop_reason == "hrrr_run_missing"
    assert store.load(fourth) is None
    v0 = RecordStore(tmp_path / "cache" / "g0" / "records").load(first)
    assert not np.allclose(store.load(first).contrast["nominal"], v0.contrast["nominal"])


def test_a_second_fetch_skips_every_date_and_renders_nothing(fetched, capsys, monkeypatch) -> None:
    tmp_path, _, _ = fetched
    calls: list[int] = []
    real = collect.day_contrasts
    monkeypatch.setattr(collect, "day_contrasts", lambda *a, **k: calls.append(1) or real(*a, **k))
    before = tree_digest(tmp_path / "cache" / "g0-v1")
    code, out, _, _ = run(capsys, tmp_path, "--engine", "v1", "--stage", "fetch")
    assert code == 1 and calls == []
    assert "skipped" in out
    assert tree_digest(tmp_path / "cache" / "g0-v1") == before


def test_a_budget_stops_between_dates_and_the_run_resumes(capsys, tmp_path) -> None:
    seed_v0(tmp_path / "cache")
    code, out, _, _ = run(
        capsys, tmp_path, "--engine", "v1", "--stage", "fetch", "--budget-seconds", "0"
    )
    assert code == 1 and "time budget" in out
    assert not (tmp_path / "cache" / "g0-v1").exists()
    code, _, _, _ = run(capsys, tmp_path, "--engine", "v1", "--stage", "fetch")
    assert v1_store(tmp_path).load(START) is not None and code == 1


def test_the_fetch_stage_does_no_network(capsys, tmp_path) -> None:
    seed_v0(tmp_path / "cache")
    # `command()` serves NoNetwork: any request would raise AssertionError.
    run(capsys, tmp_path, "--engine", "v1", "--stage", "fetch")


# --- the report stage -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reported(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("g0-v1-report")
    seed_v0(tmp_path / "cache")
    invoke(tmp_path, "--stage", "report", "--out", str(tmp_path / "v0-first"))
    invoke(tmp_path, "--engine", "v1", "--stage", "fetch", "--out", str(tmp_path / "unused"))
    invoke(tmp_path, "--engine", "v1", "--stage", "report", "--out", str(tmp_path / "v1"))
    invoke(tmp_path, "--stage", "report", "--out", str(tmp_path / "v0-again"))
    return tmp_path


def test_the_v1_report_is_written_beside_v0s_and_v0_is_unchanged(reported) -> None:
    v0_first = tree_digest(reported / "v0-first")
    assert v0_first == tree_digest(reported / "v0-again")
    assert set(v0_first) == {
        "report.md",
        "results.csv",
        "d_by_date.png",
        "d_distribution.png",
        "profile_spread.png",
    }
    assert set(tree_digest(reported / "v1")) == set(v0_first)
    assert tree_digest(reported / "v1") != v0_first


def test_the_explicit_v0_engine_is_the_default_byte_for_byte(capsys, tmp_path) -> None:
    seed_v0(tmp_path / "cache")
    run(capsys, tmp_path, "--stage", "report", out="default")
    run(capsys, tmp_path, "--stage", "report", "--engine", "v0", out="explicit")
    assert tree_digest(tmp_path / "default") == tree_digest(tmp_path / "explicit")


def test_the_report_stage_with_v1_reads_only_the_v1_store(reported, capsys, tmp_path) -> None:
    shutil.copytree(reported / "cache" / "g0-v1", tmp_path / "cache" / "g0-v1")
    assert not (tmp_path / "cache" / "g0").exists()  # no v0 store here at all
    code, _, _, out_dir = run(capsys, tmp_path, "--engine", "v1", "--stage", "report")
    assert code == 1  # the fourth date has no record in the v1 store
    assert (out_dir / "report.md").read_text() == (reported / "v1" / "report.md").read_text()


def test_the_v1_report_names_renderer_v1_where_v0_names_v0(reported) -> None:
    v0 = (reported / "v0-first" / "report.md").read_text()
    v1 = (reported / "v1" / "report.md").read_text()
    assert "renderer-v0 units" in v0 and "renderer-v1 units" not in v0
    assert "renderer-v1 units" in v1 and "renderer-v0" not in v1
    assert "renderer v1" in v1.splitlines()[0]
    assert v1.splitlines()[0] != v0.splitlines()[0]
    assert "multiple scattering" in v1.lower()
    assert "../report.md" in v1  # the v0 result sits beside it


def test_the_criterion_and_its_rule_are_the_same_in_both_reports(reported) -> None:
    def section(text: str) -> str:
        match = re.search(r"## The pre-registered criterion\n(.*?)\n## ", text, re.S)
        assert match
        return match.group(1)

    v0 = (reported / "v0-first" / "report.md").read_text()
    v1 = (reported / "v1" / "report.md").read_text()
    assert section(v0) == section(v1)
    assert "docs/gates/G0/criterion.md" in section(v1)


def test_the_results_file_has_v0s_columns_and_one_row_per_date(reported) -> None:
    v0 = (reported / "v0-first" / "results.csv").read_text().splitlines()
    v1 = (reported / "v1" / "results.csv").read_text().splitlines()
    assert v1[0] == v0[0] == ",".join(report.RESULTS_COLUMNS)
    assert [row.split(",")[:3] for row in v1[1:]] == [
        ["2025-10-01", "used", ""],
        ["2025-10-02", "used", ""],
        ["2025-10-03", "dropped", "hrrr_run_missing"],
        ["2025-10-04", "dropped", "not_collected"],
    ]


def test_the_v1_outputs_pass_the_privacy_scanner(reported) -> None:
    result = scan([reported / "v1"], root=reported / "v1")
    assert result.findings == [], [str(f) for f in result.findings]


def test_the_v1_study_is_the_same_statistic_on_the_v1_records(reported) -> None:
    store = RecordStore(reported / "cache" / "g0-v1" / "records")
    records = [r for d in range(3) if (r := store.load(START + dt.timedelta(days=d)))]
    study = analysis.build_study(records, START, START + dt.timedelta(days=3), engine="v1")
    assert study.engine == "v1"
    default = analysis.build_study(records, START, START + dt.timedelta(days=3))
    assert default.engine == "v0"
    assert study.verdict == default.verdict  # the statistic itself is not changed by the label
    text = report.render_markdown(study, whtm1_views())
    assert text == (reported / "v1" / "report.md").read_text()


def test_the_command_defaults_to_v0_and_lists_the_option(capsys) -> None:
    holder: dict = {}

    def run_spy(args: argparse.Namespace) -> int:
        holder.update(vars(args))
        return 0

    spy = dataclasses.replace(g0.COMMAND, run=run_spy)
    main(["g0", "--start", "2025-10-01", "--end", "2025-10-02", "--out", "x"], commands=[spy])
    assert holder["engine"] == "v0"
    assert main(["g0", "--help"], commands=[g0.COMMAND]) == 0
    assert "--engine" in capsys.readouterr().out
    assert (
        main(
            ["g0", "--start", "2025-10-01", "--end", "2025-10-02", "--out", "x", "--engine", "v2"],
            commands=[g0.COMMAND],
        )
        == 2
    )


def test_the_records_json_is_what_the_v0_schema_reads(reported) -> None:
    path = reported / "cache" / "g0-v1" / "records" / "2025-10-01.json"
    document = json.loads(path.read_text())
    assert document["schema_version"] == 1 and document["drop_reason"] is None
