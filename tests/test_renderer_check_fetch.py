"""Collecting each evening's inputs: G0's collector for the atmosphere, HRRR TCDC for the cloud.

The atmosphere is built exactly as G0 builds it (G0's `Collector`, its drop reasons, its
transient faults); the cloud is HRRR's total cloud cover (`TCDC`, entire atmosphere) at forecast
hour 0 of the cycle at the target hour, sampled at the 17 points of G0's transect and averaged.
An evening whose target-hour cycle is not yet two hours old is pending: it gets no record and is
never filled from a forecast.
"""

from __future__ import annotations

import dataclasses
import datetime as dt

import numpy as np
import pytest
from g0_support import gefs_files, synthetic_record
from hrrr_support import FakeTransport
from test_renderer_check_support import NOW, served_run

from lastlight import geometry
from lastlight.atmosphere import ozone_du
from lastlight.g0.collect import HRRR_FIELD_NAMES, Collector, TransientError
from lastlight.g0.record import AtmosphereRecord, DateRecord
from lastlight.inputs import gefs, hrrr
from lastlight.inputs.http import FetchError
from lastlight.renderer_check import fetch, store
from lastlight.site import REFERENCE_POINT

LAT, LON, _ = REFERENCE_POINT
EVENING = dt.date(2026, 10, 10)


def target_cycle(evening: dt.date) -> dt.datetime:
    plan = hrrr.plan_run(
        evening, LAT, LON, cycle_hour=None, forecast_hour=0, now_utc=NOW, transport=None
    )
    return dt.datetime.combine(plan.cycle_date, dt.time(plan.cycle_hour), tzinfo=dt.UTC)


class EverythingExists:
    """A transport on which every run is complete: `plan_run` then takes the newest cycle."""

    def exists(self, url: str) -> bool:
        return True

    def get(self, url, byte_range=None):
        raise AssertionError("no download is expected")


# --- the analysis is ready two hours after the target-hour cycle ----------------------------------


def test_the_plan_is_the_target_hour_analysis_once_two_hours_old() -> None:
    cycle = target_cycle(EVENING)
    plan = fetch.analysis_plan(EVENING, cycle + dt.timedelta(hours=2))
    assert plan is not None
    assert (plan.cycle_date, plan.cycle_hour, plan.forecast_hour) == (
        cycle.date(),
        cycle.hour,
        0,
    )


def test_one_second_short_of_two_hours_the_evening_is_pending() -> None:
    cycle = target_cycle(EVENING)
    assert fetch.analysis_plan(EVENING, cycle + dt.timedelta(hours=2, seconds=-1)) is None
    assert fetch.analysis_plan(EVENING, cycle) is None
    assert fetch.analysis_plan(EVENING, cycle - dt.timedelta(days=1)) is None


def test_readiness_agrees_with_what_plan_run_itself_decides() -> None:
    cycle = target_cycle(EVENING)
    for offset in (
        dt.timedelta(hours=1),
        dt.timedelta(hours=2, seconds=-1),
        dt.timedelta(hours=2),
        dt.timedelta(hours=2, seconds=1),
        dt.timedelta(hours=9),
    ):
        now = cycle + offset
        default = hrrr.plan_run(
            EVENING,
            LAT,
            LON,
            cycle_hour=None,
            forecast_hour=None,
            now_utc=now,
            transport=EverythingExists(),
        )
        assert (fetch.analysis_plan(EVENING, now) is not None) == (default.forecast_hour == 0)


def test_the_readiness_check_makes_no_request() -> None:
    fetch.analysis_plan(EVENING, NOW)  # would raise if it used the default (real) transport


# --- the cloud mean -------------------------------------------------------------------------------


def samples_with(values) -> list[hrrr.TransectSample]:
    return [
        hrrr.TransectSample(
            point_index=i,
            field="TCDC",
            lat=33.0,
            lon=-112.0,
            value=v,
            units="%",
            method="nearest-cell",
            cell=(0, 0),
            cell_distance_km=0.0,
        )
        for i, v in enumerate(values)
    ]


def test_the_cloud_is_the_mean_of_tcdc_over_the_points() -> None:
    assert fetch.cloud_percent(samples_with([0.0, 10.0, 20.0, 30.0])) == pytest.approx(15.0)


def test_a_point_without_a_value_makes_the_cloud_unknown() -> None:
    assert fetch.cloud_percent(samples_with([1.0, None, 3.0])) is None


def test_other_fields_in_the_samples_are_ignored() -> None:
    other = [dataclasses.replace(s, field="LCDC", value=99.0) for s in samples_with([5.0] * 17)]
    mixed = samples_with([5.0] * 17) + other
    assert fetch.cloud_percent(mixed) == pytest.approx(5.0)


def test_the_transect_has_seventeen_points_to_four_hundred_km() -> None:
    assert fetch.TRANSECT_POINTS == 17
    assert hrrr.TRANSECT_DISTANCES_KM[0] == 0 and hrrr.TRANSECT_DISTANCES_KM[-1] == 400
    assert len(hrrr.TRANSECT_DISTANCES_KM) == 17


# --- one evening from served archives -------------------------------------------------------------


@pytest.fixture(scope="module")
def clear_evening(tmp_path_factory):
    """The first evening collected once against served archives with a clear (5 percent) sky."""
    transport = served_run(EVENING, NOW, tcdc=5.0)
    cache = tmp_path_factory.mktemp("renderer-check-fetch")
    return transport, cache, fetch.InputFetcher(transport, cache, NOW).fetch(EVENING)


def test_an_evening_becomes_an_ok_record_with_the_atmosphere_and_the_cloud(clear_evening) -> None:
    _, _, record = clear_evening
    assert record is not None and record.ok
    assert record.date == EVENING
    assert record.cloud_mean_pct == pytest.approx(5.0, abs=1e-3)
    assert record.cloud_points == 17
    assert record.ozone_du == ozone_du(EVENING.month)


def test_aod_550_is_the_archives_total_exactly_as_g0_reads_it(clear_evening) -> None:
    transport, cache, record = clear_evening
    aerosol = gefs.fetch_aerosol(EVENING, LAT, LON, cache / "gefs", transport=transport)
    assert record.aod_550 == aerosol.total_aod_550


def test_the_atmosphere_is_the_g0_collectors_record(clear_evening, tmp_path) -> None:
    transport, _, record = clear_evening
    g0_record = Collector(transport, tmp_path / "other", NOW).collect(EVENING)
    assert g0_record.atmosphere is not None
    assert record.aod_550 == g0_record.atmosphere.aod_total
    assert record.ozone_du == g0_record.atmosphere.ozone_du


def test_the_hrrr_fields_g0_downloads_are_unchanged() -> None:
    assert HRRR_FIELD_NAMES == ("LCDC", "AOTK", "PWAT")


def test_only_the_analysis_hour_is_ever_requested(clear_evening) -> None:
    transport, _, _ = clear_evening
    urls = {call.url for call in transport.calls if "hrrr" in call.url}
    assert urls and all("wrfsfcf00" in url for url in urls)


def test_a_cloudy_sky_is_recorded_as_it_is_not_filtered_at_fetch_time(tmp_path) -> None:
    transport = served_run(EVENING, NOW)  # the recorded crop: TCDC is 100 percent
    record = fetch.InputFetcher(transport, tmp_path, NOW).fetch(EVENING)
    assert record is not None and record.ok
    assert record.cloud_mean_pct == pytest.approx(100.0)


def test_a_pending_evening_makes_no_request_and_has_no_record(tmp_path) -> None:
    cycle = target_cycle(EVENING)
    early = cycle + dt.timedelta(hours=1, minutes=59)
    transport = served_run(EVENING, NOW)
    result = fetch.InputFetcher(transport, tmp_path, early).fetch(EVENING)
    assert result is None
    assert transport.calls == []


def test_a_missing_hrrr_run_is_dropped_with_g0s_reason_before_any_cloud_fetch(tmp_path) -> None:
    transport = FakeTransport(gefs_files(EVENING))
    record = fetch.InputFetcher(transport, tmp_path, NOW).fetch(EVENING)
    assert record is not None and not record.ok
    assert record.drop_reason == "hrrr_run_missing"
    assert record.cloud_mean_pct is None and record.aod_550 is None


def test_a_missing_aerosol_run_is_dropped_with_g0s_reason(tmp_path) -> None:
    only_hrrr = served_run(EVENING, NOW, tcdc=5.0)
    transport = FakeTransport({u: b for u, b in only_hrrr.files.items() if "gefs" not in u})
    record = fetch.InputFetcher(transport, tmp_path, NOW).fetch(EVENING)
    assert record is not None and record.drop_reason == "aerosol_run_missing"


def test_a_file_without_tcdc_in_its_idx_is_dropped_as_a_missing_field(tmp_path) -> None:
    transport = served_run(EVENING, NOW, tcdc=5.0)
    for url, body in list(transport.files.items()):
        if url.endswith(".idx") and "hrrr" in url:
            kept = [ln for ln in body.decode().splitlines() if ":TCDC:" not in ln]
            transport.files[url] = ("\n".join(kept) + "\n").encode()
    record = fetch.InputFetcher(transport, tmp_path, NOW).fetch(EVENING)
    assert record is not None and record.drop_reason == "hrrr_field_missing"


def test_a_transect_point_without_a_value_is_dropped_as_tcdc_missing(tmp_path, monkeypatch) -> None:
    real = hrrr.sample_transect

    def holey(fields, points):
        samples = real(fields, points)
        if "TCDC" in fields.fields and len(fields.fields) == 1:
            samples[7] = dataclasses.replace(samples[7], value=None)
        return samples

    monkeypatch.setattr(hrrr, "sample_transect", holey)
    transport = served_run(EVENING, NOW, tcdc=5.0)
    record = fetch.InputFetcher(transport, tmp_path, NOW).fetch(EVENING)
    assert record is not None and not record.ok
    assert record.drop_reason == "tcdc_missing"
    assert record.cloud_mean_pct is None


def test_a_transient_fault_while_fetching_the_cloud_is_raised_not_recorded(
    tmp_path, monkeypatch
) -> None:
    real = hrrr.fetch_fields

    def flaky(cycle_date, cycle_hour, forecast_hour, fields, cache_dir, *, transport=None):
        if [f.name for f in fields] == ["TCDC"]:
            raise FetchError("https://example.invalid/x: HTTP 503 Service Unavailable")
        return real(cycle_date, cycle_hour, forecast_hour, fields, cache_dir, transport=transport)

    monkeypatch.setattr(hrrr, "fetch_fields", flaky)
    with pytest.raises(TransientError):
        fetch.InputFetcher(served_run(EVENING, NOW, tcdc=5.0), tmp_path, NOW).fetch(EVENING)


def test_the_cloud_is_fetched_from_the_analysis_with_the_whole_atmosphere_tcdc_spec(
    tmp_path, monkeypatch
) -> None:
    seen: list[tuple[int, list[str]]] = []
    real = hrrr.fetch_fields

    def spy(cycle_date, cycle_hour, forecast_hour, fields, cache_dir, *, transport=None):
        seen.append((forecast_hour, [f.name for f in fields]))
        return real(cycle_date, cycle_hour, forecast_hour, fields, cache_dir, transport=transport)

    monkeypatch.setattr(hrrr, "fetch_fields", spy)
    fetch.InputFetcher(served_run(EVENING, NOW, tcdc=5.0), tmp_path, NOW).fetch(EVENING)
    assert (0, ["TCDC"]) in seen
    spec = hrrr.field_spec("TCDC")
    assert (spec.variable, spec.level) == ("TCDC", "entire atmosphere")


def test_the_cloud_is_sampled_along_g0s_transect(tmp_path, monkeypatch) -> None:
    points_seen: list[list[tuple[float, float]]] = []
    real = hrrr.sample_transect

    def spy(fields, points):
        if list(fields.fields) == ["TCDC"]:
            points_seen.append(list(points))
        return real(fields, points)

    monkeypatch.setattr(hrrr, "sample_transect", spy)
    fetch.InputFetcher(served_run(EVENING, NOW, tcdc=5.0), tmp_path, NOW).fetch(EVENING)
    expected = geometry.transect(
        EVENING, LAT, LON, [float(km) for km in hrrr.TRANSECT_DISTANCES_KM]
    )
    assert points_seen == [expected]


def test_the_collector_is_g0s_and_can_be_replaced_for_tests(tmp_path) -> None:
    fetcher = fetch.InputFetcher(served_run(EVENING, NOW), tmp_path, NOW)
    assert isinstance(fetcher.collector, Collector)


# --- the run over several evenings ----------------------------------------------------------------


class Scripted:
    """A fetcher that answers from a table: a record, None (pending) or an exception."""

    def __init__(self, answers) -> None:
        self.answers = dict(answers)
        self.asked: list[dt.date] = []

    def fetch(self, evening: dt.date):
        self.asked.append(evening)
        answer = self.answers[evening]
        if isinstance(answer, Exception):
            raise answer
        return answer


def ok(day: dt.date) -> store.EveningInputs:
    return store.EveningInputs.rendered(day, aod_550=0.05, ozone_du=280.0, cloud_mean_pct=1.0)


def test_each_record_is_saved_as_it_is_made_and_pending_and_faults_leave_none(tmp_path) -> None:
    days = [dt.date(2026, 10, d) for d in (10, 11, 12, 13)]
    records = store.InputStore(tmp_path)
    scripted = Scripted(
        {
            days[0]: ok(days[0]),
            days[1]: store.EveningInputs.dropped(days[1], "aerosol_unreadable", "x"),
            days[2]: None,
            days[3]: TransientError("HRRR: HTTP 503"),
        }
    )
    summary = fetch.fetch_range(days, records, scripted, now_utc=NOW)
    assert (summary.recorded, summary.dropped, summary.skipped) == (1, 1, 0)
    assert summary.pending == (days[2],) and summary.faults == (days[3],)
    assert records.dates() == {days[0], days[1]}


def test_an_evening_with_a_record_is_skipped_without_asking(tmp_path) -> None:
    day = dt.date(2026, 10, 10)
    records = store.InputStore(tmp_path)
    records.save(ok(day))
    scripted = Scripted({})
    summary = fetch.fetch_range([day], records, scripted, now_utc=NOW)
    assert summary.skipped == 1 and scripted.asked == []


def test_the_run_reports_progress_with_dates_only(tmp_path) -> None:
    day = dt.date(2026, 10, 10)
    lines: list[str] = []
    fetch.fetch_range(
        [day], store.InputStore(tmp_path), Scripted({day: None}), now_utc=NOW, progress=lines.append
    )
    assert lines and all(line.startswith("2026-10-10") for line in lines)


def test_a_missing_run_just_after_the_evening_is_retried_but_not_later(tmp_path) -> None:
    day = dt.date(2026, 10, 10)
    records = store.InputStore(tmp_path)
    for reason in ("hrrr_run_missing", "aerosol_run_missing"):
        records.save(store.EveningInputs.dropped(day, reason, "not in the archive"))
        near = dt.datetime(2026, 10, 11, 5, tzinfo=dt.UTC)  # the day after the evening
        scripted = Scripted({day: ok(day)})
        fetch.fetch_range([day], records, scripted, now_utc=near)
        assert scripted.asked == [day], reason
        assert records.load(day).ok

        records.save(store.EveningInputs.dropped(day, reason, "not in the archive"))
        late = dt.datetime(2026, 10, 12, 0, tzinfo=dt.UTC)  # two days after: final
        scripted = Scripted({day: ok(day)})
        fetch.fetch_range([day], records, scripted, now_utc=late)
        assert scripted.asked == [], reason
        assert not records.load(day).ok


def test_other_drop_reasons_are_never_retried(tmp_path) -> None:
    day = dt.date(2026, 10, 10)
    records = store.InputStore(tmp_path)
    near = dt.datetime(2026, 10, 10, 23, tzinfo=dt.UTC)
    for reason in ("hrrr_field_missing", "hrrr_unreadable", "tcdc_missing", "aerosol_unreadable"):
        records.save(store.EveningInputs.dropped(day, reason, "x"))
        scripted = Scripted({day: ok(day)})
        fetch.fetch_range([day], records, scripted, now_utc=near)
        assert scripted.asked == [], reason


def test_a_collector_record_that_g0_could_not_build_is_stored_as_a_drop(tmp_path) -> None:
    class DroppingCollector:
        def collect(self, evening: dt.date) -> DateRecord:
            return DateRecord(
                date=evening,
                drop_reason="hrrr_field_empty",
                drop_detail="LCDC has no value",
                atmosphere=None,
                contrast={},
                floor_channels={},
                curve_lab=None,
                inputs={},
            )

    fetcher = fetch.InputFetcher(
        served_run(EVENING, NOW), tmp_path, NOW, collector=DroppingCollector()
    )
    record = fetcher.fetch(EVENING)
    assert record is not None and record.drop_reason == "hrrr_field_empty"
    assert record.drop_detail == "LCDC has no value"


def test_the_collectors_atmosphere_fields_used_are_aod_total_and_ozone(tmp_path) -> None:
    class StubCollector:
        def collect(self, evening: dt.date) -> DateRecord:
            base = synthetic_record(evening, aod=0.123)
            atmosphere = dataclasses.replace(
                base.atmosphere, aod_dust=0.0111, aod_smoke=0.0222, ozone_du=266.0
            )
            assert isinstance(atmosphere, AtmosphereRecord)
            assert np.isfinite(base.contrast["nominal"]).all()
            return dataclasses.replace(base, atmosphere=atmosphere)

    fetcher = fetch.InputFetcher(
        served_run(EVENING, NOW, tcdc=5.0), tmp_path, NOW, collector=StubCollector()
    )
    record = fetcher.fetch(EVENING)
    assert (record.aod_550, record.ozone_du) == (0.123, 266.0)
