"""Choosing a run, downloading only the wanted byte ranges, caching them, and decoding (issue #6).

Everything runs against `hrrr_support.FakeTransport`, which serves the recorded crop at the real
bucket URLs and records every request, so what is *asked for* is as testable as what comes back.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from datetime import UTC, timedelta, timezone
from pathlib import Path

import hrrr_support as support
import numpy as np
import pytest

from lastlight.inputs import hrrr
from lastlight.inputs.hrrr import (
    DEFAULT_FIELDS,
    HrrrError,
    fetch_fields,
    fetch_idx,
    fetch_messages,
    idx_url,
    latest_run,
)
from lastlight.inputs.http import ByteRange, FetchError

DATE = support.CYCLE_DATE
HOUR = support.CYCLE_HOUR


def _fetch(cache: Path, transport: support.FakeTransport, names: list[str] | None = None):
    fields = [spec for spec in DEFAULT_FIELDS if names is None or spec.name in names]
    return fetch_fields(DATE, HOUR, 0, fields, cache, transport=transport)


# --- latest_run -----------------------------------------------------------------------------------


def _bucket_with_idx_for(*cycles: tuple[dt.date, int]) -> support.FakeTransport:
    # latest_run asks for the idx of the run's last forecast hour: a run is complete only when
    # its final file has been written.
    return support.FakeTransport({idx_url(day, hour, 18): b"" for day, hour in cycles})


def test_latest_run_is_the_newest_cycle_at_least_two_hours_old() -> None:
    transport = _bucket_with_idx_for((dt.date(2026, 10, 10), 22), (dt.date(2026, 10, 10), 23))
    now = dt.datetime(2026, 10, 11, 0, 27, tzinfo=UTC)  # 22:xx cycle is 2h+ old, 23 is not

    assert latest_run(now, transport=transport) == (dt.date(2026, 10, 10), 22)


def test_latest_run_counts_a_cycle_complete_exactly_two_hours_after_it_starts() -> None:
    transport = _bucket_with_idx_for((dt.date(2026, 10, 10), 22))

    assert latest_run(dt.datetime(2026, 10, 11, 0, 0, tzinfo=UTC), transport=transport) == (
        dt.date(2026, 10, 10),
        22,
    )
    # One second earlier the 22 cycle is not yet two hours old; the first candidate is 21.
    early = _bucket_with_idx_for((dt.date(2026, 10, 10), 21))
    assert latest_run(dt.datetime(2026, 10, 10, 23, 59, 59, tzinfo=UTC), transport=early) == (
        dt.date(2026, 10, 10),
        21,
    )


def test_latest_run_rolls_the_cycle_date_back_across_midnight() -> None:
    transport = _bucket_with_idx_for((dt.date(2026, 10, 10), 23))

    assert latest_run(dt.datetime(2026, 10, 11, 1, 59, tzinfo=UTC), transport=transport) == (
        dt.date(2026, 10, 10),
        23,
    )


def test_latest_run_rolls_the_cycle_date_forward_at_midnight() -> None:
    transport = _bucket_with_idx_for((dt.date(2026, 10, 11), 0))

    assert latest_run(dt.datetime(2026, 10, 11, 2, 0, tzinfo=UTC), transport=transport) == (
        dt.date(2026, 10, 11),
        0,
    )


def test_latest_run_steps_back_an_hour_at_a_time_until_an_idx_exists() -> None:
    transport = _bucket_with_idx_for((dt.date(2026, 10, 10), 19))
    now = dt.datetime(2026, 10, 11, 0, 27, tzinfo=UTC)

    assert latest_run(now, transport=transport) == (dt.date(2026, 10, 10), 19)
    asked = [call.url for call in transport.calls]
    assert asked == [idx_url(dt.date(2026, 10, 10), hour, 18) for hour in (22, 21, 20, 19)]
    assert all(call.method == "HEAD" for call in transport.calls)


def test_latest_run_stops_at_the_first_run_that_exists() -> None:
    transport = _bucket_with_idx_for((dt.date(2026, 10, 10), 22), (dt.date(2026, 10, 10), 21))

    latest_run(dt.datetime(2026, 10, 11, 0, 27, tzinfo=UTC), transport=transport)

    assert len(transport.calls) == 1


def test_latest_run_gives_up_loudly_when_nothing_recent_exists() -> None:
    transport = support.FakeTransport({})

    with pytest.raises(HrrrError, match="no complete HRRR run"):
        latest_run(dt.datetime(2026, 10, 11, 0, 27, tzinfo=UTC), transport=transport)

    assert len(transport.calls) == hrrr.LATEST_RUN_LOOKBACK_HOURS


def test_latest_run_reads_a_non_utc_instant_as_the_instant_it_names() -> None:
    transport = _bucket_with_idx_for((dt.date(2026, 10, 10), 22))
    seven_hours_behind = timezone(timedelta(hours=-7))
    now = dt.datetime(2026, 10, 10, 17, 27, tzinfo=seven_hours_behind)  # the same instant as above

    assert latest_run(now, transport=transport) == (dt.date(2026, 10, 10), 22)


def test_latest_run_refuses_a_naive_datetime() -> None:
    with pytest.raises(ValueError, match="timezone"):
        latest_run(dt.datetime(2026, 10, 11, 0, 27), transport=support.FakeTransport({}))


# --- fetch_idx ------------------------------------------------------------------------------------


def test_fetch_idx_reads_and_parses_the_idx_of_the_requested_file() -> None:
    transport = support.crop_transport()

    entries = fetch_idx(DATE, HOUR, 0, transport=transport)

    assert [entry.variable for entry in entries][:3] == ["VIS", "PWAT", "AOTK"]
    (call,) = transport.calls
    assert call.url == idx_url(DATE, HOUR, 0)
    assert call.byte_range is None


# --- fetch_messages: byte ranges and the cache ----------------------------------------------------


def test_fetch_messages_asks_for_each_wanted_message_by_exact_byte_range(tmp_path: Path) -> None:
    transport = support.crop_transport()

    messages = fetch_messages(DATE, HOUR, 0, DEFAULT_FIELDS, tmp_path, transport=transport)

    asked = {call.byte_range for call in transport.ranged_gets()}
    # The idx gives the last message no end; every other range closes where the next begins.
    last = support.CROP_MESSAGES["HGT_CLOUD_TOP"]
    expected = {
        ByteRange(last.start, None) if byte_range == last else byte_range
        for byte_range in support.CROP_MESSAGES.values()
    }
    assert asked == expected
    assert len(transport.ranged_gets()) == len(DEFAULT_FIELDS)
    for name, data in messages.items():
        assert data == support.crop_message(name)


def test_fetch_messages_downloads_only_the_wanted_bytes_never_the_whole_file(
    tmp_path: Path,
) -> None:
    transport = support.crop_transport()

    fetch_messages(DATE, HOUR, 0, [hrrr.field_spec("TCDC")], tmp_path, transport=transport)

    assert transport.bytes_served_by_ranges() == len(support.crop_message("TCDC"))
    assert transport.bytes_served_by_ranges() < len(support.CROP_GRIB) / 4
    assert not [
        call
        for call in transport.calls
        if call.method == "GET" and not call.byte_range and call.url == hrrr.grib_url(DATE, HOUR, 0)
    ]


def test_a_second_fetch_is_served_from_the_cache_and_asks_for_no_ranges(tmp_path: Path) -> None:
    first = support.crop_transport()
    fetch_messages(DATE, HOUR, 0, DEFAULT_FIELDS, tmp_path, transport=first)
    second = support.crop_transport()

    messages = fetch_messages(DATE, HOUR, 0, DEFAULT_FIELDS, tmp_path, transport=second)

    assert second.ranged_gets() == []
    assert [call.url for call in second.calls] == [idx_url(DATE, HOUR, 0)]  # the idx is re-read
    assert messages["PWAT"] == support.crop_message("PWAT")


def test_the_cache_holds_one_file_per_url_and_range(tmp_path: Path) -> None:
    fetch_messages(DATE, HOUR, 0, DEFAULT_FIELDS, tmp_path, transport=support.crop_transport())

    files = sorted(path for path in tmp_path.iterdir())
    assert len(files) == len(DEFAULT_FIELDS)
    assert sorted(path.stat().st_size for path in files) == sorted(
        len(support.crop_message(spec.name)) for spec in DEFAULT_FIELDS
    )
    assert not [path for path in files if path.suffix == ".tmp"]


def test_a_different_url_with_the_same_range_does_not_share_a_cache_entry(
    tmp_path: Path,
) -> None:
    # The same bytes served under another cycle's URL: same range, different URL.
    one = support.crop_transport(forecast_hour=0)
    other = support.crop_transport(cycle_hour=21, forecast_hour=0)
    spec = [hrrr.field_spec("PWAT")]
    fetch_messages(DATE, HOUR, 0, spec, tmp_path, transport=one)

    fetch_messages(DATE, 21, 0, spec, tmp_path, transport=other)

    assert len(other.ranged_gets()) == 1  # not satisfied by the first cycle's cache file
    assert len(list(tmp_path.iterdir())) == 2


def test_a_damaged_cache_file_is_discarded_and_downloaded_again(tmp_path: Path) -> None:
    spec = [hrrr.field_spec("LCDC")]
    fetch_messages(DATE, HOUR, 0, spec, tmp_path, transport=support.crop_transport())
    (cached,) = tmp_path.iterdir()
    cached.write_bytes(cached.read_bytes()[:50])  # a half-written download from a crash
    again = support.crop_transport()

    messages = fetch_messages(DATE, HOUR, 0, spec, tmp_path, transport=again)

    assert messages["LCDC"] == support.crop_message("LCDC")
    assert len(again.ranged_gets()) == 1


def test_the_cache_directory_is_created_when_it_does_not_exist(tmp_path: Path) -> None:
    cache = tmp_path / "deeper" / "cache"

    fetch_messages(
        DATE, HOUR, 0, [hrrr.field_spec("VIS")], cache, transport=support.crop_transport()
    )

    assert len(list(cache.iterdir())) == 1


def test_a_range_that_is_not_a_grib_message_is_refused_and_not_cached(tmp_path: Path) -> None:
    # An idx that has gone stale against its file points into the middle of a message.
    stale = support.CROP_IDX_TEXT.replace(
        f":{support.CROP_MESSAGES['PWAT'].start}:", f":{support.CROP_MESSAGES['PWAT'].start + 7}:"
    )
    transport = support.crop_transport()
    transport.files[idx_url(DATE, HOUR, 0)] = stale.encode()

    with pytest.raises(HrrrError, match="GRIB"):
        fetch_messages(DATE, HOUR, 0, [hrrr.field_spec("PWAT")], tmp_path, transport=transport)

    assert list(tmp_path.iterdir()) == []


def test_a_missing_file_surfaces_as_a_fetch_error(tmp_path: Path) -> None:
    with pytest.raises(FetchError, match="404"):
        fetch_messages(DATE, HOUR, 0, DEFAULT_FIELDS, tmp_path, transport=support.FakeTransport({}))


# --- fetch_fields ---------------------------------------------------------------------------------


def test_fetch_fields_decodes_every_default_field_onto_one_grid(tmp_path: Path) -> None:
    fields = _fetch(tmp_path, support.crop_transport())

    assert (fields.cycle_date, fields.cycle_hour, fields.forecast_hour) == (DATE, HOUR, 0)
    assert list(fields.fields) == [spec.name for spec in DEFAULT_FIELDS]
    assert fields.grid.lats.shape == fields.grid.lons.shape == support.CROP_SHAPE
    for name, data in fields.fields.items():
        assert data.values.shape == support.CROP_SHAPE
        assert data.spec.name == name


def test_fetch_fields_values_match_the_real_file_at_the_two_known_cells(tmp_path: Path) -> None:
    fields = _fetch(tmp_path, support.crop_transport())

    for name, data in fields.fields.items():
        assert float(data.values[support.REFERENCE_CELL]) == pytest.approx(
            support.EXPECTED_AT_REFERENCE_CELL[name], abs=1e-3
        )
        assert float(data.values[support.WEST_CELL]) == pytest.approx(
            support.EXPECTED_AT_WEST_CELL[name], abs=1e-3
        )


def test_fetch_fields_returns_only_the_fields_asked_for(tmp_path: Path) -> None:
    fields = _fetch(tmp_path, support.crop_transport(), names=["PWAT", "VIS"])

    assert list(fields.fields) == ["PWAT", "VIS"]


def test_fetch_fields_refuses_a_message_that_is_not_the_field_the_idx_promised(
    tmp_path: Path,
) -> None:
    # Swap the labels of the first two idx lines: asking for VIS now downloads PWAT's bytes,
    # which are a perfectly good GRIB message of the wrong parameter.
    lines = support.CROP_IDX_TEXT.splitlines()
    swapped = (
        lines[0].replace(":VIS:surface:", ":PWAT:x:")
        + "\n"
        + lines[1].replace(
            ":PWAT:entire atmosphere (considered as a single layer):", ":VIS:surface:"
        )
    )
    transport = support.crop_transport()
    transport.files[idx_url(DATE, HOUR, 0)] = ("\n".join([swapped, *lines[2:]]) + "\n").encode()

    with pytest.raises(HrrrError, match="VIS"):
        fetch_fields(DATE, HOUR, 0, [hrrr.field_spec("VIS")], tmp_path, transport=transport)


def test_fetch_fields_refuses_a_message_from_another_forecast_hour(tmp_path: Path) -> None:
    # The crop is a forecast-hour-0 file; serving it under the hour-3 URL must not pass as hour 3.
    transport = support.crop_transport(forecast_hour=3)

    with pytest.raises(HrrrError, match="forecast hour"):
        fetch_fields(DATE, HOUR, 3, [hrrr.field_spec("PWAT")], tmp_path, transport=transport)


def test_fetch_fields_refuses_a_message_from_another_cycle(tmp_path: Path) -> None:
    transport = support.crop_transport(cycle_hour=21)

    with pytest.raises(HrrrError, match="cycle"):
        fetch_fields(DATE, 21, 0, [hrrr.field_spec("PWAT")], tmp_path, transport=transport)


def test_fetch_fields_refuses_fields_on_different_grids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_decode = hrrr.decode_message

    def shifted(data: bytes):
        decoded = real_decode(data)
        if decoded.number == 3 and decoded.category == 1:  # PWAT
            return dataclasses.replace(
                decoded,
                grid=dataclasses.replace(decoded.grid, first_lat=decoded.grid.first_lat + 1),
            )
        return decoded

    monkeypatch.setattr(hrrr, "decode_message", shifted)

    with pytest.raises(HrrrError, match="grid"):
        _fetch(tmp_path, support.crop_transport(), names=["LCDC", "PWAT"])


def test_fetch_fields_masks_cells_the_model_leaves_empty(tmp_path: Path) -> None:
    fields = _fetch(tmp_path, support.crop_transport(), names=["HGT_CLOUD_TOP"])

    values = fields.fields["HGT_CLOUD_TOP"].values
    assert isinstance(values, np.ma.MaskedArray)
    assert int(np.ma.getmaskarray(values).sum()) == support.CLOUD_TOP_MASKED_CELLS


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "minus-inf"])
def test_fetch_fields_refuses_a_non_finite_value_anywhere_in_the_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: float
) -> None:
    # The poisoned cell is the grid's far south-west corner, which no transect from the
    # reference point samples. Only a check on the whole decoded field can see it; the
    # check at sampling time reads sampled cells alone.
    real_decode = hrrr.decode_message

    def poisoned(data: bytes):
        decoded = real_decode(data)
        if (decoded.category, decoded.number) == (1, 3):  # PWAT
            values = decoded.values.copy()
            values[support.CORNER_CELL] = bad
            return dataclasses.replace(decoded, values=values)
        return decoded

    monkeypatch.setattr(hrrr, "decode_message", poisoned)

    with pytest.raises(HrrrError, match=r"PWAT.*non-finite"):
        _fetch(tmp_path, support.crop_transport(), names=["LCDC", "PWAT"])


def test_fetch_fields_with_no_fields_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="field"):
        fetch_fields(DATE, HOUR, 0, [], tmp_path, transport=support.crop_transport())


def test_fetch_fields_refuses_two_fields_with_the_same_name(tmp_path: Path) -> None:
    spec = hrrr.field_spec("VIS")

    with pytest.raises(ValueError, match="VIS"):
        fetch_fields(DATE, HOUR, 0, [spec, spec], tmp_path, transport=support.crop_transport())


def test_byte_range_dataclass_is_hashable_for_use_as_a_cache_key() -> None:
    assert len({ByteRange(1, 2), ByteRange(1, 2), ByteRange(3, None)}) == 2


# --- digests of what was read (issue #8) ----------------------------------------------------------


def test_each_decoded_field_carries_the_sha256_of_its_raw_message(tmp_path: Path) -> None:
    import hashlib

    fields = fetch_fields(
        support.CYCLE_DATE, support.CYCLE_HOUR, 0, DEFAULT_FIELDS, tmp_path,
        transport=support.crop_transport(),
    )  # fmt: skip
    assert list(fields.digests) == [spec.name for spec in DEFAULT_FIELDS]
    for spec in DEFAULT_FIELDS:
        expected = hashlib.sha256(support.crop_message(spec.name)).hexdigest()
        assert fields.digests[spec.name] == expected


def test_the_digests_do_not_change_when_the_bytes_come_from_the_cache(tmp_path: Path) -> None:
    first = fetch_fields(
        support.CYCLE_DATE, support.CYCLE_HOUR, 0, DEFAULT_FIELDS, tmp_path,
        transport=support.crop_transport(),
    )  # fmt: skip
    second = fetch_fields(
        support.CYCLE_DATE, support.CYCLE_HOUR, 0, DEFAULT_FIELDS, tmp_path,
        transport=support.crop_transport(),
    )  # fmt: skip
    assert first.digests == second.digests
