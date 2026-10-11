"""GEFS-Aerosols ingest: total and dust aerosol optical depth at 550 nm for an evening.

HRRR carries smoke only, so G0 reads dust and total aerosol from NOAA's GEFS-Aerosols (control
run, 0.25 degrees) in the public `noaa-gefs-pds` bucket, which needs no account. The flow mirrors
`lastlight.inputs.hrrr`: the `.idx` names byte ranges, only the two wanted messages are
downloaded and cached, each is checked to be the aerosol, wavelength and run that was asked for,
and the value is that of the nearest cell. Everything runs against the recorded fixtures
(`tests/fixtures/gefs`); no test touches the network.
"""

from __future__ import annotations

import datetime as dt
import hashlib

import gefs_support as support
import hrrr_support
import numpy as np
import pytest

from lastlight.atmosphere import AerosolColumn
from lastlight.inputs import gefs, grib
from lastlight.inputs.http import ByteRange, FetchError
from lastlight.site import REFERENCE_POINT

LAT, LON, _ = REFERENCE_POINT
EVENING = dt.date(2025, 9, 30)  # its sun reaches the horizon on the next UTC date
CYCLE_DATE = dt.date(2025, 10, 1)


# --- URLs ------------------------------------------------------------------------------------


def test_the_urls_are_the_chem_products_of_the_public_bucket() -> None:
    url = gefs.chem_url(dt.date(2025, 10, 1), 0, 0)
    assert url == (
        "https://noaa-gefs-pds.s3.amazonaws.com/gefs.20251001/00/chem/pgrb2ap25/"
        "gefs.chem.t00z.a2d_0p25.f000.grib2"
    )
    assert gefs.idx_url(dt.date(2025, 10, 1), 0, 0) == url + ".idx"
    assert "gefs.20251231/18/chem/pgrb2ap25/gefs.chem.t18z.a2d_0p25.f003.grib2" in gefs.chem_url(
        dt.date(2025, 12, 31), 18, 3
    )


@pytest.mark.parametrize(("cycle", "hour"), [(24, 0), (-6, 0), (7, 0), (0, -3), (0, 4), (0, 387)])
def test_a_cycle_or_forecast_hour_the_product_does_not_have_is_refused(
    cycle: int, hour: int
) -> None:
    with pytest.raises(ValueError):
        gefs.chem_url(dt.date(2025, 10, 1), cycle, hour)


# --- the run for an evening -------------------------------------------------------------------


def test_the_analysis_is_the_latest_cycle_at_or_before_the_sun_reaching_the_horizon() -> None:
    plan = gefs.aerosol_run(EVENING, LAT, LON)
    assert (plan.cycle_date, plan.forecast_hour) == (CYCLE_DATE, 0)
    # The cycle is a whole 6-hourly cycle strictly before the instant, never after it.
    from lastlight import geometry

    instant = geometry.elevation_crossings(EVENING, LAT, LON)[0.0]
    cycle = dt.datetime.combine(plan.cycle_date, dt.time(plan.cycle_hour), tzinfo=dt.UTC)
    assert cycle <= instant
    assert instant - cycle < dt.timedelta(hours=6)
    assert plan.cycle_hour % 6 == 0


@pytest.mark.parametrize("evening", [dt.date(2025, 12, 21), dt.date(2026, 6, 21)])
def test_the_run_is_on_the_utc_date_after_the_evening_at_the_reference_point(
    evening: dt.date,
) -> None:
    plan = gefs.aerosol_run(evening, LAT, LON)
    assert plan.cycle_date == evening + dt.timedelta(days=1)
    assert plan.forecast_hour == 0


def test_the_run_follows_the_longitude() -> None:
    # On the Greenwich meridian the sun reaches the horizon about six hours earlier than here.
    greenwich = gefs.aerosol_run(dt.date(2025, 10, 1), 33.45, 0.0)
    assert (greenwich.cycle_date, greenwich.cycle_hour) == (dt.date(2025, 10, 1), 12)


# --- the idx ----------------------------------------------------------------------------------


def test_the_real_idx_parses_to_32_messages_with_increasing_offsets() -> None:
    entries = gefs.parse_idx(support.REAL_IDX_TEXT)
    assert len(entries) == 32
    assert [e.number for e in entries] == list(range(1, 33))
    offsets = [e.offset for e in entries]
    assert offsets == sorted(offsets) and len(set(offsets)) == 32
    fifth = entries[4]
    assert (fifth.variable, fifth.level, fifth.forecast) == ("AOTK", "entire atmosphere", "anl")
    assert fifth.date_code == "2025100100"
    assert fifth.label == (
        "aerosol=Total aerosol:aerosol_size <2e-05:aerosol_wavelength >=5.45e-07,<=5.65e-07"
    )


def test_a_malformed_idx_line_is_an_error() -> None:
    with pytest.raises(gefs.AerosolArchiveError, match="line 2"):
        gefs.parse_idx("1:0:d=2025100100:AOTK:entire atmosphere:anl:aerosol=x\nnot an idx line\n")
    with pytest.raises(gefs.AerosolArchiveError, match="offset"):
        gefs.parse_idx(
            "1:100:d=2025100100:AOTK:entire atmosphere:anl:aerosol=x\n"
            "2:50:d=2025100100:AOTK:entire atmosphere:anl:aerosol=y\n"
        )


def test_the_two_wanted_messages_are_found_among_many_lookalikes_in_the_real_idx() -> None:
    entries = gefs.parse_idx(support.REAL_IDX_TEXT)
    ranges = gefs.select_ranges(entries, gefs.AOD_SPECS)
    # Lines 5 and 7 of the real idx; the message ends one byte before the next line's offset.
    assert ranges["TOTAL_AOD_550"] == ByteRange(4597344, 5655104)
    assert ranges["DUST_AOD_550"] == ByteRange(6710197, 7155025)
    assert list(ranges) == ["TOTAL_AOD_550", "DUST_AOD_550"]


def test_a_missing_or_doubled_message_is_an_error_not_a_guess() -> None:
    entries = gefs.parse_idx(support.REAL_IDX_TEXT)
    without_dust = [e for e in entries if "Dust dry" not in e.label]
    with pytest.raises(gefs.AerosolArchiveError, match="DUST_AOD_550"):
        gefs.select_ranges(without_dust, gefs.AOD_SPECS)
    doubled = [*entries, entries[4]]
    with pytest.raises(gefs.AerosolArchiveError, match="TOTAL_AOD_550"):
        gefs.select_ranges(doubled, gefs.AOD_SPECS)


def test_the_crops_idx_selects_the_crops_messages() -> None:
    entries = gefs.parse_idx(support.CROP_IDX_TEXT)
    assert gefs.select_ranges(entries, gefs.AOD_SPECS) == support.CROP_MESSAGES


# --- decoding a regular latitude/longitude message -------------------------------------------


def test_a_gefs_message_decodes_with_its_aerosol_wavelength_and_axes() -> None:
    total = grib.decode_latlon_message(support.crop_message("TOTAL_AOD_550"))
    assert (total.discipline, total.category, total.number) == (0, 20, 102)
    assert total.aerosol_type == 62000
    assert total.wavelength_nm == pytest.approx((545.0, 565.0))
    assert (total.reference_date, total.reference_hour, total.step_hours) == (CYCLE_DATE, 0, 0)
    assert total.values.shape == support.CROP_SHAPE
    assert total.lats.shape == (21,) and total.lons.shape == (21,)
    assert total.lats[0] == 36.0 and total.lats[-1] == 31.0  # north to south
    assert total.lons[0] == pytest.approx(-114.5) and total.lons[-1] == pytest.approx(-109.5)
    dust = grib.decode_latlon_message(support.crop_message("DUST_AOD_550"))
    assert dust.aerosol_type == 62001


def test_the_decoded_values_at_the_reference_cell_are_the_originals() -> None:
    row, col = support.REFERENCE_CELL
    total = grib.decode_latlon_message(support.crop_message("TOTAL_AOD_550"))
    dust = grib.decode_latlon_message(support.crop_message("DUST_AOD_550"))
    assert total.lats[row] == 33.5 and total.lons[col] == pytest.approx(-112.0)
    assert float(total.values[row, col]) == pytest.approx(support.TOTAL_AT_REFERENCE, abs=1e-6)
    assert float(dust.values[row, col]) == pytest.approx(support.DUST_AT_REFERENCE, abs=1e-6)


def test_a_lambert_message_is_not_a_latlon_message_and_the_reverse() -> None:
    hrrr_message = hrrr_support.crop_message("AOTK")
    with pytest.raises(grib.GribError, match="latitude/longitude"):
        grib.decode_latlon_message(hrrr_message)
    with pytest.raises(grib.GribError, match="Lambert"):
        grib.decode_message(support.crop_message("TOTAL_AOD_550"))


def test_bytes_that_are_not_one_message_are_refused() -> None:
    both = support.CROP_GRIB
    with pytest.raises(grib.GribError, match="exactly one"):
        grib.decode_latlon_message(both)
    with pytest.raises(grib.GribError):
        grib.decode_latlon_message(b"not grib at all")


# --- fetching an evening's aerosol --------------------------------------------------------------


def _fetch(tmp_path, transport=None, **kwargs):
    transport = transport or support.crop_transport(CYCLE_DATE, 0)
    return gefs.fetch_aerosol(EVENING, LAT, LON, tmp_path, transport=transport, **kwargs), transport


def test_the_values_are_those_of_the_cell_nearest_the_reference_point(tmp_path) -> None:
    sample, _ = _fetch(tmp_path)
    assert sample.total_aod_550 == pytest.approx(support.TOTAL_AT_REFERENCE, abs=1e-6)
    assert sample.dust_aod_550 == pytest.approx(support.DUST_AT_REFERENCE, abs=1e-6)
    assert sample.cell == support.REFERENCE_CELL
    assert 3.0 < sample.cell_distance_km < 12.0  # 0.05 degrees north, 0.07 east of the point
    assert (sample.cycle_date, sample.cycle_hour, sample.forecast_hour) == (CYCLE_DATE, 0, 0)


def test_the_sample_becomes_the_reductions_aerosol_column(tmp_path) -> None:
    sample, _ = _fetch(tmp_path)
    column = sample.column()
    assert isinstance(column, AerosolColumn)
    assert column.total_aod_550 == sample.total_aod_550
    assert column.dust_aod_550 == sample.dust_aod_550
    assert column.source == gefs.SOURCE_LABEL == "GEFS-Aerosols analysis"


def test_the_sample_records_the_digest_of_each_message_it_read(tmp_path) -> None:
    sample, transport = _fetch(tmp_path)
    redated = {
        name: support.edited(support.crop_message(name), cycle=(CYCLE_DATE, 0))
        for name in support.CROP_MESSAGES
    }
    assert sample.digests == {
        name: hashlib.sha256(data).hexdigest() for name, data in redated.items()
    }
    assert all(len(d) == 64 for d in sample.digests.values())


def test_only_the_idx_and_the_two_byte_ranges_are_requested(tmp_path) -> None:
    _, transport = _fetch(tmp_path)
    assert [c.method for c in transport.calls] == ["GET", "GET", "GET"]
    assert transport.calls[0].url.endswith(".grib2.idx")
    assert transport.calls[0].byte_range is None
    ranged = transport.ranged_gets()
    assert [c.byte_range for c in ranged] == [
        support.CROP_MESSAGES["TOTAL_AOD_550"],
        support.CROP_MESSAGES["DUST_AOD_550"],
    ]


def test_a_second_fetch_reads_the_ranges_from_the_cache(tmp_path) -> None:
    first, _ = _fetch(tmp_path)
    again, transport = _fetch(tmp_path)
    assert again == first
    assert len(transport.ranged_gets()) == 0  # only the small idx is read again
    assert len(transport.calls) == 1


def test_a_truncated_cached_range_is_downloaded_again(tmp_path) -> None:
    _fetch(tmp_path)
    cached = sorted(tmp_path.glob("*.grib2"))
    assert len(cached) == 2
    cached[0].write_bytes(cached[0].read_bytes()[:100])
    again, transport = _fetch(tmp_path)
    assert len(transport.ranged_gets()) == 1
    assert again.total_aod_550 == pytest.approx(support.TOTAL_AT_REFERENCE, abs=1e-6)


def test_a_run_that_is_not_in_the_archive_is_a_fetch_error_naming_the_404(tmp_path) -> None:
    empty = hrrr_support.FakeTransport({})
    with pytest.raises(FetchError, match="404"):
        gefs.fetch_aerosol(EVENING, LAT, LON, tmp_path, transport=empty)


def test_a_message_from_another_run_is_refused(tmp_path) -> None:
    wrong = dt.date(2025, 9, 29)
    transport = support.crop_transport(CYCLE_DATE, 0)
    url = gefs.chem_url(CYCLE_DATE, 0, 0)
    transport.files[url] = support.redated_crop(wrong, 0)
    with pytest.raises(gefs.AerosolArchiveError, match="cycle"):
        gefs.fetch_aerosol(EVENING, LAT, LON, tmp_path, transport=transport)


def test_a_message_that_is_not_the_aerosol_the_idx_promised_is_refused(tmp_path) -> None:
    transport = support.crop_transport(CYCLE_DATE, 0)
    # The idx says "total" first, but the bytes there are the dust message.
    url = gefs.chem_url(CYCLE_DATE, 0, 0)
    dust_then_total = support.edited(
        support.crop_message("DUST_AOD_550"), cycle=(CYCLE_DATE, 0)
    ) + support.edited(support.crop_message("TOTAL_AOD_550"), cycle=(CYCLE_DATE, 0))
    assert len(dust_then_total) == len(support.CROP_GRIB)
    transport.files[url] = dust_then_total
    transport.files[gefs.idx_url(CYCLE_DATE, 0, 0)] = support.CROP_IDX_TEXT.encode()
    with pytest.raises(gefs.AerosolArchiveError):
        gefs.fetch_aerosol(EVENING, LAT, LON, tmp_path, transport=transport)


def test_a_wrong_wavelength_is_refused(tmp_path) -> None:
    transport = support.crop_transport(CYCLE_DATE, 0)
    url = gefs.chem_url(CYCLE_DATE, 0, 0)
    blob = b"".join(
        support.edited(
            support.crop_message(name), cycle=(CYCLE_DATE, 0), scaledValueOfFirstWavelength=620
        )
        for name in support.CROP_MESSAGES
    )
    transport.files[url] = blob
    with pytest.raises(gefs.AerosolArchiveError, match="wavelength"):
        gefs.fetch_aerosol(EVENING, LAT, LON, tmp_path, transport=transport)


def test_a_point_outside_the_grid_is_an_error_not_the_nearest_edge_cell(tmp_path) -> None:
    transport = support.crop_transport(CYCLE_DATE, 0)
    with pytest.raises(gefs.AerosolArchiveError, match="outside"):
        gefs.fetch_aerosol(EVENING, 10.0, -60.0, tmp_path, transport=transport)


def test_an_optical_depth_that_is_negative_or_not_a_number_is_refused() -> None:
    with pytest.raises(gefs.AerosolArchiveError, match="TOTAL_AOD_550"):
        gefs.check_optical_depth("TOTAL_AOD_550", -0.02)
    with pytest.raises(gefs.AerosolArchiveError, match="DUST_AOD_550"):
        gefs.check_optical_depth("DUST_AOD_550", float("nan"))
    assert gefs.check_optical_depth("TOTAL_AOD_550", 0.0) == 0.0
    assert gefs.check_optical_depth("TOTAL_AOD_550", 0.3) == 0.3


def test_a_cell_the_file_leaves_empty_is_an_error() -> None:
    masked = np.ma.masked_all(support.CROP_SHAPE)
    with pytest.raises(gefs.AerosolArchiveError, match="no value"):
        gefs.cell_value("TOTAL_AOD_550", masked, support.REFERENCE_CELL)
    full = np.ma.array(np.full(support.CROP_SHAPE, 0.07))
    assert gefs.cell_value("TOTAL_AOD_550", full, support.REFERENCE_CELL) == 0.07
