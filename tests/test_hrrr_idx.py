"""HRRR `.idx` parsing and byte-range selection, URLs, and the default field list (issue #6).

The selection tests run against a real `.idx` recorded from the public bucket
(`tests/fixtures/hrrr/hrrr.t22z.wrfsfcf00.grib2.idx`, 170 messages). The expected byte ranges
below were read off its lines by hand: a message starts at its own offset and ends one byte
before the next line's offset.
"""

from __future__ import annotations

import datetime as dt

import hrrr_support as support
import pytest

from lastlight.inputs.hrrr import (
    DEFAULT_FIELDS,
    HrrrError,
    IdxEntry,
    grib_url,
    idx_url,
    parse_idx,
    select_ranges,
)
from lastlight.inputs.http import ByteRange

# (start, end) of each default field's message in the recorded idx. The end is the next line's
# offset minus one. TCDC appears twice in the file (boundary layer, entire atmosphere); the
# default is the entire atmosphere, which is the line directly after high cloud.
EXPECTED_RANGES = {
    "VIS": (2185034, 3623784),
    "PWAT": (66639229, 67666233),
    "AOTK": (67666234, 67666421),
    "LCDC": (71452294, 72222759),
    "MCDC": (72222760, 72774874),
    "HCDC": (72774875, 73220705),
    "TCDC": (73220706, 74130205),
    "HGT_CLOUD_BASE": (75244277, 77423953),
    "HGT_CLOUD_TOP": (79309484, 80386433),
}


def _spec(name: str):
    return next(spec for spec in DEFAULT_FIELDS if spec.name == name)


# --- the default fields ---------------------------------------------------------------------------


def test_default_fields_are_the_nine_the_issue_names() -> None:
    assert [spec.name for spec in DEFAULT_FIELDS] == [
        "LCDC",
        "MCDC",
        "HCDC",
        "TCDC",
        "AOTK",
        "PWAT",
        "VIS",
        "HGT_CLOUD_BASE",
        "HGT_CLOUD_TOP",
    ]


def test_default_fields_carry_the_idx_labels_and_units() -> None:
    labels = {spec.name: (spec.variable, spec.level, spec.units) for spec in DEFAULT_FIELDS}
    assert labels == {
        "LCDC": ("LCDC", "low cloud layer", "%"),
        "MCDC": ("MCDC", "middle cloud layer", "%"),
        "HCDC": ("HCDC", "high cloud layer", "%"),
        "TCDC": ("TCDC", "entire atmosphere", "%"),
        "AOTK": ("AOTK", "entire atmosphere (considered as a single layer)", "1"),
        "PWAT": ("PWAT", "entire atmosphere (considered as a single layer)", "kg m-2"),
        "VIS": ("VIS", "surface", "m"),
        "HGT_CLOUD_BASE": ("HGT", "cloud base", "gpm"),
        "HGT_CLOUD_TOP": ("HGT", "cloud top", "gpm"),
    }


# --- parsing --------------------------------------------------------------------------------------


def test_parse_idx_reads_every_message_of_a_real_idx() -> None:
    entries = parse_idx(support.REAL_IDX_TEXT)

    assert len(entries) == 170
    assert [entry.number for entry in entries] == list(range(1, 171))
    assert entries[0] == IdxEntry(
        number=1,
        offset=0,
        date_code="2026101022",
        variable="REFC",
        level="entire atmosphere",
        forecast="anl",
    )
    vis = entries[4]
    assert (vis.variable, vis.level, vis.offset) == ("VIS", "surface", 2185034)


def test_parse_idx_keeps_the_forecast_wording_of_a_later_forecast_hour() -> None:
    entries = parse_idx("5:2227812:d=2026101012:VIS:surface:6 hour fcst:\n")
    assert entries[0].forecast == "6 hour fcst"


def test_parse_idx_tolerates_blank_lines_and_a_missing_final_newline() -> None:
    text = "1:0:d=2026101022:VIS:surface:anl:\n\n2:100:d=2026101022:PWAT:x:anl:"
    assert [entry.variable for entry in parse_idx(text)] == ["VIS", "PWAT"]


@pytest.mark.parametrize(
    "bad",
    [
        "not an idx line\n",
        "1:abc:d=2026101022:VIS:surface:anl:\n",
        "x:0:d=2026101022:VIS:surface:anl:\n",
        "1:0:d=2026101022:VIS\n",
    ],
)
def test_parse_idx_refuses_a_malformed_line(bad: str) -> None:
    with pytest.raises(HrrrError, match="idx"):
        parse_idx(bad)


def test_parse_idx_refuses_offsets_that_do_not_increase() -> None:
    text = "1:500:d=2026101022:VIS:surface:anl:\n2:400:d=2026101022:PWAT:x:anl:\n"
    with pytest.raises(HrrrError, match="offset"):
        parse_idx(text)


# --- selecting byte ranges ------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(EXPECTED_RANGES))
def test_select_ranges_picks_the_right_bytes_for_each_default_field(name: str) -> None:
    entries = parse_idx(support.REAL_IDX_TEXT)

    ranges = select_ranges(entries, DEFAULT_FIELDS)

    assert ranges[name] == ByteRange(*EXPECTED_RANGES[name])


def test_select_ranges_returns_exactly_the_requested_fields_in_order() -> None:
    entries = parse_idx(support.REAL_IDX_TEXT)
    wanted = [_spec("PWAT"), _spec("LCDC")]

    assert list(select_ranges(entries, wanted)) == ["PWAT", "LCDC"]


def test_total_cloud_cover_is_the_entire_atmosphere_not_the_boundary_layer() -> None:
    entries = parse_idx(support.REAL_IDX_TEXT)
    boundary_layer = next(
        entry
        for entry in entries
        if entry.variable == "TCDC" and entry.level == "boundary layer cloud layer"
    )

    chosen = select_ranges(entries, [_spec("TCDC")])["TCDC"]

    assert chosen.start != boundary_layer.offset
    assert chosen.start == EXPECTED_RANGES["TCDC"][0]


def test_cloud_base_and_cloud_top_are_distinct_messages_of_the_same_variable() -> None:
    entries = parse_idx(support.REAL_IDX_TEXT)

    ranges = select_ranges(entries, [_spec("HGT_CLOUD_BASE"), _spec("HGT_CLOUD_TOP")])

    assert ranges["HGT_CLOUD_BASE"] != ranges["HGT_CLOUD_TOP"]
    assert ranges["HGT_CLOUD_BASE"].start == 75244277
    assert ranges["HGT_CLOUD_TOP"].start == 79309484


def test_the_last_message_of_a_file_has_an_open_ended_range() -> None:
    text = (
        "1:0:d=2026101022:PWAT:entire atmosphere (considered as a single layer):anl:\n"
        "2:100:d=2026101022:VIS:surface:anl:\n"
    )

    ranges = select_ranges(parse_idx(text), [_spec("VIS"), _spec("PWAT")])

    assert ranges["VIS"] == ByteRange(100, None)
    assert ranges["PWAT"] == ByteRange(0, 99)


def test_a_field_missing_from_the_idx_is_a_loud_error_naming_it() -> None:
    entries = parse_idx("1:0:d=2026101022:VIS:surface:anl:\n")

    with pytest.raises(HrrrError, match="LCDC"):
        select_ranges(entries, [_spec("LCDC")])


def test_a_field_that_matches_two_messages_is_a_loud_error() -> None:
    text = "1:0:d=2026101022:VIS:surface:anl:\n2:100:d=2026101022:VIS:surface:anl:\n"

    with pytest.raises(HrrrError, match="VIS"):
        select_ranges(parse_idx(text), [_spec("VIS")])


# --- URLs -----------------------------------------------------------------------------------------


def test_grib_and_idx_urls_name_the_public_bucket_layout() -> None:
    grib = (
        "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20261010/conus/hrrr.t22z.wrfsfcf00.grib2"
    )

    assert grib_url(dt.date(2026, 10, 10), 22, 0) == grib
    assert idx_url(dt.date(2026, 10, 10), 22, 0) == grib + ".idx"


def test_urls_zero_pad_the_hour_and_forecast_hour() -> None:
    url = grib_url(dt.date(2026, 1, 5), 3, 7)
    assert url.endswith("/hrrr.20260105/conus/hrrr.t03z.wrfsfcf07.grib2")


@pytest.mark.parametrize(
    ("cycle_hour", "forecast_hour"),
    [(24, 0), (-1, 0), (22, -1), (22, 19), (12, 49), (0, 49)],
)
def test_urls_refuse_hours_the_model_does_not_run(cycle_hour: int, forecast_hour: int) -> None:
    with pytest.raises(ValueError, match="hour"):
        grib_url(dt.date(2026, 10, 10), cycle_hour, forecast_hour)


@pytest.mark.parametrize("cycle_hour", [0, 6, 12, 18])
def test_the_four_long_cycles_run_to_forecast_hour_48(cycle_hour: int) -> None:
    assert grib_url(dt.date(2026, 10, 10), cycle_hour, 48).endswith("wrfsfcf48.grib2")
