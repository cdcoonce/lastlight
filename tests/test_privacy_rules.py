"""The text rules P001, P002, P003 and P005: one planted leak per rule, plus its clean twin.

Every test writes a temp file, scans it, and asserts the exact set of rule IDs, so a rule
that stops firing, or starts firing on its clean twin, turns exactly one test red.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lastlight.privacy import RULES, Finding, contains_clock_time, scan
from lastlight.site import PRIVATE_SITE_SENTINEL


def scan_text(tmp_path: Path, content: str | bytes, name: str = "note.txt") -> list[Finding]:
    path = tmp_path / name
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)
    return scan([path], root=tmp_path).findings


def rule_ids(findings: list[Finding]) -> list[str]:
    return sorted(finding.rule for finding in findings)


def test_rule_ids_and_names() -> None:
    assert RULES == {
        "P001": "clock-time",
        "P002": "sun-event-time",
        "P003": "coordinates",
        "P004": "image-metadata",
        "P005": "private-site",
    }


# --- P001 clock-time -------------------------------------------------------------------------

P001_LEAKS = [
    "17:42",
    "5:42 PM",
    "5:42pm",
    "05:42 a.m.",
    "17h42",
    "17:42:09",
    "17:42:09.125",
    "2026-10-10T00:24:00Z",
    "2026-10-10T00:24:00.5+00:00",
    "2026-10-10T00:24",
    "2026-10-10 17:42",
    "2026-10-10 17:42:09",
    "20261010T002400Z",
    "T17:42Z",
]


@pytest.mark.parametrize("leak", P001_LEAKS)
def test_p001_flags_each_clock_time_form(tmp_path: Path, leak: str) -> None:
    findings = scan_text(tmp_path, f"The frame was taken at {leak} sharp.\n")
    assert rule_ids(findings) == ["P001"]


@pytest.mark.parametrize("leak", P001_LEAKS)
def test_p001_leak_is_one_finding_even_when_it_is_a_full_datetime(
    tmp_path: Path, leak: str
) -> None:
    # A datetime is one leak, not a date-part finding plus a time-part finding.
    assert len(scan_text(tmp_path, f"{leak}\n")) == 1


def test_p001_reports_the_line_and_the_matched_text(tmp_path: Path) -> None:
    (finding,) = scan_text(tmp_path, "first line\nsecond line\nseal at 17:42 today\n")
    assert (finding.path, finding.line, finding.rule) == ("note.txt", 3, "P001")
    assert finding.excerpt == "17:42"


@pytest.mark.parametrize(
    "disguised",
    [
        "sunset-17:42.png",
        "ts17:42",
        "frame_17:42.json",
        "see/17:42",
        "time=17:42Z",
        "frame_20261010T0024",
    ],
)
def test_p001_is_not_hidden_by_the_characters_around_it(tmp_path: Path, disguised: str) -> None:
    assert "P001" in rule_ids(scan_text(tmp_path, disguised + "\n"))


@pytest.mark.parametrize(
    "clean",
    [
        "Observed on 2026-10-10 and again on 2026-10-11.",
        "python 3.12.4 and uv 0.8.0",
        "values[10:20] and rows[1:30]",
        "frame[a, 1:30]",
        "the zone is UTC-07:00 all year",
        "offsets are written GMT+5:30 and utc-07:00",
        "sha256:0123456789abcdef0123456789abcdef",
        "a 3:1 ratio and port 8080:80",
        "section 4.2 of the design",
        "delta-E 2000 of at least 4.0 in 2 of 3 bands",
    ],
)
def test_p001_clean_twins_pass(tmp_path: Path, clean: str) -> None:
    assert scan_text(tmp_path, clean + "\n") == []


def test_contains_clock_time_is_the_p001_pattern_for_other_modules() -> None:
    # The forecast schema reuses this to reject any string a scan would flag.
    assert contains_clock_time("sealed 2026-10-10T21:00:00Z")
    assert contains_clock_time("at 17:42")
    assert not contains_clock_time("2026-10-10")
    assert not contains_clock_time("sun elevation -0.8 degrees")


# --- P002 sun-event-time ---------------------------------------------------------------------


def test_p002_flags_a_sun_event_word_next_to_a_time_that_p001_does_not_see(
    tmp_path: Path,
) -> None:
    assert rule_ids(scan_text(tmp_path, "Sunset is around 5 PM tonight.\n")) == ["P002"]


@pytest.mark.parametrize(
    "leak",
    [
        "sunset around 5 PM",
        "SUNRISE about 6am",
        "dusk at 1742Z",
        "dawn 0612 hrs",
        "5 PM is when the sunset starts",
        "around 6 a.m. the dawn breaks",
    ],
)
def test_p002_flags_each_sun_event_word_near_a_time(tmp_path: Path, leak: str) -> None:
    assert rule_ids(scan_text(tmp_path, leak + "\n")) == ["P002"]


def test_p002_alongside_p001_when_the_time_has_minutes(tmp_path: Path) -> None:
    assert rule_ids(scan_text(tmp_path, "sunset 17:42\n")) == ["P001", "P002"]


def test_p002_reports_the_word_and_its_time_as_one_excerpt(tmp_path: Path) -> None:
    (finding,) = scan_text(tmp_path, "ok\nthe dusk at 5 PM\n")
    assert (finding.line, finding.rule, finding.excerpt) == (2, "P002", "dusk at 5 PM")


@pytest.mark.parametrize(
    "clean",
    [
        "The sunset was beautiful tonight.",
        "A 5 PM meeting about the forecast.",
        "dawn of a new model",
    ],
)
def test_p002_clean_twins_pass(tmp_path: Path, clean: str) -> None:
    assert scan_text(tmp_path, clean + "\n") == []


def test_p002_window_is_40_characters(tmp_path: Path) -> None:
    near = "sunset" + " " * 40 + "5 PM\n"
    far = "sunset" + " " * 41 + "5 PM\n"
    assert rule_ids(scan_text(tmp_path, near, "near.txt")) == ["P002"]
    assert scan_text(tmp_path, far, "far.txt") == []


def test_p002_window_applies_before_the_word_too(tmp_path: Path) -> None:
    near = "5 PM" + " " * 40 + "sunset\n"
    far = "5 PM" + " " * 41 + "sunset\n"
    assert rule_ids(scan_text(tmp_path, near, "near.txt")) == ["P002"]
    assert scan_text(tmp_path, far, "far.txt") == []


def test_p002_sees_sun_event_words_inside_identifiers_but_not_longer_words(
    tmp_path: Path,
) -> None:
    assert rule_ids(scan_text(tmp_path, "sunset_time = 5 PM\n", "key.txt")) == ["P002"]
    assert scan_text(tmp_path, "sunsetting at 5 PM\n", "longer.txt") == []


def test_p002_spans_a_line_break(tmp_path: Path) -> None:
    assert rule_ids(scan_text(tmp_path, "sunset\n5 PM\n")) == ["P002"]


# --- P003 coordinates ------------------------------------------------------------------------

P003_LEAKS = [
    "33.4484, -112.0740",
    "33.4484,-112.0740",
    "33.4484 N, 112.0740 W",
    "33.4484N 112.0740W",
    "33.4484°N, 112.0740°W",
    "33.4484, 112.0740",
    "-112.0740, 33.4484",
    "112.0740 W, 33.4484 N",
    "(33.4484, -112.0740)",
    "[-112.0740, 33.4484]",
    '{"lat": 33.4484, "lon": -112.0740}',
    "lat = 33.4484\nlon = -112.0740",
    "latitude 33.4484 and longitude -112.0740",
    "33.46, -112.07",
    "33.45, -112.06",
    "33.5, -112.1",
]


@pytest.mark.parametrize("leak", P003_LEAKS)
def test_p003_flags_each_coordinate_form(tmp_path: Path, leak: str) -> None:
    assert rule_ids(scan_text(tmp_path, f"camera at {leak}\n")) == ["P003"]


def test_p003_reports_the_pair_on_the_line_of_its_first_number(tmp_path: Path) -> None:
    (finding,) = scan_text(tmp_path, "a\nb\nlat = 33.4484\nlon = -112.0740\n")
    assert (finding.line, finding.rule) == (3, "P003")
    assert finding.excerpt == "33.4484 lon = -112.0740"


@pytest.mark.parametrize(
    "reference",
    [
        "33.45, -112.07",
        "33.45 N, 112.07 W",
        "33.45N 112.07W",
        "33.450, -112.070",
        "-112.07, 33.45",
        "33.45, 112.07",
        "(33.45, -112.07, 340.0)",
    ],
)
def test_p003_lets_the_public_reference_point_through(tmp_path: Path, reference: str) -> None:
    assert scan_text(tmp_path, f"reference point {reference}\n") == []


def test_p003_lets_the_reference_point_constant_through(tmp_path: Path) -> None:
    from lastlight.site import REFERENCE_POINT

    lat, lon, _elevation = REFERENCE_POINT
    assert scan_text(tmp_path, f"{lat}, {lon}\n") == []


def test_p003_flags_a_point_a_kilometre_from_the_reference(tmp_path: Path) -> None:
    # 0.01 degrees of latitude is about 1.1 km; of longitude about 0.9 km at 33.45 N.
    assert rule_ids(scan_text(tmp_path, "33.46, -112.07\n", "north.txt")) == ["P003"]
    assert rule_ids(scan_text(tmp_path, "33.45, -112.06\n", "east.txt")) == ["P003"]


@pytest.mark.parametrize(
    "outside",
    [
        "40.7128, -74.0060",  # New York
        "37.6, -112.07",  # just north of the box
        "30.9, -112.07",  # just south of the box
        "33.45, -108.9",  # just east of the box
        "33.45, -115.1",  # just west of the box
        "-33.4484, 112.0740",  # southern hemisphere
        "version 3.12.4",
        "10.0.0.1 and 33.45.1.2",
        "2.5, 3.75, 340.0",
    ],
)
def test_p003_ignores_pairs_outside_the_arizona_box(tmp_path: Path, outside: str) -> None:
    assert scan_text(tmp_path, f"{outside}\n") == []


@pytest.mark.parametrize("edge", ["31.0, -109.0", "37.5, -115.0", "37.5, 109.0", "31.0, 115.0"])
def test_p003_box_edges_are_inclusive(tmp_path: Path, edge: str) -> None:
    assert rule_ids(scan_text(tmp_path, edge + "\n")) == ["P003"]


def test_p003_needs_a_pair_not_a_single_number(tmp_path: Path) -> None:
    assert scan_text(tmp_path, "latitude 33.4484 only\n", "lat.txt") == []
    assert scan_text(tmp_path, "elevation -112.0740 m\n", "lon.txt") == []


def test_p003_does_not_pair_numbers_in_different_paragraphs(tmp_path: Path) -> None:
    assert scan_text(tmp_path, "latitude 33.4484\n\nlater, the value -112.0740\n") == []


def test_p003_does_not_pair_numbers_separated_by_other_numbers(tmp_path: Path) -> None:
    assert scan_text(tmp_path, "33.4484, 12, -112.0740\n") == []


# --- P005 private-site -----------------------------------------------------------------------


def test_p005_flags_the_sentinel_key_in_a_text_file(tmp_path: Path) -> None:
    findings = scan_text(tmp_path, f"ok\n{PRIVATE_SITE_SENTINEL} = true\n")
    assert rule_ids(findings) == ["P005"]
    assert (findings[0].line, findings[0].excerpt) == (2, PRIVATE_SITE_SENTINEL)


def test_p005_flags_the_sentinel_inside_a_binary_file(tmp_path: Path) -> None:
    blob = b"\x00\x01\x02" + PRIVATE_SITE_SENTINEL.encode() + b"\x00\xff"
    assert rule_ids(scan_text(tmp_path, blob, "blob.bin")) == ["P005"]


def test_p005_does_not_flag_the_site_file_environment_variable(tmp_path: Path) -> None:
    # `LASTLIGHT_SITE_FILE` names where the private file lives; it is not private content.
    assert scan_text(tmp_path, "export LASTLIGHT_SITE_FILE=/etc/site.toml\n") == []
    assert scan_text(tmp_path, "LASTLIGHT_PRIVATE is a prefix only\n", "prefix.txt") == []


# --- all rules on one file -------------------------------------------------------------------


def test_each_rule_is_reported_independently_in_one_file(tmp_path: Path) -> None:
    content = f"seal 17:42\ncamera 33.4484, -112.0740\n{PRIVATE_SITE_SENTINEL}\n"
    findings = scan_text(tmp_path, content)
    assert [(f.line, f.rule) for f in findings] == [(1, "P001"), (2, "P003"), (3, "P005")]


def test_a_clean_file_has_no_findings(tmp_path: Path) -> None:
    text = "Sun elevation -6 degrees on 2026-10-10 at 33.45, -112.07, 340 m.\n"
    assert scan_text(tmp_path, text) == []
