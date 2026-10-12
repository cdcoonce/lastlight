"""The G0 report, results file and figures (issue #10 acceptance criteria).

The report must state the criterion, every input source, every dropped date with its reason, the
per-band distribution, PASS or FAIL against the pre-registered criterion, the sensitivities, the
five most extreme dates, and carry figures; and the report, the results file and the figures must
pass the repository's privacy scanner: no clock time, no per-date cycle hour, no coordinate other
than the public reference point, no image metadata.
"""

from __future__ import annotations

import csv
import datetime as dt
import re

import pytest
from g0_support import dates, dropped_record, synthetic_record, whtm1_views
from PIL import Image

from lastlight.g0 import stats
from lastlight.g0.analysis import build_study
from lastlight.g0.record import RENDER_SCALES
from lastlight.g0.report import (
    FIGURE_FILES,
    OBSERVED_CONTRAST_RANGE,
    RESULTS_COLUMNS,
    write_outputs,
)
from lastlight.privacy.images import image_issues
from lastlight.privacy.scanner import scan

START = dt.date(2026, 1, 1)
MOVE = {"golden": 0.08, "early": 0.08}


def make_study(*, moved: int = 14, n_ok: int = 40, extra_drops: bool = True):
    days = dates(START, n_ok)
    records = [
        synthetic_record(
            day,
            {k: v * (1 + 0.1 * i) for k, v in MOVE.items()} if i < moved else {},
            aod=0.05 + 0.002 * i,
            dust=0.01 + 0.001 * i,
            lab_shift=float(i % 3),
        )
        for i, day in enumerate(days)
    ]
    end = START + dt.timedelta(days=n_ok - 1)
    if extra_drops:
        records.append(dropped_record(end + dt.timedelta(days=1), "hrrr_run_missing", "x"))
        records.append(dropped_record(end + dt.timedelta(days=2), "hrrr_run_missing"))
        records.append(dropped_record(end + dt.timedelta(days=3), "aerosol_run_missing"))
        end = end + dt.timedelta(days=4)  # the fourth extra date has no record: not_collected
    return build_study(records, START, end)


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    out = tmp_path_factory.mktemp("g0")
    study = make_study()
    paths = write_outputs(study, out, views=whtm1_views())
    return study, out, paths, (out / "report.md").read_text(encoding="utf-8")


# --- the files ---------------------------------------------------------------------------------


def test_the_outputs_are_the_report_the_results_and_three_figures(outputs) -> None:
    _, out, paths, _ = outputs
    assert sorted(p.name for p in out.iterdir()) == sorted(
        ["report.md", "results.csv", *FIGURE_FILES]
    )
    assert {p.name for p in paths} == {p.name for p in out.iterdir()}
    assert len(FIGURE_FILES) == 3 and all(name.endswith(".png") for name in FIGURE_FILES)


# --- what the report states --------------------------------------------------------------------


def test_the_report_quotes_the_pre_registered_criterion_and_says_where_it_is(outputs) -> None:
    *_, text = outputs
    assert stats.CRITERION_SENTENCE in text
    assert "docs/gates/G0/criterion.md" in text
    assert "0.05" in text and "2 of the 3" in text and "25 percent" in text


def test_the_verdict_is_pass_when_a_quarter_of_the_dates_meet_the_rule(outputs) -> None:
    study, _, _, text = outputs
    assert study.verdict.passed
    verdict_section = text.split("## Verdict")[1].split("\n## ")[0]
    assert "**PASS**" in verdict_section and "FAIL" not in verdict_section
    assert f"{study.verdict.meeting} of the {study.verdict.dates} dates" in verdict_section


def test_the_verdict_is_fail_when_fewer_than_a_quarter_meet_it(tmp_path) -> None:
    study = make_study(moved=5)  # 5 of 40 dates
    assert not study.verdict.passed
    write_outputs(study, tmp_path, views=whtm1_views())
    verdict_section = (tmp_path / "report.md").read_text().split("## Verdict")[1].split("\n## ")[0]
    assert "**FAIL**" in verdict_section and "PASS" not in verdict_section


def test_a_sensitivity_that_disagrees_is_reported_without_changing_the_verdict(tmp_path) -> None:
    days = dates(START, 20)
    records = [synthetic_record(day, MOVE if i < 6 else {}, low={}) for i, day in enumerate(days)]
    study = build_study(records, START, days[-1])
    write_outputs(study, tmp_path, views=whtm1_views())
    text = (tmp_path / "report.md").read_text()
    assert "**PASS**" in text.split("## Verdict")[1].split("\n## ")[0]
    sensitivity = text.split("## Pixel-scale sensitivity")[1].split("\n## ")[0]
    assert re.search(r"low[^\n]*FAIL", sensitivity) and re.search(r"high[^\n]*PASS", sensitivity)
    assert "does not change the verdict" in sensitivity


def test_the_inputs_actually_used_are_stated_with_the_run_selection_rule(outputs) -> None:
    *_, text = outputs
    inputs = text.split("## Inputs actually used")[1].split("\n## ")[0]
    for needle in (
        "HRRR",
        "forecast hour 0",
        "noaa-hrrr-bdp-pds",
        "AOTK",
        "PWAT",
        "GEFS-Aerosols",
        "noaa-gefs-pds",
        "550 nm",
        "SBUV",
        "nearest",
    ):
        assert needle in inputs, needle
    assert "whole UTC hour nearest" in inputs
    assert "latest" in inputs and "cycle" in inputs  # the aerosol rule: latest cycle before
    assert (
        "GEOS-FP" in inputs and "27 February 2026" in inputs
    )  # the archive tried first, and why not


def test_the_report_says_what_the_renderer_does_not_vary(outputs) -> None:
    *_, text = outputs
    assert "Angstrom" in text and "single-scattering albedo" in text
    assert "biases the spread low" in text or "bias the spread low" in text
    assert "cloud" in text.lower()


def test_every_date_attempted_is_counted_and_every_dropped_date_is_listed_with_its_reason(
    outputs,
) -> None:
    study, _, _, text = outputs
    dates_section = text.split("## Dates")[1].split("\n## ")[0]
    assert f"{len(study.attempted)} attempted" in dates_section
    assert f"{len(study.used)} used" in dates_section
    assert f"{len(study.dropped)} dropped" in dates_section
    for dropped in study.dropped:
        assert dropped.date.isoformat() in dates_section
    assert dates_section.count("hrrr_run_missing") >= 2
    assert "aerosol_run_missing" in dates_section and "not_collected" in dates_section
    assert "is not in the archive" in dates_section  # the reason's sentence, not just its code


def test_a_report_with_no_dropped_dates_says_so(tmp_path) -> None:
    study = make_study(extra_drops=False)
    write_outputs(study, tmp_path, views=whtm1_views())
    text = (tmp_path / "report.md").read_text()
    assert "No date was dropped" in text


def test_the_per_band_distribution_covers_all_four_bands_and_marks_late_as_not_counted(
    outputs,
) -> None:
    study, _, _, text = outputs
    section = text.split("## Distribution of D(d, s)")[1].split("\n## ")[0]
    for band in stats.BANDS:
        assert band in section
    late_row = next(line for line in section.splitlines() if line.startswith("| late"))
    assert "not counted" in late_row
    assert "10th" in section and "median" in section and "90th" in section
    assert "Dates with D >= 0.05" in section or "D >= 0.05" in section


def test_the_five_most_extreme_dates_are_listed_in_order(outputs) -> None:
    study, _, _, text = outputs
    section = text.split("## The five most extreme dates")[1].split("\n## ")[0]
    positions = [section.index(e.date.isoformat()) for e in study.extremes]
    assert len(study.extremes) == 5
    assert positions == sorted(positions)
    assert "aerosol optical depth" in section.lower()


def test_tier_b_delta_e_is_reported_without_a_criterion(outputs) -> None:
    _, _, _, text = outputs
    section = text.split("## Tier B")[1].split("\n## ")[0]
    assert "delta-E 2000" in section
    for band in stats.DELTA_E_BANDS:
        assert band in section
    assert "no pass criterion" in section.lower()


def test_the_seasonal_control_and_the_floor_are_reported(outputs) -> None:
    _, _, _, text = outputs
    assert "## Seasonal control" in text and "15 days" in text
    assert "## The channel floor" in text and "of the rendered channels" in text


def test_the_geometry_table_gives_the_rows_and_their_nominal_elevations(outputs) -> None:
    _, _, _, text = outputs
    section = text.split("## WHTM1's view")[1].split("\n## ")[0]
    for rows in ("8-28", "28-48", "48-68", "68-88", "88-108", "118-138"):
        assert rows in section
    assert "px per degree" in section and "provisional" in section


def test_the_report_links_its_figures(outputs) -> None:
    *_, text = outputs
    for name in FIGURE_FILES:
        assert f"]({name})" in text


# --- the results file --------------------------------------------------------------------------


def test_the_results_file_has_one_row_per_date_attempted(outputs) -> None:
    study, out, *_ = outputs
    with (out / "results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == list(RESULTS_COLUMNS)
    assert [r["date"] for r in rows] == [d.isoformat() for d in study.attempted]
    by_date = {r["date"]: r for r in rows}
    used = by_date[START.isoformat()]
    assert used["status"] == "used" and used["reason"] == ""
    assert float(used["D_golden"]) == pytest.approx(0.08, abs=1e-4)
    assert used["meets_rule"] == "true"
    assert used["counted_bands_at_threshold"] == "2"
    last_used = by_date[(START + dt.timedelta(days=39)).isoformat()]
    assert last_used["meets_rule"] == "false"
    dropped = next(r for r in rows if r["status"] == "dropped")
    assert dropped["reason"] in {"hrrr_run_missing", "aerosol_run_missing", "not_collected"}
    assert dropped["D_golden"] == "" and dropped["meets_rule"] == ""


def test_the_results_file_carries_flags_and_inputs_for_each_used_date(outputs) -> None:
    study, out, *_ = outputs
    with (out / "results.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert float(row["aod_total"]) == pytest.approx(0.05, abs=1e-4)
    assert float(row["aod_dust"]) == pytest.approx(0.01, abs=1e-4)
    assert row["meets_rule_scale_low"] in {"true", "false"}
    assert row["meets_rule_scale_high"] in {"true", "false"}
    assert row["seasonal_meets_rule"] in {"true", "false"}
    assert float(row["dE2000_early"]) >= 0.0
    assert int(row["floor_channels"]) == 3


# --- privacy -----------------------------------------------------------------------------------


def test_the_outputs_pass_the_privacy_scanner(outputs) -> None:
    _, out, *_ = outputs
    result = scan([out], root=out)
    assert result.files_scanned == 5
    assert result.findings == [], [str(f) for f in result.findings]


def test_the_report_has_no_clock_time_no_cycle_hour_and_no_run_url(outputs) -> None:
    *_, text = outputs
    assert not re.search(r"\b\d{1,2}:\d{2}\b", text)
    assert not re.search(r"\bt\d{2}z\b", text, re.IGNORECASE)
    assert not re.search(r"\bcycle (hour )?\d", text, re.IGNORECASE)
    assert "https://noaa-hrrr" not in text and "wrfsfc" not in text
    assert "sunset" not in text.lower()


def test_the_figures_are_valid_pngs_with_no_metadata(outputs) -> None:
    _, out, *_ = outputs
    for name in FIGURE_FILES:
        data = (out / name).read_bytes()
        assert image_issues(data, ".png") == [], name
        with Image.open(out / name) as image:
            assert image.format == "PNG"
            assert image.width >= 600 and image.height >= 300
            assert image.info == {}  # nothing attached: no text, no profile, no dpi
            colours = image.convert("RGB").getcolors(maxcolors=100000)
            assert colours is not None and len(colours) > 5


def test_writing_twice_gives_identical_files(tmp_path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    study = make_study()
    write_outputs(study, first, views=whtm1_views())
    write_outputs(study, second, views=whtm1_views())
    for name in ("report.md", "results.csv", *FIGURE_FILES):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_an_existing_output_directory_is_reused_and_the_files_replaced(tmp_path) -> None:
    study = make_study()
    (tmp_path / "report.md").write_text("old")
    write_outputs(study, tmp_path, views=whtm1_views())
    assert (tmp_path / "report.md").read_text().startswith("# G0")


# --- post hoc additions: channels, the smoke input, the aerosol by month -----------------------


def test_the_channel_decomposition_is_marked_post_hoc_and_changes_nothing(outputs) -> None:
    study, _, _, text = outputs
    section = text.split("## Channel decomposition (post hoc)")[1].split("\n## ")[0]
    assert "post hoc" in section and "not part of the pre-registered criterion" in section
    assert "does not change the verdict" in section
    assert "log(R/G)" in section and "log(B/G)" in section
    for band in stats.BANDS:
        assert band in section
    assert "**PASS**" in text.split("## Verdict")[1].split("\n## ")[0]


def test_the_channel_table_gives_the_median_d_and_the_share_meeting_the_rule(outputs) -> None:
    study, _, _, text = outputs
    section = text.split("## Channel decomposition (post hoc)")[1].split("\n## ")[0]
    blue_row = next(line for line in section.splitlines() if line.startswith("| log(B/G)"))
    meeting = study.channels["log(B/G)"].verdict.meeting
    assert f"{meeting} of {study.channels['log(B/G)'].verdict.dates}" in blue_row


def test_a_smoke_field_that_is_zero_everywhere_is_said_to_carry_nothing(outputs) -> None:
    study, _, _, text = outputs
    section = text.split("## Inputs actually used")[1].split("\n## ")[0]
    assert f"exactly zero at every point of the sightline on all {len(study.used)} dates" in section
    assert "all aerosol variation" in section


def test_a_smoke_field_with_values_is_summarised_instead(tmp_path) -> None:
    days = dates(START, 20)
    records = [synthetic_record(day, smoke=0.05 if i == 3 else 0.0) for i, day in enumerate(days)]
    write_outputs(build_study(records, START, days[-1]), tmp_path, views=whtm1_views())
    section = (tmp_path / "report.md").read_text().split("## Inputs actually used")[1]
    section = section.split("\n## ")[0]
    assert "non-zero on 1 of 20 dates" in section
    assert "exactly zero at every point" not in section


def test_the_aerosol_input_is_summarised_by_month(outputs) -> None:
    study, _, _, text = outputs
    section = text.split("## Aerosol input by month")[1].split("\n## ")[0]
    assert "2026-01" in section and "2026-02" in section
    assert "dust" in section.lower() and "median" in section
    january = next(line for line in section.splitlines() if line.startswith("| 2026-01"))
    assert f"| {sum(1 for d in study.used_dates if d.month == 1)} |" in january


# --- reading the result: what carries D, and how large the rendered contrasts are --------------


def test_the_reading_section_says_how_much_of_d_each_channel_carries(outputs) -> None:
    _, _, _, text = outputs
    reading = text.split("## Reading the result")[1]
    assert "log(B/G) carries 50 percent of the summed D(d, s) over the counted bands" in reading
    assert "nearly zero" in reading and "very sensitive" in reading  # why the blue ratio matters


def test_the_reading_section_gives_the_size_of_the_rendered_contrast_at_the_horizon(
    outputs,
) -> None:
    _, _, _, text = outputs
    reading = text.split("## Reading the result")[1]
    assert "At a sun elevation of 0" in reading
    assert "+0.00 in log(R/G) and +0.00 in log(B/G)" in reading
    assert "docs/log/2026-10-11-observed-contrasts.md" in reading
    assert "multiple scattering" in reading


def test_the_channel_floor_sentence_reads_cleanly(outputs) -> None:
    _, _, _, text = outputs
    floor = text.split("## The channel floor")[1].split("\n## ")[0]
    assert "of the rendered channels at the nominal scale" in floor
    assert "can only hide variation" not in floor


# --- the verdict section says what the PASS or FAIL is made of ---------------------------------

BLUE_OFFSET, RED_OFFSET = -2.62, 0.5


def make_blue_study(
    *, moved: int = 14, n: int = 40, smoke: float = 0.0, spread: bool = True, amount: float = 0.2
):
    """Dates that differ only in log(B/G): `moved` of them are `amount` away in golden and early.

    With `spread` the moved dates are every third one (a minority in every window, so the
    seasonal control sees them too); otherwise they are the first `moved` dates, a block that the
    windowed median partly absorbs. Every date also carries the same constant offset in each
    channel (no effect on D), so the contrast at a sun elevation of 0 is known: +0.5 in
    log(R/G) and -2.62 in log(B/G).
    """
    days = dates(START, n)
    records = [synthetic_record(day, {}, smoke=smoke) for day in days]
    shifted = range(0, n, 3) if spread else range(moved)
    for i, record in enumerate(records):
        for scale in RENDER_SCALES:
            record.contrast[scale][:, :, 0] += RED_OFFSET
            record.contrast[scale][:, :, 1] += BLUE_OFFSET
        if i in shifted:
            for band in ("golden", "early"):
                record.contrast["nominal"][list(stats.band_members(band)), :, 1] += amount
    return build_study(records, START, days[-1])


def verdict_section_of(study, tmp_path) -> str:
    write_outputs(study, tmp_path, views=whtm1_views())
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    return text.split("## Verdict")[1].split("\n## ")[0]


@pytest.fixture(scope="module")
def blue_pass(tmp_path_factory):
    study = make_blue_study()  # 14 of 40 dates (35 percent): a PASS, carried by blue alone
    return study, verdict_section_of(study, tmp_path_factory.mktemp("blue_pass"))


def test_a_pass_is_read_in_renderer_units_right_after_the_pass_line(blue_pass) -> None:
    study, section = blue_pass
    assert study.verdict.passed
    pass_line = section.index("**PASS** against the pre-registered criterion.")
    caveat = section.index("**Read this PASS in renderer-v0 units.**")
    numbers = section.index(f"{study.verdict.meeting} of the {study.verdict.dates} dates used")
    assert pass_line < caveat < numbers
    assert "FAIL" not in section


def test_the_caveat_names_the_one_channel_the_pass_rests_on_and_each_channels_share(
    blue_pass,
) -> None:
    _, section = blue_pass
    assert (
        "the PASS rests on one channel: log(B/G) alone meets the rule on 35.0 percent of dates"
        in (section)
    )
    assert "and log(R/G) alone on 0.0 percent" in section


def test_the_caveat_gives_the_seasonal_controls_channel_shares(blue_pass) -> None:
    study, section = blue_pass
    blue = study.seasonal_channels["log(B/G)"].verdict
    red = study.seasonal_channels["log(R/G)"].verdict
    assert (blue.meeting, red.meeting) == (
        14,
        0,
    )  # a third of the dates are in no window's majority
    assert (
        "the seasonal control rests on the same channel (log(B/G) alone 35.0 percent, "
        "log(R/G) alone 0.0 percent)" in section
    )


def test_the_seasonal_control_is_not_said_to_rest_on_the_channel_when_it_does_not_meet_the_bar(
    tmp_path,
) -> None:
    # The first 14 dates are a block: the median of the dates around each of them is the shifted
    # value, so the control finds only 2 of 40 dates away from it. It does not pass on blue alone.
    study = make_blue_study(spread=False)
    section = verdict_section_of(study, tmp_path)
    assert "the PASS rests on one channel: log(B/G) alone meets the rule on 35.0 percent" in section
    assert "the seasonal control rests on the same channel" not in section
    assert "for the seasonal control, log(B/G) alone 5.0 percent, log(R/G) alone 0.0 percent" in (
        section
    )


def test_the_caveat_gives_the_rendered_contrast_at_the_horizon_beside_the_observed_range(
    blue_pass,
) -> None:
    _, section = blue_pass
    assert OBSERVED_CONTRAST_RANGE == (-0.21, 0.14)
    assert "At a sun elevation of 0 the rendered lowest-band log(B/G) contrast is -2.62" in section
    assert "read between -0.21 and +0.14 in either channel" in section
    assert "so 0.05 in these units is not 0.05 on a webcam" in section


def test_the_caveat_says_where_the_aerosol_variation_comes_from(blue_pass) -> None:
    _, section = blue_pass
    assert "All aerosol variation comes from GEFS-Aerosols' modelled optical depth" in section
    assert "HRRR's smoke field was exactly zero in the analyses" in section


def test_the_caveat_says_what_the_pass_does_not_show(blue_pass) -> None:
    _, section = blue_pass
    assert "moves from day to day by more than 0.05 in its own units" in section
    assert "does not show that a camera can see that movement (G1, issue #31, measures that)" in (
        section
    )
    assert "does not yet support a claim that the forecast can beat climatology or persistence" in (
        section
    )
    assert "a multiple-scattering renderer (issue #32) re-runs this gate" in section


def test_the_verdict_section_keeps_the_numbers_and_the_rule_as_registered(blue_pass) -> None:
    study, section = blue_pass
    assert (
        f"{study.verdict.meeting} of the {study.verdict.dates} dates used (35.0 percent)" in section
    )
    assert "The rule is applied as it was written before any result existed." in section


def test_a_fail_is_read_in_renderer_units_too_and_claims_nothing_about_the_forecast(
    tmp_path,
) -> None:
    study = make_blue_study(moved=5, spread=False)  # 5 of 40 dates (12.5 percent)
    assert not study.verdict.passed
    section = verdict_section_of(study, tmp_path)
    assert section.index("**FAIL** against the pre-registered criterion.") < section.index(
        "**Read this FAIL in renderer-v0 units.**"
    )
    assert "This FAIL says the renderer's output moves from day to day by less than 0.05" in section
    assert (
        "does not show that a camera cannot see aerosol-driven movement (G1, issue #31" in section
    )
    assert "does not show that the forecast cannot beat climatology or persistence" in section
    assert "PASS" not in section
    assert "does not yet support a claim" not in section


def test_a_fail_that_one_channel_alone_would_pass_is_not_said_to_rest_on_that_channel(
    tmp_path,
) -> None:
    # Blue moves by 0.08: D is 0.08 in log(B/G) alone (a hit on 35 percent of dates) but 0.04
    # over both channels, which is below the threshold, so the verdict is a FAIL.
    study = make_blue_study(amount=0.08)
    assert not study.verdict.passed and study.channels["log(B/G)"].verdict.passed
    section = verdict_section_of(study, tmp_path)
    assert "Read this FAIL in renderer-v0 units" in section
    assert "rests on one channel" not in section
    assert "by channel, log(B/G) alone meets the rule on 35.0 percent of dates" in section


def test_when_both_channels_carry_the_result_it_is_not_said_to_rest_on_one(tmp_path) -> None:
    study = make_study()  # the same shift in both channels
    section = verdict_section_of(study, tmp_path)
    assert "Read this PASS in renderer-v0 units" in section
    assert "rests on one channel" not in section
    assert "by channel, log(B/G) alone meets the rule on 35.0 percent of dates" in section
    assert "log(R/G) alone on 35.0 percent" in section
    assert "for the seasonal control, log(B/G) alone" in section


def test_a_smoke_field_with_values_is_not_said_to_be_zero_in_the_caveat(tmp_path) -> None:
    section = verdict_section_of(make_blue_study(smoke=0.05), tmp_path)
    assert "HRRR's smoke field was exactly zero" not in section
    assert "HRRR's smoke was recorded and not rendered" in section


def test_the_caveat_adds_no_clock_time_cycle_hour_or_coordinate(tmp_path) -> None:
    study = make_blue_study()
    write_outputs(study, tmp_path, views=whtm1_views())
    result = scan([tmp_path], root=tmp_path)
    assert result.findings == [], [str(f) for f in result.findings]


# --- the other additions: attribution, the aerosol input is a model, the seasonal channels -----


def test_the_inputs_credit_noaa_and_say_the_numbers_are_derived_with_no_endorsement(
    outputs,
) -> None:
    *_, text = outputs
    inputs = text.split("## Inputs actually used")[1].split("\n## ")[0]
    assert "GEFS-Aerosols and HRRR data are from NOAA, through the AWS Open Data registry" in inputs
    assert "derived values" in inputs and "no endorsement by NOAA is implied" in inputs


def test_the_inputs_say_the_aerosol_is_a_modelled_value_at_one_cell_not_a_measurement(
    outputs,
) -> None:
    *_, text = outputs
    inputs = text.split("## Inputs actually used")[1].split("\n## ")[0]
    assert "a modelled optical depth at one 25 km grid cell, not a measurement" in inputs
    assert "was not verified" in inputs and "assimilates aerosol observations" in inputs


def test_the_seasonal_control_is_what_a_forecast_would_have_to_beat_in_renderer_units(
    outputs,
) -> None:
    study, _, _, text = outputs
    section = text.split("## Seasonal control")[1].split("\n## ")[0]
    assert "what a forecast would have to beat, in renderer units" in section
    assert "what a forecast has to beat" not in section
    for name, outcome in study.seasonal_channels.items():
        verdict = outcome.verdict
        assert f"{name} alone, {verdict.meeting} of {verdict.dates}" in section
