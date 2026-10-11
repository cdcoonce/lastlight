"""The G0 report, the per-date results file and the figures, written from a `Study`.

`write_outputs` writes four kinds of file into one directory: `report.md` (the criterion, the
inputs actually used, every dropped date with its reason, the per-band distribution of D(d, s),
PASS or FAIL against the pre-registered criterion, the sensitivities, the five most extreme
dates), `results.csv` (one machine-readable row per date attempted) and three metadata-free PNG
figures. Everything is derived from the study alone, so the same study gives the same bytes.

Privacy (`docs/design.md` section 3): the outputs carry dates, sun elevations and derived numbers
only. No clock time, no HRRR or GEFS cycle hour of any date (the run-selection rules are stated
as rules), no per-date sun azimuth, no coordinate other than the public reference point. The
camera's azimuth relative to the sun, which the renderer needs at every sweep elevation of every
date, is computed internally and never written here. The build log gives its monthly ranges, and
the fit's centre azimuth is in the report: both are functions of the public reference point and
a public camera's fit, so neither says more than those inputs already do. The tests run the
repository's own scanner over everything written here.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import numpy as np

from lastlight.cameras import load_registry
from lastlight.g0 import stats
from lastlight.g0.analysis import CHANNEL_NAMES, NOMINAL, ScaleOutcome, Study
from lastlight.g0.figures import FIGURES
from lastlight.g0.record import DROP_REASONS, RENDER_SCALES
from lastlight.g0.view import SkyView
from lastlight.geometry import ELEVATION_SWEEP

FIGURE_FILES: Final = tuple(FIGURES)

RESULTS_COLUMNS: Final = (
    "date",
    "status",
    "reason",
    "aod_total",
    "aod_dust",
    "aod_hrrr_smoke",
    "pwat_kg_m2",
    "ozone_du",
    "D_golden",
    "D_early",
    "D_mid",
    "D_late",
    "counted_bands_at_threshold",
    "meets_rule",
    "meets_rule_scale_low",
    "meets_rule_scale_high",
    "seasonal_counted_bands_at_threshold",
    "seasonal_meets_rule",
    "dE2000_early",
    "dE2000_mid",
    "dE2000_late",
    "floor_channels",
)
"""The columns of `results.csv`, in order. Dropped dates leave everything after `reason` empty."""

_CRITERION_FILE: Final = "docs/gates/G0/criterion.md"

OBSERVED_CONTRAST_RANGE: Final = (-0.21, 0.14)
"""The lowest and highest of the eight lowest-band contrasts (log(R/G) and log(B/G)) of the four
usable frames of the first observed evening, an overcast one, for sun elevations from +5 to -4.
Cited from the real-run table of `docs/log/2026-10-11-observed-contrasts.md` (PR #29), not read:
the report generator has no observation to read."""


def _percent(fraction: float) -> str:
    return f"{100.0 * fraction:.1f} percent"


def _word(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


# --- results.csv ------------------------------------------------------------------------------


def results_csv(study: Study) -> str:
    """One row per date attempted, in date order."""
    nominal, low, high = (study.outcomes[name] for name in RENDER_SCALES)
    index = {record.date: position for position, record in enumerate(study.used)}
    dropped = {record.date: record for record in study.dropped}
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(RESULTS_COLUMNS)
    for day in study.attempted:
        if day in dropped:
            padding = [""] * (len(RESULTS_COLUMNS) - 3)
            writer.writerow([day.isoformat(), "dropped", dropped[day].drop_reason, *padding])
            continue
        i = index[day]
        record = study.used[i]
        atmosphere = record.atmosphere
        assert atmosphere is not None
        pwat = "" if atmosphere.pwat_kg_m2 is None else f"{atmosphere.pwat_kg_m2:.1f}"
        writer.writerow(
            [
                day.isoformat(),
                "used",
                "",
                f"{atmosphere.aod_total:.4f}",
                f"{atmosphere.aod_dust:.4f}",
                f"{atmosphere.aod_smoke:.4f}",
                pwat,
                f"{atmosphere.ozone_du:.1f}",
                *(f"{value:.4f}" for value in nominal.distances[i]),
                int(nominal.at_threshold[i]),
                _bool(bool(nominal.meets[i])),
                _bool(bool(low.meets[i])),
                _bool(bool(high.meets[i])),
                int(study.seasonal.at_threshold[i]),
                _bool(bool(study.seasonal.meets[i])),
                *(f"{value:.3f}" for value in study.delta_e[i]),
                record.floor_channels[NOMINAL],
            ]
        )
    return buffer.getvalue()


# --- report.md --------------------------------------------------------------------------------


def _verdict_section(study: Study) -> str:
    verdict, seasonal = study.verdict, study.seasonal.verdict
    counted = ", ".join(study.counted_bands)
    lines = [
        "## Verdict",
        "",
        f"**{_word(verdict.passed)}** against the pre-registered criterion.",
        "",
        _caveat(study),
        "",
        f"{verdict.meeting} of the {verdict.dates} dates used ({_percent(verdict.fraction)}) "
        f"meet the rule: the rendered contrast differs from the period median by a D(d, s) of at "
        f"least {stats.THRESHOLD} in at least {stats.MIN_BANDS} of the "
        f"{len(study.counted_bands)} counted bands ({counted}). The criterion asks for at least "
        f"{_percent(stats.MIN_FRACTION)} of the dates. Counting every date attempted "
        f"({len(study.attempted)}), the share is {_percent(study.fraction_of_attempted)}.",
        "",
        "The rule is applied as it was written before any result existed. The pixel-scale "
        "sensitivity and the seasonal control below are reported and do not change it. For the "
        f"control, {seasonal.meeting} of the {seasonal.dates} dates "
        f"({_percent(seasonal.fraction)}) meet the rule when the median is taken over the dates "
        f"within {stats.HALF_WINDOW_DAYS} days instead of over the whole period.",
        "",
    ]
    return "\n".join(lines)


def _carrier(outcomes: Mapping[str, ScaleOutcome]) -> str | None:
    """The one channel that meets the rule's share alone; None when no channel or both do."""
    carriers = [name for name, outcome in outcomes.items() if outcome.verdict.passed]
    return carriers[0] if len(carriers) == 1 else None


def _smoke_values(study: Study) -> list[float]:
    return [record.atmosphere.aod_smoke for record in study.used if record.atmosphere]


def _horizon_contrast(study: Study) -> tuple[float, float]:
    """The median over the dates used of the lowest band's contrast at a sun elevation of 0:
    (log(R/G), log(B/G))."""
    horizon = list(ELEVATION_SWEEP).index(0.0)
    red, blue = (
        float(np.median([record.contrast[NOMINAL][horizon, 0, channel] for record in study.used]))
        for channel in (0, 1)
    )
    return red, blue


def _caveat(study: Study) -> str:
    """What the verdict is made of, directly under it: renderer units, one channel, a model.

    Every number is computed from the study except the observed range, which is cited
    (`OBSERVED_CONTRAST_RANGE`). The claims that depend on the data (that one channel carries the
    result, that the seasonal control rests on the same one, that HRRR's smoke was zero) are
    made only when the data say so.
    """
    passed = study.verdict.passed
    word = _word(passed)
    red, blue = CHANNEL_NAMES
    share = {name: _percent(study.channels[name].verdict.fraction) for name in CHANNEL_NAMES}
    split = f"{blue} alone meets the rule on {share[blue]} of dates and {red} alone on {share[red]}"
    carrier = _carrier(study.channels)
    channels = (
        f"the {word} rests on one channel: {split}"
        if passed and carrier
        else (f"by channel, {split}")
    )
    seasonal = {
        name: _percent(study.seasonal_channels[name].verdict.fraction) for name in CHANNEL_NAMES
    }
    seasonal_split = f"{blue} alone {seasonal[blue]}, {red} alone {seasonal[red]}"
    control = (
        f"the seasonal control rests on the same channel ({seasonal_split})"
        if passed and carrier and _carrier(study.seasonal_channels) == carrier
        else f"for the seasonal control, {seasonal_split}"
    )
    low, high = OBSERVED_CONTRAST_RANGE
    smoke = (
        "HRRR's smoke field was exactly zero in the analyses"
        if not any(value > 0.0 for value in _smoke_values(study))
        else "HRRR's smoke was recorded and not rendered"
    )
    first = (
        f"**Read this {word} in renderer-v0 units.** D(d, s) is computed on the "
        f"single-scattering renderer's contrast, not on camera pixels, and {channels}; {control}. "
        f"At a sun elevation of 0 the rendered lowest-band {blue} contrast is "
        f"{_horizon_contrast(study)[1]:+.2f}, where the four usable frames of the one camera "
        f"evening so far (overcast) read between {low:+.2f} and {high:+.2f}, so "
        f"{stats.THRESHOLD} in these units is not {stats.THRESHOLD} on a webcam. All aerosol "
        f"variation comes from GEFS-Aerosols' modelled optical depth; {smoke}."
    )
    if passed:
        last = (
            f"This PASS says the renderer's output moves from day to day by more than "
            f"{stats.THRESHOLD} in its own units. It does not show that a camera can see that "
            "movement (G1, issue #31, measures that), and it does not yet support a claim that "
            "the forecast can beat climatology or persistence; a multiple-scattering renderer "
            "(issue #32) re-runs this gate."
        )
    else:
        last = (
            f"This FAIL says the renderer's output moves from day to day by less than "
            f"{stats.THRESHOLD} in its own units on too many dates. It does not show that a "
            "camera cannot see aerosol-driven movement (G1, issue #31, measures that), and it "
            "does not show that the forecast cannot beat climatology or persistence; a "
            "multiple-scattering renderer (issue #32) re-runs this gate."
        )
    return f"{first} {last}"


def _criterion_section() -> str:
    counted = ", ".join(stats.COUNTED_BANDS)
    lines = [
        "## The pre-registered criterion",
        "",
        f"Committed in `{_CRITERION_FILE}` before any code that produces a result, and quoted "
        "from there:",
        "",
        f"> {stats.CRITERION_SENTENCE}",
        "",
        f"Counted bands: {counted} (`lastlight.geometry.ELEVATION_BANDS`: +6 to 0, 0 to -2 and "
        "-2 to -5 degrees of sun elevation). The band `late` (-5 to -8) is reported and not "
        "counted, because tier A scores the sun from +6 to -6 by default: city light "
        "contaminates the camera's low sky below about -6. The definitions the criterion leaves "
        'open (which sweep elevations are inside a band, the denominator of "25 percent of '
        'dates", the row to elevation mapping, the channel floor) are fixed in the same file.',
        "",
    ]
    return "\n".join(lines)


def _inputs_section(study: Study) -> str:
    # One line per bullet: Markdown needs no wrapping, and the tests look for whole phrases.
    smoke = _smoke_sentence(study)
    hrrr = (
        "- **HRRR (smoke and water vapour).** NOAA's HRRR 3 km surface files in the public bucket "
        "`noaa-hrrr-bdp-pds` (no account). Run-selection rule: the analysis (forecast hour 0) of "
        "the whole UTC hour nearest the instant the sun reaches geometric elevation 0 at the "
        "public reference point on the evening of the date, which is the past-evening path of "
        "`lastlight.inputs.hrrr.plan_run`. Three fields are downloaded by byte range: `AOTK` "
        "(aerosol optical thickness, smoke only), `PWAT` (precipitable water, recorded and not "
        "rendered) and `LCDC` (low cloud cover, which the reduction requires; recorded and not "
        "rendered). They are sampled along the great circle toward the azimuth at which the sun "
        "reaches the horizon, every 25 km out to 400 km, nearest cell. " + smoke
    )
    aerosol = (
        "- **Aerosol, total and dust.** NOAA's GEFS-Aerosols (the GEFS control run with the GOCART "
        "aerosol model), 0.25 degree 2-D aerosol product in the public bucket `noaa-gefs-pds` (no "
        "account): aerosol optical thickness in the 545 to 565 nm band (550 nm) for total aerosol "
        "and for dust, nearest cell to the reference point. Run-selection rule: the analysis "
        "(forecast hour 0) of the latest six-hourly cycle at or before the same instant. The "
        "archive's total is the aerosol optical depth that is rendered. It replaces HRRR's smoke "
        "plus the fixed background of 0.06, because the total already carries the smoke and "
        "adding HRRR's would count it twice; HRRR's smoke mean is recorded in `results.csv` and "
        "not added. The dust optical depth is recorded beside it."
    )
    ozone = (
        "- **Ozone.** The monthly climatology for 30 to 35 N from NASA's SBUV record, 2005 to 2024 "
        "(`lastlight.atmosphere.OZONE_DU_BY_MONTH`), looked up by the date's month."
    )
    analyses = (
        "- **Analyses, not forecasts.** Both archives are read at forecast hour 0, so the spread "
        "measured is the spread of the real atmosphere as the models see it, without forecast "
        "error mixed in."
    )
    model = (
        "- **The aerosol input is a model.** It is a modelled optical depth at one 25 km grid "
        "cell, not a measurement (the 0.25 degree cell nearest the public reference point), so "
        "the verdict is about a model's day-to-day variation. Whether the hour-0 state of "
        "GEFS-Aerosols assimilates aerosol observations was not verified."
    )
    geos = (
        "- **Tried first and not usable.** NASA GEOS-FP through the NCCS OPeNDAP server. Its "
        "assimilation collection of primary aerosol diagnostics (`tavg3_2d_aer_Nx`; the "
        "one-hourly collection named in the plan is not on that server) starts on 27 February "
        "2026 on the server's own time axis, checked on 2026-10-11, so it covers about seven and "
        "a half of the twelve months; the file-based GEOS-FP archive would mean downloading whole "
        "global files for every date. GEFS-Aerosols covers the whole period with one method, "
        "which a split between two aerosol models would not."
    )
    fixed = (
        "- **Held fixed by the renderer, so not part of the spread.** Surface pressure, the "
        "aerosol's Angstrom exponent (1.3), its scale height, asymmetry parameter and "
        "single-scattering albedo, and cloud (the curve is the clear-sky curve; cloud is "
        "recorded and not rendered). Real dust has a flatter spectrum and different absorption "
        "than the fixed aerosol, so a dusty day is rendered as more of the same aerosol, not as "
        "dust. These omissions bias the spread low."
    )
    attribution = (
        "- **Attribution.** GEFS-Aerosols and HRRR data are from NOAA, through the AWS Open Data "
        "registry (`noaa-gefs-pds` and `noaa-hrrr-bdp-pds`). The numbers here are derived "
        "values, not NOAA data products, and no endorsement by NOAA is implied."
    )
    bullets = [hrrr, aerosol, ozone, analyses, model, geos, fixed, attribution]
    return "## Inputs actually used\n\n" + "\n".join(bullets) + "\n"


def _smoke_sentence(study: Study) -> str:
    smoke = _smoke_values(study)
    nonzero = sum(1 for value in smoke if value > 0.0)
    if nonzero == 0:
        return (
            "HRRR's `AOTK` was exactly zero at every point of the sightline on all "
            f"{len(smoke)} dates. A check of the whole grid on two dates (2025-09-10 and "
            "2026-08-20, run on 2026-10-11) found the field zero everywhere at forecast hour 0 "
            "and non-zero at hours 6 and 12: the smoke model does not populate its analysis. "
            "HRRR therefore carried no aerosol information here, and all aerosol variation in "
            "this study comes from GEFS-Aerosols (the daily forecast reads forecast hours, which "
            "do carry smoke)."
        )
    return (
        f"HRRR's `AOTK` was non-zero on {nonzero} of {len(smoke)} dates "
        f"(largest transect mean {max(smoke):.3f})."
    )


def _aerosol_months_section(study: Study) -> str:
    months: dict[str, list] = {}
    for record in study.used:
        months.setdefault(record.date.strftime("%Y-%m"), []).append(record.atmosphere)
    rows = []
    for month, atmospheres in months.items():
        total = np.array([a.aod_total for a in atmospheres])
        dust = np.array([a.aod_dust for a in atmospheres])
        rows.append(
            [
                month,
                len(atmospheres),
                f"{np.median(total):.3f}",
                f"{total.max():.3f}",
                f"{np.median(dust):.3f}",
                f"{dust.max():.3f}",
            ]
        )
    table = _table(
        ["Month", "Dates", "Total, median", "Total, max", "Dust, median", "Dust, max"], rows
    )
    return (
        "## Aerosol input by month\n\n"
        "Aerosol optical depth at 550 nm from GEFS-Aerosols at the reference point on the dates "
        "used, the only aerosol input that varies:\n\n"
        f"{table}\n"
    )


def _channels_section(study: Study) -> str:
    rows = []
    for name, outcome in study.channels.items():
        medians = [f"{np.median(outcome.distances[:, i]):.4f}" for i in range(len(study.bands))]
        verdict = outcome.verdict
        rows.append(
            [name, *medians, f"{verdict.meeting} of {verdict.dates}", _percent(verdict.fraction)]
        )
    both = study.outcomes[NOMINAL]
    rows.append(
        [
            "both (the criterion)",
            *(f"{np.median(both.distances[:, i]):.4f}" for i in range(len(study.bands))),
            f"{both.verdict.meeting} of {both.verdict.dates}",
            _percent(both.verdict.fraction),
        ]
    )
    table = _table(
        [
            "Channel",
            *(f"median D {band}" for band in study.bands),
            "Dates meeting the rule",
            "Share",
        ],
        rows,
    )
    lines = [
        "## Channel decomposition (post hoc)",
        "",
        "This section is post hoc. It was added after the verdict was first read, and it is not "
        "part of the pre-registered criterion: D(d, s) is the mean over both channels, so this "
        "splits it. The same rule is applied to "
        "log(R/G) alone and to log(B/G) alone, with the median taken per channel. It shows how "
        "much of D each channel carries, and it does not change the verdict.",
        "",
        table,
        "",
    ]
    return "\n".join(lines)


def _view_section(views: Mapping[str, SkyView]) -> str:
    nominal = views[NOMINAL]
    contrast = load_registry()[nominal.camera_id].contrast
    assert contrast is not None
    spans = [*contrast.bands_rows, contrast.reference_rows]
    names = [*(f"band {number}" for number in range(1, len(spans))), "reference"]
    rows = []
    for position, (name, (low, high)) in enumerate(zip(names, spans, strict=True)):
        elevations = [f"{views[scale].elevations_deg[position]:.2f}" for scale in RENDER_SCALES]
        rows.append([f"{name}, {low}-{high}", *elevations])
    table = _table(
        [
            "Rows above the skyline",
            "Elevation, nominal scale",
            "Elevation, low-scale end",
            "Elevation, high-scale end",
        ],
        rows,
    )
    scales = ", ".join(f"{name} {views[name].scale_px_per_deg:.2f}" for name in RENDER_SCALES)
    skyline = f"{nominal.skyline_elevation_deg:.2f}"
    lines = [
        "## WHTM1's view",
        "",
        "Bands are pixel rows above the camera's frozen skyline, as `lastlight observe` measures "
        "them, taken over the sky columns: the columns of the frame in which every band and the "
        "reference lie inside the fit's sky mask. A band is mapped to one view elevation by its "
        f"centre row over the pixel scale, above the skyline's median elevation of {skyline} "
        f"degrees. The camera looks at the fit's centre azimuth of {nominal.azimuth_deg:.1f} "
        "degrees; each band is rendered at the relative azimuth to the sun at every sweep "
        "elevation of every date (computed internally, not published).",
        "",
        table,
        "",
        f"Pixel scale in px per degree: {scales}. The fit is provisional: the sun has not "
        "entered WHTM1's frame, so nothing independent has confirmed the scale, and the nominal "
        "value is the best of a family of nearly equal fits. That is why the statistic is "
        "recomputed at both ends of the range below.",
        "",
    ]
    return "\n".join(lines)


def _dates_section(study: Study) -> str:
    start, end = study.period
    head = (
        f"## Dates\n\nEvenings {start.isoformat()} to {end.isoformat()}: "
        f"{len(study.attempted)} attempted, {len(study.used)} used, "
        f"{len(study.dropped)} dropped.\n"
    )
    if not study.dropped:
        return head + "\nNo date was dropped.\n"
    counts: dict[str, int] = {}
    for record in study.dropped:
        assert record.drop_reason is not None
        counts[record.drop_reason] = counts.get(record.drop_reason, 0) + 1
    summary = "\n".join(f"- `{reason}`: {count}" for reason, count in sorted(counts.items()))
    rows = []
    for record in study.dropped:
        assert record.drop_reason is not None
        rows.append(
            [record.date.isoformat(), f"`{record.drop_reason}`", DROP_REASONS[record.drop_reason]]
        )
    table = _table(["Date", "Reason", "What it means"], rows)
    return f"{head}\nDropped dates by reason:\n\n{summary}\n\nEvery dropped date:\n\n{table}\n"


def _distribution_section(study: Study) -> str:
    outcome = study.outcomes[NOMINAL]
    rows = []
    for index, name in enumerate(study.bands):
        column = outcome.distances[:, index]
        reaching = int((column >= stats.THRESHOLD).sum())
        low, p10, median, p90, high = (
            float(v) for v in np.percentile(column, [0, 10, 50, 90, 100])
        )
        rows.append(
            [
                name,
                "yes" if name in study.counted_bands else "not counted",
                *(f"{v:.4f}" for v in (low, p10, median, p90, high)),
                f"{reaching} of {len(column)} ({_percent(reaching / len(column))})",
            ]
        )
    header = ["Band", "Counted", "min", "10th", "median", "90th", "max"]
    table = _table([*header, f"Dates with D >= {stats.THRESHOLD}"], rows)
    by_count = _table(
        ["Counted bands at the threshold", "Dates"],
        [
            [count, int((outcome.at_threshold == count).sum())]
            for count in range(len(study.counted_bands) + 1)
        ],
    )
    threshold = stats.THRESHOLD
    lines = [
        "## Distribution of D(d, s)",
        "",
        "D(d, s) is the mean absolute difference between a date's rendered contrast and the "
        "period median, over the sweep elevations inside the band, WHTM1's five contrast bands "
        f"and both channels. A value of {threshold} is the pre-registered guess at webcam "
        "repeatability; G1 measures the real one on the same bands.",
        "",
        table,
        "",
        f"How many of the counted bands reach {threshold} on each date; the rule needs "
        f"{stats.MIN_BANDS} or more:",
        "",
        by_count,
        "",
        "![Distribution of D(d, s) by band](d_distribution.png)",
        "",
        "![D(d, s) by date](d_by_date.png)",
        "",
        "![Rendered contrast against sun elevation](profile_spread.png)",
        "",
    ]
    return "\n".join(lines)


def _sensitivity_section(study: Study, views: Mapping[str, SkyView]) -> str:
    labels = {
        NOMINAL: "nominal (the verdict)",
        "low": "low end of the range",
        "high": "high end of the range",
    }
    rows = []
    for name in RENDER_SCALES:
        verdict = study.outcomes[name].verdict
        rows.append(
            [
                labels[name],
                f"{views[name].scale_px_per_deg:.2f}",
                f"{verdict.meeting} of {verdict.dates}",
                _percent(verdict.fraction),
                _word(verdict.passed),
            ]
        )
    table = _table(
        ["Pixel scale", "px per degree", "Dates meeting the rule", "Share", "Would be"], rows
    )
    lines = [
        "## Pixel-scale sensitivity",
        "",
        "The whole statistic is recomputed, median included, with the rows mapped to elevation "
        "at each end of the fit's `uncertainty.px_per_deg` range, the skyline kept where the fit "
        "put it. This is reported for information and does not change the verdict.",
        "",
        table,
        "",
    ]
    return "\n".join(lines)


def _seasonal_section(study: Study) -> str:
    seasonal = study.seasonal.verdict
    by_channel = " and ".join(
        f"{name} alone, {outcome.verdict.meeting} of {outcome.verdict.dates} dates "
        f"({_percent(outcome.verdict.fraction)})"
        for name, outcome in study.seasonal_channels.items()
    )
    lines = [
        "## Seasonal control",
        "",
        "The sun's azimuth and the ozone column change with the season, and the criterion takes "
        "its median over the whole period, so a calendar effect counts as variation between "
        "dates. As a control only (it was registered with the criterion, and it is not the "
        "criterion), the same rule is run with the median taken over the dates within "
        f"{stats.HALF_WINDOW_DAYS} days of each date. {seasonal.meeting} of the "
        f"{seasonal.dates} dates ({_percent(seasonal.fraction)}) meet the rule that way; against "
        f"the same {_percent(stats.MIN_FRACTION)} it would be a {_word(seasonal.passed)}. "
        f"By channel the same control gives {by_channel}. "
        "This is the closer measure of what a forecast would have to beat, in renderer units.",
        "",
    ]
    return "\n".join(lines)


def _tier_b_section(study: Study) -> str:
    rows = []
    for index, name in enumerate(stats.DELTA_E_BANDS):
        column = study.delta_e[:, index]
        rows.append(
            [
                name,
                f"{np.median(column):.2f}",
                f"{np.percentile(column, 90):.2f}",
                f"{column.max():.2f}",
            ]
        )
    table = _table(["Band", "median", "90th percentile", "max"], rows)
    lines = [
        "## Tier B: delta-E 2000 of the absolute curve",
        "",
        "Absolute colour is not scored in tier A. For tier B, delta-E 2000 of each date's "
        "absolute curve (the forecast's own curve: the solar vertical, seven view elevations "
        "from 2 to 45 degrees, each sun elevation exposed to L* 70) against the period's median "
        "curve, averaged over the sweep elevations inside each band and the views. There is no "
        "pass criterion.",
        "",
        table,
        "",
    ]
    return "\n".join(lines)


def _extremes_section(study: Study) -> str:
    rows = []
    for rank, extreme in enumerate(study.extremes, start=1):
        rows.append(
            [
                rank,
                extreme.date.isoformat(),
                f"{extreme.mean_d:.4f}",
                *(f"{value:.4f}" for value in extreme.d_by_band),
                f"{extreme.aod_total:.3f}",
                f"{extreme.aod_dust:.3f}",
                f"{extreme.aod_smoke:.3f}",
            ]
        )
    table = _table(
        [
            "Rank",
            "Date",
            "Mean D, counted bands",
            "D golden",
            "D early",
            "D mid",
            "D late",
            "Aerosol optical depth, total",
            "dust",
            "HRRR smoke",
        ],
        rows,
    )
    return (
        "## The five most extreme dates\n\n"
        "The dates with the largest mean of D(d, s) over the counted bands:\n\n"
        f"{table}\n"
    )


def _floor_section(study: Study) -> str:
    nominal, low, high = (study.floor_share[name] for name in RENDER_SCALES)
    lines = [
        "## The channel floor",
        "",
        "Single scattering near the horizon is redder than the sRGB gamut, so a rendered channel "
        "can be negative, and a logarithm needs a positive number. A channel is raised to the "
        "widest ratio an 8-bit frame can show under `lastlight.observe`'s own pixel rules (code "
        "6 against code 249, linearised; about 0.002 of the pixel's largest channel) before the "
        f"logarithm. The floor was used for {_percent(nominal)} of the rendered channels at the "
        f"nominal scale, {_percent(low)} at the low scale and {_percent(high)} at the high "
        "scale. Where it is used the logarithm is of the floor, not of the sky: those values "
        "say only that the channel is at or below the floor.",
        "",
    ]
    return "\n".join(lines)


def _reading_section(study: Study) -> str:
    columns = [study.bands.index(name) for name in study.counted_bands]
    red = float(np.mean(study.channels[CHANNEL_NAMES[0]].distances[:, columns]))
    blue = float(np.mean(study.channels[CHANNEL_NAMES[1]].distances[:, columns]))
    blue_share = 100.0 * blue / (red + blue) if red + blue > 0.0 else 0.0
    at_horizon = _horizon_contrast(study)
    observed_low, observed_high = OBSERVED_CONTRAST_RANGE
    bullets = [
        "- The verdict is about the **model's** curve: it asks whether the real atmosphere's "
        "day-to-day variation, as the archives carry it, moves the rendered contrast by more "
        "than 0.05. It says nothing about whether the model matches a camera; that is the "
        "scoring milestone, and G1 measures how much of the 0.05 is camera repeatability.",
        f"- {CHANNEL_NAMES[1]} carries {blue_share:.0f} percent of the summed D(d, s) over the "
        "counted bands (see the channel decomposition). In linear sRGB the blue channel of these "
        "reddened low skies is a small difference of large terms, nearly zero, and a ratio "
        "against a channel that is nearly zero is very sensitive to small changes in the "
        "light: the choice of a webcam proxy matters to this statistic.",
        f"- At a sun elevation of 0 the median, over the dates used, of the lowest band's contrast "
        f"is {at_horizon[0]:+.2f} in log(R/G) and {at_horizon[1]:+.2f} in log(B/G). The four "
        "usable frames of the first observed evening, an overcast one, read between "
        f"{observed_low:+.2f} and {observed_high:+.2f} for sun elevations from +5 to -4 "
        "(`docs/log/2026-10-11-observed-contrasts.md`). "
        "A single-scattering model leaves out the multiple scattering that fills the blue of a "
        "twilight sky, which is likely to exaggerate the rendered blue contrast; matching the "
        "camera is for the scoring milestone, not for G0.",
        "- The geometry is provisional. The sensitivity table shows how much the pixel scale "
        "matters.",
        "- The renderer is single scattering with fixed aerosol optics and no cloud, and the "
        "aerosol column comes from a 25 km model; real variation in aerosol type, humidity and "
        "cloud is not here.",
        "- A mean over a year includes the season. The seasonal control says how much of the "
        "spread is the calendar.",
        "- Dropped dates are listed above with their reasons; none was filled.",
    ]
    return "## Reading the result\n\n" + "\n".join(bullets) + "\n"


def render_markdown(study: Study, views: Mapping[str, SkyView]) -> str:
    """The report as one Markdown document."""
    start, end = study.period
    title = (
        "# G0: does the rendered curve vary enough to measure?\n\n"
        f"Spread study for issue #10 over the evenings {start.isoformat()} to "
        f"{end.isoformat()}, rendered from archived inputs for WHTM1's frozen geometry.\n"
    )
    parts = [
        title,
        _verdict_section(study),
        _criterion_section(),
        _inputs_section(study),
        _aerosol_months_section(study),
        _view_section(views),
        _dates_section(study),
        _distribution_section(study),
        _sensitivity_section(study, views),
        _seasonal_section(study),
        _channels_section(study),
        _tier_b_section(study),
        _extremes_section(study),
        _floor_section(study),
        _reading_section(study),
    ]
    return "\n".join(parts)


def write_outputs(study: Study, out_dir: Path | str, *, views: Mapping[str, SkyView]) -> list[Path]:
    """Write the report, the results file and the three figures into `out_dir`."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in (
        ("report.md", render_markdown(study, views)),
        ("results.csv", results_csv(study)),
    ):
        path = directory / name
        path.write_text(text, encoding="utf-8")
        written.append(path)
    for name, draw in FIGURES.items():
        path = directory / name
        path.write_bytes(draw(study))
        written.append(path)
    return written
