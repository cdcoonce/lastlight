"""The check's public outputs: `report.md`, `evenings.csv` and `frames.csv`.

Privacy (`docs/design.md` section 3): dates, sun elevations (two decimals) and derived numbers
only. No clock time, no UTC hour, no HRRR or GEFS cycle hour, no frame hash or file name, no local
path, no coordinate. The camera's azimuth relative to the sun, which the renderer needs for every
frame, is computed internally and never written. The same inputs give the same bytes: evenings
are sorted by date and every number has a fixed format (`inf` for an infinite ratio).
"""

from __future__ import annotations

import csv
import io
import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from lastlight.renderer_check import rules
from lastlight.renderer_check.evaluate import (
    COUNTED,
    DROPPED,
    NOT_CONSIDERED,
    NOT_COUNTED,
    PENDING,
    REASON_BEFORE_FIRST,
    REASON_TOO_FEW_FRAMES,
    Evaluation,
    EveningResult,
)
from lastlight.renderer_check.guards import REGISTERED_TABLE_SHA256
from lastlight.renderer_check.store import INPUT_DROP_REASONS

EVENINGS_COLUMNS: Final = (
    "date",
    "status",
    "reason",
    "frames_used",
    "cloud_mean_pct",
    "aod_550",
    "slot",
    "r_e_v1_bg",
    "r_e_v0_bg",
    "r_e_v1_bg_scale_low",
    "r_e_v1_bg_scale_high",
    "r_e_v1_rg",
    "r_e_v0_rg",
)
FRAMES_COLUMNS: Final = (
    "date",
    "frame",
    "sun_elevation_deg",
    "observed_log_bg",
    "rendered_log_bg_v0",
    "rendered_log_bg_v1",
    "observed_log_rg",
    "rendered_log_rg_v0",
    "rendered_log_rg_v1",
)

_CRITERION: Final = "criterion.md"


def _ratio(value: float | None, missing: str) -> str:
    if value is None:
        return missing
    return "inf" if math.isinf(value) else f"{value:.4f}"


def _number(value: float | None, digits: int) -> str:
    return "" if value is None else f"{value:.{digits}f}"


# --- the two files of numbers ------------------------------------------------------------------


def _evening_row(result: EveningResult) -> list[str]:
    ratios = result.ratios
    return [
        result.date.isoformat(),
        result.status,
        result.reason,
        str(result.frames_used),
        _number(result.cloud_mean_pct, 4),
        _number(result.aod_550, 4),
        "" if result.slot is None else str(result.slot),
        _ratio(ratios and ratios.v1_bg, ""),
        _ratio(ratios and ratios.v0_bg, ""),
        _ratio(ratios and ratios.v1_bg_low, ""),
        _ratio(ratios and ratios.v1_bg_high, ""),
        _ratio(ratios and ratios.v1_rg, ""),
        _ratio(ratios and ratios.v0_rg, ""),
    ]


def evenings_csv(evaluation: Evaluation) -> str:
    """One row per observed evening, in date order."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(EVENINGS_COLUMNS)
    for result in evaluation.results:
        writer.writerow(_evening_row(result))
    return buffer.getvalue()


def frames_csv(evaluation: Evaluation) -> str:
    """One row per used frame of every evening considered: the sun's elevation, the observed
    contrasts and what each renderer says (blank where the evening was not rendered)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(FRAMES_COLUMNS)
    for result in evaluation.results:
        if result.status == NOT_CONSIDERED:
            continue
        for number, pair in enumerate(result.pairs, start=1):
            writer.writerow(
                [
                    result.date.isoformat(),
                    number,
                    f"{pair.sun_elevation_deg:.2f}",
                    _number(pair.observed_bg, 6),
                    _number(pair.v0_bg, 6),
                    _number(pair.v1_bg, 6),
                    _number(pair.observed_rg, 6),
                    _number(pair.v0_rg, 6),
                    _number(pair.v1_rg, 6),
                ]
            )
    return buffer.getvalue()


# --- report.md ---------------------------------------------------------------------------------


def _frames_phrase(count: int) -> str:
    return f"{count} used frame{'' if count == 1 else 's'}"


def _status(result: EveningResult) -> str:
    if result.status == NOT_CONSIDERED:
        if result.reason == REASON_BEFORE_FIRST:
            return f"not considered: before {rules.FIRST_EVENING.isoformat()}"
        assert result.reason == REASON_TOO_FEW_FRAMES
        return f"not considered: {_frames_phrase(result.frames_used)}"
    if result.status == PENDING:
        return "pending: no input record yet"
    if result.status == DROPPED:
        return f"dropped, {result.reason}"
    if result.status == NOT_COUNTED:
        assert result.cloud_mean_pct is not None
        return f"not counted, cloud {result.cloud_mean_pct:.2f}%"
    assert result.status == COUNTED
    if result.slot is not None:
        return f"counted (first 5, number {result.slot})"
    if result.held:
        return "counted, waiting for an earlier evening's inputs"
    return "counted, after the first 5 (reported, not in the verdict)"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _evenings_table(evaluation: Evaluation) -> str:
    rows = []
    for result in evaluation.results:
        ratios = result.ratios
        rows.append(
            [
                result.date.isoformat(),
                _status(result),
                str(result.frames_used),
                _number(result.cloud_mean_pct, 2) or "n/a",
                _number(result.aod_550, 4) or "n/a",
                _ratio(ratios and ratios.v1_bg, "n/a"),
                _ratio(ratios and ratios.v0_bg, "n/a"),
                _ratio(ratios and ratios.v1_bg_low, "n/a"),
                _ratio(ratios and ratios.v1_bg_high, "n/a"),
                _ratio(ratios and ratios.v0_rg, "n/a"),
                _ratio(ratios and ratios.v1_rg, "n/a"),
            ]
        )
    return _table(
        [
            "Date",
            "Status",
            "Frames used",
            "Cloud mean (%)",
            "aod_550",
            "r_e v1",
            "r_e v0",
            "r_e v1, scale low",
            "r_e v1, scale high",
            "r_e v0, log(R/G)",
            "r_e v1, log(R/G)",
        ],
        rows,
    )


def _verdict_section(evaluation: Evaluation) -> str:
    slots = evaluation.slots
    dates = ", ".join(slot.date.isoformat() for slot in slots) or "none yet"
    if evaluation.decided:
        median = _ratio(
            statistics.median(slot.ratios.v1_bg for slot in slots if slot.ratios is not None),
            "n/a",
        )
        head = (
            f"The first {rules.COUNTED_NEEDED} counted evenings are {dates}. The median of their "
            f"v1 r_e is {median}; the window is {rules.RATIO_LOW} to {rules.RATIO_HIGH}, bounds "
            "included. The verdict is computed once, on these evenings: later counted evenings "
            "are reported beside it and never change it."
        )
    else:
        so_far = (
            f"{len(slots)} of the {rules.COUNTED_NEEDED} counted evenings the verdict needs so "
            f"far: {dates}."
            if slots
            else f"No evening is counted yet, and the verdict needs {rules.COUNTED_NEEDED} (an "
            "evening is counted when its mean cloud cover along the sightline is at most "
            f"{rules.CLOUD_LIMIT_PCT:g} percent)."
        )
        head = (
            f"{so_far} With fewer than {rules.COUNTED_NEEDED} the criterion's verdict is not yet "
            "decidable and v0 keeps painting."
        )
    pending = evaluation.pending
    if pending:
        waiting = ", ".join(p.date.isoformat() for p in pending)
        head += (
            f" {len(pending)} considered evening{'s have' if len(pending) != 1 else ' has'} no "
            f"input record yet ({waiting}); a counted evening dated after the earliest of them "
            "holds no slot until it is collected."
        )
    return f"## Verdict\n\n{head}\n"


def _renderer_section() -> str:
    return (
        "## The renderer under test\n\n"
        "Engine `v1` with the table `src/lastlight/render/data/bruneton_ms_xyz.bin`, whose "
        f"SHA-256 is `{REGISTERED_TABLE_SHA256}` (the criterion's amendment of 2026-10-12). The "
        "check hashes the bytes of the table file the renderer loads and compares them with that "
        "constant, and compares three probes of the engine with the values registered with it; "
        "it refuses to give a verdict, and writes nothing, if either differs. A report exists "
        "only because both matched.\n"
    )


def _reading_section() -> str:
    bullets = [
        "- **r_e** for an evening is the median over its used frames of the absolute rendered "
        "log(B/G) contrast of the lowest band (minus the reference's) divided by the median over "
        "the same frames of the absolute observed one: a ratio of two medians, `inf` when the "
        "observed median is 0. The verdict column is `r_e v1`.",
        "- **Reported beside the verdict, not part of it:** `r_e v0` (the same ratio for the "
        "single-scattering renderer), `r_e v1` at both ends of the fit's pixel-scale range "
        "(`scale low` and `scale high`), and the ratios for log(R/G).",
        "- **Counted** means the mean total cloud cover along the sightline, from HRRR's "
        "analysis, is at most 10 percent. Evenings not counted are shown with their cloud and "
        "their ratios, and do not enter the verdict. Dropped evenings have no ratios.",
        "- The per-frame pairs (sun elevation, observed, rendered with v0 and v1) are in "
        "`frames.csv`; the per-evening numbers in `evenings.csv`.",
    ]
    return "## Reading the table\n\n" + "\n".join(bullets) + "\n"


def _drop_reasons_section(evaluation: Evaluation) -> str:
    reasons = sorted({r.reason for r in evaluation.results if r.status == DROPPED})
    if not reasons:
        return ""
    lines = [f"- `{reason}`: {INPUT_DROP_REASONS[reason]}" for reason in reasons]
    return "## Dropped evenings\n\n" + "\n".join(lines) + "\n"


def _cannot_say_section() -> str:
    return (
        "## What this check cannot say\n\n"
        f"See [What this check cannot say]({_CRITERION}#what-this-check-cannot-say) in the "
        "criterion. It is not repeated here.\n"
    )


def render_report(evaluation: Evaluation) -> str:
    """The report as one Markdown document."""
    title = (
        "# Renderer v1: the camera magnitude check\n\n"
        f"Verdict: {evaluation.verdict}\n\n"
        f"The pre-registered check of [`{_CRITERION}`]({_CRITERION}), run by "
        "`lastlight renderer-check` over WHTM1's observed evenings. It asks one coarse question: "
        "is renderer v1's within-frame contrast within a factor of two of what the camera sees? "
        "A pass says the magnitude is plausible. It says nothing about skill.\n"
    )
    parts = [
        title,
        _verdict_section(evaluation),
        _renderer_section(),
        "## Evenings\n\n" + _evenings_table(evaluation) + "\n",
        _reading_section(),
        _drop_reasons_section(evaluation),
        _cannot_say_section(),
    ]
    return "\n".join(part for part in parts if part)


def write_outputs(evaluation: Evaluation, out_dir: Path | str) -> list[Path]:
    """Write the report and the two files of numbers into `out_dir` (made if needed)."""
    directory = Path(out_dir)
    texts = (
        ("report.md", render_report(evaluation)),
        ("evenings.csv", evenings_csv(evaluation)),
        ("frames.csv", frames_csv(evaluation)),
    )
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in texts:
        path = directory / name
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written
