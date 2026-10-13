"""The stages of `lastlight renderer-check`.

    fetch   collect each considered evening's inputs into the check's own cache (network, polite)
    report  read the cache and the observed files, and write `report.md`, `evenings.csv` and
            `frames.csv` (no network at all)
    all     both (the default)

Exit codes: 0 done; 1 done, but a considered evening has no input record yet (a network fault, an
HRRR analysis not published yet, or a run still missing from an archive within two days of the
evening; the report is still written and lists it); 2 usage or input error; 3 refusal (the
registered renderer is not the one in this checkout): nothing is written to `--out`, and in the
`all` stage nothing is fetched either.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from typing import Final

from lastlight.inputs.http import Transport

PROG: Final = "lastlight renderer-check"
STAGES: Final = ("fetch", "report", "all")
STORE_DIR: Final = "renderer-v1"
"""Under `--cache-dir`: the check's own records, and nowhere else."""


def _error(message: str) -> int:
    print(f"{PROG}: error: {message}", file=sys.stderr)
    return 2


def run_check(
    *,
    observed_dir: Path,
    out_dir: Path,
    cache_dir: Path,
    stage: str,
    transport: Transport,
    now_utc: dt.datetime,
) -> int:
    """Run the stage(s) and return the process exit code."""
    # Deferred: these import the GRIB library, which loads a native one.
    from lastlight.g0.collect import build_views
    from lastlight.renderer_check import guards, rules
    from lastlight.renderer_check.evaluate import evaluate
    from lastlight.renderer_check.fetch import InputFetcher, fetch_range, needs_fetch
    from lastlight.renderer_check.observed import ObservedError, read_observed_dir
    from lastlight.renderer_check.report import write_outputs
    from lastlight.renderer_check.store import InputRecordError, InputStore

    if stage not in STAGES:
        return _error(f"stage must be one of {STAGES}, got {stage!r}")
    try:
        evenings = read_observed_dir(observed_dir)
    except ObservedError as error:
        return _error(str(error))

    writes_report = stage in ("report", "all")
    if writes_report:
        try:
            guards.check_registered()
        except guards.Refusal as refusal:
            print(f"{PROG}: refused: {refusal}", file=sys.stderr)
            return 3

    considered = [
        evening.date
        for evening in evenings
        if evening.date >= rules.FIRST_EVENING and len(evening.frames) >= rules.MIN_USED_FRAMES
    ]
    store = InputStore(cache_dir / STORE_DIR)
    code = 0
    try:
        if stage in ("fetch", "all"):
            todo = [day for day in considered if needs_fetch(store, day, now_utc)]
            if todo:
                fetcher = InputFetcher(transport, cache_dir, now_utc)
                summary = fetch_range(considered, store, fetcher, now_utc=now_utc, progress=print)
                print(
                    f"{summary.recorded} collected, {summary.dropped} dropped, "
                    f"{summary.skipped} skipped, {len(summary.pending)} pending, "
                    f"{len(summary.faults)} faults"
                )
            missing = [day for day in considered if store.load(day) is None]
            if missing and not writes_report:  # the report stage says it for itself
                print(
                    f"{PROG}: no input record yet for "
                    f"{', '.join(day.isoformat() for day in missing)}: a network fault or an HRRR "
                    "analysis not published yet; run again later to collect it",
                    file=sys.stderr,
                )
                code = 1

        if writes_report:
            evaluation = evaluate(evenings, store, build_views())
            write_outputs(evaluation, out_dir)
            print(f"verdict: {evaluation.verdict}")
            print(f"wrote the report, evenings and frames to {out_dir}")
            if evaluation.pending:
                waiting = ", ".join(r.date.isoformat() for r in evaluation.pending)
                print(
                    f"{PROG}: no input record yet for {waiting}: a network fault or an HRRR "
                    "analysis not published yet; run again later to collect it",
                    file=sys.stderr,
                )
                code = 1
    except InputRecordError as error:
        return _error(str(error))
    return code
