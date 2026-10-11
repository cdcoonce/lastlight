"""The one test that talks to the real HRRR bucket (issue #6). Deselected by default.

Run it on purpose with `uv run pytest -m network -s tests/test_hrrr_network.py`; it prints a
short summary of what it fetched. It reads one recent `.idx` and one field's byte range.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from lastlight.inputs.hrrr import (
    DEFAULT_FIELDS,
    HrrrError,
    fetch_fields,
    fetch_idx,
    field_spec,
    latest_run,
    sample_transect,
    select_ranges,
)
from lastlight.inputs.http import UrllibTransport

REF_LAT = 33.45
REF_LON = -112.07


@pytest.mark.network
def test_a_real_recent_idx_and_one_field_from_the_public_bucket(tmp_path: Path) -> None:
    transport = UrllibTransport()
    try:
        cycle_date, cycle_hour = latest_run(dt.datetime.now(dt.UTC), transport=transport)
    except HrrrError as error:  # pragma: no cover - only if the bucket is truly empty
        pytest.fail(f"no recent HRRR run found: {error}")

    entries = fetch_idx(cycle_date, cycle_hour, 0, transport=transport)
    ranges = select_ranges(entries, DEFAULT_FIELDS)  # all nine default fields are in a real idx
    total = field_spec("TCDC")
    fields = fetch_fields(cycle_date, cycle_hour, 0, [total], tmp_path, transport=transport)
    (sample,) = sample_transect(fields, [(REF_LAT, REF_LON)])

    cached = list(tmp_path.iterdir())
    wanted = ranges["TCDC"]
    assert wanted.end is not None
    print(
        f"\nHRRR network check: cycle date {cycle_date}, cycle hour {cycle_hour}, forecast hour 0\n"
        f"  idx messages: {len(entries)}; default fields found in idx: {len(ranges)}\n"
        f"  downloaded: 1 byte range, {cached[0].stat().st_size} bytes "
        f"(of the whole file's roughly 140 MB)\n"
        f"  total cloud cover at the reference point: {sample.value} {sample.units} "
        f"(cell {sample.cell}, {sample.cell_distance_km:.2f} km from the point, {sample.method})"
    )
    assert len(entries) > 100
    assert len(cached) == 1
    assert cached[0].stat().st_size == wanted.end - wanted.start + 1
    assert cached[0].stat().st_size < 5 * 1024 * 1024
    assert sample.value is not None
    assert 0.0 <= sample.value <= 100.0
    assert sample.units == "%"
