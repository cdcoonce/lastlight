"""The per-date record the G0 run writes, and the store that makes the run resumable.

A year of archived inputs takes an hour of polite downloading, so each date's result is written
to the cache as soon as it exists, and a rerun (or the final, offline report pass) reads those
files instead of asking the archives again.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pytest
from g0_support import dropped_record, synthetic_record

from lastlight.g0.record import (
    DROP_REASONS,
    RENDER_SCALES,
    SCHEMA_VERSION,
    DateRecord,
    RecordError,
    RecordStore,
)

DAY = dt.date(2026, 3, 14)


def test_a_rendered_record_round_trips_through_json_exactly() -> None:
    record = synthetic_record(DAY, {"golden": 0.1234567891234, "mid": -0.07}, aod=0.0833333333)
    again = DateRecord.from_json(json.loads(json.dumps(record.to_json())))
    assert again.date == DAY and again.ok
    assert again.atmosphere == record.atmosphere
    for scale in RENDER_SCALES:
        np.testing.assert_array_equal(again.contrast[scale], record.contrast[scale])
    np.testing.assert_array_equal(again.curve_lab, record.curve_lab)
    assert again.floor_channels == record.floor_channels
    assert again.inputs == record.inputs


def test_a_dropped_record_round_trips_with_its_reason() -> None:
    record = dropped_record(DAY, "hrrr_run_missing", "the file is not in the archive")
    again = DateRecord.from_json(json.loads(json.dumps(record.to_json())))
    assert not again.ok
    assert (again.drop_reason, again.drop_detail) == ("hrrr_run_missing", record.drop_detail)
    assert again.atmosphere is None and again.curve_lab is None and again.contrast == {}


def test_every_drop_reason_has_a_sentence_for_the_report() -> None:
    assert {
        "hrrr_run_missing",
        "hrrr_field_missing",
        "hrrr_field_empty",
        "hrrr_unreadable",
        "aerosol_run_missing",
        "aerosol_unreadable",
        "not_collected",
    } <= set(DROP_REASONS)
    assert all(text.endswith(".") and "\n" not in text for text in DROP_REASONS.values())


def test_a_record_cannot_be_both_rendered_and_dropped() -> None:
    good = synthetic_record(DAY)
    with pytest.raises(RecordError, match="reason"):
        DateRecord(**{**good.__dict__, "drop_reason": "hrrr_run_missing"})
    with pytest.raises(RecordError, match="rendered"):
        DateRecord(
            **{
                **dropped_record(DAY, "hrrr_run_missing").__dict__,
                "curve_lab": np.zeros((29, 7, 3)),
            }
        )


def test_an_unknown_drop_reason_is_refused() -> None:
    with pytest.raises(RecordError, match="reason"):
        dropped_record(DAY, "tuesday")


def test_a_rendered_record_must_have_all_three_scales_with_the_right_shape() -> None:
    good = synthetic_record(DAY)
    with pytest.raises(RecordError, match="scales"):
        DateRecord(**{**good.__dict__, "contrast": {"nominal": good.contrast["nominal"]}})
    bad = {**good.contrast, "nominal": np.zeros((28, 5, 2))}
    with pytest.raises(RecordError, match="shape"):
        DateRecord(**{**good.__dict__, "contrast": bad})
    nan = {**good.contrast, "low": np.full((29, 5, 2), np.nan)}
    with pytest.raises(RecordError, match="finite"):
        DateRecord(**{**good.__dict__, "contrast": nan})


# --- the store ---------------------------------------------------------------------------------


def test_the_store_saves_one_file_per_date_and_loads_it_back(tmp_path) -> None:
    store = RecordStore(tmp_path / "records")
    assert store.load(DAY) is None
    record = synthetic_record(DAY, {"early": 0.2})
    store.save(record)
    assert store.path(DAY).name == "2026-03-14.json"
    assert store.load(DAY).contrast["nominal"].shape == (29, 5, 2)  # type: ignore[union-attr]
    assert store.dates() == {DAY}


def test_saving_again_replaces_the_record_and_leaves_no_partial_file(tmp_path) -> None:
    store = RecordStore(tmp_path)
    store.save(dropped_record(DAY, "hrrr_run_missing"))
    store.save(synthetic_record(DAY))
    loaded = store.load(DAY)
    assert loaded is not None and loaded.ok
    assert [p.name for p in tmp_path.iterdir()] == ["2026-03-14.json"]


def test_a_record_for_another_date_in_the_file_is_an_error(tmp_path) -> None:
    store = RecordStore(tmp_path)
    store.save(synthetic_record(DAY))
    other = dt.date(2026, 3, 15)
    store.path(DAY).rename(store.path(other))
    with pytest.raises(RecordError, match="2026-03-14"):
        store.load(other)


def test_a_corrupt_or_foreign_file_is_an_error_not_a_missing_record(tmp_path) -> None:
    store = RecordStore(tmp_path)
    store.path(DAY).write_text("{not json")
    with pytest.raises(RecordError, match="2026-03-14"):
        store.load(DAY)
    document = synthetic_record(DAY).to_json()
    document["schema_version"] = SCHEMA_VERSION + 1
    store.path(DAY).write_text(json.dumps(document))
    with pytest.raises(RecordError, match="schema"):
        store.load(DAY)


def test_the_store_ignores_files_that_are_not_dated_records(tmp_path) -> None:
    store = RecordStore(tmp_path)
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "2026-13-45.json").write_text("{}")
    store.save(synthetic_record(DAY))
    assert store.dates() == {DAY}
