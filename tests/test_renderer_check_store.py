"""The check's own store of per-evening inputs: one JSON file per evening under `renderer-v1/`.

The store holds what the report needs from the archives (aod_550, ozone, the cloud mean) or the
reason an evening was dropped. It is cache, outside git, and it is not G0's store.
"""

from __future__ import annotations

import datetime as dt
import json
import math

import pytest

from lastlight.renderer_check import store

DAY = dt.date(2026, 10, 11)


def rendered(day: dt.date = DAY, **kw) -> store.EveningInputs:
    values = {"aod_550": 0.04, "ozone_du": 281.5, "cloud_mean_pct": 3.25, "inputs": {"k": "v"}}
    values.update(kw)
    return store.EveningInputs.rendered(day, **values)


def test_a_rendered_record_round_trips(tmp_path) -> None:
    records = store.InputStore(tmp_path / "renderer-v1")
    records.save(rendered())
    loaded = records.load(DAY)
    assert loaded is not None and loaded.ok
    assert (loaded.date, loaded.aod_550, loaded.ozone_du) == (DAY, 0.04, 281.5)
    assert loaded.cloud_mean_pct == 3.25 and loaded.cloud_points == 17
    assert loaded.inputs == {"k": "v"} and loaded.drop_reason is None


def test_a_dropped_record_round_trips_with_its_reason(tmp_path) -> None:
    records = store.InputStore(tmp_path)
    records.save(store.EveningInputs.dropped(DAY, "tcdc_missing", "a point had no value"))
    loaded = records.load(DAY)
    assert loaded is not None and not loaded.ok
    assert loaded.drop_reason == "tcdc_missing" and loaded.drop_detail == "a point had no value"
    assert loaded.aod_550 is None and loaded.cloud_mean_pct is None


def test_the_reasons_are_g0s_and_tcdc_missing() -> None:
    from lastlight.g0.record import DROP_REASONS

    expected = set(DROP_REASONS) - {"not_collected"}
    assert expected <= set(store.INPUT_DROP_REASONS)
    assert "tcdc_missing" in store.INPUT_DROP_REASONS
    assert all(text.endswith(".") for text in store.INPUT_DROP_REASONS.values())


def test_an_unknown_reason_is_refused() -> None:
    with pytest.raises(store.InputRecordError, match="unknown drop reason"):
        store.EveningInputs.dropped(DAY, "because", "")


def test_a_dropped_record_cannot_carry_values_and_a_rendered_one_needs_them() -> None:
    with pytest.raises(store.InputRecordError):
        store.EveningInputs(
            DAY,
            "tcdc_missing",
            "",
            aod_550=0.1,
            ozone_du=280.0,
            cloud_mean_pct=1.0,
            cloud_points=17,
            inputs={},
        )
    with pytest.raises(store.InputRecordError):
        store.EveningInputs(
            DAY,
            None,
            "",
            aod_550=None,
            ozone_du=280.0,
            cloud_mean_pct=1.0,
            cloud_points=17,
            inputs={},
        )


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_values_are_refused(bad: float) -> None:
    for field in ("aod_550", "ozone_du", "cloud_mean_pct"):
        with pytest.raises(store.InputRecordError):
            rendered(**{field: bad})


def test_a_cloud_mean_outside_zero_to_one_hundred_is_refused() -> None:
    for bad in (-0.1, 100.1):
        with pytest.raises(store.InputRecordError):
            rendered(cloud_mean_pct=bad)
    rendered(cloud_mean_pct=0.0)
    rendered(cloud_mean_pct=100.0)


def test_the_file_is_sorted_keyed_json_with_a_trailing_newline_named_for_the_date(tmp_path) -> None:
    records = store.InputStore(tmp_path)
    records.save(rendered())
    path = records.path(DAY)
    assert path.name == "2026-10-11.json"
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    document = json.loads(text)
    assert list(document) == sorted(document)
    assert document["schema_version"] == 1


def test_saving_replaces_all_or_nothing_and_leaves_no_partial_file(tmp_path) -> None:
    records = store.InputStore(tmp_path)
    records.save(rendered(cloud_mean_pct=1.0))
    records.save(rendered(cloud_mean_pct=2.0))
    assert records.load(DAY).cloud_mean_pct == 2.0
    assert [p.name for p in tmp_path.iterdir()] == ["2026-10-11.json"]


def test_dates_lists_the_dates_with_a_file_and_ignores_other_names(tmp_path) -> None:
    records = store.InputStore(tmp_path)
    assert records.dates() == set()
    records.save(rendered(DAY))
    records.save(rendered(dt.date(2026, 10, 12)))
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "2026-13-45.json").write_text("{}")
    assert records.dates() == {DAY, dt.date(2026, 10, 12)}


def test_a_missing_record_is_none(tmp_path) -> None:
    assert store.InputStore(tmp_path / "nowhere").load(DAY) is None


def test_a_damaged_file_is_an_error_naming_it(tmp_path) -> None:
    records = store.InputStore(tmp_path)
    records.path(DAY).write_text("{broken", encoding="utf-8")
    with pytest.raises(store.InputRecordError, match="2026-10-11.json"):
        records.load(DAY)


def test_a_file_holding_another_date_is_an_error(tmp_path) -> None:
    records = store.InputStore(tmp_path)
    records.save(rendered(dt.date(2026, 10, 12)))
    records.path(DAY).write_text(
        records.path(dt.date(2026, 10, 12)).read_text(encoding="utf-8"), encoding="utf-8"
    )
    with pytest.raises(store.InputRecordError, match="not of 2026-10-11"):
        records.load(DAY)


def test_another_schema_version_is_an_error(tmp_path) -> None:
    records = store.InputStore(tmp_path)
    records.save(rendered())
    document = json.loads(records.path(DAY).read_text(encoding="utf-8"))
    document["schema_version"] = 2
    records.path(DAY).write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(store.InputRecordError, match="schema"):
        records.load(DAY)


def test_a_non_finite_value_in_the_file_is_an_error(tmp_path) -> None:
    records = store.InputStore(tmp_path)
    records.save(rendered())
    text = records.path(DAY).read_text(encoding="utf-8").replace("3.25", "NaN")
    records.path(DAY).write_text(text, encoding="utf-8")
    with pytest.raises(store.InputRecordError):
        records.load(DAY)
