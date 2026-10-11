"""Reading a ledger for the almanac: dates, facts, paintings and the verification result."""

from __future__ import annotations

import json
from pathlib import Path

import ledger_factory as factory
import pytest
import seal_helpers as helpers

from lastlight.publish import Evening, read_ledger
from lastlight.seal import ledger_paths

NEWEST = factory.DATES[-1]
OLDEST = factory.DATES[0]


def evening_of(ledger_dir: Path, date_index: int = 0) -> Evening:
    return read_ledger(ledger_dir).evenings[date_index]


def test_evenings_come_newest_first(ledger_dir: Path) -> None:
    ledger = read_ledger(ledger_dir)
    assert [evening.date for evening in ledger.evenings] == list(reversed(factory.DATES))
    assert ledger.found is True
    assert ledger.skipped_index_lines == 0


def test_an_evening_carries_the_sealed_digest_and_model_version(ledger_dir: Path) -> None:
    newest = evening_of(ledger_dir)
    sealed = ledger_paths(ledger_dir, NEWEST).forecast.read_bytes()
    assert newest.digest == helpers.digest_hex(sealed)
    assert newest.model_version == "0.0.3"
    assert newest.model_name == "lastlight-stub"


def test_an_evening_carries_what_the_forecast_says_about_itself(ledger_dir: Path) -> None:
    newest = evening_of(ledger_dir)
    assert newest.vividness == (0.42, 0.07)
    assert newest.sun_elevations == (6.0, 0.0, -4.0, -8.0)
    assert newest.view_elevations == (2.0, 5.0, 10.0)
    assert newest.notes == "stub forecast for the seal tests"


def test_a_seal_made_before_sunset_is_verified(ledger_dir: Path) -> None:
    assert all(
        evening.verified and evening.reason == "" for evening in read_ledger(ledger_dir).evenings
    )


def test_a_late_seal_is_not_verified_and_says_why(
    ledger_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    date = NEWEST.replace(day=14)
    factory.seal_date(ledger_dir, date, monkeypatch, late=True)
    late = evening_of(ledger_dir)
    assert late.date == date
    assert late.verified is False
    assert late.reason.startswith("late seal")


def test_an_edited_forecast_is_not_verified(ledger_dir: Path) -> None:
    factory.overwrite_notes(ledger_dir, NEWEST, "edited after sealing")
    edited = evening_of(ledger_dir)
    assert edited.verified is False
    assert edited.reason.startswith("digest mismatch")
    assert edited.notes == "edited after sealing"


def test_an_index_row_that_names_another_digest_is_not_verified(ledger_dir: Path) -> None:
    index = ledger_dir / "index.jsonl"
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    rows[-1]["digest"] = helpers.flip_bit(rows[-1]["digest"], 0)
    index.write_text("".join(json.dumps(row) + "\n" for row in rows))
    newest = evening_of(ledger_dir)
    assert newest.verified is False
    assert "index" in newest.reason
    assert newest.digest == helpers.digest_hex(
        ledger_paths(ledger_dir, NEWEST).forecast.read_bytes()
    )


def test_a_missing_painting_is_none_and_a_present_one_is_its_path(ledger_dir: Path) -> None:
    painting = factory.strip_path(ledger_dir, NEWEST)
    assert painting.name == "13.strip.png"
    assert evening_of(ledger_dir).strip == painting
    painting.unlink()
    assert evening_of(ledger_dir).strip is None
    assert evening_of(ledger_dir, 1).strip is not None


def test_an_indexed_date_without_its_files_is_listed_unverified(ledger_dir: Path) -> None:
    ledger_paths(ledger_dir, NEWEST).forecast.unlink()
    gone = evening_of(ledger_dir)
    index_row = json.loads((ledger_dir / "index.jsonl").read_text().splitlines()[-1])
    assert gone.date == NEWEST
    assert gone.verified is False
    assert gone.digest == index_row["digest"]
    assert (gone.notes, gone.model_name, gone.vividness, gone.sun_elevations) == ("", "", None, ())
    assert gone.model_version == index_row["model_version"]


@pytest.mark.parametrize(
    "contents",
    [
        b"not json at all",
        b"[1, 2, 3]",
        b'{"model": 5, "notes": [1], "curve": "x", "vividness": {}}',
    ],
)
def test_a_forecast_file_that_is_not_the_expected_shape_gives_no_facts_and_no_crash(
    ledger_dir: Path, contents: bytes
) -> None:
    ledger_paths(ledger_dir, NEWEST).forecast.write_bytes(contents)
    odd = evening_of(ledger_dir)
    assert odd.verified is False
    assert (odd.notes, odd.model_name, odd.vividness, odd.sun_elevations) == ("", "", None, ())
    assert odd.model_version == "0.0.3"  # falls back to the index row


def test_unusable_index_lines_are_skipped_and_counted(ledger_dir: Path) -> None:
    index = ledger_dir / "index.jsonl"
    good = index.read_text().splitlines()
    junk = [
        "not json",
        '{"date": "2026-13-45", "digest": "' + "0" * 64 + '", "model_version": "x"}',
        '{"date": "2026-10-20", "digest": "short", "model_version": "x"}',
        '{"date": "2026-10-20", "digest": "' + "0" * 64 + '"}',
        "[]",
        good[0],  # a second row for an already listed date
    ]
    index.write_text("\n".join([*junk[:3], good[0], *junk[3:], *good[1:]]) + "\n\n")
    ledger = read_ledger(ledger_dir)
    assert [evening.date for evening in ledger.evenings] == list(reversed(factory.DATES))
    assert ledger.skipped_index_lines == len(junk)


def test_a_ledger_that_is_not_there_is_empty(tmp_path: Path) -> None:
    ledger = read_ledger(tmp_path / "nothing-here")
    assert (ledger.evenings, ledger.found, ledger.skipped_index_lines) == ((), False, 0)


def test_a_ledger_directory_without_an_index_is_empty(tmp_path: Path) -> None:
    ledger = read_ledger(tmp_path)
    assert (ledger.evenings, ledger.found) == ((), True)
