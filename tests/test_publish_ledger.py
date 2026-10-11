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


def test_evenings_are_ordered_by_date_not_by_the_order_they_were_sealed(
    shuffled_ledger_dir: Path,
) -> None:
    index = (shuffled_ledger_dir / "index.jsonl").read_text().splitlines()
    listed = [json.loads(line)["date"] for line in index]
    assert listed == ["2026-10-13", "2026-10-11", "2026-10-12"], "the fixture must be unsorted"
    ledger = read_ledger(shuffled_ledger_dir)
    assert [evening.date for evening in ledger.evenings] == list(reversed(factory.DATES))
    assert [evening.model_version for evening in ledger.evenings] == ["0.0.3", "0.0.2", "0.0.1"]


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
    factory.seal_date(ledger_dir, date, monkeypatch, strip=factory.strip_png(), late=True)
    late = evening_of(ledger_dir)
    assert late.date == date
    assert late.verified is False
    assert late.reason.startswith("late seal")
    assert late.strip_verified is False, "verify stopped at the late seal and never checked it"


# --- the painting is sealed with the forecast: only `verify` says whether it checked out -------


def test_an_evening_sealed_with_its_painting_has_a_verified_strip(ledger_dir: Path) -> None:
    assert all(evening.strip_verified for evening in read_ledger(ledger_dir).evenings)


def test_an_evening_sealed_without_a_painting_verifies_but_has_no_verified_strip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "ledger"
    factory.seal_date(ledger, NEWEST, monkeypatch)
    bare = evening_of(ledger)
    assert (bare.verified, bare.reason) == (True, "")
    assert bare.strip_verified is False
    assert bare.strip is None


@pytest.mark.parametrize("how", ["swapped", "missing"])
def test_a_painting_that_no_longer_matches_the_forecast_is_not_verified(
    ledger_dir: Path, how: str
) -> None:
    if how == "swapped":
        factory.write_strip(ledger_dir, NEWEST, factory.strip_png((9, 9, 9)))
    else:
        factory.strip_path(ledger_dir, NEWEST).unlink()
    newest = evening_of(ledger_dir)
    assert newest.verified is False
    assert newest.strip_verified is False
    assert newest.reason.startswith("strip mismatch" if how == "swapped" else "no sealed strip")
    assert evening_of(ledger_dir, 1).strip_verified is True


def test_a_strip_beside_a_forecast_that_names_none_is_not_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "ledger"
    factory.seal_date(ledger, NEWEST, monkeypatch)
    factory.write_strip(ledger, NEWEST)
    planted = evening_of(ledger)
    assert (planted.verified, planted.strip_verified) == (False, False)
    assert planted.reason.startswith("an unsealed strip")


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
    assert newest.strip_verified is False, "an evening that does not verify seals nothing"
    assert "index" in newest.reason
    assert newest.digest == helpers.digest_hex(
        ledger_paths(ledger_dir, NEWEST).forecast.read_bytes()
    )


def test_a_missing_painting_is_none_and_a_present_one_is_its_path(ledger_dir: Path) -> None:
    painting = factory.strip_path(ledger_dir, NEWEST)
    assert painting.name == "13.strip.png"
    assert evening_of(ledger_dir).strip == painting
    painting.unlink()
    gone = evening_of(ledger_dir)
    assert gone.strip is None
    assert (gone.verified, gone.strip_verified) == (False, False)
    assert evening_of(ledger_dir, 1).strip is not None


@pytest.mark.parametrize("target", ["outside", "inside"])
def test_a_painting_that_is_a_symlink_is_left_out(
    ledger_dir: Path, tmp_path: Path, target: str
) -> None:
    """Whether the link leads out of the ledger or to another file inside it."""
    if target == "outside":
        real = tmp_path / "elsewhere.png"
        real.write_bytes(factory.strip_png((1, 2, 3)))
    else:
        real = factory.strip_path(ledger_dir, OLDEST)
    painting = factory.strip_path(ledger_dir, NEWEST)
    painting.unlink()
    painting.symlink_to(real)
    assert evening_of(ledger_dir).strip is None
    assert evening_of(ledger_dir, 1).strip is not None


@pytest.mark.parametrize("linked", ["2026", "2026/10"])
def test_a_painting_reached_through_a_symlinked_directory_is_left_out(
    ledger_dir: Path, tmp_path: Path, linked: str
) -> None:
    moved = tmp_path / "outside" / linked.replace("/", "-")
    moved.parent.mkdir()
    (ledger_dir / linked).rename(moved)
    (ledger_dir / linked).symlink_to(moved, target_is_directory=True)
    ledger = read_ledger(ledger_dir)
    assert [evening.date for evening in ledger.evenings] == list(reversed(factory.DATES))
    assert [evening.strip for evening in ledger.evenings] == [None, None, None]


def test_a_ledger_directory_that_is_itself_a_symlink_still_has_its_paintings(
    ledger_dir: Path, tmp_path: Path
) -> None:
    link = tmp_path / "linked-ledger"
    link.symlink_to(ledger_dir, target_is_directory=True)
    ledger = read_ledger(link)
    assert all(evening.strip is not None for evening in ledger.evenings)


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
