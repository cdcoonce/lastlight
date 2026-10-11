"""`docs/ledger.md`: how anyone checks a sealed day without trusting the author (issue #8)."""

from __future__ import annotations

from pathlib import Path

import pytest

from lastlight.privacy import contains_clock_time
from lastlight.privacy.rules import text_hits

REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER_DOC = REPO_ROOT / "docs" / "ledger.md"


@pytest.fixture(scope="module")
def text() -> str:
    return LEDGER_DOC.read_text()


@pytest.mark.parametrize(
    "needle",
    [
        "git fetch origin ledger",
        "lastlight verify",
        "--ledger-dir ledger",
        "strip_sha256",
        "DD.strip.png",
        "DD.forecast.tsr",
        "index.jsonl",
        "RFC 3161",
        "before sunset",
    ],
)
def test_the_guide_names_what_a_verifier_needs(text: str, needle: str) -> None:
    assert needle in text


def test_the_guide_says_what_the_timestamp_does_not_prove(text: str) -> None:
    assert "does not prove" in text and "signature" in text


def test_the_guide_keeps_its_signature_chain_caveat_until_verify_checks_the_signature(
    text: str,
) -> None:
    # `lastlight verify` does not validate the timestamp authority's signature (issue #25 adds
    # it). Until then the guide must say so in these words; the caveat goes when the check lands.
    assert (
        "`lastlight verify` checks the imprint and the time but does not yet validate the "
        "authority's signature chain (issue #25 will add signature verification), so until then "
        "it trusts the response as delivered."
    ) in text


def test_the_guide_spells_no_clock_time_or_position(text: str) -> None:
    assert not contains_clock_time(text)
    assert text_hits(text) == []


def test_the_readme_points_at_the_guide() -> None:
    assert "docs/ledger.md" in (REPO_ROOT / "README.md").read_text()


def test_the_design_contract_describes_the_ledger_as_built() -> None:
    design = (REPO_ROOT / "docs" / "design.md").read_text()
    assert "ledger/YYYY/MM/DD.strip.png" in design
    assert "orphan" in design and "`ledger` branch" in design
