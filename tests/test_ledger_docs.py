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


def test_the_guide_no_longer_says_the_signature_is_unchecked(text: str) -> None:
    # Until issue #25 the guide said `lastlight verify` trusted the response as delivered, in
    # words a test pinned. The check has landed, so the caveat is gone and must not come back.
    assert "does not yet validate" not in text
    assert "issue #25 will add" not in text
    assert "trusts the response as delivered" not in text


@pytest.mark.parametrize(
    "needle",
    [
        "src/lastlight/seal/certs",
        "FreeTSA",
        "pinned",
        "signature",
        "certificate chain",
        "extended key usage",
        "validity",
        "names which check failed",
        "the time the token names",
        "openssl ts -verify",
    ],
)
def test_the_guide_says_what_verify_now_checks_about_the_signature(text: str, needle: str) -> None:
    assert needle in text


def test_the_guide_spells_no_clock_time_or_position(text: str) -> None:
    assert not contains_clock_time(text)
    assert text_hits(text) == []


def test_the_readme_points_at_the_guide() -> None:
    assert "docs/ledger.md" in (REPO_ROOT / "README.md").read_text()


def test_the_design_contract_describes_the_ledger_as_built() -> None:
    design = (REPO_ROOT / "docs" / "design.md").read_text()
    assert "ledger/YYYY/MM/DD.strip.png" in design
    assert "orphan" in design and "`ledger` branch" in design
