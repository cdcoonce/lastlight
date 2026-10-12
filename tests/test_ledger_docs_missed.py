"""`docs/ledger.md` on missed evenings and GitHub's 60-day rule (issue #26).

The guide must say what the site says about an evening with no seal, document the rule that
disables scheduled workflows in a public repository, link GitHub's own pages, give the steps to
turn the workflow back on, and be honest about the one thing GitHub's documentation does not
settle (whether the bot's pushes to the `ledger` branch count as activity).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from lastlight.privacy import contains_clock_time
from lastlight.privacy.rules import text_hits

REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER_DOC = REPO_ROOT / "docs" / "ledger.md"

SCHEDULE_DOC = (
    "https://docs.github.com/en/actions/reference/workflows-and-actions/"
    "events-that-trigger-workflows#schedule"
)
ENABLE_DOC = (
    "https://docs.github.com/en/actions/how-tos/manage-workflow-runs/disable-and-enable-workflows"
)


@pytest.fixture(scope="module")
def text() -> str:
    return LEDGER_DOC.read_text()


@pytest.fixture(scope="module")
def sections(text: str) -> dict[str, str]:
    """The guide's second-level sections by heading."""
    parts = re.split(r"^## (.+)$", text, flags=re.MULTILINE)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts), 2)}


@pytest.fixture(scope="module")
def rule(sections: dict[str, str]) -> str:
    (heading,) = [name for name in sections if "60-day" in name]
    return sections[heading]


def test_the_guide_has_a_section_on_the_sixty_day_rule(sections: dict[str, str]) -> None:
    assert [name for name in sections if "60-day" in name]


def test_the_rule_is_stated_for_public_repositories_and_scheduled_workflows(rule: str) -> None:
    assert "public repository" in rule
    assert "scheduled workflow" in rule
    assert "60 days" in rule
    assert "seal.yml" in rule


def test_the_section_links_githubs_own_pages(rule: str) -> None:
    assert SCHEDULE_DOC in rule
    assert ENABLE_DOC in rule


def test_the_section_gives_both_ways_to_turn_the_workflow_back_on(rule: str) -> None:
    assert "Actions tab" in rule
    assert "gh workflow enable seal.yml -R cdcoonce/lastlight" in rule


def test_what_counts_as_activity_is_paraphrased_and_the_bot_question_is_marked_unverified(
    rule: str,
) -> None:
    assert "unverified" in rule.lower()
    assert "ledger" in rule and "bot" in rule
    assert "default branch" in rule


def test_at_most_one_short_phrase_of_githubs_text_is_quoted(rule: str) -> None:
    quotes = re.findall(r"“([^”]+)”|\"([^\"]+)\"", rule)
    phrases = [first or second for first, second in quotes]
    assert len(phrases) <= 1
    assert all(len(phrase.split()) <= 8 for phrase in phrases)


def test_the_section_says_where_each_evenings_outcome_is_recorded(rule: str) -> None:
    assert "sealed, retried or missed" in rule
    assert "build board" in rule


def test_the_guide_explains_what_the_site_says_about_a_missed_evening(
    sections: dict[str, str],
) -> None:
    (heading,) = [name for name in sections if "issed" in name and "60-day" not in name]
    body = sections[heading]
    assert "No forecast was sealed for" in body
    assert "--through" in body
    assert "never back-filled" in body or "never back-fill" in body
    assert "not a forecast" in body
    assert "retry" in body.lower() and "--attempts" in body


def test_the_guide_still_spells_no_clock_time_or_position(text: str) -> None:
    assert not contains_clock_time(text)
    assert text_hits(text) == []


def test_the_signature_caveat_is_untouched(text: str) -> None:
    assert (
        "`lastlight verify` checks the imprint and the time but does not yet validate the "
        "authority's signature chain (issue #25 will add signature verification), so until then "
        "it trusts the response as delivered."
    ) in text
