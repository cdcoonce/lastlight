"""How a finding is printed: one short line, whatever the matched text looked like."""

from __future__ import annotations

import textwrap
from pathlib import Path

from lastlight.privacy import Finding, load_allowlist, scan
from lastlight.privacy.findings import make_excerpt

LIMIT = 80


def test_a_short_excerpt_is_kept_as_it_is() -> None:
    assert make_excerpt("17:42") == "17:42"


def test_whitespace_in_an_excerpt_collapses_to_single_spaces_on_one_line() -> None:
    assert make_excerpt("sunset\n\t at   5 PM") == "sunset at 5 PM"


def test_an_excerpt_of_exactly_the_limit_is_kept_whole() -> None:
    text = "a" * LIMIT
    assert make_excerpt(text) == text


def test_an_excerpt_one_over_the_limit_is_cut_to_the_limit_and_ends_in_an_ellipsis() -> None:
    cut = make_excerpt("a" * (LIMIT + 1))
    assert cut == "a" * (LIMIT - 3) + "..."
    assert len(cut) == LIMIT


def test_a_very_long_excerpt_is_capped_at_the_limit() -> None:
    cut = make_excerpt("word " * 500)
    assert len(cut) == LIMIT
    assert cut.endswith("...")


def test_the_limit_is_measured_after_whitespace_collapses() -> None:
    padded = ("a" * 40) + (" " * 200) + ("b" * 39)  # 80 characters once collapsed
    assert make_excerpt(padded) == ("a" * 40) + " " + ("b" * 39)


def test_a_long_allowlist_error_prints_as_one_capped_line(tmp_path: Path) -> None:
    long_path = "deeply/nested/" * 20 + "seal.yml"
    body = textwrap.dedent(
        f"""\
        [[allow]]
        path = "{long_path}"
        rule = "P001"
        match = "17:42"
        reason = ""
        """
    )
    allowlist_file = tmp_path / "privacy-allowlist.toml"
    allowlist_file.write_text(body)
    allowlist = load_allowlist(allowlist_file, root=tmp_path)
    (problem,) = scan([allowlist_file], root=tmp_path, allowlist=allowlist).findings
    assert problem.rule == "A001"
    assert len(problem.excerpt) == LIMIT
    assert problem.excerpt.endswith("...")
    assert len(problem.format().splitlines()) == 1
    # The full message still reaches the allowlist matcher; only the printed excerpt is cut.
    assert long_path in problem.context


def test_a_finding_prints_as_path_line_rule_excerpt() -> None:
    finding = Finding("docs/page.md", 7, "P001", "17:42")
    assert finding.format() == "docs/page.md:7: P001: 17:42"
