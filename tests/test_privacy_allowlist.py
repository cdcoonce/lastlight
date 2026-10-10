"""`privacy-allowlist.toml`: entries suppress exactly their match, and bad entries are errors."""

from __future__ import annotations

import textwrap
from pathlib import Path

import image_factory as images
import pytest

from lastlight.privacy import Finding, load_allowlist, scan
from lastlight.site import PRIVATE_SITE_SENTINEL

LEAK_A = "17:42"
LEAK_B = "18:15"


def write_allowlist(root: Path, body: str) -> Path:
    path = root / "privacy-allowlist.toml"
    path.write_text(textwrap.dedent(body))
    return path


def entry(
    path: str, rule: str, match: str, reason: str = "fixed seal time, site-independent"
) -> str:
    return f'[[allow]]\npath = "{path}"\nrule = "{rule}"\nmatch = "{match}"\nreason = "{reason}"\n'


def scan_with(root: Path, allowlist_body: str | None, *names: str) -> list[Finding]:
    allowlist = None
    if allowlist_body is not None:
        allowlist = load_allowlist(write_allowlist(root, allowlist_body), root=root)
    return scan([root / name for name in names], root=root, allowlist=allowlist).findings


def describe(findings: list[Finding]) -> list[tuple[str, int, str]]:
    return [(f.path, f.line, f.rule) for f in findings]


# --- an entry suppresses exactly its match ---------------------------------------------------


def test_an_entry_suppresses_its_match_and_nothing_else_in_the_file(tmp_path: Path) -> None:
    (tmp_path / "seal.yml").write_text(f"cron at {LEAK_A}\nmanual at {LEAK_B}\n")

    unsuppressed = scan_with(tmp_path, None, "seal.yml")
    suppressed = scan_with(tmp_path, entry("seal.yml", "P001", LEAK_A), "seal.yml")

    assert describe(unsuppressed) == [("seal.yml", 1, "P001"), ("seal.yml", 2, "P001")]
    assert describe(suppressed) == [("seal.yml", 2, "P001")]


def test_an_entry_suppresses_every_occurrence_of_its_match_in_the_file(tmp_path: Path) -> None:
    (tmp_path / "seal.yml").write_text(f"a {LEAK_A}\nb {LEAK_A}\n")
    assert scan_with(tmp_path, entry("seal.yml", "P001", LEAK_A), "seal.yml") == []


def test_an_entry_only_applies_to_its_own_rule(tmp_path: Path) -> None:
    (tmp_path / "seal.yml").write_text(f"sunset {LEAK_A}\n")
    findings = scan_with(tmp_path, entry("seal.yml", "P001", LEAK_A), "seal.yml")
    assert describe(findings) == [("seal.yml", 1, "P002")]


def test_an_entry_only_applies_to_its_own_path(tmp_path: Path) -> None:
    (tmp_path / "seal.yml").write_text(f"at {LEAK_A}\n")
    (tmp_path / "other.yml").write_text(f"at {LEAK_A}\n")
    findings = scan_with(tmp_path, entry("seal.yml", "P001", LEAK_A), "seal.yml", "other.yml")
    assert describe(findings) == [("other.yml", 1, "P001")]


def test_an_entry_whose_match_is_not_on_the_line_suppresses_nothing(tmp_path: Path) -> None:
    (tmp_path / "seal.yml").write_text(f"at {LEAK_B}\n")
    findings = scan_with(tmp_path, entry("seal.yml", "P001", LEAK_A), "seal.yml")
    assert describe(findings) == [("seal.yml", 1, "P001")]


def test_match_is_a_literal_substring_of_the_line_not_a_pattern(tmp_path: Path) -> None:
    (tmp_path / "lock.toml").write_text('wheel = { upload-time = "2025-07-17T18:22:03Z" }\n')
    assert scan_with(tmp_path, entry("lock.toml", "P001", "upload-time"), "lock.toml") == []
    (tmp_path / "other.toml").write_text('wheel = { upload-time = "2025-07-17T18:22:03Z" }\n')
    assert scan_with(tmp_path, entry("other.toml", "P001", "upload.time"), "other.toml") != []


def test_path_may_be_a_glob(tmp_path: Path) -> None:
    (tmp_path / "ci").mkdir()
    (tmp_path / "ci" / "a.yml").write_text(f"at {LEAK_A}\n")
    (tmp_path / "b.yml").write_text(f"at {LEAK_A}\n")
    findings = scan_with(tmp_path, entry("ci/*.yml", "P001", LEAK_A), "ci/a.yml", "b.yml")
    assert describe(findings) == [("b.yml", 1, "P001")]


def test_an_entry_can_suppress_a_coordinate_pair_and_a_binary_finding(tmp_path: Path) -> None:
    (tmp_path / "geo.txt").write_text("fixture 33.4484, -112.0740\n")
    (tmp_path / "fixture.png").write_bytes(images.png_with_text())
    body = entry("geo.txt", "P003", "33.4484") + entry("fixture.png", "P004", "tEXt")
    assert scan_with(tmp_path, body, "geo.txt", "fixture.png") == []


# --- a reason is mandatory -------------------------------------------------------------------


@pytest.mark.parametrize("reason", ["", "   ", "\\t"])
def test_an_entry_with_an_empty_reason_is_an_error_and_suppresses_nothing(
    tmp_path: Path, reason: str
) -> None:
    (tmp_path / "seal.yml").write_text(f"at {LEAK_A}\n")
    findings = scan_with(tmp_path, entry("seal.yml", "P001", LEAK_A, reason), "seal.yml")
    assert sorted(f.rule for f in findings) == ["A001", "P001"]
    (error,) = (f for f in findings if f.rule == "A001")
    assert error.path == "privacy-allowlist.toml"
    assert "reason" in error.excerpt


def test_an_entry_without_a_reason_key_is_an_error(tmp_path: Path) -> None:
    body = '[[allow]]\npath = "seal.yml"\nrule = "P001"\nmatch = "17:42"\n'
    (tmp_path / "seal.yml").write_text(f"at {LEAK_A}\n")
    findings = scan_with(tmp_path, body, "seal.yml")
    assert sorted(f.rule for f in findings) == ["A001", "P001"]


def test_only_the_bad_entry_is_dropped(tmp_path: Path) -> None:
    (tmp_path / "seal.yml").write_text(f"a {LEAK_A}\nb {LEAK_B}\n")
    body = entry("seal.yml", "P001", LEAK_A) + entry("seal.yml", "P001", LEAK_B, reason="")
    findings = scan_with(tmp_path, body, "seal.yml")
    assert describe(findings) == [
        ("privacy-allowlist.toml", 6, "A001"),
        ("seal.yml", 2, "P001"),
    ]


def test_the_error_names_the_entry_by_its_position_and_fields(tmp_path: Path) -> None:
    body = entry("a.yml", "P001", "x") + entry("seal.yml", "P001", LEAK_A, reason="")
    allowlist = load_allowlist(write_allowlist(tmp_path, body), root=tmp_path)
    (problem,) = allowlist.problems
    assert "entry 2" in problem.excerpt
    assert "seal.yml" in problem.excerpt


@pytest.mark.parametrize(
    ("body", "expect"),
    [
        ('[[allow]]\nrule = "P001"\nmatch = "x"\nreason = "r"\n', "path"),
        ('[[allow]]\npath = "a"\nmatch = "x"\nreason = "r"\n', "rule"),
        ('[[allow]]\npath = "a"\nrule = "P001"\nreason = "r"\n', "match"),
        ('[[allow]]\npath = "a"\nrule = "P001"\nmatch = ""\nreason = "r"\n', "match"),
        ('[[allow]]\npath = "a"\nrule = "P009"\nmatch = "x"\nreason = "r"\n', "rule"),
        ('[[allow]]\npath = "a"\nrule = "P001"\nmatch = "x"\nreason = 7\n', "reason"),
        ('[[allow]]\npath = "a"\nrule = "P001"\nmatch = "x"\nreason = "r"\nextra = 1\n', "extra"),
    ],
)
def test_malformed_entries_are_errors(tmp_path: Path, body: str, expect: str) -> None:
    allowlist = load_allowlist(write_allowlist(tmp_path, body), root=tmp_path)
    assert allowlist.entries == ()
    (problem,) = allowlist.problems
    assert (problem.rule, expect in problem.excerpt) == ("A001", True)


def test_a_file_that_is_not_valid_toml_is_one_error(tmp_path: Path) -> None:
    allowlist = load_allowlist(write_allowlist(tmp_path, "[[allow\nnope"), root=tmp_path)
    (problem,) = allowlist.problems
    assert (problem.rule, problem.path) == ("A001", "privacy-allowlist.toml")


def test_an_unknown_top_level_key_is_an_error_not_a_silent_no_op(tmp_path: Path) -> None:
    body = entry("a", "P001", "x").replace("[[allow]]", "[[allowlist]]")
    allowlist = load_allowlist(write_allowlist(tmp_path, body), root=tmp_path)
    (problem,) = allowlist.problems
    assert "allowlist" in problem.excerpt


def test_a_valid_entry_has_no_problems(tmp_path: Path) -> None:
    allowlist = load_allowlist(write_allowlist(tmp_path, entry("a", "P001", "x")), root=tmp_path)
    assert allowlist.problems == ()
    (only,) = allowlist.entries
    assert (only.path, only.rule, only.match) == ("a", "P001", "x")


def test_an_empty_file_is_a_valid_empty_allowlist(tmp_path: Path) -> None:
    allowlist = load_allowlist(write_allowlist(tmp_path, ""), root=tmp_path)
    assert (allowlist.entries, allowlist.problems) == ((), ())


# --- the scanner's own files are exempt by path, and only those ------------------------------


def plant(root: Path, relative: str, content: str | bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)
    return path


def findings_for(root: Path, relative: str) -> list[Finding]:
    return scan([root / relative], root=root).findings


@pytest.mark.parametrize(
    "relative",
    [
        "src/lastlight/privacy/rules.py",
        "src/lastlight/privacy/nested/more.py",
        "tests/test_privacy_rules.py",
        "tests/test_privacy_whatever.py",
    ],
)
def test_the_scanners_own_source_and_tests_may_hold_the_patterns_they_define(
    tmp_path: Path, relative: str
) -> None:
    plant(tmp_path, relative, f"seal {LEAK_A}\nsunset 5 PM\ncamera 33.4484, -112.0740\n")
    assert findings_for(tmp_path, relative) == []


@pytest.mark.parametrize(
    "relative",
    [
        "src/lastlight/geometry.py",
        "src/lastlight/privacy_extra.py",
        "src/lastlight/commands/privacy_scan.py",
        "tests/test_geometry.py",
        "tests/privacy_notes.py",
        "docs/design.md",
        "other/src/lastlight/privacy/rules.py",
    ],
)
def test_every_other_path_is_scanned_normally(tmp_path: Path, relative: str) -> None:
    plant(tmp_path, relative, f"seal {LEAK_A}\n")
    assert [f.rule for f in findings_for(tmp_path, relative)] == ["P001"]


def test_the_scanner_exemption_does_not_cover_image_metadata(tmp_path: Path) -> None:
    # The name matches the exempt glob; the bytes are a PNG, which is recognised by content.
    relative = "tests/test_privacy_fixture.py"
    plant(tmp_path, relative, images.png_with_text())
    assert [f.rule for f in findings_for(tmp_path, relative)] == ["P004"]


def test_only_the_site_module_and_the_scanner_tests_may_name_the_sentinel(tmp_path: Path) -> None:
    plant(tmp_path, "src/lastlight/site.py", f'KEY = "{PRIVATE_SITE_SENTINEL}"\n')
    plant(tmp_path, "tests/test_privacy_site.py", f'KEY = "{PRIVATE_SITE_SENTINEL}"\n')
    plant(tmp_path, "src/lastlight/privacy/rules.py", f'KEY = "{PRIVATE_SITE_SENTINEL}"\n')
    plant(tmp_path, "src/lastlight/other.py", f'KEY = "{PRIVATE_SITE_SENTINEL}"\n')
    results = {
        relative: [f.rule for f in findings_for(tmp_path, relative)]
        for relative in (
            "src/lastlight/site.py",
            "tests/test_privacy_site.py",
            "src/lastlight/privacy/rules.py",
            "src/lastlight/other.py",
        )
    }
    assert results == {
        "src/lastlight/site.py": [],
        "tests/test_privacy_site.py": [],
        "src/lastlight/privacy/rules.py": ["P005"],
        "src/lastlight/other.py": ["P005"],
    }


def test_the_allowlist_file_may_quote_the_text_it_allows(tmp_path: Path) -> None:
    path = write_allowlist(tmp_path, entry("seal.yml", "P001", LEAK_A))
    assert scan([path], root=tmp_path).findings == []


def test_a_path_outside_the_root_is_never_exempt_or_allowlisted(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "privacy-allowlist.toml").write_text(f"at {LEAK_A}\n")
    allowlist = load_allowlist(write_allowlist(root, entry("*", "P001", LEAK_A)), root=root)
    findings = scan([outside / "privacy-allowlist.toml"], root=root, allowlist=allowlist).findings
    assert [f.rule for f in findings] == ["P001"]
