"""The scan of this repository's own tracked tree: what CI runs as `lastlight privacy-scan`."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from lastlight.privacy import load_allowlist, scan

REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = REPO_ROOT / "privacy-allowlist.toml"


@pytest.fixture(scope="module", autouse=True)
def _inside_a_git_checkout() -> None:
    probe = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if probe.returncode != 0:
        pytest.skip("not a git checkout; the tracked-tree scan needs `git ls-files`")


def test_the_repository_allowlist_exists_and_every_entry_is_valid() -> None:
    assert ALLOWLIST.is_file()
    allowlist = load_allowlist(ALLOWLIST, root=REPO_ROOT)
    assert [problem.format() for problem in allowlist.problems] == []
    assert allowlist.entries, "the allowlist has at least the uv.lock entry"
    assert all(entry.reason.strip() for entry in allowlist.entries)


def test_the_repository_allowlist_never_exempts_the_private_site_marker() -> None:
    # Only the site module and the scanner's own tests may name the sentinel, and both are
    # exempt by path in code. An `[[allow]]` entry for P005 would open the same door anywhere.
    allowlist = load_allowlist(ALLOWLIST, root=REPO_ROOT)
    assert [entry.path for entry in allowlist.entries if entry.rule == "P005"] == []


def test_the_scan_of_the_current_repository_tree_passes() -> None:
    allowlist = load_allowlist(ALLOWLIST, root=REPO_ROOT)
    result = scan(None, root=REPO_ROOT, allowlist=allowlist)
    assert [finding.format() for finding in result.findings] == []
    assert result.files_scanned > 10, "the default scan must actually read the tracked tree"


def test_without_the_allowlist_the_lockfile_timestamps_are_found() -> None:
    # Positive control: the passing scan above is not vacuous, and the uv.lock entry is
    # what lets it pass. uv.lock carries PyPI upload-time stamps, which are P001 matches.
    result = scan(None, root=REPO_ROOT, allowlist=None)
    assert {finding.rule for finding in result.findings if finding.path == "uv.lock"} == {"P001"}
