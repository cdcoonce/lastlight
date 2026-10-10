"""Repo-level contract from the scaffold issue: CI workflow, ignore rules, tooling config."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pyproject() -> dict[str, Any]:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())


@pytest.fixture(scope="module")
def workflow() -> dict[Any, Any]:
    return yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text())


def _triggers(workflow: dict[Any, Any]) -> dict[str, Any]:
    # YAML 1.1 reads the bare key `on` as the boolean True.
    return workflow["on"] if "on" in workflow else workflow[True]


# --- CI workflow -------------------------------------------------------------


@pytest.mark.parametrize("event", ["pull_request", "push"])
def test_ci_skips_docs_only_changes(workflow: dict[Any, Any], event: str) -> None:
    ignored = _triggers(workflow)[event]["paths-ignore"]
    assert "docs/**" in ignored
    assert "**/*.md" in ignored


def test_ci_runs_on_pull_requests_and_pushes_to_main_only(workflow: dict[Any, Any]) -> None:
    triggers = _triggers(workflow)
    assert set(triggers) == {"pull_request", "push"}
    assert triggers["push"]["branches"] == ["main"]


def test_ci_cancels_superseded_runs_per_ref(workflow: dict[Any, Any]) -> None:
    concurrency = workflow["concurrency"]
    assert "github.ref" in concurrency["group"]
    assert concurrency["cancel-in-progress"] is True


def test_ci_is_one_ubuntu_job(workflow: dict[Any, Any]) -> None:
    assert len(workflow["jobs"]) == 1
    (job,) = workflow["jobs"].values()
    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] <= 10


def test_ci_steps_are_the_gate_in_order(workflow: dict[Any, Any]) -> None:
    (job,) = workflow["jobs"].values()
    steps = job["steps"]
    assert steps[0]["uses"].startswith("actions/checkout@")
    assert steps[1]["uses"].startswith("astral-sh/setup-uv@")
    assert [step["run"] for step in steps[2:]] == [
        "uv sync --frozen",
        "uv run ruff check",
        "uv run ruff format --check",
        "uv run pytest",
        "uv run lastlight privacy-scan",
    ]


def test_ci_does_not_re_resolve_the_lockfile(workflow: dict[Any, Any]) -> None:
    (job,) = workflow["jobs"].values()
    assert job["env"]["UV_FROZEN"] == "1"


def test_ci_actions_and_uv_are_pinned(workflow: dict[Any, Any]) -> None:
    (job,) = workflow["jobs"].values()
    uses = [step["uses"] for step in job["steps"] if "uses" in step]
    for action in uses:
        assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", action), f"not a SHA pin: {action}"
    setup_uv = next(step for step in job["steps"] if step.get("uses", "").startswith("astral-sh/"))
    assert re.fullmatch(r"\d+\.\d+\.\d+", str(setup_uv["with"]["version"]))


def test_ci_grants_read_only_token(workflow: dict[Any, Any]) -> None:
    assert workflow["permissions"] == {"contents": "read"}


# --- ignore rules and lockfile -------------------------------------------------


def _gitignore_entries() -> set[str]:
    lines = (REPO_ROOT / ".gitignore").read_text().splitlines()
    return {line.strip().lstrip("/") for line in lines if line.strip() and not line.startswith("#")}


@pytest.mark.parametrize(
    "entry",
    [".venv/", "__pycache__/", "*.py[cod]", ".pytest_cache/", ".ruff_cache/", "dist/", "data/"],
)
def test_gitignore_covers_python_uv_caches_and_local_data(entry: str) -> None:
    assert entry in _gitignore_entries()


def test_lockfile_is_committed_not_ignored() -> None:
    assert (REPO_ROOT / "uv.lock").is_file()
    assert "uv.lock" not in _gitignore_entries()


# --- project metadata ------------------------------------------------------------


def test_project_metadata(pyproject: dict[str, Any]) -> None:
    project = pyproject["project"]
    assert project["name"] == "lastlight"
    assert project["requires-python"] == ">=3.12"
    assert project["scripts"] == {"lastlight": "lastlight.cli:main"}


def test_dev_group_has_the_test_runner_and_linter(pyproject: dict[str, Any]) -> None:
    dev = " ".join(pyproject["dependency-groups"]["dev"])
    assert "pytest" in dev
    assert "ruff" in dev


def test_pytest_deselects_network_tests_by_default(pyproject: dict[str, Any]) -> None:
    options = pyproject["tool"]["pytest"]["ini_options"]
    assert "-m 'not network'" in options["addopts"]
    assert any(marker.startswith("network:") for marker in options["markers"])
