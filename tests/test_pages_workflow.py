"""`.github/workflows/pages.yml`: its shape, and the shell it runs, without running Actions.

Public-repository minutes still draw down the owner's shared meter, so the workflow is one job
with narrow triggers. Two steps decide what to skip (is there a `ledger` branch, is GitHub Pages
on), and those are shell; the tests lift their `run` blocks out of the YAML and execute them
against fake `git` and `gh` programs.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "pages.yml"
PIN = re.compile(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}")


@pytest.fixture(scope="module")
def workflow() -> dict[Any, Any]:
    return yaml.safe_load(WORKFLOW.read_text())


@pytest.fixture(scope="module")
def job(workflow: dict[Any, Any]) -> dict[str, Any]:
    (only,) = workflow["jobs"].values()
    return only


def triggers(workflow: dict[Any, Any]) -> dict[str, Any]:
    # YAML 1.1 reads the bare key `on` as the boolean True.
    return workflow["on"] if "on" in workflow else workflow[True]


def step(job: dict[str, Any], step_id: str) -> dict[str, Any]:
    (found,) = [item for item in job["steps"] if item.get("id") == step_id]
    return found


def position(job: dict[str, Any], needle: str) -> int:
    """Index of the one step that runs or uses `needle`, or (as `id:NAME`) has that id."""
    if needle.startswith("id:"):
        matches = [i for i, item in enumerate(job["steps"]) if item.get("id") == needle[3:]]
    else:
        matches = [
            i
            for i, item in enumerate(job["steps"])
            if needle in item.get("run", "") or needle in item.get("uses", "")
        ]
    (index,) = matches
    return index


# --- shape ------------------------------------------------------------------------------------


def test_the_workflow_is_one_job_with_a_short_timeout(job: dict[str, Any]) -> None:
    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] <= 10


def test_it_runs_on_ledger_pushes_generator_changes_dispatch_and_after_each_seal(
    workflow: dict[Any, Any],
) -> None:
    on = triggers(workflow)
    assert set(on) == {"push", "workflow_run", "workflow_dispatch"}
    assert set(on["push"]["branches"]) == {"main", "ledger"}
    assert on["workflow_run"]["workflows"] == ["seal"]
    assert on["workflow_run"]["types"] == ["completed"]


def test_pushes_to_main_only_count_when_the_generator_or_its_inputs_change(
    workflow: dict[Any, Any],
) -> None:
    paths = set(triggers(workflow)["push"]["paths"])
    assert {
        "src/lastlight/publish/**",
        "src/lastlight/cli.py",
        "src/lastlight/commands/site.py",
        "pyproject.toml",
        "uv.lock",
        ".github/workflows/pages.yml",
    } <= paths
    # `main` never carries daily data; these only ever match pushes to the `ledger` branch.
    assert {"ledger/**"} <= paths
    assert not {"docs/**", "tests/**", "README.md"} & paths


def test_a_seal_that_failed_does_not_deploy(job: dict[str, Any]) -> None:
    condition = job["if"]
    assert "workflow_run" in condition and "conclusion == 'success'" in condition


def test_runs_are_serialised_and_a_running_deploy_is_never_cancelled(
    workflow: dict[Any, Any],
) -> None:
    """Concurrency is decided before the job's `if`, so a skipped run still takes the group.

    With cancel-in-progress on, a `workflow_run` for a failed seal (skipped by the job
    condition) would cancel a deploy that is already running. Queueing is enough: Pages keeps
    only the newest waiting run.
    """
    assert workflow["concurrency"]["cancel-in-progress"] is False
    assert workflow["concurrency"]["group"]


def test_the_token_can_only_read_the_repository_and_deploy_pages(workflow: dict[Any, Any]) -> None:
    assert workflow["permissions"] == {"contents": "read", "pages": "write", "id-token": "write"}


def test_the_job_deploys_to_the_github_pages_environment(job: dict[str, Any]) -> None:
    assert job["environment"]["name"] == "github-pages"


def test_every_action_is_pinned_to_a_commit_with_its_version_in_a_comment(
    job: dict[str, Any],
) -> None:
    uses = [item["uses"] for item in job["steps"] if "uses" in item]
    assert {name.split("@")[0] for name in uses} >= {
        "actions/checkout",
        "astral-sh/setup-uv",
        "actions/configure-pages",
        "actions/upload-pages-artifact",
        "actions/deploy-pages",
    }
    for action in uses:
        assert PIN.fullmatch(action), f"not a SHA pin: {action}"
    for line in WORKFLOW.read_text().splitlines():
        if "uses:" in line:
            assert re.search(r"@[0-9a-f]{40} # v\d+\.\d+\.\d+$", line), line


def test_the_environment_is_the_locked_one_ci_tests(job: dict[str, Any]) -> None:
    assert job["env"]["UV_FROZEN"] == "1"
    setup_uv = next(item for item in job["steps"] if item.get("uses", "").startswith("astral-sh/"))
    assert re.fullmatch(r"\d+\.\d+\.\d+", str(setup_uv["with"]["version"]))


def test_code_comes_from_main_and_data_from_the_ledger_branch(job: dict[str, Any]) -> None:
    checkouts = [
        item for item in job["steps"] if item.get("uses", "").startswith("actions/checkout")
    ]
    assert [c["with"]["ref"] for c in checkouts] == ["main", "ledger"]
    assert all(c["with"]["persist-credentials"] is False for c in checkouts)
    ledger_checkout = checkouts[1]
    assert ledger_checkout["if"] == "steps.ledger.outputs.present == 'true'"
    data = ledger_checkout["with"]["path"]
    build = next(item for item in job["steps"] if "lastlight site build" in item.get("run", ""))
    assert f"--ledger-dir {data}/ledger" in build["run"]
    assert "--out site" in build["run"]


def test_the_site_is_built_and_scanned_before_anything_is_uploaded(job: dict[str, Any]) -> None:
    build = position(job, "lastlight site build")
    scan = position(job, "lastlight privacy-scan site")
    check = position(job, "id:pages")
    configure = position(job, "actions/configure-pages")
    upload = position(job, "actions/upload-pages-artifact")
    deploy = position(job, "actions/deploy-pages")
    assert build < scan < check < configure < upload < deploy


def test_pages_steps_are_skipped_when_pages_is_off_but_the_scan_always_runs(
    job: dict[str, Any],
) -> None:
    gated = [
        item
        for item in job["steps"]
        if item.get("uses", "").split("@")[0]
        in {"actions/configure-pages", "actions/upload-pages-artifact", "actions/deploy-pages"}
    ]
    assert len(gated) == 3
    assert all(item["if"] == "steps.pages.outputs.enabled == 'true'" for item in gated)
    scan = next(item for item in job["steps"] if "privacy-scan site" in item.get("run", ""))
    assert "if" not in scan
    build = next(item for item in job["steps"] if "lastlight site build" in item.get("run", ""))
    assert "if" not in build


def test_the_artifact_is_the_site_directory(job: dict[str, Any]) -> None:
    upload = next(item for item in job["steps"] if "upload-pages-artifact" in item.get("uses", ""))
    assert upload["with"]["path"] == "site"


# --- the two decisions, run as shell --------------------------------------------------------

pytestmark_shell = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def fake_program(directory: Path, name: str) -> None:
    """A stand-in for `git` or `gh`: logs its arguments, prints and exits as the test says."""
    path = directory / name
    path.write_text(
        "#!/bin/bash\n"
        f'echo "{name} $*" >> "$FAKE_LOG"\n'
        f'printf "%s" "${{FAKE_{name.upper()}_STDOUT:-}}"\n'
        f'printf "%s" "${{FAKE_{name.upper()}_STDERR:-}}" >&2\n'
        f"exit ${{FAKE_{name.upper()}_STATUS:-0}}\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def run_step(
    job: dict[str, Any], step_id: str, tmp_path: Path, **fake: str
) -> tuple[int, dict[str, str], str, str]:
    """Run a step's shell; returns (exit code, $GITHUB_OUTPUT as a dict, stdout, the fake log)."""
    script = tmp_path / "step.sh"
    script.write_text(step(job, step_id)["run"])
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name in ("git", "gh"):
        fake_program(bin_dir, name)
    outputs = tmp_path / "output"
    outputs.write_text("")
    log = tmp_path / "log"
    log.write_text("")
    environment = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "GITHUB_OUTPUT": str(outputs),
        "GITHUB_REPOSITORY": "someone/lastlight",
        "GITHUB_SERVER_URL": "https://github.example",
        "GH_TOKEN": "not-a-token",
        "FAKE_LOG": str(log),
        **fake,
    }
    result = subprocess.run(
        ["bash", str(script)], env=environment, capture_output=True, text=True, timeout=60
    )
    parsed = dict(line.split("=", 1) for line in outputs.read_text().splitlines())
    return result.returncode, parsed, result.stdout + result.stderr, log.read_text()


@pytestmark_shell
def test_a_ledger_branch_that_exists_is_checked_out(job: dict[str, Any], tmp_path: Path) -> None:
    code, outputs, _text, log = run_step(job, "ledger", tmp_path, FAKE_GIT_STATUS="0")
    assert (code, outputs) == (0, {"present": "true"})
    assert "ls-remote --exit-code --heads https://github.example/someone/lastlight ledger" in log


@pytestmark_shell
def test_a_missing_ledger_branch_is_a_notice_not_a_failure(
    job: dict[str, Any], tmp_path: Path
) -> None:
    code, outputs, text, _log = run_step(job, "ledger", tmp_path, FAKE_GIT_STATUS="2")
    assert (code, outputs) == (0, {"present": "false"})
    assert "::notice" in text


@pytestmark_shell
def test_a_git_failure_is_not_mistaken_for_a_missing_branch(
    job: dict[str, Any], tmp_path: Path
) -> None:
    """A network error must not deploy an empty site over the real one."""
    code, outputs, text, _log = run_step(job, "ledger", tmp_path, FAKE_GIT_STATUS="128")
    assert code != 0
    assert outputs == {}
    assert "::error" in text


@pytestmark_shell
def test_pages_that_is_on_is_deployed_to(job: dict[str, Any], tmp_path: Path) -> None:
    code, outputs, _text, log = run_step(
        job, "pages", tmp_path, FAKE_GH_STATUS="0", FAKE_GH_STDOUT='{"status":"built"}'
    )
    assert (code, outputs) == (0, {"enabled": "true"})
    assert "gh api repos/someone/lastlight/pages" in log


@pytestmark_shell
def test_pages_that_is_off_skips_the_deploy_with_a_notice(
    job: dict[str, Any], tmp_path: Path
) -> None:
    code, outputs, text, _log = run_step(
        job,
        "pages",
        tmp_path,
        FAKE_GH_STATUS="1",
        FAKE_GH_STDOUT='{"message":"Not Found"}',
        FAKE_GH_STDERR="gh: Not Found (HTTP 404)",
    )
    assert (code, outputs) == (0, {"enabled": "false"})
    assert "::notice" in text and "not enabled" in text


@pytestmark_shell
@pytest.mark.parametrize("stderr", ["gh: Forbidden (HTTP 403)", "gh: Bad Gateway (HTTP 502)", ""])
def test_any_other_pages_api_error_fails_the_job(
    job: dict[str, Any], tmp_path: Path, stderr: str
) -> None:
    code, outputs, text, _log = run_step(
        job, "pages", tmp_path, FAKE_GH_STATUS="1", FAKE_GH_STDERR=stderr
    )
    assert code != 0
    assert outputs == {}
    assert "::error" in text


def test_the_pages_check_uses_the_workflow_token_and_no_write_method(job: dict[str, Any]) -> None:
    check = step(job, "pages")
    assert check["env"]["GH_TOKEN"] == "${{ github.token }}"
    assert not re.search(r"-X\s*(POST|PUT|PATCH|DELETE)|--method|-f\b|--field", check["run"])
