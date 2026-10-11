"""The structure of `.github/workflows/seal.yml` (issue #8).

The behaviour of the seal (idempotence, never late, verification) is tested where it lives, in
Python (`test_daily.py`, `test_ledger_branch.py`). What only the YAML can get wrong is checked
here: the trigger, who may write, that runs queue instead of cancelling, that every action is
pinned, and that the steps run in the order the honesty rules need (scan before commit, push
only with the token).
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from lastlight.daily import default_evening
from lastlight.geometry import elevation_crossings
from lastlight.privacy import contains_clock_time
from lastlight.site import REFERENCE_POINT

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
SEAL = WORKFLOWS / "seal.yml"

SCHEDULE_GAP_HOURS = 2.0
"""The scheduled run must start at least this long before the earliest sunset of the year."""


@pytest.fixture(scope="module")
def workflow() -> dict[Any, Any]:
    return yaml.safe_load(SEAL.read_text())


@pytest.fixture(scope="module")
def steps(workflow: dict[Any, Any]) -> list[dict[str, Any]]:
    (job,) = workflow["jobs"].values()
    return job["steps"]


def triggers(workflow: dict[Any, Any]) -> dict[str, Any]:
    return workflow["on"] if "on" in workflow else workflow[True]  # YAML 1.1 reads `on` as True


def step_index(steps: list[dict[str, Any]], needle: str) -> int:
    hits = [i for i, step in enumerate(steps) if needle in step.get("run", "")]
    assert len(hits) == 1, f"expected exactly one step running {needle!r}, found {len(hits)}"
    return hits[0]


# --- trigger, queueing, permissions -------------------------------------------------------


def test_it_runs_on_a_daily_schedule_and_on_demand_only(workflow: dict[Any, Any]) -> None:
    on = triggers(workflow)
    assert set(on) == {"schedule", "workflow_dispatch"}
    assert len(on["schedule"]) == 1


def scheduled_time(workflow: dict[Any, Any]) -> tuple[int, int]:
    (entry,) = triggers(workflow)["schedule"]
    minute, hour, day, month, weekday = entry["cron"].split()
    assert (day, month, weekday) == ("*", "*", "*"), "the seal runs every day"
    return int(hour), int(minute)


def test_the_schedule_is_a_fixed_utc_time_that_does_not_depend_on_any_site(
    workflow: dict[Any, Any],
) -> None:
    hour, minute = scheduled_time(workflow)
    assert 0 <= hour <= 23 and 0 <= minute <= 59


def test_the_scheduled_run_always_lands_before_the_evening_it_seals() -> None:
    # Whatever the season, the run at the cron hour belongs to the evening of its own UTC date
    # and starts well before the sun reaches the horizon at the reference point.
    workflow = yaml.safe_load(SEAL.read_text())
    hour, minute = scheduled_time(workflow)
    lat, lon, _elevation = REFERENCE_POINT
    smallest_gap = None
    for offset in range(0, 366, 6):
        date = dt.date(2026, 10, 11) + dt.timedelta(days=offset)
        start = dt.datetime(date.year, date.month, date.day, hour, minute, tzinfo=dt.UTC)
        assert default_evening(start) == date
        gap = elevation_crossings(date, lat, lon)[0.0] - start
        smallest_gap = gap if smallest_gap is None else min(smallest_gap, gap)
    assert smallest_gap is not None
    assert smallest_gap >= dt.timedelta(hours=SCHEDULE_GAP_HOURS)


def test_runs_queue_instead_of_cancelling_each_other(workflow: dict[Any, Any]) -> None:
    assert workflow["concurrency"] == {"group": "seal", "cancel-in-progress": False}


def test_the_workflow_is_named_seal_because_the_site_deploy_waits_on_that_name(
    workflow: dict[Any, Any],
) -> None:
    # The almanac's deploy runs on `workflow_run` of the workflow named `seal`; renaming this
    # one would silently stop the site from ever rebuilding.
    assert workflow["name"] == "seal"


def test_the_token_can_write_contents_and_nothing_else(workflow: dict[Any, Any]) -> None:
    assert workflow["permissions"] == {"contents": "write"}


def test_there_is_one_ubuntu_job_with_a_fifteen_minute_limit(workflow: dict[Any, Any]) -> None:
    assert len(workflow["jobs"]) == 1
    (job,) = workflow["jobs"].values()
    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] == 15


def test_uv_does_not_re_resolve_the_lockfile(workflow: dict[Any, Any]) -> None:
    (job,) = workflow["jobs"].values()
    assert job["env"]["UV_FROZEN"] == "1"


# --- pinned actions -----------------------------------------------------------------------


def test_every_action_is_pinned_to_a_commit_with_a_version_comment() -> None:
    uses = re.findall(r"^\s*-?\s*uses:\s*(\S+)(.*)$", SEAL.read_text(), flags=re.MULTILINE)
    assert uses, "the workflow uses at least checkout and setup-uv"
    for action, rest in uses:
        assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", action), f"not a SHA pin: {action}"
        assert re.search(r"#\s*v\d+\.\d+\.\d+", rest), f"no version comment on {action}"


def test_the_pins_are_the_ones_ci_already_uses() -> None:
    def pins(path: Path) -> dict[str, str]:
        found = re.findall(r"uses:\s*([\w.-]+/[\w.-]+)@([0-9a-f]{40})", path.read_text())
        return dict(found)

    ci, seal = pins(WORKFLOWS / "ci.yml"), pins(SEAL)
    assert seal and set(seal) <= set(ci)
    assert all(seal[name] == ci[name] for name in seal)


# --- the steps ----------------------------------------------------------------------------


def test_the_code_comes_from_main_without_leaving_credentials_behind(
    steps: list[dict[str, Any]],
) -> None:
    checkout = steps[0]
    assert checkout["uses"].startswith("actions/checkout@")
    assert checkout["with"]["ref"] == "main"
    assert checkout["with"]["persist-credentials"] is False


def test_the_steps_run_in_the_order_the_honesty_rules_need(steps: list[dict[str, Any]]) -> None:
    order = [
        step_index(steps, "uv sync --frozen"),
        step_index(steps, "lastlight ledger-branch open"),
        step_index(steps, "lastlight seal-daily"),
        step_index(steps, "lastlight privacy-scan"),
        step_index(steps, "lastlight ledger-branch publish"),
    ]
    assert order == sorted(order) and len(set(order)) == 5


def test_the_ledger_is_written_in_a_worktree_and_the_scan_covers_all_of_it(
    steps: list[dict[str, Any]],
) -> None:
    seal = steps[step_index(steps, "lastlight seal-daily")]["run"]
    scan = steps[step_index(steps, "lastlight privacy-scan")]["run"]
    assert "--ledger-dir ledger-wt/ledger" in seal
    assert "ledger-wt/ledger" in scan and "--root" in scan


def test_the_token_is_given_to_the_publish_step_alone(steps: list[dict[str, Any]]) -> None:
    carrying = [step for step in steps if "GITHUB_TOKEN" in str(step.get("env", {}))]
    assert len(carrying) == 1
    assert "lastlight ledger-branch publish" in carrying[0]["run"]
    assert carrying[0]["env"]["GITHUB_TOKEN"] == "${{ secrets.GITHUB_TOKEN }}"


def test_the_seal_does_not_know_the_site_exists(steps: list[dict[str, Any]]) -> None:
    # The site rebuilds itself when this workflow completes (`workflow_run`), so the seal job
    # neither dispatches it nor holds the permission that would let it.
    text = SEAL.read_text()
    assert "gh workflow" not in text and "pages" not in text.lower()
    assert "actions: write" not in text
    publish = steps[step_index(steps, "lastlight ledger-branch publish")]
    assert publish is steps[-1], "pushing the ledger is the last thing the job does"


def test_no_step_moves_git_history_by_hand_or_forces_anything() -> None:
    text = SEAL.read_text()
    assert "git push" not in text and "--force" not in text


def test_the_workflow_text_spells_no_clock_time() -> None:
    assert not contains_clock_time(SEAL.read_text())
