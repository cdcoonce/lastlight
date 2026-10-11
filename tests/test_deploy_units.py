"""The systemd user units and the owner's runbook (docs/runbooks/collector.md)."""

from __future__ import annotations

import configparser
import re
import shlex
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
UNIT_DIR = REPO_ROOT / "deploy" / "systemd"
RUNBOOK = REPO_ROOT / "docs" / "runbooks" / "collector.md"


def parse_unit(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None, strict=False, delimiters=("=",))
    parser.optionxform = str  # type: ignore[assignment,method-assign]  # systemd keys are case-sensitive
    parser.read_string(path.read_text())
    return parser


@pytest.fixture(scope="module")
def service() -> configparser.ConfigParser:
    return parse_unit(UNIT_DIR / "lastlight-collect.service")


@pytest.fixture(scope="module")
def timer() -> configparser.ConfigParser:
    return parse_unit(UNIT_DIR / "lastlight-collect.timer")


def test_the_timer_fires_daily_at_the_fixed_afternoon_time(
    timer: configparser.ConfigParser,
) -> None:
    match = re.fullmatch(r"\*-\*-\* (\d\d):(\d\d):(\d\d)", timer["Timer"]["OnCalendar"])
    assert match is not None
    assert tuple(int(group) for group in match.groups()) == (15, 0, 0)


def test_the_timer_catches_up_if_the_host_was_down_and_installs_under_timers_target(
    timer: configparser.ConfigParser,
) -> None:
    assert timer["Timer"]["Persistent"] == "true"
    assert timer["Install"]["WantedBy"] == "timers.target"


def test_the_timer_starts_the_collect_service(timer: configparser.ConfigParser) -> None:
    # With no Unit= the timer starts the service of the same name, lastlight-collect.service.
    assert timer["Timer"].get("Unit", "lastlight-collect.service") == "lastlight-collect.service"
    assert (UNIT_DIR / "lastlight-collect.service").is_file()


def test_the_service_runs_the_collector_from_the_projects_virtualenv(
    service: configparser.ConfigParser,
) -> None:
    command = shlex.split(service["Service"]["ExecStart"])
    assert Path(command[0]).name == "lastlight"
    assert ".venv/bin" in command[0]
    assert command[1:] == ["collect"]


def test_the_service_is_one_long_run_that_restarts_only_on_a_crash(
    service: configparser.ConfigParser,
) -> None:
    section = service["Service"]
    assert section["Type"] == "simple"
    assert section["Restart"] == "on-failure"
    assert section["RestartPreventExitStatus"] == "2"  # a refusal is configuration, not a crash


def test_the_service_is_a_user_unit(service: configparser.ConfigParser) -> None:
    assert "User" not in service["Service"]  # user units cannot set User=
    assert "Install" not in service  # the timer starts it; it is never enabled on its own


def test_the_runbook_has_the_owners_install_check_and_stop_steps() -> None:
    text = RUNBOOK.read_text()
    for needle in (
        "git clone",
        "uv sync",
        "deploy/systemd/lastlight-collect.service",
        "deploy/systemd/lastlight-collect.timer",
        "~/.config/systemd/user",
        "systemctl --user daemon-reload",
        "systemctl --user enable --now lastlight-collect.timer",
        "loginctl enable-linger",
        "systemctl --user list-timers",
        "journalctl --user -u lastlight-collect",
        "systemctl --user stop lastlight-collect.service",
        "systemctl --user disable --now lastlight-collect.timer",
        "LASTLIGHT_ARCHIVE",
    ):
        assert needle in text, f"the runbook does not mention {needle!r}"


def test_the_runbook_says_agents_never_install_the_timer() -> None:
    assert re.search(r"agents (?:never|do not) install", RUNBOOK.read_text(), re.IGNORECASE)
