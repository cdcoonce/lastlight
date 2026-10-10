"""The `lastlight` command line: version, exit codes and the subcommand registry."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import sysconfig
import textwrap
import tomllib
from pathlib import Path

import pytest

import lastlight
from lastlight.cli import Command, build_registry, discover, main

REPO_ROOT = Path(__file__).resolve().parents[1]


def _echo(args: argparse.Namespace) -> int:
    print(f"echo {args.word}")
    return 3


def _configure_echo(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("word")


ECHO = Command(name="echo", help="print a word", run=_echo, configure=_configure_echo)


# --- version -----------------------------------------------------------------


def test_package_exposes_version() -> None:
    assert lastlight.__version__ == "0.1.0"


def test_package_version_matches_pyproject() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert pyproject["project"]["version"] == lastlight.__version__


def test_version_flag_prints_name_and_version_and_returns_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--version"]) == 0
    assert capsys.readouterr().out == "lastlight 0.1.0\n"


def test_module_entry_point_prints_version_and_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "lastlight", "--version"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    assert result.stdout == "lastlight 0.1.0\n"


def test_console_script_prints_version_and_exits_zero() -> None:
    script = shutil.which("lastlight", path=sysconfig.get_path("scripts"))
    assert script is not None, "the `lastlight` console script is not installed in this env"
    result = subprocess.run([script, "--version"], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0
    assert result.stdout == "lastlight 0.1.0\n"


# --- subcommand registry -----------------------------------------------------


def test_registered_command_runs_with_parsed_arguments_and_its_exit_code(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["echo", "hello"], commands=[ECHO]) == 3
    assert capsys.readouterr().out == "echo hello\n"


def test_command_without_configure_takes_no_arguments() -> None:
    quiet = Command(name="quiet", help="do nothing", run=lambda args: 0)
    assert main(["quiet"], commands=[quiet]) == 0


def test_duplicate_command_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="echo"):
        build_registry([ECHO, ECHO])


def test_missing_subcommand_is_a_usage_error_that_lists_the_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([], commands=[ECHO]) == 2
    assert "echo" in capsys.readouterr().err


def test_unknown_subcommand_is_a_usage_error() -> None:
    assert main(["nope"], commands=[ECHO]) == 2


def test_help_lists_registered_commands_and_returns_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--help"], commands=[ECHO]) == 0
    out = capsys.readouterr().out
    assert "echo" in out
    assert "print a word" in out


def test_discover_collects_COMMAND_from_each_module_of_a_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "fake_commands"
    package.mkdir()
    (package / "__init__.py").write_text("")
    for name in ("alpha", "beta"):
        (package / f"{name}.py").write_text(
            textwrap.dedent(
                f"""
                from lastlight.cli import Command

                COMMAND = Command(name="{name}", help="the {name} command", run=lambda args: 0)
                """
            )
        )
    (package / "_private.py").write_text("raise RuntimeError('private modules are skipped')")
    monkeypatch.syspath_prepend(str(tmp_path))

    found = discover("fake_commands")

    assert sorted(command.name for command in found) == ["alpha", "beta"]


def test_discover_rejects_a_module_that_defines_no_COMMAND(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "broken_commands"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "oops.py").write_text("COMMANDS = []  # typo for COMMAND\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ValueError, match="oops"):
        discover("broken_commands")


def test_main_loads_its_registry_from_discover_when_no_commands_are_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # `main()` with no `commands=` is how the console script runs. Point `discover` at a
    # temp package so the whole path (discover -> registry -> parser -> handler) is
    # exercised without a real subcommand existing yet.
    package = tmp_path / "wired_commands"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "ping.py").write_text(
        textwrap.dedent(
            """
            from lastlight.cli import Command


            def _run(args):
                print("pong")
                return 4


            COMMAND = Command(name="ping", help="answer a ping", run=_run)
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    real_discover = discover
    monkeypatch.setattr("lastlight.cli.discover", lambda: real_discover("wired_commands"))

    assert main(["ping"]) == 4
    assert capsys.readouterr().out == "pong\n"
    assert main(["--help"]) == 0
    assert "answer a ping" in capsys.readouterr().out


def test_explicit_commands_replace_discovery_even_when_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail() -> list[Command]:
        raise AssertionError("discover() must not run when commands= is given")

    monkeypatch.setattr("lastlight.cli.discover", _fail)

    assert main(["echo", "hi"], commands=[ECHO]) == 3
    assert main(["echo", "hi"], commands=[]) == 2


def test_the_real_commands_package_is_discoverable() -> None:
    # No slice has added a subcommand yet; later slices drop a module with COMMAND into it.
    assert all(isinstance(command, Command) for command in discover())
