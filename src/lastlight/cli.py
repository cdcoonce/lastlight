"""The `lastlight` command line: a registry of subcommands that later slices extend.

A subcommand is a `Command`. `main` builds the argparse parser from the registry, so the
parser holds no command-specific code. The real registry is filled by `discover`, which
imports every module of `lastlight.commands`; tests pass their own list of commands.
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from lastlight import __version__


def _takes_no_arguments(parser: argparse.ArgumentParser) -> None:
    """Default `configure`: the subcommand has no arguments of its own."""


@dataclass(frozen=True)
class Command:
    """One subcommand: `configure` adds its arguments, `run` returns the process exit code."""

    name: str
    help: str
    run: Callable[[argparse.Namespace], int]
    configure: Callable[[argparse.ArgumentParser], None] = _takes_no_arguments


def build_registry(commands: Iterable[Command]) -> dict[str, Command]:
    """Index commands by name, refusing duplicates."""
    registry: dict[str, Command] = {}
    for command in commands:
        if command.name in registry:
            raise ValueError(f"duplicate command name: {command.name!r}")
        registry[command.name] = command
    return registry


def discover(package: str = "lastlight.commands") -> list[Command]:
    """Collect `COMMAND` from every public module of `package`, in name order."""
    root = importlib.import_module(package)
    modules = sorted(pkgutil.iter_modules(root.__path__), key=lambda info: info.name)
    found: list[Command] = []
    for info in modules:
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{package}.{info.name}")
        command = getattr(module, "COMMAND", None)
        if not isinstance(command, Command):
            raise ValueError(f"{module.__name__} must define a module-level COMMAND = Command(...)")
        found.append(command)
    return found


def build_parser(registry: dict[str, Command]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lastlight",
        description="Paint tonight's Phoenix twilight, seal the forecast, score it after dusk.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.set_defaults(handler=None)
    subparsers = parser.add_subparsers(title="commands", metavar="<command>")
    for command in registry.values():
        subparser = subparsers.add_parser(command.name, help=command.help, description=command.help)
        command.configure(subparser)
        subparser.set_defaults(handler=command.run)
    return parser


def main(argv: Sequence[str] | None = None, *, commands: Iterable[Command] | None = None) -> int:
    """Run the CLI and return the exit code; never raises SystemExit itself."""
    registry = build_registry(discover() if commands is None else commands)
    parser = build_parser(registry)
    try:
        args = parser.parse_args(argv)
    except SystemExit as stop:  # argparse exits for --version, --help and usage errors
        return int(stop.code or 0)
    if args.handler is None:
        parser.print_help(sys.stderr)
        return 2
    return args.handler(args)
