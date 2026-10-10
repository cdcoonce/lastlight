"""The conftest network guard: every test is offline unless it opts in with `network`."""

from __future__ import annotations

import socket
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def _loopback_listener() -> Iterator[tuple[str, int]]:
    """A listening socket on loopback. bind/listen are not guarded; only connect is."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        yield server.getsockname()


def test_plain_socket_connect_raises() -> None:
    # Something is really listening, so without the guard this connect would succeed.
    with (
        _loopback_listener() as address,
        socket.socket() as client,
        pytest.raises(RuntimeError, match="network") as caught,
    ):
        client.connect(address)
    # Not an OSError: code under test that retries or falls back on OSError must not hide it.
    assert not isinstance(caught.value, OSError)


def test_connect_ex_raises() -> None:
    with (
        _loopback_listener() as address,
        socket.socket() as client,
        pytest.raises(RuntimeError, match="network"),
    ):
        client.connect_ex(address)


def test_create_connection_raises() -> None:
    with _loopback_listener() as address, pytest.raises(RuntimeError, match="network"):
        socket.create_connection(address, timeout=1)


def test_the_guard_message_names_the_opt_in() -> None:
    with socket.socket() as client, pytest.raises(RuntimeError, match=r"pytest\.mark\.network"):
        client.connect(("127.0.0.1", 9))


def _inner_project(tmp_path: Path) -> Path:
    """A throwaway project holding this repo's pytest config and conftest, plus two tests."""
    project = tmp_path / "inner"
    (project / "tests").mkdir(parents=True)
    (project / "pyproject.toml").write_text((REPO_ROOT / "pyproject.toml").read_text())
    (project / "tests" / "conftest.py").write_text(
        (REPO_ROOT / "tests" / "conftest.py").read_text()
    )
    (project / "tests" / "test_inner.py").write_text(
        "import socket\n"
        "import pytest\n"
        "\n"
        "def test_offline():\n"
        "    pass\n"
        "\n"
        "@pytest.mark.network\n"
        "def test_needs_network():\n"
        "    with socket.socket() as server, socket.socket() as client:\n"
        "        server.bind(('127.0.0.1', 0))\n"
        "        server.listen(1)\n"
        "        client.connect(server.getsockname())\n"
    )
    return project


def _run_inner(project: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    # A separate interpreter, because this process already has the guard installed.
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider", *extra],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_network_marked_tests_are_deselected_by_default(tmp_path: Path) -> None:
    result = _run_inner(_inner_project(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "test_offline PASSED" in result.stdout
    assert "test_needs_network" not in result.stdout
    assert "1 deselected" in result.stdout


def test_network_marker_lifts_the_guard_when_selected(tmp_path: Path) -> None:
    result = _run_inner(_inner_project(tmp_path), "-m", "network")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "test_needs_network PASSED" in result.stdout
    assert "test_offline" not in result.stdout


def test_unregistered_markers_are_errors(tmp_path: Path) -> None:
    project = _inner_project(tmp_path)
    (project / "tests" / "test_typo.py").write_text(
        "import pytest\n\n@pytest.mark.netwerk\ndef test_typo():\n    pass\n"
    )
    result = _run_inner(project)
    assert result.returncode != 0
    assert "netwerk" in result.stdout + result.stderr
