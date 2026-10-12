"""Renderer v1, the table generator: deterministic, resumable, one node per invocation (#32).

`tools/make_bruneton_table.py` runs the Bruneton precomputation for one (`aod_550`, `ozone_du`)
node per invocation (`--node I`), keeps each node in a cache outside the repository, and writes
the table with `--assemble`. These tests use its tiny grid, which builds in seconds: the full
grid runs only on the build machine.

The container must be byte-identical for the same inputs: no timestamps, and a node order that
does not depend on the order in which the nodes were computed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from lastlight.g0.collect import build_views
from lastlight.render import bruneton, v1

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "make_bruneton_table.py"
REPORT = ROOT / "docs" / "renderer" / "bruneton-convergence.md"


def load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_bruneton_table", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("make_bruneton_table", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    return load_tool()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(tool: ModuleType, cache: Path, out: Path, order: list[int] | None = None) -> str:
    indices = list(range(tool.node_count("tiny"))) if order is None else order
    for index in indices:
        tool.compute_node("tiny", index, cache)
    return tool.assemble("tiny", cache, out)


def test_a_tiny_grid_generated_twice_has_the_same_sha256(tool: ModuleType, tmp_path: Path) -> None:
    first = build(tool, tmp_path / "a" / "cache", tmp_path / "a" / "table.bin")
    second = build(tool, tmp_path / "b" / "cache", tmp_path / "b" / "table.bin")
    assert first == sha256((tmp_path / "a" / "table.bin").read_bytes())
    assert first == second
    assert (tmp_path / "a" / "table.bin").read_bytes() == (
        tmp_path / "b" / "table.bin"
    ).read_bytes()


def test_the_bytes_do_not_depend_on_the_order_the_nodes_were_computed_in(
    tool: ModuleType, tmp_path: Path
) -> None:
    count = tool.node_count("tiny")
    assert count >= 2
    forward = build(tool, tmp_path / "f" / "cache", tmp_path / "f" / "table.bin")
    backward = build(
        tool,
        tmp_path / "r" / "cache",
        tmp_path / "r" / "table.bin",
        order=list(reversed(range(count))),
    )
    assert forward == backward


def test_assemble_refuses_while_a_node_is_missing(tool: ModuleType, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    tool.compute_node("tiny", 0, cache)
    with pytest.raises(tool.MissingNodeError, match="node"):
        tool.assemble("tiny", cache, tmp_path / "table.bin")
    assert not (tmp_path / "table.bin").exists()


def test_the_tiny_table_loads_and_records_the_tiny_resolution(
    tool: ModuleType, tmp_path: Path
) -> None:
    out = tmp_path / "table.bin"
    digest = build(tool, tmp_path / "cache", out)
    table = v1.load_table(out, expected_sha256=digest)
    assert table.metadata["resolution"] == bruneton.TINY_RESOLUTION.as_dict()
    assert table.metadata["orders"] == [2, bruneton.TINY_RESOLUTION.orders]
    assert table.metadata["ground_albedo"] == bruneton.GROUND_ALBEDO
    grid = tool.GRIDS["tiny"]
    assert table.axes["aod_550"].tolist() == list(grid.aod)
    assert table.axes["ozone_du"].tolist() == list(grid.ozone)


def test_on_its_nodes_the_table_holds_the_direct_computation(
    tool: ModuleType, tmp_path: Path
) -> None:
    """The table's value at a node is the precomputation's own ground-observer result."""
    out = tmp_path / "table.bin"
    table = v1.load_table(out, expected_sha256=build(tool, tmp_path / "cache", out))
    index = tool.node_count("tiny") - 1
    atm = tool.node_atmosphere("tiny", index)
    field = bruneton.precompute(atm, resolution=bruneton.TINY_RESOLUTION)
    i = table.axes["aod_550"].tolist().index(atm.aod_550)
    j = table.axes["ozone_du"].tolist().index(atm.ozone_du)
    for k, sun in enumerate(table.axes["sun_elevation_deg"]):
        for n, azimuth in enumerate(table.axes["rel_azimuth_deg"]):
            direct = field.multiple_xyz(
                float(sun), table.axes["view_elevation_deg"], float(azimuth)
            )
            np.testing.assert_allclose(table.values[i, j, k, :, n], direct, rtol=1e-6)


def test_the_command_line_runs_one_node_per_invocation_then_assembles(
    tool: ModuleType, tmp_path: Path
) -> None:
    cache, out = tmp_path / "cache", tmp_path / "table.bin"
    common = [sys.executable, str(TOOL), "--grid", "tiny", "--cache-dir", str(cache)]
    for index in range(tool.node_count("tiny")):
        done = subprocess.run([*common, "--node", str(index)], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
    done = subprocess.run(
        [*common, "--assemble", "--out", str(out)], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    digest = sha256(out.read_bytes())
    assert digest in done.stdout
    v1.load_table(out, expected_sha256=digest)


def test_every_tiny_node_holds_its_own_atmosphere(tool: ModuleType, tmp_path: Path) -> None:
    """Node `index` is the (aod, ozone) pair `(aod[index // n], ozone[index % n])`, `n` ozone nodes.

    The tiny grid has two nodes on each atmosphere axis, so a node index that turned `(i, j)` into
    `(j, i)` agrees with the right one at the first and the last node and differs only at the
    two between. All four are checked: the atmosphere the generator names for each, and that the
    table holds a fresh precomputation of that atmosphere at `[i, j]`.
    """
    grid = tool.GRIDS["tiny"]
    out = tmp_path / "table.bin"
    table = v1.load_table(out, expected_sha256=build(tool, tmp_path / "cache", out))
    assert tool.node_count("tiny") == len(grid.aod) * len(grid.ozone)
    for i, aod in enumerate(grid.aod):
        for j, ozone in enumerate(grid.ozone):
            index = i * len(grid.ozone) + j
            atm = tool.node_atmosphere("tiny", index)
            assert (atm.aod_550, atm.ozone_du) == (aod, ozone), f"node {index}"
            field = bruneton.precompute(atm, resolution=bruneton.TINY_RESOLUTION)
            for k, sun in enumerate(table.axes["sun_elevation_deg"]):
                for n, azimuth in enumerate(table.axes["rel_azimuth_deg"]):
                    direct = field.multiple_xyz(
                        float(sun), table.axes["view_elevation_deg"], float(azimuth)
                    )
                    np.testing.assert_allclose(
                        table.values[i, j, k, :, n], direct, rtol=1e-6, err_msg=f"node {index}"
                    )


def test_a_node_out_of_range_is_refused(tool: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="node"):
        tool.compute_node("tiny", tool.node_count("tiny"), tmp_path)


def test_the_node_cache_lives_outside_the_repository(tool: ModuleType, tmp_path: Path) -> None:
    assert Path.home() / ".cache" / "lastlight" / "bruneton" == tool.DEFAULT_CACHE_DIR
    assert ROOT not in tool.DEFAULT_CACHE_DIR.parents
    # nodes made with another grid or resolution are kept apart, never mixed
    assert tool.node_directory("tiny", tmp_path) != tool.node_directory("full", tmp_path)


def test_the_full_grid_builds_the_committed_table_at_the_tables_resolution(
    tool: ModuleType,
) -> None:
    grid = tool.GRIDS["full"]
    table = v1.default_table()
    assert grid.resolution == bruneton.RESOLUTION
    for name, nodes in (
        ("aod_550", grid.aod),
        ("ozone_du", grid.ozone),
        ("sun_elevation_deg", grid.sun),
        ("view_elevation_deg", grid.view),
        ("rel_azimuth_deg", grid.azimuth),
    ):
        assert table.axes[name].tolist() == list(nodes), name


def test_the_order_check_looks_at_whtm1_s_band_elevations(tool: ModuleType) -> None:
    nominal = build_views()["nominal"].elevations_deg
    assert len(tool.WHTM1_VIEWS) == len(nominal)
    np.testing.assert_allclose(tool.WHTM1_VIEWS, nominal, atol=0.005)


def test_the_convergence_report_is_committed_for_the_committed_table() -> None:
    text = REPORT.read_text()
    assert v1.TABLE_SHA256 in text
    for word in ("order", "doubl", "WHTM1"):
        assert word in text, word
