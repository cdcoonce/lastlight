"""The position scan: does the skyline say where in the described area the camera stands?"""

from __future__ import annotations

import pytest

from lastlight.camgeom.calibration import PositionTrial, scan_offsets, summarize_scan
from lastlight.camgeom.orientation import DEFAULT_HFOV_RANGE_DEG, Orientation


def trial(east: float, north: float, rms: float, az: float, hfov: float) -> PositionTrial:
    return PositionTrial(east, north, rms, Orientation(az, 2.0, 0.3, hfov))


def test_the_scan_grid_is_every_multiple_of_the_step_inside_the_box() -> None:
    offsets = scan_offsets(6.0, 2.0)
    assert len(offsets) == 49
    assert (0.0, 0.0) in offsets
    assert {east for east, _ in offsets} == {-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0}
    assert max(abs(v) for pair in offsets for v in pair) == 6.0


def test_a_box_smaller_than_the_step_scans_only_the_site_itself() -> None:
    assert scan_offsets(1.0, 2.0) == [(0.0, 0.0)]
    assert scan_offsets(0.0, 2.0) == [(0.0, 0.0)]


@pytest.mark.parametrize(("box", "step"), [(-1.0, 2.0), (4.0, 0.0), (4.0, -1.0)])
def test_a_nonsense_scan_box_or_step_is_refused(box: float, step: float) -> None:
    with pytest.raises(ValueError):
        scan_offsets(box, step)


def test_the_summary_reports_the_spread_of_the_residual_and_of_the_orientation() -> None:
    trials = [
        trial(0.0, 0.0, 3.6, 277.4, 72.5),
        trial(2.0, 0.0, 3.3, 288.1, 64.0),
        trial(-2.0, 0.0, 3.9, 262.0, 90.0),
    ]
    summary = summarize_scan(trials, box_km=2.0, step_km=2.0, site_rms_px=3.6)
    assert summary["box_km"] == 2.0 and summary["step_km"] == 2.0
    assert summary["positions"] == 3
    assert summary["rms_px_at_site"] == 3.6
    assert (summary["rms_px_min"], summary["rms_px_max"]) == (3.3, 3.9)
    assert summary["azimuth_deg_range"] == [262.0, 288.1]
    assert summary["hfov_deg_range"] == [64.0, 90.0]
    assert summary["adopted"] is False
    assert "held at the described site" in summary["reason"]


def test_the_summary_flags_fits_pinned_to_the_field_of_view_limits() -> None:
    low, high = DEFAULT_HFOV_RANGE_DEG
    trials = [
        trial(0.0, 0.0, 3.6, 277.4, 72.5),
        trial(2.0, 0.0, 1.9, 261.0, low),  # the lowest residual sits on the limit
        trial(-2.0, 0.0, 3.9, 262.0, high),
    ]
    summary = summarize_scan(trials, box_km=2.0, step_km=2.0, site_rms_px=3.6)
    assert summary["positions_at_hfov_bound"] == 2
    assert summary["lowest_rms_at_hfov_bound"] is True
    free = summarize_scan(trials[:1] + trials[2:], 2.0, 2.0, 3.6)
    assert free["lowest_rms_at_hfov_bound"] is False, "the 3.6 trial is the lowest of these two"


def test_the_summary_names_no_position_only_offsets_statistics() -> None:
    summary = summarize_scan([trial(0.0, 0.0, 3.0, 270.0, 70.0)], 0.0, 2.0, 3.0)
    assert set(summary) == {
        "box_km",
        "step_km",
        "positions",
        "rms_px_at_site",
        "rms_px_min",
        "rms_px_max",
        "azimuth_deg_range",
        "hfov_deg_range",
        "positions_at_hfov_bound",
        "lowest_rms_at_hfov_bound",
        "adopted",
        "reason",
    }


def test_a_scan_with_no_usable_position_is_an_error() -> None:
    with pytest.raises(ValueError, match="no position"):
        summarize_scan([], 2.0, 2.0, 3.0)
