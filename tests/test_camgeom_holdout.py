"""The held-out check: a frozen skyline laid over frames the fit was not made from."""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np
import pytest
from camgeom_synthetic import (
    HEIGHT,
    SKYLINE_COLUMNS,
    TRUE_MODEL,
    WIDTH,
    example_fit,
    paint_cloud_base,
    render_frame,
    true_skyline_y,
)
from PIL import Image

from lastlight.camgeom.fitfile import CameraFit, FrameRecord
from lastlight.camgeom.frames import frame_sha256
from lastlight.camgeom.holdout import (
    DEFAULT_BAND_ROWS,
    frame_label,
    held_out_paths,
    holdout_paths,
    holdout_rms,
)
from lastlight.camgeom.orientation import Orientation
from lastlight.camgeom.skyline import Box, default_overlay_boxes

OVERLAY = default_overlay_boxes(WIDTH, HEIGHT)
STAMP = "%Y%m%dT%H%M%SZ"
START = dt.datetime(2026, 10, 11, 0, 0, tzinfo=dt.UTC)
TRUTH_ROWS = true_skyline_y(TRUE_MODEL)
O = TRUE_MODEL  # noqa: E741


def frozen_fit(**changes) -> CameraFit:
    """A fit whose stored skyline is the true skyline of the synthetic camera."""
    rows = TRUTH_ROWS[np.array(SKYLINE_COLUMNS, dtype=int)]
    return example_fit(
        orientation=Orientation(O.azimuth_deg, O.elevation_deg, O.roll_deg, O.hfov_deg),
        skyline_y_px=tuple(float(y) for y in rows),
        **changes,
    )


def frame(seed: int = 40, **kwargs) -> np.ndarray:
    return render_frame(TRUE_MODEL, seed=seed, stamp=OVERLAY[0], **kwargs)


def test_a_frame_that_shows_the_frozen_skyline_matches_it_to_a_fraction_of_a_pixel() -> None:
    result = holdout_rms(frozen_fit(), frame(), "abcd1234")
    assert result.rms_px < 0.3
    assert result.n_columns > 700
    assert result.label == "abcd1234"
    focal = (WIDTH / 2.0) / math.tan(math.radians(O.hfov_deg) / 2.0)
    assert result.rms_deg == pytest.approx(result.rms_px / (focal * math.pi / 180.0), rel=1e-9)


def test_a_frame_whose_skyline_is_lower_by_three_rows_reads_three_pixels() -> None:
    lowered = np.roll(frame(), 3, axis=0)
    result = holdout_rms(frozen_fit(), lowered, "shifted0")
    assert result.rms_px == pytest.approx(3.0, abs=0.3)
    assert result.median_abs_px == pytest.approx(3.0, abs=0.3)


def test_the_band_decides_whether_a_cloud_edge_is_followed() -> None:
    # A cloud base 20 rows above the ridge over 200 columns. A band of 12 rows cannot see it; a
    # band of 30 can, and then the traced skyline is the cloud's, 20 rows from the frozen one.
    cloudy = paint_cloud_base(
        frame(), TRUTH_ROWS, offset_start=20, offset_end=20, x0=250, x1=450, shade_below=60.0
    )
    narrow = holdout_rms(frozen_fit(), cloudy, "narrow00", band_rows=12.0)
    wide = holdout_rms(frozen_fit(), cloudy, "wide0000", band_rows=30.0)
    assert narrow.rms_px < 0.5
    assert wide.rms_px > 5.0


def test_the_default_band_is_the_width_of_the_fits_own_last_pass() -> None:
    assert DEFAULT_BAND_ROWS == 12.0
    cloudy = paint_cloud_base(
        frame(), TRUTH_ROWS, offset_start=20, offset_end=20, x0=250, x1=450, shade_below=60.0
    )
    assert holdout_rms(frozen_fit(), cloudy, "default0").rms_px < 0.5


def test_the_boxes_the_fit_excludes_are_left_out_of_the_check_too() -> None:
    # Over 40 columns where the ridge is level, a bright strip 15 to 20 rows above it has a
    # bottom edge stronger than the ridge's own. With the strip's box excluded, as the fit's
    # timestamp box is, the trace ignores it; without the exclusion it follows the strip.
    first = min(range(60, 700), key=lambda c: np.ptp(TRUTH_ROWS[c : c + 40]))
    span = range(first, first + 40)
    box = Box(
        first,
        int(min(TRUTH_ROWS[c] for c in span)) - 21,
        first + 40,
        int(max(TRUTH_ROWS[c] for c in span)) - 10,
    )
    image = frame().astype(float)
    for column in span:
        row = int(round(TRUTH_ROWS[column]))
        image[row - 20 : row - 15, column] = 255.0
        image[row - 15 : row - 10, column] = 60.0
    image = image.astype(np.uint8)

    excluded = holdout_rms(frozen_fit(overlay_boxes=(box,)), image, "boxed000", band_rows=20.0)
    unexcluded = holdout_rms(frozen_fit(overlay_boxes=()), image, "plain000", band_rows=20.0)
    assert unexcluded.rms_px > 2.0, "the strip must bite when its box is not excluded"
    assert excluded.rms_px < 0.4


def test_a_frame_of_another_size_cannot_be_checked_against_the_fit() -> None:
    small = np.zeros((100, 200, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="800x300"):
        holdout_rms(frozen_fit(), small, "tiny0000")


@pytest.mark.parametrize("band", [0.0, -3.0, math.nan, math.inf])
def test_a_band_must_be_a_positive_number_of_rows(band: float) -> None:
    with pytest.raises(ValueError, match="band"):
        holdout_rms(frozen_fit(), frame(), "band0000", band_rows=band)


# --- which frames are held out -------------------------------------------------------------------


def named(minutes: int) -> str:
    """A collector-style frame name for a moment `minutes` after the start of the evening."""
    return f"{(START + dt.timedelta(minutes=minutes)).strftime(STAMP)}.jpg"


def save(path: Path, seed: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame(seed)).save(path, quality=92)
    return path


def test_frames_the_fit_was_frozen_on_are_left_out_and_the_rest_are_held_out(
    tmp_path: Path,
) -> None:
    used = save(tmp_path / "a" / named(10), 1)
    fresh_one = save(tmp_path / "a" / named(20), 2)
    fresh_two = save(tmp_path / "b" / named(30), 3)
    fit = frozen_fit(frames_used=(FrameRecord("2026-10-10", 8.0, frame_sha256(used)),))
    kept, skipped = held_out_paths(fit, [tmp_path / "a", tmp_path / "b"])
    assert sorted(kept) == sorted([fresh_one, fresh_two])
    assert skipped == 1


def test_a_frame_found_under_two_roots_is_counted_once(tmp_path: Path) -> None:
    one = save(tmp_path / "a" / named(10), 1)
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "copy.jpg").write_bytes(one.read_bytes())
    kept, skipped = held_out_paths(frozen_fit(), [tmp_path / "a", tmp_path / "b"])
    assert len(kept) == 1 and skipped == 0


def test_results_are_labelled_by_the_start_of_the_hash_and_never_by_name(tmp_path: Path) -> None:
    path = save(tmp_path / named(10), 5)
    (result,) = holdout_paths(frozen_fit(), [path])
    assert result.label == frame_label(frame_sha256(path)) == frame_sha256(path)[:8]
    assert "2026" not in result.label and ".jpg" not in result.label
