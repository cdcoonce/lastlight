"""`contrast_profile`: pixel rows above the frozen skyline to within-frame colour contrasts.

The key guard is the invariance test: a camera's automatic exposure and white balance act, to
first order, as one gain per colour channel on the linear signal, so a contrast (the difference
of two bands' log chromaticity) must not change when the linear channels are multiplied by
constants. The synthetic frames are built so that the test is real: the gains are applied in
linear light, the result is re-encoded with the sRGB transfer function, and nothing saturates.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from camgeom_synthetic import HEIGHT, WIDTH
from observe_support import (
    BAND_CENTRES,
    BANDS,
    BG_SLOPE,
    REFERENCE_CENTRE,
    RG_SLOPE,
    flat_fit,
    gradient_frame,
    gradient_linear,
    grey_frame,
    paint_band,
    srgb_decode,
    srgb_encode,
    to_uint8,
)

from lastlight.cameras import ContrastBands
from lastlight.camgeom.fitfile import Uncertainty
from lastlight.camgeom.mask import skyline_top_rows
from lastlight.camgeom.model import pixels_per_degree
from lastlight.camgeom.orientation import Orientation
from lastlight.camgeom.skyline import Box
from lastlight.observe.contrast import (
    DARK_LEVEL,
    LIGHT_DOME_SUN_ELEVATION_DEG,
    MIN_VALID_PIXELS,
    SATURATION_LEVEL,
    contrast_profile,
    srgb_to_linear,
)

SUN = 0.0  # a sun elevation with no light-dome exclusion
BAND_SIZE = 20 * WIDTH  # pixels in a nominal band: 20 rows by 800 columns

ORANGE = (200.0, 120.0, 60.0)


def profile(frame, fit=None, contrast=BANDS, sun=SUN):
    return contrast_profile(frame, fit or flat_fit(), contrast, sun_elevation_deg=sun)


# --- the thresholds are the specified ones -------------------------------------------------------


def test_the_thresholds_are_the_specified_ones() -> None:
    assert (SATURATION_LEVEL, DARK_LEVEL, MIN_VALID_PIXELS) == (250, 5, 50)
    assert LIGHT_DOME_SUN_ELEVATION_DEG == -4.0


# --- invariance under a gain per channel ---------------------------------------------------------

GAINS = [(1.0, 1.0, 1.0), (2.0, 1.0, 0.5), (0.6, 1.4, 1.9), (1.3, 0.4, 0.9)]


def test_a_contrast_does_not_change_when_each_linear_channel_is_multiplied_by_a_constant() -> None:
    base = profile(gradient_frame(gains=GAINS[0]))
    for gains in GAINS[1:]:
        scaled = profile(gradient_frame(gains=gains))
        for before, after in zip(base.bands, scaled.bands, strict=True):
            assert after.log_rg == pytest.approx(before.log_rg, abs=1e-6), gains
            assert after.log_bg == pytest.approx(before.log_bg, abs=1e-6), gains
            assert after.n_pixels == before.n_pixels == BAND_SIZE


def test_the_invariance_test_is_not_vacuous() -> None:
    # Positive control 1: the contrasts being compared are far from zero and differ by band.
    values = [band.log_rg for band in profile(gradient_frame()).bands]
    assert all(v > 0.05 for v in values) and len(set(values)) == len(values)

    # Positive control 2: the same gains do NOT shift the log chromaticity of the encoded codes by
    # what they shift it by in linear light, so an implementation that took logs of the codes
    # (skipping the linearisation) would change its contrasts under the gains and fail the test
    # above, by a margin far over 1e-6.
    plain = gradient_frame(gains=GAINS[0])[150, 0]
    scaled = gradient_frame(gains=GAINS[1])[150, 0]
    encoded_shift = math.log(scaled[0] / scaled[1]) - math.log(plain[0] / plain[1])
    linear_shift = math.log(2.0 / 1.0)  # what a gain of (2, 1, .5) does to log(R/G) in linear light
    assert abs(encoded_shift - linear_shift) > 0.05


def test_the_invariance_holds_through_8_bit_rounding_to_a_practical_tolerance() -> None:
    # A decoded JPEG has whole codes, so the gains cannot be undone exactly: the rounding of one
    # code is about 1 percent of the signal at these levels. The bound pins "close", not "exact".
    base = profile(to_uint8(gradient_frame(gains=GAINS[0])))
    for gains in GAINS[1:]:
        scaled = profile(to_uint8(gradient_frame(gains=gains)))
        for before, after in zip(base.bands, scaled.bands, strict=True):
            assert abs(after.log_rg - before.log_rg) < 0.03
            assert abs(after.log_bg - before.log_bg) < 0.03


def test_an_input_that_decodes_to_float_codes_and_to_uint8_gives_the_same_profile() -> None:
    frame = gradient_frame()
    from_uint8 = profile(to_uint8(frame))
    for exact, rounded in zip(profile(frame).bands, from_uint8.bands, strict=True):
        assert rounded.log_rg == pytest.approx(exact.log_rg, abs=0.03)


# --- the gradient: sign and magnitude per band ---------------------------------------------------


def test_a_horizon_to_zenith_gradient_gives_the_expected_contrast_in_every_band() -> None:
    result = profile(gradient_frame())
    assert [band.rows for band in result.bands] == list(BANDS.bands_rows)
    for band, centre in zip(result.bands, BAND_CENTRES, strict=True):
        # The median of a linear ramp over an even number of rows is its value at the centre.
        assert band.log_rg == pytest.approx(RG_SLOPE * (centre - REFERENCE_CENTRE), abs=1e-9)
        assert band.log_bg == pytest.approx(BG_SLOPE * (centre - REFERENCE_CENTRE), abs=1e-9)


def test_the_sign_is_warm_low_and_cool_high() -> None:
    result = profile(gradient_frame())
    rg = [band.log_rg for band in result.bands]
    bg = [band.log_bg for band in result.bands]
    assert all(v > 0 for v in rg) and rg == sorted(rg, reverse=True)  # redder nearer the ridge
    assert all(v < 0 for v in bg) and bg == sorted(bg)  # and less blue


def test_the_reference_band_has_no_contrast_with_itself_so_it_is_not_among_the_bands() -> None:
    result = profile(gradient_frame())
    assert result.reference.rows == BANDS.reference_rows
    assert len(result.bands) == len(BANDS.bands_rows)


def test_a_band_that_is_one_row_off_would_be_caught() -> None:
    # Positive control for the half-open convention: shifting every band one row up changes the
    # expected contrast by one slope step (0.002), far above the 1e-9 the test above holds.
    shifted = [RG_SLOPE * (centre + 1 - REFERENCE_CENTRE) for centre in BAND_CENTRES]
    got = [band.log_rg for band in profile(gradient_frame()).bands]
    assert all(abs(g - s) > 1e-3 for g, s in zip(got, shifted, strict=True))


def test_a_flat_neutral_frame_has_zero_contrast_everywhere() -> None:
    for band in profile(grey_frame()).bands:
        assert band.log_rg == pytest.approx(0.0, abs=1e-12)
        assert band.log_bg == pytest.approx(0.0, abs=1e-12)


def test_a_band_value_is_a_median_not_a_mean() -> None:
    # One tenth of the lowest band is far off colour but valid, half of it far too warm and half
    # far too cool. A median ignores both tails; a mean would move by about 0.03.
    clean = profile(gradient_frame())
    frame = paint_band(gradient_frame(), BANDS.bands_rows[0], ORANGE, columns=slice(0, 40))
    frame = paint_band(frame, BANDS.bands_rows[0], ORANGE[::-1], columns=slice(40, 80))
    dirty = profile(frame)
    assert dirty.bands[0].log_rg == pytest.approx(clean.bands[0].log_rg, abs=1e-9)
    assert dirty.bands[0].log_bg == pytest.approx(clean.bands[0].log_bg, abs=1e-9)
    assert dirty.bands[0].n_pixels == BAND_SIZE


def _colour_from_logs(log_r: float, log_g: float, log_b: float) -> np.ndarray:
    """The encoded colour of a pixel whose linear channels are 0.05 * exp(log_r, log_g, log_b)."""
    return srgb_encode(0.05 * np.exp([log_r, log_g, log_b]))


def test_a_band_value_is_the_median_of_each_pixels_ratio_not_the_ratio_of_the_medians() -> None:
    # Green varies from pixel to pixel here (the gradient holds it constant, where the median of
    # the ratios and the ratio of the medians coincide). Band 0 is three groups of columns with
    # these (log R, log G, log B) in linear light, so the per-pixel log(R/G) is -2, +1, +1 and
    # log(B/G) is -1, +2, -1, while the median of each channel alone is 1 in every channel.
    groups = [
        (slice(0, 267), (0.0, 2.0, 1.0)),
        (slice(267, 534), (1.0, 0.0, 2.0)),
        (slice(534, 800), (2.0, 1.0, 0.0)),
    ]
    frame = grey_frame()
    for columns, logs in groups:
        frame = paint_band(frame, BANDS.bands_rows[0], _colour_from_logs(*logs), columns=columns)
    band = profile(frame).bands[0]
    assert band.n_pixels == BAND_SIZE

    # Positive control: the rival statistic (the median of each channel, then the ratio) is 0 for
    # both contrasts on these pixels, far from the median of the ratios, so the test can tell them
    # apart; and so can it tell a mean (about 0) from the median.
    sizes = [(columns.stop - columns.start) * 20 for columns, _ in groups]
    channel = [np.repeat([logs[i] for _, logs in groups], sizes) for i in range(3)]
    assert np.median(channel[0]) - np.median(channel[1]) == pytest.approx(0.0)
    assert np.median(channel[2]) - np.median(channel[1]) == pytest.approx(0.0)

    assert band.log_rg == pytest.approx(1.0, abs=1e-9)
    assert band.log_bg == pytest.approx(-1.0, abs=1e-9)


# --- the skyline is read column by column --------------------------------------------------------


def test_bands_follow_the_skyline_in_each_column() -> None:
    # A skyline that steps between 150 and 190 across the frame. The colour depends only on the
    # offset above that column's own skyline, so every band must read the same ramp.
    top = np.where((np.arange(WIDTH) // 100) % 2 == 0, 150.0, 190.0)
    fit = flat_fit(skyline_y_px=tuple(top))
    d = top[None, :] - np.arange(HEIGHT)[:, None]
    frame = srgb_encode(gradient_linear(d))
    result = profile(frame, fit)
    for band, centre in zip(result.bands, BAND_CENTRES, strict=True):
        assert band.log_rg == pytest.approx(RG_SLOPE * (centre - REFERENCE_CENTRE), abs=1e-9)
        assert band.n_pixels == BAND_SIZE


def test_the_skyline_row_is_rounded_half_up_to_a_whole_row() -> None:
    knots = [150.4 if x < 400 else 150.5 for x in range(WIDTH)]
    top = skyline_top_rows(flat_fit(skyline_y_px=tuple(knots)))
    assert top[0] == 150.0 and top[-1] == 151.0


# --- columns that cannot hold a band are left out and counted ------------------------------------


def test_columns_where_a_band_leaves_the_frame_are_excluded_as_geometry_and_counted() -> None:
    # Skyline at row 40 over the first 100 columns: bands (28, 48) and above leave the top of
    # the frame there (rows 40 - 47 < 0). Band (8, 28) is rows 13 to 32 and stays inside.
    fit = flat_fit(skyline_y_px=tuple(40.0 if x < 100 else 200.0 for x in range(WIDTH)))
    result = profile(gradient_frame(), fit)
    first, second, *_rest = result.bands
    assert first.excluded.geometry == 0 and first.n_pixels == BAND_SIZE
    assert second.excluded.geometry == 100 * 20
    assert second.n_pixels == BAND_SIZE - 100 * 20
    assert second.n_excluded == 100 * 20
    assert result.reference.excluded.geometry == 100 * 20


def test_overlay_box_pixels_are_excluded_as_geometry() -> None:
    fit = flat_fit(overlay_boxes=(Box(700, 0, 800, HEIGHT),))
    result = profile(gradient_frame(), fit)
    assert all(band.excluded.geometry == 100 * 20 for band in result.bands)


# --- saturated and too-dark pixels ---------------------------------------------------------------


def _orange_band0(channel: int | None, value: float, columns=slice(0, 480)) -> np.ndarray:
    """Grey everywhere; band 0 over the first 480 columns (60 percent) is orange, with one
    channel (or none) set to `value`."""
    colour = list(ORANGE)
    if channel is not None:
        colour[channel] = value
    return paint_band(grey_frame(), BANDS.bands_rows[0], colour, columns=columns)


@pytest.mark.parametrize("channel", [0, 1, 2])
def test_a_pixel_with_any_channel_at_250_or_more_is_saturated_and_excluded(channel: int) -> None:
    result = profile(_orange_band0(channel, 250.0))
    band = result.bands[0]
    assert band.excluded.saturated == 480 * 20
    assert band.n_pixels == BAND_SIZE - 480 * 20
    # Only the grey 40 percent is left, so the contrast is the neutral grey's: zero.
    assert band.log_rg == pytest.approx(0.0, abs=1e-12)
    assert band.excluded.too_dark == 0 and band.excluded.geometry == 0


@pytest.mark.parametrize("channel", [0, 1, 2])
def test_a_pixel_with_every_channel_at_249_or_less_is_kept(channel: int) -> None:
    result = profile(_orange_band0(channel, 249.0))
    band = result.bands[0]
    assert band.excluded.saturated == 0
    assert band.n_pixels == BAND_SIZE
    assert abs(band.log_rg) > 0.01 or abs(band.log_bg) > 0.01  # the orange is in the median


def test_a_pixel_with_all_channels_at_5_or_less_is_too_dark_and_excluded() -> None:
    frame = paint_band(grey_frame(), BANDS.bands_rows[0], (5.0, 4.0, 3.0), columns=slice(0, 480))
    band = profile(frame).bands[0]
    assert band.excluded.too_dark == 480 * 20
    assert band.n_pixels == BAND_SIZE - 480 * 20
    assert band.excluded.saturated == 0
    assert band.log_rg == pytest.approx(0.0, abs=1e-12)


def test_a_pixel_with_one_channel_above_5_is_not_too_dark() -> None:
    frame = paint_band(grey_frame(), BANDS.bands_rows[0], (6.0, 5.0, 5.0), columns=slice(0, 480))
    band = profile(frame).bands[0]
    assert band.excluded.too_dark == 0
    assert band.n_pixels == BAND_SIZE
    assert band.log_rg > 0.01  # R is above G in the kept pixels


def test_a_pixel_with_a_zero_channel_cannot_be_logged_and_is_excluded_as_too_dark() -> None:
    frame = paint_band(grey_frame(), BANDS.bands_rows[0], (0.0, 100.0, 50.0), columns=slice(0, 480))
    band = profile(frame).bands[0]
    assert band.excluded.too_dark == 480 * 20
    assert math.isfinite(band.log_rg) and band.log_rg == pytest.approx(0.0, abs=1e-12)


def test_a_pixel_both_saturated_and_with_a_zero_channel_is_charged_to_saturated_only() -> None:
    # The four counts are disjoint: a pixel goes to the first reason that applies, and
    # saturation comes before too dark, so (255, 0, 100) is saturated and not also too dark.
    frame = paint_band(
        grey_frame(), BANDS.bands_rows[0], (255.0, 0.0, 100.0), columns=slice(0, 480)
    )
    band = profile(frame).bands[0]
    assert band.excluded.saturated == 480 * 20
    assert band.excluded.too_dark == 0
    assert band.n_pixels + band.n_excluded == BAND_SIZE


def test_the_reference_band_is_cleaned_the_same_way() -> None:
    frame = paint_band(
        grey_frame(), BANDS.reference_rows, (255.0, 255.0, 255.0), columns=slice(0, 10)
    )
    result = profile(frame)
    assert result.reference.excluded.saturated == 10 * 20
    assert result.reference.n_pixels == BAND_SIZE - 10 * 20


def test_each_exclusion_has_its_own_count_and_together_they_account_for_the_band() -> None:
    # Four disjoint regions of band 0: saturated, too dark, geometry (a skyline that is too
    # high in the last 40 columns, which no row of colour can undo) and the rest valid.
    top = [200.0 if x < 760 else 20.0 for x in range(WIDTH)]
    fit = flat_fit(skyline_y_px=tuple(top))
    frame = grey_frame()
    frame = paint_band(frame, BANDS.bands_rows[0], (255.0, 255.0, 255.0), columns=slice(0, 100))
    frame = paint_band(frame, BANDS.bands_rows[0], (2.0, 2.0, 2.0), columns=slice(100, 300))
    band = profile(frame, fit).bands[0]
    assert band.excluded.saturated == 100 * 20
    assert band.excluded.too_dark == 200 * 20
    assert band.excluded.geometry == 40 * 20
    assert band.excluded.light_dome == 0
    assert band.n_pixels == (760 - 300) * 20
    assert band.n_pixels + band.n_excluded == BAND_SIZE


# --- a band under 50 valid pixels is missing -----------------------------------------------------


def _band0_with_valid_pixels(count: int) -> np.ndarray:
    """Band 0 saturated white except `count` grey pixels along its first row."""
    frame = paint_band(grey_frame(), BANDS.bands_rows[0], (255.0, 255.0, 255.0))
    low = BANDS.bands_rows[0][0]
    frame[200 - low, 0:count] = 120.0
    return frame


def test_a_band_with_49_valid_pixels_is_reported_missing_and_not_filled() -> None:
    band = profile(_band0_with_valid_pixels(49)).bands[0]
    assert band.missing == "too_few_pixels"
    assert band.log_rg is None and band.log_bg is None
    assert band.n_pixels == 49
    assert band.excluded.saturated == BAND_SIZE - 49


def test_a_band_with_50_valid_pixels_is_reported() -> None:
    band = profile(_band0_with_valid_pixels(50)).bands[0]
    assert band.missing is None
    assert band.log_rg == pytest.approx(0.0, abs=1e-12)
    assert band.n_pixels == 50


def test_one_missing_band_leaves_the_others_untouched() -> None:
    result = profile(_band0_with_valid_pixels(3))
    assert result.bands[0].missing == "too_few_pixels"
    assert all(band.missing is None for band in result.bands[1:])


def test_a_missing_reference_makes_every_contrast_missing_and_says_why() -> None:
    frame = paint_band(grey_frame(), BANDS.reference_rows, (255.0, 255.0, 255.0))
    result = profile(frame)
    assert result.reference.missing == "too_few_pixels" and result.reference.n_pixels == 0
    for band in result.bands:
        assert band.log_rg is None and band.log_bg is None
        assert band.missing == "reference_missing"
        assert band.n_pixels == BAND_SIZE  # the band itself is fine, and that is still reported


def test_a_band_that_is_itself_missing_says_so_even_when_the_reference_is_missing_too() -> None:
    frame = paint_band(grey_frame(), BANDS.reference_rows, (255.0, 255.0, 255.0))
    frame = paint_band(frame, BANDS.bands_rows[0], (255.0, 255.0, 255.0))
    assert profile(frame).bands[0].missing == "too_few_pixels"


# --- the light dome ------------------------------------------------------------------------------

DOME = ContrastBands(
    skyline_margin_rows=8,
    bands_rows=BANDS.bands_rows,
    reference_rows=BANDS.reference_rows,
    light_dome_columns=((200, 800),),
)


def _dome_frame() -> np.ndarray:
    """Grey, but the lowest band is orange in columns 200 to 799: three quarters of its pixels."""
    return paint_band(grey_frame(), BANDS.bands_rows[0], ORANGE, columns=slice(200, 800))


def test_below_minus_4_the_light_dome_columns_are_excluded_from_every_band_and_counted() -> None:
    result = profile(_dome_frame(), contrast=DOME, sun=-4.5)
    assert result.light_dome_excluded is True
    for band in result.bands:
        assert band.excluded.light_dome == 600 * 20
        assert band.n_pixels == 200 * 20
    assert result.reference.excluded.light_dome == 600 * 20
    # With the dome gone, band 0 is the grey it is outside it.
    assert result.bands[0].log_rg == pytest.approx(0.0, abs=1e-12)


def test_at_minus_4_and_above_the_light_dome_is_still_in() -> None:
    for sun in (-4.0, -3.99, 0.0, 5.0):
        result = profile(_dome_frame(), contrast=DOME, sun=sun)
        assert result.light_dome_excluded is False, sun
        assert result.bands[0].excluded.light_dome == 0
        assert result.bands[0].n_pixels == BAND_SIZE
        assert result.bands[0].log_rg > 0.01  # the orange is in the median: it is a dome


def test_just_below_minus_4_the_dome_is_out() -> None:
    assert profile(_dome_frame(), contrast=DOME, sun=-4.0001).light_dome_excluded is True


def test_a_camera_with_no_dome_columns_excludes_nothing_at_any_sun_elevation() -> None:
    result = profile(_dome_frame(), contrast=BANDS, sun=-5.5)
    assert result.light_dome_excluded is False
    assert result.bands[0].excluded.light_dome == 0


def test_geometry_wins_over_the_light_dome_for_a_column_that_is_both() -> None:
    # Skyline too high in columns 700 to 799 (inside the dome): counted as geometry, once.
    fit = flat_fit(skyline_y_px=tuple(20.0 if x >= 700 else 200.0 for x in range(WIDTH)))
    band = profile(_dome_frame(), fit, contrast=DOME, sun=-5.0).bands[0]
    assert band.excluded.geometry == 100 * 20
    assert band.excluded.light_dome == 500 * 20
    assert band.n_pixels + band.n_excluded == BAND_SIZE


# --- the nominal view elevation ------------------------------------------------------------------

LENS = Orientation(274.0, 2.0, 0.0, 20.0)
SCALE = pixels_per_degree(20.0, WIDTH)  # about 39.6 pixels per degree


def _lens_fit(uncertainty):
    return flat_fit(top=200.0, orientation=LENS, uncertainty=uncertainty)


def _ridge_elevation() -> float:
    """Elevation of row 200 at the centre column, by the plain pinhole formula."""
    focal = (WIDTH / 2.0) / math.tan(math.radians(20.0) / 2.0)
    return 2.0 + math.degrees(math.atan((149.5 - 200.0) / focal))


def test_each_band_carries_a_nominal_view_elevation_from_the_fits_pixel_scale() -> None:
    uncertainty = Uncertainty((270.0, 278.0), (16.0, 24.0), (30.0, 50.0), "test", False)
    result = profile(gradient_frame(), _lens_fit(uncertainty))
    ridge = _ridge_elevation()
    for band, centre in zip(result.bands, BAND_CENTRES, strict=True):
        assert band.view_elevation is not None
        assert band.view_elevation.nominal_deg == pytest.approx(ridge + centre / SCALE, abs=0.05)


def test_the_range_comes_from_the_fits_pixel_scale_uncertainty() -> None:
    uncertainty = Uncertainty((270.0, 278.0), (16.0, 24.0), (30.0, 50.0), "test", False)
    result = profile(gradient_frame(), _lens_fit(uncertainty))
    ridge = _ridge_elevation()
    for band, centre in zip(result.bands, BAND_CENTRES, strict=True):
        low, high = band.view_elevation.range_deg
        assert low == pytest.approx(ridge + centre / 50.0, abs=0.05)  # the steepest scale
        assert high == pytest.approx(ridge + centre / 30.0, abs=0.05)  # the shallowest
        assert low <= band.view_elevation.nominal_deg <= high


def test_a_wider_scale_range_gives_a_wider_elevation_range() -> None:
    narrow = Uncertainty((270.0, 278.0), (16.0, 24.0), (38.0, 41.0), "test", False)
    wide = Uncertainty((270.0, 278.0), (16.0, 24.0), (27.0, 62.0), "test", True)
    a = profile(gradient_frame(), _lens_fit(narrow)).bands[-1].view_elevation.range_deg
    b = profile(gradient_frame(), _lens_fit(wide)).bands[-1].view_elevation.range_deg
    assert (b[1] - b[0]) > 3 * (a[1] - a[0])


def test_without_an_uncertainty_the_range_is_the_nominal_value_alone() -> None:
    result = profile(gradient_frame(), _lens_fit(None))
    view = result.bands[0].view_elevation
    assert view.range_deg == (view.nominal_deg, view.nominal_deg)


def test_nominal_elevation_rises_with_the_band() -> None:
    result = profile(gradient_frame(), _lens_fit(None))
    nominal = [band.view_elevation.nominal_deg for band in result.bands]
    assert nominal == sorted(nominal) and len(set(nominal)) == len(nominal)


# --- inputs --------------------------------------------------------------------------------------


def test_a_frame_of_the_wrong_size_is_refused_not_resized() -> None:
    with pytest.raises(ValueError, match=r"300x800|800x300"):
        profile(np.zeros((HEIGHT, WIDTH - 1, 3)))


def test_a_frame_that_is_not_three_channel_is_refused() -> None:
    with pytest.raises(ValueError, match="three channels|RGB"):
        profile(np.zeros((HEIGHT, WIDTH)))


@pytest.mark.parametrize("sun", [float("nan"), float("inf")])
def test_a_sun_elevation_that_is_not_a_number_is_refused(sun: float) -> None:
    with pytest.raises(ValueError, match="sun"):
        profile(grey_frame(), sun=sun)


def test_the_linearisation_is_the_srgb_transfer_function() -> None:
    # A pixel pair whose linear ratio is known: encoded codes of linear 0.2, 0.1, 0.4.
    codes = srgb_encode(np.array([0.2, 0.1, 0.4]))
    assert srgb_decode(codes) == pytest.approx([0.2, 0.1, 0.4], rel=1e-12)
    frame = np.broadcast_to(codes, (HEIGHT, WIDTH, 3)).copy()
    # Every band is this one colour, so each contrast is zero; make band 0 differ by exactly
    # a factor of 2 in linear R only, and its log(R/G) contrast must be log 2.
    bumped = srgb_encode(np.array([0.4, 0.1, 0.4]))
    frame = paint_band(frame, BANDS.bands_rows[0], bumped)
    band = profile(frame).bands[0]
    assert band.log_rg == pytest.approx(math.log(2.0), abs=1e-12)
    assert band.log_bg == pytest.approx(0.0, abs=1e-12)


# --- the toe of the sRGB curve --------------------------------------------------------------------
# Channel codes 6 to 10 (the lowest that are kept) lie on the linear segment of the curve, where
# linear light is code / 255 / 12.92; code 11 is the first on the power segment.


@pytest.mark.parametrize("code", [6, 7, 8, 9, 10])
def test_codes_6_to_10_are_on_the_linear_toe_of_the_srgb_curve(code: int) -> None:
    linear = srgb_to_linear(np.array([float(code)]))[0]
    assert linear == pytest.approx(code / 255 / 12.92, rel=1e-12)


def test_code_11_is_the_first_on_the_power_segment_of_the_srgb_curve() -> None:
    linear = srgb_to_linear(np.array([11.0]))[0]
    assert linear == pytest.approx(((11 / 255 + 0.055) / 1.055) ** 2.4, rel=1e-12)
    assert srgb_to_linear(np.array([10.0]))[0] < linear  # still rising across the knee


@pytest.mark.parametrize("code", [6, 7, 8, 9, 10])
def test_a_toe_pixel_is_linearised_with_the_toe_in_the_contrast_it_gives(code: int) -> None:
    # Band 0 has red on the toe and green and blue at code 100; the grey reference has log(R/G) 0,
    # so the contrast is log(red / green) in linear light, with red = code / 255 / 12.92. (A pair
    # of toe channels would hide the slope, which cancels in their ratio.)
    green = ((100 / 255 + 0.055) / 1.055) ** 2.4
    frame = paint_band(grey_frame(), BANDS.bands_rows[0], (float(code), 100.0, 100.0))
    band = profile(frame).bands[0]
    assert band.n_pixels == BAND_SIZE and band.excluded.too_dark == 0
    assert band.log_rg == pytest.approx(math.log(code / 255 / 12.92 / green), abs=1e-9)
    assert band.log_bg == pytest.approx(0.0, abs=1e-12)
