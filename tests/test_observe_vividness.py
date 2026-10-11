"""`vividness_v0` over an evening's profile series: a named placeholder, not the frozen rule.

The pre-registered vividness definition arrives in `docs/gates/vividness.md` before the first
scored evening. Until then the observed side needs *a* scalar, and this one says what it is.
"""

from __future__ import annotations

import pytest
from observe_support import BANDS, flat_fit, gradient_frame

from lastlight import observe
from lastlight.observe import vividness_v0
from lastlight.observe.contrast import (
    BandProfile,
    ContrastProfile,
    ExcludedPixels,
    ReferenceProfile,
    contrast_profile,
)


def band(rg: float | None, bg: float | None = 0.0, rows=(8, 28)) -> BandProfile:
    return BandProfile(
        rows=rows,
        log_rg=rg,
        log_bg=bg,
        n_pixels=16000 if rg is not None else 0,
        excluded=ExcludedPixels(),
        view_elevation=None,
        missing=None if rg is not None else "too_few_pixels",
    )


def frame_profile(*rgs: float | None, sun: float = 0.0) -> ContrastProfile:
    """A profile whose bands have these `log_rg` contrasts, lowest band first."""
    bands = tuple(band(rg, rows=(8 + 20 * i, 28 + 20 * i)) for i, rg in enumerate(rgs))
    reference = ReferenceProfile((118, 138), 16000, ExcludedPixels(), None)
    return ContrastProfile(sun, False, bands, reference)


def test_it_is_the_maximum_over_frames_of_the_lowest_bands_log_rg_contrast() -> None:
    series = [frame_profile(0.10), frame_profile(0.30), frame_profile(0.20)]
    assert vividness_v0(series) == pytest.approx(0.30)


def test_only_the_lowest_band_counts_not_a_higher_one() -> None:
    series = [frame_profile(0.10, 0.90), frame_profile(0.05, 0.80)]
    assert vividness_v0(series) == pytest.approx(0.10)


def test_it_reads_log_rg_not_log_bg() -> None:
    low_band = band(0.10, bg=0.70)
    reference = ReferenceProfile((118, 138), 16000, ExcludedPixels(), None)
    assert vividness_v0([ContrastProfile(0.0, False, (low_band,), reference)]) == pytest.approx(
        0.10
    )


def test_a_frame_whose_lowest_band_is_missing_is_skipped_not_counted_as_zero() -> None:
    series = [frame_profile(None, 0.9), frame_profile(-0.20)]
    assert vividness_v0(series) == pytest.approx(-0.20)


def test_the_value_is_not_clamped_at_zero() -> None:
    assert vividness_v0([frame_profile(-0.30), frame_profile(-0.10)]) == pytest.approx(-0.10)


@pytest.mark.parametrize(
    "series", [[], [frame_profile(None)], [frame_profile(None), frame_profile()]]
)
def test_with_no_usable_frame_there_is_no_value(series) -> None:
    assert vividness_v0(series) is None


def test_it_accepts_any_iterable_once() -> None:
    assert vividness_v0(iter([frame_profile(0.2), frame_profile(0.4)])) == pytest.approx(0.4)


def test_on_a_contrast_profile_of_a_frame_it_is_the_lowest_bands_contrast() -> None:
    result = contrast_profile(gradient_frame(), flat_fit(), BANDS, sun_elevation_deg=0.0)
    assert vividness_v0([result]) == pytest.approx(result.bands[0].log_rg)
    assert result.bands[0].log_rg == pytest.approx(0.22, abs=1e-9)


def test_it_is_named_as_a_v0_placeholder_everywhere_it_appears() -> None:
    assert vividness_v0.__name__ == "vividness_v0"
    assert "placeholder" in (vividness_v0.__doc__ or "").lower()
    assert "docs/gates/vividness.md" in (vividness_v0.__doc__ or "")
    assert observe.VIVIDNESS_V0_RULE == "v0-placeholder"


def test_the_forecasts_own_vividness_v0_is_a_different_function_and_still_there() -> None:
    from lastlight import vividness

    assert vividness.vividness_v0 is not vividness_v0
