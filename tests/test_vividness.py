"""The v0 vividness rule: a documented placeholder with a deliberately wide spread (issue #8).

The pre-registered definition of vividness is frozen in `docs/gates/vividness.md` before the first
scored evening. Until then the forecast needs *a* distribution, and honesty rule 3 says it must
not pretend to know more than it does: the rule is named v0 everywhere it appears and its spread
covers most of the scale, so the evenings it scores say little either way.
"""

from __future__ import annotations

import pytest

from lastlight.forecast import Vividness
from lastlight.vividness import RULE_SUMMARY, VIVIDNESS_RULE, VIVIDNESS_SD, vividness_v0


def test_the_rule_is_named_v0_and_a_placeholder() -> None:
    assert VIVIDNESS_RULE == "v0-placeholder"
    assert "v0" in RULE_SUMMARY and "placeholder" in RULE_SUMMARY


def test_the_spread_is_wide_a_quarter_of_the_whole_scale() -> None:
    assert VIVIDNESS_SD == 0.25
    assert vividness_v0(low_cloud_pct=0.0, aod_550=0.06).sd == VIVIDNESS_SD


def test_the_result_is_a_forecast_vividness() -> None:
    assert isinstance(vividness_v0(low_cloud_pct=10.0, aod_550=0.1), Vividness)


def test_clear_clean_air_sits_just_under_the_middle_of_the_scale() -> None:
    assert vividness_v0(0.0, 0.06).mean == pytest.approx(0.45 + 0.25 * 0.06 / 0.4)


def test_a_low_cloud_deck_lowers_it_by_up_to_three_tenths() -> None:
    clear = vividness_v0(0.0, 0.06).mean
    overcast = vividness_v0(100.0, 0.06).mean
    assert clear - overcast == pytest.approx(0.30)
    assert vividness_v0(50.0, 0.06).mean == pytest.approx((clear + overcast) / 2)


def test_aerosol_raises_it_up_to_a_cap_of_a_quarter() -> None:
    low = vividness_v0(0.0, 0.0).mean
    assert vividness_v0(0.0, 0.2).mean - low == pytest.approx(0.125)
    assert vividness_v0(0.0, 0.4).mean - low == pytest.approx(0.25)
    assert vividness_v0(0.0, 3.0).mean == vividness_v0(0.0, 0.4).mean


@pytest.mark.parametrize("low_cloud_pct", [0.0, 25.0, 100.0])
@pytest.mark.parametrize("aod_550", [0.0, 0.06, 0.4, 5.0])
def test_the_mean_stays_on_the_zero_to_one_scale(low_cloud_pct: float, aod_550: float) -> None:
    assert 0.0 <= vividness_v0(low_cloud_pct, aod_550).mean <= 1.0


@pytest.mark.parametrize("low_cloud_pct", [-0.1, 100.1, float("nan"), float("inf")])
def test_cloud_cover_must_be_a_percentage(low_cloud_pct: float) -> None:
    with pytest.raises(ValueError, match="low_cloud_pct"):
        vividness_v0(low_cloud_pct, 0.06)


@pytest.mark.parametrize("aod_550", [-0.01, float("nan"), float("inf")])
def test_aod_must_be_a_finite_non_negative_number(aod_550: float) -> None:
    with pytest.raises(ValueError, match="aod_550"):
        vividness_v0(0.0, aod_550)
