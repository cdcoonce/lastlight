"""From per-date records to the G0 study: which dates count, the verdict, and its sensitivities.

`build_study` takes the records the run wrote and the period, and decides nothing by hand. The
dates used are those with a rendered curve; every other date of the period is listed with its
reason, and a date with no record at all is `not_collected` (never a silent gap). The verdict is
the pre-registered rule (`lastlight.g0.stats`) over the nominal-scale contrasts of the dates
used. Around it, reported and never changing it: the same rule with the median taken at each end
of the fit's pixel-scale range, the seasonal control (median over the dates within 15 days), the
post hoc split of D by channel (log(R/G) alone, log(B/G) alone) for the whole-period median and
for the seasonal control,
delta-E 2000 of the absolute curve for tier B, how often the channel floor was used, and the five
most extreme dates.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from lastlight.g0 import stats
from lastlight.g0.record import RENDER_SCALES, DateRecord

NOMINAL = "nominal"
SEASONAL = "seasonal"
CHANNEL_NAMES = ("log(R/G)", "log(B/G)")
EXTREME_COUNT = 5


@dataclass(frozen=True)
class ScaleOutcome:
    """The statistic and the rule for one set of contrasts (a pixel scale, or the seasonal control).

    `distances` is D(d, s) for the dates used, shape (dates, 4) in `stats.BANDS` order;
    `at_threshold` counts the counted bands in which each date reaches 0.05; `meets` says which
    dates meet the rule.
    """

    name: str
    baseline: np.ndarray
    distances: np.ndarray
    at_threshold: np.ndarray
    meets: np.ndarray
    verdict: stats.Verdict


@dataclass(frozen=True)
class Extreme:
    """One of the most extreme dates: its mean D over the counted bands and its aerosol."""

    date: dt.date
    mean_d: float
    d_by_band: tuple[float, ...]
    aod_total: float
    aod_dust: float
    aod_smoke: float


@dataclass(frozen=True)
class Study:
    period: tuple[dt.date, dt.date]
    attempted: tuple[dt.date, ...]
    used: tuple[DateRecord, ...]
    dropped: tuple[DateRecord, ...]
    outcomes: dict[str, ScaleOutcome]
    seasonal: ScaleOutcome
    channels: dict[str, ScaleOutcome]
    seasonal_channels: dict[str, ScaleOutcome]
    delta_e: np.ndarray
    floor_share: dict[str, float]
    extremes: tuple[Extreme, ...]
    counted_bands: tuple[str, ...] = stats.COUNTED_BANDS
    bands: tuple[str, ...] = stats.BANDS

    @property
    def verdict(self) -> stats.Verdict:
        """The pre-registered verdict: the rule at the nominal pixel scale."""
        return self.outcomes[NOMINAL].verdict

    @property
    def fraction_of_attempted(self) -> float:
        """The share of all dates attempted that meet the rule (dropped dates cannot)."""
        return self.verdict.meeting / len(self.attempted)

    @property
    def used_dates(self) -> tuple[dt.date, ...]:
        return tuple(record.date for record in self.used)


def _outcome(name: str, contrast: np.ndarray, baseline: np.ndarray) -> ScaleOutcome:
    distances = stats.band_distances(contrast, baseline)
    return ScaleOutcome(
        name=name,
        baseline=baseline,
        distances=distances,
        at_threshold=stats.bands_at_threshold(distances),
        meets=stats.meets_rule(distances),
        verdict=stats.verdict(distances),
    )


def _not_collected(day: dt.date) -> DateRecord:
    return DateRecord(
        date=day,
        drop_reason="not_collected",
        drop_detail="",
        atmosphere=None,
        contrast={},
        floor_channels={},
        curve_lab=None,
        inputs={},
    )


def build_study(records: Sequence[DateRecord], start: dt.date, end: dt.date) -> Study:
    """The study of every date from `start` to `end` inclusive."""
    if end < start:
        raise ValueError(f"the end {end} is before the start {start}")
    by_date = {record.date: record for record in records}
    attempted = tuple(start + dt.timedelta(days=i) for i in range((end - start).days + 1))
    used: list[DateRecord] = []
    dropped: list[DateRecord] = []
    for day in attempted:
        record = by_date.get(day)
        if record is None:
            dropped.append(_not_collected(day))
        elif record.ok:
            used.append(record)
        else:
            dropped.append(record)
    if not used:
        raise ValueError("no dates used: every date of the period was dropped or not collected")

    outcomes: dict[str, ScaleOutcome] = {}
    for scale in RENDER_SCALES:
        contrast = np.stack([record.contrast[scale] for record in used])
        outcomes[scale] = _outcome(scale, contrast, stats.period_median(contrast))
    nominal = np.stack([record.contrast[NOMINAL] for record in used])
    seasonal = _outcome(
        SEASONAL, nominal, stats.windowed_median([record.date for record in used], nominal)
    )
    baseline = stats.period_median(nominal)
    channels = {
        name: _outcome(name, nominal[..., [index]], baseline[..., [index]])
        for index, name in enumerate(CHANNEL_NAMES)
    }
    seasonal_channels = {
        name: _outcome(name, nominal[..., [index]], seasonal.baseline[..., [index]])
        for index, name in enumerate(CHANNEL_NAMES)
    }
    delta_e = stats.delta_e_by_band(np.stack([record.curve_lab for record in used]))  # type: ignore[misc]

    channels_per_date = nominal.shape[1] * (nominal.shape[2] + 1) * 3  # elevations x views x RGB
    floor_share = {
        scale: sum(record.floor_channels[scale] for record in used)
        / (len(used) * channels_per_date)
        for scale in RENDER_SCALES
    }
    return Study(
        period=(start, end),
        attempted=attempted,
        used=tuple(used),
        dropped=tuple(dropped),
        outcomes=outcomes,
        seasonal=seasonal,
        channels=channels,
        seasonal_channels=seasonal_channels,
        delta_e=delta_e,
        floor_share=floor_share,
        extremes=_extremes(used, outcomes[NOMINAL]),
    )


def _extremes(used: list[DateRecord], outcome: ScaleOutcome) -> tuple[Extreme, ...]:
    """The dates with the largest mean D over the counted bands, largest first."""
    columns = [stats.BANDS.index(name) for name in stats.COUNTED_BANDS]
    mean_d = outcome.distances[:, columns].mean(axis=1)
    order = sorted(range(len(used)), key=lambda i: (-mean_d[i], used[i].date))
    chosen = []
    for index in order[:EXTREME_COUNT]:
        atmosphere = used[index].atmosphere
        assert atmosphere is not None
        chosen.append(
            Extreme(
                date=used[index].date,
                mean_d=float(mean_d[index]),
                d_by_band=tuple(float(v) for v in outcome.distances[index]),
                aod_total=atmosphere.aod_total,
                aod_dust=atmosphere.aod_dust,
                aod_smoke=atmosphere.aod_smoke,
            )
        )
    return tuple(chosen)
