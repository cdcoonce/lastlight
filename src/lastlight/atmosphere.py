"""The v0 reduction: HRRR samples along the sightline in, the renderer's parameters out.

`reduce_transect` is the one function that decides what the renderer is told about an evening's
air. It takes the samples `lastlight.inputs.hrrr.sample_transect` returns (any iterable of
objects with `point_index`, `field` and `value` attributes will do) and the evening's date, and
returns an `AtmosphereV0`: the `lastlight.render.Atmosphere` to paint, the pieces it was built
from, a cloud summary and the sealed `notes` text that says what was and was not used. The G0
spread study (#10) calls it once per archived date, so it takes no clock, no network and no
files: the same samples and date always give the same answer.

What v0 does with each input:

* **Aerosol optical depth at 550 nm** = the mean of HRRR's `AOTK` over the transect points plus
  a fixed background of `BACKGROUND_AOD` (0.06). HRRR's `AOTK` is smoke only (it is zero unless
  the smoke run is active), so the background stands in for the dust and other aerosol the
  model does not carry. The mean is unweighted: every transect point counts the same.
* **Ozone** = the monthly climatology for Phoenix's latitude below (`OZONE_DU_BY_MONTH`),
  looked up by the evening's month. HRRR carries no ozone.
* **Precipitable water** (`PWAT`, kg m-2) is averaged and recorded. The renderer has no water
  vapour term, so it is not passed to it.
* **Cloud** (`LCDC`, `MCDC`, `HCDC`, `TCDC`, `HGT_CLOUD_BASE`, `HGT_CLOUD_TOP`) is summarised in
  the notes and not rendered: v0 paints the clear-sky-plus-aerosol sky. The low cloud mean also
  feeds the placeholder vividness rule (`lastlight.vividness`).

Everything else about the atmosphere (pressure, the aerosol's spectral slope, scale height,
phase function) stays at the renderer's own defaults.

The ozone table. `OZONE_DU_BY_MONTH` is the mean total column ozone in Dobson units for the
5 degree band 30 to 35 N (Phoenix, 33.45 N, lies inside it), by calendar month, over 2005 to
2024. Source: NASA Goddard's merged SBUV total ozone record (MOD v8.7), monthly zonal means,
file `sbuv.v87.mod_v14.70-25.za.txt` at
https://acd-ext.gsfc.nasa.gov/Data_services/merged/data/ (NASA data, a US Government work,
public domain in the United States). Months with no data (0.0 in the file; only August 2022 in
this band and period) are left out of the mean; each value is rounded to 0.1 DU. The 20 source
rows are committed in `tests/fixtures/ozone/`, a test recomputes the table from them, and
`tools/make_ozone_climatology.py` rebuilds both from the NASA file.
"""

from __future__ import annotations

import datetime as dt
import math
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from lastlight.render.atmosphere import Atmosphere

if TYPE_CHECKING:  # the type only: importing hrrr would load the GRIB library
    from lastlight.inputs.hrrr import TransectSample

BACKGROUND_AOD: Final = 0.06
"""Aerosol optical depth at 550 nm added to HRRR's smoke: dust and other aerosol not in HRRR."""

OZONE_TABLE_PERIOD: Final = (2005, 2024)
"""First and last year of the source rows behind `OZONE_DU_BY_MONTH`."""

OZONE_DU_BY_MONTH: Final[tuple[float, ...]] = (
    286.6,  # January
    295.0,  # February
    307.2,  # March
    317.5,  # April
    315.0,  # May
    303.2,  # June
    295.6,  # July
    289.7,  # August
    282.4,  # September
    272.4,  # October
    270.5,  # November
    276.9,  # December
)
"""Mean total column ozone (Dobson units) at 30 to 35 N by calendar month; see the module text."""

_MONTH_NAMES: Final = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


class ReductionError(ValueError):
    """The samples do not hold what the reduction needs (a field missing or empty)."""


def ozone_du(month: int) -> float:
    """The climatological ozone column (Dobson units) for a calendar month, 1 to 12."""
    if isinstance(month, bool) or not isinstance(month, int) or not 1 <= month <= 12:
        raise ValueError(f"month must be a whole number 1 to 12, got {month!r}")
    return OZONE_DU_BY_MONTH[month - 1]


@dataclass(frozen=True)
class CloudSummary:
    """Cloud along the transect, recorded for the notes and not rendered in v0.

    The cover figures are percentages averaged over the transect points. `base_m` and `top_m`
    are medians of HRRR's cloud base and top over the `cloudy_points` points where the model
    has a cloud (geopotential metres as HRRR labels them), or None where it has none anywhere.
    A cover figure is None only if the field is absent from the samples.
    """

    points: int
    low_pct: float
    mid_pct: float | None
    high_pct: float | None
    total_pct: float | None
    base_m: float | None
    top_m: float | None
    cloudy_points: int


@dataclass(frozen=True)
class AtmosphereV0:
    """What the v0 reduction decided: the atmosphere to paint and the pieces behind it."""

    atmosphere: Atmosphere
    aod_smoke: float
    ozone_du: float
    pwat_kg_m2: float | None
    clouds: CloudSummary
    notes: str


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)  # exactly rounded, so the order of samples is moot


def reduce_transect(samples: Iterable[TransectSample], evening: dt.date) -> AtmosphereV0:
    """Reduce HRRR transect samples to the renderer's v0 parameters for the evening `evening`.

    Raises `ReductionError` if `AOTK` or `LCDC` has no value anywhere on the transect: a forecast
    painted from a missing field would read as clean, clear air, which is not evidence.
    """
    values: dict[str, list[float]] = {}
    indices: set[int] = set()
    for sample in samples:
        indices.add(sample.point_index)
        if sample.value is not None:
            values.setdefault(sample.field, []).append(sample.value)

    def mean_of(name: str, *, required: bool) -> float | None:
        found = values.get(name)
        if found:
            return _mean(found)
        if required:
            raise ReductionError(
                f"no {name} value anywhere on the transect; refusing to read that as clear air"
            )
        return None

    smoke = mean_of("AOTK", required=True)
    low = mean_of("LCDC", required=True)
    assert smoke is not None and low is not None  # required fields raise instead of returning None

    aod = smoke + BACKGROUND_AOD
    ozone = ozone_du(evening.month)
    atmosphere = Atmosphere(aod_550=aod, ozone_du=ozone)

    bases, tops = values.get("HGT_CLOUD_BASE", []), values.get("HGT_CLOUD_TOP", [])
    clouds = CloudSummary(
        points=len(indices),
        low_pct=low,
        mid_pct=mean_of("MCDC", required=False),
        high_pct=mean_of("HCDC", required=False),
        total_pct=mean_of("TCDC", required=False),
        base_m=statistics.median(bases) if bases else None,
        top_m=statistics.median(tops) if tops else None,
        cloudy_points=len(bases),
    )
    pwat = mean_of("PWAT", required=False)
    return AtmosphereV0(
        atmosphere=atmosphere,
        aod_smoke=smoke,
        ozone_du=ozone,
        pwat_kg_m2=pwat,
        clouds=clouds,
        notes=_notes(atmosphere.aod_550, smoke, ozone, evening.month, pwat, clouds),
    )


def _percent(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.0f}%"


def _notes(
    aod: float, smoke: float, ozone: float, month: int, pwat: float | None, clouds: CloudSummary
) -> str:
    """The text sealed in the forecast: what v0 used, and what it recorded without rendering.

    Whole numbers for everything but aerosol and water vapour keep the text clear of the
    privacy scanner's coordinate rule, which reads two nearby decimals as a position.
    """
    first, last = OZONE_TABLE_PERIOD
    water = (
        "Precipitable water not available"
        if pwat is None
        else f"Precipitable water {pwat:.1f} kg m-2, recorded and not rendered"
    )
    if clouds.base_m is None or clouds.top_m is None:
        heights = "no cloud base or top anywhere on the transect"
    else:
        heights = (
            f"cloud base median {clouds.base_m:.0f} m and top median {clouds.top_m:.0f} m "
            f"(HRRR geopotential metres; cloud at {clouds.cloudy_points} of {clouds.points} points)"
        )
    return (
        f"v0 reduction of HRRR along the sightline, {clouds.points} points. "
        f"Aerosol optical depth at 550 nm {aod:.3f} = {BACKGROUND_AOD:.3f} fixed background "
        f"+ {smoke:.3f} HRRR smoke (AOTK transect mean). "
        f"Ozone {ozone:.0f} DU, SBUV monthly climatology for 30-35N, {_MONTH_NAMES[month - 1]}, "
        f"{first}-{last}. "
        f"{water}. "
        f"Cloud cover transect mean: low {_percent(clouds.low_pct)}, "
        f"middle {_percent(clouds.mid_pct)}, high {_percent(clouds.high_pct)}, "
        f"total {_percent(clouds.total_pct)}; {heights}. "
        "Cloud is recorded and not rendered in v0."
    )
