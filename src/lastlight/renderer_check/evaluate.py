"""From the observed files and the collected inputs to the verdict, evening by evening.

Criterion definitions 2 to 6 and the rule. Each observed evening gets one status:

- `not_considered`: before the first evening of the criterion, or fewer than 3 used frames;
- `pending`: considered, but its inputs have not been collected yet (no record);
- `dropped`: considered, but its inputs are missing (a reason from the store) or a used frame or
  the atmosphere is outside the table v1 reads (`v1_outside_table`); dropped evenings are not
  counted and never take a first-5 slot;
- `not_counted`: rendered, but the mean cloud over the sightline is above 10 percent; reported
  with its cloud and its ratios, and it does not enter the verdict;
- `counted`: rendered and clear. The 5 earliest counted evenings in date order hold the slots 1
  to 5 and decide the verdict; later counted evenings are reported beside it.

A pending evening can still turn out to be counted, and then it would come first. So a counted
evening dated after the earliest pending one holds no slot yet (`held`): otherwise a verdict could
be computed on evenings that a later run would reorder, and the criterion computes it once.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Final

from lastlight.g0.view import SkyView
from lastlight.renderer_check import rules
from lastlight.renderer_check.observed import ObservedEvening
from lastlight.renderer_check.render import FramePair, OutsideTable, observed_only, render_evening
from lastlight.renderer_check.store import InputStore

COUNTED: Final = "counted"
NOT_COUNTED: Final = "not_counted"
DROPPED: Final = "dropped"
NOT_CONSIDERED: Final = "not_considered"
PENDING: Final = "pending"

REASON_CLOUD: Final = "cloud"
REASON_TOO_FEW_FRAMES: Final = "too_few_frames"
REASON_BEFORE_FIRST: Final = "before_first_evening"
REASON_OUTSIDE_TABLE: Final = "v1_outside_table"


@dataclass(frozen=True)
class Ratios:
    """An evening's r_e values. None: not computable (no R/G value, or a pixel-scale end that
    the table does not cover). Non-finite values are +infinity (a zero observed median)."""

    v1_bg: float
    v0_bg: float
    v1_bg_low: float | None
    v1_bg_high: float | None
    v1_rg: float | None
    v0_rg: float | None


@dataclass(frozen=True)
class EveningResult:
    """Everything the outputs say about one observed evening."""

    date: dt.date
    status: str
    reason: str
    frames_used: int
    cloud_mean_pct: float | None = None
    aod_550: float | None = None
    ratios: Ratios | None = None
    pairs: tuple[FramePair, ...] = ()
    slot: int | None = None
    held: bool = False


@dataclass(frozen=True)
class Evaluation:
    """All evenings and the verdict. `slots` are the counted evenings that decide it."""

    results: tuple[EveningResult, ...]
    verdict: str
    slots: tuple[EveningResult, ...] = field(default=())

    @property
    def pending(self) -> tuple[EveningResult, ...]:
        return tuple(r for r in self.results if r.status == PENDING)

    @property
    def decided(self) -> bool:
        return len(self.slots) >= rules.COUNTED_NEEDED


def _ratio(rendered: Sequence[float | None], observed: Sequence[float | None]) -> float | None:
    """r_e over the frames that have both values ("the same frames"); None if there are none."""
    both = [
        (r, o) for r, o in zip(rendered, observed, strict=True) if r is not None and o is not None
    ]
    if not both:
        return None
    return rules.evening_ratio([r for r, _ in both], [o for _, o in both])


def _ratios(pairs: Sequence[FramePair]) -> Ratios:
    bg = [p.observed_bg for p in pairs]
    rg = [p.observed_rg for p in pairs]
    v1_bg = _ratio([p.v1_bg for p in pairs], bg)
    v0_bg = _ratio([p.v0_bg for p in pairs], bg)
    assert v1_bg is not None and v0_bg is not None  # every pair of a rendered evening has both
    return Ratios(
        v1_bg=v1_bg,
        v0_bg=v0_bg,
        v1_bg_low=_ratio([p.v1_bg_low for p in pairs], bg),
        v1_bg_high=_ratio([p.v1_bg_high for p in pairs], bg),
        v1_rg=_ratio([p.v1_rg for p in pairs], rg),
        v0_rg=_ratio([p.v0_rg for p in pairs], rg),
    )


def _evening_result(
    evening: ObservedEvening, store: InputStore, views: Mapping[str, SkyView]
) -> EveningResult:
    used = len(evening.frames)
    if evening.date < rules.FIRST_EVENING:
        return EveningResult(evening.date, NOT_CONSIDERED, REASON_BEFORE_FIRST, used)
    if used < rules.MIN_USED_FRAMES:
        return EveningResult(evening.date, NOT_CONSIDERED, REASON_TOO_FEW_FRAMES, used)
    observed = observed_only(evening.frames)
    inputs = store.load(evening.date)
    if inputs is None:
        return EveningResult(evening.date, PENDING, "", used, pairs=observed)
    if inputs.drop_reason is not None:
        return EveningResult(evening.date, DROPPED, inputs.drop_reason, used, pairs=observed)
    assert inputs.aod_550 is not None and inputs.ozone_du is not None
    assert inputs.cloud_mean_pct is not None
    try:
        pairs = render_evening(evening.date, inputs.aod_550, inputs.ozone_du, evening.frames, views)
    except OutsideTable:
        return EveningResult(
            evening.date,
            DROPPED,
            REASON_OUTSIDE_TABLE,
            used,
            cloud_mean_pct=inputs.cloud_mean_pct,
            aod_550=inputs.aod_550,
            pairs=observed,
        )
    clear = rules.is_clear(inputs.cloud_mean_pct)
    return EveningResult(
        evening.date,
        COUNTED if clear else NOT_COUNTED,
        "" if clear else REASON_CLOUD,
        used,
        cloud_mean_pct=inputs.cloud_mean_pct,
        aod_550=inputs.aod_550,
        ratios=_ratios(pairs),
        pairs=pairs,
    )


def evaluate(
    evenings: Sequence[ObservedEvening], store: InputStore, views: Mapping[str, SkyView]
) -> Evaluation:
    """Every observed evening classified, the first-5 slots given out, and the verdict."""
    results = [_evening_result(evening, store, views) for evening in evenings]
    results.sort(key=lambda result: result.date)
    earliest_pending = min((r.date for r in results if r.status == PENDING), default=None)
    slotted: list[EveningResult] = []
    placed: list[EveningResult] = []
    for result in results:
        if result.status != COUNTED:
            placed.append(result)
            continue
        if earliest_pending is not None and result.date > earliest_pending:
            placed.append(replace(result, held=True))
        elif len(slotted) < rules.COUNTED_NEEDED:
            numbered = replace(result, slot=len(slotted) + 1)
            slotted.append(numbered)
            placed.append(numbered)
        else:
            placed.append(result)
    verdict = rules.decide([slot.ratios.v1_bg for slot in slotted if slot.ratios is not None])
    return Evaluation(results=tuple(placed), verdict=verdict, slots=tuple(slotted))
