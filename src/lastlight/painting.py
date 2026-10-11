"""Paint tonight's twilight: HRRR in, a sealed-ready forecast and its strip out.

`paint` is the daily forecast end to end, before any sealing (`docs/design.md` section 4):

1. choose the HRRR run for the evening (`inputs.hrrr.plan_run`) and fetch its fields by byte
   range, recording the SHA-256 of every raw message read;
2. sample them along the sightline toward the sunset azimuth from the public reference point;
3. reduce the samples to the renderer's v0 parameters (`lastlight.atmosphere`);
4. render the curve and the strip (`lastlight.render`);
5. put it all in a `Forecast`: every input's source, run and digest, the curve, the v0
   vividness, notes saying what was and was not used, and the SHA-256 of the strip PNG.

It does no sealing and writes no files; the caller decides where the forecast and strip go
(`lastlight forecast` writes them, `lastlight.daily.seal_daily` seals them). The HRRR transport
and the clock are arguments so tests can serve recorded files; nothing here reads the time of
day except through `now_utc`, which `plan_run` uses to pick a run and never leaves it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import os
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from lastlight import __version__, geometry
from lastlight.atmosphere import reduce_transect
from lastlight.forecast import CurvePoint, Forecast, Input, Model, ReferencePoint
from lastlight.inputs import hrrr
from lastlight.inputs.http import Transport
from lastlight.render import render_curve, render_strip, strip_png_bytes
from lastlight.site import REFERENCE_POINT
from lastlight.vividness import RULE_SUMMARY, vividness_v0

MODEL_NAME = "lastlight-v0"
"""The renderer generation: v0 is single scattering; v1 (Bruneton) will be a new model name."""

_SHA = re.compile(r"[0-9a-f]{40}")
_UNKNOWN_SHA = "unknown"


@dataclass(frozen=True)
class Painting:
    """One evening's forecast and the PNG its `strip_sha256` names (None when it names none)."""

    forecast: Forecast
    strip_png: bytes | None


def current_code_sha(*, environ: Mapping[str, str] | None = None, cwd: Path | None = None) -> str:
    """The commit of the code that is running: `LASTLIGHT_CODE_SHA`, `GITHUB_SHA`, the checkout.

    Each source counts only if it is a full 40-character commit id. `cwd` is the directory asked
    for `git rev-parse HEAD` (default: this package's own checkout). "unknown" if none answers.
    """
    environment = os.environ if environ is None else environ
    for name in ("LASTLIGHT_CODE_SHA", "GITHUB_SHA"):
        value = environment.get(name, "").strip().lower()
        if _SHA.fullmatch(value):
            return value
    where = Path(__file__).resolve().parent if cwd is None else cwd
    try:
        answer = subprocess.run(
            ["git", "-C", str(where), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return _UNKNOWN_SHA
    value = answer.stdout.strip().lower()
    return value if answer.returncode == 0 and _SHA.fullmatch(value) else _UNKNOWN_SHA


def paint(
    evening: dt.date,
    *,
    transport: Transport,
    now_utc: dt.datetime,
    cache_dir: Path | str,
    cycle_hour: int | None = None,
    forecast_hour: int | None = None,
    code_sha: str | None = None,
    with_strip: bool = True,
) -> Painting:
    """Paint the evening of `evening` at the public reference point.

    `cycle_hour` and `forecast_hour` pin the HRRR run (see `hrrr.plan_run`); by default the
    newest complete run that reaches the evening is used. With `with_strip=False` the forecast
    names no painting (`strip_sha256` null) and no PNG is made. Raises whatever the HRRR ingest
    raises (`HrrrError`, `FetchError`) and `ReductionError` for samples that cannot be reduced;
    it never paints from missing data.
    """
    lat, lon, elevation = REFERENCE_POINT
    plan = hrrr.plan_run(
        evening,
        lat,
        lon,
        cycle_hour=cycle_hour,
        forecast_hour=forecast_hour,
        now_utc=now_utc,
        transport=transport,
    )
    fields = hrrr.fetch_fields(
        plan.cycle_date,
        plan.cycle_hour,
        plan.forecast_hour,
        hrrr.DEFAULT_FIELDS,
        cache_dir,
        transport=transport,
    )
    distances = [float(km) for km in hrrr.TRANSECT_DISTANCES_KM]
    samples = hrrr.sample_transect(fields, geometry.transect(evening, lat, lon, distances))
    reduction = reduce_transect(samples, evening)

    curve = tuple(CurvePoint(**row) for row in render_curve(reduction.atmosphere))
    strip_png = (
        strip_png_bytes(render_strip([point.model_dump() for point in curve]))
        if with_strip
        else None
    )
    vividness = vividness_v0(reduction.clouds.low_pct, reduction.atmosphere.aod_550)
    forecast = Forecast(
        schema_version=1,
        date=evening,
        model=Model(
            name=MODEL_NAME,
            version=__version__,
            code_sha=current_code_sha() if code_sha is None else code_sha,
        ),
        reference_point=ReferencePoint(lat=lat, lon=lon, elev_m=elevation),
        inputs=tuple(
            Input(
                source=f"hrrr:{spec.name}",
                cycle_date=fields.cycle_date,
                cycle_hour_utc=fields.cycle_hour,
                forecast_hour=fields.forecast_hour,
                digest=fields.digests[spec.name],
            )
            for spec in hrrr.DEFAULT_FIELDS
        ),
        curve=curve,
        vividness=vividness,
        notes=f"{reduction.notes} {RULE_SUMMARY}",
        strip_sha256=None if strip_png is None else hashlib.sha256(strip_png).hexdigest(),
    )
    return Painting(forecast=forecast, strip_png=strip_png)
