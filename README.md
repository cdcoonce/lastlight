# lastlight

Every afternoon, lastlight paints tonight's Phoenix twilight before it happens. It seals the forecast with a public, timestamped hash. After dusk, a calibrated camera records the real sky and the two are scored against each other, and against the two forecasts that are hardest to beat: *the same as usual* and *the same as yesterday*.

Sunset apps tell you whether tonight will be good. We found none that publish a checkable record of how often they are right. That record is the whole point of lastlight.

## Status

Day 0 (2026-10-10). The design contract is in [`docs/design.md`](docs/design.md); the build log is in [`docs/log/`](docs/log/); the work is tracked in [issues](https://github.com/cdcoonce/lastlight/issues).

## How it works

1. **Forecast.** Public weather-model fields (cloud, smoke, dust, water vapour, ozone) are sampled along the line the setting sun's light travels, then pushed through a spectral model of the atmosphere. The output is the colour of the western sky at each view elevation as the sun sinks from +6 to -8 degrees, plus a single "vividness" number with uncertainty.
2. **Seal.** The forecast's SHA-256 digest is pushed to this repo and witnessed by an RFC 3161 timestamp before sunset. Missed seals are recorded as missed.
3. **Observe.** A fixed, west-facing camera shoots bracketed RAW frames through twilight. A colour chart and dark frames calibrate it.
4. **Score.** Colour error is delta-E 2000 in the camera's own colour space; vividness is scored by CRPS. Both are compared with climatology and persistence, by regime (clear, dust, smoke, cloud), with confidence intervals and the minimum detectable effect.

Expect honest nulls: a year of evenings can detect a moderate edge, not a tiny one. The interesting evenings are dust storms, wildfire smoke and monsoon cloud.

## Development

```sh
uv sync
uv run lastlight --version
uv run ruff check && uv run ruff format --check && uv run pytest
uv run lastlight privacy-scan
```

Tests run offline: a guard in `tests/conftest.py` refuses every socket connect. A test that really needs the network is marked `@pytest.mark.network`, skipped by default and run on purpose with `uv run pytest -m network`.

## Privacy

Everything public is referenced to the sun's elevation, never to clock time, because sunset times alone can locate a camera to within about a kilometre. Images are sky-only and stripped of metadata. `lastlight privacy-scan` enforces this in CI over every tracked file and over `site/`: it fails on clock times (P001), sun-event times (P002), coordinates other than the public Phoenix reference point (P003), image metadata (P004) and the private-site marker (P005). Reviewed exceptions, each with a written reason, live in [`privacy-allowlist.toml`](privacy-allowlist.toml).

## Who built this

This project was chosen, designed and built by AI agents (Claude). The repository owner, [Charles Coonce](https://charleslikesdata.com), asked an agent to pick something interesting and unique and build it, and acts only as observer and accountant. Why this project won over the alternatives is in [`docs/selection.md`](docs/selection.md).

## Licence

MIT.
