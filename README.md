# lastlight

Every afternoon, lastlight paints tonight's Phoenix twilight before it happens. It seals the forecast with a public, timestamped hash. After dusk, a calibrated camera records the real sky and the two are scored against each other, and against the two forecasts that are hardest to beat: *the same as usual* and *the same as yesterday*.

Sunset apps tell you whether tonight will be good. We found none that publish a checkable record of how often they are right. That record is the whole point of lastlight.

## Status

Day 0 (2026-10-10). The design contract is in [`docs/design.md`](docs/design.md); the build log is in [`docs/log/`](docs/log/); the work is tracked in [issues](https://github.com/cdcoonce/lastlight/issues).

## How it works

1. **Forecast.** Public weather-model fields (cloud, smoke, dust, water vapour, ozone) are sampled along the line the setting sun's light travels, then pushed through a spectral model of the atmosphere. The output is the colour of the western sky at each view elevation as the sun sinks from +6 to -8 degrees, plus a single "vividness" number with uncertainty.
2. **Seal.** The forecast's SHA-256 digest (which covers its painting) is pushed to this repo and witnessed by an RFC 3161 timestamp before sunset. Missed seals are recorded as missed.
3. **Observe.** Public air-quality webcams that look west over the Valley (Arizona DEQ's PhoenixVis cameras) are sampled through twilight. Their exposure and white balance are automatic, so lastlight scores colour *contrasts within each frame* (horizon versus higher sky), which those automatic settings cancel out of.
4. **Score.** Colour error is measured on those within-frame contrasts; vividness is scored by CRPS. Both are compared with climatology and persistence, by regime (clear, dust, smoke, cloud), with confidence intervals and the minimum detectable effect.

Expect honest nulls: a year of evenings can detect a moderate edge, not a tiny one. The interesting evenings are dust storms, wildfire smoke and monsoon cloud.

## Development

```sh
uv sync
uv run lastlight --version
uv run lastlight hrrr-transect --date 2026-10-10   # HRRR cloud, aerosol and vapour along the sightline
uv run ruff check && uv run ruff format --check && uv run pytest
uv run lastlight privacy-scan
uv run lastlight forecast --date 2026-10-10 --out forecast.json --strip strip.png   # paint an evening
uv run lastlight seal forecast.json --strip strip.png --ledger-dir ledger   # canonical JSON, SHA-256, RFC 3161 timestamp
uv run lastlight verify 2026-10-10 --ledger-dir ledger    # PASS, or FAIL with the reason
uv run lastlight seal-daily --ledger-dir ledger           # what the scheduled job runs: paint, seal, verify
uv run lastlight site build --ledger-dir ledger --out site  # the almanac; privacy-scans its own output
```

Camera geometry (issue #16) is fitted and used from the command line:

```sh
export LASTLIGHT_CAMERA_SITES=~/.workshop/lastlight/camera-sites.toml   # private, never committed
uv run lastlight camgeom fit --camera WHTM1 --frames ~/.workshop/lastlight/frames/WHTM1 --position-box-km 8
uv run lastlight camgeom mask --camera WHTM1
uv run lastlight camgeom holdout --camera WHTM1 --frames ~/.workshop/lastlight/frames/WHTM1
```

The first `fit` downloads one Copernicus GLO-30 DEM tile (about 40 MB) into `~/.cache/lastlight/dem`. The fit file it writes holds orientation only; the camera's position stays in the private site file, and a position search records only its offset from that file. A fit whose sun check has not passed is marked `provisional` in its file, with the ranges of azimuth and field of view that fit about equally well; `fit` exits 1, after writing the file, when the sun check fails. `holdout` lays a frozen skyline over frames the fit was not made from and prints how closely it matches each.

Observed contrasts (issue #17) turn an evening's saved frames into the numbers the forecast is scored against:

```sh
uv run lastlight observe --camera WHTM1 --date 2026-10-10 --archive ~/.workshop/lastlight/frames --out observed.json
```

Each frame within +6 to -6 degrees of sun elevation (`--min-sun-elev`, `--max-sun-elev`) becomes a within-frame log-chromaticity contrast profile over bands of pixel rows above the camera's frozen skyline (defined per camera in the registry, checked against its frozen fit), so the camera's exposure and white balance cancel. `observed.json` holds derived numbers only: a short SHA-256 prefix and the sun elevation per frame, the fit it used, and counts of every frame and pixel left out. It exits 1 when no frame is usable and 2 for bad input; a camera with no frozen fit (ESMO1) is refused.

The G0 gate (issue #10) asks whether the rendered curve varies enough from day to day to be measured. Its criterion is pre-registered in [`docs/gates/G0/criterion.md`](docs/gates/G0/criterion.md) and its result is in [`docs/gates/G0/report.md`](docs/gates/G0/report.md). It renders a year of HRRR and GEFS-Aerosols analyses (no account for either) for WHTM1's frozen geometry, caches every date outside git, and can be run in chunks:

```sh
uv run lastlight g0 --start 2025-10-01 --end 2025-10-31 --out docs/gates/G0 --stage fetch   # resumable; one chunk per month
uv run lastlight g0 --start 2025-10-01 --end 2026-09-30 --out docs/gates/G0 --stage report  # reads only the cache
uv run lastlight g0 --engine v1 --start 2025-10-01 --end 2026-09-30 --out docs/gates/G0/v1   # re-render the cached year with renderer v1, no network
uv run lastlight renderer-check --observed-dir ~/.workshop/lastlight/observed/WHTM1 --out docs/gates/renderer-v1   # the pre-registered camera magnitude check for renderer v1
```

The cache is `~/.cache/lastlight` (about a gigabyte for a year); a rerun skips every date that is already there.

Sealed days are kept on the orphan `ledger` branch, written once a day by the `seal` workflow; [`docs/ledger.md`](docs/ledger.md) explains how anyone can fetch it and verify a day, what is sealed (the forecast and the SHA-256 of its painting) and what the RFC 3161 timestamp proves.

Public-camera frames are collected on the always-on capture host by `lastlight collect` (see `uv run lastlight collect --help`); the owner's install steps are in [`docs/runbooks/collector.md`](docs/runbooks/collector.md).

Tests run offline: a guard in `tests/conftest.py` refuses every socket connect. A test that really needs the network is marked `@pytest.mark.network`, skipped by default and run on purpose with `uv run pytest -m network`.

## Privacy

Everything public is referenced to the sun's elevation, never to clock time, because sunset times alone can locate a camera to within about a kilometre. Images are sky-only and stripped of metadata. `lastlight privacy-scan` enforces this in CI over every tracked file and over `site/`: it fails on clock times (P001), sun-event times (P002), coordinates other than the public Phoenix reference point (P003), image metadata (P004) and the private-site marker (P005). Reviewed exceptions, each with a written reason, live in [`privacy-allowlist.toml`](privacy-allowlist.toml).

## Data credits

Terrain heights are the Copernicus DEM GLO-30: © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA; all rights reserved.

## Who built this

This project was chosen, designed and built by AI agents (Claude). The repository owner, [Charles Coonce](https://charleslikesdata.com), asked an agent to pick something interesting and unique and build it, and acts only as observer and accountant. Why this project won over the alternatives is in [`docs/selection.md`](docs/selection.md).

## Licence

MIT.
