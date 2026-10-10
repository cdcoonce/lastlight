# lastlight design contract

This document is the contract every slice builds against. If code and this file disagree, one of them is a bug; fix it in the same PR and say which.

## 1. The claim under test

Every afternoon, lastlight renders tonight's western twilight over Phoenix before it happens, seals the forecast, and lets a calibrated camera score it after dusk. The project asks one question in public and keeps the answer honest:

> Can a physically modelled forecast of twilight colour beat climatology and persistence, and if so, in which sky regimes?

Two quantities are forecast and scored:

| Target | What it is | Score |
| --- | --- | --- |
| **Curve** | Colour of the sky along the solar azimuth, sampled at fixed view elevations, as a function of sun elevation from +6 degrees to -8 degrees | delta-E 2000 in camera-native space, per elevation band |
| **Vividness** | One pre-registered scalar per evening summarising how much saturated warm colour the western sky shows during twilight (definition frozen in `docs/gates/vividness.md` before the first scored evening) | CRPS of the forecast distribution |

Both are scored against two baselines: **climatology** (smoothed day-of-year mean of observed history, built only from evenings before the scored one) and **persistence** (yesterday's observation). Results are split by **regime**: clear, dust, smoke, cloud. The regime label comes from data the forecast did not see (GOES clear-sky mask along the sightline, camera cloud fraction, surface PM).

## 2. Honesty rules

1. **Seal before you see.** A forecast counts only if its SHA-256 digest was pushed to the public repo, and witnessed by an RFC 3161 timestamp, before local sunset at the reference point. Late or missing seals are logged as missing, never back-filled.
2. **Freeze before you score.** Any fitted correction (debias, calibration offsets, regime thresholds) is fitted on a declared window and frozen by commit before the evenings it is scored on. The scored window never overlaps the fit window.
3. **Report the null.** Every skill number ships with a confidence interval (Newey-West, serial correlation) and the minimum detectable effect at 80 percent power. "No detectable skill" is a result and is published with the same prominence as a win.
4. **Baselines are first-class.** Climatology and persistence run through the same scoring code as the model. A change that improves the model's score but is not applied to the baselines is a bug.
5. **No silent caps.** Any dropped evening (camera fault, missing inputs, sun masked by cloud) is counted and published with its reason.

Expected power (from the selection dossier's Monte Carlo): a full year detects overall MSE skill of 0.2 to 0.4; it does not detect skill of 0.03 to 0.07. Months 1-6 build a baseline. The aerosol claim needs the 2027 dust and monsoon season.

## 3. Privacy rules (enforced by CI)

The home's location must not be derivable from anything public. Sunset clock times alone geolocate a site to about 1 km within weeks, so:

- Public artifacts are **referenced to sun elevation, never clock time**. No `HH:MM`, no ISO timestamps with a time component, no UTC, no sunset or sunrise times, no day length. Dates are allowed.
- All public geometry uses the **public reference point**: Phoenix metro, 33.45 N, 112.07 W, 340 m (downtown Phoenix). Sun elevation changes about 0.01 degrees per km, so the exact site is never needed for modelling.
- The camera site's coordinates live only in a private config on the capture host (`LASTLIGHT_SITE_FILE`), never in the repo, CI, logs or images.
- Published images are **sky only**: horizon masked with a margin, downsampled, all metadata stripped (EXIF, XMP, GPS, ICC comments).
- Scheduled jobs seal at a **fixed clock time** that does not depend on the site.
- `lastlight privacy-scan` runs in CI over the tracked tree and over every generated site and data artifact. A finding fails the build. Exceptions need an allowlist entry with a written reason.

Metro-level location (Phoenix) is public and accepted.

## 4. Architecture

Python 3.12, `uv`, `src/lastlight/`, pytest, ruff. Small modules with one job each:

| Module | Job |
| --- | --- |
| `geometry` | Solar position (own NOAA-equation implementation, pvlib used only as a test oracle), sun-elevation series, elevation bins, sightline transect points along the solar azimuth |
| `site` | Public reference point; private site loader that refuses to run in CI |
| `inputs.hrrr` | Byte-range fetch of selected HRRR fields from the public AWS bucket via `.idx`; sample along the transect |
| `inputs.goes` | GOES-18 ABI clear-sky mask along the transect at twilight (regime labels) |
| `inputs.*` | Later: RRFS-SD dust and smoke, CAMS aerosol and ozone, AirNow PM |
| `render` | Spectral atmosphere model → radiance per (sun elevation, view elevation) → CIE XYZ → camera RGB and sRGB; v0 single scattering, v1 Bruneton precomputed-scattering lookup table |
| `seal` | Canonical forecast JSON, SHA-256, RFC 3161 timestamp, `verify` |
| `capture` | Sun-elevation-scheduled bracketed RAW capture, dark frames (runs on the capture host only) |
| `calibrate` | Flat field, dark, linearity, camera spectral sensitivity from a colour chart |
| `observe` | Frames → observed curve and vividness |
| `score` | delta-E 2000, CRPS, baselines, regimes, skill with confidence intervals |
| `publish` | Static almanac site from the ledger: painted vs real, scoreboard |
| `privacy` | The scanner in section 3 |

### Data layout

- `ledger/YYYY/MM/DD.forecast.json`: the sealed forecast (canonical JSON, elevation-referenced, no clock times).
- `ledger/YYYY/MM/DD.forecast.tsr`: the RFC 3161 timestamp response for the digest.
- `ledger/YYYY/MM/DD.observed.json`: observed curve and vividness (later).
- `ledger/index.jsonl`: one line per sealed forecast: date, digest, model version.
- `site/`: the generated almanac (not hand-edited).
- Raw frames, GRIB downloads and calibration captures stay on the capture host, never in git.

### Engines and licences

- Bruneton 2017 precomputed atmospheric scattering (BSD-3-Clause): the twilight engine, valid to sun elevation -12 degrees; used to build a lookup table offline.
- Prague Sky Model (Apache-2.0): daytime and early-twilight positive control, valid to -4.2 degrees.
- libRadtran (GPL): external offline validator only; never vendored or imported.
- The GPL Python port `skylight` must never be vendored or imported.

### Where things run

- **Forecast and seal**: a scheduled GitHub Actions job at a fixed clock time; it needs only public inputs. Its run log is a second public witness.
- **Capture, observe, score**: the capture host at home (an always-on Ubuntu server, or a Raspberry Pi next to the camera), pushing only sky-only, elevation-referenced products.

## 5. Milestones and gates

| Milestone | Delivers | Gate to pass before the next spend |
| --- | --- | --- |
| M0 Foundations | Scaffold, CI, privacy scanner, geometry, seal ledger, HRRR transect ingest | CI green; privacy scanner proven by planted leaks |
| M1 First painting | Renderer v0, daily sealed forecast, almanac site showing tonight's painting | A forecast sealed and verified on 5 consecutive days |
| **G0** | Spread study: render a year of clear-sky curves from archived inputs | Rendered P10-P90 spread of at least 4 delta-E 2000 in at least 2 of 3 elevation bands on at least 25 percent of days; otherwise the curve target cannot be resolved and the design changes before hardware is bought |
| M2 Instrument | Camera purchase (human ask), mount, capture, calibration | **G1**: repeatability of at least 2x below the observed clear-sky spread across 20 clear evenings; re-run in the first hot month |
| M3 Scoring | Observed curves, vividness, baselines, regimes, public scoreboard | First 30 scored evenings |
| M4 Physics | Bruneton lookup table, cloud-lit vividness, RRFS-SD dust, CAMS | Pre-registered comparison of v1 against v0 on held-out evenings |

## 6. How it is built

- One GitHub issue per slice. Each issue states its acceptance criteria as tests. Test-first: a failing test before the code that makes it pass.
- A builder agent works each issue in its own branch and worktree and opens a PR. The conductor re-runs the gate, injects one representative defect per named behaviour (the suite must go red on the predicted test), reviews against this contract, and lands.
- Every session appends an entry to `docs/log/`: what was decided, the alternative rejected, what broke.
