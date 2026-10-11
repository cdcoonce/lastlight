# lastlight design contract

This document is the contract every slice builds against. If code and this file disagree, one of them is a bug; fix it in the same PR and say which.

## 1. The claim under test

Every afternoon, lastlight renders tonight's western twilight over Phoenix before it happens, seals the forecast, and lets a calibrated camera score it after dusk. The project asks one question in public and keeps the answer honest:

> Can a physically modelled forecast of twilight colour beat climatology and persistence, and if so, in which sky regimes?

Two quantities are forecast and scored:

| Target | What it is | Score |
| --- | --- | --- |
| **Curve** | Colour of the sky at fixed view elevations in each camera's view direction, as a function of sun elevation from +6 degrees to -8 degrees (the forecast renders that whole range; tier A scores the part its observed window keeps, by default +6 to -6 degrees, and `lastlight observe --min-sun-elev -8` extends it to -8) | Tier A: error in the within-frame log-chromaticity contrast profile, per elevation band. Tier B: delta-E 2000 in camera-native space |
| **Vividness** | One pre-registered scalar per evening summarising how much saturated warm colour the western sky shows during twilight (definition frozen in `docs/gates/vividness.md` before the first scored evening) | CRPS of the forecast distribution |

Both are scored against two baselines: **climatology** (smoothed day-of-year mean of observed history, built only from evenings before the scored one) and **persistence** (yesterday's observation). Results are split by **regime**: clear, dust, smoke, cloud. The regime label comes from data the forecast did not see (GOES clear-sky mask along the sightline, camera cloud fraction, surface PM).

### Truth tiers

| Tier | Instrument | What can be scored |
| --- | --- | --- |
| **A (primary, from 2026-10-10)** | Public, uncalibrated webcams: ADEQ PhoenixVis White Tank (`WHTM1`, looks west, 274 degrees, from Avondale) and Estrella (`ESMO1`, looks southwest from North Mountain). 800x300 JPEG, a new frame about every 15 minutes, auto exposure and white balance, no archive beyond 48 hours of hourly thumbnails. | **Within-frame colour contrasts only.** A webcam's exposure and white balance act, to first order, as one gain per colour channel for the whole frame. So the log-chromaticity difference between two sky regions of the same frame (for example `log(R/G)` in the band of rows 8 to 28 above the camera's frozen skyline minus the same in the reference band of rows 118 to 138) cancels them. The bands are pixel rows above the skyline, not degrees, because a pixel-space definition is exact and repeatable across refits, while the pixel-to-elevation scale of a public camera is not identified until the sun enters its frame (`docs/log/2026-10-11-observed-contrasts.md`). The curve target becomes a profile of these contrasts against view elevation, per frame, indexed by the sun elevation at the frame's time. Vividness is defined on the same invariants. Absolute colour ("how red is the whole sky") is not scored in tier A. |
| **B (optional, later)** | A calibrated RAW camera at the owner's home | Absolute colour, delta-E 2000 in camera-native space. Built only if tier A shows skill worth measuring more precisely; it brings back the home-privacy rules in section 3. |

The forecast computes the same invariants from its rendered spectra for each camera's own viewing geometry (camera position, azimuth, field of view, horizon line), so model and camera are compared like for like. Each camera's pixel-to-sky geometry is fitted on a declared window by matching the camera's skyline to the horizon a public terrain model predicts from the camera's position, checked against the sun's disk wherever the sun is visible, and frozen. The sun is a check, not a fit input, so that it can disagree.

## 2. Honesty rules

1. **Seal before you see.** A forecast counts only if its SHA-256 digest was pushed to the public repo, and witnessed by an RFC 3161 timestamp, before local sunset at the reference point. The forecast carries the SHA-256 of its painting (`strip_sha256`), so the one digest covers the numbers and the picture. Late or missing seals are logged as missing, never back-filled: the seal job refuses to run once the sun is at the horizon, and `lastlight verify` fails any seal whose timestamp is not strictly before it.
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
- `lastlight privacy-scan` runs in CI over the tracked tree and over every generated site and data artifact. A finding fails the build. Exceptions need an allowlist entry with a written reason. The rules are P001 clock time, P002 sun-event time, P003 coordinates (the public reference point excepted), P004 image metadata and P005 the private-site marker; the allowlist is `privacy-allowlist.toml`.

Metro-level location (Phoenix) is public and accepted.

The rules above protect a home camera site (tier B). Tier A cameras are public and their locations are published by their operators, so they leak nothing about the owner; the rules stay in force anyway, because they cost little and tier B may follow. In practice that means a frozen camera fit holds orientation only; the camera's position is read at fit time from a private site file (`LASTLIGHT_CAMERA_SITES`) and never committed.

**Third-party camera frames.** Use only cameras whose page is public and whose operator's terms do not forbid automated access; a feed whose terms forbid it (for example a consumer-camera public share) needs the operator's written consent first. Poll at most once per 5 minutes per camera, with conditional requests and an identifying User-Agent. Keep raw frames private on the capture host, never in git or CI artifacts. Publish derived numbers and small, sky-only thumbnails (at most 320 pixels wide), each credited to and linked back to the operator, and only for cameras whose registry entry has `publish_thumbnails = true` with a recorded `basis`. The site states that any operator's material is removed on request, and a removal is honoured the same day. Basis for the ADEQ PhoenixVis cameras: public page, `robots.txt` allows all, terms silent on automated use; the owner decided on 2026-10-10 to proceed without a prior permission request.

## 4. Architecture

Python 3.12, `uv`, `src/lastlight/`, pytest, ruff. Small modules with one job each:

| Module | Job |
| --- | --- |
| `geometry` | Solar position (own NOAA-equation implementation, pvlib used only as a test oracle), sun-elevation series, elevation bins, sightline transect points along the solar azimuth |
| `site` | Public reference point; private site loader that refuses to run in CI |
| `inputs.hrrr` | Byte-range fetch of selected HRRR fields from the public AWS bucket via `.idx`; sample along the transect |
| `inputs.http`, `inputs.grib` | Shared by the input modules: the HTTP transport (existence checks, byte ranges, loud failures) and the one place a GRIB2 library is imported (ecCodes) |
| `inputs.goes` | GOES-18 ABI clear-sky mask along the transect at twilight (regime labels) |
| `inputs.*` | Later: RRFS-SD dust and smoke, CAMS aerosol and ozone, AirNow PM |
| `render` | Spectral atmosphere model → radiance per (sun elevation, view elevation) → CIE XYZ → camera RGB and sRGB; v0 single scattering, v1 Bruneton precomputed-scattering lookup table |
| `forecast` | The sealed forecast schema (pydantic v2): strict fields, no time of day, only the public reference point, and the nullable `strip_sha256` that seals the painting |
| `atmosphere` | v0 reduction of HRRR transect samples to the renderer's parameters: AOD = HRRR `AOTK` plus a fixed 0.06 background, ozone from a bundled monthly climatology for 30 to 35 N (NASA SBUV), cloud summarised in the notes and not rendered; reused by the G0 spread study |
| `vividness` | The v0 vividness rule: a named placeholder with a deliberately wide sd, until the pre-registered definition and the M4 physics exist |
| `painting` | The daily forecast end to end before sealing: HRRR run, transect, reduction, curve and strip, `Forecast` with every input's source, cycle and digest |
| `daily` | The scheduled seal: paint, seal, verify; idempotent (a sealed evening is left alone) and never late (refuses once the sun is at the horizon) |
| `ledger_branch` | The orphan `ledger` branch as a git worktree: open it, commit a seal to it, push it (never forced) |
| `seal` | Canonical forecast JSON, SHA-256, RFC 3161 timestamp (`rfc3161-client`), ledger writer (forecast, timestamp, strip), `verify` (imprint, time, and the strip against `strip_sha256`; the TSA signature chain is a follow-up) |
| `cameras` | Registry of tier A cameras (operator, image URL, position, view azimuth, attribution); polite collector (conditional GET, identifying User-Agent, timeouts and retries, stale-frame rejection by Last-Modified and burned-in stamp) writing a private archive on the capture host |
| `camgeom` | Per-camera pixel-to-(azimuth, elevation) calibration by skyline matching against a public DEM horizon, checked against the sun; frozen fits in `cameras/geometry/`; sky mask |
| `observe` | Frames → within-frame log-chromaticity contrast profile over pixel-row bands above the frozen skyline (bands defined per camera in the registry, checked against its frozen fit), the `observed.json` of an evening, and a v0 vividness placeholder (tier A); later absolute colour (tier B) |
| `capture`, `calibrate` | Tier B only: RAW capture, dark frames, colour-chart spectral sensitivity |
| `score` | delta-E 2000, CRPS, baselines, regimes, skill with confidence intervals |
| `publish` | Static almanac site from the ledger: painted vs real, scoreboard |
| `privacy` | The scanner in section 3 |

### Data layout

Daily data never lives on `main`. It lives on an orphan branch named `ledger` (no history shared with `main`, no code in it), whose root holds only `ledger/`; the seal workflow is the only thing that writes to it. How to fetch and verify it is in `docs/ledger.md`.

- `ledger/YYYY/MM/DD.forecast.json`: the sealed forecast (canonical JSON, elevation-referenced, no clock times). Its `strip_sha256` is the SHA-256 of the painting below, or null for a forecast with none.
- `ledger/YYYY/MM/DD.forecast.tsr`: the RFC 3161 timestamp response for the digest.
- `ledger/YYYY/MM/DD.strip.png`: the painting (the curve as a strip, sun elevation across and view elevation up), no metadata, present exactly when `strip_sha256` is set. `verify` fails if it is missing, changed, or present without being named.
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

- **Forecast and seal**: the scheduled GitHub Actions workflow `seal` (`.github/workflows/seal.yml`), once a day at a fixed clock time that does not depend on any site; it needs only public inputs. It writes to the `ledger` branch and nothing else (`contents: write` is its only permission), and the almanac site rebuilds when it completes. Its run log is a second public witness.
- **Collect, observe, score**: the always-on capture host at home (an Ubuntu server), running the tier A collector during each evening's twilight window and pushing only derived, elevation-referenced numbers. The owner installs its timer; agents never install timers themselves.

## 5. Milestones and gates

| Milestone | Delivers | Gate to pass before the next spend |
| --- | --- | --- |
| M0 Foundations | Scaffold, CI, privacy scanner, geometry, seal ledger, HRRR transect ingest | CI green; privacy scanner proven by planted leaks |
| M1 First painting | Renderer v0, daily sealed forecast, almanac site showing tonight's painting | A forecast sealed and verified on 5 consecutive days |
| **G0** | Spread study: render a year of clear-sky curves from archived inputs | On at least 25 percent of dates, the rendered contrast profile differs from the period's median by a band-mean of at least 0.05 in log-chromaticity contrast in at least 2 of 3 twilight bands (delta-E 2000 reported alongside for tier B); otherwise the physics barely varies and the forecast cannot beat climatology on clear evenings |
| M2 Public-camera truth | Tier A collector (owner installs the timer), camera geometry calibration, contrast-profile observations, permission requests to operators (owner sends) | **G1**: on pairs of consecutive clear evenings, the observed contrast profile repeats more closely than the spread across all evenings (repeatability at most half the spread), over at least 10 pairs |
| M3 Scoring | Forecast invariants per camera, baselines, regimes, public scoreboard | First 30 scored evenings |
| M4 Physics | Bruneton lookup table, cloud-lit vividness, RRFS-SD dust, CAMS | Pre-registered comparison of v1 against v0 on held-out evenings |
| M5 (optional) Tier B | Calibrated home camera for absolute colour (spend and hands-on ask) | Only if M3 shows skill worth a better instrument |

## 6. How it is built

- One GitHub issue per slice. Each issue states its acceptance criteria as tests. Test-first: a failing test before the code that makes it pass.
- A builder agent works each issue in its own branch and worktree and opens a PR. The conductor re-runs the gate, injects one representative defect per named behaviour (the suite must go red on the predicted test), reviews against this contract, and lands.
- Every session appends an entry to `docs/log/`: what was decided, the alternative rejected, what broke.
