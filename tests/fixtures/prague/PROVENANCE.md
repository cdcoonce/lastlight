# Provenance of `sun10.json`

`sun10.json` holds CIE XYZ of the clear daytime sky as the Prague Sky Model gives it, for the
control of issue #34: renderer v1 against an independent daylight model at sun +10 degrees.
`tests/test_prague_control.py` reads it; nothing in CI needs the model, the datasets or the
network. The model's licence notice is in `NOTICE`.

## What was used

| | |
| --- | --- |
| Model | the reference implementation of the Prague Sky Model, https://github.com/PetrVevoda/pragueskymodel, commit `2385c912e9051c1258013ff8c3ce2e19e10fb917` (2023-08-27, the repository's last commit; it has no release tags). Apache-2.0, Copyright 2022 Charles University. Wilkie et al., ACM TOG 40(4) 2021; Vevoda et al., CGF 41(7) 2022. |
| Dataset used for every number | `SkyModelDataset.dat`, the authors' full dataset (README: "Full version (2.2 GB)"), 2397370196 bytes, SHA-256 `33832f99525e7833cb83ac4369026f07e7d403db2fe01b5e9fedbe92ce898727`. Google Drive file `1IShL7T3umxGOEFvyYGQpHKMTneXEvyTM`, linked from the repository's README. |
| Dataset used as a cross-check | `SkyModelDatasetGround.dat`, the authors' ground-level dataset (README: "Ground-level version (103 MB)"), 107632816 bytes, SHA-256 `76bd619dc6dfcbc900c2996436dd6cf68197c03e3982fe5580f2b109ce1c71c2`. Google Drive file `1IflyFZTJxC_N298yXq_2GK4ycIsVJZk6`. |
| Where they live | outside the repository, under `~/.cache/lastlight/prague/` (`src/`, `dl/`, `build/`). Never committed. |
| Cost | zero. The links are public and need no account. Google Drive shows a "can't scan this file for viruses" page for files this size; the tool submits that page's form, which is what its "Download anyway" button does. The repository's README gives the model's licence (Apache-2.0) and states no separate licence for the datasets; no acceptance of any other terms was asked for or given. |

The axes of the full dataset (read from its header): visibility 20, 27.6, 40, 59.4, 90, 131.8 km;
ground albedo 0, 1/3, 2/3, 1; 22 observer altitudes from 0 to 15000 m (the two around 340 m are
234.375 and 405); 30 sun elevations (the two around +10 degrees are 8.05 and 11.25); 11 spectral
channels of 40 nm from 320 to 760 nm, centred at 340, 380, ..., 740 nm. The model interpolates
linearly between neighbouring nodes of each axis, and clamps a value outside an axis to its end
without a word.

## Why the full dataset, not the ground-level one

The control puts the observer at the reference point's height, 340 m. The ground-level dataset has a
single observer altitude, 0, so it would evaluate a 340 m observer at 0 m. The full dataset has the
altitude axis. The ground-level file is kept as a check and is used so: at altitude 0 it gives, for
every query of the fixture, exactly the numbers the full dataset gives (the tool compares them and
records the result in `provenance.ground_level_dataset_matches_full_at_altitude_0`), and the
diagnostic `xyz_altitude_0` is that altitude 0 reading.

## The configuration

- Sun elevation +10 degrees (the model's "ground level solar elevation at origin"), sun azimuth 0;
  the view is at the relative azimuths 0, 90 and 180 degrees.
- Observer altitude 0.34 km. `PragueSkyModel::computeParameters` lifts the viewpoint by 50 m, so the
  driver gives it a viewpoint 290 m up; the driver prints the altitude the model ends up with and
  the tool refuses any that is not 340 m (and any sun elevation that is not 10 degrees).
- Ground albedo 0.2, v1's fixed albedo (the nodes around it are 0 and 1/3).
- View elevations: WHTM1's five bands and its reference elevation at the nominal pixel scale,
  `lastlight.g0.collect.build_views()["nominal"].elevations_deg` (1.5066, 1.9589, 2.4113, 2.8636,
  3.3159 and 3.9944 degrees), then 30 and 60 degrees as diagnostics. `test_the_fixture_views_are_the_cameras_nominal_views`
  fails if a refit of the camera moves them.
- Two atmospheres, v1's `Atmosphere(aod_550=...)` with its defaults (Angstrom 1.3, aerosol scale
  height 1.5 km, ozone 280 DU, 970 hPa): `aod_550` 0.03 and 0.1.

## The mapping from `aod_550` to Prague's visibility

Prague takes the horizontal visibility at the ground. Computed, not tuned, from
`lastlight.render.atmosphere` (`tools/prague_reference.py: visibility_km`): Koschmieder at 550 nm,
V = 3.912 / beta, with beta the extinction at the ground: v1's Rayleigh vertical optical depth at its
970 hPa (0.0927566) over its 8 km Rayleigh scale height = 0.0115946 per km, plus `aod_550` over its
1.5 km aerosol scale height.

| `aod_550` | aerosol per km | beta per km | visibility (km) | dataset nodes around it |
| --- | --- | --- | --- | --- |
| 0.03 | 0.0200000 | 0.0315946 | 123.8187 | 90 and 131.8 |
| 0.1 | 0.0666667 | 0.0782612 | 49.9864 | 40 and 59.4 |

Both lie inside Prague's valid range, 20 to 131.8 km. The 0.03 atmosphere is 8 km from the clearest
the model can describe. That the Rayleigh term is included is not stated by the model's documents
("horizontal viewing distance at ground level"); it is the only reading under which `aod_550` 0.03
is inside the range at all (aerosol alone would give 195.6 km).

## From Prague's channels to XYZ

Prague returns spectral radiance for 11 channels of 40 nm. The tool queries each channel at its
centre, treats that value as the radiance at the centre (as the authors' example renderer does),
reads the renderer's 10 nm grid (380 to 780 nm) off linearly between centres, holding the value flat
beyond the first and last centre (below 340 and above 740 nm), and integrates with the repository's
own colour-matching functions and white-card normalisation,
`lastlight.render.colorimetry.spectrum_to_xyz`. XYZ is therefore on v1's scale: a white card lit by
the extraterrestrial sun has Y = 1.

## Diagnostics (other readings of the same model; finite-only in the tests)

- `xyz_binned`: piecewise-constant channels, the model's own lookup (a wavelength takes the value of
  the 40 nm bin that holds it, zero outside 320 to 760 nm).
- `xyz_inverse_visibility`: the model interpolates linearly in V between its nodes, but extinction
  goes as 1 / V; this reads the two bracketing nodes and interpolates linearly in 1 / V instead.
- `xyz_altitude_0`: the observer at altitude 0.
- `xyz_matched_aerosol_depth`: Prague at the visibility where its own vertical aerosol optical depth
  at 550 nm (from its zenith transmittance, less v1's Rayleigh and 345 DU ozone depths, so about
  +-0.01) equals v1's `aod_550`, by bisection. For `aod_550` 0.03 Prague cannot be that clear (at
  131.8 km it still has 0.037), so that atmosphere is clamped at 131.8 km;
  `diagnostics.matched_aerosol_depth` says which.
- `zenith_transmittance`: Prague's transmittance to space along the zenith, per channel.

## Regenerating

```
uv run python tools/prague_reference.py fetch build fixture    # about 6 minutes after the downloads
uv run python tools/prague_reference.py fixture --check        # re-evaluate and compare, write nothing
uv run python tools/prague_reference.py report                 # the results tables, from the fixture
```

`fetch` clones the model at the pinned commit and downloads both datasets, verifying size and SHA-256
(2.5 GB); `build` compiles `PragueSkyModel.cpp` and lastlight's driver with the system C++17
compiler (no CMake needed); `fixture` runs the driver. `pytest -m network
tests/test_prague_control.py` runs the `--check` against the built model.
