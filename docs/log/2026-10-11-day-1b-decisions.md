# Day 1, second session: conductor decisions

The conductor's calls in the second session of 2026-10-11, each with what it rejected. Slice-level choices are in each slice's own entry: `2026-10-11-tsa-signature.md` (#25, PR #37), `2026-10-11-seal-retry-and-missed.md` (#26, PR #36), `2026-10-11-camgeom-capture.md` (#28, PR #38), `2026-10-11-renderer-v1-engine.md` (#32, PR #39) and `2026-10-11-second-evening.md`.

## The renderer v1 adoption rule, registered before v1 existed

**Decided:** the camera magnitude check that decides whether v1 paints the daily forecast is registered in `docs/gates/renderer-v1/criterion.md` (PR #33). It landed before any v1 code was written. The rule: the median, over the first 5 clear evenings, of rendered over observed |log(B/G)| lowest-band contrast lies between 0.5 and 2.0. An evening counts as clear by HRRR's total cloud cover along the G0 transect (at most 10 percent), data that neither the renderer nor the camera supplies. The verdict is taken once, on the 5 earliest counted evenings. The v1 table's SHA-256 is recorded as an amendment when v1 lands, before the check is built.

**Rejected alternatives:**

- Every evening with observe output, as the issue first said. Both evenings observed so far were overcast, and a clear-sky renderer measured against a cloud deck gives a ratio that says nothing about its physics.
- A verdict recomputed as evenings accumulate. Running the check until it passes is optional stopping.
- Registering the rule in the same PR as the analysis. The criterion should provably predate the code that produces its result.

## Issue #32 split in three

**Decided:** #32 now delivers the v1 engine only. The Prague Sky Model daylight control is #34. The magnitude check and the G0 re-run are #35, which needs the registered rule and a re-render, because G0's cache holds v0's rendered contrasts, not inputs a report stage can switch engines on.

**Rejected alternatives:** one PR for all of it (about four units, three independent review surfaces, and the control needs an external model the engine does not).

## How v1 is built

**Decided:**

- v1's XYZ is v0's single-scattering XYZ, unchanged, plus the XYZ of light scattered two or more times (ground bounce included).
- The extra light comes from a numpy port of Bruneton 2017's precomputation, run on v0's own atmosphere.
- It is stored as XYZ over `aod_550`, `ozone_du`, sun elevation, view elevation and relative azimuth. Only those two atmosphere fields vary day to day.
- The table is committed under a 2 MB cap and pinned by SHA-256.
- The engine is named in `model.name` (`lastlight-v0` or `lastlight-v1`).
- The daily forecast keeps v0 until the registered rule passes.

**Rejected alternatives:**

- Replacing v0's exact single scattering with Bruneton's interpolated single-scattering lookup.
- Spectral storage, about 14 times the size, when every consumer reads XYZ.
- Downloading the table from a release asset by pinned digest, which needs network in CI and in tests that block sockets.
- Generating it in CI, which spends minutes on every run.
- Git LFS, whose bandwidth is metered.
- Recording the engine in `model.version`, which carries the package version and is what `index.jsonl` stores.

## Timestamp fixtures that really sign

**Decided:** issue #25's tests use a throwaway CA and TSA that genuinely sign RFC 3161 responses for any bytes and any time. The real-pin oracles stay free of any test fixture:

- the recorded FreeTSA response verifies offline against the pinned pair;
- the same response with its signature altered fails;
- a token from the test authority fails under the default trust.

**Rejected alternatives:** responses recorded once from FreeTSA for fixed bytes. They tie every fixture to exact forecast bytes and to the recording date, and refreshing them needs the network.

**Decided in review:**

- The TSA certificate's extended key usage must be timestamping and nothing else, as RFC 3161 section 2.3 requires. The library already refused a multi-purpose certificate, but under the wrong reason.
- `cryptography` is declared as a direct dependency, because the package now imports it.

## The seal retries only what can heal, and the site never calls a future evening missed

**Decided:**

- `seal-daily --attempts 2` retries `HrrrError`, `FetchError`, `TsaError` and `OSError` once, after a 5 minute pause. It never retries when the pause would reach the deadline.
- The site's `through` defaults to the newest evening whose deadline has passed, so a build made before sunset cannot call that evening missed.
- In review the conductor overruled "non-blocking" on a future `--through`. Listing an evening that has not happened as "No forecast was sealed" is a false public statement. Such a `--through` is now a usage error.

**Rejected alternatives:**

- Retrying every failure: a deterministic failure cannot heal, and the pause is billed.
- Running the site workflow after a failed seal as well. The seal runs before sunset, so that build could not yet call the evening missed.

## The camera fit's capture guard votes by frame, with headroom

**Decided:**

- Issue #28 detects the least-squares capture by edge energy and refits without the captured columns. The rejected alternative was a cross-frame continuity filter, which a slow or stationary cloud bank defeats.
- The builder found that one comparison on the mean edge map refuses an existing, correct fit. A strong cloud in one of three frames costs it 65 percent there. The guard was changed to vote by frame, which mirrors the median of the traces, and the conductor accepted that change.
- Review then measured the margin. A capture on slightly wrong terrain, the realistic case at WHTM1's 1.6 pixel residual, loses only about 20.6 percent. The first margin was 20 percent, so it had no headroom. The margin is now 15 percent, with tests that hold a quarter of headroom on both sides.

## The order PRs land in is part of the review

**Decided:** issues #25 and #26 were built in parallel, with each one's files fenced off from the other. When #26 merged after #25, ten of its tests failed: nine sealed or verified without trusting the test authority, which main now requires. The tenth guarded #25's documentation paragraph against #26 editing it, and #25 had rewritten that paragraph. The conductor added `trust_test_tsa` to the two new test modules and deleted that guard. Then it re-gated the merged tree and re-injected #26's blocking mutants on it before landing.

**Rejected alternatives:** landing the reviewed head unmerged and trusting CI on the merge. The reviewed verdict was about a base that no longer existed.

## What broke

- The previous session's collector stopped when that session ended, partway through the evening window. A session takes its hand-run collector with it. This session restarted the collector within one frame interval, so nothing was lost this time. The nightly timer is still the owner's install.
- Two conductor pins were physically wrong, and builders measured them. "The multiple-scattering share rises as the sun sets" is false toward the sun at WHTM1's lowest band: it peaks near sun -2, which an independent Monte Carlo confirmed. Over the whole dome it does rise. And v0 is not a good order-1 oracle at a 1.5 degree view with a thin aerosol layer: it is 11 percent below the brute-force sum, while the Bruneton port is within 3 percent.
- The capture margin was justified from clean-terrain scenes only. The reviewer found the realistic case at the edge of the margin.
- The guard that "the default painting is unchanged by v1" first pinned the SHA-256 of the painting's canonical bytes, computed on macOS. CI on Linux produced different bytes, because v0's floating-point output differs in its last digits between the two platforms, and a hash of floating-point output is not portable. The guard now compares values (within 1e-9 relative) and decoded strip pixels against a fixture made from the base commit. That fixture reproduces the old macOS hashes exactly.
- PR #39's fix round found a 6.6 percent gap between the Bruneton port and the tool's own Monte Carlo at `aod_550` 0.1. At 0.03 the two agree to 0.5 percent. Both were written by the same builder, so only an external model can say which is right. The question is recorded on #34.
- Other sessions kept the host at a load average between 60 and 210 for most of the evening. Teeth runs on the camera-geometry suite took about 45 seconds a row and had to be split. Every gate was read from its exit code and log, never from a pipe.

## Next

The v1 engine landed (PR #39), and its table digest is recorded as the criterion's amendment. Next:

- #35: the magnitude check and the G0 re-run.
- #34: the Prague control. It also has to settle a 6.6 percent gap between the port and the Monte Carlo at `aod_550` 0.1.
- #40: v0 under-reads by up to 5.8 percent at `aod_550` 1.0, and v1 inherits it.

The verdict waits for 5 clear evenings. Every evening so far was overcast. Until the owner installs the timer, frames are collected only when a session spans the window.
