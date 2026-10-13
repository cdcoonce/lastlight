# Day 2 decisions (2026-10-12, evening session)

This session first checked the first scheduled seal of a new evening and the first evening the capture host's timer collected (`docs/log/2026-10-12-third-evening.md`). It then built three issues in parallel, each in its own worktree with a blind mutation review: the renderer v1 magnitude check and the G0 re-run with v1 (#35), the Prague daylight control (#34), and v0's heavy-aerosol quadrature (#40).

## The magnitude check pins v1's physics, not only its table

**Decided:** `lastlight renderer-check` refuses a verdict in two cases. The first is when the bytes of the v1 table hash to anything other than the digest recorded in the criterion's amendment. The second is when v1's output at three fixed probes differs from the registered values by more than a relative 1e-6. The probes are `aod_550` 0.03 to 0.2, sun -3 to +3, and WHTM1's lowest and highest views. The registered values were computed by the conductor on `main` at `807ed34` and handed to the builder, so the builder never derived its own golden. The table digest is compared against a constant in the check's own module. It is not compared against `lastlight.render.v1.TABLE_SHA256`, which would let a table and its pin change together.

**Rejected alternatives:** the digest alone, which the issue asked for. Criterion definition 8 says that a later change to v1's *physics or* table is a different renderer, and v1's single scattering is v0's code. A change to that code (#40 is exactly one) would pass a digest check and silently change the renderer under test. A SHA-256 of the rendered floats was also rejected: a hash made on macOS failed on Linux CI last session, so the pin is by value with a tolerance.

## v0's heavy-aerosol fix is an explicit quadrature, and the default stays registered

**Decided:** #40 adds `quadrature="registered" | "adaptive"` to the renderer. `"registered"` stays the default for both engines and reproduces `807ed34` bit for bit. Only the daily forecast's v0 painting asks for `"adaptive"`. G0 and the magnitude check keep the registered quadrature by not passing the keyword. Once v1's verdict is in, a v1 with the adaptive quadrature would be a new renderer. It needs its own registration before its own result.

**Decided, on the build's one deviation:** the package version stays 0.1.0. The spec asked for 0.1.1, so the index would separate paintings made before and after. The builder found that `__version__` is stamped into every painting and pinned by 8 existing tests and a fixture, which the build was not allowed to touch. Each sealed forecast already records its code SHA, which separates the two. At ordinary aerosol loads the paintings are also bit for bit the same: adaptive equals registered up to `aod_550` 0.06. The bump moves to the v1 adoption change, which has to bump the version anyway and will update those tests in the open.

**Rejected:** bumping now and editing 8 version-pinned tests in a physics PR, for a label the code SHA already carries.

**Rejected alternatives:**

- Changing the default for both engines. That changes registered v1 at aerosol loads a counted evening can have (the fingerprint's `aod_550` 0.2 probe moves), so the check would refuse, and v1 would need a new registration and 5 new counted evenings.
- Deferring #40 until v1's verdict. The daily forecast would under-read by up to 5.8 percent on a smoke evening in the meantime, for nothing, since the explicit keyword costs the registration nothing.
- Amending the criterion to tolerate the change. Writing an amendment because of a code change we want is what pre-registration exists to stop.

## observe runs retroactively, each session, on the capture host

**Decided:** the capture host's archive is never pruned (about 2.3 MB per evening for both cameras), so each session runs `lastlight observe` there for every evening not yet observed. It then copies the derived `observed.json` files (no frames) to the build machine's private observed directory, which the magnitude check reads.

**Rejected alternatives:** running observe from the collector's systemd unit after each collection. That is a change to an owner-installed unit, so it would be a hands-on ask, and nothing is lost by waiting: the frames keep. Running the check on the capture host itself was also rejected. Its clone is the timer's code and must stay on a landed `main`. The reports are committed from a build session.

## Three builds at once, on a loaded host

**Decided:** #35, #34 and #40 were built in parallel. The partition is by file: #35 owns `src/lastlight/g0/**`, the new check and the CLI; #34 owns `tools/` and the Prague fixture; #40 owns `src/lastlight/render/**` and the painting seam. The README and the docs files do not overlap.

**Rejected alternatives:** building them in series, as the issues are ordered. Their files are disjoint. #40's only interaction with #35 is the registered default, and #35's fingerprint enforces it, whichever lands first.

## The Prague control is coarse, and that is accepted

**Decided:** the Prague control (#34) keeps the tolerances the issue fixed before any comparison: 0.10 per log ratio and 0.02 in u'v'. Its blind review found two limits.

- A Rayleigh exponent of 3 instead of 4 stays inside those tolerances. The existing physics tests kill that mutant, and the magnitude check pins v1's output separately, so it is ruled a resolution limit of the control, not a missing test.
- Removing v1's multiple scattering entirely shows up only at azimuths 90 and 180. Toward the sun (azimuth 0) the control cannot tell v1 from v0. The build log's claim that the control can fail therefore holds away from the sun only.

**Rejected:** a drift pin on v1's measured deviations from Prague. It would turn a coarse external control into a second regression pin of v1, which every future registered renderer would have to re-bless, and the magnitude check's fingerprint already does that job.

## The adaptive quadrature's jump at its trigger is accepted and stated

**Decided:** adaptive gives exactly the registered result until some piece of the sightline exceeds an optical depth of 0.7, and refines from there. The blind review of #40 measured the step across that trigger: up to 4.4 percent in luminance and 1.6 dE76 in a painting row. The build had claimed about 1 percent. The step is the registered quadrature's own error at that aerosol load, which adaptive removes, so it stays. The docstring and the build log now state the measured size.

**Rejected alternatives:**

- A lower trigger. Paintings at ordinary loads would stop being bit-identical to those already sealed.
- Blending the two quadratures across a band of optical depths. That makes the result depend on a second tuning constant, to smooth an error that is better removed.

## A retryable drop is pending, never stored, while it can still be retried

**Decided:** in the magnitude check, an evening whose model run is not yet in the archive (`hrrr_run_missing` or `aerosol_run_missing`) stays pending, with no record, for two calendar days. Only after that is it stored as a final drop. The report stage sees only final records or none. Its pending rule then holds the first-5 slots of every later evening while an earlier one could still turn out counted.

The build first stored such drops and retried them later. The blind review showed what that allows. A later evening takes the slot. Then the retry records the earlier evening as counted, and the verdict flips: KEEP V0 became ADOPT on synthetic records. That is running the check again until it passes, which the criterion's "first 5" rule exists to prevent.

**Rejected alternatives:**

- Making those drops final at once. A model cycle published a few hours late would remove an evening from the check for good.
- Giving the report stage a clock to recognise a drop that can still be retried. The report would then depend on when it was run, and a byte-identical re-run would stop being a property of the cache alone.

## What landed

- **#35 (PR #44): the magnitude check and G0 with v1.** `lastlight renderer-check` reads the three observed evenings and verdicts `NOT YET DECIDABLE (0 of 5 counted evenings)`. Their mean total cloud along the sightline was 99, 31 and 50 percent, so none counts. G0 re-run with v1 still passes, but by far less. 122 of 365 dates (33.4 percent) meet the rule, against 88.5 percent with v0. At the low end of the fit's pixel-scale range it would be 22.5 percent, under the 25 percent bar (reported, not part of the verdict). Multiple scattering takes away most of the day-to-day contrast that single scattering painted. That is the tenfold exaggeration the first comparison suggested, and the reason G0's v0 pass was read in renderer units only. The v0 report is byte for byte unchanged. On the merged `main` the check's report stage reproduces the committed report byte for byte, so the adaptive quadrature from #42 did not disturb the registered v1.
- **#34 (PR #43): the Prague control.** v1 is inside both colour tolerances at sun +10, at all three azimuths. v0 misses both at azimuths 90 and 180. The aerosol gap at `aod_550` 0.1 is not settled, and neither is v1's toward-sun excess in luminance. Both are now #45.
- **#40 (PR #42): the adaptive quadrature.** The cause was the midpoint rule over long, optically thick pieces of the sightline, not the 50 m shells. Adaptive is within 2 percent of brute force up to `aod_550` 1.0 and in the thin layer, and identical to registered up to 0.06. Over the G0 year it changes the lowest band's contrast on 32 of 365 dates, by at most 0.0014.

Each PR had a blind mutation reviewer, a fix round, and the conductor's own re-injection of every blocking mutant from the committed spec before landing: 5 of 5 for #43, 8 of 8 for #42, 14 of 14 for #44.

## What broke

- **pytest's default absolute floor made a 2 percent bar vacuous.** `pytest.approx(ref, rel=0.02)` keeps an absolute tolerance of 1e-12. The spectral radiances graded at deep twilight are smaller than that, so 29 of 173 swept points were graded looser than 2 percent. A mutant 3.9 percent off still passed. Every relative bar in the new tests now passes `abs=0`.
- **A red commit is not proof that its tests stayed honest.** #44's builder reported correcting three of its red tests in the implementation commit. The reviewer counted about 20 changed assertions. None was weakened beyond what its measurement needed, but the PR body undercounted. The fixer's comment corrects it.
- **The host's load average reached about 500** while another project ran. Builders split every gate into chunks of under 10 minutes. Two chunks still ran past the limit and were moved to the background by the harness, and they finished on their own. Every gate was read from its own exit code and log, and the conductor's landing gate, run when the load had fallen, is the authority.
- **The conductor's re-injection script first missed a row** because it matched labels without their `#` prefix. It stopped on the missing row instead of guessing, which is the point of asserting that every anchor and every label occurs exactly once.

## Next

- Check the clock first. Verify each scheduled seal since this one from a fresh `ledger` clone. Then run `observe` on the capture host for every evening not yet observed, copy the outputs to the private observed directory, and run `lastlight renderer-check` (fetch, then report) and commit its report. The verdict needs 5 counted evenings. With the deck seen this week, that may take a while.
- #45: the Prague comparison at sun 0, and the toward-sun luminance excess.
- #31 (G1) once 10 consecutive clear pairs exist. It uses the same input-only clear flag as the check.
- When the check decides: adoption (or not) as its own change. That change bumps the version and updates the version-pinned tests in the open.
