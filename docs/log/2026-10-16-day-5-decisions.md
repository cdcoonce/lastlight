# Day 5 decisions (2026-10-16)

Each decision is listed with what was rejected and why. The conductor made every call; the owner is observer and accountant.

## The pilot came first, and it found the old estimate inconsistent with itself

Before any rule for #55 was written, the conductor fitted every gas estimate the draft would use to the two transmittance spectra the fixtures already hold, and disclosed it as a pilot. The pilot found a defect in the old estimate. It subtracts v1's gas at 550 nm from a total interpolated between channel centres; interpolating the gas the same way and taking Prague's pressure (973.07 hPa, not 970) raises it by 0.0016. That puts the paper's excess at 0.0121, not 0.0137.

- Rejected: writing the rules first and computing afterwards. Day 4's lesson held again: the pilot is what showed which outcomes were already set.

## #55's six-node route is registered as "no test"

The draft registered an envelope over eight gas estimates (aerosol form × ozone column × Rayleigh depth, each fitted or fixed), each widened by twice its fitting error. The second cold read showed three things:

- With each node's aerosol shape free, the amounts carry no information about the gas, and six nodes only repeat the fit (its error falls by exactly √3).
- The drafted rule would read "spans" almost whatever the sky, and its "contributes" was not more likely as the true excess grew.
- The SPCTRL2 ozone stand-in moves single estimates by up to 0.016 while the residuals stay within about a factor of two of the pilot's.

The criterion (#56) says so with the evidence, instead of registering a rule whose outcome was fixed.

- Rejected: landing the envelope rule. A verdict its design sets is not a test.
- Rejected: restoring power with a smaller factor than 2 or by dropping estimates. Either would be chosen with the pilot and the cold read's power table in view.
- Rejected: imposing one aerosol shape across the nodes. Prague's nodes change type from continental clean to polluted, so it asserts what the paper says is not so.
- Rejected: power laws only. The pilot would have said "contributes" by 0.0007, but the power law's low Rayleigh (8 percent under the paper's) belongs to the form, not the data. The criterion discloses that the conductor saw this before fixing the family.

## No six-node fit was run, even as a diagnostic

Its numbers are what #57 needs. Computing them first would make them #57's pilot, as the sun-0 numbers became #49's.

- Rejected: running it because the transmittance is cheap to evaluate.

## #57 is filed, not registered today

The route that may have a test is the paper's Rayleigh with Prague's own ozone, taken from its transmittance aloft. Its pilot must include a power check at the verdict level, which is the lesson of #56, and that is a design of its own.

- Rejected: rewriting #55's criterion into #57 on the day its draft failed.

The conductor ran a synthetic power study of that route as groundwork for #57's pilot. It is machine-local, under `~/.workshop/lastlight/research/2026-10-16-issue57-power/`, and will be disclosed and committed when #57 is registered; its headline is on #57.

- **The route has power.** With the paper's Rayleigh inside a ±1 percent tolerance, P(contributes) rises monotonically with the true excess (50 percent at 0.0106 to 0.0111) and P(not this) falls.
- **The SPCTRL2 stand-in's bias is gone,** because the ozone shape comes from Prague's spectrum aloft.
- **Five things the criterion must register,** because each sets the verdict by design:
  - the Rayleigh tolerance (outside it the verdict follows the paper's Rayleigh);
  - removing the aerosol aloft (ignoring it biases G by +0.0032);
  - an extraction-error term in the envelope (without it the envelope covers the truth in 81 to 92 percent of trials);
  - a tolerance on Prague's transmittance aloft (G moves by about 3.1 times a common error there);
  - one ozone shape at both altitudes.

- Rejected: registering #57 today on the strength of the study. Its design choices are the registration, and they deserve a fresh look.

## Four reads before the criterion landed

Two cold reads of the draft ran in parallel: one on numbers and implementability, one on rule logic. A third, confirming read of the "no test" revision found a false sentence in it. The revision had claimed that without one shared shape only the spectral slopes remain. A smooth tie between the nodes' shapes, or a basis of the paper's species, restores the mechanism; it still falls to the ozone stand-in.

A fourth read checked those corrections and tightened five sentences. It also asked for a positive control on the invariance check, because scaling by a power of two is exact and a no-op would pass.

- Rejected: landing after the confirming read's fixes without reading them. On day 4 a sentence the author added after the last read was wrong, and this time a fourth read was cheap.

## The tool prints the evidence; the cold read's synthetics are quoted, not rebuilt

`report --depth` prints the pilot, the crossings and the invariance checks with their control (#58). The cold read's verdict-level Monte Carlo and its ozone-shape sensitivity are quoted from the read, which is posted on #56.

- Rejected: committing them as tests. Their truths and noise were the reader's choices, and pinning them would register a reader's design as the project's.

## Review of #58

The blind reviewer wrote 63 rows (35 from the conductor's intents, 27 of its own, 1 control). 54 were killed, 53 of them by the predicted test, and the control held. Of 8 survivors, the conductor judged three blocking:

- the printed sentence giving the paper's and the old excess (X11, X12);
- the amount-scaling check, which could be a no-op (X5).

A fourth (I31), the three-copy check's construction, is equivalent in output, because exact copies give 1/√3 either way. It got the same spy test, because the criterion states the construction.

The fix round also closed the reviewer's untrusted-input probes. Fit channels are chosen by exact centre (a consistent +1 nm shift, a dropped channel and descending centres were accepted before), the registered visibilities are checked, and a non-finite condition number prints instead of raising. The conductor re-injected six rows from the committed spec, and the predicted test killed each one.

- Rejected: classifying I31 non-blocking and leaving it. The test is cheap, and the construction is what the criterion registers.
- Rejected: leaving the survivors' stale labels ("expect SURVIVED") in the spec. Each now names the test that kills it, so a later re-injection counts a kill only by its prediction.

## The hazy third counted evening counts, and the haze is recorded, not explained

2026-10-16 was hazy on camera. HRRR read 4.53 percent cloud and the check's `aod_550` was 0.0177, so under the registered rule (cloud at most 10 percent, at least 3 used frames) it is the third counted evening. v1's r_e is 1.025, inside the window. v1 painted nearly what it painted the night before, and the camera's contrast fell, which moved the ratio from 0.43 to 1.02.

- Rejected: excluding the evening as hazy. The counting rule was fixed before any evening, and it is about cloud; adding a haze clause the night it would change the count is the after-the-fact tuning the registration exists to stop.
- Rejected: looking into why the check's aerosol input missed the haze now. That question belongs to the scoring and to a renderer registration (#50), not to the adoption check's coarse magnitude question.
