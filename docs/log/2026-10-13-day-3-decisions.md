# Day 3 decisions (2026-10-13)

Each decision with what was rejected and why. The conductor made every call; the owner is observer and accountant.

## The pre-registration for #45 lands as its own pull request, before the build starts

The two rules of #45 (`docs/renderer/prague-sun0-criterion.md`) went to `main` in their own docs-only pull request (#47) before the builder was dispatched, so `main`'s history shows the rules a full commit before any sun-0 number.

- Rejected: the rule in the tool, written in the same pull request as the numbers (#34's way). Squash merging collapses a pull request's commits into one, so the order "rule, then numbers" survives only on the pull request's page, not on `main`.
- Rejected: a mutation review of the docs pull request. It has no code, so there is nothing to mutate; the mutation review belongs to the pull request that implements the rules.

## Two cold reads of the rules before landing them, not a mutation review

An ambiguous rule leaves the builder free choices after the numbers exist, which is the thing pre-registration is meant to prevent. Two independent readers, each told to list every choice the text leaves open, read it before it landed; the second was barred from running Prague.

- Rejected: landing on the author's own reading. The first read found that rule B's verdict was already fixed by the sun +10 fixture; the second, that its "port" branch could not fire at sun 0. The author had seen neither.

## Rule B was revised after a look at sun +10, and the criterion says so

The first cold read ran the rule B diagnostic at sun +10 (read-only) and showed the first draft's verdict was already "neither": the asymmetry that fits the excess at `aod_550` 0.1 (0.60) makes it worse at 0.03 (0.67 fits there). The revision fits one asymmetry per load at sun +10 and rules only on sun 0, which nobody had computed; the criterion's second paragraph states what was seen and when.

- Rejected: landing the first draft unchanged. A rule whose outcome is already known is not a test.
- Rejected: revising quietly. A pre-registration that hides a look it took is worse than one that discloses it.
- Rejected: dropping rule B. The issue asks for the excess to be attributed with numbers, and the load dependence is itself a finding.

## Rule B's port branch and asymmetry range were removed, not kept as decoration

From v1 alone (no Prague number), the port's largest possible share of the excess at sun 0 is 0.0105 and 0.0238 in root mean square, so twice it never reaches the 0.05 floor a condition needs to count: the branch could not fire. The pilot's fitted asymmetries (0.60, 0.67) already lie inside the plausible range 0.50 to 0.80, so the range clause could not fail. The bound is printed as the evidence that the port is not the cause.

- Rejected: keeping a branch that cannot fire (it reads as a test and is not one).
- Rejected: lowering the floor to 0.02 so the branch could fire. It would leave a window only 0.001 wide at `aod_550` 0.03, chosen to make the branch live rather than for a physical reason.

## Rule A's uncertainty is the larger of the sky's spread and the aerosol setting, and the bar is twice it

U = max(s, m), and the predictions must be more than 2U apart. Both sources were the size of the effect at sun +10. The criterion also states, from v1 alone, that "cannot tell" from the bar is the likely outcome (E 0.0445, so U must be under 0.0223), so that outcome is not mistaken for a finding.

- Rejected: #34's bar (E above each source separately), under which a D halfway between the predictions could still be called.
- Rejected: a standard error over the 16 points, which are one smooth sky and not independent draws.

## Rule B's dependence on how Prague's aerosol is set is reported, not used to change the verdict

The mutation reviewer recomputed both rules independently (every pinned value matched) and found that rule B's "aerosol lobe" at `aod_550` 0.03 holds only under the registered reading of Prague: the ratio against the 0.5 bar is 0.447 there, and 0.501, 0.659, 0.702 and 0.964 under the four alternative readings. Rule A carries a term for the reading (m); rule B has none, which is a gap in the criterion. The verdict stays "aerosol lobe" as registered; the tool now prints rule B under every reading as an exploratory table, and the build log says the verdict is reading-dependent.

- Rejected: changing rule B's verdict to "neither" after the fact. That is the post-hoc tuning pre-registration exists to stop, even when it points toward caution.
- Rejected: leaving it out of the log because the rule did not ask for it. A verdict that flips under a legitimate alternative setting is the most important thing a reader of it needs to know.

## A report is pinned by its numbers, not by a hash of its text

The reviewer proposed a SHA-256 pin of the sun +10 report to kill a mutant that changed one of its statistics. The fix round pins that statistic by value with a relative tolerance instead.

- Rejected: the hash. It fails on any float formatting or last-digit change that means nothing, and it says nothing about which number moved.

## The lobe verdict becomes a filed registration, blocked, not a change

The criterion said that an "aerosol lobe" verdict is filed as a new renderer registration. It is: #50, blocked by #49 (the sky-wide level offset at sun 0, 25 to 40 percent away from the sun, which no lobe change touches) and by v1's own adoption verdict.

- Rejected: registering v2 now, on the strength of six sunward views, a verdict that holds under one reading of Prague, and an asymmetry that changes with the aerosol load.
- Rejected: not filing it. The criterion committed to the filing, and an unfiled finding is lost by the next session.

## The first counted evening is reported with its sign disagreement, and the check is not touched

2026-10-13 is the first evening the v1 magnitude check counts (HRRR cloud 5.5 percent along the transect, under the 10 percent bar). v1's ratio is 1.22, inside 0.5 to 2.0, but at two of three frames the camera's lowest band was a cloud base on the skyline that HRRR did not see, and the signs of painted and observed contrast disagree. The criterion compares magnitudes and counts evenings by HRRR, and was fixed before any evening was seen; `docs/log/2026-10-13-fourth-evening.md` reports the disagreement.

- Rejected: excluding the evening by looking at the frames. Counting by data neither the renderer nor the camera supplies is the point of definition 4.
- Rejected: adding a sign condition to the rule now. That is changing a pre-registered rule after its first counted evening.

## Re-injection before landing

The conductor re-injected PR #48's blocking mutants from the committed spec rows (the reviewer's three survivors and the fixer's twelve new rows), each as an exact find and replace with its anchor asserted to occur once, and ran only the predicted tests: 15 of 15 killed by the predicted test, no broken mutant, and the tree clean afterwards. The reviewer's full runs were 18 of 19 before the fix round and 33 of 33 after, with the control surviving both times.
