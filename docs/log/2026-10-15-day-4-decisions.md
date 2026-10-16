# Day 4 decisions (2026-10-15)

Each decision with what was rejected and why. The conductor made every call; the owner is observer and accountant.

## The clearest evening is not counted, and the rule is not touched

2026-10-14 was the first evening clear on the ridge from the first frame to the last, and the v1 check lists it as `not considered: 2 used frames`: WHTM1's feed skipped two 15-minute slots in a row, and definition 3 of the check's criterion considers an evening only with at least 3 used frames.

- Rejected: lowering the minimum to 2 now. The minimum was fixed before any evening was seen; lowering it the day it excludes the evening we would most like to count is the after-the-fact tuning the registration exists to stop.
- Rejected: filling the gap with ESMO1's frames. ESMO1 looks southwest, not along WHTM1's sightline, and the check is defined on WHTM1.

## The #49 rules rule at sun -1 and +1, and the sun-0 numbers are published as a pilot

Every candidate in #49 except Prague's fit could already be computed at sun 0 and sun +10 from the committed fixtures and v0/v1, so the conductor computed it before writing any rule and the criterion prints it, labelled a pilot. The rules rule only on Prague at sun -1 and +1, which nobody had computed when the criterion landed.

- Rejected: ruling at sun 0. The outcome was already known.
- Rejected: sun -0.5, which the issue proposed. It is a node of Prague's sun axis but not of v1's table; v1 would interpolate in the sun there, and the curvature test would carry that interpolation.
- Rejected: writing the rules without computing the pilot. The pilot is what showed the aerosol-amount rule could not be a test (below); a rule drafted blind would have shipped it.

## Two of #49's four candidates are registered as "no test", with their evidence

The issue names four candidates. Evaluating every branch of the first draft on the pilot, and two cold reads of it, showed that two of them have no test Prague can give, and the criterion says so instead of registering rules that could not fire.

- **The aerosol amount.** The matched-depth reading removes 0.052 of the offset at (sun 0, 0.1), but the tool's depth estimate carries about ±0.01 (#34's figure, never derived) against a gap of 0.014, which puts the amount's part anywhere from 0.014 to 0.090, on both sides of any bar. Cold read 1 estimated the error at ±0.003 to ±0.005, under which the amount might clear its bar; choosing the error after seeing that would choose the verdict.
- **The aerosol profile.** The first draft tested it by the offset's growth when the load doubles. Both cold reads showed the premise false in v1's own sky: an aerosol-made change grows ×0.80 to ×1.41 from 0.1 to 0.2, changes sign between sun +1 and +10, and a constant extra amount makes a part that shrinks with the load. The rule could never say "carries", nor "contributes" at sun +1.
- **The observer and ground.** The only reading that moves Prague's observer puts it under 340 m more air and aerosol. v1 given that slab's aerosol moves by -0.068 and -0.089 at sun 0, against the reading's -0.068 and -0.096: the reading measures the slab, not the ground, so its "not this" was set by physics before any ruled number. Raising Prague's ground could only darken Prague, so the geometry can hide the offset, not cause it.

- Rejected: keeping the draft's rules because the issue asked for one per candidate. A rule that cannot fire, or whose verdict two fixed numbers decide, reads as a test and is not one.
- Rejected: a bar of twice the matched reading's error on the amount. It exceeds the amount whenever the gap is under 0.02, so the branch could never fire.
- Rejected: an albedo-0 reading for the ground. It bounds the ground's reflection, not the slab.

## Rule 3 calls a fault only when both observer heights show it, above a floor of 0.05

The test for a fault in Prague's fit at its sun-0 node asks whether sun 0's offset sits off the line through sun -1 and +1. Cold read 1 found a geometric kink that is not a fault: v1's ground observer has its horizon at sun 0, Prague's 340 m observer at -0.59 degrees. Under Prague's altitude-0 reading the horizons coincide (at the cost of the slab), so a fault is called only if the bend shows under both readings. The floor is twice the largest bend any smooth difference the cold reads could model puts on the offset (0.025, from the aerosol's scale height); the noise bar at 0.03 is 0.014.

- Rejected: the primary reading alone (its horizon is not v1's), or the altitude-0 reading alone (it carries the slab).
- Rejected: the noise bar alone (it would read a smooth bend as a fault).
- Rejected: a floor of half v1's own bend at each load (0.050 and 0.027): the question is the size of a fault, which does not depend on the load, and 0.027 sits on the smooth bends computed at 0.1.
- Rejected: a ladder of suns with a fitted curve (more runs, and a curve family chosen after the fact).

## The noise bar is twice the inverse-visibility shift, and nothing else

- Rejected: #45 rule A's m, which included the matched-depth reading (its own error is larger than what it measures).
- Rejected: a share floor such as a fifth of the offset. The pilot's amount share at sun 0 was 0.195, so any share floor would have been chosen with the pilot in view.

## A third, confirming cold read before the criterion landed

Both cold reads had read the first draft in parallel, and the revision changed two of its four rules, so the text that would land had not been read by anyone but its author. A third reader, barred from the Prague binary, read the revision. It found no blocker and asked for disclosures: rule 1's "not this" cannot fire in practice (its live outcomes are carries, contributes and cannot tell), "cannot tell" is a realistic outcome for rule 1 with no fault anywhere, and rule 3 cannot tell a fault in Prague's fit from a kink in v1's own table at its sun-0 node. It also questioned a refraction confound in rule 3's floor; the conductor checked both models instead of arguing it: v1's sun is geometric by design, and Prague's source builds the sun's direction from the elevation with no refraction term, so refraction does not enter a comparison between them. The criterion landed in its own pull request (#52) with all three reads posted on it.

- Rejected: landing on the two parallel reads of the first draft. The rules that changed most were the ones no reader had seen.
- Rejected: keeping a refraction term in the floor's list on the reader's say-so, or dropping it on the author's. Reading the model's source settled it.

## The rule governs where the criterion's own prose misread it

After the third cold read, the conductor added a sentence to the criterion's prediction paragraph: if sun +1 carried and sun -1 only contributed, rule 1 would read "cannot tell". The rule-level verdict the same file registers says otherwise: branch 2 is "every condition carries or contributes", so that mix reads "contributes". The builder found the contradiction, followed the rule, and wrote a test that says so. It did not matter for the result (all four ruled conditions contribute, and the offset at sun +1, 0.26 and 0.25, is far from ozone's carrying bars of 0.17), but it is an error in the registered text, and it is recorded here and in the build log rather than edited away.

- Rejected: amending the criterion after the ruled numbers exist. The rules are what was registered; a sentence describing them wrongly is corrected in the record, not in the registration.
- Rejected: implementing the prose. The rule is the definition the verdict is taken from; the prose is the author's reading of it.

## The command-line report is pinned to the report function's output, not to stored text

The blind reviewer of #53 found three behaviours the pins claim and no test graded: `report` could gain a trailing newline, the ozone engine could hard-code its table nodes, and a verdict could be taken on rounded values. The fix round added a test that the command's output equals what the report function returns, byte for byte; a test that the engine asks the table for the nodes v1's domain names (a spy, because a log-linear table gives the same value for any node pair on its line); and a test where rounding to four places would flip a verdict.

- Rejected: a stored copy of the report text, or a hash of it. Day 3 decided reports are pinned by their numbers; comparing the command with the function catches a stray newline without freezing any formatting.
- Rejected: leaving the hard-coded-nodes mutant as "equivalent". It is equivalent only while v1's table keeps its nodes; the pin exists so that a new table cannot silently disagree with the engine.

## Re-injection before landing

The conductor re-injected the three blocking mutants (X13, X25, X26) and rule 3's two core rows (its floor and its two-reading minimum) from the committed spec rows, each as an exact find and replace with its anchor asserted to occur once, running only the predicted test: 5 of 5 killed by the predicted test, no broken mutant, the tree clean afterwards. The reviewer's full run was 46 of 50 rows killed before the fix round and 49 of 50 after, with the control surviving both times.

## The second counted evening falls outside the window, and nothing changes

2026-10-15 counted (HRRR cloud 0.0 percent), and v1's ratio r_e is 0.432, under the rule's 0.5. With the first counted evening's 1.22, two of the five the verdict needs are in.

- Rejected: computing a provisional median, or reading anything into v1 from two evenings. The verdict is taken once, on five.
- Rejected: looking at why v1's contrast was weaker than the camera's on this evening now. That question belongs to the scoring and to a renderer registration (#50), not to the adoption check, which asks only the coarse magnitude question it registered.
