# Day 6 decisions (2026-10-17)

Each decision is listed with what was rejected and why. The conductor made every call; the owner is observer and accountant.

## The seals were verified from a fresh ledger clone, all seven public evenings

The scheduled seal for 2026-10-17 sealed on its first attempt. A fresh clone of the `ledger` branch verifies 2026-10-11 to 2026-10-17 `PASS (strip verified)`, each with exit 0.

- Rejected: verifying only the new evening. Re-verifying all seven costs seconds and would catch a rewritten ledger.

## #57's error carries an error shared by the six nodes, at both altitudes, at the scale of what the six-node fit leaves

Prague's transmittance is a rank-12 reconstruction from one singular value decomposition per altitude across every configuration (supplement §2). At 15 km the six visibility nodes barely differ, so a reconstruction error there is likely to be shared by the nodes: averaging cannot reduce it and the node-to-node scatter cannot see it. The criterion carries it at s_k, the residual scale of the six-node fit, at 15 km and at 0.34 km.

- Rejected: the six-node fit's SE alone (the day-5 estimator). Its envelope missed the truth in up to one trial in five.
- Rejected: the extraction's noise averaged over the nodes, with no shared tolerance. More power (P(contributes) at the paper's excess 0.94 against 0.30 at rms 0.0003 in the scratch comparison), but it assumes an averaging a shared reconstruction need not allow.
- Rejected: the supplement's dataset-wide maximum error, 0.014, as the tolerance. It bounds the worst point of every configuration, altitude and distance; as a tolerance it makes the verdict "spans" whatever the sky.

## The first power grid was stopped to add the shared term at 0.34 km

The draft first carried the shared tolerance aloft only. The six-node fit's SE also treats its 60 values as independent, so it divides a shared error at 0.34 km by √6 that it should not. With the aerosol guard active, that term could widen the SE by about a fifth. The grid was stopped and rerun with the term.

- Rejected: disclosing the omission with an estimate (about 4 percent from the day-5 study's gains). That estimate holds only when the two-point law is exact; in the guard regime it is several times larger.

## The power check is committed with the criterion and rules exactly as registered

`tools/issue57_power.py` runs steps 1 to 6 on synthetic truths: each sun's unrounded crossing and the verdict line, "mixed" included. It also prints the comparison of the registered error with the two rejected forms on the same trials, so the disclosure has its instrument.

- Rejected: committing the scripts with the build, as the day-5 note on #57 planned. #57's acceptance asks for the rules, the power check included, on `main` before any six-node or high-altitude number exists.
- Rejected: one crossing of 0.0095 for all suns (the day-5 study's). The registered rule rules per sun.

## The verdict compares the envelope with #55's linearly scaled crossing

- Rejected: carrying the envelope's ends into Prague for an exact Δ_M. The power check could not reproduce that verdict, and inside the 40 to 59.4 km segment the two agree to about 0.0003 in the excess at the crossing.

## The ozone at 380 and 420 nm gets an allowance tied to the ozone at 550 nm, over 0, 1 and 2 times SPCTRL2's log-linear ratio

Cold read 2 found that the paper's ozone (Gorshelev, 223 K) absorbs at 420 nm and weakly at 380 nm, where SPCTRL2's table is 0. The extraction read it as aerosol aloft, which pushed the excess up by about 0.0007 and gave false "contributes" at small residual scales. Neither s_k nor s_15 could see it. The day-5 study's "weak tail, not material" had used a tail that fell from 380 to 420 nm, shaped like aerosol.

The first form continued Prague's own O(460) and O(500) log-linearly after one pass. The rerun power check showed it failing at its own boundary: a truth at twice the continuation gave false outcomes in 51 of 1,950 trials at rms 0.0001. The first pass reads the tail as aerosol, so the continuation comes out short. Iterating to a fixed point did not cure it, because a tail continued from 460 and 500 nm trades off with a flat aerosol and the fixed point is not unique. Tied to Ō instead, the passes settle in at most 12 steps. The third grid showed no false outcome for tails of 0.5, 1 and 2 times.

- Rejected: the continuation from Prague's own O(460) and O(500), in one pass or iterated (above).
- Rejected: the reader's {0, 1} envelope. The continuation's shape is not known to be an upper bound: the reader's own range from memory runs to about 1.4 times it.
- Rejected: fetching Gorshelev's table to set the tail. It would make the rule rest on a table the paper's channels may band-average, and it would turn this issue into "What would tell instead" (the paper's own gas from its cross-sections). That route is worth its own issue.
- Rejected: dropping the 380 and 420 nm channels from the extraction. With no channel left free of ozone, the aerosol aloft could not be removed at all.

## The 15 km pressure is the U.S. Standard pressure at geometric 15 km

Prague's altitudes are geometric (both reads). 120.45 hPa is the pressure at geopotential 15 km; 121.11 hPa is the pressure at geometric 15 km.

- Rejected: keeping 120.45 hPa and saying so. The effect is small (−0.00006 in g), but the number would be wrong for no reason.

## Both reads' findings were fixed; the grid ran three times in all

The second run carried every fix; the third replaced the tail form that failed in the second. The rig now also prints the guard share, the median width, and every form's false outcomes per cell. Falsity is counted per sun, and the variants run at rms 0.0001. Cold read 1 showed that 1e-9 agreement fails for a correct implementation that orders its arithmetic differently (4e-8), so the agreement is registered at 1e-6.

## The criterion took six reads before it landed, and #61 was filed

Cold reads 1 and 2 ran in parallel on the draft. Read 3 was the confirming read of the revision. Reads 4, 5 and 6 each read the previous round's corrections, and the corrections after read 6's non-blocking points went into the head commit, as day 5 did with its last read. Each round found less. The rounds' blocking findings were the ozone tail (read 2), the condition's scope, the design's floor and the refusal list (read 3), and two over-reaching sentences (reads 4 and 5). The paper's ozone table, Gorshelev's at 223 K, turned out to be published, so its route is filed as #61.

- Rejected: landing after read 3 with the corrections unread. The project's rule is that corrections get a read, and two of the three later rounds found a false sentence.
- Rejected: fetching the table and folding it into #57. It is a different test, with its own pilot (now #57's verdict).

## The verdict stands as registered: contributes, at all four suns

The rule gives g_lo 0.011984 against crossings of 0.009444 to 0.009527, and g_hi 0.019657. A diagnostic run after the verdict showed 380 nm reading low at both altitudes. At 15 km the residual less the Rayleigh is negative, so the aerosol aloft could not be removed. The six-node fit's residual is shared by the nodes in every channel, which is what the shared tolerance was added for. Both readings of the 380 nm deficit bias the excess down: a shared Prague error leaves the aerosol aloft in the shape, and a low Prague Rayleigh behaves like the power check's dr_m2 cells. So the deficit does not explain "contributes". The build log reports it beside the verdict.

- Rejected: reading the 380 nm deficit as breaking condition 1 and withholding the verdict. The deficit is confined to 380 nm (420 nm is consistent with the Rayleigh). Its size is within condition 4's scale, and its direction is away from the outcome.
- Rejected: dropping the 380 nm channel after the fact. That would be a change made with the verdict in view.

## The reviewer's breach of the binary bar is disclosed, and the tests now stub evaluate

The blind reviewer's first form of row I45 let `report --aloft` reach the real Prague binary during tests, a few times. Nothing was recorded, and the registration had landed hours before. The fix round added an autouse stub that makes `evaluate` raise in the #57 tests, so no test reaches the binary.

- Rejected: leaving it at a note. The gap would recur for the next reviewer on a machine where the binary is built.
