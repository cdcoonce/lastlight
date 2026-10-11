# Day 1: conductor decisions

The conductor's calls on day 1, each with what it rejected. Slice-level choices are in each slice's own entry.

## Reviews interrupted by a restart

**Decided:** the adversarial reviews already posted on #19 (HRRR ingest) and #21 (collector) stand. Both were run by an independent reviewer at the exact PR heads that were later landed, with a full teeth ledger (71 of 79 and 47 of 49 mutants killed, every survivor on an unnamed or unreachable guard). Only #22 (camera geometry), whose review had posted nothing, was reviewed again from scratch.

**Rejected alternatives:** re-running all three reviews. The restart lost the conductor's copy of the verdicts, not the verdicts; re-reviewing an unchanged head buys nothing.

## HRRR: two survivors fixed before landing

**Decided:** before #19 landed, two of its reviewer's non-blocking findings were closed test-first: the decoder's forecast hour was unpinned (the only fixture is hour 0, while every forecast for a coming evening reads a later hour), and the whole-field finite check was unpinned. Both mutants were re-injected by the conductor and went red on the new tests. The transect command also takes its reference point from `lastlight.site` instead of a copy.

**Rejected alternatives:** landing as reviewed and filing follow-ups. The hour survivor sits on the path every daily forecast takes.

## The collector's evening date

**Decided:** keep the collector's folder date as the local mean solar date from longitude (the `lastlight.geometry` convention), although issue #15's clarification said the America/Phoenix civil date. At the cameras' longitude the two differ only in the half hour after civil midnight, and the polling window closes hours earlier, so every frame lands in the same folder either way. One convention across the collector, the forecast and the observation slices is worth more than matching the wording.

**Rejected alternatives:** the civil date through a time-zone database (an extra moving part with no observable difference for any polled frame).

## Where the painting lives, and sealing it

**Decided:** the `ledger` branch holds only `ledger/`, and each sealed evening's strip sits beside its forecast as `ledger/YYYY/MM/DD.strip.png`. The forecast gains a nullable `strip_sha256`, and `verify` checks the strip against it. Schema version 1 is changed in place because nothing had been sealed yet.

**Rejected alternatives:** a separate top-level `strips/` tree as issues #8 and #9 described (one evening's sealed artifacts split across two roots, and `verify` would need both); an unsealed strip (the claim "painted before it happens" would then cover only the numbers).

## The timestamp's signature is its own slice

**Decided:** `verify` checks that the timestamp covers the sealed bytes and precedes the evening, but not yet the timestamp authority's signature, so a hand-made `.tsr` would pass. That is filed as #25 rather than held against the daily seal. Every response in the ledger comes from the authority itself and is genuinely signed, so once #25 lands it verifies every evening sealed before it, and nothing is lost by sealing now.

**Rejected alternatives:** holding the first seal until signature checking exists (costs evenings and buys nothing the retroactive check does not).

## Actions minutes and the site

**Decided:** every workflow is one job with the smallest trigger set that works: the daily seal runs once, and the site deploys after a seal or a change to the site's code. Public repositories cost nothing in dollars, but their minutes still count against the owner's shared included allowance, which ran out in September and blocked private repositories. A push made with `GITHUB_TOKEN` does not start other workflows, so the site workflow listens for the `seal` workflow's successful completion (`workflow_run`) instead of a push to `ledger`, and the seal job keeps only `contents: write`.

**Rejected alternatives:** a separate site job on every push to `main`; relying on `on: push` to the `ledger` branch (it would never fire for the bot's seal); the seal job dispatching the site workflow itself (more privilege for the same effect, and with `workflow_run` as well it would deploy twice).

**Decided:** GitHub Pages is turned on by the owner, not by an agent: it is a repository setting that publishes a site. Until then the site workflow builds and privacy-scans the site and skips the upload.

**Rejected alternatives:** enabling Pages through the API from an agent session; a workflow that fails on every run until Pages exists.

## Contrast bands that fit the camera

**Decided:** tier A's contrast bands are pixel rows above each camera's frozen skyline, defined per camera in the registry, not elevations in degrees. The camera-geometry review measured WHTM1's sky at about 0.6 to 5.3 degrees: the design's example contrast (2 degrees minus 20 degrees) cannot be built, and 10 degrees and above are never in frame. The skyline is pinned in pixel space to 1.65 px, but the scale is not identified (27 to 45 px per degree inside the position prior, more beyond it) until the sun crosses the frame in spring. Rows make every observation exact and repeatable across refits. G1 needs no geometry at all, and only the forecast's mapping from rows to elevation carries the scale uncertainty, reported at both ends of its range. Light-dome exclusions are column ranges for the same reason. G0 counts the golden, early and mid sun bands and reports late without counting it, because tier A scores sun +6 to -6.

**Rejected alternatives:** degree bands up to 20 degrees (never observable); degree bands inside 2 to 5 degrees (their rows would move by up to half at every refit, so observations from different fits could not be compared); counting the late band in G0 (it would certify variation the camera is not scored on); dropping WHTM1 for a wider camera (it is the only open, west-facing camera in the metro).

## The camera's geometry is honest about what it does not know

**Decided:** the frozen WHTM1 fit stays provisional and says so in fields a program can read: a `provisional` flag, ranges for azimuth, field of view and pixel scale, and the gap between the operator's stated azimuth and the fitted one. The review found that position, field of view and azimuth trade off along a valley that a coarser grid hid. Each fit stage must be shown to change the answer on a scene where it should, or be removed.

**Rejected alternatives:** widening the position box until the residual stops falling (the operator says the camera is in Avondale, and a better residual outside the prior is the degeneracy, not a location); researching the exact mount to break the tie now (the sun crosses the frame from about April and settles the scale independently).

## The privacy gate covers the docs

**Decided:** CI runs the full gate on every pull request and every push to `main`, docs-only changes included. The scaffold had skipped `docs/**` and Markdown to save minutes, which meant the privacy scan never saw the build logs, the files where a clock time is most likely to slip in. A docs pull request costs about one billed minute, and there are few of them.

**Rejected alternatives:** a second, privacy-only job for docs paths (a second job definition for the same minute); keeping the skip and trusting the local gate (the CI scan is the one that cannot be forgotten).

## Frames for the second evening

**Decided:** the second evening's frames come from a collector the conductor ran by hand for the session, into the same private archive as the first. The nightly timer is the owner's install (runbook `docs/runbooks/collector.md`); agents never install timers.

**Rejected alternatives:** skipping the evening until the timer exists (the cameras keep no archive, so a missed evening is lost for good).

## What broke

- The lander refused #21 once: its patch changed across the base refresh. The cause was benign. The seal slice had added two README lines directly above #21's, so the hunk's context moved while its content stayed the same. Read the merge, then accept the delta.
- The review of #21 reported the branch "0 behind main". That was true when written and stale by the time of landing: the seal ledger landed in between. A verdict's base is part of the verdict.
- One of the conductor's own spot-check mutants did not parse, so its "red" was a collection error, not a kill. It was redone as a valid mutant, which went red on the predicted tests. A red suite proves nothing until you know it went red for the reason you meant.
- The host's load average passed 100 while two real-data geometry fits and three test suites ran at once. No test flaked, but every gate in this window was read from its exit code and output file, never from a pipe.

## Next

Land the almanac site (#9) and the camera geometry (#16), then build the observed contrast profiles (#17) on the row bands above, then G0 (#10). The owner's two hands-on items are batched together: installing the collector's nightly timer, and turning on GitHub Pages.
