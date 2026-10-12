# The ledger: how to check a sealed day

Every day a scheduled GitHub Actions job (`.github/workflows/seal.yml`, named `seal`) paints that evening's western twilight, seals the forecast and its painting, and commits the result to a branch called `ledger`. This page is how anyone, without trusting the author, checks that a day was sealed before it happened and that nothing has changed since.

## What is in the ledger

`ledger` is an orphan branch: it shares no history with `main` and holds no code. Its root holds one directory, `ledger/`:

```text
ledger/index.jsonl                  one line per sealed day: date, digest, model version
ledger/YYYY/MM/DD.forecast.json     the sealed forecast: canonical JSON, exactly the bytes that were hashed
ledger/YYYY/MM/DD.forecast.tsr      the RFC 3161 timestamp response for the SHA-256 of those bytes
ledger/YYYY/MM/DD.strip.png         the painting: the curve as a picture, no metadata
```

`main` never carries daily data; the seal job only ever pushes to `ledger`.

## What is sealed

The **forecast bytes** are sealed: the canonical JSON in `DD.forecast.json`, hashed with SHA-256, and that digest is what the timestamp covers. The forecast holds the date, the model and code version, the public reference point, every weather input with its source, cycle and the SHA-256 of the exact bytes read, the sky colour curve against sun elevation, the vividness forecast with its spread, and notes saying what was used. Nothing in it is a clock time; everything is referenced to sun elevation.

The **painting** is sealed through the forecast. The forecast carries `strip_sha256`, the SHA-256 of `DD.strip.png`, so the one timestamp covers the numbers and the picture. If the PNG were swapped afterwards its digest would no longer match the sealed field. A forecast with no painting has `strip_sha256` set to null and no PNG beside it.

## Verify a day

You need the verifier (this repository's code) and the `ledger` branch side by side:

```sh
git clone https://github.com/cdcoonce/lastlight.git code
cd code && uv sync && git fetch origin ledger && git worktree add --detach ../ledger-checkout origin/ledger
cd ../ledger-checkout
uv run --project ../code lastlight verify 2026-10-12 --ledger-dir ledger
```

It prints `PASS`, or `PASS (strip verified)` when the forecast names a painting and the PNG in the ledger hashes to it, or `FAIL: <reason>`. The exit code is 0 for a pass and 1 for a fail. Every check is made again from the files:

1. The SHA-256 of `DD.forecast.json` equals the message imprint inside `DD.forecast.tsr`.
2. The file is a valid forecast for that date, in canonical form (so a re-formatted copy fails).
3. The timestamp was issued strictly before geometric sunset at the public reference point on that date. Sunset is computed from the date alone by `lastlight.geometry`; the reasons `verify` prints give the lateness as a duration, never a clock time.
4. If the forecast names a painting, `DD.strip.png` exists and hashes to `strip_sha256`. A painting that is missing, changed, or present without being named fails.

You can also check the pieces by hand: `sha256sum` of the forecast file is the digest in `index.jsonl`, and `openssl ts -reply -in DD.forecast.tsr -text` prints the timestamp response.

## What the timestamp proves, and what it does not

An RFC 3161 timestamp is a signed statement by an independent timestamp authority that a given hash existed at or before the time it names. Here it proves that the exact forecast bytes, and through `strip_sha256` the exact painting, existed before sunset on that date. The seal job's public run log and the commit message `seal: YYYY-MM-DD <first 12 hex of the digest>` on the `ledger` branch are two further witnesses, which agree or do not.

It does not prove the forecast is good: whether it was right is what the scoring half of the project measures after dusk. It also does not, today, prove that the timestamp response is authentic: `lastlight verify` checks the imprint and the time but does not yet validate the authority's signature chain (issue #25 will add signature verification), so until then it trusts the response as delivered. The signature can be checked independently with `openssl ts -verify` against the authority's published certificates.

## Missing and late days

A day is sealed only if the job ran before sunset. If the job fails, or runs after the sun has reached the horizon, that day has no files and no index line, and it is recorded as missing, never back-filled. Running the job twice for a sealed day changes nothing. `lastlight verify` for a day with no seal prints `FAIL: no sealed forecast for <date>`.

## Missed evenings

A day can have no seal for reasons nobody controls: the HRRR bucket or the timestamp authority was down, the workflow was dropped or delayed, or it was switched off (next section). Two things keep that honest.

**One retry, in the command.** The workflow runs `lastlight seal-daily --attempts 2`. If the first attempt fails for a reason a pause can cure (the weather service, the timestamp authority, the disk), the command waits `--retry-pause` seconds (300 by default) and tries once more, but only while the pause still ends before the sun reaches the horizon at the reference point. The second attempt starts from a fresh reading of the clock, so the refusal to seal late applies to it exactly as it does to any run. A failure that would only repeat (a refusal, a reduction that cannot work, a seal that does not verify) is not retried, and the exit code is always the last attempt's. The run log names each attempt's failure.

**The site says so.** `lastlight site build` reports on every evening up to `--through YYYY-MM-DD`, by default the newest evening whose sunset at the reference point has passed when the site is built, and no later: an evening that has not ended is refused, since the site would call it missed. An evening between the first sealed one and that date that has no line in `index.jsonl` is missed. The archive lists each one as "No forecast was sealed for D." (D is a date, never a time), and the home page leads with that sentence when the newest evening reported on is missed and newer than every sealed one, while still showing the newest verified painting under its own date. A missed evening has no page and no painting. It is never back-filled, and a missing seal is not a forecast: nothing downstream may treat it as one. An evening sealed too late does have an index line, so the site lists it as sealed and not verified instead.

The site is rebuilt after a successful seal, which runs before sunset, so a failed evening is first called missed by the next day's build.

## The 60-day rule for scheduled workflows

GitHub turns off scheduled workflows in a public repository that has had no repository activity for 60 days (see [the `schedule` event in GitHub's documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule)). The daily `seal` workflow (`.github/workflows/seal.yml`) is scheduled, so it is subject to the rule. A workflow that has been switched off does not fail; it simply stops running, so there is no red run to notice, the `ledger` branch stops growing, and the site shows every later evening as missed.

GitHub's pages state the rule but do not define what counts as activity. Whether the seal job's own pushes to the `ledger` branch, made by the Actions bot, count is therefore unverified, and so is whether work on a branch other than the default branch counts. Scheduled workflows run only from the default branch, `main`, so the safe assumption is that a person's commit to `main` is activity and the bot's pushes to `ledger` may not be. Do not rely on the daily seal to keep its own schedule alive.

To turn the workflow back on, open the repository's Actions tab, choose `seal` in the sidebar and press the button that enables it, or run `gh workflow enable seal.yml -R cdcoonce/lastlight` ([GitHub's guide to disabling and enabling workflows](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/disable-and-enable-workflows)). Then run `seal` once by hand from the same tab. A manual run seals only while the evening is still ahead; after the sun is at the horizon it refuses, by design, and that evening stays missed.

Until scoring exists, the conductor's build board records each evening's outcome by hand: sealed, retried or missed. A retried evening shows in the run log as a first attempt that failed and a second that sealed.
