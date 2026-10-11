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

It does not prove the forecast is good: whether it was right is what the scoring half of the project measures after dusk. It also does not, today, prove that the timestamp response is authentic: `lastlight verify` checks the imprint and the time but does not yet validate the authority's signature chain (a follow-up), so until then it trusts the response as delivered. The signature can be checked independently with `openssl ts -verify` against the authority's published certificates.

## Missing and late days

A day is sealed only if the job ran before sunset. If the job fails, or runs after the sun has reached the horizon, that day has no files and no index line, and it is recorded as missing, never back-filled. Running the job twice for a sealed day changes nothing. `lastlight verify` for a day with no seal prints `FAIL: no sealed forecast for <date>`.
