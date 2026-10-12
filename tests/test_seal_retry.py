"""The daily seal retries once, in Python (issue #26).

A transient failure (the HRRR bucket, the timestamp authority, the disk) must not cost the
evening, and a retry must never produce a late seal. Both promises live in
`lastlight.daily.seal_with_retry`, which takes an injected clock and an injected sleep, so no test
here ever sleeps. The loop is tested on scripted attempts; the command is tested at the
`seal_daily` seam and end to end on the recorded HRRR crop; the workflow is checked for the one
flag it must pass.

Nothing here spells a clock time: "now" is the geometric sunset at the reference point plus or
minus a duration, as in `test_daily.py`.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import shlex
from pathlib import Path
from typing import Any

import daily_support as support
import hrrr_support
import pytest
import seal_helpers as helpers
import yaml

from lastlight import daily
from lastlight.atmosphere import ReductionError
from lastlight.cli import build_parser, build_registry, main
from lastlight.commands import seal_daily as seal_daily_command
from lastlight.daily import DailyResult, SealVerificationError, TooLateError
from lastlight.inputs.hrrr import HrrrError
from lastlight.inputs.http import FetchError
from lastlight.privacy.rules import text_hits
from lastlight.seal import AlreadySealedError, StripError, TsaError, ledger_paths, verify

SECOND = dt.timedelta(seconds=1)
PAUSE = 300.0
RESULT = DailyResult("sealed", support.EVENING, "ab" * 32)

TRANSIENT = [
    HrrrError("the HRRR index is not there yet"),
    FetchError("https://example.invalid/hrrr: HTTP 503"),
    TsaError("the timestamp authority is down"),
    OSError("no space left on the device"),
    TimeoutError("timed out"),
    ConnectionResetError("reset by peer"),
]
NON_TRANSIENT = [
    TooLateError("the sun has already reached the horizon; a late day is never back-filled"),
    ReductionError("AOTK is missing everywhere"),
    StripError("the strip is not the one the forecast names"),
    AlreadySealedError("the evening is already sealed"),
    SealVerificationError("the seal just written does not verify"),
    ValueError("the time must be timezone-aware"),
]


def names(errors: list[Exception]) -> list[str]:
    return [type(error).__name__ for error in errors]


class Clock:
    """A clock that only moves when the code sleeps. `stall` is extra time every sleep costs."""

    def __init__(self, start: dt.datetime, *, stall: float = 0.0) -> None:
        self.now = start
        self.stall = stall
        self.sleeps: list[float] = []

    def __call__(self) -> dt.datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += dt.timedelta(seconds=seconds + self.stall)


class Attempts:
    """Scripted attempts: each call returns or raises the next outcome and records its `now`."""

    def __init__(self, *outcomes: DailyResult | BaseException) -> None:
        self.outcomes = list(outcomes)
        self.nows: list[dt.datetime] = []

    def __call__(self, now: dt.datetime) -> DailyResult:
        self.nows.append(now)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def retry(
    attempts: Attempts,
    clock: Clock,
    *,
    count: int = 2,
    pause: float = PAUSE,
    start: dt.datetime | None = None,
    notes: list[str] | None = None,
) -> DailyResult:
    sink = notes if notes is not None else []
    return daily.seal_with_retry(
        attempts,
        support.EVENING,
        now_utc=start if start is not None else clock.now,
        attempts=count,
        pause_seconds=pause,
        clock=clock,
        sleep=clock.sleep,
        note=sink.append,
    )


# --- the deadline is daily's own instant ---------------------------------------------------


def test_the_deadline_is_the_sun_at_the_horizon_at_the_reference_point() -> None:
    assert daily.evening_deadline(support.EVENING) == support.sunset()


def test_the_deadline_is_the_instant_seal_daily_refuses_at(tmp_path: Path) -> None:
    deadline = daily.evening_deadline(support.EVENING)
    with pytest.raises(TooLateError):
        daily.seal_daily(
            support.EVENING,
            tmp_path / "ledger",
            transport=support.transport(),
            now_utc=deadline,
            cache_dir=tmp_path / "cache",
            cycle_hour=support.CYCLE_HOUR,
            forecast_hour=support.FORECAST_HOUR,
        )


# --- the loop: which failures are retried ---------------------------------------------------


@pytest.mark.parametrize("error", TRANSIENT, ids=names(TRANSIENT))
def test_a_transient_failure_is_retried_and_the_second_attempt_seals_once(
    error: Exception,
) -> None:
    clock = Clock(support.before_sunset())
    attempts = Attempts(error, RESULT)

    assert retry(attempts, clock, pause=42.0) is RESULT

    assert len(attempts.nows) == 2, "exactly one retry, and nothing after the success"
    assert clock.sleeps == [42.0], "the pause was requested with the configured seconds"


@pytest.mark.parametrize("error", NON_TRANSIENT, ids=names(NON_TRANSIENT))
def test_a_failure_that_cannot_heal_is_not_retried(error: Exception) -> None:
    clock = Clock(support.before_sunset())
    attempts = Attempts(error, RESULT)

    with pytest.raises(type(error)) as caught:
        retry(attempts, clock)

    assert caught.value is error
    assert len(attempts.nows) == 1
    assert clock.sleeps == []


def test_a_single_attempt_never_pauses_and_raises_the_failure() -> None:
    clock = Clock(support.before_sunset())
    attempts = Attempts(FetchError("down"), RESULT)
    with pytest.raises(FetchError):
        retry(attempts, clock, count=1)
    assert len(attempts.nows) == 1 and clock.sleeps == []


def test_a_first_attempt_that_succeeds_never_pauses() -> None:
    clock = Clock(support.before_sunset())
    attempts = Attempts(RESULT, FetchError("never reached"))
    assert retry(attempts, clock) is RESULT
    assert len(attempts.nows) == 1 and clock.sleeps == []


def test_the_last_attempts_failure_is_the_one_raised() -> None:
    clock = Clock(support.before_sunset())
    first, second = FetchError("first"), HrrrError("second")
    attempts = Attempts(first, second)
    with pytest.raises(HrrrError) as caught:
        retry(attempts, clock)
    assert caught.value is second
    assert len(attempts.nows) == 2 and clock.sleeps == [PAUSE]


def test_a_second_failure_that_cannot_heal_is_the_one_raised_and_ends_the_run() -> None:
    clock = Clock(support.before_sunset())
    refusal = ReductionError("AOTK is missing everywhere")
    attempts = Attempts(FetchError("flaky"), refusal, RESULT)
    with pytest.raises(ReductionError) as caught:
        retry(attempts, clock, count=3)
    assert caught.value is refusal
    assert len(attempts.nows) == 2


def test_with_three_attempts_each_failure_gets_its_own_pause_and_the_run_stops_at_success() -> None:
    clock = Clock(support.before_sunset(minutes=60))
    attempts = Attempts(FetchError("one"), TsaError("two"), RESULT, FetchError("never reached"))
    assert retry(attempts, clock, count=3, pause=60.0) is RESULT
    assert len(attempts.nows) == 3
    assert clock.sleeps == [60.0, 60.0]


def test_each_attempt_gets_a_fresh_now_read_after_the_pause() -> None:
    start = support.before_sunset()
    clock = Clock(start)
    attempts = Attempts(FetchError("flaky"), RESULT)
    retry(attempts, clock, start=start)
    assert attempts.nows == [start, start + dt.timedelta(seconds=PAUSE)]


def test_the_first_attempt_uses_the_now_it_was_given_not_a_clock_read() -> None:
    given = support.before_sunset(minutes=40)
    clock = Clock(support.before_sunset(minutes=5))
    attempts = Attempts(RESULT)
    retry(attempts, clock, start=given)
    assert attempts.nows == [given]


def test_zero_attempts_is_refused() -> None:
    clock = Clock(support.before_sunset())
    with pytest.raises(ValueError, match="attempts"):
        retry(Attempts(RESULT), clock, count=0)


# --- never late -----------------------------------------------------------------------------


def test_a_pause_that_would_reach_the_deadline_is_not_taken() -> None:
    deadline = support.sunset()
    clock = Clock(deadline - dt.timedelta(seconds=PAUSE))  # now + pause == the deadline exactly
    first = FetchError("flaky")
    attempts = Attempts(first, RESULT)
    notes: list[str] = []

    with pytest.raises(FetchError) as caught:
        retry(attempts, clock, notes=notes)

    assert caught.value is first
    assert len(attempts.nows) == 1, "no second attempt"
    assert clock.sleeps == [], "the pause was not taken"
    assert any("not retrying" in note for note in notes)


def test_a_pause_that_ends_one_second_before_the_deadline_is_taken() -> None:
    deadline = support.sunset()
    clock = Clock(deadline - dt.timedelta(seconds=PAUSE) - SECOND)
    attempts = Attempts(FetchError("flaky"), RESULT)
    assert retry(attempts, clock) is RESULT
    assert clock.sleeps == [PAUSE]


def test_the_deadline_is_judged_from_the_clock_after_the_failure_not_from_the_start() -> None:
    # The first attempt itself took a while: it started in good time and failed close to the
    # deadline, so the pause must not be taken.
    started = support.before_sunset(minutes=40)
    clock = Clock(support.sunset() - dt.timedelta(seconds=PAUSE / 2))
    attempts = Attempts(FetchError("slow to fail"), RESULT)
    with pytest.raises(FetchError):
        retry(attempts, clock, start=started)
    assert clock.sleeps == [] and len(attempts.nows) == 1


def test_a_failure_after_the_deadline_is_not_retried() -> None:
    clock = Clock(support.after_sunset(minutes=1))
    attempts = Attempts(FetchError("flaky"), RESULT)
    with pytest.raises(FetchError):
        retry(attempts, clock)
    assert clock.sleeps == [] and len(attempts.nows) == 1


def test_a_clock_that_is_not_timezone_aware_is_refused_before_pausing() -> None:
    clock = Clock(dt.datetime(2026, 10, 10, 12))
    attempts = Attempts(FetchError("flaky"), RESULT)
    with pytest.raises(ValueError, match="timezone"):
        retry(attempts, clock, start=support.before_sunset())
    assert clock.sleeps == []


# --- what the loop says ---------------------------------------------------------------------


def test_each_failed_attempt_is_named_and_the_pause_is_announced() -> None:
    clock = Clock(support.before_sunset())
    notes: list[str] = []
    retry(Attempts(FetchError("the bucket said 503"), RESULT), clock, notes=notes)
    text = "\n".join(notes)
    assert "attempt 1 of 2 failed" in text and "the bucket said 503" in text
    assert "300 seconds" in text
    assert text_hits(text) == []


def test_a_failure_that_cannot_heal_says_it_is_not_retried() -> None:
    clock = Clock(support.before_sunset())
    notes: list[str] = []
    with pytest.raises(ReductionError):
        retry(Attempts(ReductionError("AOTK is missing everywhere"), RESULT), clock, notes=notes)
    text = "\n".join(notes)
    assert "AOTK is missing everywhere" in text and "not retrying" in text
    assert text_hits(text) == []


def test_declining_a_pause_says_why_without_a_clock_time() -> None:
    clock = Clock(support.sunset() - dt.timedelta(seconds=10))
    notes: list[str] = []
    with pytest.raises(FetchError):
        retry(Attempts(FetchError("flaky"), RESULT), clock, notes=notes)
    text = "\n".join(notes)
    assert "not retrying" in text and "horizon" in text and "300 seconds" in text
    assert text_hits(text) == []


def test_one_attempt_is_reported_exactly_as_before() -> None:
    clock = Clock(support.before_sunset())
    notes: list[str] = []
    with pytest.raises(FetchError):
        retry(Attempts(FetchError("down")), clock, count=1, notes=notes)
    assert notes == ["down"]


# --- the command ---------------------------------------------------------------------------

COMMON = ["--cycle", "22", "--fhour", "0"]


def invoke(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    *extra: str | Path,
    transport: Any = None,
    now: dt.datetime | None = None,
    clock: Clock | None = None,
) -> tuple[int, str, str]:
    start = now if now is not None else support.before_sunset()
    the_clock = clock if clock is not None else Clock(start)

    def run(args: Any) -> int:
        return seal_daily_command.execute(
            args,
            transport=transport or support.transport(),
            now_utc=start,
            clock=the_clock,
            sleep=the_clock.sleep,
        )

    command = dataclasses.replace(seal_daily_command.COMMAND, run=run)
    argv = [
        "seal-daily", "--ledger-dir", str(tmp_path / "ledger"),
        "--cache-dir", str(tmp_path / "cache"), *COMMON, *map(str, extra),
    ]  # fmt: skip
    code = main(argv, commands=[command])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def fake_seal_daily(
    monkeypatch: pytest.MonkeyPatch, *outcomes: DailyResult | BaseException
) -> Attempts:
    """Replace `lastlight.daily.seal_daily`, the seam the command calls, with scripted attempts."""
    script = Attempts(*outcomes)

    def seal(evening: dt.date, ledger_dir: Path, **kwargs: Any) -> DailyResult:
        assert evening == support.EVENING
        return script(kwargs["now_utc"])

    monkeypatch.setattr("lastlight.daily.seal_daily", seal)
    return script


def test_help_lists_the_retry_options(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["seal-daily", "--help"], commands=[seal_daily_command.COMMAND]) == 0
    out = capsys.readouterr().out
    assert "--attempts" in out and "--retry-pause" in out


def test_the_defaults_are_one_attempt_and_a_five_minute_pause() -> None:
    parser = build_parser(build_registry([seal_daily_command.COMMAND]))
    args = parser.parse_args(["seal-daily", "--ledger-dir", "x"])
    assert args.attempts == 1
    assert args.retry_pause == 300


@pytest.mark.parametrize(
    "bad",
    [
        ["--attempts", "0"],
        ["--attempts", "-1"],
        ["--attempts", "1.5"],
        ["--attempts", "two"],
        ["--retry-pause", "-1"],
        ["--retry-pause", "nan"],
        ["--retry-pause", "inf"],
        ["--retry-pause", "soon"],
    ],
    ids=lambda bad: " ".join(bad),
)
def test_a_bad_retry_option_is_a_usage_error(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, bad: list[str]
) -> None:
    code, stdout, err = invoke(capsys, tmp_path, *bad)
    assert code == 2 and stdout == ""
    assert f"argument {bad[0]}:" in err, "rejected by the option's own check, not as unknown"
    assert helpers.snapshot(tmp_path / "ledger") == {}


def test_by_default_a_transient_failure_exits_one_without_pausing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    script = fake_seal_daily(monkeypatch, FetchError("the bucket said 503"), RESULT)
    clock = Clock(support.before_sunset())

    code, stdout, err = invoke(capsys, tmp_path, clock=clock)

    assert code == 1 and stdout == ""
    assert err == "seal-daily: the bucket said 503\n"
    assert len(script.nows) == 1 and clock.sleeps == []


def test_a_transient_failure_then_success_exits_zero_and_prints_the_seal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    script = fake_seal_daily(monkeypatch, FetchError("the bucket said 503"), RESULT)
    clock = Clock(support.before_sunset())

    code, stdout, err = invoke(
        capsys, tmp_path, "--attempts", "2", "--retry-pause", "42", clock=clock
    )

    assert code == 0, err
    assert stdout.splitlines()[-1] == f"sealed 2026-10-10 {'ab' * 32}"
    assert clock.sleeps == [42]
    assert len(script.nows) == 2
    assert script.nows[1] - script.nows[0] == dt.timedelta(seconds=42)
    assert "attempt 1 of 2 failed" in err and "the bucket said 503" in err
    assert all(line.startswith("seal-daily: ") for line in err.splitlines())
    assert text_hits(err) == []


def test_the_default_pause_is_five_minutes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    fake_seal_daily(monkeypatch, TsaError("down"), RESULT)
    clock = Clock(support.before_sunset())
    code, _stdout, err = invoke(capsys, tmp_path, "--attempts", "2", clock=clock)
    assert code == 0, err
    assert clock.sleeps == [300]


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (TooLateError("the sun is down; a late day is never back-filled"), 3),
        (ReductionError("AOTK is missing everywhere"), 1),
        (StripError("the strip is not the one the forecast names"), 1),
        (AlreadySealedError("the evening is already sealed"), 1),
        (SealVerificationError("the seal does not verify"), 1),
        (ValueError("not timezone-aware"), 1),
    ],
    ids=lambda value: type(value).__name__ if isinstance(value, Exception) else str(value),
)
def test_a_failure_that_cannot_heal_exits_at_once_with_todays_code(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    error: Exception,
    code: int,
) -> None:
    script = fake_seal_daily(monkeypatch, error, RESULT)
    clock = Clock(support.before_sunset())

    got, stdout, err = invoke(capsys, tmp_path, "--attempts", "3", clock=clock)

    assert got == code and stdout == ""
    assert str(error) in err
    assert len(script.nows) == 1 and clock.sleeps == []


def test_when_every_attempt_fails_the_exit_code_is_the_last_attempts_and_both_are_named(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    script = fake_seal_daily(monkeypatch, FetchError("first failure"), HrrrError("second failure"))
    clock = Clock(support.before_sunset())

    code, stdout, err = invoke(capsys, tmp_path, "--attempts", "2", clock=clock)

    assert code == 1 and stdout == ""
    assert "first failure" in err and "second failure" in err
    assert err.index("first failure") < err.index("second failure")
    assert len(script.nows) == 2


def test_a_retry_that_is_refused_as_too_late_exits_three(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # The first failure is transient (exit one); the second attempt is refused (exit three).
    # The exit code is the last attempt's.
    fake_seal_daily(
        monkeypatch,
        FetchError("first failure"),
        TooLateError("the sun is down; a late day is never back-filled"),
    )
    clock = Clock(support.before_sunset())

    code, stdout, err = invoke(capsys, tmp_path, "--attempts", "2", clock=clock)

    assert code == 3 and stdout == ""
    assert "first failure" in err and "never back-filled" in err


def test_a_pause_that_would_cross_the_deadline_is_not_taken_and_stderr_says_why(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    script = fake_seal_daily(monkeypatch, FetchError("the bucket said 503"), RESULT)
    now = support.sunset() - dt.timedelta(seconds=PAUSE - 1)
    clock = Clock(now)

    code, stdout, err = invoke(capsys, tmp_path, "--attempts", "2", now=now, clock=clock)

    assert code == 1 and stdout == ""
    assert "the bucket said 503" in err and "not retrying" in err and "horizon" in err
    assert len(script.nows) == 1 and clock.sleeps == []
    assert text_hits(err) == []


# --- end to end on the recorded crop -------------------------------------------------------


class FlakyTransport:
    """Fails its first request, then serves the recorded HRRR crop."""

    def __init__(self, inner: hrrr_support.FakeTransport, failures: int = 1) -> None:
        self.inner = inner
        self.failures = failures

    def _maybe_fail(self, url: str) -> None:
        if self.failures:
            self.failures -= 1
            raise FetchError(f"{url}: HTTP 503 Service Unavailable")

    def exists(self, url: str) -> bool:
        self._maybe_fail(url)
        return self.inner.exists(url)

    def get(self, url: str, byte_range: Any = None) -> bytes:
        self._maybe_fail(url)
        return self.inner.get(url, byte_range)


def test_a_flaky_first_attempt_then_a_good_one_seals_the_evening_exactly_once(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    fake = support.install_tsa(monkeypatch, support.before_sunset())
    clock = Clock(support.before_sunset())
    transport = FlakyTransport(support.transport())

    code, stdout, err = invoke(
        capsys, tmp_path, "--attempts", "2", "--retry-pause", "42", transport=transport, clock=clock
    )

    ledger = tmp_path / "ledger"
    assert code == 0, err
    assert stdout.splitlines()[-1].startswith("sealed 2026-10-10 ")
    assert clock.sleeps == [42]
    assert len(fake.calls) == 1, "one timestamp spent: the failed attempt asked for none"
    assert len((ledger / "index.jsonl").read_text().splitlines()) == 1
    assert verify(support.EVENING, ledger).ok
    assert ledger_paths(ledger, support.EVENING).strip.is_file()
    assert "attempt 1 of 2 failed" in err and "503" in err


def test_a_second_attempt_past_the_horizon_is_refused_and_seals_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # The pause fits (a minute to spare), but the runner is stalled for two more minutes, so the
    # second attempt starts after the sun is at the horizon: the existing late refusal applies.
    fake = support.install_tsa(monkeypatch, support.before_sunset())
    start = support.sunset() - dt.timedelta(seconds=PAUSE + 60)
    clock = Clock(start, stall=120.0)
    inner = support.transport()
    transport = FlakyTransport(inner)

    code, stdout, err = invoke(
        capsys, tmp_path, "--attempts", "2", now=start, transport=transport, clock=clock
    )

    assert code == 3 and stdout == ""
    assert clock.sleeps == [300]
    assert "never" in err and "back-fill" in err
    assert helpers.snapshot(tmp_path / "ledger") == {}
    assert fake.calls == [], "nothing was stamped"
    assert inner.calls == [], "nothing was fetched on the late attempt"


# --- the real entry point wires a real clock and a real sleep ------------------------------


def test_the_real_command_pauses_with_time_sleep_between_attempts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # A date far ahead keeps the deadline far from the real clock, whenever this test runs.
    evening = dt.date(2099, 1, 1)
    outcomes: list[DailyResult | BaseException] = [
        FetchError("the bucket said 503"),
        DailyResult("sealed", evening, "cd" * 32),
    ]

    def seal(day: dt.date, ledger_dir: Path, **kwargs: Any) -> DailyResult:
        assert day == evening
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    slept: list[float] = []
    monkeypatch.setattr("lastlight.daily.seal_daily", seal)
    monkeypatch.setattr("time.sleep", slept.append)

    code = main(
        [
            "seal-daily", "--ledger-dir", str(tmp_path / "ledger"), "--date", "2099-01-01",
            "--attempts", "2", "--retry-pause", "7",
        ]
    )  # fmt: skip

    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert slept == [7]
    assert captured.out.splitlines()[-1] == f"sealed 2099-01-01 {'cd' * 32}"


# --- the workflow --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
SEAL_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "seal.yml"


def seal_steps() -> list[dict[str, Any]]:
    workflow = yaml.safe_load(SEAL_WORKFLOW.read_text())
    (job,) = workflow["jobs"].values()
    return job["steps"]


def test_the_workflow_asks_for_two_attempts_and_the_command_accepts_the_line() -> None:
    (step,) = [s for s in seal_steps() if "lastlight seal-daily" in s.get("run", "")]
    words = shlex.split(step["run"])
    parser = build_parser(build_registry([seal_daily_command.COMMAND]))
    args = parser.parse_args(words[words.index("seal-daily") :])
    assert args.attempts == 2
    assert args.ledger_dir == Path("ledger-wt/ledger")


def test_the_retry_is_in_python_not_in_the_workflow() -> None:
    steps = seal_steps()
    assert len([s for s in steps if "lastlight seal-daily" in s.get("run", "")]) == 1
    for step in steps:
        assert "continue-on-error" not in step, step.get("name")
        assert "retry" not in step.get("uses", "").lower(), step.get("name")
