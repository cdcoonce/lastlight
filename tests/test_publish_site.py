"""The generated almanac (issue #9): what each page says, and that it is accessible and private.

Every test builds a site from a small sealed ledger (`ledger_factory`, the real seal module and a
recorded timestamp, offline) and asserts on the generated files. No screenshots.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import ledger_factory as factory
import pytest
import seal_helpers as helpers
from html_probe import Probe, read_page

from lastlight.privacy import contains_clock_time
from lastlight.publish import build_site
from lastlight.seal import ledger_paths, tsa
from lastlight.seal.ledger import VerifyResult

OLDEST, MIDDLE, NEWEST = factory.DATES
REPO_URL = "https://github.com/cdcoonce/lastlight"


@pytest.fixture
def site(ledger_dir: Path, tmp_path: Path) -> Path:
    out = tmp_path / "site"
    report = build_site(ledger_dir, out)
    assert report.findings == ()
    return out


def digest_of_sealed(ledger_dir: Path, date: dt.date) -> str:
    return helpers.digest_hex(ledger_paths(ledger_dir, date).forecast.read_bytes())


def html_pages(site: Path) -> list[Path]:
    return sorted(site.glob("*.html"))


# --- what is generated ----------------------------------------------------------------------


def test_the_site_is_a_home_page_an_archive_a_scoreboard_a_page_per_date_and_the_paintings(
    site: Path,
) -> None:
    generated = {path.relative_to(site).as_posix() for path in site.rglob("*") if path.is_file()}
    assert generated == {
        "index.html",
        "archive.html",
        "scoreboard.html",
        "style.css",
        *(f"{date}.html" for date in factory.DATES),
        *(f"strips/{date}.png" for date in factory.DATES),
    }


def test_the_home_page_names_the_newest_date_and_lists_three_archive_entries(site: Path) -> None:
    home = read_page(site, "index.html")
    assert "2026-10-13" in home.headings()
    assert "2026-10-11" not in home.headings()
    entries = home.find("li", "evening")
    assert len(entries) == 3
    assert [entry.text.split()[0] for entry in entries] == [
        "2026-10-13",
        "2026-10-12",
        "2026-10-11",
    ]


def test_the_home_page_shows_the_newest_painting_first(site: Path) -> None:
    home = read_page(site, "index.html")
    assert home.find("img")[0].attrs["src"] == "strips/2026-10-13.png"


def test_the_archive_lists_every_painting_newest_first(site: Path) -> None:
    archive = read_page(site, "archive.html")
    entries = archive.find("li", "evening")
    assert len(entries) == 3
    date_links = [
        href for href in archive.hrefs() if re.fullmatch(r"\d{4}-\d{2}-\d{2}\.html", href)
    ]
    assert date_links == ["2026-10-13.html", "2026-10-12.html", "2026-10-11.html"]
    assert [img.attrs["src"] for img in archive.find("img")] == [
        "strips/2026-10-13.png",
        "strips/2026-10-12.png",
        "strips/2026-10-11.png",
    ]


def test_evenings_sealed_out_of_date_order_are_still_featured_and_listed_newest_first(
    shuffled_ledger_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "site"
    assert build_site(shuffled_ledger_dir, out).findings == ()
    home = read_page(out, "index.html")
    assert "2026-10-13" in home.headings(), "the featured evening is the newest verified date"
    assert home.find("img")[0].attrs["src"] == "strips/2026-10-13.png"
    assert [entry.text.split()[0] for entry in home.find("li", "evening")] == [
        "2026-10-13",
        "2026-10-12",
        "2026-10-11",
    ]
    archive = read_page(out, "archive.html")
    date_links = [
        href for href in archive.hrefs() if re.fullmatch(r"\d{4}-\d{2}-\d{2}\.html", href)
    ]
    assert date_links == ["2026-10-13.html", "2026-10-12.html", "2026-10-11.html"]


def test_the_home_page_lists_only_recent_evenings_and_the_archive_lists_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "ledger"
    first = dt.date(2026, 11, 1)
    for offset in range(16):
        factory.seal_date(ledger, first + dt.timedelta(days=offset), monkeypatch)
    out = tmp_path / "site"
    assert build_site(ledger, out).findings == ()
    assert len(read_page(out, "index.html").find("li", "evening")) == 14
    assert len(read_page(out, "archive.html").find("li", "evening")) == 16
    assert "archive.html" in read_page(out, "index.html").hrefs()


# --- what a visitor learns -----------------------------------------------------------------


@pytest.mark.parametrize("page", ["index.html", "2026-10-13.html"])
def test_the_newest_evening_shows_its_digest_model_version_and_how_to_verify_it(
    ledger_dir: Path, site: Path, page: str
) -> None:
    probe = read_page(site, page)
    text = probe.text
    assert digest_of_sealed(ledger_dir, NEWEST) in text
    assert "0.0.3" in text
    assert "lastlight-stub" in text
    assert "Verify it yourself" in probe.headings()
    commands = "\n".join(probe.texts("pre"))
    assert "lastlight verify 2026-10-13 --ledger-dir" in commands
    assert "ledger/2026/10/13.forecast.json" in commands
    assert "git worktree add" in commands and "origin/ledger" in commands


@pytest.mark.parametrize("date", factory.DATES)
def test_every_sealed_date_has_a_page_with_its_own_digest_and_version(
    ledger_dir: Path, site: Path, date: dt.date
) -> None:
    probe = read_page(site, f"{date}.html")
    assert digest_of_sealed(ledger_dir, date) in probe.text
    assert f"0.0.{factory.DATES.index(date) + 1}" in probe.text
    assert probe.headings()[0] == f"Evening of {date}"
    assert probe.find("img")[0].attrs["src"] == f"strips/{date}.png"


def test_a_page_shows_the_forecast_notes_and_vividness(site: Path) -> None:
    text = read_page(site, "2026-10-13.html").text
    assert "stub forecast for the seal tests" in text
    assert "0.42" in text and "0.07" in text


@pytest.mark.parametrize("page", ["index.html", "2026-10-12.html"])
def test_sealed_in_time_is_shown_as_the_result_of_verification(site: Path, page: str) -> None:
    statuses = read_page(site, page).find("p", "status")
    assert statuses
    assert statuses[0].has_class("verified")
    assert statuses[0].text.startswith("Sealed before the evening: verified by lastlight verify")


def test_no_page_shows_a_time_of_day_or_when_the_timestamp_was_issued(
    ledger_dir: Path, site: Path
) -> None:
    forbidden: list[str] = []
    for date in factory.DATES:
        issued = tsa.parse_response(ledger_paths(ledger_dir, date).timestamp.read_bytes()).gen_time
        forbidden += [issued.strftime("%H:%M"), issued.strftime("%Y%m%d%H%M%S"), issued.isoformat()]
    for page in html_pages(site):
        text = page.read_text(encoding="utf-8")
        assert not contains_clock_time(text), page.name
        assert "UTC" not in text, page.name
        assert not [item for item in forbidden if item in text], page.name


def test_strips_are_labelled_in_sun_elevation_degrees_not_clock_time(site: Path) -> None:
    probe = read_page(site, "2026-10-13.html")
    ticks = [text for text in probe.texts("li") if re.fullmatch(r"[+−]?\d+(\.\d)?°", text)]
    assert ticks == ["+6°", "0°", "−4°", "−8°"]  # the forecast's own four sun elevations
    caption = probe.texts("figcaption")[0]
    assert "sun elevation" in caption.lower() and "degrees" in caption.lower()
    assert "view elevation" in caption.lower()
    assert probe.find("ol", "axis")


def test_the_scoreboard_is_a_placeholder_without_any_image(site: Path) -> None:
    probe = read_page(site, "scoreboard.html")
    assert "no scored evenings yet" in probe.text.lower()
    assert "M2" in probe.text
    assert probe.find("img") == []


def test_only_the_sealed_paintings_are_ever_embedded(site: Path) -> None:
    for page in html_pages(site):
        for img in read_page(site, page.name).find("img"):
            assert re.fullmatch(r"strips/\d{4}-\d{2}-\d{2}\.png", img.attrs["src"]), page.name


def test_the_forecast_notes_are_escaped_not_interpreted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "ledger"
    markup = '<script>alert("x")</script> & <b>bold</b>'
    factory.seal_date(ledger, OLDEST, monkeypatch, notes=markup)
    out = tmp_path / "site"
    assert build_site(ledger, out).findings == ()
    page = (out / "2026-10-11.html").read_text(encoding="utf-8")
    assert "<script" not in page and "<b>" not in page
    assert "&lt;script&gt;" in page
    assert markup in read_page(out, "2026-10-11.html").text


HOSTILE = '<script>alert("x")</script> & <img src=x onerror=alert(1)>'
HOSTILE_ESCAPED = (
    "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; &lt;img src=x onerror=alert(1)&gt;"
)


def assert_inert(out: Path, page: str, shown: str) -> None:
    """`page` writes `shown` as text: escaped in the file, no element or attribute made of it."""
    raw = (out / page).read_text(encoding="utf-8")
    assert "<script" not in raw and "onerror=alert(1)>" not in raw, page
    probe = read_page(out, page)
    assert probe.find("script") == [], page
    assert all("onerror" not in image.attrs for image in probe.find("img")), page
    assert shown in probe.text, page


def test_a_hostile_model_name_and_version_are_escaped_on_every_page_that_shows_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "ledger"
    name, version = f"{HOSTILE} name", f'{HOSTILE} v"1.0"'
    factory.seal_date(ledger, NEWEST, monkeypatch, version=version, model_name=name)
    factory.write_strip(ledger, NEWEST)
    out = tmp_path / "site"
    assert build_site(ledger, out).findings == ()
    escaped = f"{HOSTILE_ESCAPED} name {HOSTILE_ESCAPED} v&quot;1.0&quot;"
    for page in ("index.html", "2026-10-13.html"):
        assert escaped in (out / page).read_text(encoding="utf-8"), page
        assert_inert(out, page, f"{name} {version}")


def test_a_hostile_verification_reason_is_escaped_on_every_page_that_shows_it(
    ledger_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reason = f'late seal "{HOSTILE}"'
    monkeypatch.setattr(
        "lastlight.publish.ledger.verify", lambda date, ledger: VerifyResult(False, reason)
    )
    out = tmp_path / "site"
    assert build_site(ledger_dir, out).findings == ()
    for page in html_pages(out):
        raw = page.read_text(encoding="utf-8")
        assert "<script" not in raw and "onerror=alert(1)>" not in raw, page.name
        assert read_page(out, page.name).find("script") == [], page.name
    for date in factory.DATES:
        page = f"{date}.html"
        assert f"late seal &quot;{HOSTILE_ESCAPED}&quot;" in (out / page).read_text(
            encoding="utf-8"
        )
        assert_inert(out, page, reason)


# --- accessibility --------------------------------------------------------------------------


def test_every_image_has_alt_text_that_names_its_date(site: Path) -> None:
    seen = 0
    for page in html_pages(site):
        for img in read_page(site, page.name).find("img"):
            seen += 1
            alt = img.attrs.get("alt", "")
            assert alt.strip(), f"{page.name}: image without alt text"
            assert re.search(r"20\d\d-\d\d-\d\d", alt), alt
    assert seen >= 3 + 3 + 3 + 1  # archive, home recent list, date pages, home feature


def test_every_page_has_a_language_title_viewport_landmarks_and_one_h1(site: Path) -> None:
    for page in html_pages(site):
        probe = read_page(site, page.name)
        html = probe.find("html")[0]
        assert html.attrs["lang"] == "en", page.name
        viewport = [m.attrs for m in probe.find("meta") if m.attrs.get("name") == "viewport"]
        assert viewport and "width=device-width" in viewport[0]["content"], page.name
        assert probe.texts("title")[0].strip(), page.name
        assert len(probe.find("main")) == 1, page.name
        assert probe.find("nav"), page.name
        assert len(probe.find("h1")) == 1, page.name
        assert "style.css" in [m.attrs.get("href") for m in probe.find("link")], page.name


def test_the_layout_is_fluid_not_fixed_width(site: Path) -> None:
    css = (site / "style.css").read_text(encoding="utf-8")
    assert "max-width" in css
    assert not re.search(r"(?<![-\w])width:\s*\d+px", css), "no fixed pixel widths"
    assert "img" in css and "max-width: 100%" in css


def test_the_internal_links_all_resolve_and_the_external_ones_are_the_public_repository(
    site: Path,
) -> None:
    for page in html_pages(site):
        for href in read_page(site, page.name).hrefs():
            if href.startswith("#"):
                continue
            if href.startswith("https://"):
                assert href.startswith(REPO_URL), (page.name, href)
            else:
                assert not href.startswith("/"), (page.name, href)
                assert (site / href.split("#")[0]).is_file(), (page.name, href)


def _luminance(color: str) -> float:
    channels = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(first: str, second: str) -> float:
    high, low = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


@pytest.mark.parametrize("scheme", [0, 1], ids=["light", "dark"])
def test_text_and_status_colours_meet_the_wcag_aa_contrast_ratio(site: Path, scheme: int) -> None:
    css = (site / "style.css").read_text(encoding="utf-8")
    blocks = re.findall(r":root\s*\{([^}]*)\}", css)
    assert len(blocks) == 2, "one palette for light, one for prefers-color-scheme: dark"
    assert "prefers-color-scheme: dark" in css
    colors = dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})", blocks[scheme]))
    pairs = [
        ("fg", "bg"),
        ("muted", "bg"),
        ("link", "bg"),
        ("fg", "card"),
        ("muted", "card"),
        ("link", "card"),
        ("ok-fg", "ok-bg"),
        ("bad-fg", "bad-bg"),
    ]
    for text, background in pairs:
        ratio = _contrast(colors[text], colors[background])
        assert ratio >= 4.5, f"{text} on {background} is {ratio:.2f}:1"


# --- when something is wrong or missing -------------------------------------------------------


def test_an_evening_without_a_painting_gets_a_text_placeholder(
    ledger_dir: Path, tmp_path: Path
) -> None:
    factory.strip_path(ledger_dir, MIDDLE).unlink()
    out = tmp_path / "site"
    assert build_site(ledger_dir, out).findings == ()
    assert not (out / "strips" / "2026-10-12.png").exists()
    page = read_page(out, "2026-10-12.html")
    assert page.find("img") == []
    assert "No painting for this date" in page.text
    archive = read_page(out, "archive.html")
    middle = archive.find("li", "evening")[1]
    assert middle.text.startswith("2026-10-12") and "No painting for this date" in middle.text
    assert [img.attrs["src"] for img in archive.find("img")] == [
        "strips/2026-10-13.png",
        "strips/2026-10-11.png",
    ]


def test_a_late_seal_is_marked_not_verified_and_never_featured(
    ledger_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    late = dt.date(2026, 10, 14)
    factory.seal_date(ledger_dir, late, monkeypatch, late=True)
    factory.write_strip(ledger_dir, late)
    out = tmp_path / "site"
    assert build_site(ledger_dir, out).findings == ()

    home = read_page(out, "index.html")
    assert "2026-10-13" in home.headings() and "2026-10-14" not in home.headings()
    notice = home.find("p", "notice")[0].text
    assert "2026-10-14" in notice and "did not verify" in notice

    archive = read_page(out, "archive.html")
    first = archive.find("li", "evening")[0]
    assert first.text.startswith("2026-10-14") and "Not verified" in first.text

    page = read_page(out, "2026-10-14.html")
    status = page.find("p", "status")[0]
    assert status.has_class("unverified")
    assert "Not verified" in status.text and "late seal" in status.text
    assert "Sealed before the evening" not in page.text


def test_an_empty_ledger_builds_an_empty_state_site(tmp_path: Path) -> None:
    out = tmp_path / "site"
    report = build_site(tmp_path / "no-ledger-yet", out)
    assert report.findings == ()
    assert {path.name for path in out.rglob("*") if path.is_file()} == {
        "index.html",
        "archive.html",
        "scoreboard.html",
        "style.css",
    }
    home = read_page(out, "index.html")
    assert "No sealed painting yet" in home.headings()
    assert home.find("li", "evening") == []
    assert "No evenings sealed yet" in read_page(out, "archive.html").text


def test_the_home_page_of_a_ledger_with_nothing_verified_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "ledger"
    factory.seal_date(ledger, OLDEST, monkeypatch, late=True)
    out = tmp_path / "site"
    assert build_site(ledger, out).findings == ()
    home = read_page(out, "index.html")
    assert "No sealed painting yet" in home.headings()
    assert "did not verify" in home.find("p", "notice")[0].text
    assert len(home.find("li", "evening")) == 1


def test_the_archive_says_how_many_index_lines_could_not_be_read(
    ledger_dir: Path, tmp_path: Path
) -> None:
    with (ledger_dir / "index.jsonl").open("a") as index:
        index.write("garbage\n")
    out = tmp_path / "site"
    assert build_site(ledger_dir, out).ok
    notice = read_page(out, "archive.html").find("p", "notice")[0].text
    assert "1 line" in notice and "could not be read" in notice
    assert len(read_page(out, "archive.html").find("li", "evening")) == 3


def test_the_page_probe_is_a_real_html_parser() -> None:
    probe = Probe('<ul><li class="a b">x <b>y</b></li><li>z</li></ul><img src="p.png" alt="q">')
    assert probe.texts("li") == ["x y", "z"]
    assert probe.find("li", "b")[0].attrs["class"] == "a b"
    assert probe.find("img")[0].attrs == {"src": "p.png", "alt": "q"}
