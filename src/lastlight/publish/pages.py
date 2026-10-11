"""The almanac's pages: plain functions from a `Ledger` to HTML text, and the stylesheet.

Rules every page follows (docs/design.md section 3, and the accessibility the issue asks for):

- Nothing is a time of day. "Sealed in time" is the result of `lastlight verify`, never a
  timestamp; the sky is measured in sun elevation, in degrees. Dates are the only calendar
  quantity. The build runs the privacy scanner over every page before anything is published, so
  a slip here fails the build rather than reaching a visitor.
- Everything that came from the ledger is escaped. Nothing is dropped or tidied: if a note or a
  verification reason holds something the scanner rejects, the build fails and says where.
- No script and no external resource. Every image has alt text; colours are tokens with
  light and dark palettes that meet WCAG AA contrast; the layout is fluid and starts at phone
  width; pages are the same bytes every time they are built from the same ledger.
- Links are relative, so the site works from a project path such as `/lastlight/`.
"""

from __future__ import annotations

import datetime as dt
import html
import math
from collections.abc import Sequence
from importlib import resources

from lastlight.publish.ledger import Evening, Ledger

REPOSITORY_URL = "https://github.com/cdcoonce/lastlight"
GENERATOR_META = '<meta name="generator" content="lastlight site build">'
"""Marks a directory as one this build wrote, so a rebuild may replace it and nothing else."""

RECENT_EVENINGS = 14
"""How many evenings the home page lists; the archive lists them all."""

MAX_AXIS_LABELS = 8

NAVIGATION = (
    ("index.html", "Latest"),
    ("archive.html", "Archive"),
    ("scoreboard.html", "Scoreboard"),
)


def page_name(date: dt.date) -> str:
    return f"{date.isoformat()}.html"


def strip_name(date: dt.date) -> str:
    return f"strips/{date.isoformat()}.png"


def _e(text: object) -> str:
    return html.escape(str(text), quote=True)


MINUS = "\u2212"


def degrees(value: float, *, plus: bool = True) -> str:
    """An elevation as people write it: +6°, 0°, −4° (a true minus sign); `plus` adds the +."""
    magnitude = f"{abs(value):g}°"
    if value > 0:
        return "+" + magnitude if plus else magnitude
    return MINUS + magnitude if value < 0 else magnitude


# --- the shell ------------------------------------------------------------------------------


def _document(*, title: str, current: str, body: str) -> str:
    def link(href: str, label: str) -> str:
        marker = ' aria-current="page"' if href == current else ""
        return f'<li><a href="{href}"{marker}>{label}</a></li>'

    links = "\n".join(link(href, label) for href, label in NAVIGATION)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{GENERATOR_META}
<title>{_e(title)}</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
<a class="skip" href="#main">Skip to the content</a>
<header>
<p class="brand"><a href="index.html">lastlight</a>
<span>a Phoenix twilight almanac</span></p>
<nav aria-label="Main">
<ul>
{links}
</ul>
</nav>
</header>
<main id="main">
{body}
</main>
<footer>
<p>Every forecast is made for the public reference point, downtown Phoenix
(33.45 N, 112.07 W), and the sky is measured in sun elevation rather than clock time.
Sealed forecasts and their timestamps are in the
<a href="{REPOSITORY_URL}/tree/ledger">ledger branch</a> of the
<a href="{REPOSITORY_URL}">lastlight repository</a> (MIT licence).</p>
<p>lastlight was chosen, designed and built by AI agents (Claude); its owner observes and
keeps the accounts.</p>
</footer>
</body>
</html>
"""


# --- pieces ---------------------------------------------------------------------------------


def _axis(evening: Evening) -> str:
    """Tick labels under the painting: sun elevation in degrees, at the column centres."""
    suns = evening.sun_elevations
    if not suns:
        return ""
    step = max(1, math.ceil(len(suns) / MAX_AXIS_LABELS))
    ticks = "\n".join(
        f'<li style="left:{(index + 0.5) / len(suns) * 100:.1f}%">{degrees(suns[index])}</li>'
        for index in range(0, len(suns), step)
    )
    return f'<ol class="axis" aria-hidden="true">\n{ticks}\n</ol>'


def _extent(evening: Evening) -> tuple[str, str, str, str] | None:
    """(sun at left, sun at right, view at bottom, view at top), or None without a curve."""
    suns, views = evening.sun_elevations, evening.view_elevations
    if not suns or not views:
        return None
    return (
        degrees(suns[0]),
        degrees(suns[-1]),
        degrees(views[0], plus=False),
        degrees(views[-1], plus=False),
    )


def _alt(evening: Evening) -> str:
    intro = f"Painted western sky for the evening of {evening.date.isoformat()}: "
    extent = _extent(evening)
    if extent is None:
        return intro + "sun elevation across, view elevation up"
    left, right, bottom, top = extent
    return (
        f"{intro}sun elevation from {left} at the left to {right} at the right, "
        f"view elevation from {bottom} at the bottom to {top} at the top"
    )


def _caption(evening: Evening) -> str:
    extent = _extent(evening)
    if extent is None:
        return "Sun elevation in degrees runs left to right; view elevation in degrees runs upward."
    left, right, bottom, top = extent
    return (
        f"Sun elevation in degrees runs left to right, from {left} to {right}. "
        f"View elevation in degrees runs upward, from {bottom} at the bottom to {top} at the top."
    )


def _figure(evening: Evening) -> str:
    if evening.strip is None:
        return '<p class="no-painting">No painting for this date.</p>'
    return f"""<figure class="painting">
<img src="{strip_name(evening.date)}" alt="{_e(_alt(evening))}">
{_axis(evening)}
<figcaption>{_e(_caption(evening))}</figcaption>
</figure>"""


def _status(evening: Evening) -> str:
    if evening.verified:
        return (
            '<p class="status verified">Sealed before the evening: verified by '
            "<code>lastlight verify</code> when this page was built.</p>"
        )
    return (
        '<p class="status unverified">Not verified. <code>lastlight verify</code> failed '
        f"for this evening: {_e(evening.reason)}</p>"
    )


def _facts(evening: Evening) -> str:
    rows = [("Digest", f'<code class="digest">{_e(evening.digest)}</code>')]
    model = " ".join(part for part in (evening.model_name, evening.model_version) if part)
    if model:
        rows.append(("Model", _e(model)))
    if evening.vividness is not None:
        mean, spread = evening.vividness
        rows.append(("Vividness forecast", f"{mean:.2f}, spread {spread:.2f}"))
    body = "\n".join(f"<dt>{term}</dt>\n<dd>{value}</dd>" for term, value in rows)
    return f'<dl class="facts">\n{body}\n</dl>'


def _notes(evening: Evening, level: int) -> str:
    if not evening.notes:
        return ""
    return (
        f'<section class="notes">\n<h{level}>Forecast notes</h{level}>\n'
        f"<p>{_e(evening.notes)}</p>\n</section>"
    )


def _verify_section(evening: Evening, level: int) -> str:
    iso = evening.date.isoformat()
    forecast = f"{evening.date:%Y/%m}/{evening.date.day:02d}.forecast.json"
    commands = _e(
        f"git clone {REPOSITORY_URL}.git\n"
        "cd lastlight\n"
        "git worktree add ../lastlight-ledger origin/ledger\n"
        "uv sync\n"
        f"shasum -a 256 ../lastlight-ledger/ledger/{forecast}\n"
        f"uv run lastlight verify {iso} --ledger-dir ../lastlight-ledger/ledger"
    )
    return f"""<section class="verify">
<h{level}>Verify it yourself</h{level}>
<p>You do not have to trust this page. The sealed forecast and its RFC 3161 timestamp are
public files in the repository's <code>ledger</code> branch. These commands fetch them and
check the evening of {iso}.</p>
<pre><code>{commands}</code></pre>
<p>The <code>shasum</code> line must print the digest shown above.
<code>lastlight verify</code> prints PASS only when the forecast file hashes to what the
timestamp covers and the timestamp was issued before sunset at the reference point;
otherwise it prints the reason. It does not yet validate the timestamp authority's signature
chain; that check is a follow-up.</p>
</section>"""


def _evening_block(evening: Evening, level: int) -> str:
    """Painting, verification result, facts, notes and how to verify, for one evening."""
    parts = [
        _figure(evening),
        _status(evening),
        _facts(evening),
        _notes(evening, level),
        _verify_section(evening, level),
    ]
    return "\n".join(part for part in parts if part)


def _entry(evening: Evening) -> str:
    iso = evening.date.isoformat()
    if evening.strip is None:
        picture = '<span class="no-painting">No painting for this date</span>'
    else:
        picture = (
            f'<img src="{strip_name(evening.date)}" '
            f'alt="Painted western sky for the evening of {iso}" loading="lazy">'
        )
    mark = (
        '<span class="mark verified">Verified</span>'
        if evening.verified
        else '<span class="mark unverified">Not verified</span>'
    )
    return (
        f'<li class="evening"><a href="{page_name(evening.date)}">\n'
        f'<span class="date">{iso}</span>\n{picture}\n{mark}\n</a></li>'
    )


def _grid(evenings: Sequence[Evening]) -> str:
    return '<ul class="grid">\n' + "\n".join(_entry(evening) for evening in evenings) + "\n</ul>"


# --- pages ----------------------------------------------------------------------------------


def home_page(ledger: Ledger) -> str:
    evenings = ledger.evenings
    featured = next((evening for evening in evenings if evening.verified), None)
    parts = ["<h1>Latest sealed painting</h1>"]
    if evenings and not evenings[0].verified:
        newest = evenings[0].date
        parts.append(
            f'<p class="notice">The newest entry in the ledger, <a href="{page_name(newest)}">'
            f"{newest.isoformat()}</a>, did not verify, so it is not shown as a sealed painting. "
            "Its page says why.</p>"
        )
    if featured is None:
        parts.append(
            "<h2>No sealed painting yet</h2>\n"
            "<p>Each day lastlight paints the western twilight over Phoenix before it happens "
            "and seals the forecast in a public ledger. The first verified painting will "
            "appear here.</p>"
        )
    else:
        iso = featured.date.isoformat()
        parts.append(
            f'<section class="latest">\n<h2><time datetime="{iso}">{iso}</time></h2>\n'
            f'<p><a href="{page_name(featured.date)}">This evening on its own page</a></p>\n'
            f"{_evening_block(featured, 3)}\n"
            "</section>"
        )
    if evenings:
        parts.append(
            f"<section>\n<h2>Recent evenings</h2>\n{_grid(evenings[:RECENT_EVENINGS])}\n"
            '<p><a href="archive.html">All evenings</a></p>\n</section>'
        )
    return _document(
        title="lastlight · Phoenix twilight almanac",
        current="index.html",
        body="\n".join(parts),
    )


def date_page(evening: Evening) -> str:
    iso = evening.date.isoformat()
    body = f"<h1>Evening of {iso}</h1>\n{_evening_block(evening, 2)}"
    return _document(title=f"Evening of {iso} · lastlight", current="", body=body)


def archive_page(ledger: Ledger) -> str:
    parts = [
        "<h1>Archive</h1>",
        "<p>Every sealed evening, newest first. The mark says whether "
        "<code>lastlight verify</code> passed when this page was built.</p>",
    ]
    if ledger.skipped_index_lines:
        count = ledger.skipped_index_lines
        noun = "line" if count == 1 else "lines"
        parts.append(
            f'<p class="notice">{count} {noun} of the ledger index could not be read and '
            f"{'is' if count == 1 else 'are'} not listed.</p>"
        )
    parts.append(_grid(ledger.evenings) if ledger.evenings else "<p>No evenings sealed yet.</p>")
    return _document(title="Archive · lastlight", current="archive.html", body="\n".join(parts))


def scoreboard_page() -> str:
    body = f"""<h1>Scoreboard</h1>
<p>No scored evenings yet: tier A public-camera truth arrives in M2.</p>
<p>Until then every painting here is a sealed forecast and nothing more; none has been
compared with the real sky. When scoring starts, each evening will be scored against the two
forecasts that are hardest to beat, <em>the same as usual</em> and <em>the same as
yesterday</em>, split by sky regime and with confidence intervals. "No detectable skill"
will be published with the same prominence as a win. The rules are in the
<a href="{REPOSITORY_URL}/blob/main/docs/design.md">design contract</a>.</p>"""
    return _document(title="Scoreboard · lastlight", current="scoreboard.html", body=body)


# --- style ----------------------------------------------------------------------------------


def stylesheet() -> str:
    """The site's one stylesheet, `style.css` beside this module."""
    return (resources.files("lastlight.publish") / "style.css").read_text(encoding="utf-8")
