"""The G0 figures, drawn with Pillow (no plotting library) and written without any metadata.

Three figures, each a PNG as bytes (`lastlight.render.strip_png_bytes` copies the pixels into a
fresh image first, so nothing but pixel data reaches the file):

- `distribution_figure`: the distribution of D(d, s) over the dates used, one panel per band,
  with the 0.05 threshold marked and the share of dates at or above it;
- `by_date_figure`: D(d, s) against date for the counted bands, with the threshold and a strip
  marking the dates that meet the rule, so a seasonal pattern shows;
- `profile_figure`: the rendered contrast against sun elevation for the lowest and the highest
  band, the median over the dates and the 10th to 90th percentile range.

Colours are the Okabe-Ito palette (distinguishable with the common colour-vision deficiencies);
the band that is not counted (late) is grey.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence
from typing import Final

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lastlight.g0 import stats
from lastlight.g0.analysis import Study
from lastlight.geometry import ELEVATION_SWEEP
from lastlight.render import strip_png_bytes

Colour = tuple[int, int, int]

INK: Final[Colour] = (34, 34, 34)
GRID: Final[Colour] = (222, 222, 222)
PAPER: Final[Colour] = (255, 255, 255)
BAND_COLOURS: Final[dict[str, Colour]] = {
    "golden": (230, 159, 0),
    "early": (213, 94, 0),
    "mid": (0, 114, 178),
    "late": (120, 120, 120),
}
CHANNEL_NAMES: Final = ("log(R/G)", "log(B/G)")


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


class Canvas:
    """A white image with a few drawing helpers: axes in data coordinates, lines, bars, text."""

    def __init__(self, width: int, height: int) -> None:
        self.image = Image.new("RGB", (width, height), PAPER)
        self.draw = ImageDraw.Draw(self.image)
        self.small, self.medium, self.large = _font(14), _font(16), _font(20)

    def text(self, xy: tuple[float, float], text: str, *, font=None, fill=INK, anchor="la") -> None:
        self.draw.text(xy, text, fill=fill, font=font or self.small, anchor=anchor)

    def png(self) -> bytes:
        return strip_png_bytes(self.image)


class Axes:
    """A plotting box with a linear scale on each axis."""

    def __init__(
        self,
        canvas: Canvas,
        box: tuple[int, int, int, int],
        xlim: tuple[float, float],
        ylim: tuple[float, float],
    ) -> None:
        self.canvas = canvas
        self.left, self.top, self.right, self.bottom = box
        self.xlim, self.ylim = xlim, ylim

    def px(self, x: float) -> float:
        low, high = self.xlim
        return self.left + (x - low) / (high - low) * (self.right - self.left)

    def py(self, y: float) -> float:
        low, high = self.ylim
        return self.bottom - (y - low) / (high - low) * (self.bottom - self.top)

    def frame(
        self,
        xticks: Sequence[tuple[float, str]],
        yticks: Sequence[tuple[float, str]],
        *,
        xlabel: str = "",
        ylabel: str = "",
        title: str = "",
    ) -> None:
        draw, canvas = self.canvas.draw, self.canvas
        for value, label in yticks:
            y = self.py(value)
            draw.line([(self.left, y), (self.right, y)], fill=GRID, width=1)
            canvas.text((self.left - 6, y), label, anchor="rm")
        for value, label in xticks:
            x = self.px(value)
            draw.line([(x, self.top), (x, self.bottom)], fill=GRID, width=1)
            canvas.text((x, self.bottom + 6), label, anchor="ma")
        draw.rectangle([self.left, self.top, self.right, self.bottom], outline=INK, width=1)
        if title:
            canvas.text((self.left, self.top - 8), title, font=canvas.medium, anchor="lb")
        if xlabel:
            canvas.text(((self.left + self.right) / 2, self.bottom + 28), xlabel, anchor="ma")
        if ylabel:
            self.vertical_label(ylabel)

    def vertical_label(self, text: str) -> None:
        box = self.canvas.small.getbbox(text)
        label = Image.new("RGB", (box[2] + 4, box[3] + 4), PAPER)
        ImageDraw.Draw(label).text((2, 0), text, fill=INK, font=self.canvas.small)
        turned = label.rotate(90, expand=True)
        centre = (self.top + self.bottom) // 2
        self.canvas.image.paste(turned, (self.left - 62, centre - turned.height // 2))

    def line(self, xs, ys, colour: Colour, width: int = 2) -> None:
        points = [(self.px(x), self.py(y)) for x, y in zip(xs, ys, strict=True)]
        self.canvas.draw.line(points, fill=colour, width=width, joint="curve")

    def hline(self, y: float, colour: Colour, width: int = 1, dash: int = 6) -> None:
        self._dashed(
            (self.px(self.xlim[0]), self.py(y)),
            (self.px(self.xlim[1]), self.py(y)),
            colour,
            width,
            dash,
        )

    def vline(self, x: float, colour: Colour, width: int = 1, dash: int = 6) -> None:
        self._dashed(
            (self.px(x), self.py(self.ylim[0])),
            (self.px(x), self.py(self.ylim[1])),
            colour,
            width,
            dash,
        )

    def _dashed(self, a, b, colour: Colour, width: int, dash: int) -> None:
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        steps = max(int(length // dash), 1)
        for k in range(0, steps, 2):
            t0, t1 = k / steps, min((k + 1) / steps, 1.0)
            self.canvas.draw.line(
                [
                    (a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0),
                    (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1),
                ],
                fill=colour,
                width=width,
            )

    def band(self, xs, lows, highs, colour: Colour, alpha: float = 0.25) -> None:
        """A filled region between two curves, blended with the paper."""
        blended = tuple(round(c * alpha + 255 * (1 - alpha)) for c in colour)
        upper = [(self.px(x), self.py(y)) for x, y in zip(xs, highs, strict=True)]
        lower = [
            (self.px(x), self.py(y))
            for x, y in zip(reversed(list(xs)), reversed(list(lows)), strict=True)
        ]
        self.canvas.draw.polygon(upper + lower, fill=blended)

    def bars(self, edges: np.ndarray, counts: np.ndarray, colour: Colour) -> None:
        for count, low, high in zip(counts, edges[:-1], edges[1:], strict=True):
            if count:
                self.canvas.draw.rectangle(
                    [self.px(low) + 1, self.py(float(count)), self.px(high) - 1, self.py(0.0)],
                    fill=colour,
                )

    def marker(self, x: float, y: float, colour: Colour, radius: int = 3) -> None:
        cx, cy = self.px(x), self.py(y)
        self.canvas.draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=colour)


def nice_ticks(low: float, high: float, target: int = 5) -> list[float]:
    """Round tick positions covering [low, high], about `target` of them."""
    span = high - low
    if span <= 0.0:
        return [low]
    raw = span / max(target, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    first = math.ceil(low / step - 1e-9) * step
    ticks = []
    value = first
    while value <= high + 1e-9:
        ticks.append(round(value, 10))
        value += step
    return ticks


def _labels(ticks: Sequence[float], decimals: int) -> list[tuple[float, str]]:
    return [(tick, f"{tick:.{decimals}f}") for tick in ticks]


def _decimals(ticks: Sequence[float]) -> int:
    for decimals in range(0, 5):
        if all(abs(t - round(t, decimals)) < 1e-9 for t in ticks):
            return decimals
    return 4


def _legend(canvas: Canvas, x: int, y: int, entries: Sequence[tuple[str, Colour]]) -> None:
    for index, (label, colour) in enumerate(entries):
        top = y + index * 22
        canvas.draw.rectangle([x, top + 3, x + 14, top + 15], fill=colour)
        canvas.text((x + 22, top), label)


# --- figure 1: the distribution of D ---------------------------------------------------------


def distribution_figure(study: Study) -> bytes:
    """One histogram of D(d, s) per band, with the threshold and the share at or above it."""
    outcome = study.outcomes["nominal"]
    canvas = Canvas(1200, 820)
    canvas.text(
        (600, 14), "Distribution of D(d, s) over the dates used", font=canvas.large, anchor="ma"
    )
    limit = float(max(outcome.distances.max(), stats.THRESHOLD * 2)) * 1.05
    edges = np.linspace(0.0, limit, 25)
    boxes = [(90, 80, 560, 340), (680, 80, 1150, 340), (90, 450, 560, 710), (680, 450, 1150, 710)]
    for index, (name, box) in enumerate(zip(study.bands, boxes, strict=True)):
        values = outcome.distances[:, index]
        counts, _ = np.histogram(values, bins=edges)
        axes = Axes(canvas, box, (0.0, limit), (0.0, float(max(counts.max(), 1)) * 1.1))
        xticks = nice_ticks(0.0, limit, 6)
        yticks = nice_ticks(0.0, float(counts.max()) * 1.1, 4)
        counted = name in study.counted_bands
        axes.frame(
            _labels(xticks, _decimals(xticks)),
            _labels(yticks, 0),
            xlabel="D(d, s)",
            ylabel="dates",
            title=f"{name} ({'counted' if counted else 'reported only, not counted'})",
        )
        axes.bars(edges, counts, BAND_COLOURS[name])
        axes.vline(stats.THRESHOLD, INK, width=2)
        share = float((values >= stats.THRESHOLD).mean())
        canvas.text(
            (box[2] - 8, box[1] + 8),
            f"{100 * share:.0f}% of dates at or above {stats.THRESHOLD}",
            anchor="ra",
        )
    return canvas.png()


# --- figure 2: D against date ----------------------------------------------------------------


def _date_ticks(days: Sequence[dt.date]) -> list[tuple[float, str]]:
    first, last = days[0], days[-1]
    ticks: list[tuple[float, str]] = []
    month = dt.date(first.year, first.month, 1)
    while month <= last:
        if month >= first:
            ticks.append((float(month.toordinal()), month.strftime("%Y-%m")))
        month = dt.date(month.year + month.month // 12, month.month % 12 + 1, 1)
    if len(ticks) < 2:
        ticks = [
            (float(first.toordinal()), first.isoformat()),
            (float(last.toordinal()), last.isoformat()),
        ]
    return ticks


def by_date_figure(study: Study) -> bytes:
    """D(d, s) for the counted bands against the date, and the dates that meet the rule."""
    outcome = study.outcomes["nominal"]
    days = list(study.used_dates)
    ordinals = [float(day.toordinal()) for day in days]
    canvas = Canvas(1200, 640)
    canvas.text((600, 14), "D(d, s) by date, counted bands", font=canvas.large, anchor="ma")
    top = float(outcome.distances[:, :3].max())
    ylim = (0.0, max(top, stats.THRESHOLD * 2) * 1.1)
    span = (ordinals[0] - 1.0, max(ordinals[-1], ordinals[0] + 1.0) + 1.0)
    axes = Axes(canvas, (100, 70, 1150, 470), span, ylim)
    yticks = nice_ticks(*ylim, 5)
    axes.frame(_date_ticks(days), _labels(yticks, _decimals(yticks)), ylabel="D(d, s)")
    axes.hline(stats.THRESHOLD, INK, width=2)
    for index, name in enumerate(study.counted_bands):
        axes.line(ordinals, outcome.distances[:, index], BAND_COLOURS[name], width=2)
        for x, y in zip(ordinals, outcome.distances[:, index], strict=True):
            axes.marker(x, y, BAND_COLOURS[name], radius=2)
    strip = Axes(canvas, (100, 520, 1150, 560), span, (0.0, 1.0))
    canvas.draw.rectangle([strip.left, strip.top, strip.right, strip.bottom], outline=INK, width=1)
    for x, meets in zip(ordinals, outcome.meets, strict=True):
        if meets:
            canvas.draw.line(
                [(strip.px(x), strip.top + 3), (strip.px(x), strip.bottom - 3)], fill=INK, width=2
            )
    canvas.text((strip.left - 6, (strip.top + strip.bottom) / 2), "meets rule", anchor="rm")
    canvas.text(
        (strip.left, strip.bottom + 8),
        "a mark is a date on which D reaches 0.05 in at least 2 of the 3 counted bands",
    )
    _legend(canvas, 880, 80, [(name, BAND_COLOURS[name]) for name in study.counted_bands])
    return canvas.png()


# --- figure 3: the contrast profile and its spread -------------------------------------------


def profile_figure(study: Study) -> bytes:
    """Contrast against sun elevation for the lowest and highest band: median and 10th to 90th."""
    contrast = np.stack([record.contrast["nominal"] for record in study.used])  # (n, 29, 5, 2)
    elevations = list(ELEVATION_SWEEP)
    canvas = Canvas(1200, 820)
    canvas.text(
        (600, 14),
        "Rendered contrast against sun elevation, spread over the dates used",
        font=canvas.large,
        anchor="ma",
    )
    boxes = {
        (0, 0): (90, 80, 560, 340),
        (0, 1): (680, 80, 1150, 340),
        (1, 0): (90, 450, 560, 710),
        (1, 1): (680, 450, 1150, 710),
    }
    last = contrast.shape[2] - 1
    for (row, column), box in boxes.items():
        band = 0 if row == 0 else last
        which = "lowest" if row == 0 else "highest"
        values = contrast[:, :, band, column]  # (n, 29)
        low, median, high = np.percentile(values, [10, 50, 90], axis=0)
        ymin, ymax = float(values.min()), float(values.max())
        pad = max((ymax - ymin) * 0.08, 0.01)
        axes = Axes(
            canvas, box, (ELEVATION_SWEEP[-1], ELEVATION_SWEEP[0]), (ymin - pad, ymax + pad)
        )
        yticks = nice_ticks(ymin - pad, ymax + pad, 5)
        axes.frame(
            _labels([6, 4, 2, 0, -2, -4, -6, -8], 0),
            _labels(yticks, _decimals(yticks)),
            xlabel="sun elevation (degrees)",
            ylabel=CHANNEL_NAMES[column],
            title=f"{CHANNEL_NAMES[column]}, {which} band minus reference",
        )
        colour = BAND_COLOURS["mid"] if column == 0 else BAND_COLOURS["early"]
        axes.band(elevations, low, high, colour)
        axes.line(elevations, median, colour, width=3)
    canvas.text(
        (600, 790),
        "line: median over dates; shaded: 10th to 90th percentile; "
        "rendered toward the camera's azimuth",
        anchor="ma",
    )
    return canvas.png()


FIGURES: Final = {
    "d_distribution.png": distribution_figure,
    "d_by_date.png": by_date_figure,
    "profile_spread.png": profile_figure,
}
"""The figure files and the functions that draw them."""
