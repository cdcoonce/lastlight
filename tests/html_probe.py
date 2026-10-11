"""A tiny HTML reader for the almanac tests: every element with its attributes, and the text.

The site tests assert on generated files, not screenshots, so they need to ask "which images
lack alt text" and "which links point nowhere" without a browser or a parsing dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

_HIDDEN_TEXT = {"script", "style", "title"}
_VOID = {"img", "meta", "link", "br", "hr", "input"}


@dataclass
class Element:
    tag: str
    attrs: dict[str, str]
    parts: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join("".join(self.parts).split())

    def has_class(self, name: str) -> bool:
        return name in self.attrs.get("class", "").split()


class Probe(HTMLParser):
    """Parse one HTML document; `elements` keeps document order, `text` is the visible text."""

    def __init__(self, source: str) -> None:
        super().__init__(convert_charrefs=True)
        self.elements: list[Element] = []
        self._open: list[Element] = []
        self._text: list[str] = []
        self._hidden = 0
        self.feed(source)
        self.close()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        element = Element(tag, {name: value or "" for name, value in attrs})
        self.elements.append(element)
        if tag not in _VOID:
            self._open.append(element)
        if tag in _HIDDEN_TEXT:
            self._hidden += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _HIDDEN_TEXT and self._hidden:
            self._hidden -= 1
        for index in range(len(self._open) - 1, -1, -1):
            if self._open[index].tag == tag:
                del self._open[index:]
                break

    def handle_data(self, data: str) -> None:
        for element in self._open:
            element.parts.append(data)
        if not self._hidden:
            self._text.append(data)

    @property
    def text(self) -> str:
        return " ".join("".join(self._text).split())

    def find(self, tag: str, class_name: str | None = None) -> list[Element]:
        """Every `tag` element (with `class_name`, if given), in document order."""
        return [
            element
            for element in self.elements
            if element.tag == tag and (class_name is None or element.has_class(class_name))
        ]

    def texts(self, tag: str, class_name: str | None = None) -> list[str]:
        return [element.text for element in self.find(tag, class_name)]

    def headings(self) -> list[str]:
        return [element.text for element in self.elements if element.tag in {"h1", "h2", "h3"}]

    def hrefs(self) -> list[str]:
        return [element.attrs["href"] for element in self.find("a") if "href" in element.attrs]


def read_page(site: Path, name: str) -> Probe:
    return Probe((site / name).read_text(encoding="utf-8"))
