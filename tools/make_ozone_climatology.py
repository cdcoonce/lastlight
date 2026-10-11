"""Rebuild the ozone source rows and check the table against them (needs the network).

Usage: `uv run python tools/make_ozone_climatology.py`

It downloads NASA Goddard's merged SBUV total ozone record (MOD v8.7, monthly zonal means in
5 degree bands), keeps the rows of the 30 to 35 N band for 2005 to 2024, writes them verbatim
(year prefixed) to `tests/fixtures/ozone/sbuv_v87_mod_30-35N_2005-2024.txt`, and prints the
monthly means (months with no data, 0.0 in the file, left out; rounded to 0.1 DU). It exits
non-zero if those means differ from `lastlight.atmosphere.OZONE_DU_BY_MONTH`.

Source: https://acd-ext.gsfc.nasa.gov/Data_services/merged/data/sbuv.v87.mod_v14.70-25.za.txt
(NASA data, a US Government work, public domain in the United States). The file is rewritten
as new years arrive, so a rerun years from now may differ in the last digit; the committed rows
are what the table is derived from, and `tests/test_atmosphere_reduction.py` recomputes the table
from them offline.
"""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

from lastlight.atmosphere import OZONE_DU_BY_MONTH, OZONE_TABLE_PERIOD
from lastlight.inputs.http import USER_AGENT

URL = "https://acd-ext.gsfc.nasa.gov/Data_services/merged/data/sbuv.v87.mod_v14.70-25.za.txt"
BAND = ("30", "35")
OUT = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "ozone"
    / "sbuv_v87_mod_30-35N_2005-2024.txt"
)
HEADER = (
    "# NASA Goddard merged SBUV total ozone (MOD v8.7), monthly zonal means, 5 degree band 30 to "
    "35 N,\n"
    "# Dobson units, 0.0 = no data. Rows copied verbatim (year prefixed) from\n"
    f"# {URL}\n"
    "# years 2005 to 2024. Columns: year, band south edge, band north edge, then Jan to Dec.\n"
)


def band_rows(text: str, first: int, last: int) -> list[tuple[int, str]]:
    """(year, verbatim row) of the band for each year from `first` to `last`."""
    year: int | None = None
    rows = []
    for line in text.splitlines():
        title = re.match(r"(\d{4}) SBUV_V87", line)
        if title:
            year = int(title.group(1))
        elif year is not None and first <= year <= last and line.split()[:2] == list(BAND):
            rows.append((year, line))
    return rows


def monthly_means(rows: list[tuple[int, str]]) -> list[float]:
    columns: list[list[float]] = [[] for _ in range(12)]
    for _year, line in rows:
        for column, value in zip(columns, line.split()[2:], strict=True):
            if float(value) > 0.0:
                column.append(float(value))
    return [round(sum(column) / len(column), 1) for column in columns]


def main() -> int:
    first, last = OZONE_TABLE_PERIOD
    request = urllib.request.Request(URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        text = response.read().decode("ascii")
    rows = band_rows(text, first, last)
    if len(rows) != last - first + 1:
        print(f"expected {last - first + 1} rows of the band, found {len(rows)}", file=sys.stderr)
        return 1
    OUT.write_text(HEADER + "".join(f"{year} {line}\n" for year, line in rows), encoding="utf-8")
    means = monthly_means(rows)
    print("monthly means (DU), January to December:", means)
    if tuple(means) != OZONE_DU_BY_MONTH:
        print("these differ from lastlight.atmosphere.OZONE_DU_BY_MONTH", file=sys.stderr)
        return 1
    print(f"matches OZONE_DU_BY_MONTH; rows written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
