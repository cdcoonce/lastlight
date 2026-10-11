"""The bundled spectral tables: present, plausible and cited with a redistributable licence.

The design contract bans vendoring GPL physics code, and the issue requires the ozone table to be
redistributable under MIT terms. These tests keep the provenance claim next to the data.
"""

from __future__ import annotations

import csv
from importlib import resources

import pytest

DATA = resources.files("lastlight.render") / "data"


def _read_table(name: str) -> tuple[list[float], list[float]]:
    with (DATA / name).open(newline="") as handle:
        rows = list(csv.reader(handle))
    header, body = rows[0], rows[1:]
    assert header[0] == "wavelength_nm"
    return [float(r[0]) for r in body], [float(r[1]) for r in body]


@pytest.mark.parametrize("name", ["ozone_absorption.csv", "solar_irradiance.csv"])
def test_tables_cover_the_visible_range_in_increasing_wavelength_order(name: str) -> None:
    wavelengths, values = _read_table(name)
    assert wavelengths == sorted(set(wavelengths))
    assert wavelengths[0] <= 380.0
    assert wavelengths[-1] >= 780.0
    assert all(v >= 0.0 for v in values)


def test_ozone_table_peaks_in_the_chappuis_band() -> None:
    wavelengths, values = _read_table("ozone_absorption.csv")
    peak = wavelengths[values.index(max(values))]
    assert 560.0 <= peak <= 620.0


def test_data_readme_cites_the_source_and_licence_of_the_ozone_table() -> None:
    readme = (DATA / "README.md").read_text()
    assert "ozone_absorption.csv" in readme
    assert "Bird" in readme and "Riordan" in readme and "1986" in readme
    assert "BSD-3-Clause" in readme
    assert "MIT" in readme
    assert "solar_irradiance.csv" in readme
