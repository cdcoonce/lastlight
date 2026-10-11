"""The v0 atmosphere: what is in the air, and how much light each constituent removes per km.

An `Atmosphere` is four numbers that vary day to day (aerosol optical depth, ozone, surface
pressure and the aerosol's spectral slope) plus the fixed shape of the aerosol. `layer_optics`
turns it into per-km coefficients on a stack of concentric shells, which is what the
single-scattering integral in `lastlight.render.transfer` consumes.

Heights are measured above the observer, whose surface is the sphere of radius
`EARTH_RADIUS_KM`. The surface pressure in the `Atmosphere` is the pressure at the observer.

Constituents:

* Rayleigh scattering by air: cross-section from the Rayleigh formula in the form used by
  Bucholtz (1995), exponential density with an 8 km scale height, column set by pressure.
* Aerosol: extinction following the Angstrom law, exponential profile with its own scale height;
  a share `single_scatter_albedo` of the extinction scatters, the rest is absorbed.
* Ozone: absorption only, from the bundled Chappuis-band table, in a Gaussian layer centred at
  22 km.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache, lru_cache
from importlib import resources

import numpy as np

EARTH_RADIUS_KM = 6371.0
ATMOSPHERE_TOP_KM = 100.0
RAYLEIGH_SCALE_HEIGHT_KM = 8.0
OZONE_LAYER_CENTRE_KM = 22.0
OZONE_LAYER_SIGMA_KM = 6.0
"""Standard deviation of the ozone layer: a full width at half maximum of about 14 km."""

REFERENCE_WAVELENGTH_NM = 550.0
STANDARD_PRESSURE_HPA = 1013.25

WAVELENGTHS_NM = np.arange(380.0, 780.0 + 1.0, 10.0)
WAVELENGTHS_NM.flags.writeable = False

LEVELS_KM = np.concatenate(
    [
        np.linspace(0.0, 1.0, 20, endpoint=False),  # 50 m shells where the aerosol lives
        np.linspace(1.0, 5.0, 16, endpoint=False),  # 250 m shells through the boundary layer
        np.linspace(5.0, ATMOSPHERE_TOP_KM, 96),  # 1 km shells above, fine against 8 km and 6 km
    ]
)
LEVELS_KM.flags.writeable = False
"""Shell boundaries in km above the observer: 131 shells from the ground to 100 km."""

# --- Rayleigh scattering by air ----------------------------------------------------------------

_AIR_NUMBER_DENSITY_CM3 = 2.546899e19  # molecules per cm^3 at 288.15 K and 1013.25 hPa
_AIR_MOLAR_MASS_KG_MOL = 28.964e-3  # dry air with 300 ppm CO2, the Peck and Reeder composition
_AVOGADRO = 6.02214076e23
_STANDARD_GRAVITY = 9.80665


def _king_factor(wavelength_um: np.ndarray) -> np.ndarray:
    """Air's depolarisation (King) correction factor, per Bodhaine et al. (1999), 300 ppm CO2."""
    inv2 = wavelength_um**-2
    f_n2 = 1.034 + 3.17e-4 * inv2
    f_o2 = 1.096 + 1.385e-3 * inv2 + 1.448e-4 * inv2**2
    co2 = 0.03  # percent by volume
    return (78.084 * f_n2 + 20.946 * f_o2 + 0.934 * 1.0 + co2 * 1.15) / (
        78.084 + 20.946 + 0.934 + co2
    )


def rayleigh_cross_section_cm2(wavelength_nm: float | np.ndarray) -> np.ndarray:
    """Rayleigh scattering cross-section per air molecule, cm^2.

    sigma = 24 pi^3 (n^2 - 1)^2 / (lambda^4 N^2 (n^2 + 2)^2) * F, with the refractive index n of
    standard air from Peck and Reeder (1972) and the King factor F for its depolarisation.
    """
    wavelength_um = np.asarray(wavelength_nm, dtype=float) * 1e-3
    inv2 = wavelength_um**-2
    n = 1.0 + (5791817.0 / (238.0185 - inv2) + 167909.0 / (57.362 - inv2)) * 1e-8
    wavelength_cm = wavelength_um * 1e-4
    numerator = 24.0 * math.pi**3 * (n**2 - 1.0) ** 2
    denominator = wavelength_cm**4 * _AIR_NUMBER_DENSITY_CM3**2 * (n**2 + 2.0) ** 2
    return numerator / denominator * _king_factor(wavelength_um)


def rayleigh_optical_depth(
    wavelength_nm: float | np.ndarray, pressure_hpa: float = STANDARD_PRESSURE_HPA
) -> np.ndarray:
    """Vertical Rayleigh optical depth above a surface at `pressure_hpa`.

    The air column is pressure over (molar mass times gravity), so the optical depth is
    proportional to pressure. At 550 nm and 1013.25 hPa it is 0.0969; Bucholtz (1995) gives 0.0973.
    """
    column_cm2 = (
        pressure_hpa * 100.0 / (_AIR_MOLAR_MASS_KG_MOL * _STANDARD_GRAVITY) * _AVOGADRO * 1e-4
    )
    return rayleigh_cross_section_cm2(wavelength_nm) * column_cm2


# --- bundled tables ------------------------------------------------------------------------------


@cache
def _load_table(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a two-column CSV from `render/data`: wavelengths (nm) and values."""
    text = (resources.files("lastlight.render") / "data" / name).read_text()
    rows = [line.split(",") for line in text.strip().splitlines()[1:]]
    wavelengths = np.array([float(row[0]) for row in rows])
    values = np.array([float(row[1]) for row in rows])
    return wavelengths, values


def _interpolated(name: str, wavelength_nm: float | np.ndarray) -> np.ndarray:
    wavelengths, values = _load_table(name)
    return np.interp(np.asarray(wavelength_nm, dtype=float), wavelengths, values)


def ozone_absorption_per_atm_cm(wavelength_nm: float | np.ndarray) -> np.ndarray:
    """Ozone absorption coefficient k, per (atm cm) of ozone, linearly interpolated in wavelength.

    Optical depth is k times the column in atm cm; 1 Dobson unit is 0.001 atm cm.
    """
    return _interpolated("ozone_absorption.csv", wavelength_nm)


def solar_irradiance(wavelength_nm: float | np.ndarray) -> np.ndarray:
    """Extraterrestrial solar spectral irradiance at mean distance, W m^-2 nm^-1."""
    return _interpolated("solar_irradiance.csv", wavelength_nm)


# --- the atmosphere ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Atmosphere:
    """The parameters of one evening's atmosphere over the observer.

    `aod_550` is the vertical aerosol optical depth at 550 nm; `angstrom` its spectral slope;
    `aerosol_scale_height_km` the e-folding height of its exponential profile. `ozone_du` is the
    ozone column in Dobson units and `pressure_hpa` the pressure at the observer. `asymmetry` is
    the Henyey-Greenstein parameter g of the aerosol phase function and `single_scatter_albedo`
    the share of aerosol extinction that scatters instead of absorbing.
    """

    aod_550: float
    angstrom: float = 1.3
    aerosol_scale_height_km: float = 1.5
    ozone_du: float = 280
    pressure_hpa: float = 970
    asymmetry: float = 0.7
    single_scatter_albedo: float = 0.93

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.aod_550 < 0:
            raise ValueError(f"aod_550 must be >= 0, got {self.aod_550!r}")
        if self.aerosol_scale_height_km <= 0:
            raise ValueError("aerosol_scale_height_km must be > 0")
        if self.ozone_du < 0:
            raise ValueError(f"ozone_du must be >= 0, got {self.ozone_du!r}")
        if self.pressure_hpa <= 0:
            raise ValueError(f"pressure_hpa must be > 0, got {self.pressure_hpa!r}")
        if not -1.0 < self.asymmetry < 1.0:
            raise ValueError(f"asymmetry must be in (-1, 1), got {self.asymmetry!r}")
        if not 0.0 <= self.single_scatter_albedo <= 1.0:
            raise ValueError(
                f"single_scatter_albedo must be in [0, 1], got {self.single_scatter_albedo!r}"
            )


@dataclass(frozen=True, eq=False)
class LayerOptics:
    """Per-km coefficients on each shell, shape (shells, wavelengths)."""

    rayleigh_scattering: np.ndarray
    aerosol_extinction: np.ndarray
    aerosol_scattering: np.ndarray
    ozone_absorption: np.ndarray

    @property
    def extinction(self) -> np.ndarray:
        """Everything that removes light from a beam: Rayleigh, aerosol and ozone."""
        return self.rayleigh_scattering + self.aerosol_extinction + self.ozone_absorption


def _exponential_fractions(scale_height_km: float) -> np.ndarray:
    """Share of an exponential column (to infinity) that lies in each shell."""
    cumulative = np.exp(-LEVELS_KM / scale_height_km)
    return cumulative[:-1] - cumulative[1:]


def _ozone_fractions() -> np.ndarray:
    """Share of the ozone column in each shell: a Gaussian layer cut off at the ground and top."""
    z = (LEVELS_KM - OZONE_LAYER_CENTRE_KM) / (OZONE_LAYER_SIGMA_KM * math.sqrt(2.0))
    cumulative = np.array([math.erf(value) for value in z])
    return np.diff(cumulative) / (cumulative[-1] - cumulative[0])


@lru_cache(maxsize=64)
def layer_optics(atm: Atmosphere) -> LayerOptics:
    """Per-km scattering and absorption coefficients of each shell, for every wavelength.

    Each coefficient is the species' vertical optical depth times the share of its column in the
    shell, divided by the shell's thickness; summing coefficient times thickness over the shells
    returns the vertical optical depth exactly.
    """
    thickness = np.diff(LEVELS_KM)[:, None]
    wavelengths = WAVELENGTHS_NM[None, :]

    tau_rayleigh = rayleigh_optical_depth(WAVELENGTHS_NM, atm.pressure_hpa)[None, :]
    rayleigh = tau_rayleigh * _exponential_fractions(RAYLEIGH_SCALE_HEIGHT_KM)[:, None] / thickness

    tau_aerosol = atm.aod_550 * (wavelengths / REFERENCE_WAVELENGTH_NM) ** -atm.angstrom
    aerosol = tau_aerosol * _exponential_fractions(atm.aerosol_scale_height_km)[:, None] / thickness

    tau_ozone = (ozone_absorption_per_atm_cm(WAVELENGTHS_NM) * atm.ozone_du / 1000.0)[None, :]
    ozone = tau_ozone * _ozone_fractions()[:, None] / thickness

    optics = LayerOptics(
        rayleigh_scattering=rayleigh,
        aerosol_extinction=aerosol,
        aerosol_scattering=atm.single_scatter_albedo * aerosol,
        ozone_absorption=ozone,
    )
    for array in vars(optics).values():
        array.flags.writeable = False  # cached and shared: callers must not mutate it
    return optics
