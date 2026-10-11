# Bundled spectral tables

Two small tables used by the v0 renderer (`lastlight.render.atmosphere`). Both are the visible
rows (380 to 780 nm, 34 wavelengths, the native grid of the source) of the coefficient table of
the SPCTRL2 simple spectral model. The renderer interpolates them linearly onto its 10 nm grid.

| File | Column | Unit |
| --- | --- | --- |
| `ozone_absorption.csv` | `k_o3_per_atm_cm` | ozone absorption coefficient, per atm cm of ozone |
| `solar_irradiance.csv` | `irradiance_w_m2_nm` | extraterrestrial solar spectral irradiance, W m^-2 nm^-1 |

An ozone column of `D` Dobson units is `D / 1000` atm cm, so its vertical optical depth is
`k_o3 * D / 1000`. In the Chappuis band the peak is 0.12 per atm cm, a cross-section of about
4.5e-21 cm^2 per molecule. The table is zero at and below 440 nm, where the Huggins-band tail is
negligible for sky colour, and falls to zero at 780 nm.

## Source

R. E. Bird and C. Riordan, "Simple solar spectral model for direct and diffuse irradiance on
horizontal and tilted planes at the earth's surface for cloudless atmospheres", Journal of
Climate and Applied Meteorology 25, 87 to 97 (1986), also SERI/TR-215-2436 (Solar Energy
Research Institute, 1984). The ozone absorption coefficients and the extraterrestrial spectrum are
columns of the SPCTRL2 coefficient table.

The values were copied programmatically, row for row, from the copy of that table in
**pvlib-python 0.16.1** (`pvlib/spectrum/spectrl2.py`, the `_SPECTRL2_COEFFS` array), without
interpolation or rounding. The two files hold the rows with wavelength from 380 to 780 nm.

## Licence

pvlib-python is distributed under the BSD 3-Clause licence (SPDX: BSD-3-Clause), which permits redistribution of this
copy, with or without modification, provided the notice below is kept. That is compatible with
this repository's MIT licence. The underlying numbers are physical constants from a report written
at the Solar Energy Research Institute (now NREL, a US Department of Energy laboratory).

Not used, by design: libRadtran (GPL) and the GPL Python port `skylight`. Nothing here is derived
from either (`docs/design.md`, "Engines and licences").

```text
BSD 3-Clause License

Copyright (c) 2023 pvlib python Contributors
Copyright (c) 2014 PVLIB python Development Team
Copyright (c) 2013 Sandia National Laboratories

All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

  Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
