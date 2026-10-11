"""Camera geometry: which direction in the sky each pixel of a public webcam looks at.

The modules, in the order the data flows through them:

- `model`: the pinhole `CameraModel`, pixel to (azimuth, elevation) and back.
- `dem`: terrain elevations (Copernicus GLO-30) and the horizon a terrain model predicts.
- `skyline`: the sky-to-terrain edge found in a frame.
- `orientation`: the least-squares match of the two skylines that fixes azimuth, elevation,
  roll and field of view.
- `suncheck`: an independent test of the fit against the sun's disk.
- `fitfile`, `mask`: the frozen fit (`src/lastlight/cameras/geometry/<ID>.json`) and the sky
  mask drawn from it.
- `frames`, `sites`, `calibration`: the archive, the private camera site file and the
  pipeline that ties it all together for `lastlight camgeom fit`.
"""

from lastlight.camgeom.model import CameraModel

__all__ = ["CameraModel"]
