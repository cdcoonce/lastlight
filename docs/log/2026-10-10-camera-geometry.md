# Camera geometry: pixel to sky coordinates from the skyline

**Decided:** `lastlight.camgeom` maps every pixel of a public camera to a sky direction (issue #16). A pinhole `CameraModel` gives `pixel_to_sky` and `sky_to_pixel`; the orientation (azimuth, elevation, roll, field of view) is fitted by matching the skyline in the camera's frames to the horizon a public DEM predicts; the sun is an independent check; the result is frozen in `src/lastlight/cameras/geometry/WHTM1.json`; and `lastlight camgeom fit` and `mask` drive it.

**The real result (WHTM1, five frames fitted on, three held out):**

- Azimuth 288.7, elevation 1.66, roll 0.35, horizontal field of view 19.4 degrees. Skyline residual **1.65 px RMS (0.040 degrees), median 1.10 px**, over all 800 columns. The issue's target was under 2 px.
- **Held out.** The same frozen skyline, laid on three frames taken after the sun had set below the fit's cut-off (sun at -0.6, -3.8 and -6.8 degrees), gives 1.63, 1.65 and 2.01 px RMS, so the match is not an artefact of the five frames it was fitted on.
- **Sun check: skipped, and it will stay skipped until spring.** With a 19 degree lens pointed at 289, the sun (azimuth 251 to 258 on this evening) is 30 degrees outside the frame in all five frames. The sun crosses this frame at sunset from about April to early September.
- **The field of view is not pinned down.** Position and field of view trade off along a valley: eight of the 34 positions searched fit within 10 percent of the best, with azimuth 283.6 to 291.1 and field of view 17.7 to 24.9 degrees. The pixel scale (and with it the elevation of any pixel above the horizon) is therefore good to roughly 30 percent until the camera's position is known or the sun enters the frame. The fit is a first freeze, to be redone by the same command before the first scored evening.

**How the fit works, and why it has three stages:**

- **A coarse search on edge strength, then a trace, then least squares.** The first version traced the strongest edge in every column and fitted that. On real frames the strongest edges are cloud bottoms and the foreground, not the ridge. So the coarse stage scores each candidate by the edge strength the predicted curve passes through (no skyline is chosen yet), the best candidate is traced in a band around its prediction by a dynamic program that keeps the path continuous, and a robust (soft L1) least-squares fit moves azimuth, elevation, roll and field of view onto the median of the frames' traces, twice, with a narrower band each time.
- **Sky brighter than terrain is the only edge that counts.** `edge_strength` is the drop in brightness going down; a cloud's bottom or a white stamp's top never scores.
- **The timestamp is excluded, and the box is widened.** The smoothing and the comparison spread a bright stamp's edge a few rows beyond its rectangle, so an excluded box is widened by their reach. Without that, the path rode the edge just below the box.
- **Horizon from the terrain, with curvature built in.** For each azimuth, the largest angle `atan2((R + H) cos(s/R) - (R + h), (R + H) sin(s/R))` over a ray sampled every 30 m from 2 to 50 km; terrain nearer than 2 km (houses and trees beside the mast) is skipped. A synthetic cone matches the analytic elevation within 0.05 degrees; a level plain gives the geometric dip; and a test shows the flat-Earth answer would be 0.12 degrees off, so the cone test cannot pass without curvature.

**The first answer was wrong, and the data said so.** At the described site the best rectilinear fit was a 72 degree lens with 3.6 px RMS, and the leftmost 100 columns lay 10 px below the prediction: the observed ridge there sat at an elevation of -1.4 degrees, lower than any horizon can be for a camera a few metres above flat ground. A radial term would have absorbed it (2.8 px) but pegged at its bound and stretched the lens by 17 percent: a sign that the lens model, not the lens, was wrong. A scan of positions with the field-of-view range widened to 10 to 120 degrees found a long valley of fits at 1.6 to 1.9 px with 15 to 25 degree lenses, positions a few kilometres north and east, and a median residual near 1 px. A narrow lens is what a visibility camera is for. The search that is now part of `fit` (`--position-box-km`) is that scan, made repeatable: a coarse grid over a stated box, a finer grid around the best coarse position, never leaving the box, never choosing a fit pinned to a field-of-view limit.

**Position stays out of the repository.** The privacy scanner (P003) flags any coordinate pair in Arizona except the reference point, and the issue's own plan to commit a position per camera would trip it. `lastlight camgeom fit` reads position and view direction from a private site file (`--site-file` or `LASTLIGHT_CAMERA_SITES`); the frozen fit holds orientation only, and the position search records its result as an offset in kilometres from that file's point. `CameraModel` keeps `lat`, `lon` and `elev_m` as the issue specifies, and `CameraFit.camera_model(lat, lon, elev_m)` takes them from the caller; the pixel-to-sky mapping does not depend on them.

**Deviations from the issue and from `docs/design.md`:**

- The issue asks for the skyline of a clear daytime frame. The archive held no clear frame (five cloudy frames from the late afternoon). The fit therefore uses every frame with the sun at or above the horizon, takes the median of their traces, and says how many frames it used.
- `docs/design.md` said geometry is "calibrated from the sun's track". The issue makes skyline matching primary and the sun an independent check, which is better (a fit input cannot also be a test) and is now what the design says.
- The position is searched, not fixed. The issue allows it ("within a stated prior box"); the box is 8 km because the operator says "Avondale", which is not a point.

**Rejected alternatives:** a radial term in the model (not identified by this data, the diagnostic is recorded instead, see `radial_term` in the fit file); streaming the DEM with GDAL's `/vsicurl/` (a one-time 40 MB download to a cache is simpler and makes every later fit offline); a hand-written TIFF reader (rasterio reads the Copernicus tiles and merges neighbours); scipy's optimiser replaced by a hand-rolled one (the robust loss is the point); fitting the sun as an input.

**What broke:**

- Bilinear sampling beside a no-data hole returned NaN even at a node with full weight, because `0 * NaN` is NaN. Corners with no weight are now dropped from the sum.
- A DEM window ending exactly at the camera's longitude left the camera on the edge, outside the posts that were read. The window is padded, and the tiles are named from the unpadded box: padding by 2 km first pulled in a whole neighbouring 40 MB tile for the last kilometre of one ray.
- A position search that scored the best residual alone chose fits with the field of view pinned at its limit; those are excluded and counted.
- The fit file's summary held NumPy integers and booleans, which JSON refuses; the command test that runs the search caught it.
- The repository's own privacy scanner flagged this slice's code: a field-of-view range whose two limits fell inside the latitude and longitude boxes read as a coordinate pair (the range is now 10 to 120), literal grid corners in the DEM tests, and a docstring that spelled a time of day. Tests now derive positions from the reference point, and the fit tests use the scanner's own rules as their probe.

**Next:** observe (issue #17) builds on `CameraFit` and `sky_mask`. Refit WHTM1 when clear-day frames exist and again when the sun enters the frame, and freeze that fit before the first scored evening.
