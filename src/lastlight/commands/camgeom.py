"""`lastlight camgeom`: fit a public camera's pixel-to-sky geometry, cut its sky mask, and
check a frozen fit against frames it was not made from.

    lastlight camgeom fit     --camera WHTM1 --frames DIR [--site-file FILE] [--dem TIF ...]
    lastlight camgeom mask    --camera WHTM1 [--margin-deg 1.0] [--out FILE]
    lastlight camgeom holdout --camera WHTM1 --frames DIR [DIR ...] [--fit FILE] [--band 12]

The logic lives in `lastlight.camgeom`; this module only reads arguments, calls it and prints
a summary. Exit code 2 means an input was missing or unusable, and says which, in one line.
`fit` exits 1 when the sun check failed: the sun disagrees with the fit, so the file (written
anyway, so that it can be inspected) must not be trusted. A skipped sun check exits 0, and the
file says `provisional`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from lastlight.cli import Command

DEFAULT_MASK_DIR = Path("data/masks")


def _fail(message: str) -> int:
    print(f"lastlight camgeom: {message}", file=sys.stderr)
    return 2


def _configure(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="action", title="actions", required=True)

    fit = actions.add_parser(
        "fit",
        help="fit a camera's azimuth, elevation, roll and field of view and freeze the result",
        description=(
            "Match the skyline in a camera's archived frames to the horizon a public DEM "
            "predicts from the camera's position, fit azimuth, elevation, roll and field of "
            "view, check the sun, and write src/lastlight/cameras/geometry/<ID>.json."
        ),
    )
    fit.add_argument("--camera", required=True, help="camera id, for example WHTM1")
    fit.add_argument("--frames", required=True, type=Path, help="directory of the camera's frames")
    fit.add_argument(
        "--site-file",
        type=Path,
        help="TOML file with the camera's position and view direction "
        "(default: $LASTLIGHT_CAMERA_SITES)",
    )
    fit.add_argument(
        "--dem",
        type=Path,
        nargs="+",
        help="local GeoTIFF DEM tile(s) to use instead of downloading Copernicus GLO-30",
    )
    fit.add_argument(
        "--dem-cache",
        type=Path,
        help="where downloaded tiles are kept (default: $LASTLIGHT_DEM_CACHE or ~/.cache)",
    )
    fit.add_argument(
        "--out", type=Path, help="fit file to write (default: the package's cameras/geometry)"
    )
    fit.add_argument(
        "--min-sun-elevation",
        type=float,
        default=0.0,
        help="skip frames with the sun below this elevation in degrees (default 0)",
    )
    fit.add_argument(
        "--position-box-km",
        type=float,
        default=0.0,
        help="search the camera's position inside +-this many km of the site file's position "
        "and fit there (default 0: the camera stands at the site)",
    )
    fit.add_argument(
        "--position-step-km",
        type=float,
        default=1.0,
        help="grid spacing of the position search, which fits at every point of the box and "
        "then at a ring of probes one step outside it (default 1; the 8 km box of WHTM1 takes "
        "361 fits and about 10 minutes)",
    )
    fit.add_argument(
        "--frozen-on",
        type=dt.date.fromisoformat,
        default=None,
        help="the date to record as the freeze date, YYYY-MM-DD (default: today)",
    )
    fit.set_defaults(action_run=_run_fit)

    mask = actions.add_parser(
        "mask",
        help="write the sky mask of a fitted camera as a PNG",
        description="Pixels above the fitted skyline minus a margin, outside the timestamp.",
    )
    mask.add_argument("--camera", required=True, help="camera id, for example WHTM1")
    mask.add_argument(
        "--geometry-dir", type=Path, help="where the fits live (default: the package's)"
    )
    mask.add_argument(
        "--margin-deg", type=float, default=1.0, help="degrees kept clear above the skyline"
    )
    mask.add_argument(
        "--out", type=Path, help=f"PNG to write (default: {DEFAULT_MASK_DIR}/<ID>.png)"
    )
    mask.set_defaults(action_run=_run_mask)

    holdout = actions.add_parser(
        "holdout",
        help="lay a frozen skyline over frames the fit was not made from",
        description=(
            "Trace the skyline in frames the fit was not frozen on, in a band around the "
            "frozen skyline, and print how far apart the two are: one line per frame, "
            "identified by the start of its SHA-256."
        ),
    )
    holdout.add_argument("--camera", required=True, help="camera id, for example WHTM1")
    holdout.add_argument(
        "--frames",
        required=True,
        type=Path,
        nargs="+",
        help="directories of the camera's frames; frames the fit was frozen on are skipped",
    )
    holdout.add_argument(
        "--fit", type=Path, help="fit file to check (default: the package's <ID>.json)"
    )
    holdout.add_argument(
        "--band",
        type=float,
        default=12.0,
        help="half-width in rows of the band around the frozen skyline in which each frame's "
        "skyline is looked for (default 12, the width of the fit's own last pass)",
    )
    holdout.set_defaults(action_run=_run_holdout)


def _run(args: argparse.Namespace) -> int:
    return args.action_run(args)


def _run_fit(args: argparse.Namespace) -> int:
    from lastlight.camgeom import dem as dem_module
    from lastlight.camgeom.calibration import CalibrationError, calibrate_camera, view_azimuths
    from lastlight.camgeom.fitfile import GEOMETRY_DIR, write_fit
    from lastlight.camgeom.frames import discover_frames
    from lastlight.camgeom.sites import SiteFileError, load_camera_site, resolve_site_file

    try:
        site = load_camera_site(args.camera, resolve_site_file(args.site_file))
        discover_frames(args.frames)  # fail early, before any terrain is read
        if args.dem:
            missing = [path for path in args.dem if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"DEM file not found: {missing[0]}")
        azimuths = view_azimuths(site.view_azimuth_deg)
        max_range_km = 50.0
        rays = azimuths[::20]
        reach = dem_module.view_bounds(site.lat, site.lon, rays, max_range_km)
        paths = args.dem or dem_module.fetch_tiles(
            dem_module.tiles_for_bounds(reach[1], reach[3], reach[0], reach[2]),
            args.dem_cache or dem_module.default_cache_dir(),
        )
        # room for the camera to be moved, and for the ring of probes just outside the box
        reach_km = args.position_box_km + args.position_step_km if args.position_box_km > 0 else 0.0
        margin = 0.02 + reach_km / 90.0
        bounds = dem_module.view_bounds(site.lat, site.lon, rays, max_range_km, margin_deg=margin)
        terrain = dem_module.read_dem(paths, bounds)
        calibration = calibrate_camera(
            args.camera,
            site,
            args.frames,
            terrain,
            frozen_on=args.frozen_on or dt.date.today(),
            min_sun_elevation_deg=args.min_sun_elevation,
            max_range_km=max_range_km,
            position_box_km=args.position_box_km,
            position_step_km=args.position_step_km,
            stated_azimuth_deg=_stated_azimuth(args.camera),
        )
        out = args.out or GEOMETRY_DIR / f"{args.camera}.json"
        write_fit(calibration.fit, out)
    except (CalibrationError, SiteFileError, FileNotFoundError, ValueError, OSError) as error:
        return _fail(" ".join(str(error).split()))

    _print_fit_summary(args.camera, calibration, out, args.min_sun_elevation)
    if calibration.sun_check.status == "failed":
        print(
            f"lastlight camgeom: the sun check FAILED: {calibration.sun_check.reason}; "
            f"{out} was written so that it can be inspected, and must not be trusted",
            file=sys.stderr,
        )
        return 1
    return 0


def _stated_azimuth(camera_id: str) -> float | None:
    """The view direction the camera's operator states, from the registry; None if unlisted."""
    from lastlight.cameras.registry import load_registry

    camera = load_registry().get(camera_id)
    return None if camera is None else camera.view_azimuth_deg


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _print_fit_summary(camera: str, calibration, out: Path, min_sun: float) -> None:
    fit, o = calibration.fit, calibration.fit.orientation
    notes = []
    if calibration.frames_too_dark:
        notes.append(
            f"{_plural(calibration.frames_too_dark, 'frame')} skipped: "
            f"sun below {min_sun:g} degrees"
        )
    if calibration.frames_without_time:
        notes.append(f"{calibration.frames_without_time} without a time in the file name")
    aside = f" ({'; '.join(notes)})" if notes else ""
    radial = fit.radial_term or {}
    print(f"{camera}: fitted on {_plural(calibration.frames_used, 'frame')}{aside}")
    print(
        f"{camera}: azimuth {o.azimuth_deg:.2f}, elevation {o.elevation_deg:.2f}, "
        f"roll {o.roll_deg:.2f}, hfov {o.hfov_deg:.2f} degrees"
    )
    print(
        f"{camera}: skyline rms {fit.skyline_rms_px:.2f} px ({fit.skyline_rms_deg:.3f} degrees), "
        f"median {fit.skyline_median_abs_px:.2f} px, over {fit.n_columns} columns"
    )
    print(
        f"{camera}: radial term k1 {radial.get('k1', 0.0):+.3f} would give rms "
        f"{radial.get('skyline_rms_px', 0.0):.2f} px (not adopted)"
    )
    search = fit.position_search
    if search is not None:
        print(
            f"{camera}: position search over {search['positions']} positions, offset "
            f"{search['offset_east_km']:+.1f} km east {search['offset_north_km']:+.1f} km north "
            f"of the site file; {search['near_best_positions']} within 10 percent of the best "
            f"rms, azimuth {search['near_best_azimuth_deg_range'][0]:.1f} to "
            f"{search['near_best_azimuth_deg_range'][1]:.1f}, hfov "
            f"{search['near_best_hfov_deg_range'][0]:.1f} to "
            f"{search['near_best_hfov_deg_range'][1]:.1f} degrees"
            + (" (the best is on the edge of the box)" if search["best_on_box_edge"] else "")
        )
        if search["ring_positions"]:
            beyond = search["residual_falls_beyond_box"]
            print(
                f"{camera}: ring of {search['ring_positions']} probes just outside the box: best "
                f"{search['ring_rms_px_best']:.2f} px against {search['rms_px_best']:.2f} inside; "
                + ("the residual keeps falling beyond the box" if beyond else "it does not fall")
            )
    if fit.stated_azimuth_deg is not None:
        print(
            f"{camera}: fitted azimuth is {fit.azimuth_minus_stated_deg:+.1f} degrees from the "
            f"operator's stated {fit.stated_azimuth_deg:g}"
        )
    if fit.uncertainty is not None:
        u = fit.uncertainty
        print(
            f"{camera}: uncertainty (nearly equal fits): azimuth {u.azimuth_deg[0]:.1f} to "
            f"{u.azimuth_deg[1]:.1f}, hfov {u.hfov_deg[0]:.1f} to {u.hfov_deg[1]:.1f} degrees, "
            f"{u.px_per_deg[0]:.1f} to {u.px_per_deg[1]:.1f} px per degree"
            + (" (lower bounds)" if u.lower_bounds else "")
        )
    print(f"{camera}: sun check {fit.sun_check['status']}: {fit.sun_check['reason']}")
    if fit.provisional:
        print(
            f"{camera}: PROVISIONAL: the sun check has not passed, the pixel scale is unconfirmed"
        )
    print(f"{camera}: wrote {out}")


def _run_mask(args: argparse.Namespace) -> int:
    from lastlight.camgeom.fitfile import GEOMETRY_DIR, FitFileError, load_fit
    from lastlight.camgeom.mask import sky_mask, write_mask_png

    if args.margin_deg < 0:
        return _fail("the margin must not be negative")
    try:
        fit = load_fit(args.camera, args.geometry_dir or GEOMETRY_DIR)
    except FitFileError as error:
        return _fail(str(error))
    mask = sky_mask(fit, margin_deg=args.margin_deg)
    out = write_mask_png(mask, args.out or DEFAULT_MASK_DIR / f"{args.camera}.png")
    print(
        f"{args.camera}: sky mask {mask.shape[1]}x{mask.shape[0]}, "
        f"{100.0 * mask.mean():.1f}% sky with a {args.margin_deg:g} degree margin, wrote {out}"
    )
    if fit.provisional:
        print(
            f"{args.camera}: PROVISIONAL fit: the margin of {args.margin_deg:g} degrees is "
            "converted to rows with a pixel scale the sun has not confirmed"
        )
    return 0


def _run_holdout(args: argparse.Namespace) -> int:
    from lastlight.camgeom.fitfile import GEOMETRY_DIR, FitFileError, load_fit_file
    from lastlight.camgeom.holdout import held_out_paths, holdout_paths

    fit_path = args.fit or GEOMETRY_DIR / f"{args.camera}.json"
    try:
        fit = load_fit_file(fit_path)
        if fit.camera_id != args.camera:
            return _fail(f"{fit_path.name} is a fit for {fit.camera_id}, not {args.camera}")
        paths, skipped = held_out_paths(fit, args.frames)
        if not paths:
            return _fail(
                f"no held-out frame: {skipped} frames were found and all were used in the fit"
                if skipped
                else "no frame found under " + ", ".join(str(root) for root in args.frames)
            )
        results = holdout_paths(fit, paths, band_rows=args.band)
    except (FitFileError, FileNotFoundError, ValueError, OSError) as error:
        return _fail(" ".join(str(error).split()))

    left_out = f"; {_plural(skipped, 'frame')} used in the fit left out" if skipped else ""
    print(
        f"{args.camera}: frozen skyline traced within +-{args.band:g} rows on "
        f"{_plural(len(results), 'held-out frame')}{left_out}"
    )
    for index, result in enumerate(results, start=1):
        print(
            f"{args.camera}: held-out frame {index} (sha256 {result.label}): "
            f"rms {result.rms_px:.2f} px ({result.rms_deg:.3f} degrees), "
            f"median {result.median_abs_px:.2f} px, over {result.n_columns} columns"
        )
    print(f"{args.camera}: band +-{args.band:g} rows; fit rms was {fit.skyline_rms_px:.2f} px")
    return 0


COMMAND = Command(
    name="camgeom",
    help="fit a public camera's pixel-to-sky geometry, cut its sky mask, check a fit",
    run=_run,
    configure=_configure,
)
