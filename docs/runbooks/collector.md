# Runbook: the public-camera collector

For the owner. The collector archives the Arizona DEQ PhoenixVis cameras' frames (tier A truth, `docs/design.md` section 1) during each evening's twilight window. The cameras keep no usable archive, so an evening that is not collected is lost for good.

Agents never install the timer; the owner does. Nothing in this repo installs or starts it for you.

## What it does

- Every day at 15:00 local time systemd starts `lastlight collect`. The collector works out the rest: it waits until the sun at each camera falls below 8 degrees, polls every 5 minutes (the minimum; lower values are refused) until the sun is below -10 degrees, then exits 0.
- It sends an identifying `User-Agent` and conditional requests, so an unchanged frame costs a 304 and no download.
- Frames land in a private archive on this host, never in git. The command refuses (exit 2) to use an archive inside a git work tree.

```
<archive>/<camera id>/<YYYY-MM-DD>/<frame time, UTC>.jpg     the frame, named by its own Last-Modified
<archive>/<camera id>/<YYYY-MM-DD>/<frame time, UTC>.json    URL, status, kept headers, fetch time, sun elevation
<archive>/<camera id>/<YYYY-MM-DD>/log.jsonl                 one line per poll: saved, not_modified, unchanged, stale, error
```

The date folder is the local solar date of the evening, so one evening is one folder even when it passes UTC midnight. A frame whose Last-Modified is more than 20 minutes old is logged as `stale` and not saved. The default archive is `~/.local/share/lastlight/frames`; set `LASTLIGHT_ARCHIVE` to move it. The files are an estimated 50 MB a month per camera; check the real figure after a week.

## Install (once, on the always-on Ubuntu host)

1. Make sure the host has `git` and [`uv`](https://docs.astral.sh/uv/) and that its clock zone is where you want 15:00 to be. Check with `timedatectl`. The collector works at any zone, but a zone far from Phoenix just means it starts early and waits longer.

2. Clone and set up the project:

   ```sh
   cd ~
   git clone https://github.com/cdcoonce/lastlight.git
   cd lastlight
   uv sync --frozen --no-dev
   ```

   The unit expects the clone at `~/lastlight`. If you clone elsewhere, edit `WorkingDirectory` and `ExecStart` in the service file after copying it.

3. Optional: choose an archive outside the clone. The default is fine. To use another location, add `Environment=LASTLIGHT_ARCHIVE=/path/to/frames` under `[Service]` in the copied service file. It must not be inside any git work tree.

4. Install the two user units and start the timer:

   ```sh
   mkdir -p ~/.config/systemd/user
   cp deploy/systemd/lastlight-collect.service deploy/systemd/lastlight-collect.timer ~/.config/systemd/user/
   systemctl --user daemon-reload
   systemctl --user enable --now lastlight-collect.timer
   loginctl enable-linger "$USER"
   ```

   `enable-linger` lets your user units run when you are not logged in, which is what an unattended host needs.

5. Optional smoke test, on any afternoon, to see a real poll end to end. Print the sun's current elevation at the White Tank camera:

   ```sh
   cd ~/lastlight
   .venv/bin/python -c "import datetime as dt; from lastlight.cameras import load_registry; from lastlight.geometry import solar_position; p = load_registry()['WHTM1'].position; print(round(solar_position(dt.datetime.now(dt.UTC), p.lat, p.lon).elevation_deg, 1))"
   ```

   Then run the collector with a window that is open now and closes soon: start above the sun, stop about 1.5 degrees below it (for a printed 38.0, use 36.5):

   ```sh
   .venv/bin/lastlight collect --camera WHTM1 --start-elev 90 --stop-elev 36.5
   ```

   Expect one or two polls over about 5 to 10 minutes, then one summary line such as `WHTM1 2026-10-10: saved=1 not_modified=0 stale=0 errors=0`. The frame is under the archive.

## Check that it ran

- Is the timer scheduled? `systemctl --user list-timers` shows `lastlight-collect.timer` and the next run.
- Did last night's run happen? `journalctl --user -u lastlight-collect --since yesterday`. A finished evening prints one line per camera: frames saved, 304s, stale, errors. Problems are printed before it, one line each.
- Is it running now? `systemctl --user status lastlight-collect.service`. During the afternoon wait it shows as active and quiet; that is normal.
- Are frames arriving? `ls ~/.local/share/lastlight/frames/WHTM1/` then look inside the newest date folder. A good clear evening is dozens of frames per camera. `log.jsonl` says what each poll did.

What the summary means: `errors` high with everything else zero points at the network or the cameras' page (see the `detail` field in `log.jsonl`); `stale` high means the camera is serving an old picture; `not_modified` only is normal for the quiet hours when the picture does not change.

## Stop, restart, remove

- Stop tonight's run: `systemctl --user stop lastlight-collect.service`. The timer starts it again tomorrow at 15:00.
- Stop collecting until further notice: `systemctl --user disable --now lastlight-collect.timer`.
- Run one evening by hand: `systemctl --user start lastlight-collect.service`, then follow it with `journalctl --user -u lastlight-collect -f`.
- Update the code: `cd ~/lastlight && git pull && uv sync --frozen --no-dev`. If the unit files changed, copy them again and run `systemctl --user daemon-reload`.
- Remove everything: disable the timer, delete the two files in `~/.config/systemd/user/`, run `systemctl --user daemon-reload`, and delete the archive if you no longer want the frames.

## If something is wrong

- **Exit 2 and "inside a git work tree".** The archive path is under a repository. Pick another with `LASTLIGHT_ARCHIVE` or `--archive`. The unit does not retry this, by design.
- **Exit 2 and "interval must be at least 300 seconds".** Someone passed a shorter `--interval`. The cameras are polled at most once per 5 minutes, and the collector enforces it.
- **Nothing in the archive after an evening.** `journalctl --user -u lastlight-collect --since yesterday`. If the service never started, check `list-timers` and that linger is on (`loginctl show-user "$USER" | grep Linger`).
- **A camera operator asks us to stop.** Remove their camera from `src/lastlight/cameras/registry.toml`, delete that camera's folder from the archive, and tell the repo owner the same day (`docs/design.md` section 3: removal on request is honoured the same day).

## Privacy

Frames are third-party images and stay on this host: never copy them into the repository or a CI artifact. Only derived, elevation-referenced numbers and small sky-only thumbnails for cameras whose registry entry allows it (`publish_thumbnails = true` with a recorded `basis`) ever leave it, and later slices produce those.
