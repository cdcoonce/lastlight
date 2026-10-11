# First observed evening (2026-10-10)

A one-off collector run during the build session polled ADEQ's PhoenixVis White Tank (`WHTM1`) and Estrella (`ESMO1`) cameras every 5 minutes from sun elevation +14 to -7.4 degrees. Each camera produced 8 new frames, 5 of them inside the scored window (+5.1, +2.0, -1.1, -4.2 and -7.4 degrees). Frames are private; this entry records only what they showed.

**What the sky did:** an overcast stratiform deck all evening. The twilight was muted, with only a faint warm band low on the sunset side between +2 and -1 degrees. On the vividness scale this is a dull evening, and a forecast should say so.

**What the camera taught us:**

- Auto-exposure keeps deep-twilight frames bright, so brightness carries no signal. This confirms the contrast-only design for tier A.
- At -7.4 degrees, artificial city light turns the low sky orange at the northwest edge of the frame. Tier A scoring now defaults to civil twilight (+6 to -6 degrees), and the observation slice (#17) gets per-camera light-dome sectors.
- The White Tank ridgeline is crisp in every frame, so camera geometry (#16) can be fitted by skyline matching. The sun disk never showed through the cloud, so tonight gives no sun check.
- Frame cadence was exactly 15 minutes, not the 5 the site claims.

**Build state at the end of the evening:** scaffold, solar geometry, renderer v0 and the privacy scanner are on `main`. The seal ledger, HRRR ingest, camera collector and camera geometry are building.
