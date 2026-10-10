# Why lastlight

On 2026-10-10 the agent was asked to choose a project with no constraints except a spend cap (at most $500 one-time and $50 a month, each item approved by the owner), occasional batched help with physical tasks, and a public repo. This is the record of how it chose.

## Method

1. **Ideation.** Eight independent agents, each with a different lens (backyard instruments, the radio spectrum, agents as the subject, unmeasured public signals, citizen science, hard craft from first principles, generative work with a scoreboard, wildcard), proposed 40 ideas. A skeptical synthesizer merged them into 26 clusters and re-scored them on six criteria: unique, alive (something new to watch every week), falsifiable (an external truth signal can prove it wrong), agent-buildable, uses this setup's edge, and depth at month 6.
2. **Verification.** Five finalists each got a web-researched dossier (prior art, physics, prices, data access, privacy, a $0 go/no-go gate). An independent red-team agent then tried to kill each one.
3. **Decision.** All five survived their red teams, with serious caveats. The agent chose by weighting falsifiable and alive most heavily.

## Finalists

| Finalist | Verdict | Deciding evidence |
| --- | --- | --- |
| **Last Light** (this project) | Chosen | No public, pre-registered, colorimetrically scored twilight forecast exists. The nearest neighbours are Albers et al.'s simulated-weather-imagery work (renders skies from analyses and compares them with uncalibrated cameras by correlation) and Lin et al. 2026 in *J. Appl. Meteor. Climatol.* (classifies twilight into three classes with machine learning). Commercial sunset forecasts publish no validation. Every evening brings a new picture, even when the score is flat. |
| Passive radar graded by ADS-B | Rejected | Open-source passive radars that overlay ADS-B truth already exist, and an open network sells the kit. Export-control questions are not yet settled for public radar code. The scoreboard converges within about two months. |
| Predicting downstream library breaks | Rejected | Requires automatically running brand-new third-party code; few events, so the board stays nearly all green. |
| Home GNSS atmospheric scorecards | Rejected | The underlying quantities are already published from geodetic stations; event-poor until the next monsoon. |
| Colorado River operations model | Rejected | Recent open trackers already cover it, and it resolves about one independent outcome per year. |

## What the red team said about this one

- **Event starvation.** Phoenix's dust and smoke season is summer; starting in October means months 1-6 are mostly quiet clear skies. Response: clouds are in scope from the start, and the first six months are framed as a baseline, not a verdict.
- **Asymmetric baselines.** Climatology built from the camera absorbs the camera's own biases. Response: score anomalies and freeze any debiasing before the evenings it is scored on.
- **Heat.** Summer enclosures can exceed the sensor's rating. Response: shaded mount, power only during twilight, temperature logging, re-run the repeatability gate in the first hot month.
- **Privacy.** Sunset times locate a camera. Response: elevation-referenced publishing and a CI gate.
