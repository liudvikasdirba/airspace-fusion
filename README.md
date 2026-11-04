# airspace-fusion

A local, **simulation-only** single-object tracking demonstrator. It answers one question with synthetic data:
does adding a supplementary direction-measuring (bearing-only) sensor to a position-measuring sensor make
tracking better, worse, or no different?

Each experiment runs the same simulated data twice: **baseline** (position sensor only) and **fusion** (position +
bearing sensor). The results are scored against the simulated ground truth. Fusion is not assumed to win, and in
one of the four built-in scenarios it makes tracking clearly worse.

This is an educational research prototype. It connects to no real sensors, feeds, maps or other systems, and
models no real hardware.

## Quick start

Requires Python 3.10+ (3.12+ recommended; tested on 3.10.7 and 3.12.10, and in CI on 3.10, 3.12 and 3.13). From the repository root:

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install -e ".[dev]"
```

On macOS/Linux use `.venv/bin/python` in place of `.venv/Scripts/python` throughout.

Open the dashboard at http://localhost:8501:

```bash
.venv/Scripts/python -m streamlit run src/airspace_fusion/app.py --server.address localhost
```

`--server.address localhost` keeps the dashboard reachable only from this machine. `.streamlit/config.toml` sets the
same option, but Streamlit reads it only when started from the repository root.

## Using the dashboard

**Sidebar controls**

| Control | Effect |
|---|---|
| Scenario or family | One of the four built-in worlds, or one of the five families, which draw a different world for each seed (see [Scenarios](#scenarios)). The description is shown underneath; for a family, the drawn world is described above the charts. |
| Random seed | Which random version of that world is generated. The same seed always gives the same result. |
| Noise multiplier | Scales both sensors' measurement noise (0.25x to 3x). |
| Missed-observation probability | Chance that each sensor misses the object on a scan (0 to 0.8). |
| Show | Both modes, baseline only, or fusion only. |
| Show bearing rays | Draws each bearing observation as a faint line from the bearing sensor. |
| **Run simulation** | Runs with the current settings. Changing controls has no effect until you click it; the **Showing:** line states which settings are on screen. |
| Seeds / **Run multi-seed comparison** | Runs the current settings over N consecutive seeds starting at the chosen seed, and gives a verdict. |

**Page, top to bottom**

1. **Trajectory map**:
   - the ground truth is the grey dashed line
   - position observations are grey dots, and false observations are crosses (that label comes from the truth and is for display only; the tracker never sees it)
   - bearing rays are the faint lines
   - the baseline estimate is blue and the fusion estimate orange, each with 95 % uncertainty ellipses every 10 s
   - the sensors are grey squares

   Fusion's ellipses should be narrower across the bearing sensor's line of sight.
2. **Metrics (this run)**: every metric for baseline, fusion and their difference (see [Metrics](#metrics)).
3. **Error over time**: the solid lines are the actual distance from the truth; the dotted lines are each filter's
   own 95 % radius. A solid line that stays above its dotted line means that filter is over-confident.
4. **Filter consistency**: the rolling mean of each sensor's normalised innovation squared (NIS) over its last 20
   accepted updates, divided by the measurement dimension, with the 95 % band a consistent filter stays in. It uses
   no ground truth: a line above the band means the filter trusts that sensor too much. It is a display only; the
   tracker does not act on it.
5. **Multi-seed comparison** (after you run it): mean ± standard deviation per metric, and the verdict.
6. **Limitations of this simulation** (expandable).

**A single seed is an anecdote.** Draw conclusions from the multi-seed verdict.

Suggested sequence:
1. Run `reference` and compare the two estimates through the turn.
2. Run the multi-seed comparison.
3. Raise the noise multiplier, then the missed-observation probability.
4. Switch to `biased_bearing`: RMSE rises, containment collapses, and the bearing sensor's NIS climbs far above its band.
5. Switch to `distant_bearing`: the extra sensor adds almost nothing.

## Command line

One seeded simulation, both modes:

```bash
.venv/Scripts/python -m airspace_fusion.cli run --scenario reference --seed 0
```

Multi-seed comparison over all scenarios, 200 seeds by default (about 20–30 s on a 4-core laptop; `--seeds N`
for another number). Use `--scenario NAME` for one scenario or family, `--scenario families` for the five
families, and `--first-seed N` to start from a different seed:

```bash
.venv/Scripts/python -m airspace_fusion.cli compare
```

List every scenario and family, including those from scenario files, or print one as TOML to start your own file:

```bash
.venv/Scripts/python -m airspace_fusion.cli scenarios
```

```bash
.venv/Scripts/python -m airspace_fusion.cli scenarios --toml crossing_geometry
```

Sweep one family parameter: the paired comparison at each value, over the same seeds (values as a list, or
`lo:hi:n` for n evenly spaced values; `--csv` also writes the table for plotting):

```bash
.venv/Scripts/python -m airspace_fusion.cli sweep --scenario crossing_geometry --parameter crossing_angle_deg --values 5:90:6 --seeds 50 --csv sweep.csv
```

Every `compare` and `sweep` saves a **results file** under `data/results/` (change it with `--results-dir` or the
`AIRSPACE_FUSION_RESULTS_DIR` environment variable; `--no-results` saves nothing). It records:
- the scenarios or families as data, the tracker settings and the seeds
- every metric summary
- the package version, git commit (and whether there were uncommitted changes), Python version and command line
- a hash of the configuration

`reproduce` re-runs a results file and exits with status 1 if any metric differs (wall-clock time excluded):

```bash
.venv/Scripts/python -m airspace_fusion.cli reproduce data/results/<file>.json
```

`compare` and `learn` evaluate seeds in parallel worker processes, by default as many as the CPU has cores but at
most 4. Results are identical to a serial run; only `processing_time_s` differs. Each worker needs about 100 MB:
on a machine with little free memory use `--workers 2` or `--workers 1` (serial), and on a large machine more
workers can help. Comparisons of fewer than 4 seeds always run serially.

## Learning loop

`learn` tunes the tracker settings per scenario and writes them to a **rule book** that later runs start from.
Learning runs use their own seeds ([ADR 0010](docs/adr/0010-learning-evaluation.md)), so the
[Results](#results-200-seeds-0199-compare) seeds 0–199 remain an independent test:

- **training** seeds 1000–1049, reused every round;
- **held-out** seeds, fresh in every round: 2000–2049 in round 1, 2050–2099 in round 2, and so on;
- **sealed test** seeds 5000–5199, used once after the last round to report the result.

`learn` warns when fewer than 50 training or held-out seeds are used (`--train-seeds`, `--holdout-seeds`);
`--test-seeds 0` skips the sealed test.

```bash
.venv/Scripts/python -m airspace_fusion.cli learn --rounds 3
```

```bash
.venv/Scripts/python -m airspace_fusion.cli compare --rules data/learning/rules.json
```

The rule book (`data/learning/rules.json`) and an append-only JSONL run log (`data/learning/log.jsonl`) live under
the gitignored `data/` folder. Delete the rule book to start learning from scratch. `compare --rules` refuses a
path that does not exist, and both commands stop with a clear error if the rule book cannot be read.

Each round runs the same plan for every scenario. The orchestrator routes each step to an agent, retries a failed
step once, logs every step, and skips a scenario that keeps failing without stopping the run. If any scenario
failed, `learn` lists it on stderr and exits with status 1. Within a run, results are cached, so settings already
scored on the same seeds are not evaluated again.

| Step | Agent | What it does |
|---|---|---|
| 1 | research | Scores the rule book's current settings on the training seeds and lists the gaps. |
| 2 | qa | If fusion made RMSE worse, checks this on the held-out seeds; if confirmed, fusion is recommended against for that scenario and tuning scores the baseline only. |
| 3 | research | Proposes up to 4 candidate settings based on the gaps: over-confident → more process noise, under-confident → less, lost tracks → longer coast and wider gate, no gaps → a small step either way. Settings already tried are skipped. |
| 4 | experiment | Scores every candidate on the training seeds and keeps the best one that beats the current settings. |
| 5 | qa | Accepts that candidate only if its held-out score is at least 2 % better, its per-seed held-out scores are lower in a one-sided paired t-test (p < 0.05), and coverage drops by no more than 0.01. |
| 6 | content | Writes the report. |

**Score** is the mean RMSE of the modes in use, multiplied by `1 + 5 × |containment − 0.95|`: a one-point
containment shortfall costs as much as 5 % RMSE, so a candidate cannot win by becoming over-confident.

**Gaps** are containment more than 0.02 from 0.95, coverage below 0.95, more than one track initiation per run on
average, and a harmful verdict ("fusion worsens" or "fusion over-confident").

Every tried change becomes a **lesson**, whether accepted or rejected. The learning update applies the accepted
lessons to the rule book, re-checked against fixed bounds (process noise 0.05–5, gate 0.99–0.9999, coast 4–20 s),
and keeps the newest 100 lessons per scenario as memory. A run stops early when a round accepts nothing.

Ground truth is used only to score settings, which is possible only in simulation. The tracker is unchanged and
still never sees truth. The agents are deterministic Python, not language models: the same seeds and rule book
always give the same result.

After the last round, the **sealed test** scores the rule book the run started from against the final one on
seeds no decision has seen. Each rule book is scored as it would be used, with its own fusion policy.

Result of `learn --rounds 3` from an empty rule book (about 1 minute with 4 workers). Round 1 accepted one setting
per scenario; round 2 accepted nothing, so the run stopped. Sealed test, seeds 5000–5199:

| Scenario | Learned settings | Score before → after | Paired p | Verdict (final settings) |
|---|---|---|---|---|
| reference | q = 0.625 | 18.82 → 17.63 | 2e-128 | fusion improves |
| noisy_primary | q = 1.125 | 50.10 → 42.28 | 7e-123 | fusion improves |
| distant_bearing | q = 0.625 | 20.18 → 18.81 | 6e-113 | no material change |
| biased_bearing | q = 0.625, fusion recommended against | 93.05 → 18.80 | 1e-153 | fusion over-confident |

The learned settings carry over to seeds the loop never saw. `noisy_primary` gains most: its filter was the most
over-confident, and more process noise fixes that. In `biased_bearing` most of the gain comes from turning fusion
off; the loop cannot fix the bias itself, which is a sensor error, not a setting, and the verdict for fusion with
the final settings stays "fusion over-confident".

## Scenarios

The simulated world is a flat 2D frame in metres with 1 s scans for 180 s. One object starts at (−1500, 800)
heading east at 15 m/s. It flies straight, makes a 90° right turn between t = 60 s and 100 s, then flies straight
again, with a small seeded random acceleration (0.05 m/s²).

| Sensor | Location | Measures | Noise (1σ) | Missed | False observations per scan |
|---|---|---|---|---|---|
| `S1-POS` (primary) | (0, 0) | position (x, y) | 25 m per axis | 10 % | 0.2, uniform over ±4 km |
| `S2-BRG` (supplementary) | (0, −2500) | bearing only | 0.5° | 10 % | 0.2, uniform direction |

About 1 % of all emitted records are deliberately corrupted (NaN values, a timestamp without a time zone, or wrong
units) so that schema rejection is exercised.

| Scenario | Difference from the above |
|---|---|
| `reference` | none |
| `noisy_primary` | position noise 75 m |
| `distant_bearing` | bearing sensor at (0, −30000) with 2° noise |
| `biased_bearing` | bearing sensor has an unmodelled +1.5° bias (the tracker is not told) |

All values live in [`config.py`](src/airspace_fusion/config.py), and `TrackerConfig` holds the tracker settings.
For a position sensor, `bias` is added to each axis.

### Families: a different world for each seed

The four scenarios above share one trajectory and one sensor layout. A **family** draws a new world for each seed
from ranges (see [ADR 0008](docs/adr/0008-scenario-files-and-families.md)):
- **Target:** starts 1.5–2.5 km from the primary sensor in a random direction, on a crossing path (±30°) at
  10–25 m/s, and turns at up to ±3°/s for 20–60 s from 40–100 s.
- **Bearing sensor:** placed by **crossing angle** (the angle at the target between the two sensors' lines of
  sight) and **range ratio** (its distance from the target relative to the primary sensor's). Those two numbers
  decide how much a bearing adds.
- **Everything else** (noise, misses, clutter, duration) comes from the `reference` scenario.

| Family | What varies beyond the target |
|---|---|
| `crossing_geometry` | crossing angle 5–90° |
| `range_ratio` | range ratio 0.5–10 (crossing 60–90°) |
| `manoeuvre_intensity` | turns up to ±6°/s for 20–80 s (crossing 60–90°) |
| `clutter_density` | 0–5 false observations per scan on both sensors (crossing 60–90°) |
| `bias_magnitude` | unmodelled bearing bias 0–3° (crossing 60–90°) |

The same family and seed always give the same world, and both modes share it, so comparisons stay paired. A
family's results pool many geometries, so their spread across seeds is wider than for a fixed scenario. Families
draw from their own random stream, so the four scenarios above and their results are unchanged.

### Scenario files

Define your own scenarios and families in TOML and load them with `--scenario-file PATH` (repeatable) on `run`,
`compare`, `learn` and `scenarios`. Each entry starts from a `base` scenario and overrides fields:
- **`[[scenario]]` tables** use the model's fields and units (metres, seconds, radians), with `[scenario.primary]`
  and `[scenario.secondary]` for sensor fields.
- **`[[family]]` tables** give ranges as `[lo, hi]` pairs or single fixed values, with the unit in the name
  (`_m`, `_s`, `_deg`, `_mps`).
- **Errors, not guesses:** unknown keys and names that are already defined are rejected, so a typo cannot quietly
  run the wrong scenario.

[`scenarios/example.toml`](scenarios/example.toml) has one of each:

```bash
.venv/Scripts/python -m airspace_fusion.cli compare --scenario-file scenarios/example.toml --scenario wide_crossing --seeds 20
```

### Sweeps

A sweep fixes one family parameter at each of several values. A family draws every parameter of a seed in a fixed
order, so at every point each seed keeps the same target path, turn and other draws, and only the swept parameter
changes (common random numbers). Differences between points therefore come from that parameter. For example, a
crossing-angle sweep gives identical baseline results at every point, because the baseline ignores the bearing
sensor.

See [Fusion gain against crossing angle](#fusion-gain-against-crossing-angle) for the first sweep's result.

## How it works

```
simulate -> sensor records -> schema validation -> tracker (baseline, fusion) -> metrics vs truth -> verdict
```

| Module | Responsibility |
|---|---|
| `schemas.py` | Pydantic `Measurement` schema: `POSITION_2D` (2x2 covariance, m²), `RANGE_BEARING` (2x2 covariance in m/rad), `BEARING_ONLY` (angular std, rad), `DETECTION_ONLY` (no value, no uncertainty). Validates UTC timestamps, units, dimensions, finiteness, symmetric positive-definite covariances (symmetry relative to matrix scale) and probability ranges. Bearings are normalised to [−π, π). `parse_records` returns invalid records with their errors. |
| `config.py` | All scenario, sensor and tracker parameters, and the four scenarios. |
| `scenarios.py` | Scenario families (a world drawn per seed from their own random stream) and TOML scenario files. |
| `results.py` | Results files: what each `compare` and `sweep` run used and produced, with provenance; `reproduce` re-runs one. |
| `simulator.py` | Deterministic truth and sensor records from a seed. Ground truth and truth-side labels (false and corrupted record ids) are kept apart from the emitted records. |
| `estimation.py` | Single-target tracker: Kalman/extended Kalman filter in plain NumPy (Joseph-form update; the bearing model uses an exact analytic Jacobian). Receives only validated measurements and the scan schedule, never ground truth. |
| `estimation_stonesoup.py` | The same tracker with the filter steps done by Stone Soup, kept as a cross-check: tests require both to give the same results ([ADR 0007](docs/adr/0007-numpy-tracker-stone-soup-cross-check.md)). |
| `evaluation.py` | Runs both modes on the same simulation, computes metrics against truth, aggregates seeds and gives a paired verdict. |
| `learning.py` | Learning loop: score, gaps, lessons, the bounded rule book (JSON) and the JSONL run log. |
| `agents.py` | Orchestrator and the research, experiment, QA and content agents that run the learning plan. |
| `cli.py`, `app.py` | Command line and Streamlit/Plotly dashboard. |

**Tracker behaviour**

- **Motion model:** nearly constant velocity, with process noise 0.5 m²/s³ per axis. The turn is model mismatch that the process noise must absorb.
- **Each scan:**
  - predict to the scan time
  - for each sensor and measurement type, gate the observations with a chi-square test at 99.7 %
  - apply the in-gate observation with the smallest normalised innovation
  - positions are applied before bearings
- **Bearing updates** constrain only the direction across the bearing sensor's line of sight. A bearing is rejected as degenerate geometry when the predicted object is within 3σ of the bearing sensor.
- **Track start:** two position observations on consecutive scans implying at most 60 m/s. Bearings never start a track.
- **Track drop:** after 8 s without an accepted **position** update. Bearings refine a track but cannot keep it alive, since they carry no range.
- **Unsupported types:** `RANGE_BEARING` and `DETECTION_ONLY` are valid in the schema but are explicitly rejected by the tracker, never reinterpreted.

## Metrics

| Metric | Meaning | Good |
|---|---|---|
| Position RMSE (m) | Root-mean-square distance from truth, over scans **with a track** | lower |
| GOSPA (m) | Root mean square over **all** scans of single-object GOSPA (p = 2, α = 2, cut-off c = 150 m): a tracked scan costs min(error, c), a scan without a track c/√2 ≈ 106 m | lower; equals RMSE when every scan is tracked within c |
| Coverage | Share of scans with a track | near 1 |
| Truth inside 95 % ellipse | Share of estimates whose 95 % ellipse contains the truth | **≈ 0.95**; lower means over-confident |
| Missed primary-sensor updates | Scans with a track but no accepted position observation (comparable across modes) | lower |
| Prediction-only scans | Scans with a track but no accepted observation from any sensor | lower |
| Rejected observations | Rejected by the tracker: outside the gate, not nearest in gate, unsupported type, degenerate geometry, duplicate id | mostly false observations |
| Invalid records | Records from this mode's sensors that failed schema validation | equals corrupted records |
| Track initiations | Tracks started | 1; more means the track was lost and restarted |
| Processing time (s) | Tracker wall time | well under 1 s |

Read RMSE together with coverage, or use GOSPA, which combines both. Trust the ellipses only when containment is near
0.95.

**Verdict** ([ADR 0009](docs/adr/0009-verdict-v2.md)). Per-seed RMSEs are paired, and the rules apply in order:
1. **"insufficient data"**: fewer than two seeds have an RMSE in both modes, for example when no track ever forms.
2. **"fusion over-confident"**: fusion's mean 95 % containment is below 0.85 and below the baseline's, whatever RMSE does. When the baseline is calibrated no better, low containment is not fusion's fault and this rule does not apply.
3. **"fusion improves"** or **"fusion worsens"**: the paired t-test gives p < 0.05 **and** the mean relative change is at least 5 %. If every seed changes by exactly the same amount, the change is treated as certain (p = 0).
4. An improvement counts only if fusion's containment is at most 0.03 below the baseline's; otherwise the verdict is **"fusion over-confident"** too.
5. **"no material change"**: any other case.

When one run compares several scenarios (`compare`) or sweep points (`sweep`), the p-values are Holm-corrected
across them first. The CLI and the dashboard say which rule decided each verdict.

## Results (200 seeds, 0–199, `compare`)

| Scenario | Baseline RMSE (m) | Fusion RMSE (m) | Change [95 % CI] | Fusion 95 % containment | Verdict |
|---|---|---|---|---|---|
| reference | 18.70 ± 1.9 | 16.41 ± 1.6 | −2.30 m [−2.45, −2.14], −12.3 % | 0.93 | improves |
| noisy_primary | 49.86 ± 24 | 37.57 ± 25 | −12.30 m [−12.92, −11.67], −24.7 % | 0.92 | improves |
| distant_bearing | 18.70 ± 1.9 | 18.69 ± 1.9 | −0.01 m [−0.02, +0.01], −0.0 % | 0.93 | no material change |
| biased_bearing | 18.70 ± 1.9 | 37.35 ± 4.4 | +18.65 m [+18.00, +19.29], +99.7 % | 0.28 | over-confident |

Baseline containment is 0.93 in every scenario except `noisy_primary` (0.90). All p-values are below 1e-70 except
`distant_bearing` (p = 0.27), before and after the Holm correction.

Coverage is identical between modes in every scenario (0.96–0.99). GOSPA (baseline → fusion) tells the same story:
21.14 → 19.15 m (`reference`), 51.19 → 40.35 m (`noisy_primary`), 21.14 → 21.13 m (`distant_bearing`) and
21.14 → 38.42 m (`biased_bearing`). It is a few metres above RMSE because the scans before a track starts count.
In `noisy_primary` the RMSE spread across seeds (± 24 m) comes from a few seeds whose track drifts far from the
truth; GOSPA caps each scan's error at 150 m, so its spread is ± 6 m.

In `biased_bearing`, fusion also misses more primary updates (24.7 vs 20.2 per run): the bias drags the estimate far enough that correct position observations fail the gate. A bias sweep (`tests/test_statistics.py`) shows the damage is not monotonic. Fusion error grows up to about 2° of bias, then falls at 3°, because the gate rejects most grossly biased bearings. A moderate bias does the most harm. With 50 seeds, the verdict is "fusion over-confident" from 0.5° of bias,
where RMSE has changed by only +2 % but containment has fallen to 0.85, up to 2°; at 3° containment recovers
to 0.89 and the verdict is "fusion worsens".

**Filter consistency.** With a motion model matched to the simulator (no turn, matched process noise), 40 Monte Carlo runs give:
- average NEES ≈ 4 (the state dimension)
- position NIS ≈ 2 and bearing NIS ≈ 1
- 95 % containment ≈ 0.95 in both modes

See `tests/test_consistency.py`. Negative controls confirm the checks fail for an over-confident filter, and for an unmodelled bearing bias, which shows up as bearing NIS above 1.5. The dashboard's rolling bearing NIS stays inside its 95 % band in at least 85 % of windows when the filter is consistent, and in at most half of them with a 1.5° bias.

### Fusion gain against crossing angle

The **crossing angle** is the angle at the target between the two sensors' lines of sight:

```
                 bearing sensor (S2)
                /
               /
   target  ●  ) crossing angle
                                                position sensor (S1)
```

A bearing tells the tracker in which direction the target lies from S2. It says nothing about the distance along
that line, but it pins down the position *across* it: at 2 km, a 0.5° bearing error is about 17 m across the line
of sight. The position sensor measures both coordinates with 25 m error.

Sweep of the `crossing_geometry` family, 100 seeds (0–99) per angle. Every angle uses the same seeds and the same
target paths, so only the bearing sensor's position changes; that is why the baseline, which ignores the bearing
sensor, is identical at every angle.

| Crossing angle | Baseline RMSE | Fusion RMSE | Change [95 % CI] | GOSPA baseline → fusion | Fusion 95 % containment |
|---|---|---|---|---|---|
| 5° | 18.67 m | 15.94 m | −14.6 % [−3.02, −2.44] m | 21.07 → 18.70 m | 0.93 |
| 15° | 18.67 m | 16.03 m | −14.1 % [−2.92, −2.36] m | 21.07 → 18.78 m | 0.93 |
| 30° | 18.67 m | 16.14 m | −13.6 % [−2.80, −2.27] m | 21.07 → 18.86 m | 0.93 |
| 45° | 18.67 m | 16.29 m | −12.8 % [−2.61, −2.16] m | 21.07 → 19.00 m | 0.93 |
| 60° | 18.67 m | 16.36 m | −12.4 % [−2.52, −2.09] m | 21.07 → 19.06 m | 0.93 |
| 75° | 18.67 m | 16.44 m | −12.0 % [−2.44, −2.02] m | 21.07 → 19.13 m | 0.93 |
| 90° | 18.67 m | 16.44 m | −12.0 % [−2.44, −2.02] m | 21.07 → 19.13 m | 0.93 |

Fusion improves at every angle (Holm-adjusted p < 1e-30 everywhere), and the gain is **nearly flat**: it falls slightly from
14.6 % at 5° to 12.0 % from about 75° on.

The roadmap had expected the gain to *rise* with the angle. That intuition holds when the main sensor's error
depends on direction, as for a radar that is precise in range but poor across it: a bearing from the side then
covers its weak direction. Here the position sensor's error is the same in every direction, so the bearing's
information is worth about the same whichever way it points. The small fall probably comes from distance rather
than angle: the bearing sensor starts as far from the target as the position sensor at every angle, but on the
target's crossing path it stays closer on average at small angles, where it sits near the position sensor.
Testing the crossing-angle effect properly needs a direction-dependent primary sensor (range-bearing), which the
tracker does not support yet.

Reproduce it (writes a CSV and a results file that `reproduce` can check):

```bash
.venv/Scripts/python -m airspace_fusion.cli sweep --scenario crossing_geometry --parameter crossing_angle_deg --values 5,15,30,45,60,75,90 --seeds 100 --csv crossing.csv
```

## Limitations

- Single synthetic object, two stationary synthetic sensors, generic 2D frame. No real hardware, terrain or propagation.
- Nearly-constant-velocity model; the turn is absorbed by process noise (baseline 95 % containment ≈ 0.92, i.e. slightly over-confident).
- Bearing updates are linearised (EKF). Their value depends strongly on geometry: a distant bearing sensor adds almost nothing.
- Errors are assumed zero-mean, white and independent between sensors and over time. An unmodelled bias or correlated error makes fusion over-confident and **worse** than the baseline. The filter cannot correct this, although it shows up as bearing NIS well above 1.
- Association is chi-square gating plus nearest-in-gate selection for one object. False observations inside the gate can still be accepted (about 1 % of accepted bearings at 5 false observations per scan). Two-point initiation can occasionally start on two aligned false observations before the real track. There is no multi-object association or ambiguity resolution.
- RMSE is evaluated only where a track exists; GOSPA covers every scan, but its cut-off (150 m) is a choice.
- Process noise and gate settings are shared by both modes. The [Results](#results-200-seeds-0199-compare) use the untuned defaults; `learn` can tune them per scenario (see [Learning loop](#learning-loop)).

## Development and testing

Fast suite (about 20 s). It includes:
- property-based schema fuzzing (`hypothesis`)
- checks of both trackers (NumPy and Stone Soup) against textbook Kalman/EKF formulas, and of the two against
  each other
- end-to-end edge cases and module dependency rules
- CLI tests
- the golden regression

```bash
.venv/Scripts/python -m pytest
```

Slow suite (about 1 min): Monte Carlo filter-consistency and statistical-behaviour checks, the edge-settings sweep, headless dashboard tests, performance budgets, and a cross-process reproducibility check.

```bash
.venv/Scripts/python -m pytest -m slow
```

Lint and type checks, as CI runs them (the versions are pinned in `.github/workflows/tests.yml`):

```bash
.venv/Scripts/python -m pip install ruff==0.17.3 mypy==2.4.1
```

```bash
.venv/Scripts/python -m ruff check .
```

```bash
.venv/Scripts/python -m mypy
```

`tests/test_regression.py` compares per-seed metrics and a simulator-output checksum against `tests/golden/regression.json`. If a change to results is intended, regenerate it and review the JSON diff with the code change:

```bash
.venv/Scripts/python -m pytest tests/test_regression.py --update-golden
```

### Compatibility and release checks

| Check | Status |
|---|---|
| Windows, Python 3.10.7 and 3.12.10 (fresh venv, README steps) | all tests pass |
| Lowest declared dependency versions (Python 3.10) | all tests pass |
| Install from the `requirements.txt` lock (fresh venv) | tests pass |
| CI (`.github/workflows/tests.yml`): lint, type check, coverage, Ubuntu and Windows on Python 3.10/3.12/3.13, minimum versions, nightly slow suite | runs on every push and pull request; all jobs pass |

`requirements.txt` is an optional pinned lock of a known-good environment:

```bash
.venv/Scripts/python -m pip install -r requirements.txt
```

To test against the declared minimum versions, use a fresh virtual environment. On Python 3.10, first install `tomli`; the script needs it to read `pyproject.toml`. `constraints-min.txt` is gitignored.

```bash
python -m pip install tomli
```

```bash
python ci/minimum_constraints.py > constraints-min.txt
```

```bash
python -m pip install -e ".[dev]" -c constraints-min.txt
```

Manual dashboard checklist, per release (everything else is covered by `tests/test_dashboard.py`):
- the page renders in light and dark themes, with a readable legend and visible sensors and ellipses
- at about 800 px wide with the sidebar open, the metrics table fits and the page has no horizontal scroll
- tooltips show units
- the server listens on 127.0.0.1 only: `netstat -an | findstr 8501` on Windows, `ss -ltn | grep 8501` on Linux

## Documentation, changelog and versions

- [CHANGELOG.md](CHANGELOG.md): what changed in each version, with upcoming changes under Unreleased.
- [docs/](docs/README.md): an index of the project documents, including the
  [roadmap to April 2027](docs/roadmap.md) with what is done so far, the [versioning policy](docs/versioning.md),
  the [release process](docs/releasing.md) and the architecture decision records in [docs/adr/](docs/adr/).
- The version is defined in `src/airspace_fusion/__init__.py`. Print it with `airspace-fusion --version`, and
  check that `CITATION.cff` and the changelog agree with it:

```bash
.venv/Scripts/python ci/release.py --check
```

## License

MIT. See [LICENSE](LICENSE).
