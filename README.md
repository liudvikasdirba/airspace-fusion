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

Requires Python 3.10+ (3.12+ recommended; tested on 3.10.7 and 3.12.10). From the repository root:

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
| Scenario | One of the four built-in worlds (see [Scenarios](#scenarios)); its description is shown underneath. |
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
4. **Multi-seed comparison** (after you run it): mean ± standard deviation per metric, and the verdict.
5. **Limitations of this simulation** (expandable).

**A single seed is an anecdote.** Draw conclusions from the multi-seed verdict.

Suggested sequence:
1. Run `reference` and compare the two estimates through the turn.
2. Run the multi-seed comparison.
3. Raise the noise multiplier, then the missed-observation probability.
4. Switch to `biased_bearing`: RMSE rises and containment collapses.
5. Switch to `distant_bearing`: the extra sensor adds almost nothing.

## Command line

One seeded simulation, both modes:

```bash
.venv/Scripts/python -m airspace_fusion.cli run --scenario reference --seed 0
```

Multi-seed comparison over all scenarios (about 45 s for 20 seeds). Use `--scenario NAME` for one scenario and
`--first-seed N` to start from a different seed:

```bash
.venv/Scripts/python -m airspace_fusion.cli compare --seeds 20
```

## Learning loop

`learn` tunes the tracker settings per scenario and writes them to a **rule book** that later runs start from.
Learning runs use their own seeds (training 1000–1005 and held-out 2000–2005 by default), so the
[Results](#results-20-seeds-019-compare---seeds-20) seeds 0–19 remain an independent test.

```bash
.venv/Scripts/python -m airspace_fusion.cli learn --rounds 3
```

```bash
.venv/Scripts/python -m airspace_fusion.cli compare --seeds 20 --rules data/learning/rules.json
```

The rule book (`data/learning/rules.json`) and an append-only JSONL run log (`data/learning/log.jsonl`) live under
the gitignored `data/` folder. Delete the rule book to start learning from scratch.

Each round runs the same plan for every scenario. The orchestrator routes each step to an agent, retries a failed
step once, logs every step, and skips a scenario that keeps failing without stopping the run:

| Step | Agent | What it does |
|---|---|---|
| 1 | research | Scores the rule book's current settings on the training seeds and lists the gaps. |
| 2 | qa | If fusion made RMSE worse, checks this on the held-out seeds; if confirmed, fusion is recommended against for that scenario and tuning scores the baseline only. |
| 3 | research | Proposes up to 4 candidate settings based on the gaps: over-confident → more process noise, under-confident → less, lost tracks → longer coast and wider gate, no gaps → a small step either way. Settings already tried are skipped. |
| 4 | experiment | Scores every candidate on the training seeds and keeps the best one that beats the current settings. |
| 5 | qa | Accepts that candidate only if its held-out score is at least 2 % better and coverage drops by no more than 0.01. |
| 6 | content | Writes the report. |

**Score** is the mean RMSE of the modes in use, multiplied by `1 + 5 × |containment − 0.95|`: a one-point
containment shortfall costs as much as 5 % RMSE, so a candidate cannot win by becoming over-confident.

**Gaps** are containment more than 0.02 from 0.95, coverage below 0.95, more than one track initiation per run on
average, and a "fusion worsens" verdict.

Every tried change becomes a **lesson**, whether accepted or rejected. The learning update applies the accepted
lessons to the rule book, re-checked against fixed bounds (process noise 0.05–5, gate 0.99–0.9999, coast 4–20 s),
and keeps all lessons as memory. A run stops early when a round accepts nothing.

Ground truth is used only to score settings, which is possible only in simulation. The tracker is unchanged and
still never sees truth. The agents are deterministic Python, not language models: the same seeds and rule book
always give the same result.

Result of `learn --rounds 3` from an empty rule book (about 90 s; converges after round 2), checked on seeds 0–19:

| Scenario | Learned settings | Baseline RMSE (m) | Fusion RMSE (m) | Baseline 95 % containment |
|---|---|---|---|---|
| reference | q = 0.625 | 18.83 → 18.62 | 16.70 → 16.56 | 0.92 → 0.94 |
| noisy_primary | q = 0.75 | 49.67 → 48.38 | 37.83 → 37.25 | 0.88 → 0.91 |
| distant_bearing | q = 0.625 | 18.83 → 18.62 | 18.83 → 18.62 | 0.92 → 0.94 |
| biased_bearing | q = 0.625, fusion recommended against | 18.83 → 18.62 | (37.06 → 36.83) | 0.92 → 0.94 |

The gains are small but appear on seeds the loop never saw. The verdicts do not change. In `biased_bearing`, the
loop learns that fusion should not be used, but it cannot fix the bias: the bias is a sensor error, not a setting.

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

All values live in [`config.py`](src/airspace_fusion/config.py). Edit `SCENARIOS` to add scenarios, or
`TrackerConfig` to change tracker settings. For a position sensor, `bias` is added to each axis.

## How it works

```
simulate -> sensor records -> schema validation -> tracker (baseline, fusion) -> metrics vs truth -> verdict
```

| Module | Responsibility |
|---|---|
| `schemas.py` | Pydantic `Measurement` schema: `POSITION_2D` (2x2 covariance, m²), `RANGE_BEARING` (2x2 covariance in m/rad), `BEARING_ONLY` (angular std, rad), `DETECTION_ONLY` (no value, no uncertainty). Validates UTC timestamps, units, dimensions, finiteness, symmetric positive-definite covariances (symmetry relative to matrix scale) and probability ranges. Bearings are normalised to [−π, π). `parse_records` returns invalid records with their errors. |
| `config.py` | All scenario, sensor and tracker parameters, and the four scenarios. |
| `simulator.py` | Deterministic truth and sensor records from a seed. Ground truth and truth-side labels (false and corrupted record ids) are kept apart from the emitted records. |
| `estimation.py` | Single-target tracker built on Stone Soup (Kalman predictor, extended Kalman updater; the bearing model uses an exact analytic Jacobian). Receives only validated measurements and the scan schedule, never ground truth. |
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
| Coverage | Share of scans with a track | near 1 |
| Truth inside 95 % ellipse | Share of estimates whose 95 % ellipse contains the truth | **≈ 0.95**; lower means over-confident |
| Missed primary-sensor updates | Scans with a track but no accepted position observation (comparable across modes) | lower |
| Prediction-only scans | Scans with a track but no accepted observation from any sensor | lower |
| Rejected observations | Rejected by the tracker: outside the gate, not nearest in gate, unsupported type, degenerate geometry, duplicate id | mostly false observations |
| Invalid records | Records from this mode's sensors that failed schema validation | equals corrupted records |
| Track initiations | Tracks started | 1; more means the track was lost and restarted |
| Processing time (s) | Tracker wall time | well under 1 s |

Read RMSE together with coverage, and trust the ellipses only when containment is near 0.95.

**Verdict.** Per-seed RMSEs are paired.
- **"fusion improves"** or **"fusion worsens"**: the paired t-test gives p < 0.05 **and** the mean relative change is at least 5 %. If every seed changes by exactly the same amount, the change is treated as certain (p = 0).
- **"no material change"**: any other case.
- **"insufficient data"**: fewer than two seeds have an RMSE in both modes, for example when no track ever forms.

## Results (20 seeds, 0–19, `compare --seeds 20`)

| Scenario | Baseline RMSE (m) | Fusion RMSE (m) | Change [95 % CI] | Fusion 95 % containment | Verdict |
|---|---|---|---|---|---|
| reference | 18.83 ± 1.7 | 16.70 ± 1.6 | −2.13 m [−2.51, −1.76], −11.3 % | 0.92 | improves |
| noisy_primary | 49.67 ± 5.3 | 37.83 ± 5.7 | −11.83 m [−13.54, −10.12], −23.8 % | 0.91 | improves |
| distant_bearing | 18.83 ± 1.7 | 18.83 ± 1.7 | −0.00 m [−0.02, +0.02], −0.0 % | 0.92 | no material change |
| biased_bearing | 18.83 ± 1.7 | 37.06 ± 4.0 | +18.23 m [+16.33, +20.13], +96.8 % | 0.28 | worsens |

Coverage is identical between modes in every scenario (0.96–0.99).

In `biased_bearing`, fusion also misses more primary updates (24.5 vs 19.45 per run): the bias drags the estimate far enough that correct position observations fail the gate. A bias sweep (`tests/test_statistics.py`) shows the damage is not monotonic. Fusion error grows up to about 2° of bias, then falls at 3°, because the gate rejects most grossly biased bearings. A moderate bias does the most harm.

**Filter consistency.** With a motion model matched to the simulator (no turn, matched process noise), 40 Monte Carlo runs give:
- average NEES ≈ 4 (the state dimension)
- position NIS ≈ 2 and bearing NIS ≈ 1
- 95 % containment ≈ 0.95 in both modes

See `tests/test_consistency.py`. Negative controls confirm the checks fail for an over-confident filter, and for an unmodelled bearing bias, which shows up as bearing NIS above 1.5.

## Limitations

- Single synthetic object, two stationary synthetic sensors, generic 2D frame. No real hardware, terrain or propagation.
- Nearly-constant-velocity model; the turn is absorbed by process noise (baseline 95 % containment ≈ 0.92, i.e. slightly over-confident).
- Bearing updates are linearised (EKF). Their value depends strongly on geometry: a distant bearing sensor adds almost nothing.
- Errors are assumed zero-mean, white and independent between sensors and over time. An unmodelled bias or correlated error makes fusion over-confident and **worse** than the baseline. The filter cannot correct this, although it shows up as bearing NIS well above 1.
- Association is chi-square gating plus nearest-in-gate selection for one object. False observations inside the gate can still be accepted (about 1 % of accepted bearings at 5 false observations per scan). Two-point initiation can occasionally start on two aligned false observations before the real track. There is no multi-object association or ambiguity resolution.
- RMSE is evaluated only where a track exists.
- Process noise and gate settings are shared by both modes. The [Results](#results-20-seeds-019-compare---seeds-20) use the untuned defaults; `learn` can tune them per scenario (see [Learning loop](#learning-loop)).

## Development and testing

Fast suite (about 25 s). It includes:
- property-based schema fuzzing (`hypothesis`)
- checks of the tracker against textbook Kalman/EKF formulas written in plain NumPy
- end-to-end edge cases and module dependency rules
- CLI tests
- the golden regression

```bash
.venv/Scripts/python -m pytest
```

Slow suite (about 3 min): Monte Carlo filter-consistency and statistical-behaviour checks, the edge-settings sweep, headless dashboard tests, performance budgets, and a cross-process reproducibility check.

```bash
.venv/Scripts/python -m pytest -m slow
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
| CI (`.github/workflows/tests.yml`): Ubuntu and Windows, Python 3.10/3.12, minimum versions, nightly slow suite | **not yet run** (needs a GitHub remote) |

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
  [versioning policy](docs/versioning.md), the [release process](docs/releasing.md) and the architecture
  decision records in [docs/adr/](docs/adr/).
- The version is defined in `src/airspace_fusion/__init__.py`. Print it with `airspace-fusion --version`, and
  check that `CITATION.cff` and the changelog agree with it:

```bash
.venv/Scripts/python ci/release.py --check
```

## License

MIT. See [LICENSE](LICENSE).
