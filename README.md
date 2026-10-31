# airspace-fusion

A local, **simulation-only** single-object tracking demonstrator. It asks one question with synthetic data:
does adding a supplementary direction-measuring (bearing-only) sensor to a position-measuring sensor improve
tracking, make it worse, or make no measurable difference?

This is an educational research prototype. It connects to no real sensors, feeds, maps or other systems.

## Install

Requires Python 3.10+ (3.12+ recommended). From the repository root:

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install -e ".[dev]"
```

(On macOS/Linux use `.venv/bin/python`.) `requirements.txt` is an optional pinned lock of a known-good environment.

## Run

Tests:

```bash
.venv/Scripts/python -m pytest
```

One seeded simulation, both modes, from the command line:

```bash
.venv/Scripts/python -m airspace_fusion.cli run --scenario reference --seed 0
```

Multi-seed baseline vs fusion comparison over all scenarios (about 45 s for 20 seeds):

```bash
.venv/Scripts/python -m airspace_fusion.cli compare --seeds 20
```

Dashboard (served on http://localhost:8501, bound to localhost by `.streamlit/config.toml`):

```bash
.venv/Scripts/python -m streamlit run src/airspace_fusion/app.py
```

## What it does

| Module | Responsibility |
|---|---|
| `schemas.py` | Pydantic `Measurement` schema: `POSITION_2D` (2x2 covariance, m²), `RANGE_BEARING` (2x2 covariance in m/rad), `BEARING_ONLY` (angular std, rad), `DETECTION_ONLY` (no value, no uncertainty). Validates UTC timestamps, units, dimensions, finiteness, symmetric positive-definite covariances, probability ranges. Bearings are normalised to [-π, π). `parse_records` returns invalid records with error messages. |
| `config.py` | All scenario, sensor and tracker parameters, and four named scenarios. |
| `simulator.py` | Deterministic truth (straight, 90° turn, straight, plus small seeded random acceleration), two stationary sensors with noise, missed observations, false observations, optional unmodelled bias, and a small fraction of deliberately corrupted records. Truth and truth-side labels are kept apart from the emitted records. |
| `estimation.py` | Stone Soup Kalman predictor and extended Kalman updater; chi-square gating, nearest-in-gate selection per sensor, two-point initiation, coasting and track drop. Receives only validated measurements and the scan schedule. |
| `evaluation.py` | Runs both modes on the *same* simulation, computes metrics against truth, aggregates multiple seeds and gives a paired verdict. |
| `cli.py`, `app.py` | Command line and Streamlit/Plotly dashboard. |

**Modes.** *Baseline*: measurements from the position sensor `S1-POS` only. *Fusion*: `S1-POS` plus the bearing
sensor `S2-BRG`. Both use one simulation per seed, so truth and primary measurements are identical.

**Metrics.** Position RMSE (only at scans with a track), coverage (fraction of scans with a track), 95 %
containment (fraction of estimates whose 95 % ellipse contains truth; ≈0.95 means the stated uncertainty is
honest), missed updates (scans with a track but no accepted primary-sensor measurement - comparable across
modes), prediction-only scans (no sensor updated the track), rejected observations (gate, unsupported type,
degenerate geometry, duplicate id), invalid records from the mode's own sensors (schema), track initiations,
processing time.

**Verdict.** Per-seed RMSEs are paired; fusion "improves" or "worsens" only if a paired t-test gives p < 0.05
**and** the mean relative change is at least 5 %. Otherwise "no material change".

## Results (20 seeds, 0–19, `compare --seeds 20`)

| Scenario | Baseline RMSE (m) | Fusion RMSE (m) | Change [95 % CI] | Fusion 95 % containment | Verdict |
|---|---|---|---|---|---|
| reference | 18.83 ± 1.7 | 16.70 ± 1.6 | −2.13 m [−2.51, −1.76], −11.3 % | 0.92 | improves |
| noisy_primary | 49.67 ± 5.3 | 37.84 ± 5.7 | −11.83 m [−13.54, −10.12], −23.8 % | 0.91 | improves |
| distant_bearing | 18.83 ± 1.7 | 18.83 ± 1.7 | −0.00 m [−0.02, +0.02], −0.0 % | 0.92 | no material change |
| biased_bearing | 18.83 ± 1.7 | 37.50 ± 5.0 | +18.66 m [+16.39, +20.94], +99.1 % | 0.28 | worsens |

Coverage is identical between modes (0.96–0.99). In `biased_bearing` fusion also misses more primary updates
(25.0 vs 19.5 per run): the bias drags the estimate far enough that correct position measurements fail the gate.
Run the command to reproduce.

## Limitations

- Single synthetic object, two stationary synthetic sensors, generic 2D frame. No real hardware, terrain or propagation.
- Nearly-constant-velocity model; the turn is model mismatch absorbed by process noise (baseline 95 % containment ≈0.92, i.e. slightly over-confident).
- Bearing updates are linearised (EKF). Information depends strongly on geometry: a distant bearing sensor adds almost nothing; a target close to the bearing sensor relative to its uncertainty is rejected as degenerate rather than processed.
- Errors are assumed zero-mean, white and independent between sensors and over time. An unmodelled bias or correlated error makes fusion over-confident and **worse** than the baseline (`biased_bearing`). The filter cannot detect this itself.
- Association is chi-square gating plus nearest-in-gate selection for one object. False observations inside the gate can still be accepted; there is no multi-object association, track management beyond initiation/drop, or ambiguity resolution.
- `RANGE_BEARING` and `DETECTION_ONLY` are validated by the schema but explicitly rejected by the estimator (`rejected_unsupported_type`).
- RMSE is evaluated only where a track exists; read it together with coverage.
- Process noise and gate settings were chosen once and are shared by both modes; they were not tuned per scenario.
