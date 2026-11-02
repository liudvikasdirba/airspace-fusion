# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) as described in
[docs/versioning.md](docs/versioning.md). Add entries under **Unreleased** as you change things;
`python ci/release.py X.Y.Z` turns them into a release.

## [Unreleased]

## [0.4.0] - 2026-11-02

M2 finished: scenario families that draw a different world for each seed, TOML scenario files, a `sweep`
command, results files that `reproduce` can re-run, the GOSPA metric and a rolling NIS consistency chart. Fusion
improves at every crossing angle from 5 to 90 deg, with a nearly flat gain.

### Added

- GOSPA metric (`gospa_m`): root mean square over all scans of single-object GOSPA with p = 2, α = 2 and a 150 m
  cut-off, so scans without a track count against a mode, unlike RMSE. Shown by `run`, `compare` and the
  dashboard, recorded in the golden file and checked against Stone Soup's `GOSPAMetric`. The verdict still uses
  RMSE.
- Dashboard "Filter consistency" chart: the rolling mean of each sensor's NIS per measurement dimension over its
  last 20 accepted updates, with the 95 % band of a consistent filter. It needs no ground truth and makes an
  unmodelled bearing bias visible during a single run. Display only; the tracker does not act on it.
  `estimation.nis_series` computes it.
- Scenario families: a base scenario plus ranges, drawing a different world for each seed (target start, heading,
  speed and turn; the bearing sensor placed by crossing angle and range ratio; optionally bearing bias and
  clutter). Five built-in families: `crossing_geometry`, `range_ratio`, `manoeuvre_intensity`, `clutter_density`
  and `bias_magnitude`. Families draw from their own random stream, so the built-in scenarios and published
  results are unchanged ([ADR 0008](docs/adr/0008-scenario-files-and-families.md)).
- Scenario files: TOML `[[scenario]]` and `[[family]]` entries loaded with `--scenario-file PATH` on `run`,
  `compare`, `learn` and the new `scenarios` command, which lists everything and prints any entry as TOML.
  `scenarios/example.toml` shows both kinds.
- `--scenario families` selects the five built-in families; `run`, `compare` and the dashboard accept families.
- `sweep` command: the paired comparison at each value of one family parameter, over the same seeds and the same
  other draws (common random numbers), as a table and optionally CSV (`--csv`).
- Results files: every `compare` and `sweep` saves a JSON record under `data/results/` with the scenarios or
  families, tracker settings, seeds, all metric summaries, a configuration hash and provenance (package version,
  git commit, Python version, command line). `--results-dir`, `AIRSPACE_FUSION_RESULTS_DIR` and `--no-results`
  control it. `reproduce PATH` re-runs a record and exits with status 1 if any metric differs.

### Changed

- Roadmap: M2 is done. Its crossing-angle exit criterion now reports what the sweep showed (fusion improves at every
  angle; the gain is nearly flat) instead of expecting the gain to rise, with the reason. The README has a "Fusion
  gain against crossing angle" section with a 100-seed sweep.
- `--scenario` names are checked after parsing, because scenario files add names: an unknown name now gives
  "unknown scenario" instead of argparse's "invalid choice". `all` still means the four built-in scenarios.
- New dependency on Python 3.10 only: `tomli>=1.1.0` (Python 3.11+ reads TOML with the standard library).
  `ci/minimum_constraints.py` keeps environment markers.

## [0.3.0] - 2026-11-02

M1 finished: about 40 % faster runs with unchanged results, a per-seed run-time benchmark, sturdier parallel
workers, and CI on Node.js 24 actions. The M1 speed targets are revised in the roadmap.

### Added

- Per-seed run-time benchmark (`tests/test_benchmark.py`, slow suite). It appends per-scenario median, 90th
  percentile and fastest seed times, with commit, versions and machine, to `data/benchmarks/run_times.jsonl`,
  and warns when a median is more than 1.5 times the last entry from the same machine. The nightly CI slow job
  uploads the file.

### Fixed

- `learn` now stops with the "worker process stopped unexpectedly" error when a worker dies. Before, the learning
  loop retried every step against the broken worker pool and reported every scenario as failed, without the
  advice to use fewer `--workers`.
- `--workers` above 61 no longer crashes on Windows, where Python allows at most 61 worker processes; the
  number is capped.
- Ctrl-C during a parallel `compare` or `learn` drops the queued seeds instead of waiting for them, and exits
  with status 130 and the message "interrupted".
- The default number of workers counts the CPUs the process may use (its CPU affinity, for example a
  container's cpuset), not every CPU in the machine.

### Changed

- One run (`run_single`, both modes) takes about 50 ms instead of about 80 ms on the development machine, and
  `compare --seeds 1000` about 99 s instead of 117 s. Covariance validation uses a cached closed-form 2x2 check;
  the tracker uses an unrolled 4x4 Cholesky check and slices instead of selection-matrix products; the simulator
  and metrics loop over plain floats. Results are unchanged (golden file and Stone Soup cross-check pass).
- Worker processes no longer import SciPy (chi-square gate quantiles are computed in closed form), which halves
  their start-up, and run BLAS single-threaded unless `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` or
  `MKL_NUM_THREADS` is set.
- CI uses `actions/checkout`, `actions/setup-python` and `actions/upload-artifact` v7, which run on
  Node.js 24; the v4/v5 versions ran on the deprecated Node.js 20.
- Parallel runs send seeds to workers in batches, so the scenario and tracker settings are not pickled once
  per seed.
- Performance budgets allow about six times the development machine's timings (was four), to avoid failures
  on busy CI runners.
- Roadmap: M1 is done. Its speed targets are revised to what pure Python and NumPy reach on the development
  machine (one run under 60 ms instead of under 17 ms; 1,000 seeds under 110 s instead of under 60 s), with the
  reasons and when to revisit them; the later target rows use the same figure.
- Documentation: the [roadmap](docs/roadmap.md) records progress (M0 done, most of M1 done in 0.2.0, with
  notes where results differ from the plan); the [release process](docs/releasing.md) describes the pull
  request, tag and GitHub release steps actually used; the README has current test times and the lint and type
  check commands.

## [0.2.0] - 2026-11-01

Fast evaluation and a learning loop: seeds run in parallel, the tracker's filter maths runs in NumPy with
Stone Soup as a cross-check (about 5x faster per run, same results), and the package is type-checked in CI.

### Added

- `learn` command: a closed learning loop in which research, experiment, QA and content agents tune the
  tracker settings per scenario on training seeds. A change is accepted only if it also wins on held-out
  seeds. Results go to a bounded JSON rule book (newest 100 lessons per scenario) and a JSONL run log under
  `data/learning/`. Evaluations are cached within a run. `learn` exits with status 1 if any scenario failed.
  An unreadable rule book stops the command with an error that names the file.
- `compare --rules PATH` evaluates the settings from a learned rule book; the file must exist.
- `compare` and `learn` evaluate seeds in parallel worker processes: `--workers N`, default the CPU count
  but at most 4 (each worker needs about 100 MB). Results are identical to a serial run; only
  `processing_time_s` differs. Comparisons of fewer than 4 seeds run serially, because starting workers
  would cost more than it saves. If a worker stops unexpectedly (usually from running out of memory), the
  command says so and exits with status 1. In code: `compare_modes(..., executor)` and `seed_executor`.
- `estimation_stonesoup.StoneSoupTracker`: the tracker with its filter steps done by Stone Soup, kept as a
  cross-check. Tests require it and the NumPy tracker to make the same decision about every measurement and to
  agree within 1e-9 ([ADR 0007](docs/adr/0007-numpy-tracker-stone-soup-cross-check.md)).
- `airspace-fusion --version`.
- Type checking: mypy runs over the package in CI (new `types` job, mypy pinned) and passes. Only the untyped
  third-party packages (SciPy, Stone Soup, Plotly, pandas) are exempt from import checks; settings are in
  `pyproject.toml`.
- CI tests Python 3.13 on Ubuntu and Windows as a required target (it was an experimental Ubuntu-only job).
- `Measurement.covariance()` and `Measurement.angular_std()` return the uncertainty in the form the tracker
  needs and raise `ValueError` if the measurement carries the other kind.
- Versioning: the package version is defined once, in `airspace_fusion.__version__`. `ci/release.py`
  checks that `CITATION.cff` and this changelog agree with it and cuts releases.
- Rule books record their JSON format version and the package version that wrote them. A rule book
  in an unknown format fails to load instead of being misread.
- Documentation set under `docs/`: index, architecture decision records, versioning policy and release
  process. Tests check that the docs' metadata, index and links stay consistent.
- [Roadmap to April 2027](docs/roadmap.md) (draft): the November 2026 audit findings and nine milestones
  from 0.2.0 to 1.0.0, with dates, deliverables and exit criteria.

### Changed

- The tracker's filter steps (predict, innovation, Joseph-form update) are written directly in NumPy instead of
  going through Stone Soup. Results are unchanged (the golden regression passes as before); one run takes about
  60 ms instead of 330 ms. `SingleTargetTracker` internals now use `TrackState` instead of Stone Soup's
  `GaussianState`, and `AnalyticBearingModel` moved to `estimation_stonesoup`. Performance budgets are tightened.
- `Estimate.position` and `Estimate.position_covariance` raise `ValueError` when there is no track instead of
  returning `None`; check `has_track` first, as all existing callers already did.

### Fixed

- The golden regression's simulator checksum rounds floats to 10 significant digits, so it no longer
  depends on last-bit differences between C maths libraries (it failed on Linux CI while passing on Windows).
  CI also reports failing tests as annotations, which can be read without signing in to GitHub.

## [0.1.0] - 2026-10-31

First release: a simulation-only demonstrator comparing a position-only baseline tracker with position +
bearing fusion against simulated ground truth.

### Added

- Pydantic measurement schema with type-specific uncertainty, UTC timestamps and unit, dimension,
  covariance and probability validation. Invalid records are returned with their errors.
- Deterministic seeded simulator: one object, two synthetic sensors, missed and false observations,
  optional unmodelled bias and deliberately corrupted records. Ground truth is kept separate from records.
- Stone Soup Kalman/EKF single-target tracker with chi-square gating, nearest-in-gate selection,
  two-point initiation and track drop.
- Evaluation: RMSE, coverage, 95 % containment, missed, rejected and invalid counts, timing, and a
  multi-seed paired comparison with a verdict.
- Four scenarios: `reference`, `noisy_primary`, `distant_bearing`, `biased_bearing`.
- Command line (`run`, `compare`) and a Streamlit/Plotly dashboard bound to localhost.
- Tests: property-based schema fuzzing, textbook Kalman/EKF oracle checks, Monte Carlo filter
  consistency (NEES/NIS), golden regression, settings sweep, statistical behaviour, headless
  dashboard tests, performance budgets and module dependency rules.
- CI workflow: Ubuntu and Windows on Python 3.10/3.12, minimum dependency versions, a nightly slow
  suite, ruff lint and coverage. `ci/minimum_constraints.py` and a `requirements.txt` lock.
- MIT license and `CITATION.cff`.

### Fixed

Found during development, before the release:

- Bearing-only scans no longer delay track start in fusion mode.
- Observations are gated per sensor and measurement type, so dimensions never mix.
- The paired verdict handles zero-variance differences (previously a NaN p-value).
- `wrap_angle` could return +π for angles just below −π.
- The covariance symmetry check is relative to the matrix scale.
- Bearing updates use an exact analytic Jacobian instead of a finite difference.
- Bearings alone can no longer keep a diverged track alive; the coast timeout counts from the last
  accepted position update.
- Sensor overrides are validated instead of copied unchecked.
- Dashboard: sensor markers visible in dark mode; the metrics table fits at about 800 px.
