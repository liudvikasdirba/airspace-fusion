# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) as described in
[docs/versioning.md](docs/versioning.md). Add entries under **Unreleased** as you change things;
`python ci/release.py X.Y.Z` turns them into a release.

## [Unreleased]

### Added

- `learn` command: a closed learning loop in which research, experiment, QA and content agents tune the
  tracker settings per scenario on training seeds. A change is accepted only if it also wins on held-out
  seeds. Results go to a bounded JSON rule book (newest 100 lessons per scenario) and a JSONL run log under
  `data/learning/`. Evaluations are cached within a run. `learn` exits with status 1 if any scenario failed.
  An unreadable rule book stops the command with an error that names the file.
- `compare --rules PATH` evaluates the settings from a learned rule book; the file must exist.
- `airspace-fusion --version`.
- Versioning: the package version is defined once, in `airspace_fusion.__version__`. `ci/release.py`
  checks that `CITATION.cff` and this changelog agree with it and cuts releases.
- Rule books record their JSON format version and the package version that wrote them. A rule book
  in an unknown format fails to load instead of being misread.
- Documentation set under `docs/`: index, architecture decision records, versioning policy and release
  process. Tests check that the docs' metadata, index and links stay consistent.

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
