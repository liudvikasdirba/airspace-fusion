# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) as described in
[docs/versioning.md](docs/versioning.md). Add entries under **Unreleased** as you change things;
`python ci/release.py X.Y.Z` turns them into a release.

## [Unreleased]

### Changed

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
