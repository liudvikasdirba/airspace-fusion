---
title: "0007: NumPy tracker by default, Stone Soup as the cross-check"
status: accepted
updated: 2026-11-01
---

# 0007: NumPy tracker by default, Stone Soup as the cross-check

## Context

A run took about 0.33 s, nearly all of it Stone Soup wrapper overhead: about 13,000 `__array_ufunc__` calls
per run for 4x4 matrices. Verdicts rested on 20 seeds and learning on 6 + 6 seeds because more was too slow
(roadmap finding 2). The filter itself is a few lines of linear algebra, already written out independently
in `tests/test_estimation_oracle.py`. Stone Soup is still valuable as a second, widely used implementation to
check against.

## Decision

`SingleTargetTracker` in `estimation.py` does predict, innovation and update directly in NumPy, with the same
equations as Stone Soup's `KalmanPredictor` and `ExtendedKalmanUpdater` (Joseph form, symmetrised, analytic
bearing Jacobian, innovation wrapped to [-pi, pi)). `run_tracker` and everything above it use it.

`StoneSoupTracker` in `estimation_stonesoup.py` subclasses it and replaces only those three filter steps with
Stone Soup. Gating, nearest-neighbour selection, initiation and track drop are shared code, so a difference
between the two can only come from the filter maths.

The choice is a class, not a `TrackerConfig` setting as the roadmap first proposed: tracker settings are saved
in rule books and tuned by the learning loop, and which implementation computes them is neither.

## Consequences

- One run (both modes) takes about 60 ms instead of 330 ms; the tracker's share per scan fell from about 1.2 ms
  to 0.14 ms. What remains is mostly schema validation (pydantic), the simulator and metric computation.
- `tests/test_tracker_equivalence.py` runs both trackers on the golden seeds of every scenario and on stress
  cases (heavy clutter, noisy sensors, unusual settings). They must make the same decision about every
  measurement and agree on estimates and NIS within 1e-9. The oracle tests run against both.
- The golden regression passed without being regenerated.
- Stone Soup stays a runtime dependency because the cross-check ships with the package. Worker processes no
  longer import it, since only `estimation_stonesoup` does.
- A change to the filter equations must be made in `estimation.py`; the equivalence test then shows whether
  Stone Soup agrees. A deliberate departure from Stone Soup needs a new ADR.
