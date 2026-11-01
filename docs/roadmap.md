---
title: Roadmap to April 2027
status: draft
updated: 2026-11-01
---

# Roadmap to April 2027

This document records the November 2026 audit of the project and the plan that follows from it: nine
milestones from November 2026 to the 1.0.0 release at the end of April 2027. It is a `draft`: a plan, not a
commitment. Review it at every release and update the dates and the `updated` field when the plan changes.

## Progress

Finished deliverables and exit criteria are ticked below, with a note where the result differs from the plan.

| Milestone | Status (1 November 2026) |
|---|---|
| M0 Ship what exists | **Done.** 0.2.0 released and tagged on 1 November, ahead of the 2 – 8 November window. |
| M1 Fast and parallel | **Mostly done early**, shipped in 0.2.0. Open: the per-seed benchmark test, and the two speed targets (one run takes about 60 ms, not under 17 ms; 1,000 seeds take 117 s, not under 60 s). |
| M2 – M8 | Not started. |

## Where the project stands

| Measure | At the audit (1 Nov 2026, morning) | After 0.2.0 (1 Nov 2026) |
|---|---|---|
| Source / tests | about 2,200 lines in `src/`, about 2,600 lines in `tests/` | about 2,470 and 2,800 lines |
| Fast suite | 249 tests, all passing, about 30 s on Windows / Python 3.12 | 323 tests, about 21 s; slow suite 65 tests, about 60 s |
| Lint and types | ruff clean; no type checking | ruff and mypy clean, both in CI |
| Cost of one run (`run_single`, both modes) | about 0.33 s | about 60 ms |
| `compare --seeds 20` (all scenarios) | about 45 s | about 6 s (4 workers) |
| Scenarios | 4, all sharing one trajectory and one sensor geometry | unchanged |
| Package version | 0.1.0, with the learning loop and docs under Unreleased | 0.2.0 |
| `main` branch | only "Initial project structure"; all work is on `chore/repo-hygiene` | all work merged (pull requests 1 – 4) |
| Git tags | none | `v0.1.0`, `v0.2.0` (with a GitHub release) |

### Strengths to keep

- Ground truth is isolated from the tracker, and module dependency tests enforce it
  ([ADR 0003](adr/0003-ground-truth-isolation.md)).
- Filter consistency is tested (NEES/NIS), with negative controls that prove the checks can fail.
- The tracker is checked against textbook Kalman/EKF formulas, and a portable golden regression guards results.
- Negative results are reported honestly: a biased sensor makes fusion worse, and the README says so.
- Decisions are recorded as ADRs; versions and release notes are checked automatically.

### Audit findings

Ordered by importance. Each one is addressed by the milestone named in the fourth column.

| # | Finding | Effect | Milestone | Status |
|---|---|---|---|---|
| 1 | No work is on `main`, there are no tags, and the README says CI has "not yet run". | The project has not actually shipped. | M0 | ✅ Resolved in 0.2.0 |
| 2 | One run costs about 0.33 s, about 1 ms per sensor update, almost all of it Stone Soup wrapper overhead (about 13,000 `__array_ufunc__` calls per run). Seeds run one after another. | Verdicts rest on 20 seeds and learning on 6+6 seeds and 4 candidates. | M1 | Partly: about 5× faster per run and seeds run in parallel (0.2.0) |
| 3 | Every scenario shares one start point, heading, turn and sensor layout. Seeds vary only noise and a 0.05 m/s² random acceleration. | "Fusion improves RMSE by 11 %" holds for one geometry, not in general. | M2 | Open |
| 4 | The verdict uses RMSE only. In `biased_bearing`, fusion containment drops to 0.28 but the verdict ignores it. RMSE counts only scans with a track. No correction for comparing several scenarios. | The verdict can miss the worst failure, an over-confident filter. | M3 | Open |
| 5 | The learning loop reuses the same 6 held-out seeds for every round and candidate. 6 seeds cannot reliably resolve the 2 % acceptance margin. | Some risk of overfitting to the held-out seeds. | M3 | Open |
| 6 | The constant-velocity model absorbs the turn as process noise; baseline containment is 0.92. | The baseline filter is slightly over-confident by design. | M4 | Open |
| 7 | Fusion cannot detect or recover from a degraded sensor at runtime, although bearing NIS above 1.5 already reveals the bias. | The project can show that fusion fails but not prevent it. | M5, M6 | Open |
| 8 | The rule book is keyed by scenario name, which a real system would not know. Nothing tests generalisation to unseen scenarios. | Learned settings are tied to labelled worlds. | M7 | Open |
| 9 | Results are compared only with each other, never with a theoretical best. | "Better than baseline" is the strongest claim possible. | M8 | Open |
| 10 | No static type checking; scenarios can only be defined in Python; results are not stored with provenance; Python 3.13 is experimental in CI. | Harder to contribute to and to reproduce. | M0, M2, M8 | Partly: mypy and Python 3.13 done (0.2.0); scenario files and provenance open |

## Ambition

Today the project asks: *does adding a bearing sensor help?* By April 2027 it should answer a larger question:

> **Can a fusion system tell, without ground truth, when a sensor is hurting it, correct for it, and
> how close does it come to the best that is theoretically possible?**

It stays a simulation-only research and teaching tool ([ADR 0002](adr/0002-simulation-only-scope.md)).
Simulation is the advantage: every claim can be checked against exact truth, over thousands of seeds and
many geometries.

By 1.0.0 the project will be a **reproducible benchmark for trustworthy sensor fusion**: a versioned scenario
suite, a fast reference tracker, a sensor-health monitor, bias correction, label-free learning, a theoretical
lower bound to score against, and a plugin interface so that other trackers can be scored the same way.

## Timeline

Weeks start on Mondays. Each milestone ends in a minor release (see [versioning](versioning.md)).
Estimates assume one maintainer working part-time; items marked *stretch* move to the next milestone or to
[After April 2027](#after-april-2027) if they do not fit.

| Milestone | Release | Dates | Theme |
|---|---|---|---|
| M0 | 0.2.0 | 2 – 8 Nov 2026 | Ship what exists |
| M1 | 0.3.0 | 9 – 29 Nov 2026 | Fast and parallel |
| M2 | 0.4.0 | 30 Nov – 20 Dec 2026 | Varied worlds |
| — | — | 21 Dec 2026 – 3 Jan 2027 | Holiday buffer: catch-up, documentation only |
| M3 | 0.5.0 | 4 – 17 Jan 2027 | Honest scoring |
| M4 | 0.6.0 | 18 – 31 Jan 2027 | Manoeuvres (IMM) |
| M5 | 0.7.0 | 1 – 21 Feb 2027 | Sensor health monitor |
| M6 | 0.8.0 | 22 Feb – 7 Mar 2027 | Bias estimation |
| M7 | 0.9.0 | 8 – 28 Mar 2027 | Learning without labels |
| M8 | 1.0.0 | 29 Mar – 30 Apr 2027 | Benchmark and 1.0 (release candidate by 11 Apr) |

```
Nov 2026        Dec 2026        Jan 2027        Feb 2027        Mar 2027        Apr 2027
|M0|---M1---|----M2----|buffer|-M3-|-M4-|----M5----|-M6-|----M7----|--------M8--------|
 ship  fast   varied worlds    score IMM   health    bias  label-free   benchmark 1.0
```

Dependencies: M1 (speed) and M2 (scenario variety) come first, because every later milestone needs many seeds
over many geometries. M3 must land before M5–M7, so that their results are judged by the new verdict.
M5 comes before M6 (bias estimation is triggered by the monitor) and M7 (the monitor's statistics are the
learning features). M8 needs all of them.

## M0 — Ship what exists (0.2.0, 2 – 8 Nov 2026) — done 1 Nov 2026

**Goal:** the current work is on `main`, released, tagged and tested by CI.

Deliverables:
- [x] Open a pull request from `chore/repo-hygiene` to `main`; fix whatever CI finds on Ubuntu and Windows.
  *Merged as pull request 1. CI found one failure: the golden checksum depended on last-bit differences between
  C maths libraries; it now rounds to 10 significant digits.*
- [x] Tag `v0.1.0` on the commit that matches the 0.1.0 changelog entry (`e7ed74d`, "Add lint and coverage CI
  jobs and CITATION.cff"; confirm before tagging).
- [x] Release 0.2.0 from the Unreleased entries (learning loop, `--version`, docs) with `ci/release.py`, and tag it.
  *Also published as a GitHub release. 0.2.0 includes most of M1 as well.*
- [x] Replace the README's "not yet run" CI line with the real status.
- [x] Add mypy (strict for `src/`) to the lint job; make Python 3.13 a required CI target.
  *mypy runs in default mode, not strict, in its own `types` job (it needs the dependencies installed, pinned by
  `requirements.txt`). Strict mode stays the 0.4.0 target in [Targets](#targets). Python 3.13 runs on Ubuntu
  and Windows.*
- [x] Add this roadmap to the docs index.

Exit criteria:
- [x] All CI jobs green on `main`.
- [x] Tags `v0.1.0` and `v0.2.0` exist.
- [x] mypy passes with no ignores except for untyped third-party imports (SciPy, Stone Soup, Plotly, pandas).

## M1 — Fast and parallel (0.3.0, 9 – 29 Nov 2026)

**Goal:** make large experiments cheap, so that later verdicts can rest on hundreds or thousands of seeds.

Status: most deliverables shipped early, in 0.2.0. What is left decides when 0.3.0 is cut.

Deliverables:
- [x] A plain NumPy tracker kernel with the same behaviour as the current one (same gating, selection, initiation,
  coast and Joseph-form update). The textbook oracle in `tests/test_estimation_oracle.py` is the starting point.
- [x] The Stone Soup tracker stays as a second implementation, selectable with a setting, and a test runs both on
  the golden seeds and requires identical results.
  *Selected by class (`StoneSoupTracker`), not by a `TrackerConfig` setting, so that the choice never enters rule
  books or the learning loop. `tests/test_tracker_equivalence.py` also covers stress cases; the largest
  difference seen is 4.6e-13.*
- [x] Seeds are evaluated in parallel across processes (`--workers N`, default: number of cores), with output
  identical to a serial run.
  *The default is the number of cores but at most 4: each worker needs about 100 MB, and 8 workers ran an 8 GB
  machine out of memory. A crashed worker gives a clear error instead of a traceback.*
- [x] The learning loop's evaluation cache and agents use the parallel path.
- [ ] A benchmark test that records run time per seed, so that speed regressions show up.
  *Not yet. The performance budgets in `tests/test_performance.py` were tightened 4 – 8×, but they record nothing.*
- [x] ADR 0007: NumPy kernel is the default, Stone Soup is the cross-check.

Exit criteria:
- [x] The golden regression passes unchanged on the new kernel (results agree to 1e-9 m).
- [ ] One run at least 20× faster than today (below 17 ms).
  *About 60 ms, 5.5× faster. The tracker itself is about 9× faster (0.14 ms per scan); most of what remains is
  schema validation (pydantic), the simulator and metric computation.*
- [ ] `compare --seeds 1000` over all four scenarios in under 60 s on the development machine.
  *117 s with 4 workers.*
- [x] Serial and parallel runs give byte-identical CLI output.
  *Identical except the `processing_time_s` row, which is wall-clock time; tests compare everything else.*

Risk: subtle differences from Stone Soup (angle wrapping, symmetrisation). Mitigation: compare the two
implementations step by step on the golden seeds and on the hypothesis-generated edge cases.

## M2 — Varied worlds (0.4.0, 30 Nov – 20 Dec 2026)

**Goal:** results describe how fusion behaves across geometries, not in one fixed world.

Deliverables:
- Scenario files in TOML, read with `--scenario-file PATH` and validated by the existing pydantic models.
  The four built-in scenarios stay and are reproduced exactly (golden file unchanged).
- A scenario generator that draws, per seed, the start point, heading, speed, turn time and rate, and the
  bearing sensor's position, from ranges given in a scenario *family*.
- At least five built-in families, for example: `crossing_geometry` (angle between the sensors' lines of sight
  from 5° to 90°), `range_ratio`, `manoeuvre_intensity`, `clutter_density`, `bias_magnitude`.
- A `sweep` command that varies one parameter and reports the paired verdict at each value, as a table and as
  CSV for plotting.
- A results file for every `compare` and `sweep` run, recording package version, git commit, seeds, a hash of
  the configuration, and all metrics. Saved under `data/results/`.
- A README section showing fusion gain against crossing angle, with the geometry explained.
- ADR 0008: scenario file format and families.

Exit criteria: at least five families; a `sweep` over crossing angle shows the expected trend (gain rising with
crossing angle) with confidence intervals; any results file can be re-run to reproduce its numbers exactly.

## M3 — Honest scoring (0.5.0, 4 – 17 Jan 2027)

**Goal:** a verdict that cannot call an over-confident filter an improvement, and learning that cannot overfit
its held-out seeds.

Deliverables:
- GOSPA as an additional metric (single object; penalises missing tracks as well as error), so that coverage
  and error are judged together.
- Verdict v2:
  - "fusion improves" also requires fusion's 95 % containment to be no more than 0.03 below the baseline's
  - a new verdict, "fusion over-confident", when containment falls below 0.85 whatever the RMSE does
  - Holm correction when several scenarios or sweep points are compared at once
  - default seed count raised from 20 to 200
- Learning loop:
  - fresh held-out seeds for every round
  - a sealed test set, used once, to report the final result of a `learn` run
  - at least 50 seeds per training and held-out evaluation
  - acceptance by a paired test on held-out seeds as well as the 2 % margin
- README results regenerated with the changes and the numbers that moved listed in the changelog.
- ADR 0009: verdict v2 and learning evaluation.

Exit criteria: `biased_bearing` gets "fusion over-confident" or "fusion worsens", never "no material change";
on the four built-in scenarios the other verdicts are unchanged; learning results reproduce on the sealed test set.

## M4 — Manoeuvres (0.6.0, 18 – 31 Jan 2027)

**Goal:** remove the motion-model mismatch so that containment problems come from sensors, not from the turn.

Deliverables:
- An interacting multiple model (IMM) tracker with constant-velocity and coordinated-turn models, in the NumPy
  kernel, selectable per run. The constant-velocity tracker stays the default until the IMM passes its tests.
- NEES/NIS consistency tests extended to the IMM, including a negative control.
- The `manoeuvre_intensity` family used to show where the IMM helps and where it does not.
- The learning loop can choose between the two motion models.

Exit criteria: baseline 95 % containment in `reference` within 0.95 ± 0.02 (today 0.92); RMSE during the turn
lower than with the constant-velocity model; no loss on straight segments beyond 5 %.

Risk: IMM tuning takes longer than two weeks. Mitigation: a single coordinated-turn rate is enough for 0.6.0;
a learned turn rate can follow later.

## M5 — Sensor health monitor (0.7.0, 1 – 21 Feb 2027)

**Goal:** fusion detects, without ground truth, that a sensor is hurting it, and stops trusting it.

Deliverables:
- A monitor per sensor that tests normalised innovations over a sliding window (windowed chi-square test and
  a CUSUM), using only data the tracker sees.
- Three states per sensor: trusted, suspect (measurements down-weighted), excluded (ignored until its
  innovations are consistent again).
- A new metric set: detection delay, false alarms per run, time spent excluded.
- ROC curves (detection delay against false-alarm rate) from the `bias_magnitude` family.
- A module dependency rule so that the monitor cannot import the simulator or evaluation.
- ADR 0010: sensor exclusion and how it interacts with [ADR 0004](adr/0004-position-updates-keep-tracks-alive.md).

Exit criteria:
- 1.5° bearing bias detected with a median delay of 30 s or less.
- False alarms on unbiased scenarios in at most 1 % of runs.
- In `biased_bearing`, fusion with the monitor is no more than 5 % worse than the baseline (today 97 % worse).
- In `reference`, at least 90 % of today's fusion gain is kept.

## M6 — Bias estimation (0.8.0, 22 Feb – 7 Mar 2027)

**Goal:** go beyond excluding a biased sensor: estimate the bias and keep using the sensor.

Deliverables:
- The filter state augmented with a bias per bearing sensor, switched on by the monitor when a sensor becomes
  suspect, with a check that the bias is observable from the geometry.
- A Schmidt-Kalman option that accounts for bias uncertainty without estimating it.
- Covariance intersection as a fusion option for sensors whose errors may be correlated in unknown ways.
- Consistency tests for each option, including a negative control.

Exit criteria: in `biased_bearing`, fusion RMSE at or below the baseline's with containment of at least 0.90;
the bias estimate within ±0.3° of the true bias by the end of the run, on at least 90 % of seeds.

*Stretch:* covariance intersection may move to M8 if bias estimation takes the full two weeks.

## M7 — Learning without labels (0.9.0, 8 – 28 Mar 2027)

**Goal:** the learning loop chooses settings from what the tracker observes, not from the scenario's name,
and it is tested on worlds it has never seen.

Deliverables:
- Context features computed from tracker-visible data only: innovation statistics per sensor, monitor states,
  update rates, clutter rate, manoeuvre indicators.
- A rule book keyed by context (for example bins or a small decision tree over the features) instead of by
  scenario name. Rule book format 2, with a clear error when a format 1 file is loaded.
- Leave-one-family-out evaluation: train on all families but one, test on the one left out.
- The agents keep their deterministic design and their logs, lessons and bounds.
- ADR 0011: context-keyed learning. It supersedes [ADR 0005](adr/0005-deterministic-learning-agents.md) in part;
  determinism and held-out acceptance carry over.

Exit criteria: on held-out families the learned policy beats the defaults on at least two thirds of families
and is never worse by more than 2 %; same seeds and rule book still give the same result.

## M8 — Benchmark and 1.0 (1.0.0, 29 Mar – 30 Apr 2027)

**Goal:** a stable, citable benchmark that other people can run their trackers on.

Deliverables (29 Mar – 11 Apr, release candidate):
- Posterior Cramér-Rao lower bound for the single-object scenarios, including missed detections, and every
  tracker reported as "% above the bound".
- Benchmark suite `suite-2027.1`: a frozen, versioned set of scenario families, seeds and metrics, with a
  published results file for every built-in tracker.
- A tracker plugin interface (Python entry points): a plugin receives validated measurements and the scan
  schedule, exactly like the built-in tracker, and is scored by the same evaluation.
- Compatibility freeze: the versioning policy updated for 1.0, and the compatibility surface listed in
  [versioning](versioning.md) reviewed and confirmed.
- `1.0.0rc1` tagged by 11 April.

Deliverables (12 – 30 Apr, release):
- A documentation site built from `docs/` (for example MkDocs), with a tutorial: "Why fusion can make things
  worse".
- A JOSS paper (`paper.md` and `paper.bib`) submitted after the release.
- 1.0.0 released and tagged; a retrospective added to this roadmap.

Exit criteria: every built-in scenario has a PCRLB and a reported gap; a sample plugin in `tests/` runs through
the full benchmark; the release candidate has had two weeks without changes to the compatibility surface.

*Stretch:* sensor placement optimisation (choose where to put the bearing sensor by Fisher information, then
validate the choice by Monte Carlo).

## Targets

| Target | At the audit | 0.2.0 (Nov) | 0.4.0 (Dec) | 0.7.0 (Feb) | 1.0.0 (Apr) |
|---|---|---|---|---|---|
| Seeds behind each verdict | 20 | 20 (200 take 26 s) | 1,000 possible | 200 by default | 200 by default |
| Time for 1,000 seeds × 4 scenarios | about 37 min | 117 s | under 60 s | under 60 s | under 60 s |
| Scenario families | 0 (4 fixed scenarios) | 0 | 5 | 6+ | suite-2027.1 frozen |
| Baseline containment, `reference` | 0.92 | 0.92 | 0.92 | 0.95 ± 0.02 | 0.95 ± 0.02 |
| Fusion vs baseline RMSE, `biased_bearing` | +97 % | +97 % | +97 % | ≤ +5 % | ≤ 0 % |
| Bias detection delay / false alarms | none | none | none | ≤ 30 s / ≤ 1 % | ≤ 30 s / ≤ 1 % |
| Learned policy on unseen families | not tested | not tested | not tested | not tested | better on ≥ 2/3, never > 2 % worse |
| Comparison with a theoretical bound | none | none | none | none | PCRLB for every scenario |
| Static typing | none | mypy (default mode) | mypy strict | mypy strict | mypy strict |

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| One maintainer, part-time: milestones slip | high | The holiday buffer; stretch items move first; M1 and M2 are protected because everything depends on them. |
| The NumPy kernel drifts from Stone Soup | medium | Both kept; identical-result test on golden and fuzzed inputs. |
| Published numbers change often and confuse users | medium | Every change to README results gets a changelog entry naming the numbers that moved; golden diffs reviewed with the code. |
| Statistical claims overreach | medium | Verdict v2, Holm correction, sealed test sets, CIs on every published number. |
| Scope creep into multi-object tracking before 1.0 | medium | Multi-object work is listed after April 2027 and needs a new milestone in this document first. |
| Pressure to connect real data | low | Out of scope by [ADR 0002](adr/0002-simulation-only-scope.md); changing that needs a superseding ADR. |

## Working practice

- One branch and pull request per milestone (or per deliverable for large milestones), merged to `main` with CI green.
- Every release follows [the release process](releasing.md): changelog entries, `ci/release.py`, tag.
- Every decision listed above as an ADR is written **before** its code is merged.
- At every release, this roadmap is reviewed: finished items marked, dates moved where needed, `updated` changed.
- Every new module gets a dependency rule in `tests/test_integration.py`
  ([ADR 0003](adr/0003-ground-truth-isolation.md)).

## After April 2027

Not planned in detail; candidates for the next roadmap:

- Several objects: data association with JPDA or GM-PHD, crossing and closely spaced objects.
- Any number of sensors, moving sensors, sensors with different scan rates.
- Sensor placement optimisation, if it did not fit in M8.
- The dashboard running in the browser with no install (Pyodide), and teaching notebooks.
- A public results table for the benchmark suite, with contributed trackers.
