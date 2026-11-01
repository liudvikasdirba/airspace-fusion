"""Performance budgets (L9; slow). Best-of-N wall time with generous limits to tolerate slower machines.

Reference timings on the development machine with the NumPy tracker: single run (both modes) ~60 ms;
20-seed x 4-scenario compare ~5 s of work in one process; heavy clutter ~0.17 s; ~0.14 ms per scan.
Budgets leave about four times that, for slower CI runners.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest

from airspace_fusion.config import SCENARIOS, TrackerConfig, default_primary, default_secondary
from airspace_fusion.estimation import run_tracker
from airspace_fusion.evaluation import run_single
from airspace_fusion.schemas import parse_records
from airspace_fusion.simulator import simulate

pytestmark = pytest.mark.slow


def best_time(fn: Callable[[], object], repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def test_single_run_under_quarter_second():
    elapsed = best_time(lambda: run_single(SCENARIOS["reference"], TrackerConfig(), 0))
    assert elapsed < 0.25, elapsed


def test_full_comparison_budget():
    """20 seeds x 4 scenarios must stay under 20 s in one process; extrapolated from one seed of every scenario."""
    per_seed = best_time(lambda: [run_single(s, TrackerConfig(), 0) for s in SCENARIOS.values()], repeats=2)
    projected = per_seed * 20
    assert projected < 20.0, projected


def test_heavy_clutter_under_one_second():
    scenario = SCENARIOS["reference"].model_copy(update={
        "primary": default_primary(false_observations_per_scan=5.0),
        "secondary": default_secondary(false_observations_per_scan=5.0),
    })
    elapsed = best_time(lambda: run_single(scenario, TrackerConfig(), 0))
    assert elapsed < 1.0, elapsed


def test_tracker_time_per_scan():
    sim = simulate(SCENARIOS["reference"], 0)
    measurements, _ = parse_records(sim.records)
    result = run_tracker(measurements, sim.report_times, TrackerConfig())
    per_scan_ms = result.processing_time_s / len(sim.report_times) * 1000.0
    assert per_scan_ms < 0.6, per_scan_ms  # ~0.14 ms on the development machine with both sensors
