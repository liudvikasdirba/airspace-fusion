"""Performance budgets (L9; slow). Best-of-N wall time with generous limits to tolerate slower machines.

Reference timings on the development machine: single run (both modes) ~0.4 s;
20-seed x 4-scenario compare ~30 s of tracking (plus process start-up); ~1.2 ms per scan.
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


def test_single_run_under_one_second():
    elapsed = best_time(lambda: run_single(SCENARIOS["reference"], TrackerConfig(), 0))
    assert elapsed < 1.0, elapsed


def test_full_comparison_budget():
    """20 seeds x 4 scenarios must stay under 60 s; extrapolated from one seed of every scenario."""
    per_seed = best_time(lambda: [run_single(s, TrackerConfig(), 0) for s in SCENARIOS.values()], repeats=2)
    projected = per_seed * 20
    assert projected < 60.0, projected


def test_heavy_clutter_under_three_seconds():
    scenario = SCENARIOS["reference"].model_copy(update={
        "primary": default_primary(false_observations_per_scan=5.0),
        "secondary": default_secondary(false_observations_per_scan=5.0),
    })
    elapsed = best_time(lambda: run_single(scenario, TrackerConfig(), 0))
    assert elapsed < 3.0, elapsed


def test_tracker_time_per_scan():
    sim = simulate(SCENARIOS["reference"], 0)
    measurements, _ = parse_records(sim.records)
    result = run_tracker(measurements, sim.report_times, TrackerConfig())
    per_scan_ms = result.processing_time_s / len(sim.report_times) * 1000.0
    assert per_scan_ms < 5.0, per_scan_ms  # ~1.2 ms on the development machine with both sensors
