"""Evaluation of baseline and fusion tracking against ground truth.

This is the only module (besides plotting) that combines tracker output with truth.
Every number is computed from an actual simulation run.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from itertools import repeat

import numpy as np
from scipy import stats

from airspace_fusion.config import ScenarioConfig, TrackerConfig
from airspace_fusion.estimation import TrackerResult, run_tracker
from airspace_fusion.schemas import Measurement, RejectedRecord, parse_records
from airspace_fusion.simulator import GroundTruth, SimulationOutput, simulate

# 95 % probability contour of a 2D Gaussian (chi-square, 2 dof).
CHI2_2DOF_95 = float(stats.chi2.ppf(0.95, 2))
# Relative RMSE change below which a difference is not considered material.
MATERIAL_RELATIVE_CHANGE = 0.05
SIGNIFICANCE_LEVEL = 0.05
# Below this many seeds per comparison, starting worker processes costs more than it saves.
MIN_PARALLEL_SEEDS = 4


class Mode(str, Enum):
    BASELINE = "baseline"
    FUSION = "fusion"


def sensors_for_mode(scenario: ScenarioConfig, mode: Mode) -> tuple[str, ...]:
    if mode is Mode.BASELINE:
        return (scenario.primary.sensor_id,)
    return (scenario.primary.sensor_id, scenario.secondary.sensor_id)


@dataclass(frozen=True)
class RunMetrics:
    position_rmse_m: float  # NaN if no estimates exist
    coverage: float  # fraction of scan times with a track estimate
    containment_95: float  # fraction of estimates whose 95 % ellipse contains the truth
    missed_updates: int  # scans with a track but no accepted primary-sensor update (comparable across modes)
    prediction_only_scans: int  # scans with a track but no accepted update from any sensor
    rejected_observations: int  # rejected by the tracker (gate, unsupported, geometry, duplicate)
    invalid_observations: int  # records from this mode's sensors rejected by schema validation
    track_initiations: int
    processing_time_s: float


def evaluate_run(
    result: TrackerResult, truth: GroundTruth, invalid_count: int, primary_sensor_id: str
) -> RunMetrics:
    truth_index = {t: i for i, t in enumerate(truth.timestamps)}
    sq_errors: list[float] = []
    contained = 0
    for est in result.estimates:
        if est.timestamp not in truth_index:
            raise ValueError(f"estimate at {est.timestamp} has no ground-truth sample")
        if not est.has_track:
            continue
        error = est.position - truth.positions_m[truth_index[est.timestamp]]
        sq_errors.append(float(error @ error))
        if float(error @ np.linalg.solve(est.position_covariance, error)) <= CHI2_2DOF_95:
            contained += 1
    n_est = len(sq_errors)
    n_total = len(result.estimates)
    return RunMetrics(
        position_rmse_m=math.sqrt(sum(sq_errors) / n_est) if n_est else math.nan,
        coverage=n_est / n_total if n_total else math.nan,
        containment_95=contained / n_est if n_est else math.nan,
        missed_updates=result.missed_updates_for(primary_sensor_id),
        prediction_only_scans=result.prediction_only_scans,
        rejected_observations=result.rejected_count,
        invalid_observations=invalid_count,
        track_initiations=result.track_initiations,
        processing_time_s=result.processing_time_s,
    )


@dataclass(frozen=True)
class ModeRun:
    result: TrackerResult
    metrics: RunMetrics


@dataclass(frozen=True)
class SingleRun:
    simulation: SimulationOutput
    measurements: list[Measurement]
    invalid_records: list[RejectedRecord]
    modes: dict[Mode, ModeRun]


def run_single(scenario: ScenarioConfig, tracker: TrackerConfig, seed: int) -> SingleRun:
    """Simulate once and run both modes on exactly the same observations and truth."""
    sim = simulate(scenario, seed)
    measurements, invalid = parse_records(sim.records)
    modes: dict[Mode, ModeRun] = {}
    for mode in Mode:
        sensors = sensors_for_mode(scenario, mode)
        result = run_tracker(measurements, sim.report_times, tracker, sensor_ids=sensors)
        # Records with an unreadable sensor_id cannot be attributed, so they count against every mode.
        n_invalid = sum(1 for r in invalid if r.sensor_id is None or r.sensor_id in sensors)
        metrics = evaluate_run(result, sim.truth, n_invalid, scenario.primary.sensor_id)
        modes[mode] = ModeRun(result, metrics)
    return SingleRun(sim, measurements, invalid, modes)


class Verdict(str, Enum):
    IMPROVES = "fusion improves"
    WORSENS = "fusion worsens"
    NO_MATERIAL_CHANGE = "no material change"
    INSUFFICIENT_DATA = "insufficient data"


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    std: float
    n: int


def summarise(values: Sequence[float]) -> MetricSummary:
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return MetricSummary(math.nan, math.nan, 0)
    return MetricSummary(float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0, int(arr.size))


@dataclass(frozen=True)
class ComparisonSummary:
    scenario: str
    seeds: tuple[int, ...]
    metrics: dict[Mode, dict[str, MetricSummary]]
    rmse_delta_mean_m: float  # fusion - baseline, paired by seed
    rmse_delta_ci95_m: tuple[float, float]
    rmse_relative_change: float
    rmse_p_value: float
    verdict: Verdict


def paired_verdict(
    baseline: Sequence[float], fusion: Sequence[float]
) -> tuple[float, tuple[float, float], float, float, Verdict]:
    """Paired comparison of per-seed RMSE. Lower is better. Seeds where either value is NaN are dropped."""
    pairs = [(b, f) for b, f in zip(baseline, fusion) if not (math.isnan(b) or math.isnan(f))]
    if len(pairs) < 2:
        return math.nan, (math.nan, math.nan), math.nan, math.nan, Verdict.INSUFFICIENT_DATA
    b = np.array([p[0] for p in pairs])
    f = np.array([p[1] for p in pairs])
    d = f - b
    mean_d = float(d.mean())
    rel = mean_d / float(b.mean())
    if np.allclose(d, 0.0):
        return mean_d, (mean_d, mean_d), rel, 1.0, Verdict.NO_MATERIAL_CHANGE
    if np.allclose(d, d[0]):
        # Identical non-zero change on every seed: no variance, so the t-test is undefined
        # (it would return NaN), but the change is certain rather than insignificant.
        half, p_value = 0.0, 0.0
    else:
        sem = float(d.std(ddof=1)) / math.sqrt(d.size)
        half = float(stats.t.ppf(0.975, d.size - 1)) * sem
        p_value = float(stats.ttest_rel(f, b).pvalue)
    if p_value < SIGNIFICANCE_LEVEL and abs(rel) >= MATERIAL_RELATIVE_CHANGE:
        verdict = Verdict.IMPROVES if mean_d < 0 else Verdict.WORSENS
    else:
        verdict = Verdict.NO_MATERIAL_CHANGE
    return mean_d, (mean_d - half, mean_d + half), rel, p_value, verdict


METRIC_NAMES = (
    "position_rmse_m",
    "coverage",
    "containment_95",
    "missed_updates",
    "prediction_only_scans",
    "rejected_observations",
    "invalid_observations",
    "track_initiations",
    "processing_time_s",
)


def seed_metrics(scenario: ScenarioConfig, tracker: TrackerConfig, seed: int) -> dict[Mode, RunMetrics]:
    """Both modes' metrics for one seed. Module-level, so worker processes can run it."""
    run = run_single(scenario, tracker, seed)
    return {mode: run.modes[mode].metrics for mode in Mode}


@contextmanager
def seed_executor(workers: int, seeds_per_comparison: int) -> Iterator[Executor | None]:
    """A process pool for ``compare_modes``, or None when running serially is cheaper.

    Create it once per command and pass it to every comparison: starting the worker processes takes
    about a second, because each one imports NumPy, SciPy and Stone Soup.
    """
    if workers < 2 or seeds_per_comparison < MIN_PARALLEL_SEEDS:
        yield None
        return
    with ProcessPoolExecutor(max_workers=min(workers, seeds_per_comparison)) as executor:
        yield executor


def compare_modes(
    scenario: ScenarioConfig, tracker: TrackerConfig, seeds: Sequence[int], executor: Executor | None = None
) -> ComparisonSummary:
    """Paired baseline/fusion comparison over ``seeds``. With an ``executor``, seeds run in parallel.

    Seeds are independent and results are collected in seed order, so the summary is the same either way;
    only ``processing_time_s`` (wall time) can differ.
    """
    seeds = tuple(seeds)
    if executor is None:
        per_seed = [seed_metrics(scenario, tracker, seed) for seed in seeds]
    else:
        per_seed = list(executor.map(seed_metrics, repeat(scenario), repeat(tracker), seeds))
    per_mode = {mode: [metrics[mode] for metrics in per_seed] for mode in Mode}
    metrics = {
        mode: {name: summarise([float(getattr(m, name)) for m in runs]) for name in METRIC_NAMES}
        for mode, runs in per_mode.items()
    }
    delta, ci, rel, p, verdict = paired_verdict(
        [m.position_rmse_m for m in per_mode[Mode.BASELINE]],
        [m.position_rmse_m for m in per_mode[Mode.FUSION]],
    )
    return ComparisonSummary(scenario.name, seeds, metrics, delta, ci, rel, p, verdict)
