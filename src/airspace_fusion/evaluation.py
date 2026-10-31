"""Evaluation of baseline and fusion tracking against ground truth.

This is the only module (besides plotting) that combines tracker output with truth.
Every number is computed from an actual simulation run.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

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
    missed_updates: int
    rejected_observations: int  # rejected by the tracker (gate, unsupported, geometry, duplicate)
    invalid_observations: int  # rejected by schema validation before tracking
    track_initiations: int
    processing_time_s: float


def evaluate_run(result: TrackerResult, truth: GroundTruth, invalid_count: int) -> RunMetrics:
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
        missed_updates=result.missed_updates,
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
        result = run_tracker(measurements, sim.report_times, tracker, sensor_ids=sensors_for_mode(scenario, mode))
        modes[mode] = ModeRun(result, evaluate_run(result, sim.truth, len(invalid)))
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


def paired_verdict(baseline: Sequence[float], fusion: Sequence[float]) -> tuple[float, tuple[float, float], float, float, Verdict]:
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
    "rejected_observations",
    "invalid_observations",
    "track_initiations",
    "processing_time_s",
)


def compare_modes(scenario: ScenarioConfig, tracker: TrackerConfig, seeds: Sequence[int]) -> ComparisonSummary:
    per_mode: dict[Mode, list[RunMetrics]] = {mode: [] for mode in Mode}
    for seed in seeds:
        run = run_single(scenario, tracker, seed)
        for mode in Mode:
            per_mode[mode].append(run.modes[mode].metrics)
    metrics = {
        mode: {name: summarise([float(getattr(m, name)) for m in runs]) for name in METRIC_NAMES}
        for mode, runs in per_mode.items()
    }
    delta, ci, rel, p, verdict = paired_verdict(
        [m.position_rmse_m for m in per_mode[Mode.BASELINE]],
        [m.position_rmse_m for m in per_mode[Mode.FUSION]],
    )
    return ComparisonSummary(scenario.name, tuple(seeds), metrics, delta, ci, rel, p, verdict)
