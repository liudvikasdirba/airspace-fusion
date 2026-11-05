"""Evaluation of baseline and fusion tracking against ground truth.

This is the only module (besides plotting) that combines tracker output with truth.
Every number is computed from an actual simulation run.
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Iterator, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import Enum
from itertools import repeat

import numpy as np

from airspace_fusion.config import ScenarioConfig, TrackerConfig
from airspace_fusion.estimation import TrackerResult, chi2_quantile, run_tracker
from airspace_fusion.scenarios import ScenarioFamily, ScenarioSource
from airspace_fusion.schemas import Measurement, RejectedRecord, parse_records
from airspace_fusion.simulator import GroundTruth, SimulationOutput, simulate

# 95 % probability contour of a 2D Gaussian (chi-square, 2 dof).
CHI2_2DOF_95 = chi2_quantile(0.95, 2)
# GOSPA cut-off c (p = 2, alpha = 2). A scan without a track costs c / sqrt(2); an estimate further than c from
# the truth costs c, the same as a missed object plus a false track. 150 m is about 3x the worst scenario RMSE,
# so a correctly held track is almost never cut off.
GOSPA_CUTOFF_M = 150.0
# Relative RMSE change below which a difference is not considered material.
MATERIAL_RELATIVE_CHANGE = 0.05
SIGNIFICANCE_LEVEL = 0.05
# Verdict v2 (ADR 0009). Below this mean 95 % containment fusion is over-confident, whatever its RMSE does.
OVER_CONFIDENT_CONTAINMENT = 0.85
# An RMSE improvement counts only if fusion's containment is at most this much below the baseline's.
MAX_CONTAINMENT_DROP = 0.03
DEFAULT_SEEDS = 200
# Below this many seeds per comparison, starting worker processes costs more than it saves.
MIN_PARALLEL_SEEDS = 4
# Seeds per comparison are sent to workers in about this many batches: few enough that the scenario and
# tracker are not pickled once per seed, enough to keep every worker busy until the end.
PARALLEL_BATCHES = 64
BLAS_THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
# ProcessPoolExecutor refuses more workers than this on Windows.
MAX_WORKERS = 61 if sys.platform == "win32" else None


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
    gospa_m: float  # RMS over all scans of single-object GOSPA; missing tracks count, unlike RMSE
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
    truth_xy = truth.positions_m.tolist()
    sq_errors: list[float] = []
    contained = 0
    for est in result.estimates:
        if est.timestamp not in truth_index:
            raise ValueError(f"estimate at {est.timestamp} has no ground-truth sample")
        if est.state is None or est.covariance is None:
            continue
        # Plain floats and a closed-form 2x2 inverse: per estimate, NumPy's call overhead dominates the work.
        tx, ty = truth_xy[truth_index[est.timestamp]]
        ex, ey = float(est.state[0]) - tx, float(est.state[2]) - ty
        sq_errors.append(ex * ex + ey * ey)
        p = est.covariance
        a, b, d = float(p[0, 0]), float(p[0, 2]), float(p[2, 2])
        det = a * d - b * b
        if not det > 0.0:
            raise FloatingPointError(f"position covariance at {est.timestamp} is not positive definite")
        if (d * ex * ex - 2.0 * b * ex * ey + a * ey * ey) / det <= CHI2_2DOF_95:
            contained += 1
    n_est = len(sq_errors)
    n_total = len(result.estimates)
    return RunMetrics(
        position_rmse_m=math.sqrt(sum(sq_errors) / n_est) if n_est else math.nan,
        gospa_m=gospa(sq_errors, n_total - n_est),
        coverage=n_est / n_total if n_total else math.nan,
        containment_95=contained / n_est if n_est else math.nan,
        missed_updates=result.missed_updates_for(primary_sensor_id),
        prediction_only_scans=result.prediction_only_scans,
        rejected_observations=result.rejected_count,
        invalid_observations=invalid_count,
        track_initiations=result.track_initiations,
        processing_time_s=result.processing_time_s,
    )


def gospa(sq_errors: Sequence[float], scans_without_track: int, cutoff_m: float = GOSPA_CUTOFF_M) -> float:
    """Root mean square over scans of GOSPA (p = 2, alpha = 2) for one true object, NaN if there are no scans.

    The object exists at every scan. A scan with a track costs min(d, c)^2, where d is the position error;
    a scan without one costs c^2 / 2 (one missed object). With every scan tracked within c, this equals the RMSE.
    """
    n_total = len(sq_errors) + scans_without_track
    if n_total == 0:
        return math.nan
    c2 = cutoff_m * cutoff_m
    total = sum(min(e, c2) for e in sq_errors) + scans_without_track * c2 / 2.0
    return math.sqrt(total / n_total)


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


# After a turn ends, a filter needs a while to settle; those scans count as neither turn nor straight.
SETTLING_S = 20.0
# The first scans of a run are track start-up; they count as neither.
STARTUP_S = 10.0


def segment_squared_errors(run: SingleRun, mode: Mode) -> dict[str, list[float]]:
    """Squared position errors of one mode, split into "turn" scans (between the scenario's turn start and end)
    and "straight" scans (from STARTUP_S to the turn start, and from SETTLING_S after the turn end).

    Pool lists over seeds and take the root of the mean to get segment RMSEs (the M4 exit criteria).
    """
    scenario, truth = run.simulation.scenario, run.simulation.truth
    t0 = truth.timestamps[0]
    segments: dict[str, list[float]] = {"turn": [], "straight": []}
    for k, estimate in enumerate(run.modes[mode].result.estimates):
        if estimate.state is None:
            continue
        t = (estimate.timestamp - t0).total_seconds()
        if scenario.turn_start_s <= t < scenario.turn_end_s:
            segment = "turn"
        elif STARTUP_S <= t < scenario.turn_start_s or t >= scenario.turn_end_s + SETTLING_S:
            segment = "straight"
        else:
            continue
        dx, dy = estimate.state[0] - truth.positions_m[k, 0], estimate.state[2] - truth.positions_m[k, 1]
        segments[segment].append(float(dx * dx + dy * dy))
    return segments


class Verdict(str, Enum):
    IMPROVES = "fusion improves"
    WORSENS = "fusion worsens"
    OVER_CONFIDENT = "fusion over-confident"
    NO_MATERIAL_CHANGE = "no material change"
    INSUFFICIENT_DATA = "insufficient data"

    @property
    def harmful(self) -> bool:
        """Fusion should not be used: it is less accurate, or it misstates its own uncertainty."""
        return self in (Verdict.WORSENS, Verdict.OVER_CONFIDENT)


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
    # Holm-adjusted p-value when this comparison was one of several in a command run; None for a single one.
    # The verdict uses it in place of ``rmse_p_value``.
    rmse_p_adjusted: float | None = None
    # Every seed's metrics, in seed order, for paired tests across settings (the learning loop). Not part of
    # equality: wall-clock time differs between otherwise identical runs.
    per_seed: dict[Mode, tuple[RunMetrics, ...]] = field(default_factory=dict, compare=False, repr=False)

    @property
    def decision_p_value(self) -> float:
        """The RMSE p-value the verdict rests on: Holm-adjusted when there is one, raw otherwise."""
        return self.rmse_p_value if self.rmse_p_adjusted is None else self.rmse_p_adjusted

    def mean_containment(self) -> dict[Mode, float]:
        return mean_containment(self.metrics)


def mean_containment(metrics: dict[Mode, dict[str, MetricSummary]]) -> dict[Mode, float]:
    """Each mode's mean 95 % containment, the input to the containment rules of ADR 0009."""
    return {mode: metrics[mode]["containment_95"].mean for mode in Mode}


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
        from scipy import stats  # imported here so that worker processes, which never need it, start faster

        sem = float(d.std(ddof=1)) / math.sqrt(d.size)
        half = float(stats.t.ppf(0.975, d.size - 1)) * sem
        p_value = float(stats.ttest_rel(f, b).pvalue)
    return mean_d, (mean_d - half, mean_d + half), rel, p_value, rmse_verdict(mean_d, rel, p_value)


def rmse_verdict(mean_delta: float, relative_change: float, p_value: float) -> Verdict:
    """Verdict from the paired RMSE test alone: significant (p < 0.05) and material (at least 5 %)."""
    if p_value < SIGNIFICANCE_LEVEL and abs(relative_change) >= MATERIAL_RELATIVE_CHANGE:
        return Verdict.IMPROVES if mean_delta < 0 else Verdict.WORSENS
    return Verdict.NO_MATERIAL_CHANGE


def verdict_v2(rmse: Verdict, baseline_containment: float, fusion_containment: float) -> Verdict:
    """Add the containment rules of ADR 0009 to the RMSE verdict.

    Fusion is over-confident when its mean 95 % containment is below 0.85 and below the baseline's, or when it
    improves RMSE but its containment falls more than 0.03 below the baseline's: the accuracy gain then comes
    with an uncertainty the filter understates. A fusion filter at least as well calibrated as the baseline is
    never called over-confident: the verdict compares the two modes, and switching fusion off would not help.
    Without a containment value for both modes the RMSE verdict stands.
    """
    if rmse is Verdict.INSUFFICIENT_DATA or math.isnan(baseline_containment) or math.isnan(fusion_containment):
        return rmse
    if fusion_containment < min(OVER_CONFIDENT_CONTAINMENT, baseline_containment):
        return Verdict.OVER_CONFIDENT
    if rmse is Verdict.IMPROVES and fusion_containment < baseline_containment - MAX_CONTAINMENT_DROP:
        return Verdict.OVER_CONFIDENT
    return rmse


def summary_verdict(summary: ComparisonSummary, p_value: float) -> Verdict:
    """Verdict v2 of ``summary`` with ``p_value`` as the RMSE test's p-value."""
    if summary.verdict is Verdict.INSUFFICIENT_DATA:
        return summary.verdict
    rmse = rmse_verdict(summary.rmse_delta_mean_m, summary.rmse_relative_change, p_value)
    containment = summary.mean_containment()
    return verdict_v2(rmse, containment[Mode.BASELINE], containment[Mode.FUSION])


def with_adjusted_p(summary: ComparisonSummary, p_adjusted: float) -> ComparisonSummary:
    """``summary`` with its Holm-adjusted p-value and the verdict that follows from it."""
    return replace(summary, rmse_p_adjusted=p_adjusted, verdict=summary_verdict(summary, p_adjusted))


def explain_verdict(summary: ComparisonSummary) -> str:
    """Why ``summary`` got its verdict, in one sentence."""
    p = summary.decision_p_value
    p_text = "p" if summary.rmse_p_adjusted is None else "Holm-adjusted p"
    containment = summary.mean_containment()
    if summary.verdict is Verdict.INSUFFICIENT_DATA:
        return "fewer than two seeds have an RMSE in both modes"
    if summary.verdict is Verdict.OVER_CONFIDENT:
        if containment[Mode.FUSION] < min(OVER_CONFIDENT_CONTAINMENT, containment[Mode.BASELINE]):
            return (f"fusion's 95 % containment is {containment[Mode.FUSION]:.2f}, below "
                    f"{OVER_CONFIDENT_CONTAINMENT:.2f} and below the baseline's ({containment[Mode.BASELINE]:.2f})")
        return (f"RMSE improves ({p_text} = {p:.3g}), but fusion's 95 % containment "
                f"({containment[Mode.FUSION]:.2f}) is more than {MAX_CONTAINMENT_DROP:.2f} "
                f"below the baseline's ({containment[Mode.BASELINE]:.2f})")
    material = abs(summary.rmse_relative_change) >= MATERIAL_RELATIVE_CHANGE
    significant = p < SIGNIFICANCE_LEVEL
    if summary.verdict is Verdict.NO_MATERIAL_CHANGE:
        reasons = ([] if significant else [f"{p_text} = {p:.3g} is not below {SIGNIFICANCE_LEVEL}"]) + (
            [] if material else [f"the change ({summary.rmse_relative_change:+.1%}) is under "
                                 f"{MATERIAL_RELATIVE_CHANGE:.0%}"])
        return " and ".join(reasons)
    return (f"RMSE {summary.rmse_relative_change:+.1%} with {p_text} = {p:.3g}; fusion's 95 % containment "
            f"{containment[Mode.FUSION]:.2f} (baseline {containment[Mode.BASELINE]:.2f})")


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order. NaN stays NaN and does not count as a test."""
    tested = sorted((p, i) for i, p in enumerate(p_values) if not math.isnan(p))
    adjusted = [math.nan] * len(p_values)
    running = 0.0
    for rank, (p, i) in enumerate(tested):
        running = max(running, min(1.0, (len(tested) - rank) * p))  # step-down: never below an earlier one
        adjusted[i] = running
    return adjusted


def with_holm(summaries: Sequence[ComparisonSummary]) -> list[ComparisonSummary]:
    """Correct the RMSE tests of several comparisons from one command run for multiple testing.

    Each summary gets its Holm-adjusted p-value and the verdict that follows from it. A single comparison
    is returned unchanged.
    """
    if len(summaries) < 2:
        return list(summaries)
    adjusted = holm_adjust([s.rmse_p_value for s in summaries])
    return [with_adjusted_p(s, p) for s, p in zip(summaries, adjusted)]


METRIC_NAMES = (
    "position_rmse_m",
    "gospa_m",
    "coverage",
    "containment_95",
    "missed_updates",
    "prediction_only_scans",
    "rejected_observations",
    "invalid_observations",
    "track_initiations",
    "processing_time_s",
)


def seed_metrics(scenario: ScenarioSource, tracker: TrackerConfig, seed: int) -> dict[Mode, RunMetrics]:
    """Both modes' metrics for one seed, in the world ``scenario`` gives that seed. Module-level, for workers."""
    run = run_single(scenario.for_seed(seed), tracker, seed)
    return {mode: run.modes[mode].metrics for mode in Mode}


@contextmanager
def seed_executor(workers: int, seeds_per_comparison: int) -> Iterator[Executor | None]:
    """A process pool for ``compare_modes``, or None when running serially is cheaper.

    Create it once per command and pass it to every comparison: starting the worker processes takes
    a moment, because each one imports NumPy and pydantic.
    """
    if workers < 2 or seeds_per_comparison < MIN_PARALLEL_SEEDS:
        yield None
        return
    workers = min(workers, seeds_per_comparison, MAX_WORKERS or workers)
    # The tracker multiplies 4x4 matrices: BLAS threads cannot help there, and with several workers they only
    # compete for the same cores. Workers inherit this environment; a value the user set is kept.
    for variable in BLAS_THREAD_VARIABLES:
        os.environ.setdefault(variable, "1")
    executor = ProcessPoolExecutor(max_workers=workers)
    try:
        yield executor
    except BaseException:
        # Interrupted or failed: drop queued seeds instead of waiting for them, then re-raise.
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    executor.shutdown(wait=True)


def compare_modes(
    scenario: ScenarioSource, tracker: TrackerConfig, seeds: Sequence[int], executor: Executor | None = None
) -> ComparisonSummary:
    """Paired baseline/fusion comparison over ``seeds``. With an ``executor``, seeds run in parallel.

    ``scenario`` is a fixed scenario or a family; a family draws a different world for each seed, and both modes
    of a seed always share that world, so the comparison stays paired.

    Seeds are independent and results are collected in seed order, so the summary is the same either way;
    only ``processing_time_s`` (wall time) can differ.
    """
    seeds = tuple(seeds)
    if executor is None:
        per_seed = [seed_metrics(scenario, tracker, seed) for seed in seeds]
    else:
        chunksize = max(1, len(seeds) // PARALLEL_BATCHES)
        per_seed = list(executor.map(seed_metrics, repeat(scenario), repeat(tracker), seeds, chunksize=chunksize))
    per_mode = {mode: [metrics[mode] for metrics in per_seed] for mode in Mode}
    metrics = {
        mode: {name: summarise([float(getattr(m, name)) for m in runs]) for name in METRIC_NAMES}
        for mode, runs in per_mode.items()
    }
    delta, ci, rel, p, rmse = paired_verdict(
        [m.position_rmse_m for m in per_mode[Mode.BASELINE]],
        [m.position_rmse_m for m in per_mode[Mode.FUSION]],
    )
    containment = mean_containment(metrics)
    verdict = verdict_v2(rmse, containment[Mode.BASELINE], containment[Mode.FUSION])
    return ComparisonSummary(
        scenario.name, seeds, metrics, delta, ci, rel, p, verdict,
        per_seed={mode: tuple(runs) for mode, runs in per_mode.items()},
    )


@dataclass(frozen=True)
class SweepPoint:
    value: float
    family: ScenarioFamily  # the family with the swept parameter fixed at ``value``
    summary: ComparisonSummary


def sweep(
    family: ScenarioFamily,
    parameter: str,
    values: Sequence[float],
    tracker: TrackerConfig,
    seeds: Sequence[int],
    executor: Executor | None = None,
) -> list[SweepPoint]:
    """Paired comparison at each value of one family parameter, over the same seeds.

    The family draws every other parameter of a seed identically at every point (common random numbers), so
    differences between points come from the swept parameter, not from different worlds. Verdicts are
    Holm-corrected across the points (ADR 0009).
    """
    if not values:
        raise ValueError("a sweep needs at least one value")
    points = []
    for value in values:
        if not math.isfinite(value):
            raise ValueError(f"sweep values must be finite, got {value}")
        fixed = family.with_fixed(parameter, value)
        points.append(SweepPoint(value, fixed, compare_modes(fixed, tracker, seeds, executor)))
    corrected = with_holm([point.summary for point in points])  # the points are one family of tests
    return [replace(point, summary=summary) for point, summary in zip(points, corrected)]
