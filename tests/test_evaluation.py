import concurrent.futures
import math
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import airspace_fusion.evaluation as evaluation
from airspace_fusion.config import SCENARIOS, TrackerConfig
from airspace_fusion.estimation import Disposition, Estimate, TrackerResult
from airspace_fusion.evaluation import (
    MIN_PARALLEL_SEEDS,
    Mode,
    Verdict,
    compare_modes,
    evaluate_run,
    paired_verdict,
    run_single,
    seed_executor,
    sensors_for_mode,
)
from airspace_fusion.simulator import GroundTruth

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
TIMES = tuple(T0 + timedelta(seconds=k) for k in range(4))


def truth() -> GroundTruth:
    return GroundTruth(
        timestamps=TIMES,
        positions_m=np.zeros((4, 2)),
        velocities_mps=np.zeros((4, 2)),
        false_observation_ids=frozenset(),
        corrupted_record_ids=frozenset(),
    )


def estimate(t: datetime, x: float | None, var: float = 1.0, sensors: tuple[str, ...] = ("P",)) -> Estimate:
    if x is None:
        return Estimate(t, None, None, ())
    ids = tuple(f"{s}-m" for s in sensors)
    return Estimate(t, np.array([x, 0.0, 0.0, 0.0]), np.eye(4) * var, ids, frozenset(sensors))


def test_evaluate_run_known_values():
    result = TrackerResult(
        estimates=[estimate(TIMES[0], None), estimate(TIMES[1], 3.0), estimate(TIMES[2], 4.0, sensors=("B",)),
                   estimate(TIMES[3], 0.5, sensors=())],
        dispositions={"a": Disposition.ACCEPTED, "b": Disposition.OUTSIDE_GATE, "c": Disposition.NO_TRACK},
        track_initiations=1,
        processing_time_s=0.01,
    )
    m = evaluate_run(result, truth(), invalid_count=2, primary_sensor_id="P")
    assert m.position_rmse_m == pytest.approx(math.sqrt((9 + 16 + 0.25) / 3))
    assert m.coverage == pytest.approx(0.75)
    # 95 % ellipse with unit variance has radius sqrt(5.99) ~ 2.45: only the 0.5 m error is contained.
    assert m.containment_95 == pytest.approx(1 / 3)
    assert m.missed_updates == 2  # primary absent at t=2 (bearing only) and t=3 (nothing)
    assert m.prediction_only_scans == 1
    assert m.rejected_observations == 1
    assert m.invalid_observations == 2


def test_evaluate_run_without_estimates():
    result = TrackerResult([estimate(t, None) for t in TIMES], {}, 0, 0.0)
    m = evaluate_run(result, truth(), 0, "P")
    assert math.isnan(m.position_rmse_m) and m.coverage == 0.0
    empty = evaluate_run(TrackerResult([], {}, 0, 0.0), truth(), 0, "P")
    assert math.isnan(empty.coverage)


def test_evaluate_run_rejects_misaligned_timestamps():
    result = TrackerResult([estimate(T0 + timedelta(seconds=0.5), 1.0)], {}, 1, 0.0)
    with pytest.raises(ValueError, match="no ground-truth sample"):
        evaluate_run(result, truth(), 0, "P")


def test_paired_verdict_cases():
    base = [10.0, 11.0, 12.0, 10.5, 11.5]
    assert paired_verdict(base, [b - 2.0 + 0.1 * i for i, b in enumerate(base)])[-1] is Verdict.IMPROVES
    assert paired_verdict(base, [b + 2.0 + 0.1 * i for i, b in enumerate(base)])[-1] is Verdict.WORSENS
    assert paired_verdict(base, base)[-1] is Verdict.NO_MATERIAL_CHANGE
    # Consistent but tiny (< 5 %) change is statistically significant yet not material.
    assert paired_verdict(base, [b - 0.1 + 0.001 * i for i, b in enumerate(base)])[-1] is Verdict.NO_MATERIAL_CHANGE
    # Large but inconsistent change is not significant.
    assert paired_verdict(base, [b + d for b, d in zip(base, [5, -5, 4, -4, 1])])[-1] is Verdict.NO_MATERIAL_CHANGE
    # Identical change on every seed has zero variance: still a certain, material change.
    delta, ci, rel, p, verdict = paired_verdict(base, [b - 2.0 for b in base])
    assert verdict is Verdict.IMPROVES and p == 0.0 and ci == (pytest.approx(-2.0), pytest.approx(-2.0))
    assert paired_verdict([1.0], [0.5])[-1] is Verdict.INSUFFICIENT_DATA
    assert paired_verdict([1.0, math.nan], [0.5, 0.4])[-1] is Verdict.INSUFFICIENT_DATA


def test_run_single_uses_identical_truth_and_inputs():
    scenario = SCENARIOS["reference"]
    run = run_single(scenario, TrackerConfig(), seed=2)
    baseline, fusion = run.modes[Mode.BASELINE].result, run.modes[Mode.FUSION].result
    primary, secondary = scenario.primary.sensor_id, scenario.secondary.sensor_id
    assert sensors_for_mode(scenario, Mode.BASELINE) == (primary,)
    assert all(mid.startswith(primary) for mid in baseline.dispositions)
    assert any(mid.startswith(secondary) for mid in fusion.dispositions)
    # Every primary measurement is presented to both modes.
    assert {k for k in fusion.dispositions if k.startswith(primary)} == set(baseline.dispositions)
    assert [e.timestamp for e in baseline.estimates] == [e.timestamp for e in fusion.estimates]
    assert list(run.simulation.truth.timestamps) == [e.timestamp for e in baseline.estimates]


def test_invalid_records_attributed_to_mode_sensors():
    scenario = SCENARIOS["reference"].model_copy(update={"corrupted_record_probability": 0.2})
    run = run_single(scenario, TrackerConfig(), seed=2)
    primary = scenario.primary.sensor_id
    n_primary = sum(r.sensor_id == primary for r in run.invalid_records)
    assert 0 < n_primary < len(run.invalid_records)
    assert run.modes[Mode.BASELINE].metrics.invalid_observations == n_primary
    assert run.modes[Mode.FUSION].metrics.invalid_observations == len(run.invalid_records)


def test_metrics_respond_to_simulation_inputs():
    cfg = TrackerConfig()
    clean = run_single(SCENARIOS["reference"], cfg, seed=0).modes[Mode.BASELINE].metrics
    noisy = run_single(SCENARIOS["noisy_primary"], cfg, seed=0).modes[Mode.BASELINE].metrics
    assert noisy.position_rmse_m > clean.position_rmse_m * 1.5


def test_compare_modes_reproducible_and_complete():
    scenario = SCENARIOS["reference"].model_copy(update={"duration_s": 40.0})
    a = compare_modes(scenario, TrackerConfig(), seeds=[0, 1, 2])
    b = compare_modes(scenario, TrackerConfig(), seeds=[0, 1, 2])
    for mode in Mode:
        for name, summary in a.metrics[mode].items():
            assert summary.n == 3
            if name != "processing_time_s":
                assert summary == b.metrics[mode][name]
    assert a.verdict == b.verdict and a.rmse_delta_mean_m == b.rmse_delta_mean_m


def without_timing(summary):
    return {mode: {k: v for k, v in m.items() if k != "processing_time_s"} for mode, m in summary.metrics.items()}


@pytest.mark.parametrize("batches", [64, 2])  # 2 batches: several seeds per task
def test_parallel_comparison_matches_serial(batches, monkeypatch):
    monkeypatch.setattr(evaluation, "PARALLEL_BATCHES", batches)
    scenario = SCENARIOS["biased_bearing"].model_copy(update={"duration_s": 40.0})
    seeds = range(3, 3 + MIN_PARALLEL_SEEDS + 1)
    serial = compare_modes(scenario, TrackerConfig(), seeds)
    with seed_executor(2, len(seeds)) as executor:
        assert executor is not None
        parallel = compare_modes(scenario, TrackerConfig(), seeds, executor)
    assert parallel.seeds == serial.seeds == tuple(seeds)
    assert without_timing(parallel) == without_timing(serial)
    assert (parallel.rmse_delta_mean_m, parallel.rmse_delta_ci95_m, parallel.rmse_p_value, parallel.verdict) == (
        serial.rmse_delta_mean_m, serial.rmse_delta_ci95_m, serial.rmse_p_value, serial.verdict
    )


@pytest.mark.parametrize("workers, seeds", [(1, 100), (2, MIN_PARALLEL_SEEDS - 1)])
def test_seed_executor_runs_serially_when_parallel_does_not_pay(workers, seeds):
    with seed_executor(workers, seeds) as executor:
        assert executor is None


def test_seed_executor_respects_the_platform_worker_limit(monkeypatch):
    monkeypatch.setattr(evaluation, "MAX_WORKERS", 2)
    with seed_executor(64, 100) as executor:
        assert executor is not None and executor._max_workers == 2


def test_interrupted_seed_executor_drops_queued_work():
    started = time.perf_counter()
    with pytest.raises(KeyboardInterrupt), seed_executor(2, MIN_PARALLEL_SEEDS) as executor:
        assert executor is not None
        futures = [executor.submit(time.sleep, 0.5) for _ in range(40)]  # about 10 s of work for 2 workers
        raise KeyboardInterrupt
    assert time.perf_counter() - started < 5.0  # did not wait for the queued seeds
    concurrent.futures.wait(futures, timeout=10)  # the pool cancels in its own thread
    assert sum(f.cancelled() for f in futures) >= 30


def test_disposition_counts_are_derived():
    result = TrackerResult([], {"a": Disposition.ACCEPTED, "b": Disposition.ACCEPTED}, 0, 0.0)
    assert result.disposition_counts == Counter({Disposition.ACCEPTED: 2})
