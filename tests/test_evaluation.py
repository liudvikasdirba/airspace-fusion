import math
from collections import Counter
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from airspace_fusion.config import SCENARIOS, TrackerConfig
from airspace_fusion.estimation import Disposition, Estimate, TrackerResult
from airspace_fusion.evaluation import (
    Mode,
    Verdict,
    compare_modes,
    evaluate_run,
    paired_verdict,
    run_single,
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


def estimate(t: datetime, x: float | None, var: float = 1.0, updated: tuple[str, ...] = ("m",)) -> Estimate:
    if x is None:
        return Estimate(t, None, None, ())
    return Estimate(t, np.array([x, 0.0, 0.0, 0.0]), np.eye(4) * var, updated)


def test_evaluate_run_known_values():
    result = TrackerResult(
        estimates=[estimate(TIMES[0], None), estimate(TIMES[1], 3.0), estimate(TIMES[2], 4.0, updated=()),
                   estimate(TIMES[3], 0.5)],
        dispositions={"a": Disposition.ACCEPTED, "b": Disposition.OUTSIDE_GATE, "c": Disposition.NO_TRACK},
        track_initiations=1,
        processing_time_s=0.01,
    )
    m = evaluate_run(result, truth(), invalid_count=2)
    assert m.position_rmse_m == pytest.approx(math.sqrt((9 + 16 + 0.25) / 3))
    assert m.coverage == pytest.approx(0.75)
    # 95 % ellipse with unit variance has radius sqrt(5.99) ~ 2.45: only the 0.5 m error is contained.
    assert m.containment_95 == pytest.approx(1 / 3)
    assert m.missed_updates == 1
    assert m.rejected_observations == 1
    assert m.invalid_observations == 2


def test_evaluate_run_without_estimates():
    result = TrackerResult([estimate(t, None) for t in TIMES], {}, 0, 0.0)
    m = evaluate_run(result, truth(), 0)
    assert math.isnan(m.position_rmse_m) and m.coverage == 0.0
    empty = evaluate_run(TrackerResult([], {}, 0, 0.0), truth(), 0)
    assert math.isnan(empty.coverage)


def test_evaluate_run_rejects_misaligned_timestamps():
    result = TrackerResult([estimate(T0 + timedelta(seconds=0.5), 1.0)], {}, 1, 0.0)
    with pytest.raises(ValueError, match="no ground-truth sample"):
        evaluate_run(result, truth(), 0)


def test_paired_verdict_cases():
    base = [10.0, 11.0, 12.0, 10.5, 11.5]
    assert paired_verdict(base, [b - 2.0 + 0.1 * i for i, b in enumerate(base)])[-1] is Verdict.IMPROVES
    assert paired_verdict(base, [b + 2.0 + 0.1 * i for i, b in enumerate(base)])[-1] is Verdict.WORSENS
    assert paired_verdict(base, base)[-1] is Verdict.NO_MATERIAL_CHANGE
    # Consistent but tiny (< 5 %) change is statistically significant yet not material.
    assert paired_verdict(base, [b - 0.1 + 0.001 * i for i, b in enumerate(base)])[-1] is Verdict.NO_MATERIAL_CHANGE
    # Large but inconsistent change is not significant.
    assert paired_verdict(base, [b + d for b, d in zip(base, [5, -5, 4, -4, 1])])[-1] is Verdict.NO_MATERIAL_CHANGE
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
    assert run.modes[Mode.BASELINE].metrics.invalid_observations == len(run.invalid_records)


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


def test_disposition_counts_are_derived():
    result = TrackerResult([], {"a": Disposition.ACCEPTED, "b": Disposition.ACCEPTED}, 0, 0.0)
    assert result.disposition_counts == Counter({Disposition.ACCEPTED: 2})
