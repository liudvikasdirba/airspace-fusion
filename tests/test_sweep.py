"""Edge-configuration sweep (slow; run with ``pytest -m slow``).

Every scenario at the extremes of the dashboard's noise and missed-observation controls,
over several seeds: the pipeline must not raise and every metric must be well-formed.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from airspace_fusion.config import SCENARIOS, TrackerConfig, with_sensor_overrides
from airspace_fusion.evaluation import Mode, run_single

pytestmark = pytest.mark.slow

NOISE_MULTIPLIERS = (0.25, 3.0)
MISS_PROBABILITIES = (0.0, 0.8)
SEEDS = (0, 1, 2)
COUNT_METRICS = (
    "missed_updates",
    "prediction_only_scans",
    "rejected_observations",
    "invalid_observations",
    "track_initiations",
)


@pytest.mark.parametrize("miss", MISS_PROBABILITIES)
@pytest.mark.parametrize("noise", NOISE_MULTIPLIERS)
@pytest.mark.parametrize("scenario_name", list(SCENARIOS))
def test_extreme_settings_produce_well_formed_metrics(scenario_name, noise, miss):
    scenario = with_sensor_overrides(SCENARIOS[scenario_name], noise, miss)
    n_scans = scenario.num_steps
    for seed in SEEDS:
        run = run_single(scenario, TrackerConfig(), seed)
        for mode in Mode:
            m = run.modes[mode].metrics
            context = f"{scenario_name} noise={noise} miss={miss} seed={seed} {mode.value}: {m}"
            assert len(run.modes[mode].result.estimates) == n_scans, context
            assert 0.0 <= m.coverage <= 1.0, context
            # RMSE and containment are NaN exactly when no estimate exists, finite otherwise.
            if m.coverage == 0.0:
                assert math.isnan(m.position_rmse_m) and math.isnan(m.containment_95), context
            else:
                assert math.isfinite(m.position_rmse_m) and m.position_rmse_m >= 0.0, context
                assert 0.0 <= m.containment_95 <= 1.0, context
            for name in COUNT_METRICS:
                value = getattr(m, name)
                assert isinstance(value, int) and value >= 0, context
            assert m.prediction_only_scans <= m.missed_updates <= n_scans, context
            assert m.processing_time_s >= 0.0, context
            assert set(dataclasses.asdict(m)) >= set(COUNT_METRICS), context


def test_overrides_apply_to_both_sensors():
    base = SCENARIOS["reference"]
    s = with_sensor_overrides(base, 3.0, 0.8)
    assert s.primary.noise_std == pytest.approx(3.0 * base.primary.noise_std)
    assert s.secondary.noise_std == pytest.approx(3.0 * base.secondary.noise_std)
    assert s.primary.detection_probability == pytest.approx(0.2)
    assert s.secondary.detection_probability == pytest.approx(0.2)
    assert with_sensor_overrides(base) == base
