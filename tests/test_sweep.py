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
from airspace_fusion.scenarios import FAMILIES

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


@pytest.mark.parametrize("family_name", list(FAMILIES))
def test_every_family_world_runs_with_well_formed_metrics(family_name):
    """Many drawn geometries per family, at default settings: no exceptions, sane metrics, tracks form."""
    family = FAMILIES[family_name]
    coverage = []
    for seed in range(25):
        world = family.for_seed(seed)
        run = run_single(world, TrackerConfig(), seed)
        for mode in Mode:
            m = run.modes[mode].metrics
            context = f"{family_name} seed={seed} {mode.value}: {world.description}"
            assert 0.0 <= m.coverage <= 1.0, context
            if m.coverage > 0.0:
                assert math.isfinite(m.position_rmse_m) and math.isfinite(m.gospa_m), context
            coverage.append(m.coverage)
    assert sum(coverage) / len(coverage) > 0.8, family_name  # drawn worlds are trackable, not degenerate
