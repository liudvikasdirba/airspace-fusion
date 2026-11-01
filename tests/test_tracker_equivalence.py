"""The NumPy tracker against the Stone Soup tracker on the same inputs (ADR 0007).

Both share gating, selection and initiation, so any difference comes from the filter maths. They must
make the same decision about every measurement and agree on every estimate and NIS within 1e-9.
"""

from __future__ import annotations

import numpy as np
import pytest

from airspace_fusion.config import SCENARIOS, TrackerConfig, default_primary, default_secondary, with_sensor_overrides
from airspace_fusion.estimation import SingleTargetTracker, TrackerResult
from airspace_fusion.estimation_stonesoup import StoneSoupTracker
from airspace_fusion.schemas import parse_records
from airspace_fusion.simulator import simulate

TOLERANCE = 1e-9
DEFAULT_TRACKER = TrackerConfig()


def assert_same(fast: TrackerResult, reference: TrackerResult) -> None:
    assert fast.dispositions == reference.dispositions
    assert fast.track_initiations == reference.track_initiations
    assert fast.duplicate_count == reference.duplicate_count
    assert fast.update_nis.keys() == reference.update_nis.keys()
    for mid, nis in reference.update_nis.items():
        assert fast.update_nis[mid] == pytest.approx(nis, rel=TOLERANCE, abs=TOLERANCE), mid
    assert len(fast.estimates) == len(reference.estimates)
    for a, b in zip(fast.estimates, reference.estimates):
        assert (a.timestamp, a.updated_by, a.updated_sensors) == (b.timestamp, b.updated_by, b.updated_sensors)
        assert a.has_track == b.has_track
        if a.has_track:
            np.testing.assert_allclose(a.state, b.state, rtol=TOLERANCE, atol=TOLERANCE)
            np.testing.assert_allclose(a.covariance, b.covariance, rtol=TOLERANCE, atol=TOLERANCE)


def compare_trackers(scenario, seed: int, tracker: TrackerConfig = DEFAULT_TRACKER) -> None:
    sim = simulate(scenario, seed)
    measurements, _ = parse_records(sim.records)
    for sensors in ({scenario.primary.sensor_id}, None):  # baseline mode, fusion mode
        selected = [m for m in measurements if sensors is None or m.sensor_id in sensors]
        assert_same(
            SingleTargetTracker(tracker).run(selected, sim.report_times),
            StoneSoupTracker(tracker).run(selected, sim.report_times),
        )


@pytest.mark.parametrize("name", list(SCENARIOS))
@pytest.mark.parametrize("seed", [0, 1])
def test_same_results_on_golden_seeds(name, seed):
    compare_trackers(SCENARIOS[name], seed)


@pytest.mark.slow
@pytest.mark.parametrize("name", list(SCENARIOS))
@pytest.mark.parametrize("seed", range(2, 5))
def test_same_results_on_remaining_golden_seeds(name, seed):
    compare_trackers(SCENARIOS[name], seed)


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(3))
def test_same_results_under_stress(seed):
    """Heavy clutter, noisy sensors, frequent misses and unusual settings exercise gating and track drop."""
    clutter = SCENARIOS["reference"].model_copy(update={
        "primary": default_primary(false_observations_per_scan=5.0),
        "secondary": default_secondary(false_observations_per_scan=5.0),
    })
    compare_trackers(clutter, seed)
    noisy = with_sensor_overrides(SCENARIOS["distant_bearing"], noise_multiplier=3.0, miss_probability=0.4)
    compare_trackers(noisy, seed, TrackerConfig(process_noise_m2_s3=5.0, gate_probability=0.9999, max_coast_s=3.0))
