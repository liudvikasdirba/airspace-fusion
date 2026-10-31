import math

import numpy as np
import pytest

from airspace_fusion.config import SCENARIOS, default_primary, default_secondary
from airspace_fusion.schemas import Measurement, MeasurementType, parse_records
from airspace_fusion.simulator import simulate

REFERENCE = SCENARIOS["reference"]


def test_reproducible_with_same_seed():
    a = simulate(REFERENCE, seed=7)
    b = simulate(REFERENCE, seed=7)
    np.testing.assert_array_equal(a.truth.positions_m, b.truth.positions_m)
    # repr() so that NaNs in deliberately corrupted records compare equal.
    assert repr(a.records) == repr(b.records)


def test_different_seeds_differ():
    a = simulate(REFERENCE, seed=1)
    b = simulate(REFERENCE, seed=2)
    assert not np.array_equal(a.truth.positions_m, b.truth.positions_m)


def test_truth_independent_of_sensor_settings():
    noisy = REFERENCE.model_copy(update={"primary": default_primary(noise_std=100.0, detection_probability=0.3)})
    a = simulate(REFERENCE, seed=3)
    b = simulate(noisy, seed=3)
    np.testing.assert_array_equal(a.truth.positions_m, b.truth.positions_m)


def test_truth_follows_configured_motion():
    scenario = REFERENCE.model_copy(update={"random_acceleration_std_mps2": 0.0})
    out = simulate(scenario, seed=0)
    v = out.truth.velocities_mps
    speed = np.linalg.norm(v, axis=1)
    np.testing.assert_allclose(speed, 15.0, rtol=1e-9)
    heading_change = math.atan2(v[-1, 1], v[-1, 0]) - math.atan2(v[0, 1], v[0, 0])
    assert heading_change == pytest.approx(math.radians(-90.0), abs=1e-6)
    assert len(out.truth.timestamps) == scenario.num_steps == len(out.truth.positions_m)


def test_records_validate_and_contain_no_truth():
    out = simulate(REFERENCE.model_copy(update={"corrupted_record_probability": 0.0}), seed=4)
    valid, rejected = parse_records(out.records)
    assert not rejected
    assert len(valid) == len(out.records) > 0
    for record in out.records:
        assert set(record) <= set(Measurement.model_fields)


def test_noise_statistics_match_configuration():
    scenario = REFERENCE.model_copy(
        update={
            "duration_s": 2000.0,
            "corrupted_record_probability": 0.0,
            "primary": default_primary(detection_probability=1.0, false_observations_per_scan=0.0),
            "secondary": default_secondary(detection_probability=1.0, false_observations_per_scan=0.0),
        }
    )
    out = simulate(scenario, seed=11)
    valid, _ = parse_records(out.records)
    pos = np.array([m.measurement_value for m in valid if m.measurement_type is MeasurementType.POSITION_2D])
    assert len(pos) == scenario.num_steps
    assert np.std(pos - out.truth.positions_m) == pytest.approx(25.0, rel=0.05)

    brg = np.array([m.measurement_value[0] for m in valid if m.measurement_type is MeasurementType.BEARING_ONLY])
    d = out.truth.positions_m - np.array(scenario.secondary.position_m)
    err = (brg - np.arctan2(d[:, 1], d[:, 0]) + np.pi) % (2 * np.pi) - np.pi
    assert np.std(err) == pytest.approx(scenario.secondary.noise_std, rel=0.05)
    assert abs(np.mean(err)) < 3 * scenario.secondary.noise_std / math.sqrt(len(err))


def test_missed_and_false_observations():
    scenario = REFERENCE.model_copy(
        update={
            "duration_s": 1000.0,
            "corrupted_record_probability": 0.0,
            "primary": default_primary(detection_probability=0.5, false_observations_per_scan=1.0),
        }
    )
    out = simulate(scenario, seed=5)
    primary = [r for r in out.records if r["sensor_id"] == scenario.primary.sensor_id]
    n_false = sum(r["measurement_id"] in out.truth.false_observation_ids for r in primary)
    n_true = len(primary) - n_false
    assert n_true / scenario.num_steps == pytest.approx(0.5, abs=0.06)
    assert n_false / scenario.num_steps == pytest.approx(1.0, abs=0.12)


def test_corrupted_records_are_rejected_by_schema():
    out = simulate(REFERENCE.model_copy(update={"corrupted_record_probability": 0.2}), seed=6)
    _, rejected = parse_records(out.records)
    assert {r.measurement_id for r in rejected} == set(out.truth.corrupted_record_ids)
    assert rejected


def test_scan_interval():
    scenario = REFERENCE.model_copy(update={"secondary": default_secondary(scan_interval_steps=3)})
    out = simulate(scenario, seed=8)
    times = {r["timestamp_utc"] for r in out.records if r["sensor_id"] == scenario.secondary.sensor_id}
    allowed = set(out.truth.timestamps[::3])
    assert times <= allowed
