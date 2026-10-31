import ast
import inspect
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

import airspace_fusion.estimation as estimation
from airspace_fusion.config import SCENARIOS, TrackerConfig
from airspace_fusion.estimation import Disposition, SingleTargetTracker, run_tracker
from airspace_fusion.schemas import Measurement, parse_records
from airspace_fusion.simulator import simulate
from stonesoup.types.array import StateVector
from stonesoup.types.state import GaussianState

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
CFG = TrackerConfig()
_ids = iter(range(10**9))


def at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def pos(t: float, x: float, y: float, std: float = 10.0, sensor: str = "P", mid: str | None = None) -> Measurement:
    return Measurement(
        measurement_id=mid or f"p{next(_ids)}",
        sensor_id=sensor,
        timestamp_utc=at(t),
        sensor_position=(0.0, 0.0),
        measurement_type="POSITION_2D",
        measurement_value=(x, y),
        measurement_units=("m", "m"),
        measurement_uncertainty={"kind": "covariance", "matrix": ((std**2, 0.0), (0.0, std**2))},
    )


def brg(t: float, sensor_xy: tuple[float, float], target_xy: tuple[float, float], std: float = 0.005,
        offset: float = 0.0) -> Measurement:
    angle = math.atan2(target_xy[1] - sensor_xy[1], target_xy[0] - sensor_xy[0]) + offset
    return Measurement(
        measurement_id=f"b{next(_ids)}",
        sensor_id="B",
        timestamp_utc=at(t),
        sensor_position=sensor_xy,
        measurement_type="BEARING_ONLY",
        measurement_value=(angle,),
        measurement_units=("rad",),
        measurement_uncertainty={"kind": "angular_std", "std_rad": std},
    )


def straight_line(n: int, skip: set[int] = frozenset(), noise: float = 0.0, seed: int = 0) -> list[Measurement]:
    rng = np.random.default_rng(seed)
    return [
        pos(k, 100.0 + 10.0 * k + rng.normal(0, noise), 50.0 + 5.0 * k + rng.normal(0, noise))
        for k in range(n)
        if k not in skip
    ]


def grid(n: int) -> list[datetime]:
    return [at(k) for k in range(n)]


def test_empty_observations():
    result = run_tracker([], grid(10), CFG)
    assert len(result.estimates) == 10
    assert not any(e.has_track for e in result.estimates)
    assert result.prediction_only_scans == 0 and result.rejected_count == 0
    assert run_tracker([], [], CFG).estimates == []


def test_tracks_straight_line():
    result = run_tracker(straight_line(30, noise=5.0), grid(30), CFG)
    assert result.track_initiations == 1
    assert [e.timestamp for e in result.estimates] == grid(30)
    final = result.estimates[-1]
    np.testing.assert_allclose(final.position, [100 + 290, 50 + 145], atol=15.0)
    np.testing.assert_allclose(final.state[[1, 3]], [10.0, 5.0], atol=2.0)


def test_missing_observations_coast_and_inflate_uncertainty():
    result = run_tracker(straight_line(30, skip={10, 11, 12, 13}), grid(30), CFG)
    assert all(e.has_track for e in result.estimates[1:])
    assert result.prediction_only_scans == 4
    assert result.missed_updates_for("P") == 4
    variances = [np.trace(e.position_covariance) for e in result.estimates[9:14]]
    assert all(b > a for a, b in zip(variances, variances[1:]))
    # Coasting follows the constant-velocity prediction.
    np.testing.assert_allclose(result.estimates[13].position, [230.0, 115.0], atol=5.0)


def test_track_dropped_after_long_gap_and_reinitiated():
    gap = set(range(10, 25))
    result = run_tracker(straight_line(40, skip=gap), grid(40), CFG)
    has_track = [e.has_track for e in result.estimates]
    assert not has_track[0]  # one scan is not enough to initiate
    assert has_track[1] and not has_track[20] and has_track[-1]
    assert result.track_initiations == 2


def test_bearings_alone_never_initiate():
    bearings = [brg(k, (0.0, 0.0), (1000.0 + k, 500.0)) for k in range(10)]
    result = run_tracker(bearings, grid(10), CFG)
    assert not any(e.has_track for e in result.estimates)
    assert set(result.dispositions.values()) == {Disposition.NO_TRACK}


def test_report_time_validation():
    with pytest.raises(ValueError, match="UTC"):
        run_tracker([], [datetime(2026, 1, 1)], CFG)
    with pytest.raises(ValueError, match="increasing"):
        run_tracker([], [at(1), at(1)], CFG)
    with pytest.raises(ValueError, match="increasing"):
        run_tracker([], [at(2), at(1)], CFG)


def test_input_order_does_not_matter():
    ms = straight_line(20, noise=3.0) + [brg(k, (0.0, -2000.0), (100 + 10 * k, 50 + 5 * k)) for k in range(20)]
    shuffled = ms.copy()
    random.Random(1).shuffle(shuffled)
    a = run_tracker(ms, grid(20), CFG)
    b = run_tracker(shuffled, grid(20), CFG)
    for ea, eb in zip(a.estimates, b.estimates):
        np.testing.assert_array_equal(ea.state, eb.state)


def test_off_grid_measurements_processed_with_correct_timestamps():
    ms = [pos(k + 0.5, 100.0 + 10.0 * (k + 0.5), 50.0) for k in range(20)]
    result = run_tracker(ms, grid(21), CFG)
    assert [e.timestamp for e in result.estimates] == grid(21)
    assert result.disposition_counts[Disposition.ACCEPTED] == 18
    # At t=20 the filter has predicted 0.5 s beyond the last update at 19.5 s.
    assert result.estimates[-1].position[0] == pytest.approx(300.0, abs=3.0)


def test_unsupported_types_rejected_explicitly():
    base = pos(0, 0, 0).model_dump()
    rb = Measurement.model_validate({**base, "measurement_id": "rb", "measurement_type": "RANGE_BEARING",
                                     "measurement_value": (100.0, 0.1), "measurement_units": ("m", "rad")})
    det = Measurement.model_validate({**base, "measurement_id": "det", "measurement_type": "DETECTION_ONLY",
                                      "measurement_value": (), "measurement_units": (),
                                      "measurement_uncertainty": None})
    result = run_tracker(straight_line(5) + [rb, det], grid(5), CFG)
    assert result.dispositions["rb"] == Disposition.UNSUPPORTED_TYPE
    assert result.dispositions["det"] == Disposition.UNSUPPORTED_TYPE
    assert result.rejected_count == 2


def test_duplicate_ids_rejected_and_counted():
    ms = straight_line(5)
    result = run_tracker(ms + [ms[3], ms[3], ms[1]], grid(5), CFG)
    assert result.duplicate_count == 3
    assert result.rejected_count == 3
    assert set(result.dispositions) == {m.measurement_id for m in ms}


def test_bearing_only_scan_does_not_reset_initiation():
    # Primary misses scan 1; a bearing arrives then. Initiation from scans 0 and 2 must still happen,
    # exactly as it would without the bearing sensor.
    ms = straight_line(6, skip={1})
    bearing = brg(1, (0.0, -2000.0), (110.0, 55.0))
    without = run_tracker(ms, grid(6), CFG)
    with_bearing = run_tracker(ms + [bearing], grid(6), CFG)
    assert without.estimates[2].has_track
    assert [e.has_track for e in with_bearing.estimates] == [e.has_track for e in without.estimates]


def test_sensor_with_mixed_types_gated_per_type():
    ms = straight_line(20)
    truth_10 = (200.0, 100.0)
    bearing = brg(10, (0.0, -2000.0), truth_10).model_copy(update={"sensor_id": "P", "measurement_id": "mixed_b"})
    result = run_tracker(ms + [bearing], grid(20), CFG)
    # Both the position and the bearing from sensor "P" at t=10 are used; neither displaces the other.
    assert result.dispositions["mixed_b"] == Disposition.ACCEPTED
    assert len(result.estimates[10].updated_by) == 2
    assert result.estimates[10].updated_sensors == {"P"}


def test_far_false_observation_gated_out():
    false = pos(10, 3000.0, -3000.0, mid="false")
    result = run_tracker(straight_line(20) + [false], grid(20), CFG)
    assert result.dispositions["false"] == Disposition.OUTSIDE_GATE
    assert result.estimates[10].updated_by != ("false",)


def test_nearest_in_gate_selected():
    near = pos(10, 201.0, 100.0, mid="near")
    farther = pos(10, 215.0, 110.0, mid="farther")
    result = run_tracker(straight_line(20, skip={10}) + [farther, near], grid(20), CFG)
    assert result.dispositions["near"] == Disposition.ACCEPTED
    assert result.dispositions["farther"] == Disposition.NOT_SELECTED


def test_bearing_update_reduces_cross_range_uncertainty_only():
    tracker = SingleTargetTracker(CFG)
    prior = GaussianState(StateVector([1000.0, 0.0, 0.0, 0.0]), np.diag([100.0, 1.0, 100.0, 1.0]), timestamp=at(0))
    # Bearing sensor at origin looks along +x: bearing constrains y (cross-range), not x (range).
    m = brg(0, (0.0, 0.0), (1000.0, 0.0), std=0.001)
    post, used = tracker._update_with_sensor(prior, [m], at(0), {})
    assert used == [m]
    p = np.asarray(post.covar)
    assert p[2, 2] < 10.0  # y variance strongly reduced (1 mrad at 1 km = 1 m)
    assert p[0, 0] == pytest.approx(100.0, rel=1e-6)  # x variance unchanged


def test_position_covariance_weighting():
    tracker = SingleTargetTracker(CFG)
    prior = GaussianState(StateVector([0.0, 0.0, 0.0, 0.0]), np.diag([100.0, 1.0, 100.0, 1.0]), timestamp=at(0))
    tight, _ = tracker._update_with_sensor(prior, [pos(0, 5.0, 5.0, std=1.0)], at(0), {})
    loose, _ = tracker._update_with_sensor(prior, [pos(0, 5.0, 5.0, std=30.0)], at(0), {})
    assert np.asarray(tight.covar)[0, 0] < np.asarray(loose.covar)[0, 0]
    assert tight.state_vector[0] > loose.state_vector[0]  # pulled further toward the precise measurement


def test_bearing_angle_wraparound():
    # Target passes behind the bearing sensor's -x axis, where the bearing jumps between +pi and -pi.
    sensor = (0.0, 0.0)
    truth = [(-2000.0, 40.0 - 4.0 * k) for k in range(21)]
    ms = [pos(k, x, y, std=20.0) for k, (x, y) in enumerate(truth)]
    bearings = [brg(k, sensor, xy) for k, xy in enumerate(truth)]
    values = [b.measurement_value[0] for b in bearings]
    assert max(values) > 3.1 and min(values) < -3.1
    result = run_tracker(ms + bearings, grid(21), CFG)
    bearing_dispositions = [result.dispositions[b.measurement_id] for b in bearings]
    assert bearing_dispositions[2:] == [Disposition.ACCEPTED] * 19
    assert abs(result.estimates[-1].position[1] - truth[-1][1]) < 5.0


def test_degenerate_bearing_geometry_rejected():
    ms = straight_line(10)
    target_at_6 = (160.0, 80.0)
    on_top = brg(6, target_at_6, (target_at_6[0] + 1.0, target_at_6[1]))
    on_top = on_top.model_copy(update={"measurement_id": "on_top"})
    result = run_tracker(ms + [on_top], grid(10), CFG)
    assert result.dispositions["on_top"] == Disposition.DEGENERATE_GEOMETRY


def test_sensor_selection():
    ms = straight_line(10) + [brg(k, (0.0, -2000.0), (100 + 10 * k, 50 + 5 * k)) for k in range(10)]
    result = run_tracker(ms, grid(10), CFG, sensor_ids={"P"})
    assert all(mid.startswith("p") for mid in result.dispositions)


def test_estimation_module_has_no_truth_access():
    tree = ast.parse(Path(estimation.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    } | {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert not any(name.startswith(("airspace_fusion.simulator", "airspace_fusion.evaluation")) for name in imported)
    params = set(inspect.signature(run_tracker).parameters)
    assert params == {"measurements", "report_times", "config", "sensor_ids"}


def test_tracker_output_independent_of_truth_values():
    sim = simulate(SCENARIOS["reference"], seed=3)
    measurements, _ = parse_records(sim.records)
    a = run_tracker(measurements, sim.report_times, CFG)
    sim.truth.positions_m[:] = 0.0  # corrupting truth must not affect the estimator
    b = run_tracker(measurements, sim.report_times, CFG)
    for ea, eb in zip(a.estimates, b.estimates):
        assert (ea.state is None) == (eb.state is None)
        if ea.state is not None:
            np.testing.assert_array_equal(ea.state, eb.state)
