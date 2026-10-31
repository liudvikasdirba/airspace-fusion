"""Deterministic 2D single-object simulator with two stationary synthetic sensors.

Output is split into two parts that must never be mixed:

* ``SimulationOutput.records`` - the raw records the sensors emit. This is the
  only thing the tracking components may consume.
* ``SimulationOutput.truth`` - ground truth trajectory plus truth-side labels
  (which records are false observations). Used only by evaluation and plots.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from airspace_fusion.config import EPOCH_UTC, ScenarioConfig, SensorConfig
from airspace_fusion.schemas import EXPECTED_UNITS, MeasurementType, wrap_angle


@dataclass(frozen=True)
class GroundTruth:
    timestamps: tuple[datetime, ...]
    positions_m: np.ndarray  # shape (N, 2): x, y
    velocities_mps: np.ndarray  # shape (N, 2): vx, vy
    false_observation_ids: frozenset[str]
    corrupted_record_ids: frozenset[str]


@dataclass(frozen=True)
class SimulationOutput:
    scenario: ScenarioConfig
    seed: int
    truth: GroundTruth
    records: list[dict[str, Any]] = field(repr=False)

    @property
    def report_times(self) -> tuple[datetime, ...]:
        """The sensor scan grid. Known to the tracker by design, not derived from truth."""
        return scan_times(self.scenario)


def scan_times(scenario: ScenarioConfig) -> tuple[datetime, ...]:
    return tuple(EPOCH_UTC + timedelta(seconds=k * scenario.time_step_s) for k in range(scenario.num_steps))


def simulate_truth(scenario: ScenarioConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the target motion on the scan grid. Returns positions and velocities, each (N, 2)."""
    dt = scenario.time_step_s
    n = scenario.num_steps
    pos = np.zeros((n, 2))
    vel = np.zeros((n, 2))
    pos[0] = scenario.initial_position_m
    vel[0] = scenario.initial_velocity_mps
    for k in range(1, n):
        t_prev = (k - 1) * dt
        v = vel[k - 1]
        if scenario.turn_start_s <= t_prev < scenario.turn_end_s:
            angle = scenario.turn_rate_rad_s * dt
            c, s = math.cos(angle), math.sin(angle)
            v_new = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])
        else:
            v_new = v.copy()
        accel = rng.normal(0.0, scenario.random_acceleration_std_mps2, size=2)
        v_new = v_new + accel * dt
        pos[k] = pos[k - 1] + 0.5 * (v + v_new) * dt
        vel[k] = v_new
    return pos, vel


def _record(
    sensor: SensorConfig, measurement_id: str, timestamp: datetime, value: tuple[float, ...]
) -> dict[str, Any]:
    if sensor.measurement_type is MeasurementType.POSITION_2D:
        var = sensor.noise_std**2
        uncertainty: dict[str, Any] = {"kind": "covariance", "matrix": ((var, 0.0), (0.0, var))}
    else:
        uncertainty = {"kind": "angular_std", "std_rad": sensor.noise_std}
    return {
        "measurement_id": measurement_id,
        "sensor_id": sensor.sensor_id,
        "timestamp_utc": timestamp,
        "sensor_position": sensor.position_m,
        "measurement_type": sensor.measurement_type.value,
        "measurement_value": value,
        "measurement_units": EXPECTED_UNITS[sensor.measurement_type],
        "measurement_uncertainty": uncertainty,
        "confidence": None,
    }


def _true_value(sensor: SensorConfig, target_xy: np.ndarray, rng: np.random.Generator) -> tuple[float, ...]:
    if sensor.measurement_type is MeasurementType.POSITION_2D:
        noise = rng.normal(0.0, sensor.noise_std, size=2) + sensor.bias
        return (float(target_xy[0] + noise[0]), float(target_xy[1] + noise[1]))
    dx, dy = target_xy - np.asarray(sensor.position_m)
    bearing = math.atan2(dy, dx) + sensor.bias + rng.normal(0.0, sensor.noise_std)
    return (wrap_angle(bearing),)


def _false_value(sensor: SensorConfig, scenario: ScenarioConfig, rng: np.random.Generator) -> tuple[float, ...]:
    if sensor.measurement_type is MeasurementType.POSITION_2D:
        return (float(rng.uniform(*scenario.surveillance_x_m)), float(rng.uniform(*scenario.surveillance_y_m)))
    return (float(rng.uniform(-math.pi, math.pi)),)


def _corrupt(record: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """Introduce one schema violation, mimicking a malformed sensor message."""
    corrupted = dict(record)
    choice = int(rng.integers(3))
    if choice == 0:
        corrupted["measurement_value"] = tuple(float("nan") for _ in record["measurement_value"])
    elif choice == 1:
        corrupted["timestamp_utc"] = record["timestamp_utc"].replace(tzinfo=None)
    else:
        corrupted["measurement_units"] = ("deg",) * len(record["measurement_units"])
    return corrupted


def simulate(scenario: ScenarioConfig, seed: int) -> SimulationOutput:
    """Run one reproducible simulation. Identical (scenario, seed) gives identical output."""
    root = np.random.SeedSequence(seed)
    truth_seed, primary_seed, secondary_seed, corrupt_seed = root.spawn(4)
    # Independent streams: changing one sensor's settings does not perturb truth or the other sensor.
    truth_rng = np.random.default_rng(truth_seed)
    sensor_rngs = {
        scenario.primary.sensor_id: np.random.default_rng(primary_seed),
        scenario.secondary.sensor_id: np.random.default_rng(secondary_seed),
    }
    corrupt_rng = np.random.default_rng(corrupt_seed)

    positions, velocities = simulate_truth(scenario, truth_rng)
    times = scan_times(scenario)

    records: list[dict[str, Any]] = []
    false_ids: set[str] = set()
    corrupted_ids: set[str] = set()
    counter = 0
    for k, timestamp in enumerate(times):
        for sensor in (scenario.primary, scenario.secondary):
            if k % sensor.scan_interval_steps != 0:
                continue
            rng = sensor_rngs[sensor.sensor_id]
            scan: list[tuple[tuple[float, ...], bool]] = []
            if rng.random() < sensor.detection_probability:
                scan.append((_true_value(sensor, positions[k], rng), False))
            for _ in range(rng.poisson(sensor.false_observations_per_scan)):
                scan.append((_false_value(sensor, scenario, rng), True))
            # Shuffle so the true observation's position in the scan carries no information.
            for idx in rng.permutation(len(scan)):
                value, is_false = scan[idx]
                counter += 1
                mid = f"{sensor.sensor_id}-{counter:06d}"
                record = _record(sensor, mid, timestamp, value)
                if is_false:
                    false_ids.add(mid)
                if corrupt_rng.random() < scenario.corrupted_record_probability:
                    record = _corrupt(record, corrupt_rng)
                    corrupted_ids.add(mid)
                records.append(record)

    truth = GroundTruth(
        timestamps=times,
        positions_m=positions,
        velocities_mps=velocities,
        false_observation_ids=frozenset(false_ids),
        corrupted_record_ids=frozenset(corrupted_ids),
    )
    return SimulationOutput(scenario=scenario, seed=seed, truth=truth, records=records)
