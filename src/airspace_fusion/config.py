"""Centralised configuration for scenarios and the tracker.

Units: metres, seconds, radians (angles shown in degrees only in the UI).
Everything here is synthetic: sensor parameters do not describe real hardware.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, model_validator

from airspace_fusion.schemas import MeasurementType

# Fixed epoch so that timestamps (and therefore runs) are reproducible.
EPOCH_UTC = datetime(2026, 1, 1, tzinfo=timezone.utc)

PRIMARY_SENSOR_ID = "S1-POS"
SECONDARY_SENSOR_ID = "S2-BRG"


class SensorConfig(BaseModel):
    """A stationary synthetic sensor.

    ``noise_std`` is in metres for POSITION_2D and radians for BEARING_ONLY.
    ``bias`` is an *unmodelled* systematic error in the same units, added by the
    simulator only; the tracker is never told about it. For POSITION_2D it is added
    to each axis, so the offset magnitude is ``bias * sqrt(2)``.
    """

    model_config = ConfigDict(frozen=True)

    sensor_id: str
    position_m: tuple[float, float]
    measurement_type: MeasurementType
    noise_std: float = Field(gt=0.0)
    bias: float = 0.0
    detection_probability: float = Field(default=0.9, ge=0.0, le=1.0)
    false_observations_per_scan: float = Field(default=0.2, ge=0.0)
    scan_interval_steps: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _supported_type(self) -> SensorConfig:
        if self.measurement_type not in (MeasurementType.POSITION_2D, MeasurementType.BEARING_ONLY):
            raise ValueError("the simulator only generates POSITION_2D and BEARING_ONLY sensors")
        return self


class ScenarioConfig(BaseModel):
    """One synthetic object, two stationary sensors, deterministic given a seed."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""
    duration_s: float = Field(default=180.0, gt=0.0)
    time_step_s: float = Field(default=1.0, gt=0.0)

    # Target motion: constant velocity, then a constant-rate turn, then constant velocity.
    initial_position_m: tuple[float, float] = (-1500.0, 800.0)
    initial_velocity_mps: tuple[float, float] = (15.0, 0.0)
    turn_start_s: float = 60.0
    turn_end_s: float = 100.0
    turn_rate_rad_s: float = math.radians(-2.25)  # -90 degrees over 40 s
    random_acceleration_std_mps2: float = Field(default=0.05, ge=0.0)

    primary: SensorConfig
    secondary: SensorConfig

    # Uniform clutter region for false position observations.
    surveillance_x_m: tuple[float, float] = (-4000.0, 4000.0)
    surveillance_y_m: tuple[float, float] = (-4000.0, 4000.0)

    # Fraction of emitted records deliberately corrupted, to exercise schema rejection.
    corrupted_record_probability: float = Field(default=0.01, ge=0.0, le=1.0)

    @property
    def num_steps(self) -> int:
        return int(round(self.duration_s / self.time_step_s)) + 1


class TrackerConfig(BaseModel):
    """Single-target tracker settings, shared by the baseline and fusion modes."""

    model_config = ConfigDict(frozen=True)

    # Continuous white-noise acceleration intensity per axis for the constant-velocity model, m^2/s^3.
    process_noise_m2_s3: float = Field(default=0.5, gt=0.0)
    # Probability mass of the chi-square validation gate.
    gate_probability: float = Field(default=0.997, gt=0.0, lt=1.0)
    # Two-point initiation: reject candidate pairs implying a faster speed.
    max_initiation_speed_mps: float = Field(default=60.0, gt=0.0)
    # Drop the track after this long without an accepted position update.
    max_coast_s: float = Field(default=8.0, gt=0.0)


def with_sensor_overrides(
    base: ScenarioConfig, noise_multiplier: float = 1.0, miss_probability: float | None = None
) -> ScenarioConfig:
    """Scale both sensors' noise and optionally set a common missed-observation probability.

    The overridden sensors are re-validated (``model_copy`` alone would skip validation), so
    out-of-range values raise ``pydantic.ValidationError`` instead of producing a silent bad run.
    """
    sensors = {}
    for key in ("primary", "secondary"):
        sensor: SensorConfig = getattr(base, key)
        update: dict[str, object] = {"noise_std": sensor.noise_std * noise_multiplier}
        if miss_probability is not None:
            update["detection_probability"] = 1.0 - miss_probability
        sensors[key] = SensorConfig.model_validate(sensor.model_dump() | update)
    return base.model_copy(update=sensors)


def default_primary(**overrides: object) -> SensorConfig:
    values: dict[str, object] = dict(
        sensor_id=PRIMARY_SENSOR_ID,
        position_m=(0.0, 0.0),
        measurement_type=MeasurementType.POSITION_2D,
        noise_std=25.0,
    )
    values.update(overrides)
    return SensorConfig(**values)


def default_secondary(**overrides: object) -> SensorConfig:
    values: dict[str, object] = dict(
        sensor_id=SECONDARY_SENSOR_ID,
        position_m=(0.0, -2500.0),
        measurement_type=MeasurementType.BEARING_ONLY,
        noise_std=math.radians(0.5),
    )
    values.update(overrides)
    return SensorConfig(**values)


SCENARIOS: dict[str, ScenarioConfig] = {
    s.name: s
    for s in (
        ScenarioConfig(
            name="reference",
            description="Well-separated sensors, accurate unbiased bearing sensor.",
            primary=default_primary(),
            secondary=default_secondary(),
        ),
        ScenarioConfig(
            name="noisy_primary",
            description="Primary position noise tripled (75 m); bearing sensor unchanged.",
            primary=default_primary(noise_std=75.0),
            secondary=default_secondary(),
        ),
        ScenarioConfig(
            name="distant_bearing",
            description="Bearing sensor 30 km away with 2 deg noise: little cross-range information.",
            primary=default_primary(),
            secondary=default_secondary(position_m=(0.0, -30000.0), noise_std=math.radians(2.0)),
        ),
        ScenarioConfig(
            name="biased_bearing",
            description="Bearing sensor has an unmodelled 1.5 deg bias (a correlated error the filter ignores).",
            primary=default_primary(),
            secondary=default_secondary(bias=math.radians(1.5)),
        ),
    )
}

DEFAULT_SCENARIO = "reference"
