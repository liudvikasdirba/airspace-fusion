"""Validated schema for synthetic sensor observations.

All positions are in a shared local 2D Cartesian frame, in metres.
Angles are in radians, measured counter-clockwise from the +x axis
(``atan2(dy, dx)``), and normalised to the interval [-pi, pi).

Each measurement type carries an uncertainty representation that matches
its measurement model:

* ``POSITION_2D``    value ``[x_m, y_m]``            -> 2x2 covariance in m^2
* ``RANGE_BEARING``  value ``[range_m, bearing_rad]`` -> 2x2 covariance in (m, rad) units
* ``BEARING_ONLY``   value ``[bearing_rad]``          -> angular standard deviation in rad
* ``DETECTION_ONLY`` value ``[]``                     -> no measurement uncertainty

A bearing-only or detection-only observation never carries a Cartesian
position or covariance: none is invented here or downstream.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from enum import Enum
from typing import Annotated, Any, Literal, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

PROBABILITY_SUM_TOLERANCE = 1e-6
SYMMETRY_TOLERANCE = 1e-9


class MeasurementType(str, Enum):
    POSITION_2D = "POSITION_2D"
    RANGE_BEARING = "RANGE_BEARING"
    BEARING_ONLY = "BEARING_ONLY"
    DETECTION_ONLY = "DETECTION_ONLY"


EXPECTED_UNITS: dict[MeasurementType, tuple[str, ...]] = {
    MeasurementType.POSITION_2D: ("m", "m"),
    MeasurementType.RANGE_BEARING: ("m", "rad"),
    MeasurementType.BEARING_ONLY: ("rad",),
    MeasurementType.DETECTION_ONLY: (),
}


def wrap_angle(angle_rad: float) -> float:
    """Normalise an angle to [-pi, pi)."""
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


class CovarianceUncertainty(BaseModel):
    """Gaussian covariance in the units of the measurement vector (squared)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["covariance"] = "covariance"
    matrix: tuple[tuple[float, ...], ...]

    @field_validator("matrix")
    @classmethod
    def _valid_covariance(cls, matrix: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
        array = np.asarray(matrix, dtype=float)
        if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] == 0:
            raise ValueError(f"covariance must be a non-empty square matrix, got shape {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError("covariance contains non-finite values")
        if not np.allclose(array, array.T, atol=SYMMETRY_TOLERANCE, rtol=0.0):
            raise ValueError("covariance must be symmetric")
        try:
            np.linalg.cholesky(array)
        except np.linalg.LinAlgError as exc:
            raise ValueError("covariance must be positive definite") from exc
        return matrix

    def as_array(self) -> np.ndarray:
        return np.asarray(self.matrix, dtype=float)


class AngularUncertainty(BaseModel):
    """One-sigma angular uncertainty of a direction measurement, in radians."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["angular_std"] = "angular_std"
    std_rad: float = Field(gt=0.0, lt=math.pi)

    @field_validator("std_rad")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("angular standard deviation must be finite")
        return value


Uncertainty = Annotated[Union[CovarianceUncertainty, AngularUncertainty], Field(discriminator="kind")]


class Measurement(BaseModel):
    """A single synthetic sensor observation, as seen by the tracking components."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    measurement_id: str = Field(min_length=1)
    sensor_id: str = Field(min_length=1)
    timestamp_utc: datetime
    sensor_position: tuple[float, float] = Field(description="Sensor location in the local frame, metres")
    measurement_type: MeasurementType
    measurement_value: tuple[float, ...]
    measurement_units: tuple[str, ...]
    measurement_uncertainty: Uncertainty | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    simulated_class_probabilities: dict[str, float] | None = Field(
        default=None,
        description=(
            "Simulated classification labels. Not a tracking measurement and "
            "ignored by the state estimator."
        ),
    )

    @field_validator("timestamp_utc")
    @classmethod
    def _timezone_aware_utc(cls, value: datetime) -> datetime:
        offset = value.utcoffset()
        if offset is None:
            raise ValueError("timestamp_utc must be timezone-aware")
        if offset != timedelta(0):
            raise ValueError(f"timestamp_utc must be in UTC, got offset {offset}")
        return value

    @field_validator("sensor_position")
    @classmethod
    def _finite_position(cls, value: tuple[float, float]) -> tuple[float, float]:
        if not all(math.isfinite(v) for v in value):
            raise ValueError("sensor_position must be finite")
        return value

    @field_validator("measurement_value")
    @classmethod
    def _finite_value(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not all(math.isfinite(v) for v in value):
            raise ValueError("measurement_value contains non-finite values")
        return value

    @field_validator("simulated_class_probabilities")
    @classmethod
    def _valid_probabilities(cls, value: dict[str, float] | None) -> dict[str, float] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("simulated_class_probabilities must not be empty when provided")
        for label, probability in value.items():
            if not label:
                raise ValueError("class labels must be non-empty strings")
            if not (math.isfinite(probability) and 0.0 <= probability <= 1.0):
                raise ValueError(f"probability for class {label!r} must be in [0, 1], got {probability}")
        total = sum(value.values())
        if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
            raise ValueError(f"class probabilities must sum to 1, got {total}")
        return value

    @model_validator(mode="after")
    def _consistent_with_type(self) -> Measurement:
        mtype = self.measurement_type
        expected_units = EXPECTED_UNITS[mtype]
        if self.measurement_units != expected_units:
            raise ValueError(f"{mtype.value} requires units {expected_units}, got {self.measurement_units}")
        if len(self.measurement_value) != len(expected_units):
            raise ValueError(
                f"{mtype.value} requires {len(expected_units)} values, got {len(self.measurement_value)}"
            )

        uncertainty = self.measurement_uncertainty
        if mtype in (MeasurementType.POSITION_2D, MeasurementType.RANGE_BEARING):
            if not isinstance(uncertainty, CovarianceUncertainty):
                raise ValueError(f"{mtype.value} requires a covariance uncertainty")
            if uncertainty.as_array().shape != (2, 2):
                raise ValueError(f"{mtype.value} requires a 2x2 covariance")
        elif mtype is MeasurementType.BEARING_ONLY:
            if not isinstance(uncertainty, AngularUncertainty):
                raise ValueError("BEARING_ONLY requires an angular (radian) uncertainty, not a covariance")
        elif mtype is MeasurementType.DETECTION_ONLY:
            if uncertainty is not None:
                raise ValueError("DETECTION_ONLY carries no measurement value and must not carry an uncertainty")

        if mtype is MeasurementType.RANGE_BEARING and self.measurement_value[0] < 0.0:
            raise ValueError("range must be non-negative")

        # Normalise bearings so downstream code sees one consistent angle convention.
        if mtype is MeasurementType.BEARING_ONLY:
            object.__setattr__(self, "measurement_value", (wrap_angle(self.measurement_value[0]),))
        elif mtype is MeasurementType.RANGE_BEARING:
            r, b = self.measurement_value
            object.__setattr__(self, "measurement_value", (r, wrap_angle(b)))
        return self


class RejectedRecord(BaseModel):
    """A raw record that failed schema validation, with the reason."""

    model_config = ConfigDict(frozen=True)

    measurement_id: str | None
    errors: tuple[str, ...]


def parse_records(records: list[dict[str, Any]]) -> tuple[list[Measurement], list[RejectedRecord]]:
    """Validate raw sensor records. Invalid records are returned with their errors, never dropped silently."""
    valid: list[Measurement] = []
    rejected: list[RejectedRecord] = []
    for record in records:
        try:
            valid.append(Measurement.model_validate(record))
        except ValidationError as exc:
            errors = tuple(f"{'.'.join(str(p) for p in e['loc']) or '<record>'}: {e['msg']}" for e in exc.errors())
            mid = record.get("measurement_id") if isinstance(record, dict) else None
            rejected.append(RejectedRecord(measurement_id=mid if isinstance(mid, str) else None, errors=errors))
    return valid, rejected
