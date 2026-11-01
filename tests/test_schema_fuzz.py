"""Property-based tests for the measurement schema (L1: S1-S5)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from airspace_fusion.schemas import Measurement, MeasurementType, parse_records, wrap_angle

UTC = timezone.utc
FIELDS = list(Measurement.model_fields)

finite = st.floats(allow_nan=False, allow_infinity=False, min_value=-1e7, max_value=1e7)
any_float = st.floats(allow_nan=True, allow_infinity=True)
junk_scalar = st.one_of(st.none(), st.booleans(), st.integers(), any_float, st.text(max_size=8))
junk_value = st.recursive(
    junk_scalar,
    lambda inner: st.one_of(st.lists(inner, max_size=4), st.dictionaries(st.text(max_size=6), inner, max_size=3)),
    max_leaves=10,
)
aware_times = st.datetimes(
    min_value=datetime(2000, 1, 1), max_value=datetime(2100, 1, 1), timezones=st.just(UTC)
)


# -- S1: junk never escapes as anything other than a rejected record ---------------------------

@given(
    st.lists(
        st.one_of(
            st.dictionaries(st.sampled_from(FIELDS + ["extra"]), junk_value, max_size=len(FIELDS) + 1),
            junk_value,
        ),
        max_size=5,
    )
)
def test_parse_records_never_raises(records):
    valid, rejected = parse_records(records)
    assert len(valid) + len(rejected) == len(records)
    assert all(r.errors for r in rejected)


# -- S2: valid measurements round-trip -------------------------------------------------------

def spd_matrix(scale: float, a: float, b: float, c: float) -> tuple[tuple[float, ...], ...]:
    m = np.array([[a, 0.0], [b, c]]) * scale
    cov = m @ m.T + np.eye(2) * scale**2 * 1e-3
    cov = (cov + cov.T) / 2.0
    return tuple(tuple(float(v) for v in row) for row in cov)


@st.composite
def valid_records(draw) -> dict:
    mtype = draw(st.sampled_from(list(MeasurementType)))
    base = {
        "measurement_id": draw(st.text(min_size=1, max_size=10)),
        "sensor_id": draw(st.text(min_size=1, max_size=10)),
        "timestamp_utc": draw(aware_times),
        "sensor_position": (draw(finite), draw(finite)),
        "measurement_type": mtype.value,
        "confidence": draw(st.one_of(st.none(), st.floats(0.0, 1.0))),
    }
    unit = st.floats(0.1, 10.0)
    cov = {"kind": "covariance", "matrix": spd_matrix(draw(st.floats(0.01, 1e3)), draw(unit), draw(unit), draw(unit))}
    if mtype is MeasurementType.POSITION_2D:
        base.update(measurement_value=(draw(finite), draw(finite)), measurement_units=("m", "m"),
                    measurement_uncertainty=cov)
    elif mtype is MeasurementType.RANGE_BEARING:
        base.update(measurement_value=(draw(st.floats(0.0, 1e6)), draw(finite)), measurement_units=("m", "rad"),
                    measurement_uncertainty=cov)
    elif mtype is MeasurementType.BEARING_ONLY:
        base.update(measurement_value=(draw(finite),), measurement_units=("rad",),
                    measurement_uncertainty={"kind": "angular_std", "std_rad": draw(st.floats(1e-6, 3.0))})
    else:
        base.update(measurement_value=(), measurement_units=(), measurement_uncertainty=None)
    return base


@given(valid_records())
def test_valid_records_accepted_and_round_trip(record):
    m = Measurement.model_validate(record)
    again = Measurement.model_validate(m.model_dump())
    assert again == m
    if m.measurement_type in (MeasurementType.BEARING_ONLY, MeasurementType.RANGE_BEARING):
        bearing = m.measurement_value[-1]
        assert -math.pi <= bearing < math.pi


# -- S3: angle wrapping ----------------------------------------------------------------------

@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e9, max_value=1e9))
def test_wrap_angle_properties(angle):
    wrapped = wrap_angle(angle)
    assert -math.pi <= wrapped < math.pi
    assert wrap_angle(wrapped) == wrapped
    # Precision of the reduction degrades with |angle|; compare on the unit circle with a scaled tolerance.
    tol = 1e-12 * max(1.0, abs(angle))
    assert math.cos(wrapped) == pytest.approx(math.cos(angle), abs=tol)
    assert math.sin(wrapped) == pytest.approx(math.sin(angle), abs=tol)


@pytest.mark.parametrize(
    "angle",
    [-math.pi, math.pi, math.nextafter(-math.pi, -4.0), math.nextafter(math.pi, 4.0),
     math.nextafter(-math.pi, 0.0), math.nextafter(math.pi, 0.0), -1e-17, 1e-17, 3 * math.pi, -3 * math.pi],
)
def test_wrap_angle_boundaries(angle):
    wrapped = wrap_angle(angle)
    assert -math.pi <= wrapped < math.pi
    assert wrap_angle(wrapped) == wrapped


# -- S4: covariance validation ----------------------------------------------------------------

def position_with(matrix) -> dict:
    return {
        "measurement_id": "m",
        "sensor_id": "s",
        "timestamp_utc": datetime(2026, 1, 1, tzinfo=UTC),
        "sensor_position": (0.0, 0.0),
        "measurement_type": "POSITION_2D",
        "measurement_value": (1.0, 2.0),
        "measurement_units": ("m", "m"),
        "measurement_uncertainty": {"kind": "covariance", "matrix": matrix},
    }


eigen = st.floats(1e-4, 1e8)
rotation = st.floats(0.0, math.pi)


def rotated(l1: float, l2: float, theta: float) -> np.ndarray:
    r = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    return r @ np.diag([l1, l2]) @ r.T


def as_tuple(m: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(v) for v in row) for row in m)


@given(eigen, eigen, rotation)
def test_spd_covariances_accepted(l1, l2, theta):
    # Products like R diag R^T are symmetric only to rounding; that must not cause rejection.
    Measurement.model_validate(position_with(as_tuple(rotated(l1, l2, theta))))


@given(eigen, st.floats(1e-3, 1e8), rotation)
def test_indefinite_covariances_rejected(l1, negative, theta):
    cov = rotated(l1, -negative, theta)
    cov = (cov + cov.T) / 2.0
    with pytest.raises(ValidationError, match="positive definite"):
        Measurement.model_validate(position_with(as_tuple(cov)))


@given(eigen, eigen, rotation, st.floats(1e-6, 1e-2))
def test_materially_asymmetric_covariances_rejected(l1, l2, theta, relative_skew):
    cov = rotated(l1, l2, theta)
    cov = (cov + cov.T) / 2.0
    cov[0, 1] += relative_skew * max(l1, l2)
    with pytest.raises(ValidationError, match="symmetric"):
        Measurement.model_validate(position_with(as_tuple(cov)))


# -- S5: timestamps -------------------------------------------------------------------------

@given(st.datetimes(min_value=datetime(2000, 1, 1), max_value=datetime(2100, 1, 1),
                    timezones=st.one_of(st.none(), st.timezones())))
@settings(max_examples=200)
def test_only_utc_timestamps_accepted(ts):
    record = position_with(((1.0, 0.0), (0.0, 1.0))) | {"timestamp_utc": ts}
    offset = ts.utcoffset()
    if offset == timedelta(0):
        assert Measurement.model_validate(record).timestamp_utc == ts
    else:
        expected = "timezone-aware" if offset is None else "UTC"
        with pytest.raises(ValidationError, match=expected):
            Measurement.model_validate(record)


@given(st.integers(-14 * 60, 14 * 60).filter(lambda m: m != 0))
def test_fixed_offsets_rejected(minutes):
    ts = datetime(2026, 1, 1, tzinfo=timezone(timedelta(minutes=minutes)))
    with pytest.raises(ValidationError, match="UTC"):
        Measurement.model_validate(position_with(((1.0, 0.0), (0.0, 1.0))) | {"timestamp_utc": ts})
