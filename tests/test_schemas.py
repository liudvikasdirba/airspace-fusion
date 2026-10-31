import math
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from airspace_fusion.schemas import Measurement, MeasurementType, parse_records, wrap_angle

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def position_record(**overrides):
    record = {
        "measurement_id": "m1",
        "sensor_id": "S1",
        "timestamp_utc": T0,
        "sensor_position": (0.0, 0.0),
        "measurement_type": "POSITION_2D",
        "measurement_value": (100.0, 200.0),
        "measurement_units": ("m", "m"),
        "measurement_uncertainty": {"kind": "covariance", "matrix": ((25.0, 0.0), (0.0, 25.0))},
    }
    record.update(overrides)
    return record


def bearing_record(**overrides):
    record = {
        "measurement_id": "b1",
        "sensor_id": "S2",
        "timestamp_utc": T0,
        "sensor_position": (0.0, -1000.0),
        "measurement_type": "BEARING_ONLY",
        "measurement_value": (0.5,),
        "measurement_units": ("rad",),
        "measurement_uncertainty": {"kind": "angular_std", "std_rad": 0.01},
    }
    record.update(overrides)
    return record


def test_valid_position_and_bearing():
    p = Measurement.model_validate(position_record())
    b = Measurement.model_validate(bearing_record())
    assert p.measurement_type is MeasurementType.POSITION_2D
    assert b.measurement_uncertainty.std_rad == 0.01


def test_naive_timestamp_rejected():
    with pytest.raises(ValidationError, match="timezone-aware"):
        Measurement.model_validate(position_record(timestamp_utc=datetime(2026, 1, 1)))


def test_non_utc_timestamp_rejected():
    tz = timezone(timedelta(hours=2))
    with pytest.raises(ValidationError, match="UTC"):
        Measurement.model_validate(position_record(timestamp_utc=datetime(2026, 1, 1, 2, tzinfo=tz)))


@pytest.mark.parametrize(
    "matrix, message",
    [
        (((25.0, 1.0), (0.0, 25.0)), "symmetric"),
        (((25.0, 0.0), (0.0, -1.0)), "positive definite"),
        (((0.0, 0.0), (0.0, 0.0)), "positive definite"),
        (((25.0, 30.0), (30.0, 25.0)), "positive definite"),
        (((float("nan"), 0.0), (0.0, 1.0)), "non-finite"),
        (((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), "square"),
    ],
)
def test_invalid_covariance_rejected(matrix, message):
    with pytest.raises(ValidationError, match=message):
        Measurement.model_validate(position_record(measurement_uncertainty={"kind": "covariance", "matrix": matrix}))


def test_position_requires_2x2_covariance():
    cov3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    with pytest.raises(ValidationError, match="2x2"):
        Measurement.model_validate(position_record(measurement_uncertainty={"kind": "covariance", "matrix": cov3}))


def test_bearing_rejects_cartesian_covariance():
    cov = {"kind": "covariance", "matrix": ((1.0, 0.0), (0.0, 1.0))}
    with pytest.raises(ValidationError, match="angular"):
        Measurement.model_validate(bearing_record(measurement_uncertainty=cov))


def test_position_rejects_angular_uncertainty():
    with pytest.raises(ValidationError, match="covariance"):
        Measurement.model_validate(position_record(measurement_uncertainty={"kind": "angular_std", "std_rad": 0.1}))


@pytest.mark.parametrize("std", [0.0, -0.1, math.pi, float("inf")])
def test_invalid_angular_std(std):
    with pytest.raises(ValidationError):
        Measurement.model_validate(bearing_record(measurement_uncertainty={"kind": "angular_std", "std_rad": std}))


def test_wrong_units_and_dimensions_rejected():
    with pytest.raises(ValidationError, match="units"):
        Measurement.model_validate(bearing_record(measurement_units=("deg",)))
    with pytest.raises(ValidationError, match="requires 2 values"):
        Measurement.model_validate(position_record(measurement_value=(1.0,)))
    with pytest.raises(ValidationError, match="non-finite"):
        Measurement.model_validate(position_record(measurement_value=(1.0, float("inf"))))


def test_bearing_wraparound_normalised():
    m = Measurement.model_validate(bearing_record(measurement_value=(3 * math.pi / 2,)))
    assert m.measurement_value[0] == pytest.approx(-math.pi / 2)
    m = Measurement.model_validate(bearing_record(measurement_value=(math.pi,)))
    assert m.measurement_value[0] == pytest.approx(-math.pi)


@pytest.mark.parametrize("angle", [0.0, 1.0, -1.0, math.pi, -math.pi, 7.0, -7.0, 100.0])
def test_wrap_angle_range_and_equivalence(angle):
    wrapped = wrap_angle(angle)
    assert -math.pi <= wrapped < math.pi
    assert math.cos(wrapped) == pytest.approx(math.cos(angle))
    assert math.sin(wrapped) == pytest.approx(math.sin(angle), abs=1e-12)


def test_range_bearing_schema():
    rb = {
        **position_record(),
        "measurement_type": "RANGE_BEARING",
        "measurement_value": (500.0, 4.0),
        "measurement_units": ("m", "rad"),
    }
    m = Measurement.model_validate(rb)
    assert m.measurement_value[1] == pytest.approx(4.0 - 2 * math.pi)
    with pytest.raises(ValidationError, match="non-negative"):
        Measurement.model_validate({**rb, "measurement_value": (-1.0, 0.0)})


def test_detection_only_has_no_value_or_uncertainty():
    det = {
        **position_record(),
        "measurement_type": "DETECTION_ONLY",
        "measurement_value": (),
        "measurement_units": (),
        "measurement_uncertainty": None,
        "confidence": 0.7,
    }
    m = Measurement.model_validate(det)
    assert m.measurement_value == ()
    with pytest.raises(ValidationError, match="requires 0 values"):
        Measurement.model_validate({**det, "measurement_value": (1.0, 2.0), "measurement_units": ()})
    with pytest.raises(ValidationError, match="must not carry an uncertainty"):
        Measurement.model_validate({**det, "measurement_uncertainty": {"kind": "angular_std", "std_rad": 0.1}})


def test_confidence_and_class_probabilities():
    ok = Measurement.model_validate(
        position_record(confidence=0.8, simulated_class_probabilities={"a": 0.25, "b": 0.75})
    )
    assert ok.simulated_class_probabilities == {"a": 0.25, "b": 0.75}
    with pytest.raises(ValidationError):
        Measurement.model_validate(position_record(confidence=1.5))
    with pytest.raises(ValidationError, match="sum to 1"):
        Measurement.model_validate(position_record(simulated_class_probabilities={"a": 0.5, "b": 0.6}))
    with pytest.raises(ValidationError, match=r"\[0, 1\]"):
        Measurement.model_validate(position_record(simulated_class_probabilities={"a": 1.5, "b": -0.5}))


def test_unknown_fields_and_types_rejected():
    with pytest.raises(ValidationError):
        Measurement.model_validate(position_record(true_position=(1.0, 2.0)))
    with pytest.raises(ValidationError):
        Measurement.model_validate(position_record(measurement_type="ACOUSTIC"))


def test_parse_records_reports_errors():
    records = [position_record(), bearing_record(measurement_units=("deg",)), {"measurement_id": "x"}]
    valid, rejected = parse_records(records)
    assert [m.measurement_id for m in valid] == ["m1"]
    assert [r.measurement_id for r in rejected] == ["b1", "x"]
    assert [r.sensor_id for r in rejected] == ["S2", None]
    assert all(r.errors for r in rejected)
    assert parse_records([]) == ([], [])
