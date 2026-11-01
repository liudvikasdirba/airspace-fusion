import math

import pytest
from pydantic import ValidationError

from airspace_fusion.config import SCENARIOS, with_sensor_overrides

BASE = SCENARIOS["reference"]


def test_overrides_apply_to_both_sensors():
    s = with_sensor_overrides(BASE, 3.0, 0.8)
    assert s.primary.noise_std == pytest.approx(3.0 * BASE.primary.noise_std)
    assert s.secondary.noise_std == pytest.approx(3.0 * BASE.secondary.noise_std)
    assert s.primary.detection_probability == pytest.approx(0.2)
    assert s.secondary.detection_probability == pytest.approx(0.2)
    assert with_sensor_overrides(BASE) == BASE


def test_overrides_keep_other_settings():
    biased = SCENARIOS["biased_bearing"]
    s = with_sensor_overrides(biased, 2.0)
    assert s.secondary.bias == biased.secondary.bias == pytest.approx(math.radians(1.5))
    assert s.secondary.position_m == biased.secondary.position_m
    assert s.primary.detection_probability == biased.primary.detection_probability
    assert s.name == biased.name and s.duration_s == biased.duration_s


@pytest.mark.parametrize(
    "noise, miss, field",
    [
        (0.0, None, "noise_std"),
        (-1.0, None, "noise_std"),
        (1.0, 1.5, "detection_probability"),
        (1.0, -0.1, "detection_probability"),
    ],
)
def test_invalid_overrides_rejected(noise, miss, field):
    with pytest.raises(ValidationError, match=field):
        with_sensor_overrides(BASE, noise, miss)


def test_boundary_overrides_accepted():
    assert with_sensor_overrides(BASE, 1.0, 0.0).primary.detection_probability == 1.0
    assert with_sensor_overrides(BASE, 1.0, 1.0).primary.detection_probability == 0.0
