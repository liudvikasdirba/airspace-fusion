"""Shared test scenarios."""

from __future__ import annotations

from airspace_fusion.config import SCENARIOS, ScenarioConfig, TrackerConfig, default_primary, default_secondary

# Simulator per-step random acceleration for the matched-model scenario, m/s^2.
ACCEL_STD_MPS2 = 0.3


def matched_scenario(**overrides: object) -> ScenarioConfig:
    """Reference geometry with a motion model the tracker matches: no turn, no clutter, no misses.

    Pair it with ``matched_tracker()``: q = sigma^2 for the simulator's per-step acceleration sigma (dt = 1 s).
    """
    values: dict[str, object] = dict(
        turn_rate_rad_s=0.0,
        random_acceleration_std_mps2=ACCEL_STD_MPS2,
        corrupted_record_probability=0.0,
        primary=default_primary(detection_probability=1.0, false_observations_per_scan=0.0),
        secondary=default_secondary(detection_probability=1.0, false_observations_per_scan=0.0),
    )
    values.update(overrides)
    return SCENARIOS["reference"].model_copy(update=values)


def matched_tracker(**overrides: object) -> TrackerConfig:
    return TrackerConfig(**({"process_noise_m2_s3": ACCEL_STD_MPS2**2} | overrides))
