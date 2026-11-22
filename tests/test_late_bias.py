"""A bias that starts partway through a run: the sensor setting, the late_bias family and its metrics."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import timedelta

import pytest
from pydantic import ValidationError

from airspace_fusion.cli import main
from airspace_fusion.config import SCENARIOS, HealthMonitorConfig, SensorConfig, TrackerConfig
from airspace_fusion.evaluation import Mode, run_single, sensor_health_metrics
from airspace_fusion.health import HealthReport, HealthTransition, SensorHealth
from airspace_fusion.scenarios import FAMILIES, SWEEPABLE_PARAMETERS, ScenarioFamily
from airspace_fusion.schemas import parse_records
from airspace_fusion.simulator import simulate

BIASED = SCENARIOS["biased_bearing"]
LATE = FAMILIES["late_bias"]
MONITORED = TrackerConfig(health_monitor=HealthMonitorConfig())


def bearings(scenario, seed=0):
    """Bearing values of the true detections, by time (s), with the false observations left out."""
    sim = simulate(scenario, seed)
    t0 = sim.truth.timestamps[0]
    measurements, _ = parse_records(sim.records)
    return {
        (m.timestamp_utc - t0).total_seconds(): m.measurement_value[0]
        for m in measurements
        if m.sensor_id == scenario.secondary.sensor_id and m.measurement_id not in sim.truth.false_observation_ids
    }


def with_bias_start(scenario, start_s):
    return scenario.model_copy(update={"secondary": scenario.secondary.model_copy(update={"bias_start_s": start_s})})


def test_bias_applies_only_from_its_start():
    unbiased = bearings(BIASED.model_copy(update={"secondary": BIASED.secondary.model_copy(update={"bias": 0.0})}))
    always = bearings(BIASED)
    late = bearings(with_bias_start(BIASED, 60.0))
    assert late.keys() == always.keys() == unbiased.keys()  # the same random draws either way
    for t, value in late.items():
        assert value == pytest.approx(always[t] if t >= 60.0 else unbiased[t], abs=1e-12), t


def test_bias_start_is_left_out_of_the_serialised_config_while_zero():
    """So configurations and hashes from before the setting existed are unchanged."""
    assert "bias_start_s" not in BIASED.secondary.model_dump()
    late = BIASED.secondary.model_copy(update={"bias_start_s": 45.0})
    assert late.model_dump()["bias_start_s"] == 45.0
    assert SensorConfig.model_validate(late.model_dump()) == late
    assert SensorConfig.model_validate(BIASED.secondary.model_dump()).bias_start_s == 0.0
    with pytest.raises(ValidationError):
        BIASED.secondary.model_copy(update={"bias_start_s": -1.0}).model_validate(
            BIASED.secondary.model_dump() | {"bias_start_s": -1.0})


def test_late_bias_family_draws_the_onset_and_keeps_every_other_draw():
    assert len(FAMILIES) == 6
    without_onset = LATE.model_copy(update={"bias_start_s": None})
    for seed in range(20):
        world = LATE.for_seed(seed)
        assert 30.0 <= world.secondary.bias_start_s <= 150.0
        assert world.secondary.bias == pytest.approx(math.radians(1.5))
        assert f"bias from {world.secondary.bias_start_s:.0f} s" in world.description
        plain = without_onset.for_seed(seed)  # the onset is drawn last: nothing else moves
        assert plain.secondary.bias_start_s == 0.0
        assert world.model_copy(update={"secondary": plain.secondary, "description": plain.description}) == plain
    onsets = {round(LATE.for_seed(seed).secondary.bias_start_s) for seed in range(50)}
    assert len(onsets) > 20  # spread over the range, not one value


def test_bias_start_is_sweepable_and_validated():
    assert "bias_start_s" in SWEEPABLE_PARAMETERS
    fixed = LATE.with_fixed("bias_start_s", 90.0)
    assert {fixed.for_seed(s).secondary.bias_start_s for s in range(5)} == {90.0}
    with pytest.raises(ValidationError, match="bias_start_s must not be negative"):
        ScenarioFamily.model_validate(LATE.model_dump() | {"bias_start_s": {"lo": -5.0, "hi": 10.0}})


def with_transitions(scenario, transitions):
    run = run_single(scenario, MONITORED, 0)
    fusion = run.modes[Mode.FUSION]
    report = HealthReport(tuple(transitions), {}, {}, {})
    modes = dict(run.modes)
    modes[Mode.FUSION] = replace(fusion, result=replace(fusion.result, health=report))
    return replace(run, modes=modes)


def test_detection_delay_counts_from_the_onset_and_earlier_alarms_are_false():
    scenario = with_bias_start(BIASED, 80.0)
    sensor = scenario.secondary.sensor_id
    t0 = simulate(scenario, 0).truth.timestamps[0]

    def at(s):
        return t0 + timedelta(seconds=s)

    run = with_transitions(scenario, [
        HealthTransition(at(20), sensor, SensorHealth.TRUSTED, SensorHealth.SUSPECT, "cusum"),  # before the onset
        HealthTransition(at(50), sensor, SensorHealth.SUSPECT, SensorHealth.TRUSTED, "consistent"),
        HealthTransition(at(93), sensor, SensorHealth.TRUSTED, SensorHealth.SUSPECT, "cusum"),
        HealthTransition(at(97), sensor, SensorHealth.SUSPECT, SensorHealth.EXCLUDED, "cusum"),
    ])
    metrics = sensor_health_metrics(run)[sensor]
    assert metrics.biased
    assert metrics.detection_delay_s == 13.0
    assert metrics.false_alarms == 1


def test_a_bias_from_the_start_is_measured_as_before():
    sensor = BIASED.secondary.sensor_id
    t0 = simulate(BIASED, 0).truth.timestamps[0]
    run = with_transitions(BIASED, [
        HealthTransition(t0 + timedelta(seconds=9), sensor, SensorHealth.TRUSTED, SensorHealth.SUSPECT, "cusum"),
    ])
    metrics = sensor_health_metrics(run)[sensor]
    assert (metrics.detection_delay_s, metrics.false_alarms) == (9.0, 0)


def test_the_monitor_catches_a_late_bias_in_a_real_run():
    scenario = with_bias_start(BIASED, 90.0)
    metrics = sensor_health_metrics(run_single(scenario, MONITORED, 0))[scenario.secondary.sensor_id]
    assert 0.0 <= metrics.detection_delay_s <= 60.0
    assert metrics.false_alarms == 0


def test_families_group_lists_six(capsys):
    assert main(["scenarios"]) == 0
    out = capsys.readouterr().out
    assert "late_bias" in out


@pytest.mark.slow
@pytest.mark.parametrize("motion_model", ["cv", "imm"])
def test_late_bias_is_detected_like_a_bias_from_the_start(motion_model):
    """README "A bias that starts later": on seeds 0-199, all detected, median 9.4 s (CV) / 8.9 s (IMM) from the onset,
    no false alarms. Checked here on 60 seeds against the M5 bars (median at most 30 s, false alarms at most 1 %)."""
    from airspace_fusion.evaluation import health_summary

    tracker = TrackerConfig(motion_model=motion_model, health_monitor=HealthMonitorConfig())
    s = health_summary(LATE, tracker, range(60))
    assert s.detection_rate >= 0.95, s
    assert s.median_delay_s <= 30.0, s
    assert s.runs_with_false_alarm <= 1, s
