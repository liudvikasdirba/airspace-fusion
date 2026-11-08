"""Sensor health monitor: the tests and state machine, its place in the tracker, and its metrics (ADR 0012)."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from scipy import stats

from airspace_fusion.cli import main
from airspace_fusion.config import SCENARIOS, HealthMonitorConfig, TrackerConfig, default_primary
from airspace_fusion.estimation import Disposition, Innovation, down_weighted
from airspace_fusion.estimation_stonesoup import StoneSoupTracker
from airspace_fusion.evaluation import (
    Mode,
    health_roc,
    health_summary,
    run_single,
    scaled_thresholds,
    sensor_health_metrics,
)
from airspace_fusion.health import (
    HealthMonitor,
    HealthReport,
    HealthTransition,
    SensorHealth,
    SensorMonitor,
    chi2_upper_quantile,
)
from airspace_fusion.results import build_record, config_hash
from airspace_fusion.scenarios import FAMILIES
from airspace_fusion.schemas import parse_records
from airspace_fusion.simulator import simulate

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
MONITOR = HealthMonitorConfig()
NO_PROBATION = HealthMonitorConfig(probation_samples=0)
MONITORED = TrackerConfig(health_monitor=MONITOR)


def feed(monitor: SensorMonitor, samples, protected: bool = False) -> list[SensorHealth]:
    states = []
    for value in samples:
        monitor.observe((float(value),), protected)
        states.append(monitor.state)
    return states


@pytest.mark.parametrize("dof", [20, 40])
@pytest.mark.parametrize("tail", [1e-2, 1e-4, 1e-6])
def test_chi2_upper_quantile_is_close_to_exact(dof, tail):
    assert chi2_upper_quantile(tail, dof) == pytest.approx(stats.chi2.isf(tail, dof), rel=0.02)


def test_chi2_upper_quantile_rejects_bad_arguments():
    with pytest.raises(ValueError):
        chi2_upper_quantile(0.0, 20)
    with pytest.raises(ValueError):
        chi2_upper_quantile(0.5, 0)


def test_config_rejects_limits_in_the_wrong_order():
    with pytest.raises(ValueError):
        HealthMonitorConfig(suspect_cusum=10.0, exclude_cusum=5.0)
    with pytest.raises(ValueError):
        HealthMonitorConfig(suspect_window_probability=1e-9, exclude_window_probability=1e-6)


def test_consistent_innovations_raise_no_alarm():
    rng = np.random.default_rng(1)
    for _ in range(20):
        monitor = SensorMonitor("s", NO_PROBATION, 1)
        assert set(feed(monitor, rng.normal(size=200))) == {SensorHealth.TRUSTED}


def test_a_persistent_shift_goes_through_suspect_to_excluded_by_cusum():
    monitor = SensorMonitor("s", NO_PROBATION, 1)
    states = feed(monitor, [2.5] * 40)
    assert states.index(SensorHealth.SUSPECT) < states.index(SensorHealth.EXCLUDED)
    assert states[-1] is SensorHealth.EXCLUDED


def test_negative_shifts_count_too():
    monitor = SensorMonitor("s", NO_PROBATION, 1)
    assert feed(monitor, [-2.5] * 40)[-1] is SensorHealth.EXCLUDED


def test_inflated_noise_without_a_shift_is_caught_by_the_windowed_test():
    # Alternating signs keep the CUSUM near zero; only the sum of squares can see this.
    monitor = SensorMonitor("s", NO_PROBATION, 1)
    feed(monitor, [3.0, -3.0] * 15)
    assert monitor.state is not SensorHealth.TRUSTED
    assert monitor.cusum < NO_PROBATION.suspect_cusum


def test_excluded_sensor_stays_out_while_shifted_and_returns_through_suspect():
    rng = np.random.default_rng(2)
    monitor = SensorMonitor("s", NO_PROBATION, 1)
    assert feed(monitor, [2.5] * 40)[-1] is SensorHealth.EXCLUDED
    assert set(feed(monitor, 2.5 + rng.normal(size=100))) == {SensorHealth.EXCLUDED}
    states = feed(monitor, rng.normal(size=120))
    assert SensorHealth.SUSPECT in states
    assert states.index(SensorHealth.SUSPECT) > 0  # never straight back to trusted
    assert states[-1] is SensorHealth.TRUSTED


def test_no_test_fires_before_min_samples():
    monitor = SensorMonitor("s", HealthMonitorConfig(probation_samples=0, min_samples=30), 1)
    assert set(feed(monitor, [3.0] * 29)) == {SensorHealth.TRUSTED}


def test_a_protected_sensor_stays_trusted_and_its_alarms_are_counted():
    monitor = SensorMonitor("s", NO_PROBATION, 1)
    assert set(feed(monitor, [3.0] * 40, protected=True)) == {SensorHealth.TRUSTED}
    assert monitor.held_alarms > 0


def test_the_last_position_sensor_is_protected_but_a_second_one_is_not():
    bad = (3.0, 3.0)
    alone = HealthMonitor(NO_PROBATION)
    for k in range(40):
        assert alone.observe(T0 + timedelta(seconds=k), "p1", bad, True) is SensorHealth.TRUSTED

    pair = HealthMonitor(NO_PROBATION)
    for k in range(40):
        t = T0 + timedelta(seconds=k)
        pair.observe(t, "p1", bad, True)
        pair.observe(t, "p2", (0.1, -0.1), True)
    assert pair.state("p1") is SensorHealth.EXCLUDED
    assert pair.state("p2") is SensorHealth.TRUSTED
    for k in range(40, 80):  # p2 is now the last usable position sensor: it cannot follow p1 out
        assert pair.observe(T0 + timedelta(seconds=k), "p2", bad, True) is SensorHealth.TRUSTED


def test_probation_tests_a_sensor_before_it_is_used():
    monitor = HealthMonitor(HealthMonitorConfig(probation_samples=3))
    states = [monitor.observe(T0 + timedelta(seconds=k), "b", (0.1,), False) for k in range(5)]
    assert states[:2] == [SensorHealth.EXCLUDED, SensorHealth.EXCLUDED]
    assert states[3:] == [SensorHealth.TRUSTED, SensorHealth.TRUSTED]
    assert monitor.report().transitions == ()  # being on probation is not an alarm


def test_probation_continues_while_the_samples_look_shifted_but_not_past_one_window():
    # A shift too small to raise an alarm must not keep the sensor out for ever without anything being reported.
    config = HealthMonitorConfig(probation_samples=3, window_samples=20)
    monitor = HealthMonitor(config)
    states = [monitor.observe(T0 + timedelta(seconds=k), "b", (1.2,), False) for k in range(21)]
    assert states[10] is SensorHealth.EXCLUDED
    assert states[20] is SensorHealth.TRUSTED
    assert monitor.report().transitions == ()


def test_normalised_innovation_has_identity_covariance_and_keeps_the_nis():
    s = np.array([[900.0, 200.0], [200.0, 400.0]])
    inverse = np.linalg.inv(s)
    rng = np.random.default_rng(3)
    samples = rng.multivariate_normal([0.0, 0.0], s, size=20000)
    whitened = []
    for vector in samples:
        innovation = Innovation(vector, np.eye(2, 4), np.eye(2), np.zeros((4, 2)), inverse)
        z = innovation.normalised()
        assert z[0] ** 2 + z[1] ** 2 == pytest.approx(innovation.nis)
        whitened.append(z)
    assert np.cov(np.array(whitened).T) == pytest.approx(np.eye(2), abs=0.05)


def test_normalised_bearing_innovation():
    innovation = Innovation(np.array([0.02]), np.zeros((1, 4)), np.eye(1), np.zeros((4, 1)), np.array([[2500.0]]))
    assert innovation.normalised() == pytest.approx((1.0,))


def test_down_weighting_scales_the_variance_of_both_measurement_types():
    measurements, _ = parse_records(simulate(SCENARIOS["reference"], 0).records)
    by_id = {"pos": next(m for m in measurements if len(m.measurement_value) == 2),
             "brg": next(m for m in measurements if len(m.measurement_value) == 1)}
    assert down_weighted(by_id["pos"], 4.0).covariance() == pytest.approx(by_id["pos"].covariance() * 4.0)
    assert down_weighted(by_id["brg"], 4.0).angular_std() == pytest.approx(by_id["brg"].angular_std() * 2.0)
    assert down_weighted(by_id["pos"], 4.0).measurement_value == by_id["pos"].measurement_value


# -- in the tracker ------------------------------------------------------------------------------------------------


def test_monitor_never_touches_the_baseline():
    plain = run_single(SCENARIOS["biased_bearing"], TrackerConfig(), 0).modes[Mode.BASELINE]
    monitored = run_single(SCENARIOS["biased_bearing"], MONITORED, 0).modes[Mode.BASELINE]
    assert monitored.result.health is None
    assert monitored.metrics.position_rmse_m == plain.metrics.position_rmse_m


@pytest.mark.parametrize("motion_model", ["cv", "imm"])
def test_biased_bearing_is_excluded_and_the_track_survives_on_positions(motion_model):
    """ADR 0004 and 0012: excluding bearings removes nothing that keeps a track alive."""
    tracker = TrackerConfig(motion_model=motion_model, health_monitor=MONITOR)
    run = run_single(SCENARIOS["biased_bearing"], tracker, 1)
    fusion, baseline = run.modes[Mode.FUSION], run.modes[Mode.BASELINE]
    assert fusion.result.health.final_states[SCENARIOS["biased_bearing"].secondary.sensor_id] is SensorHealth.EXCLUDED
    assert Disposition.SENSOR_EXCLUDED in fusion.result.dispositions.values()
    assert fusion.metrics.coverage == baseline.metrics.coverage
    assert fusion.metrics.position_rmse_m < 1.25 * baseline.metrics.position_rmse_m


def test_monitor_off_produces_no_report():
    assert run_single(SCENARIOS["reference"], TrackerConfig(), 0).modes[Mode.FUSION].result.health is None


def test_stone_soup_cross_check_refuses_a_monitor():
    with pytest.raises(ValueError):
        StoneSoupTracker(MONITORED)


def test_unset_monitor_does_not_enter_results_records():
    record = build_record("compare", [], TrackerConfig(), [0], [])
    assert "health_monitor" not in record["tracker"]
    assert config_hash(record)
    assert "health_monitor" in build_record("compare", [], MONITORED, [0], [])["tracker"]


# -- metrics -------------------------------------------------------------------------------------------------------


def with_report(scenario_name: str, transitions):
    scenario = SCENARIOS[scenario_name]
    run = run_single(scenario, MONITORED, 0)
    fusion = run.modes[Mode.FUSION]
    report = HealthReport(tuple(transitions), {}, {}, {scenario.primary.sensor_id: 7})
    modes = dict(run.modes)
    modes[Mode.FUSION] = replace(fusion, result=replace(fusion.result, health=report))
    return replace(run, modes=modes), scenario


def at(seconds: float) -> datetime:
    return simulate(SCENARIOS["reference"], 0).truth.timestamps[0] + timedelta(seconds=seconds)


def transition(seconds, sensor, previous, state, reason="cusum"):
    return HealthTransition(at(seconds), sensor, previous, state, reason)


TRUSTED, SUSPECT, EXCLUDED = SensorHealth.TRUSTED, SensorHealth.SUSPECT, SensorHealth.EXCLUDED


def test_health_metrics_of_an_unbiased_sensor():
    sensor = SCENARIOS["reference"].secondary.sensor_id
    run, scenario = with_report("reference", [
        transition(20, sensor, TRUSTED, SUSPECT),
        transition(25, sensor, SUSPECT, EXCLUDED),
        transition(55, sensor, EXCLUDED, SUSPECT, "consistent"),
        transition(80, sensor, SUSPECT, TRUSTED, "consistent"),
        transition(100, sensor, TRUSTED, SUSPECT),
    ])
    metrics = sensor_health_metrics(run)
    assert metrics[sensor].false_alarms == 2
    assert math.isnan(metrics[sensor].detection_delay_s)
    assert metrics[sensor].time_excluded_s == 30.0
    assert metrics[scenario.primary.sensor_id].held_alarms == 7


def test_health_metrics_of_a_biased_sensor_excluded_until_the_end():
    sensor = SCENARIOS["biased_bearing"].secondary.sensor_id
    run, scenario = with_report("biased_bearing", [
        transition(9, sensor, TRUSTED, SUSPECT),
        transition(12, sensor, SUSPECT, EXCLUDED),
    ])
    metrics = sensor_health_metrics(run)[sensor]
    assert metrics.biased and metrics.false_alarms == 0
    assert metrics.detection_delay_s == 9.0
    assert metrics.time_excluded_s == scenario.duration_s - 12.0


def test_health_metrics_are_empty_without_a_monitor():
    assert sensor_health_metrics(run_single(SCENARIOS["reference"], TrackerConfig(), 0)) == {}


def test_health_summary_needs_a_monitor():
    with pytest.raises(ValueError):
        health_summary(SCENARIOS["reference"], TrackerConfig(), [0])


def test_health_summary_counts_runs():
    summary = health_summary(SCENARIOS["biased_bearing"], MONITORED, range(3))
    assert (summary.runs, summary.biased_runs, summary.detected_runs) == (3, 3, 3)
    assert 0.0 < summary.median_delay_s <= 30.0
    assert summary.mean_time_excluded_s > 100.0
    assert summary.false_alarm_rate == 0.0


def test_a_biased_position_sensor_is_not_attributed():
    """A documented limit (ADR 0012): the position sensor is the reference, so its own bias falls on the bearing."""
    scenario = SCENARIOS["reference"].model_copy(update={"primary": default_primary(bias=60.0)})
    report = run_single(scenario, MONITORED, 0).modes[Mode.FUSION].result.health
    assert report.final_states[scenario.primary.sensor_id] is SensorHealth.TRUSTED
    assert report.final_states[scenario.secondary.sensor_id] is SensorHealth.EXCLUDED


def test_scaled_thresholds_scale_only_the_cusum_limits():
    scaled = scaled_thresholds(MONITOR, 2.0)
    assert scaled.suspect_cusum == 2.0 * MONITOR.suspect_cusum
    assert scaled.exclude_cusum == 2.0 * MONITOR.exclude_cusum
    assert scaled.window_samples == MONITOR.window_samples


def test_roc_has_a_point_per_threshold_and_a_summary_per_bias():
    points = health_roc(FAMILIES["bias_magnitude"], MONITORED, [1.0, 2.0], [1.5, 3.0], range(2))
    assert [p.threshold_scale for p in points] == [1.0, 2.0]
    assert all(set(p.detection) == {1.5, 3.0} for p in points)
    assert all(0.0 <= p.false_alarm_rate <= 1.0 for p in points)


def test_roc_needs_a_monitor():
    with pytest.raises(ValueError):
        health_roc(FAMILIES["bias_magnitude"], TrackerConfig(), [1.0], [1.5], range(1))


# -- command line --------------------------------------------------------------------------------------------------


def test_cli_run_reports_health(capsys):
    assert main(["run", "--scenario", "biased_bearing", "--health-monitor"]) == 0
    assert "health S2-BRG: biased=True" in capsys.readouterr().out


def test_cli_run_without_the_flag_reports_no_health(capsys):
    assert main(["run", "--scenario", "reference"]) == 0
    assert "health S2-BRG" not in capsys.readouterr().out


def test_cli_health(capsys):
    assert main(["health", "--scenario", "biased_bearing", "--seeds", "2", "--workers", "1"]) == 0
    out = capsys.readouterr().out
    assert "biased_bearing" in out and "2/2" in out


def test_cli_roc_writes_csv(tmp_path, capsys):
    path = tmp_path / "roc.csv"
    assert main(["roc", "--seeds", "1", "--scales", "1.0", "--workers", "1", "--csv", str(path)]) == 0
    assert "CSV written" in capsys.readouterr().out
    assert path.read_text(encoding="utf-8").startswith("threshold_scale,")


def test_cli_roc_rejects_a_fixed_scenario():
    with pytest.raises(SystemExit):
        main(["roc", "--scenario", "reference", "--seeds", "1"])
