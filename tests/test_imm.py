"""IMM tracker (M4, ADR 0011): constant velocity plus coordinated turns at +/- a fixed rate."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from airspace_fusion.cli import main
from airspace_fusion.config import SCENARIOS, TrackerConfig
from airspace_fusion.estimation import IMMTracker, SingleTargetTracker, run_tracker
from airspace_fusion.evaluation import Mode, run_single
from airspace_fusion.schemas import parse_records
from airspace_fusion.simulator import simulate

REFERENCE = SCENARIOS["reference"]
IMM = TrackerConfig(motion_model="imm")


def test_default_tracker_is_still_constant_velocity():
    run = run_single(REFERENCE, TrackerConfig(), 0)
    assert all(e.mode_probabilities is None for e in run.modes[Mode.BASELINE].result.estimates)


@pytest.mark.parametrize("omega_deg", [3.0, -2.25, 10.0])
@pytest.mark.parametrize("dt", [0.5, 1.0, 3.0])
def test_coordinated_turn_transition_is_exact(omega_deg, dt):
    """F of the turn model moves a target exactly along a circle at the turn rate."""
    tracker = IMMTracker(TrackerConfig(motion_model="imm", imm_turn_rate_deg_s=abs(omega_deg)))
    model = 1 if omega_deg > 0 else 2
    omega = math.radians(omega_deg)
    f, _ = tracker._model_transition(model, dt)
    speed, heading = 20.0, 0.3
    x = np.array([100.0, speed * math.cos(heading), -50.0, speed * math.sin(heading)])
    for step in range(1, 30):
        x = f @ x
        t = step * dt
        new_heading = heading + omega * t
        # Closed-form position on the circle of radius speed / omega.
        px = 100.0 + speed / omega * (math.sin(new_heading) - math.sin(heading))
        py = -50.0 - speed / omega * (math.cos(new_heading) - math.cos(heading))
        np.testing.assert_allclose(x, [px, speed * math.cos(new_heading), py, speed * math.sin(new_heading)],
                                   rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("scenario", ["reference", "biased_bearing"])
def test_imm_with_identical_models_reduces_to_constant_velocity(scenario):
    """Three practically identical models must give the constant-velocity tracker's estimates."""
    q = 0.5
    same = TrackerConfig(motion_model="imm", imm_turn_rate_deg_s=1e-7, imm_stay_probability=0.9,
                         imm_straight_process_noise_m2_s3=q, imm_turn_process_noise_m2_s3=q)
    sim = simulate(SCENARIOS[scenario], 1)
    measurements, _ = parse_records(sim.records)
    cv = run_tracker(measurements, sim.report_times, TrackerConfig(process_noise_m2_s3=q))
    imm = run_tracker(measurements, sim.report_times, same)
    assert cv.dispositions == imm.dispositions
    for a, b in zip(cv.estimates, imm.estimates):
        assert a.has_track == b.has_track
        if a.has_track:
            np.testing.assert_allclose(b.state, a.state, rtol=0, atol=1e-6)
            np.testing.assert_allclose(b.covariance, a.covariance, rtol=1e-6, atol=1e-6)


def test_mode_probabilities_follow_the_turn():
    """reference turns right (-2.25 deg/s) from 60 s to 100 s; straight before and after."""
    turning, straight = [], []
    for seed in range(5):
        run = run_single(REFERENCE, IMM, seed)
        t0 = run.simulation.truth.timestamps[0]
        for mode in Mode:
            for e in run.modes[mode].result.estimates:
                if e.mode_probabilities is None:
                    continue
                assert len(e.mode_probabilities) == 3
                assert math.isclose(sum(e.mode_probabilities), 1.0, abs_tol=1e-9)
                assert min(e.mode_probabilities) >= 0.0
                t = (e.timestamp - t0).total_seconds()
                if 70 <= t < 100:
                    turning.append(e.mode_probabilities)
                elif 20 <= t < 55 or t >= 130:
                    straight.append(e.mode_probabilities)
    turning_mean, straight_mean = np.mean(turning, axis=0), np.mean(straight, axis=0)
    assert turning_mean[2] > 0.5 > turning_mean[1]  # the right-turn model takes over during the turn
    assert straight_mean[0] > 0.6  # constant velocity dominates on straight flight


def test_estimates_without_a_track_have_no_mode_probabilities():
    run = run_single(REFERENCE, IMM, 0)
    for e in run.modes[Mode.BASELINE].result.estimates:
        assert (e.mode_probabilities is None) == (not e.has_track)


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"motion_model": "ca"}, "motion_model"),
        ({"imm_turn_rate_deg_s": 0.0}, "greater than 0"),
        ({"imm_turn_rate_deg_s": 45.0}, "less than or equal to 30"),
        ({"imm_stay_probability": 1.0}, "less than 1"),
        ({"imm_turn_process_noise_m2_s3": -1.0}, "greater than 0"),
    ],
)
def test_invalid_imm_settings(settings, message):
    with pytest.raises(ValidationError, match=message):
        TrackerConfig(**settings)


def test_tracker_class_follows_the_setting():
    assert type(IMMTracker(IMM)) is IMMTracker
    assert isinstance(IMMTracker(IMM), SingleTargetTracker)  # gating, initiation and track drop are shared


def test_cli_motion_model_option(capsys, results_in_tmp):
    assert main(["run", "--seed", "0", "--motion-model", "imm"]) == 0
    imm_out = capsys.readouterr().out
    assert main(["run", "--seed", "0"]) == 0
    assert capsys.readouterr().out != imm_out
    assert main(["compare", "--scenario", "reference", "--seeds", "2", "--motion-model", "imm"]) == 0
    from airspace_fusion.results import load_record

    (path,) = results_in_tmp.glob("*-compare-*.json")
    assert load_record(path)["tracker"]["motion_model"] == "imm"
