"""Independent-oracle checks (E1-E2): the Stone Soup-based tracker against textbook formulas in plain NumPy."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from stonesoup.types.array import StateVector
from stonesoup.types.state import GaussianState, State

from airspace_fusion.config import TrackerConfig
from airspace_fusion.estimation import AnalyticBearingModel, Disposition, SingleTargetTracker
from airspace_fusion.schemas import Measurement, wrap_angle

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
N_CASES = 25


def oracle_predict(x: np.ndarray, p: np.ndarray, dt: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    f1 = np.array([[1.0, dt], [0.0, 1.0]])
    q1 = q * np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])
    f = np.kron(np.eye(2), f1)  # state order [x, vx, y, vy]
    qm = np.kron(np.eye(2), q1)
    return f @ x, f @ p @ f.T + qm


def oracle_update(x, p, innovation, h, r) -> tuple[np.ndarray, np.ndarray, float]:
    s = h @ p @ h.T + r
    k = p @ h.T @ np.linalg.inv(s)
    i_kh = np.eye(4) - k @ h
    nis = float(innovation @ np.linalg.solve(s, innovation))
    return x + k @ innovation, i_kh @ p @ i_kh.T + k @ r @ k.T, nis


def random_prior(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([rng.uniform(-3000, 3000), rng.uniform(-30, 30), rng.uniform(-3000, 3000), rng.uniform(-30, 30)])
    a = rng.normal(size=(4, 4)) * np.array([10.0, 1.0, 10.0, 1.0])
    p = a @ a.T + np.diag([4.0, 0.1, 4.0, 0.1])
    return x, (p + p.T) / 2.0


def time_step(rng: np.random.Generator, low: float, high: float) -> float:
    # Timestamps have microsecond resolution; give the oracle exactly the interval the tracker sees.
    return timedelta(seconds=float(rng.uniform(low, high))).total_seconds()


def measurement(**fields) -> Measurement:
    base = dict(measurement_id="m", sensor_id="s", timestamp_utc=T0, sensor_position=(0.0, 0.0))
    return Measurement.model_validate(base | fields)


def run_tracker_step(x, p, dt, m, q):
    tracker = SingleTargetTracker(TrackerConfig(process_noise_m2_s3=q, gate_probability=0.999999))
    prior = GaussianState(StateVector(x), p, timestamp=T0)
    dispositions: dict[str, str] = {}
    nis: dict[str, float] = {}
    t = T0 + timedelta(seconds=dt)
    posterior, used = tracker._update_with_sensor(prior, [m.model_copy(update={"timestamp_utc": t})], t,
                                                  dispositions, nis)
    assert used and dispositions["m"] == Disposition.ACCEPTED, dispositions
    return np.asarray(posterior.state_vector, dtype=float).ravel(), np.asarray(posterior.covar, dtype=float), nis["m"]


@pytest.mark.parametrize("case", range(N_CASES))
def test_position_update_matches_textbook_kalman(case):
    """E1: one predict + position update equals the closed-form Kalman filter (Joseph form)."""
    rng = np.random.default_rng(1000 + case)
    x, p = random_prior(rng)
    dt = time_step(rng, 0.1, 5.0)
    q = float(rng.uniform(0.01, 5.0))
    a = rng.normal(size=(2, 2)) * rng.uniform(1.0, 50.0)
    r = a @ a.T + np.eye(2)
    r = (r + r.T) / 2.0
    xp, pp = oracle_predict(x, p, dt, q)
    h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    z = h @ xp + rng.multivariate_normal(np.zeros(2), h @ pp @ h.T + r) * 0.5
    want_x, want_p, want_nis = oracle_update(xp, pp, z - h @ xp, h, r)

    m = measurement(measurement_type="POSITION_2D", measurement_value=tuple(z), measurement_units=("m", "m"),
                    measurement_uncertainty={"kind": "covariance", "matrix": tuple(map(tuple, r))})
    got_x, got_p, got_nis = run_tracker_step(x, p, dt, m, q)
    np.testing.assert_allclose(got_x, want_x, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(got_p, want_p, rtol=1e-9, atol=1e-9)
    assert got_nis == pytest.approx(want_nis, rel=1e-9)


def bearing_jacobian(x: np.ndarray, sensor: np.ndarray) -> np.ndarray:
    """Analytic d/dx atan2(y - sy, x - sx), derived independently of the tracker code."""
    dx, dy = x[0] - sensor[0], x[2] - sensor[1]
    r2 = dx * dx + dy * dy
    return np.array([[-dy / r2, 0.0, dx / r2, 0.0]])


def test_analytic_jacobian_agrees_with_central_difference():
    rng = np.random.default_rng(7)
    for _ in range(50):
        x, _, sensor = bearing_case(rng, near_wrap=False)
        step = 1e-3
        numeric = np.zeros((1, 4))
        for i in range(4):
            d = np.zeros(4)
            d[i] = step
            hi = math.atan2(x[2] + d[2] - sensor[1], x[0] + d[0] - sensor[0])
            lo = math.atan2(x[2] - d[2] - sensor[1], x[0] - d[0] - sensor[0])
            numeric[0, i] = wrap_angle(hi - lo) / (2 * step)
        np.testing.assert_allclose(bearing_jacobian(x, sensor), numeric, rtol=1e-6, atol=1e-12)


def test_tracker_bearing_model_uses_exact_jacobian():
    rng = np.random.default_rng(8)
    for _ in range(50):
        x, _, sensor = bearing_case(rng, near_wrap=False)
        model = AnalyticBearingModel(ndim_state=4, mapping=(0, 2), noise_covar=np.array([[1e-4]]),
                                     translation_offset=StateVector(sensor))
        np.testing.assert_allclose(model.jacobian(State(StateVector(x))), bearing_jacobian(x, sensor),
                                   rtol=1e-12, atol=0.0)


def bearing_case(rng: np.random.Generator, near_wrap: bool):
    x, p = random_prior(rng)
    sensor = np.array([rng.uniform(-1000, 1000), rng.uniform(-1000, 1000)])
    rng_m = rng.uniform(800.0, 6000.0)
    true_bearing = math.pi - 1e-3 if near_wrap else rng.uniform(-math.pi, math.pi)
    x[0], x[2] = sensor[0] + rng_m * math.cos(true_bearing), sensor[1] + rng_m * math.sin(true_bearing)
    if near_wrap:
        x[1] = x[3] = 0.0  # keep the predicted position on the wrap line
    return x, p, sensor


@pytest.mark.parametrize("case", range(N_CASES))
def test_bearing_update_matches_hand_coded_ekf(case):
    """E2: one predict + bearing update equals a hand-coded EKF (analytic Jacobian, wrapped innovation)."""
    rng = np.random.default_rng(2000 + case)
    near_wrap = case % 5 == 0  # every fifth case puts the target just below +pi, measured past -pi
    x, p, sensor = bearing_case(rng, near_wrap)
    dt = time_step(rng, 0.1, 3.0)
    q = float(rng.uniform(0.01, 2.0))
    std = float(rng.uniform(1e-3, 2e-2))
    xp, pp = oracle_predict(x, p, dt, q)
    predicted = math.atan2(xp[2] - sensor[1], xp[0] - sensor[0])
    offset = 4e-3 if near_wrap else float(rng.normal(0.0, std))
    z = wrap_angle(predicted + offset)
    if near_wrap:
        assert z < 0.0 < predicted  # the measurement and prediction sit on opposite sides of +-pi
    h = bearing_jacobian(xp, sensor)
    want_x, want_p, want_nis = oracle_update(xp, pp, np.array([wrap_angle(z - predicted)]), h, np.array([[std**2]]))

    m = measurement(measurement_type="BEARING_ONLY", measurement_value=(z,), measurement_units=("rad",),
                    measurement_uncertainty={"kind": "angular_std", "std_rad": std},
                    sensor_position=tuple(sensor))
    got_x, got_p, got_nis = run_tracker_step(x, p, dt, m, q)
    np.testing.assert_allclose(got_x, want_x, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(got_p, want_p, rtol=1e-9, atol=1e-9)
    assert got_nis == pytest.approx(want_nis, rel=1e-9)
