"""The single-target tracker with its filter steps done by Stone Soup, as a cross-check (ADR 0007).

``StoneSoupTracker`` keeps the gating, selection, initiation and track drop of
``estimation.SingleTargetTracker`` and replaces only predict, innovation and update with Stone
Soup's ``KalmanPredictor`` and ``ExtendedKalmanUpdater`` (Joseph form, symmetrised). It is the
implementation the project started with. Tests run both trackers on the same inputs and require
the same dispositions and estimates within 1e-9; the NumPy tracker is about 20 times faster.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.measurement.nonlinear import Cartesian2DToBearing
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.types.angle import Bearing
from stonesoup.types.array import StateVector
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.state import GaussianState
from stonesoup.updater.kalman import ExtendedKalmanUpdater

from airspace_fusion.config import TrackerConfig
from airspace_fusion.estimation import POSITION_INDICES, SingleTargetTracker, TrackState
from airspace_fusion.schemas import Measurement, MeasurementType, wrap_angle


class AnalyticBearingModel(Cartesian2DToBearing):
    """``Cartesian2DToBearing`` with the exact Jacobian of ``atan2``.

    Stone Soup's default Jacobian is a forward finite difference, which is off by up to
    ~4e-4 relative for this model. A rotation offset only shifts the bearing by a
    constant, so it does not affect the Jacobian.
    """

    def jacobian(self, state, **kwargs) -> np.ndarray:
        x = np.asarray(state.state_vector, dtype=float).ravel()
        offset = np.asarray(self.translation_offset, dtype=float).ravel()
        ix, iy = self.mapping
        dx, dy = x[ix] - offset[0], x[iy] - offset[1]
        r2 = dx * dx + dy * dy
        if r2 == 0.0:
            raise FloatingPointError("bearing Jacobian undefined at the sensor position")
        jac = np.zeros((1, self.ndim_state))
        jac[0, ix] = -dy / r2
        jac[0, iy] = dx / r2
        return jac


def to_stonesoup(state: TrackState) -> GaussianState:
    return GaussianState(StateVector(state.mean), state.covar, timestamp=state.timestamp)


def from_stonesoup(state: GaussianState) -> TrackState:
    return TrackState(
        np.asarray(state.state_vector, dtype=float).ravel(), np.asarray(state.covar, dtype=float), state.timestamp
    )


class StoneSoupTracker(SingleTargetTracker):
    def __init__(self, config: TrackerConfig) -> None:
        super().__init__(config)
        q = config.process_noise_m2_s3
        self.transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q), ConstantVelocity(q)])
        self.predictor = KalmanPredictor(self.transition_model)
        self.updater = ExtendedKalmanUpdater(
            measurement_model=None, use_joseph_cov=True, force_symmetric_covariance=True
        )

    @staticmethod
    def _detection(m: Measurement) -> Detection:
        if m.measurement_type is MeasurementType.POSITION_2D:
            model = LinearGaussian(
                ndim_state=4, mapping=POSITION_INDICES, noise_covar=m.covariance()
            )
            vector = StateVector(list(m.measurement_value))
        elif m.measurement_type is MeasurementType.BEARING_ONLY:
            model = AnalyticBearingModel(
                ndim_state=4,
                mapping=POSITION_INDICES,
                noise_covar=np.array([[m.angular_std() ** 2]]),
                translation_offset=StateVector(list(m.sensor_position)),
            )
            vector = StateVector([Bearing(m.measurement_value[0])])
        else:  # guarded by the caller; never reinterpret another type
            raise ValueError(f"unsupported measurement type {m.measurement_type}")
        return Detection(vector, timestamp=m.timestamp_utc, measurement_model=model)

    def _predict(self, state: TrackState, t: datetime) -> TrackState:
        return from_stonesoup(self.predictor.predict(to_stonesoup(state), timestamp=t))

    def _nis(self, prior: TrackState, m: Measurement) -> tuple[float, object]:
        detection = self._detection(m)
        prediction = to_stonesoup(prior)
        mp = self.updater.predict_measurement(prediction, detection.measurement_model)
        innovation = np.asarray(detection.state_vector, dtype=float) - np.asarray(mp.state_vector, dtype=float)
        if m.measurement_type is MeasurementType.BEARING_ONLY:
            innovation[0, 0] = wrap_angle(innovation[0, 0])
        s = np.asarray(mp.covar, dtype=float)
        nis = (innovation.T @ np.linalg.solve(s, innovation)).item()
        if not np.isfinite(nis):
            raise FloatingPointError(f"non-finite NIS for measurement {m.measurement_id}")
        return nis, SingleHypothesis(prediction, detection, measurement_prediction=mp)

    def _update(self, prior: TrackState, m: Measurement, hypothesis: object) -> TrackState:
        assert isinstance(hypothesis, SingleHypothesis)
        return from_stonesoup(self.updater.update(hypothesis))
