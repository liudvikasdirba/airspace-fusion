"""Single-object state estimation using Stone Soup Kalman components.

Model and assumptions (deliberately simple):

* State ``[x, vx, y, vy]`` in metres and m/s (Stone Soup ordering).
* Nearly-constant-velocity motion: continuous white-noise acceleration with
  intensity ``process_noise_m2_s3`` per axis. Turns are model mismatch that the
  process noise must absorb.
* POSITION_2D: linear Gaussian measurement model, covariance taken from the
  measurement itself (Kalman update).
* BEARING_ONLY: ``atan2`` measurement from the reporting sensor's position with
  the reported angular variance, linearised by an extended Kalman update.
  Innovations are wrapped to [-pi, pi) (Stone Soup ``Bearing`` arithmetic).
* Exactly one object. Each sensor contributes at most one measurement per scan:
  the in-gate candidate with the smallest normalised innovation squared (NIS).
  This is gating plus nearest-neighbour selection, not multi-object association.
* Measurement errors are assumed zero-mean, white and independent across
  sensors and time. A biased or correlated sensor violates this silently from
  the filter's point of view: it will be over-confident.
* Track initiation needs two position measurements on consecutive scans that
  imply a plausible speed. Bearings alone never initiate a track.
* A track is dropped after ``max_coast_s`` without an accepted update.

This module only ever receives validated ``Measurement`` objects and the scan
schedule. It has no access to ground truth.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
from scipy.stats import chi2
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
from airspace_fusion.schemas import Measurement, MeasurementType, wrap_angle

POSITION_INDICES = (0, 2)
SUPPORTED_TYPES = (MeasurementType.POSITION_2D, MeasurementType.BEARING_ONLY)
# Bearing linearisation is unreliable when the predicted target is this close to the
# bearing sensor relative to the position uncertainty (in standard deviations).
MIN_BEARING_RANGE_SIGMAS = 3.0


class Disposition:
    ACCEPTED = "accepted"
    INITIATION = "used_for_initiation"
    OUTSIDE_GATE = "rejected_outside_gate"
    NOT_SELECTED = "rejected_not_nearest_in_gate"
    UNSUPPORTED_TYPE = "rejected_unsupported_type"
    DEGENERATE_GEOMETRY = "rejected_degenerate_geometry"
    NO_TRACK = "unused_no_track"

    REJECTED = (OUTSIDE_GATE, NOT_SELECTED, UNSUPPORTED_TYPE, DEGENERATE_GEOMETRY)


@dataclass(frozen=True)
class Estimate:
    timestamp: datetime
    state: np.ndarray | None  # [x, vx, y, vy], or None when no track exists
    covariance: np.ndarray | None  # 4x4
    updated_by: tuple[str, ...]  # measurement ids accepted at this timestamp
    updated_sensors: frozenset[str] = frozenset()  # sensors that contributed those measurements

    @property
    def has_track(self) -> bool:
        return self.state is not None

    @property
    def position(self) -> np.ndarray | None:
        return None if self.state is None else self.state[list(POSITION_INDICES)]

    @property
    def position_covariance(self) -> np.ndarray | None:
        if self.covariance is None:
            return None
        return self.covariance[np.ix_(POSITION_INDICES, POSITION_INDICES)]


@dataclass
class TrackerResult:
    estimates: list[Estimate]
    dispositions: dict[str, str]  # measurement_id -> Disposition
    track_initiations: int
    processing_time_s: float
    duplicate_count: int = 0  # records whose measurement_id was already seen; not processed
    # Normalised innovation squared of each accepted update, for filter consistency checks.
    update_nis: dict[str, float] = field(default_factory=dict)

    @property
    def disposition_counts(self) -> Counter[str]:
        return Counter(self.dispositions.values())

    @property
    def rejected_count(self) -> int:
        counts = self.disposition_counts
        return sum(counts[r] for r in Disposition.REJECTED) + self.duplicate_count

    @property
    def prediction_only_scans(self) -> int:
        """Report times at which a track existed but no measurement from any sensor was accepted."""
        return sum(1 for e in self.estimates if e.has_track and not e.updated_by)

    def missed_updates_for(self, sensor_id: str) -> int:
        """Report times at which a track existed but ``sensor_id`` contributed no accepted measurement."""
        return sum(1 for e in self.estimates if e.has_track and sensor_id not in e.updated_sensors)


class SingleTargetTracker:
    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        q = config.process_noise_m2_s3
        self.transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q), ConstantVelocity(q)])
        self.predictor = KalmanPredictor(self.transition_model)
        self.updater = ExtendedKalmanUpdater(
            measurement_model=None, use_joseph_cov=True, force_symmetric_covariance=True
        )
        self.max_coast = timedelta(seconds=config.max_coast_s)
        # Chi-square gate thresholds by measurement dimension (1 = bearing, 2 = position).
        self.gate_thresholds = {dof: float(chi2.ppf(config.gate_probability, dof)) for dof in (1, 2)}

    # -- measurement models -------------------------------------------------

    @staticmethod
    def _detection(m: Measurement) -> Detection:
        if m.measurement_type is MeasurementType.POSITION_2D:
            model = LinearGaussian(
                ndim_state=4, mapping=POSITION_INDICES, noise_covar=m.measurement_uncertainty.as_array()
            )
            vector = StateVector(list(m.measurement_value))
        elif m.measurement_type is MeasurementType.BEARING_ONLY:
            model = Cartesian2DToBearing(
                ndim_state=4,
                mapping=POSITION_INDICES,
                noise_covar=np.array([[m.measurement_uncertainty.std_rad**2]]),
                translation_offset=StateVector(list(m.sensor_position)),
            )
            vector = StateVector([Bearing(m.measurement_value[0])])
        else:  # guarded by the caller; never reinterpret another type
            raise ValueError(f"unsupported measurement type {m.measurement_type}")
        return Detection(vector, timestamp=m.timestamp_utc, measurement_model=model)

    def _nis(self, prior: GaussianState, m: Measurement) -> tuple[float, SingleHypothesis]:
        detection = self._detection(m)
        mp = self.updater.predict_measurement(prior, detection.measurement_model)
        innovation = np.asarray(detection.state_vector, dtype=float) - np.asarray(mp.state_vector, dtype=float)
        if m.measurement_type is MeasurementType.BEARING_ONLY:
            innovation[0, 0] = wrap_angle(innovation[0, 0])
        s = np.asarray(mp.covar, dtype=float)
        nis = (innovation.T @ np.linalg.solve(s, innovation)).item()
        if not np.isfinite(nis):
            raise FloatingPointError(f"non-finite NIS for measurement {m.measurement_id}")
        return nis, SingleHypothesis(prior, detection, measurement_prediction=mp)

    @staticmethod
    def _bearing_geometry_ok(prior: GaussianState, m: Measurement) -> bool:
        pos = np.asarray(prior.state_vector, dtype=float).ravel()[list(POSITION_INDICES)]
        rng = float(np.linalg.norm(pos - np.asarray(m.sensor_position)))
        pos_cov = np.asarray(prior.covar, dtype=float)[np.ix_(POSITION_INDICES, POSITION_INDICES)]
        sigma = float(np.sqrt(np.max(np.linalg.eigvalsh(pos_cov))))
        return rng > MIN_BEARING_RANGE_SIGMAS * sigma

    @staticmethod
    def _check_state(state: GaussianState) -> None:
        x = np.asarray(state.state_vector, dtype=float)
        p = np.asarray(state.covar, dtype=float)
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(p))):
            raise FloatingPointError("non-finite state or covariance after update")
        try:
            np.linalg.cholesky(p)
        except np.linalg.LinAlgError as exc:
            raise FloatingPointError("state covariance lost positive definiteness") from exc

    # -- initiation ----------------------------------------------------------

    def _initiate(
        self, previous: Sequence[Measurement], current: Sequence[Measurement]
    ) -> tuple[GaussianState, Measurement, Measurement] | None:
        """Two-point differencing on consecutive scans; picks the slowest plausible pair."""
        best: tuple[float, Measurement, Measurement] | None = None
        for a in previous:
            for b in current:
                dt = (b.timestamp_utc - a.timestamp_utc).total_seconds()
                if dt <= 0.0 or dt > self.config.max_coast_s:
                    continue
                speed = float(np.hypot(*(np.subtract(b.measurement_value, a.measurement_value)))) / dt
                if speed <= self.config.max_initiation_speed_mps and (best is None or speed < best[0]):
                    best = (speed, a, b)
        if best is None:
            return None
        _, a, b = best
        dt = (b.timestamp_utc - a.timestamp_utc).total_seconds()
        # state = A @ [z_a; z_b]; covariance propagated exactly through the linear map.
        mapping = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [-1.0 / dt, 0.0, 1.0 / dt, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, -1.0 / dt, 0.0, 1.0 / dt],
            ]
        )
        z = np.concatenate([a.measurement_value, b.measurement_value])
        r = np.zeros((4, 4))
        r[:2, :2] = a.measurement_uncertainty.as_array()
        r[2:, 2:] = b.measurement_uncertainty.as_array()
        state = GaussianState(StateVector(mapping @ z), mapping @ r @ mapping.T, timestamp=b.timestamp_utc)
        self._check_state(state)
        return state, a, b

    # -- main loop -------------------------------------------------------------

    def run(self, measurements: Iterable[Measurement], report_times: Sequence[datetime]) -> TrackerResult:
        started = time.perf_counter()
        _validate_report_times(report_times)
        report_set = set(report_times)

        dispositions: dict[str, str] = {}
        update_nis: dict[str, float] = {}
        duplicates = 0
        by_time: dict[datetime, list[Measurement]] = defaultdict(list)
        for m in measurements:
            if m.measurement_id in dispositions:
                duplicates += 1
                continue
            if m.measurement_type not in SUPPORTED_TYPES:
                dispositions[m.measurement_id] = Disposition.UNSUPPORTED_TYPE
                continue
            dispositions[m.measurement_id] = Disposition.NO_TRACK  # provisional
            by_time[m.timestamp_utc].append(m)

        track: GaussianState | None = None
        last_update: datetime | None = None
        previous_positions: list[Measurement] = []
        initiations = 0
        estimates: list[Estimate] = []

        for t in sorted(set(by_time) | report_set):
            scan = sorted(
                by_time.get(t, []),
                key=lambda m: (SUPPORTED_TYPES.index(m.measurement_type), m.sensor_id, m.measurement_id),
            )
            positions = [m for m in scan if m.measurement_type is MeasurementType.POSITION_2D]
            accepted: list[Measurement] = []

            if track is not None and last_update is not None and t - last_update > self.max_coast:
                track = None

            if track is not None:
                state = track
                for group in _group_by_sensor_and_type(scan):
                    state, used = self._update_with_sensor(state, group, t, dispositions, update_nis)
                    accepted.extend(used)
                track = state if state.timestamp == t else self.predictor.predict(state, timestamp=t)
                if accepted:
                    last_update = t
            else:
                started_track = self._initiate(previous_positions, positions)
                if started_track is not None:
                    track, a, b = started_track
                    dispositions[a.measurement_id] = Disposition.INITIATION
                    dispositions[b.measurement_id] = Disposition.INITIATION
                    accepted.append(b)
                    last_update = t
                    initiations += 1

            # Only scans that contain positions replace the initiation history, so other
            # sensors' observations cannot change how (or when) a track is started.
            if positions:
                previous_positions = positions

            if t in report_set:
                estimates.append(
                    Estimate(
                        timestamp=t,
                        state=None if track is None else np.asarray(track.state_vector, dtype=float).ravel(),
                        covariance=None if track is None else np.asarray(track.covar, dtype=float),
                        updated_by=tuple(m.measurement_id for m in accepted),
                        updated_sensors=frozenset(m.sensor_id for m in accepted),
                    )
                )

        return TrackerResult(
            estimates=estimates,
            dispositions=dispositions,
            track_initiations=initiations,
            processing_time_s=time.perf_counter() - started,
            duplicate_count=duplicates,
            update_nis=update_nis,
        )

    def _update_with_sensor(
        self,
        state: GaussianState,
        group: list[Measurement],
        t: datetime,
        dispositions: dict[str, str],
        update_nis: dict[str, float] | None = None,
    ) -> tuple[GaussianState, list[Measurement]]:
        """Gate one sensor's measurements of one type and apply the nearest in-gate one."""
        prior = self.predictor.predict(state, timestamp=t)
        threshold = self.gate_thresholds[len(group[0].measurement_value)]
        candidates: list[tuple[float, Measurement, SingleHypothesis]] = []
        for m in group:
            if m.measurement_type is MeasurementType.BEARING_ONLY and not self._bearing_geometry_ok(prior, m):
                dispositions[m.measurement_id] = Disposition.DEGENERATE_GEOMETRY
                continue
            nis, hypothesis = self._nis(prior, m)
            if nis > threshold:
                dispositions[m.measurement_id] = Disposition.OUTSIDE_GATE
            else:
                candidates.append((nis, m, hypothesis))
        if not candidates:
            return prior, []
        candidates.sort(key=lambda c: (c[0], c[1].measurement_id))
        chosen_nis, chosen, hypothesis = candidates[0]
        for _, m, _ in candidates[1:]:
            dispositions[m.measurement_id] = Disposition.NOT_SELECTED
        posterior = self.updater.update(hypothesis)
        self._check_state(posterior)
        dispositions[chosen.measurement_id] = Disposition.ACCEPTED
        if update_nis is not None:
            update_nis[chosen.measurement_id] = chosen_nis
        return posterior, [chosen]


def _group_by_sensor_and_type(scan: list[Measurement]) -> list[list[Measurement]]:
    """One group per (sensor, measurement type), so gating never mixes measurement dimensions."""
    groups: dict[tuple[str, MeasurementType], list[Measurement]] = {}
    for m in scan:  # scan is already ordered: positions first, then bearings
        groups.setdefault((m.sensor_id, m.measurement_type), []).append(m)
    return list(groups.values())


def _validate_report_times(report_times: Sequence[datetime]) -> None:
    for t in report_times:
        if t.utcoffset() != timedelta(0):
            raise ValueError("report times must be timezone-aware UTC datetimes")
    for a, b in zip(report_times, report_times[1:]):
        if b <= a:
            raise ValueError("report times must be strictly increasing")


def run_tracker(
    measurements: Iterable[Measurement],
    report_times: Sequence[datetime],
    config: TrackerConfig,
    sensor_ids: Iterable[str] | None = None,
) -> TrackerResult:
    """Run the tracker on the measurements from ``sensor_ids`` (all sensors if None)."""
    allowed = None if sensor_ids is None else set(sensor_ids)
    selected = [m for m in measurements if allowed is None or m.sensor_id in allowed]
    return SingleTargetTracker(config).run(selected, report_times)
