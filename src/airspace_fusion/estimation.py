"""Single-object state estimation: a Kalman/EKF filter written directly in NumPy.

Model and assumptions (deliberately simple):

* State ``[x, vx, y, vy]`` in metres and m/s (Stone Soup ordering).
* Nearly-constant-velocity motion: continuous white-noise acceleration with
  intensity ``process_noise_m2_s3`` per axis. Turns are model mismatch that the
  process noise must absorb.
* POSITION_2D: linear Gaussian measurement model, covariance taken from the
  measurement itself (Kalman update).
* BEARING_ONLY: ``atan2`` measurement from the reporting sensor's position with
  the reported angular variance, linearised by an extended Kalman update with the
  exact analytic Jacobian. Innovations are wrapped to [-pi, pi).
* Covariance updates use the Joseph form and are then symmetrised.
* Exactly one object. Each sensor contributes at most one measurement of each type per scan:
  the in-gate candidate with the smallest normalised innovation squared (NIS).
  This is gating plus nearest-neighbour selection, not multi-object association.
* Measurement errors are assumed zero-mean, white and independent across
  sensors and time. A biased or correlated sensor violates this silently from
  the filter's point of view: it will be over-confident.
* Track initiation needs two position measurements on consecutive scans that
  imply a plausible speed. Bearings alone never initiate a track.
* A track is dropped after ``max_coast_s`` without an accepted *position* update.
  Bearings refine a track but cannot confirm it: they carry no range information,
  so on their own they could keep a diverged track alive indefinitely.

The filter equations are the same as Stone Soup's ``KalmanPredictor`` and
``ExtendedKalmanUpdater``. ``estimation_stonesoup.StoneSoupTracker`` runs the same
gating and initiation logic on Stone Soup and serves as a cross-check (ADR 0007).

This module only ever receives validated ``Measurement`` objects and the scan
schedule. It has no access to ground truth.
"""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
from scipy.stats import chi2

from airspace_fusion.config import TrackerConfig
from airspace_fusion.schemas import Measurement, MeasurementType, wrap_angle

POSITION_INDICES = (0, 2)
POSITION_SLICE = slice(0, None, 2)  # the same indices; slicing is much cheaper than np.ix_
SUPPORTED_TYPES = (MeasurementType.POSITION_2D, MeasurementType.BEARING_ONLY)
# Bearing linearisation is unreliable when the predicted target is this close to the
# bearing sensor relative to the position uncertainty (in standard deviations).
MIN_BEARING_RANGE_SIGMAS = 3.0


POSITION_MATRIX = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
IDENTITY = np.eye(4)


@dataclass(frozen=True)
class TrackState:
    """Gaussian track state: mean ``[x, vx, y, vy]``, 4x4 covariance, and the time they refer to.

    The arrays are never modified after construction; every filter step makes new ones.
    """

    mean: np.ndarray
    covar: np.ndarray
    timestamp: datetime


@dataclass(frozen=True)
class Innovation:
    """One measurement against a predicted state: what the Kalman update needs, computed once."""

    vector: np.ndarray  # measurement minus predicted measurement (bearings wrapped to [-pi, pi))
    jacobian: np.ndarray  # H, measurement dimension x 4
    noise: np.ndarray  # R
    cross_covar: np.ndarray  # P H^T
    covar: np.ndarray  # S = H P H^T + R
    covar_inverse: np.ndarray

    @property
    def nis(self) -> float:
        return float(self.vector @ self.covar_inverse @ self.vector)


def small_inverse(s: np.ndarray) -> np.ndarray:
    """Inverse of a 1x1 or 2x2 matrix in closed form; ``np.linalg.inv`` costs more in overhead than in work."""
    if s.shape == (1, 1):
        return 1.0 / s
    (a, b), (c, d) = s
    det = a * d - b * c
    if not det > 0.0:  # an innovation covariance is positive definite; also catches NaN
        raise FloatingPointError("innovation covariance is not positive definite")
    return np.array([[d, -b], [-c, a]]) / det


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
    def position(self) -> np.ndarray:
        """Estimated ``[x, y]``. Raises ValueError when there is no track: check ``has_track`` first."""
        if self.state is None:
            raise ValueError(f"no track at {self.timestamp}")
        return self.state[POSITION_SLICE].copy()

    @property
    def position_covariance(self) -> np.ndarray:
        """2x2 covariance of ``position``. Raises ValueError when there is no track."""
        if self.covariance is None:
            raise ValueError(f"no track at {self.timestamp}")
        return self.covariance[POSITION_SLICE, POSITION_SLICE].copy()


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
    """Gating, nearest-neighbour selection, initiation and track drop around a NumPy Kalman/EKF filter.

    The filter steps are ``_predict``, ``_nis`` and ``_update``; a subclass can replace them
    (``estimation_stonesoup.StoneSoupTracker`` does) and keep everything else.
    """

    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self.max_coast = timedelta(seconds=config.max_coast_s)
        # Chi-square gate thresholds by measurement dimension (1 = bearing, 2 = position).
        self.gate_thresholds = {dof: float(chi2.ppf(config.gate_probability, dof)) for dof in (1, 2)}
        self._transitions: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    # -- filter steps -----------------------------------------------------------

    def _transition(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Transition matrix F and process noise Q over ``dt`` seconds; cached, since scans are regular."""
        cached = self._transitions.get(dt)
        if cached is None:
            q = self.config.process_noise_m2_s3
            f1 = np.array([[1.0, dt], [0.0, 1.0]])
            adt = abs(dt)
            q1 = np.array([[adt**3 / 3, adt**2 / 2], [adt**2 / 2, adt]]) * q
            f, noise = np.zeros((4, 4)), np.zeros((4, 4))
            f[:2, :2] = f[2:, 2:] = f1
            noise[:2, :2] = noise[2:, 2:] = q1
            cached = self._transitions[dt] = (f, noise)
        return cached

    def _predict(self, state: TrackState, t: datetime) -> TrackState:
        if state.timestamp == t:
            return state  # F = I and Q = 0: nothing changes
        f, noise = self._transition((t - state.timestamp).total_seconds())
        return TrackState(f @ state.mean, f @ state.covar @ f.T + noise, t)

    def _innovation(self, prior: TrackState, m: Measurement) -> Innovation:
        x = prior.mean
        if m.measurement_type is MeasurementType.POSITION_2D:
            jacobian = POSITION_MATRIX
            noise = m.covariance()
            vector = np.array(m.measurement_value) - x[POSITION_SLICE]
        elif m.measurement_type is MeasurementType.BEARING_ONLY:
            dx, dy = x[0] - m.sensor_position[0], x[2] - m.sensor_position[1]
            r2 = dx * dx + dy * dy
            if r2 == 0.0:
                raise FloatingPointError("bearing Jacobian undefined at the sensor position")
            jacobian = np.array([[-dy / r2, 0.0, dx / r2, 0.0]])  # exact derivative of atan2
            noise = np.array([[m.angular_std() ** 2]])
            vector = np.array([wrap_angle(m.measurement_value[0] - math.atan2(dy, dx))])
        else:  # guarded by the caller; never reinterpret another type
            raise ValueError(f"unsupported measurement type {m.measurement_type}")
        cross_covar = prior.covar @ jacobian.T
        covar = jacobian @ cross_covar + noise
        return Innovation(vector, jacobian, noise, cross_covar, covar, small_inverse(covar))

    def _nis(self, prior: TrackState, m: Measurement) -> tuple[float, object]:
        """NIS of ``m`` against ``prior``, plus whatever ``_update`` needs to apply it."""
        innovation = self._innovation(prior, m)
        nis = innovation.nis
        if not np.isfinite(nis):
            raise FloatingPointError(f"non-finite NIS for measurement {m.measurement_id}")
        return nis, innovation

    def _update(self, prior: TrackState, m: Measurement, innovation: object) -> TrackState:
        """Extended Kalman update with the Joseph-form covariance, then symmetrised."""
        assert isinstance(innovation, Innovation)
        gain = innovation.cross_covar @ innovation.covar_inverse
        i_kh = IDENTITY - gain @ innovation.jacobian
        covar = i_kh @ prior.covar @ i_kh.T + gain @ innovation.noise @ gain.T
        return TrackState(prior.mean + gain @ innovation.vector, (covar + covar.T) / 2, m.timestamp_utc)

    @staticmethod
    def _bearing_geometry_ok(prior: TrackState, m: Measurement) -> bool:
        x, p = prior.mean, prior.covar
        rng = math.hypot(x[0] - m.sensor_position[0], x[2] - m.sensor_position[1])
        # Largest eigenvalue of the 2x2 position covariance, in closed form.
        half_trace, half_diff = (p[0, 0] + p[2, 2]) / 2, (p[0, 0] - p[2, 2]) / 2
        sigma = math.sqrt(half_trace + math.hypot(half_diff, p[0, 2]))
        return rng > MIN_BEARING_RANGE_SIGMAS * sigma

    @staticmethod
    def _check_state(state: TrackState) -> None:
        # A sum is finite only if every term is (the values are nowhere near overflow); cheaper than isfinite.
        if not math.isfinite(float(state.mean.sum() + state.covar.sum())):
            raise FloatingPointError("non-finite state or covariance after update")
        try:
            np.linalg.cholesky(state.covar)
        except np.linalg.LinAlgError as exc:
            raise FloatingPointError("state covariance lost positive definiteness") from exc

    # -- initiation ----------------------------------------------------------

    def _initiate(
        self, previous: Sequence[Measurement], current: Sequence[Measurement]
    ) -> tuple[TrackState, Measurement, Measurement] | None:
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
        r[:2, :2] = a.covariance()
        r[2:, 2:] = b.covariance()
        state = TrackState(mapping @ z, mapping @ r @ mapping.T, b.timestamp_utc)
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

        track: TrackState | None = None
        last_position_update: datetime | None = None
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

            if (
                track is not None
                and last_position_update is not None
                and t - last_position_update > self.max_coast
            ):
                track = None

            if track is not None:
                state = track
                for group in _group_by_sensor_and_type(scan):
                    state, used = self._update_with_sensor(state, group, t, dispositions, update_nis)
                    accepted.extend(used)
                track = self._predict(state, t)
                if any(m.measurement_type is MeasurementType.POSITION_2D for m in accepted):
                    last_position_update = t
            else:
                started_track = self._initiate(previous_positions, positions)
                if started_track is not None:
                    track, a, b = started_track
                    dispositions[a.measurement_id] = Disposition.INITIATION
                    dispositions[b.measurement_id] = Disposition.INITIATION
                    accepted.append(b)
                    last_position_update = t
                    initiations += 1

            # Only scans that contain positions replace the initiation history, so other
            # sensors' observations cannot change how (or when) a track is started.
            if positions:
                previous_positions = positions

            if t in report_set:
                estimates.append(
                    Estimate(
                        timestamp=t,
                        state=None if track is None else track.mean,
                        covariance=None if track is None else track.covar,
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
        state: TrackState,
        group: list[Measurement],
        t: datetime,
        dispositions: dict[str, str],
        update_nis: dict[str, float] | None = None,
    ) -> tuple[TrackState, list[Measurement]]:
        """Gate one sensor's measurements of one type and apply the nearest in-gate one."""
        prior = self._predict(state, t)
        threshold = self.gate_thresholds[len(group[0].measurement_value)]
        candidates: list[tuple[float, Measurement, object]] = []
        for m in group:
            if m.measurement_type is MeasurementType.BEARING_ONLY and not self._bearing_geometry_ok(prior, m):
                dispositions[m.measurement_id] = Disposition.DEGENERATE_GEOMETRY
                continue
            nis, payload = self._nis(prior, m)
            if nis > threshold:
                dispositions[m.measurement_id] = Disposition.OUTSIDE_GATE
            else:
                candidates.append((nis, m, payload))
        if not candidates:
            return prior, []
        candidates.sort(key=lambda c: (c[0], c[1].measurement_id))
        chosen_nis, chosen, payload = candidates[0]
        for _, m, _ in candidates[1:]:
            dispositions[m.measurement_id] = Disposition.NOT_SELECTED
        posterior = self._update(prior, chosen, payload)
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
