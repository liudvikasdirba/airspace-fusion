"""Sensor health monitor: tests each sensor's normalised innovations for consistency (ADR 0012).

A sensor is judged only from what the tracker sees: the whitened innovation of its nearest measurement against the
track, one sample per scan. If the filter's model is right these samples are independent standard normal, so

* the sum of squares over a sliding window is chi-square (the windowed test, which reacts to any inflation of the
  innovations), and
* a two-sided CUSUM per component stays small (which reacts to a persistent shift, the signature of a bias, long
  before the windowed test does).

Each sensor is trusted, suspect (its measurements are down-weighted) or excluded (its measurements are ignored).
An excluded sensor keeps being tested against the track, which no longer contains it, and comes back through
suspect once its innovations are consistent again.

Every sample is capped at the tracker's gate. A gross bias then counts as a run of the largest allowed values rather
than disappearing behind the gate, and one clutter point cannot trip a test on its own.

This module knows nothing about truth, the simulator or the evaluation; a dependency test keeps it that way.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from statistics import NormalDist

from airspace_fusion.config import HealthMonitorConfig


class SensorHealth(str, Enum):
    TRUSTED = "trusted"
    SUSPECT = "suspect"
    EXCLUDED = "excluded"


def chi2_upper_quantile(tail_probability: float, dof: int) -> float:
    """The value a chi-square variable exceeds with ``tail_probability`` (Wilson-Hilferty approximation).

    Within about 1 % of the exact quantile from 20 degrees of freedom up, and avoids importing ``scipy.stats``, which
    costs worker processes about 1.5 s at start-up.
    """
    if not 0.0 < tail_probability < 1.0:
        raise ValueError(f"tail probability must lie strictly between 0 and 1, got {tail_probability}")
    if dof < 1:
        raise ValueError(f"degrees of freedom must be at least 1, got {dof}")
    z = -NormalDist().inv_cdf(tail_probability)  # 1 - p would round to 1 for the smallest allowed tails
    term = 2.0 / (9.0 * dof)
    return float(dof * (1.0 - term + z * math.sqrt(term)) ** 3)


@dataclass(frozen=True)
class HealthTransition:
    """One sensor changing state."""

    timestamp: datetime
    sensor_id: str
    previous: SensorHealth
    state: SensorHealth
    reason: str  # which test fired, or "consistent"


@dataclass(frozen=True)
class HealthReport:
    """What the monitor did during one tracker run."""

    transitions: tuple[HealthTransition, ...]
    samples: dict[str, int]  # innovations tested, per sensor
    final_states: dict[str, SensorHealth]
    held_alarms: dict[str, int]  # samples at which a test fired on a protected sensor (no action was taken)

    def for_sensor(self, sensor_id: str) -> tuple[HealthTransition, ...]:
        return tuple(t for t in self.transitions if t.sensor_id == sensor_id)


def _worst(states: Iterable[SensorHealth]) -> SensorHealth:
    return max(states, key=tuple(SensorHealth).index, default=SensorHealth.TRUSTED)


class SensorMonitor:
    """The windowed chi-square test, CUSUM and state machine of one sensor."""

    def __init__(self, sensor_id: str, config: HealthMonitorConfig, dof: int) -> None:
        self.sensor_id = sensor_id
        self.config = config
        self.dof = dof
        self.state = SensorHealth.TRUSTED
        self.held_alarms = 0  # samples at which a test fired on a sensor the monitor may not act on
        self.samples = 0
        self.samples_on_track = 0  # since the current track started
        self.samples_in_state = 0
        self.on_probation = config.probation_samples > 0  # only tested, not yet used; see ``_end_probation_if_due``
        window_dof = config.window_samples * dof
        self.suspect_sum = chi2_upper_quantile(config.suspect_window_probability, window_dof)
        self.exclude_sum = chi2_upper_quantile(config.exclude_window_probability, window_dof)
        self.consistent_sum = chi2_upper_quantile(config.consistent_window_probability, window_dof)
        self._window: deque[tuple[float, ...]] = deque(maxlen=config.window_samples)
        self._window_sum = 0.0
        self._was_protected = False
        self._high = [0.0] * dof  # CUSUM of positive shifts, per component
        self._low = [0.0] * dof  # CUSUM of negative shifts, per component

    @property
    def cusum(self) -> float:
        return max(*self._high, *self._low)

    @property
    def window_sum(self) -> float:
        """Sum of squares of the samples in the window."""
        return self._window_sum

    def _push(self, sample: tuple[float, ...]) -> None:
        self.samples += 1
        self.samples_on_track += 1
        self.samples_in_state += 1
        self._window.append(sample)
        self._window_sum = sum(v * v for s in self._window for v in s)
        drift = self.config.cusum_drift
        for i, value in enumerate(sample):
            self._high[i] = max(0.0, self._high[i] + value - drift)
            self._low[i] = max(0.0, self._low[i] - value - drift)

    def _consistent(self) -> bool:
        """The window is full, not inflated and not shifted, and the current state has lasted long enough."""
        if self.samples_in_state < self.config.release_samples or len(self._window) < self.config.window_samples:
            return False
        if self.window_sum > self.consistent_sum:
            return False
        return not self._shifted()

    def _fired(self, cusum_limit: float, sum_limit: float) -> str | None:
        if self.cusum >= cusum_limit:
            return "cusum"
        if len(self._window) == self.config.window_samples and self.window_sum >= sum_limit:
            return "chi-square"
        return None

    def _reset_cusum(self) -> None:
        self._high = [0.0] * self.dof
        self._low = [0.0] * self.dof

    def _clear_tests(self) -> None:
        self._reset_cusum()
        self._window.clear()
        self._window_sum = 0.0

    def start_new_track(self) -> None:
        """Samples against the previous track say nothing about this one, and its state holds no bias yet."""
        self._clear_tests()
        self.samples_on_track = 0
        self.on_probation = self.config.probation_samples > 0
        self._was_protected = False

    def _shifted(self) -> bool:
        """Some component's window mean is further from zero than ``consistent_mean_sigmas`` standard errors."""
        limit = self.config.consistent_mean_sigmas * math.sqrt(len(self._window))
        return any(abs(sum(sample[i] for sample in self._window)) > limit for i in range(self.dof))

    def _end_probation_if_due(self) -> None:
        """Probation ends after ``probation_samples`` unless the samples look shifted, and after a full window anyway:
        a bias too small for the alarms must not keep a sensor out without anything being reported."""
        if self.samples_on_track >= self.config.window_samples or (
            self.samples_on_track >= self.config.probation_samples and not self._shifted()
        ):
            self.on_probation = False

    def observe(self, sample: tuple[float, ...], protected: bool) -> tuple[SensorHealth, str] | None:
        """Add one normalised innovation. Returns the new state and the reason, if the state changed.

        A protected sensor is the track's only source of position updates, so it is trusted whatever its state was.
        What it accumulates meanwhile is counted but dropped when protection ends: the monitor could not act on it.
        """
        if self._was_protected and not protected:
            self._clear_tests()
        self._was_protected = protected
        self._push(sample)
        if self.on_probation:
            self._end_probation_if_due()
        if protected:
            self.held_alarms += self._fired(self.config.suspect_cusum, self.suspect_sum) is not None
            return None if self.state is SensorHealth.TRUSTED else self._enter(SensorHealth.TRUSTED, "protected")
        if self.samples_on_track < self.config.min_samples:
            return None
        config = self.config
        new: SensorHealth | None = None
        reason = ""
        if self.state is SensorHealth.EXCLUDED:
            if self._consistent():
                new, reason = SensorHealth.SUSPECT, "consistent"
        else:
            fired = self._fired(config.exclude_cusum, self.exclude_sum)
            if fired is not None:
                new, reason = SensorHealth.EXCLUDED, fired
            elif self.state is SensorHealth.TRUSTED:
                fired = self._fired(config.suspect_cusum, self.suspect_sum)
                if fired is not None:
                    new, reason = SensorHealth.SUSPECT, fired
            elif self._consistent():
                new, reason = SensorHealth.TRUSTED, "consistent"
        if new is None:
            return None
        if reason == "consistent":
            self._reset_cusum()  # the alarm is over: start accumulating afresh
        return self._enter(new, reason)

    def _enter(self, state: SensorHealth, reason: str) -> tuple[SensorHealth, str]:
        self.state = state
        self.samples_in_state = 0
        return state, reason


class HealthMonitor:
    """One ``SensorMonitor`` per sensor and measurement kind, plus the rules that protect the track (ADR 0012).

    Only position updates keep a track alive (ADR 0004), and the position sensors are the reference the others are
    judged against. So the last usable position sensor is protected: it is still tested, and its alarms are counted,
    but it is trusted. A position sensor not heard for ``max_position_silence_s`` no longer counts as usable; the
    tracker keeps this below its coast limit, so a remaining sensor is reinstated before the track is dropped.
    Other sensors start each track on probation: tested against the track, but not used to update it.
    """

    def __init__(self, config: HealthMonitorConfig, max_position_silence_s: float) -> None:
        self.config = config
        self.max_position_silence_s = max_position_silence_s
        self.monitors: dict[tuple[str, bool], SensorMonitor] = {}  # by (sensor_id, is_position)
        self.position_last_heard: dict[str, datetime] = {}
        self.transitions: list[HealthTransition] = []

    def state(self, sensor_id: str) -> SensorHealth:
        """The sensor's worst state, if it reports more than one kind of measurement."""
        return _worst(m.state for (s, _), m in self.monitors.items() if s == sensor_id)

    def _position_state(self, sensor_id: str) -> SensorHealth:
        monitor = self.monitors.get((sensor_id, True))
        return SensorHealth.TRUSTED if monitor is None else monitor.state

    def _protected(self, sensor_id: str, timestamp: datetime) -> bool:
        """No other position sensor is both usable and recently heard."""
        return not any(
            other != sensor_id
            and (timestamp - heard).total_seconds() <= self.max_position_silence_s
            and self._position_state(other) is not SensorHealth.EXCLUDED
            for other, heard in self.position_last_heard.items()
        )

    def not_for_initiation(self, timestamp: datetime, position_sensors: Iterable[str]) -> set[str]:
        """Of the sensors with positions in this scan, those that may not start a track: excluded, with an alternative.

        Without a track no innovations are tested, so the last position sensor must be able to start one.
        """
        reporting = set(position_sensors)
        for sensor_id in reporting:
            self.position_last_heard[sensor_id] = timestamp
        return {
            s for s in reporting
            if self._position_state(s) is SensorHealth.EXCLUDED and not self._protected(s, timestamp)
        }

    def start_new_track(self) -> None:
        for monitor in self.monitors.values():
            monitor.start_new_track()

    def observe(
        self, timestamp: datetime, sensor_id: str, sample: tuple[float, ...], is_position: bool
    ) -> SensorHealth:
        """Test one normalised innovation and return how to treat the sensor's measurement in this scan."""
        key = (sensor_id, is_position)
        monitor = self.monitors.get(key)
        if monitor is None:
            monitor = self.monitors[key] = SensorMonitor(sensor_id, self.config, len(sample))
        if is_position:
            self.position_last_heard[sensor_id] = timestamp
        previous = monitor.state
        changed = monitor.observe(sample, is_position and self._protected(sensor_id, timestamp))
        if changed is not None:
            self.transitions.append(HealthTransition(timestamp, sensor_id, previous, *changed))
        if not is_position and monitor.on_probation:
            return SensorHealth.EXCLUDED  # tested, but not yet used, even if it has already raised an alarm
        return monitor.state

    def report(self) -> HealthReport:
        samples: dict[str, int] = {}
        held_alarms: dict[str, int] = {}
        for (sensor_id, _), m in self.monitors.items():
            samples[sensor_id] = samples.get(sensor_id, 0) + m.samples
            held_alarms[sensor_id] = held_alarms.get(sensor_id, 0) + m.held_alarms
        return HealthReport(
            transitions=tuple(self.transitions),
            samples=samples,
            final_states={sensor_id: self.state(sensor_id) for sensor_id in samples},
            held_alarms=held_alarms,
        )
