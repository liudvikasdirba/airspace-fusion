"""Scenario families and scenario files (ADR 0008).

A ``ScenarioConfig`` is one fixed world. A ``ScenarioFamily`` is a built-in base scenario plus ranges for the
parts that change: for each seed it draws a concrete ``ScenarioConfig`` (target start, heading, speed, turn,
and the bearing sensor's placement by crossing angle and range ratio). Both offer ``for_seed(seed)``, so the
evaluation code takes either.

Draws come from their own random stream (spawn key ``FAMILY_STREAM`` of the seed), separate from the
simulator's streams, so the built-in scenarios are unaffected. Every parameter consumes exactly one draw, in a
fixed order, even when its range is a single value: fixing one parameter (as a sweep does) leaves every other
drawn parameter of a seed unchanged.

Scenario files are TOML with ``[[scenario]]`` and ``[[family]]`` tables; each starts from a ``base`` scenario
and overrides fields. Scenario fields use the model's units (metres, seconds, radians); family ranges say
their unit in their name (``_m``, ``_s``, ``_deg``, ``_mps``). Unknown keys are errors, not ignored.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Iterable, Mapping
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from pydantic import BaseModel, BeforeValidator, ConfigDict, ValidationError, field_validator, model_validator

from airspace_fusion.config import SCENARIOS, ScenarioConfig, SensorConfig, with_sensor_overrides

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.10
    import tomli as tomllib

# Spawn key of the family's random stream; ``simulate`` uses keys 0-3 of the same seed for truth and sensors.
FAMILY_STREAM = 100


class ParameterRange(BaseModel):
    """A closed interval ``[lo, hi]`` to draw uniformly from; ``lo == hi`` fixes the value."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lo: float
    hi: float

    @model_validator(mode="after")
    def _ordered(self) -> ParameterRange:
        if not (math.isfinite(self.lo) and math.isfinite(self.hi)):
            raise ValueError("range bounds must be finite")
        if self.lo > self.hi:
            raise ValueError(f"range lower bound {self.lo} is above the upper bound {self.hi}")
        return self

    def at(self, u: float) -> float:
        """The value at fraction ``u`` in [0, 1) of the range."""
        return self.lo + u * (self.hi - self.lo)


def _to_range(value: Any) -> Any:
    """Accept a number (fixed value), a ``[lo, hi]`` pair, or a mapping/ParameterRange."""
    if isinstance(value, bool):
        raise ValueError("a range must be a number or a [lo, hi] pair")
    if isinstance(value, (int, float)):
        return {"lo": value, "hi": value}
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"a range needs exactly two values [lo, hi], got {len(value)}")
        return {"lo": value[0], "hi": value[1]}
    return value


Range = Annotated[ParameterRange, BeforeValidator(_to_range)]


def _range(lo: float, hi: float) -> ParameterRange:
    return ParameterRange(lo=lo, hi=hi)


class ScenarioFamily(BaseModel):
    """A base scenario plus parameter ranges; ``for_seed`` draws one concrete world per seed.

    Geometry, all relative to the primary sensor at ``base.primary.position_m``:

    * the target starts ``start_range_m`` away, in direction ``start_azimuth_deg`` (counter-clockwise from +x);
    * it heads along the tangent (a crossing path, counter-clockwise around the primary sensor), turned by
      ``heading_offset_deg``, at ``speed_mps``; it turns at ``turn_rate_deg_s`` from ``turn_start_s`` for
      ``turn_duration_s``;
    * the bearing sensor sits ``range_ratio`` times the target's start range from the target's start point, so
      that the angle at the target between the two sensors' lines of sight is ``crossing_angle_deg`` at the
      start. When ``crossing_angle_deg`` is not set, the base scenario's bearing sensor position is kept (and
      ``range_ratio`` is unused).

    ``bearing_bias_deg`` and ``false_observations_per_scan`` (both sensors) replace the base values when set.
    Clutter stays uniform over the base scenario's surveillance area.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str = ""
    base: ScenarioConfig
    start_range_m: Range = _range(1500.0, 2500.0)
    start_azimuth_deg: Range = _range(0.0, 360.0)
    heading_offset_deg: Range = _range(-30.0, 30.0)
    speed_mps: Range = _range(10.0, 25.0)
    turn_start_s: Range = _range(40.0, 100.0)
    turn_duration_s: Range = _range(20.0, 60.0)
    turn_rate_deg_s: Range = _range(-3.0, 3.0)
    crossing_angle_deg: Range | None = None
    range_ratio: Range = _range(1.0, 1.0)
    bearing_bias_deg: Range | None = None
    false_observations_per_scan: Range | None = None

    @field_validator("base", mode="before")
    @classmethod
    def _base_by_name(cls, value: Any) -> Any:
        if isinstance(value, str):
            if value not in SCENARIOS:
                raise ValueError(f"unknown base scenario {value!r}; built-in scenarios: {', '.join(SCENARIOS)}")
            return SCENARIOS[value]
        return value

    @model_validator(mode="after")
    def _ranges_make_sense(self) -> ScenarioFamily:
        checks = [
            (self.start_range_m.lo > 0.0, "start_range_m must be positive"),
            (self.speed_mps.lo >= 0.0, "speed_mps must not be negative"),
            (self.turn_start_s.lo >= 0.0, "turn_start_s must not be negative"),
            (self.turn_duration_s.lo >= 0.0, "turn_duration_s must not be negative"),
            (self.range_ratio.lo > 0.0, "range_ratio must be positive"),
        ]
        if self.crossing_angle_deg is not None:
            checks.append((0.0 < self.crossing_angle_deg.lo and self.crossing_angle_deg.hi < 180.0,
                           "crossing_angle_deg must lie strictly between 0 and 180"))
        if self.false_observations_per_scan is not None:
            checks.append((self.false_observations_per_scan.lo >= 0.0,
                           "false_observations_per_scan must not be negative"))
        problems = [message for ok, message in checks if not ok]
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def with_fixed(self, parameter: str, value: float) -> ScenarioFamily:
        """This family with one range fixed at ``value``: every seed's other draws stay the same (a sweep point)."""
        if parameter not in SWEEPABLE_PARAMETERS:
            raise ValueError(f"cannot sweep {parameter!r}; sweepable parameters: {', '.join(SWEEPABLE_PARAMETERS)}")
        return ScenarioFamily.model_validate(self.model_dump() | {parameter: value})

    def for_seed(self, seed: int) -> ScenarioConfig:
        """The world this family draws for ``seed``: deterministic, and independent of the simulator's streams."""
        rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(FAMILY_STREAM,)))
        # One draw per parameter, always in this order (see the module docstring).
        u = iter(rng.random(11).tolist())
        start_range = self.start_range_m.at(next(u))
        azimuth = math.radians(self.start_azimuth_deg.at(next(u)))
        heading = azimuth + math.pi / 2 + math.radians(self.heading_offset_deg.at(next(u)))
        speed = self.speed_mps.at(next(u))
        turn_start = self.turn_start_s.at(next(u))
        turn_duration = self.turn_duration_s.at(next(u))
        turn_rate = math.radians(self.turn_rate_deg_s.at(next(u)))
        u_crossing, u_ratio, u_bias, u_clutter = next(u), next(u), next(u), next(u)

        px, py = self.base.primary.position_m
        sx, sy = px + start_range * math.cos(azimuth), py + start_range * math.sin(azimuth)
        secondary: dict[str, Any] = {}
        details = [f"start {start_range:.0f} m at {math.degrees(azimuth):.0f} deg", f"speed {speed:.1f} m/s",
                   f"turn {math.degrees(turn_rate):+.2f} deg/s from {turn_start:.0f} s for {turn_duration:.0f} s"]
        if self.crossing_angle_deg is not None:
            crossing = math.radians(self.crossing_angle_deg.at(u_crossing))
            distance = self.range_ratio.at(u_ratio) * start_range
            towards_primary = azimuth + math.pi  # direction from the target's start to the primary sensor
            direction = towards_primary + crossing
            secondary["position_m"] = (sx + distance * math.cos(direction), sy + distance * math.sin(direction))
            details.append(f"crossing {math.degrees(crossing):.0f} deg, range ratio {distance / start_range:.2f}")
        if self.bearing_bias_deg is not None:
            bias = math.radians(self.bearing_bias_deg.at(u_bias))
            secondary["bias"] = bias
            details.append(f"bearing bias {math.degrees(bias):.2f} deg")
        clutter: dict[str, Any] = {}
        if self.false_observations_per_scan is not None:
            clutter["false_observations_per_scan"] = self.false_observations_per_scan.at(u_clutter)
            details.append(f"{clutter['false_observations_per_scan']:.2f} false observations per scan")

        return ScenarioConfig.model_validate(
            self.base.model_dump()
            | {
                "name": self.name,
                "description": f"{self.name} seed {seed}: " + "; ".join(details),
                "initial_position_m": (sx, sy),
                "initial_velocity_mps": (speed * math.cos(heading), speed * math.sin(heading)),
                "turn_start_s": turn_start,
                "turn_end_s": turn_start + turn_duration,
                "turn_rate_rad_s": turn_rate,
                "primary": _sensor(self.base.primary, clutter),
                "secondary": _sensor(self.base.secondary, secondary | clutter),
            }
        )


# Family fields that are ranges, i.e. what a sweep can fix (``with_fixed``).
SWEEPABLE_PARAMETERS = (
    "start_range_m",
    "start_azimuth_deg",
    "heading_offset_deg",
    "speed_mps",
    "turn_start_s",
    "turn_duration_s",
    "turn_rate_deg_s",
    "crossing_angle_deg",
    "range_ratio",
    "bearing_bias_deg",
    "false_observations_per_scan",
)


def _sensor(sensor: SensorConfig, update: Mapping[str, Any]) -> SensorConfig:
    return SensorConfig.model_validate(sensor.model_dump() | dict(update))


ScenarioSource = ScenarioConfig | ScenarioFamily


def with_overrides(source: ScenarioSource, noise_multiplier: float, miss_probability: float | None) -> ScenarioSource:
    """``config.with_sensor_overrides`` for a scenario, or for a family's base scenario."""
    if isinstance(source, ScenarioFamily):
        return ScenarioFamily.model_validate(
            source.model_dump() | {"base": with_sensor_overrides(source.base, noise_multiplier, miss_probability)}
        )
    return with_sensor_overrides(source, noise_multiplier, miss_probability)


_REFERENCE = SCENARIOS["reference"]
FAMILIES: dict[str, ScenarioFamily] = {
    f.name: f
    for f in (
        ScenarioFamily(
            name="crossing_geometry",
            description="Crossing angle between the sensors' lines of sight drawn from 5 to 90 deg.",
            base=_REFERENCE,
            crossing_angle_deg=_range(5.0, 90.0),
        ),
        ScenarioFamily(
            name="range_ratio",
            description="Bearing sensor 0.5 to 10 times the primary sensor's distance from the target.",
            base=_REFERENCE,
            crossing_angle_deg=_range(60.0, 90.0),
            range_ratio=_range(0.5, 10.0),
        ),
        ScenarioFamily(
            name="manoeuvre_intensity",
            description="Turns from straight flight to 6 deg/s, either direction, for 20 to 80 s.",
            base=_REFERENCE,
            turn_rate_deg_s=_range(-6.0, 6.0),
            turn_duration_s=_range(20.0, 80.0),
            crossing_angle_deg=_range(60.0, 90.0),
        ),
        ScenarioFamily(
            name="clutter_density",
            description="0 to 5 false observations per scan on both sensors.",
            base=_REFERENCE,
            crossing_angle_deg=_range(60.0, 90.0),
            false_observations_per_scan=_range(0.0, 5.0),
        ),
        ScenarioFamily(
            name="bias_magnitude",
            description="Unmodelled bearing bias drawn from 0 to 3 deg.",
            base=_REFERENCE,
            crossing_angle_deg=_range(60.0, 90.0),
            bearing_bias_deg=_range(0.0, 3.0),
        ),
    )
}


# -- scenario files ------------------------------------------------------------------------------------------


def builtin_sources() -> dict[str, ScenarioSource]:
    """The built-in scenarios followed by the built-in families, by name."""
    sources: dict[str, ScenarioSource] = dict(SCENARIOS)
    sources.update(FAMILIES)
    return sources


class ScenarioFileError(ValueError):
    """A scenario file cannot be read or contains an invalid entry."""


SENSOR_KEYS = ("primary", "secondary")


def _check_keys(table: Mapping[str, Any], allowed: Iterable[str], where: str) -> None:
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {', '.join(unknown)}")


def _scenario_from_table(table: Mapping[str, Any], known: Mapping[str, ScenarioSource]) -> ScenarioConfig:
    table = dict(table)
    base_name = table.pop("base", None)
    _check_keys(table, ScenarioConfig.model_fields, "scenario")
    values: dict[str, Any] = {}
    if base_name is not None:
        base = known.get(base_name)
        if not isinstance(base, ScenarioConfig):
            raise ValueError(f"scenario base {base_name!r} is not a known scenario")
        values = base.model_dump()
    for key in SENSOR_KEYS:
        if key in table:
            _check_keys(table[key], SensorConfig.model_fields, f"scenario.{key}")
            values[key] = values.get(key, {}) | dict(table.pop(key))
    return ScenarioConfig.model_validate(values | table)


def _family_from_table(table: Mapping[str, Any], known: Mapping[str, ScenarioSource]) -> ScenarioFamily:
    table = dict(table)
    base = table.get("base")
    if isinstance(base, str):
        resolved = known.get(base)
        if not isinstance(resolved, ScenarioConfig):
            raise ValueError(f"family base {base!r} is not a known scenario")
        table["base"] = resolved
    return ScenarioFamily.model_validate(table)


def parse_scenario_toml(text: str, known: Mapping[str, ScenarioSource] | None = None) -> dict[str, ScenarioSource]:
    """Entries of a scenario file, by name. ``known`` names may be used as bases but not redefined."""
    known = dict(builtin_sources() if known is None else known)
    data = tomllib.loads(text)
    _check_keys(data, ("scenario", "family"), "scenario file")
    entries: dict[str, ScenarioSource] = {}
    for kind, build in (("scenario", _scenario_from_table), ("family", _family_from_table)):
        tables = data.get(kind, [])
        if not isinstance(tables, list):
            raise ValueError(f"use [[{kind}]] (an array of tables), not [{kind}]")
        for index, table in enumerate(tables, start=1):
            try:
                entry = build(table, known | entries)
            except (ValueError, ValidationError) as exc:
                raise ValueError(f"[[{kind}]] #{index} ({table.get('name', 'unnamed')}): {exc}") from exc
            if entry.name in known or entry.name in entries:
                raise ValueError(f"[[{kind}]] #{index}: the name {entry.name!r} is already defined")
            entries[entry.name] = entry
    return entries


def load_scenario_file(path: Path, known: Mapping[str, ScenarioSource] | None = None) -> dict[str, ScenarioSource]:
    try:
        return parse_scenario_toml(path.read_text(encoding="utf-8"), known)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError) as exc:
        raise ScenarioFileError(f"{path}: {exc}") from exc


def available_sources(files: Iterable[Path] = ()) -> dict[str, ScenarioSource]:
    """Built-in scenarios and families, plus the entries of each scenario file (names must be unique)."""
    sources = builtin_sources()
    for path in files:
        sources |= load_scenario_file(path, sources)
    return sources


def _toml_value(value: Any) -> str:
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(float(value)) if isinstance(value, float) else str(value)
    if isinstance(value, str):
        return json.dumps(value)  # JSON string escapes are valid TOML basic strings
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"cannot write {value!r} to TOML")


def _toml_table(header: str, values: Mapping[str, Any]) -> list[str]:
    lines = [header]
    lines += [f"{key} = {_toml_value(value)}" for key, value in values.items() if value is not None]
    return lines


def _scenario_toml(scenario: ScenarioConfig) -> list[str]:
    values = scenario.model_dump()
    sensors = {key: values.pop(key) for key in SENSOR_KEYS}
    lines = _toml_table("[[scenario]]", values)
    for key in SENSOR_KEYS:
        lines += ["", *_toml_table(f"[scenario.{key}]", sensors[key])]
    return lines


def to_toml(source: ScenarioSource) -> str:
    """A scenario file defining ``source``; ``parse_scenario_toml`` reads it back to an equal object.

    A built-in scenario is written under its own name and must be renamed before loading next to the
    built-ins. A family whose base is not built-in is preceded by that base scenario.
    """
    if isinstance(source, ScenarioConfig):
        return "\n".join(_scenario_toml(source)) + "\n"
    lines: list[str] = []
    base_is_builtin = SCENARIOS.get(source.base.name) == source.base
    if not base_is_builtin:
        lines += [*_scenario_toml(source.base), ""]
    values: dict[str, Any] = {"name": source.name, "description": source.description, "base": source.base.name}
    unset = []
    for field in ScenarioFamily.model_fields:
        value = getattr(source, field)
        if isinstance(value, ParameterRange):
            values[field] = value.lo if value.lo == value.hi else [value.lo, value.hi]
        elif value is None:
            unset.append(field)  # TOML has no null: an unset optional range is simply left out
    lines += _toml_table("[[family]]", values)
    lines += [f"# {field}: not set (keeps the base scenario's value)" for field in unset]
    return "\n".join(lines) + "\n"
