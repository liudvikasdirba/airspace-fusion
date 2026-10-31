"""Learning loop: run logs, evaluation, lessons and the learned rule book.

The loop turns evaluation results into reusable tracker settings:

    logs -> evaluation (score, gaps) -> lessons -> learning update (rule book) -> next run

* **Score**: mean position RMSE of the modes in use, inflated by how far 95 % containment
  is from 0.95, so a candidate cannot win by becoming over-confident.
* **Gaps**: measurable shortfalls (over- or under-confidence, low coverage, track restarts,
  fusion making things worse) that steer which candidate settings are worth trying.
* **Lessons**: one per tried change, kept whether accepted or rejected. Settings already tried
  are not proposed again, so later runs start from what earlier runs learned. Only the newest
  ``MAX_LESSONS_PER_SCENARIO`` lessons per scenario are kept, so the rule book stays bounded and
  long-forgotten rejections can eventually be tried again.
* **Rule book**: per-scenario ``TrackerConfig`` defaults and fusion policy, persisted as
  JSON. Only lessons accepted on held-out seeds are applied, within ``TUNABLE_BOUNDS``.

Scoring uses ground truth, so it only exists in simulation. The tracker itself is unchanged
and still never sees truth: the rule book only chooses its settings.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from airspace_fusion import __version__
from airspace_fusion.config import TrackerConfig
from airspace_fusion.evaluation import ComparisonSummary, Mode, Verdict

# Version of the rule book's JSON layout. Increase it on any incompatible change, so that an
# old file fails to load with a clear error instead of being misread.
RULE_BOOK_FORMAT = 1

TARGET_CONTAINMENT = 0.95
# Score penalty per unit of |containment - 0.95|: one percentage point costs as much as 5 % RMSE.
CONTAINMENT_WEIGHT = 5.0
# Containment within this distance of the target is not reported as a gap.
CONTAINMENT_TOLERANCE = 0.02
MIN_COVERAGE = 0.95
# Mean track initiations per run above this means tracks are lost and restarted.
MAX_TRACK_INITIATIONS = 1.05
# A candidate must beat the incumbent's held-out score by this relative margin.
MIN_SCORE_IMPROVEMENT = 0.02
# ...and may not lose more than this much coverage in any scored mode.
MAX_COVERAGE_LOSS = 0.01

# Lesson memory kept per scenario; older lessons are dropped by the learning update.
MAX_LESSONS_PER_SCENARIO = 100

# Constraints on what the learning update may change, and the allowed range of each setting.
TUNABLE_BOUNDS: dict[str, tuple[float, float]] = {
    "process_noise_m2_s3": (0.05, 5.0),
    "gate_probability": (0.99, 0.9999),
    "max_coast_s": (4.0, 20.0),
}


class GapKind(str, Enum):
    OVER_CONFIDENT = "over_confident"
    UNDER_CONFIDENT = "under_confident"
    LOW_COVERAGE = "low_coverage"
    TRACK_RESTARTS = "track_restarts"
    FUSION_HARMFUL = "fusion_harmful"
    STEP_FAILED = "step_failed"


@dataclass(frozen=True)
class Gap:
    kind: GapKind
    scenario: str
    mode: Mode | None
    value: float
    target: float

    def describe(self) -> str:
        where = f" ({self.mode.value})" if self.mode else ""
        return f"{self.kind.value}{where}: {self.value:.3g} vs target {self.target:.3g}"


def scored_modes(fusion_enabled: bool) -> tuple[Mode, ...]:
    return (Mode.BASELINE, Mode.FUSION) if fusion_enabled else (Mode.BASELINE,)


def score(summary: ComparisonSummary, modes: Sequence[Mode]) -> float:
    """Lower is better. ``inf`` when any scored mode never formed a track."""
    terms = []
    for mode in modes:
        rmse = summary.metrics[mode]["position_rmse_m"].mean
        containment = summary.metrics[mode]["containment_95"].mean
        if math.isnan(rmse) or math.isnan(containment):
            return math.inf
        terms.append(rmse * (1.0 + CONTAINMENT_WEIGHT * abs(containment - TARGET_CONTAINMENT)))
    return sum(terms) / len(terms)


def find_gaps(summary: ComparisonSummary, modes: Sequence[Mode]) -> list[Gap]:
    gaps: list[Gap] = []
    for mode in modes:
        m = summary.metrics[mode]
        containment = m["containment_95"].mean
        if containment < TARGET_CONTAINMENT - CONTAINMENT_TOLERANCE:
            gaps.append(Gap(GapKind.OVER_CONFIDENT, summary.scenario, mode, containment, TARGET_CONTAINMENT))
        elif containment > TARGET_CONTAINMENT + CONTAINMENT_TOLERANCE:
            gaps.append(Gap(GapKind.UNDER_CONFIDENT, summary.scenario, mode, containment, TARGET_CONTAINMENT))
        if m["coverage"].mean < MIN_COVERAGE:
            gaps.append(Gap(GapKind.LOW_COVERAGE, summary.scenario, mode, m["coverage"].mean, MIN_COVERAGE))
        if m["track_initiations"].mean > MAX_TRACK_INITIATIONS:
            gaps.append(Gap(GapKind.TRACK_RESTARTS, summary.scenario, mode, m["track_initiations"].mean, 1.0))
    if Mode.FUSION in modes and summary.verdict is Verdict.WORSENS:
        gaps.append(Gap(GapKind.FUSION_HARMFUL, summary.scenario, Mode.FUSION, summary.rmse_relative_change, 0.0))
    return gaps


def _clamp(name: str, value: float) -> float:
    lo, hi = TUNABLE_BOUNDS[name]
    return min(max(value, lo), hi)


def _round(value: float) -> float:
    return float(f"{value:.4g}")


def same_settings(a: TrackerConfig, b: TrackerConfig) -> bool:
    return all(math.isclose(getattr(a, k), getattr(b, k), rel_tol=1e-6) for k in TUNABLE_BOUNDS)


def propose_candidates(
    tracker: TrackerConfig, gaps: Iterable[Gap], tried: Sequence[TrackerConfig], max_candidates: int
) -> list[TrackerConfig]:
    """Gap-directed neighbours of ``tracker``, within bounds, excluding settings already tried.

    Over-confidence asks for more process noise, under-confidence for less; lost tracks ask for a
    longer coast and a wider gate. With no gaps, a small step either way explores RMSE alone.
    """
    kinds = {g.kind for g in gaps}
    q = tracker.process_noise_m2_s3
    changes: list[dict[str, float]] = []
    if GapKind.OVER_CONFIDENT in kinds:
        changes += [{"process_noise_m2_s3": q * 1.5}, {"process_noise_m2_s3": q * 2.25}]
    if GapKind.UNDER_CONFIDENT in kinds:
        changes += [{"process_noise_m2_s3": q / 1.5}, {"process_noise_m2_s3": q / 2.25}]
    if kinds & {GapKind.TRACK_RESTARTS, GapKind.LOW_COVERAGE}:
        changes += [{"max_coast_s": tracker.max_coast_s * 1.5},
                    {"gate_probability": 1.0 - (1.0 - tracker.gate_probability) / 3.0}]
    if not kinds & {GapKind.OVER_CONFIDENT, GapKind.UNDER_CONFIDENT}:
        changes += [{"process_noise_m2_s3": q * 1.25}, {"process_noise_m2_s3": q / 1.25}]

    candidates: list[TrackerConfig] = []
    for change in changes:
        update = {k: _round(_clamp(k, v)) for k, v in change.items()}
        candidate = TrackerConfig.model_validate(tracker.model_dump() | update)
        if any(same_settings(candidate, c) for c in (tracker, *candidates, *tried)):
            continue
        candidates.append(candidate)
        if len(candidates) == max_candidates:
            break
    return candidates


class LessonAction(str, Enum):
    SET_TRACKER = "set_tracker"
    DISABLE_FUSION = "disable_fusion"


class Lesson(BaseModel):
    """One tried change and its held-out outcome. Rejected lessons are kept as negative memory."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    scenario: str
    action: LessonAction
    triggers: tuple[GapKind, ...]
    tracker: TrackerConfig | None = None  # the candidate settings, for SET_TRACKER
    accepted: bool
    reason: str
    holdout_score_before: float | None = None  # None when not finite
    holdout_score_after: float | None = None

    @model_validator(mode="after")
    def _tracker_matches_action(self) -> Lesson:
        if (self.action is LessonAction.SET_TRACKER) != (self.tracker is not None):
            raise ValueError("a tracker is required for set_tracker lessons and only for them")
        return self


def finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


class RuleBookError(ValueError):
    """A rule book file is missing (when required) or cannot be read."""


def _newest_per_scenario(lessons: Sequence[Lesson], limit: int) -> tuple[Lesson, ...]:
    kept: list[Lesson] = []
    counts: dict[str, int] = {}
    for lesson in reversed(lessons):
        counts[lesson.scenario] = counts.get(lesson.scenario, 0) + 1
        if counts[lesson.scenario] <= limit:
            kept.append(lesson)
    return tuple(reversed(kept))


class RuleBook(BaseModel):
    """Learned defaults and policies, plus the lesson memory they came from."""

    model_config = ConfigDict(frozen=True)

    format_version: Literal[1] = RULE_BOOK_FORMAT
    version: int = Field(default=0, ge=0)  # increases with every learning update that changes a rule
    package_version: str = __version__  # airspace-fusion release that last wrote this rule book
    trackers: dict[str, TrackerConfig] = Field(default_factory=dict)
    fusion_disabled: tuple[str, ...] = ()
    lessons: tuple[Lesson, ...] = ()

    @model_validator(mode="after")
    def _within_bounds(self) -> RuleBook:
        for scenario, tracker in self.trackers.items():
            for name, (lo, hi) in TUNABLE_BOUNDS.items():
                if not lo <= getattr(tracker, name) <= hi:
                    raise ValueError(f"{scenario}: {name}={getattr(tracker, name)} outside [{lo}, {hi}]")
        return self

    def tracker_for(self, scenario: str) -> TrackerConfig:
        return self.trackers.get(scenario, TrackerConfig())

    def fusion_enabled(self, scenario: str) -> bool:
        return scenario not in self.fusion_disabled

    def tried_trackers(self, scenario: str) -> list[TrackerConfig]:
        """Settings not worth proposing again: every tried candidate, and the factory default once replaced."""
        tried = [lesson.tracker for lesson in self.lessons if lesson.scenario == scenario and lesson.tracker]
        if scenario in self.trackers:
            tried.append(TrackerConfig())
        return tried

    def with_lessons(self, lessons: Sequence[Lesson]) -> RuleBook:
        """Learning update: apply the accepted lessons (re-validated against the bounds) and remember the
        newest ``MAX_LESSONS_PER_SCENARIO`` lessons per scenario."""
        trackers = dict(self.trackers)
        disabled = list(self.fusion_disabled)
        for lesson in lessons:
            if not lesson.accepted:
                continue
            if lesson.action is LessonAction.SET_TRACKER:
                trackers[lesson.scenario] = lesson.tracker
            elif lesson.scenario not in disabled:
                disabled.append(lesson.scenario)
        changed = trackers != self.trackers or disabled != list(self.fusion_disabled)
        return RuleBook.model_validate(
            dict(
                version=self.version + int(changed),
                package_version=__version__,
                trackers=trackers,
                fusion_disabled=tuple(disabled),
                lessons=_newest_per_scenario((*self.lessons, *lessons), MAX_LESSONS_PER_SCENARIO),
            )
        )

    @classmethod
    def load(cls, path: Path, must_exist: bool = False) -> RuleBook:
        """Read a rule book. A missing file is an empty rule book unless ``must_exist`` is set."""
        if not path.exists():
            if must_exist:
                raise RuleBookError(f"rule book {path} does not exist")
            return cls()
        try:
            return cls.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValidationError) as exc:
            raise RuleBookError(
                f"cannot read rule book {path} (delete it to start learning from scratch): {exc}"
            ) from exc

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, float):
        return finite_or_none(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class RunLog:
    """Structured record of a learning run, kept in memory and optionally appended to a JSONL file."""

    def __init__(self, run_id: str, path: Path | None = None) -> None:
        self.run_id = run_id
        self.path = path
        self.entries: list[dict[str, Any]] = []
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: str, **fields: Any) -> dict[str, Any]:
        entry = {"time": datetime.now(timezone.utc).isoformat(), "run_id": self.run_id, "event": event}
        entry |= {k: _jsonable(v) for k, v in fields.items()}
        self.entries.append(entry)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, allow_nan=False) + "\n")
        return entry

    def events(self, event: str) -> list[dict[str, Any]]:
        return [e for e in self.entries if e["event"] == event]
