"""Results files: every ``compare`` and ``sweep`` run, saved with what is needed to reproduce it.

A record is JSON with the command, the scenarios or families (as data, not names, so a later change to a
built-in cannot silently change what a record means), the tracker settings, the seeds, every metric summary,
and provenance: package version, git commit (and whether the working tree had uncommitted changes), Python
version and the command line. ``config_hash`` is a SHA-256 over what determines the numbers (command, sources,
sweep points, tracker, seeds and the Holm family), so two records with the same hash must agree. ``reproduce``
re-runs a record and lists every metric that differs.

Each entry stores its Holm-adjusted p-value and, at record level, ``multiple_comparisons`` holds the raw
p-values of every comparison in the command run and the positions of this record's entries among them. A
``compare`` with a rule book can split one run over several records, so the whole family is stored and
``reproduce`` can redo the correction exactly. The family decides the adjusted p-values and verdicts, so it is
part of ``config_hash`` (format 3); format 2 left it out.

Wall-clock ``processing_time_s`` is recorded but never compared.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from collections.abc import Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from airspace_fusion import __version__
from airspace_fusion.config import ScenarioConfig, TrackerConfig
from airspace_fusion.evaluation import ComparisonSummary, compare_modes, holm_adjust, with_adjusted_p
from airspace_fusion.scenarios import ScenarioFamily, ScenarioSource

RESULTS_FORMAT = 3
NON_DETERMINISTIC = ("processing_time_s",)
TOLERANCE = 1e-9


@dataclass(frozen=True)
class ResultEntry:
    source: ScenarioSource
    summary: ComparisonSummary
    point: dict[str, float] | None = None  # the swept parameter and its value, for sweep entries


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def summary_to_dict(summary: ComparisonSummary) -> dict[str, Any]:
    return {
        "scenario": summary.scenario,
        "seeds": list(summary.seeds),
        "metrics": {
            mode.value: {
                name: {"mean": _finite_or_none(m.mean), "std": _finite_or_none(m.std), "n": m.n}
                for name, m in metrics.items()
            }
            for mode, metrics in summary.metrics.items()
        },
        "rmse_delta_mean_m": _finite_or_none(summary.rmse_delta_mean_m),
        "rmse_delta_ci95_m": [_finite_or_none(v) for v in summary.rmse_delta_ci95_m],
        "rmse_relative_change": _finite_or_none(summary.rmse_relative_change),
        "rmse_p_value": _finite_or_none(summary.rmse_p_value),
        "rmse_p_adjusted": None if summary.rmse_p_adjusted is None else _finite_or_none(summary.rmse_p_adjusted),
        "verdict": summary.verdict.value,
    }


def source_to_dict(source: ScenarioSource) -> dict[str, Any]:
    kind = "family" if isinstance(source, ScenarioFamily) else "scenario"
    return {"kind": kind, "config": source.model_dump(mode="json")}


def source_from_dict(data: dict[str, Any]) -> ScenarioSource:
    if data.get("kind") == "family":
        return ScenarioFamily.model_validate(data["config"])
    if data.get("kind") == "scenario":
        return ScenarioConfig.model_validate(data["config"])
    raise ValueError(f"unknown source kind {data.get('kind')!r}")


def _git(*args: str) -> str | None:
    try:
        done = subprocess.run(["git", *args], cwd=Path(__file__).parent, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def config_hash(record: dict[str, Any]) -> str:
    """SHA-256 over what determines a record's numbers."""
    determining = {
        "command": record["command"],
        "tracker": record["tracker"],
        "seeds": record["seeds"],
        "entries": [{"source": e["source"], "point": e["point"]} for e in record["entries"]],
        # The other comparisons of the run set this record's Holm-adjusted p-values and verdicts.
        "multiple_comparisons": record["multiple_comparisons"],
    }
    return hashlib.sha256(json.dumps(determining, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ComparisonFamily:
    """The Holm family a record's entries belong to: every raw p-value of the command run, in run order."""

    p_values: list[float]
    positions: list[int]  # where this record's entries sit in ``p_values``


def build_record(
    command: str,
    entries: Sequence[ResultEntry],
    tracker: TrackerConfig,
    seeds: Sequence[int],
    argv: Sequence[str],
    family: ComparisonFamily | None = None,
) -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    record: dict[str, Any] = {
        "format_version": RESULTS_FORMAT,
        "command": command,
        "created": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "package_version": __version__,
        "git_commit": commit,
        "git_dirty": None if commit is None else bool(_git("status", "--porcelain", "--untracked-files=no")),
        "python": platform.python_version(),
        "argv": list(argv),
        "tracker": tracker.model_dump(mode="json"),
        "seeds": list(seeds),
        "entries": [
            {"source": source_to_dict(e.source), "point": e.point, "summary": summary_to_dict(e.summary)}
            for e in entries
        ],
        "multiple_comparisons": None if family is None else {
            "method": "holm",
            "p_values": [_finite_or_none(p) for p in family.p_values],
            "positions": list(family.positions),
        },
    }
    record["config_hash"] = config_hash(record)
    return record


def save_record(record: dict[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(record["created"]).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / f"{stamp}-{record['command']}-{record['config_hash'][:12]}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


class ResultsFileError(ValueError):
    """A results file cannot be read, or was written in an unknown format."""


def load_record(path: Path) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResultsFileError(f"cannot read results file {path}: {exc}") from exc
    if not isinstance(record, dict) or record.get("format_version") != RESULTS_FORMAT:
        raise ResultsFileError(f"{path}: not a results file in format {RESULTS_FORMAT}")
    try:
        matches = record.get("config_hash") == config_hash(record)
    except (KeyError, TypeError) as exc:
        raise ResultsFileError(f"{path}: results file is missing {exc}") from exc
    if not matches:
        raise ResultsFileError(f"{path}: config_hash does not match the recorded configuration (file was edited?)")
    _check_family(record, path)
    return record


def _check_family(record: dict[str, Any], path: Path) -> None:
    """The Holm family must place every entry exactly once, so that ``reproduce`` checks them all."""
    family = record["multiple_comparisons"]
    if family is None:
        return
    p_values, positions = family.get("p_values"), family.get("positions")
    if not (isinstance(p_values, list) and isinstance(positions, list)
            and all(type(i) is int for i in positions)):
        raise ResultsFileError(f"{path}: multiple_comparisons needs lists p_values and positions")
    if len(positions) != len(record["entries"]):
        raise ResultsFileError(f"{path}: multiple_comparisons has {len(positions)} positions for "
                               f"{len(record['entries'])} entries")
    if len(set(positions)) != len(positions) or not all(0 <= i < len(p_values) for i in positions):
        raise ResultsFileError(f"{path}: multiple_comparisons positions must be distinct indices into p_values")


def _differences(expected: Any, actual: Any, where: str) -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            return [f"{where}: keys differ ({sorted(expected)} vs {sorted(actual)})"]
        found = []
        for key in expected:
            if key in NON_DETERMINISTIC:
                continue
            found += _differences(expected[key], actual[key], f"{where}.{key}")
        return found
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{where}: length {len(expected)} vs {len(actual)}"]
        return [d for i, (e, a) in enumerate(zip(expected, actual)) for d in _differences(e, a, f"{where}[{i}]")]
    if isinstance(expected, float) and isinstance(actual, float):
        if math.isclose(expected, actual, rel_tol=TOLERANCE, abs_tol=TOLERANCE):
            return []
    elif expected == actual:
        return []
    return [f"{where}: recorded {expected!r}, now {actual!r}"]


def reproduce(record: dict[str, Any], executor: Executor | None = None) -> list[str]:
    """Re-run every entry of a record; returns the metrics that differ (empty when it reproduces)."""
    tracker = TrackerConfig.model_validate(record["tracker"])
    seeds = record["seeds"]
    summaries = [compare_modes(source_from_dict(e["source"]), tracker, seeds, executor) for e in record["entries"]]
    family = record["multiple_comparisons"]
    if family is not None:
        # Redo the Holm correction with this record's re-run p-values in their places in the recorded family.
        p_values = [math.nan if p is None else p for p in family["p_values"]]
        for position, summary in zip(family["positions"], summaries, strict=True):
            p_values[position] = summary.rmse_p_value
        adjusted = holm_adjust(p_values)
        summaries = [with_adjusted_p(s, adjusted[pos]) for pos, s in zip(family["positions"], summaries, strict=True)]
    found = []
    for index, (entry, summary) in enumerate(zip(record["entries"], summaries, strict=True)):
        found += _differences(entry["summary"], summary_to_dict(summary), f"entries[{index}]")
    return found
