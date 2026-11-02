"""Per-seed run-time benchmark (slow): records how long one seed takes, so speed regressions show up.

Each run appends one JSON line to ``data/benchmarks/run_times.jsonl`` (gitignored; override the path
with ``AIRSPACE_FUSION_BENCHMARK_LOG``), with per-scenario median, 90th percentile and fastest seed times
and what was measured: git commit, package and Python versions, machine. CI uploads the file as an artifact.

A scenario whose median is more than ``REGRESSION_FACTOR`` times the last entry from the same machine and
Python version raises a warning rather than a failure: timings on a shared machine are noisy. The hard
limits are the budgets in ``test_performance.py``.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pytest

import airspace_fusion
from airspace_fusion.config import SCENARIOS, TrackerConfig
from airspace_fusion.evaluation import run_single

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "data" / "benchmarks" / "run_times.jsonl"
SEEDS = range(12)
REGRESSION_FACTOR = 1.5


class PerformanceRegressionWarning(UserWarning):
    pass


def git(*args: str) -> str | None:
    try:
        done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def seed_times_ms(scenario_name: str) -> list[float]:
    scenario, tracker = SCENARIOS[scenario_name], TrackerConfig()
    run_single(scenario, tracker, SEEDS[0])  # warm-up: caches, imports, first-call costs
    times = []
    for seed in SEEDS:
        start = time.perf_counter()
        run_single(scenario, tracker, seed)
        times.append((time.perf_counter() - start) * 1000.0)
    return times


def summarise(times: list[float]) -> dict[str, float]:
    ordered = sorted(times)
    return {
        "median_ms": round(statistics.median(ordered), 2),
        "p90_ms": round(ordered[min(len(ordered) - 1, round(0.9 * (len(ordered) - 1)))], 2),
        "min_ms": round(ordered[0], 2),
    }


def same_setup(a: dict, b: dict) -> bool:
    return all(a.get(k) == b.get(k) for k in ("machine", "python"))


def previous_entry(log: Path, current: dict) -> dict | None:
    if not log.exists():
        return None
    for line in reversed(log.read_text(encoding="utf-8").splitlines()):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue  # a damaged line must not stop the benchmark
        if same_setup(entry, current):
            return entry
    return None


def regressions(previous: dict | None, current: dict) -> list[str]:
    if previous is None:
        return []
    slower = []
    for name, now in current["scenarios"].items():
        before = previous.get("scenarios", {}).get(name)
        if before and now["median_ms"] > REGRESSION_FACTOR * before["median_ms"]:
            slower.append(f"{name}: median {now['median_ms']} ms vs {before['median_ms']} ms "
                          f"at {str(previous.get('commit'))[:7]}")
    return slower


def test_record_per_seed_run_time():
    log = Path(os.environ.get("AIRSPACE_FUSION_BENCHMARK_LOG", DEFAULT_LOG))
    entry = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),  # measured with uncommitted changes
        "package_version": airspace_fusion.__version__,
        "python": platform.python_version(),
        "machine": f"{platform.node()} | {platform.system()} {platform.machine()} | {os.cpu_count()} CPUs",
        "seeds": len(SEEDS),
        "scenarios": {name: summarise(seed_times_ms(name)) for name in SCENARIOS},
    }
    previous = previous_entry(log, entry)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    print(json.dumps(entry["scenarios"], indent=1), file=sys.stderr)

    for name, summary in entry["scenarios"].items():
        assert 0.0 < summary["min_ms"] <= summary["median_ms"] <= summary["p90_ms"], (name, summary)
    for message in regressions(previous, entry):
        warnings.warn(f"slower than the last benchmark on this machine: {message}", PerformanceRegressionWarning,
                      stacklevel=1)


def test_regression_detection():
    base = {"machine": "m", "python": "3.12", "commit": "abc1234",
            "scenarios": {"reference": {"median_ms": 50.0}}}
    assert regressions(base, base | {"scenarios": {"reference": {"median_ms": 70.0}}}) == []
    slower = regressions(base, base | {"scenarios": {"reference": {"median_ms": 80.0}}})
    assert len(slower) == 1 and "abc1234" in slower[0]
    assert regressions(None, base) == []


def test_previous_entry_matches_machine_and_skips_damaged_lines(tmp_path):
    log = tmp_path / "run_times.jsonl"
    mine = {"machine": "m", "python": "3.12", "scenarios": {}}
    log.write_text("\n".join([json.dumps(mine | {"commit": "old"}), "not json",
                              json.dumps({"machine": "other", "python": "3.12", "commit": "x"})]) + "\n",
                   encoding="utf-8")
    assert previous_entry(log, mine)["commit"] == "old"
    assert previous_entry(tmp_path / "missing.jsonl", mine) is None
