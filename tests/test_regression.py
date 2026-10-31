"""Golden-file regression: locks simulator output and per-seed metrics for every scenario.

A failure means results changed. If the change is intended, regenerate with
``pytest tests/test_regression.py --update-golden`` and review the JSON diff with the code change.
Processing time is excluded because it is not deterministic.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path

import pytest

from airspace_fusion.config import SCENARIOS, TrackerConfig
from airspace_fusion.evaluation import Mode, run_single
from airspace_fusion.simulator import simulate

GOLDEN = Path(__file__).parent / "golden" / "regression.json"
SEEDS = range(5)
TOLERANCE = 1e-9
NON_DETERMINISTIC = {"processing_time_s"}


# Significant digits kept when fingerprinting simulator output. Exact float text is not portable:
# atan2, sin, cos and log may differ in the last bit between C maths libraries (e.g. glibc vs MSVC).
DIGEST_SIGNIFICANT_DIGITS = 10


def canonical(value: object) -> object:
    """JSON-friendly form of a record with floats rounded, so the digest is the same on every platform."""
    if isinstance(value, float):
        return f"{value:.{DIGEST_SIGNIFICANT_DIGITS}g}"
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)  # datetimes


def records_digest(scenario_name: str, seed: int) -> str:
    records = simulate(SCENARIOS[scenario_name], seed).records
    text = json.dumps(canonical(records), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_digest_ignores_last_bit_differences():
    record = {"value": (0.1 + 0.2, 1.0), "when": "t"}
    nudged = {"value": (math.nextafter(0.1 + 0.2, 1.0), 1.0), "when": "t"}
    assert canonical(record) == canonical(nudged)
    assert canonical(record) != canonical({"value": (0.3001, 1.0), "when": "t"})


def current_results() -> dict:
    tracker = TrackerConfig()
    metrics: dict = {}
    for name, scenario in SCENARIOS.items():
        for seed in SEEDS:
            run = run_single(scenario, tracker, seed)
            for mode in Mode:
                values = dataclasses.asdict(run.modes[mode].metrics)
                metrics.setdefault(name, {}).setdefault(str(seed), {})[mode.value] = {
                    k: v for k, v in values.items() if k not in NON_DETERMINISTIC
                }
    return {
        "records_sha256": {name: records_digest(name, 0) for name in SCENARIOS},
        "metrics": metrics,
    }


@pytest.fixture(scope="module")
def results(request) -> dict:
    current = current_results()
    if request.config.getoption("--update-golden"):
        GOLDEN.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return current


@pytest.fixture(scope="module")
def golden(results) -> dict:
    if not GOLDEN.exists():
        pytest.fail(f"{GOLDEN} missing; generate it with --update-golden")
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def test_simulator_records_unchanged(results, golden):
    assert results["records_sha256"] == golden["records_sha256"]


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_metrics_unchanged(results, golden, scenario):
    expected = golden["metrics"][scenario]
    actual = results["metrics"][scenario]
    assert actual.keys() == expected.keys()
    mismatches = []
    for seed, modes in expected.items():
        for mode, values in modes.items():
            assert actual[seed][mode].keys() == values.keys(), f"metric set changed for {scenario}/{seed}/{mode}"
            for metric, want in values.items():
                got = actual[seed][mode][metric]
                same = (math.isnan(want) and math.isnan(got)) or math.isclose(
                    got, want, rel_tol=TOLERANCE, abs_tol=TOLERANCE
                )
                if not same:
                    mismatches.append(f"seed {seed} {mode} {metric}: expected {want!r}, got {got!r}")
    assert not mismatches, "\n".join(mismatches)
