"""End-to-end pipeline tests (L4: I1, I3-I6). I2 (extreme settings sweep) lives in test_sweep.py."""

from __future__ import annotations

import ast
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

import airspace_fusion
from airspace_fusion.config import SCENARIOS, TrackerConfig, default_primary, default_secondary
from airspace_fusion.estimation import Disposition
from airspace_fusion.evaluation import Mode, Verdict, compare_modes, run_single

REFERENCE = SCENARIOS["reference"]
PACKAGE_DIR = Path(airspace_fusion.__file__).parent


@pytest.mark.slow
def test_cli_output_reproducible_across_processes():
    """I1: two independent interpreter processes produce identical results (timing excluded)."""
    cmd = [sys.executable, "-m", "airspace_fusion.cli", "compare", "--scenario", "reference", "--seeds", "3"]

    def run() -> str:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=True)
        return "\n".join(line for line in done.stdout.splitlines() if not line.startswith("processing_time_s"))

    first, second = run(), run()
    assert "VERDICT" in first
    assert first == second


def test_only_false_observations():
    """I3: sensors that never detect the object, only clutter, must not crash or invent a track."""
    scenario = REFERENCE.model_copy(update={
        "primary": default_primary(detection_probability=0.0, false_observations_per_scan=1.0),
        "secondary": default_secondary(detection_probability=0.0, false_observations_per_scan=1.0),
    })
    summary = compare_modes(scenario, TrackerConfig(), seeds=[0, 1, 2])
    for mode in Mode:
        coverage = summary.metrics[mode]["coverage"]
        assert coverage.mean < 0.1, coverage  # at most brief tracks on coincidentally aligned clutter
    if all(summary.metrics[m]["position_rmse_m"].n == 0 for m in Mode):
        assert summary.verdict is Verdict.INSUFFICIENT_DATA


def test_single_scan_scenario():
    """I4a: a one-scan scenario cannot initiate a track; metrics degrade gracefully."""
    scenario = REFERENCE.model_copy(update={"duration_s": 0.4})
    assert scenario.num_steps == 1
    run = run_single(scenario, TrackerConfig(), seed=0)
    for mode in Mode:
        m = run.modes[mode].metrics
        assert m.coverage == 0.0 and math.isnan(m.position_rmse_m) and m.track_initiations == 0
    assert compare_modes(scenario, TrackerConfig(), seeds=[0, 1]).verdict is Verdict.INSUFFICIENT_DATA


def test_sparse_bearing_sensor():
    """I4b: a bearing sensor scanning every 7th step is used only at those scans."""
    scenario = REFERENCE.model_copy(update={"secondary": default_secondary(scan_interval_steps=7)})
    run = run_single(scenario, TrackerConfig(), seed=1)
    fusion = run.modes[Mode.FUSION].result
    bearing_sensor = scenario.secondary.sensor_id
    t0 = run.simulation.truth.timestamps[0]
    used_at = [
        (e.timestamp - t0).total_seconds() for e in fusion.estimates if bearing_sensor in e.updated_sensors
    ]
    assert used_at, "bearing updates expected"
    assert all(t % 7 == 0 for t in used_at)
    assert run.modes[Mode.FUSION].metrics.coverage == run.modes[Mode.BASELINE].metrics.coverage


def test_all_records_corrupted():
    """I5: every record malformed - all rejected by the schema, attributed to sensors, nothing tracked."""
    scenario = REFERENCE.model_copy(update={"corrupted_record_probability": 1.0})
    run = run_single(scenario, TrackerConfig(), seed=0)
    assert run.measurements == []
    assert len(run.invalid_records) == len(run.simulation.records) > 0
    primary = scenario.primary.sensor_id
    n_primary = sum(r.sensor_id == primary for r in run.invalid_records)
    assert run.modes[Mode.BASELINE].metrics.invalid_observations == n_primary
    assert run.modes[Mode.FUSION].metrics.invalid_observations == len(run.invalid_records)
    for mode in Mode:
        result = run.modes[mode].result
        assert not any(e.has_track for e in result.estimates)
        assert result.dispositions == {}
        assert run.modes[mode].metrics.coverage == 0.0


def test_corrupted_records_never_reach_tracker():
    scenario = REFERENCE.model_copy(update={"corrupted_record_probability": 0.3})
    run = run_single(scenario, TrackerConfig(), seed=4)
    corrupted = run.simulation.truth.corrupted_record_ids
    for mode in Mode:
        assert not corrupted & set(run.modes[mode].result.dispositions)
    assert set(run.modes[Mode.FUSION].result.dispositions.values()) <= set(vars(Disposition).values())


# -- I6: module dependency rules --------------------------------------------------------------

# Which internal modules each module may import. Ground truth (simulator) and tracker output
# (estimation) meet only in evaluation; the estimator never sees the simulator.
ALLOWED_IMPORTS = {
    "schemas": set(),
    "config": {"schemas"},
    "simulator": {"config", "schemas"},
    "estimation": {"config", "schemas"},
    "estimation_stonesoup": {"config", "estimation", "schemas"},
    "evaluation": {"config", "estimation", "schemas", "simulator"},
    "learning": {"config", "evaluation"},
    "agents": {"config", "evaluation", "learning"},
    "cli": {"agents", "config", "evaluation", "learning"},
    "app": {"config", "estimation", "evaluation", "schemas"},
}


def internal_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return {m.split(".")[1] for m in names if re.match(r"airspace_fusion\.\w+", m)}


def test_every_module_has_a_dependency_rule():
    modules = {p.stem for p in PACKAGE_DIR.glob("*.py") if p.stem != "__init__"}
    assert modules == set(ALLOWED_IMPORTS)


@pytest.mark.parametrize("module", sorted(ALLOWED_IMPORTS))
def test_module_dependencies(module):
    actual = internal_imports(PACKAGE_DIR / f"{module}.py")
    assert actual <= ALLOWED_IMPORTS[module], f"{module} imports {actual - ALLOWED_IMPORTS[module]}"
