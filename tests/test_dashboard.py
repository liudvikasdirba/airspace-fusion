"""Automated dashboard tests (L7) using Streamlit's headless AppTest harness (slow)."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import airspace_fusion
from airspace_fusion.config import SCENARIOS, TrackerConfig, with_sensor_overrides
from airspace_fusion.evaluation import METRIC_NAMES, Mode, compare_modes, run_single

pytestmark = pytest.mark.slow

APP = str(Path(airspace_fusion.__file__).parent / "app.py")
TRAJECTORY_TRACES = {
    "Bearing observations",
    "Position observations",
    "False position observations (truth label)",
    "Ground truth",
    "Baseline (position sensor only)",
    "baseline 95% ellipse (every 10 s)",
    "Fusion (position + bearing)",
    "fusion 95% ellipse (every 10 s)",
}


def start() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def figures(at: AppTest) -> tuple[dict, dict]:
    trajectory, error = (json.loads(c.proto.spec) for c in at.get("plotly_chart"))
    return trajectory, error


def trace_names(fig: dict) -> set[str]:
    return {t["name"] for t in fig["data"] if t.get("name")}


def widget(at: AppTest, kind: str, label: str):
    matches = [w for w in getattr(at, kind) if w.label.startswith(label)]
    assert len(matches) == 1, (kind, label, [w.label for w in getattr(at, kind)])
    return matches[0]


def click(at: AppTest, label: str) -> AppTest:
    widget(at, "button", label).click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def showing(at: AppTest) -> str:
    return next(m.value for m in at.markdown if m.value.startswith("**Showing:**"))


def test_default_page_renders_complete_simulation():
    at = start()
    trajectory, error = figures(at)
    assert trace_names(trajectory) == TRAJECTORY_TRACES
    assert {"baseline: actual position error", "fusion: actual position error"} <= trace_names(error)
    table = at.dataframe[0].value
    assert list(table.columns) == ["baseline", "fusion", "delta"]
    assert len(table) == len(METRIC_NAMES)
    assert "scenario `reference`, seed 0" in showing(at)
    assert [e.label for e in at.expander] == ["Limitations of this simulation"]
    # Equal axis scaling, so distances and ellipses are not distorted.
    assert trajectory["layout"]["yaxis"]["scaleanchor"] == "x"


@pytest.mark.parametrize("view, hidden", [("Baseline only", "fusion"), ("Fusion only", "baseline")])
def test_view_selection_hides_other_mode(view, hidden):
    at = start()
    widget(at, "radio", "Show").set_value(view)
    at.run()
    for fig in figures(at):
        names = {n.lower() for n in trace_names(fig)}
        assert not any(n.startswith(hidden) for n in names), names


def test_bearing_rays_toggle():
    at = start()
    widget(at, "checkbox", "Show bearing rays").uncheck()
    at.run()
    trajectory, _ = figures(at)
    assert "Bearing observations" not in trace_names(trajectory)


def test_ellipses_drawn_every_ten_tracked_scans():
    at = start()
    trajectory, _ = figures(at)
    run = run_single(SCENARIOS["reference"].model_copy(), TrackerConfig(), 0)
    for mode in Mode:
        trace = next(t for t in trajectory["data"] if t.get("name") == f"{mode.value} 95% ellipse (every 10 s)")
        n_ellipses = sum(1 for x in trace["x"] if x is None)
        tracked = sum(e.has_track for e in run.modes[mode].result.estimates)
        assert n_ellipses == math.ceil(tracked / 10)


def test_run_button_shows_metrics_for_selected_settings():
    at = start()
    widget(at, "selectbox", "Scenario").set_value("biased_bearing")
    widget(at, "number_input", "Random seed").set_value(5)
    widget(at, "slider", "Noise multiplier").set_value(1.5)
    widget(at, "slider", "Missed-observation").set_value(0.3)
    at.run()
    # Changing controls alone keeps showing the previous run until Run is clicked.
    assert "scenario `reference`, seed 0" in showing(at)
    click(at, "Run simulation")
    assert "scenario `biased_bearing`, seed 5" in showing(at)
    assert "miss probability 0.30" in showing(at)

    scenario = with_sensor_overrides(SCENARIOS["biased_bearing"], 1.5, 0.3)
    expected = run_single(scenario, TrackerConfig(), 5)
    table = at.dataframe[0].value
    for row, name in zip(table.itertuples(), METRIC_NAMES):
        if name == "processing_time_s":
            continue
        for column, mode in (("baseline", Mode.BASELINE), ("fusion", Mode.FUSION)):
            want = getattr(expected.modes[mode].metrics, name)
            got = getattr(row, column)
            assert (math.isnan(want) and math.isnan(got)) or got == pytest.approx(want), (name, mode)


def test_multi_seed_comparison_matches_library():
    at = start()
    widget(at, "selectbox", "Scenario").set_value("noisy_primary")
    widget(at, "number_input", "Random seed").set_value(3)
    widget(at, "number_input", "Seeds for multi-seed").set_value(3)
    at.run()
    click(at, "Run multi-seed comparison")
    expected = compare_modes(with_sensor_overrides(SCENARIOS["noisy_primary"], 1.0, 0.1), TrackerConfig(), [3, 4, 5])
    verdict = next(m.value for m in at.markdown if m.value.startswith("**Verdict:"))
    assert f"**Verdict: {expected.verdict.value}.**" in verdict
    assert f"{expected.rmse_delta_mean_m:+.2f} m" in verdict
    subheader = " ".join(h.value for h in at.subheader)
    assert "`noisy_primary`, seeds 3-5 (n=3)" in subheader
    table = at.dataframe[1].value
    assert len(table) == len(METRIC_NAMES)
    assert all(re.fullmatch(r"\S+ ± \S+", cell) for cell in table["baseline"])


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_extreme_settings_render(scenario):
    at = start()
    widget(at, "selectbox", "Scenario").set_value(scenario)
    widget(at, "slider", "Noise multiplier").set_value(3.0)
    widget(at, "slider", "Missed-observation").set_value(0.8)
    at.run()
    click(at, "Run simulation")
    trajectory, error = figures(at)
    assert trajectory["data"] and error["data"]


def test_distant_sensor_does_not_squash_trajectory_plot():
    at = start()
    widget(at, "selectbox", "Scenario").set_value("distant_bearing")
    at.run()
    click(at, "Run simulation")
    trajectory, _ = figures(at)
    y_min, y_max = trajectory["layout"]["yaxis"]["range"]
    assert y_min > -10_000.0  # the bearing sensor at y = -30 km is left off-plot
    assert y_max - y_min < 10_000.0
