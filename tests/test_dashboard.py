"""Automated dashboard tests (L7) using Streamlit's headless AppTest harness (slow)."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import airspace_fusion
from airspace_fusion.config import SCENARIOS, HealthMonitorConfig, TrackerConfig, with_sensor_overrides
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


def figures(at: AppTest) -> tuple[dict, dict, dict]:
    trajectory, error, nis = (json.loads(c.proto.spec) for c in at.get("plotly_chart"))
    return trajectory, error, nis


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
    trajectory, error, nis = figures(at)
    assert trace_names(trajectory) == TRAJECTORY_TRACES
    assert {"baseline: actual position error", "fusion: actual position error"} <= trace_names(error)
    # Rolling NIS: the baseline has only the position sensor; one 95 % band per panel.
    assert trace_names(nis) == {"baseline: S1-POS", "fusion: S1-POS", "fusion: S2-BRG"}
    assert len([shape for shape in nis["layout"]["shapes"] if shape["type"] == "rect"]) == 2
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
    trajectory, *_ = figures(at)
    assert "Bearing observations" not in trace_names(trajectory)


def test_ellipses_drawn_every_ten_tracked_scans():
    at = start()
    trajectory, *_ = figures(at)
    run = run_single(SCENARIOS["reference"].model_copy(), TrackerConfig(), 0)
    for mode in Mode:
        trace = next(t for t in trajectory["data"] if t.get("name") == f"{mode.value} 95% ellipse (every 10 s)")
        n_ellipses = sum(1 for x in trace["x"] if x is None)
        tracked = sum(e.has_track for e in run.modes[mode].result.estimates)
        assert n_ellipses == math.ceil(tracked / 10)


def test_family_runs_and_shows_the_drawn_world():
    at = start()
    widget(at, "selectbox", "Scenario").set_value("crossing_geometry")
    widget(at, "number_input", "Random seed").set_value(4)
    at.run()
    click(at, "Run simulation")
    captions = " ".join(c.value for c in at.caption)
    assert "World: crossing_geometry seed 4:" in captions
    assert "different world for each seed" in captions
    trajectory, *_ = figures(at)
    assert trajectory["data"]


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
    trajectory, error, nis = figures(at)
    assert trajectory["data"] and error["data"]
    assert len(nis["layout"]["annotations"]) == 2  # both panel titles, even when there are no updates


def test_distant_sensor_does_not_squash_trajectory_plot():
    at = start()
    widget(at, "selectbox", "Scenario").set_value("distant_bearing")
    at.run()
    click(at, "Run simulation")
    trajectory, *_ = figures(at)
    y_min, y_max = trajectory["layout"]["yaxis"]["range"]
    assert y_min > -10_000.0  # the bearing sensor at y = -30 km is left off-plot
    assert y_max - y_min < 10_000.0


# -- tracker controls: IMM and sensor health monitor -----------------------------------------------------------


def charts(at: AppTest) -> list[dict]:
    return [json.loads(c.proto.spec) for c in at.get("plotly_chart")]


def chart_with(at: AppTest, name: str) -> dict | None:
    return next((fig for fig in charts(at) if name in trace_names(fig)), None)


def test_default_tracker_shows_no_imm_or_health_charts():
    at = start()
    assert len(charts(at)) == 3
    assert "tracker constant velocity, health monitor off" in showing(at)
    assert widget(at, "radio", "Motion model").value == "Constant velocity"
    assert widget(at, "checkbox", "Sensor health monitor").value is False


def test_imm_shows_its_model_probabilities():
    at = start()
    widget(at, "radio", "Motion model").set_value("IMM (constant velocity + turns)")
    at.run()
    assert "tracker constant velocity" in showing(at)  # nothing changes until Run is clicked
    click(at, "Run simulation")
    assert "tracker IMM" in showing(at)
    fig = chart_with(at, "right turn")
    assert fig is not None
    assert trace_names(fig) == {"constant velocity", "left turn", "right turn"}
    assert len([t for t in fig["data"] if t["name"] == "right turn"]) == 2  # one panel per mode
    # The right-turn model dominates during the reference turn (60-100 s), as in tests/test_imm.py.
    trace = next(t for t in fig["data"] if t["name"] == "right turn")
    during = [p for x, p in zip(trace["x"], trace["y"]) if p is not None and 70 <= x < 100]
    assert sum(during) / len(during) > 0.5
    expected = run_single(SCENARIOS["reference"], TrackerConfig(motion_model="imm"), 0)
    table = at.dataframe[0].value
    rmse = next(row for row, name in zip(table.itertuples(), METRIC_NAMES) if name == "position_rmse_m")
    assert rmse.fusion == pytest.approx(expected.modes[Mode.FUSION].metrics.position_rmse_m)


def test_health_monitor_timeline_and_table_for_a_biased_bearing():
    at = start()
    widget(at, "selectbox", "Scenario").set_value("biased_bearing")
    widget(at, "checkbox", "Sensor health monitor").check()
    at.run()
    click(at, "Run simulation")
    assert "health monitor on (fusion only)" in showing(at)
    fig = chart_with(at, "excluded (ignored)")
    assert fig is not None
    excluded = next(t for t in fig["data"] if t["name"] == "excluded (ignored)")
    bearing = SCENARIOS["biased_bearing"].secondary.sensor_id
    assert bearing in excluded["y"]
    assert SCENARIOS["biased_bearing"].primary.sensor_id not in excluded["y"]  # the last position sensor stays
    probation = next(t for t in fig["data"] if t["name"] == "on probation (tested, not used)")
    assert bearing in probation["y"]
    table = at.dataframe[1].value
    assert set(table.index) == {SCENARIOS["biased_bearing"].primary.sensor_id, bearing}
    assert bool(table.loc[bearing, "biased (truth label)"]) is True
    assert table.loc[bearing, "first alarm (s)"] <= 30.0


def test_health_chart_follows_the_view_and_the_monitor_setting():
    at = start()
    widget(at, "checkbox", "Sensor health monitor").check()
    at.run()
    click(at, "Run simulation")
    assert chart_with(at, "trusted") is not None
    widget(at, "radio", "Show").set_value("Baseline only")
    at.run()
    assert chart_with(at, "trusted") is None  # the monitor never runs on the baseline
    widget(at, "radio", "Show").set_value("Both")
    widget(at, "checkbox", "Sensor health monitor").uncheck()
    at.run()
    click(at, "Run simulation")
    assert chart_with(at, "trusted") is None
    assert any("Turn on the sensor health monitor" in c.value for c in at.caption)


def test_multi_seed_comparison_uses_and_names_the_tracker():
    at = start()
    widget(at, "radio", "Motion model").set_value("IMM (constant velocity + turns)")
    widget(at, "checkbox", "Sensor health monitor").check()
    widget(at, "number_input", "Seeds for multi-seed").set_value(2)
    at.run()
    click(at, "Run multi-seed comparison")
    assert any(c.value == "Tracker IMM, health monitor on (fusion only)." for c in at.caption)
    tracker = TrackerConfig(motion_model="imm", health_monitor=HealthMonitorConfig())
    expected = compare_modes(with_sensor_overrides(SCENARIOS["reference"], 1.0, 0.1), tracker, [0, 1])
    verdict = next(m.value for m in at.markdown if m.value.startswith("**Verdict:"))
    assert f"{expected.rmse_delta_mean_m:+.2f} m" in verdict
