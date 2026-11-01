"""Minimal local Streamlit dashboard. Run with: streamlit run src/airspace_fusion/app.py"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from airspace_fusion.config import DEFAULT_SCENARIO, SCENARIOS, TrackerConfig, with_sensor_overrides
from airspace_fusion.estimation import Estimate
from airspace_fusion.evaluation import CHI2_2DOF_95, METRIC_NAMES, Mode, SingleRun, compare_modes, run_single
from airspace_fusion.schemas import MeasurementType

# Categorical slots 1-2 of the validated reference palette; truth and observations stay neutral.
MODE_COLORS = {Mode.BASELINE: "#2a78d6", Mode.FUSION: "#eb6834"}
MODE_LABELS = {Mode.BASELINE: "Baseline (position sensor only)", Mode.FUSION: "Fusion (position + bearing)"}
# Plotly colours cannot follow the Streamlit theme, so neutrals are mid-greys that keep >= 3:1
# contrast on both the light (white) and dark (#0e1117) backgrounds.
TRUTH_COLOR = "#8a8984"
OBS_COLOR = "#9d9c97"
SENSOR_OUTLINE = "#f0efec"
ELLIPSE_EVERY_N = 10

LIMITATIONS = """
**This is a synthetic, educational simulation - not an operational system.**

- One synthetic object, two stationary synthetic sensors, generic 2D frame in metres. No real hardware, terrain or
  propagation.
- Nearly-constant-velocity Kalman filter; the turn is model mismatch absorbed by process noise.
- Bearing updates use an extended Kalman update (linearised `atan2`). Poor geometry (target near the bearing sensor
  or far away) gives little or unreliable information; very close geometry is explicitly rejected.
- The filter assumes zero-mean, independent, white errors. A biased or correlated sensor makes fusion over-confident
  and can make it **worse** than the baseline (see the `biased_bearing` scenario).
- Association is gating + nearest-neighbour for a single object. False observations inside the gate can still be
  accepted. No multi-object association.
- RMSE is computed only where a track exists; compare it together with coverage.
- "False observation" markers use truth-side labels for display only; the tracker never sees them.
"""


def ellipse_xy(mean: np.ndarray, cov: np.ndarray, scale2: float = CHI2_2DOF_95) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(cov)
    theta = np.linspace(0.0, 2.0 * math.pi, 48)
    circle = np.stack([np.cos(theta), np.sin(theta)])
    points = vectors @ (np.sqrt(np.maximum(values, 0.0) * scale2)[:, None] * circle)
    return mean[0] + points[0], mean[1] + points[1]


def trajectory_figure(run: SingleRun, modes: list[Mode], show_bearings: bool) -> go.Figure:
    scenario = run.simulation.scenario
    truth = run.simulation.truth
    fig = go.Figure()

    if show_bearings:
        offsets = truth.positions_m - np.array(scenario.secondary.position_m)
        ray_len = 1.3 * float(np.max(np.linalg.norm(offsets, axis=1)))
        xs: list[float | None] = []
        ys: list[float | None] = []
        sx, sy = scenario.secondary.position_m
        for m in run.measurements:
            if m.measurement_type is MeasurementType.BEARING_ONLY:
                b = m.measurement_value[0]
                xs += [sx, sx + ray_len * math.cos(b), None]
                ys += [sy, sy + ray_len * math.sin(b), None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=OBS_COLOR, width=1), opacity=0.12,
                                 name="Bearing observations", hoverinfo="skip"))

    positions = [m for m in run.measurements if m.measurement_type is MeasurementType.POSITION_2D]
    real = [m for m in positions if m.measurement_id not in truth.false_observation_ids]
    false = [m for m in positions if m.measurement_id in truth.false_observation_ids]
    for subset, name, symbol in ((real, "Position observations", "circle"),
                                 (false, "False position observations (truth label)", "x")):
        fig.add_trace(go.Scatter(
            x=[m.measurement_value[0] for m in subset], y=[m.measurement_value[1] for m in subset],
            mode="markers", marker=dict(color=OBS_COLOR, size=8 if symbol == "x" else 5, symbol=symbol,
                                        opacity=0.6), name=name,
            hovertemplate="%{x:.0f} m, %{y:.0f} m<extra>" + name + "</extra>"))

    fig.add_trace(go.Scatter(x=truth.positions_m[:, 0], y=truth.positions_m[:, 1], mode="lines",
                             line=dict(color=TRUTH_COLOR, width=2, dash="dash"), name="Ground truth",
                             hovertemplate="truth %{x:.0f} m, %{y:.0f} m<extra></extra>"))

    for mode in modes:
        ests = [e for e in run.modes[mode].result.estimates if e.has_track]
        color = MODE_COLORS[mode]
        # Break the line where the track is absent, rather than joining across gaps.
        xs, ys = [], []
        for e in run.modes[mode].result.estimates:
            xs.append(e.position[0] if e.has_track else None)
            ys.append(e.position[1] if e.has_track else None)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=2),
                                 name=MODE_LABELS[mode], legendgroup=mode.value,
                                 hovertemplate="%{x:.0f} m, %{y:.0f} m<extra>" + mode.value + "</extra>"))
        ex: list[float | None] = []
        ey: list[float | None] = []
        for e in ests[::ELLIPSE_EVERY_N]:
            x, y = ellipse_xy(e.position, e.position_covariance)
            ex += [*x, None]
            ey += [*y, None]
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color=color, width=1),
                                 name=f"{mode.value} 95% ellipse (every {ELLIPSE_EVERY_N} s)",
                                 legendgroup=mode.value, hoverinfo="skip"))

    for sensor in (scenario.primary, scenario.secondary):
        fig.add_trace(go.Scatter(x=[sensor.position_m[0]], y=[sensor.position_m[1]], mode="markers+text",
                                 marker=dict(symbol="square", size=11, color=TRUTH_COLOR,
                                             line=dict(color=SENSOR_OUTLINE, width=1.5)),
                                 text=[f"{sensor.sensor_id} ({sensor.measurement_type.value})"],
                                 textposition="bottom center", showlegend=False, hoverinfo="text"))

    pad = 600.0
    xr = (float(truth.positions_m[:, 0].min()) - pad, float(truth.positions_m[:, 0].max()) + pad)
    yr = (float(truth.positions_m[:, 1].min()) - pad, float(truth.positions_m[:, 1].max()) + pad)
    sensor_xy = np.array([scenario.primary.position_m, scenario.secondary.position_m])
    near = np.all(np.abs(sensor_xy) < 6000, axis=1)  # keep distant sensors from squashing the plot
    if near.any():
        xr = (min(xr[0], sensor_xy[near, 0].min() - pad), max(xr[1], sensor_xy[near, 0].max() + pad))
        yr = (min(yr[0], sensor_xy[near, 1].min() - pad), max(yr[1], sensor_xy[near, 1].max() + pad))
    fig.update_xaxes(title="x (m)", range=xr, showgrid=True, gridwidth=1, zeroline=False)
    fig.update_yaxes(title="y (m)", range=yr, scaleanchor="x", scaleratio=1, zeroline=False)
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=30, b=10), hovermode="closest",
                      legend=dict(orientation="h", yanchor="top", y=-0.12))
    return fig


def error_figure(run: SingleRun, modes: list[Mode]) -> go.Figure:
    truth = run.simulation.truth
    t0 = truth.timestamps[0]
    fig = go.Figure()
    for mode in modes:
        color = MODE_COLORS[mode]
        t: list[float] = []
        err: list[float | None] = []  # None breaks the line where there is no track
        bound: list[float | None] = []
        for k, e in enumerate(run.modes[mode].result.estimates):
            t.append((e.timestamp - t0).total_seconds())
            if e.has_track:
                err.append(float(np.linalg.norm(e.position - truth.positions_m[k])))
                bound.append(_radius95(e))
            else:
                err.append(None)
                bound.append(None)
        fig.add_trace(go.Scatter(x=t, y=err, mode="lines", line=dict(color=color, width=2),
                                 name=f"{mode.value}: actual position error",
                                 hovertemplate="t=%{x:.0f} s  error %{y:.1f} m<extra>" + mode.value + "</extra>"))
        fig.add_trace(go.Scatter(x=t, y=bound, mode="lines", line=dict(color=color, width=1, dash="dot"),
                                 name=f"{mode.value}: filter's 95% radius (major axis)",
                                 hovertemplate="t=%{x:.0f} s 95% radius %{y:.1f} m<extra>" + mode.value + "</extra>"))
    fig.update_xaxes(title="time (s)")
    fig.update_yaxes(title="metres", rangemode="tozero")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="top", y=-0.25))
    return fig


def _radius95(e: Estimate) -> float:
    return float(math.sqrt(CHI2_2DOF_95 * np.max(np.linalg.eigvalsh(e.position_covariance))))


METRIC_LABELS = {
    "position_rmse_m": "Position RMSE (m)",
    "coverage": "Coverage (share of scans with a track)",
    "containment_95": "Truth inside 95% ellipse (~0.95 = honest)",
    "missed_updates": "Missed primary-sensor updates",
    "prediction_only_scans": "Prediction-only scans (no sensor updated)",
    "rejected_observations": "Rejected observations (tracker)",
    "invalid_observations": "Invalid records (this mode's sensors)",
    "track_initiations": "Track initiations",
    "processing_time_s": "Processing time (s)",
}


def metrics_table(run: SingleRun) -> pd.DataFrame:
    rows = []
    for name in METRIC_NAMES:
        b = getattr(run.modes[Mode.BASELINE].metrics, name)
        f = getattr(run.modes[Mode.FUSION].metrics, name)
        rows.append({"metric": METRIC_LABELS[name], "baseline": b, "fusion": f, "delta": f - b})
    return pd.DataFrame(rows).set_index("metric")


def main() -> None:
    st.set_page_config(page_title="Synthetic tracking: baseline vs fusion", layout="wide")
    st.title("Synthetic single-object tracking: baseline vs fusion")
    st.caption("Simulation-only educational prototype. All data is synthetic.")

    with st.sidebar:
        st.header("Simulation")
        scenario_name = st.selectbox("Scenario", list(SCENARIOS), index=list(SCENARIOS).index(DEFAULT_SCENARIO))
        st.caption(SCENARIOS[scenario_name].description)
        seed = st.number_input("Random seed", min_value=0, max_value=1_000_000, value=0, step=1)
        noise = st.slider("Noise multiplier (both sensors)", 0.25, 3.0, 1.0, 0.25)
        miss = st.slider("Missed-observation probability (both sensors)", 0.0, 0.8, 0.1, 0.05)
        view = st.radio("Show", ["Both", "Baseline only", "Fusion only"], horizontal=True)
        show_bearings = st.checkbox("Show bearing rays", value=True)
        run_clicked = st.button("Run simulation", type="primary", width="stretch")
        st.divider()
        n_seeds = st.number_input("Seeds for multi-seed comparison", min_value=2, max_value=100, value=10)
        compare_clicked = st.button("Run multi-seed comparison", width="stretch")

    scenario = with_sensor_overrides(SCENARIOS[scenario_name], noise, miss)
    tracker = TrackerConfig()
    if run_clicked or "run" not in st.session_state:
        st.session_state["run"] = run_single(scenario, tracker, int(seed))
    run: SingleRun = st.session_state["run"]
    modes = {"Both": list(Mode), "Baseline only": [Mode.BASELINE], "Fusion only": [Mode.FUSION]}[view]

    sim_scn = run.simulation.scenario
    st.markdown(
        f"**Showing:** scenario `{sim_scn.name}`, seed {run.simulation.seed}, "
        f"position noise {sim_scn.primary.noise_std:.0f} m, "
        f"bearing noise {math.degrees(sim_scn.secondary.noise_std):.2f} deg, "
        f"miss probability {1 - sim_scn.primary.detection_probability:.2f}. Both modes use the same truth and records."
    )

    # Stacked rather than side by side: the metrics table and legends need the full width on narrow windows.
    st.plotly_chart(trajectory_figure(run, modes, show_bearings), width="stretch")
    st.subheader("Metrics (this run)")
    st.dataframe(metrics_table(run).style.format("{:.4g}"), width="stretch")
    st.caption("Lower RMSE is better. RMSE only covers scans where a track exists - read it with coverage.")
    st.plotly_chart(error_figure(run, modes), width="stretch")

    if compare_clicked:
        seeds = list(range(int(seed), int(seed) + int(n_seeds)))
        with st.spinner(f"Running {len(seeds)} seeds x 2 modes..."):
            st.session_state["comparison"] = compare_modes(scenario, tracker, seeds)
    if "comparison" in st.session_state:
        s = st.session_state["comparison"]
        st.subheader(f"Multi-seed comparison: `{s.scenario}`, seeds {s.seeds[0]}-{s.seeds[-1]} (n={len(s.seeds)})")
        rows = [{"metric": METRIC_LABELS[name],
                 **{m.value: f"{s.metrics[m][name].mean:.4g} ± {s.metrics[m][name].std:.2g}" for m in Mode}}
                for name in METRIC_NAMES]
        st.dataframe(pd.DataFrame(rows).set_index("metric"), width="stretch")
        st.markdown(
            f"**Verdict: {s.verdict.value}.** Paired RMSE change (fusion - baseline) "
            f"{s.rmse_delta_mean_m:+.2f} m (95% CI {s.rmse_delta_ci95_m[0]:+.2f} to {s.rmse_delta_ci95_m[1]:+.2f}), "
            f"{s.rmse_relative_change:+.1%}, paired t-test p = {s.rmse_p_value:.3g}. "
            "A change counts as material only if p < 0.05 and |relative change| >= 5%."
        )
        st.caption("Settings in the sidebar apply to the next comparison run; this table reflects the last one run.")

    with st.expander("Limitations of this simulation", expanded=False):
        st.markdown(LIMITATIONS)


main()
