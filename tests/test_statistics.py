"""Statistical behaviour tests (M4-M7; slow, run with ``pytest -m slow``).

Bounds come from a calibration study and are stated next to each test. Where the
result was surprising (bias response above ~2 deg), the test asserts what the
tracker actually does and the docstring explains why.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from airspace_fusion.config import SCENARIOS, TrackerConfig, default_primary, default_secondary
from airspace_fusion.estimation import Disposition
from airspace_fusion.evaluation import Mode, Verdict, compare_modes, run_single
from helpers import matched_scenario, matched_tracker

pytestmark = pytest.mark.slow


def test_gate_leakage_matches_gate_probability():
    """M4: with a matched model, the share of true measurements rejected by the gate is ~(1 - gate probability).

    Calibration (30 runs): 21/5354 baseline, 43/10708 fusion against a nominal 0.3 %.
    """
    tracker = matched_tracker()
    expected = 1.0 - tracker.gate_probability
    rejected = {m: 0 for m in Mode}
    gated = {m: 0 for m in Mode}
    for seed in range(30):
        run = run_single(matched_scenario(), tracker, seed)
        for mode in Mode:
            for disposition in run.modes[mode].result.dispositions.values():
                if disposition in (Disposition.ACCEPTED, Disposition.OUTSIDE_GATE):
                    gated[mode] += 1
                    rejected[mode] += disposition == Disposition.OUTSIDE_GATE
    for mode in Mode:
        # Two-sided 99.9 % binomial interval around the nominal rate.
        lo, hi = stats.binom.ppf([0.0005, 0.9995], gated[mode], expected)
        assert lo <= rejected[mode] <= hi, (mode, rejected[mode], gated[mode], lo, hi)


def test_fusion_never_increases_uncertainty_and_usually_reduces_error():
    """M5: with a matched model and unbiased bearings, adding bearings only adds information.

    The covariance check disables gating, because otherwise the two modes can make different
    accept/reject decisions and their covariances are no longer comparable.
    """
    no_gate = matched_tracker(gate_probability=1.0 - 1e-12)
    worst = math.inf
    for seed in range(10):
        run = run_single(matched_scenario(), no_gate, seed)
        base, fused = run.modes[Mode.BASELINE].result, run.modes[Mode.FUSION].result
        for eb, ef in zip(base.estimates[10:], fused.estimates[10:]):
            difference = eb.covariance - ef.covariance
            worst = min(worst, float(np.min(np.linalg.eigvalsh(difference))) / float(np.trace(eb.covariance)))
    assert worst >= -1e-6, worst  # baseline covariance - fusion covariance is positive semi-definite

    wins = 0
    seeds = range(30)
    for seed in seeds:
        run = run_single(matched_scenario(), matched_tracker(), seed)
        wins += run.modes[Mode.FUSION].metrics.position_rmse_m <= run.modes[Mode.BASELINE].metrics.position_rmse_m
    assert wins >= 0.9 * len(seeds), wins  # calibration: 29/30


BIASES_DEG = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)


@pytest.fixture(scope="module")
def bias_sweep():
    results = []
    for bias in BIASES_DEG:
        scenario = SCENARIOS["reference"].model_copy(
            update={"secondary": default_secondary(bias=math.radians(bias))}
        )
        summary = compare_modes(scenario, TrackerConfig(), range(8))
        bearing_rejected = bearing_total = 0
        for seed in range(8):
            run = run_single(scenario, TrackerConfig(), seed)
            false_ids = run.simulation.truth.false_observation_ids
            for mid, disposition in run.modes[Mode.FUSION].result.dispositions.items():
                # True bearings only: clutter is rejected regardless of bias.
                if (
                    mid.startswith(scenario.secondary.sensor_id)
                    and mid not in false_ids
                    and disposition in (Disposition.ACCEPTED, Disposition.OUTSIDE_GATE)
                ):
                    bearing_total += 1
                    bearing_rejected += disposition == Disposition.OUTSIDE_GATE
        results.append((bias, summary, bearing_rejected / bearing_total))
    return results


def test_bias_degrades_fusion_up_to_two_degrees(bias_sweep):
    """M6a: fusion error grows with unmodelled bearing bias, at least up to 2 deg (calibration: 16.2 -> 47.7 m)."""
    rmse = [s.metrics[Mode.FUSION]["position_rmse_m"].mean for bias, s, _ in bias_sweep if bias <= 2.0]
    assert all(b > a for a, b in zip(rmse, rmse[1:])), rmse


def test_large_bias_is_increasingly_gated_out(bias_sweep):
    """M6b: the gate rejects a growing share of true bearings as bias grows.

    This is why fusion error falls again at 3 deg (calibration: 24 m): a gross bias is mostly
    rejected and fusion reverts toward the baseline, so a moderate bias is the most damaging.
    """
    by_bias = {bias: fraction for bias, _, fraction in bias_sweep}
    # Below 0.5 deg rejections are ~1 % (about ten per thousand) and differences are counting noise.
    trend = [fraction for bias, fraction in by_bias.items() if bias >= 0.5]
    assert all(b > a for a, b in zip(trend, trend[1:])), by_bias
    assert by_bias[0.0] < 0.02 and by_bias[3.0] > 0.5, by_bias


def test_verdict_flips_from_improves_to_worsens(bias_sweep):
    """M6c: verdict improves at small bias, worsens at moderate bias (calibration flip between 0.25 and 1 deg)."""
    verdicts = {bias: s.verdict for bias, s, _ in bias_sweep}
    assert verdicts[0.0] is Verdict.IMPROVES and verdicts[0.25] is Verdict.IMPROVES
    assert all(verdicts[b] is Verdict.WORSENS for b in (1.0, 1.5, 2.0)), verdicts
    first_worse = min(b for b, v in verdicts.items() if v is Verdict.WORSENS)
    assert 0.25 < first_worse <= 1.0


@pytest.mark.parametrize("rate", [0.2, 1.0, 5.0])
def test_clutter_is_rarely_accepted_and_coverage_holds(rate):
    """M7: false observations per scan at 0.2, 1 and 5 on both sensors.

    Calibration: accepted-false fraction <= 1.2 % (fusion at 5/scan, from bearing clutter inside the
    angular gate); coverage >= 0.96. RMSE is checked by median because a track can occasionally be
    initiated on two aligned false observations before the real one (a known two-point limitation).
    """
    scenario = SCENARIOS["reference"].model_copy(update={
        "primary": default_primary(false_observations_per_scan=rate),
        "secondary": default_secondary(false_observations_per_scan=rate),
    })
    for mode in Mode:
        accepted = accepted_false = 0
        coverage, rmse = [], []
        for seed in range(8):
            run = run_single(scenario, TrackerConfig(), seed)
            false_ids = run.simulation.truth.false_observation_ids
            for mid, disposition in run.modes[mode].result.dispositions.items():
                if disposition in (Disposition.ACCEPTED, Disposition.INITIATION):
                    accepted += 1
                    accepted_false += mid in false_ids
            coverage.append(run.modes[mode].metrics.coverage)
            rmse.append(run.modes[mode].metrics.position_rmse_m)
        assert accepted_false / accepted <= 0.025, (mode, rate, accepted_false, accepted)
        assert min(coverage) >= 0.95, (mode, rate, coverage)
        assert float(np.median(rmse)) <= 25.0, (mode, rate, rmse)
