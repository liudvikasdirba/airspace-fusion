"""Monte Carlo filter-consistency checks (slow; run with ``pytest -m slow``).

The scenario is set up so the tracker's motion model matches the simulator: no turn,
no false or missed observations, and process noise q = sigma^2 for the simulator's
per-step random acceleration sigma (dt = 1 s). The simulator holds each acceleration
constant over a step, so the filter's position process noise is slightly larger than
the truth's (dt^3/3 vs dt^4/4): the filter should be consistent or marginally
conservative, never over-confident.

Bounds were calibrated from a 100-run study (baseline ANEES 3.99, fusion 4.11,
position NIS 1.94, bearing NIS 0.97, containment 0.947 / 0.940) and widened for the
40 runs used here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest
from scipy import stats

from airspace_fusion.config import SCENARIOS, ScenarioConfig, TrackerConfig, default_primary, default_secondary
from airspace_fusion.evaluation import CHI2_2DOF_95, Mode, run_single
from airspace_fusion.schemas import MeasurementType

pytestmark = pytest.mark.slow

ACCEL_STD_MPS2 = 0.3
N_RUNS = 40
# Two-point initiation can be delayed a few scans when noise makes the first pair imply an
# implausible speed (> max_initiation_speed_mps); measure consistency after start-up.
SKIP_INITIAL_SCANS = 10


@dataclass(frozen=True)
class ConsistencyStats:
    anees: float
    fraction_of_scans_in_band: float
    nis_position: float
    nis_bearing: float
    containment_95: float


def matched_scenario(**overrides: object) -> ScenarioConfig:
    values: dict[str, object] = dict(
        turn_rate_rad_s=0.0,
        random_acceleration_std_mps2=ACCEL_STD_MPS2,
        corrupted_record_probability=0.0,
        primary=default_primary(detection_probability=1.0, false_observations_per_scan=0.0),
        secondary=default_secondary(detection_probability=1.0, false_observations_per_scan=0.0),
    )
    values.update(overrides)
    return SCENARIOS["reference"].model_copy(update=values)


def consistency(scenario: ScenarioConfig, tracker: TrackerConfig, runs: int) -> dict[Mode, ConsistencyStats]:
    nees: dict[Mode, list[list[float]]] = {m: [] for m in Mode}
    nis: dict[Mode, dict[MeasurementType, list[float]]] = {
        m: {MeasurementType.POSITION_2D: [], MeasurementType.BEARING_ONLY: []} for m in Mode
    }
    contained: dict[Mode, list[bool]] = {m: [] for m in Mode}
    for seed in range(runs):
        run = run_single(scenario, tracker, seed)
        truth = run.simulation.truth
        types = {m.measurement_id: m.measurement_type for m in run.measurements}
        for mode in Mode:
            result = run.modes[mode].result
            row = []
            for k, e in enumerate(result.estimates[SKIP_INITIAL_SCANS:], start=SKIP_INITIAL_SCANS):
                assert e.has_track, "matched scenario without clutter or misses must track continuously"
                x = np.array([truth.positions_m[k, 0], truth.velocities_mps[k, 0],
                              truth.positions_m[k, 1], truth.velocities_mps[k, 1]])
                d = e.state - x
                row.append(float(d @ np.linalg.solve(e.covariance, d)))
                p = e.position - truth.positions_m[k]
                contained[mode].append(float(p @ np.linalg.solve(e.position_covariance, p)) <= CHI2_2DOF_95)
            nees[mode].append(row)
            for mid, value in result.update_nis.items():
                nis[mode][types[mid]].append(value)

    # Per-scan average NEES over N runs, scaled by N, is chi-square with 4N dof if consistent.
    lo, hi = stats.chi2.ppf([0.025, 0.975], 4 * runs) / runs
    out = {}
    for mode in Mode:
        table = np.array(nees[mode])
        per_scan = table.mean(axis=0)
        bearings = nis[mode][MeasurementType.BEARING_ONLY]
        out[mode] = ConsistencyStats(
            anees=float(table.mean()),
            fraction_of_scans_in_band=float(np.mean((per_scan >= lo) & (per_scan <= hi))),
            nis_position=float(np.mean(nis[mode][MeasurementType.POSITION_2D])),
            nis_bearing=float(np.mean(bearings)) if bearings else math.nan,
            containment_95=float(np.mean(contained[mode])),
        )
    return out


@pytest.fixture(scope="module")
def matched() -> dict[Mode, ConsistencyStats]:
    return consistency(matched_scenario(), TrackerConfig(process_noise_m2_s3=ACCEL_STD_MPS2**2), N_RUNS)


@pytest.mark.parametrize("mode", list(Mode))
def test_nees_consistent(matched, mode):
    """M1: state error is consistent with the reported 4x4 covariance."""
    s = matched[mode]
    assert 3.6 <= s.anees <= 4.4, s
    assert s.fraction_of_scans_in_band >= 0.85, s  # 0.95 expected; scans are correlated in time


@pytest.mark.parametrize("mode", list(Mode))
def test_nis_consistent(matched, mode):
    """M2: innovations match their predicted covariance (gating truncates the mean slightly below dof)."""
    s = matched[mode]
    assert 1.8 <= s.nis_position <= 2.1, s
    if mode is Mode.FUSION:
        assert 0.9 <= s.nis_bearing <= 1.1, s
    else:
        assert math.isnan(s.nis_bearing)


@pytest.mark.parametrize("mode", list(Mode))
def test_containment_95(matched, mode):
    """M3: the 95 % position ellipse contains the truth about 95 % of the time."""
    assert 0.93 <= matched[mode].containment_95 <= 0.97, matched[mode]


def test_detects_overconfident_filter():
    """Negative control: with process noise 100x too small the checks must fail."""
    s = consistency(matched_scenario(), TrackerConfig(process_noise_m2_s3=ACCEL_STD_MPS2**2 / 100), 10)
    assert s[Mode.BASELINE].anees > 4.4
    assert s[Mode.BASELINE].containment_95 < 0.93


def test_bias_shows_in_bearing_nis():
    """Negative control: an unmodelled 1.5 deg bearing bias inflates the bearing NIS, not the position NIS."""
    biased = matched_scenario(
        secondary=default_secondary(detection_probability=1.0, false_observations_per_scan=0.0,
                                    bias=math.radians(1.5))
    )
    s = consistency(biased, TrackerConfig(process_noise_m2_s3=ACCEL_STD_MPS2**2), 10)
    assert s[Mode.FUSION].nis_bearing > 1.5
    assert s[Mode.FUSION].containment_95 < 0.93
    assert 1.8 <= s[Mode.BASELINE].nis_position <= 2.1
