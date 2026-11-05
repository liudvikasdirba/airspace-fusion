---
title: "0011: IMM tracker with fixed-rate coordinated turns"
status: accepted
updated: 2026-11-05
---

# 0011: IMM tracker with fixed-rate coordinated turns

## Context

The tracker's nearly-constant-velocity model treats the 90° turn in every scenario as noise. Its process noise
(0.5 m²/s³) is a compromise: too much for straight flight, too little for the turn. Baseline 95 % containment
in `reference` was 0.93 and RMSE during the turn about 23 m (roadmap finding 6). Containment problems from the
motion model then mix with the ones that M5 and M6 need to attribute to sensors.

## Decision

`IMMTracker` (in `estimation.py`) is an interacting multiple model filter with three models in the same state
`[x, vx, y, vy]`: constant velocity, and coordinated turns at +ω and −ω with a fixed ω (`imm_turn_rate_deg_s`,
default 3°/s). With a known turn rate every model is linear, so each is the existing Kalman filter with its own
transition matrix. Each scan the models are mixed through a Markov matrix (`imm_stay_probability`, default 0.99),
predicted, updated with the measurement and re-weighted by its likelihood, computed in log space. A new track
starts as constant velocity. Bearing updates use each model's own linearisation.

The IMM subclasses `SingleTargetTracker` and overrides only the filter steps, as the Stone Soup cross-check does.
`IMMState.mean` and `.covar` hold the moment-matched combined estimate, so gating, nearest-neighbour selection,
initiation, track drop and every metric are shared code. Estimates carry the mode probabilities.

It is selected by `TrackerConfig.motion_model = "imm"` (`--motion-model imm` on `run`, `compare` and `sweep`). The
default stays `"cv"`, so published results and the golden file are unchanged. The IMM has its own process noises
(`imm_straight_process_noise_m2_s3`, `imm_turn_process_noise_m2_s3`, both 0.05): its straight model needs far less
noise than the constant-velocity tracker, which must absorb turns as noise, and `process_noise_m2_s3` keeps one
meaning. The defaults were tuned on `reference` (60 seeds) against the M4 exit criteria.

## Consequences

- In `reference` (60 seeds) the IMM brings baseline containment from 0.93 to 0.96, turn RMSE from 23 to 17 m and
  straight-segment RMSE from 15.8 to 13.8 m. Baseline RMSE drops 9–16 % in every built-in scenario (40 seeds).
- With matched noise it is consistent or mildly conservative (average NEES about 3.3 against 4), because mixing
  in the turn models widens its covariance on straight flight. A negative control with too little noise fails.
- A single fixed turn rate is the limit: turns faster than about ω are tracked with lower RMSE but stay as
  over-confident as with the constant-velocity model (containment about 0.77 at 4.5–6°/s). A learned or
  multiple turn rate would address this.
- Its tighter straight model trusts an unmodelled bearing bias more: fusion containment in `biased_bearing`
  falls from 0.29 to 0.16. Detecting such a sensor is M5's job.
- The IMM costs about six times the constant-velocity filter (about 0.57 against 0.10 ms per scan with both
  sensors): three models, each mixed and filtered in small NumPy steps. Stone Soup is not a cross-check for
  it; `tests/test_imm.py` checks it against the constant-velocity tracker (three identical models must agree)
  and the turn model against the exact circle.
