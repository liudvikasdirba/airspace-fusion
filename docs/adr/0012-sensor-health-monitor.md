---
title: "0012: Sensor health monitor and sensor exclusion"
status: accepted
updated: 2026-11-08
---

# 0012: Sensor health monitor and sensor exclusion

## Context

An unmodelled bearing bias makes fusion worse than the baseline and over-confident, and the filter cannot tell: in
`biased_bearing` fusion RMSE is about twice the baseline's. The bias is visible in the innovations, so a monitor can
detect it without ground truth. But excluding a sensor interacts with [ADR 0004](0004-position-updates-keep-tracks-alive.md):
only position updates keep a track alive, so the monitor must never be able to starve a track of them.

With two sensors there is also an attribution problem. When a bearing and a position disagree, the innovations alone
cannot say which one is wrong.

## Decision

`health.py` holds one monitor per sensor and measurement kind. It sees one sample per sensor per scan, the whitened innovation of that
sensor's nearest measurement against the track, capped at the gate, and imports nothing but `config`
(`tests/test_integration.py` enforces this). Two tests run on the samples: a chi-square test on the sum of squares over
a 20-sample window, and a two-sided CUSUM per component. Each sensor is **trusted**, **suspect** (variance multiplied
by 4) or **excluded** (ignored, disposition `rejected_sensor_excluded`, which `rejected_observations` does not count,
so that metric stays comparable with runs without the monitor). An excluded sensor is still tested against
the track, which no longer contains it, and returns through suspect after 20 samples with a consistent window.

Three rules go beyond the tests:

- **The last usable position sensor is protected.** Another position sensor is an alternative only if it is not
  excluded and has been heard within half the coast limit, so a remaining sensor is reinstated before a track would be
  dropped. A protected sensor is tested and its alarms are counted (`held_alarms`), but it is trusted, whatever state
  it was in. When protection ends, what it accumulated is dropped, since the monitor could not act on it. This is how
  the monitor respects ADR 0004: it can never remove the updates that keep a track alive. An excluded position sensor
  cannot start a track either while another is usable. Protection also fixes the attribution: position sensors are the
  reference that other sensors are judged against. Excluding bearings costs a track nothing, since bearings never kept
  it alive.
- **Other sensors start each track on probation.** Their first 4 samples against the track are tested but not used,
  even if they raise an alarm, and probation lasts until the window mean is not shifted, for at most one window. Once a
  bias is in the state the filter absorbs it and the innovations shrink, so the first samples are the best evidence
  available. A new track also clears every sensor's tests, because samples against the old track say nothing about
  this one; a sensor's state carries over. Probation is not an alarm and does not count as time excluded.
- **The baseline has no monitor.** It has one sensor and nothing to fall back on, so baseline results are identical
  with the monitor on or off.

The monitor is opt-in (`TrackerConfig.health_monitor`, `--health-monitor`), so the golden file and published results
do not change, and an unset monitor is left out of results records, so existing configuration hashes hold. Stone Soup
has no monitor and refuses a configuration with one.

An *alarm* is any departure from trusted, including to suspect. *Detection delay* runs from the start of the run,
when the bias is already present, to the first alarm on a biased sensor. A *false alarm* is an alarm on an unbiased
sensor. Only `evaluation.py` knows which sensors are biased.

## Consequences

Settings were tuned on seeds 0–299. A first held-out block (10000–10199) with 3 probation samples missed one
criterion: the IMM in `biased_bearing` was 5.5 % worse than the baseline. Development seeds showed the cause, biased
sensors admitted after a lucky start, so probation went to 4 samples and that block was retired rather than reused.
No setting has been chosen from blocks 20000–20199 and 40000–40199, and the results below pool them.

| Exit criterion (400 seeds) | Constant velocity | IMM |
|---|---|---|
| 1.5° bias: median delay ≤ 30 s | 11 s, 400/400 detected | 11 s, 400/400 detected |
| False alarms in ≤ 1 % of unbiased runs | 3/1,200 | 1/1,200 |
| `biased_bearing` ≤ 5 % worse than baseline | +0.9 %, containment 0.93 | +1.4 %, containment 0.95 |
| `reference` keeps ≥ 90 % of the fusion gain | 92.6 % | 90.6 % |

The gain criterion is not settled for the IMM. Its 95 % bootstrap interval is 87.5–93.0 %, and the blocks alone give
88.7 % and 92.6 %; for constant velocity the interval is 90.1–94.6 %, with 90.6 % and 94.5 %. Both blocks are reported
because 20000–20199 alone misses the criterion with the IMM and 40000–40199 alone meets it. A code review changed the
monitor between the two measurements, but the same code on the first block gives the same gain as before, so the
difference is seed variation. Probation is the trade-off: every scan without bearings costs gain in unbiased worlds, and
a shorter probation lets more biases in.

All false alarms were in `noisy_primary`, where a 75 m position sensor is a poor reference: 0.75 % of runs with constant
velocity, and on development seeds up to 1.3 % with the IMM. The 1 % criterion holds pooled over scenarios, not
necessarily for that scenario alone.

Limits:

- A biased **position** sensor is not attributed. It looks like a bad bearing, so the bearing is excluded and fusion
  falls back to the baseline's error. With 60 m per axis, fusion without the monitor has lower RMSE (70 m against
  91 m, constant velocity, 30 seeds): excluding the bearing removed a sensor that was helping.
- Small biases go undetected. At 0.5° only about a quarter of runs raise an alarm at the default thresholds.
- Probation is only at the start of a track. A sensor that goes bad later has its bias partly absorbed, which makes
  detection slower than these figures, and no scenario tests it.
- The built-in scenarios have one position sensor, so the rules for several are covered by unit tests only.
- Turns faster than the IMM's turn rate have not been tested for false alarms.

`airspace-fusion health` and `airspace-fusion roc` report the monitor's record over many seeds.
