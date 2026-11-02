---
title: "0008: Scenario files and scenario families"
status: accepted
updated: 2026-11-02
---

# 0008: Scenario files and scenario families

## Context

All four built-in scenarios share one trajectory and one sensor layout; seeds vary only noise and a small random
acceleration (roadmap finding 3). "Fusion improves RMSE by 11 %" therefore holds for one geometry. M2 needs
results across geometries, scenarios defined without editing Python, and, next, sweeps that vary one parameter
while everything else is held fixed.

## Decision

**Families.** `ScenarioFamily` (`scenarios.py`) is a base `ScenarioConfig` plus uniform ranges for the target's
start (range and direction from the primary sensor), heading (relative to a crossing path), speed and turn, and
optionally the bearing sensor's placement, bearing bias and clutter. The bearing sensor is placed by **crossing
angle** (the angle at the target's start between the two sensors' lines of sight) and **range ratio** (its
distance from the target relative to the primary sensor's), because those two numbers decide how much a bearing
adds; absolute coordinates would not. `family.for_seed(seed)` draws one concrete `ScenarioConfig`.
`ScenarioConfig.for_seed` returns the scenario itself, so `compare_modes`, worker processes, the learning loop
and the dashboard take either. Both modes of a seed share its drawn world, so comparisons stay paired.

**Random streams.** A family draws from spawn key 100 of the seed's `SeedSequence`; the simulator uses keys 0–3.
Built-in scenarios, and therefore the golden file, are unchanged. Every parameter consumes exactly one draw in
a fixed order, even when its range is a single value or unused. Fixing one parameter thus leaves every other
drawn parameter of a seed unchanged (common random numbers), which sweeps rely on.

**Built-in families.** `crossing_geometry` (5–90°), `range_ratio` (0.5–10), `manoeuvre_intensity` (up to
6 °/s), `clutter_density` (0–5 per scan) and `bias_magnitude` (0–3°), all on the `reference` base.

**Files.** TOML, read with `--scenario-file PATH` (repeatable) on `run`, `compare`, `learn` and `scenarios`.
`[[scenario]]` and `[[family]]` tables start from a `base` (a built-in scenario or one defined earlier) and
override fields. Scenario fields use the model's units (metres, seconds, radians), so a file field and the Python
field are the same thing; family ranges carry their unit in their name (`_m`, `_s`, `_deg`, `_mps`) and are
`[lo, hi]` pairs or single values. Unknown keys and redefined names are errors. `airspace-fusion scenarios --toml
NAME` prints any entry in this format, and `to_toml` output reads back to an equal object.

**Names.** `--scenario all` keeps meaning the four built-in scenarios, so published results are unchanged;
`--scenario families` selects the five families. Names are checked after parsing, because file entries are not
known to argparse.

## Consequences

- Results can be stated per family, and the next step (a `sweep` command) can vary one parameter with every
  seed's other draws held fixed.
- A family verdict pools many geometries. The paired test stays valid, but the spread across seeds now includes
  geometry, so it is wider than for a fixed scenario; per-family results should be read with that in mind.
- Clutter stays uniform over the base scenario's surveillance area (±4 km). Fast, distant targets can leave that
  area, where they see no position clutter.
- `tomli` is a dependency on Python 3.10, which has no `tomllib`.
- `scenarios.py` depends only on `config.py`; the module dependency test enforces it.
