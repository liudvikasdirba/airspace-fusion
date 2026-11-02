"""Scenario families and scenario files (M2, ADR 0008)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from airspace_fusion.cli import main
from airspace_fusion.config import SCENARIOS, TrackerConfig
from airspace_fusion.evaluation import Mode, compare_modes, seed_executor
from airspace_fusion.scenarios import (
    FAMILIES,
    ParameterRange,
    ScenarioFamily,
    ScenarioFileError,
    available_sources,
    builtin_sources,
    load_scenario_file,
    parse_scenario_toml,
    to_toml,
    with_overrides,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "scenarios" / "example.toml"
REFERENCE = SCENARIOS["reference"]


def crossing_angle_at_start(world) -> float:
    """Angle at the target's start point between the lines of sight to the two sensors, in degrees."""
    (sx, sy), (px, py), (bx, by) = world.initial_position_m, world.primary.position_m, world.secondary.position_m
    a = math.atan2(py - sy, px - sx) - math.atan2(by - sy, bx - sx)
    return abs((math.degrees(a) + 180.0) % 360.0 - 180.0)


def family(**ranges) -> ScenarioFamily:
    return ScenarioFamily.model_validate({"name": "f", "base": "reference"} | ranges)


# -- drawing worlds ------------------------------------------------------------------------------------------


def test_fixed_scenario_is_the_same_for_every_seed():
    assert REFERENCE.for_seed(0) is REFERENCE and REFERENCE.for_seed(99) is REFERENCE


@pytest.mark.parametrize("name", list(FAMILIES))
def test_family_draws_are_deterministic_and_vary_by_seed(name):
    f = FAMILIES[name]
    assert f.for_seed(7) == f.for_seed(7)
    assert f.for_seed(7) != f.for_seed(8)
    world = f.for_seed(7)
    assert world.name == name and world.description.startswith(f"{name} seed 7: ")


@pytest.mark.parametrize("name", list(FAMILIES))
def test_drawn_worlds_respect_the_ranges(name):
    f = FAMILIES[name]
    for seed in range(60):
        w = f.for_seed(seed)
        px, py = w.primary.position_m
        start_range = math.hypot(w.initial_position_m[0] - px, w.initial_position_m[1] - py)
        assert f.start_range_m.lo - 1e-9 <= start_range <= f.start_range_m.hi + 1e-9
        assert f.speed_mps.lo - 1e-9 <= math.hypot(*w.initial_velocity_mps) <= f.speed_mps.hi + 1e-9
        assert f.turn_start_s.lo <= w.turn_start_s <= f.turn_start_s.hi
        assert f.turn_duration_s.lo - 1e-9 <= w.turn_end_s - w.turn_start_s <= f.turn_duration_s.hi + 1e-9
        assert f.turn_rate_deg_s.lo - 1e-9 <= math.degrees(w.turn_rate_rad_s) <= f.turn_rate_deg_s.hi + 1e-9
        assert f.crossing_angle_deg is not None
        assert f.crossing_angle_deg.lo - 1e-6 <= crossing_angle_at_start(w) <= f.crossing_angle_deg.hi + 1e-6
        bearing_distance = math.hypot(w.secondary.position_m[0] - w.initial_position_m[0],
                                      w.secondary.position_m[1] - w.initial_position_m[1])
        assert f.range_ratio.lo - 1e-9 <= bearing_distance / start_range <= f.range_ratio.hi + 1e-9
        if f.bearing_bias_deg is not None:
            assert f.bearing_bias_deg.lo - 1e-9 <= math.degrees(w.secondary.bias) <= f.bearing_bias_deg.hi + 1e-9
        else:
            assert w.secondary.bias == REFERENCE.secondary.bias
        if f.false_observations_per_scan is not None:
            assert w.primary.false_observations_per_scan == w.secondary.false_observations_per_scan
        # Everything the family does not draw comes from the base scenario.
        assert w.primary.noise_std == REFERENCE.primary.noise_std
        assert w.duration_s == REFERENCE.duration_s


def test_fixing_one_parameter_leaves_the_others_unchanged():
    """Common random numbers: what a sweep relies on to compare its points seed by seed."""
    narrow, wide = family(crossing_angle_deg=20.0), family(crossing_angle_deg=80.0)
    for seed in range(20):
        a, b = narrow.for_seed(seed), wide.for_seed(seed)
        assert a.initial_position_m == b.initial_position_m
        assert a.initial_velocity_mps == b.initial_velocity_mps
        assert (a.turn_start_s, a.turn_end_s, a.turn_rate_rad_s) == (b.turn_start_s, b.turn_end_s, b.turn_rate_rad_s)
        assert crossing_angle_at_start(a) == pytest.approx(20.0) and crossing_angle_at_start(b) == pytest.approx(80.0)


def test_unset_crossing_keeps_the_base_bearing_sensor():
    w = family().for_seed(3)
    assert w.secondary.position_m == REFERENCE.secondary.position_m


def test_overrides_apply_to_a_family_base():
    f = with_overrides(FAMILIES["bias_magnitude"], 2.0, 0.5)
    assert isinstance(f, ScenarioFamily)
    w = f.for_seed(1)
    assert w.primary.noise_std == pytest.approx(2.0 * REFERENCE.primary.noise_std)
    assert w.secondary.detection_probability == pytest.approx(0.5)
    assert w.secondary.bias == FAMILIES["bias_magnitude"].for_seed(1).secondary.bias  # the drawn bias stays
    assert with_overrides(REFERENCE, 1.0, None) == REFERENCE


def test_families_are_hashable_and_picklable():
    import pickle

    f = FAMILIES["crossing_geometry"]
    assert {f: 1}[f] == 1  # evaluation caches key on (scenario, tracker, seeds)
    assert pickle.loads(pickle.dumps(f)) == f  # worker processes receive families


@pytest.mark.parametrize(
    "ranges, message",
    [
        ({"speed_mps": [30.0, 10.0]}, "above the upper bound"),
        ({"speed_mps": [1.0, 2.0, 3.0]}, "exactly two values"),
        ({"start_range_m": 0.0}, "start_range_m must be positive"),
        ({"crossing_angle_deg": [0.0, 90.0]}, "strictly between 0 and 180"),
        ({"crossing_angle_deg": 180.0}, "strictly between 0 and 180"),
        ({"range_ratio": -1.0}, "range_ratio must be positive"),
        ({"false_observations_per_scan": [-1.0, 1.0]}, "must not be negative"),
        ({"turn_duration_s": [-5.0, 5.0]}, "must not be negative"),
        ({"speed_mps": True}, "number or a"),
        ({"speed_mps": [math.inf, 1.0]}, "finite"),
        ({"spead_mps": 10.0}, "spead_mps"),
    ],
)
def test_invalid_families_are_rejected(ranges, message):
    with pytest.raises(ValidationError, match=message):
        family(**ranges)


def test_unknown_base_is_rejected():
    with pytest.raises(ValidationError, match="unknown base scenario"):
        ScenarioFamily.model_validate({"name": "f", "base": "nowhere"})


def test_parameter_range():
    r = ParameterRange(lo=2.0, hi=4.0)
    assert r.at(0.0) == 2.0 and r.at(0.5) == 3.0
    assert ParameterRange(lo=5.0, hi=5.0).at(0.9) == 5.0


# -- evaluation with families --------------------------------------------------------------------------------


def test_compare_family_serial_and_parallel_agree():
    f = FAMILIES["crossing_geometry"]
    seeds = list(range(6))
    serial = compare_modes(f, TrackerConfig(), seeds)
    with seed_executor(2, len(seeds)) as executor:
        parallel = compare_modes(f, TrackerConfig(), seeds, executor)
    assert serial.scenario == "crossing_geometry"
    for mode in Mode:
        for name, summary in serial.metrics[mode].items():
            if name != "processing_time_s":
                assert parallel.metrics[mode][name] == summary, (mode, name)
    assert parallel.verdict == serial.verdict


# -- scenario files ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(FAMILIES))
def test_family_toml_round_trip(name):
    assert parse_scenario_toml(to_toml(FAMILIES[name]), known=SCENARIOS)[name] == FAMILIES[name]


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_builtin_scenarios_round_trip_exactly(name):
    """A built-in scenario written to TOML and read back is identical, down to every float."""
    text = to_toml(SCENARIOS[name]).replace(f'name = "{name}"', f'name = "{name}_copy"', 1)
    loaded = parse_scenario_toml(text, known=builtin_sources())[f"{name}_copy"]
    assert loaded.model_copy(update={"name": name}) == SCENARIOS[name]


def test_family_with_a_file_base_round_trips():
    entries = parse_scenario_toml(EXAMPLE.read_text(encoding="utf-8"))
    custom = ScenarioFamily(name="g", base=entries["close_bearing_sensor"], speed_mps=ParameterRange(lo=5, hi=6))
    text = to_toml(custom)
    assert text.index("[[scenario]]") < text.index("[[family]]")  # the base is written first
    assert parse_scenario_toml(text)["g"] == custom


def test_example_file_loads_and_runs(capsys):
    entries = load_scenario_file(EXAMPLE)
    assert set(entries) == {"close_bearing_sensor", "wide_crossing"}
    close = entries["close_bearing_sensor"]
    assert close.secondary.position_m == (0.0, -1000.0)
    assert close.model_copy(update={"name": "reference", "description": REFERENCE.description,
                                    "secondary": REFERENCE.secondary}) == REFERENCE
    assert main(["run", "--scenario-file", str(EXAMPLE), "--scenario", "wide_crossing", "--seed", "2"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("scenario=wide_crossing seed=2 ") and "world: wide_crossing seed 2:" in out


@pytest.mark.parametrize(
    "text, message",
    [
        ("[[scenario]]\nname = 'x'\nbase = 'reference'\nduraton_s = 5\n", "unknown key\\(s\\) duraton_s"),
        ("[[scenario]]\nname = 'x'\nbase = 'reference'\n[scenario.primary]\nnoise = 3\n", "scenario.primary"),
        ("[[scenario]]\nname = 'x'\nbase = 'nowhere'\n", "not a known scenario"),
        ("[[scenario]]\nname = 'reference'\nbase = 'reference'\n", "already defined"),
        ("[[family]]\nname = 'crossing_geometry'\nbase = 'reference'\n", "already defined"),
        ("[[family]]\nname = 'a'\nbase = 'reference'\n[[family]]\nname = 'a'\nbase = 'reference'\n",
         "already defined"),
        ("[[family]]\nname = 'a'\nbase = 'nowhere'\n", "not a known scenario"),
        ("[[family]]\nname = 'a'\nbase = 'reference'\nspeed_mps = [9, 1]\n", "above the upper bound"),
        ("[scenario]\nname = 'x'\n", "array of tables"),
        ("[[widget]]\nname = 'x'\n", "unknown key\\(s\\) widget"),
    ],
)
def test_invalid_scenario_files_are_rejected(text, message):
    with pytest.raises(ValueError, match=message):
        parse_scenario_toml(text)


def test_load_errors_name_the_file(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("[[scenario]\nname = ", encoding="utf-8")
    with pytest.raises(ScenarioFileError, match="bad.toml"):
        load_scenario_file(bad)
    with pytest.raises(ScenarioFileError, match="missing.toml"):
        load_scenario_file(tmp_path / "missing.toml")


def test_files_can_build_on_each_other_but_not_redefine(tmp_path):
    first, second = tmp_path / "a.toml", tmp_path / "b.toml"
    first.write_text("[[scenario]]\nname = 'base_a'\nbase = 'reference'\nduration_s = 120.0\n", encoding="utf-8")
    second.write_text("[[family]]\nname = 'fam_b'\nbase = 'base_a'\n", encoding="utf-8")
    sources = available_sources([first, second])
    assert sources["fam_b"].for_seed(0).duration_s == 120.0
    with pytest.raises(ScenarioFileError, match="already defined"):
        available_sources([first, first])


# -- command line --------------------------------------------------------------------------------------------


def test_cli_lists_scenarios_and_families(capsys):
    assert main(["scenarios", "--scenario-file", str(EXAMPLE)]) == 0
    out = capsys.readouterr().out
    for name in [*SCENARIOS, *FAMILIES, "close_bearing_sensor", "wide_crossing"]:
        assert f"  {name} " in out


def test_cli_toml_output_loads_back(tmp_path, capsys):
    assert main(["scenarios", "--toml", "range_ratio"]) == 0
    text = capsys.readouterr().out.replace('name = "range_ratio"', 'name = "my_ratio"')
    path = tmp_path / "mine.toml"
    path.write_text(text, encoding="utf-8")
    loaded = load_scenario_file(path)["my_ratio"]
    assert loaded.model_copy(update={"name": "range_ratio"}) == FAMILIES["range_ratio"]


def test_cli_compare_families(capsys):
    assert main(["compare", "--scenario", "families", "--seeds", "2", "--workers", "1"]) == 0
    out = capsys.readouterr().out
    for name in FAMILIES:
        assert f"== {name} (family: a different world for each seed):" in out
    assert out.count("VERDICT: ") == len(FAMILIES)


def test_cli_bad_scenario_file_is_a_usage_error(tmp_path, capsys):
    bad = tmp_path / "bad.toml"
    bad.write_text("[[scenario]]\nname = 'x'\nbase = 'nowhere'\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["compare", "--scenario-file", str(bad), "--seeds", "2"])
    assert exc.value.code == 2
    assert "bad.toml" in capsys.readouterr().err
