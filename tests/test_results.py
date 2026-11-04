"""Sweeps and results files (M2): common random numbers, provenance, and reproducing a run."""

from __future__ import annotations

import csv
import json
import math

import pytest

from airspace_fusion.cli import SWEEP_CSV_COLUMNS, main
from airspace_fusion.config import SCENARIOS, TrackerConfig
from airspace_fusion.evaluation import Mode, Verdict, compare_modes, holm_adjust, sweep, with_holm
from airspace_fusion.results import (
    RESULTS_FORMAT,
    ComparisonFamily,
    ResultEntry,
    ResultsFileError,
    build_record,
    config_hash,
    load_record,
    reproduce,
    save_record,
    source_from_dict,
    source_to_dict,
    summary_to_dict,
)
from airspace_fusion.scenarios import FAMILIES, SWEEPABLE_PARAMETERS, ParameterRange, ScenarioFamily

CROSSING = FAMILIES["crossing_geometry"]
SEEDS = list(range(4))


# -- sweeps --------------------------------------------------------------------------------------------------


def test_sweepable_parameters_are_exactly_the_range_fields():
    ranges = {name for name, field in ScenarioFamily.model_fields.items() if "ParameterRange" in str(field.annotation)}
    assert set(SWEEPABLE_PARAMETERS) == ranges


def test_with_fixed():
    fixed = CROSSING.with_fixed("crossing_angle_deg", 30.0)
    assert fixed.crossing_angle_deg == ParameterRange(lo=30.0, hi=30.0)
    assert fixed.speed_mps == CROSSING.speed_mps
    with pytest.raises(ValueError, match="cannot sweep 'base'"):
        CROSSING.with_fixed("base", 1.0)
    with pytest.raises(ValueError, match="strictly between 0 and 180"):
        CROSSING.with_fixed("crossing_angle_deg", 0.0)


def test_sweep_points_share_everything_but_the_swept_parameter():
    points = sweep(CROSSING, "crossing_angle_deg", [10.0, 80.0], TrackerConfig(), SEEDS)
    assert [p.value for p in points] == [10.0, 80.0]
    # The baseline ignores the bearing sensor, whose position is all that the crossing angle changes:
    # with common random numbers its results are identical at every point.
    first, second = (p.summary.metrics[Mode.BASELINE] for p in points)
    assert {k: v for k, v in first.items() if k != "processing_time_s"} == {
        k: v for k, v in second.items() if k != "processing_time_s"}
    assert points[0].summary.metrics[Mode.FUSION] != points[1].summary.metrics[Mode.FUSION]


def test_sweep_rejects_bad_values():
    with pytest.raises(ValueError, match="at least one value"):
        sweep(CROSSING, "crossing_angle_deg", [], TrackerConfig(), SEEDS)
    with pytest.raises(ValueError, match="finite"):
        sweep(CROSSING, "crossing_angle_deg", [math.nan], TrackerConfig(), SEEDS)


# -- results records -----------------------------------------------------------------------------------------


def record_for(source=SCENARIOS["reference"], seeds=SEEDS, point=None, tracker=None) -> dict:
    tracker = tracker or TrackerConfig()
    summary = compare_modes(source, tracker, seeds)
    return build_record("compare", [ResultEntry(source, summary, point)], tracker, seeds, ["compare"])


@pytest.mark.parametrize("source", [SCENARIOS["biased_bearing"], FAMILIES["range_ratio"]], ids=lambda s: s.name)
def test_sources_round_trip_through_json(source):
    assert source_from_dict(json.loads(json.dumps(source_to_dict(source)))) == source
    with pytest.raises(ValueError, match="unknown source kind"):
        source_from_dict({"kind": "widget", "config": {}})


def test_summary_json_is_strict_and_complete():
    summary = compare_modes(SCENARIOS["reference"], TrackerConfig(), [0])  # one seed: no verdict, NaN p-value
    data = summary_to_dict(summary)
    json.dumps(data, allow_nan=False)  # NaN became null
    assert data["verdict"] == Verdict.INSUFFICIENT_DATA.value and data["rmse_p_value"] is None
    assert set(data["metrics"]) == {m.value for m in Mode}


def test_record_provenance_and_hash():
    record = record_for()
    assert record["format_version"] == RESULTS_FORMAT
    assert record["package_version"] and record["python"] and record["argv"] == ["compare"]
    assert record["seeds"] == SEEDS and record["tracker"] == TrackerConfig().model_dump(mode="json")
    assert record["config_hash"] == config_hash(record)
    # The hash covers what determines the numbers, and nothing else.
    assert record_for()["config_hash"] == record["config_hash"]
    assert record_for(seeds=[0, 1, 2])["config_hash"] != record["config_hash"]
    assert record_for(tracker=TrackerConfig(process_noise_m2_s3=0.7))["config_hash"] != record["config_hash"]


def test_save_load_and_reproduce(tmp_path):
    path = save_record(record_for(FAMILIES["clutter_density"]), tmp_path)
    assert path.name.endswith(f"-compare-{load_record(path)['config_hash'][:12]}.json")
    assert reproduce(load_record(path)) == []


def test_reproduce_reports_changed_numbers(tmp_path):
    record = record_for()
    record["entries"][0]["summary"]["metrics"]["fusion"]["position_rmse_m"]["mean"] += 0.5
    differences = reproduce(record)  # summaries are not hashed: editing one is caught here, not at load
    assert len(differences) == 1 and "fusion.position_rmse_m.mean" in differences[0]


def test_load_rejects_bad_files(tmp_path):
    good = save_record(record_for(), tmp_path)
    edited = json.loads(good.read_text(encoding="utf-8"))
    edited["seeds"] = [9, 10]
    (tmp_path / "edited.json").write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(ResultsFileError, match="config_hash"):
        load_record(tmp_path / "edited.json")
    (tmp_path / "old.json").write_text(json.dumps({"format_version": 99}), encoding="utf-8")
    with pytest.raises(ResultsFileError, match="format"):
        load_record(tmp_path / "old.json")
    (tmp_path / "junk.json").write_text("{", encoding="utf-8")
    with pytest.raises(ResultsFileError, match="cannot read"):
        load_record(tmp_path / "junk.json")


# -- command line --------------------------------------------------------------------------------------------


def test_cli_sweep_csv_results_and_reproduce(tmp_path, capsys, results_in_tmp):
    csv_path = tmp_path / "out" / "sweep.csv"
    args = ["sweep", "--scenario", "bias_magnitude", "--parameter", "bearing_bias_deg", "--values", "0:1.5:3",
            "--seeds", "4", "--workers", "1", "--csv", str(csv_path)]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "sweep of bearing_bias_deg in family bias_magnitude" in out
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [float(r["value"]) for r in rows] == [0.0, 0.75, 1.5]
    assert tuple(rows[0]) == SWEEP_CSV_COLUMNS
    # Up to about 2 deg a growing unmodelled bias makes fusion steadily worse (beyond that the gate rejects
    # most biased bearings and fusion recovers; see tests/test_statistics.py).
    fusion = [float(r["fusion_rmse_m"]) for r in rows]
    assert fusion[0] < fusion[1] < fusion[2]
    (record_path,) = results_in_tmp.glob("*-sweep-*.json")
    record = load_record(record_path)
    assert [e["point"] for e in record["entries"]] == [{"bearing_bias_deg": v} for v in (0.0, 0.75, 1.5)]
    assert record["argv"] == args
    assert main(["reproduce", str(record_path), "--workers", "1"]) == 0
    assert "reproduced: every metric matches" in capsys.readouterr().out


def test_cli_compare_saves_results_unless_asked_not_to(tmp_path, capsys, results_in_tmp):
    assert main(["compare", "--scenario", "reference", "--seeds", "2", "--no-results"]) == 0
    assert not results_in_tmp.exists()
    explicit = tmp_path / "explicit"
    assert main(["compare", "--scenario", "reference", "--seeds", "2", "--results-dir", str(explicit)]) == 0
    (path,) = explicit.glob("*-compare-*.json")
    assert f"results saved to {path}" in capsys.readouterr().out


def test_cli_reproduce_failure_exits_1(tmp_path, capsys):
    record = record_for()
    record["entries"][0]["summary"]["verdict"] = "fusion worsens"
    path = save_record(record, tmp_path)
    assert main(["reproduce", str(path), "--workers", "1"]) == 1
    assert "NOT REPRODUCED" in capsys.readouterr().err


@pytest.mark.parametrize(
    "args, message",
    [
        (["sweep", "--scenario", "reference", "--parameter", "speed_mps", "--values", "10"], "fixed scenario"),
        (["sweep", "--scenario", "crossing_geometry", "--parameter", "crossing_angle_deg", "--values", "0,30"],
         "strictly between 0 and 180"),
        (["sweep", "--scenario", "crossing_geometry", "--parameter", "crossing_angle_deg", "--values", "5:90:1"],
         "lo:hi:n with n >= 2"),
        (["sweep", "--scenario", "crossing_geometry", "--parameter", "colour", "--values", "1"], "invalid choice"),
        (["reproduce", "no/such/file.json"], "cannot read results file"),
    ],
)
def test_cli_sweep_and_reproduce_usage_errors(args, message, capsys):
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_sweep_points_are_holm_corrected():
    points = sweep(CROSSING, "crossing_angle_deg", [10.0, 80.0], TrackerConfig(), SEEDS)
    raw = [p.summary.rmse_p_value for p in points]
    assert [p.summary.rmse_p_adjusted for p in points] == pytest.approx(holm_adjust(raw))
    assert all(p.summary.rmse_p_adjusted >= p.summary.rmse_p_value for p in points)


def test_reproduce_redoes_holm_across_records(tmp_path):
    """A compare run split over two records (as a rule book does) keeps the whole family, and reproduces."""
    sources = [SCENARIOS["reference"], SCENARIOS["distant_bearing"], SCENARIOS["noisy_primary"]]
    tracker = TrackerConfig()
    summaries = with_holm([compare_modes(s, tracker, SEEDS) for s in sources])
    raw = [s.rmse_p_value for s in summaries]
    records = [
        build_record("compare", [ResultEntry(sources[i], summaries[i]) for i in positions], tracker, SEEDS,
                     ["compare"], ComparisonFamily(raw, positions))
        for positions in ([0, 2], [1])
    ]
    for record in records:
        assert record["multiple_comparisons"]["positions"] in ([0, 2], [1])
        assert reproduce(load_record(save_record(record, tmp_path))) == []
    # The other record's p-values take part in the correction: change one and this record's adjusted p moves.
    edited = json.loads(json.dumps(records[1]))
    edited["multiple_comparisons"]["p_values"][0] = 0.5
    assert any("rmse_p_adjusted" in d for d in reproduce(edited))


def test_single_comparison_records_have_no_family():
    record = record_for()
    assert record["multiple_comparisons"] is None
    assert record["entries"][0]["summary"]["rmse_p_adjusted"] is None
