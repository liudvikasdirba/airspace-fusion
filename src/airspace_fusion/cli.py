"""Command-line entry point: simulations, multi-seed baseline/fusion comparisons, sweeps and learning."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import NoReturn

from airspace_fusion import __version__
from airspace_fusion.agents import LearningTask, Orchestrator, Skills, describe_tracker
from airspace_fusion.config import DEFAULT_SCENARIO, SCENARIOS, HealthMonitorConfig, TrackerConfig
from airspace_fusion.evaluation import (
    DEFAULT_SEEDS,
    METRIC_NAMES,
    MIN_PARALLEL_SEEDS,
    ComparisonSummary,
    Mode,
    SweepPoint,
    compare_modes,
    explain_verdict,
    health_roc,
    health_summary,
    run_single,
    seed_executor,
    sensor_health_metrics,
    sweep,
    with_holm,
)
from airspace_fusion.learning import RuleBook, RuleBookError, RunLog
from airspace_fusion.results import (
    ComparisonFamily,
    ResultEntry,
    ResultsFileError,
    build_record,
    load_record,
    reproduce,
    save_record,
)
from airspace_fusion.scenarios import (
    FAMILIES,
    SWEEPABLE_PARAMETERS,
    ScenarioFamily,
    ScenarioFileError,
    ScenarioSource,
    available_sources,
    to_toml,
)

DEFAULT_RULES_PATH = Path("data/learning/rules.json")
DEFAULT_LOG_PATH = Path("data/learning/log.jsonl")
DEFAULT_RESULTS_DIR = Path("data/results")
# Fewer training or held-out seeds than this cannot resolve the 2 % acceptance margin reliably (ADR 0010).
MIN_RELIABLE_LEARNING_SEEDS = 50
RESULTS_DIR_ENV = "AIRSPACE_FUSION_RESULTS_DIR"  # overrides the default results folder


def _usable_cpus() -> int:
    """CPUs this process may run on (its CPU affinity, e.g. a container's cpuset), not all CPUs in the machine."""
    process_cpu_count = getattr(os, "process_cpu_count", None)  # Python 3.13+
    if process_cpu_count is not None:
        return process_cpu_count() or 1
    sched_getaffinity = getattr(os, "sched_getaffinity", None)  # Linux
    if sched_getaffinity is not None:
        return len(sched_getaffinity(0)) or 1
    return os.cpu_count() or 1


# Each worker holds its own NumPy/SciPy/pydantic (about 100 MB). Beyond 4, gains are small and starting
# many workers at once ran an 8 GB machine out of memory, so more must be asked for explicitly.
DEFAULT_WORKERS = min(_usable_cpus(), 4)
WORKERS_HELP = (
    f"processes that evaluate seeds in parallel (default: {DEFAULT_WORKERS}; 1 runs serially). "
    f"Fewer than {MIN_PARALLEL_SEEDS} seeds per comparison always run serially."
)


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


SCENARIO_HELP = (
    "a scenario or family name (list them with 'airspace-fusion scenarios'); 'all' is the four built-in scenarios, "
    "'families' the five built-in families"
)


def _sources(args: argparse.Namespace) -> dict[str, ScenarioSource]:
    try:
        return available_sources(args.scenario_file or ())
    except ScenarioFileError as exc:
        _usage_error(args, str(exc))


def _selected(args: argparse.Namespace, groups: bool = True) -> list[ScenarioSource]:
    """The scenarios or families named by ``--scenario``: one name, or 'all' / 'families' where allowed."""
    sources = _sources(args)
    if groups and args.scenario == "all":
        return list(SCENARIOS.values())
    if groups and args.scenario == "families":
        return list(FAMILIES.values())
    if args.scenario not in sources:
        _usage_error(args, f"unknown scenario {args.scenario!r}; list them with 'airspace-fusion scenarios'")
    return [sources[args.scenario]]


def _run(args: argparse.Namespace) -> None:
    (source,) = _selected(args, groups=False)
    scenario = source.for_seed(args.seed)
    run = run_single(scenario, _tracker(args), args.seed)
    print(f"scenario={scenario.name} seed={args.seed} records={len(run.simulation.records)} "
          f"valid={len(run.measurements)} invalid={len(run.invalid_records)}")
    if isinstance(source, ScenarioFamily):
        print(f"world: {scenario.description}")
    print(f"{'metric':<24}{'baseline':>14}{'fusion':>14}")
    for name in METRIC_NAMES:
        values = [getattr(run.modes[m].metrics, name) for m in Mode]
        print(f"{name:<24}" + "".join(f"{v:>14.4g}" for v in values))
    for sensor in sensor_health_metrics(run).values():
        print(f"health {sensor.sensor_id}: biased={sensor.biased} detection_delay_s={sensor.detection_delay_s:.0f} "
              f"false_alarms={sensor.false_alarms} time_excluded_s={sensor.time_excluded_s:.0f} "
              f"held_alarms={sensor.held_alarms}")


MOTION_MODEL_HELP = (
    "tracker motion model: cv (nearly constant velocity, the default) or imm (constant velocity plus "
    "coordinated turns; ADR 0011). With --rules, replaces the rule book's model"
)


def _tracker(args: argparse.Namespace, base: TrackerConfig | None = None) -> TrackerConfig:
    """Default (or rule-book) tracker settings, with --motion-model and --health-monitor applied when given."""
    tracker = base or TrackerConfig()
    update: dict[str, object] = {}
    if args.motion_model is not None:
        update["motion_model"] = args.motion_model
    if args.health_monitor:
        update["health_monitor"] = HealthMonitorConfig()
    return TrackerConfig.model_validate(tracker.model_dump() | update) if update else tracker


def _usage_error(args: argparse.Namespace, message: str) -> NoReturn:
    parser: argparse.ArgumentParser = args.parser
    parser.error(message)  # prints usage and exits with status 2


def _load_rules(args: argparse.Namespace, must_exist: bool) -> RuleBook:
    try:
        return RuleBook.load(args.rules, must_exist=must_exist)
    except RuleBookError as exc:
        _usage_error(args, str(exc))


def _results_dir(args: argparse.Namespace) -> Path:
    return args.results_dir or Path(os.environ.get(RESULTS_DIR_ENV) or DEFAULT_RESULTS_DIR)


def _save_results(
    args: argparse.Namespace,
    command: str,
    entries: list[ResultEntry],
    tracker: TrackerConfig,
    seeds: list[int],
    family: ComparisonFamily | None = None,
) -> None:
    if args.no_results:
        return
    path = save_record(build_record(command, entries, tracker, seeds, args.argv, family), _results_dir(args))
    print(f"results saved to {path} (check with: airspace-fusion reproduce {path})")


def _compare(args: argparse.Namespace) -> None:
    sources = _selected(args)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    # An explicitly named rule book must exist: silently falling back to defaults would mislabel the results.
    rules = _load_rules(args, must_exist=True) if args.rules else None
    print(f"seeds {seeds[0]}..{seeds[-1]} (n={len(seeds)}); values are mean +/- sample std across seeds\n")
    runs: list[tuple[ScenarioSource, TrackerConfig, ComparisonSummary]] = []
    with seed_executor(args.workers, len(seeds)) as executor:
        for number, source in enumerate(sources, 1):
            tracker = _tracker(args, rules.tracker_for(source.name) if rules else None)
            runs.append((source, tracker, compare_modes(source, tracker, seeds, executor)))
            if len(sources) > 1:  # verdicts wait for every p-value, so show progress meanwhile
                print(f"compared {source.name} ({number}/{len(sources)})", file=sys.stderr, flush=True)
    # Every comparison of the run is one family of tests (ADR 0009).
    summaries = with_holm([summary for _, _, summary in runs])
    if len(summaries) > 1:
        print(f"verdicts are Holm-corrected across the {len(summaries)} comparisons of this run\n")
    # A results record holds one tracker; a rule book with per-scenario settings gives one record per setting.
    by_tracker: dict[TrackerConfig, list[tuple[int, ResultEntry]]] = {}
    for index, ((source, tracker, _), summary) in enumerate(zip(runs, summaries)):
        _print_comparison(source, tracker, summary, rules)
        by_tracker.setdefault(tracker, []).append((index, ResultEntry(source, summary)))
    raw = [summary.rmse_p_value for summary in summaries]
    for tracker, items in by_tracker.items():
        family = ComparisonFamily(raw, [i for i, _ in items]) if len(summaries) > 1 else None
        _save_results(args, "compare", [entry for _, entry in items], tracker, seeds, family)


def _print_comparison(
    source: ScenarioSource, tracker: TrackerConfig, s: ComparisonSummary, rules: RuleBook | None
) -> None:
    name = source.name
    kind = " (family: a different world for each seed)" if isinstance(source, ScenarioFamily) else ""
    print(f"== {name}{kind}: {source.description}")
    if rules:
        print(f"learned rule book v{rules.version}: {describe_tracker(tracker)}, fusion "
              f"{'allowed' if rules.fusion_enabled(name) else 'recommended against'}")
    print(f"{'metric':<24}{'baseline':>22}{'fusion':>22}")
    for metric in METRIC_NAMES:
        cells = [s.metrics[m][metric] for m in Mode]
        print(f"{metric:<24}" + "".join(f"{f'{c.mean:.4g} +/- {c.std:.2g}':>22}" for c in cells))
    print(f"RMSE delta (fusion - baseline): {s.rmse_delta_mean_m:+.2f} m "
          f"[95% CI {s.rmse_delta_ci95_m[0]:+.2f}, {s.rmse_delta_ci95_m[1]:+.2f}], "
          f"relative {s.rmse_relative_change:+.1%}, paired t-test p={s.rmse_p_value:.3g}"
          + ("" if s.rmse_p_adjusted is None else f", Holm-adjusted p={s.rmse_p_adjusted:.3g}"))
    print(f"VERDICT: {s.verdict.value} ({explain_verdict(s)})\n")


def _sweep_values(text: str) -> list[float]:
    """Comma-separated values; ``lo:hi:n`` stands for n evenly spaced values from lo to hi."""
    values: list[float] = []
    for token in text.split(","):
        token = token.strip()
        try:
            if ":" in token:
                lo, hi, n = token.split(":")
                count = int(n)
                if count < 2:
                    raise ValueError
                values += [float(lo) + (float(hi) - float(lo)) * k / (count - 1) for k in range(count)]
            else:
                values.append(float(token))
        except ValueError:
            raise argparse.ArgumentTypeError(f"bad value {token!r}: use numbers, or lo:hi:n with n >= 2") from None
    return values


SWEEP_CSV_COLUMNS = (
    "family", "parameter", "value", "seeds", "baseline_rmse_m", "baseline_rmse_std", "fusion_rmse_m",
    "fusion_rmse_std", "rmse_delta_m", "rmse_delta_ci95_low", "rmse_delta_ci95_high", "rmse_relative_change",
    "p_value", "p_adjusted", "verdict", "baseline_gospa_m", "fusion_gospa_m", "baseline_coverage", "fusion_coverage",
    "baseline_containment_95", "fusion_containment_95",
)


def _sweep_row(family: str, parameter: str, point: SweepPoint) -> dict[str, object]:
    s = point.summary
    base, fused = s.metrics[Mode.BASELINE], s.metrics[Mode.FUSION]
    return {
        "family": family, "parameter": parameter, "value": point.value, "seeds": len(s.seeds),
        "baseline_rmse_m": base["position_rmse_m"].mean, "baseline_rmse_std": base["position_rmse_m"].std,
        "fusion_rmse_m": fused["position_rmse_m"].mean, "fusion_rmse_std": fused["position_rmse_m"].std,
        "rmse_delta_m": s.rmse_delta_mean_m, "rmse_delta_ci95_low": s.rmse_delta_ci95_m[0],
        "rmse_delta_ci95_high": s.rmse_delta_ci95_m[1], "rmse_relative_change": s.rmse_relative_change,
        "p_value": s.rmse_p_value, "p_adjusted": s.rmse_p_adjusted, "verdict": s.verdict.value,
        "baseline_gospa_m": base["gospa_m"].mean, "fusion_gospa_m": fused["gospa_m"].mean,
        "baseline_coverage": base["coverage"].mean, "fusion_coverage": fused["coverage"].mean,
        "baseline_containment_95": base["containment_95"].mean,
        "fusion_containment_95": fused["containment_95"].mean,
    }


def _sweep(args: argparse.Namespace) -> None:
    (source,) = _selected(args, groups=False)
    if not isinstance(source, ScenarioFamily):
        _usage_error(args, f"{source.name!r} is a fixed scenario; sweep needs a family ('airspace-fusion scenarios')")
    try:  # check every point before running any
        for value in args.values:
            source.with_fixed(args.parameter, value)
    except ValueError as exc:
        _usage_error(args, str(exc))
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    tracker = _tracker(args)
    print(f"sweep of {args.parameter} in family {source.name}; seeds {seeds[0]}..{seeds[-1]} (n={len(seeds)}); "
          "every point uses the same seeds and the same other draws\n")
    with seed_executor(args.workers, len(seeds)) as executor:
        points = sweep(source, args.parameter, args.values, tracker, seeds, executor)
    print(f"{args.parameter:>22}  {'baseline RMSE':>14}  {'fusion RMSE':>12}  {'change [95% CI]':>28}  "
          f"{'Holm p':>8}  {'fusion cont.':>12}  verdict")
    for point in points:
        s = point.summary
        base, fused = s.metrics[Mode.BASELINE], s.metrics[Mode.FUSION]
        low, high = s.rmse_delta_ci95_m
        change = f"{s.rmse_relative_change:+.1%} [{low:+.2f}, {high:+.2f}] m"
        print(f"{point.value:>22.4g}  {base['position_rmse_m'].mean:>12.2f} m  {fused['position_rmse_m'].mean:>10.2f} m"
              f"  {change:>28}  {s.decision_p_value:>8.2g}  {fused['containment_95'].mean:>12.2f}  {s.verdict.value}")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SWEEP_CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(_sweep_row(source.name, args.parameter, point) for point in points)
        print(f"\nCSV written to {args.csv}")
    entries = [ResultEntry(p.family, p.summary, {args.parameter: p.value}) for p in points]
    family = ComparisonFamily([p.summary.rmse_p_value for p in points], list(range(len(points))))
    _save_results(args, "sweep", entries, tracker, seeds, family if len(points) > 1 else None)


def _reproduce(args: argparse.Namespace) -> int:
    try:
        record = load_record(args.path)
    except ResultsFileError as exc:
        _usage_error(args, str(exc))
    count = len(record["entries"])
    dirty = ", with uncommitted changes" if record["git_dirty"] else ""
    print(f"re-running {count} entr{'y' if count == 1 else 'ies'} of a {record['command']} run from "
          f"{record['created']} (package {record['package_version']}, commit {str(record['git_commit'])[:12]}{dirty})")
    with seed_executor(args.workers, len(record["seeds"])) as executor:
        differences = reproduce(record, executor)
    if differences:
        print(f"NOT REPRODUCED: {len(differences)} value(s) differ", file=sys.stderr)
        for line in differences[:20]:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("reproduced: every metric matches (wall-clock time excluded)")
    return 0


def _monitored_tracker(args: argparse.Namespace) -> TrackerConfig:
    args.health_monitor = True  # these commands are about the monitor, so it is always on
    return _tracker(args)


def _health(args: argparse.Namespace) -> None:
    sources = _selected(args)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    tracker = _monitored_tracker(args)
    print(f"seeds {seeds[0]}..{seeds[-1]} (n={len(seeds)}), motion model {tracker.motion_model}\n")
    print(f"{'scenario':<20}{'false-alarm runs':>18}{'detected':>12}{'median delay':>14}{'time excluded':>15}")
    with seed_executor(args.workers, len(seeds)) as executor:
        for source in sources:
            s = health_summary(source, tracker, seeds, executor)
            detected = f"{s.detected_runs}/{s.biased_runs}" if s.biased_runs else "-"
            delay = f"{s.median_delay_s:.0f} s" if s.detected_runs else "-"
            print(f"{s.scenario:<20}{f'{s.runs_with_false_alarm}/{s.runs}':>18}{detected:>12}{delay:>14}"
                  f"{f'{s.mean_time_excluded_s:.0f} s':>15}")


ROC_CSV_COLUMNS = ("threshold_scale", "false_alarm_rate", "bias_deg", "detection_rate", "median_delay_s", "seeds")
ROC_BIASES_DEG = (0.5, 1.0, 1.5, 2.0, 3.0)
ROC_SCALES = (0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0)


def _roc(args: argparse.Namespace) -> None:
    (source,) = _selected(args, groups=False)
    if not isinstance(source, ScenarioFamily):
        _usage_error(args, f"{source.name!r} is a fixed scenario; roc needs a family with a bearing bias")
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    tracker = _monitored_tracker(args)
    print(f"ROC of the health monitor on family {source.name}, seeds {seeds[0]}..{seeds[-1]} (n={len(seeds)}), "
          f"motion model {tracker.motion_model}\ndelay: median s from run start; in brackets the detection rate\n")
    with seed_executor(args.workers, len(seeds)) as executor:
        points = health_roc(source, tracker, args.scales, ROC_BIASES_DEG, seeds, executor)
    print(f"{'scale':>6}{'false alarms':>14}" + "".join(f"{f'{b:g} deg':>14}" for b in ROC_BIASES_DEG))
    for point in points:
        cells = [f"{f'{s.median_delay_s:.0f}' if s.detected_runs else '-'} ({s.detection_rate:.2f})"
                 for s in point.detection.values()]
        print(f"{point.threshold_scale:>6.2f}{point.false_alarm_rate:>14.3f}" + "".join(f"{c:>14}" for c in cells))
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ROC_CSV_COLUMNS)
            writer.writeheader()
            for point in points:
                for bias, s in point.detection.items():
                    writer.writerow({
                        "threshold_scale": point.threshold_scale, "false_alarm_rate": point.false_alarm_rate,
                        "bias_deg": bias, "detection_rate": s.detection_rate, "median_delay_s": s.median_delay_s,
                        "seeds": s.runs,
                    })
        print(f"\nCSV written to {args.csv}")


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be at least 0, got {value}")
    return value


def _learn(args: argparse.Namespace) -> int:
    sources = _selected(args)
    try:
        task = LearningTask(
            scenarios=tuple(sources),
            train_seeds=tuple(range(args.train_first_seed, args.train_first_seed + args.train_seeds)),
            holdout_seeds=tuple(range(args.holdout_first_seed, args.holdout_first_seed + args.holdout_seeds)),
            max_candidates=args.max_candidates,
            test_seeds=tuple(range(args.test_first_seed, args.test_first_seed + args.test_seeds)),
        )
        task.check_rounds(args.rounds)
    except ValueError as exc:
        _usage_error(args, str(exc))
    if min(args.train_seeds, args.holdout_seeds) < MIN_RELIABLE_LEARNING_SEEDS:
        print(f"warning: fewer than {MIN_RELIABLE_LEARNING_SEEDS} training or held-out seeds; accepted settings "
              "may not carry over to new seeds", file=sys.stderr)
    # Microseconds keep run ids distinct when several runs start within the same second.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    rules = _load_rules(args, must_exist=False)
    seeds_per_comparison = min(len(task.train_seeds), len(task.holdout_seeds))
    with seed_executor(args.workers, seeds_per_comparison) as executor:
        skills = Skills(evaluate=partial(compare_modes, executor=executor))
        orchestrator = Orchestrator(rules, RunLog(run_id, args.log), skills=skills, rules_path=args.rules)
        outcome = orchestrator.run(task, rounds=args.rounds)
    print(f"run {run_id}: train seeds {task.train_seeds[0]}..{task.train_seeds[-1]}, "
          f"fresh held-out seeds every round from {task.holdout_seeds[0]}"
          + (f", sealed test seeds {task.test_seeds[0]}..{task.test_seeds[-1]}" if task.test_seeds else "")
          + "\n")
    print("\n".join(outcome.report))
    accepted = sum(lesson.accepted for lesson in outcome.lessons)
    print(f"\nrule book v{outcome.rules.version} saved to {args.rules} "
          f"({accepted} of {len(outcome.lessons)} lessons accepted in {outcome.rounds_run} round(s)); log: {args.log}")
    if outcome.failures:
        print(f"FAILED scenarios: {', '.join(g.scenario for g in outcome.failures)}", file=sys.stderr)
        return 1
    return 0


def _scenarios(args: argparse.Namespace) -> None:
    sources = _sources(args)
    if args.toml:
        if args.toml not in sources:
            _usage_error(args, f"unknown scenario {args.toml!r}")
        print(to_toml(sources[args.toml]), end="")
        return
    from_files = [s for s in sources.values() if s.name not in SCENARIOS and s.name not in FAMILIES]
    groups: list[tuple[str, list[ScenarioSource]]] = [
        ("Built-in scenarios", list(SCENARIOS.values())),
        ("Built-in families (a different world for each seed)", list(FAMILIES.values())),
        ("From scenario files", from_files),
    ]
    for title, entries in groups:
        if entries:
            print(title)
            for entry in entries:
                kind = "family" if isinstance(entry, ScenarioFamily) else "scenario"
                print(f"  {entry.name:<22} {kind:<9} {entry.description}")
            print()


def main(argv: list[str] | None = None) -> int:
    """Run the command line. Returns the process exit status (non-zero when a learning run had failures)."""
    parser = argparse.ArgumentParser(prog="airspace-fusion", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="single seeded simulation, both modes")
    run_p.add_argument("--scenario", default=DEFAULT_SCENARIO, help="a scenario or family name")
    run_p.add_argument("--seed", type=int, default=0)
    run_p.set_defaults(func=_run, parser=run_p)

    cmp_p = sub.add_parser("compare", help="multi-seed baseline vs fusion comparison")
    cmp_p.add_argument("--scenario", default="all", help=SCENARIO_HELP)
    cmp_p.add_argument("--seeds", type=_positive_int, default=DEFAULT_SEEDS)
    cmp_p.add_argument("--first-seed", type=int, default=0)
    cmp_p.add_argument("--rules", type=Path, help="use the tracker settings from a learned rule book")
    cmp_p.add_argument("--workers", type=_positive_int, default=DEFAULT_WORKERS, help=WORKERS_HELP)
    cmp_p.set_defaults(func=_compare, parser=cmp_p)

    learn_p = sub.add_parser("learn", help="tune tracker settings per scenario; results go to a rule book")
    learn_p.add_argument("--scenario", default="all", help=SCENARIO_HELP)
    learn_p.add_argument("--rounds", type=_positive_int, default=2)
    learn_p.add_argument("--train-seeds", type=_positive_int, default=50)
    learn_p.add_argument("--train-first-seed", type=int, default=1000)
    learn_p.add_argument("--holdout-seeds", type=_positive_int, default=50,
                         help="held-out seeds per round; each round uses the next block")
    learn_p.add_argument("--holdout-first-seed", type=int, default=2000)
    learn_p.add_argument("--test-seeds", type=_non_negative_int, default=200,
                         help="sealed test seeds, used once after learning (0 skips the test)")
    learn_p.add_argument("--test-first-seed", type=int, default=5000)
    learn_p.add_argument("--max-candidates", type=_positive_int, default=4)
    learn_p.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH)
    learn_p.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH)
    learn_p.add_argument("--workers", type=_positive_int, default=DEFAULT_WORKERS, help=WORKERS_HELP)
    learn_p.set_defaults(func=_learn, parser=learn_p)

    swp_p = sub.add_parser("sweep", help="paired comparison at each value of one family parameter")
    swp_p.add_argument("--scenario", required=True, help="the family to sweep")
    swp_p.add_argument("--parameter", required=True, choices=SWEEPABLE_PARAMETERS)
    swp_p.add_argument("--values", required=True, type=_sweep_values,
                       help="comma-separated values, or lo:hi:n for n evenly spaced values (e.g. 5:90:6)")
    swp_p.add_argument("--seeds", type=_positive_int, default=DEFAULT_SEEDS)
    swp_p.add_argument("--first-seed", type=int, default=0)
    swp_p.add_argument("--csv", type=Path, metavar="PATH", help="also write the table as CSV")
    swp_p.add_argument("--workers", type=_positive_int, default=DEFAULT_WORKERS, help=WORKERS_HELP)
    swp_p.set_defaults(func=_sweep, parser=swp_p)

    rep_p = sub.add_parser("reproduce", help="re-run a results file and check that every metric matches")
    rep_p.add_argument("path", type=Path, help="a results file written by compare or sweep")
    rep_p.add_argument("--workers", type=_positive_int, default=DEFAULT_WORKERS, help=WORKERS_HELP)
    rep_p.set_defaults(func=_reproduce, parser=rep_p)

    health_p = sub.add_parser("health", help="how the sensor health monitor behaves over many seeds (ADR 0012)")
    health_p.add_argument("--scenario", default="all", help=SCENARIO_HELP)
    health_p.add_argument("--seeds", type=_positive_int, default=DEFAULT_SEEDS)
    health_p.add_argument("--first-seed", type=int, default=0)
    health_p.add_argument("--workers", type=_positive_int, default=DEFAULT_WORKERS, help=WORKERS_HELP)
    health_p.set_defaults(func=_health, parser=health_p)

    roc_p = sub.add_parser("roc", help="health monitor detection delay against false-alarm rate, from a family")
    roc_p.add_argument("--scenario", default="bias_magnitude", help="a family with a bearing bias")
    roc_p.add_argument("--seeds", type=_positive_int, default=DEFAULT_SEEDS)
    roc_p.add_argument("--first-seed", type=int, default=0)
    roc_p.add_argument("--scales", type=_sweep_values, default=list(ROC_SCALES),
                       help="multipliers of the CUSUM alarm limits (comma-separated, or lo:hi:n)")
    roc_p.add_argument("--csv", type=Path, metavar="PATH", help="also write the table as CSV")
    roc_p.add_argument("--workers", type=_positive_int, default=DEFAULT_WORKERS, help=WORKERS_HELP)
    roc_p.set_defaults(func=_roc, parser=roc_p)

    for subparser in (cmp_p, swp_p):
        subparser.add_argument("--results-dir", type=Path, metavar="PATH",
                               help=f"where to save the results file (default: ${RESULTS_DIR_ENV} or "
                                    f"{DEFAULT_RESULTS_DIR})")
        subparser.add_argument("--no-results", action="store_true", help="do not save a results file")

    scn_p = sub.add_parser("scenarios", help="list scenarios and families, or print one as TOML")
    scn_p.add_argument("--toml", metavar="NAME", help="print this scenario or family as a scenario file")
    scn_p.set_defaults(func=_scenarios, parser=scn_p)

    for subparser in (run_p, cmp_p, swp_p, health_p, roc_p):
        subparser.add_argument("--motion-model", choices=("cv", "imm"), help=MOTION_MODEL_HELP)

    for subparser in (run_p, cmp_p, swp_p):
        subparser.add_argument("--health-monitor", action="store_true",
                               help="test each sensor's innovations and stop trusting inconsistent ones (ADR 0012)")

    for subparser in (run_p, cmp_p, swp_p, learn_p, scn_p, health_p, roc_p):
        subparser.add_argument("--scenario-file", type=Path, action="append", metavar="PATH",
                               help="TOML file with extra [[scenario]] and [[family]] entries (repeatable)")

    args = parser.parse_args(argv)
    args.argv = list(sys.argv[1:] if argv is None else argv)  # recorded in results files
    try:
        return args.func(args) or 0
    except BrokenProcessPool:
        print("error: a worker process stopped unexpectedly, usually because the machine ran out of memory. "
              "Run again with fewer --workers, or --workers 1.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130  # the conventional status for Ctrl-C


if __name__ == "__main__":
    sys.exit(main())
