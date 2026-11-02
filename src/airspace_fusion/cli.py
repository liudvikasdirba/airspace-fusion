"""Command-line entry point: run one simulation or a multi-seed baseline/fusion comparison."""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import Executor
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import NoReturn

from airspace_fusion import __version__
from airspace_fusion.agents import LearningTask, Orchestrator, Skills, describe_tracker
from airspace_fusion.config import DEFAULT_SCENARIO, SCENARIOS, TrackerConfig
from airspace_fusion.evaluation import (
    METRIC_NAMES,
    MIN_PARALLEL_SEEDS,
    Mode,
    compare_modes,
    run_single,
    seed_executor,
)
from airspace_fusion.learning import RuleBook, RuleBookError, RunLog
from airspace_fusion.scenarios import (
    FAMILIES,
    ScenarioFamily,
    ScenarioFileError,
    ScenarioSource,
    available_sources,
    to_toml,
)

DEFAULT_RULES_PATH = Path("data/learning/rules.json")
DEFAULT_LOG_PATH = Path("data/learning/log.jsonl")


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
    run = run_single(scenario, TrackerConfig(), args.seed)
    print(f"scenario={scenario.name} seed={args.seed} records={len(run.simulation.records)} "
          f"valid={len(run.measurements)} invalid={len(run.invalid_records)}")
    if isinstance(source, ScenarioFamily):
        print(f"world: {scenario.description}")
    print(f"{'metric':<24}{'baseline':>14}{'fusion':>14}")
    for name in METRIC_NAMES:
        values = [getattr(run.modes[m].metrics, name) for m in Mode]
        print(f"{name:<24}" + "".join(f"{v:>14.4g}" for v in values))


def _usage_error(args: argparse.Namespace, message: str) -> NoReturn:
    parser: argparse.ArgumentParser = args.parser
    parser.error(message)  # prints usage and exits with status 2


def _load_rules(args: argparse.Namespace, must_exist: bool) -> RuleBook:
    try:
        return RuleBook.load(args.rules, must_exist=must_exist)
    except RuleBookError as exc:
        _usage_error(args, str(exc))


def _compare(args: argparse.Namespace) -> None:
    sources = _selected(args)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    # An explicitly named rule book must exist: silently falling back to defaults would mislabel the results.
    rules = _load_rules(args, must_exist=True) if args.rules else None
    print(f"seeds {seeds[0]}..{seeds[-1]} (n={len(seeds)}); values are mean +/- sample std across seeds\n")
    with seed_executor(args.workers, len(seeds)) as executor:
        for source in sources:
            _print_comparison(source, seeds, rules, executor)


def _print_comparison(
    source: ScenarioSource, seeds: list[int], rules: RuleBook | None, executor: Executor | None
) -> None:
    name = source.name
    tracker = rules.tracker_for(name) if rules else TrackerConfig()
    s = compare_modes(source, tracker, seeds, executor)
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
          f"relative {s.rmse_relative_change:+.1%}, paired t-test p={s.rmse_p_value:.3g}")
    print(f"VERDICT: {s.verdict.value}\n")


def _learn(args: argparse.Namespace) -> int:
    sources = _selected(args)
    try:
        task = LearningTask(
            scenarios=tuple(sources),
            train_seeds=tuple(range(args.train_first_seed, args.train_first_seed + args.train_seeds)),
            holdout_seeds=tuple(range(args.holdout_first_seed, args.holdout_first_seed + args.holdout_seeds)),
            max_candidates=args.max_candidates,
        )
    except ValueError as exc:
        _usage_error(args, str(exc))
    # Microseconds keep run ids distinct when several runs start within the same second.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    rules = _load_rules(args, must_exist=False)
    seeds_per_comparison = min(len(task.train_seeds), len(task.holdout_seeds))
    with seed_executor(args.workers, seeds_per_comparison) as executor:
        skills = Skills(evaluate=partial(compare_modes, executor=executor))
        orchestrator = Orchestrator(rules, RunLog(run_id, args.log), skills=skills, rules_path=args.rules)
        outcome = orchestrator.run(task, rounds=args.rounds)
    print(f"run {run_id}: train seeds {task.train_seeds[0]}..{task.train_seeds[-1]}, "
          f"held-out seeds {task.holdout_seeds[0]}..{task.holdout_seeds[-1]}\n")
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
    cmp_p.add_argument("--seeds", type=_positive_int, default=20)
    cmp_p.add_argument("--first-seed", type=int, default=0)
    cmp_p.add_argument("--rules", type=Path, help="use the tracker settings from a learned rule book")
    cmp_p.add_argument("--workers", type=_positive_int, default=DEFAULT_WORKERS, help=WORKERS_HELP)
    cmp_p.set_defaults(func=_compare, parser=cmp_p)

    learn_p = sub.add_parser("learn", help="tune tracker settings per scenario; results go to a rule book")
    learn_p.add_argument("--scenario", default="all", help=SCENARIO_HELP)
    learn_p.add_argument("--rounds", type=_positive_int, default=2)
    learn_p.add_argument("--train-seeds", type=_positive_int, default=6)
    learn_p.add_argument("--train-first-seed", type=int, default=1000)
    learn_p.add_argument("--holdout-seeds", type=_positive_int, default=6)
    learn_p.add_argument("--holdout-first-seed", type=int, default=2000)
    learn_p.add_argument("--max-candidates", type=_positive_int, default=4)
    learn_p.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH)
    learn_p.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH)
    learn_p.add_argument("--workers", type=_positive_int, default=DEFAULT_WORKERS, help=WORKERS_HELP)
    learn_p.set_defaults(func=_learn, parser=learn_p)

    scn_p = sub.add_parser("scenarios", help="list scenarios and families, or print one as TOML")
    scn_p.add_argument("--toml", metavar="NAME", help="print this scenario or family as a scenario file")
    scn_p.set_defaults(func=_scenarios, parser=scn_p)

    for subparser in (run_p, cmp_p, learn_p, scn_p):
        subparser.add_argument("--scenario-file", type=Path, action="append", metavar="PATH",
                               help="TOML file with extra [[scenario]] and [[family]] entries (repeatable)")

    args = parser.parse_args(argv)
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
