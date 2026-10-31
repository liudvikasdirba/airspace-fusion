"""Command-line entry point: run one simulation or a multi-seed baseline/fusion comparison."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from airspace_fusion import __version__
from airspace_fusion.agents import LearningTask, Orchestrator, describe_tracker
from airspace_fusion.config import DEFAULT_SCENARIO, SCENARIOS, TrackerConfig
from airspace_fusion.evaluation import METRIC_NAMES, Mode, compare_modes, run_single
from airspace_fusion.learning import RuleBook, RunLog

DEFAULT_RULES_PATH = Path("data/learning/rules.json")
DEFAULT_LOG_PATH = Path("data/learning/log.jsonl")


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _run(args: argparse.Namespace) -> None:
    scenario = SCENARIOS[args.scenario]
    run = run_single(scenario, TrackerConfig(), args.seed)
    print(f"scenario={scenario.name} seed={args.seed} records={len(run.simulation.records)} "
          f"valid={len(run.measurements)} invalid={len(run.invalid_records)}")
    print(f"{'metric':<24}{'baseline':>14}{'fusion':>14}")
    for name in METRIC_NAMES:
        values = [getattr(run.modes[m].metrics, name) for m in Mode]
        print(f"{name:<24}" + "".join(f"{v:>14.4g}" for v in values))


def _compare(args: argparse.Namespace) -> None:
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    rules = RuleBook.load(args.rules) if args.rules else None
    print(f"seeds {seeds[0]}..{seeds[-1]} (n={len(seeds)}); values are mean +/- sample std across seeds\n")
    for name in names:
        tracker = rules.tracker_for(name) if rules else TrackerConfig()
        s = compare_modes(SCENARIOS[name], tracker, seeds)
        print(f"== {name}: {SCENARIOS[name].description}")
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


def _learn(args: argparse.Namespace) -> None:
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    try:
        task = LearningTask(
            scenarios=tuple(SCENARIOS[n] for n in names),
            train_seeds=tuple(range(args.train_first_seed, args.train_first_seed + args.train_seeds)),
            holdout_seeds=tuple(range(args.holdout_first_seed, args.holdout_first_seed + args.holdout_seeds)),
            max_candidates=args.max_candidates,
        )
    except ValueError as exc:
        args.parser.error(str(exc))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    orchestrator = Orchestrator(RuleBook.load(args.rules), RunLog(run_id, args.log), rules_path=args.rules)
    outcome = orchestrator.run(task, rounds=args.rounds)
    print(f"run {run_id}: train seeds {task.train_seeds[0]}..{task.train_seeds[-1]}, "
          f"held-out seeds {task.holdout_seeds[0]}..{task.holdout_seeds[-1]}\n")
    print("\n".join(outcome.report))
    accepted = sum(lesson.accepted for lesson in outcome.lessons)
    print(f"\nrule book v{outcome.rules.version} saved to {args.rules} "
          f"({accepted} of {len(outcome.lessons)} lessons accepted in {outcome.rounds_run} round(s)); log: {args.log}")
    if outcome.failures:
        print(f"FAILED scenarios: {', '.join(g.scenario for g in outcome.failures)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="airspace-fusion", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="single seeded simulation, both modes")
    run_p.add_argument("--scenario", choices=list(SCENARIOS), default=DEFAULT_SCENARIO)
    run_p.add_argument("--seed", type=int, default=0)
    run_p.set_defaults(func=_run)

    cmp_p = sub.add_parser("compare", help="multi-seed baseline vs fusion comparison")
    cmp_p.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="all")
    cmp_p.add_argument("--seeds", type=_positive_int, default=20)
    cmp_p.add_argument("--first-seed", type=int, default=0)
    cmp_p.add_argument("--rules", type=Path, help="use the tracker settings from a learned rule book")
    cmp_p.set_defaults(func=_compare)

    learn_p = sub.add_parser("learn", help="tune tracker settings per scenario; results go to a rule book")
    learn_p.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="all")
    learn_p.add_argument("--rounds", type=_positive_int, default=2)
    learn_p.add_argument("--train-seeds", type=_positive_int, default=6)
    learn_p.add_argument("--train-first-seed", type=int, default=1000)
    learn_p.add_argument("--holdout-seeds", type=_positive_int, default=6)
    learn_p.add_argument("--holdout-first-seed", type=int, default=2000)
    learn_p.add_argument("--max-candidates", type=_positive_int, default=4)
    learn_p.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH)
    learn_p.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH)
    learn_p.set_defaults(func=_learn, parser=learn_p)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
