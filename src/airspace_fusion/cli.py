"""Command-line entry point: run one simulation or a multi-seed baseline/fusion comparison."""

from __future__ import annotations

import argparse

from airspace_fusion.config import DEFAULT_SCENARIO, SCENARIOS, TrackerConfig
from airspace_fusion.evaluation import METRIC_NAMES, Mode, compare_modes, run_single


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
    print(f"seeds {seeds[0]}..{seeds[-1]} (n={len(seeds)}); values are mean +/- sample std across seeds\n")
    for name in names:
        s = compare_modes(SCENARIOS[name], TrackerConfig(), seeds)
        print(f"== {name}: {SCENARIOS[name].description}")
        print(f"{'metric':<24}{'baseline':>22}{'fusion':>22}")
        for metric in METRIC_NAMES:
            cells = [s.metrics[m][metric] for m in Mode]
            print(f"{metric:<24}" + "".join(f"{f'{c.mean:.4g} +/- {c.std:.2g}':>22}" for c in cells))
        print(f"RMSE delta (fusion - baseline): {s.rmse_delta_mean_m:+.2f} m "
              f"[95% CI {s.rmse_delta_ci95_m[0]:+.2f}, {s.rmse_delta_ci95_m[1]:+.2f}], "
              f"relative {s.rmse_relative_change:+.1%}, paired t-test p={s.rmse_p_value:.3g}")
        print(f"VERDICT: {s.verdict.value}\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="airspace-fusion", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="single seeded simulation, both modes")
    run_p.add_argument("--scenario", choices=list(SCENARIOS), default=DEFAULT_SCENARIO)
    run_p.add_argument("--seed", type=int, default=0)
    run_p.set_defaults(func=_run)

    cmp_p = sub.add_parser("compare", help="multi-seed baseline vs fusion comparison")
    cmp_p.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="all")
    cmp_p.add_argument("--seeds", type=int, default=20)
    cmp_p.add_argument("--first-seed", type=int, default=0)
    cmp_p.set_defaults(func=_compare)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
