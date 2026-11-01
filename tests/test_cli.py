"""Command-line interface tests (L6)."""

import re
from concurrent.futures.process import BrokenProcessPool

import pytest

import airspace_fusion.cli as cli
from airspace_fusion.cli import main
from airspace_fusion.config import SCENARIOS
from airspace_fusion.evaluation import METRIC_NAMES, Verdict


def run_cli(args, capsys) -> str:
    main(args)  # returns normally on success (exit code 0)
    out = capsys.readouterr().out
    assert out.isascii(), "output must be plain ASCII for the Windows console"
    return out


def metric_rows(out: str) -> dict[str, list[str]]:
    rows = {}
    for line in out.splitlines():
        parts = line.split()
        if parts and parts[0] in METRIC_NAMES:
            rows.setdefault(parts[0], []).append(line)
    return rows


@pytest.mark.parametrize(
    "args, message",
    [
        (["compare", "--seeds", "0"], "must be at least 1"),
        (["compare", "--seeds", "-3"], "must be at least 1"),
        (["compare", "--seeds", "two"], "invalid"),
        (["compare", "--workers", "0"], "must be at least 1"),
        (["learn", "--workers", "-1"], "must be at least 1"),
        (["run", "--seed", "1.5"], "invalid int value"),
        (["run", "--scenario", "nope"], "invalid choice"),
        (["compare", "--scenario", "nope"], "invalid choice"),
        ([], "required"),
        (["bogus"], "invalid choice"),
    ],
)
def test_bad_arguments_exit_with_usage_error(args, message, capsys):
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_run_every_scenario(scenario, capsys):
    out = run_cli(["run", "--scenario", scenario, "--seed", "3"], capsys)
    assert out.startswith(f"scenario={scenario} seed=3 ")
    rows = metric_rows(out)
    assert set(rows) == set(METRIC_NAMES)
    for (line,) in rows.values():
        values = line.split()[1:]
        assert len(values) == 2, line  # baseline, fusion
        for v in values:
            float(v)  # parses as a number (nan allowed)


def test_compare_single_scenario_with_first_seed(capsys):
    out = run_cli(["compare", "--scenario", "distant_bearing", "--seeds", "2", "--first-seed", "5"], capsys)
    assert out.startswith("seeds 5..6 (n=2)")
    assert out.count("== ") == 1 and "== distant_bearing:" in out
    rows = metric_rows(out)
    assert set(rows) == set(METRIC_NAMES)
    for (line,) in rows.values():
        assert len(re.findall(r"\S+ \+/- \S+", line)) == 2, line
    verdicts = re.findall(r"^VERDICT: (.+)$", out, flags=re.MULTILINE)
    assert len(verdicts) == 1 and verdicts[0] in {v.value for v in Verdict}
    assert re.search(r"paired t-test p=", out)


def test_compare_output_same_with_parallel_workers(capsys):
    args = ["compare", "--scenario", "reference", "--seeds", "4", "--first-seed", "7"]

    def without_timing(out: str) -> list[str]:
        return [line for line in out.splitlines() if not line.startswith("processing_time_s")]

    serial = run_cli([*args, "--workers", "1"], capsys)
    parallel = run_cli([*args, "--workers", "2"], capsys)
    assert without_timing(parallel) == without_timing(serial)


def test_crashed_worker_gives_clear_error(monkeypatch, capsys):
    def broken(*args, **kwargs):
        raise BrokenProcessPool("a child process terminated abruptly")

    monkeypatch.setattr(cli, "compare_modes", broken)
    assert main(["compare", "--scenario", "reference", "--seeds", "2"]) == 1
    err = capsys.readouterr().err
    assert "worker process stopped unexpectedly" in err and "--workers 1" in err


def test_crashed_worker_during_learn_gives_clear_error(monkeypatch, tmp_path, capsys):
    """The orchestrator retries failing steps, but a broken worker pool must stop the run with the same advice."""
    def broken(*args, **kwargs):
        raise BrokenProcessPool("a child process terminated abruptly")

    monkeypatch.setattr(cli, "compare_modes", broken)
    paths = ["--rules", str(tmp_path / "rules.json"), "--log", str(tmp_path / "log.jsonl")]
    assert main(["learn", "--scenario", "reference", *paths]) == 1
    err = capsys.readouterr().err
    assert "worker process stopped unexpectedly" in err and "--workers 1" in err


def test_interrupt_exits_with_status_130(monkeypatch, capsys):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "compare_modes", interrupted)
    assert main(["compare", "--scenario", "reference", "--seeds", "2"]) == 130
    assert "interrupted" in capsys.readouterr().err


def test_default_workers_fit_the_usable_cpus():
    assert 1 <= cli.DEFAULT_WORKERS <= min(cli._usable_cpus(), 4)


def test_compare_all_scenarios(capsys):
    out = run_cli(["compare", "--seeds", "2"], capsys)
    assert re.findall(r"^== (\w+):", out, flags=re.MULTILINE) == list(SCENARIOS)
    assert out.count("VERDICT: ") == len(SCENARIOS)
