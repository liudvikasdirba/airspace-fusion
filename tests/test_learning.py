"""Learning loop and agent orchestration: scoring, gaps, lessons, rule book, orchestrator."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest
from pydantic import ValidationError

import airspace_fusion.agents as agents
from airspace_fusion.agents import PLAN, LearningTask, Orchestrator, SealedTestResult, Skills
from airspace_fusion.cli import main
from airspace_fusion.config import SCENARIOS, TrackerConfig
from airspace_fusion.evaluation import METRIC_NAMES, ComparisonSummary, MetricSummary, Mode, RunMetrics, Verdict
from airspace_fusion.learning import (
    MAX_LESSONS_PER_SCENARIO,
    TUNABLE_BOUNDS,
    Gap,
    GapKind,
    Lesson,
    LessonAction,
    RuleBook,
    RuleBookError,
    RunLog,
    find_gaps,
    paired_improvement_p,
    propose_candidates,
    same_settings,
    score,
    seed_scores,
)

REFERENCE = SCENARIOS["reference"]


def seed_noise(seed: int) -> float:
    """Per-seed RMSE offset, the same for every setting: like common random numbers in a real run."""
    return float((seed * 7) % 5 - 2)


def run_metrics(rmse: float, containment: float) -> RunMetrics:
    return RunMetrics(rmse, rmse, 0.99, containment, 0, 0, 0, 0, 1, 0.0)


def summary(
    rmse: dict[Mode, float], containment: dict[Mode, float], coverage: float = 0.99, initiations: float = 1.0,
    verdict: Verdict = Verdict.IMPROVES, name: str = "reference", seeds: tuple[int, ...] = (0, 1),
) -> ComparisonSummary:
    def metrics(mode: Mode) -> dict[str, MetricSummary]:
        values = dict.fromkeys(METRIC_NAMES, 0.0) | {
            "position_rmse_m": rmse[mode], "containment_95": containment[mode], "coverage": coverage,
            "track_initiations": initiations,
        }
        return {k: MetricSummary(v, 0.0, 6) for k, v in values.items()}

    delta = rmse[Mode.FUSION] - rmse[Mode.BASELINE]
    per_seed = {m: tuple(run_metrics(rmse[m] + seed_noise(s), containment[m]) for s in seeds) for m in Mode}
    return ComparisonSummary(name, seeds, {m: metrics(m) for m in Mode}, delta, (delta, delta),
                             delta / rmse[Mode.BASELINE], 0.01, verdict, per_seed=per_seed)


def both(value: float) -> dict[Mode, float]:
    return dict.fromkeys(Mode, value)


# -- score and gaps ------------------------------------------------------------------------------


def test_score_penalises_containment_gap_in_both_directions():
    assert score(summary(both(20.0), both(0.95)), list(Mode)) == pytest.approx(20.0)
    assert score(summary(both(20.0), both(0.92)), list(Mode)) == pytest.approx(20.0 * 1.15)
    assert score(summary(both(20.0), both(0.98)), list(Mode)) == pytest.approx(20.0 * 1.15)


def test_score_uses_only_requested_modes_and_is_inf_without_a_track():
    s = summary({Mode.BASELINE: 10.0, Mode.FUSION: 30.0}, both(0.95))
    assert score(s, [Mode.BASELINE]) == pytest.approx(10.0)
    assert score(s, list(Mode)) == pytest.approx(20.0)
    assert score(summary({Mode.BASELINE: math.nan, Mode.FUSION: 1.0}, both(0.95)), list(Mode)) == math.inf


def test_gaps():
    s = summary(both(20.0), {Mode.BASELINE: 0.90, Mode.FUSION: 0.99}, coverage=0.9, initiations=1.5,
                verdict=Verdict.WORSENS)
    kinds = {(g.kind, g.mode) for g in find_gaps(s, list(Mode))}
    assert kinds == {
        (GapKind.OVER_CONFIDENT, Mode.BASELINE), (GapKind.UNDER_CONFIDENT, Mode.FUSION),
        (GapKind.LOW_COVERAGE, Mode.BASELINE), (GapKind.LOW_COVERAGE, Mode.FUSION),
        (GapKind.TRACK_RESTARTS, Mode.BASELINE), (GapKind.TRACK_RESTARTS, Mode.FUSION),
        (GapKind.FUSION_HARMFUL, Mode.FUSION),
    }
    # Fusion gaps are not reported when fusion is not in use.
    assert all(g.mode is Mode.BASELINE for g in find_gaps(s, [Mode.BASELINE]))
    assert find_gaps(summary(both(20.0), both(0.96)), list(Mode)) == []


# -- candidate proposals -------------------------------------------------------------------------


def gap(kind: GapKind) -> Gap:
    return Gap(kind, "reference", Mode.BASELINE, 0.0, 0.0)


def test_over_confidence_raises_process_noise_and_under_confidence_lowers_it():
    base = TrackerConfig()
    up = propose_candidates(base, [gap(GapKind.OVER_CONFIDENT)], [], 4)
    assert [c.process_noise_m2_s3 for c in up] == [0.75, 1.125]
    down = propose_candidates(base, [gap(GapKind.UNDER_CONFIDENT)], [], 4)
    assert all(c.process_noise_m2_s3 < base.process_noise_m2_s3 for c in down)


def test_lost_tracks_lengthen_coast_and_widen_gate():
    base = TrackerConfig()
    cands = propose_candidates(base, [gap(GapKind.TRACK_RESTARTS)], [], 4)
    assert any(c.max_coast_s > base.max_coast_s for c in cands)
    assert any(c.gate_probability > base.gate_probability for c in cands)


def test_proposals_stay_in_bounds_skip_tried_and_respect_the_cap():
    top = TrackerConfig(process_noise_m2_s3=TUNABLE_BOUNDS["process_noise_m2_s3"][1])
    cands = propose_candidates(top, [gap(GapKind.OVER_CONFIDENT)], [], 4)
    assert cands == []  # both steps clamp to the incumbent
    base = TrackerConfig()
    first = propose_candidates(base, [], [], 4)
    assert len(first) == 2
    again = propose_candidates(base, [], [first[0]], 4)
    assert not any(same_settings(c, first[0]) for c in again)
    assert len(propose_candidates(base, [gap(GapKind.OVER_CONFIDENT), gap(GapKind.TRACK_RESTARTS)], [], 3)) == 3


# -- rule book -----------------------------------------------------------------------------------


def lesson(accepted: bool, q: float = 0.75, scenario: str = "reference") -> Lesson:
    return Lesson(run_id="r", scenario=scenario, action=LessonAction.SET_TRACKER, triggers=(),
                  tracker=TrackerConfig(process_noise_m2_s3=q), accepted=accepted, reason="test")


def test_rule_book_applies_only_accepted_lessons_but_remembers_all():
    rules = RuleBook().with_lessons([lesson(False, 0.4), lesson(True, 0.75)])
    assert rules.version == 1
    assert rules.tracker_for("reference").process_noise_m2_s3 == 0.75
    assert rules.tracker_for("other") == TrackerConfig()
    assert [t.process_noise_m2_s3 for t in rules.tried_trackers("reference")] == [0.4, 0.75, 0.5]
    assert rules.tried_trackers("other") == []
    assert len(rules.lessons) == 2
    assert RuleBook().with_lessons([lesson(False)]).version == 0  # memory only, no rule change


def test_rule_book_fusion_policy():
    disable = Lesson(run_id="r", scenario="biased_bearing", action=LessonAction.DISABLE_FUSION, triggers=(),
                     accepted=True, reason="test")
    rules = RuleBook().with_lessons([disable, disable])
    assert not rules.fusion_enabled("biased_bearing") and rules.fusion_enabled("reference")
    assert rules.fusion_disabled == ("biased_bearing",)


def test_rule_book_rejects_out_of_bounds_settings_and_inconsistent_lessons():
    with pytest.raises(ValidationError, match="outside"):
        RuleBook().with_lessons([lesson(True, q=50.0)])
    with pytest.raises(ValidationError, match="tracker is required"):
        Lesson(run_id="r", scenario="x", action=LessonAction.SET_TRACKER, triggers=(), accepted=True, reason="")


def test_rule_book_round_trip(tmp_path):
    path = tmp_path / "sub" / "rules.json"
    assert RuleBook.load(path) == RuleBook()
    rules = RuleBook().with_lessons([lesson(True), lesson(False, 0.4)])
    rules.save(path)
    assert RuleBook.load(path) == rules


def test_lesson_memory_keeps_newest_per_scenario():
    n = MAX_LESSONS_PER_SCENARIO
    older = [lesson(False, 0.1 + i * 0.001) for i in range(n)]
    newer = [lesson(False, 0.3 + i * 0.001) for i in range(10)]
    other = [lesson(False, 0.2, scenario="noisy_primary")]
    rules = RuleBook().with_lessons(older + other).with_lessons(newer)
    reference = [x for x in rules.lessons if x.scenario == "reference"]
    assert len(reference) == n
    assert reference[-10:] == newer and reference[0] == older[10]  # the 10 oldest were dropped
    assert [x for x in rules.lessons if x.scenario == "noisy_primary"] == other


def test_rule_book_load_errors(tmp_path):
    missing = tmp_path / "missing.json"
    assert RuleBook.load(missing) == RuleBook()
    with pytest.raises(RuleBookError, match="does not exist"):
        RuleBook.load(missing, must_exist=True)
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text('{"trackers": {', encoding="utf-8")
    with pytest.raises(RuleBookError, match="cannot read rule book .*corrupt.json"):
        RuleBook.load(corrupt)


def test_run_log_writes_strict_json_lines(tmp_path):
    path = tmp_path / "log.jsonl"
    log = RunLog("r1", path)
    log.record("evaluation", score=math.inf, verdict=Verdict.WORSENS, gaps=[gap(GapKind.OVER_CONFIDENT)],
               tracker=TrackerConfig())
    log.record("step_done", step="a.b")
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [e["event"] for e in lines] == ["evaluation", "step_done"]
    assert lines[0]["score"] is None and lines[0]["verdict"] == "fusion worsens"
    assert lines[0]["gaps"][0]["kind"] == "over_confident"
    assert lines[0]["tracker"]["process_noise_m2_s3"] == 0.5
    assert log.events("step_done")[0]["step"] == "a.b"


# -- orchestrator with a synthetic evaluator ----------------------------------------------------


class FakeEvaluator:
    """RMSE is a bowl around q = 0.8 and containment rises with q, so more process noise helps up to 0.8."""

    def __init__(self, verdict: Verdict = Verdict.IMPROVES, fail_times: int = 0) -> None:
        self.calls: list[tuple[str, float, tuple[int, ...]]] = []
        self.verdict = verdict
        self.fail_times = fail_times

    def __call__(self, scenario, tracker, seeds):
        if self.fail_times:
            self.fail_times -= 1
            raise RuntimeError("transient failure")
        q = tracker.process_noise_m2_s3
        self.calls.append((scenario.name, q, tuple(seeds)))
        rmse = 20.0 + 40.0 * (q - 0.8) ** 2
        containment = min(0.95, 0.88 + 0.1 * q)
        return summary({Mode.BASELINE: rmse, Mode.FUSION: rmse * 0.9}, both(containment), verdict=self.verdict,
                       name=scenario.name, seeds=tuple(seeds))


def task(**overrides) -> LearningTask:
    values = dict(scenarios=(REFERENCE,), train_seeds=(1, 2), holdout_seeds=(3, 4))
    return LearningTask(**(values | overrides))


def test_task_rules():
    with pytest.raises(ValueError, match="disjoint"):
        task(holdout_seeds=(2, 3))
    with pytest.raises(ValueError, match="two seeds"):
        task(train_seeds=(1,))
    with pytest.raises(ValueError, match="scenario"):
        task(scenarios=())


def test_closed_loop_improves_and_converges(tmp_path):
    evaluator = FakeEvaluator()
    log = RunLog("r1")
    path = tmp_path / "rules.json"
    outcome = Orchestrator(RuleBook(), log, Skills(evaluate=evaluator), rules_path=path).run(task(), rounds=5)

    q = outcome.rules.tracker_for("reference").process_noise_m2_s3
    assert 0.6 < q <= 1.0
    assert outcome.rules.version >= 1
    assert outcome.rounds_run < 5  # stops once a round accepts nothing
    assert RuleBook.load(path) == outcome.rules
    accepted = [lesson for lesson in outcome.lessons if lesson.accepted]
    assert accepted and all(lesson.holdout_score_after < lesson.holdout_score_before for lesson in accepted)
    # Accepted settings were checked on the held-out seeds; no setting is tried twice.
    assert any(seeds == (3, 4) for _, _, seeds in evaluator.calls)
    # Results are cached: no (scenario, settings, seeds) combination is evaluated twice.
    assert len(evaluator.calls) == len(set(evaluator.calls))
    tried = [c for c in (lesson.tracker for lesson in outcome.lessons) if c is not None]
    assert len({c.process_noise_m2_s3 for c in tried}) == len(tried)
    # Every planned step ran and was tracked.
    assert {e["step"] for e in log.events("step_done")} == {f"{a}.{s}" for a, s in PLAN}
    assert log.events("learning_update")


def test_next_run_starts_from_learned_rules():
    first = Orchestrator(RuleBook(), RunLog("r1"), Skills(evaluate=FakeEvaluator())).run(task(), rounds=1)
    evaluator = FakeEvaluator()
    Orchestrator(first.rules, RunLog("r2"), Skills(evaluate=evaluator)).run(task(), rounds=1)
    learned_q = first.rules.tracker_for("reference").process_noise_m2_s3
    assert evaluator.calls[0] == ("reference", learned_q, (1, 2))


@pytest.mark.parametrize("verdict", [Verdict.WORSENS, Verdict.OVER_CONFIDENT])
def test_harmful_fusion_is_disabled_after_holdout_confirmation(verdict):
    outcome = Orchestrator(RuleBook(), RunLog("r"), Skills(evaluate=FakeEvaluator(verdict))).run(task())
    assert not outcome.rules.fusion_enabled("reference")
    (fusion_lesson,) = [x for x in outcome.lessons if x.action is LessonAction.DISABLE_FUSION]
    assert fusion_lesson.accepted and fusion_lesson.triggers == (GapKind.FUSION_HARMFUL,)


def test_transient_failure_is_retried():
    log = RunLog("r")
    outcome = Orchestrator(RuleBook(), log, Skills(evaluate=FakeEvaluator(fail_times=1))).run(task())
    assert outcome.failures == ()
    assert len(log.events("step_error")) == 1


def test_persistent_failure_skips_the_scenario_without_stopping_the_run():
    log = RunLog("r")
    outcome = Orchestrator(RuleBook(), log, Skills(evaluate=FakeEvaluator(fail_times=99)), max_retries=2).run(task())
    assert [g.scenario for g in outcome.failures] == ["reference"]
    assert len(log.events("step_error")) == 3
    assert outcome.rules == RuleBook()
    assert any("FAILED at research.diagnose" in line for line in outcome.report)


def test_retried_fusion_step_records_its_lesson_once(monkeypatch):
    real_find_gaps = agents.find_gaps
    failed = []

    def flaky_find_gaps(summary_, modes):
        # Fail once, on the baseline-only re-diagnosis that happens after fusion is found harmful.
        if tuple(modes) == (Mode.BASELINE,) and not failed:
            failed.append(True)
            raise RuntimeError("transient failure")
        return real_find_gaps(summary_, modes)

    monkeypatch.setattr(agents, "find_gaps", flaky_find_gaps)
    log = RunLog("r")
    outcome = Orchestrator(RuleBook(), log, Skills(evaluate=FakeEvaluator(Verdict.WORSENS))).run(task())
    assert failed and len(log.events("step_error")) == 1
    assert len([x for x in outcome.lessons if x.action is LessonAction.DISABLE_FUSION]) == 1
    assert not outcome.rules.fusion_enabled("reference")


class FailingFor(FakeEvaluator):
    def __init__(self, bad: str) -> None:
        super().__init__()
        self.bad = bad

    def __call__(self, scenario, tracker, seeds):
        if scenario.name == self.bad:
            raise RuntimeError("always fails")
        return super().__call__(scenario, tracker, seeds)


def test_scenario_failing_in_several_rounds_is_reported_once():
    bad = REFERENCE.model_copy(update={"name": "bad"})
    outcome = Orchestrator(RuleBook(), RunLog("r"), Skills(evaluate=FailingFor("bad"))).run(
        task(scenarios=(REFERENCE, bad)), rounds=4)
    assert outcome.rounds_run >= 2  # the healthy scenario keeps the run going
    assert [g.scenario for g in outcome.failures] == ["bad"]


# -- real simulation, end to end ------------------------------------------------------------------


def test_learn_on_short_real_scenario():
    short = REFERENCE.model_copy(update={"name": "short", "duration_s": 40.0})
    log = RunLog("r")
    outcome = Orchestrator(RuleBook(), log).run(task(scenarios=(short,), max_candidates=2), rounds=1)
    assert outcome.failures == ()
    assert outcome.lessons
    assert all(math.isfinite(e["score"]) for e in log.events("screened"))


def test_cli_learn_then_compare_with_rules(tmp_path, capsys):
    rules, log = tmp_path / "rules.json", tmp_path / "log.jsonl"
    main(["learn", "--scenario", "distant_bearing", "--rounds", "1", "--train-seeds", "2", "--holdout-seeds", "2",
          "--max-candidates", "1", "--rules", str(rules), "--log", str(log)])
    out = capsys.readouterr().out
    assert out.isascii()
    assert "== distant_bearing:" in out and "rule book v" in out
    assert RuleBook.load(rules).lessons
    assert log.read_text(encoding="utf-8").count("\n") >= len(PLAN)

    assert main(["compare", "--scenario", "distant_bearing", "--seeds", "2", "--rules", str(rules)]) == 0
    assert "learned rule book v" in capsys.readouterr().out


@pytest.mark.slow
def test_cli_learn_same_with_parallel_workers(tmp_path, capsys):
    """Parallel evaluation must not change what is learned."""
    books = []
    for workers in ("1", "2"):
        rules = tmp_path / f"rules-{workers}.json"
        main(["learn", "--scenario", "reference", "--rounds", "1", "--train-seeds", "4", "--holdout-seeds", "4",
              "--max-candidates", "2", "--rules", str(rules), "--log", str(tmp_path / "log.jsonl"),
              "--workers", workers])
        capsys.readouterr()
        book = json.loads(rules.read_text(encoding="utf-8"))
        for lesson in book["lessons"]:
            lesson.pop("run_id")  # derived from the start time
        books.append(book)
    assert books[0]["lessons"]
    assert books[0] == books[1]


def test_cli_compare_rejects_missing_or_corrupt_rule_book(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["compare", "--scenario", "reference", "--seeds", "2", "--rules", str(tmp_path / "typo.json")])
    assert exc.value.code == 2
    assert "does not exist" in capsys.readouterr().err

    corrupt = tmp_path / "rules.json"
    corrupt.write_text("not json", encoding="utf-8")
    for argv in (["compare", "--seeds", "2", "--rules", str(corrupt)],
                 ["learn", "--rules", str(corrupt), "--log", str(tmp_path / "log.jsonl")]):
        with pytest.raises(SystemExit) as exc:
            main(argv)
        assert exc.value.code == 2
        assert "cannot read rule book" in capsys.readouterr().err


def test_cli_learn_exits_non_zero_when_scenarios_fail(tmp_path, capsys, monkeypatch):
    def broken(self, work, ctx):
        raise FloatingPointError("numerical failure")

    monkeypatch.setattr(agents.ResearchAgent, "diagnose", broken)
    status = main(["learn", "--scenario", "reference", "--rounds", "1", "--train-seeds", "2", "--holdout-seeds", "2",
                   "--rules", str(tmp_path / "rules.json"), "--log", str(tmp_path / "log.jsonl")])
    assert status == 1
    assert "FAILED scenarios: reference" in capsys.readouterr().err


def test_cli_learn_rejects_overlapping_seeds(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["learn", "--train-first-seed", "0", "--holdout-first-seed", "3", "--rules", str(tmp_path / "r.json")])
    assert exc.value.code == 2
    assert "disjoint" in capsys.readouterr().err


# -- learning evaluation (ADR 0010) ----------------------------------------------------------------------------


def test_holdout_seeds_are_fresh_every_round():
    t = task(holdout_seeds=(10, 11, 12))
    assert [t.holdout_seeds_for(r) for r in range(3)] == [(10, 11, 12), (13, 14, 15), (16, 17, 18)]
    t.check_rounds(3)
    with pytest.raises(ValueError, match="round 2's held-out seeds"):
        task(train_seeds=(5, 6), holdout_seeds=(3, 4)).check_rounds(2)  # round 2 would use 5, 6


def test_task_seed_sets_must_be_disjoint():
    with pytest.raises(ValueError, match="sealed test"):
        task(test_seeds=(1, 9))
    with pytest.raises(ValueError, match="held-out and sealed test"):
        task(test_seeds=(4, 9))
    with pytest.raises(ValueError, match="two seeds"):
        task(test_seeds=(9,))
    with pytest.raises(ValueError, match="round 3's held-out seeds"):
        task(test_seeds=(7, 8)).check_rounds(3)  # held-out (3, 4), (5, 6), (7, 8)


def test_each_round_judges_on_its_own_holdout_seeds():
    evaluator = FakeEvaluator()
    Orchestrator(RuleBook(), RunLog("r"), Skills(evaluate=evaluator)).run(task(), rounds=3)
    holdouts = {seeds for _, _, seeds in evaluator.calls if seeds not in ((1, 2),)}
    assert (3, 4) in holdouts and (5, 6) in holdouts
    assert all(seeds in {(3, 4), (5, 6), (7, 8)} for seeds in holdouts), holdouts


def test_seed_scores_and_paired_improvement():
    seeds = tuple(range(6))
    worse = summary(both(20.0), both(0.95), seeds=seeds)
    better = summary(both(18.0), both(0.95), seeds=seeds)
    assert seed_scores(worse, [Mode.BASELINE]) == [20.0 + seed_noise(s) for s in seeds]
    assert paired_improvement_p(worse, better, [Mode.BASELINE]) == 0.0  # every seed 2 m better: certain
    assert paired_improvement_p(better, worse, [Mode.BASELINE]) == 1.0
    noisy = replace_per_seed(better, {Mode.BASELINE: [18.0, 25.0, 15.0, 22.0, 17.0, 21.0]})
    p = paired_improvement_p(worse, noisy, [Mode.BASELINE])
    assert 0.05 < p < 1.0  # mixed per-seed changes: not significant
    with pytest.raises(ValueError, match="same seeds"):
        paired_improvement_p(worse, summary(both(18.0), both(0.95), seeds=(7, 8)), [Mode.BASELINE])
    with pytest.raises(ValueError, match="per-seed"):
        seed_scores(ComparisonSummary("s", (0, 1), {}, 0.0, (0.0, 0.0), 0.0, 1.0, Verdict.IMPROVES), [Mode.BASELINE])


def test_seed_scores_average_to_the_score():
    """The paired test and the 2 % margin judge the same quantity (ADR 0010)."""
    seeds = tuple(range(5))  # seed_noise sums to zero over these, so the per-seed RMSEs average to the mean
    s = summary({Mode.BASELINE: 20.0, Mode.FUSION: 16.0}, {Mode.BASELINE: 0.90, Mode.FUSION: 0.80}, seeds=seeds)
    for modes in ([Mode.BASELINE], [Mode.BASELINE, Mode.FUSION]):
        assert sum(seed_scores(s, modes)) / len(seeds) == pytest.approx(score(s, modes))
    # Calibration comes from the mean containment: per-seed spread around the target is not penalised.
    spread = replace(s, per_seed={m: tuple(replace(x, containment_95=c) for x, c in zip(s.per_seed[m], [0.85, 1.0] * 3))
                                  for m in Mode})
    assert seed_scores(spread, [Mode.BASELINE]) == seed_scores(s, [Mode.BASELINE])


def replace_per_seed(s: ComparisonSummary, rmse: dict[Mode, list[float]]) -> ComparisonSummary:
    per_seed = dict(s.per_seed) | {
        mode: tuple(run_metrics(r, m.containment_95) for r, m in zip(values, s.per_seed[mode]))
        for mode, values in rmse.items()
    }
    return replace(s, per_seed=per_seed)


class NoisyHoldout(FakeEvaluator):
    """Held-out scores improve on average but not consistently across seeds."""

    def __call__(self, scenario, tracker, seeds):
        s = super().__call__(scenario, tracker, seeds)
        if tuple(seeds) == (1, 2):
            return s
        jitter = [(-1) ** k * 15.0 * tracker.process_noise_m2_s3 for k in range(len(seeds))]
        return replace_per_seed(s, {m: [x.position_rmse_m + j for x, j in zip(s.per_seed[m], jitter)] for m in Mode})


def test_candidate_needs_a_significant_paired_improvement():
    outcome = Orchestrator(RuleBook(), RunLog("r"), Skills(evaluate=NoisyHoldout())).run(task())
    tracker_lessons = [x for x in outcome.lessons if x.action is LessonAction.SET_TRACKER and x.holdout_score_after]
    assert tracker_lessons and not any(x.accepted for x in tracker_lessons)
    assert all("not significant" in x.reason for x in tracker_lessons)


def test_sealed_test_compares_start_and_final_rule_books_once():
    evaluator = FakeEvaluator()
    outcome = Orchestrator(RuleBook(), RunLog("r"), Skills(evaluate=evaluator)).run(
        task(test_seeds=(50, 51, 52)), rounds=3)
    (result,) = outcome.sealed_test
    assert isinstance(result, SealedTestResult) and result.changed
    assert result.score_after < result.score_before and result.p_value < 0.05
    assert "improves" in result.describe()
    assert any("sealed test" in line for line in outcome.report)
    # Used once, after learning: two evaluations (start and final settings), and nothing before them.
    test_calls = [i for i, (_, _, seeds) in enumerate(evaluator.calls) if seeds == (50, 51, 52)]
    assert len(test_calls) == 2 and test_calls[0] == len(evaluator.calls) - 2


def test_sealed_test_scores_each_rule_book_with_its_own_fusion_policy():
    outcome = Orchestrator(RuleBook(), RunLog("r"), Skills(evaluate=FakeEvaluator(Verdict.OVER_CONFIDENT))).run(
        task(test_seeds=(50, 51)))
    (result,) = outcome.sealed_test
    assert not outcome.rules.fusion_enabled("reference") and result.changed
    assert result.verdict_after == Verdict.OVER_CONFIDENT.value


def test_no_sealed_test_without_test_seeds():
    assert Orchestrator(RuleBook(), RunLog("r"), Skills(evaluate=FakeEvaluator())).run(task()).sealed_test == ()
