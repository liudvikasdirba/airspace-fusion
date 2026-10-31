"""Learning loop and agent orchestration: scoring, gaps, lessons, rule book, orchestrator."""

from __future__ import annotations

import json
import math

import pytest
from pydantic import ValidationError

from airspace_fusion.agents import PLAN, LearningTask, Orchestrator, Skills
from airspace_fusion.cli import main
from airspace_fusion.config import SCENARIOS, TrackerConfig
from airspace_fusion.evaluation import METRIC_NAMES, ComparisonSummary, MetricSummary, Mode, Verdict
from airspace_fusion.learning import (
    TUNABLE_BOUNDS,
    Gap,
    GapKind,
    Lesson,
    LessonAction,
    RuleBook,
    RunLog,
    find_gaps,
    propose_candidates,
    same_settings,
    score,
)

REFERENCE = SCENARIOS["reference"]


def summary(
    rmse: dict[Mode, float], containment: dict[Mode, float], coverage: float = 0.99, initiations: float = 1.0,
    verdict: Verdict = Verdict.IMPROVES, name: str = "reference",
) -> ComparisonSummary:
    def metrics(mode: Mode) -> dict[str, MetricSummary]:
        values = dict.fromkeys(METRIC_NAMES, 0.0) | {
            "position_rmse_m": rmse[mode], "containment_95": containment[mode], "coverage": coverage,
            "track_initiations": initiations,
        }
        return {k: MetricSummary(v, 0.0, 6) for k, v in values.items()}

    delta = rmse[Mode.FUSION] - rmse[Mode.BASELINE]
    return ComparisonSummary(name, (0, 1), {m: metrics(m) for m in Mode}, delta, (delta, delta),
                             delta / rmse[Mode.BASELINE], 0.01, verdict)


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
                       name=scenario.name)


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


def test_harmful_fusion_is_disabled_after_holdout_confirmation():
    outcome = Orchestrator(RuleBook(), RunLog("r"), Skills(evaluate=FakeEvaluator(Verdict.WORSENS))).run(task())
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

    main(["compare", "--scenario", "distant_bearing", "--seeds", "2", "--rules", str(rules)])
    assert "learned rule book v" in capsys.readouterr().out


def test_cli_learn_rejects_overlapping_seeds(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["learn", "--train-first-seed", "0", "--holdout-first-seed", "3", "--rules", str(tmp_path / "r.json")])
    assert exc.value.code == 2
    assert "disjoint" in capsys.readouterr().err
