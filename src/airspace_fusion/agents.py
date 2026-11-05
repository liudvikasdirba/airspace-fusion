"""Execution side of the learning loop: task, orchestrator, agents and skills.

    task -> rule book -> orchestrator (plan, route, retry, track) -> agents -> skills -> report
                 ^                                                                   |
                 +---------- learning update <- lessons <- evaluation <- run log <---+

Each scenario in a task runs the same plan of agent steps:

1. ``research.diagnose``   score the current rule-book settings on training seeds, find gaps
2. ``qa.fusion_policy``    if fusion looked harmful, confirm on held-out seeds and disable it
3. ``research.propose``    gap-directed candidate settings, skipping ones already tried
4. ``experiment.screen``   score each candidate on training seeds, keep the best
5. ``qa.holdout``          accept the best only if it also wins on held-out seeds
6. ``content.report``      plain-text summary

After every round, the lessons update the rule book and the next round starts from it.
The agents are deterministic Python: there is no language model, and the same task, seeds and
rule book always give the same result.

Seeds (ADR 0010): training seeds are reused every round; held-out seeds are fresh in every round, so no
round's decision is judged on seeds an earlier decision saw; sealed test seeds are used once, after the last
round, to report how the final rule book compares with the one the run started from. They never steer a
decision.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field, replace
from pathlib import Path

from airspace_fusion import __version__
from airspace_fusion.config import TrackerConfig
from airspace_fusion.evaluation import ComparisonSummary, Mode, compare_modes
from airspace_fusion.learning import (
    ACCEPTANCE_SIGNIFICANCE,
    MAX_COVERAGE_LOSS,
    MIN_SCORE_IMPROVEMENT,
    Gap,
    GapKind,
    Lesson,
    LessonAction,
    RuleBook,
    RunLog,
    find_gaps,
    finite_or_none,
    paired_improvement_p,
    propose_candidates,
    score,
    scored_modes,
)
from airspace_fusion.scenarios import ScenarioSource


@dataclass(frozen=True)
class LearningTask:
    """What to learn on. ``holdout_seeds`` are round 1's held-out seeds; later rounds shift them (see below)."""

    scenarios: tuple[ScenarioSource, ...]
    train_seeds: tuple[int, ...]
    holdout_seeds: tuple[int, ...]
    max_candidates: int = 4
    test_seeds: tuple[int, ...] = ()  # sealed: used once, after the last round; empty skips the test

    def __post_init__(self) -> None:
        if not self.scenarios:
            raise ValueError("a learning task needs at least one scenario")
        if len(self.train_seeds) < 2 or len(self.holdout_seeds) < 2:
            raise ValueError("training and held-out sets each need at least two seeds for a paired verdict")
        if len(self.test_seeds) == 1:
            raise ValueError("the sealed test set needs at least two seeds for a paired test (or none)")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be at least 1")
        self.check_rounds(1)

    def holdout_seeds_for(self, round_index: int) -> tuple[int, ...]:
        """Held-out seeds of round ``round_index`` (0-based): round 1's seeds shifted by a whole block per round."""
        stride = max(self.holdout_seeds) - min(self.holdout_seeds) + 1
        return tuple(seed + round_index * stride for seed in self.holdout_seeds)

    def check_rounds(self, rounds: int) -> None:
        """Raise ValueError unless training, every round's held-out seeds and the test seeds are disjoint."""
        train, test = set(self.train_seeds), set(self.test_seeds)
        if train & test:
            raise ValueError("training and sealed test seeds must be disjoint")
        for round_index in range(rounds):
            holdout = set(self.holdout_seeds_for(round_index))
            where = f" (round {round_index + 1}'s held-out seeds)" if round_index else ""
            if holdout & train:
                raise ValueError(f"training and held-out seeds must be disjoint{where}")
            if holdout & test:
                raise ValueError(f"held-out and sealed test seeds must be disjoint{where}")


Evaluate = Callable[[ScenarioSource, TrackerConfig, Sequence[int]], ComparisonSummary]


@dataclass(frozen=True)
class Skills:
    """Reusable capabilities the agents call. ``evaluate`` is the only expensive one.

    ``evaluate`` must be deterministic in (scenario, tracker, seeds): the orchestrator caches its results.
    """

    evaluate: Evaluate = compare_modes


class CachedEvaluate:
    """Memoises evaluations for one orchestrator, so settings scored in an earlier step or round are reused."""

    def __init__(self, evaluate: Evaluate) -> None:
        self._evaluate = evaluate
        self._results: dict[tuple[ScenarioSource, TrackerConfig, tuple[int, ...]], ComparisonSummary] = {}

    def __call__(self, scenario: ScenarioSource, tracker: TrackerConfig, seeds: Sequence[int]) -> ComparisonSummary:
        key = (scenario, tracker, tuple(seeds))
        if key not in self._results:
            self._results[key] = self._evaluate(scenario, tracker, seeds)
        return self._results[key]


@dataclass
class ScenarioWork:
    """Everything the agents learn about one scenario during one round."""

    scenario: ScenarioSource
    incumbent: TrackerConfig
    fusion_enabled: bool
    tried: list[TrackerConfig]
    train: ComparisonSummary | None = None
    holdout: ComparisonSummary | None = None
    gaps: list[Gap] = field(default_factory=list)
    candidates: list[TrackerConfig] = field(default_factory=list)
    best: tuple[float, TrackerConfig] | None = None  # (training score, settings)
    lessons: list[Lesson] = field(default_factory=list)
    failed_step: str | None = None
    report: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.fusion_enabled_at_start = self.fusion_enabled

    @property
    def name(self) -> str:
        return self.scenario.name

    @property
    def modes(self) -> tuple[Mode, ...]:
        return scored_modes(self.fusion_enabled)

    @property
    def train_summary(self) -> ComparisonSummary:
        """The training evaluation; the research agent's diagnosis must have run first."""
        if self.train is None:
            raise RuntimeError(f"{self.name}: no training evaluation yet")
        return self.train


@dataclass(frozen=True)
class StepContext:
    task: LearningTask
    skills: Skills
    log: RunLog
    holdout_seeds: tuple[int, ...]  # this round's


def describe_tracker(t: TrackerConfig) -> str:
    common = f"gate={t.gate_probability:.4g} coast={t.max_coast_s:.4g}s"
    if t.motion_model == "imm":
        return (f"imm q_cv={t.imm_straight_process_noise_m2_s3:.4g} q_ct={t.imm_turn_process_noise_m2_s3:.4g} "
                + common)
    return f"q={t.process_noise_m2_s3:.4g} " + common


class ResearchAgent:
    name = "research"

    def diagnose(self, work: ScenarioWork, ctx: StepContext) -> None:
        work.train = ctx.skills.evaluate(work.scenario, work.incumbent, ctx.task.train_seeds)
        work.gaps = find_gaps(work.train, work.modes)
        ctx.log.record("evaluation", scenario=work.name, seeds="train", score=score(work.train, work.modes),
                       verdict=work.train.verdict, gaps=work.gaps)

    def propose(self, work: ScenarioWork, ctx: StepContext) -> None:
        work.candidates = propose_candidates(work.incumbent, work.gaps, work.tried, ctx.task.max_candidates)
        ctx.log.record("candidates", scenario=work.name, candidates=work.candidates)


class ExperimentAgent:
    name = "experiment"

    def screen(self, work: ScenarioWork, ctx: StepContext) -> None:
        # Results are committed only after every candidate ran, so a retried step does not duplicate lessons.
        incumbent_score = score(work.train_summary, work.modes)
        losers: list[Lesson] = []
        best: tuple[float, TrackerConfig] | None = None
        for candidate in work.candidates:
            summary = ctx.skills.evaluate(work.scenario, candidate, ctx.task.train_seeds)
            s = score(summary, work.modes)
            ctx.log.record("screened", scenario=work.name, tracker=candidate, score=s)
            if s >= incumbent_score:
                losers.append(_tracker_lesson(ctx, work, candidate, False, "no better on training seeds"))
            elif best is None or s < best[0]:
                best = (s, candidate)
        work.lessons.extend(losers)
        work.best = best


class QAAgent:
    name = "qa"

    def _holdout_incumbent(self, work: ScenarioWork, ctx: StepContext) -> ComparisonSummary:
        if work.holdout is None:
            work.holdout = ctx.skills.evaluate(work.scenario, work.incumbent, ctx.holdout_seeds)
        return work.holdout

    def fusion_policy(self, work: ScenarioWork, ctx: StepContext) -> None:
        if not any(g.kind is GapKind.FUSION_HARMFUL for g in work.gaps):
            return
        holdout = self._holdout_incumbent(work, ctx)
        confirmed = holdout.verdict.harmful
        lesson = Lesson(
            run_id=ctx.log.run_id,
            scenario=work.name,
            action=LessonAction.DISABLE_FUSION,
            triggers=(GapKind.FUSION_HARMFUL,),
            accepted=confirmed,
            reason=f"held-out verdict: {holdout.verdict.value} ({holdout.rmse_relative_change:+.1%} RMSE)",
            holdout_score_before=finite_or_none(score(holdout, scored_modes(True))),
            holdout_score_after=finite_or_none(score(holdout, scored_modes(False))),
        )
        # Tune for the baseline only from here on: chasing a harmful sensor's containment is pointless.
        # Everything is computed before anything is committed, so a retried step cannot record the lesson twice.
        gaps = find_gaps(work.train_summary, scored_modes(False)) if confirmed else work.gaps
        work.lessons.append(lesson)
        if confirmed:
            work.fusion_enabled = False
            work.gaps = gaps

    def holdout(self, work: ScenarioWork, ctx: StepContext) -> None:
        if work.best is None:
            return
        _, candidate = work.best
        before = self._holdout_incumbent(work, ctx)
        after = ctx.skills.evaluate(work.scenario, candidate, ctx.holdout_seeds)
        s_before, s_after = score(before, work.modes), score(after, work.modes)
        p_value = paired_improvement_p(before, after, work.modes)
        problems = []
        if not s_after < s_before * (1.0 - MIN_SCORE_IMPROVEMENT):
            problems.append(f"held-out score {s_after:.4g} not {MIN_SCORE_IMPROVEMENT:.0%} better than {s_before:.4g}")
        if not p_value < ACCEPTANCE_SIGNIFICANCE:  # also rejects NaN (too few seeds with a track)
            problems.append(f"held-out improvement not significant (paired p = {p_value:.3g})")
        for mode in work.modes:
            lost = before.metrics[mode]["coverage"].mean - after.metrics[mode]["coverage"].mean
            if lost > MAX_COVERAGE_LOSS:
                problems.append(f"{mode.value} coverage drops by {lost:.3f}")
        reason = "; ".join(problems) or f"held-out score {s_before:.4g} -> {s_after:.4g} (paired p = {p_value:.3g})"
        work.lessons.append(_tracker_lesson(ctx, work, candidate, not problems, reason, s_before, s_after))


class ContentAgent:
    name = "content"

    def report(self, work: ScenarioWork, ctx: StepContext) -> None:
        lines = [f"== {work.name}: {describe_tracker(work.incumbent)}, fusion "
                 f"{'enabled' if work.fusion_enabled_at_start else 'disabled'}"]
        if work.train is not None:
            gaps = find_gaps(work.train, scored_modes(work.fusion_enabled_at_start))
            lines.append(f"  training score {score(work.train, work.modes):.4g}; "
                         f"gaps: {', '.join(g.describe() for g in gaps) or 'none'}")
        lines.append(f"  candidates tried: {len(work.candidates)}")
        for lesson in work.lessons:
            what = describe_tracker(lesson.tracker) if lesson.tracker else lesson.action.value
            lines.append(f"  {'ACCEPTED' if lesson.accepted else 'rejected'} {what}: {lesson.reason}")
        work.report.extend(lines)


def _tracker_lesson(
    ctx: StepContext, work: ScenarioWork, candidate: TrackerConfig, accepted: bool, reason: str,
    before: float = float("nan"), after: float = float("nan"),
) -> Lesson:
    return Lesson(
        run_id=ctx.log.run_id,
        scenario=work.name,
        action=LessonAction.SET_TRACKER,
        triggers=tuple(dict.fromkeys(g.kind for g in work.gaps)),
        tracker=candidate,
        accepted=accepted,
        reason=reason,
        holdout_score_before=finite_or_none(before),
        holdout_score_after=finite_or_none(after),
    )


PLAN: tuple[tuple[str, str], ...] = (
    ("research", "diagnose"),
    ("qa", "fusion_policy"),
    ("research", "propose"),
    ("experiment", "screen"),
    ("qa", "holdout"),
    ("content", "report"),
)


@dataclass(frozen=True)
class SealedTestResult:
    """One scenario on the sealed test seeds: the rule book the run started from against the final one.

    Each rule book is scored as it would be used, so with its own fusion policy. ``p_value`` is the one-sided
    paired test that the final rule book scores lower; NaN when the two rule books agree for this scenario.
    """

    scenario: str
    changed: bool
    score_before: float
    score_after: float
    p_value: float
    verdict_after: str  # verdict v2 of the final settings on the test seeds

    def describe(self) -> str:
        if not self.changed:
            return f"{self.scenario}: unchanged; score {self.score_after:.4g}, verdict {self.verdict_after}"
        outcome = "improves" if self.p_value < ACCEPTANCE_SIGNIFICANCE else "no significant change"
        return (f"{self.scenario}: score {self.score_before:.4g} -> {self.score_after:.4g}, {outcome} "
                f"(paired p = {self.p_value:.3g}); verdict {self.verdict_after}")


@dataclass(frozen=True)
class LearningOutcome:
    rules: RuleBook
    rounds_run: int
    lessons: tuple[Lesson, ...]
    report: tuple[str, ...]
    failures: tuple[Gap, ...]
    sealed_test: tuple[SealedTestResult, ...] = ()


class Orchestrator:
    """Plans each scenario's steps, routes them to agents, retries failures and tracks everything in the log."""

    def __init__(
        self, rules: RuleBook, log: RunLog, skills: Skills | None = None, max_retries: int = 1,
        rules_path: Path | None = None,
    ) -> None:
        self.rules = rules
        self.log = log
        base = skills or Skills()
        self.skills = replace(base, evaluate=CachedEvaluate(base.evaluate))
        self.max_retries = max_retries
        self.rules_path = rules_path
        self.agents = {a.name: a for a in (ResearchAgent(), ExperimentAgent(), QAAgent(), ContentAgent())}

    def run(self, task: LearningTask, rounds: int = 1) -> LearningOutcome:
        task.check_rounds(rounds)
        start_rules = self.rules
        all_lessons: list[Lesson] = []
        report: list[str] = []
        failed: dict[str, Gap] = {}  # one entry per scenario, however many rounds it failed in
        rounds_run = 0
        self.log.record("task", package_version=__version__, scenarios=[s.name for s in task.scenarios],
                        train_seeds=task.train_seeds, holdout_seeds=task.holdout_seeds, rounds=rounds,
                        test_seeds=task.test_seeds, rules_version=self.rules.version)
        for round_index in range(rounds):
            rounds_run += 1
            holdout = task.holdout_seeds_for(round_index)
            ctx = StepContext(task, self.skills, self.log, holdout)
            report.append(f"-- round {round_index + 1} (rule book v{self.rules.version}; "
                          f"held-out seeds {holdout[0]}..{holdout[-1]})")
            works = [self._execute(scenario, ctx) for scenario in task.scenarios]
            lessons = [lesson for w in works for lesson in w.lessons]
            for w in works:
                if w.failed_step:
                    failed.setdefault(w.name, Gap(GapKind.STEP_FAILED, w.name, None, 1.0, 0.0))
            for w in works:
                report.extend(w.report)
            self.rules = self.rules.with_lessons(lessons)
            all_lessons += lessons
            accepted = [lesson for lesson in lessons if lesson.accepted]
            self.log.record("learning_update", round=round_index + 1, rules_version=self.rules.version,
                            accepted=len(accepted), rejected=len(lessons) - len(accepted))
            if self.rules_path is not None:
                self.rules.save(self.rules_path)
            if not accepted:
                report.append("-- nothing new accepted; stopping")
                break
        sealed = self._sealed_test(task, start_rules, failed) if task.test_seeds else []
        if sealed:
            report.append(f"-- sealed test, seeds {task.test_seeds[0]}..{task.test_seeds[-1]} (used once, after "
                          f"learning): rule book v{start_rules.version} -> v{self.rules.version}")
            report.extend(f"   {result.describe()}" for result in sealed)
        return LearningOutcome(self.rules, rounds_run, tuple(all_lessons), tuple(report), tuple(failed.values()),
                               tuple(sealed))

    def _sealed_test(self, task: LearningTask, start: RuleBook, failed: dict[str, Gap]) -> list[SealedTestResult]:
        """Compare the starting and final rule books once on the sealed test seeds; nothing is learned from it."""
        results = []
        for scenario in task.scenarios:
            if scenario.name in failed:
                continue
            name = scenario.name
            before_tracker, after_tracker = start.tracker_for(name), self.rules.tracker_for(name)
            before_modes = scored_modes(start.fusion_enabled(name))
            after_modes = scored_modes(self.rules.fusion_enabled(name))
            after = self.skills.evaluate(scenario, after_tracker, task.test_seeds)
            changed = before_tracker != after_tracker or before_modes != after_modes
            before = self.skills.evaluate(scenario, before_tracker, task.test_seeds) if changed else after
            # Each rule book is scored as used: mean per-seed score over its own modes, paired by seed.
            p_value = paired_improvement_p(before, after, before_modes, after_modes) if changed else float("nan")
            result = SealedTestResult(name, changed, score(before, before_modes), score(after, after_modes),
                                      p_value, after.verdict.value)
            self.log.record("sealed_test", scenario=name, changed=changed, score_before=result.score_before,
                            score_after=result.score_after, p_value=p_value, verdict=after.verdict)
            results.append(result)
        return results

    def _execute(self, scenario: ScenarioSource, ctx: StepContext) -> ScenarioWork:
        work = ScenarioWork(
            scenario=scenario,
            incumbent=self.rules.tracker_for(scenario.name),
            fusion_enabled=self.rules.fusion_enabled(scenario.name),
            tried=self.rules.tried_trackers(scenario.name),
        )
        for agent_name, step in PLAN:
            if not self._attempt(agent_name, step, work, ctx):
                work.failed_step = f"{agent_name}.{step}"
                work.report.append(f"== {work.name}: FAILED at {work.failed_step}; see the run log")
                break
        return work

    def _attempt(self, agent_name: str, step: str, work: ScenarioWork, ctx: StepContext) -> bool:
        action = getattr(self.agents[agent_name], step)
        for attempt in range(1 + self.max_retries):
            try:
                action(work, ctx)
            except BrokenProcessPool:
                raise  # the worker pool is gone: every later step would fail too, so stop the whole run
            except Exception as exc:  # recorded and retried; a scenario that keeps failing is skipped, not fatal
                self.log.record("step_error", scenario=work.name, step=f"{agent_name}.{step}", attempt=attempt,
                                error=repr(exc))
                continue
            self.log.record("step_done", scenario=work.name, step=f"{agent_name}.{step}", attempt=attempt)
            return True
        return False
