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
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from airspace_fusion import __version__
from airspace_fusion.config import ScenarioConfig, TrackerConfig
from airspace_fusion.evaluation import ComparisonSummary, Mode, Verdict, compare_modes
from airspace_fusion.learning import (
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
    propose_candidates,
    score,
    scored_modes,
)


@dataclass(frozen=True)
class LearningTask:
    scenarios: tuple[ScenarioConfig, ...]
    train_seeds: tuple[int, ...]
    holdout_seeds: tuple[int, ...]
    max_candidates: int = 4

    def __post_init__(self) -> None:
        if not self.scenarios:
            raise ValueError("a learning task needs at least one scenario")
        if len(self.train_seeds) < 2 or len(self.holdout_seeds) < 2:
            raise ValueError("training and held-out sets each need at least two seeds for a paired verdict")
        if set(self.train_seeds) & set(self.holdout_seeds):
            raise ValueError("training and held-out seeds must be disjoint")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be at least 1")


Evaluate = Callable[[ScenarioConfig, TrackerConfig, Sequence[int]], ComparisonSummary]


@dataclass(frozen=True)
class Skills:
    """Reusable capabilities the agents call. ``evaluate`` is the only expensive one."""

    evaluate: Evaluate = compare_modes


@dataclass
class ScenarioWork:
    """Everything the agents learn about one scenario during one round."""

    scenario: ScenarioConfig
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


@dataclass(frozen=True)
class StepContext:
    task: LearningTask
    skills: Skills
    log: RunLog


def describe_tracker(t: TrackerConfig) -> str:
    return f"q={t.process_noise_m2_s3:.4g} gate={t.gate_probability:.4g} coast={t.max_coast_s:.4g}s"


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
        incumbent_score = score(work.train, work.modes)
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
            work.holdout = ctx.skills.evaluate(work.scenario, work.incumbent, ctx.task.holdout_seeds)
        return work.holdout

    def fusion_policy(self, work: ScenarioWork, ctx: StepContext) -> None:
        if not any(g.kind is GapKind.FUSION_HARMFUL for g in work.gaps):
            return
        holdout = self._holdout_incumbent(work, ctx)
        confirmed = holdout.verdict is Verdict.WORSENS
        work.lessons.append(Lesson(
            run_id=ctx.log.run_id,
            scenario=work.name,
            action=LessonAction.DISABLE_FUSION,
            triggers=(GapKind.FUSION_HARMFUL,),
            accepted=confirmed,
            reason=f"held-out verdict: {holdout.verdict.value} ({holdout.rmse_relative_change:+.1%} RMSE)",
            holdout_score_before=finite_or_none(score(holdout, scored_modes(True))),
            holdout_score_after=finite_or_none(score(holdout, scored_modes(False))),
        ))
        if confirmed:
            # Tune for the baseline only from here on: chasing a harmful sensor's containment is pointless.
            work.fusion_enabled = False
            work.gaps = find_gaps(work.train, work.modes)

    def holdout(self, work: ScenarioWork, ctx: StepContext) -> None:
        if work.best is None:
            return
        _, candidate = work.best
        before = self._holdout_incumbent(work, ctx)
        after = ctx.skills.evaluate(work.scenario, candidate, ctx.task.holdout_seeds)
        s_before, s_after = score(before, work.modes), score(after, work.modes)
        problems = []
        if not s_after < s_before * (1.0 - MIN_SCORE_IMPROVEMENT):
            problems.append(f"held-out score {s_after:.4g} not {MIN_SCORE_IMPROVEMENT:.0%} better than {s_before:.4g}")
        for mode in work.modes:
            lost = before.metrics[mode]["coverage"].mean - after.metrics[mode]["coverage"].mean
            if lost > MAX_COVERAGE_LOSS:
                problems.append(f"{mode.value} coverage drops by {lost:.3f}")
        reason = "; ".join(problems) or f"held-out score {s_before:.4g} -> {s_after:.4g}"
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
class LearningOutcome:
    rules: RuleBook
    rounds_run: int
    lessons: tuple[Lesson, ...]
    report: tuple[str, ...]
    failures: tuple[Gap, ...]


class Orchestrator:
    """Plans each scenario's steps, routes them to agents, retries failures and tracks everything in the log."""

    def __init__(
        self, rules: RuleBook, log: RunLog, skills: Skills | None = None, max_retries: int = 1,
        rules_path: Path | None = None,
    ) -> None:
        self.rules = rules
        self.log = log
        self.skills = skills or Skills()
        self.max_retries = max_retries
        self.rules_path = rules_path
        self.agents = {a.name: a for a in (ResearchAgent(), ExperimentAgent(), QAAgent(), ContentAgent())}

    def run(self, task: LearningTask, rounds: int = 1) -> LearningOutcome:
        ctx = StepContext(task, self.skills, self.log)
        all_lessons: list[Lesson] = []
        report: list[str] = []
        failures: list[Gap] = []
        rounds_run = 0
        self.log.record("task", package_version=__version__, scenarios=[s.name for s in task.scenarios],
                        train_seeds=task.train_seeds, holdout_seeds=task.holdout_seeds, rounds=rounds,
                        rules_version=self.rules.version)
        for round_index in range(rounds):
            rounds_run += 1
            report.append(f"-- round {round_index + 1} (rule book v{self.rules.version})")
            works = [self._execute(scenario, ctx) for scenario in task.scenarios]
            lessons = [lesson for w in works for lesson in w.lessons]
            failures += [Gap(GapKind.STEP_FAILED, w.name, None, 1.0, 0.0) for w in works if w.failed_step]
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
        return LearningOutcome(self.rules, rounds_run, tuple(all_lessons), tuple(report), tuple(failures))

    def _execute(self, scenario: ScenarioConfig, ctx: StepContext) -> ScenarioWork:
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
            except Exception as exc:  # recorded and retried; a scenario that keeps failing is skipped, not fatal
                self.log.record("step_error", scenario=work.name, step=f"{agent_name}.{step}", attempt=attempt,
                                error=repr(exc))
                continue
            self.log.record("step_done", scenario=work.name, step=f"{agent_name}.{step}", attempt=attempt)
            return True
        return False
