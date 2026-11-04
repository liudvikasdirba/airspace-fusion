---
title: "0010: Learning evaluation: fresh held-out seeds, paired acceptance and a sealed test set"
status: accepted
updated: 2026-11-04
---

# 0010: Learning evaluation: fresh held-out seeds, paired acceptance and a sealed test set

## Context

[ADR 0005](0005-deterministic-learning-agents.md) accepts a candidate setting when it beats the current one by at
least 2 % on held-out seeds. Roadmap finding 5 showed two weaknesses. Every round and every candidate reused the
same 6 held-out seeds, so repeated decisions could fit those seeds: the more candidates were tried, the more
likely one won there by chance. And 6 seeds cannot resolve a 2 % difference: seed-to-seed spread in RMSE is about
10 %. The loop also never reported its final result on seeds that no decision had touched.

## Decision

This ADR replaces ADR 0005's seed and acceptance rules; the rest of ADR 0005 (deterministic agents, lessons,
bounds) stands.

- **Training seeds** (1000 onwards, 50 by default) are reused every round, as before.
- **Held-out seeds are fresh in every round.** Round *r* uses round 1's held-out block shifted by *r − 1* block
  widths (`LearningTask.holdout_seeds_for`): 2000–2049, then 2050–2099, and so on. A decision is never judged on
  seeds an earlier decision saw. Within a round the incumbent and every candidate share the round's seeds, so
  comparisons stay paired.
- **Acceptance needs a paired test** as well as the 2 % margin: the candidate's per-seed held-out scores must be
  lower than the incumbent's in a one-sided paired t-test with p < 0.05 (`paired_improvement_p`). The per-seed
  score is the same formula as the score, applied to one seed. Comparison summaries now keep every seed's metrics
  (`ComparisonSummary.per_seed`) for this.
- **A sealed test set** (5000–5199, 200 seeds by default) is used once, after the last round, to compare the rule
  book the run started from with the final one, scenario by scenario. Each rule book is scored as it would be
  used, with its own fusion policy. The result is reported and logged; nothing is learned from it.
- **At least 50** training and held-out seeds by default. `learn` warns below 50 but does not refuse, so quick
  experiments stay possible.
- Training, every round's held-out seeds and the test seeds must be disjoint; `LearningTask.check_rounds`
  enforces it before a run starts. Disabling fusion still needs a harmful held-out verdict
  ([ADR 0009](0009-verdict-v2.md)); learning-loop comparisons are single decisions and are not Holm-corrected.

## Consequences

- A `learn --rounds 3` run from an empty rule book takes about a minute with 4 workers instead of about 10 s.
- In that run, round 1 accepted one setting per scenario, each with paired p < 0.001, and disabled fusion for
  `biased_bearing`. Round 2 rejected every candidate: none was 2 % better on its fresh held-out seeds, and none
  passed the paired test (p from 0.09 to 0.98). On the sealed test every accepted change improved the score,
  with p < 1e-11.
- The sealed seeds stay clean only if they are not used to choose settings by hand either. Change them
  (`--test-first-seed`) after they have informed a decision.
- The rule book format is unchanged; old rule books still load.
