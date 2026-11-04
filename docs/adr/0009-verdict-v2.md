---
title: "0009: Verdict v2: containment rules and multiple-comparison correction"
status: accepted
updated: 2026-11-04
---

# 0009: Verdict v2: containment rules and multiple-comparison correction

## Context

The first verdict compared per-seed RMSE only (paired t-test, p < 0.05, at least 5 % change). Roadmap finding 4
listed three problems:

- It ignores the filter's honesty. In `biased_bearing`, fusion's 95 % containment is 0.28: the filter claims an
  accuracy it does not have. That is the most dangerous failure in the project, and the verdict only caught it
  because the bias also happened to raise RMSE. A smaller bias could improve RMSE while leaving the filter
  over-confident.
- `compare` over several scenarios and `sweep` over several points test many hypotheses at once, each at
  p < 0.05, so some "significant" results are expected by chance.
- 20 seeds give wide intervals, so small effects are hard to tell from noise.

GOSPA (added on 2 November) combines error with missing tracks. Coverage is the same in both modes by
construction ([ADR 0004](0004-position-updates-keep-tracks-alive.md): only position updates keep a track alive),
so GOSPA and RMSE rank the modes the same way.

## Decision

**Verdicts.** `Verdict` gains **"fusion over-confident"**. A comparison's verdict is decided in this order
(`verdict_v2` and `rmse_verdict` in `evaluation.py`):

1. Fewer than two seeds with an RMSE in both modes: **"insufficient data"** (unchanged).
2. Fusion's mean 95 % containment below **0.85** and below the baseline's: **"fusion over-confident"**, whatever
   RMSE does. (Amended 4 November: the rule first ignored the baseline, so a fusion filter better calibrated
   than a badly calibrated baseline was still called harmful and the learning loop switched it off. The verdict
   compares the two modes, so low containment that fusion shares with the baseline is not held against it.)
3. RMSE verdict as before (p < 0.05 and at least 5 % change), with the Holm-adjusted p-value when the comparison
   is one of several.
4. If that RMSE verdict is "fusion improves" but fusion's containment is more than **0.03** below the
   baseline's: **"fusion over-confident"**. The accuracy gain is bought with uncertainty the filter understates.

When either mode has no containment value the RMSE verdict stands. "Fusion worsens" and "fusion over-confident"
are both **harmful** (`Verdict.harmful`); the learning loop treats either as a reason to stop using fusion.

**Multiple comparisons.** Every comparison of one `compare` or `sweep` run is one family. Its RMSE p-values are
Holm-adjusted (`holm_adjust`, `with_holm`), and the verdict uses the adjusted value. A single comparison is not
adjusted. The learning loop's comparisons stay unadjusted: each held-out check is a separate decision, and
learning evaluation is reworked separately (see the roadmap, M3).

**Seeds.** `compare` and `sweep` default to **200** seeds instead of 20.

**GOSPA** stays a reported metric and does not enter the verdict: while coverage cannot differ between modes it
would repeat the RMSE verdict. Revisit this if a later change lets fusion affect track continuity.

**Results files** move to format 2. Each entry stores its raw and adjusted p-value, and the record stores
`multiple_comparisons`: the raw p-values of the whole run and the positions of its own entries among them. A
`compare` with a rule book splits a run over several records, so `reproduce` needs the whole family to redo the
correction. Format 1 files are rejected, because their verdicts follow the old rules. Format 3 (amended
4 November) adds the family to `config_hash`, since it decides the adjusted p-values and verdicts, and
`load_record` checks that the family places every entry exactly once.

## Consequences

- On the four built-in scenarios with 200 seeds, `biased_bearing` is "fusion over-confident" (containment 0.28);
  `reference` and `noisy_primary` still improve and `distant_bearing` still shows no material change.
- A bearing-bias sweep (50 seeds) shows the case the old verdict missed: at 0.5° fusion RMSE changes by only
  +2 % ("no material change" before) while containment falls to 0.85, so the verdict is now "fusion
  over-confident". It stays so up to 2°; at 3° the gate rejects most biased bearings, containment recovers to
  0.89 and the verdict is "fusion worsens".
- A verdict now says why it was given (`explain_verdict`), in the CLI and the dashboard.
- The 0.85 and 0.03 thresholds are choices, not derived values. 0.85 is far enough below 0.95 that sampling
  noise with 200 seeds cannot reach it for a consistent filter; 0.03 tolerates the baseline's own shortfall from
  the turn (about 0.93). Both are constants in `evaluation.py`.
- `compare` prints its scenarios after all of them have run, since the correction needs every p-value.
- Old results files cannot be reproduced with this version; re-run them.
