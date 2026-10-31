---
title: "0005: Deterministic learning agents with held-out acceptance"
status: accepted
updated: 2026-10-31
---

# 0005: Deterministic learning agents with held-out acceptance

## Context

The learning loop (`agents.py`, `learning.py`) follows a closed-loop "agent learning system" design: an
orchestrator, research, experiment, QA and content agents, logs, evaluation, lessons and a learning update.
Such systems usually use language-model agents. This project is offline and reproducible by design
(see [ADR 0002](0002-simulation-only-scope.md)). Tuning on the same seeds that are used to judge the result
would also overfit.

## Decision

The agents are plain, deterministic Python. Candidate settings are scored on training seeds (1000 onwards by
default) and accepted only if they also win, by at least 2 %, on separate held-out seeds (2000 onwards). The
README results use seeds 0–19, which the loop never sees. Accepted changes are re-checked against fixed bounds
before they enter the rule book, and every tried change is kept as a lesson.

## Consequences

The same seeds and rule book always give the same result, with no network access or API keys. Gains are
small but carry over to unseen seeds. The loop can only change tracker settings: it can advise against fusion
when a sensor is biased, but it cannot correct the bias.
