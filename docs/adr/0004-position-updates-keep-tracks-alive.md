---
title: "0004: Only position updates keep a track alive"
status: accepted
updated: 2026-10-31
---

# 0004: Only position updates keep a track alive

## Context

A bias sweep found that bearing-only updates kept a diverged track alive indefinitely: one seed with a 3° bias
drifted to 375 m RMSE. A bearing carries no range information, so it cannot confirm that the track is still
on the object.

## Decision

A track is dropped after `max_coast_s` without an accepted **position** update. Bearings refine a track but
cannot keep it alive, and they never start one.

## Consequences

Fusion cannot make coverage longer than the baseline's. Diverged tracks are dropped and restarted, which
shows up as extra track initiations. Golden results for seeds 0–4 were unchanged.
