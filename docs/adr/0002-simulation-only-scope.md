---
title: "0002: Simulation-only scope"
status: accepted
updated: 2026-10-31
---

# 0002: Simulation-only scope

## Context

The project asks whether a bearing-only sensor improves position tracking. The question can be answered
with synthetic data, where the ground truth is known exactly.

## Decision

The project is an educational, simulation-only prototype. It connects to no real sensors, feeds, maps or
other systems, and its sensor parameters do not describe real hardware. The dashboard listens on localhost only.

## Consequences

Every metric can be scored against exact truth, and every run is reproducible from a seed. The results say
nothing about real hardware; the README's Limitations section lists what is not modelled. Adding real data
sources would need a new ADR that supersedes this one.
