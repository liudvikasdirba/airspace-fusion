---
title: "0001: Record architecture decisions"
status: accepted
updated: 2026-10-31
---

# 0001: Record architecture decisions

## Context

Several choices in this project look arbitrary without their history. For example, why bearings cannot keep a
track alive, or why the learning agents are not language models. Commit messages record these reasons, but they
are hard to find.

## Decision

Significant decisions are recorded as numbered ADRs in `docs/adr/`, using `template.md`. An accepted ADR is not
rewritten. A changed decision gets a new ADR, and the old one is marked `superseded` with `superseded-by`.

## Consequences

Each decision has one place to look. `tests/test_docs.py` enforces the numbering, metadata, index entries and
supersession links. Writing an ADR adds a small cost to each significant change.
