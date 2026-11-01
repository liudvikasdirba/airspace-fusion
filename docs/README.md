---
title: Documentation index
status: accepted
updated: 2026-11-01
---

# Documentation index

This folder holds the project's long-lived documents. The top-level [README](../README.md) is the user
guide (install, dashboard, command line, results). The [CHANGELOG](../CHANGELOG.md) records what changed in
each version.

## How documents are managed

- **Metadata.** Every document starts with a front-matter block giving its `title`, `status` and the date
  it was last `updated` (YYYY-MM-DD). The title is repeated as the document's only `#` heading.
- **Status.** `draft` (proposed, not yet relied on), `accepted` (current), `superseded` (replaced; names its
  replacement in `superseded-by`) or `deprecated` (no longer applies, kept for history).
- **Decisions** are recorded as numbered architecture decision records (ADRs) in [adr/](adr/), copied from
  [adr/template.md](adr/template.md). An accepted ADR is never rewritten: a changed decision gets a new ADR,
  and the old one becomes `superseded`.
- **This index** lists every document with its status. Add a row when you add a document, and update the
  row when its status changes.
- **Checks.** `tests/test_docs.py` (part of the fast suite) fails when metadata is missing or invalid, this
  index is out of date, ADR numbering has a gap, a relative link is broken, or the version sources disagree.
- **Changes** to documents that affect users get a CHANGELOG entry under Unreleased.

## Documents

| Document | Status | Summary |
|---|---|---|
| [Versioning policy](versioning.md) | accepted | What each version number means and where versions are recorded |
| [Release process](releasing.md) | accepted | Steps to cut, check and tag a release |
| [Roadmap to April 2027](roadmap.md) | draft | November 2026 audit and the milestones to 1.0.0 |
| [ADR 0001: Record architecture decisions](adr/0001-record-architecture-decisions.md) | accepted | Why and how decisions are recorded |
| [ADR 0002: Simulation-only scope](adr/0002-simulation-only-scope.md) | accepted | No real sensors, feeds or hardware |
| [ADR 0003: Keep ground truth away from the tracker](adr/0003-ground-truth-isolation.md) | accepted | Truth and tracker output meet only in evaluation |
| [ADR 0004: Only position updates keep a track alive](adr/0004-position-updates-keep-tracks-alive.md) | accepted | Bearings refine a track but cannot keep it alive |
| [ADR 0005: Deterministic learning agents with held-out acceptance](adr/0005-deterministic-learning-agents.md) | accepted | The learning loop uses no language model |
| [ADR 0006: Single source for the package version](adr/0006-single-version-source.md) | accepted | `__version__` is the source; other files are checked against it |
| [ADR 0007: NumPy tracker by default, Stone Soup as the cross-check](adr/0007-numpy-tracker-stone-soup-cross-check.md) | accepted | Filter maths in NumPy; Stone Soup must agree within 1e-9 |
