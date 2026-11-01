---
title: "0006: Single source for the package version"
status: accepted
updated: 2026-10-31
---

# 0006: Single source for the package version

## Context

The version appeared in both `pyproject.toml` and `CITATION.cff`, with nothing to keep them in step. There was
no changelog, and no way for a saved rule book to say which code or which format wrote it.

## Decision

`__version__` in `src/airspace_fusion/__init__.py` is the only place the package version is written by hand.
`pyproject.toml` reads it as a dynamic version. `ci/release.py` cuts releases: it updates `__init__.py` and
`CITATION.cff` and dates the CHANGELOG's Unreleased section. Rule books carry a `format_version` and the
`package_version` that wrote them. The policy is in [versioning.md](../versioning.md).

## Consequences

`tests/test_docs.py` fails when `__init__.py`, `CITATION.cff` and `CHANGELOG.md` disagree. After a release, an
editable install must be reinstalled before `importlib.metadata` reports the new version. `airspace-fusion
--version` reads `__version__` directly, so it is always current.
