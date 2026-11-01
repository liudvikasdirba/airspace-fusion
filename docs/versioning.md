---
title: Versioning policy
status: accepted
updated: 2026-10-31
---

# Versioning policy

Releases are numbered `MAJOR.MINOR.PATCH` following [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).
While the major version is 0, the project is a research prototype and a minor release may break compatibility.
Such breaks are listed under **Changed** or **Removed** in the [CHANGELOG](../CHANGELOG.md).

## What is versioned

The compatibility surface is:

- the command line: commands, options and the layout of their output
- the dashboard's controls
- `ScenarioConfig`, `SensorConfig` and `TrackerConfig` fields, and the built-in scenarios
- metric definitions and the verdict rules
- the rule book JSON format (`format_version`) and the run-log events

## Which number to raise

| Change | From 1.0.0 | Before 1.0.0 |
|---|---|---|
| Removes or renames something in the compatibility surface, or changes the rule book format | MAJOR | MINOR |
| Adds a command, option, scenario, metric or setting; changes published results on purpose (golden file regenerated) | MINOR | MINOR |
| Fixes a bug without changing intended results; documentation only | PATCH | PATCH |

A change that alters the 20-seed results in the README always gets a CHANGELOG entry saying which numbers moved.

## Where versions are recorded

| Version | Where | Changed by |
|---|---|---|
| Package version | `__version__` in `src/airspace_fusion/__init__.py`; `pyproject.toml` reads it | `ci/release.py` |
| Citation version and date | `CITATION.cff` | `ci/release.py` (checked by tests) |
| Release notes | `CHANGELOG.md` | you, under Unreleased; `ci/release.py` dates them |
| Git tag | `vX.Y.Z` on the release commit | you, after the release commit |
| Rule book format | `format_version` in the rule book JSON (`RULE_BOOK_FORMAT`) | code change, with a MINOR (later MAJOR) bump |
| Rule book contents | `version` in the rule book JSON; increases with every learning update that changes a rule | `learn` |
| Package that wrote a rule book or run | `package_version` in the rule book and in the run log's `task` event | `learn` |

`airspace-fusion --version` prints the installed package version.
