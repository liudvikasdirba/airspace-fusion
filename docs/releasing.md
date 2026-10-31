---
title: Release process
status: accepted
updated: 2026-10-31
---

# Release process

1. Make sure every user-visible change has an entry under **Unreleased** in the [CHANGELOG](../CHANGELOG.md).
   Use the subsections `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed` and `Security`.
2. Choose the new version using the [versioning policy](versioning.md).
3. Run the full test suites:

   ```bash
   .venv/Scripts/python -m pytest
   ```

   ```bash
   .venv/Scripts/python -m pytest -m slow
   ```

4. If results changed on purpose, regenerate the golden file, review its diff, and refresh the README results.
5. Cut the release. This moves the Unreleased entries into a dated section and sets the version in
   `__init__.py` and `CITATION.cff`:

   ```bash
   .venv/Scripts/python ci/release.py 0.2.0
   ```

6. Reinstall so the installed metadata shows the new version, then check it:

   ```bash
   .venv/Scripts/python -m pip install -e ".[dev]"
   ```

   ```bash
   .venv/Scripts/python ci/release.py --check
   ```

7. Commit and tag:

   ```bash
   git commit -am "Release 0.2.0"
   ```

   ```bash
   git tag -a v0.2.0 -m "Release 0.2.0"
   ```

8. Push the commit and the tag when a remote exists. CI must pass on the tagged commit.

`python ci/release.py --check` only verifies that the version sources agree, and changes nothing.
