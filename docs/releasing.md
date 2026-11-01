---
title: Release process
status: accepted
updated: 2026-11-01
---

# Release process

1. Make sure every user-visible change has an entry under **Unreleased** in the [CHANGELOG](../CHANGELOG.md).
   Use the subsections `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed` and `Security`.
2. Choose the new version using the [versioning policy](versioning.md), and start a branch from an up-to-date
   `main`:

   ```bash
   git checkout -b release/0.3.0
   ```

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
   .venv/Scripts/python ci/release.py 0.3.0
   ```

6. Reinstall so the installed metadata shows the new version, then check it:

   ```bash
   .venv/Scripts/python -m pip install -e ".[dev]"
   ```

   ```bash
   .venv/Scripts/python ci/release.py --check
   ```

   Add a one-line summary under the new version heading, as earlier releases have.

7. Commit, push the branch and open a pull request into `main`:

   ```bash
   git commit -am "Release 0.3.0"
   ```

   ```bash
   git push -u origin release/0.3.0
   ```

   ```bash
   gh pr create --base main --title "Release 0.3.0"
   ```

8. When every CI job is green, merge the pull request, then tag the merge commit on `main` and push the tag:

   ```bash
   git checkout main
   ```

   ```bash
   git pull --ff-only
   ```

   ```bash
   git tag -a v0.3.0 -m "Release 0.3.0"
   ```

   ```bash
   git push origin v0.3.0
   ```

9. Publish a GitHub release for the tag, with the version's CHANGELOG section as its notes (saved to a file
   first, here `notes.md`):

   ```bash
   gh release create v0.3.0 --verify-tag --title "0.3.0" --notes-file notes.md
   ```

10. Review the [roadmap](roadmap.md): tick what the release finished and move dates where needed.

`python ci/release.py --check` only verifies that the version sources agree, and changes nothing.
