"""Check version consistency, or cut a release.

The package version lives in src/airspace_fusion/__init__.py (pyproject.toml reads it from there).
CITATION.cff and CHANGELOG.md must agree with it.

Usage: python ci/release.py --check
       python ci/release.py 0.2.0 [--date YYYY-MM-DD]

A release moves the CHANGELOG "Unreleased" entries into a new dated section and sets the new
version in __init__.py and CITATION.cff. It does not commit or tag; it prints the git commands.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INIT = Path("src/airspace_fusion/__init__.py")
CITATION = Path("CITATION.cff")
CHANGELOG = Path("CHANGELOG.md")

SEMVER = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)")
VERSION_LINE = re.compile(r'^__version__ = "([^"]*)"$', re.MULTILINE)
HEADING = re.compile(r"^## \[(?P<version>[^\]]+)\](?: - (?P<date>\d{4}-\d{2}-\d{2}))?$")
SUBSECTIONS = ("Added", "Changed", "Deprecated", "Removed", "Fixed", "Security")


class ReleaseError(Exception):
    pass


def parse_semver(text: str) -> tuple[int, int, int]:
    match = SEMVER.fullmatch(text)
    if not match:
        raise ReleaseError(f"{text!r} is not a MAJOR.MINOR.PATCH version")
    return int(match[1]), int(match[2]), int(match[3])


def read_version(root: Path) -> str:
    match = VERSION_LINE.search((root / INIT).read_text(encoding="utf-8"))
    if not match:
        raise ReleaseError(f"no __version__ line in {INIT}")
    return match[1]


def citation_field(text: str, key: str) -> str | None:
    match = re.search(rf"^{key}: ?\"?([^\"\n]*)\"?$", text, re.MULTILINE)
    return match[1] if match else None


@dataclass(frozen=True)
class Section:
    version: str
    date: str | None
    body: str


def parse_changelog(text: str) -> tuple[str, list[Section]]:
    """Split CHANGELOG.md into its preamble and ``## [...]`` sections, validating their order and format."""
    lines = text.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.startswith("## ")]
    if not starts:
        raise ReleaseError("CHANGELOG.md has no '## [...]' sections")
    sections = []
    for k, i in enumerate(starts):
        heading = lines[i].rstrip("\n")
        match = HEADING.fullmatch(heading)
        if not match:
            raise ReleaseError(f"bad changelog heading {heading!r}; use '## [X.Y.Z] - YYYY-MM-DD' or '## [Unreleased]'")
        end = starts[k + 1] if k + 1 < len(starts) else len(lines)
        sections.append(Section(match["version"], match["date"], "".join(lines[i + 1:end])))

    if sections[0].version != "Unreleased" or sections[0].date:
        raise ReleaseError("the first changelog section must be '## [Unreleased]' without a date")
    released = sections[1:]
    previous: tuple[tuple[int, int, int], str] | None = None
    for s in released:
        if s.date is None:
            raise ReleaseError(f"changelog section {s.version} has no release date")
        date.fromisoformat(s.date)
        current = (parse_semver(s.version), s.date)
        if previous and not (current[0] < previous[0] and current[1] <= previous[1]):
            raise ReleaseError(f"changelog section {s.version} is out of order (newest first)")
        previous = current
    for s in sections:
        for sub in re.findall(r"^### (.+)$", s.body, re.MULTILINE):
            if sub not in SUBSECTIONS:
                raise ReleaseError(f"changelog section {s.version}: unknown subsection {sub!r}")
    return "".join(lines[: starts[0]]), sections


def has_entries(body: str) -> bool:
    return any(line.startswith("- ") for line in body.splitlines())


def check(root: Path = ROOT) -> list[str]:
    """Return every inconsistency between the version sources (empty when all agree)."""
    problems = []
    try:
        version = read_version(root)
        parse_semver(version)
        _, sections = parse_changelog((root / CHANGELOG).read_text(encoding="utf-8"))
    except ReleaseError as exc:
        return [str(exc)]
    released = {s.version: s for s in sections[1:]}
    if version not in released:
        problems.append(f"CHANGELOG.md has no section for the current version {version}")
    elif sections[1].version != version:
        problems.append(f"the newest CHANGELOG.md release is {sections[1].version}, but the package is {version}")
    citation = (root / CITATION).read_text(encoding="utf-8")
    if citation_field(citation, "version") != version:
        problems.append(f"CITATION.cff version {citation_field(citation, 'version')!r} != package version {version!r}")
    if version in released and citation_field(citation, "date-released") != released[version].date:
        problems.append(f"CITATION.cff date-released != CHANGELOG.md date for {version}")
    return problems


def release(new_version: str, release_date: str, root: Path = ROOT) -> None:
    problems = check(root)
    if problems:
        raise ReleaseError("fix these before releasing:\n  " + "\n  ".join(problems))
    current = read_version(root)
    if parse_semver(new_version) <= parse_semver(current):
        raise ReleaseError(f"new version {new_version} must be greater than {current}")
    date.fromisoformat(release_date)

    changelog_path = root / CHANGELOG
    preamble, sections = parse_changelog(changelog_path.read_text(encoding="utf-8"))
    if not has_entries(sections[0].body):
        raise ReleaseError("the Unreleased changelog section has no entries")
    if release_date < sections[1].date:
        raise ReleaseError(f"release date {release_date} is before the previous release ({sections[1].date})")
    new_sections = [
        Section("Unreleased", None, "\n"),
        Section(new_version, release_date, sections[0].body),
        *sections[1:],
    ]
    changelog_path.write_text(preamble + "".join(
        f"## [{s.version}]{f' - {s.date}' if s.date else ''}\n{s.body}" for s in new_sections
    ), encoding="utf-8")

    init = root / INIT
    init.write_text(VERSION_LINE.sub(f'__version__ = "{new_version}"', init.read_text(encoding="utf-8")),
                    encoding="utf-8")
    citation = root / CITATION
    text = citation.read_text(encoding="utf-8")
    text = re.sub(r"^version: .*$", f"version: {new_version}", text, flags=re.MULTILINE)
    text = re.sub(r"^date-released: .*$", f"date-released: {release_date}", text, flags=re.MULTILINE)
    citation.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("version", nargs="?", help="new MAJOR.MINOR.PATCH version")
    parser.add_argument("--check", action="store_true", help="only check that the version sources agree")
    parser.add_argument("--date", default=date.today().isoformat(), help="release date (default: today)")
    args = parser.parse_args(argv)
    if args.check == bool(args.version):
        parser.error("give either --check or a new version")
    if args.check:
        problems = check()
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        if not problems:
            print(f"version {read_version(ROOT)}: __init__.py, CITATION.cff and CHANGELOG.md agree")
        return 1 if problems else 0
    try:
        release(args.version, args.date)
    except ReleaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"released {args.version} ({args.date}) in {INIT}, {CITATION} and {CHANGELOG}. Next:")
    print("  python -m pip install -e .   # refresh installed metadata")
    print(f'  git commit -am "Release {args.version}"')
    print(f'  git tag -a v{args.version} -m "Release {args.version}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
