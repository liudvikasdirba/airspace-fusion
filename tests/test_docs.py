"""Document management and versioning: doc metadata, index, ADRs, links, changelog and version sources."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
from datetime import date
from pathlib import Path

import pytest

import airspace_fusion
from airspace_fusion.cli import main
from airspace_fusion.learning import RULE_BOOK_FORMAT, RuleBook, RuleBookError

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ADR = DOCS / "adr"
STATUSES = {"draft", "accepted", "superseded", "deprecated"}

spec = importlib.util.spec_from_file_location("release", ROOT / "ci" / "release.py")
release = importlib.util.module_from_spec(spec)
sys.modules["release"] = release  # dataclasses look the module up while it executes
spec.loader.exec_module(release)


def front_matter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, f"{path.relative_to(ROOT)} has no front matter"
    fields = {}
    for line in match[1].splitlines():
        key, sep, value = line.partition(":")
        assert sep, f"{path.relative_to(ROOT)}: bad front-matter line {line!r}"
        fields[key.strip()] = value.strip().strip('"')
    return fields


def managed_docs() -> list[Path]:
    return sorted(p for p in DOCS.rglob("*.md") if p != ADR / "template.md")


def index_rows() -> dict[str, str]:
    """Linked path (relative to docs/) -> status, from the index table."""
    text = (DOCS / "README.md").read_text(encoding="utf-8")
    return dict(re.findall(r"^\| \[[^\]]+\]\(([^)]+)\) \| (\w+) \|", text, re.MULTILINE))


# -- document metadata and index -----------------------------------------------------------------


@pytest.mark.parametrize("path", managed_docs(), ids=lambda p: str(p.relative_to(DOCS)))
def test_document_metadata(path):
    meta = front_matter(path)
    assert {"title", "status", "updated"} <= set(meta)
    assert meta["status"] in STATUSES
    assert date.fromisoformat(meta["updated"]) <= date.today()
    headings = re.findall(r"^# (.+)$", path.read_text(encoding="utf-8"), re.MULTILINE)
    assert headings == [meta["title"]], "the title must be the document's only '#' heading"
    if meta["status"] == "superseded":
        assert "superseded-by" in meta


def test_index_lists_every_document_with_its_status():
    rows = index_rows()
    docs = {
        p.relative_to(DOCS).as_posix(): front_matter(p)["status"] for p in managed_docs() if p != DOCS / "README.md"
    }
    assert set(rows) == set(docs), "docs/README.md must list exactly the documents in docs/"
    assert rows == docs


def test_adr_numbering_and_supersession():
    adrs = sorted(p for p in ADR.glob("*.md") if p.name != "template.md")
    numbers = []
    for path in adrs:
        match = re.fullmatch(r"(\d{4})-[a-z0-9-]+\.md", path.name)
        assert match, f"ADR file name {path.name} must be NNNN-slug.md"
        numbers.append(int(match[1]))
        meta = front_matter(path)
        assert meta["title"].startswith(f"{match[1]}: ")
        if "superseded-by" in meta:
            assert meta["status"] == "superseded"
            assert any(p.name.startswith(meta["superseded-by"] + "-") for p in adrs)
    assert numbers == list(range(1, len(numbers) + 1)), "ADR numbers must be contiguous from 0001"


def link_targets(path: Path) -> list[str]:
    text = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.DOTALL)
    return re.findall(r"\]\(([^)\s]+)\)", text)


@pytest.mark.parametrize(
    "path", [ROOT / "README.md", ROOT / "CHANGELOG.md", *managed_docs(), ADR / "template.md"],
    ids=lambda p: str(p.relative_to(ROOT)),
)
def test_relative_links_resolve(path):
    for target in link_targets(path):
        if re.match(r"[a-z]+:", target) or target.startswith("#"):
            continue
        resolved = (path.parent / target.split("#")[0]).resolve()
        assert resolved.exists(), f"{path.relative_to(ROOT)} links to missing {target}"


# -- versions --------------------------------------------------------------------------------------


def test_version_sources_agree():
    assert release.check() == []
    assert release.read_version(ROOT) == airspace_fusion.__version__


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"airspace-fusion {airspace_fusion.__version__}"


def test_rule_book_records_versions_and_rejects_unknown_format(tmp_path):
    book = RuleBook()
    assert book.format_version == RULE_BOOK_FORMAT
    assert book.package_version == airspace_fusion.__version__
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(book.model_dump(mode="json") | {"format_version": RULE_BOOK_FORMAT + 1}))
    with pytest.raises(RuleBookError, match="format_version"):
        RuleBook.load(path)


# -- release script --------------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path):
    for rel in (release.INIT, release.CITATION, release.CHANGELOG):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    return tmp_path


def changelog(repo: Path, unreleased: str) -> None:
    path = repo / release.CHANGELOG
    text = path.read_text(encoding="utf-8")
    start = text.index("## [Unreleased]")
    end = text.index("## [", start + 1)
    path.write_text(text[:start] + "## [Unreleased]\n" + unreleased + text[end:], encoding="utf-8")


def test_release_moves_unreleased_entries_and_bumps_every_source(repo):
    changelog(repo, "\n### Added\n\n- A thing.\n\n")
    current = release.read_version(repo)
    release.release("9.0.0", "2099-01-02", root=repo)

    assert release.read_version(repo) == "9.0.0"
    citation = (repo / release.CITATION).read_text(encoding="utf-8")
    assert "version: 9.0.0" in citation and "date-released: 2099-01-02" in citation
    _, sections = release.parse_changelog((repo / release.CHANGELOG).read_text(encoding="utf-8"))
    assert [s.version for s in sections[:3]] == ["Unreleased", "9.0.0", current]
    assert not release.has_entries(sections[0].body)
    assert "- A thing." in sections[1].body and sections[1].date == "2099-01-02"
    assert release.check(repo) == []


@pytest.mark.parametrize(
    "version, release_date, unreleased, message",
    [
        ("0.0.9", "2099-01-02", "\n- x\n\n", "must be greater"),
        ("v1.0", "2099-01-02", "\n- x\n\n", "MAJOR.MINOR.PATCH"),
        ("9.0.0", "2099-01-02", "\n\n", "no entries"),
        ("9.0.0", "2000-01-01", "\n- x\n\n", "before the previous release"),
    ],
)
def test_release_refuses_bad_input(repo, version, release_date, unreleased, message):
    changelog(repo, unreleased)
    with pytest.raises(release.ReleaseError, match=message):
        release.release(version, release_date, root=repo)
    assert release.check(repo) == []  # nothing was half-written


def test_check_reports_disagreeing_sources(repo):
    citation = repo / release.CITATION
    citation.write_text(citation.read_text(encoding="utf-8").replace("\nversion: ", "\nversion: 7"), encoding="utf-8")
    assert any("CITATION.cff version" in p for p in release.check(repo))


@pytest.mark.parametrize(
    "bad, message",
    [
        ("## [0.1.0]\n", "no release date"),
        ("## 0.1.0 - 2026-10-31\n", "bad changelog heading"),
        ("## [Unreleased]\n### Notes\n- x\n", "unknown subsection"),
    ],
)
def test_changelog_format_errors(bad, message):
    text = "# Changelog\n\n## [Unreleased]\n\n" + bad if not bad.startswith("## [Unreleased]") else bad
    with pytest.raises(release.ReleaseError, match=message):
        release.parse_changelog(text)


# (A compact date such as 20261031 in a heading is already rejected as a bad heading.)
@pytest.mark.parametrize("bad_date", ["2026-13-31", "2026-02-30"])
def test_invalid_dates_are_reported_not_raised(repo, bad_date):
    path = repo / release.CHANGELOG
    text = path.read_text(encoding="utf-8")
    current = release.read_version(repo)
    heading = re.search(rf"^## \[{re.escape(current)}\] - .*$", text, re.MULTILINE)[0]
    path.write_text(text.replace(heading, f"## [{current}] - {bad_date}"), encoding="utf-8")
    problems = release.check(repo)
    assert len(problems) == 1 and "not a valid YYYY-MM-DD date" in problems[0]


@pytest.mark.parametrize("bad_date", ["2099-13-01", "20991231"])
def test_release_rejects_invalid_release_date(repo, bad_date):
    changelog(repo, "\n- x\n\n")
    with pytest.raises(release.ReleaseError, match="not a valid YYYY-MM-DD date"):
        release.release("9.0.0", bad_date, root=repo)
    assert release.check(repo) == []


def test_changelog_releases_must_be_newest_first():
    text = "## [Unreleased]\n\n## [0.1.0] - 2026-01-01\n\n## [0.2.0] - 2026-02-01\n"
    with pytest.raises(release.ReleaseError, match="out of order"):
        release.parse_changelog(text)
