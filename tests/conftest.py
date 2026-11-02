import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Rewrite tests/golden/*.json from the current code instead of comparing against it.",
    )


@pytest.fixture(autouse=True)
def results_in_tmp(tmp_path, monkeypatch):
    """compare and sweep save results files by default; keep the ones tests make out of the repository."""
    monkeypatch.setenv("AIRSPACE_FUSION_RESULTS_DIR", str(tmp_path / "results"))
    return tmp_path / "results"
