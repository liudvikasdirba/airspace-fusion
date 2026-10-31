import pytest

from airspace_fusion.cli import main


@pytest.mark.parametrize("seeds", ["0", "-3"])
def test_compare_rejects_non_positive_seed_count(seeds, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["compare", "--seeds", seeds])
    assert exc.value.code == 2
    assert "must be at least 1" in capsys.readouterr().err


def test_run_prints_metrics(capsys):
    main(["run", "--seed", "0"])
    out = capsys.readouterr().out
    assert "position_rmse_m" in out and "prediction_only_scans" in out
