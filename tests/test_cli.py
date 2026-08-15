from argparse import Namespace

from port_opt.backtest import cli


def test_cli_has_all_four_experiments_and_default_component_grid():
    parser = cli.build_parser()

    factor = parser.parse_args(["factor"])
    tail_risk = parser.parse_args(["tail-risk"])
    dimension = parser.parse_args(["pca-dimension"])
    all_experiments = parser.parse_args(["all"])

    assert factor.num_principal_components == 1
    assert factor.hac_lag == 20
    assert tail_risk.tail_loss_weights == (1.0, 0.1, 0.01)
    assert tail_risk.tail_loss_weight is None
    assert dimension.components == "1,2,3,4,5"
    assert dimension.end_date == "2023-12-31"
    assert all_experiments.output_directory.name == "research_outputs"


def test_cli_parses_component_grid_and_rejects_invalid_values():
    assert cli._parse_components("1, 3,5") == (1, 3, 5)
    try:
        cli._parse_components("one,two")
    except Exception as error:
        assert "components" in str(error)
    else:
        raise AssertionError("invalid components should be rejected")


def test_cli_parses_tail_loss_weight_grid_and_single_value_override():
    parser = cli.build_parser()

    assert parser.parse_args(
        ["tail-risk", "--tail-loss-weights", "1, 0.2,0.01"]
    ).tail_loss_weights == (1.0, 0.2, 0.01)
    assert (
        parser.parse_args(["tail-risk", "--tail-loss-weight", "0.2"]).tail_loss_weight
        == 0.2
    )


def test_all_forwards_the_same_tail_risk_defaults_as_the_tail_risk_command(
    monkeypatch, tmp_path
):
    from port_opt.backtest import (
        covariance_experiment,
        pca_dimension_experiment,
        tail_risk_experiment,
        xle_experiment,
    )

    captured_kwargs = {}

    def capture_run(label):
        def run(**kwargs):
            captured_kwargs[label] = kwargs
            return object()

        return run

    def save_visuals(*_args):
        return {}

    monkeypatch.setattr(
        xle_experiment,
        "run_xle_pca_historical_mean_experiment",
        capture_run("factor"),
    )
    monkeypatch.setattr(xle_experiment, "save_xle_experiment_visuals", save_visuals)
    monkeypatch.setattr(
        covariance_experiment,
        "run_xle_covariance_experiment",
        capture_run("covariance"),
    )
    monkeypatch.setattr(
        covariance_experiment, "save_xle_covariance_experiment_visuals", save_visuals
    )
    monkeypatch.setattr(
        tail_risk_experiment, "run_xle_tail_risk_experiment", capture_run("tail-risk")
    )
    monkeypatch.setattr(
        tail_risk_experiment, "save_xle_tail_risk_experiment_visuals", save_visuals
    )
    monkeypatch.setattr(
        pca_dimension_experiment,
        "run_xle_pca_dimension_experiment",
        capture_run("pca-dimension"),
    )
    monkeypatch.setattr(
        pca_dimension_experiment,
        "save_xle_pca_dimension_experiment_visuals",
        save_visuals,
    )
    monkeypatch.setattr(cli, "_print_result", lambda *_args: None)

    cli._run_all(Namespace(output_directory=tmp_path))

    tail_risk = cli.build_parser().parse_args(["tail-risk"])
    assert captured_kwargs["tail-risk"]["cvar_percentile"] == tail_risk.cvar_percentile
    assert (
        captured_kwargs["tail-risk"]["tail_loss_weights"] == tail_risk.tail_loss_weights
    )
    assert captured_kwargs["pca-dimension"]["num_principal_components"] == (
        1,
        2,
        3,
        4,
        5,
    )
    assert "components" not in captured_kwargs["pca-dimension"]
