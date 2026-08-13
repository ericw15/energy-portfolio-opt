from port_opt.backtest import cli


def test_cli_has_all_four_experiments_and_default_component_grid():
    parser = cli.build_parser()

    factor = parser.parse_args(["factor"])
    dimension = parser.parse_args(["pca-dimension"])
    all_experiments = parser.parse_args(["all"])

    assert factor.num_principal_components == 1
    assert factor.hac_lag == 20
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
