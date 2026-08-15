"""Single command-line launch point for the four XLE research experiments."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

_FINAL_COMMON_DEFAULTS = {
    "training_start": "2021-01-01",
    "backtest_start": "2024-01-01",
    "end_date": "2026-08-01",
    "lookback_periods": 504,
    "rebalance_frequency": 21,
    "risk_free_rate": 0.04 / 252,
    "num_principal_components": 1,
    "max_download_attempts": 3,
    "retry_delay_seconds": 1.0,
}
_FACTOR_DEFAULTS = {"hac_lag": 20}
_COVARIANCE_DEFAULTS = {"ewma_half_life": 63, "hac_lag": 20}
_TAIL_RISK_DEFAULTS = {
    "cvar_percentile": 0.95,
    "tail_loss_weights": (1.0, 0.1, 0.01),
}
_PCA_DIMENSION_DEFAULTS = {
    "training_start": "2018-01-01",
    "backtest_start": "2021-01-01",
    "end_date": "2023-12-31",
    "lookback_periods": 504,
    "rebalance_frequency": 21,
    "risk_free_rate": 0.04 / 252,
    "components": "1,2,3,4,5",
    "max_download_attempts": 3,
    "retry_delay_seconds": 1.0,
}


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--training-start", default=_FINAL_COMMON_DEFAULTS["training_start"]
    )
    parser.add_argument(
        "--backtest-start", default=_FINAL_COMMON_DEFAULTS["backtest_start"]
    )
    parser.add_argument("--end-date", default=_FINAL_COMMON_DEFAULTS["end_date"])
    parser.add_argument(
        "--lookback-periods",
        type=int,
        default=_FINAL_COMMON_DEFAULTS["lookback_periods"],
    )
    parser.add_argument(
        "--rebalance-frequency",
        type=int,
        default=_FINAL_COMMON_DEFAULTS["rebalance_frequency"],
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=_FINAL_COMMON_DEFAULTS["risk_free_rate"],
    )
    parser.add_argument(
        "--num-principal-components",
        type=int,
        default=_FINAL_COMMON_DEFAULTS["num_principal_components"],
    )
    parser.add_argument(
        "--max-download-attempts",
        type=int,
        default=_FINAL_COMMON_DEFAULTS["max_download_attempts"],
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=_FINAL_COMMON_DEFAULTS["retry_delay_seconds"],
    )


def _add_output_argument(parser: argparse.ArgumentParser, default: str) -> None:
    parser.add_argument("--output-directory", type=Path, default=Path(default))


def _parse_tail_loss_weights(value: str) -> tuple[float, ...]:
    try:
        weights = tuple(float(weight.strip()) for weight in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "tail-loss-weights must be comma-separated numbers, for example 1,0.1,0.01"
        ) from error
    if not weights:
        raise argparse.ArgumentTypeError("tail-loss-weights must not be empty")
    return weights


def build_parser() -> argparse.ArgumentParser:
    """Build the stable CLI surface without importing experiment modules eagerly."""
    parser = argparse.ArgumentParser(
        prog="python -m port_opt.backtest.cli",
        description="Run XLE portfolio research experiments.",
    )
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    factor = subparsers.add_parser(
        "factor", help="Run the factor-construction experiment."
    )
    _add_common_arguments(factor)
    factor.add_argument("--hac-lag", type=int, default=_FACTOR_DEFAULTS["hac_lag"])
    _add_output_argument(factor, "research_outputs")

    covariance = subparsers.add_parser(
        "covariance", help="Run the covariance-construction experiment."
    )
    _add_common_arguments(covariance)
    covariance.add_argument(
        "--ewma-half-life", type=int, default=_COVARIANCE_DEFAULTS["ewma_half_life"]
    )
    covariance.add_argument(
        "--hac-lag", type=int, default=_COVARIANCE_DEFAULTS["hac_lag"]
    )
    _add_output_argument(covariance, "covariance_research_outputs")

    tail_risk = subparsers.add_parser(
        "tail-risk", help="Run the tail-risk objective experiment."
    )
    _add_common_arguments(tail_risk)
    tail_risk.add_argument(
        "--cvar-percentile",
        type=float,
        default=_TAIL_RISK_DEFAULTS["cvar_percentile"],
    )
    tail_risk.add_argument(
        "--tail-loss-weights",
        type=_parse_tail_loss_weights,
        default=_TAIL_RISK_DEFAULTS["tail_loss_weights"],
        help="Comma-separated tail-loss weights to compare.",
    )
    tail_risk.add_argument(
        "--tail-loss-weight",
        type=float,
        default=None,
        help="Deprecated single-value override for --tail-loss-weights.",
    )
    _add_output_argument(tail_risk, "tail_risk_research_outputs")

    dimension = subparsers.add_parser(
        "pca-dimension", help="Run the PCA-dimension development experiment."
    )
    dimension.add_argument(
        "--training-start", default=_PCA_DIMENSION_DEFAULTS["training_start"]
    )
    dimension.add_argument(
        "--backtest-start", default=_PCA_DIMENSION_DEFAULTS["backtest_start"]
    )
    dimension.add_argument("--end-date", default=_PCA_DIMENSION_DEFAULTS["end_date"])
    dimension.add_argument(
        "--lookback-periods",
        type=int,
        default=_PCA_DIMENSION_DEFAULTS["lookback_periods"],
    )
    dimension.add_argument(
        "--rebalance-frequency",
        type=int,
        default=_PCA_DIMENSION_DEFAULTS["rebalance_frequency"],
    )
    dimension.add_argument(
        "--risk-free-rate",
        type=float,
        default=_PCA_DIMENSION_DEFAULTS["risk_free_rate"],
    )
    dimension.add_argument(
        "--components", default=_PCA_DIMENSION_DEFAULTS["components"]
    )
    dimension.add_argument(
        "--max-download-attempts",
        type=int,
        default=_PCA_DIMENSION_DEFAULTS["max_download_attempts"],
    )
    dimension.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=_PCA_DIMENSION_DEFAULTS["retry_delay_seconds"],
    )
    _add_output_argument(dimension, "pca_dimension_research_outputs")

    all_experiments = subparsers.add_parser(
        "all", help="Run every experiment with its own documented defaults."
    )
    _add_output_argument(all_experiments, "research_outputs")
    return parser


def _parse_components(value: str) -> tuple[int, ...]:
    try:
        components = tuple(int(component.strip()) for component in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "components must be comma-separated integers, for example 1,2,3"
        ) from error
    if not components:
        raise argparse.ArgumentTypeError("components must not be empty")
    return components


def _print_result(result: object, paths: dict[str, Path]) -> None:
    print("Performance metrics:")
    print(result.performance_metrics)
    print("Implementation metrics:")
    print(result.implementation_metrics)
    print("Outputs:")
    print(*paths.values(), sep="\n")


def _run_factor(args: argparse.Namespace) -> None:
    from .xle_experiment import (
        run_xle_pca_historical_mean_experiment,
        save_xle_experiment_visuals,
    )

    result = run_xle_pca_historical_mean_experiment(
        training_start=args.training_start,
        backtest_start=args.backtest_start,
        end_date=args.end_date,
        lookback_periods=args.lookback_periods,
        rebalance_frequency=args.rebalance_frequency,
        risk_free_rate=args.risk_free_rate,
        num_principal_components=args.num_principal_components,
        max_download_attempts=args.max_download_attempts,
        retry_delay_seconds=args.retry_delay_seconds,
        hac_lag=args.hac_lag,
    )
    _print_result(result, save_xle_experiment_visuals(result, args.output_directory))


def _run_covariance(args: argparse.Namespace) -> None:
    from .covariance_experiment import (
        run_xle_covariance_experiment,
        save_xle_covariance_experiment_visuals,
    )

    result = run_xle_covariance_experiment(
        training_start=args.training_start,
        backtest_start=args.backtest_start,
        end_date=args.end_date,
        lookback_periods=args.lookback_periods,
        rebalance_frequency=args.rebalance_frequency,
        risk_free_rate=args.risk_free_rate,
        num_principal_components=args.num_principal_components,
        ewma_half_life=args.ewma_half_life,
        max_download_attempts=args.max_download_attempts,
        retry_delay_seconds=args.retry_delay_seconds,
        hac_lag=args.hac_lag,
    )
    _print_result(
        result, save_xle_covariance_experiment_visuals(result, args.output_directory)
    )


def _run_tail_risk(args: argparse.Namespace) -> None:
    from .tail_risk_experiment import (
        run_xle_tail_risk_experiment,
        save_xle_tail_risk_experiment_visuals,
    )

    result = run_xle_tail_risk_experiment(
        training_start=args.training_start,
        backtest_start=args.backtest_start,
        end_date=args.end_date,
        lookback_periods=args.lookback_periods,
        rebalance_frequency=args.rebalance_frequency,
        risk_free_rate=args.risk_free_rate,
        num_principal_components=args.num_principal_components,
        cvar_percentile=args.cvar_percentile,
        tail_loss_weights=args.tail_loss_weights,
        tail_loss_weight=args.tail_loss_weight,
        max_download_attempts=args.max_download_attempts,
        retry_delay_seconds=args.retry_delay_seconds,
    )
    _print_result(
        result, save_xle_tail_risk_experiment_visuals(result, args.output_directory)
    )


def _run_pca_dimension(args: argparse.Namespace) -> None:
    from .pca_dimension_experiment import (
        run_xle_pca_dimension_experiment,
        save_xle_pca_dimension_experiment_visuals,
    )

    result = run_xle_pca_dimension_experiment(
        training_start=args.training_start,
        backtest_start=args.backtest_start,
        end_date=args.end_date,
        lookback_periods=args.lookback_periods,
        rebalance_frequency=args.rebalance_frequency,
        risk_free_rate=args.risk_free_rate,
        num_principal_components=_parse_components(args.components),
        max_download_attempts=args.max_download_attempts,
        retry_delay_seconds=args.retry_delay_seconds,
    )
    _print_result(
        result,
        save_xle_pca_dimension_experiment_visuals(result, args.output_directory),
    )


def _run_all(args: argparse.Namespace) -> None:
    """Run documented defaults without mixing development and final periods."""
    from .covariance_experiment import (
        run_xle_covariance_experiment,
        save_xle_covariance_experiment_visuals,
    )
    from .pca_dimension_experiment import (
        run_xle_pca_dimension_experiment,
        save_xle_pca_dimension_experiment_visuals,
    )
    from .tail_risk_experiment import (
        run_xle_tail_risk_experiment,
        save_xle_tail_risk_experiment_visuals,
    )
    from .xle_experiment import (
        run_xle_pca_historical_mean_experiment,
        save_xle_experiment_visuals,
    )

    runners = (
        (
            "factor",
            run_xle_pca_historical_mean_experiment,
            save_xle_experiment_visuals,
            {**_FINAL_COMMON_DEFAULTS, **_FACTOR_DEFAULTS},
        ),
        (
            "covariance",
            run_xle_covariance_experiment,
            save_xle_covariance_experiment_visuals,
            {**_FINAL_COMMON_DEFAULTS, **_COVARIANCE_DEFAULTS},
        ),
        (
            "tail-risk",
            run_xle_tail_risk_experiment,
            save_xle_tail_risk_experiment_visuals,
            {**_FINAL_COMMON_DEFAULTS, **_TAIL_RISK_DEFAULTS},
        ),
        (
            "pca-dimension",
            run_xle_pca_dimension_experiment,
            save_xle_pca_dimension_experiment_visuals,
            {
                **{
                    key: value
                    for key, value in _PCA_DIMENSION_DEFAULTS.items()
                    if key != "components"
                },
                "num_principal_components": _parse_components(
                    _PCA_DIMENSION_DEFAULTS["components"]
                ),
            },
        ),
    )
    for label, run_experiment, save_visuals, kwargs in runners:
        result = run_experiment(**kwargs)
        print(f"\n=== {label} ===")
        _print_result(result, save_visuals(result, args.output_directory / label))


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    dispatch = {
        "factor": _run_factor,
        "covariance": _run_covariance,
        "tail-risk": _run_tail_risk,
        "pca-dimension": _run_pca_dimension,
        "all": _run_all,
    }
    dispatch[args.experiment](args)


if __name__ == "__main__":
    main()
