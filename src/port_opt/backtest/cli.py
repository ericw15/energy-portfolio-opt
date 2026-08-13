"""Single command-line launch point for the four XLE research experiments."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--training-start", default="2021-01-01")
    parser.add_argument("--backtest-start", default="2024-01-01")
    parser.add_argument("--end-date", default="2026-08-01")
    parser.add_argument("--lookback-periods", type=int, default=504)
    parser.add_argument("--rebalance-frequency", type=int, default=21)
    parser.add_argument("--risk-free-rate", type=float, default=0.04 / 252)
    parser.add_argument("--num-principal-components", type=int, default=1)
    parser.add_argument("--max-download-attempts", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=1.0)


def _add_output_argument(parser: argparse.ArgumentParser, default: str) -> None:
    parser.add_argument("--output-directory", type=Path, default=Path(default))


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
    factor.add_argument("--hac-lag", type=int, default=20)
    _add_output_argument(factor, "research_outputs")

    covariance = subparsers.add_parser(
        "covariance", help="Run the covariance-construction experiment."
    )
    _add_common_arguments(covariance)
    covariance.add_argument("--ewma-half-life", type=int, default=63)
    covariance.add_argument("--hac-lag", type=int, default=20)
    _add_output_argument(covariance, "covariance_research_outputs")

    tail_risk = subparsers.add_parser(
        "tail-risk", help="Run the tail-risk objective experiment."
    )
    _add_common_arguments(tail_risk)
    tail_risk.add_argument("--cvar-percentile", type=float, default=0.95)
    tail_risk.add_argument("--tail-loss-weight", type=float, default=1.0)
    _add_output_argument(tail_risk, "tail_risk_research_outputs")

    dimension = subparsers.add_parser(
        "pca-dimension", help="Run the PCA-dimension development experiment."
    )
    dimension.add_argument("--training-start", default="2018-01-01")
    dimension.add_argument("--backtest-start", default="2021-01-01")
    dimension.add_argument("--end-date", default="2023-12-31")
    dimension.add_argument("--lookback-periods", type=int, default=504)
    dimension.add_argument("--rebalance-frequency", type=int, default=21)
    dimension.add_argument("--risk-free-rate", type=float, default=0.04 / 252)
    dimension.add_argument("--components", default="1,2,3,4,5")
    dimension.add_argument("--max-download-attempts", type=int, default=3)
    dimension.add_argument("--retry-delay-seconds", type=float, default=1.0)
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
            {"num_principal_components": 1},
        ),
        (
            "covariance",
            run_xle_covariance_experiment,
            save_xle_covariance_experiment_visuals,
            {"num_principal_components": 1},
        ),
        (
            "tail-risk",
            run_xle_tail_risk_experiment,
            save_xle_tail_risk_experiment_visuals,
            {"num_principal_components": 1},
        ),
        (
            "pca-dimension",
            run_xle_pca_dimension_experiment,
            save_xle_pca_dimension_experiment_visuals,
            {},
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
