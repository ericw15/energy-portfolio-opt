# Energy Portfolio Research

> A small, transparent codebase for testing how portfolio-construction choices
> behave in a fixed set of energy equities.

This is a starting point for comparing a few ideas, it is not a claim to have found an investable edge.

## Start here

This project uses Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python -m port_opt.backtest.cli factor
```

The command downloads market data, runs the factor-construction experiment, and
writes charts and CSV summaries to `research_outputs/`.

To run the full set of experiments:

```bash
uv run python -m port_opt.backtest.cli all --output-directory research_outputs
```

Available experiments are `factor`, `covariance`, `tail-risk`, and
`pca-dimension`. Use `--help` to see the options for any one of them.

## What it explores

The experiments hold most choices fixed and vary one component at a time:

- factor inputs for covariance estimation;
- sample, PCA, and recency-weighted covariance estimates; and
- a simple tail-loss adjustment to a Sharpe-like objective.

Each compares optimized portfolios with equal-weight constituents and XLE.
The code is useful for learning, extending a walk-forward backtest, or making a
careful side-by-side comparison of portfolio rules.

## Read results with care

These are intentionally narrow historical experiments: a static energy-equity
universe, historical mean returns, long-only fully invested portfolios, and
gross returns. They do not model trading costs, taxes, execution, or
point-in-time index membership, and they are not investment advice.

The methods, results, and limitations belong in the
[research report](RESEARCH_REPORT.md). It is the source of record for research
claims; this README deliberately does not repeat them.

## Where to look next

```text
src/port_opt/backtest/cli.py    command-line entry point
src/port_opt/strategy/          portfolio and covariance implementations
research_outputs/               generated charts and tables
DEVELOPMENT_NOTES.md            code structure and research conventions
```

For contributors and future experiments, see
[development notes](DEVELOPMENT_NOTES.md).
