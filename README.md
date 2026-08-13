# Energy portfolio research

This repository is becoming a modular research environment for comparing energy
portfolios with stated baselines. Historical data acquisition, return/covariance
estimation, portfolio construction, and walk-forward evaluation are intentionally
separate so that a research result can swap one component without changing the rest.

## Current backtesting contract

`port_opt.backtest.run_walk_forward_backtest` accepts a clean daily return panel
and a callable that maps *only* in-sample returns to a labelled weight series. It
rebalances by available return observations (usually trading days), holds weights
out of sample, and returns daily portfolio returns, daily weights, rebalance audit
records, and turnover. This is the backbone for comparing estimators and objectives.

```python
from port_opt.backtest import run_walk_forward_backtest

result = run_walk_forward_backtest(
    returns,
    lambda train: train.mean().clip(lower=0).pipe(lambda x: x / x.sum()),
    lookback_periods=252,
    rebalance_frequency=21,
)
print(result.sharpe_ratio(risk_free_rate=0.04 / 252))
```

## Energy research experiment

The original focal experiment is preserved and upgraded in
[`xle_experiment.py`](src/port_opt/backtest/xle_experiment.py). It compares
the PCA-factor, commodity-factor-only, PCA-plus-U.S.-commodity-factor, and
sample-covariance Markowitz portfolios, all using full in-sample historical
mean returns, for the XLE constituent universe against equal weight and the XLE
ETF. It can write
`return_comparison_historical_means.png`:

```bash
uv run python -m port_opt.backtest.xle_experiment
```

It retains the original static `XLE_TICKERS` list. That makes the experiment easy
to reproduce, but it should not be interpreted as a historical constituent
simulation until the universe is made point-in-time. The default configuration
uses a 504-observation rolling estimation window (roughly two trading years), a
21-observation rebalance cadence, and a 2021 training start so evaluation can
begin in 2024. These are explicit parameters, not calendar-day assumptions.
Failed or empty market-data downloads retry three times by default; callers can
configure the attempt count and delay.

`PCA_Historical_Mean_Strategy` is the strategy used by this experiment. It
combines PCA-factor covariance with each asset's full in-sample daily arithmetic
mean return and then constructs a long-only, fully invested maximum-Sharpe
portfolio. Both covariance and expected returns use the same 504-observation
trailing in-sample panel at each rebalance; there is no fitted expected-return
model. The comparison also includes `Markowitz_Portfolio`, which uses the same
historical means and optimizer but the sample covariance matrix. It therefore
isolates the covariance estimator under an otherwise identical protocol.
`PCA_Commodity_Factor_Strategy` adds WTI crude oil, Henry Hub natural gas, and
RBOB gasoline daily returns to the existing PCA factors; commodity factors are
observable risk inputs, not portfolio holdings.
`Commodity_Factor_Strategy` uses those same factors without PCA, making it the
direct comparator for whether the PCA factors contribute information beyond
commodity-price exposure.


## Covariance-construction experiment

[`covariance_experiment.py`](src/port_opt/backtest/covariance_experiment.py)
is a separate main experiment for the covariance question. It compares sample
covariance Markowitz, ordinary PCA covariance, Ledoit--Wolf covariance, EWMA
covariance and EWMA-PCA covariance, alongside the
equal-weight constituent and XLE baselines. Expected returns, long-only
maximum-Sharpe construction, estimation windows, and rebalance dates are held
identical across every optimized strategy.

EWMA-PCA fits the PCA factor covariance model using exponentially recency-
weighted observations throughout. The default EWMA half-life is 63 available
trading observations and is configurable.

This experiment produces a growth comparison, a six-panel risk/return
comparison, a compact turnover/concentration comparison, and the supporting
performance and implementation metric CSVs:

```bash
uv run python -m port_opt.backtest.covariance_experiment
```

## Tail-risk objective experiment

[`tail_risk_experiment.py`](src/port_opt/backtest/tail_risk_experiment.py)
isolates the portfolio-objective question. It holds the PCA covariance model,
historical-mean expected returns, estimation window, rebalance schedule, and
long-only constraints fixed. It compares ordinary maximum Sharpe with a
tail-adjusted version, plus equal-weight constituents and the XLE baseline.

For weights $w$, historical mean returns $mu$, covariance estimate $Sigma$,
and daily risk-free rate $r_f$, the ordinary strategy maximizes:

$$
\frac{w^\top\mu-r_f}{\sqrt{w^\top\Sigma w}}.
$$

The tail-adjusted strategy maximizes:

$$
\frac{w^\top\mu-r_f}{\sqrt{w^\top\Sigma w}+\lambda L_\alpha(w)}.
$$

Here $R_t$ is the vector of in-sample constituent returns on day $t$ and
$r_{p,t}(w)=R_t^\top w$ is the portfolio return. $L_\alpha(w)$ is the positive
empirical expected-tail-loss: the negative of the mean portfolio return on days
at or below its $(1-\alpha)$ quantile. The default is $\alpha=0.95$ and
$\lambda=1$. Thus volatility and expected tail loss are both daily-return risk
quantities in the denominator; $\lambda=0$ exactly recovers ordinary Sharpe.

```bash
uv run python -m port_opt.backtest.tail_risk_experiment
```

## PCA-dimension development experiment

[`pca_dimension_experiment.py`](src/port_opt/backtest/pca_dimension_experiment.py)
asks how many retained PCA components are useful. It compares otherwise identical
historical-mean maximum-Sharpe PCA portfolios over a fixed component grid: by
default, one through five components. Equal-weight constituents and XLE remain
reference baselines.

This is a development experiment, not a fourth competing final result. Its
default 2018--2023 period ends before the 2024 onward evaluation period used by
the main experiments. Choose and lock a component count based on this earlier
period before reporting its performance in a final experiment; choosing it from
the same final period would be hyperparameter selection on the test set.

```bash
uv run python -m port_opt.backtest.pca_dimension_experiment
```

## Covariance-estimator development study

`run_covariance_estimator_study` is deliberately separate from the XLE
experiment. Supply an earlier development-only return panel—never the final
portfolio evaluation period—and it compares rolling forecasts from sample
covariance, Ledoit--Wolf shrinkage, EWMA covariance over a fixed half-life grid,
and PCA covariance. If an exactly date-aligned factor-return panel is supplied,
it also evaluates PCA-plus-factor covariance.

At every rebalance each candidate fits to the same prior window. The primary,
presentation-first comparison holds a fixed equal-weight portfolio over the
following rebalance block: it compares predicted daily variance,
$w_{EW}^\top\hat\Sigma w_{EW}$, with the sample variance realized over that
block. Lower mean absolute and root mean squared variance errors are better;
the variance calibration ratio should be near one.

As a secondary technical diagnostic, the study retains mean Gaussian
quasi-log-likelihood, $\log\det(\hat\Sigma_t)+e_t^\top\hat\Sigma_t^{-1}e_t$;
lower is better. This is a covariance-forecast scoring rule, not a claim that
returns are Gaussian. The study also reports condition numbers, minimum
eigenvalues, and numerical regularization required only for scoring. Select and
lock a model configuration here before using it in the untouched portfolio
experiment.

```python
from port_opt.backtest import run_covariance_estimator_study

study = run_covariance_estimator_study(
    development_returns,
    factor_returns=development_commodity_returns,
    lookback_periods=504,
    rebalance_frequency=21,
    ewma_half_lives=(63, 126, 252),
)
print(study.metrics)
```

`save_fixed_portfolio_variance_comparison` writes a two-panel bar chart from
that same result: root mean squared equal-weight variance error on the left and
the realized-to-predicted variance calibration ratio on the right. The dashed
line at one is perfect calibration. Each estimator uses the same color in both
panels; EWMA half-life candidates use related blue shades.

```python
from port_opt.backtest import save_fixed_portfolio_variance_comparison

save_fixed_portfolio_variance_comparison(
    study, "covariance_study_outputs/fixed-portfolio-variance-comparison.png"
)
```

## Result tables

Every experiment result includes two tables. `performance_metrics` compares all
portfolio strategies and baselines over their shared out-of-sample dates:
cumulative return, daily and annualized mean return, annualized volatility,
Sharpe and Sortino ratios, maximum drawdown, worst daily return, positive-day
rate, and empirical five-percent tail-return and expected-shortfall measures.
The empirical tail metrics make no normal-return assumption.

`implementation_metrics` applies only to modeled portfolios, because target
turnover is not defined consistently for the XLE and equal-weight baselines. It
reports rebalances, target turnover excluding initial allocation, annualized
target turnover, maximum weight, and effective number of assets. The standard
experiment run writes both tables to `research_outputs` as CSV files.

### Metric definitions

Let $r_t$ be a series' out-of-sample daily simple return, $T$ the number of
observations, $r_f$ the daily risk-free rate, and $N=252$ trading periods per
year. All performance metrics use the same out-of-sample dates for every series.

| Metric | Definition and interpretation |
| --- | --- |
| `observations` | Number of out-of-sample daily returns, $T$. |
| `cumulative_return` | Total compounded return: $\prod_{t=1}^{T}(1+r_t)-1$. |
| `mean_daily_return` | Arithmetic mean daily return: $\bar r=T^{-1}\sum_t r_t$. |
| `annualized_arithmetic_return` | Daily arithmetic mean scaled by $N$: $N\bar r$. It is not compounded growth. |
| `annualized_geometric_return` | Annualized compounded growth: $\left(\prod_t(1+r_t)\right)^{N/T}-1$. |
| `annualized_volatility` | Daily sample volatility annualized: $\sqrt{N}\operatorname{sd}(r_t)$, using `ddof=1`. |
| `sharpe_ratio` | $\sqrt{N}(\bar r-r_f)/\operatorname{sd}(r_t)$. It measures excess return per unit of total volatility. |
| `sortino_ratio` | $N(\bar r-r_f)/\left[\sqrt{N}\sqrt{T^{-1}\sum_t\min(r_t-r_f,0)^2}\right]$. It penalizes downside deviations but not upside variation. |
| `maximum_drawdown` | The minimum of $W_t/\max_{s\leq t}W_s-1$, where $W_t=\prod_{s\leq t}(1+r_s)$. It is the largest peak-to-trough compounded loss. |
| `worst_daily_return` | $\min_t r_t$: the single most negative daily return. |
| `positive_day_fraction` | $T^{-1}\sum_t\mathbf{1}\{r_t>0\}$. It describes frequency, not the magnitude, of positive days. |
| `tail_return_quantile_5pct` | Empirical fifth percentile of daily returns. Five percent of observed days fall at or below this threshold. |
| `tail_expected_shortfall_5pct` | Mean return among days at or below the empirical fifth-percentile threshold. It describes the average realized severity of the worst five percent of days. |

The tail metrics are empirical order statistics; they do not assume normally
distributed returns. Their names change consistently if a non-default tail
probability is selected in `summarize_performance`.

For a modeled strategy, let $w_{i,j}$ be its target weight in asset $j$ at
rebalance $i$. The first allocation is funding, not turnover. For every later
rebalance, target turnover is:

$$
\operatorname{turnover}_i=\frac{1}{2}\sum_j|w_{i,j}-w_{i-1,j}|.
$$

| Implementation metric | Definition and interpretation |
| --- | --- |
| `out_of_sample_observations` | Number of daily strategy returns after initial fitting. |
| `rebalances` | Number of target-weight estimates applied in the backtest. |
| `total_turnover_ex_initial` | $\sum_{i=2}^{I}\operatorname{turnover}_i$. A value of 2.0 represents target reallocations totaling 200% of portfolio value after funding. |
| `mean_rebalance_turnover_ex_initial` | Average $\operatorname{turnover}_i$ across non-initial rebalances. |
| `annualized_turnover` | $N\sum_{i=2}^{I}\operatorname{turnover}_i/T$. It scales observed target turnover to a 252-observation year. |
| `mean_max_weight` | Mean over out-of-sample days of $\max_j w_{t,j}$; higher values indicate more persistent concentration. |
| `maximum_weight` | Maximum observed $\max_j w_{t,j}$; the greatest single-asset target allocation. |
| `mean_effective_number_assets` | Mean over days of $1/\sum_j w_{t,j}^{2}$. It equals the actual asset count under equal weights and declines as the portfolio concentrates. |

Turnover is based on target weights only. It does not model intervening weight
drift, transaction costs, bid-ask spreads, market impact, or taxes.

### Comparing strategies statistically

When a research question is whether one strategy earned a higher average daily
return than another, compare their paired daily active returns,
$d_t=r_t^{A}-r_t^{B}$, rather than testing one final cumulative-return value.
Use a two-sided Newey--West/HAC test of whether $\mathbb{E}[d_t]=0$ and report
the mean active return, its annualized equivalent, a HAC confidence interval,
and the p-value. The HAC lag should be selected and recorded before inspection;
the rebalance holding interval is a natural starting point when weights are
held over multiple days.

Restrict such tests to a small, explicitly pre-specified set of comparisons and
state whether p-values are unadjusted. Statistical significance concerns the
evaluated sample's mean daily return only; it is not evidence by itself of
superior drawdown, tail loss, turnover, or general future performance.

The factor-construction experiment writes the pre-specified PCA-plus-commodity
versus PCA comparison, and the covariance-construction experiment writes the
pre-specified EWMA-PCA versus PCA comparison, to `statistical-tests.csv`.
Both use a two-sided test with a 20-day HAC lag by default; `hac_lag` is an
explicit experiment parameter and is recorded in the output.

## Visual diagnostics

The historical-mean experiment retains PCA and Markowitz covariance estimates
from a selected rebalance record and the daily returns of all six comparison
series. `save_xle_experiment_visuals` therefore writes every standard chart from
that one result: a growth comparison, PCA-versus-Markowitz and
PCA-versus-commodity-factor and PCA-versus-commodity-only covariance heatmaps
on a common color scale within each comparison, separate return histograms, a
six-panel risk/return comparison, and a compact implementation comparison. The
module entry point runs this function
automatically.

The risk/return figure compares annualized geometric return, annualized
volatility, Sharpe ratio, Sortino ratio, maximum drawdown, and empirical 5%
expected shortfall. These answer distinct questions: return growth is not shown
there because it has its own time-series figure; maximum drawdown is a
path-dependent loss; and expected shortfall is the average daily loss on the
worst five percent of observed days. The implementation figure compares
annualized target turnover, mean maximum target weight, and mean effective
number of assets for optimized strategies only.

```python
from port_opt.backtest.xle_experiment import (
    run_xle_pca_historical_mean_experiment,
    save_xle_experiment_visuals,
)

experiment = run_xle_pca_historical_mean_experiment(
    num_principal_components=3,
    ewma_half_life=63,
    covariance_rebalance_index=0,
)
save_xle_experiment_visuals(experiment, "research_outputs")
print(experiment.performance_metrics)
print(experiment.implementation_metrics)
```

## Research assumptions to make explicit in every result

- Returns are daily simple returns. Input data must already be aligned to a shared
  trading calendar with missing and non-finite values resolved.
- The optimizer convention is fully invested and long-only unless a strategy says
  otherwise. Current Sharpe optimization uses a daily risk-free rate.
- A rebalance uses data ending the prior trading observation; weights apply from the
  next observed return. This excludes look-ahead but does not yet model execution lag.
- Reported gross returns exclude transaction costs, slippage, taxes, borrowing,
  management fees, and corporate-action/data-vendor differences. Turnover records
  half the absolute change in target weights at each rebalance; it does not yet
  account for intervening weight drift, so it is an input to—not a substitute for—a
  future execution-cost model.
- Annualized Sharpe uses 252 periods/year and sample volatility (`ddof=1`).
- Baselines must use the same date index, return convention, and cost assumptions
  as the portfolio being compared.

## Next research building blocks

Maintenance:
* Expand the pytest suite across estimators, optimizers, and data adapters.

Research and Experiment Design
* Add paired in-sample diagnostics to the walk-forward result (out-of-sample daily
  returns and Sharpe are now supported).
* statistical tests (are X returns really higher than baselines, do X returns follow an expected, known distribution)
* Visual products -- graph each portfolio strategy by growth over time, and add
  covariance and allocation explainers for the historical-mean strategy.


Portfolio Design
* Make risk and expected-return estimators composable with objectives (minimum
  variance, target return, CVaR, tracking error, turnover-aware optimization).

Report Construction
* Automatically construct a report with findings. Center around hedging portfolio before and during real-world event (strait of hormuz closure)
