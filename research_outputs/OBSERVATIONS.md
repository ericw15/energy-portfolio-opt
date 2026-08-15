# Research-output observations

This note records potential observations from every current research product under
`research_outputs/`. It is a reading aid, not a claim of causality or a portfolio
recommendation. Results are gross of trading costs. Turnover and concentration
should therefore be considered alongside the return and risk figures. No TSV files
were present in this directory at the time of review. The tabular products are CSVs.

## Factor-construction experiment

Products reviewed: `factor/growth-comparison.png`,
`factor/risk-return-comparison.png`, `factor/implementation-comparison.png`, the
three covariance heatmaps, all six return histograms, and the three CSV tables.

- The PCA and PCA-plus-U.S.-commodity-factor strategies have almost identical
  cumulative paths and rounded risk-adjusted metrics. The commodity addition ended
  with slightly higher cumulative return (111.4% versus 111.0%) but the paired
  HAC test estimates only a 0.10% annualized active mean return, with a two-sided
  p-value of 0.570. This sample therefore does not establish a mean-return benefit
  from the additional commodity factors.
- The heatmaps make the preceding result unsurprising: PCA and PCA-plus-commodity
  covariance estimates are visually very similar at the displayed rebalance date.
  They should not be treated as identical merely because the rounded display looks
  similar, but no large visual structural change is apparent.
- The commodity-only model had lower observed volatility (23.5% annualized), a
  less severe maximum drawdown (-19.4%), and a less negative 5% expected shortfall
  (-3.5%) than the PCA variants, at the cost of lower geometric return (27.9%) and
  Sharpe ratio (1.00). Its covariance heatmap has materially lower off-diagonal
  values than the PCA heatmap at the displayed date. This is a candidate explanation
  for its more diversified allocations, not proof of one.
- Implementation is meaningfully different for the commodity-only strategy: mean
  maximum weight is 35.5% and effective number of assets is 3.86, versus roughly
  55% and 2.34 for the two PCA variants. Its annualized turnover is nevertheless
  similar (134.8% versus 137.2% and 138.8%).
- The sample-covariance Markowitz strategy had higher turnover (228.6%), a higher
  mean maximum weight (69.3%), and fewer effective holdings (1.98) than the PCA
  strategies, while also posting lower geometric return, Sharpe, and downside
  metrics in this run.
- The histograms show nontrivial extreme daily observations for every plotted
  series. They are descriptive only: the binning and one sample period do not
  justify a distributional conclusion.

## Covariance-construction experiment

Products reviewed: `covariance/growth-comparison.png`,
`covariance/risk-return-comparison.png`,
`covariance/implementation-comparison.png`, and the three CSV tables.

- In the realized sample, EWMA-PCA has the highest Sharpe (1.14), Sortino (1.55),
  and the least adverse drawdown (-19.2%) and 5% expected shortfall (-3.7%) among
  the optimized covariance variants. Plain PCA has the slightly higher geometric
  return (33.7% versus 33.5%). This is a risk/return trade-off, rather than a clean
  dominance on every measure.
- The paired HAC test of EWMA-PCA minus PCA gives a -0.29% annualized active mean
  return estimate with p = 0.907. It provides no evidence that their average daily
  returns differ, despite their different observed risk profiles.
- EWMA-PCA is less concentrated than the other optimized variants (48.9% mean
  maximum weight and 2.64 effective assets), but it turns over more than plain PCA
  (171.0% versus 133.9% annualized). Pure EWMA has the highest turnover (267.1%).
- Pure EWMA deserves attention as the simplest non-factor alternative. Relative
  to sample-covariance Markowitz, it has lower realized volatility (26.1% versus
  26.6%), less negative 5% expected shortfall (-3.99% versus -4.09%), and slightly
  lower concentration. It gives up geometric return and Sharpe, has worse drawdown,
  and turns over more. It is a simple risk trade-off, not a clear overall winner.
- The growth chart shows all four optimized variants substantially above the two
  passive baselines in this particular period. It does not test whether those
  differences are statistically reliable.

## Tail-risk-objective experiment

Products reviewed: `tail-risk/growth-comparison.png`,
`tail-risk/risk-return-comparison.png`, `tail-risk/implementation-comparison.png`,
and the two CSV tables.

- With lambda = 1, tail-adjusted Sharpe underperformed maximum Sharpe on every
  reported return and downside-risk metric in this run: 30.7% versus 33.7%
  geometric return, 1.01 versus 1.12 Sharpe, -22.4% versus -20.5% maximum
  drawdown, and -4.0% versus -3.9% 5% expected shortfall.
- Lower positive weights move the strategy close to maximum Sharpe. Lambda = 0.1
  has 33.0% geometric return and lambda = 0.01 has 33.5%, but neither improves
  maximum Sharpe on the reported drawdown or expected-shortfall measures.
- The objective change is operationally real rather than a duplicate strategy.
  With lambda = 1, it increases annualized target turnover from 133.9% to
  222.1%, raises mean maximum weight from 54.8% to 62.2%, and at least once
  reaches a 100% allocation. Lambda = 0.01 is much closer to maximum Sharpe in
  both turnover and concentration, as expected from its smaller penalty.
- The lower weights were added after the initial lambda = 1 result. This is an
  exploratory sensitivity check, not a clean calibration study. It is evidence
  about this parameterization and period, not evidence that CVaR-aware objectives
  cannot help under another calibration or constraints.

## PCA-component-count development experiment

Products reviewed: `pca-dimension/growth-comparison.png`,
`pca-dimension/risk-return-comparison.png`,
`pca-dimension/implementation-comparison.png`, and the two CSV tables.

- Within the one- through five-component PCA grid, one component produced the
  highest observed geometric return (44.5%) and Sharpe ratio (1.05). The paths are
  close, particularly after the common large movements, so the difference should
  not be overstated.
- The PC counts have nearly identical realized volatility (about 38.1%--38.2%),
  drawdown (about -32.8% to -32.9%), and 5% expected shortfall (about -5.3%). Two
  components have the least negative expected shortfall by a very small margin.
  One component does not uniformly dominate every risk statistic.
- Changing the component count does not materially improve implementability in
  this grid: annualized turnover remains roughly 293%--301%, mean maximum weights
  remain roughly 72%--74%, and the effective number of holdings stays below 1.83.
- This is explicitly a development-period selection exercise (753 observations),
  not independent evidence for the one-component choice. Its appropriate use is
  to motivate a prespecified component count for the later out-of-sample themes,
  not to report the grid winner as a final performance result.

## Products without an additional standalone conclusion

The growth, risk-return, and implementation figures in each theme visually agree
with their corresponding CSV tables. The observations above capture their
incremental content. The plots improve comparison and auditability but do not add
separate inferential evidence beyond the two factor/covariance paired tests. In
particular, no statistical-test product currently accompanies the tail-risk or
PCA-dimension comparisons.
