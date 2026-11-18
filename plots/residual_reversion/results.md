# Residual reversion - results

**1,201 stocks - 99 configurations - 2004-01-01 to 2023-12-29**

Net results include trading fees. Dollar figures illustrate a $1 million starting portfolio, rebased at the start of each listed period.

## Baseline

5 PCA factors; 500-day PCA window; 100-day factor regression; 60-day residual score; long/short 20% each; daily rebalancing; 5 bp one-way fees.

| Period | Sharpe | Net return | P&L | Max drawdown | Daily turnover | Notional traded |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 2.70 | 419.0% | $4,190,320 | -8.3% | 34.1% | $5,126.3m |
| Development | 3.52 | 301.3% | $3,013,210 | -3.7% | 34.6% | $3,174.3m |
| Holdout | 1.28 | 29.3% | $293,309 | -8.3% | 32.7% | $486.4m |

Notional traded is buys plus sells, not share volume or market liquidity. It grows with simulated portfolio equity.

## Main comparisons

| Configuration | Overall Sharpe | Holdout Sharpe | Overall net return | Holdout CAGR |
|---|---:|---:|---:|---:|
| Baseline | 2.70 | 1.28 | 419.0% | 5.3% |
| Development-selected | 3.25 | 1.36 | 848.0% | 6.4% |
| OU threshold | 2.13 | 1.10 | 131.6% | 2.8% |
| Calibrated, 1 day | 1.15 | 0.19 | 37.2% | 0.4% |
| Alpha benchmark | -1.14 | -0.99 | -54.6% | -4.4% |
| Extra execution day | 0.19 | 0.20 | 10.1% | 0.6% |
| 10 bp per side | 1.30 | 0.27 | 119.9% | 1.0% |

Development-selected configuration: **Rank: K=5, PCA=500d, regression=252d, score=20d, rebalance=1d**. Selection used development Sharpe only.

## Rebalance / forecast horizons

| Rule | Interval | Overall Sharpe | Holdout Sharpe | Overall average gross |
|---|---:|---:|---:|---:|
| Rank | 1 days | 2.70 | 1.28 | 0.85 |
| Rank | 5 days | 1.47 | 1.16 | 0.86 |
| Rank | 20 days | 0.80 | 0.73 | 0.86 |
| Threshold | 1 days | 2.12 | 1.24 | 0.88 |
| Threshold | 5 days | 1.25 | 0.90 | 0.87 |
| Threshold | 20 days | 0.72 | 0.79 | 0.87 |
| Calibrated | 1 days | 1.15 | 0.19 | 0.06 |
| Calibrated | 5 days | 1.13 | 0.88 | 0.29 |
| Calibrated | 20 days | 0.78 | 0.74 | 0.27 |

## Calibration check - whole evaluation period

| Forward horizon | Reversal rank IC | Forecast IC | Forecast MSE / zero-forecast MSE |
|---|---:|---:|---:|
| 1 days | 0.020 | 0.020 | 0.999610 |
| 5 days | 0.026 | 0.021 | 0.999593 |
| 20 days | 0.022 | 0.013 | 1.000030 |

Positive IC indicates predictive alignment; an MSE ratio below 1 improves on a zero-return forecast. These are gross forecast diagnostics, not net trading profits.

## Interpretation

- **Dates:** development is 2004-01-01 to 2018-12-31; holdout is 2019-01-01 to 2023-12-29. The start is the first full year after the longest required warm-up, replacing the arbitrary 2005 cutoff.
- **Regression:** ordinary rolling OLS (`numpy.linalg.lstsq`) of each stock on an intercept and PCA factor returns. It is fully refitted every five days, or when PCA changes; coefficients are reused between refits. It is not recursive regression. The separate calibration spline efficiently maintains rolling X-transpose-X and X-transpose-y statistics and solves ridge every 21 matured training dates.
- **Timing:** signals use today's information for subsequent returns. Forecast labels mature after 1/5/20 days and use frozen-origin buy-and-hold hedges. Calibrated trading uses the matching rebalance interval.
- **Neutrality:** beta and dollar neutrality apply at rebalance. Exposures drift between scheduled trades. Gross and position limits also apply at rebalance; sparse signals may leave cash idle.
- **Execution:** signal-close fills are idealized. Compare the delay and cost rows before treating performance as executable. Cash earns zero interest; no impact or stock-specific borrow constraints are modeled.
- **Research limits:** the historical holdout has already been inspected. No nested selection or deflated-Sharpe correction is claimed. Universe survivorship is unverified; the market benchmark is an equal-weight proxy.

## Detailed outputs

- [All configuration statistics](tables/strategy_statistics.csv)
- [Signal and prediction quantile statistics](tables/calibration_statistics.csv)
- [Performance plots](performance/) - [Parameter heatmaps](parameter_sweeps/) - [Trading-rule comparisons](trading_rules/)
- [Calibration plots](calibration/) - [Risk plots](risk/)
- [Run configuration and data definitions](data/run_configuration.json)

Calibration deciles are formed at the signal date; crossing-period labels are excluded from split-period views. Error bars use daily group means and approximate 95% Newey-West intervals with 21 lags. Whole-period views are descriptive.
