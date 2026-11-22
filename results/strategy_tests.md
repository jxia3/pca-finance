## 0. Residual snapback variants

Each variant from its own first trade (April 2002) to December 2023. Market beta is against the equal-weighted universe, with 5bp fees.

| Variant | Sharpe (5bp) | Sharpe (no fees) | Return (5bp) | Volatility (5bp) | Market beta | Max drawdown (5bp) |
|---|---|---|---|---|---|---|
| Equal-weight benchmark (weekly) | 0.82 | 0.83 | 14.7% | 18.8% | 1.00 | -53.0% |
| Default snapback (long only) | 0.82 | 1.03 | 19.1% | 25.2% | 1.08 | -58.5% |
| Stricter cutoff (score <= -2.5) | 0.67 | 0.86 | 15.3% | 26.5% | 1.08 | -74.8% |
| Hybrid, 70% unhedged | 0.87 | 1.14 | 15.7% | 18.8% | 0.77 | -45.0% |
| Hybrid, 50% unhedged | 0.91 | 1.26 | 13.1% | 14.8% | 0.57 | -36.1% |
| Hybrid, 30% unhedged | 0.94 | 1.39 | 10.4% | 11.3% | 0.36 | -30.3% |
| Dynamic volatility mix | 0.96 | 1.39 | 11.3% | 11.9% | 0.31 | -27.8% |
| Fully hedged snapback (long-short) | 0.76 | 1.38 | 6.1% | 8.2% | 0.06 | -21.5% |
| Earlier version: projection hedge of the long picks | 0.83 | 1.23 | 10.2% | 12.7% | 0.12 | -28.7% |
| Earlier version: dynamic mix, unsmoothed, projection hedge | 1.00 | 1.33 | 13.4% | 13.5% | 0.33 | -32.8% |
| Contrarian Treynor-Black |  |  |  |  | 0.04 |  |
| PCA pairs trading |  |  |  |  | 0.02 |  |

### Residual diagnostics (lagged factor model, every stock and day)

| Measure | Value |
|---|---|
| Mean residual (bp/day) | -0.003 |
| Newey-West t of daily cross-sectional mean | -0.02 |
| Median residual std (daily) | 1.8% |
| 60-day mean residual, 10th / 90th percentile across stocks (daily) | -0.25% / 0.24% |
| Median autocorrelation, lag 1 / lag 2 | -0.034 / -0.009 |
| Median Var(3-day sum) / (3 Var(1 day)) | 0.95 |
| Std of snapback score | 1.08 |

Hedged runs trade from day 565 (long-short) and 565 (projection), 5472 and 5472 days.

## 1. Regime robustness

### Contrarian Treynor-Black

| Year | Return (no fees) | Return (5bp) | Sharpe (no fees) | Sharpe (5bp) |
|---|---|---|---|---|
| 2002 | 5.6% | 4.2% | 1.93 | 1.45 |
| 2003 | 3.6% | 2.2% | 1.64 | 1.01 |
| 2004 | 2.6% | 1.2% | 1.51 | 0.71 |
| 2005 | 0.7% | -0.7% | 0.40 | -0.40 |
| 2006 | 2.4% | 1.1% | 1.29 | 0.60 |
| 2007 | -1.1% | -2.4% | -0.38 | -0.85 |
| 2008 | 15.6% | 13.7% | 3.00 | 2.68 |
| 2009 | 23.4% | 21.5% | 4.83 | 4.48 |
| 2010 | 2.2% | 0.8% | 1.14 | 0.42 |
| 2011 | 2.3% | 0.8% | 0.90 | 0.34 |
| 2012 | 5.3% | 3.9% | 2.58 | 1.91 |
| 2013 | 0.2% | -1.1% | 0.14 | -0.63 |
| 2014 | 2.7% | 1.3% | 1.62 | 0.77 |
| 2015 | -0.3% | -1.6% | -0.14 | -0.85 |
| 2016 | 5.5% | 4.1% | 2.50 | 1.88 |
| 2017 | 0.3% | -0.9% | 0.14 | -0.44 |
| 2018 | 1.7% | 0.4% | 0.70 | 0.18 |
| 2019 | 5.1% | 3.7% | 2.46 | 1.82 |
| 2020 | 8.1% | 6.5% | 1.26 | 1.03 |
| 2021 | 1.6% | 0.1% | 0.61 | 0.04 |
| 2022 | 1.6% | 0.2% | 0.44 | 0.06 |
| 2023 | -0.8% | -2.1% | -0.22 | -0.61 |

| Measure | Value |
|---|---|
| Years positive (no fees / 5bp) | 19 / 16 of 22 |
| Share of total profit from best year (2009) | 25.2% |
| Share from best two years (2009, 2008) | 42.5% |
| Sharpe, all years (no fees / 5bp) | 1.31 / 0.85 |
| Sharpe excluding 2009 (no fees / 5bp) | 1.07 / 0.59 |
| Sharpe excluding 2009 and 2008 (no fees / 5bp) | 0.91 / 0.41 |
| Sharpe 2002-2012 / 2013-2023 (5bp) | 1.41 / 0.32 |
| Max drawdown (no fees / 5bp) | -5.2% / -6.1% |

### PCA pairs trading

| Year | Return (no fees) | Return (5bp) | Sharpe (no fees) | Sharpe (5bp) |
|---|---|---|---|---|
| 2002 | 13.8% | 13.0% | 3.52 | 3.33 |
| 2003 | 0.9% | 0.1% | 0.33 | 0.07 |
| 2004 | 2.6% | 1.9% | 1.02 | 0.74 |
| 2005 | 1.4% | 0.7% | 0.69 | 0.34 |
| 2006 | 1.6% | 0.8% | 0.85 | 0.47 |
| 2007 | 4.8% | 4.0% | 1.77 | 1.48 |
| 2008 | 7.2% | 6.2% | 1.53 | 1.34 |
| 2009 | 5.5% | 4.8% | 2.02 | 1.79 |
| 2010 | 4.8% | 4.1% | 2.49 | 2.13 |
| 2011 | 2.1% | 1.3% | 1.12 | 0.72 |
| 2012 | -0.6% | -1.2% | -0.42 | -0.86 |
| 2013 | 1.8% | 1.1% | 0.94 | 0.59 |
| 2014 | 1.1% | 0.4% | 0.79 | 0.30 |
| 2015 | 0.1% | -0.6% | 0.07 | -0.33 |
| 2016 | 7.8% | 7.0% | 3.80 | 3.42 |
| 2017 | 3.9% | 3.2% | 2.28 | 1.90 |
| 2018 | 3.6% | 2.8% | 2.06 | 1.63 |
| 2019 | 4.5% | 3.9% | 2.71 | 2.31 |
| 2020 | 9.5% | 8.6% | 1.91 | 1.74 |
| 2021 | 2.7% | 2.1% | 1.14 | 0.89 |
| 2022 | 0.7% | -0.1% | 0.32 | -0.04 |
| 2023 | 2.2% | 1.6% | 1.05 | 0.77 |

| Measure | Value |
|---|---|
| Years positive (no fees / 5bp) | 21 / 19 of 22 |
| Share of total profit from best year (2002) | 16.4% |
| Share from best two years (2002, 2020) | 27.8% |
| Sharpe, all years (no fees / 5bp) | 1.45 / 1.17 |
| Sharpe excluding 2002 (no fees / 5bp) | 1.31 / 1.02 |
| Sharpe excluding 2002 and 2020 (no fees / 5bp) | 1.28 / 0.97 |
| Sharpe 2002-2012 / 2013-2023 (5bp) | 1.18 / 1.16 |
| Max drawdown (no fees / 5bp) | -3.6% / -3.9% |

### Residual snapback

| Year | Return (no fees) | Return (5bp) | Sharpe (no fees) | Sharpe (5bp) |
|---|---|---|---|---|
| 2002 | 14.1% | 9.9% | 0.77 | 0.59 |
| 2003 | 64.3% | 56.1% | 3.28 | 2.94 |
| 2004 | 59.0% | 50.9% | 2.79 | 2.48 |
| 2005 | 27.1% | 20.7% | 1.90 | 1.50 |
| 2006 | 36.2% | 29.4% | 2.26 | 1.89 |
| 2007 | -14.9% | -19.1% | -0.79 | -1.07 |
| 2008 | -20.7% | -24.6% | -0.18 | -0.28 |
| 2009 | 104.7% | 94.3% | 1.87 | 1.74 |
| 2010 | 35.7% | 28.9% | 1.58 | 1.33 |
| 2011 | 1.5% | -3.5% | 0.19 | 0.01 |
| 2012 | 45.8% | 38.5% | 2.48 | 2.16 |
| 2013 | 58.9% | 51.0% | 3.26 | 2.92 |
| 2014 | 20.4% | 14.3% | 1.36 | 1.00 |
| 2015 | 11.8% | 6.1% | 0.68 | 0.41 |
| 2016 | 28.5% | 22.1% | 1.60 | 1.29 |
| 2017 | 13.8% | 8.1% | 1.16 | 0.73 |
| 2018 | 13.2% | 7.5% | 0.85 | 0.53 |
| 2019 | 34.3% | 27.6% | 2.06 | 1.72 |
| 2020 | 20.5% | 14.5% | 0.62 | 0.52 |
| 2021 | 24.1% | 17.9% | 1.35 | 1.05 |
| 2022 | 1.2% | -3.9% | 0.18 | -0.01 |
| 2023 | 31.9% | 25.3% | 1.49 | 1.24 |

| Measure | Value |
|---|---|
| Years positive (no fees / 5bp) | 20 / 18 of 22 |
| Share of total profit from best year (2009) | 14.6% |
| Share from best two years (2009, 2003) | 24.7% |
| Sharpe, all years (no fees / 5bp) | 1.03 / 0.82 |
| Sharpe excluding 2009 (no fees / 5bp) | 0.97 / 0.75 |
| Sharpe excluding 2009 and 2003 (no fees / 5bp) | 0.90 / 0.68 |
| Sharpe 2002-2012 / 2013-2023 (5bp) | 0.86 / 0.78 |
| Max drawdown (no fees / 5bp) | -54.6% / -58.5% |

### Residual snapback (hedged)

| Year | Return (no fees) | Return (5bp) | Sharpe (no fees) | Sharpe (5bp) |
|---|---|---|---|---|
| 2002 | 13.8% | 9.7% | 1.97 | 1.41 |
| 2003 | 5.2% | -0.0% | 0.71 | 0.03 |
| 2004 | 17.6% | 11.7% | 2.50 | 1.69 |
| 2005 | 21.6% | 15.5% | 3.78 | 2.78 |
| 2006 | 19.7% | 13.7% | 3.49 | 2.49 |
| 2007 | -1.4% | -6.3% | -0.18 | -0.95 |
| 2008 | 25.7% | 19.6% | 1.67 | 1.32 |
| 2009 | 27.1% | 20.9% | 1.98 | 1.57 |
| 2010 | 2.7% | -2.5% | 0.44 | -0.35 |
| 2011 | 15.6% | 9.8% | 2.30 | 1.48 |
| 2012 | 12.8% | 7.2% | 1.91 | 1.11 |
| 2013 | 7.5% | 2.2% | 1.63 | 0.49 |
| 2014 | 7.8% | 2.4% | 1.56 | 0.50 |
| 2015 | 1.4% | -3.8% | 0.26 | -0.63 |
| 2016 | 8.9% | 3.4% | 1.44 | 0.59 |
| 2017 | 5.9% | 0.6% | 1.05 | 0.14 |
| 2018 | 17.7% | 11.9% | 3.20 | 2.16 |
| 2019 | 4.3% | -0.9% | 0.69 | -0.11 |
| 2020 | 23.4% | 18.0% | 1.26 | 1.01 |
| 2021 | 10.2% | 4.7% | 1.37 | 0.66 |
| 2022 | 9.4% | 4.0% | 1.10 | 0.50 |
| 2023 | 1.3% | -3.7% | 0.21 | -0.47 |

| Measure | Value |
|---|---|
| Years positive (no fees / 5bp) | 21 / 16 of 22 |
| Share of total profit from best year (2009) | 10.1% |
| Share from best two years (2009, 2008) | 19.7% |
| Sharpe, all years (no fees / 5bp) | 1.38 / 0.76 |
| Sharpe excluding 2009 (no fees / 5bp) | 1.34 / 0.70 |
| Sharpe excluding 2009 and 2008 (no fees / 5bp) | 1.34 / 0.65 |
| Sharpe 2002-2012 / 2013-2023 (5bp) | 1.06 / 0.45 |
| Max drawdown (no fees / 5bp) | -19.6% / -21.5% |

## 2. Parameter sensitivity

Variants are compared over a common period starting 2003-12-26 for Treynor-Black and pairs trading and 2004-03-29 for residual snapback (the 1000-day PCA window needs four years of history). Selection = 2002-2017, holdout = 2018-2023, both with 5bp fees.

### Contrarian Treynor-Black

| Parameter | Value | Sharpe (no fees) | Sharpe (5bp) | Return (5bp) | Sharpe selection | Sharpe holdout |
|---|---|---|---|---|---|---|
| default |  | 1.27 | 0.81 | 2.4% | 1.08 | 0.40 |
| PCA window | 1000 | 1.17 | 0.72 | 2.1% | 0.96 | 0.36 |
| PCA window | 250 | 1.25 | 0.78 | 2.3% | 1.06 | 0.37 |
| alpha window | 125 | 1.55 | 0.93 | 2.7% | 1.13 | 0.64 |
| alpha window | 500 | 1.15 | 0.79 | 2.2% | 0.88 | 0.64 |
| alpha window | 60 | 1.93 | 1.00 | 2.8% | 1.13 | 0.83 |
| score cap | 2 | 1.27 | 0.81 | 2.4% | 1.10 | 0.39 |
| score cap | 5 | 1.27 | 0.80 | 2.3% | 1.06 | 0.41 |

### PCA pairs trading

| Parameter | Value | Sharpe (no fees) | Sharpe (5bp) | Return (5bp) | Sharpe selection | Sharpe holdout |
|---|---|---|---|---|---|---|
| default |  | 1.36 | 1.07 | 2.6% | 1.04 | 1.15 |
| PCA window | 1000 | 1.54 | 1.28 | 3.3% | 1.10 | 1.66 |
| PCA window | 250 | 1.21 | 0.88 | 2.0% | 0.76 | 1.13 |
| entry z | 1.5 | 1.59 | 1.21 | 3.4% | 1.20 | 1.25 |
| entry z | 2.5 | 1.21 | 1.01 | 2.0% | 0.92 | 1.19 |
| entry z | 3 | 1.15 | 1.03 | 1.5% | 0.99 | 1.14 |
| exit z | 0 | 1.25 | 0.98 | 2.5% | 0.97 | 1.02 |
| exit z | 1 | 1.49 | 1.15 | 2.6% | 1.05 | 1.37 |
| num pairs | 100 | 1.73 | 1.38 | 2.7% | 1.51 | 1.20 |
| num pairs | 20 | 0.68 | 0.48 | 1.7% | 0.43 | 0.58 |
| reform days | 5 | 1.54 | 1.14 | 2.5% | 1.06 | 1.32 |
| reform days | 63 | 1.42 | 1.19 | 3.1% | 1.03 | 1.55 |
| selection | correlation | 1.89 | 1.35 | 1.6% | 1.37 | 1.33 |
| spread window | 120 | 1.29 | 1.08 | 2.6% | 1.06 | 1.14 |
| spread window | 20 | 1.93 | 1.43 | 3.4% | 1.63 | 1.05 |
| spread window | 40 | 1.59 | 1.25 | 3.1% | 1.27 | 1.20 |

### Residual snapback

| Parameter | Value | Sharpe (no fees) | Sharpe (5bp) | Return (5bp) | Sharpe selection | Sharpe holdout |
|---|---|---|---|---|---|---|
| default |  | 0.96 | 0.76 | 17.5% | 0.83 | 0.62 |
| PCA window | 1000 | 0.99 | 0.78 | 17.8% | 0.83 | 0.69 |
| PCA window | 250 | 0.95 | 0.74 | 17.0% | 0.79 | 0.66 |
| hedge mix | 30% unhedged | 1.35 | 0.90 | 10.1% | 1.00 | 0.71 |
| hedge mix | 50% unhedged | 1.20 | 0.86 | 12.5% | 0.95 | 0.69 |
| hedge mix | 70% unhedged | 1.08 | 0.81 | 14.7% | 0.89 | 0.66 |
| hedge mix | dynamic | 1.31 | 0.88 | 10.3% | 1.06 | 0.51 |
| max positions | 10 | 0.85 | 0.67 | 16.2% | 0.75 | 0.52 |
| max positions | 50 | 1.02 | 0.81 | 17.9% | 0.87 | 0.68 |
| rebalance | daily | 1.42 | 0.84 | 21.1% | 0.96 | 0.60 |
| regression window | 250 | 1.04 | 0.84 | 20.4% | 0.91 | 0.71 |
| regression window | 60 | 0.95 | 0.75 | 16.8% | 0.83 | 0.59 |
| score threshold | -1.5 | 1.03 | 0.83 | 19.1% | 0.91 | 0.67 |
| score threshold | -2.5 | 0.77 | 0.58 | 12.7% | 0.45 | 0.83 |
| score threshold | -3 | 0.65 | 0.48 | 10.3% | 0.33 | 0.77 |
| signal days | 1 | 1.24 | 1.04 | 25.7% | 1.23 | 0.69 |
| signal days | 5 | 1.04 | 0.84 | 20.0% | 0.92 | 0.68 |
| volatility window | 120 | 0.98 | 0.78 | 18.4% | 0.81 | 0.72 |
| volatility window | 20 | 0.82 | 0.61 | 12.8% | 0.67 | 0.51 |

### Residual snapback (hedged)

| Parameter | Value | Sharpe (no fees) | Sharpe (5bp) | Return (5bp) | Sharpe selection | Sharpe holdout |
|---|---|---|---|---|---|---|
| default |  | 1.38 | 0.76 | 6.1% | 0.87 | 0.59 |
| PCA window | 1000 | 1.39 | 0.75 | 5.8% | 0.84 | 0.61 |
| PCA window | 250 | 1.38 | 0.72 | 5.5% | 0.75 | 0.69 |
| max positions | 10 | 1.34 | 0.79 | 7.1% | 0.80 | 0.76 |
| max positions | 50 | 1.50 | 0.83 | 6.3% | 0.87 | 0.77 |
| rebalance | daily | 2.46 | 0.89 | 9.4% | 0.96 | 0.83 |
| regression window | 250 | 1.31 | 0.82 | 8.4% | 1.06 | 0.68 |
| regression window | 60 | 1.21 | 0.56 | 4.2% | 0.64 | 0.43 |
| score threshold | -1.5 | 1.59 | 0.91 | 6.9% | 1.15 | 0.68 |
| score threshold | -2.5 | 1.16 | 0.68 | 6.5% | 0.89 | 0.31 |
| score threshold | -3 | 0.68 | 0.36 | 3.5% | 0.44 | 0.20 |
| signal days | 1 | 2.06 | 1.46 | 13.0% | 1.71 | 1.03 |
| signal days | 5 | 1.38 | 0.80 | 6.7% | 0.52 | 1.40 |
| volatility window | 120 | 1.51 | 0.93 | 8.0% | 1.03 | 0.78 |
| volatility window | 20 | 1.63 | 0.92 | 6.6% | 0.88 | 1.02 |

## 3. Trading costs

| Strategy | Daily turnover | Annual turnover | Gross return (mean, annual) | Fee drag at 5bp (annual) | Share of edge lost at 5bp | Break-even fee |
|---|---|---|---|---|---|---|
| Contrarian Treynor-Black | 10.9% | 28x | 3.8% | 1.4% | 35.8% | 14.0bp |
| PCA pairs trading | 5.6% | 14x | 3.6% | 0.7% | 19.5% | 25.7bp |
| Residual snapback | 40.8% | 103x | 25.8% | 5.2% | 20.0% | 25.0bp |
| Residual snapback (hedged) | 40.3% | 102x | 11.3% | 5.1% | 44.9% | 11.1bp |

| Sharpe at fee of | 0bp | 2bp | 5bp | 10bp | 20bp |
|---|---|---|---|---|---|
| Contrarian Treynor-Black | 1.31 | 1.13 | 0.85 | 0.38 | -0.57 |
| PCA pairs trading | 1.45 | 1.33 | 1.17 | 0.88 | 0.32 |
| Residual snapback | 1.03 | 0.94 | 0.82 | 0.62 | 0.21 |
| Residual snapback (hedged) | 1.38 | 1.13 | 0.76 | 0.14 | -1.05 |

## 4. Calibration

### Contrarian Treynor-Black

Signal = the strategy's score (minus the capped z-score of alpha / residual variance). Target = each stock's return over the next h days minus the cross-sectional average. Deciles are formed each day; decile 10 is what the strategy buys most.

| Horizon | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1d | -3.0 | -2.0 | -1.3 | -1.0 | 0.6 | 2.1 | 1.8 | 1.6 | 1.0 | 0.1 |
| 5d | -10.7 | -6.6 | -1.7 | -1.0 | 3.1 | 6.2 | 3.6 | 3.6 | 2.9 | 0.7 |
| 20d | -34.4 | -17.4 | -0.5 | 2.1 | 14.2 | 22.1 | 10.0 | 6.9 | 3.4 | -6.1 |

Mean excess return by signal decile, in bp (D1 = most shorted, D10 = most bought).

| Horizon (days) | Return per 1 s.d. of signal | t-stat (Newey-West) |
|---|---|---|
| 1 | 1.10bp | 5.2 |
| 5 | 3.39bp | 3.4 |
| 20 | 7.70bp | 1.8 |

### PCA pairs trading

Signal = minus the spread z-score (positive = buy the first stock, short the second). Target = change in the log price ratio of the pair over the next h days, in the direction of the signal's first leg.

| Spread z-score | Next 1d (bp) | Next 5d (bp) | Next 20d (bp) |
|---|---|---|---|
| z < -3 | 48.4 (n=2,251) | 108.4 (n=2,251) | 157.9 (n=2,244) |
| -3 to -2 | 17.7 (n=16,811) | 20.8 (n=16,799) | 43.9 (n=16,751) |
| -2 to -1 | 7.0 (n=56,696) | 13.5 (n=56,642) | 8.3 (n=56,437) |
| -1 to 0 | 3.0 (n=64,245) | 5.0 (n=64,199) | -0.8 (n=64,030) |
| 0 to 1 | -3.5 (n=64,170) | -6.3 (n=64,124) | -31.8 (n=63,963) |
| 1 to 2 | -7.2 (n=55,074) | -14.3 (n=55,041) | -53.1 (n=54,921) |
| 2 to 3 | -18.3 (n=15,567) | -46.2 (n=15,559) | -101.7 (n=15,524) |
| z > 3 | -59.7 (n=2,036) | -143.8 (n=2,035) | -267.5 (n=2,030) |

The strategy buys the spread below z = -2 and sells it above z = 2, so returns should fall as z rises.

| Horizon (days) | Spread return per 1 s.d. of signal | t-stat (Newey-West) |
|---|---|---|
| 1 | 9.62bp | 13.3 |
| 5 | 19.31bp | 7.3 |
| 20 | 38.99bp | 4.7 |

### Residual snapback

Signal = minus the snapback score (the last 3 days of residuals over the prior 60-day residual volatility, scaled by sqrt 3), for every stock on every day. Target = each stock's return over the next h days minus the cross-sectional average.

| Snapback score | Next 1d (bp) | Next 5d (bp) | Next 20d (bp) |
|---|---|---|---|
| z < -3 | 13.1 (n=52,973) | 46.6 (n=52,961) | 67.1 (n=52,860) |
| -3 to -2 | 10.8 (n=118,870) | 25.8 (n=118,848) | 40.4 (n=118,564) |
| -2 to -1 | 9.7 (n=653,346) | 17.4 (n=653,085) | 24.5 (n=651,181) |
| -1 to 0 | 3.6 (n=2,514,925) | 5.9 (n=2,512,752) | 9.1 (n=2,505,816) |
| 0 to 1 | -4.2 (n=2,393,685) | -6.8 (n=2,391,677) | -9.5 (n=2,385,179) |
| 1 to 2 | -7.7 (n=648,547) | -15.2 (n=648,276) | -27.0 (n=646,495) |
| 2 to 3 | -9.3 (n=132,428) | -22.9 (n=132,388) | -34.4 (n=132,036) |
| z > 3 | -18.3 (n=59,500) | -42.8 (n=59,483) | -39.4 (n=59,324) |

The strategy buys up to 20 of the lowest scores at or below -2, so returns should fall as the score rises.

| Horizon (days) | Excess return per 1 s.d. of signal | t-stat (Newey-West) |
|---|---|---|
| 1 | 5.96bp | 21.6 |
| 5 | 11.38bp | 13.3 |
| 20 | 17.06bp | 10.7 |

## 5. Holdout

Choices are made using only 2002-2017, then evaluated once on 2018-2023 (Sharpe with 5bp fees).

| Strategy | Choice | Sharpe 2002-2017 | Sharpe 2018-2023 |
|---|---|---|---|
| Contrarian Treynor-Black | Default settings | 1.08 | 0.40 |
| Contrarian Treynor-Black | Best setting on selection period (alpha window = 60) | 1.13 | 0.83 |
| Contrarian Treynor-Black | Best K on selection period (K = 20) | 1.11 | 0.63 |
| PCA pairs trading | Default settings | 1.04 | 1.15 |
| PCA pairs trading | Best setting on selection period (spread window = 20) | 1.63 | 1.05 |
| PCA pairs trading | Best K on selection period (K = 15) | 1.41 | 0.58 |
| Residual snapback | Default settings | 0.83 | 0.62 |
| Residual snapback | Best setting on selection period (signal days = 1) | 1.23 | 0.69 |
| Residual snapback | Best K on selection period (K = 5) | 0.91 | 0.62 |
| Residual snapback (hedged) | Default settings | 0.87 | 0.59 |
| Residual snapback (hedged) | Best setting on selection period (signal days = 1) | 1.71 | 1.03 |
| Residual snapback (hedged) | Best K on selection period (K = 3) | 0.98 | 0.79 |
| Treynor-Black, following alpha (not contrarian) | Sign check | -1.63 (no fees) | -0.77 (no fees) |

## 6. Number of factors K

Out-of-sample = PCA and alphas re-estimated on trailing windows (no look-ahead). In-sample = PCA and alphas estimated once on the full 2000-2023 sample, then traded over the same period (look-ahead). Sharpe ratios; 5bp fees only where stated.

| K | T-B out-of-sample | T-B out-of-sample (5bp) | T-B in-sample (contrarian) | T-B in-sample (following alpha) | Pairs out-of-sample | Pairs out-of-sample (5bp) | Pairs in-sample |
|---|---|---|---|---|---|---|---|
| 1 | 0.95 | 0.68 | -1.54 | 1.54 | 0.78 | 0.52 | 0.62 |
| 2 | 0.86 | 0.54 | -2.12 | 2.12 | 1.39 | 1.11 | 0.69 |
| 3 | 0.87 | 0.51 | -2.13 | 2.13 | 1.13 | 0.86 | 1.13 |
| 5 | 1.31 | 0.85 | -2.18 | 2.18 | 1.45 | 1.17 | 1.93 |
| 8 | 1.46 | 0.92 | -2.76 | 2.76 | 1.41 | 1.09 | 2.18 |
| 10 | 1.49 | 0.91 | -2.77 | 2.77 | 1.53 | 1.18 | 2.40 |
| 15 | 1.58 | 0.94 | -3.29 | 3.29 | 1.51 | 1.14 | 2.39 |
| 20 | 1.66 | 0.96 | -3.42 | 3.42 | 1.60 | 1.18 | 2.15 |

Residual snapback. In-sample = PCA and regression estimated once on the full sample; residuals are still computed day by day from those fixed coefficients.

| K | Snapback out-of-sample | Snapback out-of-sample (5bp) | Snapback in-sample | Hedged out-of-sample | Hedged out-of-sample (5bp) | Hedged in-sample |
|---|---|---|---|---|---|---|
| 1 | 1.01 | 0.81 | 1.03 | 1.41 | 0.89 | 1.40 |
| 2 | 1.02 | 0.82 | 1.01 | 1.31 | 0.78 | 1.52 |
| 3 | 1.03 | 0.82 | 1.07 | 1.53 | 0.91 | 1.64 |
| 5 | 1.03 | 0.82 | 1.18 | 1.38 | 0.76 | 1.84 |
| 8 | 1.00 | 0.79 | 1.13 | 1.50 | 0.77 | 1.86 |
| 10 | 0.98 | 0.77 | 1.16 | 1.38 | 0.65 | 1.93 |
| 15 | 1.01 | 0.80 | 1.12 | 1.01 | 0.29 | 1.88 |
| 20 | 0.98 | 0.77 | 1.12 | 0.46 | -0.21 | 1.97 |

