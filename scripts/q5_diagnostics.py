"""From the repository root, run: python -m scripts.q5_diagnostics."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from src.config import DAILY_RETURNS_FILE
from src.data import load_daily_returns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", type=Path,
                        default=Path("question_figures/q3_factor_returns.csv"))
    parser.add_argument("--return-units", choices=("decimal", "percent"),
                        default="decimal", help="Units in materials/daily_returns.csv.")
    args = parser.parse_args()
    for path in (Path(DAILY_RETURNS_FILE), args.factors):
        if not path.is_file():
            parser.error(f"Cannot find {path}. Run from the repository root.")

    names = [f"PC{k}" for k in range(1, 6)]
    with args.factors.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not set(["date", *names]).issubset(reader.fieldnames or []):
            parser.error("The Q3 CSV must contain date and PC1 through PC5 columns.")
        factor_rows = {}
        for row in reader:
            day = np.datetime64(row["date"], "D")
            if day in factor_rows:
                parser.error(f"Duplicate factor date: {day}.")
            factor_rows[day] = [float(row[name]) for name in names]

    daily = load_daily_returns()
    dates = daily.dates.astype("datetime64[D]")
    if len(set(dates)) != len(dates) or set(dates) != set(factor_rows):
        parser.error("The stock and Q3 factor CSVs must contain the same unique dates.")
    factors = np.asarray([factor_rows[day] for day in dates])
    returns = daily.returns / 100 if args.return_units == "percent" else daily.returns
    if returns.ndim != 2 or returns.shape[1] < 5 or len(dates) <= len(names) + 1:
        parser.error("Need at least five stocks and seven trading days.")
    if not np.isfinite(returns).all() or not np.isfinite(factors).all():
        parser.error("The stock and factor returns must all be finite.")

    design = np.column_stack((np.ones(len(dates)), factors))
    coefficients, residual_ss, rank, _ = np.linalg.lstsq(design, returns, rcond=None)
    if rank != design.shape[1]:
        parser.error("The intercept and five factor columns must have full rank.")
    alphas = coefficients[0]
    betas = coefficients[1:]
    mean_returns = returns.mean(axis=0)
    np.testing.assert_allclose(alphas, mean_returns - factors.mean(axis=0) @ betas,
                               rtol=1e-9, atol=1e-12)
    total_ss = np.square(returns - mean_returns).sum(axis=0)
    r_squared = np.full(returns.shape[1], np.nan)
    variable = total_ss > 0
    r_squared[variable] = 1 - residual_ss[variable] / total_ss[variable]

    alpha_bp = 10_000 * alphas
    raw_bp = 10_000 * mean_returns
    p5, median, p95 = np.percentile(alpha_bp, [5, 50, 95])
    rank_corr = spearmanr(raw_bp, alpha_bp).statistic
    quintile_count = len(alphas) // 5
    top_raw = np.argsort(-raw_bp, kind="stable")[:quintile_count]
    top_alpha = np.argsort(-alpha_bp, kind="stable")[:quintile_count]
    overlap = len(np.intersect1d(top_raw, top_alpha))
    print(f"Matched {len(dates):,} dates, {len(alphas):,} stocks")
    print(f"Alpha p5 / median / p95 (bp/day): {p5:.3f} / {median:.3f} / {p95:.3f}")
    print(f"Median stock-level in-sample R-squared: {np.nanmedian(r_squared):.4f}")
    print(f"Spearman rank correlation (raw mean vs. alpha): {rank_corr:.4f}")
    print(f"Top-quintile overlap (raw mean vs. alpha): {overlap}/{quintile_count}")

    output = Path("question_figures")
    output.mkdir(exist_ok=True)
    estimates_path = output / "q5_stock_alphas.csv"
    with estimates_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["stock_id", "mean_daily_return", "daily_alpha", "daily_alpha_bp",
                         "r_squared", *(f"beta_{name}" for name in names)])
        writer.writerows((stock_id, raw, alpha, bp, r2, *stock_betas)
                         for stock_id, raw, alpha, bp, r2, stock_betas in
                         zip(daily.stock_ids, mean_returns, alphas, alpha_bp,
                             r_squared, betas.T))

    fig, (hist, scatter) = plt.subplots(1, 2, figsize=(9.6, 3.2), layout="constrained")
    blue, rust = "#1a5677", "#a24e2e"
    hist.hist(alpha_bp, bins="fd", color=blue, edgecolor="white", linewidth=0.45)
    hist.axvline(0, color="#59626a", lw=1, label="Zero")
    hist.axvline(median, color=rust, lw=1.3, ls="--", label=f"Median: {median:.2f} bp")
    hist.set(title="Distribution of estimated alphas", xlabel="Daily alpha (bp)",
             ylabel="Stocks")
    hist.legend(frameon=False, fontsize=8)

    low, high = min(raw_bp.min(), alpha_bp.min()), max(raw_bp.max(), alpha_bp.max())
    scatter.scatter(raw_bp, alpha_bp, color=blue, s=9, alpha=.42, edgecolors="none")
    scatter.plot([low, high], [low, high], color=rust, ls="--", lw=1, label="Equal values")
    scatter.set(title="Mean return versus estimated alpha",
                xlabel="Mean daily stock return (bp)", ylabel="Daily alpha (bp)")
    scatter.legend(frameon=False, fontsize=8)
    for ax in (hist, scatter):
        ax.grid(axis="y", color="#dce1e6", linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    figure_path = output / "q5_stock_alphas.png"
    fig.savefig(figure_path, dpi=250)
    plt.close(fig)
    print(f"Saved {estimates_path} and {figure_path}")


if __name__ == "__main__":
    main()
