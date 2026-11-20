"""From the repository root, run: python -m scripts.q2_diagnostics."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np

from src.config import DAILY_RETURNS_FILE, TICKERS_FILE
from src.data import load_daily_returns, load_tickers
from src.models.pca import correlation_pca


def main():
    for filename in (DAILY_RETURNS_FILE, TICKERS_FILE):
        if not Path(filename).is_file():
            raise SystemExit(f"Missing {filename}. Run from the repository root.")

    daily = load_daily_returns()
    returns = daily.returns
    component_count = 5
    if returns.ndim != 2 or min(returns.shape) <= component_count:
        raise SystemExit("Q2 needs at least six dates and six stock return series.")
    if not np.isfinite(returns).all():
        raise SystemExit("Apply the Q1 missing-return rule before fitting Q2.")
    if np.any(returns.std(axis=0, ddof=1) == 0):
        raise SystemExit("Constant return columns cannot be standardized.")

    # Use the same full-spectrum fit as Q1, then keep its leading five directions.
    fit = correlation_pca(returns, returns.shape[1])
    components = fit.components[:, :component_count].copy()
    columns = np.arange(component_count)
    anchors = np.abs(components).argmax(axis=0)
    components *= np.where(components[anchors, columns] < 0, -1, 1)
    raw_return_weights = components / fit.std_devs[:, None]
    weights = raw_return_weights / np.abs(raw_return_weights).sum(axis=0)
    # The correlation matrix has one unit of variance per stock.
    variance_shares = fit.eigenvalues[:component_count] / returns.shape[1]

    tickers = load_tickers()
    labels = np.array([tickers.get(int(stock_id), str(stock_id))
                       for stock_id in daily.stock_ids])
    shown_count = min(6, returns.shape[1])
    top_count = min(10, returns.shape[1])
    gross_rank = np.argsort(-np.abs(weights), axis=0, kind="stable")
    loading_rank = np.argsort(-np.abs(components), axis=0, kind="stable")
    print(f"Full sample: {returns.shape[0]:,} dates, {returns.shape[1]:,} stocks")
    print(f"Five-component explained variance: {variance_shares.sum():.2%}")
    print(f"PC   Variance    Long   Short*      Net   Top {top_count} gross   Top {top_count} overlap")

    fig, axes = plt.subplots(2, 3, sharex=True, figsize=(11.2, 4.2), layout="constrained")
    limit = np.abs(weights).max() * 1.18
    for k, ax in enumerate(axes.flat[:component_count]):
        portfolio = weights[:, k]
        ranked = gross_rank[:top_count, k]
        long = portfolio[portfolio > 0].sum()
        short = np.abs(portfolio[portfolio < 0]).sum()
        top_gross = np.abs(portfolio[ranked]).sum()
        overlap = np.intersect1d(ranked, loading_rank[:top_count, k]).size
        print(f"{k + 1:>2}  {variance_shares[k]:>9.2%}  {long:>6.1%}  {short:>6.1%}  "
              f"{portfolio.sum():>+7.1%}  {top_gross:>12.2%}  {overlap:>8}/{top_count}")

        selected = ranked[:shown_count][::-1]
        positions = np.arange(shown_count)
        values = portfolio[selected]
        ax.barh(positions, values, color=np.where(values >= 0, "#1e5474", "#9b4b26"))
        ax.set_yticks(positions, labels[selected])
        ax.axvline(0, color="#4f5963", linewidth=0.8)
        ax.set_xlim(-limit, limit)
        ax.set_title(f"PC{k + 1}  |  {variance_shares[k]:.2%} of variance", loc="left",
                     fontsize=10)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, symmetric=True))
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=1))
        ax.tick_params(axis="x", labelbottom=True, labelsize=8)
        ax.tick_params(axis="y", labelsize=9, length=0)
        ax.spines[["top", "right", "left"]].set_visible(False)

    print("*Short exposure is shown as a positive magnitude.")
    print("Overlap counts names appearing in both top lists, before and after "
          "inverse-volatility scaling.")
    axes.flat[-1].set_axis_off()
    fig.supxlabel("Portfolio weight (% of gross exposure)", fontsize=10)
    output = Path("question_figures/q2_eigenportfolios.png")
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=250)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
