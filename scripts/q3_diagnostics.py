"""From the repository root, run: python -m scripts.q3_diagnostics."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.config import DAILY_RETURNS_FILE
from src.data import load_daily_returns
from src.models.pca import correlation_pca


def main():
    if not Path(DAILY_RETURNS_FILE).is_file():
        raise SystemExit(f"Missing {DAILY_RETURNS_FILE}. Run from the repository root.")

    daily = load_daily_returns()
    returns = daily.returns
    component_count = 5
    window = 63
    if returns.ndim != 2 or returns.shape[0] < window or returns.shape[1] < component_count:
        raise SystemExit(f"Q3 needs at least {window} dates and five stock return series.")
    if not np.isfinite(returns).all() or np.any(returns.std(axis=0, ddof=1) == 0):
        raise SystemExit("The return panel needs finite values and nonconstant stock series.")
    if np.any(np.diff(daily.dates) <= np.timedelta64(0, "D")):
        raise SystemExit("Daily return dates must be unique and increasing.")

    # Match the fitted PCA and sign convention used for Q2.
    fit = correlation_pca(returns, returns.shape[1])
    components = fit.components[:, :component_count].copy()
    anchors = np.abs(components).argmax(axis=0)
    signs = np.sign(components[anchors, np.arange(component_count)])
    components *= signs
    weights = components / fit.std_devs[:, None]
    factors = returns @ weights

    centered_scores = ((returns - fit.means) / fit.std_devs) @ components
    np.testing.assert_allclose(factors - factors.mean(axis=0), centered_scores,
                               atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(factors.var(axis=0, ddof=1),
                               fit.eigenvalues[:component_count], atol=1e-9, rtol=1e-9)
    rows, cols = np.triu_indices(component_count, k=1)

    scaled_corr = np.asarray(np.corrcoef(factors, rowvar=False), dtype=float)
    unadjusted_corr = np.asarray(
        np.corrcoef(returns @ components, rowvar=False), dtype=float
    )
    correlations = scaled_corr[rows, cols]
    unadjusted_correlations = unadjusted_corr[rows, cols]
    print(f"Largest absolute factor correlation, without scaling: "
          f"{np.abs(unadjusted_correlations).max():.6f}")
    print(f"Largest absolute factor correlation, with inverse-volatility scaling: "
          f"{np.abs(correlations).max():.6f}")

    names = [f"PC{k + 1}" for k in range(component_count)]
    rolling_volatility = sliding_window_view(factors, window, axis=0).std(axis=-1, ddof=1)
    rolling_dates = daily.dates[window - 1:]
    full_volatility = factors.std(axis=0, ddof=1)
    peaks = rolling_volatility.argmax(axis=0)
    print(f"\nFactor   Daily mean   Daily SD   Peak {window}-day SD   Window end")
    for k, name in enumerate(names):
        print(f"{name:>6}   {factors[:, k].mean():>10.4f}   {full_volatility[k]:>8.4f}"
              f"   {rolling_volatility[peaks[k], k]:>14.4f}   {rolling_dates[peaks[k]]}")

    csv_path = Path("question_figures/q3_factor_returns.csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["date", *names])
        writer.writerows((str(date), *row) for date, row in zip(daily.dates, factors))

    fig, ax = plt.subplots(figsize=(10.5, 3.0), layout="constrained")
    colors = ("#104c73", "#aa4f30", "#307d76", "#78518c", "#60656e")
    for k, (name, color) in enumerate(zip(names, colors)):
        ax.plot(rolling_dates, rolling_volatility[:, k], label=name, color=color,
                linewidth=1.6 if k == 0 else 0.95)
    ax.set(ylabel="Daily standard deviation (factor units)", xlabel="Window end",
           ylim=(0, None), xlim=(rolling_dates[0], daily.dates[-1]))
    ax.legend(loc="upper right", ncols=5, frameon=False)
    ax.grid(axis="y", color="#dce1e6", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    figure_path = Path("question_figures/q3_factor_volatility.png")
    figure_path.parent.mkdir(exist_ok=True)
    fig.savefig(figure_path, dpi=250)
    plt.close(fig)
    print(f"\nSaved {csv_path} and {figure_path}")


if __name__ == "__main__":
    main()
