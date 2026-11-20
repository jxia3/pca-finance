"""Run from the repo root: python scripts/q6_diagnostics.py."""

import argparse
import csv
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


PC_NAMES = ("PC1", "PC2", "PC3", "PC4", "PC5")
BENCHMARK_NAMES = ("Market\n(Mkt–RF)", "Value\n(HML)", "Momentum\n(Mom)")


def load_french_returns(path, columns):
    """Read selected columns from the daily French files, skipping their preamble."""
    result = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.reader(stream):
            day = row[0].strip() if row else ""
            if len(day) == 8 and day.isdigit():
                values = tuple(float(row[column]) for column in columns)
                if all(np.isfinite(value) and value > -99 for value in values):
                    result[date.fromisoformat(day)] = values
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", type=Path,
                        default=Path("question_figures/q3_factor_returns.csv"))
    parser.add_argument("--ff3", type=Path,
                        default=Path("materials/F-F_Research_Data_Factors_daily.csv"))
    parser.add_argument("--momentum", type=Path,
                        default=Path("materials/F-F_Momentum_Factor_daily.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("question_figures/q6_pca_vs_standard_factors.png"))
    args = parser.parse_args()
    for path in (args.factors, args.ff3, args.momentum):
        if not path.is_file():
            parser.error(f"Missing {path}. Run from the repo root.")

    with args.factors.open(newline="", encoding="utf-8-sig") as stream:
        factor_rows = list(csv.DictReader(stream))
    by_date = {date.fromisoformat(row["date"]): tuple(float(row[pc]) for pc in PC_NAMES)
               for row in factor_rows}
    if len(by_date) != len(factor_rows):
        parser.error("The Q3 factor CSV contains duplicate dates.")

    market_value = load_french_returns(args.ff3, (1, 3))
    momentum = load_french_returns(args.momentum, (1,))
    dates = sorted(by_date.keys() & market_value.keys() & momentum.keys())
    if len(dates) < 5:
        parser.error("At least five overlapping trading days are required.")
    pcs = np.asarray([by_date[day] for day in dates])
    benchmarks = np.asarray([market_value[day] + momentum[day] for day in dates])
    combined = np.column_stack((pcs, benchmarks))
    if not np.isfinite(combined).all() or np.any(combined.std(axis=0) == 0):
        parser.error("All matched factor series must be finite and nonconstant.")

    correlations = np.corrcoef(combined, rowvar=False)[:len(PC_NAMES), len(PC_NAMES):]
    design = np.column_stack((np.ones(len(dates)), benchmarks))
    coefficients, _, rank, _ = np.linalg.lstsq(design, pcs, rcond=None)
    if rank != design.shape[1]:
        parser.error("The three benchmark factors and intercept must have full rank.")
    residual = pcs - design @ coefficients
    total_ss = np.square(pcs - pcs.mean(axis=0)).sum(axis=0)
    r2 = 1 - np.square(residual).sum(axis=0) / total_ss

    print(f"Matched {len(dates):,} of {len(by_date):,} dates: {dates[0]} to {dates[-1]}")
    print("        Mkt-RF       HML       Mom    Joint R^2")
    for pc, values, fit in zip(PC_NAMES, correlations, r2):
        print(f"{pc:>3}  " + "  ".join(f"{value:>8.4f}" for value in values)
              + f"    {fit:>8.4f}")

    fig, (heatmap, bars) = plt.subplots(1, 2, figsize=(9.3, 3.25), layout="constrained",
                                        width_ratios=(1.1, 1))
    image = heatmap.imshow(correlations, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    heatmap.set(title="Daily Pearson correlations")
    heatmap.set_xticks(range(len(BENCHMARK_NAMES)), BENCHMARK_NAMES)
    heatmap.set_yticks(range(len(PC_NAMES)), PC_NAMES)
    for (row, column), value in np.ndenumerate(correlations):
        heatmap.text(column, row, f"{value:+.2f}", ha="center", va="center",
                     color="white" if abs(value) >= .6 else "#172531", fontsize=10)
    fig.colorbar(image, ax=heatmap, shrink=.82, ticks=[-1, 0, 1])

    positions = np.arange(len(PC_NAMES))
    bars.barh(positions, r2, color="#1a5677", height=.56)
    bars.set_yticks(positions, PC_NAMES)
    bars.invert_yaxis()
    bars.set(title="Joint explanatory power", xlabel=r"In-sample $R^2$", xlim=(0, 1))
    bars.xaxis.set_major_formatter(PercentFormatter(1))
    bars.grid(axis="x", color="#dce1e6", linewidth=.7)
    bars.set_axisbelow(True)
    bars.spines[["top", "right"]].set_visible(False)
    for position, value in zip(positions, r2):
        bars.text(value + .015 if value <= .9 else value - .015, position, f"{value:.1%}",
                  va="center", ha="left" if value <= .9 else "right",
                  color="#172531" if value <= .9 else "white", fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=260)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
