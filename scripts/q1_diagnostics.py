"""From the repository root, run: python -m scripts.q4_diagnostics."""

import argparse
import csv
import io
from datetime import date
from pathlib import Path
from zipfile import ZipFile

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


DEFAULT_MARKET_CSV = Path("materials/F-F_Research_Data_Factors_daily.csv")
ROLLING_WINDOW = 252


def load_market_returns(path):
    """Return the benchmark vintage and daily total market returns in percent."""
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as bundle:
            member = next(name for name in bundle.namelist() if name.lower().endswith(".csv"))
            content = bundle.read(member).decode("utf-8-sig")
    else:
        content = path.read_text(encoding="utf-8-sig")

    returns = {}
    for row in csv.reader(io.StringIO(content)):
        if row and len(row[0].strip()) == 8 and row[0].strip().isdigit():
            day = date.fromisoformat(row[0].strip())
            excess_pct, risk_free_pct = float(row[1]), float(row[4])
            if min(excess_pct, risk_free_pct) > -99:
                returns[day] = excess_pct + risk_free_pct
    return content.splitlines()[0], returns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", type=Path,
                        default=Path("question_figures/q3_factor_returns.csv"))
    market_input = parser.add_mutually_exclusive_group()
    market_input.add_argument("--market-csv", type=Path,
                              help=f"French daily factors CSV (default: {DEFAULT_MARKET_CSV}).")
    market_input.add_argument("--market-zip", type=Path,
                              help="Use a local French daily factors ZIP instead.")
    parser.add_argument("--output", type=Path,
                        default=Path("question_figures/q4_pc1_market.png"))
    args = parser.parse_args()
    if not args.factors.is_file():
        parser.error(f"{args.factors} was not found. Set --factors to your Q3 CSV path.")

    with args.factors.open(newline="", encoding="utf-8-sig") as stream:
        rows = [(date.fromisoformat(row["date"]), float(row["PC1"]))
                for row in csv.DictReader(stream)]
    if len({day for day, _ in rows}) != len(rows):
        parser.error("The Q3 CSV contains duplicate dates.")
    if not np.isfinite([score for _, score in rows]).all():
        parser.error("The Q3 CSV contains non-finite PC1 values.")

    market_path = args.market_csv or args.market_zip or DEFAULT_MARKET_CSV
    if not market_path.is_file():
        parser.error(f"{market_path} was not found. Place the benchmark CSV there "
                     "or specify its location with --market-csv.")
    vintage, market = load_market_returns(market_path)
    matched = [(day, market[day], score) for day, score in sorted(rows) if day in market]
    if len(matched) < ROLLING_WINDOW:
        parser.error(f"At least {ROLLING_WINDOW} overlapping trading dates are required.")

    dates, market_values, pc1_values = zip(*matched)
    x = np.asarray(market_values)
    y = np.asarray(pc1_values)
    full_corr = np.corrcoef(x, y)[0, 1]
    x_windows = sliding_window_view(x, ROLLING_WINDOW)
    y_windows = sliding_window_view(y, ROLLING_WINDOW)
    x_centered = x_windows - x_windows.mean(axis=1, keepdims=True)
    y_centered = y_windows - y_windows.mean(axis=1, keepdims=True)
    rolling_corr = (
        (x_centered * y_centered).sum(axis=1)
        / (np.linalg.norm(x_centered, axis=1) * np.linalg.norm(y_centered, axis=1))
    )
    rolling_dates = dates[ROLLING_WINDOW - 1:]
    low, high = int(rolling_corr.argmin()), int(rolling_corr.argmax())
    print(vintage)
    print(f"Matched {len(matched):,} of {len(rows):,} Q3 dates: {dates[0]} to {dates[-1]}")
    print(f"Pearson correlation: {full_corr:.6f}")
    print(f"OLS R-squared: {full_corr**2:.6f}")
    print(f"Trailing {ROLLING_WINDOW}-day median: {np.median(rolling_corr):.6f}")
    print(f"Rolling minimum: {rolling_corr[low]:.6f} ({rolling_dates[low]})")
    print(f"Rolling maximum: {rolling_corr[high]:.6f} ({rolling_dates[high]})")

    fig, (scatter, series) = plt.subplots(1, 2, figsize=(9.6, 3.1), layout="constrained")
    scatter.scatter(x, y, s=6, alpha=.24, color="#1a5677", edgecolors="none")
    slope, intercept = np.polyfit(x, y, 1)
    line_x = np.array([x.min(), x.max()])
    scatter.plot(line_x, intercept + slope * line_x, color="#a24e2e", linewidth=1.5)
    scatter.set(title=f"Daily returns (r = {full_corr:.3f})",
                xlabel="U.S. market return (%)", ylabel="PC1 return (factor units)")
    series.plot(rolling_dates, rolling_corr, linewidth=1.3, color="#1a5677")
    series.axhline(full_corr, linestyle="--", linewidth=1.1, color="#a24e2e",
                   label="Full sample")
    y_bottom = max(-1, np.floor(min(rolling_corr.min(), full_corr) * 10) / 10)
    series.set(title=f"Trailing {ROLLING_WINDOW}-trading-day correlation",
               xlabel="Window end", ylabel="Pearson correlation", ylim=(y_bottom, 1.005))
    series.legend(loc="lower right", frameon=False, fontsize=9)
    series.xaxis.set_major_locator(mdates.YearLocator(5))
    series.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in (scatter, series):
        ax.grid(axis="y", color="#dce1e6", linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=260)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()