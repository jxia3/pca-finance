import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import numpy as np

from src import data
from strategy_tests import (
    DEVELOPMENT, HOLDOUT_START, K_VALUES, TRADE_FEE, Z_LABELS, Series, alpha_calibration, common_starts,
    load_results, log_price_paths, market_beta, net_returns, newey_west_t, pair_calibration,
    rolling_residual_means, sharpe, snapback, snapback_calibration, snapback_scores, weekly_rebalance_dates,
)

FIGURE_DIR = "writeup/figures"
STRATEGIES = {"tb": "Contrarian Treynor-Black", "pairs": "PCA pairs trading", "snap": "Residual snapback"}
# Validated categorical slots 1 to 3, color always identifies the strategy
COLORS = {"tb": "#2a78d6", "pairs": "#eb6834", "snap": "#1baf7a"}
# Within residual snapback the hedged version is drawn hollow, dashed or hatched in the same color
LONG_LABEL = "Long only"

def period_legend(axes: Axes, labels: tuple[str, str], loc: str) -> None:
    # Hollow = earlier or no-fee value, filled = later or after-fee value, drawn in neutral ink
    handles = [Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor="white",
                      markeredgecolor=SECONDARY, markeredgewidth=1.5),
               Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor=SECONDARY,
                      markeredgecolor="white", markeredgewidth=1.5)]
    axes.legend(handles, labels, loc=loc, fontsize=8)
HEDGED_LABEL = "Fully hedged"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SHADE = "#f0efec"
CRISES = (("2008-01-01", "2009-12-31", "2008-09"), ("2020-02-01", "2020-12-31", "2020"))

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": SECONDARY,
    "axes.titlecolor": INK,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": SECONDARY,
    "ytick.labelcolor": SECONDARY,
    "legend.frameon": False,
    "legend.labelcolor": SECONDARY,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "savefig.facecolor": "white",
})

def save(figure: plt.Figure, name: str) -> None:
    os.makedirs(FIGURE_DIR, exist_ok=True)
    for extension in ("pdf", "png"):
        figure.savefig(os.path.join(FIGURE_DIR, f"{name}.{extension}"), dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"saved {FIGURE_DIR}/{name}.pdf")

def zero_line(axes: Axes, horizontal: bool = True) -> None:
    if horizontal:
        axes.axhline(0.0, color=BASELINE, linewidth=0.9, zorder=1)
    else:
        axes.axvline(0.0, color=BASELINE, linewidth=0.9, zorder=1)

def cumulative_returns(key: str, base: Series, dates: np.ndarray) -> None:
    figure, axes = plt.subplots(figsize=(6.5, 3.2))
    for start, end, label in CRISES:
        axes.axvspan(np.datetime64(start), np.datetime64(end), color=SHADE, zorder=0, linewidth=0)
        axes.text(np.datetime64(start), 0.995, f" {label}", transform=axes.get_xaxis_transform(),
                  va="top", ha="left", color=MUTED, fontsize=8)
    x = dates[base.start:]
    handles = []
    for fee, weight, alpha, label in ((0.0, 2.0, 1.0, "No fees"), (TRADE_FEE, 1.4, 0.45, "5bp fees")):
        values = np.cumprod(1.0 + net_returns(base, fee))
        handles += axes.plot(x, values, color=COLORS[key], linewidth=weight, alpha=alpha, label=label, zorder=3)
        axes.annotate(f"{values[-1]:.2f}", (x[-1], values[-1]), xytext=(4, 0), textcoords="offset points",
                      va="center", color=SECONDARY, fontsize=8)
    axes.axhline(1.0, color=BASELINE, linewidth=0.9, zorder=1)
    axes.set_ylabel("Value of $1 invested")
    axes.set_title(f"{STRATEGIES[key]}: cumulative returns, 2002-2023")
    axes.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    save(figure, f"cumulative_returns_{key}")

def snapback_cumulative(series: dict[str, Series], dates: np.ndarray) -> None:
    # Long-only default grows with the market, so compare it with the benchmark it rebalances against, on a log scale
    figure, axes = plt.subplots(figsize=(6.5, 3.2))
    for start, end, label in CRISES:
        axes.axvspan(np.datetime64(start), np.datetime64(end), color=SHADE, zorder=0, linewidth=0)
        axes.text(np.datetime64(start), 0.995, f" {label}", transform=axes.get_xaxis_transform(),
                  va="top", ha="left", color=MUTED, fontsize=8)
    start = series["snap|base|"].start
    lines = (
        ("snap|base|", "Long only, 5bp fees", COLORS["snap"], 2.0, 1.0, "-"),
        ("snaph|base|", "Fully hedged, 5bp fees", COLORS["snap"], 1.4, 1.0, (0, (4, 2))),
        ("bench|base|", "Equal-weight benchmark, 5bp fees", MUTED, 1.4, 1.0, "-"),
    )
    handles = []
    for name, label, color, weight, alpha, style in lines:
        s = series[name]
        # Measured from the snapback's first trade so all three lines start at $1
        values = np.cumprod(1.0 + net_returns(s, TRADE_FEE, np.arange(len(s.growth)) >= start))
        x = dates[max(start, s.start):]
        handles += axes.plot(x, values, color=color, linewidth=weight, alpha=alpha, linestyle=style, label=label,
                             zorder=3)
        axes.annotate(f"{values[-1]:.1f}", (x[-1], values[-1]), xytext=(4, 0), textcoords="offset points",
                      va="center", color=SECONDARY, fontsize=8)
    axes.set_yscale("log")
    axes.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:g}"))
    axes.axhline(1.0, color=BASELINE, linewidth=0.9, zorder=1)
    axes.set_ylabel("Value of $1 invested (log scale)")
    axes.set_title(f"{STRATEGIES['snap']}: cumulative returns, 2002-2023")
    axes.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    save(figure, "cumulative_returns_snap")

def annual_returns(key: str, base: Series, dates: np.ndarray, hedged: Series | None = None) -> None:
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    active_years = np.unique(years[base.start:])
    values = np.array([np.prod(1.0 + net_returns(base, 0.0, years == year)) - 1.0 for year in active_years])
    figure, axes = plt.subplots(figsize=(6.5, 2.8))
    if hedged is None:
        axes.bar(active_years, values * 100, width=0.6, color=COLORS[key], zorder=3)
    else:
        hedged_values = np.array([np.prod(1.0 + net_returns(hedged, 0.0, years == year)) - 1.0
                                  for year in active_years])
        axes.bar(active_years - 0.19, values * 100, width=0.36, color=COLORS[key], label=LONG_LABEL, zorder=3)
        axes.bar(active_years + 0.19, hedged_values * 100, width=0.36, color="white", edgecolor=COLORS[key],
                 hatch="/////", linewidth=0.8, label=HEDGED_LABEL, zorder=3)
        axes.legend(loc="upper right", fontsize=8)
    zero_line(axes)
    axes.grid(axis="x", visible=False)
    axes.set_xticks(active_years)
    axes.set_xticklabels([f"'{year % 100:02d}" for year in active_years])
    axes.set_ylabel("Annual return, no fees (%)")
    axes.set_title(f"{STRATEGIES[key]}: annual returns")
    save(figure, f"annual_returns_{key}")

def calibration(key: str, results: dict[str, Any], log_prices: np.ndarray) -> None:
    figure, axes = plt.subplots(figsize=(4.8, 3.0))
    if key == "tb":
        decile_means, _ = alpha_calibration(results, log_prices, 5)
        axes.bar(np.arange(1, 11), decile_means * 1e4, width=0.6, color=COLORS[key], zorder=3)
        axes.set_xticks(np.arange(1, 11))
        axes.set_xlabel("Signal decile (1 = most shorted, 10 = most bought)")
        axes.set_ylabel("Next 5-day excess return (bp)")
    elif key == "snap":
        buckets, _ = snapback_calibration(snapback_scores(results["snapback_residuals"]), log_prices, 5)
        axes.bar(np.arange(len(buckets)), [np.mean(bucket) * 1e4 for bucket in buckets], width=0.6,
                 color=COLORS[key], zorder=3)
        axes.set_xticks(np.arange(len(buckets)))
        axes.set_xticklabels([label.replace("z ", "").replace(" to ", " to\n") for label in Z_LABELS], fontsize=7.5)
        axes.set_xlabel("Snapback score (buys at or below -2)")
        axes.set_ylabel("Next 5-day excess return (bp)")
    else:
        buckets, _ = pair_calibration(results, log_prices, 1)
        axes.bar(np.arange(len(buckets)), [np.mean(bucket) * 1e4 for bucket in buckets], width=0.6,
                 color=COLORS[key], zorder=3)
        axes.set_xticks(np.arange(len(buckets)))
        axes.set_xticklabels([label.replace("z ", "").replace(" to ", " to\n") for label in Z_LABELS], fontsize=7.5)
        axes.set_xlabel("Spread z-score (buy below -2, sell above 2)")
        axes.set_ylabel("Next-day change in log price ratio (bp)")
    zero_line(axes)
    axes.grid(axis="x", visible=False)
    axes.set_title(f"{STRATEGIES[key]}: calibration")
    save(figure, f"calibration_{key}")

def fee_sensitivity(key: str, base: Series, hedged: Series | None = None) -> None:
    fees = np.linspace(0.0, 0.0030, 61)
    figure, axes = plt.subplots(figsize=(4.8, 3.0))
    axes.axvline(TRADE_FEE * 1e4, color=BASELINE, linewidth=0.9, zorder=1)
    axes.text(TRADE_FEE * 1e4, 0.99, " 5bp assumption", transform=axes.get_xaxis_transform(),
              va="top", ha="left", color=MUTED, fontsize=8)
    lines = [(base, "-", True, LONG_LABEL)] + ([] if hedged is None else [(hedged, (0, (4, 2)), False, HEDGED_LABEL)])
    for s, style, filled, label in lines:
        axes.plot(fees * 1e4, [sharpe(net_returns(s, fee)) for fee in fees], color=COLORS[key], linewidth=2,
                  linestyle=style, label=label, zorder=3)
        active = np.arange(len(s.growth)) >= s.start
        break_even = np.mean(s.growth[active] - 1.0) / np.mean(s.growth[active] * s.turnover[active])
        axes.plot(break_even * 1e4, 0.0, "o", markersize=7, color=COLORS[key] if filled else "white",
                  markeredgecolor="white" if filled else COLORS[key], markeredgewidth=1.5, zorder=4)
        axes.annotate(f"break-even {break_even * 1e4:.0f}bp", (break_even * 1e4, 0.0),
                      xytext=(6, 6) if filled else (-6, -8), textcoords="offset points",
                      ha="left" if filled else "right", va="bottom" if filled else "top", color=SECONDARY, fontsize=8)
    if hedged is not None:
        axes.legend(loc="upper right", fontsize=8)
    zero_line(axes)
    axes.set_xlabel("Transaction cost per dollar traded (bp)")
    axes.set_ylabel("Sharpe ratio")
    axes.set_title(f"{STRATEGIES[key]}: Sharpe ratio vs. costs")
    save(figure, f"fee_sensitivity_{key}")

def parameter_sensitivity(key: str, series: dict[str, Series], dates: np.ndarray) -> None:
    sensitivity = {name: s for name, s in series.items() if name.split("|")[0] in ("tb", "pairs", "snap", "snaph")}
    start = common_starts(sensitivity)[key]
    common = np.arange(len(dates)) >= start
    rows = []
    for name, s in sensitivity.items():
        strategy, param, value = name.split("|")
        if strategy != key:
            continue
        label = "Default" if param == "base" else f"{param[0].upper()}{param[1:]} = {value.replace('unhedged', 'long only')}"
        order = float(value) if value.replace(".", "").isdigit() else 0.0
        rows.append((param != "base", param, order, label, sharpe(net_returns(s, TRADE_FEE, common)), name))
    # Reversed so the first row is drawn at the top
    ordered = sorted(rows, key=lambda row: row[:3])[::-1]
    default = next(row[4] for row in ordered if not row[0])
    positions = np.arange(len(ordered))

    figure, axes = plt.subplots(figsize=(5.0, 0.24 * len(ordered) + 1.1))
    axes.axvline(default, color=COLORS[key], linewidth=0.9, alpha=0.4, zorder=1)
    axes.plot([row[4] for row in ordered], positions, "o", markersize=6.5, color=COLORS[key],
              markeredgecolor="white", markeredgewidth=1.5, zorder=3, label=LONG_LABEL if key == "snap" else None)
    hedged_values = []
    if key == "snap":
        # Same parameter on the hedged version, hedge mixes have no hedged counterpart
        for row in ordered:
            name = row[5].replace("snap|", "snaph|", 1)
            hedged_values.append(sharpe(net_returns(series[name], TRADE_FEE, common)) if name in series else np.nan)
        axes.plot(hedged_values, positions, "o", markersize=6.5, color="white", markeredgecolor=COLORS[key],
                  markeredgewidth=1.5, zorder=3, label=HEDGED_LABEL)
        axes.legend(loc="upper right", fontsize=8)
    axes.set_yticks(positions)
    axes.set_yticklabels([row[3] for row in ordered], fontsize=8)
    for tick, row in zip(axes.get_yticklabels(), ordered):
        if not row[0]:
            tick.set_fontweight("bold")
            tick.set_color(INK)
    axes.grid(axis="y", visible=False)
    axes.set_ylim(-0.7, len(ordered) - 0.3)
    values = [row[4] for row in ordered] + [value for value in hedged_values if np.isfinite(value)]
    axes.set_xlim(min(0.0, min(values) - 0.1), max(1.5, max(values) + 0.05))
    zero_line(axes, horizontal=False)
    period = dates[start].astype("datetime64[M]").item().strftime("%b %Y")
    axes.set_xlabel(f"Sharpe ratio with 5bp fees ({period}-2023)")
    axes.set_title(f"{STRATEGIES[key]}: parameter sensitivity")
    save(figure, f"parameter_sensitivity_{key}")

def draw_k_lines(axes: Axes, key: str, series: dict[str, Series], lines: tuple) -> None:
    for name, label, filled in lines:
        # Strategies that never trade at some K (hedged snapback at K = 20) leave a gap
        values = [sharpe(net_returns(series[f"k{k}|{name}"], 0.0)) for k in K_VALUES]
        axes.plot(K_VALUES, values, color=COLORS[key], linewidth=2 if filled else 1.2, alpha=1.0 if filled else 0.7,
                  marker="o", markersize=6, markerfacecolor=COLORS[key] if filled else "white",
                  markeredgecolor="white" if filled else COLORS[key], markeredgewidth=1.5 if not filled else 0,
                  label=label, zorder=3)
    zero_line(axes)
    axes.set_ylim(bottom=0.0)
    axes.set_xticks(K_VALUES)
    axes.set_xlabel("Number of factors K")

def k_sweep(key: str, series: dict[str, Series]) -> None:
    if key == "snap":
        # One panel per version so each keeps the out-of-sample vs in-sample comparison of the other strategies
        figure, panels = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=True)
        for axes, (name, title) in zip(panels, (("snap", LONG_LABEL), ("snaph", HEDGED_LABEL))):
            draw_k_lines(axes, key, series, ((f"{name}|oos", "Out-of-sample (rolling estimates)", True),
                                             (f"{name}|is", "In-sample (look-ahead)", False)))
            axes.set_title(title, fontsize=9, fontweight="normal", color=SECONDARY)
        panels[0].set_ylabel("Sharpe ratio, no fees")
        panels[0].set_ylim(0.0, 2.1)
        panels[0].legend(loc="lower left", fontsize=8)
        figure.suptitle(f"{STRATEGIES[key]}: Sharpe ratio by K", x=0.02, ha="left", fontsize=10, fontweight="bold",
                        color=INK)
        figure.tight_layout()
        save(figure, f"k_sweep_{key}")
        return
    lines = (
        (f"{key}|oos", "Out-of-sample (rolling estimates)", True),
        ("tb follow|is" if key == "tb" else f"{key}|is",
         "In-sample, following alpha (look-ahead)" if key == "tb" else "In-sample (look-ahead)", False),
    )
    figure, axes = plt.subplots(figsize=(4.8, 3.0))
    draw_k_lines(axes, key, series, lines)
    axes.set_ylabel("Sharpe ratio, no fees")
    axes.set_title(f"{STRATEGIES[key]}: Sharpe ratio by K")
    axes.legend(loc="lower right", fontsize=8)
    save(figure, f"k_sweep_{key}")

def development(series: dict[str, Series]) -> None:
    # Table 1: each step towards contrarian Treynor-Black, the line between the dots is the cost of fees
    labels = list(DEVELOPMENT)[::-1]
    positions = np.arange(len(labels))
    figure, axes = plt.subplots(figsize=(6.0, 2.6))
    for position, label in zip(positions, labels):
        s = series[DEVELOPMENT[label]]
        gross, net = sharpe(net_returns(s, 0.0)), sharpe(net_returns(s, TRADE_FEE))
        axes.plot([net, gross], [position, position], color=COLORS["tb"], linewidth=1.2, alpha=0.5, zorder=2)
        axes.plot(gross, position, "o", markersize=7, color="white", markeredgecolor=COLORS["tb"],
                  markeredgewidth=1.5, zorder=3, label="No fees" if position == 0 else None)
        axes.plot(net, position, "o", markersize=7, color=COLORS["tb"], markeredgecolor="white",
                  markeredgewidth=1.5, zorder=4, label="5bp fees" if position == 0 else None)
        axes.annotate(f"{net:.2f}", (min(net, gross), position), xytext=(-8, 0), textcoords="offset points",
                      ha="right", va="center", color=SECONDARY, fontsize=8)
    axes.set_yticks(positions)
    axes.set_yticklabels(labels, fontsize=8)
    axes.get_yticklabels()[0].set_fontweight("bold")
    axes.get_yticklabels()[0].set_color(INK)
    axes.grid(axis="y", visible=False)
    axes.set_xlim(-1.8, 1.6)
    axes.set_ylim(-0.6, len(labels) - 0.4)
    zero_line(axes, horizontal=False)
    axes.set_xlabel("Sharpe ratio, 2002-2023")
    axes.set_title("Development of contrarian Treynor-Black")
    period_legend(axes, ("No fees", "5bp fees"), "lower left")
    save(figure, "development_tb")

def snapback_variants(series: dict[str, Series], market: np.ndarray) -> None:
    # Table 2: Sharpe ratio against market beta, with the two market-neutral strategies for reference
    points = (
        ("bench|base|", "Equal-weight benchmark", MUTED, True, (6, -12)),
        ("snap|base|", "Long only (default)", COLORS["snap"], True, (6, 4)),
        ("snap|score threshold|-2.5", "Stricter cutoff", COLORS["snap"], True, (6, -4)),
        ("snap|hedge mix|70% unhedged", "70% long only", COLORS["snap"], True, (6, 2)),
        ("snap|hedge mix|50% unhedged", "50% long only", COLORS["snap"], True, (6, 2)),
        ("snap|hedge mix|30% unhedged", "30% long only", COLORS["snap"], True, (6, 2)),
        ("snap|hedge mix|dynamic", "Dynamic mix", COLORS["snap"], True, (-8, 8)),
        ("snaph|base|", "Fully hedged (long-short)", COLORS["snap"], False, (8, -8)),
        ("cmp|projection hedge|", "Earlier hedge (projection)", MUTED, False, (10, 0)),
        ("tb|base|", "Treynor-Black", COLORS["tb"], True, (6, 11)),
        ("pairs|base|", "Pairs trading", COLORS["pairs"], True, (6, -4)),
    )
    figure, axes = plt.subplots(figsize=(5.6, 3.4))
    for name, label, color, filled, offset in points:
        s = series[name]
        beta, value = market_beta(s, market), sharpe(net_returns(s, TRADE_FEE))
        axes.plot(beta, value, "o", markersize=8, color=color if filled else "white",
                  markeredgecolor="white" if filled else color, markeredgewidth=1.5, zorder=3)
        axes.annotate(label, (beta, value), xytext=offset, textcoords="offset points",
                      ha="left" if offset[0] > 0 else "right", va="center", color=SECONDARY, fontsize=7.5)
    axes.set_xlim(-0.1, 1.35)
    axes.set_ylim(0.5, 1.3)
    axes.set_xlabel("Market beta (vs. equal-weighted universe)")
    axes.set_ylabel("Sharpe ratio, 5bp fees")
    axes.set_title("Residual snapback variants: Sharpe ratio vs. market exposure")
    save(figure, "snapback_variants")

def regime_stability(series: dict[str, Series], dates: np.ndarray) -> None:
    # Table 4: how concentrated profits are, and how the Sharpe ratio changed between the two halves
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    first_half = years <= 2012
    keys = (("tb", "tb|base|"), ("pairs", "pairs|base|"), ("snap", "snap|base|"), ("snaph", "snaph|base|"))
    labels = ("Treynor-Black", "Pairs trading", f"Snapback, {LONG_LABEL.lower()}", f"Snapback, {HEDGED_LABEL.lower()}")
    positions = np.arange(len(keys))[::-1]
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.4, 2.6), sharey=True)
    for position, (key, name), label in zip(positions, keys, labels):
        s = series[name]
        color, filled = COLORS[key[:4] if key.startswith("snap") else key], key != "snaph"
        log_pnl = {year: float(np.sum(np.log1p(net_returns(s, 0.0, years == year)))) for year in np.unique(years[s.start:])}
        share = sum(sorted(log_pnl.values())[-2:]) / sum(log_pnl.values())
        left.barh(position, share * 100, height=0.55, color=color if filled else "white", edgecolor=color,
                  hatch=None if filled else "/////", linewidth=0.8, zorder=3)
        left.annotate(f"{share:.1%}", (share * 100, position), xytext=(4, 0), textcoords="offset points",
                      va="center", color=SECONDARY, fontsize=8)
        early, late = sharpe(net_returns(s, TRADE_FEE, first_half)), sharpe(net_returns(s, TRADE_FEE, ~first_half))
        right.annotate("", (late, position), (early, position),
                       arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": 1.4, "shrinkA": 4, "shrinkB": 4})
        right.plot(early, position, "o", markersize=7, color="white", markeredgecolor=color, markeredgewidth=1.5,
                   zorder=3, label="2002-2012" if position == positions[0] else None)
        right.plot(late, position, "o", markersize=7, color=color, markeredgecolor="white", markeredgewidth=1.5,
                   zorder=3, label="2013-2023" if position == positions[0] else None)
    left.set_yticks(positions)
    left.set_yticklabels(labels, fontsize=8)
    for axes in (left, right):
        axes.grid(axis="y", visible=False)
    left.set_xlim(0, 55)
    left.set_xlabel("Share of profit from best two years (%)")
    right.set_xlim(0, 1.6)
    right.set_xlabel("Sharpe ratio, 5bp fees")
    period_legend(right, ("2002-2012", "2013-2023"), "lower right")
    left.set_title("Concentration", fontsize=9, fontweight="normal", color=SECONDARY)
    right.set_title("First vs. second half", fontsize=9, fontweight="normal", color=SECONDARY)
    figure.suptitle("Stability of returns", x=0.02, ha="left", fontsize=10, fontweight="bold", color=INK)
    figure.tight_layout()
    save(figure, "regime_stability")

def calibration_slopes(results: dict[str, Any], log_prices: np.ndarray) -> None:
    # Table 9: significance of each signal by horizon (slopes are in different units, t-stats are comparable)
    horizons = (1, 5, 20)
    scores = snapback_scores(results["snapback_residuals"])
    signals = {
        "tb": lambda h: alpha_calibration(results, log_prices, h)[1],
        "pairs": lambda h: pair_calibration(results, log_prices, h)[1],
        "snap": lambda h: snapback_calibration(scores, log_prices, h)[1],
    }
    figure, axes = plt.subplots(figsize=(4.8, 3.0))
    axes.axhline(2.0, color=BASELINE, linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
    axes.text(0.85, 2.0, "t = 2", ha="left", va="bottom", color=MUTED, fontsize=8)
    for key, slopes in signals.items():
        values = [newey_west_t(slopes(h), h) for h in horizons]
        axes.plot(horizons, values, color=COLORS[key], linewidth=2, marker="o", markersize=6,
                  markeredgecolor="white", markeredgewidth=1.5, zorder=3)
        axes.annotate(STRATEGIES[key], (horizons[-1], values[-1]), xytext=(6, 0), textcoords="offset points",
                      va="center", color=SECONDARY, fontsize=8)
    axes.set_xscale("log")
    axes.set_xticks(horizons)
    axes.set_xticklabels([f"{h}d" for h in horizons])
    axes.minorticks_off()
    axes.set_xlim(0.8, 60)
    axes.set_ylim(bottom=0.0)
    axes.set_xlabel("Horizon (trading days)")
    axes.set_ylabel("Newey-West t-statistic of slope")
    axes.set_title("Calibration: significance by horizon")
    save(figure, "calibration_slopes")

def holdout(series: dict[str, Series], dates: np.ndarray) -> None:
    # Table 10: Sharpe ratio on 2002-2017 (hollow) and on the 2018-2023 holdout (filled)
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    selection, test = years < HOLDOUT_START, years >= HOLDOUT_START
    sensitivity = {name: s for name, s in series.items() if name.split("|")[0] in ("tb", "pairs", "snap", "snaph")}
    starts = common_starts(sensitivity)
    groups = (("tb", "Treynor-Black"), ("pairs", "Pairs trading"), ("snap", f"Snapback, {LONG_LABEL.lower()}"),
              ("snaph", f"Snapback, {HEDGED_LABEL.lower()}"))
    rows = []
    for key, group in groups:
        common = (np.arange(len(dates)) >= starts[key]) & selection
        candidates = {name: s for name, s in sensitivity.items() if name.startswith(f"{key}|")}
        best = max((n for n in candidates if "|hedge mix|" not in n),
                   key=lambda n: sharpe(net_returns(candidates[n], TRADE_FEE, common)))
        best_k = max(K_VALUES, key=lambda k: sharpe(net_returns(series[f"k{k}|{key}|oos"], TRADE_FEE, selection)))
        entries = [("Default", series[f"{key}|base|"], common),
                   (f"Best setting ({best.split('|')[1]} {best.split('|')[2]})", candidates[best], common),
                   (f"Best K (K = {best_k})", series[f"k{best_k}|{key}|oos"], selection)]
        if key == "snap":
            mixes = {n: s for n, s in candidates.items() if "|hedge mix|" in n}
            mix = max(mixes, key=lambda n: sharpe(net_returns(mixes[n], TRADE_FEE, common)))
            entries.append((f"Best hedge mix ({mix.split('|')[2].replace('unhedged', 'long only')})", mixes[mix], common))
        for index, (label, s, period) in enumerate(entries):
            rows.append((group if index == 0 else "", label, key,
                         sharpe(net_returns(s, TRADE_FEE, period)), sharpe(net_returns(s, TRADE_FEE, test))))
    positions = np.arange(len(rows))[::-1]
    figure, axes = plt.subplots(figsize=(6.2, 0.27 * len(rows) + 1.2))
    for position, (_, _, key, before, after) in zip(positions, rows):
        color = COLORS["snap" if key == "snaph" else key]
        axes.plot([before, after], [position, position], color=color, linewidth=1.2, alpha=0.5, zorder=2)
        axes.plot(before, position, "o", markersize=7, color="white", markeredgecolor=color, markeredgewidth=1.5,
                  zorder=3, label=f"2002-{HOLDOUT_START - 1}" if position == positions[0] else None)
        axes.plot(after, position, "o", markersize=7, color=color, markeredgecolor="white", markeredgewidth=1.5,
                  zorder=4, label=f"{HOLDOUT_START}-2023 (holdout)" if position == positions[0] else None)
    axes.set_yticks(positions)
    axes.set_yticklabels([f"{group}: {label}" if group else label for group, label, *_ in rows], fontsize=7.5)
    for tick, row in zip(axes.get_yticklabels(), rows):
        if row[0]:
            tick.set_color(INK)
    axes.grid(axis="y", visible=False)
    axes.set_ylim(-0.7, len(rows) - 0.3)
    axes.set_xlim(0.0, 1.8)
    axes.set_xlabel("Sharpe ratio, 5bp fees")
    axes.set_title("Selection period vs. holdout")
    period_legend(axes, (f"2002-{HOLDOUT_START - 1}", f"{HOLDOUT_START}-2023 (holdout)"), "upper right")
    save(figure, "holdout")

def residual_means(results: dict[str, Any], dates: np.ndarray) -> None:
    # Is the lagged factor model unbiased? Each stock's trailing 60-day mean residual, summarized across stocks
    days, means = rolling_residual_means(results["snapback_residuals"])
    x = dates[days]
    low, high = np.nanpercentile(means, 10, axis=1) * 1e4, np.nanpercentile(means, 90, axis=1) * 1e4
    figure, axes = plt.subplots(figsize=(6.5, 3.0))
    axes.fill_between(x, low, high, color=COLORS["snap"], alpha=0.18, linewidth=0, label="10th-90th percentile of stocks")
    axes.plot(x, np.nanmean(means, axis=1) * 1e4, color=COLORS["snap"], linewidth=1.6, label="Average across stocks")
    zero_line(axes)
    axes.set_ylabel("60-day mean residual (bp/day)")
    axes.set_title("Residual snapback: 60-day mean residual of the lagged factor model")
    axes.legend(loc="upper left", fontsize=8, ncol=2)
    axes.set_ylim(-80, 80)
    save(figure, "residual_means")

def hedge_allocation(daily_returns: data.DailyReturns) -> None:
    # Dynamic mix: share in the hedged portfolio at each weekly rebalance, raw percentile vs. smoothed
    weekly = weekly_rebalance_dates(daily_returns.dates)
    raw = snapback("lagged", daily_returns, dynamic=True, projection=True)._original_fraction_by_date
    smoothed = snapback("lagged", daily_returns, dynamic=True)._original_fraction_by_date
    figure, axes = plt.subplots(figsize=(6.5, 3.0))
    for allocation, label, width, alpha in ((raw, "Raw percentile (earlier version)", 0.8, 0.45),
                                            (smoothed, "Smoothed (126 pseudo-readings each side)", 1.6, 1.0)):
        x = np.array([date for date in weekly if date in allocation])
        axes.plot(x, [100 * (1.0 - allocation[date]) for date in x], color=COLORS["snap"], linewidth=width,
                  alpha=alpha, label=label)
    axes.axvline(daily_returns.dates[500], color=BASELINE, linewidth=0.9, linestyle=(0, (4, 2)))
    axes.text(daily_returns.dates[500], 103, " trading starts", color=MUTED, fontsize=7.5, va="bottom")
    axes.set_ylim(0, 128)
    axes.set_yticks([0, 25, 50, 75, 100])
    axes.set_ylabel("Share in hedged portfolio (%)")
    axes.set_title("Residual snapback: dynamic volatility mix")
    axes.legend(loc="upper right", fontsize=8, ncol=2)
    save(figure, "hedge_allocation")

def main() -> None:
    daily_returns = data.load_daily_returns()
    results = load_results()
    series: dict[str, Series] = results["series"]
    log_prices = log_price_paths(daily_returns.returns)
    for key in STRATEGIES:
        base = series[f"{key}|base|"]
        hedged = series["snaph|base|"] if key == "snap" else None
        if key == "snap":
            snapback_cumulative(series, daily_returns.dates)
        else:
            cumulative_returns(key, base, daily_returns.dates)
        annual_returns(key, base, daily_returns.dates, hedged)
        parameter_sensitivity(key, series, daily_returns.dates)
        fee_sensitivity(key, base, hedged)
        calibration(key, results, log_prices)
        k_sweep(key, series)
    development(series)
    snapback_variants(series, daily_returns.returns.mean(axis=1))
    regime_stability(series, daily_returns.dates)
    calibration_slopes(results, log_prices)
    holdout(series, daily_returns.dates)
    residual_means(results, daily_returns.dates)
    hedge_allocation(daily_returns)

if __name__ == "__main__":
    main()
