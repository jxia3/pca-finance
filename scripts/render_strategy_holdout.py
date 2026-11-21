"""Render saved holdout outputs. Does not import or execute trading models."""

import hashlib
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEST = Path("report/holdout_results")
NAMES = ("Treynor--Black", "PCA pairs")
DISPLAY_NAMES = {
    "Treynor--Black": "TB (factor-hedged, 60d regression)",
    "PCA pairs": "PCA pairs (20d spread)",
}
TABLE_NAMES = {"Treynor--Black": "Hedged TB, 60d", "PCA pairs": "PCA pairs, 20d"}
MODES = ("constant_mark",)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render():
    receipt = json.loads((DEST / "run_receipt.json").read_text())
    if receipt["status"] != "complete":
        raise RuntimeError("No complete evaluation to render.")
    for filename, digest in receipt["output_hashes"].items():
        if sha(DEST / filename) != digest:
            raise RuntimeError(f"Saved results changed: {filename}")
    frames = [pd.read_csv(DEST / f"{m}_daily.csv") for m in MODES]
    rows = pd.concat(frames)
    summary = json.loads((DEST / "summary.json").read_text())
    figures = Path("report/figures")
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    colors = ["#275e80", "#ad593a"]
    fig, axis = plt.subplots(figsize=(8, 3.6), layout="constrained")
    axes = [axis]
    for ax, mode, title in zip(
        axes,
        MODES,
        ("Holdout performance of the selected strategies",),
    ):
        for strategy, color in zip(NAMES, colors):
            g = rows[(rows["mode"] == mode) & (rows.strategy == strategy)]
            dates = pd.to_datetime(g.date)
            plot_dates = pd.DatetimeIndex(["2023-12-29"]).append(
                pd.DatetimeIndex(dates)
            )
            ax.plot(
                plot_dates,
                np.r_[0, (g.value - 1) * 100],
                color=color,
                label=DISPLAY_NAMES[strategy],
            )
        ax.axhline(0, color="#999999", lw=0.6)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.2)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    axes[0].set_ylabel("Net cumulative return (%)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(figures / "holdout_performance.png", dpi=220)
    plt.close(fig)
    write_section(summary)
    with zipfile.ZipFile(
        "report/holdout_overleaf.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as bundle:
        for name in (
            "holdout_results.tex",
            "figures/holdout_performance.png",
        ):
            bundle.write(Path("report") / name, name)
    print(
        pd.DataFrame(summary)[
            [
                "mode",
                "strategy",
                "net_return",
                "sharpe",
                "drawdown",
                "final_frozen_gross",
            ]
        ].to_string(index=False)
    )


def write_section(summary):
    """Results-only section. Existing report content is never rewritten."""
    table = []
    for s in summary:
        table.append(
            f"{TABLE_NAMES[s['strategy']]} & {s['net_return']:.2%} & {s['cagr']:.2%} & {s['sharpe']:.2f} & {s['drawdown']:.2%} \\\\".replace(
                "%", r"\%"
            )
        )
    replacements = {}
    for s in summary:
        if s["mode"] == "constant_mark":
            prefix = "TB" if s["strategy"] == NAMES[0] else "PAIRS"
            replacements[prefix + "SHARPE"] = f"{s['sharpe']:.2f}"
            for token, key in [('RETURN','net_return'), ('FEES','fees_per_initial_dollar'),
                               ('TURNOVER','mean_turnover'), ('GROSS','mean_gross')]:
                replacements[prefix + token] = f"{s[key]:.2%}".replace('%', r'\%')
            replacements[prefix + 'PNL'] = f"{s['net_return'] + s['fees_per_initial_dollar']:.2%}".replace('%', r'\%')
            replacements[prefix + "AFFECTEDDAYS"] = str(s["affected_days"])
            for token, key in [
                ("MAXFROZEN", "max_frozen_gross"),
                ("ENDFROZEN", "final_frozen_gross"),
                ("LONGFROZEN", "final_frozen_long"),
                ("SHORTFROZEN", "final_frozen_short"),
            ]:
                replacements[prefix + token] = f"{s[key]:.2%}".replace("%", r"\%")
    section = Path("report/holdout_results.template.tex").read_text(encoding="utf-8")
    section = section.replace("TABLEROWS", "\n".join(table))
    for token, value in replacements.items():
        section = section.replace(token, value)
    Path("report/holdout_results.tex").write_text(section, encoding="utf-8")


if __name__ == "__main__":
    render()
