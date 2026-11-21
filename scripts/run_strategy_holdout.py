"""Fixed-parameter holdout evaluation, with an immutable run receipt.

Plan and commands (repository root):
  python -m scripts.run_strategy_holdout prepare
  python -m unittest discover -s tests -p test_holdout.py
  python -m scripts.run_strategy_holdout run
  python -m scripts.render_strategy_holdout

Prepare records parameters, input/code hashes and the gap convention without
evaluating returns. Run refuses to overwrite a previous run. Render regenerates
figures and the copy-pastable LaTeX section from saved results, without retrading.
Only the selected strategies and gap-aware accounting are evaluated.
"""

import os

for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import hashlib
import json
import platform
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from src.data import DailyReturns
from src.engine import Engine, ModelEntry
from src.holdout.accounting import HoldoutSimulator
from src.holdout.models import HoldoutModels

DEST = Path("report/holdout_results")
CONFIG = {
    "k": 5,
    "pca_window": 500,
    "pca_refit": 5,
    "regression_window": 60,
    "regression_refit": 1,
    "score_cap": 3.0,
    "gross": 1.0,
    "fee": 0.0005,
    "pairs": 50,
    "formation_window": 250,
    "formation_interval": 21,
    "spread_window": 20,
    "entry": 2.0,
    "exit": 0.5,
}
NAMES = ("Treynor--Black", "PCA pairs")
MODES = ("constant_mark",)
SELECTED_CONFIG = {**CONFIG, "regression_window": 60, "spread_window": 20}


def settings(profile):
    """Each profile has an independent immutable receipt and fixed parameters."""
    if profile == "selected":
        return SELECTED_CONFIG.copy(), DEST
    raise ValueError(f"Unknown profile: {profile}")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_files():
    return [
        Path(__file__),
        Path("src/engine.py"),
        Path("src/models/pca.py"),
        Path("src/modules/simulator.py"),
        Path("src/data.py"),
        Path("src/models/model.py"),
        Path("src/strategies/strategy.py"),
        Path("src/modules/module.py"),
        *sorted(Path("src/holdout_strategies").glob("*.py")),
        *sorted(Path("src/holdout").glob("*.py")),
    ]


def hashes():
    return {
        str(p.relative_to(Path.cwd()) if p.is_absolute() else p).replace(
            "\\", "/"
        ): sha(p)
        for p in source_files()
    }


def prepare(profile="selected"):
    config, dest = settings(profile)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "run_receipt.json").exists():
        raise RuntimeError("A run already exists. Use render, not another evaluation.")
    plan = {
        "parameters": config,
        "source_branch": "origin/kayla",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "origin/kayla"], text=True
        ).strip(),
        "current_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "code_hashes": hashes(),
        "input_hashes": {
            p: sha(p)
            for p in (
                "materials/daily_returns.csv",
                "holdout/reconstruction/holdout_daily_returns.csv",
                "holdout/reconstruction/holdout_tickers.csv",
                "holdout/reconstruction/changes.csv",
                "holdout/reconstruction/reviewed_cutoffs.csv",
                "scripts/reconstruct_holdout.py",
                "holdout/source_evidence.zip",
            )
        },
        "modes": list(MODES),
        "start": "2024-01-02",
        "end": "2026-11-17",
        "initialization": "500 original observations for warm-up; start holdout in cash; first trades at first holdout close.",
        "eligibility": "500 consecutive finite returns greater than -1, positive variance; current and past data only.",
        "gap_accounting": "Missing returns mark held positions unchanged. Trading is prohibited on missing dates. No advance liquidation inferred from future gaps.",
        "resumption": "Apply the first observed one-day return to the carried mark. Missing-interval returns are unrecovered. If ineligible, close at this proxy mark.",
        "frozen_capital": "Frozen positions retain signed dollar values. Active gross budget is max(0,1-frozen gross). Whole-book neutrality is not guaranteed.",
        "membership": "Refit PCA/factor on membership change. Remove ineligible pairs immediately, replace only at scheduled formation. Idle pair budget remains cash.",
        "source_changes": "Only import paths differ in the isolated Kayla copies. Missing-data adaptation and accounting are in src/holdout.",
        "accounting": "Self-financing dollar trades, fees deducted from equity, frozen dollar holdings preserved. Flat initial and no forced final liquidation. Zero cash interest, no borrow cost.",
        "selection": (
            "Earlier selection-period winner: TB regression60, pairs spread20, report-recommended K5; one contrarian factor-neutral TB configuration."
            if profile == "selected"
            else "Report baseline defaults and K5."
        ),
        "profile": profile,
        "prior_evaluation": (
            "Corrected-data rerun. Earlier evaluations of this period were inspected. Strategy parameters remain fixed from the pre-existing report. Data repairs use price evidence, not strategy performance. Superseded run artifacts were removed at the user's request."
            if profile == "selected"
            else "None."
        ),
        "finality": "One gap-aware execution per selected strategy. No sweeps or complete-case comparison.",
    }
    (dest / "protocol.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print("Prepared frozen protocol. No strategy performance evaluated.")


def audit_gaps(frame):
    """Count contiguous absent-return spells, including leading and trailing gaps."""
    records = []
    for stock in frame:
        missing = frame[stock].isna().to_numpy()
        starts = np.flatnonzero(missing & ~np.r_[False, missing[:-1]])
        ends = np.flatnonzero(missing & ~np.r_[missing[1:], False])
        for a, b in zip(starts, ends):
            records.append(
                {
                    "stock_id": stock,
                    "start": str(frame.index[a]),
                    "end": str(frame.index[b]),
                    "length": int(b - a + 1),
                    "resumes": bool(b < len(frame) - 1),
                    "next_date": str(frame.index[b + 1]) if b < len(frame) - 1 else "",
                }
            )
    gaps = pd.DataFrame(
        records, columns=["stock_id", "start", "end", "length", "resumes", "next_date"]
    )
    gaps["resumes"] = gaps["resumes"].astype(bool)
    summary = {
        "missing_cells": int(frame.isna().sum().sum()),
        "stocks_with_gaps": int(frame.isna().any().sum()),
        "complete_stocks": int(frame.notna().all().sum()),
        "missing_spells": len(gaps),
        "resuming_spells": int(gaps.resumes.sum()),
        "resuming_stocks": int(gaps.loc[gaps.resumes, "stock_id"].nunique()),
        "resuming_missing_cells": int(gaps.loc[gaps.resumes, "length"].sum()),
        "resuming_median_days": float(gaps.loc[gaps.resumes, "length"].median()),
        "trailing_stocks": int((~gaps.resumes).sum()),
        "all_missing_stocks": int(frame.isna().all().sum()),
    }
    return gaps, summary


def audit_cutoffs(frame, summary, dest):
    """Separate documented exclusions from gaps with unresolved attribution."""
    cutoffs = pd.read_csv("holdout/reconstruction/reviewed_cutoffs.csv")
    excluded = 0
    for row in cutoffs.itertuples():
        values = frame.loc[frame.index >= row.exclude_from, str(row.permno)]
        if values.notna().any():
            raise ValueError(f"Observed returns after reviewed cutoff: {row.permno}")
        excluded += len(values)
    summary.update(
        documented_cutoffs=len(cutoffs),
        missing_at_or_after_cutoff=excluded,
        other_missing_cells=summary["missing_cells"] - excluded,
    )
    cutoffs.to_csv(dest / "reviewed_cutoffs.csv", index=False)


class Evaluate:
    def __init__(self, data, config):
        self.data = data
        self.simulators = [HoldoutSimulator(data, None, config["fee"]) for _ in NAMES]
        self.rows = []
        self.events = []

    def update(self, timestamp, returns, outputs):
        if timestamp < self.data.dates[0]:
            return
        decision = outputs["signals"][-1]
        raw = returns[-1]
        for index, sim in enumerate(self.simulators):
            previous = sim.positions.copy()
            old_value = sim.total_value
            marked = np.where(np.isfinite(raw), raw, 0.0)
            for stock in np.flatnonzero(
                (~np.isfinite(raw)) & (np.abs(previous) > 1e-12)
            ):
                self.events.append(
                    {
                        "date": str(timestamp),
                        "strategy": NAMES[index],
                        "stock_id": int(self.data.stock_ids[stock]),
                        "previous_weight": float(previous[stock]),
                        "marked_dollars_per_initial_dollar": float(
                            old_value * previous[stock]
                        ),
                    }
                )
            sim.update(timestamp, raw[None, :], {"target": decision.weights[index]})
            target = sim.positions
            frozen = np.where(~np.isfinite(raw), target, 0.0)
            j = sim.history_index - 1
            if sim.total_value <= 0 or not np.isfinite(sim.total_value):
                raise RuntimeError("Nonpositive or invalid equity.")
            assert np.allclose(sim.history.trades[j][~np.isfinite(raw)], 0, atol=1e-12)
            frozen_dollars = old_value * np.sum(np.abs(previous[~np.isfinite(raw)]))
            full_exposure = (
                float(np.max(np.abs(decision.betas @ target[decision.eligible])))
                if decision.betas is not None
                else 0.0
            )
            self.rows.append(
                {
                    "date": str(timestamp),
                    "strategy": NAMES[index],
                    "value": sim.total_value,
                    "net_return": sim.total_value / old_value - 1,
                    "gross_return": float(previous @ marked),
                    "fee": sim.history.fees[j],
                    "turnover": float(np.abs(sim.history.trades[j]).sum()),
                    "gross": float(np.abs(target).sum()),
                    "net": float(target.sum()),
                    "active_factor_exposure": full_exposure,
                    "frozen_gross": float(np.abs(frozen).sum()),
                    "frozen_dollars": float(frozen_dollars),
                    "frozen_long": float(np.maximum(frozen, 0).sum()),
                    "frozen_short": float(np.maximum(-frozen, 0).sum()),
                    "frozen_count": int(
                        np.sum((~np.isfinite(raw)) & (np.abs(previous) > 1e-12))
                    ),
                    "eligible": int(decision.eligible.sum()),
                    "selected_pairs": decision.selected_pairs,
                    "pca_refit": decision.refit,
                }
            )


def statistics(rows):
    result = []
    for (mode, strategy), group in rows.groupby(["mode", "strategy"], sort=False):
        r = group.net_return.to_numpy()
        values = np.r_[1.0, group.value.to_numpy()]
        vol = np.std(r, ddof=1)
        result.append(
            {
                "mode": mode,
                "strategy": strategy,
                "days": len(r),
                "net_return": float(values[-1] - 1),
                "cagr": float(values[-1] ** (252 / len(r)) - 1),
                "sharpe": float(np.mean(r) / vol * np.sqrt(252)) if vol > 0 else 0.0,
                "drawdown": float(np.min(values / np.maximum.accumulate(values) - 1)),
                "annual_volatility": float(vol * np.sqrt(252)),
                "mean_turnover": float(group.turnover.mean()),
                "fees_per_initial_dollar": float(group.fee.sum()),
                "mean_gross": float(group.gross.mean()),
                "mean_abs_net": float(group.net.abs().mean()),
                "max_frozen_gross": float(group.frozen_gross.max()),
                "final_frozen_gross": float(group.frozen_gross.iloc[-1]),
                "final_frozen_long": float(group.frozen_long.iloc[-1]),
                "final_frozen_short": float(group.frozen_short.iloc[-1]),
                "affected_days": int((group.frozen_count > 0).sum()),
                "min_eligible": int(group.eligible.min()),
                "max_eligible": int(group.eligible.max()),
            }
        )
    return result


def run(profile="selected"):
    config, dest = settings(profile)
    plan = json.loads((dest / "protocol.json").read_text())
    if plan["parameters"] != config or plan["code_hashes"] != hashes():
        raise RuntimeError("Protocol/code mismatch. Prepare before the first run.")
    for path, digest in plan["input_hashes"].items():
        if sha(path) != digest:
            raise RuntimeError(f"Input changed: {path}")
    receipt = {
        "started_utc": datetime.now(UTC).isoformat(),
        "status": "started",
        "protocol_sha256": sha(dest / "protocol.json"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
    }
    # Exclusive creation prevents a second execution, even after interruption.
    with (dest / "run_receipt.json").open("x", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)
    original = pd.read_csv("materials/daily_returns.csv", index_col=0)
    holdout = pd.read_csv("holdout/reconstruction/holdout_daily_returns.csv", index_col=0)
    assert original.columns.equals(holdout.columns)
    assert (
        str(holdout.index[0]) == plan["start"] and str(holdout.index[-1]) == plan["end"]
    )
    assert original.index[-1] < holdout.index[0]
    gaps, gap_summary = audit_gaps(holdout)
    audit_cutoffs(holdout, gap_summary, dest)
    gaps.to_csv(dest / "gap_spells.csv", index=False)
    (dest / "gap_summary.json").write_text(json.dumps(gap_summary, indent=2))
    combined = pd.concat([original.iloc[-config["pca_window"] :], holdout])
    data = DailyReturns(
        dates=combined.index.to_numpy(dtype="datetime64[D]"),
        stock_ids=combined.columns.to_numpy(dtype=int),
        returns=combined.to_numpy(),
    )
    evaluation_data = DailyReturns(
        dates=holdout.index.to_numpy(dtype="datetime64[D]"),
        stock_ids=data.stock_ids,
        returns=holdout.to_numpy(),
    )
    histories, frames, event_frames = {}, [], []
    for mode in MODES:
        allowed = np.ones(len(data.stock_ids), bool)
        model = HoldoutModels(config, allowed, plan["start"])
        evaluator = Evaluate(evaluation_data, config)
        Engine(data, [ModelEntry("signals", model)], [evaluator]).run()
        frame = pd.DataFrame(evaluator.rows)
        frame["mode"] = mode
        frame.to_csv(dest / f"{mode}_daily.csv", index=False)
        frames.append(frame)
        for index, sim in enumerate(evaluator.simulators):
            histories[f"{mode}_{index}_positions"] = sim.history.positions
            histories[f"{mode}_{index}_trades"] = sim.history.trades
        if evaluator.events:
            events = pd.DataFrame(evaluator.events)
            events["mode"] = mode
            event_frames.append(events)
        print(f"Completed fixed evaluation: {mode}", flush=True)
    np.savez_compressed(
        dest / "positions_and_trades.npz",
        **histories,
        dates=evaluation_data.dates,
        stock_ids=data.stock_ids,
    )
    pd.concat(event_frames, ignore_index=True).to_csv(
        dest / "missing_position_marks.csv", index=False
    ) if event_frames else pd.DataFrame(
        columns=[
            "date",
            "strategy",
            "stock_id",
            "previous_weight",
            "marked_dollars_per_initial_dollar",
            "mode",
        ]
    ).to_csv(dest / "missing_position_marks.csv", index=False)
    rows = pd.concat(frames, ignore_index=True)
    summary = statistics(rows)
    (dest / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(summary).to_csv(dest / "summary.csv", index=False)
    receipt.update(
        status="complete",
        completed_utc=datetime.now(UTC).isoformat(),
        output_hashes={
            p.name: sha(p)
            for p in dest.iterdir()
            if p.is_file() and p.name != "run_receipt.json"
        },
    )
    (dest / "run_receipt.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument(
        "--profile", choices=["selected"], default="selected"
    )
    args = parser.parse_args()
    {"prepare": prepare, "run": run}[args.action](args.profile)
