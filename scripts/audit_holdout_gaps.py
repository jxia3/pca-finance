"""Offline gap investigation using retained evidence and reviewed event links.

Run: python -m scripts.audit_holdout_gaps
No network requests, archived code execution, return edits, or strategy runs.
Reviewed event links are evidence annotations, not replacement price data.
"""

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DEST = Path("holdout/gap_investigation")
EVIDENCE = Path("holdout/source_evidence.zip")


def read_csv(archive, name):
    with archive.open(name) as stream:
        return pd.read_csv(stream)


def quote_diagnostics(archive, returns, tickers):
    """Diagnose six heavily affected names without accepting rejected quotes."""
    bulk = pd.read_parquet(io.BytesIO(archive.read("raw/recovery/dolt_ohlcv.parquet")))
    rows = []
    for ticker in ("AGM.A", "AKO.A", "CYAN", "DXYN", "SFES", "PTN"):
        stock = str(tickers.index[tickers.eq(ticker)][0])
        path = f"raw/recovery/yahoo/{ticker}.json"
        chart = json.loads(archive.read(path))["chart"]["result"][0]
        dates = pd.to_datetime(chart["timestamp"], unit="s", utc=True)
        dates = dates.tz_convert("America/New_York").strftime("%Y-%m-%d")
        close = pd.Series(
            chart["indicators"]["quote"][0]["close"], index=dates
        ).sort_index()
        # Shift on the market calendar including the preceding original session.
        calendar = pd.Index(["2023-12-29"]).append(returns.index)
        close = close.reindex(calendar)
        consecutive = (close.notna() & close.shift().notna()).reindex(returns.index)
        missing = returns[stock].isna()
        row = {
            "stock_id": stock,
            "ticker": ticker,
            "missing_cells": int(missing.sum()),
            "missing_with_two_yahoo_closes": int((missing & consecutive).sum()),
            "cached_source": path,
        }
        symbol = "SFE" if ticker == "SFES" else ticker
        bulk_close = (
            bulk.loc[bulk.act_symbol.eq(symbol)].set_index("date").close.astype(float)
        )
        bulk_close.index = bulk_close.index.astype(str)
        bulk_close = bulk_close.reindex(calendar)
        bulk_pair = bulk_close.notna() & bulk_close.shift().notna()
        row["missing_with_two_dolt_closes"] = int(
            (missing & bulk_pair.reindex(returns.index)).sum()
        )
        if ticker in ("CYAN", "DXYN", "SFES"):
            table = read_csv(archive, f"raw/recovery/otc_tables/{ticker}.csv")
            prices = pd.to_numeric(table.set_index("date")["close"]).reindex(calendar)
            two_prices = prices.notna() & prices.shift().notna()
            match = prices.sub(close).abs().le(0.00501)
            both_match = match & match.shift(fill_value=False)
            row["missing_without_two_corroborating_closes"] = int(
                (missing & ~two_prices.reindex(returns.index)).sum()
            )
            row["missing_with_corroborating_price_disagreement"] = int(
                (
                    missing
                    & two_prices.reindex(returns.index)
                    & ~both_match.reindex(returns.index)
                ).sum()
            )
            row["missing_with_two_matching_closes"] = int(
                (missing & both_match.reindex(returns.index)).sum()
            )
            raw_return = prices / prices.shift() - 1
            row["matching_close_gaps_exceeding_50pct_filter"] = int(
                (
                    missing
                    & both_match.reindex(returns.index)
                    & raw_return.reindex(returns.index).abs().gt(0.5)
                ).sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    returns = pd.read_csv("holdout/holdout_daily_returns.csv", index_col=0)
    tickers = pd.read_csv("holdout/holdout_tickers.csv", index_col=0).ticker
    events = pd.read_csv(DEST / "reviewed_events.csv")
    records = []
    for stock in returns:
        missing = returns[stock].isna().to_numpy()
        starts = np.flatnonzero(missing & ~np.r_[False, missing[:-1]])
        ends = np.flatnonzero(missing & ~np.r_[missing[1:], False])
        for start, end in zip(starts, ends):
            records.append(
                {
                    "stock_id": int(stock),
                    "ticker": tickers[int(stock)],
                    "start": returns.index[start],
                    "end": returns.index[end],
                    "length": int(end - start + 1),
                    "resumes": end < len(returns) - 1,
                }
            )
    spells = pd.DataFrame(records)
    per_stock = (
        spells.groupby(["stock_id", "ticker"])
        .agg(
            missing_cells=("length", "sum"),
            spells=("length", "size"),
            resuming_spells=("resumes", "sum"),
        )
        .reset_index()
    )
    per_stock = per_stock.merge(events, how="left", on=["stock_id", "ticker"])
    with zipfile.ZipFile(EVIDENCE) as archive:
        cutoffs = read_csv(archive, "audit/reviewed_security_cutoffs.csv")
        diagnostics = quote_diagnostics(archive, returns, tickers)
        codes = pd.read_parquet(
            io.BytesIO(archive.read("audit/extension_source_codes.parquet"))
        )
        source_names = json.loads(archive.read("audit/source_codes.json"))
    # Disjoint counts: old cutoffs take precedence over newly reviewed events.
    old = pd.DataFrame(False, index=returns.index, columns=returns.columns)
    new = old.copy()
    flagged = []
    for row in cutoffs.itertuples():
        old.loc[old.index >= row.exclude_from, str(row.permno)] = True
    for row in events.loc[events.category.eq("additional_termination")].itertuples():
        stock = str(row.stock_id)
        new.loc[new.index >= row.event_date, stock] = True
        after = returns.loc[returns.index >= row.event_date, stock].dropna()
        for date, value in after.items():
            code = str(int(codes.at[date, stock]))
            flagged.append(
                {
                    "stock_id": row.stock_id,
                    "ticker": row.ticker,
                    "date": date,
                    "accepted_return": value,
                    "source": source_names[code]["source"],
                    "event_source": row.source,
                }
            )
    missing = returns.isna()
    old_count = int((missing & old).sum().sum())
    new_count = int((missing & new & ~old).sum().sum())
    resume = spells.loc[spells.resumes]
    concentration = resume.groupby("ticker").length.sum().sort_values(ascending=False)
    summary = {
        "dates": len(returns),
        "stocks": len(returns.columns),
        "entirely_missing_date_rows": int(missing.all(axis=1).sum()),
        "missing_cells": int(missing.sum().sum()),
        "spells": len(spells),
        "resuming_spells": len(resume),
        "resuming_cells": int(resume.length.sum()),
        "top_five_resuming_cells": int(concentration.head(5).sum()),
        "top_five_resuming_tickers": concentration.head(5).index.tolist(),
        "previously_documented_cutoff_cells": old_count,
        "additional_termination_cells": new_count,
        "remaining_cells_outside_these_cutoffs": int(missing.sum().sum())
        - old_count
        - new_count,
        "new_termination_cases_reviewed": int(
            events.category.eq("additional_termination").sum()
        ),
        "scope": "Targeted event review; remaining cells are not all diagnosed. OTC event context does not establish a cause for every later blank.",
    }
    spells.to_csv(DEST / "gap_spells.csv", index=False)
    per_stock.sort_values("missing_cells", ascending=False).to_csv(
        DEST / "per_stock.csv", index=False
    )
    diagnostics.to_csv(DEST / "cached_quote_diagnostics.csv", index=False)
    pd.DataFrame(flagged).to_csv(DEST / "post_event_observations.csv", index=False)
    (DEST / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()
