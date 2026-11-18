import hashlib
import io
import json
import zipfile
import pandas as pd

from build import EVIDENCE_FILE, RETURNS_FILE, TICKERS_FILE, load_inputs, sha256

def verify() -> None:
    data = load_inputs()
    for path in [RETURNS_FILE, TICKERS_FILE]:
        assert sha256(path) == data.checksums[path.name], path.name

    returns = pd.read_csv(RETURNS_FILE, index_col="date")
    tickers = pd.read_csv(TICKERS_FILE, index_col="stock_id", dtype=str)
    pd.testing.assert_frame_equal(returns, data.returns, check_exact=True)
    pd.testing.assert_frame_equal(tickers, data.tickers, check_exact=True)
    assert returns.index.intersection(data.original_returns.index).empty
    assert returns.columns.is_unique and tickers.index.is_unique
    assert tickers.index.tolist() == returns.columns.tolist()
    assert tickers.ticker.notna().all() and tickers.ticker.is_unique
    assert tickers.ticker.str.strip().ne("").all()
    assert (pd.to_datetime(returns.index).day_of_week < 5).all()

    with zipfile.ZipFile(EVIDENCE_FILE) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names))
        assert all(not name.startswith("/") and ".." not in name.split("/") for name in names)
        assert archive.testzip() is None
        manifest = json.loads(archive.read("audit/raw_evidence_manifest.json"))
        assert {"raw/" + entry["member"] for entry in manifest} == {
            name for name in names if name.startswith("raw/") and not name.endswith("/")
        }
        for entry in manifest:
            contents = archive.read("raw/" + entry["member"])
            assert len(contents) == entry["bytes"], entry["member"]
            assert hashlib.sha256(contents).hexdigest() == entry["sha256"], entry["member"]
        source_codes = pd.read_parquet(
            io.BytesIO(archive.read("audit/extension_source_codes.parquet")),
        )
        source_definitions = json.loads(archive.read("audit/source_codes.json"))
        coverage = pd.read_csv(io.BytesIO(archive.read("audit/coverage.csv")), dtype={"permno": str})
        calendar = pd.read_csv(io.BytesIO(archive.read("audit/trading_calendar.csv")), dtype=str)
        chart = json.loads(archive.read("raw/recovery/yahoo/SPY.json"))["chart"]["result"][0]
        quote_dates = pd.to_datetime(chart["timestamp"], unit="s", utc=True)
        quote_dates = quote_dates.tz_convert("America/New_York").strftime("%Y-%m-%d")
        market_dates = quote_dates[
            (quote_dates >= returns.index.min()) & (quote_dates <= returns.index.max())
        ]

    assert source_codes.columns.equals(returns.columns)
    assert source_codes.index.equals(returns.index)
    assert ((source_codes > 0) == returns.notna()).all().all()
    assert set(source_codes.to_numpy().ravel()) <= {0, *map(int, source_definitions)}
    assert calendar.date.tolist() == returns.index.tolist()
    assert market_dates.tolist() == returns.index.tolist()
    assert coverage.permno.tolist() == returns.columns.tolist()
    assert coverage.observations.tolist() == returns.notna().sum().tolist()
    assert coverage.missing_calendar_sessions.tolist() == returns.isna().sum().tolist()
    print(
        f"PASS: {len(returns)} holdout dates, {len(tickers)} IDs, unchanged original, "
        f"exact CSV values, {len(data.cutoffs)} trading cutoffs, "
        f"{len(manifest)} raw evidence files and observation source attribution."
    )

if __name__ == "__main__":
    verify()
