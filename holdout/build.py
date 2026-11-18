from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile
import numpy as np
import pandas as pd

HOLDOUT_DIR = Path(__file__).resolve().parent
ROOT_DIR = HOLDOUT_DIR.parent
ORIGINAL_FILE = ROOT_DIR / "materials/daily_returns.csv"
EVIDENCE_FILE = HOLDOUT_DIR / "source_evidence.zip"
RETURNS_FILE = HOLDOUT_DIR / "holdout_daily_returns.csv"
TICKERS_FILE = HOLDOUT_DIR / "holdout_tickers.csv"

@dataclass(kw_only=True)
class HoldoutData:
    original_returns: pd.DataFrame
    returns: pd.DataFrame
    tickers: pd.DataFrame
    cutoffs: pd.DataFrame
    checksums: dict[str, str]

def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()

def load_checksums() -> dict[str, str]:
    readme = (HOLDOUT_DIR / "README.md").read_text(encoding="utf-8")
    matches = re.findall(
        r"^\| `([^`]+)` \| `([0-9a-f]{64})` \|$",
        readme,
        flags=re.MULTILINE,
    )
    checksums = dict(matches)
    expected_files = {
        "materials/daily_returns.csv",
        "source_evidence.zip",
        "holdout_daily_returns.csv",
        "holdout_tickers.csv",
    }
    assert len(matches) == len(expected_files) and set(checksums) == expected_files
    return checksums

def load_inputs() -> HoldoutData:
    checksums = load_checksums()
    assert sha256(ORIGINAL_FILE) == checksums["materials/daily_returns.csv"]
    assert sha256(EVIDENCE_FILE) == checksums[EVIDENCE_FILE.name]
    original_returns = pd.read_csv(ORIGINAL_FILE, index_col="date")

    with zipfile.ZipFile(EVIDENCE_FILE) as archive:
        returns = pd.read_parquet(
            io.BytesIO(archive.read("snapshot/accepted_extension.parquet")),
        )
        mapping = pd.read_csv(
            io.BytesIO(archive.read("audit/mapping.csv")),
            dtype=str,
        )
        cutoffs = pd.read_csv(
            io.BytesIO(archive.read("audit/reviewed_security_cutoffs.csv")),
            dtype=str,
        )

    assert original_returns.columns.equals(returns.columns)
    assert mapping.permno.is_unique and set(mapping.permno) == set(returns.columns)
    assert cutoffs.permno.is_unique and cutoffs.permno.isin(returns.columns).all()
    assert returns.index.min() > original_returns.index.max()
    assert returns.index.is_unique and returns.index.is_monotonic_increasing
    assert not np.isinf(returns.to_numpy()).any()
    assert not (returns < -1).any().any()
    for cutoff in cutoffs.itertuples():
        post_cutoff = returns.loc[returns.index >= cutoff.exclude_from, cutoff.permno]
        assert post_cutoff.isna().all(), cutoff.permno

    tickers = mapping.set_index("permno").loc[returns.columns, ["ticker"]]
    tickers.index.name = "stock_id"
    return HoldoutData(
        original_returns=original_returns,
        returns=returns,
        tickers=tickers,
        cutoffs=cutoffs,
        checksums=checksums,
    )

def build() -> None:
    data = load_inputs()
    data.returns.to_csv(RETURNS_FILE, index_label="date")
    data.tickers.to_csv(TICKERS_FILE)
    pd.testing.assert_frame_equal(
        pd.read_csv(RETURNS_FILE, index_col="date"),
        data.returns,
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(TICKERS_FILE, index_col="stock_id", dtype=str),
        data.tickers,
        check_exact=True,
    )
    for path in [RETURNS_FILE, TICKERS_FILE]:
        assert sha256(path) == data.checksums[path.name], path.name

    report = {
        "holdout_rows": len(data.returns),
        "stocks": len(data.returns.columns),
        "first_date": data.returns.index.min(),
        "last_date": data.returns.index.max(),
        "observations": int(data.returns.notna().sum().sum()),
        "stocks_with_data": int(data.returns.notna().any().sum()),
        "latest_day_stocks": int(data.returns.iloc[-1].notna().sum()),
        "reviewed_cutoffs_checked": len(data.cutoffs),
    }
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    build()
