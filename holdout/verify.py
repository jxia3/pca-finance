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

    with zipfile.ZipFile(EVIDENCE_FILE) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("audit/raw_evidence_manifest.json"))
        for entry in manifest:
            contents = archive.read("raw/" + entry["member"])
            assert len(contents) == entry["bytes"], entry["member"]
            assert hashlib.sha256(contents).hexdigest() == entry["sha256"], entry["member"]
        source_codes = pd.read_parquet(
            io.BytesIO(archive.read("audit/extension_source_codes.parquet")),
        )

    assert source_codes.columns.equals(returns.columns)
    assert source_codes.index.equals(returns.index)
    assert ((source_codes > 0) == returns.notna()).all().all()
    print(
        f"PASS: {len(returns)} holdout dates, {len(tickers)} IDs, unchanged original, "
        f"exact CSV values, {len(data.cutoffs)} trading cutoffs, "
        f"{len(manifest)} raw evidence files and observation source attribution."
    )

if __name__ == "__main__":
    verify()
