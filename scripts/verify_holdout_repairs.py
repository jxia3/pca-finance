"""Verify every reconstructed cell and the retained source evidence offline."""
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    root = Path('holdout/reconstruction')
    summary = json.loads((root / 'summary.json').read_text())
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest(Path('holdout/source_evidence.zip')) == summary['input_archive_sha256']
    assert digest(root / 'holdout_daily_returns.csv') == summary['output_sha256']
    with zipfile.ZipFile('holdout/source_evidence.zip') as archive:
        original = pd.read_parquet(io.BytesIO(archive.read('snapshot/accepted_extension.parquet')))
    repaired = pd.read_csv(root / 'holdout_daily_returns.csv', index_col=0)
    ledger = pd.read_csv(root / 'changes.csv', dtype={'stock_id': str})
    assert not ledger.duplicated(['stock_id', 'date']).any()
    replay = original.copy()
    for row in ledger.itertuples():
        prior = replay.loc[row.date, row.stock_id]
        assert (pd.isna(prior) and pd.isna(row.old_return)) or np.isclose(prior, row.old_return)
        replay.loc[row.date, row.stock_id] = row.new_return
    pd.testing.assert_frame_equal(replay, repaired, check_exact=False, rtol=1e-12, atol=1e-15)
    for row in pd.read_csv(root / 'reviewed_cutoffs.csv').itertuples():
        assert repaired.loc[repaired.index >= row.exclude_from, str(row.permno)].isna().all()
    for record in json.loads((root / 'download_manifest.json').read_text()):
        if record['status'] == 'ok':
            assert digest(root / 'raw' / (record['symbol'] + '.json')) == record['sha256']
    assert repaired.notna().sum().sum() - original.notna().sum().sum() == 870
    print('Verified cell ledger, unchanged observations, cutoffs, dataset and source hashes.')


if __name__ == '__main__':
    main()
