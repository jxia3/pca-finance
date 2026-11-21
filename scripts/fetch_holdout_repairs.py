"""Fetch bounded JSON responses from Yahoo Finance for incomplete histories.

Only the named Yahoo chart endpoint is used. Existing successful responses
are reused. No downloaded code is executed and no package is installed.
"""
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

DEST = Path('holdout/reconstruction/raw')


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    returns = pd.read_csv('holdout/holdout_daily_returns.csv', index_col=0)
    labels = pd.read_csv('holdout/holdout_tickers.csv', index_col=0).ticker
    symbols = [labels[int(stock)].replace('.', '-') for stock in returns.columns[returns.isna().any()]]
    symbols = sorted(set(symbols + ['OMCC', 'SSUP', 'PTNT']))
    records = []
    start = int(datetime(2022, 1, 1, tzinfo=UTC).timestamp())
    end = int(datetime(2026, 11, 18, tzinfo=UTC).timestamp())
    for i, symbol in enumerate(symbols):
        # OMCC was fetched during the initial identity review with this endpoint.
        # Reconstruction still truncates every source at the evaluation end date.
        query_end = 1795046400 if symbol == 'OMCC' else end
        url = (f'https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe="")}'
               f'?period1={start}&period2={query_end}&interval=1d&events=div%2Csplits')
        path = DEST / f'{symbol}.json'
        record = {'symbol': symbol, 'url': url}
        try:
            if not path.exists():
                with urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=15) as response:
                    payload = response.read(4_000_001)
                if len(payload) > 4_000_000:
                    raise ValueError('Response exceeded size limit')
                json.loads(payload)
                path.write_bytes(payload)
                time.sleep(.2)
            payload = path.read_bytes()
            chart = json.loads(payload)['chart']
            result = chart.get('result')
            record.update(status='ok' if result else 'provider_error',
                          sha256=hashlib.sha256(payload).hexdigest(),
                          observations=len(result[0].get('timestamp', [])) if result else 0,
                          error=chart.get('error'))
        except Exception as error:
            record.update(status='unavailable', error=str(error))
        records.append(record)
        if (i+1) % 20 == 0:
            print(f'Reviewed {i+1}/{len(symbols)} Yahoo histories', flush=True)
    (DEST.parent/'download_manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(pd.Series([r['status'] for r in records]).value_counts().to_string())


if __name__ == '__main__':
    main()
