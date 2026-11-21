"""Reconstruct missing returns from price evidence, without imputation.

Reads the original frozen snapshot and cached/new Yahoo JSON. Acceptance is
based on historical agreement and split-aware rounding bounds, never strategy
performance. Produces a staged dataset and a cell-level change log.
"""
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DEST = Path('holdout/reconstruction')
ALIASES = {'85705': 'OMCC', '52396': 'SSUP'}
ROUNDING_NAMES = {'10661', '10886', '46340', '79702', '85705'}


def chart_frame(payload, calendar):
    chart = json.loads(payload).get('chart', {}).get('result')
    if not chart or len(chart[0].get('timestamp', [])) < 30:
        return None
    chart = chart[0]
    dates = pd.to_datetime(chart['timestamp'], unit='s', utc=True)
    dates = dates.tz_convert('America/New_York').strftime('%Y-%m-%d')
    quote = chart['indicators']['quote'][0]
    frame = pd.DataFrame({'close': quote['close'], 'volume': quote['volume']}, index=dates)
    if not frame.index.is_unique:
        raise ValueError('Duplicate chart dates')
    adjusted = chart['indicators'].get('adjclose', [{}])[0].get('adjclose')
    frame['adjusted'] = adjusted if adjusted is not None else frame.close
    frame = frame.reindex(calendar)
    frame['cash'] = 0.
    frame['precision'] = .00501
    for event in chart.get('events', {}).get('dividends', {}).values():
        day = pd.Timestamp(event['date'], unit='s', tz='UTC').tz_convert('America/New_York').strftime('%Y-%m-%d')
        if day in frame.index:
            frame.loc[day, 'cash'] += event['amount']
    for event in chart.get('events', {}).get('splits', {}).values():
        day = pd.Timestamp(event['date'], unit='s', tz='UTC').tz_convert('America/New_York').strftime('%Y-%m-%d')
        frame.loc[frame.index < day, 'precision'] *= event['denominator'] / event['numerator']
    frame['traded_pair'] = frame.volume.gt(0) & frame.volume.shift().gt(0)
    frame['cash_return'] = (frame.close + frame.cash) / frame.close.shift() - 1
    frame['adjusted_return'] = frame.adjusted / frame.adjusted.shift() - 1
    frame['lower'] = (frame.close-frame.precision+frame.cash)/(frame.close.shift()+frame.precision.shift())-1
    frame['upper'] = (frame.close+frame.precision+frame.cash)/(frame.close.shift()-frame.precision.shift())-1
    return frame


def agreement(reference, frame, method):
    candidate = frame[method].reindex(reference.index)
    valid = reference.notna() & candidate.notna() & frame.traded_pair.reindex(reference.index, fill_value=False)
    actual, estimated = reference[valid], candidate[valid]
    if len(actual) < 2:
        return {'pairs': len(actual), 'correlation': 0., 'within_5bp': 0., 'rounding_agreement': 0.}
    return {'pairs': len(actual), 'correlation': float(actual.corr(estimated)),
            'within_5bp': float((actual-estimated).abs().le(.0005).mean()),
            'rounding_agreement': float(actual.between(frame.lower.reindex(actual.index)-1e-6,
                                                       frame.upper.reindex(actual.index)+1e-6).mean())}


def qualification(stock, score, method):
    if score['pairs'] >= 100 and score['correlation'] >= .995 and score['within_5bp'] >= .95:
        return 'historical_agreement_on_traded_sessions'
    if stock == '80168' and score['pairs'] >= 78 and score['correlation'] >= .99999 and score['within_5bp'] == 1:
        return 'reviewed_class_A_short_overlap'
    if stock in ROUNDING_NAMES and method == 'cash_return' and score['pairs'] >= 400 and score['correlation'] >= .98 and score['rounding_agreement'] >= .98:
        return 'split_aware_cent_rounding_exception'
    return None


def candidates(archive, stock, ticker):
    symbols = {ticker, ticker.replace('.', '-'), ALIASES.get(stock, ticker)}
    # Aliased chart identity has an official link in the separate event review.
    for path in sorted((DEST/'raw').glob('*.json')):
        if path.stem in symbols:
            yield str(path), path.read_bytes()
    for name in archive.namelist():
        if name.endswith('.json') and Path(name).stem in symbols and any(
            part in name for part in ('/raw/', '/yahoo/', '/latest_refresh/')
        ):
            yield name, archive.read(name)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv('materials/daily_returns.csv', index_col=0).loc['2022-01-01':]
    tickers = pd.read_csv('holdout/holdout_tickers.csv', index_col=0, dtype=str)
    with zipfile.ZipFile('holdout/source_evidence.zip') as archive:
        base = pd.read_parquet(io.BytesIO(archive.read('snapshot/accepted_extension.parquet')))
        cutoffs = pd.read_csv(archive.open('audit/reviewed_security_cutoffs.csv'), dtype=str)
        events = pd.read_csv('holdout/gap_investigation/reviewed_events.csv', dtype=str)
        added = events.loc[events.category.eq('additional_termination'), ['stock_id','ticker','event_date','source']]
        added = added.rename(columns={'stock_id':'permno','event_date':'exclude_from'})
        cutoffs = pd.concat([cutoffs, added], ignore_index=True).sort_values('exclude_from').drop_duplicates('permno')
        result = base.copy()
        changes, reviews, large = [], [], []
        for row in cutoffs.itertuples():
            for day, old in result.loc[result.index >= row.exclude_from, row.permno].dropna().items():
                changes.append({'stock_id':row.permno,'date':day,'old_return':old,'new_return':np.nan,
                                'reason':'after_verified_last_trading_day','source':row.source})
            result.loc[result.index >= row.exclude_from, row.permno] = np.nan
        calendar = original.index.union(base.index).sort_values()
        for stock in base.columns[base.isna().any()]:
            ticker = tickers.loc[stock,'ticker']
            eligible = pd.Series(True, index=base.index)
            boundary = cutoffs.loc[cutoffs.permno.eq(stock)]
            if not boundary.empty:
                eligible.loc[eligible.index >= boundary.iloc[0].exclude_from] = False
            options = []
            for source, payload in candidates(archive, stock, ticker):
                frame = chart_frame(payload, calendar)
                if frame is None:
                    continue
                if stock == '85705':
                    # OMCC/NICK has stale repeated closes despite positive volume.
                    # Reject both the repeated close and the following return.
                    # The remaining overlap agrees within cent-rounding bounds.
                    changed = frame.close.diff().ne(0)
                    frame['traded_pair'] &= changed & changed.shift(fill_value=False)
                for method in ('cash_return','adjusted_return'):
                    score = agreement(original[stock],frame,method)
                    rule = qualification(stock,score,method)
                    bridge = agreement(base[stock],frame,method)
                    # A materially inconsistent continuation is not accepted merely
                    # because the earlier original sample happened to agree.
                    bridge_ok = bridge['pairs'] < 20 or (bridge['correlation'] >= .98 and
                                 (bridge['within_5bp'] >= .95 or (method=='cash_return' and bridge['rounding_agreement']>=.98)))
                    reviews.append({'stock_id':stock,'ticker':ticker,'source':source,'method':method,
                                    **score,**{'bridge_'+k:v for k,v in bridge.items()},
                                    'rule':rule,'bridge_ok':bridge_ok})
                    if rule and bridge_ok:
                        values = frame[method].reindex(base.index).where(frame.traded_pair.reindex(base.index))
                        values = values.where(eligible & np.isfinite(values) & values.gt(-1))
                        # Reviewed 64% CYAN move: retained two-source closing prices.
                        reviewed_large = {
                            '10661': ['2024-04-29', '2024-07-17', '2025-12-02'],
                            '79702': ['2025-07-23', '2025-08-06', '2025-08-18', '2025-11-21'],
                        }
                        allowed_large = values.index.isin(reviewed_large.get(stock, []))
                        needs_review = values.abs().gt(.5) & ~allowed_large & result[stock].isna()
                        for day, value in values[needs_review].items():
                            large.append({'stock_id':stock,'ticker':ticker,'date':day,'return':value,'source':source})
                        values = values.mask(needs_review)
                        options.append((score['within_5bp'], bridge['within_5bp'], source, rule, method, values))
            # Fixed quality ordering; this does not inspect portfolio performance.
            options.sort(key=lambda option:(-option[0],-option[1],option[2],option[4]))
            for _,_,source,rule,method,values in options:
                additions = values[result[stock].isna() & values.notna()]
                for day,value in additions.items():
                    changes.append({'stock_id':stock,'date':day,'old_return':np.nan,'new_return':value,
                                    'reason':rule+':'+method,'source':source})
                result.loc[additions.index,stock] = additions
        pd.DataFrame(reviews).to_csv(DEST/'source_validation.csv',index=False)
        pd.DataFrame(large).drop_duplicates().to_csv(DEST/'large_moves_for_review.csv',index=False)
        pd.DataFrame(changes).to_csv(DEST/'changes.csv',index=False)
        cutoffs.to_csv(DEST/'reviewed_cutoffs.csv',index=False)
        result.to_csv(DEST/'holdout_daily_returns.csv',index_label='date')
        tickers.loc['85705','ticker']='OMCC'
        tickers.to_csv(DEST/'holdout_tickers.csv')
        assert result.columns.equals(base.columns) and result.index.equals(base.index)
        assert np.isfinite(result.to_numpy()[result.notna().to_numpy()]).all()
        assert not (result < -1).any().any()
        summary = {'initial_missing':int(base.isna().sum().sum()),'final_missing':int(result.isna().sum().sum()),
                   'added_returns':int((base.isna()&result.notna()).sum().sum()),
                   'removed_returns':int((base.notna()&result.isna()).sum().sum()),
                   'stocks_with_gaps':int(result.isna().any().sum()),'reviewed_cutoffs':len(cutoffs),
                   'input_archive_sha256':hashlib.sha256(Path('holdout/source_evidence.zip').read_bytes()).hexdigest(),
                   'output_sha256':hashlib.sha256((DEST/'holdout_daily_returns.csv').read_bytes()).hexdigest()}
        (DEST/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
