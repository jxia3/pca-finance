# Holdout stock returns

Daily returns for **January 2, 2024–November 17, 2026**, covering the same 1,201 PERMNO IDs as `materials/daily_returns.csv`. Original-period observations are excluded from the holdout file. The original data remains unchanged.

This is a frozen snapshot with documented gaps (`frozen_holdout_snapshot_with_documented_gaps`). Public-source returns are proxies, not an exact CRSP continuation. Some histories and latest-ticker reviews remain incomplete.

## Files

| File | Contents |
| --- | --- |
| `holdout_daily_returns.csv` | `date`, followed by the original 1,201 ID columns in their original order. |
| `holdout_tickers.csv` | `stock_id,ticker`, in return-column order. |
| `source_evidence.zip` | Raw evidence, detailed audit records and the accepted snapshot. |
| `build.py` | Rebuild the two CSVs from the retained snapshot. |
| `verify.py` | Verify the CSVs, original data, trading cutoffs and source evidence. |
| `requirements.txt` | Pinned dependencies. |

Returns are decimals: `0.01` means 1%. Blank cells mean missing observations, not zero returns. Missing returns are never imputed. Successor stocks, fund NAVs and unverified settlements are not substituted for the original security. No daily merger-settlement or delisting return is inferred.

Ticker labels are the most recent supported working labels, or last-known labels for terminated securities. Selected renames have issuer evidence; the latest-label review is not complete across all IDs.

## Coverage and validation

| Measure | Result |
| --- | --- |
| First holdout date | 2024-01-02 |
| Last holdout date | 2026-11-17 |
| Dates | 723 |
| Stock IDs | 1,201 |
| Accepted return observations | 792,369 |
| IDs with at least one holdout observation | 1,199 |
| IDs with a return on the final date | 1,007 |
| Original-period rows in the holdout | 0 |
| Original file unchanged | Yes; SHA-256 verified |
| CSV round-trip preserves values exactly | Yes |
| Missing values imputed | No |
| Reviewed trading cutoffs checked | 14 |

RPT and DISH have no standalone post-2023 exchange trading expected. Having some observations for an ID does not mean its history is complete. Per-ID coverage and gaps are retained in the evidence archive.

### Delistings and trading cutoffs

There are **14 documented security cutoffs** in this snapshot. They include acquisitions, reorganizations and trading suspensions, so this is not a complete count of exchange delistings. Another 180 IDs lack a return on the final date and still require coverage or termination review. A missing final observation alone does not establish a delisting.

| Original security | PERMNO | Returns excluded from |
| --- | --- | --- |
| DISH | 81696 | 2023-12-31 |
| RPT | 75416 | 2024-01-02 |
| CWBC | 84215 | 2024-04-01 |
| TSRI | 74500 | 2024-07-01 |
| PRFT | 87086 | 2024-10-02 |
| HMNF | 80633 | 2024-10-10 |
| SRCL | 83906 | 2024-11-04 |
| B | 39693 | 2025-01-27 |
| HYB | 75263 | 2025-02-12 |
| LGF.A | 86408 | 2025-05-07 |
| TGI | 84041 | 2025-07-25 |
| K | 26825 | 2025-12-11 |
| XOMA | 10333 | 2026-07-14 |
| ALOT | 15537 | 2026-08-27 |

All cells on or after these cutoffs are blank in the CSV (`NaN` in pandas). They were **not filled with zero**. Actual observed zero returns elsewhere remain zero. Both scripts check that no nonmissing return occurs after a reviewed cutoff. The original ID column is retained, and the acquiring company's returns are not appended. Terminal cash payments, contingent rights and delisting returns have not been reconstructed, so a backtest needs an explicit policy for positions still held at these cutoffs.

## Agreement with the original returns

Candidates were compared with the original data over shared historical dates. The main automatic screen required at least 100 pairs, correlation of at least 0.995, and at least 95% of return differences within 0.0005: five basis points, or 0.05 percentage points.

| Passing source group | Candidate IDs | Median correlation | Median within 5 bp | Lowest within 5 bp | Fewest overlap pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Yahoo baseline | 1,009 | 0.9999996876468229 | 100% | 95% | 249 |
| Dolt | 187 | 0.9999999999163979 | 100% | 99% | 206 |
| Defeatbeta | 58 | 0.9999998983528967 | 100% | 95% | 500 |

The groups overlap, so their ID counts must not be added. These are historical screening results, not independent verification of every later observation. Manual exceptions include SKM's depositary distributions, MHD's original/vendor discrepancy and the short overlap for illiquid AGM.A. Supporting reviews are retained in `source_evidence.zip`.

## Sources

- [Yahoo Finance](https://finance.yahoo.com/): public chart data.
- [Defeatbeta](https://huggingface.co/datasets/defeatbeta/yahoo-finance-data): bulk prices, dividends and splits.
- [DoltHub stocks](https://www.dolthub.com/repositories/post-no-preference/stocks): prices and corporate actions.
- [FINSABER](https://huggingface.co/datasets/finsaber-team/FINSABER-V2-Data): academic price data.
- Public tables from Quiver and StockAnalysis; SEC filings, exchange and issuer records, and BNY and Citi depositary notices for identity and corporate-action checks.

Exact source URLs, downloaded-file hashes and available pinned revisions are retained in the evidence archive. Alternative Kaggle and AmirTrader datasets were screened but contributed no accepted returns.

## Evidence archive

| Directory | Contents |
| --- | --- |
| `raw/` | 3,475 source responses and extracts, issuer notices, target-specific bulk data and source manifests. |
| `audit/` | Mapping details, per-stock coverage and gaps, original-overlap checks, observation source codes, raw-file checksums, historical algorithm references and the removed-file inventory. |
| `snapshot/` | Frozen accepted extension and academic target extracts for reproducibility. |

Full-market duplicates, failed transfers, obsolete probes and intermediate working matrices were removed. Historical algorithm references are audit material; use the scripts below to rebuild the delivered files. No downloader remains running.

## Rebuild and verify

Run from the repository root. Python 3.12 is recommended.

```sh
python -m pip install -r holdout/requirements.txt
python holdout/build.py
python holdout/verify.py
```

The build uses the unchanged original CSV and the frozen accepted snapshot. It makes no network requests and does not execute archived code. Verification checks output hashes, exact CSV values, exclusion of original dates, raw evidence hashes and per-observation attribution. New data requires separate review and validation before updating this snapshot.

## SHA-256 checksums

The scripts read this table to verify inputs and outputs. The original source path is relative to the repository root; other paths are relative to `holdout/`.

| File | SHA-256 |
| --- | --- |
| `materials/daily_returns.csv` | `45f7c2bc6439e5da8ef9ed4b204bfe4be2bf5653a3a88e5e79fa5007c322477d` |
| `source_evidence.zip` | `03c2bd4fe876fc5ae784c62defedd90285586c92c99e95ef9bd140f77c7afbbc` |
| `holdout_daily_returns.csv` | `750bd99b29fb94203dcd873a7c89166b4b778bbf7688925bef3bbfd8d981e6d8` |
| `holdout_tickers.csv` | `c11b4c8b81e691d4351fb851e2a504a4bf140877e1e6dd89e7bc57fb3bcdd8d3` |
