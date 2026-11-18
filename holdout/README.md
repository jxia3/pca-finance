# Holdout stock returns

Daily returns for **January 2, 2024–November 17, 2026**, covering the same 1,201 PERMNO IDs as `materials/daily_returns.csv`. Original-period observations are excluded from the holdout file. The original data remains unchanged.

This is a validated public-source snapshot with unresolved gaps. It is not a complete return history for every stock. Public-source returns are proxies, not an exact CRSP continuation. Some histories and latest-ticker reviews remain incomplete.

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
| Accepted return observations | 830,987 |
| Missing return cells, including dates after trading cutoffs | 37,336 |
| IDs with at least one holdout observation | 1,199 |
| IDs with a return on the final date | 1,086 |
| Original-period rows in the holdout | 0 |
| Original file unchanged | Yes; SHA-256 verified |
| CSV round-trip preserves values exactly | Yes |
| Missing values imputed | No |
| Reviewed trading cutoffs checked | 55 |

RPT and DISH have no standalone post-2023 exchange trading expected. Having some observations for an ID does not mean its history is complete. Per-ID coverage and gaps are retained in the evidence archive.

The final consistency audit verified matching ID order, unique dates and tickers, weekday dates, exact agreement with the frozen snapshot, unchanged original data, all retained raw-file hashes and observation source attribution. No nonmissing return is infinite or below −100%. There are 16,444 observed zero returns and 46 daily moves exceeding 50% in absolute value; these were not clipped or replaced. These checks establish file consistency, not independent certification of every market observation.

### Delistings and trading cutoffs

There are **55 documented security cutoffs**. They include acquisitions, reorganizations and trading suspensions; this is not a complete count of exchange delistings. Another 60 IDs lack a final-date return without a reviewed cutoff. A missing final observation alone does not establish a delisting.

| Original security | PERMNO | Returns excluded from |
| --- | --- | --- |
| DISH | 81696 | 2023-12-31 |
| RPT | 75416 | 2024-01-02 |
| IMGN | 75860 | 2024-02-12 |
| CWBC | 84215 | 2024-04-01 |
| MFV | 75840 | 2024-04-11 |
| PXD | 75241 | 2024-05-03 |
| SCX | 18438 | 2024-05-23 |
| TSRI | 74500 | 2024-07-01 |
| WIRE | 77785 | 2024-07-02 |
| MCBC | 87501 | 2024-08-01 |
| ATRI | 11587 | 2024-08-21 |
| SWN | 63765 | 2024-10-01 |
| PRFT | 87086 | 2024-10-02 |
| HMNF | 80633 | 2024-10-10 |
| SRCL | 83906 | 2024-11-04 |
| KSM | 75465 | 2024-11-14 |
| MRO | 15069 | 2024-11-22 |
| USAP | 81184 | 2025-01-23 |
| B | 39693 | 2025-01-27 |
| HYB | 75263 | 2025-02-12 |
| LGTY | 13777 | 2025-04-07 |
| LGF.A | 86408 | 2025-05-07 |
| FLIC | 11006 | 2025-06-02 |
| DMF | 75392 | 2025-06-18 |
| TGI | 84041 | 2025-07-25 |
| SGMA | 80274 | 2025-07-28 |
| BRKL | 85860 | 2025-09-01 |
| JEQ | 77860 | 2025-10-13 |
| BPT | 75622 | 2025-10-21 |
| K | 26825 | 2025-12-11 |
| NXC | 77692 | 2026-01-12 |
| NXN | 77693 | 2026-01-12 |
| MHN | 85339 | 2026-02-09 |
| MUE | 86754 | 2026-02-09 |
| BKN | 78936 | 2026-02-23 |
| MQT | 77925 | 2026-02-23 |
| MVF | 75354 | 2026-02-23 |
| MVT | 79139 | 2026-02-23 |
| MYD | 77152 | 2026-02-23 |
| NOM | 79222 | 2026-04-27 |
| NQP | 76572 | 2026-04-27 |
| FCO | 77458 | 2026-05-18 |
| AMWD | 10501 | 2026-05-29 |
| CMU | 71722 | 2026-06-08 |
| CXE | 75455 | 2026-06-08 |
| CXH | 75623 | 2026-06-08 |
| CIF | 75336 | 2026-06-22 |
| MCR | 75642 | 2026-06-22 |
| XOMA | 10333 | 2026-07-14 |
| ELSE | 32062 | 2026-07-31 |
| AVB | 80381 | 2026-08-17 |
| ALOT | 15537 | 2026-08-27 |
| LEG | 60943 | 2026-08-27 |
| BZH | 80193 | 2026-10-20 |
| LXP | 79770 | 2026-11-02 |

All cells on or after these cutoffs are blank in the CSV (`NaN` in pandas). They were **not filled with zero**. Actual observed zero returns elsewhere remain zero. Both scripts check that no nonmissing return occurs after a reviewed cutoff. The original ID column is retained, and the acquiring company's returns are not appended. Terminal cash payments, contingent rights and delisting returns have not been reconstructed, so a backtest needs an explicit policy for positions still held at these cutoffs.

## Recovery and remaining gaps

The November 18 recovery added 38,720 validated observations and removed 102 observations on or after documented trading cutoffs. Other previously accepted values were preserved exactly. The latest completed session is November 17, 2026; 1,086 IDs have a return on that date.

Of the 37,336 blank cells, 18,018 fall on or after reviewed cutoffs. The other 19,318 cells span 101 IDs and remain unresolved. That remainder includes additional ended securities whose exact cutoffs have not been reviewed, OTC histories, missing quotes for thinly traded securities and observations rejected by source-agreement checks. It must not be described as 19,318 confirmed missing trading-day returns. `audit/gaps.csv` in the evidence archive lists every ID and its status.

The bulk download was interrupted by the remote server after 540,016,640 bytes. Complete CSV rows were recovered from the partial archive: 29,567,804 market-wide price rows were scanned, and 1,410,154 target rows were retained. Dividends and splits were also retained. Separate Dolt API requests supplied the remaining available dates through November 16. Yahoo supplied validated November 17 quotes. No incomplete CSV row was accepted, and the partial ZIP was not treated as a complete download.

Additional Quiver requests returned rate-limit responses; those requests contributed no data. Conflicting quotes were not averaged. Missing prices were not carried forward, and returns across multi-session gaps were not assigned to a single day.

### Reviewed recovery exceptions

- ALOT: 488 original-period comparisons all differ by less than 0.000001. A reviewed bridge exception allows a maximum later difference below seven basis points across 545 pairs; 97.25% are within five basis points. This recovered 120 observations before its acquisition.
- PTN: SEC filings document its OTC interval and November 12, 2025 relisting. Only consecutive post-relisting Dolt closing prices corroborated by Yahoo within half a cent were added. A later reuse of `PTNT` was excluded. The OTC gap remains unresolved.
- SENEA: its November 17, 2026 return includes the issuer-confirmed five-for-one split. The preceding price basis was checked against 20 recent Yahoo observations.
- CYAN and DXYN: 201 OTC returns use consecutive Quiver closing prices corroborated by Yahoo within half a cent. Historical cent-rounding intervals contain 98.4% and 100% of 500 original returns, respectively. These are explicit rounding exceptions to the five-basis-point screen, not evidence of exact CRSP agreement. Disagreeing quotes remain excluded.

ChartExchange closing prices, independently retrieved from its public OTC tables and corroborated by Yahoo within half a cent on both consecutive dates, supplied another 706 returns for CYAN, DXYN and SFES. SFES historical cent-rounding agreement is 99% across 500 pairs. Its April 21, 2026 return includes the issuer-confirmed $1.40 special dividend. These are also rounding exceptions. SEC and OCC notices support the updated SFES and BPPTU labels. BPPTU last traded on October 20, 2025; its unrecovered OTC interval remains blank.

The 14 newly accepted daily moves above 50% were corroborated using closing-price pairs. None was clipped. Public sources may share upstream feeds, so corroboration is not necessarily independent exchange verification.

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

Exact source URLs, downloaded-file hashes and available pinned revisions are retained in the evidence archive. Downloads were data and documents from the listed public providers, issuers, exchanges and regulators. No dataset-supplied executable, macro or archived script was executed. This is a description of the retrieval procedure, not a malware-scan certification. Alternative Kaggle and AmirTrader datasets were screened but contributed no accepted returns.

## Evidence archive

| Directory | Contents |
| --- | --- |
| `raw/` | 3,652 source responses and extracts, issuer notices, target-specific bulk data and source manifests. |
| `audit/` | Mapping details, per-stock coverage and gaps, original-overlap checks, observation source codes, raw-file checksums, historical algorithm references and the removed-file inventory. |
| `snapshot/` | Frozen accepted extension and academic target extracts for reproducibility. |

The clean package retains target extracts and transfer metadata, not the large partial ZIP or full-market working CSV. Recovery audits and processing references are under `audit/recovery/`; source responses and target extracts are under `raw/recovery/`. Historical algorithm references are audit material; use the scripts below to rebuild the delivered files. No downloader remains running.

Price tables are retained as CSV extracts. Redundant Quiver HTML pages were removed where a nonempty extracted table exists; their original hashes and URLs are recorded in `audit/archive_compaction.json`. Parquet files were recompressed with Zstandard and checked for exact table and schema equality. All return observations, corporate-action evidence, source attribution and rebuild inputs are preserved.

## Rebuild and verify

Run from the repository root. Python 3.12 is recommended.

```sh
python -m pip install -r holdout/requirements.txt
python holdout/build.py
python holdout/verify.py
```

The build uses the unchanged original CSV and the frozen accepted snapshot. It makes no network requests and does not execute archived code. Verification checks output hashes, exact CSV values, exclusion of original dates, complete raw-evidence manifests, archive paths, per-ID coverage counts and per-observation attribution. The 723-date calendar also matches the retained SPY daily chart calendar exactly. New data requires separate review and validation before updating this snapshot.

## SHA-256 checksums

The scripts read this table to verify inputs and outputs. The original source path is relative to the repository root; other paths are relative to `holdout/`.

| File | SHA-256 |
| --- | --- |
| `materials/daily_returns.csv` | `45f7c2bc6439e5da8ef9ed4b204bfe4be2bf5653a3a88e5e79fa5007c322477d` |
| `source_evidence.zip` | `4442db06435c146c420ca189803610c73cae1b668af7467b9cc31f5092520f06` |
| `holdout_daily_returns.csv` | `77c186daa037bed1e7e9b1ebd5ddaeb6190b48ee99557135089ed8a4dbbd272a` |
| `holdout_tickers.csv` | `764455c64e8cb5648a03759ab3bbc6e0126ce1398472a3814703094bb27ff0d9` |
