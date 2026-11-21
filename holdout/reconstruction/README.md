# Reconstructed holdout input

The current evaluation reads `holdout/reconstruction/holdout_daily_returns.csv`.
The CSV in the parent directory remains the original source snapshot, so its
existing build and verification tools retain their original meaning.

Run from the repository root:

```
python -m scripts.reconstruct_holdout
python -m scripts.verify_holdout_repairs
```

Reconstruction is offline and deterministic against the retained evidence.
`changes.csv` records every changed cell. `source_validation.csv` records
historical and continuation agreement for each candidate method. Original
observed returns are preserved except for two PDCO observations after its
verified trading cutoff. The source archive is immutable.

The reconstruction adds 872 returns and removes those two observations, leaving
36,466 missing cells. It uses both archived Yahoo responses and the bounded
Yahoo chart responses in `raw/`. The download manifest records URLs and hashes.
No missing price is interpolated or carried forward into model input.

Acceptance requires consecutive sessions with positive reported volume and
agreement with the original returns. For reviewed low-price stocks, cent
rounding is evaluated on the appropriate split-adjusted price scale. The four
original exceptions are CYAN, DXYN, SFES and PTN. AGM.A has a shorter but exact
78-pair historical overlap on traded sessions. OMCC continues NICK's identity,
as documented in the SEC notice retained in the event review. Its repeated
closes sometimes conflict with original returns despite positive volume, so
both a repeated close and the following price pair are excluded. The remaining
431 historical and 147 continuation pairs agree within cent-rounding bounds.
Only 318 missing OMCC/NICK returns pass this conservative screen.

Seven reviewed large moves in CYAN and PTN are retained. Their consecutive
positive-volume closes imply the large moves, with no split on those dates.
The PTN calculation uses prices already adjusted for the recorded reverse
splits. A large percentage move alone is not evidence of an error in a penny
stock. Yahoo rounding and provider revisions remain measurement limitations.
Cached responses from the same provider are not independent confirmation.

Known security cutoffs prevent appending stale quotes after termination.
Remaining blanks include verified terminations, untraded/stale prices, and
unresolved coverage gaps. They are not all identified as delistings. Merger
consideration and unobserved terminal payoffs are not invented. These data
repairs are fixed before the corrected-data strategy rerun and do not use
strategy performance to select sources or parameters.
