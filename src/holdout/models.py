"""Past-data-only universe adaptation for the fixed Kayla strategies."""

from dataclasses import dataclass

import numpy as np

from src.holdout_strategies.factor import FactorModel
from src.holdout_strategies.pairs_model import PairSignals, PairsModel
from src.holdout_strategies.pairs_trading import PairsTrading
from src.holdout_strategies.treynor_black import TreynorBlack
from src.models.pca import PCAModel


@dataclass
class Decision:
    weights: np.ndarray
    eligible: np.ndarray
    betas: np.ndarray | None
    selected_pairs: int
    refit: bool


class HoldoutModels:
    """Causal missing-data adapter around the unchanged imported models."""

    def __init__(self, config, allowed, start):
        self.c = config
        self.allowed = np.asarray(allowed, dtype=bool)
        self.start = np.datetime64(start)
        self.ids = np.array([], dtype=int)
        self.pca = self.factor = self.pair_model = None
        self.global_pairs = np.empty((0, 2), dtype=int)
        self.tb = TreynorBlack(
            "factor",
            score_limit=config["score_cap"],
            contrarian=True,
            factor_neutral=True,
            gross_leverage=config.get("gross", 1.0),
        )
        self.pairs = PairsTrading(
            "pairs",
            entry_threshold=config["entry"],
            exit_threshold=config["exit"],
            gross_leverage=config.get("gross", 1.0),
        )

    def update(self, timestamp, returns, model_outputs):
        c = self.c
        empty = np.zeros((2, returns.shape[1]))
        if len(returns) < c["pca_window"]:
            return Decision(empty, np.zeros(returns.shape[1], bool), None, 0, False)
        history = returns[-c["pca_window"] :]
        eligible = np.all(np.isfinite(history) & (history > -1), axis=0) & self.allowed
        candidate = np.flatnonzero(eligible)
        eligible[candidate[np.std(history[:, candidate], axis=0) == 0]] = False
        ids = np.flatnonzero(eligible)
        if len(ids) < max(2 * c["pairs"], c["k"] + 2):
            self.pca = self.factor = self.pair_model = None
            self.global_pairs = np.empty((0, 2), int)
            self.ids = ids
            return Decision(empty, eligible, None, 0, False)
        changed = not np.array_equal(ids, self.ids) or self.pca is None
        if changed:
            staleness = (
                self.pair_model.staleness
                if self.pair_model
                else c["formation_interval"]
            )
            self.pca = PCAModel(c["k"], c["pca_window"], c["pca_refit"])
            self.factor = FactorModel(
                "pca", c["k"], c["regression_window"], c["regression_refit"]
            )
            self.pair_model = PairsModel(
                "pca",
                selection="loadings",
                num_pairs=c["pairs"],
                formation_window=c["formation_window"],
                formation_interval=c["formation_interval"],
                spread_window=c["spread_window"],
            )
            # Preserve pair identity and the formation clock as the universe changes.
            lookup = np.full(returns.shape[1], -1, int)
            lookup[ids] = np.arange(len(ids))
            surviving = self.global_pairs[np.all(eligible[self.global_pairs], axis=1)]
            self.pair_model.pairs = lookup[surviving]
            self.pair_model.staleness = staleness
            self.ids = ids
        subset = history[:, ids]
        refit = self.pca.staleness >= c["pca_refit"]
        pca = self.pca.update(timestamp, subset, {})
        attribution = self.factor.update(timestamp, subset, {"pca": [pca]})
        pair_signal = self.pair_model.update(timestamp, subset, {"pca": [pca]})
        self.global_pairs = ids[pair_signal.pairs]
        if timestamp >= self.start:
            # Both recommended strategies trade daily and do not use current
            # positions to calculate their signal. Accounting supplies actual drift.
            scores = attribution.factor_intercept / attribution.residual_variance
            if np.isfinite(scores).all() and np.std(scores) > 1e-14:
                empty[0, ids] = self.tb.update(
                    timestamp, {"factor": [attribution]}, np.zeros(len(ids))
                )
            signals = (
                PairSignals(pairs=self.global_pairs, z_scores=pair_signal.z_scores)
                if len(self.global_pairs)
                else None
            )
            empty[1] = self.pairs.update(
                timestamp, {"pairs": [signals]}, np.zeros(returns.shape[1])
            )
            # Keep the original 1/(2*50) stock budget if missing data removes pairs.
            empty[1] *= len(self.global_pairs) / c["pairs"]
        assert np.isfinite(empty).all()
        assert np.max(np.abs(empty.sum(axis=1))) < 1e-8
        assert np.max(np.abs(empty).sum(axis=1)) <= 1 + 1e-8
        assert np.max(np.abs(attribution.betas @ empty[0, ids])) < 1e-8
        return Decision(
            empty, eligible, attribution.betas, len(self.global_pairs), refit
        )
