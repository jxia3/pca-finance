import numpy as np
from typing import Any

from src.holdout_strategies.factor import ReturnAttribution
from src.holdout_strategies.portfolio import neutralize, scale_gross
from src.strategies.strategy import Strategy

class TreynorBlack(Strategy):
    factor_name: str
    gross_leverage: float
    rebalance: int
    factor_neutral: bool
    score_limit: float
    contrarian: bool

    staleness: int

    def __init__(
        self,
        factor_name: str,
        gross_leverage: float = 1.0,
        rebalance: int = 1,
        factor_neutral: bool = True,
        score_limit: float = 3.0,
        contrarian: bool = False,
    ) -> None:
        assert gross_leverage > 0 and rebalance > 0 and score_limit > 0

        self.factor_name = factor_name
        self.gross_leverage = gross_leverage
        self.rebalance = rebalance
        self.factor_neutral = factor_neutral
        self.score_limit = score_limit
        self.contrarian = contrarian

        self.staleness = rebalance

    def update(
        self,
        timestamp: np.datetime64,
        model_outputs: dict[str, Any],
        positions: np.ndarray,
    ) -> np.ndarray:
        attribution: ReturnAttribution = model_outputs[self.factor_name][-1]
        alphas = attribution.factor_intercept
        residual_variance = attribution.residual_variance
        if not np.all(np.isfinite(alphas)) or not np.all(residual_variance > 0):
            return np.zeros_like(positions)
        if self.staleness < self.rebalance:
            self.staleness += 1
            return positions

        # Weight each stock by alpha per unit of idiosyncratic variance
        scores = alphas / residual_variance
        # Winsorize the cross-sectional z-scores so a few stocks cannot dominate the book
        z_scores = (scores - np.mean(scores)) / np.std(scores)
        z_scores = np.clip(z_scores, -self.score_limit, self.score_limit)
        if self.contrarian:
            # Bet that past alpha reverts: long the low alpha stocks and short the high alpha stocks
            z_scores = -z_scores
        weights = neutralize(z_scores, attribution.betas if self.factor_neutral else None)

        self.staleness = 1
        return scale_gross(weights, self.gross_leverage)
