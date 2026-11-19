import numpy as np
from typing import Any

from src.models.factor import ReturnAttribution
from src.strategies.portfolio import neutralize, scale_gross
from src.strategies.strategy import Strategy

class MultiHorizonAlpha(Strategy):
    short_factor_name: str
    long_factor_name: str
    gross_leverage: float
    rebalance: int
    factor_neutral: bool
    contrarian: bool
    score_limit: float

    staleness: int

    def __init__(
        self,
        short_factor_name: str,
        long_factor_name: str,
        gross_leverage: float = 1.0,
        rebalance: int = 1,
        factor_neutral: bool = True,
        contrarian: bool = True,
        score_limit: float = 3.0,
    ) -> None:
        assert gross_leverage > 0 and rebalance > 0 and score_limit > 0

        self.short_factor_name = short_factor_name
        self.long_factor_name = long_factor_name
        self.gross_leverage = gross_leverage
        self.rebalance = rebalance
        self.factor_neutral = factor_neutral
        self.contrarian = contrarian
        self.score_limit = score_limit

        self.staleness = rebalance

    def update(
        self,
        timestamp: np.datetime64,
        model_outputs: dict[str, Any],
        positions: np.ndarray,
    ) -> np.ndarray:
        short_attribution: ReturnAttribution = model_outputs[self.short_factor_name][-1]
        long_attribution: ReturnAttribution = model_outputs[self.long_factor_name][-1]
        short_alphas = short_attribution.factor_intercept
        long_alphas = long_attribution.factor_intercept
        if not np.all(np.isfinite(short_alphas)) or not np.all(np.isfinite(long_alphas)):
            return np.zeros_like(positions)
        if self.staleness < self.rebalance:
            self.staleness += 1
            return positions

        short_scores = self._z_scores(short_alphas)
        long_scores = self._z_scores(long_alphas)
        # Only trade stocks where both horizons agree on the sign of alpha, sized by the weaker signal
        agree = np.sign(short_scores) == np.sign(long_scores)
        scores = np.where(agree, np.sign(long_scores) * np.minimum(np.abs(short_scores), np.abs(long_scores)), 0.0)
        if self.contrarian:
            # Bet that past alpha reverts: long the low alpha stocks and short the high alpha stocks
            scores = -scores
        weights = neutralize(scores, long_attribution.betas if self.factor_neutral else None)

        self.staleness = 1
        return scale_gross(weights, self.gross_leverage)

    def _z_scores(self, alphas: np.ndarray) -> np.ndarray:
        z_scores = (alphas - np.mean(alphas)) / np.std(alphas)
        return np.clip(z_scores, -self.score_limit, self.score_limit)