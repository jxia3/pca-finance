import numpy as np
from typing import Any

from src.models.factor import ReturnAttribution
from src.strategies.portfolio import neutralize, scale_gross
from src.strategies.strategy import Strategy

class AlphaQuintile(Strategy):
    factor_name: str
    quantile: float
    gross_leverage: float
    rebalance: int
    factor_neutral: bool
    contrarian: bool

    staleness: int

    def __init__(
        self,
        factor_name: str,
        quantile: float = 0.2,
        gross_leverage: float = 1.0,
        rebalance: int = 1,
        factor_neutral: bool = False,
        contrarian: bool = False,
    ) -> None:
        assert 0 < quantile and quantile <= 0.5 and gross_leverage > 0 and rebalance > 0

        self.factor_name = factor_name
        self.quantile = quantile
        self.gross_leverage = gross_leverage
        self.rebalance = rebalance
        self.factor_neutral = factor_neutral
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
        if not np.all(np.isfinite(alphas)):
            return np.zeros_like(positions)
        if self.staleness < self.rebalance:
            self.staleness += 1
            return positions

        # Equal weight long the top quantile of alpha and short the bottom quantile
        num_selected = max(1, int(len(alphas) * self.quantile))
        order = np.argsort(alphas)
        weights = np.zeros_like(positions)
        weights[order[-num_selected:]] = 1.0
        weights[order[:num_selected]] = -1.0
        if self.contrarian:
            # Bet that past alpha reverts: long the low alpha stocks and short the high alpha stocks
            weights = -weights
        if self.factor_neutral:
            weights = neutralize(weights, attribution.betas)

        self.staleness = 1
        return scale_gross(weights, self.gross_leverage)