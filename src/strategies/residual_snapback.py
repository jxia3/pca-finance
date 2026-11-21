from collections.abc import Iterable
from typing import Any
import numpy as np
from src.data import DailyReturns
from src.strategies.strategy import Strategy


class ResidualSnapback(Strategy):
    signal_days = 3
    volatility_days = 60
    score_threshold = -2.0
    max_positions = 20

    def __init__(
        self,
        factor_name: str,
        rebalance_dates: Iterable[np.datetime64],
        hedge_out_components: bool = False,
        *,
        hybrid: bool = False,
        hybrid_parameter: float = 0.5,
        dynamic_vol: bool = False,
        daily_returns: DailyReturns | None = None,
    ) -> None:
        """In hybrid mode, allocate hybrid_parameter to the original portfolio."""
        if not 0.0 <= hybrid_parameter <= 1.0:
            raise ValueError("hybrid_parameter must be between 0 and 1.")
        self.factor_name = factor_name
        self.rebalance_dates = frozenset(rebalance_dates)
        self.hedge_out_components = hedge_out_components
        self.hybrid = hybrid
        self.hybrid_parameter = hybrid_parameter
        self.dynamic_vol = dynamic_vol
        self._original_fraction_by_date: dict[np.datetime64, float] = {}
        if dynamic_vol:
            if daily_returns is None:
                raise ValueError("daily_returns is required when dynamic_vol is enabled.")
            self._original_fraction_by_date = self._volatility_allocations(daily_returns)

    def update(
        self,
        timestamp: np.datetime64,
        model_outputs: dict[str, Any],
        positions: np.ndarray,
    ) -> np.ndarray:
        if timestamp not in self.rebalance_dates:
            return positions

        target = np.zeros_like(positions)
        history = model_outputs[self.factor_name]
        required_days = self.volatility_days + self.signal_days
        if len(history) < required_days:
            return target

        residuals = np.stack([output.residuals for output in history[-required_days:]])
        valid = np.flatnonzero(np.all(np.isfinite(residuals), axis=0))
        prior = residuals[:-self.signal_days, valid]
        volatility = np.std(prior, axis=0, ddof=1)
        varying = volatility > 0
        candidates = valid[varying]
        recent = residuals[-self.signal_days:, candidates].sum(axis=0)
        scores = recent / (volatility[varying] * np.sqrt(self.signal_days))
        ranked = np.argsort(scores, kind="stable")[:self.max_positions]
        selected = candidates[ranked[scores[ranked] <= self.score_threshold]]
        if selected.size:
            target[selected] = 1.0 / selected.size
            if self.dynamic_vol:
                original_fraction = self._original_fraction_by_date.get(timestamp, 1.0)
            elif self.hybrid:
                original_fraction = self.hybrid_parameter
            elif self.hedge_out_components:
                original_fraction = 0.0
            else:
                original_fraction = 1.0
            if original_fraction < 1.0:
                hedged = self._hedged_weights(
                    target[selected], history[-1].factor_loadings, selected
                )
                target[selected] = (
                    original_fraction * target[selected]
                    + (1.0 - original_fraction) * hedged
                )
        return target

    def _volatility_allocations(self, daily_returns: DailyReturns) -> dict[np.datetime64, float]:
        window = 30
        min_history = 252
        allocations = {}
        if len(daily_returns.dates) < window + min_history:
            return allocations

        market_returns = daily_returns.returns.mean(axis=1)
        trailing_returns = np.lib.stride_tricks.sliding_window_view(market_returns, window)
        volatility = trailing_returns.std(axis=1, ddof=1)
        for index, timestamp in enumerate(daily_returns.dates[window - 1:]):
            if timestamp not in self.rebalance_dates or index < min_history:
                continue
            earlier = volatility[:index]
            current = volatility[index]
            below = np.count_nonzero(earlier < current)
            tied = np.count_nonzero(earlier == current)
            # Give tied readings half weight in the historical percentile rank.
            hedged_fraction = (below + 0.5 * tied) / index
            allocations[timestamp] = 1.0 - hedged_fraction
        return allocations

    @staticmethod
    def _hedged_weights(
        weights: np.ndarray, loadings: np.ndarray | None, selected: np.ndarray
    ) -> np.ndarray:
        if loadings is None or not np.all(np.isfinite(loadings[:, selected])):
            return np.zeros_like(weights)

        # Project equal weights onto the null space of the selected stocks' loadings.
        exposures = loadings[:, selected]
        _, singular_values, right_vectors = np.linalg.svd(exposures, full_matrices=True)
        tolerance = np.finfo(exposures.dtype).eps * max(exposures.shape)
        rank = np.count_nonzero(singular_values > tolerance * singular_values[0])
        null_space = right_vectors[rank:]
        hedged = null_space.T @ (null_space @ weights)
        gross = np.abs(hedged).sum()
        if gross <= tolerance:
            return np.zeros_like(weights)
        return hedged / gross
