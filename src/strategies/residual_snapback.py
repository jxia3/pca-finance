from collections.abc import Iterable
from typing import Any
import numpy as np
from scipy.optimize import LinearConstraint, minimize
from src.data import DailyReturns
from src.strategies.strategy import Strategy


class ResidualSnapback(Strategy):
    signal_days = 3
    volatility_days = 60
    score_threshold = -2.0
    max_positions = 20
    max_hedged_positions_per_side = 40

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
            original_fraction = self._original_fraction_by_date.get(timestamp, 0.5)
        elif self.hybrid:
            original_fraction = self.hybrid_parameter
        elif self.hedge_out_components:
            original_fraction = 0.0
        else:
            original_fraction = 1.0

        if original_fraction == 1.0:
            return target
        hedged = self._hedged_weights(candidates, scores, history[-1].factor_loadings, target.size)
        return original_fraction * target + (1.0 - original_fraction) * hedged

    def _volatility_allocations(self, daily_returns: DailyReturns) -> dict[np.datetime64, float]:
        window = 30
        prior_per_side = 126
        allocations = {}
        if len(daily_returns.dates) < window:
            return allocations

        market_returns = daily_returns.returns.mean(axis=1)
        trailing_returns = np.lib.stride_tricks.sliding_window_view(market_returns, window)
        volatility = trailing_returns.std(axis=1, ddof=1)
        for index, timestamp in enumerate(daily_returns.dates[window - 1:]):
            if timestamp not in self.rebalance_dates:
                continue
            earlier = volatility[:index]
            current = volatility[index]
            below = np.count_nonzero(earlier < current)
            tied = np.count_nonzero(earlier == current)
            # Give tied readings half weight in the historical percentile rank.
            hedged_fraction = (below + 0.5 * tied + prior_per_side) / (
                index + 2 * prior_per_side
            )
            allocations[timestamp] = 1.0 - hedged_fraction
        return allocations

    def _hedged_weights(
        self, candidates: np.ndarray, scores: np.ndarray,
        loadings: np.ndarray | None, num_stocks: int,
    ) -> np.ndarray:
        target = np.zeros(num_stocks)
        if loadings is None:
            return target

        valid = np.isfinite(scores) & np.all(np.isfinite(loadings[:, candidates]), axis=0)
        candidates, scores = candidates[valid], scores[valid]
        count = self.max_hedged_positions_per_side
        long_order = np.argsort(scores, kind="stable")
        short_order = np.argsort(-scores, kind="stable")
        longs = candidates[long_order[scores[long_order] < 0][:count]]
        shorts = candidates[short_order[scores[short_order] > 0][:count]]
        if not longs.size or not shorts.size:
            return target

        selected = np.concatenate((longs, shorts))
        signs = np.repeat([1.0, -1.0], [longs.size, shorts.size])
        desired = np.repeat([0.5 / longs.size, 0.5 / shorts.size], [longs.size, shorts.size])
        exposures = loadings[:, selected] * signs
        _, singular, vectors = np.linalg.svd(exposures, full_matrices=False)
        tolerance = np.finfo(float).eps * max(exposures.shape) * singular[0]
        basis = vectors[singular > tolerance]
        # Optimize nonnegative magnitudes, keeping the score-implied signs fixed.
        constraints = np.vstack((np.ones(selected.size), basis))
        rhs = np.zeros(len(constraints))
        rhs[0] = 1.0

        def objective(magnitudes: np.ndarray) -> tuple[float, np.ndarray]:
            difference = magnitudes - desired
            return 0.5 * float(difference @ difference), difference

        solution = minimize(
            objective, desired, jac=True, method="SLSQP", bounds=[(0, 1)] * selected.size,
            constraints=LinearConstraint(constraints, rhs, rhs),
            options={"ftol": 1e-10},
        )
        magnitudes = np.clip(solution.x, 0, 1)
        if solution.success and np.max(np.abs(constraints @ magnitudes - rhs)) <= 1e-8:
            weights = signs * magnitudes
            if np.max(np.abs(loadings[:, selected] @ weights)) <= 1e-8:
                target[selected] = weights
        return target
