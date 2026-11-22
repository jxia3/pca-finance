from typing import Any
import numpy as np
from scipy.optimize import minimize
from src.data import DailyReturns
from src.strategies.residual_snapback import ResidualSnapback


class LongShortSnapback(ResidualSnapback):
    """Residual snapback whose hedged portfolio is a factor-neutral long-short book.

    The hedged portfolio takes up to 2 * max_positions (40 by default) of the lowest scores at or below
    the threshold as long candidates and as many of the highest scores at or above minus the threshold as
    short candidates. It starts from 50% of capital in equal longs and 50% in equal shorts, then finds the
    closest weights with zero exposure to every factor, gross exposure 1 and each trade's sign preserved.
    The dynamic mix smooths the volatility percentile with pseudo_readings on each side, so it starts at 50/50.
    """

    pseudo_readings = 126

    def update(
        self,
        timestamp: np.datetime64,
        model_outputs: dict[str, Any],
        positions: np.ndarray,
    ) -> np.ndarray:
        if timestamp not in self.rebalance_dates:
            return positions

        target = np.zeros_like(positions)
        scored = self._scores(model_outputs)
        if scored is None:
            return target
        candidates, scores = scored
        history = model_outputs[self.factor_name]

        if self.dynamic_vol:
            original_fraction = self._original_fraction_by_date.get(timestamp, 0.5)
        elif self.hybrid:
            original_fraction = self.hybrid_parameter
        elif self.hedge_out_components:
            original_fraction = 0.0
        else:
            original_fraction = 1.0

        if original_fraction > 0.0:
            ranked = np.argsort(scores, kind="stable")[:self.max_positions]
            selected = candidates[ranked[scores[ranked] <= self.score_threshold]]
            if selected.size:
                target[selected] += original_fraction / selected.size
        if original_fraction < 1.0:
            target += (1.0 - original_fraction) * self._long_short_weights(
                scores, candidates, history[-1].betas, positions.size
            )
        return target

    def _scores(self, model_outputs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
        # Stocks with a valid score and their scores, None before enough history
        history = model_outputs[self.factor_name]
        required_days = self.volatility_days + self.signal_days
        if len(history) < required_days:
            return None
        residuals = np.stack([output.residuals for output in history[-required_days:]])
        valid = np.flatnonzero(np.all(np.isfinite(residuals), axis=0))
        volatility = np.std(residuals[:-self.signal_days, valid], axis=0, ddof=1)
        varying = volatility > 0
        candidates = valid[varying]
        recent = residuals[-self.signal_days:, candidates].sum(axis=0)
        return candidates, recent / (volatility[varying] * np.sqrt(self.signal_days))

    def _long_short_weights(
        self, scores: np.ndarray, candidates: np.ndarray, loadings: np.ndarray | None, num_stocks: int
    ) -> np.ndarray:
        weights = np.zeros(num_stocks)
        count = 2 * self.max_positions
        order = np.argsort(scores, kind="stable")
        longs = order[:count][scores[order[:count]] <= self.score_threshold]
        highest = order[::-1][:count]
        shorts = highest[scores[highest] >= -self.score_threshold]
        if not longs.size or not shorts.size or loadings is None:
            return weights
        stocks = candidates[np.r_[longs, shorts]]
        if not np.all(np.isfinite(loadings[:, stocks])):
            return weights

        # With signs fixed, w = signs * u for u >= 0, and gross exposure sum |w| = sum u is linear
        signs = np.r_[np.ones(longs.size), -np.ones(shorts.size)]
        start = np.r_[np.full(longs.size, 0.5 / longs.size), np.full(shorts.size, 0.5 / shorts.size)]
        exposures = loadings[:, stocks] * signs
        # Rescaling each factor row leaves B w = 0 unchanged and keeps the constraints well conditioned
        norms = np.linalg.norm(exposures, axis=1, keepdims=True)
        exposures = exposures / np.where(norms > 0, norms, 1.0)
        matrix = np.vstack((exposures, np.ones(stocks.size)))
        bounds = np.r_[np.zeros(len(exposures)), 1.0]
        result = minimize(
            lambda u: 0.5 * np.sum((u - start) ** 2),
            start,
            jac=lambda u: u - start,
            method="SLSQP",
            bounds=[(0.0, None)] * stocks.size,
            constraints={"type": "eq", "fun": lambda u: matrix @ u - bounds, "jac": lambda u: matrix},
            options={"ftol": 1e-12, "maxiter": 500},
        )
        solution = np.clip(result.x, 0.0, None)
        if not result.success or np.max(np.abs(matrix @ solution - bounds)) > 1e-6:
            return weights
        weights[stocks] = signs * solution / solution.sum()
        return weights

    def _volatility_allocations(self, daily_returns: DailyReturns) -> dict[np.datetime64, float]:
        window = 30
        market_returns = daily_returns.returns.mean(axis=1)
        trailing_returns = np.lib.stride_tricks.sliding_window_view(market_returns, window)
        volatility = trailing_returns.std(axis=1, ddof=1)
        allocations = {}
        for index, timestamp in enumerate(daily_returns.dates[window - 1:]):
            if timestamp not in self.rebalance_dates:
                continue
            earlier = volatility[:index]
            current = volatility[index]
            below = np.count_nonzero(earlier < current) + 0.5 * np.count_nonzero(earlier == current)
            # Laplace smoothing: pseudo readings on each side pull early percentiles towards 50%
            hedged_fraction = (below + self.pseudo_readings) / (index + 2 * self.pseudo_readings)
            allocations[timestamp] = 1.0 - hedged_fraction
        return allocations


class CumulativeSnapback(LongShortSnapback):
    """Snapback trading rules applied to the cumulative residual score.

    The score is today's running sum of residuals relative to its previous levels (ResidualSignalModel),
    so a stock that has drifted below the factor model over weeks qualifies even if its last three days
    were in line with the model.
    """

    def __init__(self, factor_name: str, signal_name: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(factor_name, *args, **kwargs)
        self.signal_name = signal_name

    def _scores(self, model_outputs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
        score = model_outputs[self.signal_name][-1].score
        candidates = np.flatnonzero(np.isfinite(score))
        return (candidates, score[candidates]) if candidates.size else None
