from typing import Any

import numpy as np
from scipy.optimize import minimize

from src.strategies.strategy import Strategy


def neutral_weights(
    target: np.ndarray, betas: np.ndarray, max_weight: float = 0.02) -> np.ndarray:
    """Project desired weights onto neutrality and sign bounds via a small dual.

    Positions may only shrink, preserving eligibility, direction and gross limits.
    Cash is always feasible. Failed numerical neutrality checks return cash.
    """
    result = np.zeros_like(target)
    if not np.isfinite(target).all() or max_weight <= 0:
        raise ValueError("Targets must be finite and max_weight must be positive.")
    selected = np.flatnonzero((target != 0) & np.isfinite(betas).all(axis=1))

    if len(selected) <= betas.shape[1] + 1:
        return result

    exposures = np.column_stack((np.ones(len(selected)), betas[selected]))
    u, singular, _ = np.linalg.svd(exposures, full_matrices=False)
    basis = u[:, singular > singular[0] * 1e-10]
    scale = np.max(np.abs(target[selected]))
    desired = target[selected] / scale
    bound = np.minimum(np.abs(desired), max_weight / scale)
    lower = np.where(desired < 0, -bound, 0)
    upper = np.where(desired > 0, bound, 0)

    def objective(multiplier: np.ndarray) -> tuple[float, np.ndarray]:
        shift = basis @ multiplier
        weight = np.clip(desired - shift, lower, upper)
        loss = -0.5 * np.sum((weight - desired) ** 2) - shift @ weight
        return float(loss), -basis.T @ weight

    fit = minimize(
        objective,
        np.zeros(basis.shape[1]),
        jac=True,
        method="L-BFGS-B",
        options={"gtol": 1e-10, "ftol": 1e-15, "maxiter": 200},
    )
    weight = np.clip(desired - basis @ fit.x, lower, upper) * scale

    if np.max(np.abs(exposures.T @ weight)) <= 1e-8:
        result[selected] = weight

    return result


class ResidualReversion(Strategy):
    def __init__(
        self,
        factor_name: str,
        signal_name: str,
        mode: str = "rank",
        forecast_name: str = "forecast",
        fraction: float = 0.2,
        entry: float = 1.5,
        exit: float = 0.5,
        gross: float = 1.0,
        max_weight: float = 0.02,
        min_edge: float = 0.0,
        trade_fee: float = 0.0005,
    ) -> None:
        assert mode in ("rank", "threshold", "calibrated", "alpha")
        assert 0 < fraction <= 0.5 and 0 <= exit < entry
        assert gross > 0 and max_weight > 0 and min_edge >= 0 and trade_fee >= 0
        self.factor_name = factor_name
        self.signal_name = signal_name
        self.mode = mode
        self.forecast_name = forecast_name
        self.fraction = fraction
        self.entry = entry
        self.exit = exit
        self.gross = gross
        self.max_weight = max_weight
        self.min_edge = min_edge
        self.trade_fee = trade_fee

    def update(
        self,
        timestamp: np.datetime64,
        model_outputs: dict[str, Any],
        positions: np.ndarray,
    ) -> np.ndarray:
        factor = model_outputs[self.factor_name][-1]
        score = model_outputs[self.signal_name][-1].score
        direction = np.zeros_like(positions)
        valid = np.isfinite(score) & np.isfinite(factor.betas).all(axis=1)

        if self.mode in ("rank", "alpha"):
            signal = -score if self.mode == "rank" else factor.factor_intercept
            selected = np.flatnonzero(valid & np.isfinite(signal))
            count = int(len(selected) * self.fraction)

            if count == 0:
                return direction

            order = selected[np.argsort(signal[selected], kind="stable")]
            direction[order[:count]] = -1
            direction[order[-count:]] = 1
        elif self.mode == "threshold":
            keep_long = (positions > 1e-10) & (score < -self.exit)
            keep_short = (positions < -1e-10) & (score > self.exit)
            direction[valid & ((score < -self.entry) | keep_long)] = 1
            direction[valid & ((score > self.entry) | keep_short)] = -1
        else:
            prediction = model_outputs[self.forecast_name][-1].expected_return
            eligible = valid & np.isfinite(prediction)
            eligible &= np.abs(prediction) > self.min_edge + 2 * self.trade_fee
            direction[eligible] = np.sign(prediction[eligible])

        # Fixed stock budgets let sparse threshold signals leave capital idle.
        budget = self.gross / max(1, 2 * self.fraction * len(positions))
        target = direction * min(budget, self.max_weight)
        target *= min(1.0, self.gross / max(np.abs(target).sum(), 1e-16))
        return neutral_weights(target, factor.betas, self.max_weight)
