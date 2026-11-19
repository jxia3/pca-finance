"""Constrained alpha allocation with diagonal residual risk and L1 turnover."""

from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from src.strategies.strategy import Strategy


class FactorNeutralAlpha(Strategy):
    """Maximize alpha - risk_aversion/2 * residual variance - turnover cost.

    Inputs are daily return estimates. turnover_cost is a penalty per dollar
    traded (not a fee deduction; the simulator charges actual fees separately).
    Gross exposure is an upper bound, so the strategy may retain cash.
    Neutrality holds at rebalances; return drift can change exposures between them.
    """

    def __init__(self, factor_name: str, risk_aversion: float = 10.0,
                 turnover_cost: float = 0.0005, gross_leverage: float = 1.0,
                 position_cap: float = 0.05, rebalance: int = 5) -> None:
        values = [risk_aversion, turnover_cost, gross_leverage, position_cap]
        if (not np.all(np.isfinite(values)) or risk_aversion <= 0
                or turnover_cost < 0 or gross_leverage <= 0 or position_cap <= 0
                or not isinstance(rebalance, int) or rebalance < 1):
            raise ValueError("Invalid optimizer parameters")
        self.factor_name = factor_name
        self.risk_aversion = risk_aversion
        self.turnover_cost = turnover_cost
        self.gross_leverage = gross_leverage
        self.position_cap = position_cap
        self.rebalance = rebalance
        self.staleness = rebalance

    def update(self, timestamp: np.datetime64, model_outputs: dict[str, Any],
               positions: np.ndarray) -> np.ndarray:
        attribution = model_outputs[self.factor_name][-1]
        alpha = attribution.factor_intercept
        variance = attribution.residual_variance
        betas = attribution.betas
        if (not all(np.all(np.isfinite(x)) for x in (alpha, variance, betas))
                or np.any(variance <= 0)):
            self.staleness = self.rebalance
            return np.zeros_like(positions)
        if self.staleness < self.rebalance:
            self.staleness += 1
            return positions.copy()

        n = len(positions)
        # Orthonormal rows avoid numerical scaling problems and redundant factors.
        exposures = np.vstack((np.ones(n), betas))
        _, singular, vectors = np.linalg.svd(exposures, full_matrices=False)
        rank = np.sum(singular > singular[0] * max(exposures.shape) * np.finfo(float).eps)
        exposure_rows = vectors[:rank]

        # x = [weights, absolute weights, absolute trades]. Linear epigraph
        # constraints represent both L1 terms exactly, including at zero.
        eye = np.eye(n)
        zero = np.zeros((n, n))
        inequalities = np.vstack((
            np.hstack((-eye, eye, zero)), np.hstack((eye, eye, zero)),
            np.hstack((-eye, zero, eye)), np.hstack((eye, zero, eye)),
            np.concatenate((np.zeros(n), np.ones(n), np.zeros(n)))[None, :],
        ))
        lower = np.concatenate((np.zeros(2*n), -positions, positions, [-np.inf]))
        upper = np.concatenate((np.full(4*n, np.inf), [self.gross_leverage]))
        constraints = [
            LinearConstraint(np.hstack((exposure_rows, np.zeros((rank, 2*n)))), 0, 0),
            LinearConstraint(inequalities, lower, upper),
        ]
        bounds = Bounds(np.concatenate((np.full(n, -self.position_cap), np.zeros(2*n))),
                        np.concatenate((np.full(n, self.position_cap), np.full(2*n, np.inf))))

        # Scaling improves solver accuracy for daily-return-sized objectives.
        scale = max(float(np.max(np.abs(alpha))), self.turnover_cost, 1e-6)

        def objective(x: np.ndarray) -> float:
            w = x[:n]
            return float((-alpha @ w + 0.5*self.risk_aversion*np.sum(variance*w*w)
                          + self.turnover_cost*np.sum(x[2*n:])) / scale)

        def gradient(x: np.ndarray) -> np.ndarray:
            return np.concatenate(((-alpha + self.risk_aversion*variance*x[:n]),
                                   np.zeros(n), np.full(n, self.turnover_cost))) / scale

        initial = np.concatenate((np.zeros(2*n), np.abs(positions)))
        result = minimize(objective, initial, jac=gradient, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"ftol": 1e-9, "maxiter": 500})
        weights = result.x[:n]
        if (not result.success or not np.all(np.isfinite(weights))
                or np.max(np.abs(exposure_rows @ weights)) > 1e-7
                or np.sum(np.abs(weights)) > self.gross_leverage + 1e-7
                or np.max(np.abs(weights)) > self.position_cap + 1e-7):
            raise RuntimeError(f"Alpha optimization failed: {result.message}")
        self.staleness = 1
        return weights.copy()
