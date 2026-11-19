import unittest
from types import SimpleNamespace

import numpy as np

from src.strategies.factor_neutral_alpha import FactorNeutralAlpha


class OptimizerTests(unittest.TestCase):
    def run_strategy(self, alpha, betas, positions=None, **kwargs):
        attribution = SimpleNamespace(factor_intercept=np.array(alpha, dtype=float),
                                      residual_variance=np.full(len(alpha), 0.01),
                                      betas=np.array(betas, dtype=float))
        strategy = FactorNeutralAlpha("factor", position_cap=0.4, **kwargs)
        weights = strategy.update(np.datetime64("2020-01-01"), {"factor": [attribution]},
                                  np.zeros(len(alpha)) if positions is None else positions)
        return weights, strategy, attribution

    def test_analytic_unconstrained_solution(self):
        alpha = np.array([0.01, -0.01, -0.01, 0.01])
        weights, _, _ = self.run_strategy(alpha, [[-1, -1, 1, 1]], turnover_cost=0)
        np.testing.assert_allclose(weights, alpha / 0.1, atol=1e-6)

    def test_caps_gross_and_redundant_exposures(self):
        betas = np.array([[-1, -1, 1, 1], [-2, -2, 2, 2]])
        weights, _, _ = self.run_strategy([1, -1, -1, 1], betas, turnover_cost=0,
                                         gross_leverage=0.6)
        self.assertLessEqual(np.abs(weights).sum(), 0.6000001)
        self.assertLessEqual(np.abs(weights).max(), 0.4000001)
        self.assertAlmostEqual(weights.sum(), 0, places=7)
        np.testing.assert_allclose(betas @ weights, 0, atol=1e-7)
        self.assertGreater(weights[0], 0)

    def test_turnover_penalty_and_rebalance(self):
        previous = np.array([0.1, -0.1, -0.1, 0.1])
        weights, strategy, attribution = self.run_strategy(
            [0, 0, 0, 0], [[-1, -1, 1, 1]], positions=previous, turnover_cost=1)
        np.testing.assert_allclose(weights, previous, atol=1e-6)
        drifted = weights * 1.01
        result = strategy.update(np.datetime64("2020-01-02"), {"factor": [attribution]}, drifted)
        np.testing.assert_array_equal(result, drifted)
        attribution.factor_intercept[:] = np.nan
        np.testing.assert_array_equal(strategy.update(np.datetime64("2020-01-03"),
                                      {"factor": [attribution]}, drifted), np.zeros(4))


if __name__ == "__main__":
    unittest.main()
