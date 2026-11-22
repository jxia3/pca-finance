import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from src.data import DailyReturns
from src.engine import ModelEntry
from src.modules.simulator import Simulator
from src.strategies.long_short_snapback import CumulativeSnapback, LongShortSnapback
from src.strategies.residual_snapback import ResidualSnapback
from strategy_tests import (
    TRADE_FEE, Delayed, InSampleFactor, LatestEngine, net_returns, snapback, snapback_models, snapback_scores, to_series,
    weekly_rebalance_dates,
)


def synthetic_returns(num_days: int = 260, num_stocks: int = 40, seed: int = 0) -> DailyReturns:
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0003, 0.01, (num_days, 1))
    returns = market * rng.uniform(0.5, 1.5, num_stocks) + rng.normal(0.0, 0.015, (num_days, num_stocks))
    dates = np.datetime64("2020-01-01") + np.arange(num_days)
    dates = dates[np.is_busday(dates)][:num_days]
    returns = returns[:len(dates)]
    return DailyReturns(dates=dates, stock_ids=np.arange(num_stocks), returns=returns)


class SnapbackScoreTests(unittest.TestCase):
    def test_scores_match_strategy_selection(self):
        rng = np.random.default_rng(1)
        residuals = rng.normal(0.0, 0.02, (120, 30)) * rng.uniform(0.5, 2.0, 30)
        residuals[:10, 3] = np.nan
        residuals[70:73, :5] -= 0.12
        scores = snapback_scores(residuals)

        strategy = ResidualSnapback("factor", [np.datetime64("2020-01-01")])
        history = [SimpleNamespace(residuals=row) for row in residuals]
        for day in (62, 72, 90, 119):
            weights = strategy.update(np.datetime64("2020-01-01"), {"factor": history[:day + 1]}, np.zeros(30))
            ranked = np.argsort(np.where(np.isfinite(scores[day]), scores[day], np.inf), kind="stable")
            expected = [i for i in ranked[:strategy.max_positions] if scores[day, i] <= strategy.score_threshold]
            np.testing.assert_array_equal(np.flatnonzero(weights), np.sort(expected))
        # A score needs 63 finite residuals, stock 3 is missing its first 10
        self.assertTrue(np.isnan(scores[61]).all())
        self.assertTrue(np.isnan(scores[71, 3]))
        self.assertTrue(np.isfinite(scores[72, 3]))


class LongShortSnapbackTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        self.residuals = rng.normal(0.0, 0.02, (63, 200)) * rng.uniform(0.5, 2.0, 200)
        self.residuals[60:, :30] -= 0.06
        self.residuals[60:, 30:60] += 0.06
        self.betas = rng.normal(1.0, 0.5, (5, 200))
        self.history = [SimpleNamespace(residuals=row, betas=self.betas) for row in self.residuals]
        self.date = np.datetime64("2020-01-01")

    def weights(self, **options):
        strategy = LongShortSnapback("factor", [self.date], **options)
        return strategy.update(self.date, {"factor": self.history}, np.zeros(200))

    def test_hedged_book_is_factor_neutral_and_keeps_trade_signs(self):
        weights = self.weights(hedge_out_components=True)
        scores = snapback_scores(self.residuals)[-1]
        np.testing.assert_allclose(self.betas @ weights, 0.0, atol=1e-8)
        self.assertAlmostEqual(np.abs(weights).sum(), 1.0)
        self.assertTrue(np.all(scores[weights > 0] <= -2.0))
        self.assertTrue(np.all(scores[weights < 0] >= 2.0))
        self.assertLessEqual(np.count_nonzero(weights > 0), 40)
        self.assertLessEqual(np.count_nonzero(weights < 0), 40)

    def test_long_only_and_hybrid_match_their_parts(self):
        long_only = self.weights()
        base = ResidualSnapback("factor", [self.date]).update(self.date, {"factor": self.history}, np.zeros(200))
        np.testing.assert_array_equal(long_only, base)
        hedged = self.weights(hedge_out_components=True)
        np.testing.assert_allclose(self.weights(hybrid=True, hybrid_parameter=0.3), 0.3 * long_only + 0.7 * hedged)

    def test_cumulative_score_uses_signal_with_snapback_rules(self):
        rng = np.random.default_rng(3)
        score = rng.normal(0.0, 1.0, 200)
        score[:5] = [-4.0, -3.0, -2.5, np.nan, -2.1]
        outputs = {"factor": self.history, "signal": [SimpleNamespace(score=score)]}
        strategy = CumulativeSnapback("factor", "signal", [self.date])
        weights = strategy.update(self.date, outputs, np.zeros(200))
        finite = np.flatnonzero(np.isfinite(score))
        ranked = finite[np.argsort(score[finite], kind="stable")][:20]
        expected = np.sort(ranked[score[ranked] <= -2.0])
        np.testing.assert_array_equal(np.flatnonzero(weights), expected)
        self.assertEqual(weights[3], 0.0)
        hedged = CumulativeSnapback("factor", "signal", [self.date], hedge_out_components=True).update(
            self.date, outputs, np.zeros(200))
        np.testing.assert_allclose(self.betas @ hedged, 0.0, atol=1e-8)
        self.assertTrue(np.all(score[hedged > 0] <= -2.0) and np.all(score[hedged < 0] >= 2.0))

    def test_smoothed_volatility_percentile(self):
        daily_returns = synthetic_returns(num_days=400)
        dates = daily_returns.dates[[29, 30, 250]]
        allocations = LongShortSnapback("factor", dates, dynamic_vol=True,
                                        daily_returns=daily_returns)._original_fraction_by_date
        self.assertAlmostEqual(allocations[dates[0]], 0.5)  # No earlier readings
        volatility = np.lib.stride_tricks.sliding_window_view(daily_returns.returns.mean(axis=1), 30).std(
            axis=1, ddof=1)
        index = 250 - 29
        below = np.count_nonzero(volatility[:index] < volatility[index])
        self.assertAlmostEqual(allocations[dates[2]], 1.0 - (below + 126) / (index + 252))


class HarnessTests(unittest.TestCase):
    def test_weekly_dates_are_last_trading_day_of_each_week(self):
        dates = np.array(["2024-01-02", "2024-01-05", "2024-01-08", "2024-01-12", "2024-01-16"], dtype="datetime64[D]")
        np.testing.assert_array_equal(weekly_rebalance_dates(dates), dates[[1, 3, 4]])

    def test_delayed_trades_each_target_one_day_late_and_holds_otherwise(self):
        class Weekly:
            def __init__(self):
                self.day = 0

            def update(self, timestamp, model_outputs, positions):
                self.day += 1
                return np.full(2, float(self.day)) if self.day % 3 == 1 else positions

        delayed = Delayed(Weekly())
        held = np.zeros(2)
        outputs = [delayed.update(None, {}, held) for _ in range(5)]
        self.assertIs(outputs[0], held)
        np.testing.assert_array_equal(outputs[1], [1.0, 1.0])
        self.assertIs(outputs[2], held)
        self.assertIs(outputs[3], held)
        np.testing.assert_array_equal(outputs[4], [4.0, 4.0])

    def test_fees_applied_afterwards_match_simulator(self):
        daily_returns = synthetic_returns()
        runs = {}
        for fee in (0.0, TRADE_FEE):
            simulator = Simulator(daily_returns, snapback("lagged", daily_returns, max_positions=5,
                                                          score_threshold=-1.0), fee)
            models = [ModelEntry(entry.name, entry.model) for entry in snapback_models(pca_window=60,
                                                                                         regression_window=40)]
            LatestEngine(daily_returns, models, [simulator]).run()
            runs[fee] = simulator.get_history()
        series = to_series(runs[0.0])
        self.assertGreater(np.sum(series.turnover), 0.0)
        expected = runs[TRADE_FEE].total_value[1:] / runs[TRADE_FEE].total_value[:-1] - 1.0
        np.testing.assert_allclose(net_returns(series, TRADE_FEE), expected[series.start:], atol=1e-12)

    def test_in_sample_factor_residuals_use_full_sample_fit(self):
        daily_returns = synthetic_returns()
        model = InSampleFactor(daily_returns.returns, 2, start=50)
        self.assertTrue(np.isnan(model.update(None, daily_returns.returns[:49], {}).residuals).all())
        attribution = model.update(None, daily_returns.returns[:120], {})
        fitted = model.coefficients[0] + (daily_returns.returns[119] @ model.weights) @ model.coefficients[1:]
        np.testing.assert_allclose(attribution.residuals, daily_returns.returns[119] - fitted)


if __name__ == "__main__":
    unittest.main()
