"""Synthetic checks only. These tests do not evaluate holdout performance."""

import unittest

import numpy as np

from src.data import DailyReturns
from src.holdout.accounting import HoldoutSimulator
from src.holdout.models import HoldoutModels
from src.holdout_strategies.factor import FactorModel
from src.holdout_strategies.pairs_model import PairsModel
from src.holdout_strategies.pairs_trading import PairsTrading
from src.holdout_strategies.treynor_black import TreynorBlack
from src.models.pca import PCAModel


class AccountingTests(unittest.TestCase):
    def make_book(self):
        data = DailyReturns(
            dates=np.arange("2020-01-01", "2020-01-05", dtype="datetime64[D]"),
            stock_ids=np.arange(3),
            returns=np.zeros((4, 3)),
        )
        return HoldoutSimulator(data, None, 0.0005)

    def test_entry_fee_and_self_financing(self):
        book = self.make_book()
        book.update(None, np.zeros((1, 3)), {"target": np.array([0.5, -0.5, 0])})
        self.assertAlmostEqual(book.total_value, 1 / 1.0005, places=12)
        self.assertAlmostEqual(book.total_value + book.history.fees[1], 1, places=12)
        np.testing.assert_allclose(book.positions, [0.5, -0.5, 0])

    def test_frozen_long_and_short_dollars_survive_fees(self):
        for sign in (1, -1):
            book = self.make_book()
            book.update(
                None, np.zeros((1, 3)), {"target": np.array([0.4, -0.4, 0]) * sign}
            )
            old_dollars = book.total_value * book.positions[0]
            old_value = book.total_value
            old_weights = book.positions.copy()
            raw = np.array([np.nan, 0.03, -0.02])
            book.update(None, raw[None, :], {"target": np.array([0, 0.5, -0.5])})
            self.assertAlmostEqual(
                book.total_value * book.positions[0], old_dollars, places=12
            )
            self.assertEqual(book.history.trades[2, 0], 0)
            self.assertAlmostEqual(np.abs(book.positions).sum(), 1, places=12)
            self.assertAlmostEqual(
                book.total_value + book.history.fees[2],
                old_value * (1 + old_weights @ np.nan_to_num(raw)),
                places=12,
            )
            book.update(None, np.array([[0.02, 0, 0]]), {"target": np.zeros(3)})
            np.testing.assert_array_equal(book.positions, 0)
            self.assertNotEqual(book.history.trades[3, 0], 0)


class ModelTests(unittest.TestCase):
    def test_clean_data_matches_unadapted_source_strategies(self):
        config = {
            "k": 2,
            "pca_window": 30,
            "pca_refit": 5,
            "regression_window": 20,
            "regression_refit": 1,
            "score_cap": 3.0,
            "pairs": 4,
            "formation_window": 20,
            "formation_interval": 7,
            "spread_window": 10,
            "entry": 2.0,
            "exit": 0.5,
        }
        values = np.random.default_rng(73).normal(0, 0.01, (70, 20))
        dates = np.arange("2020-01-01", "2020-03-11", dtype="datetime64[D]")
        adapter = HoldoutModels(config, np.ones(20, bool), dates[30])
        pca = PCAModel(2, 30, 5)
        factor = FactorModel("pca", 2, 20, 1)
        pairs = PairsModel(
            "pca",
            num_pairs=4,
            formation_window=20,
            formation_interval=7,
            spread_window=10,
        )
        tb = TreynorBlack("factor", contrarian=True)
        trading = PairsTrading("pairs")
        for i, date in enumerate(dates):
            history = values[: i + 1]
            actual = adapter.update(date, history, {})
            if i < 29:
                continue
            components = pca.update(date, history, {})
            attribution = factor.update(date, history, {"pca": [components]})
            signals = pairs.update(date, history, {"pca": [components]})
            if i >= 30:
                expected = np.array(
                    [
                        tb.update(date, {"factor": [attribution]}, np.zeros(20)),
                        trading.update(date, {"pairs": [signals]}, np.zeros(20)),
                    ]
                )
                np.testing.assert_allclose(actual.weights, expected, atol=1e-7)

    def test_gap_exclusion_reentry_and_prefix_invariance(self):
        config = {
            "k": 2,
            "pca_window": 30,
            "pca_refit": 5,
            "regression_window": 20,
            "regression_refit": 1,
            "score_cap": 3.0,
            "pairs": 4,
            "formation_window": 20,
            "formation_interval": 7,
            "spread_window": 10,
            "entry": 2.0,
            "exit": 0.5,
        }
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, (95, 20))
        returns[40, 0] = np.nan
        dates = np.arange("2020-01-01", "2020-04-05", dtype="datetime64[D]")

        def evaluate(values):
            model = HoldoutModels(config, np.ones(20, bool), dates[30])
            return [
                model.update(date, values[: i + 1], {}) for i, date in enumerate(dates)
            ]

        baseline = evaluate(returns)
        altered = returns.copy()
        altered[80:] *= 3
        other = evaluate(altered)
        self.assertTrue(baseline[39].eligible[0])
        self.assertFalse(baseline[40].eligible[0])
        self.assertFalse(baseline[69].eligible[0])
        self.assertTrue(baseline[70].eligible[0])
        for first, second in zip(baseline[:80], other[:80]):
            np.testing.assert_allclose(first.weights, second.weights, atol=1e-7)
        for decision in baseline[40:70]:
            np.testing.assert_array_equal(decision.weights[:, 0], 0)


if __name__ == "__main__":
    unittest.main()
