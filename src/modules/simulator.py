from dataclasses import dataclass
import numpy as np
from typing import Any

from src.data import DailyReturns
from src.modules.module import Module
from src.strategies.strategy import Strategy

@dataclass(kw_only=True)
class PortfolioHistory:
    total_value: np.ndarray
    positions: np.ndarray
    trades: np.ndarray
    fees: np.ndarray

class Simulator(Module):
    num_stocks: int
    strategy: Strategy
    trade_fee: float

    total_value: float
    positions: np.ndarray
    history: PortfolioHistory
    history_index: int

    def __init__(self, daily_returns: DailyReturns, strategy: Strategy, trade_fee: float) -> None:
        num_days, num_stocks = daily_returns.returns.shape
        self.num_stocks = num_stocks
        self.strategy = strategy
        self.trade_fee = trade_fee

        self.total_value = 1.0
        self.positions = np.zeros(num_stocks, dtype=np.float64)
        self.history = PortfolioHistory(
            total_value=np.zeros(num_days + 1, dtype=np.float64),
            positions=np.zeros((num_days + 1, num_stocks), dtype=np.float64),
            trades=np.zeros((num_days + 1, num_stocks), dtype=np.float64),
            fees=np.zeros(num_days + 1, dtype=np.float64),
        )
        self.history.total_value[0] = self.total_value
        self.history_index = 1

    def get_history(self) -> PortfolioHistory:
        return self.history

    def update(self, timestamp: np.datetime64, returns: np.ndarray, model_outputs: dict[str, Any]) -> None:
        if self.total_value <= 0.0:
            return

        current_returns = returns[-1]
        assert current_returns.shape == (self.num_stocks,)
        growth = 1.0 + float(self.positions @ current_returns)
        self.total_value *= growth
        if self.total_value <= 0.0:
            self._terminate()
            return

        current_positions = self.positions * (1.0 + current_returns) / growth
        target_positions = self.strategy.update(timestamp, model_outputs, current_positions.copy())
        assert target_positions.shape == (self.num_stocks,)
        assert np.all(np.isfinite(target_positions))

        trades = target_positions - current_positions
        fees = self.total_value * float(np.sum(np.abs(trades))) * self.trade_fee
        self.total_value -= fees
        self.positions = target_positions.copy()

        self.history.total_value[self.history_index] = self.total_value
        self.history.positions[self.history_index] = self.positions
        self.history.trades[self.history_index] = trades
        self.history.fees[self.history_index] = fees

        if self.total_value <= 0.0:
            self._terminate()
            return
        self.history_index += 1

    def _terminate(self) -> None:
        self.positions = np.zeros_like(self.positions)
        self.history.total_value[self.history_index:] = self.total_value
        self.history.positions[self.history_index:].fill(0.0)
        self.history.trades[self.history_index + 1:].fill(0.0)
        self.history.fees[self.history_index + 1:].fill(0.0)