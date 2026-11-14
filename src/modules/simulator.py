import numpy as np
from typing import Any

from src.modules.module import Module
from src.strategies.strategy import Strategy

class Simulator(Module):
    num_stocks: int
    strategy: Strategy
    positions: np.ndarray

    def __init__(self, num_stocks: int, strategy: Strategy) -> None:
        assert num_stocks > 0
        self.num_stocks = num_stocks
        self.strategy = strategy
        self.positions = np.zeros(num_stocks, dtype=np.float64)

    def update(self, timestamp: np.datetime64, returns: np.ndarray, model_outputs: dict[str, Any]) -> None:
        self.strategy.update(timestamp, model_outputs, self.positions)
        # TODO: position accounting