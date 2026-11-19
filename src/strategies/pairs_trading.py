import numpy as np
from typing import Any

from src.models.pairs import PairSignals
from src.strategies.strategy import Strategy

class PairsTrading(Strategy):
    pairs_name: str
    entry_threshold: float
    exit_threshold: float
    gross_leverage: float

    pairs: np.ndarray | None
    directions: np.ndarray | None

    def __init__(
        self,
        pairs_name: str,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        gross_leverage: float = 1.0,
    ) -> None:
        assert 0 <= exit_threshold and exit_threshold < entry_threshold and gross_leverage > 0

        self.pairs_name = pairs_name
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.gross_leverage = gross_leverage

        self.pairs = None
        self.directions = None

    def update(
        self,
        timestamp: np.datetime64,
        model_outputs: dict[str, Any],
        positions: np.ndarray,
    ) -> np.ndarray:
        signals: PairSignals | None = model_outputs[self.pairs_name][-1]
        if signals is None:
            self.pairs = None
            self.directions = None
            return np.zeros_like(positions)
        if self.pairs is None or not np.array_equal(self.pairs, signals.pairs):
            self.directions = self._carry_directions(signals.pairs)
            self.pairs = signals.pairs
        assert self.directions is not None

        # Direction 1 is long the first stock and short the second, -1 is the reverse
        z_scores = signals.z_scores
        self.directions[(self.directions == 1) & (z_scores > -self.exit_threshold)] = 0
        self.directions[(self.directions == -1) & (z_scores < self.exit_threshold)] = 0
        self.directions[(self.directions == 0) & (z_scores > self.entry_threshold)] = -1
        self.directions[(self.directions == 0) & (z_scores < -self.entry_threshold)] = 1

        # Commit a fixed slice of capital to each pair so idle pairs hold cash
        pair_weight = self.gross_leverage / (2 * len(self.pairs))
        weights = np.zeros_like(positions)
        np.add.at(weights, self.pairs[:, 0], self.directions * pair_weight)
        np.add.at(weights, self.pairs[:, 1], -self.directions * pair_weight)
        return weights

    def _carry_directions(self, pairs: np.ndarray) -> np.ndarray:
        # Keep open trades on pairs that survive re-formation and close the rest
        directions = np.zeros(len(pairs), dtype=np.int64)
        if self.pairs is None or self.directions is None:
            return directions
        previous = {(int(first), int(second)): direction for (first, second), direction in zip(self.pairs, self.directions)}
        for index, (first, second) in enumerate(pairs):
            directions[index] = previous.get((int(first), int(second)), 0)
        return directions