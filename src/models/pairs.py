from dataclasses import dataclass
import numpy as np
from typing import Any

from src.models.model import Model
from src.models.pca import CorrelationPCA

@dataclass(kw_only=True)
class PairSignals:
    pairs: np.ndarray
    z_scores: np.ndarray

class PairsModel(Model):
    pca_name: str
    selection: str
    num_pairs: int
    formation_window: int
    formation_interval: int
    spread_window: int

    pairs: np.ndarray | None
    staleness: int

    def __init__(
        self,
        pca_name: str,
        selection: str = "loadings",
        num_pairs: int = 50,
        formation_window: int = 250,
        formation_interval: int = 21,
        spread_window: int = 60,
    ) -> None:
        assert selection in ("loadings", "correlation")
        assert num_pairs > 0 and formation_window > 1 and formation_interval > 0 and spread_window > 1

        self.pca_name = pca_name
        self.selection = selection
        self.num_pairs = num_pairs
        self.formation_window = formation_window
        self.formation_interval = formation_interval
        self.spread_window = spread_window

        self.pairs = None
        self.staleness = formation_interval

    def update(
        self,
        timestamp: np.datetime64,
        returns: np.ndarray,
        model_outputs: dict[str, Any],
    ) -> PairSignals | None:
        assert returns.ndim == 2
        return_pca: CorrelationPCA | None = model_outputs[self.pca_name][-1]
        if return_pca is None or len(returns) < max(self.formation_window, self.spread_window):
            return None

        if self.staleness >= self.formation_interval:
            if self.selection == "loadings":
                # Stocks with similar exposure to every principal component have a nearly factor neutral spread
                loadings = return_pca.components * np.sqrt(return_pca.eigenvalues)
                squared_norms = np.sum(loadings ** 2, axis=1)
                distances = squared_norms[:, np.newaxis] + squared_norms[np.newaxis, :] - 2.0 * loadings @ loadings.T
            else:
                distances = -np.corrcoef(returns[-self.formation_window:], rowvar=False)
            self.pairs = match_pairs(distances, self.num_pairs)
            self.staleness = 0

        assert self.pairs is not None
        self.staleness += 1
        # Spread is the log price ratio of each pair over the spread window
        log_prices = np.cumsum(np.log1p(returns[-self.spread_window:]), axis=0)
        spreads = log_prices[:, self.pairs[:, 0]] - log_prices[:, self.pairs[:, 1]]
        std_devs = np.std(spreads, axis=0, ddof=1)
        z_scores = np.where(std_devs > 0, (spreads[-1] - np.mean(spreads, axis=0)) / std_devs, 0.0)

        return PairSignals(pairs=self.pairs, z_scores=z_scores)

def match_pairs(distances: np.ndarray, num_pairs: int) -> np.ndarray:
    assert distances.ndim == 2 and distances.shape[0] == distances.shape[1]
    num_stocks = distances.shape[0]
    assert 2 * num_pairs <= num_stocks

    # Greedily take the closest remaining pair so each stock is in at most one pair
    rows, columns = np.triu_indices(num_stocks, k=1)
    order = np.argsort(distances[rows, columns], kind="stable")
    used = np.zeros(num_stocks, dtype=bool)
    pairs = []
    for index in order:
        first, second = rows[index], columns[index]
        if used[first] or used[second]:
            continue
        used[first] = used[second] = True
        pairs.append((first, second))
        if len(pairs) == num_pairs:
            break

    return np.array(pairs, dtype=np.int64)