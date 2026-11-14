import numpy as np
from typing import Any

from src.strategies.strategy import Strategy

class AlphaPersistence(Strategy):
    pca_name: str

    def __init__(self, pca_name: str) -> None:
        self.pca_name = pca_name

    def update(
        self,
        timestamp: np.datetime64,
        model_outputs: dict[str, Any],
        positions: np.ndarray,
    ) -> np.ndarray:
        print("Strategy:", timestamp)