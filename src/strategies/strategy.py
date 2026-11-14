import numpy as np
from typing import Any, Protocol

class Strategy(Protocol):
    def update(
        self,
        timestamp: np.datetime64,
        model_outputs: dict[str, Any],
        positions: np.ndarray,
    ) -> np.ndarray:
        ...