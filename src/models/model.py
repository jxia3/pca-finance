import numpy as np
from typing import Any, Protocol

class Model(Protocol):
    def update(self, timestamp: np.datetime64, returns: np.ndarray, model_outputs: dict[str, Any]) -> Any:
        ...