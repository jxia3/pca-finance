from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.models.model import Model


@dataclass(kw_only=True)
class ResidualSignal:
    level: np.ndarray
    score: np.ndarray
    half_life: np.ndarray


def ou_score(history: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit daily AR(1) levels, rejecting invalid continuous-time OU fits."""
    x, y = history[:-1], history[1:]
    centered = x - x.mean(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.sum(centered * (y - y.mean(axis=0)), axis=0) / np.sum(
            centered**2, axis=0
        )
        a = y.mean(axis=0) - b * x.mean(axis=0)
        variance = np.sum((y - a - b * x) ** 2, axis=0) / (len(x) - 2)
        half_life = -np.log(2.0) / np.log(b)
        score = (current - a / (1 - b)) / np.sqrt(variance / (1 - b**2))

    valid = (b > 0) & (b < 1) & (variance > 1e-16)
    valid &= (half_life >= 1) & (half_life <= len(history) / 2)
    return np.where(valid, score, np.nan), np.where(valid, half_life, np.nan)


class ResidualSignalModel(Model):
    """Accumulate each latest residual once; never rewrite historical residuals.

    Today's level is compared with the preceding window of levels. Refits can
    change the residual definition: this is an adaptive signal, not a fixed hedge.
    """

    def __init__(
        self, factor_name: str, window: int = 60, use_ou: bool = False
    ) -> None:
        assert window >= 5
        self.factor_name = factor_name
        self.window = window
        self.use_ou = use_ou
        self.level: np.ndarray | None = None
        self.history: deque[np.ndarray] = deque(maxlen=window)

    def update(
        self,
        timestamp: np.datetime64,
        returns: np.ndarray,
        model_outputs: dict[str, Any],
    ) -> ResidualSignal:
        residual = model_outputs[self.factor_name][-1].residuals

        if self.level is None:
            self.level = np.zeros_like(residual)

        self.level = np.where(np.isfinite(residual), self.level + residual, 0.0)
        current = np.where(np.isfinite(residual), self.level, np.nan)
        score = np.full_like(current, np.nan)
        half_life = np.full_like(current, np.nan)

        if len(self.history) == self.window:
            history = np.asarray(self.history)

            if self.use_ou:
                score, half_life = ou_score(history, current)
            else:
                deviation = history.std(axis=0, ddof=1)
                np.divide(
                    current - history.mean(axis=0),
                    deviation,
                    out=score,
                    where=np.isfinite(deviation) & (deviation > 1e-8),
                )

        self.history.append(current.copy())
        return ResidualSignal(level=current, score=score, half_life=half_life)
