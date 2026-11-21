from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.models.model import Model


@dataclass(kw_only=True)
class ResidualForecast:
    expected_return: np.ndarray
    training_days: np.ndarray


@dataclass
class PendingForecast:
    features: np.ndarray
    betas: np.ndarray
    weights: np.ndarray
    growth: np.ndarray
    age: int = 0


class ResidualCalibrationModel(Model):
    """Pooled ridge spline predicting frozen-origin buy-and-hold hedge returns.

    Origin-date features, PCA weights and betas remain frozen until the
    forward return is fully observed. Training dates have equal weight. Only mature labels
    enter the rolling training set; the intercept is not a hedge expense.
    """

    def __init__(
        self,
        signal_name: str,
        factor_name: str,
        pca_name: str,
        window: int = 756,
        min_days: int = 126,
        ridge: float = 0.01,
        recalculate: int = 21,
        horizon: int = 1,
    ) -> None:
        assert 0 < min_days <= window and ridge > 0 and recalculate > 0
        assert horizon >= 1
        self.signal_name = signal_name
        self.factor_name = factor_name
        self.pca_name = pca_name
        self.window = window
        self.min_days = min_days
        self.ridge = ridge
        self.recalculate = recalculate
        self.horizon = horizon
        self.pending: deque[PendingForecast] = deque()
        self.moments: deque[tuple[np.ndarray, np.ndarray]] = deque()
        self.gram = np.zeros((5, 5))
        self.target = np.zeros(5)
        self.coefficients: np.ndarray | None = None
        self.staleness = recalculate

    @staticmethod
    def features(score: np.ndarray) -> np.ndarray:
        z = np.clip(score, -5, 5)
        return np.column_stack(
            (np.ones_like(z), z, *(np.maximum(z - k, 0) for k in (-2, 0, 2)))
        )

    def update(
        self,
        timestamp: np.datetime64,
        returns: np.ndarray,
        model_outputs: dict[str, Any],
    ) -> ResidualForecast:
        for observation in self.pending:
            observation.growth *= 1 + returns[-1]
            observation.age += 1

        if self.pending and self.pending[0].age == self.horizon:
            observation = self.pending.popleft()
            realized = observation.growth - 1
            features = observation.features
            label = realized - (realized @ observation.weights) @ observation.betas.T
            valid = np.isfinite(features).all(axis=1) & np.isfinite(label)

            if valid.sum() >= 20:
                x, y = features[valid], label[valid]
                gram, target = x.T @ x / len(y), x.T @ y / len(y)
                self.gram += gram
                self.target += target
                self.moments.append((gram, target))
                self.staleness += 1

                if len(self.moments) > self.window:
                    old_gram, old_target = self.moments.popleft()
                    self.gram -= old_gram
                    self.target -= old_target

        if len(self.moments) >= self.min_days and self.staleness >= self.recalculate:
            penalty = np.diag([0, self.ridge, self.ridge, self.ridge, self.ridge])
            self.coefficients = np.linalg.solve(
                self.gram / len(self.moments) + penalty,
                self.target / len(self.moments),
            )
            self.staleness = 0

        score = model_outputs[self.signal_name][-1].score
        features = self.features(score)
        pca = model_outputs[self.pca_name][-1]
        if pca is not None:
            betas = model_outputs[self.factor_name][-1].betas
            if np.isfinite(betas).all() and np.isfinite(score).any():
                self.pending.append(
                    PendingForecast(
                        features=features.copy(),
                        betas=betas.copy(),
                        weights=pca.components / pca.std_devs[:, None],
                        growth=np.ones_like(score),
                    )
                )

        prediction = np.full_like(score, np.nan)

        if self.coefficients is not None:
            prediction = features @ self.coefficients

        return ResidualForecast(
            expected_return=prediction, training_days=np.array(len(self.moments))
        )
