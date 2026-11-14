from dataclasses import dataclass
import numpy as np
from typing import Any

from src.models.model import Model
from src.models.pca import CorrelationPCA

@dataclass(kw_only=True)
class ReturnAttribution:
    factor_returns: np.ndarray
    factor_intercept: np.ndarray
    factor_projection: np.ndarray
    residuals: np.ndarray

class FactorModel(Model):
    pca_name: str
    num_components: int
    window: int
    recalculate: int

    active_pca: CorrelationPCA | None
    coefficients: np.ndarray | None
    staleness: int

    def __init__(self, pca_name: str, num_components: int, window: int, recalculate: int) -> None:
        assert 0 < num_components and num_components < window and recalculate > 0

        self.pca_name = pca_name
        self.num_components = num_components
        self.window = window
        self.recalculate = recalculate

        self.active_pca = None
        self.coefficients = None
        self.staleness = recalculate

    def update(
        self,
        timestamp: np.datetime64,
        returns: np.ndarray,
        model_outputs: dict[str, Any],
    ) -> ReturnAttribution:
        assert returns.ndim == 2
        return_pca: CorrelationPCA | None = model_outputs[self.pca_name][-1]
        if return_pca is None or len(returns) < self.window:
            return self._empty_attribution(returns.shape[1])
        assert return_pca.components.shape[1] == self.num_components

        weights = return_pca.components / return_pca.std_devs[:, np.newaxis]
        if self.staleness >= self.recalculate or return_pca is not self.active_pca:
            realized_factors = returns[-self.window:] @ weights
            features = np.hstack((
                np.ones((len(realized_factors), 1), dtype=np.float64),
                realized_factors,
            ))
            self.coefficients = np.linalg.lstsq(features, returns[-self.window:], rcond=None)[0]
            self.active_pca = return_pca
            self.staleness = 0

        assert self.coefficients is not None
        self.staleness += 1
        factor_returns = returns[-1] @ weights
        factor_intercept = self.coefficients[0]
        factor_projection = factor_returns @ self.coefficients[1:]
        residuals = returns[-1] - factor_intercept - factor_projection

        return ReturnAttribution(
            factor_returns=factor_returns,
            factor_intercept=factor_intercept,
            factor_projection=factor_projection,
            residuals=residuals,
        )

    def _empty_attribution(self, num_stocks: int) -> ReturnAttribution:
        return ReturnAttribution(
            factor_returns=np.full(self.num_components, np.nan, dtype=np.float64),
            factor_intercept=np.full(num_stocks, np.nan, dtype=np.float64),
            factor_projection=np.full(num_stocks, np.nan, dtype=np.float64),
            residuals=np.full(num_stocks, np.nan, dtype=np.float64),
        )