from dataclasses import dataclass
import numpy as np
import scipy
from scipy.sparse.linalg import LinearOperator
from typing import Any

from src.models.model import Model

@dataclass(kw_only=True)
class CorrelationPCA:
    means: np.ndarray
    std_devs: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray

@dataclass(kw_only=True)
class CovariancePCA:
    means: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray

class PCAModel(Model):
    num_components: int
    window: int
    recalculate: int

    return_pca: CorrelationPCA | None
    staleness: int

    def __init__(self, num_components: int, window: int, recalculate: int) -> None:
        assert 0 < num_components and num_components < window and recalculate > 0

        self.num_components = num_components
        self.window = window
        self.recalculate = recalculate

        self.return_pca = None
        self.staleness = recalculate

    def update(
        self,
        timestamp: np.datetime64,
        returns: np.ndarray,
        model_outputs: dict[str, Any],
    ) -> CorrelationPCA | None:
        assert returns.ndim == 2
        if len(returns) < self.window:
            return None
        if self.staleness >= self.recalculate:
            self.return_pca = correlation_pca(returns[-self.window:], self.num_components)
            self.staleness = 0
        self.staleness += 1
        return self.return_pca

def correlation_pca(returns: np.ndarray, num_components: int) -> CorrelationPCA:
    assert returns.ndim == 2 and returns.shape[0] >= 2
    assert 0 < num_components and num_components <= returns.shape[1]

    means = np.mean(returns, axis=0)
    std_devs = np.std(returns, axis=0, ddof=1)
    assert np.all(std_devs > 0)
    pca = covariance_pca((returns - means) / std_devs, num_components)

    return CorrelationPCA(
        means=means,
        std_devs=std_devs,
        components=pca.components,
        eigenvalues=pca.eigenvalues,
    )

def covariance_pca(returns: np.ndarray, num_components: int) -> CovariancePCA:
    assert returns.ndim == 2 and returns.shape[0] >= 2
    assert 0 < num_components and num_components <= returns.shape[1]
    if num_components == returns.shape[1]:
        return basic_covariance_pca(returns, num_components)

    num_days, num_stocks = returns.shape
    means = np.mean(returns, axis=0)
    centered = returns - means
    covariance = LinearOperator(
        shape=(num_stocks, num_stocks),
        matvec=lambda v: centered.T @ (centered @ v) / (num_days - 1), # type: ignore
        dtype=np.float64,
    )
    eigenvalues, components = scipy.sparse.linalg.eigsh(
        A=covariance,
        k=num_components,
        which="LM",
        tol=1e-10,
    )
    order = np.argsort(eigenvalues, descending=True)

    return CovariancePCA(
        means=means,
        components=components[:, order],
        eigenvalues=eigenvalues[order],
    )

def basic_covariance_pca(returns: np.ndarray, num_components: int) -> CovariancePCA:
    assert returns.ndim == 2 and returns.shape[0] >= 2
    assert 0 < num_components and num_components <= returns.shape[1]

    means = np.mean(returns, axis=0)
    covariance = np.cov(returns, rowvar=False)
    eigenvalues, components = np.linalg.eigh(covariance)
    eigenvalues = eigenvalues[::-1][:num_components]
    components = components[:, ::-1][:, :num_components]

    return CovariancePCA(
        means=means,
        components=components,
        eigenvalues=eigenvalues,
    )